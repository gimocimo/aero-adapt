"""Txt2Fly's language layer (spec 06 §9): plain words → a typed plan, a clarifying question or
a refusal, through Claude with structured outputs.

The model only translates. Its reply must match a schema made for the site: a place is an enum
of the site's place ids, so it cannot name a place that does not exist, and the reply is one of
plan, clarify or refuse. Numbers are left to the builder and the checker, which explain their
limits. Everything after the reply is the same deterministic code as for a typed plan: build,
check, preview, confirm.

The prompt:

* **system**, cached: the rules and the plan reference with worked examples (the same for
  every site, ``prompt_rules.md``), then the site (places with their ground and top heights,
  no-fly areas, the height band) and the aircraft (speeds, roll limit, tightest circles). Both
  are fixed for a session, so they are written once and then read from the cache.
* **each instruction**: the state now (position, height, wind, the tightest circle in this
  wind, battery, the active plan), after the last few turns of the conversation.

Translators:

* ``ClaudeTranslator``: the API. The SDK reads ``ANTHROPIC_API_KEY`` from the environment of
  the process; the key is never stored, logged or passed around here.
* ``StandIn``: a few fixed phrasings, for tests and for flying without a key.
* ``Recorded``: replies recorded from the API, for CI (which never calls it).
"""

from __future__ import annotations

import functools
import json
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, create_model

from aeroadapt.txt2fly.builder import COMPASS, Aircraft, _speeds, r_min
from aeroadapt.txt2fly.energy import RESERVE, energy_model
from aeroadapt.txt2fly.gazetteer import Gazetteer, Place, UnknownPlace
from aeroadapt.txt2fly.plan import Plan
from aeroadapt.txt2fly.terrain import relief

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

TIMEOUT_S = 30.0
MAX_TURNS = 6  # turns of the conversation sent with each instruction
MAX_PLACES = 120
STREET_KINDS = ("road", "street", "lane")  # left out of the model's list on a big site
NO_TOP_KINDS = ("home", "safe zone", "field", "airstrip", "river", "road", "street", "lane",
                "park", "green")  # fmt: skip
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class Setting:
    """A model configuration (spec 06 §11)."""

    name: str
    model: str
    thinking: bool  # adaptive thinking; False turns it off
    effort: str
    max_tokens: int
    fallbacks: bool = False  # the server-side refusal fallback (Opus)


SETTINGS = {
    # The default: Haiku 5.5 without thinking, for latency (the owner's choice, 21 Oct 2026).
    "haiku": Setting("haiku", "claude-haiku-5-5", False, "low", 4096),
    "haiku-think": Setting("haiku-think", "claude-haiku-5-5", True, "low", 16000),
    # The reference: what a larger model adds. Opus 5.5 always thinks; medium is its default.
    "opus": Setting("opus", "claude-opus-5-5", True, "medium", 16000, fallbacks=True),
}


# --------------------------------------------------------------------------- replies
@dataclass(frozen=True)
class Clarify:
    question: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class Refuse:
    reason: str


Reply = Plan | Clarify | Refuse


@dataclass(frozen=True)
class Turn:
    """One exchange, as it was sent: the state and text, and the model's reply (JSON)."""

    user: str
    reply: str


@dataclass
class Translation:
    reply: Reply
    raw: dict[str, Any]  # the reply as the model wrote it
    translator: str
    model: str | None = None  # the model that answered (a fallback may differ)
    latency_s: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    turn: Turn | None = None  # to append to the conversation


class TranslatorError(RuntimeError):
    """The translator could not answer: shown to the operator; the flight carries on."""


@dataclass
class Context:
    """What the translator is told besides the text."""

    gaz: Gazetteer
    aircraft: Aircraft
    active: dict[str, Any] | None = None  # the active plan, as ``report_json`` gives it
    history: Sequence[Turn] = ()
    time_s: float | None = None


class Translator(Protocol):
    name: str

    async def translate(self, ctx: Context, text: str) -> Translation: ...


# --------------------------------------------------------------------------- the schema
class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


@functools.cache
def reply_model(ids: tuple[str, ...]) -> type[BaseModel]:
    """The reply schema for a site: the plan of ``plan.py`` with places limited to ``ids`` (and
    no map points), or a question, or a refusal. Its fields mirror the plan's, so a reply's plan
    validates as a ``Plan``. Built at run time (the ids are the site's), hence ``create_model``.
    Unions are plain: each variant's literal tells them apart, and the API's schemas have no
    discriminator keyword."""
    lit: Any = Literal  # the values are only known at run time

    def m(name: str, **fields: Any) -> Any:
        return create_model(name, __base__=_M, **fields)

    place = m("PlaceRef", kind=(lit["place"], ...), id=(lit[ids], ...))
    here = m("Here", kind=(lit["here"], ...))
    home = m("Home", kind=(lit["home"], ...))
    offset = m(
        "Offset",
        kind=(lit["offset"], ...),
        origin=(lit[(*ids, "here", "home")], ...),
        bearing_deg=(
            float,
            Field(description="true bearing from the origin, clockwise from north"),
        ),
        distance_m=(float, ...),
    )
    target = place | here | home | offset
    opt = (float | None, None)
    steps = [
        m("Goto", do=(lit["goto"], ...), target=(target, ...), alt_m=opt),
        m(
            "PassOver",
            do=(lit["pass_over"], ...),
            target=(target, ...),
            alt_m=opt,
            heading=(lit["along", "into_wind"] | float, "along"),
        ),
        m(
            "Orbit",
            do=(lit["orbit"], ...),
            target=(target, ...),
            radius_m=opt,
            alt_m=opt,
            direction=(lit["cw", "ccw"], "cw"),
            turns=opt,
            duration_s=opt,
        ),
        m(
            "Patrol",
            do=(lit["patrol"], ...),
            targets=(list[target], ...),
            visit=(lit["pass_over", "goto"], "pass_over"),
            alt_m=opt,
            laps=(int | None, None),
        ),
        m(
            "SetSpeed",
            do=(lit["set_speed"], ...),
            speed=(lit["slow", "cruise", "fast"] | float, ...),
        ),
        m("Hold", do=(lit["hold"], ...), radius_m=opt),
        m("ReturnHome", do=(lit["return_home"], ...)),
    ]
    step = steps[0]
    for s in steps[1:]:
        step = step | s
    plan = m("TPlan", summary=(str, ...), assumptions=(list[str], ...), steps=(list[step], ...))
    replies = (
        m("PlanReply", kind=(lit["plan"], ...), plan=(plan, ...))
        | m(
            "ClarifyReply",
            kind=(lit["clarify"], ...),
            question=(str, ...),
            options=(list[str], ...),
        )
        | m("RefuseReply", kind=(lit["refuse"], ...), reason=(str, ...))
    )
    return m("Reply", reply=(replies, ...))


def to_reply(raw: dict[str, Any]) -> Reply:
    """The model's JSON as a typed reply (a plan validates as ``plan.Plan``)."""
    r = raw["reply"]
    if r["kind"] == "plan":
        return Plan.model_validate(r["plan"])
    if r["kind"] == "clarify":
        return Clarify(str(r["question"]), tuple(str(o) for o in r.get("options") or ())[:4])
    return Refuse(str(r["reason"]))


# --------------------------------------------------------------------------- the prompt
@functools.cache
def _rules() -> str:
    return (Path(__file__).with_name("prompt_rules.md")).read_text()


def prompt_places(gaz: Gazetteer) -> list[Place]:
    """The places the model may name: on a big site (a city), its streets are left out; a name
    given twice keeps its first place."""
    places = list(gaz.places.values())
    if len(places) > MAX_PLACES:
        places = [p for p in places if p.kind not in STREET_KINDS]
    seen: set[str] = set()
    out = []
    for p in places:
        if p.name.lower() not in seen:
            seen.add(p.name.lower())
            out.append(p)
    return out[:MAX_PLACES]


def _compass(deg: float) -> str:
    return COMPASS[round((deg % 360.0) / 45.0) % 8]


def _from_home(p: np.ndarray, home: np.ndarray) -> str:
    d = np.asarray(p, float) - home
    dist = float(np.hypot(*d))
    if dist < 25.0:
        return "at home"
    brg = math.degrees(math.atan2(d[1], d[0]))
    return f"{dist:.0f} m {_compass(brg)} of home (N {d[0]:.0f}, E {d[1]:.0f})"


def place_heights(gaz: Gazetteer, p: Place) -> tuple[float, float | None]:
    """A place's ground and top, metres above home (the top: None where it means nothing)."""
    rel = relief(gaz.site)
    if rel is None:
        top = p.height_m
        return 0.0, top
    ground = float(rel.ground(p.point[None])[0])
    if p.kind in NO_TOP_KINDS:
        return ground, None
    top = float(rel.top(p.point[None])[0])
    if p.height_m is not None:
        top = max(top, ground + p.height_m)
    return ground, top


_TRIVIAL_ALIAS = re.compile(r"^((safe )?zone )?[A-Z]\d*$")  # "zone A", "A": the id says it


def site_block(gaz: Gazetteer, ac: Aircraft) -> str:
    """The site and the aircraft: fixed for a session (cached with the rules)."""
    site, ops = gaz.site, gaz.site.operations
    assert ops is not None
    lines = [f"# This site: {site.name}", ""]
    lines.append(
        "Positions are from home, in metres north (N) and east (E). Heights are metres above home."
    )
    band = f"Plans fly at least {ops.min_agl_m:g} m above the ground and at most "
    band += f"{ops.ceiling_agl_m:g} m above it"
    if ops.ceiling_alt_m is not None:
        band += f", and never above {ops.ceiling_alt_m:g} m above home"
    lines += [band + ", inside the geofence.", ""]
    fence = gaz.geofence - gaz.home
    lines.append(
        f"The geofence spans N {fence[:, 0].min():.0f} to {fence[:, 0].max():.0f}, "
        f"E {fence[:, 1].min():.0f} to {fence[:, 1].max():.0f}."
    )
    if gaz.no_fly:
        lines += ["", "No-fly areas (the planner keeps out of them):"]
        lines += [f"- {json.dumps(a.name)}: {a.reason}" for a in gaz.no_fly]
    lines += ["", "Places (id: name, kind, where; ground and top in metres above home):"]
    for p in prompt_places(gaz):
        ground, top = place_heights(gaz, p)
        h = f"ground {round(ground)} m" + (f", top {round(top)} m" if top is not None else "")
        also = [a for a in p.aliases if not _TRIVIAL_ALIAS.match(a)]
        aka = f" (also {', '.join(json.dumps(a) for a in also)})" if also else ""
        lines.append(f"- `{p.id}`: {json.dumps(p.name)}{aka}, {p.kind}, "
                     f"{_from_home(p.point, gaz.home)}; {h}")  # fmt: skip
    cfg, af = ac.stack_cfg, ac.airframe
    sp = _speeds(cfg)
    roll = cfg.l1.roll_limit_deg
    model = energy_model(af, site, cfg.tecs.airspeed_min_mps, cfg.tecs.airspeed_max_mps)
    lines += [
        "",
        f"# The aircraft: {af.name} (`{af.id}`)",
        "",
        f"- Speeds: slow {sp['slow']:g} m/s, cruise {sp['cruise']:g} m/s, fast {sp['fast']:g} m/s.",
        f"- Roll limit {roll:g}°. Tightest circle in calm air: "
        + ", ".join(f"{r_min(v, 0.0, roll):.0f} m {k}" for k, v in sp.items())
        + ". Wind makes it wider: the state gives it for the wind now.",
        f"- Battery {model.capacity_mah:.0f} mAh, of which {RESERVE:.0%} is kept in reserve.",
    ]
    return "\n".join(lines)


def system_blocks(gaz: Gazetteer, ac: Aircraft) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": _rules()},
        {"type": "text", "text": site_block(gaz, ac), "cache_control": {"type": "ephemeral"}},
    ]


def _active(active: dict[str, Any]) -> str:
    steps = []
    for i, s in enumerate(active.get("steps") or []):
        extra = []
        if s.get("radius") is not None:
            extra.append(f"radius {s['radius']:g} m")
        if s.get("alt") is not None:
            extra.append(f"at {s['alt']:g} m")
        steps.append(f"{i + 1}. {s['label']}" + (f" ({', '.join(extra)})" if extra else ""))
    plan = json.dumps(active.get("plan"), separators=(",", ":"))
    return f"{active.get('summary') or 'a plan'}. Steps: {' '.join(steps)} As a plan: {plan}"


def state_text(ctx: Context) -> str:
    """The state now, as the planner knows it (estimates, not the simulator's truth)."""
    ac, gaz = ctx.aircraft, ctx.gaz
    pos = np.asarray(ac.pos_ne, float)
    lines = ["# State now", ""]
    if ctx.time_s is not None:
        lines.append(f"- Time: {ctx.time_s:.0f} s into the flight.")
    lines.append(f"- Aircraft: {_from_home(pos, gaz.home)}, {ac.alt_agl:.0f} m above home, "
                 f"flying {ac.course_deg % 360:03.0f}° ({_compass(ac.course_deg)}).")  # fmt: skip
    near = min(prompt_places(gaz), key=lambda p: float(np.hypot(*(p.point - pos))))
    d = near.point - pos
    lines.append(f"- Nearest place: {json.dumps(near.name)}, {np.hypot(*d):.0f} m "
                 f"{_compass(math.degrees(math.atan2(d[1], d[0])))}.")  # fmt: skip
    w = ac.wind_speed
    frm = math.degrees(math.atan2(-ac.wind_ne[1], -ac.wind_ne[0])) % 360.0
    lines.append(f"- Wind: {w:.0f} m/s from {frm:03.0f}°." if w >= 0.5 else "- Wind: calm.")
    sp = _speeds(ac.stack_cfg)
    roll = ac.stack_cfg.l1.roll_limit_deg
    lines.append(
        "- Tightest circle in this wind: "
        + ", ".join(f"{r_min(v, w, roll):.0f} m {k}" for k, v in sp.items())
        + "."
    )
    cfg = ac.stack_cfg.tecs
    model = energy_model(ac.airframe, gaz.site, cfg.airspeed_min_mps, cfg.airspeed_max_mps)
    left = 1.0 - ac.used_mah / model.capacity_mah
    usable = model.capacity_mah * (1.0 - RESERVE) - ac.used_mah
    amps = float(model.current(np.asarray(cfg.airspeed_cruise_mps), np.asarray(0.0)))
    minutes = max(usable, 0.0) / (amps * 1000.0) * 60.0
    lines.append(f"- Battery: {left:.0%} left, about {minutes:.0f} min at cruise to the reserve.")
    lines.append(f"- Active plan: {_active(ctx.active)}" if ctx.active else "- Active plan: none.")
    return "\n".join(lines)


def turn_text(ctx: Context, text: str) -> str:
    return f"{state_text(ctx)}\n\n<operator>\n{text.strip()}\n</operator>"


def _messages(ctx: Context, user: str) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    for t in list(ctx.history)[-MAX_TURNS:]:
        msgs += [{"role": "user", "content": t.user}, {"role": "assistant", "content": t.reply}]
    return [*msgs, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- Claude
class ClaudeTranslator:
    """Claude through the official SDK (``anthropic``, the ``txt2fly`` extra)."""

    def __init__(self, setting: str = "haiku", client: AsyncAnthropic | None = None) -> None:
        self.setting = SETTINGS[setting]
        self.name = f"claude:{self.setting.name}"
        self._client = client

    @property
    def client(self) -> AsyncAnthropic:
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(timeout=TIMEOUT_S, max_retries=1)
        return self._client

    async def translate(self, ctx: Context, text: str) -> Translation:
        import anthropic

        s = self.setting
        ids = tuple(p.id for p in prompt_places(ctx.gaz))
        user = turn_text(ctx, text)
        kwargs: dict[str, Any] = {
            "model": s.model,
            "max_tokens": s.max_tokens,
            "system": system_blocks(ctx.gaz, ctx.aircraft),
            "messages": _messages(ctx, user),
            "output_format": reply_model(ids),
            "output_config": {"effort": s.effort},
            "thinking": {"type": "adaptive"} if s.thinking else {"type": "disabled"},
        }
        t0 = time.perf_counter()
        resp: Any = None
        for attempt in range(2):
            try:
                if s.fallbacks:
                    resp = await self.client.beta.messages.parse(
                        **kwargs, betas=[FALLBACK_BETA], fallbacks="default"
                    )
                else:
                    resp = await self.client.messages.parse(**kwargs)
            except anthropic.APITimeoutError as e:
                raise TranslatorError(f"the language model took over {TIMEOUT_S:g} s") from e
            except anthropic.RateLimitError as e:
                raise TranslatorError("the language model is busy (rate limit); try again") from e
            except anthropic.AuthenticationError as e:
                raise TranslatorError("the API key was not accepted") from e
            except anthropic.APIStatusError as e:
                raise TranslatorError(f"the language model failed ({e.status_code})") from e
            except anthropic.APIConnectionError as e:
                raise TranslatorError("the language model could not be reached") from e
            except ValueError as e:  # the reply did not validate against the schema
                raise TranslatorError("the reply did not match the plan format") from e
            if resp.stop_reason == "max_tokens" and attempt == 0:
                kwargs["max_tokens"] *= 2  # one retry with more room
                continue
            break
        assert resp is not None
        latency = time.perf_counter() - t0
        u = resp.usage
        usage = {
            "input": u.input_tokens,
            "output": u.output_tokens,
            "cache_read": u.cache_read_input_tokens or 0,
            "cache_write": u.cache_creation_input_tokens or 0,
        }
        if resp.stop_reason == "refusal":
            raw: dict[str, Any] = {
                "reply": {"kind": "refuse", "reason": "The language model declined this request."}
            }
        elif resp.stop_reason == "max_tokens" or resp.parsed_output is None:
            raise TranslatorError("the language model's reply was cut short")
        else:
            raw = resp.parsed_output.model_dump(mode="json")
        return Translation(
            to_reply(raw), raw, self.name, resp.model, latency, usage, resp.stop_reason,
            Turn(user, json.dumps(raw, separators=(",", ":"))),
        )  # fmt: skip


# --------------------------------------------------------------------------- stand-ins
_UNSAFE = re.compile(
    r"\b(ignore|disable|turn off|switch off|override|remove|bypass|raise|lift)\b.*"
    r"\b(geofence|fence|no[- ]fly|ceiling|limits?|contingency|failsafe|reserve|rules?)\b"
)
_HOME = re.compile(r"\b(go|come|return|fly|head)( back)? home\b|\brtl\b|^home$")
_ALT = re.compile(r"\bat (\d+(?:\.\d+)?) ?(?:m|metres|meters)\b")
_RADIUS = re.compile(r"\b(\d+(?:\.\d+)?) ?m (?:circle|radius)|\bradius (?:of )?(\d+(?:\.\d+)?)")
_VERBS = (
    ("orbit", re.compile(r"^(?:circle|orbit|loiter (?:over|around|round)|fly a?round) (.+)")),
    ("pass_over", re.compile(r"^(?:fly by|fly over|pass over|overfly|make a pass over) (.+)")),
    ("patrol", re.compile(r"^patrol (.+)")),
    ("goto", re.compile(r"^(?:go to|fly to|head to|head for) (.+)")),
)  # fmt: skip


class StandIn:
    """A few fixed phrasings, no model: "circle the church at 80 m", "fly by field A",
    "patrol fields A, B and C", "go to the mast", "hold", "go home", joined by "then". For tests
    and for flying without an API key."""

    name = "stand-in"

    def _place(self, gaz: Gazetteer, phrase: str) -> dict[str, Any] | None:
        phrase = re.sub(r"\b(at|with|in|then)\b.*$", "", phrase).strip(" ,.")
        for cand in (phrase, phrase.removeprefix("the "), "field " + phrase.split()[-1]):
            try:
                return {"kind": "place", "id": gaz.place(cand).id}
            except UnknownPlace:
                continue
        return None

    def _step(self, gaz: Gazetteer, clause: str) -> dict[str, Any] | str:
        if _HOME.search(clause):
            return {"do": "return_home"}
        if re.match(r"^(hold|wait|stop|loiter here|circle here)\b", clause):
            return {"do": "hold"}
        alt = _ALT.search(clause)
        for do, rx in _VERBS:
            m = rx.match(clause)
            if m is None:
                continue
            rest = m.group(1)
            step: dict[str, Any] = {"do": do}
            if alt:
                step["alt_m"] = float(alt.group(1))
            if do == "patrol":
                names = re.split(r",\s*|\s+and\s+", re.sub(r"\bat .*$", "", rest))
                names = [re.sub(r"^fields? ", "", n).strip() for n in names if n.strip()]
                targets = [self._place(gaz, n) for n in names]
                if not targets or any(t is None for t in targets):
                    return f"I don't know all of these places: {rest}"
                step["targets"] = targets
                return step
            target = self._place(gaz, rest)
            if target is None:
                return f"I don't know a place called '{rest}'"
            step["target"] = target
            rad = _RADIUS.search(clause)
            if do == "orbit" and rad:
                step["radius_m"] = float(rad.group(1) or rad.group(2))
            return step
        return "I can only read a few fixed phrasings without the language model"

    async def translate(self, ctx: Context, text: str) -> Translation:
        t = text.strip().lower()
        if _UNSAFE.search(t):
            raw = {"reply": {"kind": "refuse", "reason": "Safety settings can't be changed by "
                             "a plan. I can plan a flight inside them."}}  # fmt: skip
        elif re.search(r"\bland\b", t):
            raw = {"reply": {"kind": "refuse", "reason": "Plans can't land the aircraft."}}
        else:
            steps: list[dict[str, Any]] = []
            problem = None
            for clause in re.split(r",? (?:and )?then |; ", t):
                s = self._step(ctx.gaz, clause.strip(" ,."))
                if isinstance(s, str):
                    problem = s
                    break
                steps.append(s)
            if problem is not None or not steps:
                raw = {"reply": {"kind": "clarify", "question": f"{problem or 'Say again?'}. "
                                 "Try: 'circle the church at 80 m'.", "options": []}}  # fmt: skip
            else:
                raw = {
                    "reply": {
                        "kind": "plan",
                        "plan": {
                            "summary": text.strip()[:1].upper() + text.strip()[1:],
                            "assumptions": [],
                            "steps": steps,
                        },
                    }
                }
        return Translation(to_reply(raw), raw, self.name,
                           turn=Turn(turn_text(ctx, text), json.dumps(raw)))  # fmt: skip


class Recorded:
    """Replies recorded from the API (``{"<site>|<text>": raw}``), for CI."""

    name = "recorded"

    def __init__(self, path: str | Path) -> None:
        self.replies: dict[str, dict[str, Any]] = json.loads(Path(path).read_text())

    async def translate(self, ctx: Context, text: str) -> Translation:
        key = f"{ctx.gaz.site.id}|{text.strip()}"
        if key not in self.replies:
            raise TranslatorError(f"no recorded reply for '{text.strip()}'")
        raw = self.replies[key]
        return Translation(to_reply(raw), raw, self.name,
                           turn=Turn(turn_text(ctx, text), json.dumps(raw)))  # fmt: skip


def available() -> bool:
    """Whether the API translator can run: the SDK is installed and a credential is set."""
    import os

    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
