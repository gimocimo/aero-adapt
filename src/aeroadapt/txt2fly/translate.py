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
  sizes and extents, lines with their course; no-fly areas with their extents and how high
  they reach; the height band) and the aircraft (speeds, roll limit, tightest circles). Both
  are fixed for a session, so they are written once and then read from the cache.
* **the map** (a setting, off by default), cached: a picture of the site
  (``configs/sites/<site>.map.png``, drawn by ``tools/site_map.py`` from the same data), first
  in the conversation.
* **each instruction**: the state now (position, height, wind, the tightest circle in this
  wind, battery, the active plan), after the last few turns of the conversation.
* **the review** (``translate_checked``): when the builder could not build the plan, or it
  fails a hard check, the model is shown what would be flown and why, and answers once more:
  the same plan, a corrected one, or a question to the operator.

Translators:

* ``ClaudeTranslator``: the API. The SDK reads ``ANTHROPIC_API_KEY`` from the environment of
  the process; the key is never stored, logged or passed around here. An organisation-wide key
  also needs ``ANTHROPIC_WORKSPACE_ID`` (sent as the ``anthropic-workspace-id`` header).
* ``StandIn``: a few fixed phrasings, for tests and for flying without a key.
* ``Recorded``: replies recorded from the API, for CI (which never calls it).
"""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import math
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, create_model

from aeroadapt.scenarios.site import SITE_DIR
from aeroadapt.txt2fly.builder import COMPASS, Aircraft, _speeds, r_min
from aeroadapt.txt2fly.energy import RESERVE, energy_model
from aeroadapt.txt2fly.gazetteer import (
    ABOVE_ZONE_M,
    FIELD_WORDS,
    Gazetteer,
    NoFlyArea,
    Place,
    UnknownPlace,
)
from aeroadapt.txt2fly.plan import (
    Around,
    Follow,
    Goto,
    Hold,
    Orbit,
    PassOver,
    Patrol,
    Plan,
    ReturnHome,
    SetSpeed,
    Survey,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

    from aeroadapt.txt2fly.checker import Report

TIMEOUT_S = 60.0  # a site's first request compiles its schema (~30 s, then cached a day)
MAX_TURNS = 6  # turns of the conversation sent with each instruction
MAX_PLACES = 120
STREET_KINDS = ("road", "street", "lane")  # left out of the model's list on a big site
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
    map: bool = False  # send the site's map picture (docs/06 §17: off for Haiku, see V32)


SETTINGS = {
    # Haiku 5.5 without thinking: the default until 25 Oct 2026 (the owner's choice, for latency).
    "haiku": Setting("haiku", "claude-haiku-5-5", False, "low", 4096),
    # The default since 25 Oct 2026 (the owner's choice): adaptive thinking meets every V32 target,
    # relative positions included, for about 0.5 s more (docs/validation/txt2fly_spatial.md).
    "haiku-think": Setting("haiku-think", "claude-haiku-5-5", True, "low", 16000),
    # With the map picture: measured, not the default (about 2 s slower, and Haiku trusted the
    # picture's few labels over the list: docs/validation/txt2fly_spatial.md).
    "haiku-map": Setting("haiku-map", "claude-haiku-5-5", False, "low", 4096, map=True),
    # The reference: what a larger model adds. Opus 5.5 always thinks; medium is its default.
    "opus": Setting("opus", "claude-opus-5-5", True, "medium", 16000, fallbacks=True),
}
DEFAULT = "haiku-think"  # the setting `serve` and `txt2fly ask|eval` use unless told


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
    review: str | None = None  # what the model was shown of its first plan, if anything
    first: dict[str, Any] | None = None  # its first reply, when the review changed it


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


STEP_CLASSES = {"goto": Goto, "pass_over": PassOver, "orbit": Orbit, "patrol": Patrol,
                "around": Around, "follow": Follow, "survey": Survey, "set_speed": SetSpeed,
                "hold": Hold, "return_home": ReturnHome}  # fmt: skip


@functools.cache
def reply_model(ids: tuple[str, ...], kinds: tuple[str, ...] = ()) -> type[BaseModel]:
    """The reply schema the model writes, for a site: flat, because the API compiles a
    schema into a grammar and nested unions (seven step kinds, each with four kinds of target)
    make it too large. One reply object (``kind`` plan, clarify or refuse, with the fields of
    each), one step object (``do`` and every step's optional fields), one target object
    (``kind`` and its fields). Places stay an enum of the site's ids, so the model cannot name
    a place that does not exist. ``nest`` turns a reply into the typed form, where the plan
    validates as ``plan.Plan`` exactly as a typed one."""
    lit: Any = Literal  # the values are only known at run time

    def m(name: str, **fields: Any) -> Any:
        return create_model(name, __base__=_M, **fields)

    # Every field required (null where it does not apply): a small model writing a long list
    # of optional fields skips some, the stated height among them.
    num = (float | None, ...)
    target = m(
        "Target",
        kind=(
            lit["place", "here", "home", "offset", "nearest", "furthest", "tallest", "beyond"],
            ...,
        ),
        id=(lit[ids] | None, ...),
        origin=(lit[(*ids, "here", "home")] | None, ...),
        bearing_deg=num,
        distance_m=num,
        among=((lit[kinds] if kinds else str) | None, ...),
    )
    step = m(
        "Step",
        do=(lit[tuple(STEP_CLASSES)], ...),
        target=(target | None, ...),
        targets=(list[target] | None, ...),
        alt_m=num,
        radius_m=num,
        direction=(lit["cw", "ccw"] | None, ...),
        turns=num,
        duration_s=num,
        heading=(lit["along", "into_wind"] | float | None, ...),
        visit=(lit["pass_over", "goto"] | None, ...),
        laps=(int | None, ...),
        speed=(lit["slow", "cruise", "fast"] | float | None, ...),
        standoff_m=num,
        start=(target | None, ...),
        end=(target | None, ...),
        side=(lit["over", "north", "south", "east", "west"] | None, ...),
        offset_m=num,
        spacing_m=num,
    )
    return m(
        "Reply",
        kind=(lit["plan", "clarify", "refuse"], ...),
        summary=(str | None, ...),
        assumptions=(list[str] | None, ...),
        steps=(list[step] | None, ...),
        question=(str | None, ...),
        options=(list[str] | None, ...),
        reason=(str | None, ...),
    )


def site_schema(gaz: Gazetteer) -> type[BaseModel]:
    """The reply schema for a site: its place ids, and its kinds of place (with the words for
    its fields, which are its safe zones)."""
    places = prompt_places(gaz)
    kinds = {p.kind for p in places} | set(FIELD_WORDS)
    return reply_model(tuple(p.id for p in places), tuple(sorted(kinds)))


def _target(t: dict[str, Any]) -> dict[str, Any]:
    if t["kind"] == "place":
        return {"kind": "place", "id": t.get("id")}
    if t["kind"] == "offset":
        keys = ("origin", "bearing_deg", "distance_m")
        return {"kind": "offset", **{k: t.get(k) for k in keys}}
    if t["kind"] in ("nearest", "furthest"):
        return {"kind": t["kind"], "among": t.get("among"), "origin": t.get("origin") or "here"}
    if t["kind"] == "tallest":
        return {"kind": "tallest", "among": t.get("among")}
    if t["kind"] == "beyond":
        out = {"kind": "beyond", "id": t.get("id"), "origin": t.get("origin") or "here"}
        return out | ({"distance_m": t["distance_m"]} if t.get("distance_m") is not None else {})
    return {"kind": t["kind"]}


def nest(flat: dict[str, Any]) -> dict[str, Any]:
    """A reply as the model writes it (flat) in the typed form (``{"reply": {...}}``): each
    step keeps the fields its kind has, unset ones taking the plan's defaults."""
    kind = flat["kind"]
    if kind == "clarify":
        return {"reply": {"kind": "clarify", "question": flat.get("question") or "Which one?",
                          "options": list(flat.get("options") or [])}}  # fmt: skip
    if kind == "refuse":
        return {"reply": {"kind": "refuse", "reason": flat.get("reason") or ""}}
    steps = []
    for s in flat.get("steps") or []:
        fields = STEP_CLASSES[s["do"]].model_fields
        s = dict(s)
        # One place given in the other field: unambiguous, so taken as meant.
        if "targets" in fields and not s.get("targets") and s.get("target"):
            s["targets"] = [s["target"]]
        if "target" in fields and not s.get("target") and len(s.get("targets") or []) == 1:
            s["target"] = s["targets"][0]
        step = {k: v for k, v in s.items() if k in fields and v is not None}
        for k in ("target", "start", "end"):
            if k in step:
                step[k] = _target(step[k])
        if "targets" in step:
            step["targets"] = [_target(t) for t in step["targets"]]
        steps.append(step)
    assumptions = list(flat.get("assumptions") or [])
    plan = {"summary": flat.get("summary") or "", "assumptions": assumptions, "steps": steps}
    return {"reply": {"kind": "plan", "plan": plan}}


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


def _span(pts: np.ndarray, home: np.ndarray) -> str:
    """Where an outline lies: its extent from home and its size."""
    d = np.asarray(pts, float) - home
    lo, hi = d.min(axis=0), d.max(axis=0)
    return (f"N {lo[0]:.0f} to {hi[0]:.0f}, E {lo[1]:.0f} to {hi[1]:.0f} (about "
            f"{hi[1] - lo[1]:.0f} m east-west by {hi[0] - lo[0]:.0f} m north-south)")  # fmt: skip


def _course(line: np.ndarray, home: np.ndarray) -> str:
    """A line's length and course: its ends and a few points between."""
    from shapely.geometry import LineString

    from aeroadapt.txt2fly.areas import points_along

    ls = LineString(line)
    pts = points_along(ls, 6) - home
    via = (
        " via " + ", ".join(f"(N {n:.0f}, E {e:.0f})" for n, e in pts[1:-1]) if len(pts) > 2 else ""
    )
    return (f"a line {ls.length:.0f} m long from (N {pts[0, 0]:.0f}, E {pts[0, 1]:.0f}){via} "
            f"to (N {pts[-1, 0]:.0f}, E {pts[-1, 1]:.0f})")  # fmt: skip


def _no_fly(gaz: Gazetteer) -> list[str]:
    """The no-fly areas, one line per name (the 22 turbines as one), with the place that
    stands for each, where it lies and how high it reaches."""
    groups: dict[str, list[NoFlyArea]] = {}
    for a in gaz.no_fly:
        groups.setdefault(a.name, []).append(a)
    ids = {p.id for p in gaz.places.values()}
    lines = []
    for name, areas in groups.items():
        # The place for it: its own id (one area), else a place of the same name.
        pid = areas[0].id if len(areas) == 1 and areas[0].id in ids else None
        pid = pid or next((p.id for p in gaz.places.values() if p.name == name), None)
        ref = f" (`{pid}`)" if pid else ""
        a0 = areas[0]
        if a0.ceiling_m is None:
            height = "closed at all heights"
        elif a0.base_m is None:
            height = (f"closed up to {a0.ceiling_m:g} m above the ground: a plan may cross it "
                      f"{a0.ceiling_m + ABOVE_ZONE_M:g} m or more above the ground")  # fmt: skip
        else:
            tops = sorted(a.top_m for a in areas if a.top_m is not None)
            top = f"{tops[0]:.0f}" if tops[-1] - tops[0] < 1 else f"{tops[0]:.0f} to {tops[-1]:.0f}"
            height = (f"closed up to {top} m above home: a plan may cross it "
                      f"{ABOVE_ZONE_M:g} m higher")  # fmt: skip
        count = f"{len(areas)} areas, " if len(areas) > 1 else ""
        where = _span(np.vstack([r for a in areas for r in a.polygons]), gaz.home)
        lines.append(f"- {json.dumps(name)}{ref}: {a0.reason}; {count}{height}; {where}")
    return lines


def place_heights(gaz: Gazetteer, p: Place) -> tuple[float, float | None]:
    """A place's ground and top, metres above home (the top: None where it means nothing)."""
    return gaz.heights(p)


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
        lines += ["", "No-fly areas (the planner keeps out of them, or crosses one closed up to "
                      "a height above it):"]  # fmt: skip
        lines += _no_fly(gaz)
    lines += ["", "Places (id: name, kind, where; ground and top in metres above home; an area's "
                  "extent and size, a line's length and course). The safe zones are the landing "
                  "fields, where the aircraft may land after a failure:"]  # fmt: skip
    for p in prompt_places(gaz):
        ground, top = place_heights(gaz, p)
        h = f"ground {round(ground)} m" + (f", top {round(top)} m" if top is not None else "")
        also = [a for a in p.aliases if not _TRIVIAL_ALIAS.match(a)]
        aka = f" (also {', '.join(json.dumps(a) for a in also)})" if also else ""
        shape = ""
        if p.line is not None:
            shape = f"; {_course(p.line, gaz.home)}"
        elif p.polygon is not None and p.kind not in STREET_KINDS:
            shape = f"; spans {_span(p.polygon, gaz.home)}"
        lines.append(f"- `{p.id}`: {json.dumps(p.name)}{aka}, {p.kind}, "
                     f"{_from_home(p.point, gaz.home)}; {h}{shape}")  # fmt: skip
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


@functools.cache
def site_map(site_id: str) -> tuple[str, float] | None:
    """The site's map picture (PNG, base64) and its grid spacing, if one has been drawn."""
    png, meta = SITE_DIR / f"{site_id}.map.png", SITE_DIR / f"{site_id}.map.json"
    if not png.exists() or not meta.exists():
        return None
    grid = float(json.loads(meta.read_text())["grid_m"])
    return base64.b64encode(png.read_bytes()).decode(), grid


MAP_NOTE = (
    "The map of this site, drawn from the data in the system prompt: north up; a grid every "
    "{grid:g} m, labelled as positions are given (N and E, metres from home); the ground "
    "(fields, woods, buildings, water), rivers and roads; safe zones outlined in green; no-fly "
    "areas in red where closed at all heights, orange where closed up to a height; the "
    "geofence dashed; home a blue star. Use it for sizes, shapes and what lies where."
)


def _messages(ctx: Context, user: str, with_map: bool = False) -> list[dict[str, Any]]:
    """The last turns and the instruction; with the map first (cached: it opens every
    request the same way, whatever the turns after it)."""
    msgs: list[dict[str, Any]] = []
    for t in list(ctx.history)[-MAX_TURNS:]:
        msgs += [{"role": "user", "content": t.user}, {"role": "assistant", "content": t.reply}]
    msgs.append({"role": "user", "content": user})
    found = site_map(ctx.gaz.site.id) if with_map else None
    if found is not None:
        data, grid = found
        image = {"type": "image", "cache_control": {"type": "ephemeral"},
                 "source": {"type": "base64", "media_type": "image/png", "data": data}}  # fmt: skip
        msgs[0] = {"role": "user", "content": [
            image, {"type": "text", "text": MAP_NOTE.format(grid=grid)},
            {"type": "text", "text": msgs[0]["content"]}]}  # fmt: skip
    return msgs


# --------------------------------------------------------------------------- Claude
class ClaudeTranslator:
    """Claude through the official SDK (``anthropic``, the ``txt2fly`` extra)."""

    def __init__(self, setting: str = DEFAULT, client: AsyncAnthropic | None = None) -> None:
        self.setting = SETTINGS[setting]
        self.name = f"claude:{self.setting.name}"
        self._client = client

    @property
    def client(self) -> AsyncAnthropic:
        if self._client is None:
            import os

            from anthropic import AsyncAnthropic

            # An organisation-wide key (not scoped to a workspace) must name one per request.
            ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
            headers = {"anthropic-workspace-id": ws} if ws else None
            self._client = AsyncAnthropic(timeout=TIMEOUT_S, max_retries=1, default_headers=headers)
        return self._client

    async def aclose(self) -> None:
        """Close the HTTP client (in the event loop that used it)."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def translate(self, ctx: Context, text: str) -> Translation:
        user = turn_text(ctx, text)
        return await self._ask(ctx, _messages(ctx, user, with_map=self.setting.map), user)

    async def revise(self, ctx: Context, text: str, first: Translation, review: str) -> Translation:
        """The model's second answer, having seen what was made of its first plan."""
        assert first.turn is not None
        user = turn_text(ctx, text)
        msgs = _messages(ctx, user, with_map=self.setting.map)
        msgs += [{"role": "assistant", "content": first.turn.reply},
                 {"role": "user", "content": review}]  # fmt: skip
        return await self._ask(ctx, msgs, user)

    async def _ask(
        self, ctx: Context, messages: list[dict[str, Any]], user: str, mend: bool = True
    ) -> Translation:
        import anthropic

        s = self.setting
        kwargs: dict[str, Any] = {
            "model": s.model,
            "max_tokens": s.max_tokens,
            "system": system_blocks(ctx.gaz, ctx.aircraft),
            "messages": messages,
            "output_format": site_schema(ctx.gaz),
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
                why = str(getattr(e, "message", "") or "")[:300]
                raise TranslatorError(f"the language model failed ({e.status_code}): {why}") from e
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
            flat: dict[str, Any] = {
                "kind": "refuse",
                "reason": "The language model declined this request.",
            }
        elif resp.stop_reason == "max_tokens" or resp.parsed_output is None:
            raise TranslatorError("the language model's reply was cut short")
        else:
            flat = resp.parsed_output.model_dump(mode="json", exclude_none=True)
        raw = nest(flat)
        try:
            reply = to_reply(raw)
        except ValueError as e:  # e.g. a patrol without its places
            where = " ".join(line.strip() for line in str(e).splitlines()[1:3])[:200]
            if mend:  # once: the model sees what is missing and writes the reply again
                again = [*messages, {"role": "assistant", "content": json.dumps(flat)},
                         {"role": "user", "content": MEND.format(where=where)}]  # fmt: skip
                t = await self._ask(ctx, again, user, mend=False)
                t.latency_s += latency
                t.usage = {k: usage.get(k, 0) + t.usage.get(k, 0) for k in {*usage, *t.usage}}
                return t
            raise TranslatorError(f"the reply did not make a valid plan: {where}") from e
        # The conversation keeps the reply as the model wrote it.
        turn = Turn(user, json.dumps(flat, separators=(",", ":")))
        return Translation(reply, raw, self.name, resp.model, latency, usage,
                           resp.stop_reason, turn)  # fmt: skip


MEND = (
    "That reply is not a valid plan ({where}). Write it again, complete: every step with the "
    "fields its kind needs (a patrol's places in `targets`, a target's `among` for nearest or "
    "furthest)."
)


# --------------------------------------------------------------------------- the review
REVIEW_ASK = (
    "Check this against what the operator asked. If it does that, reply with your plan "
    "unchanged: a wider circle or loop round the same place, a route round a no-fly area, or a "
    "height raised to clear the ground, or a circle moved but still passing over the place "
    "keeps the meaning. If not (a circle moved off the place, a different place, a failed "
    "check), reply once more: a corrected plan (other "
    "steps, places or numbers that do what they asked, a check's suggestion), or a question to "
    "the operator (`clarify`) saying what stands in the way, when no plan can do what they "
    "asked. The site's rules are fixed: do not plan round them."
)


def review_text(report: Report | None, error: str | None = None) -> str | None:
    """What was made of a plan, for the model to check against the request: when it cannot be
    built, or fails a hard check (what the model can mend). None when it passes: what the
    builder changed to keep it within the rules is in its notes, for the operator."""
    lines = ["<builder>"]
    if report is None or report.built.errors:
        why = [error] if report is None else report.built.errors
        lines += ["Your plan cannot be built:", *(f"- {e}" for e in why if e)]
    else:
        b = report.built
        failed = [c for c in report.checks if c.hard and not c.ok]
        if not failed:
            return None
        lines.append("Your plan, built for this aircraft in this wind, would fly:")
        for i, g in enumerate(b.steps):
            extra = [f"at {g.alt:g} m" if g.alt is not None else "",
                     f"radius {g.radius:g} m" if g.radius is not None else ""]  # fmt: skip
            more = ", ".join(x for x in extra if x)
            lines.append(f"{i + 1}. {g.label}" + (f" ({more})" if more else ""))
        if b.notes:
            lines += ["What the builder changed or chose, and why:", *(f"- {n}" for n in b.notes)]
        if failed:
            lines.append("It fails these checks, so it cannot fly as it is:")
            for c in failed:
                tip = f" Suggestion: {c.suggestion}." if c.suggestion else ""
                lines.append(f"- {c.rule} {c.title}: {c.reason}{tip}")
    return "\n".join([*lines, "</builder>", REVIEW_ASK])


async def translate_checked(
    translator: Translator,
    ctx: Context,
    text: str,
    check: Callable[[Plan], Awaitable[Report]],
) -> tuple[Translation, Report | None, str | None]:
    """Translate, build and check: (the translation, its report, why it cannot be built). When
    the builder could not build the plan, or it fails a hard check, the model sees what was
    made of it and answers once more (a translator that can: the API's). The conversation
    keeps the final answer."""
    t = await translator.translate(ctx, text)
    report, error = await check_reply(t.reply, check)
    review = review_text(report, error) if isinstance(t.reply, Plan) else None
    revise = getattr(translator, "revise", None)
    if review is None or revise is None:
        return t, report, error
    t2: Translation = await revise(ctx, text, t, review)
    t2.review, t2.latency_s = review, t.latency_s + t2.latency_s
    t2.usage = {k: t.usage.get(k, 0) + t2.usage.get(k, 0) for k in {*t.usage, *t2.usage}}
    if t2.reply == t.reply:  # kept: the first report stands
        return t2, report, error
    t2.first = t.raw
    report, error = await check_reply(t2.reply, check)
    return t2, report, error


async def check_reply(
    reply: Reply, check: Callable[[Plan], Awaitable[Report]]
) -> tuple[Report | None, str | None]:
    if not isinstance(reply, Plan):
        return None, None
    try:
        return await check(reply), None
    except ValueError as e:  # a plan the builder cannot take at all (K1)
        return None, str(e)


def in_thread(check: Callable[[Plan], Report]) -> Callable[[Plan], Awaitable[Report]]:
    """A build-and-check function, run off the event loop."""

    async def run(plan: Plan) -> Report:
        return await asyncio.to_thread(check, plan)

    return run


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
