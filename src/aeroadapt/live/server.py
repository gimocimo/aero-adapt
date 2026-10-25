"""Live server (M2 spec §9): one ``LiveSession`` per WebSocket connection, paced to wall-clock
time × a speed factor, streaming the replay viz columns and events as msgpack.

Server → client: ``hello`` (manifest, the scenario and airframe catalogue, the current
airframe's glTF), ``frames`` (columns of new rows), ``events``, ``status`` (time, speed,
paused, done, last error, the session's faults, the wind set).
Client → server: ``inject_fault{spec}`` (a fault spec without trigger), ``clear_fault{index}``,
``set_wind{speed_mps, from_deg}``, ``set_speed{x}``, ``pause``, ``resume``,
``reset{scenario, airframe}``.

A world site's ``hello`` also carries its world (docs/07 §8): terrain, fields, woods, hedges,
trees, buildings, for the stylised view.

Txt2Fly (docs/06 §10): ``hello`` carries the site's operating area (geofence, no-fly areas,
places); ``propose_plan{plan}`` answers ``plan_proposal`` (the plan built and checked from the
aircraft's state now); ``confirm_plan{id, acknowledged}`` flies it from the next chunk
(``plan_active`` follows); ``cancel_plan{id}``. ``ask{text}`` sends plain words to the
translator (docs/06 §9) and answers ``ask_reply`` (kind plan, clarify, refuse or error); a plan
also comes as a ``plan_proposal``, exactly as a typed one. The conversation stays with the
connection, and is saved beside the recording (``conversation.json``); the session log itself
records only the plans flown, so a replay never calls the model.

Airframes: the scenario's own, and the sample fleet (``configs/fleets/sample``) with the
stack configurations V20 tuned for them (``stack_cfgs.jsonl``).

It binds to localhost only. Every session is recorded under ``runs/live/<scenario>-<n>``.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import itertools
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from aeroadapt.airframes import Airframe
from aeroadapt.airframes.fleet import fleet_path, load_fleet
from aeroadapt.live.session import CHUNK_S, LiveSession
from aeroadapt.scenarios.schema import SCENARIO_DIR, load_scenario
from aeroadapt.scenarios.site import load_site
from aeroadapt.stack.config import StackCfg
from aeroadapt.txt2fly import translate as tr
from aeroadapt.txt2fly.gazetteer import load_gazetteer, site_layers
from aeroadapt.txt2fly.plan import Plan

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "runs" / "live"
SAMPLE_FLEET = REPO / "configs" / "fleets" / "sample"
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
_session_ids = itertools.count(1)


@dataclass
class Catalogue:
    """What a client may fly: scenarios, and fleet airframes with their tuned stacks."""

    scenarios: list[str]
    fleet: str | None = None  # the bundle the airframes come from (repository-relative)
    airframes: dict[str, Airframe] = field(default_factory=dict)
    stacks: dict[str, StackCfg] = field(default_factory=dict)
    models: dict[str, bytes] = field(default_factory=dict)
    translator: tr.Translator | None = None
    translator_note: str = ""

    @classmethod
    def load(cls, fleet: Path = SAMPLE_FLEET) -> Catalogue:
        cat = cls(sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml")), str(fleet.relative_to(REPO)))
        cfgs = fleet / "stack_cfgs.jsonl"
        if not cfgs.exists():
            return cat
        stacks = {}
        for line in cfgs.read_text().splitlines():
            row = json.loads(line)
            stacks[row["airframe"]] = StackCfg.model_validate(row["stack"])
        for af in load_fleet(fleet):
            if af.id in stacks:
                cat.airframes[af.id] = af
                cat.stacks[af.id] = stacks[af.id]
                glb = fleet / "gltf" / f"{af.id}.glb"
                if glb.exists():
                    cat.models[af.id] = glb.read_bytes()
        return cat

    def listing(self) -> list[dict[str, Any]]:
        return [
            {"id": k, "v_cruise_mps": self.stacks[k].tecs.airspeed_cruise_mps}
            for k in self.airframes
        ]

    def session(self, scenario_id: str, airframe_id: str | None) -> LiveSession:
        sc = load_scenario(scenario_id)
        if airframe_id is None:
            return LiveSession(sc)
        return LiveSession(
            sc, self.airframes[airframe_id], self.stacks[airframe_id], fleet=self.fleet
        )


@functools.lru_cache(maxsize=8)
def _fleet_model(fleet: str | None, airframe_id: str) -> bytes | None:
    """The glTF model of a scenario's own fleet airframe (london-city's city trainer)."""
    if fleet is None:
        return None
    glb = fleet_path(fleet) / "gltf" / f"{airframe_id}.glb"
    return glb.read_bytes() if glb.exists() else None


@functools.lru_cache(maxsize=4)
def _world(world_id: str | None) -> dict[str, Any] | None:
    """A world site's world for the browser (docs/07 §8), built once per process."""
    if world_id is None:
        return None
    from aeroadapt.world import load_world
    from aeroadapt.world.export import viewer_payload

    return viewer_payload(load_world(world_id))


def _pack(msg: dict[str, Any]) -> bytes:
    return msgpack.packb(msg, use_bin_type=True)  # type: ignore[no-any-return]


def _frames(viz: dict[str, np.ndarray]) -> dict[str, Any]:
    cols = {
        k: {"dtype": "i4" if v.dtype == np.int32 else "f4", "data": v.tobytes()}
        for k, v in viz.items()
    }
    return {"type": "frames", "n": len(viz["t"]), "cols": cols}


class Connection:
    """Simulation work runs in a worker thread, so the event loop keeps answering pings."""

    def __init__(
        self,
        ws: ServerConnection,
        catalogue: Catalogue,
        scenario_id: str,
        airframe_id: str | None,
        session: LiveSession,
    ) -> None:
        self.ws = ws
        self.catalogue = catalogue
        self.scenario_id = scenario_id
        self.airframe_id = airframe_id
        self.speed = 1.0
        self.paused = False
        self.error: str | None = None
        self.session = session
        self.turns: list[tr.Turn] = []  # what the translator is shown of the conversation
        self.conversation: list[dict[str, Any]] = []  # the record, saved with the session
        self._asks: set[asyncio.Task[None]] = set()

    async def send(self, msg: dict[str, Any]) -> None:
        await self.ws.send(_pack(msg))

    async def status(self) -> None:
        s = self.session
        await self.send({"type": "status", "time_s": s.time_s, "speed": self.speed,
                         "paused": self.paused, "done": s.done, "error": self.error,
                         "faults": s.fault_list(), "wind": s.wind})  # fmt: skip

    async def hello(self) -> None:
        cat = self.catalogue
        await self.send({
            "type": "hello",
            "manifest": self.session.manifest(),
            "scenario": self.scenario_id,
            "scenarios": cat.scenarios,
            "airframe": self.airframe_id,
            "airframes": cat.listing(),
            "model": cat.models.get(self.airframe_id or "") or _fleet_model(
                self.session.resolved.scenario.fleet, self.session.resolved.airframe.id
            ),
            "site": site_layers(self.session.resolved.site),
            "world": _world(self.session.resolved.site.world),
            "plan": self.session.active_plan,
            "translator": {"name": cat.translator.name if cat.translator else None,
                           "note": cat.translator_note},
        })  # fmt: skip

    async def ask(self, text: str) -> None:
        """Plain words → the translator → a proposal, a question or a refusal. Runs beside the
        flight, which carries on whatever happens here."""
        cat, session = self.catalogue, self.session
        reply: dict[str, Any] = {"type": "ask_reply", "text": text}
        if cat.translator is None or session.resolved.site.operations is None:
            reply.update(kind="error", message="no translator here: type a plan instead")
            await self.send(reply)
            return
        ctx = tr.Context(load_gazetteer(session.resolved.site), session.aircraft(),
                         session.active_plan, tuple(self.turns), session.time_s)  # fmt: skip
        try:
            t, report, build_error = await tr.translate_checked(
                cat.translator, ctx, text, tr.in_thread(session.check_plan)
            )
        except tr.TranslatorError as e:
            reply.update(kind="error", message=str(e))
            self.conversation.append({"time_s": session.time_s, "text": text, "error": str(e)})
            await self.send(reply)
            return
        if session is not self.session:  # the scenario changed while the model was thinking
            return
        if t.turn is not None:
            self.turns.append(t.turn)
        self.conversation.append(
            {
                "time_s": session.time_s,
                "text": text,
                "reply": t.raw,
                "translator": t.translator,
                "model": t.model,
                "latency_s": round(t.latency_s, 3),
                "usage": t.usage,
                **({"review": t.review} if t.review else {}),
                **({"first_reply": t.first} if t.first else {}),
            }
        )
        reply.update(translator=t.translator, latency_s=round(t.latency_s, 3))
        if isinstance(t.reply, tr.Clarify):
            reply.update(kind="clarify", question=t.reply.question, options=list(t.reply.options))
        elif isinstance(t.reply, tr.Refuse):
            reply.update(kind="refuse", reason=t.reply.reason)
        else:
            assert isinstance(t.reply, Plan)
            if report is None:
                err, prop = (build_error or "").splitlines()[0] if build_error else None, None
            else:
                err, prop = session.propose(t.reply, report)
            if prop is None:
                reply.update(kind="error", message=err or "the plan could not be built")
            else:
                reply.update(kind="plan", proposal=prop["id"], summary=t.reply.summary)
                await self.send({"type": "plan_proposal", "proposal": prop})
        await self.send(reply)

    async def receive(self) -> None:
        async for raw in self.ws:
            msg = msgpack.unpackb(raw, raw=False) if isinstance(raw, bytes) else {}
            cmd = msg.get("cmd")
            self.error = None
            if cmd == "inject_fault":
                self.error = self.session.inject(msg.get("spec", {}))
            elif cmd == "clear_fault":
                self.error = self.session.clear(int(msg.get("index", -1)))
            elif cmd == "set_wind":
                self.error = self.session.set_wind(
                    float(msg.get("speed_mps", 0.0)), float(msg.get("from_deg", 0.0))
                )
            elif cmd == "set_speed" and float(msg.get("x", 1.0)) in SPEEDS:
                self.speed = float(msg["x"])
            elif cmd == "pause":
                self.paused = True
            elif cmd == "resume":
                self.paused = False
            elif cmd == "propose_plan":
                err, prop = await asyncio.to_thread(self.session.propose, msg.get("plan") or {})
                self.error = err
                if prop is not None:
                    await self.send({"type": "plan_proposal", "proposal": prop})
            elif cmd == "confirm_plan":
                ack = [str(x) for x in msg.get("acknowledged") or []]
                self.error = await asyncio.to_thread(self.session.confirm, str(msg.get("id")), ack)
            elif cmd == "cancel_plan":
                self.session.cancel(str(msg.get("id")))
            elif cmd == "ask" and str(msg.get("text") or "").strip():
                task = asyncio.create_task(self.ask(str(msg["text"]).strip()[:500]))
                self._asks.add(task)
                task.add_done_callback(self._asks.discard)
            elif cmd == "reset":
                scenario = str(msg.get("scenario") or self.scenario_id)
                airframe = msg.get("airframe", self.airframe_id) or None
                if scenario not in self.catalogue.scenarios:
                    self.error = f"unknown scenario {scenario}"
                elif airframe is not None and airframe not in self.catalogue.airframes:
                    self.error = f"unknown airframe {airframe}"
                else:
                    self.record()
                    self.scenario_id, self.airframe_id = scenario, airframe
                    self.turns, self.conversation = [], []
                    self.session = await asyncio.to_thread(
                        self.catalogue.session, scenario, airframe
                    )
                    await self.hello()
            await self.status()

    async def run(self) -> None:
        next_wall = time.perf_counter()
        while True:
            if self.paused or self.session.done:
                await asyncio.sleep(0.05)
                next_wall = time.perf_counter()
                continue
            session = self.session
            cols = await asyncio.to_thread(session.step)
            await self.send(_frames(session.viz(cols)))
            events = await asyncio.to_thread(session.new_events)
            if events:
                await self.send({"type": "events", "events": events})
            if session.plan_changed:
                session.plan_changed = False
                await self.send({"type": "plan_active", "plan": session.active_plan})
            if session.done:
                await self.status()
            next_wall += CHUNK_S / self.speed
            await asyncio.sleep(max(0.0, next_wall - time.perf_counter()))

    def record(self) -> Path | None:
        if not self.session.chunks:
            return None
        out = RUNS / f"{self.scenario_id}-{next(_session_ids):03d}"
        path = self.session.record(out)
        if self.conversation:
            (path / "conversation.json").write_text(json.dumps(self.conversation, indent=1) + "\n")
        return path


async def _handler(ws: ServerConnection, catalogue: Catalogue, scenario_id: str) -> None:
    session = await asyncio.to_thread(catalogue.session, scenario_id, None)
    conn = Connection(ws, catalogue, scenario_id, None, session)
    try:
        await conn.hello()
        await conn.status()
    except ConnectionClosed:  # the client left before the session started (a page reload)
        return
    tasks = [asyncio.create_task(conn.receive()), asyncio.create_task(conn.run())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            exc = None if t.cancelled() else t.exception()
            if isinstance(exc, Exception) and not isinstance(exc, ConnectionClosed):
                traceback.print_exception(exc)
    except ConnectionClosed:
        pass
    finally:
        for t in tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError, ConnectionClosed):
                await t
        out = conn.record()
        if out is not None:
            print(f"session recorded → {out}", flush=True)


def _warm_up(catalogue: Catalogue, scenario_id: str, fleet: bool = True) -> None:
    """Compile the closed loop before the first browser connects: once for the scenario's
    airframe and once for tabulated (fleet) aerodynamics; and the plan preview."""
    session = catalogue.session(scenario_id, None)
    session.step()
    if session.resolved.site.operations is not None:
        session.propose({"steps": [{"do": "return_home"}]})  # the short preview
        session.propose({"steps": [{"do": "hold"}]})  # … and the long one
    for af in itertools.islice(catalogue.airframes, 1 if fleet else 0):
        catalogue.session(scenario_id, af).step()


def _other_grounds(catalogue: Catalogue, scenario_id: str) -> list[str]:
    """One scenario per ground grid (flat, or each world) other than this scenario's."""

    def ground(s: str) -> str | None:
        return load_site(load_scenario(s).site).world

    seen, out = {ground(scenario_id)}, []
    for s in catalogue.scenarios:
        g = ground(s)
        if g not in seen:
            seen.add(g)
            out.append(s)
    return out


def make_translator(choice: str) -> tuple[tr.Translator | None, str]:
    """The translator ``serve --translator`` asks for, and a note for the operator. ``auto``:
    Claude when the SDK and a key are there, else the stand-in."""
    if choice == "none":
        return None, "no translator: type plans instead"
    if choice == "stand-in" or (choice == "auto" and not tr.available()):
        note = "" if choice == "stand-in" else "no API key: "
        return tr.StandIn(), note + "a stand-in reads a few fixed phrasings"
    setting = tr.DEFAULT if choice in ("auto", "claude") else choice
    t = tr.ClaudeTranslator(setting)
    return t, f"Claude ({t.setting.model}{', adaptive thinking' if t.setting.thinking else ''})"


async def run_server(
    scenario_id: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    warm_others: bool = True,
    translator: str = "auto",
) -> None:
    catalogue = Catalogue.load()
    catalogue.translator, catalogue.translator_note = make_translator(translator)
    print(f"translator: {catalogue.translator_note}", flush=True)
    print(f"compiling the closed loop ({len(catalogue.airframes)} fleet airframes)…", flush=True)
    await asyncio.to_thread(_warm_up, catalogue, scenario_id)

    async def handler(ws: ServerConnection) -> None:
        await _handler(ws, catalogue, scenario_id)

    async with serve(handler, host, port, max_size=None) as server:
        print(
            f"live server on ws://{host}:{port} (scenario {scenario_id}); Ctrl-C to stop",
            flush=True,
        )
        # Sites with other ground grids (flat ones, worlds) compile their own closed loop:
        # one scenario of each, in the background, so switching to it later is quick.
        others = _other_grounds(catalogue, scenario_id) if warm_others else []
        warm = asyncio.gather(*(asyncio.to_thread(_warm_up, catalogue, s, False) for s in others))
        await server.serve_forever()
        warm.cancel()
