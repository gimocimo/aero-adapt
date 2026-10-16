"""Live server (M2 spec §9): one ``LiveSession`` per WebSocket connection, paced to wall-clock
time × a speed factor, streaming the replay viz columns and events as msgpack.

Server → client: ``hello`` (manifest, the scenario and airframe catalogue, the current
airframe's glTF), ``frames`` (columns of new rows), ``events``, ``status`` (time, speed,
paused, done, last error, the session's faults, the wind set).
Client → server: ``inject_fault{spec}`` (a fault spec without trigger), ``clear_fault{index}``,
``set_wind{speed_mps, from_deg}``, ``set_speed{x}``, ``pause``, ``resume``,
``reset{scenario, airframe}``.

Airframes: the scenario's own, and the sample fleet (``configs/fleets/sample``) with the
stack configurations V20 tuned for them (``stack_cfgs.jsonl``).

It binds to localhost only. Every session is recorded under ``runs/live/<scenario>-<n>``.
"""

from __future__ import annotations

import asyncio
import contextlib
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
from aeroadapt.airframes.fleet import load_fleet
from aeroadapt.live.session import CHUNK_S, LiveSession
from aeroadapt.scenarios.schema import SCENARIO_DIR, load_scenario
from aeroadapt.stack.config import StackCfg

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
            "model": cat.models.get(self.airframe_id or ""),
        })  # fmt: skip

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
            if session.done:
                await self.status()
            next_wall += CHUNK_S / self.speed
            await asyncio.sleep(max(0.0, next_wall - time.perf_counter()))

    def record(self) -> Path | None:
        if not self.session.chunks:
            return None
        out = RUNS / f"{self.scenario_id}-{next(_session_ids):03d}"
        return self.session.record(out)


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


def _warm_up(catalogue: Catalogue, scenario_id: str) -> None:
    """Compile the closed loop before the first browser connects: once for the scenario's
    airframe and once for tabulated (fleet) aerodynamics."""
    catalogue.session(scenario_id, None).step()
    for af in itertools.islice(catalogue.airframes, 1):
        catalogue.session(scenario_id, af).step()


async def run_server(scenario_id: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    catalogue = Catalogue.load()
    print(f"compiling the closed loop ({len(catalogue.airframes)} fleet airframes)…", flush=True)
    await asyncio.to_thread(_warm_up, catalogue, scenario_id)

    async def handler(ws: ServerConnection) -> None:
        await _handler(ws, catalogue, scenario_id)

    async with serve(handler, host, port, max_size=None) as server:
        print(
            f"live server on ws://{host}:{port} (scenario {scenario_id}); Ctrl-C to stop",
            flush=True,
        )
        await server.serve_forever()
