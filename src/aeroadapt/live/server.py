"""Live server (M2 spec §9): one ``LiveSession`` per WebSocket connection, paced to wall-clock
time × a speed factor, streaming the replay viz columns and events as msgpack.

Server → client: ``hello`` (manifest), ``frames`` (columns of new rows), ``events``,
``status`` (time, speed, paused, done, last error).
Client → server: ``inject_fault`` (a fault spec without trigger), ``set_speed``, ``pause``,
``resume``, ``reset``.

It binds to localhost only. Every session is recorded under ``runs/live/<scenario>-<n>``.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
import traceback
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from aeroadapt.live.session import CHUNK_S, LiveSession
from aeroadapt.scenarios.schema import load_scenario

RUNS = Path(__file__).resolve().parents[3] / "runs" / "live"
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
_session_ids = itertools.count(1)


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

    def __init__(self, ws: ServerConnection, scenario_id: str, session: LiveSession) -> None:
        self.ws = ws
        self.scenario_id = scenario_id
        self.speed = 1.0
        self.paused = False
        self.error: str | None = None
        self.session = session

    async def send(self, msg: dict[str, Any]) -> None:
        await self.ws.send(_pack(msg))

    async def status(self) -> None:
        s = self.session
        await self.send({"type": "status", "time_s": s.time_s, "speed": self.speed,
                         "paused": self.paused, "done": s.done, "error": self.error})  # fmt: skip

    async def hello(self) -> None:
        await self.send({"type": "hello", "manifest": self.session.manifest()})

    async def receive(self) -> None:
        async for raw in self.ws:
            msg = msgpack.unpackb(raw, raw=False) if isinstance(raw, bytes) else {}
            cmd = msg.get("cmd")
            self.error = None
            if cmd == "inject_fault":
                self.error = self.session.inject(msg.get("spec", {}))
            elif cmd == "set_speed" and float(msg.get("x", 1.0)) in SPEEDS:
                self.speed = float(msg["x"])
            elif cmd == "pause":
                self.paused = True
            elif cmd == "resume":
                self.paused = False
            elif cmd == "reset":
                self.record()
                self.scenario_id = str(msg.get("scenario", self.scenario_id))
                self.session = await _new_session(self.scenario_id)
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


async def _new_session(scenario_id: str) -> LiveSession:
    return await asyncio.to_thread(LiveSession, load_scenario(scenario_id))


async def _handler(ws: ServerConnection, scenario_id: str) -> None:
    conn = Connection(ws, scenario_id, await _new_session(scenario_id))
    await conn.hello()
    await conn.status()
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


def _warm_up(scenario_id: str) -> None:
    """Compile the closed loop once, before the first browser connects."""
    LiveSession(load_scenario(scenario_id)).step()


async def run_server(scenario_id: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    print("compiling the closed loop…", flush=True)
    await asyncio.to_thread(_warm_up, scenario_id)
    async with serve(lambda ws: _handler(ws, scenario_id), host, port, max_size=None) as server:
        print(
            f"live server on ws://{host}:{port} (scenario {scenario_id}); Ctrl-C to stop",
            flush=True,
        )
        await server.serve_forever()
