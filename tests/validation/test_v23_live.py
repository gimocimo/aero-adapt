"""V23 (M2 spec §10): live mode. Faults take effect at the next control step; a session's
command log re-flies offline to an identical recording; the server speaks the protocol."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("websockets")
pytest.importorskip("msgpack")

import msgpack
from websockets.asyncio.client import connect

from aeroadapt.live.session import LiveSession, replay_commands
from aeroadapt.scenarios.schema import load_scenario
from aeroadapt.telemetry.recorder import FILES
from aeroadapt.telemetry.schema import FAULT_BITS


def _fly(session: LiveSession, seconds: float) -> None:
    for _ in range(round(seconds / 0.1)):
        session.step()


def test_injection_takes_effect_at_the_next_control_step() -> None:
    s = LiveSession(load_scenario("m1-nominal"))
    _fly(s, 2.0)
    t_inject = s.time_s
    assert s.inject({"type": "propulsion_fault", "kind": "partial_loss", "magnitude": 0.5}) is None
    assert s.inject({"type": "motor_cut"}) is None
    assert "only one motor_cut" in (s.inject({"type": "motor_cut"}) or "")
    _fly(s, 1.0)
    cols = s.all_columns()
    t = cols["time_us"] * 1e-6
    bits = cols["labels.fault_bits"]
    on = (bits & FAULT_BITS["motor_cut"]) > 0
    first = t[np.argmax(on)]
    assert first == pytest.approx(t_inject + 0.01, abs=1e-6)  # the next 100 Hz step
    assert not on[t < first].any()
    assert (bits[on] & FAULT_BITS["propulsion:partial_loss"]).all()


def test_command_log_refly_is_byte_identical(tmp_path: Path) -> None:
    s = LiveSession(load_scenario("m1-nominal"))
    _fly(s, 3.0)
    assert (
        s.inject({"type": "sensor_fault", "kind": "gyro_bias", "magnitude": 0.02, "axis": 2})
        is None
    )
    _fly(s, 2.0)
    a = s.record(tmp_path / "live")
    commands = json.loads((a / "manifest.json").read_text())["live_commands"]
    b = replay_commands(load_scenario("m1-nominal"), commands)
    while b.time_s < s.time_s - 1e-6:
        b.step()
    b_dir = b.record(tmp_path / "offline")
    for name in FILES:
        assert (a / name).read_bytes() == (b_dir / name).read_bytes(), name


def test_server_protocol() -> None:
    """The real CLI server in a subprocess: hello, then frames, and an injected fault."""
    port = 8798
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "aeroadapt.cli", "serve", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:  # wait until the closed loop is compiled and listening
            if "live server on" in line:
                break
        types = asyncio.run(_exchange(port))
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    assert types[0] == "hello" and "frames" in types and types[-1] == "status-after-inject"


async def _exchange(port: int) -> list[str]:
    types: list[str] = []
    async with connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
        for _ in range(12):
            m = msgpack.unpackb(await asyncio.wait_for(ws.recv(), 60), raw=False)
            types.append(m["type"])
            if m["type"] == "frames":
                t = np.frombuffer(m["cols"]["t"]["data"], np.float32)
                assert len(t) == m["n"] and np.all(np.diff(t) > 0)
        cmd = msgpack.packb({"cmd": "inject_fault", "spec": {"type": "motor_cut"}})
        assert isinstance(cmd, bytes)
        await ws.send(cmd)
        while True:
            m = msgpack.unpackb(await asyncio.wait_for(ws.recv(), 60), raw=False)
            if m["type"] == "status":
                assert m["error"] is None
                types.append("status-after-inject")
                return types
