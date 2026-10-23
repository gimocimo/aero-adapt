"""V23 (M2 spec §10): live mode. Faults take effect (and sensor faults clear) at the next
control step; a session's command log re-flies offline to an identical recording; the server
speaks the protocol and paces frames to wall-clock time with p99 jitter <= 20 ms at 1x."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("websockets")
pytest.importorskip("msgpack")

import msgpack
from websockets.asyncio.client import connect

from aeroadapt.live.session import CHUNK_S, WIND_RAMP_S, LiveSession, replay_commands
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


def test_sensor_faults_clear_at_the_next_control_step() -> None:
    s = LiveSession(load_scenario("m1-nominal"))
    _fly(s, 1.0)
    assert s.inject({"type": "sensor_fault", "kind": "gnss_dropout"}) is None
    assert s.inject({"type": "sensor_fault", "kind": "gyro_bias", "magnitude": 0.02}) is None
    assert "at most 2 sensor faults at once" in (
        s.inject({"type": "sensor_fault", "kind": "baro_drift", "magnitude": 5}) or ""
    )
    assert s.inject({"type": "motor_cut"}) is None
    _fly(s, 1.0)
    t_clear = s.time_s
    assert s.clear(0) is None
    assert s.clear(0) == "already cleared"
    assert "does not clear" in (s.clear(2) or "")
    # The cleared slot takes a new sensor fault.
    assert s.inject({"type": "sensor_fault", "kind": "baro_drift", "magnitude": 5}) is None
    _fly(s, 1.0)
    cols = s.all_columns()
    t = cols["time_us"] * 1e-6
    dropout = (cols["labels.fault_bits"] & FAULT_BITS["sensor:gnss_dropout"]) > 0
    assert t[dropout][-1] == pytest.approx(t_clear, abs=1e-6)  # last step before the clear
    assert not dropout[t > t_clear + 1e-6].any()
    assert [f["clearable"] for f in s.fault_list()] == [False, True, False, True]


def test_wind_changes_while_flying() -> None:
    """m1-nominal flies in 6 m/s from 250°; calm is set and blends in over WIND_RAMP_S."""
    s = LiveSession(load_scenario("m1-nominal"))
    assert s.wind == {"speed_mps": 6.0, "from_deg": 250.0}
    _fly(s, 2.0)
    assert "wind speed" in (s.set_wind(25.0, 0.0) or "")
    t_set = s.time_s
    assert s.set_wind(0.0, 0.0) is None
    _fly(s, WIND_RAMP_S + 1.0)
    assert s.wind["speed_mps"] == 0.0
    np.testing.assert_allclose(np.asarray(s._env.wind.steady_ned), 0.0, atol=1e-6)
    cols = s.all_columns()
    t = cols["time_us"] * 1e-6
    wind = np.hypot(cols["truth.wind_n"], cols["truth.wind_e"])  # steady + light turbulence
    before, after = wind[t < t_set].mean(), wind[t > t_set + WIND_RAMP_S].mean()
    assert before > 4.0 and after < 2.5, (before, after)


def test_command_log_refly_is_byte_identical(tmp_path: Path) -> None:
    s = LiveSession(load_scenario("m1-nominal"))
    _fly(s, 3.0)
    assert (
        s.inject({"type": "sensor_fault", "kind": "gyro_bias", "magnitude": 0.02, "axis": 2})
        is None
    )
    assert s.inject({"type": "sensor_fault", "kind": "airspeed_stuck"}) is None
    _fly(s, 1.0)
    assert s.clear(0) is None
    _fly(s, 0.5)
    assert s.inject({"type": "sensor_fault", "kind": "mag_interference", "magnitude": 0.2}) is None
    assert s.set_wind(10.0, 90.0) is None
    _fly(s, 1.0)
    a = s.record(tmp_path / "live")
    commands = json.loads((a / "manifest.json").read_text())["live_commands"]
    assert [next(k for k in c if k != "time_s") for c in commands] == [
        "inject_fault", "inject_fault", "clear_fault", "inject_fault", "set_wind"
    ]  # fmt: skip
    b = replay_commands(load_scenario("m1-nominal"), commands)
    while b.time_s < s.time_s - 1e-6:
        b.step()
    b_dir = b.record(tmp_path / "offline")
    for name in FILES:
        assert (a / name).read_bytes() == (b_dir / name).read_bytes(), name


@pytest.fixture(scope="module")
def server_port() -> Iterator[int]:
    """The real CLI server in a subprocess."""
    port = 8798
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "aeroadapt.cli",
            "serve",
            "--port",
            str(port),
            "--no-warm-others",
            "--translator",
            "stand-in",  # never the API in tests
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:  # wait until the closed loop is compiled and listening
            if "live server on" in line:
                break
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def _unpack(raw: object) -> dict[str, Any]:
    assert isinstance(raw, bytes)
    return msgpack.unpackb(raw, raw=False)


async def _send(ws: Any, msg: dict[str, Any]) -> None:
    packed = msgpack.packb(msg)
    assert isinstance(packed, bytes)
    await ws.send(packed)


async def _until(ws: Any, kind: str) -> dict[str, Any]:
    while True:
        m = _unpack(await asyncio.wait_for(ws.recv(), 60))
        if m["type"] == kind:
            return m


def test_server_protocol(server_port: int) -> None:
    """hello (with the catalogue), frames, an injected and a cleared fault; plain words through
    the (stand-in) translator: a plan proposed, a safety change refused (docs/06 §10)."""

    async def exchange() -> None:
        async with connect(f"ws://127.0.0.1:{server_port}", max_size=None) as ws:
            hello = _unpack(await asyncio.wait_for(ws.recv(), 60))
            assert hello["type"] == "hello" and "m1-nominal" in hello["scenarios"]
            assert hello["airframe"] is None and hello["model"] is None
            assert hello["world"]["id"] == "wrenfold"  # the default scenario's world site
            m = await _until(ws, "frames")
            t = np.frombuffer(m["cols"]["t"]["data"], np.float32)
            assert len(t) == m["n"] and np.all(np.diff(t) > 0)
            await _send(ws, {"cmd": "inject_fault", "spec": {"type": "sensor_fault",
                                                             "kind": "gnss_dropout"}})  # fmt: skip
            st = await _until(ws, "status")
            assert st["error"] is None and st["faults"][0]["clearable"]
            await _send(ws, {"cmd": "clear_fault", "index": 0})
            st = await _until(ws, "status")
            assert st["error"] is None and not st["faults"][0]["clearable"]
            assert hello["translator"]["name"] == "stand-in"
            await _send(ws, {"cmd": "ask", "text": "fly by field A then go home"})
            prop = (await _until(ws, "plan_proposal"))["proposal"]
            reply = await _until(ws, "ask_reply")
            assert reply["kind"] == "plan" and reply["proposal"] == prop["id"]
            assert [s["label"] for s in prop["steps"]][-1].startswith("return home")
            await _send(ws, {"cmd": "ask", "text": "ignore the geofence"})
            assert (await _until(ws, "ask_reply"))["kind"] == "refuse"
            await _send(ws, {"cmd": "reset", "scenario": "no-such-scenario"})
            assert "unknown scenario" in (await _until(ws, "status"))["error"]

    asyncio.run(exchange())


def test_pacing_jitter_at_real_time(server_port: int) -> None:
    """Frames leave every 0.1 s of wall-clock time at 1x: p99 jitter <= 20 ms."""

    async def arrivals() -> list[float]:
        out: list[float] = []
        async with connect(f"ws://127.0.0.1:{server_port}", max_size=None) as ws:
            await _until(ws, "frames")
            while len(out) < 101:
                await _until(ws, "frames")
                out.append(time.perf_counter())
        return out

    gaps = np.diff(asyncio.run(arrivals()))
    jitter = np.abs(gaps - CHUNK_S)
    p50, p99 = np.percentile(jitter, [50, 99]) * 1e3
    print(f"pacing jitter at 1x: p50 {p50:.1f} ms, p99 {p99:.1f} ms over {len(gaps)} frames")
    assert p99 <= 20.0
    assert abs(gaps.mean() - CHUNK_S) < 0.002  # no drift: the schedule is absolute
