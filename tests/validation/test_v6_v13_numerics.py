"""V6 (integrator convergence), V7 (float32 vs float64) and V13 (determinism), spec §12."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aeroadapt.airframes import AirframeParams
from aeroadapt.core import quaternion as Q
from aeroadapt.scenarios.runner import record, run_scenario
from aeroadapt.scenarios.schema import Scenario, load_scenario
from aeroadapt.sim.aerojax import BM_PARITY, controls_from_vector, rollout
from aeroadapt.sim.aerojax.trim import trim

REPO = Path(__file__).resolve().parents[2]
T_END = 30.0
DTS = (0.01, 0.005, 0.0025)
DT_REF = 0.0005


def _doublet(params: AirframeParams, dt: float) -> np.ndarray:
    """Open-loop elevator doublet from cruise trim; returns the pitch angle every 10 ms."""
    tr = trim(params, BM_PARITY, 25.0, h=100.0)
    n = round(T_END / dt)
    t = dt * np.arange(n)
    u = np.tile(tr.controls, (n, 1))
    u[(t >= 2.0) & (t < 3.0), 1] += np.radians(3.0)
    u[(t >= 3.0) & (t < 4.0), 1] -= np.radians(3.0)
    traj = jax.jit(rollout, static_argnums=5)(
        params, BM_PARITY, tr.state, controls_from_vector(jnp.asarray(u)), jnp.zeros((n, 3)), dt
    )
    every = round(0.01 / dt)
    _, pitch, _ = Q.to_euler(traj.quat[::every])
    return np.asarray(pitch)


@pytest.mark.full
def test_v6_rk4_converges_at_fourth_order(params: AirframeParams) -> None:
    ref = _doublet(params, DT_REF)
    errs = [float(np.max(np.abs(_doublet(params, dt) - ref))) for dt in DTS]
    slope = np.polyfit(np.log(DTS), np.log(errs), 1)[0]
    assert 3.5 <= slope <= 4.5, (errs, slope)
    assert errs[1] <= 1e-3  # the chosen physics step, 5 ms


def _flight_in_subprocess(x64: bool, out: Path) -> dict[str, np.ndarray]:
    code = (
        "import sys, numpy as np\n"
        "from aeroadapt.scenarios.runner import run_scenario\n"
        "from aeroadapt.scenarios.schema import load_scenario\n"
        "s = load_scenario('m1-nominal').model_copy(update={'duration_s': 60.0})\n"
        "c = run_scenario(s).cols\n"
        "np.savez(sys.argv[1], **{k.replace('.', '__'): v for k, v in c.items()})\n"
    )
    env = {**os.environ, "JAX_ENABLE_X64": "1" if x64 else "0"}
    subprocess.run([sys.executable, "-c", code, str(out)], check=True, env=env, cwd=REPO)
    with np.load(out) as z:
        return {k.replace("__", "."): z[k] for k in z.files}


def precision_divergence(tmp: Path) -> tuple[float, float]:
    """(max position divergence m, max attitude divergence deg) of a 60 s closed-loop flight
    flown in float32 vs float64, same scenario and seed."""
    f32 = _flight_in_subprocess(False, tmp / "f32.npz")
    f64 = _flight_in_subprocess(True, tmp / "f64.npz")
    n = min(len(f32["time_us"]), len(f64["time_us"]))
    pos = np.stack([f32[f"truth.pos_{a}"][:n] - f64[f"truth.pos_{a}"][:n] for a in "ned"], 1)
    ang = np.stack([f32[f"truth.{a}"][:n] - f64[f"truth.{a}"][:n]
                    for a in ("roll", "pitch", "yaw")], 1)  # fmt: skip
    ang = (ang + np.pi) % (2 * np.pi) - np.pi
    return float(np.linalg.norm(pos, axis=1).max()), float(np.degrees(np.abs(ang)).max())


@pytest.mark.full
def test_v7_float32_tracks_float64(tmp_path: Path) -> None:
    pos, att = precision_divergence(tmp_path)
    # Spec expectation: ≤ 1 m and ≤ 0.5°. Light turbulence and sensor noise are included, so
    # the closed loop amplifies rounding differences; float32 is the training precision.
    assert pos <= 1.0 and att <= 0.5, (pos, att)


def test_v13_same_seed_same_bundle(tmp_path: Path) -> None:
    d = load_scenario("m1-motorcut-classical").model_dump(mode="json")
    d["duration_s"] = 70.0
    sc = Scenario.model_validate(d)
    a = record(run_scenario(sc), tmp_path / "a")
    b = record(run_scenario(sc), tmp_path / "b")
    for name in ("manifest.json", "frames.parquet", "events.jsonl", "viz.arrow"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    events = [json.loads(line) for line in (a / "events.jsonl").read_text().splitlines()]
    assert any(e["type"] == "failure_detected" for e in events)
