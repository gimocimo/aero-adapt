"""D4 (M2 spec): the held-out Rascal 110, configured from its datasheet, flown in JSBSim.

The Rascal's JSBSim model is GPL data fetched into a user cache (ADR-006), never committed;
these tests skip where it has not been fetched (``uv run --extra jsbsim python
tools/rascal_flights.py`` fetches it, and flies the full campaign).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jsbsim")

from aeroadapt.harness.host_flight import fly_vehicle
from aeroadapt.scenarios.resolve import resolve
from aeroadapt.scenarios.schema import load_scenario
from aeroadapt.sim.jsbsim.fdm import JSBSimFDM
from aeroadapt.sim.jsbsim.rascal import CACHE_ROOT, DERIVED, MODEL_SIGNS, derived_model
from aeroadapt.vehicle.jsbsim_vehicle import JSBSimVehicle

pytestmark = pytest.mark.skipif(
    not (CACHE_ROOT / "aircraft" / "Rascal" / "Rascal.xml").exists(),
    reason="the Rascal is not in the cache (tools/rascal_flights.py fetches it)",
)


def test_the_derived_model_takes_our_surfaces_and_thrust(tmp_path: Path) -> None:
    """The control system and engine are ours; the signs of the Rascal's control powers are
    MODEL_SIGNS (positive aileron rolls right, elevator pitches down, rudder yaws left)."""
    import shutil

    shutil.copytree(CACHE_ROOT / "aircraft" / "Rascal", tmp_path / "aircraft" / "Rascal")
    root = derived_model(tmp_path)
    f = JSBSimFDM(DERIVED, root=root)
    assert f["inertia/weight-lbs"] == pytest.approx(13.0)  # no fuel tank, no engine mass

    def accel(surfaces: tuple[float, float, float]) -> np.ndarray:
        f.initialise(140.0, 22.0, alpha=0.03, surfaces=surfaces)
        return np.array([f[f"accelerations/{a}dot-rad_sec2"] for a in "pqr"])

    base = accel((0.0, 0.0, 0.0))
    for i, sign in enumerate(MODEL_SIGNS):
        s = [0.0, 0.0, 0.0]
        s[i] = 0.1
        assert np.sign(accel((s[0], s[1], s[2]))[i] - base[i]) == sign
    f.initialise(140.0, 22.0, alpha=0.03)
    f["external_reactions/thrust/magnitude"] = 5.0
    f.step()
    assert f["forces/fbx-total-lbs"] > 0.0  # thrust beats the drag at this speed and α


def test_the_stock_stack_flies_the_rascal_in_jsbsim() -> None:
    """The M1 loop's first 90 s, in the scenario's wind (6 m/s) and light turbulence: on
    track, at height and speed, with nothing but the datasheet configuration."""
    sc = load_scenario("rascal-nominal").model_copy(update={"duration_s": 90.0})
    r = resolve(sc)
    vehicle = JSBSimVehicle(
        r.airframe, r.site, r.env, r.faults, r.config, model=DERIVED, root=derived_model(),
        model_signs=MODEL_SIGNS,
    )  # fmt: skip
    c = fly_vehicle(r, vehicle).cols
    t = c["time_us"] * 1e-6
    settled = t > 20.0
    wind = np.stack([c["truth.wind_n"], c["truth.wind_e"]], axis=1)
    np.testing.assert_allclose(wind.mean(axis=0), np.asarray(r.env.wind.steady_ned)[:2], atol=1.0)
    assert c["truth.tas"][0] == pytest.approx(14.5, abs=0.5)
    assert not c["done"].any()
    assert c["wp_idx"].max() >= 1
    assert np.abs(c["truth.alt_agl"][settled] - 120.0).max() < 10.0
    assert np.abs(c["truth.tas"][settled] - 14.5).max() < 5.0
    assert np.degrees(np.abs(c["est.roll"] - c["truth.roll"])[settled]).max() < 3.0
