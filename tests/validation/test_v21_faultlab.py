"""V21 (M2 spec §10): faultlab v1. Every kind (a) produces its intended observable effect and
(b) changes nothing before its onset; labels switch exactly at onset.

All kinds fly the same 25 s Aerosonde flight (calm air, stock stack, fault at 10 s), batched
in one vmap with a fault-free twin flown from the same seed.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike
from pydantic import TypeAdapter

from aeroadapt.airframes import Airframe, to_vehicle_params
from aeroadapt.faultlab import build_fault_config, no_faults
from aeroadapt.faultlab.schema import FaultSpec
from aeroadapt.faultlab.slots import (
    ACTUATOR_KINDS,
    AIRFRAME_KINDS,
    ENERGY_KINDS,
    PROPULSION_KINDS,
    SENSOR_KINDS,
)
from aeroadapt.harness.closed_loop import LoopLog, closed_loop, setup_loop
from aeroadapt.scenarios import load_site
from aeroadapt.sim.aerojax import make_env
from aeroadapt.stack.types import make_mission
from aeroadapt.vehicle.protocol import InitialCondition

ONSET, SECONDS = 10.0, 25.0
SPECS: dict[str, dict[str, Any]] = {
    "airspeed_bias": dict(type="sensor_fault", kind="airspeed_bias", magnitude=5.0),
    "airspeed_stuck": dict(type="sensor_fault", kind="airspeed_stuck"),
    "airspeed_drift": dict(type="sensor_fault", kind="airspeed_drift", magnitude=0.5),
    "gyro_bias": dict(type="sensor_fault", kind="gyro_bias", magnitude=0.05, axis=2),
    "accel_bias": dict(type="sensor_fault", kind="accel_bias", magnitude=1.0, axis=0),
    "imu_vibration": dict(type="sensor_fault", kind="imu_vibration", magnitude=8.0),
    "mag_interference": dict(type="sensor_fault", kind="mag_interference", magnitude=0.2, axis=1),
    "baro_drift": dict(type="sensor_fault", kind="baro_drift", magnitude=20.0),
    "gnss_dropout": dict(type="sensor_fault", kind="gnss_dropout", magnitude=0.0),
    "gnss_drift": dict(type="sensor_fault", kind="gnss_drift", magnitude=2.0, axis=1),
    "gnss_spoof": dict(type="sensor_fault", kind="gnss_spoof", magnitude=0.4, axis=0),
    "hard_over": dict(type="actuator_fault", kind="hard_over", channel="rudder", magnitude=1.0),
    "floating": dict(type="actuator_fault", kind="floating", channel="elevator"),
    "slow": dict(type="actuator_fault", kind="slow", channel="aileron", magnitude=0.1),
    "reduced_throw": dict(
        type="actuator_fault", kind="reduced_throw", channel="aileron", magnitude=0.3
    ),
    "backlash": dict(type="actuator_fault", kind="backlash", channel="elevator", magnitude=3.0),
    "reversed": dict(type="actuator_fault", kind="reversed", channel="aileron"),
    "partial_loss": dict(type="propulsion_fault", kind="partial_loss", magnitude=0.4),
    "prop_damage": dict(type="propulsion_fault", kind="prop_damage", magnitude=0.3),
    "thermal_derate": dict(
        type="propulsion_fault", kind="thermal_derate", magnitude=0.5, ramp_s=2.0
    ),
    "cell_failure": dict(type="energy_fault", kind="cell_failure"),
    "abnormal_sag": dict(type="energy_fault", kind="abnormal_sag", magnitude=8.0),
    "capacity_loss": dict(type="energy_fault", kind="capacity_loss", magnitude=0.6),
    "cg_shift": dict(type="airframe_fault", kind="cg_shift", magnitude=0.03),
    "payload_drop": dict(type="airframe_fault", kind="payload_drop", magnitude=0.25),
    "wing_damage": dict(type="airframe_fault", kind="wing_damage", magnitude=0.15, side=1),
}
_SPEC = TypeAdapter(FaultSpec)


@pytest.fixture(scope="module")
def runs(aerosonde: Airframe) -> dict[str, LoopLog]:
    site = load_site("cambridge_west")
    env = make_env(
        alt_origin_msl_m=site.ground_msl_m, magnetic_ned_gauss=site.magnetic.field_ned_gauss
    )
    su = setup_loop(aerosonde, site, env, InitialCondition(), seed=3)
    configs = [no_faults()] + [
        build_fault_config([_SPEC.validate_python({**s, "trigger": {"at_time_s": ONSET}})])
        for s in SPECS.values()
    ]
    faults = jax.tree.map(lambda *x: jnp.stack([jnp.asarray(v) for v in x]), *configs)
    mission = make_mission([(3000.0, 0.0, 120.0)])
    n = round(SECONDS / su.config.dt_control)

    def one(f):
        return closed_loop(su.vp, su.env, f, su.sp, mission, su.sim0, su.stack0, su.z0,
                           jax.random.PRNGKey(5), n_steps=n, config=su.config)[2]  # fmt: skip

    logs = jax.tree.map(np.asarray, jax.vmap(one)(faults))
    names = ["baseline", *SPECS]
    return {k: jax.tree.map(lambda x, i=i: x[i], logs) for i, k in enumerate(names)}


def _spread(x: ArrayLike) -> float:
    """Sum over axes of each axis's standard deviation."""
    return float(np.std(x, axis=0).sum())


def _after(log: LoopLog, delay: float = 0.0) -> np.ndarray:
    return np.asarray(log.t) >= ONSET + delay


def test_every_v1_kind_is_covered() -> None:
    kinds = {*SENSOR_KINDS[1:], *ACTUATOR_KINDS[1:], *PROPULSION_KINDS[1:], *ENERGY_KINDS[1:],
             *AIRFRAME_KINDS[1:]}  # fmt: skip
    assert kinds == set(SPECS)
    # With the three M1 faults (pitot blockage with two drain variants, servo jam, motor cut)
    # the library has 26 + 4 = 30 kinds, plus link loss planned (M2 spec §7.1).
    assert len(kinds) == 26


@pytest.mark.parametrize("name", list(SPECS))
def test_no_effect_before_onset_and_labels(runs: dict[str, LoopLog], name: str) -> None:
    base, run = runs["baseline"], runs[name]
    before = ~_after(run)
    # Identical to machine precision: the batch's SIMD lanes may round one ulp differently.
    for field in ("pos_ned", "quat", "surfaces", "cmd", "va"):
        np.testing.assert_allclose(
            np.asarray(getattr(run, field))[before],
            np.asarray(getattr(base, field))[before],
            rtol=1e-12,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        run.sensors.accel[before], base.sensors.accel[before], rtol=1e-12, atol=1e-12
    )
    lab = run.labels
    family = SPECS[name]["type"].split("_")[0]
    kind = np.asarray(getattr(lab, f"{family}_kind")).reshape(len(run.t), -1).max(axis=1)
    assert np.all(kind[before] == 0) and np.all(kind[_after(run)] > 0)


def test_sensor_effects(runs: dict[str, LoopLog]) -> None:
    b = runs["baseline"]
    a = _after(b, 2.0)
    d = lambda k, f: np.asarray(f(runs[k]))[a] - np.asarray(f(b))[a]  # noqa: E731
    ias = lambda log: np.sqrt(2 * np.maximum(log.sensors.airspeed_dp_pa, 0) / 1.225)  # noqa: E731
    # The stack flies the biased reading, so compare the sensor with the true airspeed.
    bias = ias(runs["airspeed_bias"])[a] - runs["airspeed_bias"].va[a]
    assert np.median(bias) - np.median(ias(b)[a] - b.va[a]) == pytest.approx(5.0, abs=0.5)
    stuck = runs["airspeed_stuck"].sensors.airspeed_dp_pa[a]
    assert np.ptp(stuck) == 0.0 and np.ptp(b.sensors.airspeed_dp_pa[a]) > 1.0
    drift = ias(runs["airspeed_drift"])[a] - runs["airspeed_drift"].va[a]
    assert drift[-1] - (ias(b)[a] - b.va[a])[-1] > 5.0
    assert np.mean(d("gyro_bias", lambda r: r.sensors.gyro[:, 2])) == pytest.approx(0.05, abs=0.01)
    assert np.mean(d("accel_bias", lambda r: r.sensors.accel[:, 0])) == pytest.approx(1.0, abs=0.3)
    assert _spread(runs["imu_vibration"].sensors.accel[a]) > 3 * _spread(b.sensors.accel[a])
    assert np.mean(d("mag_interference", lambda r: r.sensors.mag[:, 1])) == pytest.approx(
        0.2, abs=0.02
    )
    # The stack holds the drifting baro height, so the aircraft climbs ~Δp / (ρ g) for real.
    climb = d("baro_drift", lambda r: -r.pos_ned[:, 2])
    assert climb[-1] > 15.0
    assert not np.any(runs["gnss_dropout"].sensors.fresh.gnss[_after(b, 0.5)])
    # The navigator follows the drifting fix, so compare the GNSS error (reading − truth).
    drift = d("gnss_drift", lambda r: r.sensors.gnss_pos_ned[:, 1] - r.pos_ned[:, 1])
    assert drift[-1] == pytest.approx(2.0 * (SECONDS - ONSET), rel=0.15)
    spoof_v = d("gnss_spoof", lambda r: r.sensors.gnss_vel_ned[:, 0])
    assert spoof_v[-1] == pytest.approx(0.4 * (SECONDS - ONSET), rel=0.15)


def test_actuator_effects(runs: dict[str, LoopLog], aerosonde: Airframe) -> None:
    act = to_vehicle_params(aerosonde).actuators
    a = _after(runs["baseline"], 1.0)
    rud = runs["hard_over"].surfaces[a, 2]
    assert np.abs(rud[-1]) == pytest.approx(act.max_deflection[2], rel=0.02)
    fl = runs["floating"]
    np.testing.assert_allclose(fl.surfaces[a, 1], -0.6 * fl.alpha[a], atol=math.radians(1.5))
    slow = np.abs(np.diff(runs["slow"].surfaces[a, 0])) / 0.01
    assert slow.max() <= 0.1 * act.rate_limit[0] + 1e-6
    assert np.abs(runs["reduced_throw"].surfaces[a, 0]).max() <= 0.3 * act.max_deflection[0] + 1e-9
    assert np.abs(runs["backlash"].surfaces[a, 1] - runs["baseline"].surfaces[a, 1]).max() > 1e-4
    rev = runs["reversed"]
    # The aileron now moves against its command: the aircraft rolls the wrong way and departs.
    assert np.abs(rev.euler[a, 0]).max() > math.radians(20.0)


def test_propulsion_energy_airframe_effects(runs: dict[str, LoopLog]) -> None:
    b = runs["baseline"]
    a = _after(b, 0.5)
    late = _after(b, 5.0)
    # Within 30 ms of onset, before TECS reacts with more throttle.
    first = _after(b)
    ratio = runs["partial_loss"].thrust[first][1:4] / b.thrust[first][1:4]
    assert np.median(ratio) == pytest.approx(0.6, abs=0.05)
    assert _spread(runs["prop_damage"].sensors.accel[a]) > 2 * _spread(b.sensors.accel[a])
    assert runs["thermal_derate"].throttle_actual[late].max() <= 0.5 + 1e-6
    dv = b.battery_v[a] - runs["cell_failure"].battery_v[a]
    assert np.median(dv) > 3.0  # one LiPo cell
    assert np.median(b.battery_v[a] - runs["abnormal_sag"].battery_v[a]) > 1.0
    used = lambda r: r.soc[a][0] - r.soc[a][-1]  # noqa: E731
    assert used(runs["capacity_loss"]) > 2.0 * used(b)
    # Airframe faults move the trim: elevator (CG, mass) and aileron (wing damage).
    assert np.abs(runs["cg_shift"].cmd[late, 1].mean() - b.cmd[late, 1].mean()) > 0.02
    assert np.abs(runs["payload_drop"].cmd[late, 1].mean() - b.cmd[late, 1].mean()) > 0.01
    assert np.abs(runs["wing_damage"].cmd[late, 0].mean() - b.cmd[late, 0].mean()) > 0.02


def test_weather_events_from_a_scenario() -> None:
    """Gust (1−cos), low-level shear and a thermal with its sink ring, built from YAML."""
    from aeroadapt.scenarios.resolve import build_events
    from aeroadapt.scenarios.schema import EventsCfg
    from aeroadapt.sim.aerojax.atmosphere import event_wind

    ev = build_events(
        EventsCfg.model_validate(
            {
                "gust": {"at_time_s": 5.0, "duration_s": 2.0, "speed_mps": 6.0, "from_deg": 90.0},
                "shear": {"height_m": 50.0, "speed_change_mps": -4.0, "from_deg": 0.0},
                "thermals": [{"centre_ne_m": [100.0, 0.0], "radius_m": 60.0, "w_peak_mps": 3.0}],
            }
        )
    )
    w = lambda pos, t: np.asarray(event_wind(ev, jnp.asarray(pos), -pos[2], jnp.asarray(t)))  # noqa: E731
    far = [5000.0, 5000.0, -200.0]
    np.testing.assert_allclose(w(far, 4.9), 0.0, atol=1e-12)  # before the gust, above shear
    np.testing.assert_allclose(w(far, 6.0), [0.0, -6.0, 0.0], atol=1e-9)  # peak, from the east
    np.testing.assert_allclose(w([5000.0, 5000.0, -25.0], 0.0), [2.0, 0.0, 0.0], atol=1e-9)
    assert w([100.0, 0.0, -200.0], 0.0)[2] == pytest.approx(-3.0)  # updraft (NED: negative)
    assert w([100.0 + 90.0, 0.0, -200.0], 0.0)[2] > 0.0  # the sink ring
