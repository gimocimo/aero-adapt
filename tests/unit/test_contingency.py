"""Classical contingency v0: glide model, geometry, reachability and detection."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from aeroadapt.airframes import Airframe
from aeroadapt.scenarios import load_site
from aeroadapt.stack.config import StackParams, build_stack_params, load_stack_cfg, zones_from_site
from aeroadapt.stack.contingency import (
    CIRCLE,
    DOWNWIND,
    FINAL,
    NOMINAL,
    STRAIGHT_IN,
    assess_zones,
    cont_init,
    contingency_step,
    glide_ratio_at,
    ground_speed,
    speed_for_glide_ratio,
    zone_patterns,
)
from aeroadapt.stack.navigator import Estimate

DT = 0.01
RHO = jnp.asarray(1.2)


@pytest.fixture(scope="module")
def sp(aerosonde: Airframe) -> StackParams:
    return build_stack_params(
        load_stack_cfg("aerosonde_bm"), aerosonde, load_site("cambridge_west"), DT
    )


def _est(pos=(0.0, 0.0), h=120.0, course_deg=0.0, wind=(0.0, 0.0), airspeed=25.0) -> Estimate:
    c = math.radians(course_deg)
    vel = np.array([math.cos(c), math.sin(c), 0.0]) * airspeed
    a = jnp.asarray
    return Estimate(
        quat=a([1.0, 0.0, 0.0, 0.0]), roll=a(0.0), pitch=a(0.05), yaw=a(c), omega=a(np.zeros(3)),
        accel=a([0.5, 0.0, -9.8]), pos_ned=a([*pos, -h]), vel_ned=a(vel), alt_agl=a(h),
        climb_rate=a(0.0), ias=a(airspeed), tas=a(airspeed), airspeed=a(airspeed),
        airspeed_healthy=a(True), wind_ned=a([*wind, 0.0]), groundspeed=a(airspeed),
        course=a(c),
    )  # fmt: skip


# --------------------------------------------------------------------------- glide model
def test_polar_peaks_at_best_glide(sp: StackParams) -> None:
    cp = sp.cont
    v_bg = cp.v_bg_rho0 * math.sqrt(1.225 / float(RHO))
    assert float(glide_ratio_at(cp, RHO, v_bg)) == pytest.approx(cp.ld_max)
    for v in (0.8 * v_bg, 1.3 * v_bg):
        assert float(glide_ratio_at(cp, RHO, v)) < cp.ld_max


@pytest.mark.parametrize("v", [19.0, 21.0, 24.0])
def test_speed_for_glide_ratio_inverts_the_polar(sp: StackParams, v: float) -> None:
    ld = glide_ratio_at(sp.cont, RHO, v)
    assert float(speed_for_glide_ratio(sp.cont, RHO, ld, 15.0, 30.0)) == pytest.approx(v, rel=1e-4)


def test_wind_triangle() -> None:
    north = jnp.array([1.0, 0.0])
    assert float(ground_speed(20.0, jnp.zeros(2), north)) == pytest.approx(20.0)
    assert float(ground_speed(20.0, jnp.array([-5.0, 0.0]), north)) == pytest.approx(15.0)
    assert float(ground_speed(20.0, jnp.array([0.0, 12.0]), north)) == pytest.approx(16.0)
    assert float(ground_speed(10.0, jnp.array([0.0, 12.0]), north)) == -1.0  # cannot hold track


# --------------------------------------------------------------------------- geometry
def test_landing_is_into_wind_and_aims_inside_the_field(sp: StackParams) -> None:
    zones = zones_from_site(load_site("cambridge_west"))
    wind = jnp.array([4.0, 0.0])  # blowing north
    d, touchdown, final_start, usable = zone_patterns(sp.cont, zones, wind, jnp.zeros(2))
    assert bool(np.all(np.asarray(usable)[zones.valid]))
    for i in np.flatnonzero(zones.valid):
        assert float(d[i] @ wind) < 0  # headwind on final
        to_centre = np.asarray(zones.centre[i] - touchdown[i])
        assert 0 < float(to_centre @ np.asarray(d[i])) < zones.half_length[i]  # T inside
        np.testing.assert_allclose(
            np.asarray(touchdown[i] - final_start[i]),
            sp.cont.final_length * np.asarray(d[i]),
            atol=1e-6,
        )


def test_reachability_grows_with_height_and_shrinks_with_distance(sp: StackParams) -> None:
    zones = zones_from_site(load_site("cambridge_west"))
    m_low, _ = assess_zones(sp.cont, zones, jnp.zeros(2), jnp.asarray(60.0), jnp.zeros(2), RHO)
    m_high, score = assess_zones(
        sp.cont, zones, jnp.zeros(2), jnp.asarray(150.0), jnp.zeros(2), RHO
    )
    valid = zones.valid
    assert np.all(np.asarray(m_high)[valid] > np.asarray(m_low)[valid])
    assert np.all(np.isneginf(np.asarray(m_high)[~valid]))
    assert np.isfinite(float(jnp.max(score)))
    far, _ = assess_zones(
        sp.cont, zones, jnp.array([3000.0, 0.0]), jnp.asarray(150.0), jnp.zeros(2), RHO
    )
    assert np.all(np.asarray(far)[valid] < 0)


# --------------------------------------------------------------------------- detection
def _run_detector(sp: StackParams, rpm_fn, throttle: float, seconds: float):
    s = cont_init(jnp.float64)
    est = _est()
    t_detect = None
    for k in range(round(seconds / DT)):
        t = k * DT
        s, out = contingency_step(
            sp,
            s,
            est,
            t=jnp.asarray(t),
            rho=RHO,
            esc_rpm=jnp.asarray(rpm_fn(t)),
            battery_v=jnp.asarray(48.0),
            throttle_prev=jnp.asarray(throttle),
            dt=DT,
        )
        if bool(out.active) and t_detect is None:
            t_detect = t
    return s, t_detect


def test_esc_detector_latency(sp: StackParams) -> None:
    """rpm collapses (τ = 0.3 s spin-down, as in the simulator) at t = 1 s."""
    rpm0 = sp.cont.kv_rpm_per_volt * 48.0 * 0.68 * 0.98
    rpm = lambda t: rpm0 if t < 1.0 else rpm0 * math.exp(-(t - 1.0) / 0.3)  # noqa: E731
    s, t_detect = _run_detector(sp, rpm, 0.68, 3.0)
    assert t_detect is not None
    assert t_detect - 1.0 <= 1.0  # V12: ESC latency ≤ 1.0 s
    assert int(s.phase) in (CIRCLE, DOWNWIND, FINAL, STRAIGHT_IN)


def test_esc_detector_ignores_healthy_motor_and_low_throttle(sp: StackParams) -> None:
    rpm0 = sp.cont.kv_rpm_per_volt * 48.0 * 0.68 * 0.96  # worst loaded/no-load ratio
    s, t_detect = _run_detector(sp, lambda t: rpm0, 0.68, 3.0)
    assert t_detect is None and int(s.phase) == NOMINAL
    s, t_detect = _run_detector(sp, lambda t: 0.0, 0.1, 3.0)  # idle: rpm meaningless
    assert t_detect is None


def test_disabled_contingency_never_engages(sp: StackParams) -> None:
    off = sp._replace(cont=sp.cont._replace(enabled=False))
    _, t_detect = _run_detector(off, lambda t: 0.0, 0.8, 3.0)
    assert t_detect is None
