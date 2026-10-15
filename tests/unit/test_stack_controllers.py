"""Stack controllers in isolation: TECS, attitude loops and mission sequencing."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from aeroadapt.airframes import Airframe
from aeroadapt.scenarios import load_site
from aeroadapt.stack.attitude import att_init, attitude_step
from aeroadapt.stack.autopilot import MODE_AUTO, MODE_RTL, ApState, _mission_step
from aeroadapt.stack.config import StackParams, build_stack_params, load_stack_cfg
from aeroadapt.stack.tecs import tecs_init, tecs_step
from aeroadapt.stack.types import LAT_LOITER, LAT_WAYPOINT, make_mission

DT = 0.01


@pytest.fixture(scope="module")
def sp(aerosonde: Airframe) -> StackParams:
    return build_stack_params(
        load_stack_cfg("aerosonde_bm"), aerosonde, load_site("cambridge_west"), DT
    )


def _tecs(sp: StackParams, *, h=120.0, v=25.0, hgt=120.0, spd=25.0, gliding=False, n=1, s=None):
    p = sp.tecs
    s = tecs_init(p, jnp.asarray(h), jnp.asarray(v)) if s is None else s
    out = None
    for _ in range(n):
        s, out = tecs_step(
            p,
            s,
            height=jnp.asarray(h),
            climb_rate=jnp.asarray(0.0),
            tas=jnp.asarray(v),
            accel_x=jnp.asarray(9.80665 * math.sin(p.pitch_trim)),
            pitch=jnp.asarray(p.pitch_trim),
            roll=jnp.asarray(0.0),
            hgt_target=jnp.asarray(hgt),
            spd_target=jnp.asarray(spd),
            gliding=jnp.asarray(gliding),
            dt=DT,
        )
    assert out is not None
    return s, out


# --------------------------------------------------------------------------- TECS
def test_tecs_holds_trim_with_no_error(sp: StackParams) -> None:
    _, out = _tecs(sp, n=50)
    assert float(out.throttle) == pytest.approx(sp.tecs.thr_cruise, abs=1e-3)
    assert float(out.pitch_dem) == pytest.approx(sp.tecs.pitch_trim, abs=1e-4)


def test_tecs_climb_demand_raises_throttle_and_pitch(sp: StackParams) -> None:
    _, out = _tecs(sp, hgt=140.0, n=200)
    assert float(out.throttle) > sp.tecs.thr_cruise + 0.1
    assert float(out.pitch_dem) > sp.tecs.pitch_trim + math.radians(2)


def test_tecs_speed_demand_trades_pitch_down(sp: StackParams) -> None:
    _, out = _tecs(sp, spd=30.0, n=200)
    assert float(out.throttle) > sp.tecs.thr_cruise
    assert float(out.pitch_dem) < sp.tecs.pitch_trim


def test_tecs_gliding_cuts_throttle_and_flies_speed(sp: StackParams) -> None:
    _, fast = _tecs(sp, v=25.0, spd=20.0, gliding=True, n=100)
    assert float(fast.throttle) == 0.0
    assert float(fast.pitch_dem) > sp.tecs.pitch_trim  # too fast: pitch up to slow down
    _, slow = _tecs(sp, v=17.0, spd=20.0, gliding=True, n=100)
    assert float(slow.pitch_dem) < float(fast.pitch_dem)


def test_tecs_integrator_clamp_never_steps(sp: StackParams) -> None:
    """A sudden large demand change must not jump the pitch integrator (AP_TECS #4066)."""
    s, _ = _tecs(sp, n=20)
    before = float(s.integ_pitch)
    s, _ = _tecs(sp, hgt=400.0, spd=35.0, s=s, n=1)
    span = (sp.tecs.pitch_max - sp.tecs.pitch_min + 0.16) * 25.0 * sp.tecs.time_const * 9.80665
    assert abs(float(s.integ_pitch) - before) <= 0.1 * span + 1e-6


def test_tecs_underspeed_flies_speed_first(sp: StackParams) -> None:
    _, out = _tecs(sp, v=0.85 * sp.tecs.v_min, hgt=200.0, n=50)
    assert bool(out.underspeed)
    assert float(out.pitch_dem) < sp.tecs.pitch_trim  # nose down despite the climb demand


# --------------------------------------------------------------------------- attitude
def _att(sp: StackParams, **kw):
    base = dict(
        roll=0.0,
        pitch=sp.tecs.pitch_trim,
        omega=np.zeros(3),
        accel_y=0.0,
        airspeed=sp.att.v_cruise,
        roll_dem=0.0,
        pitch_dem=sp.tecs.pitch_trim,
    )
    base.update(kw)
    args = {k: jnp.asarray(v) for k, v in base.items()}
    return attitude_step(sp.att, att_init(jnp.float64), dt=DT, **args)[1]


def test_attitude_outputs_servo_trims_at_trim(sp: StackParams) -> None:
    out = _att(sp)
    a = sp.att
    assert float(out.aileron) == pytest.approx(a.trim_ail, abs=1e-6)
    assert float(out.rudder) == pytest.approx(a.trim_rud, abs=1e-6)
    assert float(out.elevator) == pytest.approx(a.trim_ele_a + a.trim_ele_b, abs=1e-6)


def test_attitude_signs(sp: StackParams) -> None:
    trim = _att(sp)
    assert float(_att(sp, roll_dem=0.3).aileron) > float(trim.aileron)  # roll right
    assert float(_att(sp, pitch_dem=sp.tecs.pitch_trim + 0.1).elevator) > float(trim.elevator)
    assert float(_att(sp, omega=np.array([0.0, 0.0, 0.2])).rudder) < float(trim.rudder)  # damper
    assert float(_att(sp, accel_y=0.0, roll=0.3, roll_dem=0.3).rudder) > float(trim.rudder)


def test_speed_scaler_raises_gains_when_slow(sp: StackParams) -> None:
    cruise = _att(sp, roll_dem=0.2)
    slow = _att(sp, roll_dem=0.2, airspeed=0.7 * sp.att.v_cruise)
    trim = _att(sp)
    assert float(slow.aileron - trim.aileron) > 1.5 * float(cruise.aileron - trim.aileron)


# --------------------------------------------------------------------------- mission
def test_mission_switches_legs_loops_and_rtl(sp: StackParams) -> None:
    wps = [(500.0, 0.0, 120.0), (500.0, 500.0, 130.0)]
    ap = ApState(jnp.asarray(MODE_AUTO), jnp.asarray(0), jnp.zeros(2))
    ap, tgt = _mission_step(sp, ap, jnp.array([100.0, 0.0]), make_mission(wps))
    assert int(ap.wp_idx) == 0 and int(tgt.lateral) == LAT_WAYPOINT
    ap, tgt = _mission_step(sp, ap, jnp.array([480.0, 0.0]), make_mission(wps))  # within radius
    assert int(ap.wp_idx) == 1
    np.testing.assert_allclose(ap.leg_a, [500.0, 0.0])
    assert float(tgt.alt) == 130.0
    # Passing the last waypoint: loop back to the first …
    ap_loop, _ = _mission_step(sp, ap, jnp.array([500.0, 510.0]), make_mission(wps, loop=True))
    assert int(ap_loop.wp_idx) == 0 and int(ap_loop.mode) == MODE_AUTO
    # … or return to launch.
    ap_rtl, tgt = _mission_step(sp, ap, jnp.array([500.0, 510.0]), make_mission(wps, loop=False))
    assert int(ap_rtl.mode) == MODE_RTL and int(tgt.lateral) == LAT_LOITER
