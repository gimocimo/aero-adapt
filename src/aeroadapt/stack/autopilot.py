"""The stock flight stack core (spec §9.2): navigator → modes/mission → L1 + TECS → attitude,
with the classical contingency able to take over guidance. Pure JAX: ``stack_step`` is jit-,
scan- and vmap-able.

Modes:
* **AUTO**: fly the waypoint legs; at the end, loop or go to RTL.
* **RTL**: loiter over home.
* **GUIDED**: follow an externally supplied :class:`Target` each step. Test harnesses use
  it for step responses; later a learned high-level policy will drive it.

Any mode hands over to the contingency once it has latched a motor failure.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from aeroadapt.core.messages import SensorArrays
from aeroadapt.stack.attitude import AttOut, AttState, att_init, attitude_step
from aeroadapt.stack.config import StackParams
from aeroadapt.stack.contingency import ContOut, ContState, cont_init, contingency_step
from aeroadapt.stack.guidance import heading_hold, l1_loiter, l1_waypoint, roll_from_lat_acc
from aeroadapt.stack.navigator import R_AIR, Estimate, NavState, nav_init, nav_step
from aeroadapt.stack.tecs import TecsOut, TecsState, tecs_init, tecs_step
from aeroadapt.stack.types import (
    LAT_DIRECT,
    LAT_LOITER,
    LAT_WAYPOINT,
    VERT_DIRECT,
    VERT_GLIDE,
    VERT_TECS,
    Mission,
    Target,
    target,
)

MODE_AUTO, MODE_RTL, MODE_GUIDED = 0, 1, 2
MODE_NAMES = ("AUTO", "RTL", "GUIDED")


class ApState(NamedTuple):
    mode: Array
    wp_idx: Array  # waypoint being flown to
    leg_a: Array  # (2,) start of the current leg


class StackState(NamedTuple):
    nav: NavState
    ap: ApState
    tecs: TecsState
    att: AttState
    cont: ContState
    throttle: Array  # last throttle command (the contingency's detector needs it)


class StackOut(NamedTuple):
    cmd: Array  # (4,) aileron, elevator, rudder, throttle (normalised, ADR-001)
    est: Estimate
    target: Target
    mode: Array
    wp_idx: Array
    roll_dem: Array
    pitch_dem: Array
    lat_acc: Array
    xtrack: Array
    tecs: TecsOut
    att: AttOut
    cont: ContOut


def stack_init(sp: StackParams, z: SensorArrays, mode: int = MODE_AUTO) -> StackState:
    nav = nav_init(sp.nav, z)
    dtype = nav.pos.dtype
    height = -nav.pos[2]
    return StackState(
        nav=nav,
        ap=ApState(jnp.asarray(mode, jnp.int32), jnp.zeros((), jnp.int32), nav.pos[:2]),
        tecs=tecs_init(sp.tecs, height, nav.tas),
        att=att_init(dtype),
        cont=cont_init(dtype),
        throttle=jnp.asarray(sp.tecs.thr_cruise, dtype),
    )


def _mission_step(
    sp: StackParams, ap: ApState, pos: Array, mission: Mission
) -> tuple[ApState, Target]:
    """AUTO leg sequencing (acceptance radius or passing the waypoint) and the RTL loiter."""
    wps = jnp.asarray(mission.wps)  # may arrive as host NumPy when closed over
    b = wps[ap.wp_idx, :2]
    ab = b - ap.leg_a
    reached = jnp.logical_or(jnp.linalg.norm(pos - b) < sp.l1.wp_radius, (pos - b) @ ab > 0.0)
    switch = jnp.logical_and(ap.mode == MODE_AUTO, reached)
    nxt = ap.wp_idx + 1
    done = nxt >= mission.n_wp
    mode = jnp.where(
        jnp.logical_and(switch, jnp.logical_and(done, ~mission.loop)), MODE_RTL, ap.mode
    )
    wp_idx = jnp.where(switch, jnp.where(done, 0, nxt), ap.wp_idx).astype(jnp.int32)
    leg_a = jnp.where(switch, b, ap.leg_a)
    ap = ApState(mode.astype(jnp.int32), wp_idx, leg_a)

    wp = wps[ap.wp_idx]
    t_auto = target(
        lateral=LAT_WAYPOINT,
        a=ap.leg_a,
        b=wp[:2],
        roll_limit=sp.l1.roll_limit,
        vertical=VERT_TECS,
        alt=wp[2],
        speed=mission.speed,
    )
    t_rtl = target(
        lateral=LAT_LOITER,
        centre=jnp.asarray(mission.home)[:2],
        radius=sp.l1.loiter_radius,
        direction=1.0,
        roll_limit=sp.l1.roll_limit,
        vertical=VERT_TECS,
        alt=jnp.asarray(mission.home)[2],
        speed=mission.speed,
    )
    return ap, jax.tree.map(lambda a, r: jnp.where(ap.mode == MODE_AUTO, a, r), t_auto, t_rtl)


def stack_step(
    sp: StackParams,
    s: StackState,
    z: SensorArrays,
    mission: Mission,
    guided: Target | None = None,
) -> tuple[StackState, StackOut]:
    dt = sp.dt
    nav, est = nav_step(sp.nav, s.nav, z, dt)
    pos, vel = est.pos_ned[:2], est.vel_ned[:2]

    # ------------------------------------------------------------- what to fly
    ap, tgt = _mission_step(sp, s.ap, pos, mission)
    if guided is not None:
        tgt = jax.tree.map(lambda g, m: jnp.where(ap.mode == MODE_GUIDED, g, m), guided, tgt)
    rho = z.baro_pa / (R_AIR * z.baro_temp_k)
    cont, cout = contingency_step(
        sp,
        s.cont,
        est,
        t=nav.t,
        rho=rho,
        esc_rpm=z.esc_rpm,
        battery_v=z.battery_v,
        throttle_prev=s.throttle,
        dt=dt,
    )
    tgt = jax.tree.map(lambda c, m: jnp.where(cout.active, c, m), cout.target, tgt)

    # ------------------------------------------------------------- lateral
    wp = l1_waypoint(sp.l1, pos, vel, tgt.a, tgt.b)
    lo = l1_loiter(sp.l1, pos, vel, tgt.centre, tgt.radius, tgt.direction)
    hd = heading_hold(est.yaw, tgt.heading, est.groundspeed)
    lat_acc = jnp.where(
        tgt.lateral == LAT_WAYPOINT,
        wp.lat_acc,
        jnp.where(tgt.lateral == LAT_LOITER, lo.lat_acc, hd),
    )
    xtrack = jnp.where(
        tgt.lateral == LAT_WAYPOINT,
        wp.xtrack_err,
        jnp.where(tgt.lateral == LAT_LOITER, lo.xtrack_err, 0.0),
    )
    roll_dem = jnp.where(
        tgt.lateral == LAT_DIRECT,
        jnp.clip(tgt.roll_dem, -tgt.roll_limit, tgt.roll_limit),
        roll_from_lat_acc(lat_acc, tgt.roll_limit, sp.nav.gravity),
    )

    # ------------------------------------------------------------- vertical
    gliding = tgt.vertical == VERT_GLIDE
    height = est.alt_agl
    tecs, tout = tecs_step(
        sp.tecs,
        s.tecs,
        height=height,
        climb_rate=est.climb_rate,
        tas=est.airspeed,
        accel_x=est.accel[0],
        pitch=est.pitch,
        roll=est.roll,
        hgt_target=jnp.where(gliding, height, tgt.alt),
        spd_target=tgt.speed,
        gliding=gliding,
        dt=dt,
    )
    direct = tgt.vertical == VERT_DIRECT
    # Outside TECS modes keep its demands on the aircraft, so re-engaging is bumpless.
    tecs_reset = tecs_init(sp.tecs, height, est.airspeed)._replace(pitch_dem=est.pitch)
    tecs = jax.tree.map(lambda r, t: jnp.where(direct, r, t), tecs_reset, tecs)
    pitch_dem = jnp.where(direct, tgt.pitch_dem, tout.pitch_dem)
    throttle = jnp.where(direct, tgt.throttle, tout.throttle)
    throttle = jnp.where(jnp.logical_and(cout.active, ~direct), 0.0, throttle)

    # ------------------------------------------------------------- attitude
    att, aout = attitude_step(
        sp.att,
        s.att,
        roll=est.roll,
        pitch=est.pitch,
        omega=est.omega,
        accel_y=est.accel[1],
        airspeed=est.airspeed,
        roll_dem=roll_dem,
        pitch_dem=pitch_dem,
        dt=dt,
    )
    cmd = jnp.stack([aout.aileron, aout.elevator, aout.rudder, jnp.clip(throttle, 0.0, 1.0)])

    state = StackState(nav, ap, tecs, att, cont, cmd[3])
    out = StackOut(
        cmd, est, tgt, ap.mode, ap.wp_idx, roll_dem, pitch_dem, lat_acc, xtrack, tout, aout, cout
    )
    return state, out
