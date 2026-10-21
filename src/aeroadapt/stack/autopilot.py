"""The stock flight stack core (spec §9.2): navigator → modes/mission → L1 + TECS → attitude,
with the classical contingency able to take over guidance. Pure JAX: ``stack_step`` is jit-,
scan- and vmap-able.

Modes:
* **AUTO**: fly the mission items (waypoints, loiters, jumps); at the end, loop or go to RTL.
* **RTL**: loiter over home. A geofence breach also switches to RTL (ArduPlane's
  ``FENCE_ACTION`` = RTL), and so does the battery monitor reaching the reserve
  (``BATT_FS_LOW_ACT`` = RTL); a new mission (``ap_init``) re-engages AUTO.
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
    ITEM_FLYOVER,
    ITEM_JUMP,
    ITEM_LOITER_TURNS,
    ITEM_LOITER_UNLIM,
    ITEM_RTL,
    ITEM_WAYPOINT,
    LAT_DIRECT,
    LAT_LOITER,
    LAT_WAYPOINT,
    MAX_ITEMS,
    VERT_DIRECT,
    VERT_GLIDE,
    VERT_TECS,
    Mission,
    Target,
    target,
)

MODE_AUTO, MODE_RTL, MODE_GUIDED = 0, 1, 2
MODE_NAMES = ("AUTO", "RTL", "GUIDED")


LOITER_CAPTURE = 0.1  # on the circle: within this fraction of the radius (at least 10 m)
LOITER_EXIT_COS = 0.866  # leave a finished loiter heading within 30° of the next item


class ApState(NamedTuple):
    mode: Array
    wp_idx: Array  # mission item being flown
    leg_a: Array  # (2,) start of the current leg
    leg_h: Array  # height at the start of the current leg (m AGL)
    captured: Array  # () bool, on the current loiter's circle
    turned: Array  # () rad turned around the loiter centre since capture
    bearing: Array  # () last bearing from the loiter centre
    t_capture: Array  # () time of capture
    jumps: Array  # (MAX_ITEMS,) int, jumps taken by each JUMP item


def ap_init(mode: int | Array, pos: Array, height: Array) -> ApState:
    """Start of a mission: item 0, with the first leg starting where the aircraft is."""
    z = jnp.zeros_like(height)
    return ApState(
        jnp.asarray(mode, jnp.int32),
        jnp.zeros((), jnp.int32),
        pos,
        height,
        jnp.zeros((), bool),
        z,
        z,
        z,
        jnp.zeros(MAX_ITEMS, jnp.int32),
    )


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
        ap=ap_init(mode, nav.pos[:2], height),
        tecs=tecs_init(sp.tecs, height, nav.tas),
        att=att_init(dtype),
        cont=cont_init(dtype),
        throttle=jnp.asarray(sp.tecs.thr_cruise, dtype),
    )


def inside_polygon(p: Array, verts: Array, n: Array) -> Array:
    """Even-odd test of the point ``p`` against the first ``n`` vertices of ``verts``."""
    j = jnp.arange(verts.shape[0])
    a = verts
    b = verts[(j + 1) % jnp.maximum(n, 1)]
    straddles = (a[:, 1] > p[1]) != (b[:, 1] > p[1])
    de = jnp.where(straddles, b[:, 1] - a[:, 1], 1.0)
    north = a[:, 0] + (p[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / de
    crossings = jnp.logical_and(jnp.logical_and(j < n, straddles), p[0] < north)
    return jnp.sum(crossings) % 2 == 1


def _next_item(mission: Mission, i: Array, jumps: Array) -> tuple[Array, Array, Array]:
    """The item after ``i``: (index, jumps, end of mission). A JUMP item is resolved on the
    way (taken while it has jumps left, otherwise skipped); missions never chain jumps."""
    kind, prm = jnp.asarray(mission.kind), jnp.asarray(mission.param)
    n = i + 1
    k = jnp.minimum(n, MAX_ITEMS - 1)
    is_jump = jnp.logical_and(n < mission.n_wp, kind[k] == ITEM_JUMP)
    repeats = prm[k, 2]
    take = jnp.logical_and(is_jump, jnp.logical_or(repeats < 0, jumps[k] < repeats))
    jumps = jnp.where(take, jumps.at[k].add(1), jumps)
    after = jnp.where(is_jump, n + 1, n)
    end = jnp.logical_and(~take, after >= mission.n_wp)
    idx = jnp.where(take, prm[k, 1].astype(jnp.int32), jnp.where(end, 0, after))
    return idx.astype(jnp.int32), jumps, end


def _mission_step(
    sp: StackParams,
    ap: ApState,
    pos: Array,
    mission: Mission,
    *,
    vel: Array | None = None,
    t: Array | float = 0.0,
    height: Array | None = None,
    used_mah: Array | None = None,
) -> tuple[ApState, Target]:
    """AUTO item sequencing, the RTL loiter and the geofence.

    * WAYPOINT: done within the acceptance radius or on passing it (fly-by); FLYOVER: only on
      passing it. As ArduPlane's glide slope: a leg that changes altitude by at least
      ``glide_slope_min`` moves the height target along the leg in proportion to the progress
      made, so the climb or descent is spread over the leg; a smaller change is flown at once.
    * LOITER_TURNS / LOITER_TIME: circle the item; turns or time count from capture (on the
      circle), and the loiter ends once they are done and the course points at the next item
      (ArduPlane's "heading required" exit; at once if that item is inside the circle).
      LOITER_UNLIM never ends.
    * JUMP: resolved when sequencing past it; RTL: switches to the RTL mode.
    * Leaving a loiter starts the next leg where the aircraft is; leaving a waypoint, at it.
    * Failsafes (when ``height`` / ``used_mah`` are given): leaving the geofence or climbing
      above its ceiling, or using the battery down to the mission's reserve, switches to RTL.
    """
    wps = jnp.asarray(mission.wps)  # may arrive as host NumPy when closed over
    kind, prm = jnp.asarray(mission.kind), jnp.asarray(mission.param)
    vel = jnp.zeros(2, pos.dtype) if vel is None else vel
    i = ap.wp_idx
    k = kind[i]
    b = wps[i, :2]
    ab = b - ap.leg_a
    passed = (pos - b) @ ab > 0.0
    near = jnp.linalg.norm(pos - b) < sp.l1.wp_radius

    # Loiter progress around the item's centre.
    is_loiter = jnp.logical_and(k >= ITEM_LOITER_TURNS, k <= ITEM_LOITER_UNLIM)
    radius = jnp.abs(prm[i, 0])
    direction = jnp.where(prm[i, 0] < 0.0, -1.0, 1.0)
    r_vec = pos - b
    bearing = jnp.arctan2(r_vec[1], r_vec[0])
    band = jnp.maximum(LOITER_CAPTURE * radius, 10.0)
    on_circle = jnp.abs(jnp.linalg.norm(r_vec) - radius) < band
    captured = jnp.logical_or(ap.captured, jnp.logical_and(is_loiter, on_circle))
    newly = jnp.logical_and(captured, ~ap.captured)
    step = (bearing - ap.bearing + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    turned = jnp.where(captured & ~newly, ap.turned + direction * step, 0.0)
    t_capture = jnp.where(newly, t, ap.t_capture)
    nxt, jumps, end = _next_item(mission, i, ap.jumps)
    # The next item's position (home for RTL); one inside the circle ends the loiter at once
    # (as ArduPlane's verify_loiter_heading does), since the course never points at it.
    nxt_pos = jnp.where(kind[nxt] == ITEM_RTL, jnp.asarray(mission.home)[:2], wps[nxt, :2])
    to_next = nxt_pos - pos
    heading_ok = jnp.logical_or(
        vel @ to_next >= LOITER_EXIT_COS * jnp.linalg.norm(vel) * jnp.linalg.norm(to_next),
        jnp.linalg.norm(nxt_pos - b) < radius,
    )
    counted = jnp.where(k == ITEM_LOITER_TURNS, turned >= 2.0 * jnp.pi * prm[i, 1],
                        t - t_capture >= prm[i, 1])  # fmt: skip
    loiter_done = jnp.logical_and(
        jnp.logical_and(captured, k != ITEM_LOITER_UNLIM), jnp.logical_and(counted, heading_ok)
    )
    reached = jnp.where(
        k == ITEM_WAYPOINT,
        jnp.logical_or(near, passed),
        jnp.where(k == ITEM_FLYOVER, passed, jnp.logical_and(is_loiter, loiter_done)),
    )
    switch = jnp.logical_and(ap.mode == MODE_AUTO, reached)
    to_rtl = jnp.logical_or(jnp.logical_and(end, ~mission.loop), kind[nxt] == ITEM_RTL)
    mode = jnp.where(jnp.logical_and(switch, to_rtl), MODE_RTL, ap.mode)

    # Geofence (on the estimated position, as ArduPlane): a breach switches to RTL.
    if height is not None:
        out = jnp.logical_and(
            mission.n_fence >= 3, ~inside_polygon(pos, jnp.asarray(mission.fence), mission.n_fence)
        )
        high = jnp.logical_and(mission.fence_alt > 0.0, height > mission.fence_alt)
        breach = jnp.logical_and(jnp.logical_or(out, high), mode != MODE_RTL)
        mode = jnp.where(breach, MODE_RTL, mode)
    if used_mah is not None:  # battery failsafe on the monitor's consumed charge
        low = jnp.logical_and(mission.rtl_mah > 0.0, used_mah >= mission.rtl_mah)
        mode = jnp.where(jnp.logical_and(low, mode == MODE_AUTO), MODE_RTL, mode)

    leg_start = jnp.where(is_loiter, pos, b)
    ap = ApState(
        mode.astype(jnp.int32),
        jnp.where(switch, nxt, i).astype(jnp.int32),
        jnp.where(switch, leg_start, ap.leg_a),
        jnp.where(switch, wps[i, 2], ap.leg_h).astype(ap.leg_h.dtype),
        jnp.logical_and(captured, ~switch),
        jnp.where(switch, 0.0, turned).astype(ap.turned.dtype),
        bearing.astype(ap.bearing.dtype),
        jnp.where(switch, 0.0, t_capture).astype(ap.t_capture.dtype),
        jnp.where(switch, jumps, ap.jumps),
    )

    i = ap.wp_idx
    k = kind[i]
    wp = wps[i]
    speed = jnp.where(prm[i, 3] > 0.0, prm[i, 3], mission.speed)
    leg = wp[:2] - ap.leg_a
    progress = jnp.clip((pos - ap.leg_a) @ leg / jnp.maximum(leg @ leg, 1.0), 0.0, 1.0)
    dh = wp[2] - ap.leg_h
    gs = sp.l1.glide_slope_min
    sloped = jnp.logical_and(gs > 0.0, jnp.abs(dh) >= gs)
    alt = jnp.where(sloped, ap.leg_h + dh * progress, wp[2])
    t_auto = target(
        lateral=LAT_WAYPOINT,
        a=ap.leg_a,
        b=wp[:2],
        roll_limit=sp.l1.roll_limit,
        vertical=VERT_TECS,
        alt=alt,
        speed=speed,
    )
    t_loiter = target(
        lateral=LAT_LOITER,
        centre=wp[:2],
        radius=jnp.abs(prm[i, 0]),
        direction=jnp.where(prm[i, 0] < 0.0, -1.0, 1.0),
        roll_limit=sp.l1.roll_limit,
        vertical=VERT_TECS,
        alt=wp[2],
        speed=speed,
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
    loitering = jnp.logical_and(k >= ITEM_LOITER_TURNS, k <= ITEM_LOITER_UNLIM)
    t_item = jax.tree.map(lambda lo, a: jnp.where(loitering, lo, a), t_loiter, t_auto)
    return ap, jax.tree.map(lambda a, r: jnp.where(ap.mode == MODE_AUTO, a, r), t_item, t_rtl)


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
    ap, tgt = _mission_step(
        sp, s.ap, pos, mission, vel=vel, t=nav.t, height=est.alt_agl, used_mah=z.battery_mah
    )
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
