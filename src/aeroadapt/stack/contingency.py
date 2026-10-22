"""Classical contingency v0 (spec §9.3), the seed of baseline B2. Pure JAX state machine.

1. **Detect** a motor failure from ESC telemetry (rpm far below what the commanded throttle
   should give), or, on airframes without ESC telemetry, with a CUSUM test on the
   total-energy-rate residual against TECS's throttle model.
2. **Assess** every pre-surveyed zone: the height needed to glide to the start of its final
   approach and down the final, using the nominal glide ratio scaled by the wind-triangle
   ground-to-air speed ratio, plus pattern and safety margins. Pick the best reachable zone
   (maximum quality × margin). The geometry is fixed at selection: land into wind, aim a
   quarter of the field in, and a descent circle tangent to the final course at the
   final-start point F, on the side the aircraft is on.
3. **Fly a glider's pattern.** At selection, estimate the height on arrival at the
   circuit's downwind point D (diametrically opposite F, heading downwind). Too high: hold
   on a descent circle tangent to the downwind leg abeam the aircraft, and leave onto the
   leg once the height fits. Within range: join the
   **downwind leg** (parallel to the final, offset by the circle's diameter) directly.
   Too low for a circuit: straight in. Turn base when the height matches the path
   still to fly (base turn + a final lengthened by the downwind distance). Then final and
   flare. The downwind leg makes the absorbable height *continuous*: one lap costs more
   height than a final can absorb, so a circle-only pattern always has gaps that wind
   turns into short or long landings. Energy is managed with the **speed to fly**: the
   airspeed whose glide ratio matches the remaining path over the remaining height, from
   the nominal parabolic polar. On the base turn and on the circle's last lap the aircraft
   may cut in early when low (flexible base). If no zone is reachable: glide straight
   ahead, into wind.

Speeds respect the B&M Aerosonde's elevator authority: it saturates near best glide
(~16 m/s), so gliding is floored at ``glide_speed_min`` and the final approach at
``V_APPROACH`` (≈1.3 × the elevator-limited minimum), leaving authority to flare.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from aeroadapt.stack.config import ContParams, StackParams, Zones
from aeroadapt.stack.guidance import cross2
from aeroadapt.stack.navigator import Estimate
from aeroadapt.stack.types import (
    LAT_HEADING,
    LAT_LOITER,
    LAT_WAYPOINT,
    VERT_DIRECT,
    VERT_GLIDE,
    Target,
    target,
)

NOMINAL, CIRCLE, FINAL, FLARE, STRAIGHT_IN, DOWNWIND, BASE = 0, 1, 2, 3, 4, 5, 6
PHASE_NAMES = ("NOMINAL", "CIRCLE", "FINAL", "FLARE", "STRAIGHT_IN", "DOWNWIND", "BASE")
RHO0 = 1.225
G = 9.80665
WINDOW_DIST_M = 40.0  # a pattern point counts as reached within this distance …
WINDOW_HEADING_RAD = 0.6  # … with the course within ±34° of the course there
V_PATTERN_MAX = 25.0  # fastest speed-to-fly on circle and downwind (bank stays moderate)
V_FINAL_MAX = 26.0  # fastest speed-to-fly on base and final
V_APPROACH = 20.0  # slowest base and final (flare authority, see module docstring)
# The pattern is planned at this speed's glide ratio: ~20% stretch margin to the slowest
# final, and long landings have more room than short ones (aim point a quarter in).
V_PLAN = 22.5
DOWNWIND_PLAN_M = 400.0  # nominal longest downwind leg past D …
DOWNWIND_ROOM_MIN_M = 200.0  # … but always this much more after joining late
DOWNWIND_MAX_M = 1000.0  # safety cap
REPLAN_DIST_M = 900.0
HOLD_MARGIN_M = 0.0  # hold only when this much above what the downwind leg can absorb
# Achieved glide vs the nominal polar in light turbulence (control activity, gust drag):
# every plan uses ratios discounted by this factor.
GLIDE_EFFICIENCY = 0.92
MAX_LAPS = 8
CIRCLE_EFFICIENCY = 0.85  # glide in a turn vs straight (load factor, wind drift)
CIRCLE_ROLL_LIMIT_RAD = 0.61  # 35°: steep banks cost glide (sink grows as n^1.5)
FINAL_ROLL_LIMIT_RAD = 0.44  # 25°
INTERCEPT_FACTOR = 0.41  # extra path per metre of lateral offset at a 45° intercept (√2 − 1)
FLARE_PITCH_MAX_RAD = 0.21  # 12°
FLARE_SEC = 2.0
ENERGY_ARM_S = 20.0  # energy-residual detector armed after navigator convergence
FLOAT_M = 60.0  # the glide path aims this far short of the touchdown point (flare float)
STOP_M = 60.0  # a landing direction needs this much of the zone left beyond the touchdown
# Over obstacles, the glide aims this far past the displaced threshold: on the planned glide
# (L/D about 12 on final) it crosses the threshold about 6 m up, room for arriving low.
THRESHOLD_BUFFER_M = 75.0


class ContState(NamedTuple):
    failed: Array  # bool, latched
    t_detect: Array  # time of detection (s since stack start), -1 before
    esc_t: Array  # persistence timers
    energy_cusum: Array  # CUSUM of the energy-rate deficit (m)
    edot_lp: Array  # filtered total-energy rate, m/s climb equivalent
    phase: Array
    zone: Array  # selected zone (-1 = none)
    land_dir: Array  # (2,) unit vector of the landing course
    touchdown: Array  # (2,) aim point T
    final_start: Array  # (2,) F: where the final starts (moves back by the downwind length)
    centre: Array  # (2,) descent circle, later the base-turn centre
    side: Array  # +1: the pattern lies to the right of the final course
    direction: Array  # +1 clockwise
    dw_start: Array  # (2,) the circuit's downwind point D (fixed at selection)
    hold_exit: Array  # (2,) where the descent circle meets the downwind leg
    straight_hdg: Array
    flare_pitch: Array  # flare pitch integrator


class ContOut(NamedTuple):
    active: Array  # bool: the contingency owns the guidance target
    phase: Array
    zone: Array
    target: Target
    margin: Array  # (Z,) height margin per zone now (m); -inf if invalid/unflyable
    speed_to_fly: Array
    laps: Array  # planned full laps of the descent circle still to fly
    h_plan: Array  # planned height at the next decision point (D on the circle)
    ld_required: Array  # glide ratio the remaining pattern needs


def cont_init(dtype=jnp.float32) -> ContState:
    z, z2 = jnp.zeros((), dtype), jnp.zeros(2, dtype)
    return ContState(
        failed=jnp.zeros((), bool),
        t_detect=z - 1,
        esc_t=z,
        energy_cusum=z,
        edot_lp=z,
        phase=jnp.zeros((), jnp.int32),
        zone=jnp.full((), -1, jnp.int32),
        land_dir=z2,
        touchdown=z2,
        final_start=z2,
        centre=z2,
        side=z + 1,
        direction=z + 1,
        dw_start=z2,
        hold_exit=z2,
        straight_hdg=z,
        flare_pitch=z,
    )


# --------------------------------------------------------------------------- glide model
def best_glide_speed(cp: ContParams, rho: Array) -> Array:
    return jnp.maximum(cp.v_bg_rho0 * jnp.sqrt(RHO0 / rho), cp.v_glide_min)


def glide_ratio_at(cp: ContParams, rho: Array, v: Array | float) -> Array:
    """Glide ratio at airspeed ``v`` on the parabolic polar: L/D = 2·LDmax / (x + 1/x),
    x = (V/V_bg)²."""
    x = (v / (cp.v_bg_rho0 * jnp.sqrt(RHO0 / rho))) ** 2
    return 2.0 * cp.ld_max / (x + 1.0 / x)


def speed_for_glide_ratio(
    cp: ContParams, rho: Array, ld_req: Array, v_lo: Array | float, v_hi: float
) -> Array:
    """Airspeed in [v_lo, v_hi] whose glide ratio is closest to ``ld_req`` (inverse of
    :func:`glide_ratio_at` on the fast side of the polar)."""
    r = jnp.clip(ld_req / cp.ld_max, 0.05, 1.0)
    x = 1.0 / r + jnp.sqrt(jnp.maximum(1.0 / r**2 - 1.0, 0.0))
    v = cp.v_bg_rho0 * jnp.sqrt(RHO0 / rho) * jnp.sqrt(x)
    return jnp.clip(v, v_lo, v_hi)


def ground_speed(v_air: Array | float, wind: Array, course_unit: Array) -> Array:
    """Ground speed along ``course_unit`` (…, 2) at airspeed ``v_air`` (wind triangle);
    -1 where the crosswind cannot be beaten."""
    w_along = course_unit @ wind
    w_cross = course_unit[..., 0] * wind[1] - course_unit[..., 1] * wind[0]
    under = v_air**2 - w_cross**2
    return jnp.where(under > 0, w_along + jnp.sqrt(jnp.maximum(under, 0.0)), -1.0)


def zone_patterns(
    cp: ContParams, zones: Zones, wind: Array, pos: Array
) -> tuple[Array, Array, Array, Array]:
    """Per zone: landing course unit vector (the axis direction with the headwind; in light
    wind the one whose final starts nearer the aircraft), aim point a quarter of the field
    in, the final-start point, and whether that landing is usable.

    Where the site survey found obstacles on an approach (docs/07 §5), the glide's aim point
    (``FLOAT_M`` before the touchdown) moves at least ``THRESHOLD_BUFFER_M`` past the
    displaced threshold, so the planned glide crosses the threshold several metres up, and
    the touchdown moves with it; a direction is usable while ``STOP_M`` of the zone remains
    beyond the touchdown. If the into-wind direction is not usable, the other one is (a
    downwind landing beats none)."""
    d1 = jnp.stack([jnp.cos(zones.axis), jnp.sin(zones.axis)], -1)  # (Z, 2)
    along_wind = d1 @ wind  # < 0: d1 is into wind
    # Light wind: approach from the side the aircraft is on (F = T − L·d lies upwind of T).
    nearer = ((pos[None, :] - zones.centre) * d1).sum(-1) < 0  # aircraft behind the centre
    prefer_d1 = jnp.where(jnp.abs(along_wind) > 1.0, along_wind < 0, nearer)
    half = jnp.asarray(zones.half_length)[:, None]
    aim_back = zones.thr_back - THRESHOLD_BUFFER_M
    td_back = jnp.where(
        aim_back >= 0.5 * half + FLOAT_M, 0.5 * half, aim_back - FLOAT_M
    )  # (Z, 2): touchdown, metres before the centre, landing along +axis / −axis
    ok = td_back >= STOP_M - half
    use_d1 = jnp.where(prefer_d1, ok[:, 0] | ~ok[:, 1], ok[:, 0] & ~ok[:, 1])
    d = jnp.where(use_d1[:, None], d1, -d1)
    back = jnp.where(use_d1, td_back[:, 0], td_back[:, 1])
    touchdown = jnp.asarray(zones.centre) - back[:, None] * d
    final_start = touchdown - cp.final_length * d
    return d, touchdown, final_start, jnp.where(use_d1, ok[:, 0], ok[:, 1])


def assess_zones(
    cp: ContParams, zones: Zones, pos: Array, height: Array, wind: Array, rho: Array
) -> tuple[Array, Array]:
    """(height margin, selection score) per zone; both -inf where unreachable or invalid.
    ``height`` is above the site datum; each zone's margin counts from its own ground."""
    d, _, final_start, usable = zone_patterns(cp, zones, wind, pos)
    to_f = final_start - pos[None, :]
    dist_f = jnp.linalg.norm(to_f, axis=-1)
    va = best_glide_speed(cp, rho)
    vg_leg = ground_speed(va, wind, to_f / jnp.maximum(dist_f, 1e-3)[:, None])
    vg_final = ground_speed(va, wind, d)
    flyable = jnp.logical_and(vg_leg > 1.0, vg_final > 1.0)
    ld = GLIDE_EFFICIENCY * glide_ratio_at(cp, rho, va)
    h_req = (
        dist_f / (ld * jnp.maximum(vg_leg, 1.0) / va)
        + cp.final_length / (ld * jnp.maximum(vg_final, 1.0) / va)
        + cp.h_pattern
        + cp.h_margin
    )
    ok = jnp.logical_and(jnp.logical_and(zones.valid, flyable), usable)
    margin = jnp.where(ok, height - zones.elev - h_req, -jnp.inf)
    score = jnp.where(margin >= 0, zones.quality * margin, -jnp.inf)  # spec §9.3
    return margin, score


def _select(cp: ContParams, zones: Zones, s: ContState, est: Estimate, score: Array) -> ContState:
    """Fix the pattern for the best zone, or straight-in into wind if none is reachable."""
    best = jnp.argmax(score)
    ok = jnp.isfinite(score[best])
    wind = est.wind_ned[:2]
    d, touchdown, final_start, _ = zone_patterns(cp, zones, wind, est.pos_ned[:2])
    dk, fk = d[best], final_start[best]
    # Circle tangent to the final course at F, on the side the aircraft is on.
    side = jnp.where(cross2(dk, est.pos_ned[:2] - fk) >= 0, 1.0, -1.0)
    right = jnp.stack([-dk[1], dk[0]])
    into_wind = jnp.arctan2(-wind[1], -wind[0])
    return s._replace(
        phase=jnp.where(ok, CIRCLE, STRAIGHT_IN).astype(jnp.int32),
        zone=jnp.where(ok, best, -1).astype(jnp.int32),
        land_dir=dk,
        touchdown=touchdown[best],
        final_start=fk,
        centre=fk + side * cp.circle_radius * right,
        side=side,
        direction=side,
        dw_start=fk + 2.0 * cp.circle_radius * side * right,
        hold_exit=fk + 2.0 * cp.circle_radius * side * right,
        straight_hdg=jnp.where(jnp.linalg.norm(wind) > 2.0, into_wind, est.yaw),
    )


def _arc_to(centre: Array, direction: Array, radius: float, pos: Array, point: Array) -> Array:
    """Arc length along the circle (in the flying direction) from ``pos`` to ``point``."""
    r_now, r_p = pos - centre, point - centre
    ang = jnp.arctan2(cross2(r_now, r_p), r_now @ r_p)  # + = clockwise from now to point
    return jnp.mod(direction * ang, 2 * jnp.pi) * radius


def _in_window(pos: Array, course: Array, point: Array, heading: Array) -> Array:
    return jnp.logical_and(
        jnp.linalg.norm(point - pos) < WINDOW_DIST_M, course @ heading > jnp.cos(WINDOW_HEADING_RAD)
    )


# --------------------------------------------------------------------------- step
def contingency_step(
    sp: StackParams,
    s: ContState,
    est: Estimate,
    *,
    t: Array,
    rho: Array,
    esc_rpm: Array,
    battery_v: Array,
    throttle_prev: Array,
    dt: float,
) -> tuple[ContState, ContOut]:
    cp, zones, tp = sp.cont, sp.zones, sp.tecs
    pos, h, wind = est.pos_ned[:2], est.alt_agl, est.wind_ned[:2]
    radius = cp.circle_radius

    # ------------------------------------------------------------- detection
    expected_rpm = cp.kv_rpm_per_volt * battery_v * throttle_prev
    low_rpm = jnp.logical_and(
        throttle_prev > cp.rpm_min_throttle, esc_rpm < cp.rpm_fraction * expected_rpm
    )
    # The held (sample-and-hold) ESC reading counts every step it stays low.
    esc_t = jnp.where(jnp.logical_and(cp.esc_enabled, low_rpm), s.esc_t + dt, 0.0)
    # Backup: total-energy rate (climb equivalent) far below what the commanded throttle should
    # give on TECS's own model: STEdot = (throttle − trim(V))·(climb_max + sink_min), less the
    # induced drag of a bank (TECS_RLL2THR). A cut loses ≈2.5 m/s at cruise thrust.
    edot = est.climb_rate + est.airspeed * (est.accel[0] - G * jnp.sin(est.pitch)) / G
    edot_lp = s.edot_lp + (dt / cp.energy_tau) * (edot - s.edot_lp)
    thr_trim = tp.thr_cruise + tp.thr_slope * (est.airspeed - tp.v_cruise)
    cos_r = jnp.maximum(jnp.cos(est.roll), 0.2)
    turn = tp.rll2thr * (1.0 / cos_r**2 - 1.0) / G
    expected = (throttle_prev - thr_trim) * (tp.climb_max + tp.sink_min) - turn
    # CUSUM: accumulate the deficit beyond the slack k, decide at h. Gusts move the residual by
    # ~1.4 m/s RMS for seconds at a time; a cut is a sustained ≈3 m/s step. Armed once the
    # navigator has converged (the estimate swings by several m/s while it aligns).
    armed = jnp.logical_and(
        jnp.logical_and(t > ENERGY_ARM_S, throttle_prev > cp.rpm_min_throttle),
        jnp.logical_not(cp.esc_enabled),
    )
    deficit = expected - edot_lp
    energy_cusum = jnp.where(
        armed, jnp.maximum(s.energy_cusum + (deficit - cp.energy_slack) * dt, 0.0), 0.0
    )
    detect = jnp.logical_and(
        cp.enabled,
        jnp.logical_or(esc_t >= cp.esc_persist, energy_cusum >= cp.energy_threshold),
    )
    newly = jnp.logical_and(detect, ~s.failed)
    s = s._replace(
        failed=jnp.logical_or(s.failed, detect),
        t_detect=jnp.where(newly, t, s.t_detect),
        esc_t=esc_t,
        energy_cusum=energy_cusum,
        edot_lp=edot_lp,
    )

    # ------------------------------------------------------------- (re)selection
    margin, score = assess_zones(cp, zones, pos, h, wind, rho)
    sel = jnp.clip(s.zone, 0, zones.valid.shape[0] - 1)
    en_route = jnp.linalg.norm(pos - s.centre) > radius + 80.0
    # Still far away and the chosen zone's margin is used up: choose again. Near the zone the
    # circuit planner (and its flexible cut-ins) is the better judge: never abandon there.
    far = jnp.linalg.norm(pos - s.touchdown) > REPLAN_DIST_M
    lost = jnp.logical_and(jnp.logical_and(s.phase == CIRCLE, far), margin[sel] < -cp.h_margin)
    s = jax.tree.map(
        lambda new, old: jnp.where(jnp.logical_or(newly, lost), new, old),
        _select(cp, zones, s, est, score),
        s,
    )

    # From here on, heights count from the chosen zone's ground (its landing altitude, as
    # ArduPlane's land waypoint); straight in without a zone, from the datum.
    sel = jnp.clip(s.zone, 0, zones.valid.shape[0] - 1)
    h = h - jnp.where(s.zone >= 0, jnp.asarray(zones.elev)[sel], 0.0)

    # ------------------------------------------------------------- pattern geometry
    d = s.land_dir
    right = jnp.stack([-d[1], d[0]])
    course = jnp.stack([jnp.cos(est.course), jnp.sin(est.course)])
    v_bg = best_glide_speed(cp, rho)
    ld_plan = GLIDE_EFFICIENCY * glide_ratio_at(cp, rho, V_PLAN)
    # Air path per metre of ground path into wind (final) and downwind, at the planning speed.
    r_f = V_PLAN / jnp.maximum(ground_speed(V_PLAN, wind, d), 1.0)
    r_dw = V_PLAN / jnp.maximum(ground_speed(V_PLAN, wind, -d), 1.0)
    base_air = jnp.pi * radius / CIRCLE_EFFICIENCY
    lap = 2 * jnp.pi * radius / CIRCLE_EFFICIENCY
    # Along-track distance still to fly to the glide aim point, and the offset from the line.
    aim = s.touchdown - FLOAT_M * d
    along = (aim - pos) @ d
    lat_off = cross2(d, pos - aim)
    final_air = (jnp.maximum(along, 0.0) + INTERCEPT_FACTOR * jnp.abs(lat_off)) * r_f

    # ------------------------------------------------------------- circle: plan to reach D
    # Height wanted at D: the pattern from D with a mid-length downwind leg, at mid speed.
    l_final = jnp.maximum((aim - s.final_start) @ d, 0.0)  # final length to the aim point
    dw_hold = (s.hold_exit - s.dw_start) @ (-d)  # downwind run already made at the exit
    room = jnp.maximum(DOWNWIND_PLAN_M - dw_hold, DOWNWIND_ROOM_MIN_M)
    h_plan = ((l_final + dw_hold) * r_f + base_air + 0.4 * room * (r_dw + r_f)) / ld_plan
    h_dw_max = ((l_final + dw_hold) * r_f + base_air + room * (r_dw + r_f)) / ld_plan
    at_d = _in_window(pos, course, s.hold_exit, -d)
    arc_d = jnp.where(at_d, 0.0, _arc_to(s.centre, s.direction, radius, pos, s.hold_exit))
    leg = jnp.maximum(jnp.linalg.norm(pos - s.centre) - radius, 0.0)
    n = jnp.arange(MAX_LAPS + 1)
    spare = jnp.maximum(h - h_plan, 1.0)
    ld_n = (leg + arc_d / CIRCLE_EFFICIENCY + n * lap) / spare
    ld_fast = GLIDE_EFFICIENCY * glide_ratio_at(cp, rho, V_PATTERN_MAX)
    ld_fast_final = glide_ratio_at(cp, rho, V_FINAL_MAX)
    ld_top = GLIDE_EFFICIENCY * glide_ratio_at(cp, rho, v_bg)
    cost = (
        jnp.abs(jnp.log(ld_n / ld_plan))
        + 10.0 * jnp.maximum(jnp.log(ld_n / ld_top), 0.0)
        + 10.0 * jnp.maximum(jnp.log(ld_fast / ld_n), 0.0)
    )
    laps = n[jnp.argmin(cost)]
    ld_circle = (leg + arc_d / CIRCLE_EFFICIENCY + laps * lap) / spare

    # ------------------------------------------------------------- downwind: when to turn base
    # Distance flown downwind past D (negative while joining upwind of D: an early base turn
    # shortens the final, down to 40% of its nominal length).
    dw_run = jnp.maximum((pos - s.dw_start) @ (-d), -0.6 * l_final)
    off_dw_line = jnp.abs(cross2(-d, pos - s.dw_start))
    established = jnp.logical_and(off_dw_line < 40.0, course @ (-d) > 0.7)
    # Path if the base turn started now: base + final lengthened by the downwind run.
    path_turn_now = base_air + (l_final + dw_run) * r_f
    join = jnp.where(established, 0.0, off_dw_line * (1.0 + INTERCEPT_FACTOR))
    ld_downwind = (join + path_turn_now) / jnp.maximum(h, 1.0)

    # ------------------------------------------------------------- base: plan to reach F
    arc_f = _arc_to(s.centre, s.direction, radius, pos, s.final_start)
    at_f = _in_window(pos, course, s.final_start, d)
    arc_f = jnp.where(at_f, 0.0, arc_f)
    ld_base = (arc_f / CIRCLE_EFFICIENCY + l_final * r_f) / jnp.maximum(h, 1.0)
    ld_final = final_air / jnp.maximum(h, 1.0)

    # ------------------------------------------------------------- transitions
    ph = s.phase
    # Flexible base: on the half of a turn heading along the landing course, cut in as soon as
    # the intercept-plus-final path needs the mid-range glide ratio.
    good_half = jnp.logical_and(course @ d > 0.3, along > 0.5 * cp.final_length)
    low_now = h <= final_air / ld_plan
    circle_to_dw = jnp.logical_and(
        jnp.logical_and(ph == CIRCLE, ~en_route), jnp.logical_and(at_d, h <= h_dw_max)
    )
    circle_cut = jnp.logical_and(
        jnp.logical_and(ph == CIRCLE, ~en_route),
        jnp.logical_and(jnp.logical_and(good_half, laps == 0), low_now),
    )
    dw_to_base = jnp.logical_and(
        jnp.logical_and(ph == DOWNWIND, established),
        jnp.logical_or(h <= path_turn_now / ld_plan, dw_run >= DOWNWIND_MAX_M),
    )
    # Still joining the downwind leg but already low: straight in instead.
    dw_cut = jnp.logical_and(jnp.logical_and(ph == DOWNWIND, ~established), low_now)
    base_to_final = jnp.logical_and(
        ph == BASE, jnp.logical_or(at_f, jnp.logical_and(good_half, low_now))
    )
    phase = jnp.where(circle_to_dw, DOWNWIND, ph)
    phase = jnp.where(
        jnp.logical_or(jnp.logical_or(circle_cut, base_to_final), dw_cut), FINAL, phase
    )
    phase = jnp.where(dw_to_base, BASE, phase)
    # Turning base after a downwind run: the base turn and the final shift back by the run
    # (geometry from the fixed downwind point D: the circle may have been a hold elsewhere).
    side_r = s.side * radius * right
    s = s._replace(
        final_start=jnp.where(dw_to_base, s.dw_start - 2.0 * side_r - dw_run * d, s.final_start),
        centre=jnp.where(dw_to_base, s.dw_start - side_r - dw_run * d, s.centre),
    )
    # Cutting in: the final course starts abeam the aircraft and L1 intercepts it.
    cut = jnp.logical_or(jnp.logical_or(circle_cut, jnp.logical_and(base_to_final, ~at_f)), dw_cut)
    s = s._replace(final_start=jnp.where(cut, aim - jnp.maximum(along, 1.0) * d, s.final_start))
    # Flare at the flare height or two seconds before impact, whichever is higher
    # (LAND_FLARE_ALT / LAND_FLARE_SEC).
    flare_h = jnp.maximum(cp.flare_alt, FLARE_SEC * jnp.maximum(-est.climb_rate, 0.0))
    flare_now = jnp.logical_and(jnp.logical_or(phase == FINAL, phase == STRAIGHT_IN), h <= flare_h)
    s = s._replace(
        phase=jnp.where(flare_now, FLARE, phase).astype(jnp.int32),
        flare_pitch=jnp.where(flare_now, est.pitch, s.flare_pitch),
    )

    # ------------------------------------------------------------- entry on (re)selection
    # Join the downwind leg abeam the aircraft (it may already be downwind of D). Hold on the
    # circle first if the height exceeds what the downwind leg left from there can absorb;
    # go straight in if it is too low for a base turn.
    dw_join = jnp.clip((pos - s.dw_start) @ (-d), -0.6 * l_final, DOWNWIND_MAX_M)
    join_dist = jnp.abs(cross2(-d, pos - s.dw_start)) * (1.0 + INTERCEPT_FACTOR) + radius
    h_at_join = h - join_dist / ld_top
    room_join = jnp.maximum(DOWNWIND_PLAN_M - dw_join, DOWNWIND_ROOM_MIN_M)
    h_join_max = (base_air + (l_final + dw_join + room_join) * r_f + room_join * r_dw) / ld_plan
    h_circuit_min = (base_air + (l_final + dw_join) * r_f) / ld_top
    # Straight in only from the approach side, well before the aim point, and only if not
    # too high for it (the circuit absorbs more).
    approach_side = jnp.logical_and(
        along > 0.6 * cp.final_length, h <= final_air / (GLIDE_EFFICIENCY * ld_fast_final)
    )
    entry = jnp.where(
        h_at_join > h_join_max + HOLD_MARGIN_M,
        CIRCLE,
        jnp.where(jnp.logical_or(h_at_join >= h_circuit_min, ~approach_side), DOWNWIND, FINAL),
    )
    # Hold circle tangent to the downwind leg abeam the aircraft (leaving room downwind).
    dw_hold_new = jnp.clip(dw_join, -0.6 * l_final, DOWNWIND_MAX_M - DOWNWIND_ROOM_MIN_M)
    hold_exit = s.dw_start - dw_hold_new * d
    selected = jnp.logical_and(jnp.logical_or(newly, lost), s.zone >= 0)
    hold = jnp.logical_and(selected, entry == CIRCLE)
    s = s._replace(
        hold_exit=jnp.where(hold, hold_exit, s.hold_exit),
        centre=jnp.where(hold, hold_exit - s.side * radius * right, s.centre),
    )
    s = s._replace(
        phase=jnp.where(selected, entry, s.phase).astype(jnp.int32),
        final_start=jnp.where(
            jnp.logical_and(selected, entry == FINAL),
            aim - jnp.maximum(along, 1.0) * d,
            s.final_start,
        ),
    )

    # ------------------------------------------------------------- targets
    headwind = jnp.maximum(-(wind @ d), 0.0)
    # Near the limit of the glide en route, add the headwind rule of thumb (max range).
    to_c = (s.centre - pos) / jnp.maximum(jnp.linalg.norm(s.centre - pos), 1.0)
    wind_against = jnp.maximum(-(wind @ to_c), 0.0)
    stretch = ld_circle >= 0.95 * ld_top
    v_circle = speed_for_glide_ratio(
        cp, rho, ld_circle / GLIDE_EFFICIENCY, v_bg, V_PATTERN_MAX
    ) + jnp.where(jnp.logical_and(en_route, stretch), cp.headwind_gain * wind_against, 0.0)
    v_downwind = speed_for_glide_ratio(
        cp, rho, jnp.maximum(ld_downwind, ld_plan) / GLIDE_EFFICIENCY, v_bg, V_PLAN
    )
    v_base = speed_for_glide_ratio(cp, rho, ld_base / GLIDE_EFFICIENCY, V_APPROACH, V_FINAL_MAX)
    # The headwind speed bonus maximises ground range: only when stretching the glide.
    ld_appr = GLIDE_EFFICIENCY * glide_ratio_at(cp, rho, V_APPROACH)
    v_final = speed_for_glide_ratio(
        cp, rho, ld_final / GLIDE_EFFICIENCY, V_APPROACH, V_FINAL_MAX
    ) + jnp.where(ld_final >= 0.95 * ld_appr, cp.headwind_gain * headwind, 0.0)
    beyond = s.touchdown + 500.0 * d  # L1 tracks the line, no waypoint switching
    t_circle = target(
        lateral=LAT_LOITER,
        centre=s.centre,
        radius=radius,
        direction=s.direction,
        roll_limit=CIRCLE_ROLL_LIMIT_RAD,
        vertical=VERT_GLIDE,
        speed=v_circle,
    )
    dw_line_a = s.dw_start + l_final * d  # abeam the aim point: joins may start upwind of D
    t_downwind = target(
        lateral=LAT_WAYPOINT,
        a=dw_line_a,
        b=s.dw_start - DOWNWIND_MAX_M * d,
        roll_limit=CIRCLE_ROLL_LIMIT_RAD,
        vertical=VERT_GLIDE,
        speed=v_downwind,
    )
    t_base = t_circle._replace(speed=jnp.asarray(v_base))
    t_final = target(
        lateral=LAT_WAYPOINT,
        a=s.final_start,
        b=beyond,
        roll_limit=FINAL_ROLL_LIMIT_RAD,
        vertical=VERT_GLIDE,
        speed=v_final,
    )
    t_straight = target(
        lateral=LAT_HEADING,
        heading=s.straight_hdg,
        roll_limit=FINAL_ROLL_LIMIT_RAD,
        vertical=VERT_GLIDE,
        speed=jnp.maximum(v_bg, V_APPROACH) + cp.headwind_gain * jnp.linalg.norm(wind),
    )
    # Flare: arrest the sink rate with pitch (P + I on the sink error), wings near level.
    sink_err = -est.climb_rate - cp.flare_sink
    in_flare = s.phase == FLARE
    flare_pitch = jnp.where(
        in_flare,
        jnp.clip(
            s.flare_pitch + 2.0 * cp.flare_pitch_gain * sink_err * dt, -0.1, FLARE_PITCH_MAX_RAD
        ),
        s.flare_pitch,
    )
    s = s._replace(flare_pitch=flare_pitch)
    flare_dem = jnp.clip(flare_pitch + cp.flare_pitch_gain * sink_err, -0.1, FLARE_PITCH_MAX_RAD)
    zone_ok = s.zone >= 0
    t_flare = target(
        lateral=jnp.where(zone_ok, LAT_WAYPOINT, LAT_HEADING),
        a=s.final_start,
        b=beyond,
        heading=s.straight_hdg,
        roll_limit=cp.land_roll_limit,
        vertical=VERT_DIRECT,
        pitch_dem=flare_dem,
        throttle=0.0,
    )
    tgt = _pick(
        s.phase,
        {
            CIRCLE: t_circle,
            DOWNWIND: t_downwind,
            BASE: t_base,
            FINAL: t_final,
            FLARE: t_flare,
            STRAIGHT_IN: t_straight,
        },
        t_circle,
    )
    ph = s.phase
    stf = _pick_scalar(
        ph, {CIRCLE: v_circle, DOWNWIND: v_downwind, BASE: v_base, FINAL: v_final}, 0.0
    )
    ld_req = _pick_scalar(
        ph, {CIRCLE: ld_circle, DOWNWIND: ld_downwind, BASE: ld_base, FINAL: ld_final}, 0.0
    )
    return s, ContOut(s.failed, ph, s.zone, tgt, margin, stf, laps, h_plan, ld_req)


def _pick(phase: Array, by_phase: dict[int, Target], default: Target) -> Target:
    out = default
    for k, tk in by_phase.items():
        out = jax.tree.map(lambda a, b, k=k: jnp.where(phase == k, a, b), tk, out)
    return out


def _pick_scalar(phase: Array, by_phase: dict[int, Array], default: float) -> Array:
    out = jnp.asarray(default)
    for k, v in by_phase.items():
        out = jnp.where(phase == k, v, out)
    return out
