"""Total Energy Control System (Lambregts; structure of ArduPilot ``AP_TECS``). Pure JAX.

* **Throttle** controls the *total* specific energy rate (potential + kinetic).
* **Pitch** controls the energy *balance* (potential vs kinetic), weighted by ``spd_weight``.

Gliding (motor out or throttle forced to zero) and underspeed protection both move all the
weight to speed: pitch then controls airspeed only.

Throttle and pitch trims are scheduled with the speed demand, from the airframe's trim
table: throttle linear in V, pitch trim linear in (V_cruise/V)² (level-flight α). The
integrators then only absorb model error.

As in AP_TECS, the height and speed demands are rate-limited and then lagged, and the
energy-rate demands are the derivatives of the *lagged* demands, so a step in either target
never produces a one-step spike. The integrator clamps follow AP_TECS too: they bound the
integrator's *movement* but never step it (ArduPilot issue #4066).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from aeroadapt.stack.config import TecsParams

G = 9.80665
HGT_LAG_S = 2.0  # first-order lag on the rate-limited height demand (AP_TECS: 0.05 at 10 Hz)
SPD_LAG_S = 1.0  # same for the speed demand
VDOT_LP_S = 0.5
PITCH_MARGIN_RAD = 0.0783  # integrator may hold pitch up to 4.5° beyond its limits (AP_TECS)


class TecsState(NamedTuple):
    hgt_dem_rl: Array  # rate-limited height demand
    hgt_dem: Array  # … then lagged: the demand TECS controls to
    spd_dem_rl: Array
    spd_dem: Array
    integ_thr: Array
    integ_pitch: Array  # on the energy-balance error (m²/s²)
    vdot_lp: Array  # filtered longitudinal acceleration
    pitch_dem: Array
    underspeed: Array


class TecsOut(NamedTuple):
    pitch_dem: Array
    throttle: Array
    underspeed: Array
    ste_error: Array  # total energy error (m²/s²)
    seb_error: Array  # energy balance error


def tecs_init(p: TecsParams, height: Array, tas: Array) -> TecsState:
    z = jnp.zeros_like(height)
    tas = jnp.clip(tas, p.v_min, p.v_max)
    return TecsState(height, height, tas, tas, z, z, z, z + p.pitch_trim, jnp.zeros((), bool))


def _clamp_no_step(x_new: Array, x_old: Array, lo: Array, hi: Array) -> Array:
    """Clamp that may stop an integrator moving further out of [lo, hi] but never moves it
    in by itself."""
    return jnp.clip(x_new, jnp.minimum(lo, x_old), jnp.maximum(hi, x_old))


def tecs_step(
    p: TecsParams,
    s: TecsState,
    *,
    height: Array,
    climb_rate: Array,
    tas: Array,
    accel_x: Array,  # body-x specific force
    pitch: Array,
    roll: Array,
    hgt_target: Array,
    spd_target: Array,
    gliding: Array,  # bool: throttle forced to zero, pitch flies speed
    dt: float,
) -> tuple[TecsState, TecsOut]:
    v = jnp.maximum(tas, 3.0)
    # Longitudinal acceleration along the flight path: f_x − g·sin θ, low-passed.
    vdot = accel_x - G * jnp.sin(pitch)
    vdot_lp = s.vdot_lp + (dt / VDOT_LP_S) * (vdot - s.vdot_lp)

    # ------------------------------------------------------------- demands
    underspeed = jnp.logical_and(
        ~gliding,
        jnp.logical_or(
            jnp.logical_and(s.underspeed, tas < p.v_min * 1.15),
            tas < p.v_min * 0.9,
        ),
    )
    # Underspeed: stop asking for height, fly speed first.
    dh = jnp.clip(hgt_target - s.hgt_dem_rl, -p.sink_max * dt, p.climb_max * dt)
    hgt_rl = jnp.where(underspeed, height, s.hgt_dem_rl + dh)
    hgt_dem = jnp.where(underspeed, height, s.hgt_dem + (dt / HGT_LAG_S) * (hgt_rl - s.hgt_dem))
    hgt_rate_dem = jnp.where(underspeed, 0.0, (hgt_dem - s.hgt_dem) / dt)
    # Gliding may fly below the powered minimum (best glide is slower than ARSPD_FBW_MIN).
    spd_target = jnp.clip(spd_target, jnp.where(gliding, 0.8 * p.v_min, p.v_min), p.v_max)
    # Speed-demand rate limits from the energy-rate limits (AP_TECS velRateMax/Min).
    rate_up = 0.5 * p.climb_max * G / v
    rate_dn = 0.5 * p.sink_min * G / v
    spd_rl = s.spd_dem_rl + jnp.clip(spd_target - s.spd_dem_rl, -rate_dn * dt, rate_up * dt)
    spd_dem = s.spd_dem + (dt / SPD_LAG_S) * (spd_rl - s.spd_dem)
    spd_rate_dem = (spd_dem - s.spd_dem) / dt

    spe_dem, spe = G * hgt_dem, G * height
    ske_dem, ske = 0.5 * spd_dem**2, 0.5 * v**2
    spedot_dem = G * hgt_rate_dem
    skedot_dem = v * spd_rate_dem
    spedot, skedot = G * climb_rate, v * vdot_lp

    # ------------------------------------------------------------- throttle (total energy)
    stedot_max, stedot_min = p.climb_max * G, -p.sink_min * G
    ste_error = (spe_dem - spe) + (ske_dem - ske)
    # Induced drag in a bank (TECS_RLL2THR): extra energy rate needed at load factor n.
    cos_r = jnp.maximum(jnp.cos(roll), 0.2)
    roll_comp = p.rll2thr * (1.0 / cos_r**2 - 1.0)
    stedot_dem = jnp.clip(spedot_dem + skedot_dem, stedot_min, stedot_max)
    stedot_error = stedot_dem - spedot - skedot
    k_thr2ste = stedot_max - stedot_min  # throttle range 0..1
    k_ste2thr = 1.0 / (p.time_const * k_thr2ste)
    thr_trim = p.thr_cruise + p.thr_slope * (spd_dem - p.v_cruise)
    ff = thr_trim + (stedot_dem + roll_comp) / k_thr2ste
    thr_unsat = (ste_error + stedot_error * p.thr_damp) * k_ste2thr + ff
    integ_thr = _clamp_no_step(
        s.integ_thr + ste_error * p.integ_gain * dt * k_ste2thr,
        s.integ_thr,
        -thr_unsat,
        1.0 - thr_unsat,
    )
    throttle = jnp.where(gliding, 0.0, jnp.clip(thr_unsat + integ_thr, 0.0, 1.0))
    integ_thr = jnp.where(gliding, 0.0, integ_thr)

    # ------------------------------------------------------------- pitch (energy balance)
    ske_w = jnp.where(jnp.logical_or(gliding, underspeed), 2.0, jnp.clip(p.spd_weight, 0, 2))
    spe_w = 2.0 - ske_w
    seb_error = (spe_dem * spe_w - ske_dem * ske_w) - (spe * spe_w - ske * ske_w)
    sebdot_dem = spedot_dem * spe_w - skedot_dem * ske_w
    sebdot_error = sebdot_dem - (spedot * spe_w - skedot * ske_w)
    gain_inv = v * p.time_const * G
    pitch_trim = p.pitch_trim + p.pitch_trim_b * ((p.v_cruise / spd_dem) ** 2 - 1.0)
    temp = seb_error + sebdot_error * p.ptch_damp + sebdot_dem * p.time_const
    lim_lo = (p.pitch_min - PITCH_MARGIN_RAD - pitch_trim) * gain_inv - temp
    lim_hi = (p.pitch_max + PITCH_MARGIN_RAD - pitch_trim) * gain_inv - temp
    span = jnp.maximum(lim_hi - lim_lo, 0.0)
    # No more than 10% of the range per step: single glitches cannot wind the integrator.
    delta = jnp.clip(seb_error * p.integ_gain * dt, -0.1 * span, 0.1 * span)
    integ_pitch = _clamp_no_step(s.integ_pitch + delta, s.integ_pitch, lim_lo, lim_hi)
    pitch_unc = pitch_trim + (temp + integ_pitch) / gain_inv
    max_rate = p.vert_acc / v * dt  # pitch-rate limit from the vertical acceleration limit
    pitch_dem = s.pitch_dem + jnp.clip(
        jnp.clip(pitch_unc, p.pitch_min, p.pitch_max) - s.pitch_dem, -max_rate, max_rate
    )

    state = TecsState(
        hgt_rl, hgt_dem, spd_rl, spd_dem, integ_thr, integ_pitch, vdot_lp, pitch_dem, underspeed
    )
    return state, TecsOut(pitch_dem, throttle, underspeed, ste_error, seb_error)
