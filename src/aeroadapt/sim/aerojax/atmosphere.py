"""Atmosphere: ISA troposphere, steady wind with an optional boundary-layer profile, Dryden
turbulence (MIL-HDBK-1797 low-altitude model; M1 spec §6.5) and discrete weather events
(M2 spec §7.2): a 1−cos gust, low-level wind shear, and thermals or sink."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from aeroadapt.sim.aerojax import noise

# ISA constants
T0 = 288.15  # K
P0 = 101_325.0  # Pa
LAPSE = 0.0065  # K/m
R_AIR = 287.052_87  # J/(kg K)
G0 = 9.806_65  # m/s^2
ISA_EXP = G0 / (R_AIR * LAPSE)  # ≈ 5.25588

FT = 0.3048
KT = 0.514_444


class IsaParams(NamedTuple):
    temp_offset_k: float  # ISA deviation ΔT
    qnh_pa: float  # sea-level pressure
    alt_origin_msl_m: float  # MSL altitude of the site origin (ground)


class WindParams(NamedTuple):
    steady_ned: Array  # (3,) m/s, air-mass velocity at/above the reference height
    bl_alpha: float  # power-law exponent below bl_ref_m (0 disables the profile)
    bl_ref_m: float  # reference height (AGL) of the profile


class TurbulenceParams(NamedTuple):
    w20_mps: float  # wind speed at 20 ft; σ_w = 0.1·W20 (0 disables turbulence)


class DrydenState(NamedTuple):
    u: Array  # () first-order channel
    v: Array  # (2,) second-order channel states
    w: Array  # (2,)


def isa(isa_p: IsaParams, alt_msl_m: ArrayLike) -> tuple[Array, Array, Array]:
    """(temperature K, static pressure Pa, density kg/m³) at a geometric MSL altitude.

    Pressure follows the standard lapse from QNH; ΔT shifts temperature (and so density),
    the usual "ISA deviation" convention.
    """
    t_std = T0 - LAPSE * jnp.asarray(alt_msl_m)
    pressure = isa_p.qnh_pa * (t_std / T0) ** ISA_EXP
    temp = t_std + isa_p.temp_offset_k
    return temp, pressure, pressure / (R_AIR * temp)


def steady_wind(wind: WindParams, h_agl: Array) -> Array:
    """Steady wind at height ``h_agl``; horizontal components follow a power law below the
    reference height when ``bl_alpha > 0``."""
    ratio = jnp.clip(h_agl, 0.5, wind.bl_ref_m) / wind.bl_ref_m
    factor = jnp.where(wind.bl_alpha > 0, ratio**wind.bl_alpha, 1.0)
    return wind.steady_ned * jnp.array([1.0, 1.0, 0.0]) * factor + wind.steady_ned * jnp.array(
        [0.0, 0.0, 1.0]
    )


def dryden_scales(w20_mps: ArrayLike, h_agl: ArrayLike) -> tuple[Array, Array, Array, Array]:
    """MIL-HDBK-1797 low-altitude Dryden: (σ_u = σ_v, σ_w, L_u = L_v, L_w) in SI units."""
    h_ft = jnp.clip(jnp.asarray(h_agl) / FT, 10.0, 1000.0)
    k = 0.177 + 0.000823 * h_ft
    sigma_w = 0.1 * jnp.asarray(w20_mps)
    sigma_uv = sigma_w / k**0.4
    l_uv = h_ft / k**1.2 * FT
    l_w = h_ft * FT
    return sigma_uv, sigma_w, l_uv, l_w


def dryden_zero() -> DrydenState:
    return DrydenState(jnp.zeros(()), jnp.zeros(2), jnp.zeros(2))


def _second_order(x: Array, sigma: Array, tau: Array, dt: float, noise: Array):
    """Dryden lateral/vertical channel, H(s) ∝ (1 + √3 τs)/(1 + τs)², τ = L/V.

    Controllable canonical form with a double pole at -1/τ; the transition matrix is exact
    and the noise uses Euler-Maruyama (unit intensity). The output gain is set from the
    stationary covariance (P11 = τ³/4, P22 = τ/4, P12 = 0) so that Var(y) = σ² exactly.
    """
    e = jnp.exp(-dt / tau)
    phi = e * jnp.array([[1 + dt / tau, dt], [-dt / tau**2, 1 - dt / tau]])
    x_next = phi @ x + jnp.array([0.0, 1.0]) * jnp.sqrt(dt) * noise
    gain = sigma / tau**1.5
    y = gain * (x_next[0] + jnp.sqrt(3.0) * tau * x_next[1])
    return x_next, y


def dryden_step(
    state: DrydenState, turb: TurbulenceParams, va: Array, h_agl: Array, dt: float, key: Array
) -> tuple[DrydenState, Array]:
    """Advance the turbulence filters by ``dt`` and return the body-axis gust (u, v, w)."""
    sigma_uv, sigma_w, l_uv, l_w = dryden_scales(turb.w20_mps, h_agl)
    v_ref = jnp.maximum(va, 1.0)  # frozen-turbulence speed; guard against V → 0
    n = noise.normal(key, (3,), state.v.dtype)

    a = jnp.exp(-v_ref * dt / l_uv)  # exact AR(1) discretisation of the u channel
    u = a * state.u + sigma_uv * jnp.sqrt(1 - a**2) * n[0]
    v_state, v = _second_order(state.v, sigma_uv, l_uv / v_ref, dt, n[1])
    w_state, w = _second_order(state.w, sigma_w, l_w / v_ref, dt, n[2])
    return DrydenState(u, v_state, w_state), jnp.stack([u, v, w])


def dryden_stationary(
    turb: TurbulenceParams, va: Array, h_agl: Array, key: Array, dtype=jnp.float32
) -> DrydenState:
    """Sample the filters from their stationary distribution, so turbulence starts developed
    rather than spinning up from calm."""
    sigma_uv, _, l_uv, l_w = dryden_scales(turb.w20_mps, h_agl)
    v_ref = jnp.maximum(va, 1.0)
    n = noise.normal(key, (5,), dtype)
    tau_v, tau_w = l_uv / v_ref, l_w / v_ref
    return DrydenState(
        u=sigma_uv * n[0],
        v=jnp.stack([jnp.sqrt(tau_v**3 / 4) * n[1], jnp.sqrt(tau_v / 4) * n[2]]),
        w=jnp.stack([jnp.sqrt(tau_w**3 / 4) * n[3], jnp.sqrt(tau_w / 4) * n[4]]),
    )


# --------------------------------------------------------------------------- weather events
MAX_THERMALS = 2


class EnvEvents(NamedTuple):
    """Host-side event data (NumPy); traced like any other pytree leaf under ``jit``."""

    gust_onset_s: np.ndarray  # ()
    gust_duration_s: np.ndarray  # () > 0
    gust_ned: np.ndarray  # (3,) peak gust velocity (air-mass motion), m/s
    shear_height_m: np.ndarray  # () the shear acts below this height (0: none)
    shear_ned: np.ndarray  # (3,) wind change at the ground, fading to zero at the height
    thermal_ne: np.ndarray  # (T, 2) centres, site frame
    thermal_radius_m: np.ndarray  # (T,)
    thermal_w_mps: np.ndarray  # (T,) peak updraft (negative: sink)
    thermal_onset_s: np.ndarray  # (T,)


def no_events() -> EnvEvents:
    t = MAX_THERMALS
    z, z3 = np.array(0.0), np.zeros(3)
    return EnvEvents(
        z, np.array(1.0), z3, z, z3, np.zeros((t, 2)), np.ones(t), np.zeros(t), np.zeros(t)
    )


def event_wind(ev: EnvEvents, pos_ned: Array, h_agl: Array, t: Array) -> Array:
    """NED air-mass velocity from the events at a point and time (zero with no events)."""
    tau = (t - ev.gust_onset_s) / ev.gust_duration_s
    shape = jnp.where((tau >= 0.0) & (tau <= 1.0), 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * tau)), 0.0)
    wind = jnp.asarray(ev.gust_ned) * shape
    below = jnp.where(
        ev.shear_height_m > 0.0,
        jnp.clip(1.0 - h_agl / jnp.maximum(ev.shear_height_m, 1e-6), 0, 1),
        0.0,
    )
    wind = wind + ev.shear_ned * below
    # Thermals: a bell-shaped core with a ring of sink, w = w₀ (1 − x²) exp(−x²), x = r / R.
    d = pos_ned[:2] - ev.thermal_ne
    x2 = jnp.sum(d * d, axis=-1) / ev.thermal_radius_m**2
    w = jnp.sum(jnp.where(t >= ev.thermal_onset_s, ev.thermal_w_mps * (1 - x2) * jnp.exp(-x2), 0.0))
    return wind + jnp.stack([jnp.zeros_like(w), jnp.zeros_like(w), -w])
