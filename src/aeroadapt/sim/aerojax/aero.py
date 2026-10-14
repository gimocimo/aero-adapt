"""Aerodynamic model: Beard & McLain (2012) ch. 4, nonlinear lift/drag with stall blending."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

from aeroadapt.airframes.params import AirframeParams
from aeroadapt.core import quaternion
from aeroadapt.sim.aerojax.state import RigidBodyState, SurfaceControls

VA_MIN = 1e-6  # guard against division by zero at (unphysical) zero airspeed


class AirData(NamedTuple):
    va: Array  # airspeed, m/s
    alpha: Array  # angle of attack, rad
    beta: Array  # sideslip, rad
    v_air_body: Array  # (3,) air-relative velocity in body axes


def air_data(state: RigidBodyState, wind_ned: Array) -> AirData:
    v_air = state.vel_body - quaternion.ned_to_body(state.quat, wind_ned)
    ur, vr, wr = v_air[..., 0], v_air[..., 1], v_air[..., 2]
    va = jnp.linalg.norm(v_air, axis=-1)
    alpha = jnp.arctan2(wr, ur)
    beta = jnp.arcsin(jnp.clip(vr / jnp.maximum(va, VA_MIN), -1.0, 1.0))
    return AirData(va, alpha, beta, v_air)


def stall_sigma(p: AirframeParams, alpha: Array) -> Array:
    """B&M blending function σ(α) (eq. 4.10), evaluated in log-sum-exp form.

    σ = (1 + a + b) / ((1 + a)(1 + b)), where a = exp(-M(α-α0)) and b = exp(M(α+α0)).
    Because a·b = exp(2Mα0) is constant, σ = exp(LSE(0, ln a, ln b) - LSE(0, ln a, ln b, 2Mα0)).
    This form stays finite in float32 for any α, whereas the direct form overflows once
    M·|α| ≳ 88.
    """
    alpha = jnp.asarray(alpha)
    ln_a = -p.M * (alpha - p.alpha0)
    ln_b = p.M * (alpha + p.alpha0)
    zero = jnp.zeros_like(alpha)
    ln_ab = jnp.full_like(alpha, 2.0 * p.M * p.alpha0)
    ln_num = logsumexp(jnp.stack([zero, ln_a, ln_b]), axis=0)
    ln_den = logsumexp(jnp.stack([zero, ln_a, ln_b, ln_ab]), axis=0)
    return jnp.exp(ln_num - ln_den)


def lift_coefficient(p: AirframeParams, alpha: Array) -> Array:
    """C_L(α), B&M eq. 4.9: linear lift blended into flat-plate lift past stall."""
    s = stall_sigma(p, alpha)
    flat_plate = 2.0 * jnp.sign(alpha) * jnp.sin(alpha) ** 2 * jnp.cos(alpha)
    return (1.0 - s) * (p.C_L_0 + p.C_L_alpha * alpha) + s * flat_plate


def drag_coefficient(p: AirframeParams, alpha: Array) -> Array:
    """C_D(α), B&M eq. 4.11: parasitic + induced drag."""
    return p.C_D_p + (p.C_L_0 + p.C_L_alpha * alpha) ** 2 / (jnp.pi * p.e_oswald * p.AR)


def aero_forces_moments(
    p: AirframeParams, rho: float, air: AirData, omega: Array, u: SurfaceControls
) -> tuple[Array, Array]:
    """Aerodynamic force and moment in body axes (excluding gravity and propulsion)."""
    va, alpha, beta = air.va, air.alpha, air.beta
    pr, qr, rr = omega[..., 0], omega[..., 1], omega[..., 2]
    two_va = 2.0 * jnp.maximum(va, VA_MIN)
    qbar_s = 0.5 * rho * va**2 * p.S

    lift = qbar_s * (
        lift_coefficient(p, alpha) + p.C_L_q * p.c / two_va * qr + p.C_L_delta_e * u.delta_e
    )
    drag = qbar_s * (
        drag_coefficient(p, alpha) + p.C_D_q * p.c / two_va * qr + p.C_D_delta_e * u.delta_e
    )
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    fx = -ca * drag + sa * lift
    fz = -sa * drag - ca * lift
    fy = qbar_s * (
        p.C_Y_0
        + p.C_Y_beta * beta
        + p.C_Y_p * p.b / two_va * pr
        + p.C_Y_r * p.b / two_va * rr
        + p.C_Y_delta_a * u.delta_a
        + p.C_Y_delta_r * u.delta_r
    )

    ell = (
        qbar_s
        * p.b
        * (
            p.C_ell_0
            + p.C_ell_beta * beta
            + p.C_ell_p * p.b / two_va * pr
            + p.C_ell_r * p.b / two_va * rr
            + p.C_ell_delta_a * u.delta_a
            + p.C_ell_delta_r * u.delta_r
        )
    )
    m = (
        qbar_s
        * p.c
        * (p.C_m_0 + p.C_m_alpha * alpha + p.C_m_q * p.c / two_va * qr + p.C_m_delta_e * u.delta_e)
    )
    n = (
        qbar_s
        * p.b
        * (
            p.C_n_0
            + p.C_n_beta * beta
            + p.C_n_p * p.b / two_va * pr
            + p.C_n_r * p.b / two_va * rr
            + p.C_n_delta_a * u.delta_a
            + p.C_n_delta_r * u.delta_r
        )
    )
    return jnp.stack([fx, fy, fz], axis=-1), jnp.stack([ell, m, n], axis=-1)
