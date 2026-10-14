"""Trim and linearisation (V4, V5).

Trim solves for a steady flight condition given airspeed ``va``, flight-path angle ``gamma``
and turn radius ``radius`` (inf = straight). There are seven unknowns, z = (α, θ, φ, δa, δe,
δr, δt), with β = 0 (coordinated). The seven residuals are u̇ = v̇ = ẇ = ṗ = q̇ = ṙ = 0, plus
climb rate = Va·sin γ. Body rates are those of a steady turn, ω = ψ̇·(−sin θ, sin φ cos θ,
cos φ cos θ), so φ̇ = θ̇ = 0 by construction. The attitude θ is solved for rather than
approximated as α + γ, which is exact in turning flight too.

Runs in float64 regardless of the global JAX setting.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.optimize import least_squares

from aeroadapt.airframes.params import AirframeParams
from aeroadapt.core import quaternion
from aeroadapt.sim.aerojax.environment import Environment
from aeroadapt.sim.aerojax.model import state_derivative
from aeroadapt.sim.aerojax.state import RigidBodyState, SurfaceControls


class TrimResult(NamedTuple):
    state: RigidBodyState
    controls: np.ndarray  # (4,) float64: δa, δe, δr (rad, B&M signs) and δt
    alpha: float
    theta: float
    phi: float
    residual: float  # max |residual|
    success: bool


def _trim_state(z: Array, va: float, psi_dot: float, h: float) -> RigidBodyState:
    alpha, theta, phi = z[0], z[1], z[2]
    vel = va * jnp.stack([jnp.cos(alpha), jnp.zeros_like(alpha), jnp.sin(alpha)])
    omega = psi_dot * jnp.stack(
        [-jnp.sin(theta), jnp.sin(phi) * jnp.cos(theta), jnp.cos(phi) * jnp.cos(theta)]
    )
    q = quaternion.from_euler(phi, theta, jnp.zeros_like(phi))
    return RigidBodyState(jnp.array([0.0, 0.0, -h]), vel, q, omega)


def trim(
    p: AirframeParams,
    env: Environment,
    va: float,
    gamma: float = 0.0,
    radius: float = math.inf,
    h: float = 100.0,
    v_supply: float | None = None,
) -> TrimResult:
    psi_dot = 0.0 if math.isinf(radius) else va * math.cos(gamma) / radius
    with jax.enable_x64(True):
        zero3 = jnp.zeros(3)

        def residual(z: Array) -> Array:
            s = _trim_state(z, va, psi_dot, h)
            u = SurfaceControls(z[3], z[4], z[5], z[6])
            vs = None if v_supply is None else jnp.asarray(v_supply)
            d = state_derivative(p, env, s, u, zero3, v_supply=vs)
            climb = -d.pos_ned[2]
            return jnp.concatenate(
                [d.vel_body, d.omega, ((climb - va * jnp.sin(gamma)) / va)[None]]
            )

        fun = jax.jit(residual)
        jac = jax.jit(jax.jacfwd(residual))
        phi0 = 0.0 if psi_dot == 0.0 else math.atan(va * psi_dot / float(env.gravity))
        z0 = np.array([0.05, 0.05 + gamma, phi0, 0.0, -0.1, 0.0, 0.6])

        def jacobian(z: np.ndarray) -> np.ndarray:
            return np.asarray(jac(jnp.asarray(z)))

        sol = least_squares(
            lambda z: np.asarray(fun(jnp.asarray(z))),
            z0,
            jac=jacobian,  # pyright: ignore[reportArgumentType]  (scipy stubs omit callables)
            method="lm",
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
            max_nfev=200,
        )
        z = jnp.asarray(sol.x)
        res = float(np.max(np.abs(sol.fun)))
        state = jax.tree_util.tree_map(np.asarray, _trim_state(z, va, psi_dot, h))
    controls = np.asarray(sol.x[3:7], dtype=np.float64)
    ok = bool(res < 1e-8 and 0.0 <= sol.x[6] <= 1.0)
    return TrimResult(state, controls, float(sol.x[0]), float(sol.x[1]), float(sol.x[2]), res, ok)


# --------------------------------------------------------------------------- linearisation
LON = (3, 5, 10, 7, 2)  # u, w, q, θ, p_d   (Euler-state indices)
LAT = (4, 9, 11, 6, 8)  # v, p, r, φ, ψ


def _euler_model(p: AirframeParams, env: Environment, v_supply: float | None):
    """f(x12, u4) on the Euler state x = [pn, pe, pd, u, v, w, φ, θ, ψ, p, q, r]."""

    def f(x: Array, u: Array) -> Array:
        phi, theta, psi = x[6], x[7], x[8]
        s = RigidBodyState(x[0:3], x[3:6], quaternion.from_euler(phi, theta, psi), x[9:12])
        vs = None if v_supply is None else jnp.asarray(v_supply)
        d = state_derivative(p, env, s, SurfaceControls(*u), jnp.zeros(3), v_supply=vs)
        pr, qr, rr = x[9], x[10], x[11]
        euler_dot = jnp.stack(
            [
                pr + (qr * jnp.sin(phi) + rr * jnp.cos(phi)) * jnp.tan(theta),
                qr * jnp.cos(phi) - rr * jnp.sin(phi),
                (qr * jnp.sin(phi) + rr * jnp.cos(phi)) / jnp.cos(theta),
            ]
        )
        return jnp.concatenate([d.pos_ned, d.vel_body, euler_dot, d.omega])

    return f


class LinearModel(NamedTuple):
    a: np.ndarray  # (12, 12) on the Euler state
    b: np.ndarray  # (12, 4) inputs (δa, δe, δr, δt)
    a_lon: np.ndarray  # (5, 5): u, w, q, θ, p_d
    b_lon: np.ndarray  # (5, 2): δe, δt
    a_lat: np.ndarray  # (5, 5): v, p, r, φ, ψ
    b_lat: np.ndarray  # (5, 2): δa, δr


def linearise(
    p: AirframeParams, env: Environment, tr: TrimResult, v_supply: float | None = None
) -> LinearModel:
    with jax.enable_x64(True):
        phi, theta, psi = quaternion.to_euler(jnp.asarray(tr.state.quat))
        x = jnp.concatenate(
            [
                jnp.asarray(tr.state.pos_ned),
                jnp.asarray(tr.state.vel_body),
                jnp.stack([phi, theta, psi]),
                jnp.asarray(tr.state.omega),
            ]
        )
        u = jnp.asarray(tr.controls)
        f = _euler_model(p, env, v_supply)
        a = np.asarray(jax.jacfwd(f, 0)(x, u))
        b = np.asarray(jax.jacfwd(f, 1)(x, u))
    lon, lat = np.array(LON), np.array(LAT)
    return LinearModel(
        a,
        b,
        a[np.ix_(lon, lon)],
        b[np.ix_(lon, [1, 3])],
        a[np.ix_(lat, lat)],
        b[np.ix_(lat, [0, 2])],
    )
