"""Assembled fixed-wing model: forces, moments, state derivative and one integration step.

Everything here is pure and ``jit``/``vmap``-compatible (ADR-002).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from aeroadapt.airframes.params import AirframeParams
from aeroadapt.core import quaternion
from aeroadapt.sim.aerojax import rigid_body
from aeroadapt.sim.aerojax.aero import AirData, aero_forces_moments, air_data
from aeroadapt.sim.aerojax.environment import Environment
from aeroadapt.sim.aerojax.integrators import rk4
from aeroadapt.sim.aerojax.propulsion import PropulsionOutput, propeller, windmill_thrust
from aeroadapt.sim.aerojax.state import RigidBodyState, SurfaceControls


class ForcesMoments(NamedTuple):
    force_body: Array  # (3,) N, total (gravity + aero + propulsion)
    moment_body: Array  # (3,) N·m, total
    air: AirData
    prop: PropulsionOutput


def forces_moments(
    p: AirframeParams,
    env: Environment,
    s: RigidBodyState,
    u: SurfaceControls,
    wind_ned: Array,
    gust_body: Array | None = None,
    v_supply: Array | None = None,
    motor_cut: Array | None = None,
) -> ForcesMoments:
    """Total body force and moment.

    The optional inputs default to the Beard & McLain model used for N-version parity: no
    gust, ideal battery (``V_max``) and a healthy motor.
    """
    air = air_data(s, wind_ned, gust_body)
    f_aero, m_aero = aero_forces_moments(p, env.rho, air, s.omega, u)
    weight_ned = jnp.zeros_like(s.pos_ned).at[..., 2].set(p.mass * env.gravity)
    f_grav = quaternion.ned_to_body(s.quat, weight_ned)
    prop = propeller(p, env.rho, air.va, u.delta_t, v_supply)
    thrust, torque = prop.thrust, prop.torque
    if motor_cut is not None:  # windmilling propeller, no motor torque (spec §8.2)
        thrust = jnp.where(motor_cut, windmill_thrust(p, env.rho, air.va), thrust)
        torque = jnp.where(motor_cut, 0.0, torque)
    zero = jnp.zeros_like(thrust)
    f_prop = jnp.stack([thrust, zero, zero], axis=-1)
    # The propeller spins positively about +x_body, so the airframe feels -torque (ADR-001).
    m_prop = jnp.stack([-torque, zero, zero], axis=-1)
    prop = prop._replace(thrust=thrust, torque=torque)  # report what actually acts
    return ForcesMoments(f_aero + f_grav + f_prop, m_aero + m_prop, air, prop)


def state_derivative(
    p: AirframeParams,
    env: Environment,
    s: RigidBodyState,
    u: SurfaceControls,
    wind_ned: Array,
    gust_body: Array | None = None,
    v_supply: Array | None = None,
    motor_cut: Array | None = None,
) -> RigidBodyState:
    fm = forces_moments(p, env, s, u, wind_ned, gust_body, v_supply, motor_cut)
    return rigid_body.derivatives(p, s, fm.force_body, fm.moment_body)


def step(
    p: AirframeParams,
    env: Environment,
    s: RigidBodyState,
    u: SurfaceControls,
    wind_ned: Array,
    dt: float,
    gust_body: Array | None = None,
    v_supply: Array | None = None,
    motor_cut: Array | None = None,
) -> RigidBodyState:
    """Advance one RK4 step with zero-order-hold inputs, then renormalise the quaternion."""

    def f(x: RigidBodyState) -> RigidBodyState:
        return state_derivative(p, env, x, u, wind_ned, gust_body, v_supply, motor_cut)

    nxt = rk4(f, s, dt)
    return nxt._replace(quat=quaternion.normalize(nxt.quat))
