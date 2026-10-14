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
from aeroadapt.sim.aerojax.propulsion import PropulsionOutput, propeller
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
) -> ForcesMoments:
    air = air_data(s, wind_ned)
    f_aero, m_aero = aero_forces_moments(p, env.rho, air, s.omega, u)
    weight_ned = jnp.zeros_like(s.pos_ned).at[..., 2].set(p.mass * env.gravity)
    f_grav = quaternion.ned_to_body(s.quat, weight_ned)
    prop = propeller(p, env.rho, air.va, u.delta_t)
    zero = jnp.zeros_like(prop.thrust)
    f_prop = jnp.stack([prop.thrust, zero, zero], axis=-1)
    # The propeller spins positively about +x_body, so the airframe feels -torque (ADR-001).
    m_prop = jnp.stack([-prop.torque, zero, zero], axis=-1)
    return ForcesMoments(f_aero + f_grav + f_prop, m_aero + m_prop, air, prop)


def state_derivative(
    p: AirframeParams,
    env: Environment,
    s: RigidBodyState,
    u: SurfaceControls,
    wind_ned: Array,
) -> RigidBodyState:
    fm = forces_moments(p, env, s, u, wind_ned)
    return rigid_body.derivatives(p, s, fm.force_body, fm.moment_body)


def step(
    p: AirframeParams,
    env: Environment,
    s: RigidBodyState,
    u: SurfaceControls,
    wind_ned: Array,
    dt: float,
) -> RigidBodyState:
    """Advance one RK4 step with zero-order-hold inputs, then renormalise the quaternion."""
    nxt = rk4(lambda x: state_derivative(p, env, x, u, wind_ned), s, dt)
    return nxt._replace(quat=quaternion.normalize(nxt.quat))
