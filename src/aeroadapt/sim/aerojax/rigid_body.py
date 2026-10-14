"""6-DoF rigid-body equations of motion (B&M eqs. 3.14-3.17, quaternion attitude)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from aeroadapt.airframes.params import AirframeParams
from aeroadapt.core import quaternion
from aeroadapt.sim.aerojax.state import RigidBodyState


def derivatives(
    p: AirframeParams, s: RigidBodyState, force_body: Array, moment_body: Array
) -> RigidBodyState:
    """State derivative for a body with inertia ``J = [[Jx,0,-Jxz],[0,Jy,0],[-Jxz,0,Jz]]``."""
    pos_dot = quaternion.body_to_ned(s.quat, s.vel_body)
    vel_dot = -jnp.cross(s.omega, s.vel_body) + force_body / p.mass
    quat_dot = quaternion.derivative(s.quat, s.omega)

    gam = p.Jx * p.Jz - p.Jxz**2
    g1 = p.Jxz * (p.Jx - p.Jy + p.Jz) / gam
    g2 = (p.Jz * (p.Jz - p.Jy) + p.Jxz**2) / gam
    g3 = p.Jz / gam
    g4 = p.Jxz / gam
    g5 = (p.Jz - p.Jx) / p.Jy
    g6 = p.Jxz / p.Jy
    g7 = ((p.Jx - p.Jy) * p.Jx + p.Jxz**2) / gam
    g8 = p.Jx / gam

    pr, qr, rr = s.omega[..., 0], s.omega[..., 1], s.omega[..., 2]
    ell, m, n = moment_body[..., 0], moment_body[..., 1], moment_body[..., 2]
    omega_dot = jnp.stack(
        [
            g1 * pr * qr - g2 * qr * rr + g3 * ell + g4 * n,
            g5 * pr * rr - g6 * (pr**2 - rr**2) + m / p.Jy,
            g7 * pr * qr - g1 * qr * rr + g4 * ell + g8 * n,
        ],
        axis=-1,
    )
    return RigidBodyState(pos_dot, vel_dot, quat_dot, omega_dot)
