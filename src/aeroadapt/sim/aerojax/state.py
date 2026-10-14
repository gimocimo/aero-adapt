"""Simulator state and control containers (pytrees)."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


class RigidBodyState(NamedTuple):
    pos_ned: Array  # (3,) m, NED relative to the site origin
    vel_body: Array  # (3,) m/s, body FRD velocity relative to the ground (u, v, w)
    quat: Array  # (4,) scalar-first unit quaternion, body → NED
    omega: Array  # (3,) rad/s, body rates (p, q, r)


class SurfaceControls(NamedTuple):
    """Physical control inputs in Beard & McLain sign conventions (ADR-001).

    These are *actual* deflections. From week 2 the actuator pipeline produces them from
    normalised ArduPilot-style commands.
    """

    delta_a: Array  # rad
    delta_e: Array  # rad; positive = trailing edge down (nose-down moment)
    delta_r: Array  # rad
    delta_t: Array  # throttle in [0, 1]


def state_to_vector(s: RigidBodyState) -> Array:
    """Pack into the 13-vector [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]."""
    return jnp.concatenate([s.pos_ned, s.vel_body, s.quat, s.omega], axis=-1)


def state_from_vector(x: ArrayLike) -> RigidBodyState:
    x = jnp.asarray(x)
    return RigidBodyState(x[..., 0:3], x[..., 3:6], x[..., 6:10], x[..., 10:13])


def controls_from_vector(u: ArrayLike) -> SurfaceControls:
    u = jnp.asarray(u)
    return SurfaceControls(u[..., 0], u[..., 1], u[..., 2], u[..., 3])
