"""Frame conventions and conversions (ADR-001).

* World frame: NED local tangent plane at the site origin.
* Body frame: FRD (forward, right, down).
* Wind vectors are air-mass velocity (the direction the air moves *towards*); aviation
  "wind from" direction/speed is converted only at the boundary (YAML / UI).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def ned_to_enu(v_ned: ArrayLike) -> Array:
    v = jnp.asarray(v_ned)
    return jnp.stack([v[..., 1], v[..., 0], -v[..., 2]], axis=-1)


def enu_to_ned(v_enu: ArrayLike) -> Array:
    v = jnp.asarray(v_enu)
    return jnp.stack([v[..., 1], v[..., 0], -v[..., 2]], axis=-1)


def frd_to_flu(v_frd: ArrayLike) -> Array:
    """Body FRD → FLU (forward, left, up), the frame CesiumJS uses for models."""
    v = jnp.asarray(v_frd)
    return jnp.stack([v[..., 0], -v[..., 1], -v[..., 2]], axis=-1)


def wind_from_to_ned(from_deg: ArrayLike, speed_mps: ArrayLike) -> Array:
    """Horizontal wind blowing *from* ``from_deg`` (clockwise from true north) → NED vector."""
    chi = jnp.deg2rad(jnp.asarray(from_deg))
    speed = jnp.asarray(speed_mps)
    return jnp.stack([-speed * jnp.cos(chi), -speed * jnp.sin(chi), jnp.zeros_like(speed)], axis=-1)


def ned_to_wind_from(wind_ned: ArrayLike) -> tuple[Array, Array]:
    """Inverse of :func:`wind_from_to_ned` (horizontal part): ``(from_deg in [0, 360), speed)``."""
    w = jnp.asarray(wind_ned)
    speed = jnp.hypot(w[..., 0], w[..., 1])
    from_deg = jnp.mod(jnp.rad2deg(jnp.arctan2(-w[..., 1], -w[..., 0])), 360.0)
    return from_deg, speed
