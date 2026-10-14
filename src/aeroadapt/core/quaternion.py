"""Attitude quaternions (ADR-001).

Convention: Hamilton, scalar-first ``q = [e0, e1, e2, e3]``, rotating vectors from the
body frame (FRD) to the NED frame. Euler angles are ZYX (yaw ψ, pitch θ, roll φ).
These match Beard & McLain, "Small Unmanned Aircraft", ch. 2-3 and Appendix B.

All functions are pure ``jax.numpy`` and work under ``jit``/``vmap``; they accept NumPy
arrays too. They are dtype-agnostic (float32 or float64).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def normalize(q: ArrayLike) -> Array:
    """Return the unit quaternion in the direction of ``q``."""
    q = jnp.asarray(q)
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


def conjugate(q: ArrayLike) -> Array:
    q = jnp.asarray(q)
    return q * jnp.array([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def multiply(q: ArrayLike, r: ArrayLike) -> Array:
    """Hamilton product ``q ⊗ r``."""
    q = jnp.asarray(q)
    r = jnp.asarray(r)
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r0, r1, r2, r3 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return jnp.stack(
        [
            q0 * r0 - q1 * r1 - q2 * r2 - q3 * r3,
            q0 * r1 + q1 * r0 + q2 * r3 - q3 * r2,
            q0 * r2 - q1 * r3 + q2 * r0 + q3 * r1,
            q0 * r3 + q1 * r2 - q2 * r1 + q3 * r0,
        ],
        axis=-1,
    )


def to_dcm(q: ArrayLike) -> Array:
    """Direction cosine matrix ``R_nb`` (body → NED) of a unit quaternion."""
    q = jnp.asarray(q)
    e0, e1, e2, e3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    row0 = jnp.stack(
        [e0**2 + e1**2 - e2**2 - e3**2, 2 * (e1 * e2 - e3 * e0), 2 * (e1 * e3 + e2 * e0)], axis=-1
    )
    row1 = jnp.stack(
        [2 * (e1 * e2 + e3 * e0), e0**2 - e1**2 + e2**2 - e3**2, 2 * (e2 * e3 - e1 * e0)], axis=-1
    )
    row2 = jnp.stack(
        [2 * (e1 * e3 - e2 * e0), 2 * (e2 * e3 + e1 * e0), e0**2 - e1**2 - e2**2 + e3**2], axis=-1
    )
    return jnp.stack([row0, row1, row2], axis=-2)


def body_to_ned(q: ArrayLike, v_body: ArrayLike) -> Array:
    """Rotate a body-frame vector into NED."""
    return jnp.einsum("...ij,...j->...i", to_dcm(q), jnp.asarray(v_body))


def ned_to_body(q: ArrayLike, v_ned: ArrayLike) -> Array:
    """Rotate a NED vector into the body frame."""
    return jnp.einsum("...ji,...j->...i", to_dcm(q), jnp.asarray(v_ned))


def from_euler(phi: ArrayLike, theta: ArrayLike, psi: ArrayLike) -> Array:
    """Quaternion from ZYX Euler angles (roll φ, pitch θ, yaw ψ), in radians."""
    phi, theta, psi = jnp.asarray(phi), jnp.asarray(theta), jnp.asarray(psi)
    cp, sp = jnp.cos(phi / 2), jnp.sin(phi / 2)
    ct, st = jnp.cos(theta / 2), jnp.sin(theta / 2)
    cy, sy = jnp.cos(psi / 2), jnp.sin(psi / 2)
    return jnp.stack(
        [
            cy * ct * cp + sy * st * sp,
            cy * ct * sp - sy * st * cp,
            cy * st * cp + sy * ct * sp,
            sy * ct * cp - cy * st * sp,
        ],
        axis=-1,
    )


def to_euler(q: ArrayLike) -> tuple[Array, Array, Array]:
    """ZYX Euler angles ``(phi, theta, psi)`` of a unit quaternion, in radians."""
    q = jnp.asarray(q)
    e0, e1, e2, e3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    phi = jnp.arctan2(2 * (e0 * e1 + e2 * e3), e0**2 + e3**2 - e1**2 - e2**2)
    theta = jnp.arcsin(jnp.clip(2 * (e0 * e2 - e1 * e3), -1.0, 1.0))
    psi = jnp.arctan2(2 * (e0 * e3 + e1 * e2), e0**2 + e1**2 - e2**2 - e3**2)
    return phi, theta, psi


def derivative(q: ArrayLike, omega_body: ArrayLike) -> Array:
    """Quaternion kinematics ``q̇ = ½ q ⊗ [0, ω]`` for body angular rate ``ω`` (rad/s)."""
    omega = jnp.asarray(omega_body)
    pure = jnp.concatenate([jnp.zeros_like(omega[..., :1]), omega], axis=-1)
    return 0.5 * multiply(q, pure)
