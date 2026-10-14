"""V1 (part): quaternion and attitude conventions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from aeroadapt.core import quaternion as Q

roll = st.floats(-3.1, 3.1)
pitch = st.floats(-1.55, 1.55)
yaw = st.floats(-3.1, 3.1)


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


@settings(max_examples=200, deadline=None)
@given(roll, pitch, yaw)
def test_dcm_equals_zyx_elementary_rotations(phi: float, theta: float, psi: float) -> None:
    expected = _rz(psi) @ _ry(theta) @ _rx(phi)  # body → NED
    np.testing.assert_allclose(Q.to_dcm(Q.from_euler(phi, theta, psi)), expected, atol=1e-12)


@settings(max_examples=200, deadline=None)
@given(roll, pitch, yaw)
def test_euler_roundtrip(phi: float, theta: float, psi: float) -> None:
    got = np.array(Q.to_euler(Q.from_euler(phi, theta, psi)))
    np.testing.assert_allclose(_wrap(got - np.array([phi, theta, psi])), 0.0, atol=1e-10)


def test_dcm_is_a_rotation() -> None:
    q = Q.normalize(jax.random.normal(jax.random.PRNGKey(0), (500, 4)))
    r = Q.to_dcm(q)
    eye = jnp.broadcast_to(jnp.eye(3), r.shape)
    np.testing.assert_allclose(r @ jnp.swapaxes(r, -1, -2), eye, atol=1e-12)
    np.testing.assert_allclose(jnp.linalg.det(r), 1.0, atol=1e-12)


def test_product_composes_rotations_and_vector_roundtrip() -> None:
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(1), 3)
    q = Q.normalize(jax.random.normal(k1, (200, 4)))
    r = Q.normalize(jax.random.normal(k2, (200, 4)))
    v = jax.random.normal(k3, (200, 3))
    np.testing.assert_allclose(Q.to_dcm(Q.multiply(q, r)), Q.to_dcm(q) @ Q.to_dcm(r), atol=1e-12)
    np.testing.assert_allclose(Q.ned_to_body(q, Q.body_to_ned(q, v)), v, atol=1e-12)
    np.testing.assert_allclose(
        Q.to_dcm(Q.multiply(q, Q.conjugate(q))),
        jnp.broadcast_to(jnp.eye(3), (200, 3, 3)),
        atol=1e-12,
    )


def test_kinematics_match_dcm_rate() -> None:
    """q̇ = ½ q⊗[0, ω] must imply Ṙ = R [ω]×, the DCM kinematics, by an independent route."""
    q = Q.from_euler(0.3, -0.2, 1.1)
    omega = jnp.array([0.4, -0.7, 0.25])
    q_dot = Q.derivative(q, omega)
    r_dot = jnp.einsum("ijk,k->ij", jax.jacfwd(Q.to_dcm)(q), q_dot)
    wx = jnp.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
    np.testing.assert_allclose(r_dot, Q.to_dcm(q) @ wx, atol=1e-12)


def test_known_attitudes() -> None:
    # Yaw 90°: body forward points east.
    np.testing.assert_allclose(
        Q.body_to_ned(Q.from_euler(0.0, 0.0, np.pi / 2), jnp.array([1.0, 0, 0])),
        [0, 1, 0],
        atol=1e-12,
    )
    # Pitch up 30°: body forward has a negative down (i.e. upward) component.
    fwd = Q.body_to_ned(Q.from_euler(0.0, np.pi / 6, 0.0), jnp.array([1.0, 0, 0]))
    np.testing.assert_allclose(fwd, [np.cos(np.pi / 6), 0, -0.5], atol=1e-12)
    # Roll right 45°: the right wing (body +y) points partly down (+d).
    right = Q.body_to_ned(Q.from_euler(np.pi / 4, 0.0, 0.0), jnp.array([0, 1.0, 0]))
    np.testing.assert_allclose(right, [0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-12)
