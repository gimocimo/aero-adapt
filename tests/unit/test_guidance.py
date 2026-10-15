"""L1 guidance on a point-mass kinematic model: signs, convergence, loiter radius."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aeroadapt.stack.config import L1Params
from aeroadapt.stack.guidance import l1_loiter, l1_waypoint, roll_from_lat_acc

P = L1Params(
    period=15.0, damping=0.75, wp_radius=60.0, roll_limit=math.radians(45), loiter_radius=150.0
)
V, DT, G = 25.0, 0.05, 9.80665


def _fly(law, pos0, heading0, n):
    """Coordinated-turn point mass: heading rate = g·tan(φ)/V with φ from the guidance law."""

    def body(x, _):
        pos, psi = x
        vel = V * jnp.array([jnp.cos(psi), jnp.sin(psi)])
        out = law(pos, vel)
        phi = roll_from_lat_acc(out.lat_acc, P.roll_limit)
        psi = psi + G * jnp.tan(phi) / V * DT
        return (pos + vel * DT, psi), (pos, out.xtrack_err, phi)

    _, traj = jax.lax.scan(body, (jnp.asarray(pos0, float), jnp.asarray(heading0, float)), None, n)
    return [np.asarray(t) for t in traj]


def test_turns_towards_the_track() -> None:
    a, b = jnp.array([0.0, 0.0]), jnp.array([1000.0, 0.0])  # northbound leg
    east_of_track = l1_waypoint(P, jnp.array([100.0, 50.0]), jnp.array([V, 0.0]), a, b)
    west_of_track = l1_waypoint(P, jnp.array([100.0, -50.0]), jnp.array([V, 0.0]), a, b)
    assert float(east_of_track.lat_acc) < 0 < float(west_of_track.lat_acc)  # left / right
    assert float(east_of_track.xtrack_err) > 0  # + = right of track


@pytest.mark.parametrize("offset", [-150.0, 80.0])
def test_waypoint_leg_converges(offset: float) -> None:
    a, b = jnp.array([0.0, 0.0]), jnp.array([5000.0, 0.0])
    _, xtrack, _ = _fly(lambda p, v: l1_waypoint(P, p, v, a, b), [0.0, offset], 0.3, 1600)
    assert np.max(np.abs(xtrack[-400:])) < 1.0  # converged within ~60 s
    assert np.max(np.abs(xtrack)) <= abs(offset) + 30  # no large overshoot


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_loiter_holds_radius_and_direction(direction: float) -> None:
    c = jnp.array([0.0, 0.0])
    law = lambda p, v: l1_loiter(P, p, v, c, jnp.asarray(150.0), jnp.asarray(direction))  # noqa: E731
    _, xtrack, phi = _fly(law, [600.0, -300.0], 0.0, 3000)
    assert np.sqrt(np.mean(xtrack[-1200:] ** 2)) < 5.0
    assert np.sign(phi[-1]) == direction  # clockwise = right bank
