"""Lateral guidance: L1 nonlinear guidance (Park, Deyst & How 2004) with ArduPilot's
``AP_L1_Control`` parameterisation (period, damping). Pure JAX, 2-D in the NE plane.

2-D cross product convention: ``cross(a, b) = a_n·b_e − a_e·b_n``. A positive lateral
acceleration demand means "turn right" (roll right).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from aeroadapt.stack.config import L1Params

TWO_PI = 2.0 * jnp.pi


def cross2(a: Array, b: Array) -> Array:
    return a[0] * b[1] - a[1] * b[0]


def _unit(v: Array) -> Array:
    return v / jnp.maximum(jnp.linalg.norm(v), 1e-6)


class LateralOut(NamedTuple):
    lat_acc: Array  # m/s², + = right
    xtrack_err: Array  # m, + = right of track (waypoint) / outside circle (loiter)
    nav_bearing: Array  # rad, direction the controller is steering towards


def l1_distance(p: L1Params, groundspeed: Array) -> Array:
    return jnp.maximum(p.damping * p.period * groundspeed / jnp.pi, 0.1)


def l1_waypoint(p: L1Params, pos: Array, vel: Array, a: Array, b: Array) -> LateralOut:
    """Track the straight leg A → B (AP_L1_Control::update_waypoint)."""
    v = jnp.maximum(jnp.linalg.norm(vel), 0.1)
    l1 = l1_distance(p, v)
    k_l1 = 4.0 * p.damping**2
    ab = _unit(b - a)
    a_air = pos - a
    # Far behind A (beyond L1 and more than 135° off the leg direction): capture A first.
    dist_a = jnp.linalg.norm(a_air)
    behind = jnp.logical_and(dist_a > l1, (a_air @ ab) / jnp.maximum(dist_a, 1.0) < -0.7071)
    to_a = _unit(-a_air)
    nu_capture = jnp.arctan2(cross2(vel, to_a), vel @ to_a)
    # On the leg: cross-track geometry (Nu1) plus velocity angle (Nu2).
    nu2 = jnp.arctan2(cross2(vel, ab), vel @ ab)
    xtrack = cross2(a_air, ab)  # + = left of track (aircraft to the left of AB)
    nu1 = jnp.arcsin(jnp.clip(xtrack / l1, -0.7071, 0.7071))
    nu = jnp.where(behind, nu_capture, nu1 + nu2)
    nu = jnp.clip(nu, -jnp.pi / 2, jnp.pi / 2)
    lat_acc = k_l1 * v**2 / l1 * jnp.sin(nu)
    bearing = jnp.arctan2(ab[1], ab[0])
    return LateralOut(lat_acc, -xtrack, bearing)


def l1_loiter(
    p: L1Params, pos: Array, vel: Array, centre: Array, radius: Array, direction: Array
) -> LateralOut:
    """Circle ``centre`` at ``radius``; ``direction`` +1 = clockwise (right turns), -1 = ccw.

    Inside or near the circle: PD on the radial error plus the centripetal feed-forward
    (as in AP_L1_Control::update_loiter). Far outside: L1 capture towards the centre.
    """
    v = jnp.maximum(jnp.linalg.norm(vel), 0.1)
    l1 = l1_distance(p, v)
    k_l1 = 4.0 * p.damping**2
    omega = TWO_PI / p.period
    kx, kv = omega**2, 2.0 * p.damping * omega
    r_vec = pos - centre
    r = jnp.maximum(jnp.linalg.norm(r_vec), 1e-3)
    r_unit = r_vec / r
    tangent = direction * jnp.array([-r_unit[1], r_unit[0]])
    xtrack = r - radius  # + = outside
    v_radial = vel @ r_unit  # + = moving outwards
    v_tangent = vel @ tangent
    acc_circle = direction * (
        kx * xtrack + kv * v_radial + v_tangent**2 / jnp.maximum(0.5 * radius, radius + xtrack)
    )
    to_c = -r_unit
    nu = jnp.clip(jnp.arctan2(cross2(vel, to_c), vel @ to_c), -jnp.pi / 2, jnp.pi / 2)
    acc_capture = k_l1 * v**2 / l1 * jnp.sin(nu)
    use_capture = xtrack > l1
    lat_acc = jnp.where(use_capture, acc_capture, acc_circle)
    bearing = jnp.arctan2(tangent[1], tangent[0])
    return LateralOut(lat_acc, xtrack, bearing)


def heading_hold(heading: Array, target: Array, groundspeed: Array, gain: float = 0.6) -> Array:
    """Lateral acceleration to turn towards ``target`` heading (used for straight-in landings)."""
    err = (target - heading + jnp.pi) % TWO_PI - jnp.pi
    return gain * jnp.maximum(groundspeed, 5.0) * err


def roll_from_lat_acc(lat_acc: Array, roll_limit: Array | float, gravity: float = 9.80665) -> Array:
    return jnp.clip(jnp.arctan(lat_acc / gravity), -roll_limit, roll_limit)
