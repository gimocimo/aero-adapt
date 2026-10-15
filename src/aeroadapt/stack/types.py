"""Shared stack types: the guidance target every mode produces, and the mission."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

MAX_WAYPOINTS = 16

# Lateral guidance modes
LAT_DIRECT, LAT_WAYPOINT, LAT_LOITER, LAT_HEADING = 0, 1, 2, 3
# Vertical modes
VERT_DIRECT, VERT_TECS, VERT_GLIDE = 0, 1, 2


class Target(NamedTuple):
    lateral: Array  # LAT_*
    roll_dem: Array  # LAT_DIRECT
    a: Array  # (2,) LAT_WAYPOINT leg start
    b: Array  # (2,) LAT_WAYPOINT leg end
    centre: Array  # (2,) LAT_LOITER
    radius: Array
    direction: Array  # +1 clockwise, -1 counter-clockwise
    heading: Array  # LAT_HEADING
    roll_limit: Array
    vertical: Array  # VERT_*
    pitch_dem: Array  # VERT_DIRECT
    throttle: Array  # VERT_DIRECT
    alt: Array  # VERT_TECS height target (AGL)
    speed: Array  # VERT_TECS / VERT_GLIDE airspeed target


def target(**kw) -> Target:
    """A Target with defaults for unspecified fields (host or traced values)."""
    z2 = jnp.zeros(2)
    base = dict(
        lateral=LAT_DIRECT,
        roll_dem=0.0,
        a=z2,
        b=z2,
        centre=z2,
        radius=100.0,
        direction=1.0,
        heading=0.0,
        roll_limit=0.7854,
        vertical=VERT_DIRECT,
        pitch_dem=0.0,
        throttle=0.0,
        alt=100.0,
        speed=25.0,
    )
    base.update(kw)
    return Target(**{k: jnp.asarray(v) for k, v in base.items()})


class Mission(NamedTuple):
    wps: Array  # (MAX_WAYPOINTS, 3): north, east, alt AGL
    n_wp: Array  # () int
    loop: Array  # () bool
    speed: Array  # () cruise airspeed
    home: Array  # (3,) north, east, alt AGL (RTL loiter)


def make_mission(
    waypoints: Sequence[tuple[float, float, float]],
    *,
    loop: bool = True,
    speed: float = 25.0,
    home: tuple[float, float, float] = (0.0, 0.0, 120.0),
) -> Mission:
    if not 1 <= len(waypoints) <= MAX_WAYPOINTS:
        raise ValueError(f"1..{MAX_WAYPOINTS} waypoints required")
    wps = np.zeros((MAX_WAYPOINTS, 3))
    wps[: len(waypoints)] = np.asarray(waypoints, dtype=float)
    return Mission(
        jnp.asarray(wps),
        jnp.asarray(len(waypoints)),
        jnp.asarray(loop),
        jnp.asarray(speed),
        jnp.asarray(home, dtype=float),
    )
