"""Shared stack types: the guidance target every mode produces, and the mission.

A mission is a fixed-shape table of items in the manner of ArduPlane's mission commands
(Txt2Fly spec, doc 06 §8): waypoints (fly-by or fly-over), loiters (turns, time or unlimited),
jumps and RTL, with an optional geofence and battery-reserve RTL. ``make_mission`` builds the
plain waypoint missions of M1 and M2, which fly exactly as before.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

MAX_ITEMS = 32  # mission items
MAX_FENCE = 16  # geofence polygon vertices

# Mission item kinds (the MAVLink commands they correspond to are in ``aeroadapt.txt2fly``).
ITEM_WAYPOINT, ITEM_FLYOVER, ITEM_LOITER_TURNS, ITEM_LOITER_TIME, ITEM_LOITER_UNLIM = 0, 1, 2, 3, 4
ITEM_JUMP, ITEM_RTL = 5, 6
ITEM_NAMES = ("WAYPOINT", "FLYOVER", "LOITER_TURNS", "LOITER_TIME", "LOITER_UNLIM", "JUMP", "RTL")

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
    wps: Array  # (MAX_ITEMS, 3): north, east, alt AGL of each item (loiter: the centre)
    n_wp: Array  # () int, number of items
    loop: Array  # () bool: after the last item, back to the first (else RTL)
    speed: Array  # () cruise airspeed
    home: Array  # (3,) north, east, alt AGL (RTL loiter)
    kind: Array  # (MAX_ITEMS,) int, ITEM_*
    # (MAX_ITEMS, 4): loiter radius (+ clockwise, − anticlockwise); p1 = turns (LOITER_TURNS),
    # seconds (LOITER_TIME) or the target item (JUMP); p2 = jumps to take (−1: forever);
    # airspeed (0: the mission's)
    param: Array
    fence: Array  # (MAX_FENCE, 2) geofence polygon, north/east
    n_fence: Array  # () int, vertices in use (0: no horizontal fence)
    fence_alt: Array  # () maximum height AGL (0: none)
    rtl_mah: Array  # () battery monitor's consumed charge that triggers RTL (0: none)


class Item(NamedTuple):
    """One mission item, as host values."""

    kind: int
    north: float = 0.0
    east: float = 0.0
    alt: float = 0.0
    radius: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    speed: float = 0.0


def mission_from_items(
    items: Sequence[Item],
    *,
    speed: float = 25.0,
    home: tuple[float, float, float] = (0.0, 0.0, 120.0),
    loop: bool = False,
    fence: Sequence[tuple[float, float]] = (),
    fence_alt: float = 0.0,
    rtl_mah: float = 0.0,
) -> Mission:
    if not 1 <= len(items) <= MAX_ITEMS:
        raise ValueError(f"1..{MAX_ITEMS} mission items required")
    if fence and not 3 <= len(fence) <= MAX_FENCE:
        raise ValueError(f"a geofence needs 3..{MAX_FENCE} vertices")
    wps = np.zeros((MAX_ITEMS, 3))
    kind = np.full(MAX_ITEMS, ITEM_WAYPOINT, dtype=np.int32)
    param = np.zeros((MAX_ITEMS, 4))
    for i, it in enumerate(items):
        wps[i] = (it.north, it.east, it.alt)
        kind[i] = it.kind
        param[i] = (it.radius, it.p1, it.p2, it.speed)
    poly = np.zeros((MAX_FENCE, 2))
    poly[: len(fence)] = np.asarray(fence, dtype=float).reshape(-1, 2)
    # NumPy scalars, not Python ones: no weakly typed leaves, so missions built anywhere
    # share one compiled loop (a live mission swap must not recompile).
    return Mission(
        jnp.asarray(wps),
        jnp.asarray(np.int32(len(items))),
        jnp.asarray(np.bool_(loop)),
        jnp.asarray(np.float64(speed)),
        jnp.asarray(home, dtype=float),
        jnp.asarray(kind),
        jnp.asarray(param),
        jnp.asarray(poly),
        jnp.asarray(np.int32(len(fence))),
        jnp.asarray(np.float64(fence_alt)),
        jnp.asarray(np.float64(rtl_mah)),
    )


def mission_to_dict(m: Mission) -> dict:
    """A mission as JSON-able data (session logs), the inverse of :func:`mission_from_dict`."""
    n, nf = int(m.n_wp), int(m.n_fence)
    wps, kind, prm = np.asarray(m.wps), np.asarray(m.kind), np.asarray(m.param)
    return {
        "items": [
            [int(kind[i]), *(float(x) for x in wps[i]), *(float(x) for x in prm[i])]
            for i in range(n)
        ],
        "speed": float(m.speed),
        "home": [float(x) for x in np.asarray(m.home)],
        "loop": bool(m.loop),
        "fence": [[float(a), float(b)] for a, b in np.asarray(m.fence)[:nf]],
        "fence_alt": float(m.fence_alt),
        "rtl_mah": float(m.rtl_mah),
    }


def mission_from_dict(d: dict) -> Mission:
    items = [Item(int(r[0]), *map(float, r[1:8])) for r in d["items"]]
    return mission_from_items(
        items, speed=d["speed"], home=tuple(d["home"]), loop=d["loop"],
        fence=[tuple(v) for v in d["fence"]], fence_alt=d["fence_alt"], rtl_mah=d["rtl_mah"],
    )  # fmt: skip


def make_mission(
    waypoints: Sequence[tuple[float, float, float]],
    *,
    loop: bool = True,
    speed: float = 25.0,
    home: tuple[float, float, float] = (0.0, 0.0, 120.0),
) -> Mission:
    """A plain waypoint mission (fly-by waypoints, then loop or RTL)."""
    items = [Item(ITEM_WAYPOINT, *map(float, w)) for w in waypoints]
    return mission_from_items(items, speed=speed, home=home, loop=loop)
