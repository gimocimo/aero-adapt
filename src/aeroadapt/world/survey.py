"""The safe-zone survey of a world (docs/07 §5).

Every landing ground the world names becomes a safe zone: its field inset 15 m from the
hedges, its (level) ground height, and, for landing along its long axis either way, the
**displaced threshold**. As at an aerodrome, an approach surface rises from the threshold
at 1:15 (about 3.8°); every obstacle under it, hedge, tree, building or rising ground, must
sit at least 3 m below it within a corridor 20 m either side of the centre line, widening by
10 % of the distance. The threshold is the first point of the zone where that holds: the
glide may aim at the ground no earlier.

1:15 is flatter than the gliding Aerosonde can fly on final (L/D at most 13.5 with its
windmilling propeller, after the stack's 0.92 efficiency), so an aircraft that reaches the
aim point on the glide passes above the surface, and the obstacles, all the way in.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from shapely.geometry import Polygon

from aeroadapt.core.polygons import principal_chords
from aeroadapt.world.generate import ZONE_INSET_M, World

APPROACH_SLOPE = 15.0  # horizontal metres per metre of height
APPROACH_CLEARANCE_M = 3.0
CORRIDOR_HALF_M = 20.0
CORRIDOR_SPLAY = 0.10
LOOK_BACK_M = 900.0  # how far out along the approach obstacles are surveyed
STEP_M = 2.0

QUALITY = {
    "airstrip": 1.0,
    "pasture": 0.9,
    "meadow": 0.85,
    "barley": 0.8,
    "stubble": 0.8,
    "wheat": 0.7,
    "oilseed": 0.6,
    "ploughed": 0.5,
}
SURFACE = {
    "airstrip": "mown grass airstrip",
    "pasture": "grazed pasture",
    "meadow": "hay meadow",
    "barley": "barley",
    "stubble": "stubble",
    "wheat": "winter wheat",
    "oilseed": "oilseed rape",
    "ploughed": "ploughed",
}


def _threshold(world: World, centre: np.ndarray, d: np.ndarray, hl: float, elev: float) -> float:
    """Distance before the centre of the displaced threshold for landing along ``d``
    (inf: no obstacle under the approach surface)."""
    obs = world.obstacles()
    right = np.array([-d[1], d[0]])
    s = np.arange(-hl - LOOK_BACK_M, -hl + STEP_M, STEP_M)  # along d, from the centre
    worst = -np.inf
    for k in range(-12, 13):
        frac = k / 12.0
        half = CORRIDOR_HALF_M + CORRIDOR_SPLAY * np.abs(s + hl)
        p = centre[None, :] + s[:, None] * d[None, :] + (frac * half)[:, None] * right[None, :]
        ground = world.dtm.at(p[:, 0], p[:, 1])
        top = ground + obs.at(p[:, 0], p[:, 1]) - elev
        need = s + APPROACH_SLOPE * (top + APPROACH_CLEARANCE_M)  # the earliest aim point
        stand = top > 0.25  # something above the zone's level
        if stand.any():
            worst = max(worst, float(need[stand].max()))
    return -worst  # metres before the centre; inf when nothing stands


def survey_zones(world: World) -> list[dict]:
    out = []
    for plot in world.zone_plots():
        spec = next(z for z in world.spec.zones if z.id == plot.zone)
        inner = plot.polygon.buffer(-ZONE_INSET_M, join_style="mitre")
        if inner.geom_type != "Polygon":
            inner = max(getattr(inner, "geoms", [inner]), key=lambda g: g.area)
        inner = inner.minimum_rotated_rectangle if plot.kind == "airstrip" else inner
        ring = np.asarray(cast(Polygon, inner.simplify(2.0)).exterior.coords)[:-1]
        centre, axis, hl, _ = principal_chords(ring)
        elev = float(world.dtm.at(centre[0], centre[1]))
        d = np.array([math.cos(axis), math.sin(axis)])
        thr = [_threshold(world, centre, sign * d, hl, elev) for sign in (1.0, -1.0)]
        cover = plot.cover
        out.append({
            "id": spec.id,
            "polygon_ne_m": [(round(float(n), 2), round(float(e), 2)) for n, e in ring],
            "surface": f"{SURFACE.get(cover, cover)} ({spec.name or plot.name or 'field'})",
            "quality": QUALITY.get(cover, 0.6),
            "elevation_m": round(elev, 2),
            "threshold_back_m": tuple(None if math.isinf(t) else round(t, 1) for t in thr),
        })  # fmt: skip
    return out
