"""Terrain and obstacles under planned tracks (docs/07 §7).

On a world site the planner knows the ground and everything standing on it, as a ground
station with terrain and obstacle data does. Plan heights stay above home (the frame the
autopilot flies, ArduPlane's relative altitude); along the whole track the aircraft must keep

* at least ``min_agl`` above the ground and at most ``ceiling`` above it, and
* at least 30 m above any tree, hedge, building or mast,

within the 30 m tracking margin either side. The grids here are conservative for that: each
10 m cell holds the highest (or, for the ceiling, the lowest) value within the margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from aeroadapt.scenarios.site import Site

CLEARANCE_M = 30.0  # above any known feature (K4)
MARGIN_M = 30.0  # the tracking margin (K3): the track may stray this far
CELL_M = 10.0


@dataclass(frozen=True)
class Relief:
    n0: float  # centre of cell (0, 0)
    e0: float
    hi_ground: np.ndarray  # highest ground within the margin
    lo_ground: np.ndarray  # lowest ground within the margin
    hi_surface: np.ndarray  # highest ground or obstacle top within the margin
    obstacle: np.ndarray  # whether the highest surface there is an obstacle (not ground)

    def _idx(self, ne: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.atleast_2d(np.asarray(ne, float))
        rows, cols = self.hi_ground.shape
        i = np.clip(np.rint((p[:, 0] - self.n0) / CELL_M).astype(int), 0, rows - 1)
        j = np.clip(np.rint((p[:, 1] - self.e0) / CELL_M).astype(int), 0, cols - 1)
        return i, j

    def floor(self, ne: np.ndarray, min_agl: float) -> np.ndarray:
        """The lowest height (above home) a track through each point may fly."""
        i, j = self._idx(ne)
        return np.maximum(self.hi_ground[i, j] + min_agl, self.hi_surface[i, j] + CLEARANCE_M)

    def ceiling(self, ne: np.ndarray, ceiling_agl: float) -> np.ndarray:
        i, j = self._idx(ne)
        return self.lo_ground[i, j] + ceiling_agl

    def by_obstacle(self, ne: np.ndarray, min_agl: float) -> np.ndarray:
        """Whether the floor at each point is set by an obstacle (else by the ground)."""
        i, j = self._idx(ne)
        return self.hi_surface[i, j] + CLEARANCE_M > self.hi_ground[i, j] + min_agl

    def ground(self, ne: np.ndarray) -> np.ndarray:
        """The highest ground within the margin (what min_agl is measured from)."""
        i, j = self._idx(ne)
        return self.hi_ground[i, j]

    def top(self, ne: np.ndarray) -> np.ndarray:
        """The highest ground or obstacle top within the margin."""
        i, j = self._idx(ne)
        return self.hi_surface[i, j]


def _disc(radius_cells: int) -> np.ndarray:
    r = radius_cells
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= r * r


_CACHE: dict[str, Relief] = {}


def relief(site: Site) -> Relief | None:
    """The site's relief grids (None on a flat site)."""
    if site.world is None:
        return None
    if site.world not in _CACHE:
        from aeroadapt.world import load_world

        w = load_world(site.world)
        obs = w.obstacles()
        k = round(CELL_M / obs.cell)  # obstacle cells per planner cell
        rows, cols = obs.agl.shape[0] // k, obs.agl.shape[1] // k
        blocks = obs.agl[: rows * k, : cols * k].reshape(rows, k, cols, k)
        top = blocks.max(axis=(1, 3)) * obs.quantum  # highest obstacle in each 10 m cell
        n0 = obs.n0 + (k - 1) * obs.cell / 2
        e0 = obs.e0 + (k - 1) * obs.cell / 2
        nn, ee = np.meshgrid(n0 + CELL_M * np.arange(rows), e0 + CELL_M * np.arange(cols),
                             indexing="ij")  # fmt: skip
        # The ground over each cell: its highest and lowest corner (bilinear between them).
        half = CELL_M / 2
        corners = [w.dtm.at(nn + a, ee + b) for a in (-half, half) for b in (-half, half)]
        g_hi, g_lo = np.max(corners, axis=0), np.min(corners, axis=0)
        fp = _disc(int(np.ceil(MARGIN_M / CELL_M)) + 1)
        surface = g_hi + top
        hi_surface = ndimage.maximum_filter(surface, footprint=fp, mode="nearest")
        hi_ground = ndimage.maximum_filter(g_hi, footprint=fp, mode="nearest")
        _CACHE[site.world] = Relief(
            n0, e0, hi_ground,
            ndimage.minimum_filter(g_lo, footprint=fp, mode="nearest"),
            hi_surface, hi_surface > hi_ground + 0.5,
        )  # fmt: skip
    return _CACHE[site.world]


LANDMARK_KINDS = ("wood", "village", "farm", "church", "mast", "silo", "airstrip", "hangar",
                  "pub", "hall")  # fmt: skip


def landmark(gaz, p: np.ndarray) -> str | None:  # gaz: Gazetteer (no import cycle)
    """The named feature at or near a point, for explanations ("Hanger Wood")."""
    from aeroadapt.txt2fly import geometry as geo

    best, name = 80.0, None
    for place in gaz.places.values():
        if place.kind not in LANDMARK_KINDS:
            continue
        if place.polygon is not None:
            if geo.inside(p[None], place.polygon)[0]:
                return place.name
            d = float(geo.distance_to_boundary(p[None], place.polygon)[0])
        else:
            d = float(np.linalg.norm(place.point - p))
        if d < best:
            best, name = d, place.name
    return name


def climb_out(alt: np.ndarray, floor: np.ndarray) -> int:
    """Where a track that starts below the floor (the aircraft is already too low: nothing a
    plan can change) first reaches it; 0 if it starts above. A track that never reaches the
    floor gets no allowance (0), so a plan that stays low still fails."""
    if alt.size == 0 or alt[0] >= floor[0] - 1.0:
        return 0
    up = np.nonzero(alt >= floor - 1.0)[0]
    return int(up[0]) if up.size else 0


def ceil5(x: float) -> float:
    return float(5.0 * np.ceil(x / 5.0 - 1e-9))
