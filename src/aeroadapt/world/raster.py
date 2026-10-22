"""Obstacle raster (docs/07 §4): the height of whatever stands on the ground (trees, hedges,
buildings, masts) above the terrain, on a 2 m grid.

Rasterisation is conservative: a cell is marked when its centre lies within half a cell
diagonal of a feature, so every point of every feature falls in a marked cell (the raster
over-states footprints by at most one cell). Heights are rounded up to the next 0.25 m and
stored as uint8 (0 to 63.75 m), small enough to sit on the GPU next to the terrain grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely

from aeroadapt.world.generate import World

CELL_M = 2.0
QUANTUM_M = 0.25
HALF_DIAG = CELL_M * math.sqrt(0.5)


@dataclass(frozen=True)
class Obstacles:
    n0: float  # centre of cell (0, 0)
    e0: float
    cell: float
    quantum: float
    agl: np.ndarray  # (rows, cols) uint8: height above the terrain, in quanta

    def at(self, n: np.ndarray | float, e: np.ndarray | float) -> np.ndarray:
        """Obstacle height above the ground (m) in the cell containing each point."""
        i = np.rint((np.asarray(n, float) - self.n0) / self.cell).astype(int)
        j = np.rint((np.asarray(e, float) - self.e0) / self.cell).astype(int)
        rows, cols = self.agl.shape
        inside = (i >= 0) & (j >= 0) & (i < rows) & (j < cols)
        v = self.agl[np.clip(i, 0, rows - 1), np.clip(j, 0, cols - 1)]
        return np.where(inside, v * self.quantum, 0.0)


class _Canvas:
    def __init__(self, world: World) -> None:
        n0, e0, n1, e1 = world.bounds
        self.n = np.arange(n0 + CELL_M / 2, n1, CELL_M)
        self.e = np.arange(e0 + CELL_M / 2, e1, CELL_M)
        self.top = np.zeros((len(self.n), len(self.e)))  # metres above the terrain
        self.dtm = world.dtm

    def window(self, minx: float, miny: float, maxx: float, maxy: float):
        """Index slices and cell centres covering a bounding box."""
        i0 = max(int(np.searchsorted(self.n, minx - CELL_M)), 0)
        i1 = min(int(np.searchsorted(self.n, maxx + CELL_M)), len(self.n))
        j0 = max(int(np.searchsorted(self.e, miny - CELL_M)), 0)
        j1 = min(int(np.searchsorted(self.e, maxy + CELL_M)), len(self.e))
        nn, ee = np.meshgrid(self.n[i0:i1], self.e[j0:j1], indexing="ij")
        return (slice(i0, i1), slice(j0, j1)), nn, ee

    def put(self, sl, mask: np.ndarray, height: np.ndarray | float) -> None:
        cur = self.top[sl]
        self.top[sl] = np.where(mask, np.maximum(cur, height), cur)


def rasterise(world: World) -> Obstacles:
    c = _Canvas(world)
    # Woods: the canopy is solid.
    for w in world.woods:
        sl, nn, ee = c.window(*w.polygon.bounds)
        mask = shapely.contains_xy(w.polygon.buffer(HALF_DIAG), nn, ee)
        c.put(sl, mask, w.canopy)
    # Trees outside the woods (hedgerows, river banks, gardens): crowns as cylinders.
    t = world.trees
    for k in np.flatnonzero(t.wood < 0):
        r = float(t.crown[k]) + HALF_DIAG
        n, e = float(t.n[k]), float(t.e[k])
        sl, nn, ee = c.window(n - r, e - r, n + r, e + r)
        mask = (nn - n) ** 2 + (ee - e) ** 2 <= r * r
        c.put(sl, mask, float(t.base[k] + t.height[k]) - c.dtm.at(nn, ee))
    # Hedges.
    for h in world.hedges:
        (a, b) = np.asarray(h.line.coords)[[0, -1]]
        reach = h.width / 2 + HALF_DIAG
        sl, nn, ee = c.window(min(a[0], b[0]) - reach, min(a[1], b[1]) - reach,
                              max(a[0], b[0]) + reach, max(a[1], b[1]) + reach)  # fmt: skip
        ab = b - a
        tt = np.clip(((nn - a[0]) * ab[0] + (ee - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
        d2 = (nn - a[0] - tt * ab[0]) ** 2 + (ee - a[1] - tt * ab[1]) ** 2
        c.put(sl, d2 <= reach * reach, h.height)
    # Buildings: gable roofs (ridge along the long axis), flat roofs, towers with spires.
    for bd in world.buildings:
        fp = bd.footprint()
        sl, nn, ee = c.window(*fp.bounds)
        mask = shapely.contains_xy(fp.buffer(HALF_DIAG), nn, ee)
        u = np.array([math.cos(bd.heading), math.sin(bd.heading)])
        x = (nn - bd.n) * u[0] + (ee - bd.e) * u[1]
        y = -(nn - bd.n) * u[1] + (ee - bd.e) * u[0]
        hw = bd.width / 2
        # The highest point of the roof within each cell (half a diagonal nearer the ridge).
        if bd.roof == "gable":
            yy = np.clip(np.abs(y) - HALF_DIAG, 0, hw)
            z = bd.eaves + (bd.ridge - bd.eaves) * (1.0 - yy / hw)
        elif bd.roof == "spire":
            rho = np.maximum(np.hypot(np.clip(x, -hw, hw), np.clip(y, -hw, hw)) - HALF_DIAG, 0)
            z = np.maximum(bd.eaves, bd.eaves + (bd.ridge - bd.eaves) * (1.0 - rho / hw))
        else:
            z = np.full_like(x, bd.ridge)
        c.put(sl, mask, bd.base + z - c.dtm.at(nn, ee))
    # Masts and silos.
    for s in world.structures:
        r = s.radius + HALF_DIAG
        sl, nn, ee = c.window(s.n - r, s.e - r, s.n + r, s.e + r)
        mask = (nn - s.n) ** 2 + (ee - s.e) ** 2 <= r * r
        c.put(sl, mask, s.base + s.height - c.dtm.at(nn, ee))
    q = np.ceil(np.clip(c.top, 0.0, 255 * QUANTUM_M) / QUANTUM_M).astype(np.uint8)
    return Obstacles(float(c.n[0]), float(c.e[0]), CELL_M, QUANTUM_M, q)
