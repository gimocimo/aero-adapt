"""Obstacle raster (docs/07 §4): the height of whatever stands on the ground (trees, hedges,
buildings, masts, turbines, pylons, cooling towers, wires) above the terrain, on a 2 m grid
(or the grid of a world's measured obstacles, merged in: docs/08 §4).

Rasterisation is conservative: a cell is marked when its centre lies within half a cell
diagonal of a feature, so every point of every feature falls in a marked cell (the raster
over-states footprints by at most one cell). Heights are rounded up to the next 0.25 m and
stored as uint16 (to 16 km), small enough to sit on the GPU next to the terrain grid.

The grid holds one height per cell, so what stands in the air is taken down to the ground:
a wire is a wall up to the wire, a turbine's rotor a wall up to its blade tips, across the
wind. The aircraft cannot pass under either here (it would not in practice, either).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely

from aeroadapt.world.generate import World

CELL_M = 2.0
QUANTUM_M = 0.25


@dataclass(frozen=True)
class Obstacles:
    n0: float  # centre of cell (0, 0)
    e0: float
    cell: float
    quantum: float
    agl: np.ndarray  # (rows, cols) uint16: height above the terrain, in quanta

    def at(self, n: np.ndarray | float, e: np.ndarray | float) -> np.ndarray:
        """Obstacle height above the ground (m) in the cell containing each point."""
        i = np.rint((np.asarray(n, float) - self.n0) / self.cell).astype(int)
        j = np.rint((np.asarray(e, float) - self.e0) / self.cell).astype(int)
        rows, cols = self.agl.shape
        inside = (i >= 0) & (j >= 0) & (i < rows) & (j < cols)
        v = self.agl[np.clip(i, 0, rows - 1), np.clip(j, 0, cols - 1)]
        return np.where(inside, v * self.quantum, 0.0)


class _Canvas:
    def __init__(self, world: World, cell: float) -> None:
        n0, e0, n1, e1 = world.bounds
        self.cell = cell
        self.half = cell * math.sqrt(0.5)  # half a cell's diagonal
        self.n = np.arange(n0 + cell / 2, n1, cell)
        self.e = np.arange(e0 + cell / 2, e1, cell)
        self.top = np.zeros((len(self.n), len(self.e)))  # metres above the terrain
        self.dtm = world.dtm

    def window(self, minx: float, miny: float, maxx: float, maxy: float):
        """Index slices and cell centres covering a bounding box."""
        i0 = max(int(np.searchsorted(self.n, minx - self.cell)), 0)
        i1 = min(int(np.searchsorted(self.n, maxx + self.cell)), len(self.n))
        j0 = max(int(np.searchsorted(self.e, miny - self.cell)), 0)
        j1 = min(int(np.searchsorted(self.e, maxy + self.cell)), len(self.e))
        nn, ee = np.meshgrid(self.n[i0:i1], self.e[j0:j1], indexing="ij")
        return (slice(i0, i1), slice(j0, j1)), nn, ee

    def put(self, sl, mask: np.ndarray, height: np.ndarray | float) -> None:
        cur = self.top[sl]
        self.top[sl] = np.where(mask, np.maximum(cur, height), cur)


def rasterise(world: World) -> Obstacles:
    m = world.measured
    c = _Canvas(world, CELL_M if m is None else m.cell)
    half = c.half  # half a cell's diagonal
    # Woods: the canopy is solid.
    for w in world.woods:
        sl, nn, ee = c.window(*w.polygon.bounds)
        mask = shapely.contains_xy(w.polygon.buffer(half), nn, ee)
        c.put(sl, mask, w.canopy)
    # Trees outside the woods (hedgerows, river banks, gardens): crowns as cylinders.
    t = world.trees
    for k in np.flatnonzero(t.wood < 0):
        r = float(t.crown[k]) + half
        n, e = float(t.n[k]), float(t.e[k])
        sl, nn, ee = c.window(n - r, e - r, n + r, e + r)
        mask = (nn - n) ** 2 + (ee - e) ** 2 <= r * r
        c.put(sl, mask, float(t.base[k] + t.height[k]) - c.dtm.at(nn, ee))
    # Hedges.
    for h in world.hedges:
        (a, b) = np.asarray(h.line.coords)[[0, -1]]
        reach = h.width / 2 + half
        sl, nn, ee = c.window(min(a[0], b[0]) - reach, min(a[1], b[1]) - reach,
                              max(a[0], b[0]) + reach, max(a[1], b[1]) + reach)  # fmt: skip
        ab = b - a
        tt = np.clip(((nn - a[0]) * ab[0] + (ee - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
        d2 = (nn - a[0] - tt * ab[0]) ** 2 + (ee - a[1] - tt * ab[1]) ** 2
        c.put(sl, d2 <= reach * reach, h.height)
    # Buildings: gable roofs (ridge along the long axis), flat roofs, towers with spires; a
    # footprint building's spire or dome rises from its centroid.
    for bd in world.buildings:
        fp = bd.footprint()
        sl, nn, ee = c.window(*fp.bounds)
        mask = shapely.contains_xy(fp.buffer(half), nn, ee)
        if bd.outline is not None:
            cn, ce = fp.centroid.x, fp.centroid.y
            reach = max(float(np.hypot(*(np.asarray(bd.outline) - (cn, ce)).T).max()), 1.0)
            rho = np.clip((np.hypot(nn - cn, ee - ce) - half) / reach, 0.0, 1.0)
            rise = bd.ridge - bd.eaves
            if bd.roof == "skillion":
                # A plane falling from the ridge to the eaves along the downslope heading.
                along = (nn * math.cos(bd.slope) + ee * math.sin(bd.slope)) - half
                ring = np.asarray(bd.outline)
                d = ring[:, 0] * math.cos(bd.slope) + ring[:, 1] * math.sin(bd.slope)
                t = np.clip((along - d.min()) / max(float(np.ptp(d)), 1e-6), 0.0, 1.0)
                z = bd.ridge - rise * t
            elif bd.roof == "spire":
                z = bd.eaves + rise * (1.0 - rho)
            elif bd.roof == "dome":
                z = bd.eaves + rise * np.sqrt(1.0 - rho**2)
            else:
                z = np.full_like(nn, bd.ridge)
            c.put(sl, mask, bd.base + z - c.dtm.at(nn, ee))
            continue
        u = np.array([math.cos(bd.heading), math.sin(bd.heading)])
        x = (nn - bd.n) * u[0] + (ee - bd.e) * u[1]
        y = -(nn - bd.n) * u[1] + (ee - bd.e) * u[0]
        hw = bd.width / 2
        # The highest point of the roof within each cell (half a diagonal nearer the ridge).
        if bd.roof == "gable":
            yy = np.clip(np.abs(y) - half, 0, hw)
            z = bd.eaves + (bd.ridge - bd.eaves) * (1.0 - yy / hw)
        elif bd.roof == "spire":
            rho = np.maximum(np.hypot(np.clip(x, -hw, hw), np.clip(y, -hw, hw)) - half, 0)
            z = np.maximum(bd.eaves, bd.eaves + (bd.ridge - bd.eaves) * (1.0 - rho / hw))
        else:
            z = np.full_like(x, bd.ridge)
        c.put(sl, mask, bd.base + z - c.dtm.at(nn, ee))
    # Masts, silos, pylons and cooling towers (cylinders on their foot), turbines (the tower
    # to its hub, the rotor a wall across the wind up to the blade tips).
    for s in world.structures:
        top = s.hub if s.kind == "turbine" else s.height
        r = s.radius + half
        sl, nn, ee = c.window(s.n - r, s.e - r, s.n + r, s.e + r)
        mask = (nn - s.n) ** 2 + (ee - s.e) ** 2 <= r * r
        c.put(sl, mask, s.base + top - c.dtm.at(nn, ee))
        if s.kind == "turbine":
            across = np.array([-math.sin(s.heading), math.cos(s.heading)]) * s.rotor / 2
            a, b = np.array([s.n, s.e]) - across, np.array([s.n, s.e]) + across
            _wall(c, a, b, 1.5, lambda nn, ee, s=s: s.base + s.height - c.dtm.at(nn, ee))
    # Wires: walls up to the wire.
    for w in world.wires:
        p = w.points
        for k in range(len(p) - 1):
            a, b = p[k], p[k + 1]
            _wall(
                c,
                a[:2],
                b[:2],
                0.5,
                lambda nn, ee, a=a, b=b: _along(a, b, nn, ee, c.half) - c.dtm.at(nn, ee),
            )
    # What was measured on the ground (on this same grid).
    if m is not None:
        if m.agl.shape != c.top.shape or abs(m.n0 - c.n[0]) > 1e-6 or abs(m.e0 - c.e[0]) > 1e-6:
            raise ValueError("measured obstacles are not on the world's obstacle grid")
        c.top = np.maximum(c.top, m.agl * m.quantum)
    q = np.ceil(np.clip(c.top, 0.0, 65535 * QUANTUM_M) / QUANTUM_M).astype(np.uint16)
    return Obstacles(float(c.n[0]), float(c.e[0]), c.cell, QUANTUM_M, q)


def _along(
    a: np.ndarray, b: np.ndarray, nn: np.ndarray, ee: np.ndarray, half_diag: float
) -> np.ndarray:
    """The height of the segment a → b (n, e, z) over each point: the higher end's within a
    cell's reach of it, else linear between them."""
    ab = b[:2] - a[:2]
    t = np.clip(((nn - a[0]) * ab[0] + (ee - a[1]) * ab[1]) / max(float(ab @ ab), 1e-9), 0, 1)
    return (
        a[2]
        + t * (b[2] - a[2])
        + abs(b[2] - a[2]) * min(1.0, 2 * half_diag / max(float(np.hypot(*ab)), 1e-9))
    )


def _wall(c: _Canvas, a: np.ndarray, b: np.ndarray, half_width: float, height) -> None:
    """Mark the cells within ``half_width`` (and half a diagonal) of the segment a → b (n, e)
    with ``height(nn, ee)`` above the ground."""
    reach = half_width + c.half
    sl, nn, ee = c.window(min(a[0], b[0]) - reach, min(a[1], b[1]) - reach,
                          max(a[0], b[0]) + reach, max(a[1], b[1]) + reach)  # fmt: skip
    ab = b - a
    t = np.clip(((nn - a[0]) * ab[0] + (ee - a[1]) * ab[1]) / max(float(ab @ ab), 1e-9), 0, 1)
    d2 = (nn - a[0] - t * ab[0]) ** 2 + (ee - a[1] - t * ab[1]) ** 2
    mask = d2 <= reach * reach
    if mask.any():
        c.put(sl, mask, np.where(mask, height(nn, ee), 0.0))
