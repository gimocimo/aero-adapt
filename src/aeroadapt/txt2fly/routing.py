"""Clearance and routing over a site (Txt2Fly spec 06 §6).

A clearance map holds, on a 10 m grid over the operating area, the distance from each point to
the nearest no-fly area or the geofence's edge (negative inside a no-fly area or outside the
fence). The planner uses it to try candidates quickly and to route around no-fly areas; the
checker still measures every final track exactly. An area closed only up to a height counts
for a track flown too low to cross it: there is a map for each set of areas closed at a
height (``clearance_map(gaz, alt)``).
"""

from __future__ import annotations

import heapq
import math
from typing import cast

import numpy as np
from scipy import ndimage

from aeroadapt.txt2fly import geometry as geo
from aeroadapt.txt2fly.gazetteer import Gazetteer

GRID_M = 10.0
ROUTE_STEP = 2  # routing searches every 2nd grid cell (20 m)
CORRIDOR_M = 30.0  # a route is searched with this much more room first (fewer, longer legs)


class _Grids:
    """A site's distance grids, one for the geofence and one for each no-fly area (made when
    first needed)."""

    def __init__(self, gaz: Gazetteer) -> None:
        lo = gaz.geofence.min(axis=0) - 100.0
        hi = gaz.geofence.max(axis=0) + 100.0
        self.lo = lo
        self.n = np.arange(lo[0], hi[0] + GRID_M, GRID_M)
        self.e = np.arange(lo[1], hi[1] + GRID_M, GRID_M)
        nn, ee = np.meshgrid(self.n, self.e, indexing="ij")
        self.shape = nn.shape
        self.pts = np.stack([nn.ravel(), ee.ravel()], 1)
        self.fence = -geo.signed_distance(self.pts, gaz.geofence).reshape(self.shape)
        self.gaz = gaz
        self._areas: dict[int, np.ndarray] = {}

    def area(self, k: int) -> np.ndarray:
        if k not in self._areas:
            d = self.gaz.no_fly[k].signed_distance(self.pts).reshape(self.shape)
            self._areas[k] = d.astype(np.float32)
        return self._areas[k]


class ClearanceMap:
    def __init__(self, gaz: Gazetteer, closed: tuple[int, ...] | None = None) -> None:
        """The map for the no-fly areas ``closed`` (indices into ``gaz.no_fly``; None: all)."""
        g = _grids(gaz)
        self.lo, self.n, self.e = g.lo, g.n, g.e
        d = g.fence
        for k in range(len(gaz.no_fly)) if closed is None else closed:
            d = np.minimum(d, g.area(k))
        self.d = d
        self.closed = closed
        self.gaz = gaz
        self._reach: dict[float, np.ndarray] = {}

    def at(self, points: np.ndarray) -> np.ndarray:
        """Clearance at points (bilinear; a lower bound to within half a cell's diagonal)."""
        p = np.atleast_2d(np.asarray(points, float))
        x = (p[:, 0] - self.lo[0]) / GRID_M
        y = (p[:, 1] - self.lo[1]) / GRID_M
        inside = (x >= 0) & (y >= 0) & (x <= len(self.n) - 1) & (y <= len(self.e) - 1)
        x = np.clip(x, 0, len(self.n) - 1.001)
        y = np.clip(y, 0, len(self.e) - 1.001)
        i, j = x.astype(int), y.astype(int)
        fx, fy = x - i, y - j
        d = self.d
        v = (d[i, j] * (1 - fx) * (1 - fy) + d[i + 1, j] * fx * (1 - fy)
             + d[i, j + 1] * (1 - fx) * fy + d[i + 1, j + 1] * fx * fy)  # fmt: skip
        return np.where(inside, v, -1e3)

    def line(self, a: np.ndarray, b: np.ndarray) -> float:
        """Least clearance along the segment a-b."""
        k = max(int(np.linalg.norm(np.asarray(b) - np.asarray(a)) / 5.0), 2)
        return float(self.at(np.linspace(a, b, k)).min())

    def reachable(self, need: float) -> np.ndarray:
        """Grid cells with ``need`` of clearance that connect to home without passing closer
        (a pocket between a motorway and a campus is clear but out of reach)."""
        key = round(need, 1)
        if key not in self._reach:
            free = self.d >= need
            found = cast(tuple, ndimage.label(free, structure=np.ones((3, 3))))
            labels = np.asarray(found[0])
            nn, ee = np.meshgrid(self.n, self.e, indexing="ij")
            if not free.any():
                self._reach[key] = free
            else:
                home = self.gaz.home
                k = np.argmin(np.where(free, (nn - home[0]) ** 2 + (ee - home[1]) ** 2, np.inf))
                self._reach[key] = labels == labels.ravel()[k]
        return self._reach[key]

    def at_reachable(self, points: np.ndarray, need: float) -> np.ndarray:
        """Whether each point's nearest grid cell is reachable at ``need``."""
        p = np.atleast_2d(np.asarray(points, float))
        i = np.clip(np.round((p[:, 0] - self.lo[0]) / GRID_M).astype(int), 0, len(self.n) - 1)
        j = np.clip(np.round((p[:, 1] - self.lo[1]) / GRID_M).astype(int), 0, len(self.e) - 1)
        return self.reachable(need)[i, j]

    def nearest_clear(self, p: np.ndarray, need: float) -> np.ndarray | None:
        """The reachable grid point nearest ``p`` with at least ``need`` of clearance."""
        nn, ee = np.meshgrid(self.n, self.e, indexing="ij")
        ok = self.reachable(need)
        if not ok.any():
            return None
        cand = np.stack([nn[ok], ee[ok]], 1)
        return cand[int(np.argmin(np.linalg.norm(cand - np.asarray(p), axis=1)))]

    def route(self, a: np.ndarray, b: np.ndarray, need: float) -> list[np.ndarray] | None:
        """Waypoints (a and b excluded) of the shortest route from a to b that keeps ``need``
        of clearance (A* on a 20 m grid, then pulled taut). [] if the straight line is clear;
        None if no route exists. A start or end inside the margin is allowed (leaving it).
        The search first keeps ``CORRIDOR_M`` more: a path that hugs the clearance it needs
        pulls taut into many short legs, one with room to spare into few."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        if self.line(a, b) >= need:
            return []
        for corridor in (need + CORRIDOR_M, need):
            way = self._route(a, b, need, corridor)
            if way is not None:
                return way
        return None

    def _route(
        self, a: np.ndarray, b: np.ndarray, need: float, corridor: float
    ) -> list[np.ndarray] | None:
        d = self.d[::ROUTE_STEP, ::ROUTE_STEP]
        gn, ge = self.n[::ROUTE_STEP], self.e[::ROUTE_STEP]
        free = d >= corridor
        cell = GRID_M * ROUTE_STEP
        # A start or end in a tight spot: within 300 m of it, the route may keep the clearance
        # the spot itself has (leaving it directly, rather than via a distant clear cell).
        nn, ee = np.meshgrid(gn, ge, indexing="ij")
        for p in (a, b):
            here = float(self.at(p[None])[0])
            if here < corridor:
                near = (nn - p[0]) ** 2 + (ee - p[1]) ** 2 < 300.0**2
                free |= near & (d >= min(here, corridor) - 5.0)

        def idx(p: np.ndarray) -> tuple[int, int]:
            i = int(np.clip(round((p[0] - gn[0]) / cell), 0, len(gn) - 1))
            j = int(np.clip(round((p[1] - ge[0]) / cell), 0, len(ge) - 1))
            return i, j

        def snap(p: np.ndarray) -> tuple[int, int] | None:
            i, j = idx(p)
            if free[i, j]:
                return i, j
            ii, jj = np.nonzero(free)
            if not len(ii):
                return None
            k = int(np.argmin((gn[ii] - p[0]) ** 2 + (ge[jj] - p[1]) ** 2))
            return int(ii[k]), int(jj[k])

        s, g = snap(a), snap(b)
        if s is None or g is None:
            return None
        moves = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                 if (di, dj) != (0, 0)]  # fmt: skip
        h = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])  # noqa: E731
        best = {s: 0.0}
        came: dict[tuple[int, int], tuple[int, int]] = {}
        heap = [(h(s), s)]
        while heap:
            _, c = heapq.heappop(heap)
            if c == g:
                break
            for di, dj, w in moves:
                q = (c[0] + di, c[1] + dj)
                if not (0 <= q[0] < len(gn) and 0 <= q[1] < len(ge)) or not free[q]:
                    continue
                cost = best[c] + w
                if cost < best.get(q, math.inf):
                    best[q], came[q] = cost, c
                    heapq.heappush(heap, (cost + h(q), q))
        if g not in best:
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(came[cells[-1]])
        path = [a] + [np.array([gn[i], ge[j]]) for i, j in reversed(cells)] + [b]
        # Pull taut: from each kept point, jump to the farthest point still in clear sight. A
        # line from or to a tight start or end keeps what the search allowed there.
        here = {0: float(self.at(a[None])[0]), len(path) - 1: float(self.at(b[None])[0])}
        tight = [p for p, c in ((a, here[0]), (b, here[len(path) - 1])) if c < need]

        def enough(k: int, m: int) -> float:
            for p in tight:
                if min(np.linalg.norm(path[k] - p), np.linalg.norm(path[m] - p)) < 300.0:
                    ends = self.at(np.array([path[k], path[m]]))
                    return min(need, float(ends.min()) - 5.0)
            return need

        out, k = [], 0
        while k < len(path) - 1:
            far = k + 1
            for m in range(len(path) - 1, k, -1):
                if self.line(path[k], path[m]) >= enough(k, m) or m == k + 1:
                    far = m
                    break
            if far != len(path) - 1:
                out.append(path[far])
            k = far
        return out


_GRIDS: dict[str, _Grids] = {}
_MAPS: dict[tuple[str, tuple[int, ...]], ClearanceMap] = {}


def _grids(gaz: Gazetteer) -> _Grids:
    key = gaz.site.model_dump_json()
    if key not in _GRIDS:
        _GRIDS[key] = _Grids(gaz)
    return _GRIDS[key]


def closed_at(gaz: Gazetteer, alt: float | None) -> tuple[int, ...]:
    """The no-fly areas (indices) a track at ``alt`` above home keeps out of (None: all)."""
    return tuple(k for k, a in enumerate(gaz.no_fly) if alt is None or a.closed_at(alt))


def clearance_map(gaz: Gazetteer, alt: float | None = None) -> ClearanceMap:
    """The map for a track at ``alt`` above home (None: every area, at all heights)."""
    key = (gaz.site.model_dump_json(), closed_at(gaz, alt))
    if key not in _MAPS:
        _MAPS[key] = ClearanceMap(gaz, key[1])
    return _MAPS[key]
