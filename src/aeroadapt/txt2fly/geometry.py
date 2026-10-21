"""Plane geometry in the site's local north/east metres (NumPy, host side).

Polygons are (K, 2) arrays of (north, east) vertices, without the closing vertex repeated.
"""

from __future__ import annotations

import numpy as np


def signed_area(poly: np.ndarray) -> float:
    """Shoelace area; positive when the vertices run clockwise seen from above (N, E axes)."""
    n, e = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(n * np.roll(e, -1) - np.roll(n, -1) * e))


def centroid(poly: np.ndarray) -> np.ndarray:
    n, e = poly[:, 0], poly[:, 1]
    cross = n * np.roll(e, -1) - np.roll(n, -1) * e
    a = cross.sum() / 2.0
    if abs(a) < 1e-9:
        return poly.mean(axis=0)
    cn = ((n + np.roll(n, -1)) * cross).sum() / (6.0 * a)
    ce = ((e + np.roll(e, -1)) * cross).sum() / (6.0 * a)
    return np.array([cn, ce])


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; vertices in a consistent turning order."""
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(pts) < 3:
        return pts

    def half(seq: np.ndarray) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for p in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _dp(points: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on an open polyline (keeps both ends)."""
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    ab = b - a
    length = np.linalg.norm(ab)
    rel = points[1:-1] - a
    if length < 1e-9:
        d = np.linalg.norm(rel, axis=1)
    else:
        d = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / length
    k = int(np.argmax(d))
    if d[k] <= tol:
        return np.array([a, b])
    left = _dp(points[: k + 2], tol)
    right = _dp(points[k + 1 :], tol)
    return np.vstack([left[:-1], right])


def simplify(poly: np.ndarray, max_vertices: int = 16) -> tuple[np.ndarray, float]:
    """Simplify a closed ring to at most ``max_vertices``; returns it and the tolerance used
    (every original vertex lies within that distance of the simplified ring)."""
    poly = np.asarray(poly, dtype=float)
    if len(poly) <= max_vertices:
        return poly, 0.0
    # Split the ring at its two mutually farthest vertices so both halves keep their ends.
    i = 0
    j = int(np.argmax(np.linalg.norm(poly - poly[i], axis=1)))
    ring = np.vstack([poly[i:], poly[:i]])
    tol = 0.5
    while True:
        a = _dp(ring[: j + 1], tol)
        b = _dp(np.vstack([ring[j:], ring[:1]]), tol)
        out = np.vstack([a[:-1], b[:-1]])
        if len(out) <= max_vertices:
            return out, tol
        tol *= 1.5


def offset_convex(poly: np.ndarray, d: float) -> np.ndarray:
    """Move every edge of a convex polygon outwards by ``d`` (vertices at the intersections
    of the moved edges): the result contains the polygon grown by ``d`` in every direction."""
    poly = np.asarray(poly, dtype=float)
    sign = 1.0 if signed_area(poly) > 0 else -1.0
    out = []
    k = len(poly)
    for i in range(k):
        p0, p1, p2 = poly[i - 1], poly[i], poly[(i + 1) % k]
        lines = []
        for a, b in ((p0, p1), (p1, p2)):
            t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            normal = sign * np.array([t[1], -t[0]])  # outwards
            lines.append((a + d * normal, t))
        (a1, t1), (a2, t2) = lines
        den = t1[0] * t2[1] - t1[1] * t2[0]
        if abs(den) < 1e-9:  # collinear edges: the vertex just moves out with them
            out.append(p1 + (a2 - p1))
            continue
        s = ((a2[0] - a1[0]) * t2[1] - (a2[1] - a1[1]) * t2[0]) / den
        out.append(a1 + s * t1)
    return np.array(out)


def inside(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for (N, 2) points."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    a = np.asarray(poly, dtype=float)
    b = np.roll(a, -1, axis=0)
    pn, pe = pts[:, :1], pts[:, 1:2]
    straddles = (a[None, :, 1] > pe) != (b[None, :, 1] > pe)
    de = np.where(straddles, b[None, :, 1] - a[None, :, 1], 1.0)
    north = a[None, :, 0] + (pe - a[None, :, 1]) * (b[None, :, 0] - a[None, :, 0]) / de
    return (np.sum(straddles & (pn < north), axis=1) % 2) == 1


def distance_to_boundary(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point to the polygon's boundary."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    a = np.asarray(poly, dtype=float)
    ab = np.roll(a, -1, axis=0) - a
    ap = pts[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12)[None], 0.0, 1.0)
    closest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - closest, axis=-1).min(axis=1)


def signed_distance(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance to the boundary, negative inside the polygon."""
    d = distance_to_boundary(points, poly)
    return np.where(inside(points, poly), -d, d)


def long_axis(poly: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit vector along a polygon's longest extent (principal axis of its outline, sampled
    every metre or so) and the half-length of the polygon along it."""
    a = np.asarray(poly, dtype=float)
    b = np.roll(a, -1, axis=0)
    n = np.maximum((np.linalg.norm(b - a, axis=1)).astype(int), 1)
    pts = np.vstack([np.linspace(p, q, k, endpoint=False) for p, q, k in zip(a, b, n, strict=True)])
    c = pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((pts - c).T))
    u = vecs[:, -1] / np.linalg.norm(vecs[:, -1])
    proj = (a - centroid(a)) @ u
    return u, float(max(proj.max(), -proj.min()))


def polyline_buffer(line: np.ndarray, d: float) -> np.ndarray:
    """A convex polygon containing every point within ``d`` of the polyline (the hull of
    its vertices grown by ``d``): conservative for nearly straight lines such as a motorway."""
    return offset_convex(convex_hull(np.asarray(line, dtype=float)), d)


class PolygonSet:
    """The union of many polygons as one edge list: vectorised signed distances, exact outside
    the union and negative inside it (where polygons overlap, the depth inside is to the
    nearest edge of any of them, so only its sign is meaningful). Points farther than
    ``cutoff`` from the set's bounding box get the distance to the box, a lower bound."""

    def __init__(self, polygons: list[np.ndarray] | tuple[np.ndarray, ...], cutoff: float = 500.0):
        polys = [np.asarray(p, dtype=float) for p in polygons]
        self.a = np.vstack(polys)
        self.b = np.vstack([np.roll(p, -1, axis=0) for p in polys])
        sizes = np.array([len(p) for p in polys])
        self.starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        self.lo, self.hi = self.a.min(axis=0), self.a.max(axis=0)
        self.cutoff = cutoff

    def signed_distance(self, points: np.ndarray, chunk: int = 512) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        gap = np.maximum(np.maximum(self.lo - pts, pts - self.hi), 0.0)
        out = np.hypot(gap[:, 0], gap[:, 1])
        near = np.nonzero(out <= self.cutoff)[0]
        ab = self.b - self.a
        ab2 = np.maximum((ab * ab).sum(-1), 1e-12)
        for k in range(0, len(near), chunk):
            idx = near[k : k + chunk]
            p = pts[idx]
            ap = p[:, None, :] - self.a[None]
            t = np.clip((ap * ab[None]).sum(-1) / ab2[None], 0.0, 1.0)
            d2 = ((ap - t[..., None] * ab[None]) ** 2).sum(-1)
            d = np.sqrt(d2.min(axis=1))
            pe = p[:, 1:2]
            straddles = (self.a[None, :, 1] > pe) != (self.b[None, :, 1] > pe)
            de = np.where(straddles, self.b[None, :, 1] - self.a[None, :, 1], 1.0)
            north = self.a[None, :, 0] + (pe - self.a[None, :, 1]) * ab[None, :, 0] / de
            cross = (straddles & (p[:, :1] < north)).astype(np.int32)
            parity = np.add.reduceat(cross, self.starts, axis=1) % 2
            out[idx] = np.where(parity.any(axis=1), -d, d)
        return out
