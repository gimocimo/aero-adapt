# pyright: reportAttributeAccessIssue=false, reportReturnType=false
# (shapely's stubs type its results loosely)
"""Shapes for the area and line steps (docs/06 §17): a loop round an area, a stretch of a line
and its offset to one side, survey lines over an area. Pure geometry in metres, north and east;
the builder turns the shapes into mission items and checks them against the no-fly areas.

A mission holds 32 items, so loops and lines are simplified to a few waypoints: each vertex of
a loop is moved only outward (the loop is drawn ``tol`` further out, then simplified by
``tol``), so it never comes closer to the area than asked.
"""

from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

TOLERANCES = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)


def _largest(g: BaseGeometry) -> Polygon:
    parts = [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon"]
    return max(parts, key=lambda p: p.area)


def loop_round(area: BaseGeometry, standoff: float, r_turn: float, max_points: int) -> np.ndarray:
    """A closed loop (k, 2) outside ``area`` by at least ``standoff``, its bends no tighter than
    ``r_turn`` (concave notches narrower than a turn are bridged), with at most ``max_points``
    vertices; the first is not repeated."""
    ring = np.zeros((0, 2))
    for tol in TOLERANCES:
        grown = area.buffer(standoff + tol + r_turn, quad_segs=8).buffer(-r_turn, quad_segs=8)
        outline = _largest(grown).exterior.simplify(tol)
        ring = np.asarray(outline.coords)[:-1]
        if len(ring) <= max_points:
            break
    return ring


def clockwise(ring: np.ndarray) -> bool:
    """Whether a loop runs clockwise as seen on a map (north up, east right)."""
    x, y = ring[:, 1], ring[:, 0]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) < 0.0


def oriented(ring: np.ndarray, cw: bool, start: np.ndarray) -> np.ndarray:
    """The loop running the way asked, starting at its vertex nearest ``start``."""
    if clockwise(ring) != cw:
        ring = ring[::-1]
    k = int(np.argmin(np.linalg.norm(ring - np.asarray(start), axis=1)))
    return np.roll(ring, -k, axis=0)


def enclosing_circle(area: BaseGeometry) -> tuple[np.ndarray, float]:
    """The smallest circle round an area: centre (north, east) and radius."""
    c = shapely.minimum_bounding_circle(area).centroid
    return np.array([c.x, c.y]), float(shapely.minimum_bounding_radius(area))


def section(
    line: LineString, a: np.ndarray | None, b: np.ndarray | None, near: np.ndarray
) -> LineString:
    """The stretch of ``line`` from ``a`` to ``b`` (their nearest points on it). Without ``a``
    it starts at an end: the one nearer ``near``, or, given ``b``, the one farther from ``b``;
    without ``b`` it runs to the other end."""
    length = line.length
    db = None if b is None else line.project(Point(*b))
    if a is not None:
        da = line.project(Point(*a))
    elif db is not None:
        da = 0.0 if db > length / 2 else length
    else:
        ends = np.asarray([line.coords[0], line.coords[-1]])
        da = 0.0 if np.linalg.norm(ends[0] - near) <= np.linalg.norm(ends[1] - near) else length
    if db is None:
        db = length if da < length / 2 else 0.0
    part = substring(line, min(da, db), max(da, db))
    if da > db:
        part = LineString(np.asarray(part.coords)[::-1])
    return part


COMPASS = {"north": (1.0, 0.0), "south": (-1.0, 0.0), "east": (0.0, 1.0), "west": (0.0, -1.0)}


def offset_side(line: LineString, side: str, distance: float) -> LineString:
    """``line`` moved ``distance`` towards a compass side (whichever offset lies that way)."""
    want = np.asarray(COMPASS[side])
    mid = np.asarray(line.interpolate(0.5, normalized=True).coords[0])
    best, score = line, -math.inf
    for d in (distance, -distance):
        cand = line.offset_curve(d, quad_segs=8, join_style="round")
        if cand.is_empty:
            continue
        if cand.geom_type == "MultiLineString":
            cand = max(cand.geoms, key=lambda g: g.length)
        m = np.asarray(cand.interpolate(0.5, normalized=True).coords[0])
        s = float((m - mid) @ want)
        if s > score:
            best, score = cand, s
    # offset_curve keeps the direction for one side and may reverse it for the other
    ends = np.asarray(line.coords)[[0, -1]]
    b = np.asarray(best.coords)
    if np.linalg.norm(b[0] - ends[0]) > np.linalg.norm(b[-1] - ends[0]):
        best = LineString(b[::-1])
    return best


def points_along(line: LineString, max_points: int) -> np.ndarray:
    """A line's shape in at most ``max_points`` vertices (both ends kept)."""
    pts = np.asarray(line.coords)
    for tol in TOLERANCES:
        pts = np.asarray(line.simplify(tol).coords)
        if len(pts) <= max_points:
            break
    return pts


def survey_lines(
    area: BaseGeometry, spacing: float, axis: np.ndarray, r_turn: float, run_out: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Lines ``spacing`` apart across ``area``, along ``axis`` (a unit vector), each run on past
    the area by ``run_out``. Flown in an order the aircraft can turn for: a line, then the
    line ``k`` further on (k spacings at least a turn's width), back and forth, then the
    lines in between (the passes interleave)."""
    poly = _largest(area) if area.geom_type != "Polygon" else area
    across = np.array([-axis[1], axis[0]])
    pts = np.asarray(poly.exterior.coords)
    c = np.asarray(poly.centroid.coords[0])
    proj = (pts - c) @ across
    reach = float(np.max(np.abs((pts - c) @ axis))) + run_out + 10.0
    offsets = np.arange(proj.min() + spacing / 2, proj.max(), spacing)
    if not len(offsets):
        offsets = np.array([0.5 * (proj.min() + proj.max())])
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for o in offsets:
        p = c + o * across
        cut = LineString([p - reach * axis, p + reach * axis]).intersection(poly)
        if cut.is_empty:
            continue
        seg = max(getattr(cut, "geoms", [cut]), key=lambda g: g.length)
        a, b = np.asarray(seg.coords)[[0, -1]]
        if (b - a) @ axis < 0:
            a, b = b, a
        lines.append((a - run_out * axis, b + run_out * axis))
    k = max(1, math.ceil(2.0 * r_turn / spacing))
    order = [i for start in range(k) for i in range(start, len(lines), k)]
    out = []
    for j, i in enumerate(order):
        a, b = lines[i]
        out.append((a, b) if j % 2 == 0 else (b, a))
    return out
