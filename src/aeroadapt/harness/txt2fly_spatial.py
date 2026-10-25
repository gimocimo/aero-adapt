"""Outcome checks for Txt2Fly's spatial evaluation (``configs/txt2fly/eval_spatial.yaml``):
what would be flown, measured against the map's shapes (``txt2fly.features``), whatever steps
the plan used. The track is the builder's preview: the mission flown by a point-mass aircraft
with the stack's own guidance in the current wind."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from aeroadapt.stack.types import (
    ITEM_JUMP,
    ITEM_LOITER_TIME,
    ITEM_LOITER_TURNS,
    ITEM_LOITER_UNLIM,
    ITEM_WAYPOINT,
)
from aeroadapt.txt2fly import features
from aeroadapt.txt2fly import geometry as geo
from aeroadapt.txt2fly.checker import Report
from aeroadapt.txt2fly.gazetteer import Gazetteer

MARGIN_M = 30.0  # the tracking margin: a stretch closer than this to a no-fly area is not flyable
STEP_M = 25.0  # sampling of lines and areas
LOITERS = (ITEM_LOITER_TURNS, ITEM_LOITER_TIME, ITEM_LOITER_UNLIM)


def _pts(a: np.ndarray) -> Any:
    return shapely.points(np.asarray(a, float))


def _flyable(gaz: Gazetteer, xy: np.ndarray) -> np.ndarray:
    ok = geo.inside(xy, gaz.geofence)
    for a in gaz.no_fly:
        ok &= a.signed_distance(xy) >= MARGIN_M
    return ok


def _shape(gaz: Gazetteer, ref: str) -> BaseGeometry:
    geom = features.shape(gaz.site, ref)
    if geom is None:
        raise ValueError(f"the evaluation names an unknown place: {ref}")
    return geom


def _area(geom: BaseGeometry) -> BaseGeometry:
    return geom.buffer(MARGIN_M) if geom.geom_type in ("Point", "LineString") else geom


def encloses(geom: BaseGeometry, ne: np.ndarray, max_out: float) -> tuple[str | None, np.ndarray]:
    """None if the track goes once round the area (its bearings from the middle cover a full
    turn), outside it and within ``max_out`` of its edge; else why not. Also the samples that
    count (near the area)."""
    area = _area(geom)
    pts = _pts(ne)
    near = shapely.distance(area, pts) <= max_out
    if not near.any():
        return "never comes near it", near
    first = int(np.argmax(near))
    inside = shapely.contains(area, pts[first:])
    if inside.mean() > 0.02:
        return f"flies over it ({inside.mean():.0%} of the way)", near
    c = area.centroid if area.contains(area.centroid) else area.representative_point()
    rel = ne[near] - np.array([c.x, c.y])
    ang = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    turns = float(ang.max() - ang.min()) / (2 * math.pi)  # the bearings it covers round it
    if turns < 0.9:
        return f"goes {turns:.2f} of the way round it within {max_out:g} m", near
    return None, near


def follows(
    gaz: Gazetteer, geom: BaseGeometry, ne: np.ndarray, share: float, within: float
) -> tuple[str | None, np.ndarray]:
    # An area's edge is measured just outside it, where a no-fly area's can be flown.
    line = geom if features.is_line(geom) else geom.buffer(MARGIN_M + 10.0).boundary
    s = np.arange(0.0, line.length, STEP_M)
    xy = shapely.get_coordinates(line.interpolate(s))
    keep = _flyable(gaz, xy)
    track = LineString(ne)
    mask = shapely.distance(line, _pts(ne)) <= within
    if keep.sum() < 3:
        return "no stretch of it can be flown", mask
    got = float((shapely.distance(track, _pts(xy[keep])) <= within).mean())
    if got < share:
        return f"follows {got:.0%} of it (wanted {share:.0%})", mask
    return None, mask


def covers(
    gaz: Gazetteer, geom: BaseGeometry, ne: np.ndarray, share: float, within: float
) -> tuple[str | None, np.ndarray]:
    area = _area(geom)
    x0, y0, x1, y1 = area.bounds
    g = np.stack(np.meshgrid(np.arange(x0, x1, STEP_M), np.arange(y0, y1, STEP_M)), -1)
    g = g.reshape(-1, 2)
    g = g[shapely.contains(area, _pts(g))]
    g = g[_flyable(gaz, g)] if len(g) else g
    mask = shapely.contains(area.buffer(within), _pts(ne))
    if len(g) < 3:
        return "none of it can be flown", mask
    got = float((shapely.distance(LineString(ne), _pts(g)) <= within).mean())
    if got < share:
        return f"covers {got:.0%} of it (wanted {share:.0%})", mask
    return None, mask


def last_circle(report: Report) -> np.ndarray | None:
    """The middle of the plan's last circle, or of its last loop round a place (an ``around``
    step: waypoints and a JUMP back)."""
    b = report.built
    loiters = [k for k, it in enumerate(b.items) if it.kind in LOITERS]
    jumps = [k for k, it in enumerate(b.items) if it.kind == ITEM_JUMP]
    if jumps and (not loiters or jumps[-1] > loiters[-1]):
        loop = [it for it in b.items[int(b.items[jumps[-1]].p1) : jumps[-1]]
                if it.kind == ITEM_WAYPOINT]  # fmt: skip
        if len(loop) >= 3:
            return np.mean([[it.north, it.east] for it in loop], axis=0)
    return (
        None if not loiters else np.array([b.items[loiters[-1]].north, b.items[loiters[-1]].east])
    )


def visits(
    gaz: Gazetteer, refs: list[str], ne: np.ndarray, within: float
) -> tuple[str | None, np.ndarray]:
    pts = _pts(ne)
    k = 0
    mask = np.zeros(len(ne), bool)
    for ref in refs:
        d = shapely.distance(_shape(gaz, ref), pts)
        hit = np.flatnonzero(d[k:] <= within)
        if not len(hit):
            return f"never comes within {within:g} m of {ref} (after the places before it)", mask
        k += int(hit[0])
        mask |= d <= within
    return None, mask


def outcome(expect: dict[str, Any], report: Report, gaz: Gazetteer) -> str | None:
    """None if what would be flown does what the case asks; else why not."""
    pv = report.built.preview
    if pv is None or len(pv.ne) < 2:
        return "nothing would be flown"
    ne = np.asarray(pv.ne, float)
    within = float(expect.get("within_m", 0) or 0)
    masks: list[np.ndarray] = []
    if "encloses" in expect:
        why, m = encloses(_shape(gaz, expect["encloses"]), ne, float(expect.get("max_out_m", 1200)))
        if why:
            return f"around {expect['encloses']}: {why}"
        masks.append(m)
    if "follows" in expect:
        why, m = follows(gaz, _shape(gaz, expect["follows"]), ne, float(expect.get("share", 0.6)),
                         within or 150.0)  # fmt: skip
        if why:
            return f"along {expect['follows']}: {why}"
        masks.append(m)
    if "covers" in expect:
        why, m = covers(gaz, _shape(gaz, expect["covers"]), ne, float(expect.get("share", 0.7)),
                        within or 80.0)  # fmt: skip
        if why:
            return f"over {expect['covers']}: {why}"
        masks.append(m)
    if "visits" in expect:
        why, m = visits(gaz, list(expect["visits"]), ne, within or 150.0)
        if why:
            return why
        masks.append(m)
    if "circles_near" in expect or "circles_at" in expect:
        c = last_circle(report)
        if c is None:
            return "the plan never circles"
        if "circles_near" in expect:
            ref, tol = expect["circles_near"]
            refs = ref if isinstance(ref, list) else [ref]  # any of them
            d = min(float(_shape(gaz, x).distance(Point(*c))) for x in refs)
            if d > tol:
                return f"circles {d:.0f} m from {' or '.join(refs)} (wanted within {tol} m)"
        else:
            at = expect["circles_at"]
            b = math.radians(float(at["bearing"]))
            p = gaz.place(at["place"]).point + float(at["distance"]) * np.array(
                [math.cos(b), math.sin(b)]
            )
            d = float(np.linalg.norm(c - p))
            if d > float(at["tol"]):
                return f"circles {d:.0f} m from the point asked (wanted within {at['tol']} m)"
    if "loop_alt" in expect and masks:
        m = masks[0]
        alt = float(np.median(np.asarray(pv.alt, float)[m])) if m.any() else math.nan
        if not abs(alt - float(expect["loop_alt"])) <= 15.0:
            return f"flies it at {alt:.0f} m (asked {expect['loop_alt']} m)"
    return None
