# pyright: reportAttributeAccessIssue=false, reportReturnType=false
# (shapely's stubs type its results loosely)
"""The shapes of a site's places (docs/06 §17): an outline for an area, a line for a river or a
road, a point otherwise, so that "around the city" or "along the river" can be planned and
judged against the map as it is.

The gazetteer's places come as points from several sources (the site's own list, its safe
zones, its world or OpenStreetMap extract). Their shapes come from, in order:

1. a no-fly area with the same id ("the city" and its no-fly area are the same place);
2. on a world site, the world itself: rivers and named roads as lines; woods, plots and named
   buildings as outlines; a town or village as its buildings drawn together (each grown by
   30 m, the part nearest its point); a wind farm as the ground its turbines sweep; a solar
   farm as its fields of panels;
3. the gazetteer's own outline (safe zones, OpenStreetMap areas);
4. a point.

``nofly:<id>`` names a no-fly area itself.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from aeroadapt.scenarios.site import Site

if TYPE_CHECKING:
    from aeroadapt.txt2fly.gazetteer import NoFlyArea, Place

TOWN_KINDS = ("town", "village", "city", "hamlet")
BUILDING_REACH_M = 700.0  # how far from a town's point its buildings are gathered
BUILDING_GROW_M = 30.0  # each building grown by this to draw a town together


def _poly(ring: np.ndarray) -> Polygon:
    return Polygon(np.asarray(ring, float)).buffer(0)


def _nearest_part(geom: BaseGeometry, p: Point) -> BaseGeometry:
    parts = list(getattr(geom, "geoms", [geom]))
    return min(parts, key=lambda g: g.distance(p))


@functools.cache
def _world_shapes(world_id: str) -> dict[str, BaseGeometry]:
    from aeroadapt.world import load_world

    w = load_world(world_id)
    out: dict[str, BaseGeometry] = {}
    if w.river is not None:
        out[w.river.id] = w.river.line
    for road in w.roads:
        if road.name:
            out[road.id] = road.line
    for wood in w.woods:
        out[wood.id] = wood.polygon
    for plot in w.plots:
        if plot.name:
            out.setdefault(plot.id, plot.polygon)
    for b in w.buildings:
        if b.name:
            out.setdefault(b.name.lower().replace(" ", "-").replace("'", ""), b.footprint())
    for s in w.structures:
        out.setdefault(s.id, Point(s.n, s.e).buffer(max(s.radius, 5.0)))
    centres = np.array([[b.n, b.e] for b in w.buildings]) if w.buildings else np.zeros((0, 2))
    for place in w.spec.places:
        p = Point(*place.point_ne_m)
        if place.kind in TOWN_KINDS and len(centres):
            near = np.linalg.norm(centres - np.asarray(place.point_ne_m), axis=1) < BUILDING_REACH_M
            if near.any():
                grown = unary_union([w.buildings[i].footprint().buffer(BUILDING_GROW_M)
                                     for i in np.flatnonzero(near)])  # fmt: skip
                out[place.id] = _nearest_part(grown, p).simplify(10.0)
        elif place.kind == "wind farm":
            t = [s for s in w.structures if s.kind == "turbine"]
            if t:
                reach = max(s.rotor for s in t) / 2 + 30.0
                hull = MultiPoint([(s.n, s.e) for s in t]).convex_hull
                out[place.id] = hull.buffer(reach).simplify(10.0)
        elif place.kind == "solar farm":
            panels = [
                q.polygon for q in w.plots if q.cover == "solar" and q.polygon.distance(p) < 500
            ]
            if panels:
                out[place.id] = unary_union(panels).simplify(5.0)
    return out


def ring_of(geom: BaseGeometry) -> np.ndarray:
    """One outline for an area: its own, or its convex hull's if it comes in pieces."""
    g = geom if geom.geom_type == "Polygon" else geom.convex_hull
    return np.asarray(g.exterior.coords)[:-1]  # pyright: ignore[reportAttributeAccessIssue]


def enrich(site: Site, places: dict[str, Place], no_fly: tuple[NoFlyArea, ...]) -> dict[str, Place]:
    """The places with their shapes (the rules above: their own outline first, then a no-fly
    area with the same id, then the world's), and a place for each no-fly area that no place
    stands for (one per name: "wind turbines" for all 22)."""
    from aeroadapt.txt2fly.gazetteer import Place

    world = _world_shapes(site.world) if site.world is not None else {}
    nf = {a.id: unary_union([_poly(r) for r in a.polygons]) for a in no_fly}
    out: dict[str, Place] = {}
    for p in places.values():
        polygon, line = p.polygon, p.line
        if polygon is None and line is None:
            g = nf.get(p.id) or world.get(p.id)
            if g is not None and is_line(g):
                line = np.asarray(g.coords if g.geom_type == "LineString" else
                                  max(g.geoms, key=lambda x: x.length).coords)  # fmt: skip
            elif g is not None and g.geom_type in ("Polygon", "MultiPolygon"):
                polygon = ring_of(g.simplify(5.0))
        out[p.id] = replace(p, polygon=polygon, line=line)
    names = {p.name.lower() for p in out.values()}
    groups: dict[str, list[NoFlyArea]] = {}
    for a in no_fly:
        if a.id not in out and a.name.lower() not in names:
            groups.setdefault(a.name, []).append(a)
    for name, areas in groups.items():
        g = unary_union([nf[a.id] for a in areas])
        c = g.representative_point()
        pid = f"nofly-{areas[0].id}"
        out[pid] = Place(pid, name, "no-fly area", np.array([c.x, c.y]),
                         ring_of(g.simplify(5.0)), source="no-fly area")  # fmt: skip
    return out


def no_fly_shape(site: Site, area_id: str) -> BaseGeometry | None:
    """A no-fly area's ground, as one geometry (None if the site has none of that id)."""
    from aeroadapt.txt2fly.gazetteer import load_gazetteer

    gaz = load_gazetteer(site)
    polys = [_poly(p) for a in gaz.no_fly if a.id == area_id for p in a.polygons]
    return unary_union(polys) if polys else None


def shape(site: Site, ref: str) -> BaseGeometry | None:
    """The shape of a place (or ``nofly:<id>``) as the gazetteer has it; None if unknown."""
    from aeroadapt.txt2fly.gazetteer import UnknownPlace, load_gazetteer

    if ref.startswith("nofly:"):
        return no_fly_shape(site, ref.removeprefix("nofly:"))
    try:
        place = load_gazetteer(site).place(ref)
    except UnknownPlace:
        return None
    if place.line is not None:
        return LineString(place.line)
    if place.polygon is not None:
        return _poly(place.polygon)
    return Point(*place.point)


def is_line(geom: BaseGeometry) -> bool:
    return isinstance(geom, LineString) or geom.geom_type == "MultiLineString"
