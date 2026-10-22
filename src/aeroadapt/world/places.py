"""A world's gazetteer entries (docs/07 §6): named places and the areas a site may declare
no-fly, in the same shape as a site's OpenStreetMap extract (``tools/fetch_gazetteer.py``),
so the Txt2Fly gazetteer reads either."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from aeroadapt.world.generate import World

SETTLEMENT_BUFFER_M = 30.0  # houses and gardens, and the people in them
YARD_BUFFER_M = 30.0  # farmyards: people and livestock
MAST_BUFFER_M = 80.0  # a mast's stay wires reach out about half its height


def _ring(poly) -> list[list[float]]:
    return np.round(np.asarray(poly.exterior.coords)[:-1], 1).tolist()


def _pt(p) -> list[float]:
    return [round(float(p.x), 1), round(float(p.y), 1)]


def world_gazetteer(world: World) -> dict:
    """``{"features": [...], "areas": [...], "attribution": None}``."""
    feats: list[dict] = []
    spec = world.spec
    if spec.village is not None:
        v = next(p for p in world.plots if p.kind == "settlement")
        feats.append(
            {
                "id": spec.village.id,
                "name": spec.village.name,
                "kind": "village",
                "point_ne_m": _pt(v.polygon.centroid),
                "polygon_ne_m": _ring(v.polygon.simplify(5.0)),
            }
        )
        green = next((p for p in world.plots if p.kind == "green"), None)
        if green is not None:
            feats.append(
                {
                    "id": f"{spec.village.id}-green",
                    "name": green.name,
                    "kind": "green",
                    "point_ne_m": _pt(green.polygon.centroid),
                }
            )
    # Named buildings; a church's tower joins its nave (the place is the whole church).
    named = [b for b in world.buildings if b.name]
    church = [b for b in named if b.kind in ("nave", "tower")]
    if church:
        nave = next((b for b in church if b.kind == "nave"), church[0])
        shape = unary_union([b.footprint() for b in church])
        feats.append(
            {
                "id": "church",
                "name": nave.name,
                "kind": "church",
                "point_ne_m": [round(nave.n, 1), round(nave.e, 1)],
                "polygon_ne_m": _ring(shape.convex_hull),
                "height_m": round(max(b.base + b.ridge for b in church) - nave.base, 1),
            }
        )
    for b in named:
        if b.kind in ("nave", "tower") or b.name is None:
            continue
        feats.append(
            {
                "id": b.name.lower().replace(" ", "-").replace("'", ""),
                "name": b.name,
                "kind": b.kind,
                "point_ne_m": [round(b.n, 1), round(b.e, 1)],
                "polygon_ne_m": _ring(b.footprint()),
                "height_m": round(b.ridge, 1),
            }
        )
    for f in spec.farms:
        yard = next(p for p in world.plots if p.id == f.id)
        tall = max(
            (
                b.ridge
                for b in world.buildings
                if yard.polygon.buffer(5.0).contains(Point(b.n, b.e))
            ),
            default=0.0,
        )
        feats.append(
            {
                "id": f.id,
                "name": f.name,
                "kind": "farm",
                "point_ne_m": _pt(yard.polygon.centroid),
                "polygon_ne_m": _ring(yard.polygon),
                "height_m": round(tall, 1),
            }
        )
    for w in world.woods:
        feats.append(
            {
                "id": w.id,
                "name": w.name,
                "kind": "wood",
                "point_ne_m": _pt(w.polygon.centroid),
                "polygon_ne_m": _ring(w.polygon.simplify(5.0)),
                "height_m": w.canopy,
            }
        )
    for s in world.structures:
        feats.append(
            {
                "id": s.id,
                "name": s.name,
                "kind": s.kind,
                "point_ne_m": [round(s.n, 1), round(s.e, 1)],
                "height_m": s.height,
            }
        )
    if spec.airstrip is not None:
        strip = next(p for p in world.plots if p.kind == "airstrip")
        feats.append(
            {
                "id": spec.airstrip.id,
                "name": spec.airstrip.name,
                "kind": "airstrip",
                "point_ne_m": _pt(strip.polygon.centroid),
                "polygon_ne_m": _ring(strip.polygon),
            }
        )
    if world.river is not None:
        r = world.river
        feats.append(
            {
                "id": r.id,
                "name": r.name,
                "kind": "river",
                "point_ne_m": _pt(r.line.interpolate(0.5, normalized=True)),
            }
        )
        for road in world.roads:
            x = road.line.intersection(r.line)
            if not x.is_empty and road.name:
                p = x if x.geom_type == "Point" else x.centroid
                feats.append(
                    {
                        "id": f"{road.id}-bridge",
                        "name": f"{road.name} bridge",
                        "kind": "bridge",
                        "point_ne_m": _pt(p),
                    }
                )
    for road in world.roads:
        if road.name:
            feats.append(
                {
                    "id": road.id,
                    "name": road.name,
                    "kind": road.kind,
                    "point_ne_m": _pt(road.line.interpolate(0.5, normalized=True)),
                }
            )
    for p in spec.places:
        feats.append({"id": p.id, "name": p.name, "kind": p.kind, "point_ne_m": list(p.point_ne_m)})

    areas: list[dict] = []
    for p in world.plots:
        if p.kind == "settlement":
            a = p.polygon.buffer(SETTLEMENT_BUFFER_M).simplify(3.0)
            areas.append({"id": f"settlement-{p.id}", "polygons_ne_m": [_ring(a)]})
        elif p.kind == "yard":
            a = p.polygon.buffer(YARD_BUFFER_M).simplify(3.0)
            areas.append({"id": f"yard-{p.id}", "polygons_ne_m": [_ring(a)]})
    for s in world.structures:
        if s.kind == "mast":
            a = Point(s.n, s.e).buffer(MAST_BUFFER_M, quad_segs=6)
            areas.append({"id": f"structure-{s.id}", "polygons_ne_m": [_ring(a)]})
    return {"features": feats, "areas": areas, "attribution": None}


def what_is_at(world: World, n: float, e: float) -> str:
    """What stands, or what lies, at a point: for collision and touchdown reports."""
    p = Point(n, e)
    # Within 4 m: a wingtip (1.5 m) and the obstacle grid's rounding (up to 2.4 m).
    for b in world.buildings:
        if b.footprint().buffer(4.0).contains(p):
            return b.name or f"a {b.kind}"
    for s in world.structures:
        if p.distance(Point(s.n, s.e)) <= s.radius + 4.0:
            return s.name
    for w in world.woods:
        if w.polygon.buffer(4.0).contains(p):
            return f"{w.name} (trees)"
    t = world.trees
    if len(t):
        d = np.hypot(t.n - n, t.e - e) - t.crown
        k = int(np.argmin(d))
        if d[k] <= 4.0:
            return "a tree"
    for h in world.hedges:
        if h.line.distance(p) <= h.width / 2 + 4.0:
            return "a hedge"
    if world.river is not None and world.river.line.distance(p) <= world.river.width / 2:
        return world.river.name
    for r in world.roads:
        if r.line.distance(p) <= r.width / 2:
            return r.name or f"a {r.kind}"
    for plot in world.plots:
        if plot.polygon.contains(p):
            if plot.zone:
                return f"{plot.name or 'field'} (zone {plot.zone})"
            if plot.kind == "field":
                return f"a field ({plot.cover})"
            return plot.name or plot.kind
    return "open ground"
