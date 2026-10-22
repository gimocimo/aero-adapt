# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportOperatorIssue=false, reportReturnType=false, reportOptionalMemberAccess=false
# pyright: reportGeneralTypeIssues=false
# (Drawing tools over shapely, whose set operations are typed as returning a BaseGeometry; the
# code relies on them being polygons or lines at run time, which the checker cannot narrow.)
"""Operations on the sketch (configs/sketches/<name>.yaml ``ops``): the airstrip the UAV flies
from, the fields it may land in, and the airspace it must keep out of.

The airstrip: a grass strip in a field (or two neighbouring fields made one, their hedge
grubbed up) as near the middle of the map as there is one long, level and clear enough,
lined up as near the prevailing wind as the field allows: a mown runway with white markers
at its corners, a windsock beside its middle, a hangar and a clubhouse on an apron near one
end, a track to the road. The trees in the field and in each approach are felled.

Landing fields: of the fields, the empty ones (grass, stubble or ploughed: no standing crop),
big and long enough, level, with no tree standing in them (only by the hedges), away from
power lines, masts, turbines and no-fly zones; the best, spread over the map.

No-fly zones: volumes the UAV must not enter (a plan and the heights between which it
applies): the power station, the city centre (its historic centre and towers) and the
arena at all heights;
the city's built-up core and a corridor along each high-voltage line up to 100 m above the
ground; flat-topped cylinders round each wind turbine and each mast (and its stays). Besides,
a geofence inside the map's edge, and a ceiling over all.
"""

from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points, unary_union

from sketch_city import Building, City, Part, _frame, _lines, _polys, _rect
from sketch_country import Ground

GRASS = ("pasture", "meadow", "stubble")


def _bearing(th: float) -> float:
    return math.degrees(th) % 360


def runway_name(th: float) -> str:
    """Runway designators from its bearing, e.g. 04/22."""
    a = round(_bearing(th) / 10) % 36 or 36
    b = (a + 18 - 1) % 36 + 1
    lo, hi = sorted((a, b))
    return f"{lo:02d}/{hi:02d}"


def _obstacles(city: City):
    wires = [LineString(np.asarray(w["pts"])[:, :2]) for w in city.wires]
    masts = [Point(m["n"], m["e"]).buffer(m["height"] * 0.65 + 10) for m in city.masts]
    return {
        "wires": unary_union(wires) if wires else Polygon(),
        "masts": unary_union(masts) if masts else Polygon(),
        "turbines": unary_union([Point(t["n"], t["e"]).buffer(1) for t in city.turbines])
        if city.turbines
        else Polygon(),
        "homes": unary_union(
            [
                b.parts[0].footprint
                for b in city.buildings
                if b.kind in ("house", "old", "terrace", "block", "modern", "tower")
            ]
        ),
        "buildings": unary_union([b.parts[0].footprint for b in city.buildings]),
    }


def _trees_in(city: City, geom) -> int:
    n = 0
    for arr in (city.forest, city.trees):
        if len(arr):
            n += int(shapely.contains_xy(geom, arr[:, 0], arr[:, 1]).sum())
    return n


def _airstrip(cfg: dict, city: City, ground: Ground, obs: dict, zones):
    """The best field (or pair of neighbouring fields) for the strip, and the strip in it."""
    a = cfg.get("airstrip", {})
    want = float(a.get("runway_m", 320))
    shortest = float(a.get("min_runway_m", 260))
    middle = np.asarray(a.get("near_ne_m", [0, 0]), float)
    wind_axis = math.radians(a.get("wind_from_deg", 225.0)) % math.pi
    fields = [(i, c, p) for i, (c, p, _) in enumerate(city.fields) if c not in ("orchard", "solar")]
    tree = shapely.STRtree([p for _, _, p in fields])
    cands = [([i], p) for i, _, p in fields]
    for k, (i, _, p) in enumerate(fields):  # and pairs that share a long hedge
        for j in tree.query(p.buffer(3)):
            # Neighbours: their edges (a hedge apart) run together for 40 m or more.
            if j > k and p.buffer(2).intersection(fields[j][2].buffer(2)).area > 160:
                u = unary_union([p.buffer(2), fields[j][2].buffer(2)]).buffer(-2)
                if isinstance(u, Polygon):
                    cands.append(([i, fields[j][0]], u))
    nofly = unary_union([z["polygon"] for z in zones])
    closed = unary_union([z["polygon"] for z in zones if z["ceiling_m"] is None])
    best = None
    for idx, u in cands:
        if u.area < 4e4:
            continue
        c, th, length, width = _frame(u)
        if u.area / u.minimum_rotated_rectangle.area < 0.7:
            continue
        run = min(want, length - 60)
        if run < shortest or width < 70:
            continue
        ctr = np.asarray(u.buffer(-25).centroid.coords[0]) if not u.buffer(-25).is_empty else c
        strip = _rect(ctr, th, run, 30)
        while run >= shortest and not u.buffer(-4).contains(strip):
            run -= 10
            strip = _rect(ctr, th, run, 30)
        if run < shortest:
            continue
        ax = np.array([math.cos(th), math.sin(th)])
        ends = np.array([ctr - ax * run / 2, ctr + ax * run / 2, ctr])
        z = ground.at(ground.h, ends)
        if abs(z[1] - z[0]) / run > 0.02 or float(np.mean(ground.at(ground.slope, ends))) > 0.03:
            continue
        approach = unary_union(
            [_rect(ctr + s * ax * (run / 2 + 120), th, 240, 60) for s in (-1, 1)]
        )
        if approach.intersects(obs["buildings"]) or strip.buffer(60).intersects(obs["homes"]):
            continue
        if (
            u.distance(obs["wires"]) < 150
            or u.distance(obs["masts"]) < 300
            or u.distance(obs["turbines"]) < 400
        ):
            continue
        if u.intersects(nofly) or approach.intersects(nofly) or u.distance(closed) < 150:
            continue
        off = abs((th - wind_axis + math.pi / 2) % math.pi - math.pi / 2)  # from the wind's line
        score = (
            float(np.hypot(*(ctr - middle))) / 100 + math.degrees(off) / 12 + 0.5 * (len(idx) - 1)
        )
        if best is None or score < best[0]:
            best = (score, idx, u, ctr, th, run, approach)
    if best is None:
        return None
    _, idx, u, ctr, th, run, approach = best
    ax = np.array([math.cos(th), math.sin(th)])
    side = np.array([-ax[1], ax[0]])
    runway = _rect(ctr, th, run, 18)
    markers = [
        _rect(ctr + s * ax * (run / 2 - 2) + t * side * 9, th, 4, 1.2)
        for s in (-1, 1)
        for t in (-1, 1)
    ]
    markers += [_rect(ctr + s * ax * (run / 2 - 2), th, 1.2, 10) for s in (-1, 1)]  # threshold bars
    # The windsock beside the middle; the apron, hangar and clubhouse near one end.
    for sgn in (1, -1):
        sock = ctr + side * sgn * 32
        apron_c = ctr + ax * (run / 2 - 45) + side * sgn * 48
        apron = _rect(apron_c, th, 55, 40)
        if u.buffer(-3).contains(apron) and u.buffer(-3).contains(Point(sock)):
            break
    buildings = [
        Building([Part(_rect(sock, th, 0.35, 0.35), 0, 6.0)], "steel"),
        Building([Part(_rect(sock + ax * 1.8, th, 3.6, 0.9), 5.0, 5.9)], "windsock"),
        Building(
            [Part(_rect(apron_c + side * sgn * 8 - ax * 8, th, 22, 16), 0, 6.0, "gable", 2.2)],
            "hangar",
            "hangar",
        ),
        Building(
            [Part(_rect(apron_c + side * sgn * 8 + ax * 15, th, 10, 6), 0, 3.2, "gable", 2.0)],
            "house",
        ),
    ]
    roads = unary_union(
        [ln for _, ln in city.roads] + [ln for k, ln, _ in city.streets if k in ("lane", "track")]
    )
    gate = np.asarray(nearest_points(u.exterior, Point(apron_c))[0].coords[0])
    to = np.asarray(nearest_points(Point(gate), roads)[1].coords[0])
    streets = [("track", LineString([apron_c, gate, to]), 1.8)]
    places = [("runway", "", runway), *[("marker", "", m) for m in markers], ("apron", "", apron)]
    return {
        "field": u,
        "fields": idx,
        "runway": runway,
        "heading": _bearing(th),
        "name": runway_name(th),
        "length": run,
        "centre": ctr.tolist(),
        "buildings": buildings,
        "places": places,
        "streets": streets,
        "fell": unary_union([u.buffer(-1.5), approach]),
    }


def _landable(cfg: dict, city: City, ground: Ground, obs: dict, zones, skip, fence) -> list[dict]:
    """The fields a UAV may land in: the best, spread over the map, all inside the geofence."""
    lc = cfg.get("landable", {})
    crops = set(lc.get("crops", GRASS))
    nofly = unary_union([z["polygon"] for z in zones])  # (to land, the UAV comes down through it)
    closed = unary_union([z["polygon"] for z in zones if z["ceiling_m"] is None])  # at all heights
    out = []
    for i, (crop, poly, _) in enumerate(city.fields):
        if (
            crop not in crops
            or i in skip
            or poly.area < lc.get("min_ha", 3.0) * 1e4
            or not fence.contains(poly.buffer(30))
        ):
            continue
        c, th, length, width = _frame(poly)
        if length < lc.get("min_length_m", 200) or width < lc.get("min_width_m", 80):
            continue
        q = np.asarray(poly.exterior.coords)
        if float(np.mean(ground.at(ground.slope, np.vstack([q, [c]])))) > lc.get("max_slope", 0.05):
            continue
        if _trees_in(city, poly.buffer(-lc.get("clear_in_m", 12))) > 0:  # trees by the hedge only
            continue
        if (
            poly.distance(obs["wires"]) < 100
            or poly.distance(obs["masts"]) < 200
            or poly.distance(obs["turbines"]) < 250
        ):
            continue
        if poly.distance(nofly) < lc.get("from_zones_m", 30) or poly.distance(closed) < lc.get(
            "from_closed_m", 100
        ):
            continue
        ax = np.array([math.cos(th), math.sin(th)])
        ends = unary_union(
            [_rect(c + s * ax * (length / 2 + 15), th, 30, width * 0.8) for s in (-1, 1)]
        )
        edge = 0.0  # the tallest tree just past either end
        for arr, height in (
            (city.forest, lambda a: a[:, 3]),
            (city.trees, lambda a: a[:, 3] + 0.9 * a[:, 2]),
        ):
            if len(arr):
                m = shapely.contains_xy(ends, arr[:, 0], arr[:, 1])
                if m.any():
                    edge = max(edge, float(height(arr[m]).max()))
        score = length / 100 + poly.area / 4e4 - edge / 6
        out.append(
            {
                "index": i,
                "polygon": poly,
                "crop": crop,
                "area_ha": round(poly.area / 1e4, 1),
                "length_m": round(length),
                "heading": round(_bearing(th)),
                "name": runway_name(th),
                "centre": c.tolist(),
                "edge_trees_m": round(edge),
                "score": score,
            }
        )
    out.sort(key=lambda f: -f["score"])
    chosen: list[dict] = []
    for f in out:
        if len(chosen) >= lc.get("count", 22):
            break
        if all(
            np.hypot(*(np.asarray(f["centre"]) - g["centre"])) > lc.get("spacing_m", 380)
            for g in chosen
        ):
            chosen.append(f)
    chosen.sort(
        key=lambda f: (-round(f["centre"][0] / 900), f["centre"][1])
    )  # north to south, west to east
    for k, f in enumerate(chosen):
        f["id"] = f"L{k + 1}"
    return chosen


def _island(channels) -> Polygon:
    """The island: the land between the river's arm and its main channel."""
    from scipy.spatial import KDTree

    main = channels[0]
    arm = next((c for c in channels if c.name == "arm"), None)
    if arm is None:
        return Polygon()
    _, (i0, i1) = KDTree(main.line).query([arm.line[0], arm.line[-1]])
    lo, hi = sorted((int(i0), int(i1)))
    return Polygon([*arm.line, *main.line[lo : hi + 1][::-1]]).buffer(0)


def _centre(nz: dict, city: City, channels=None) -> list[dict]:
    """One zone over the city centre: the historic centre and the towers. Its blocks: every
    block with an old-town building or a tower in it, and the sites of the cathedral, castle
    and town hall; each taken out to the middle of the streets round it, blocks a street
    apart joined, corners kept square; the whole island, and the river (with its quays) round
    it and between the blocks, so the centre is one zone across the water."""
    pts = {"old": [], "tower": []}
    for b in city.buildings:
        if b.kind in pts:
            pts[b.kind].append(b.parts[0].footprint.centroid)
    tree = shapely.STRtree([*pts["old"], *pts["tower"]])
    chosen = [blk for blk in city.blocks if len(tree.query(blk, predicate="contains"))]
    names = set(nz.get("sites", ["Cathedral", "Castle", "Town Hall"]))
    chosen += [p for _, n, p in city.places if n in names]
    half, r = nz.get("to_street_m", 5.0), nz.get("join_m", 8.0)
    u = unary_union([p.buffer(half, join_style="mitre") for p in chosen])
    u = u.buffer(r, join_style="mitre").buffer(-r, join_style="mitre")
    across = nz.get("across_river_m", 70.0)
    span = u.buffer(across, join_style="mitre").buffer(-across, join_style="mitre")
    river = span.intersection(
        city.water.buffer(nz.get("quays_m", 45.0), join_style="mitre")
    )  # the water, quays and embankments
    parts = [u, river]
    if channels:  # the whole island, and both channels round it out to the quays on the far banks
        isl = _island(channels)
        reach = nz.get("round_island_m", 60.0)
        parts.append(
            isl.buffer(reach, join_style="mitre").intersection(
                unary_union([isl, city.water.buffer(nz.get("quays_m", 45.0))])
            )
        )
    seam = nz.get("seam_m", 12.0)  # no slit left between the river and the blocks
    u = unary_union(parts).buffer(seam, join_style="mitre").buffer(-seam, join_style="mitre")
    out = []
    for p in sorted(_polys(u), key=lambda g: -g.area):
        if p.area < 2e4:
            continue
        out.append(
            {
                "name": "city centre",
                "group": "city centre (historic centre and towers)",
                "polygon": Polygon(p.exterior).simplify(1.5),
                "floor_m": 0,
                "ceiling_m": None,
            }
        )
    return out


def _city_core(nz: dict, city: City, inside=()) -> list[dict]:
    """One zone over the city's built-up core (its blocks of townhouses, terraces, city
    blocks, the old town and the towers; not the outskirts of detached houses, nor the
    villages), with the zones ``inside`` it (the centre) and the river where it runs through:
    each block taken out to the middle of the streets round it, blocks a street apart joined,
    corners kept square; the river and its quays between the banks; one outer edge, nothing
    left out inside it."""
    half, r = nz.get("to_street_m", 5.0), nz.get("join_m", 8.0)
    u = unary_union([b.buffer(half, join_style="mitre") for b in city.blocks])
    u = u.buffer(r, join_style="mitre").buffer(-r, join_style="mitre")
    u = unary_union([p for p in _polys(u) if p.area >= nz.get("min_ha", 4.0) * 1e4] + list(inside))
    # The parks, greens and squares the blocks wrap round (the arena's grounds among them).
    near = u.buffer(15)
    for kind, _, pl in city.places:
        if kind in ("park", "garden", "green", "square", "pitch"):
            ring = pl.exterior if isinstance(pl, Polygon) else pl.boundary
            if ring.intersection(near).length > nz.get("enclosed", 0.5) * ring.length:
                u = unary_union([u, pl.buffer(half, join_style="mitre")])
    inlet = nz.get("inlet_m", 30.0)  # inlets narrower than twice this closed across their mouths
    u = u.buffer(inlet, join_style="mitre").buffer(-inlet, join_style="mitre")
    across = nz.get("across_river_m", 90.0)
    span = u.buffer(across, join_style="mitre").buffer(-across, join_style="mitre")
    river = span.intersection(city.water.buffer(nz.get("quays_m", 45.0), join_style="mitre"))
    seam = nz.get("seam_m", 12.0)
    u = unary_union([u, river]).buffer(seam, join_style="mitre").buffer(-seam, join_style="mitre")
    p = max(_polys(u), key=lambda g: g.area)
    return [
        {
            "name": "city",
            "group": "city",
            "polygon": Polygon(p.exterior).simplify(1.5),
            "floor_m": 0,
            "ceiling_m": nz.get("ceiling_m", 100.0),
        }
    ]


def _zones(cfg: dict, city: City, ground: Ground, channels=None) -> list[dict]:
    """The no-fly zones: (name, plan, floor and ceiling in metres; None: no ceiling). The
    ceiling is above the ground under each point, or, for a zone with ``datum_m`` (a
    cylinder round a turbine or mast), above the ground at its centre: a flat top."""
    zs: list[dict] = []
    nz = cfg.get("no_fly", {})
    plant = next((p for k, _, p in city.places if k == "plant"), None)
    if plant is not None:
        zs.append({"name": "power station", "polygon": plant, "floor_m": 0, "ceiling_m": None})
    if nz.get("centre"):
        zs += _centre(nz["centre"], city, channels)
    arena = [b for b in city.buildings if b.name == "Stadium"]
    if arena:
        zs.append(
            {
                "name": "arena",
                "polygon": unary_union([p.footprint for p in arena[0].parts]).convex_hull.buffer(
                    30
                ),
                "floor_m": 0,
                "ceiling_m": None,
            }
        )

    def base(p) -> float:  # the ground at a cylinder's centre: its flat top is measured from here
        return round(float(ground.at(ground.h, np.asarray(p, float))[0]), 2)

    for k, t in enumerate(city.turbines):
        zs.append(
            {
                "name": f"turbine {k + 1}",
                "group": "wind turbines",
                "polygon": Point(t["n"], t["e"]).buffer(nz.get("turbine_radius_m", 100.0), 24),
                "floor_m": 0,
                "ceiling_m": nz.get("turbine_ceiling_m", 200.0),
                "datum_m": base((t["n"], t["e"])),
            }
        )
    mz = nz.get("masts", {"ring_m": 20, "above_m": 50})
    for m in city.masts:  # the mast and its stays
        zs.append(
            {
                "name": m["name"],
                "group": "masts",
                "polygon": Point(m["n"], m["e"]).buffer(m["height"] * 0.65 + mz["ring_m"], 24),
                "floor_m": 0,
                "ceiling_m": m["height"] + mz["above_m"],
                "datum_m": base((m["n"], m["e"])),
            }
        )
    pz = nz.get("power_lines", {"half_width_m": 40, "ceiling_m": 100})
    if city.wires:
        lines = unary_union([LineString(np.asarray(w["pts"])[:, :2]) for w in city.wires])
        for p in _polys(lines.buffer(pz["half_width_m"])):
            zs.append(
                {
                    "name": "power line",
                    "group": "power lines",
                    "polygon": p.simplify(5),
                    "floor_m": 0,
                    "ceiling_m": pz["ceiling_m"],
                }
            )
    if nz.get("city"):
        zs += _city_core(nz["city"], city, [z["polygon"] for z in zs if z["name"] == "city centre"])
    return zs


def build_ops(cfg: dict, sk: dict, city: City, h, ns, es, channels=None) -> dict:
    ground = Ground(h, ns, es)
    zones = _zones(cfg, city, ground, channels)
    obs = _obstacles(city)
    (n0, e0), (n1, e1) = sk["bounds_ne_m"]
    inset = cfg.get("geofence_inset_m", 150)
    fence = box(n0 + inset, e0 + inset, n1 - inset, e1 - inset)
    strip = _airstrip(cfg, city, ground, obs, zones)
    skip = set(strip["fields"]) if strip else set()
    landable = _landable(cfg, city, ground, obs, zones, skip, fence)
    return {
        "airstrip": strip,
        "landable": landable,
        "zones": zones,
        "geofence": fence,
        "ceiling_m": cfg.get("ceiling_m", 120.0),
    }


def rezone(cfg: dict, sk: dict, city: City, h, ns, es, state: dict, channels=None) -> list[str]:
    """Draw the no-fly zones again (from ``cfg``) over locked operations, keeping the airstrip
    and the landing fields; returns any field or approach a new zone now takes in."""
    ground = Ground(h, ns, es)
    state["zones"] = _zones(cfg, city, ground, channels)
    state["ceiling_m"] = cfg.get("ceiling_m", state["ceiling_m"])
    state["ceiling_ref"] = cfg.get("ceiling_ref", state.get("ceiling_ref", "ground"))
    lc = cfg.get("landable", {})
    nofly = unary_union([z["polygon"] for z in state["zones"]])
    closed = unary_union([z["polygon"] for z in state["zones"] if z["ceiling_m"] is None])
    problems = [
        f["id"]
        for f in state["landable"]
        if f["polygon"].distance(nofly) < lc.get("from_zones_m", 30)
        or f["polygon"].distance(closed) < lc.get("from_closed_m", 100)
    ]
    s = state["airstrip"]
    if s is not None and (s["field"].intersects(nofly) or s["field"].distance(closed) < 150):
        problems.append("airstrip")
    return problems


def apply_ops(city: City, state: dict) -> None:
    s = state["airstrip"]
    if s is not None:
        u = s["field"]
        shapely.prepare(u)
        city.fields = [f for f in city.fields if f[1].intersection(u).area < 0.5 * f[1].area] + [
            ("airstrip", u, "airstrip")
        ]
        inner = u.buffer(-1.5)
        city.hedges = [
            seg
            for ln in city.hedges
            for seg in ([ln] if not ln.intersects(inner) else _lines(ln.difference(inner)))
        ]
        fell = s["fell"]
        for name in ("forest", "trees"):
            arr = getattr(city, name)
            if len(arr):
                setattr(city, name, arr[~shapely.contains_xy(fell, arr[:, 0], arr[:, 1])])
        city.buildings += s["buildings"]
        city.places += s["places"]
        city.streets += s["streets"]
    city.ops = state


# --- locking ----------------------------------------------------------------------------


def save_ops(state: dict, path) -> None:
    import gzip
    import json

    from sketch_city import _wkt, building_record

    s = state["airstrip"]
    strip = (
        None
        if s is None
        else {
            **{
                k: v
                for k, v in s.items()
                if k not in ("field", "runway", "buildings", "places", "streets", "fell")
            },
            "field": _wkt(s["field"]),
            "runway": _wkt(s["runway"]),
            "fell": _wkt(s["fell"]),
            "buildings": [building_record(b) for b in s["buildings"]],
            "places": [[k, n, _wkt(p)] for k, n, p in s["places"]],
            "streets": [[k, _wkt(ln), hw] for k, ln, hw in s["streets"]],
        }
    )
    data = {
        "airstrip": strip,
        "geofence": _wkt(state["geofence"]),
        "ceiling_m": state["ceiling_m"],
        "ceiling_ref": state.get("ceiling_ref", "ground"),
        "landable": [
            {**{k: v for k, v in f.items() if k != "polygon"}, "polygon": _wkt(f["polygon"])}
            for f in state["landable"]
        ],
        "zones": [
            {**{k: v for k, v in z.items() if k != "polygon"}, "polygon": _wkt(z["polygon"])}
            for z in state["zones"]
        ],
    }
    with gzip.open(path, "wt") as f:
        json.dump(data, f, separators=(",", ":"))


def load_ops(path) -> dict:
    import gzip
    import json

    from sketch_city import building_from

    with gzip.open(path, "rt") as f:
        d = json.load(f)
    g = shapely.from_wkt
    s = d["airstrip"]
    if s is not None:
        s = {
            **s,
            "field": g(s["field"]),
            "runway": g(s["runway"]),
            "fell": g(s["fell"]),
            "buildings": [building_from(b) for b in s["buildings"]],
            "places": [(k, n, g(w)) for k, n, w in s["places"]],
            "streets": [(k, g(w), hw) for k, w, hw in s["streets"]],
        }
    return {
        "airstrip": s,
        "geofence": g(d["geofence"]),
        "ceiling_m": d["ceiling_m"],
        "ceiling_ref": d.get("ceiling_ref", "ground"),
        "landable": [{**f, "polygon": g(f["polygon"])} for f in d["landable"]],
        "zones": [{**z, "polygon": g(z["polygon"])} for z in d["zones"]],
    }


# --- drawing ----------------------------------------------------------------------------


CLOSED, LIMITED, FENCE, FIELD, STRIP = "#e5484d", "#f08a24", "#7a3fb0", "#178a3e", "#2a62c9"


def zone_colour(z: dict) -> str:
    return CLOSED if z["ceiling_m"] is None else LIMITED


def zone_label(z: dict) -> str:
    if z["ceiling_m"] is None:
        return z.get("group", z["name"]) + "\nall heights"
    over = " over its foot" if z.get("datum_m") is not None else ""
    return z.get("group", z["name"]) + f"\nup to {z['ceiling_m']:.0f} m{over}"


def draw_ops(ax, city: City, detail: bool = True) -> None:
    """The operations overlay, top down: no-fly zones (red: at all heights; orange: up to a
    height), landing fields (green, numbered), the airstrip (blue) and the geofence."""
    from matplotlib.patches import Patch
    from matplotlib.patches import Polygon as MPoly

    ops = getattr(city, "ops", None)
    if not ops:
        return
    fs = 8 if detail else 6
    labelled: set[str] = set()
    for z in sorted(ops["zones"], key=lambda z: z["ceiling_m"] is None):  # the closed ones on top
        col = zone_colour(z)
        for p in _polys(z["polygon"]):
            ax.add_patch(
                MPoly(
                    np.asarray(p.exterior.coords)[:, ::-1],
                    closed=True,
                    zorder=9,
                    fc=(*_rgb(col), 0.2),
                    ec=col,
                    lw=1.2,
                )
            )
        group = z.get("group", z["name"])
        if group in labelled:
            continue
        labelled.add(group)
        c = np.asarray(z["polygon"].representative_point().coords[0])
        ax.annotate(
            zone_label(z),
            xy=(c[1], c[0]),
            xytext=(18, 18),
            textcoords="offset points",
            fontsize=fs - 1,
            color="#5a1a10",
            zorder=11,
            annotation_clip=True,
            bbox={"fc": "white", "ec": col, "alpha": 0.85, "pad": 1.5, "lw": 0.6},
            arrowprops={"arrowstyle": "-", "color": col, "lw": 0.7},
        )
    for f in ops["landable"]:
        p = f["polygon"]
        ax.add_patch(
            MPoly(
                np.asarray(p.exterior.coords)[:, ::-1],
                closed=True,
                zorder=9,
                fc=(*_rgb(FIELD), 0.25),
                ec=FIELD,
                lw=1.6,
            )
        )
        c = np.asarray(p.representative_point().coords[0])
        ax.text(
            c[1],
            c[0],
            f["id"],
            fontsize=fs,
            fontweight="bold",
            color="#0d4d24",
            ha="center",
            va="center",
            zorder=11,
            bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 0.8},
            clip_on=True,
        )
    s = ops["airstrip"]
    if s is not None:
        ax.add_patch(
            MPoly(
                np.asarray(s["field"].exterior.coords)[:, ::-1],
                closed=True,
                zorder=9,
                fc=(*_rgb(STRIP), 0.15),
                ec=STRIP,
                lw=1.8,
            )
        )
        ax.add_patch(
            MPoly(
                np.asarray(s["runway"].exterior.coords)[:, ::-1],
                closed=True,
                zorder=9.5,
                fc=STRIP,
                ec="none",
                alpha=0.7,
            )
        )
        c = s["centre"]
        ax.annotate(
            f"airstrip {s['name']} ({s['length']:.0f} m)",
            xy=(c[1], c[0]),
            xytext=(-30, -34),
            textcoords="offset points",
            fontsize=fs,
            fontweight="bold",
            color="#173d80",
            zorder=11,
            bbox={"fc": "white", "ec": STRIP, "alpha": 0.9, "pad": 1.5, "lw": 0.6},
            arrowprops={"arrowstyle": "-", "color": STRIP, "lw": 0.8},
        )
    g = np.asarray(ops["geofence"].exterior.coords)
    ax.plot(g[:, 1], g[:, 0], ls=(0, (6, 4)), color=FENCE, lw=1.3, zorder=9)
    ax.legend(
        handles=[
            Patch(fc=(*_rgb(CLOSED), 0.3), ec=CLOSED, label="no-fly zone, at all heights"),
            Patch(fc=(*_rgb(LIMITED), 0.3), ec=LIMITED, label="no-fly zone, up to a height"),
            Patch(
                fc=(*_rgb(FIELD), 0.3), ec=FIELD, label=f"landing field ({len(ops['landable'])})"
            ),
            Patch(fc=(*_rgb(STRIP), 0.25), ec=STRIP, label="airstrip"),
            Patch(
                fc="none",
                ec=FENCE,
                ls="--",
                label=f"geofence; ceiling {ops['ceiling_m']:.0f} m "
                + ("altitude (flat)" if ops.get("ceiling_ref") == "datum" else "above the ground"),
            ),
        ],
        loc="lower left",
        fontsize=fs,
        framealpha=0.9,
    )


def ceiling_alt(ops: dict) -> float | None:
    """The ceiling as an altitude above the map's datum (flat), if it is one."""
    return ops["ceiling_m"] if ops.get("ceiling_ref") == "datum" else None


def zone_volumes(ops: dict, view: np.ndarray, project, ground, open_top: float | None = None):
    """The no-fly zones and the geofence as see-through volumes for an isometric drawing:
    (behind: polygons and colours drawn before what stands on the ground; in front: after;
    labels: (x, y, text, colour)). Walls follow the ground; a zone with no ceiling is drawn
    up to the map's ceiling."""
    hv = view[:2] / max(np.linalg.norm(view[:2]), 1e-9)
    open_top = ops["ceiling_m"] if open_top is None else open_top  # no ceiling: up to the map's
    behind, front, labels = [], [], []
    done: set[str] = set()
    flat = ceiling_alt(
        ops
    )  # a flat ceiling: zones with no ceiling of their own, and the fence, go up to it
    vols = [
        (
            z["polygon"],
            z["ceiling_m"] if z["ceiling_m"] is not None else open_top,
            zone_colour(z),
            z,
        )
        for z in ops["zones"]
    ]
    vols.append((ops["geofence"], ops["ceiling_m"], FENCE, None))
    for poly, top, col, z in sorted(vols, key=lambda v: -sum(v[0].centroid.coords[0])):
        datum = None if z is None else z.get("datum_m")
        if flat is not None and (z is None or z["ceiling_m"] is None):
            datum, top = 0.0, flat
        for p in _polys(poly):
            ring = np.asarray(shapely.segmentize(p.exterior, 30.0).coords)
            if shapely.is_ccw(p.exterior):
                ring = ring[
                    ::-1
                ]  # clockwise (in north, east): outward normals to the left of travel
            g = np.asarray(ground(ring[:, 0], ring[:, 1]), float)
            lid = (
                np.full(len(ring), datum + top) if datum is not None else g + top
            )  # flat, or following the ground
            a_fill, a_edge = (0.05, 0.5) if z is None else (0.13, 0.75)
            rgb = _rgb(col)
            for k in range(len(ring) - 1):
                d = ring[k + 1] - ring[k]
                nrm = np.array([d[1], -d[0]])  # outward, for a clockwise ring
                q = np.array(
                    [
                        [*ring[k], g[k]],
                        [*ring[k + 1], g[k + 1]],
                        [*ring[k + 1], lid[k + 1]],
                        [*ring[k], lid[k]],
                    ]
                )
                x, y = project(q[:, 0], q[:, 1], q[:, 2])
                quad = np.column_stack([x, y])
                (front if nrm @ hv > 0 else behind).append((quad, (*rgb, a_fill), (0, 0, 0, 0)))
            x, y = project(ring[:, 0], ring[:, 1], g)  # the foot, outlined on the ground …
            behind.append((np.column_stack([x, y]), (*rgb, 0.0), (*rgb, a_edge)))
            x, y = project(ring[:, 0], ring[:, 1], lid)  # … and the top, filled lightly
            front.append((np.column_stack([x, y]), (*rgb, a_fill * 0.8), (*rgb, a_edge)))
        if z is not None and z.get("group", z["name"]) not in done:
            done.add(z.get("group", z["name"]))
            c = np.asarray(poly.representative_point().coords[0])
            gx = (
                datum
                if datum is not None
                else float(np.atleast_1d(ground(np.array([c[0]]), np.array([c[1]])))[0])
            )
            x, y = project(np.array([c[0]]), np.array([c[1]]), np.array([gx + top]))
            labels.append((float(x[0]), float(y[0]), zone_label(z).replace("\n", ", "), col))
    return behind, front, labels


def _rgb(c: str) -> tuple[float, float, float]:
    return tuple(int(c[i : i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]
