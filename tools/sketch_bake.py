# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportAssignmentType=false
# pyright: reportOperatorIssue=false
# (Drawing tools over shapely, whose set operations are typed as returning a BaseGeometry.)
"""Bake a map sketch into a world the simulator flies (docs/07 §14): every layer as it is
locked, written as ``configs/worlds/<name>.world.npz`` (read by ``aeroadapt.world.baked``),
its spec ``configs/worlds/<name>.yaml``, and the operations block of
``configs/sites/<name>.yaml`` (home at the airstrip, the geofence and ceiling, the no-fly
zones with their heights). The site's safe zones come from the world's survey: the airstrip
(zone A) and the landing fields (L1, L2, …).

What changes on the way: heights become metres above the site datum, the ground at the
map's centre; each part of a building becomes a box from the ground to its top (with its
roof: gable, flat, or a hipped or spired top); shrubs and fruit trees become small
broadleaves, willows the simulator's willows; the woods' glades are cut out of them.

Usage: ``uv run python tools/sketch_bake.py testmap``
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import shapely
import yaml
from shapely.geometry import Polygon
from shapely.ops import substring, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
import itertools

from aeroadapt.world.baked import ROOFS, save_baked
from aeroadapt.world.generate import Dtm, _funnels
from sketch_city import TONES, _frame, _lines, _polys
from sketch_map import CELL, prepare
from sketch_power import FOOT, PEAK

ROOT = Path(__file__).resolve().parents[1]
HEDGE_SEED = 31
WOOD_CANOPY_M = 24.0  # the woods' solid canopy for the simulator (their trees reach 13-27 m)
TREE_KIND = {0: 0, 1: 1, 2: 0, 3: 2, 4: 0}  # sketch → world: shrub and fruit tree as broadleaf
ROOF = {"gable": "gable", "flat": "flat", "hip": "spire"}
GROUND_COVER = {  # places in the town and on the land, by the colour they are painted
    "park": "park",
    "garden": "park",
    "green": "green",
    "square": "paved",
    "pitch": "pitch",
    "yard": "garden",
    "pad": "gravel",
    "plant": "park",
    "apron": "paved",
    "switchyard": "gravel",
    "car_park": "paved",
    "basin": "water",
    "runway": "runway",
    "marker": "marker",
}
STREETS = {  # sketch street kind → (world road kind, width)
    "street": "street",
    "quay": "street",
    "lane": "lane",
    "track": "track",
    "service": "track",
    "plant_road": "street",
    "access": "lane",
}


def _ring(p: Polygon, step: float = 0.0) -> dict:
    p = p.simplify(step) if step else p
    return {
        "outer": np.round(np.asarray(p.exterior.coords)[:-1], 2).tolist(),
        "holes": [np.round(np.asarray(r.coords)[:-1], 2).tolist() for r in p.interiors],
    }


def bake(name: str) -> dict:
    sk, ns, es, h, channels, _anchors, city = prepare(name)
    assert city is not None and city.ops
    datum = float(Dtm(float(ns[0]), float(es[0]), CELL, h).at(0.0, 0.0))
    dtm = Dtm(float(ns[0]), float(es[0]), CELL, h - datum)

    def ground(n, e) -> np.ndarray:
        return np.atleast_1d(dtm.at(np.asarray(n, float), np.asarray(e, float)))

    meta: dict = {
        "plots": [],
        "woods": [],
        "roads": [],
        "structures": [],
        "wires": [],
        "zones": [],
        "places": [],
    }
    ops = city.ops

    # Landing grounds: the airstrip (A) and the landing fields (L1 …).
    zones: list[tuple[str, Polygon, str, str]] = []
    if ops["airstrip"] is not None:
        a = ops["airstrip"]
        zones.append(("A", a["field"], "airstrip", f"the airstrip ({a['name']})"))
    for f in ops["landable"]:
        zones.append((f["id"], f["polygon"], f["crop"], f"landing field {f['id']}"))

    def zone_of(poly: Polygon) -> tuple[str, str] | None:
        for zid, zp, _, zname in zones:
            if (
                poly.intersection(zp).area > 0.9 * zp.area
                and zp.intersection(poly).area > 0.6 * poly.area
            ):
                return zid, zname
        return None

    # Land cover: water, fields (the landing grounds marked), the town's ground and places.
    from sketch_city import _band_polygon

    water = unary_union([_band_polygon(c.line, c.width) for c in channels])
    for k, p in enumerate(_polys(water)):
        meta["plots"].append(
            {
                "id": f"water-{k}",
                "kind": "water",
                "cover": "water",
                "zone": None,
                "name": None,
                **_ring(p, 0.5),
            }
        )
    for k, (crop, poly, _) in enumerate(city.fields):
        for q in _polys(poly if poly.is_valid else poly.buffer(0)):
            z = zone_of(q)
            kind = "airstrip" if crop == "airstrip" else "field"
            meta["plots"].append(
                {
                    "id": f"field-{k}",
                    "kind": kind,
                    "cover": crop,
                    "zone": z[0] if z else None,
                    "name": z[1] if z else None,
                    **_ring(q, 0.5),
                }
            )
    for k, b in enumerate(city.blocks):
        for q in _polys(b):
            meta["plots"].append(
                {
                    "id": f"block-{k}",
                    "kind": "settlement",
                    "cover": "settlement",
                    "zone": None,
                    "name": None,
                    **_ring(q, 0.5),
                }
            )
    for k, (kind, pname, poly) in enumerate(city.places):
        cover = GROUND_COVER.get(kind)
        if cover is None:
            continue
        for q in _polys(poly if poly.is_valid else poly.buffer(0)):
            if q.area < 4:
                continue
            meta["plots"].append(
                {
                    "id": f"place-{k}",
                    "kind": "green" if cover in ("park", "green") else "paved",
                    "cover": cover,
                    "zone": None,
                    "name": pname or None,
                    **_ring(q, 0.3),
                }
            )

    # Woods (glades cut out) and every tree.
    glades = unary_union(city.glades) if city.glades else Polygon()
    woods = [
        g
        for w in city.woods
        for g in _polys((w if w.is_valid else w.buffer(0)).difference(glades))
        if g.area > 400
    ]
    f = city.forest
    wood_of = np.full(len(f), -1, np.int16)
    for wi, w in enumerate(woods):
        inside = shapely.contains_xy(w, f[:, 0], f[:, 1]) & (wood_of < 0)
        wood_of[inside] = wi
    # The big woods get names (by where they are); copses and small woods stay unnamed.
    big = sorted(range(len(woods)), key=lambda k: -woods[k].area)[:6]
    names = {}
    for k in big:
        c = np.asarray(woods[k].centroid.coords[0])
        ns_ = "north" if c[0] > 600 else "south" if c[0] < -600 else ""
        ew = "east" if c[1] > 600 else "west" if c[1] < -600 else ""
        where = f"{ns_}-{ew}" if ns_ and ew else ns_ or ew or "central"
        label = f"the {where}ern woods".replace("-ern", "-").replace("centralern", "central")
        if label in names.values():
            label = f"{label} ({woods[k].area / 1e4:.0f} ha)"
        names[k] = label
    for wi, w in enumerate(woods):
        conifer = (f[wood_of == wi, 4] == 1).mean() > 0.5 if (wood_of == wi).any() else False
        meta["woods"].append(
            {
                "id": f"wood-{wi}",
                "name": names.get(wi, ""),
                "kind": "conifer" if conifer else "broadleaf",
                "canopy": WOOD_CANOPY_M,
                **_ring(w, 1.0),
            }
        )
    t = city.trees  # the town's and villages' trees: (n, e, crown radius, crown centre height)
    trees = [
        np.column_stack(
            [
                t[:, 0],
                t[:, 1],
                t[:, 3] + 0.9 * t[:, 2],
                t[:, 2],
                np.zeros(len(t)),
                np.full(len(t), -1),
            ]
        )
    ]
    kind = np.vectorize(TREE_KIND.get)(f[:, 4].astype(int))
    trees.append(np.column_stack([f[:, 0], f[:, 1], f[:, 3], f[:, 2], kind, wood_of]))
    trees = np.vstack(trees)
    # A landing ground, as in every world (docs/07 §3): no tree on it, within 20 m of its
    # hedges, or in its approach funnels (the woods stay whole).
    clear = unary_union(
        [z.buffer(20.0) for _, z, _, _ in zones]
        + [f_ for _, z, _, _ in zones for f_ in _funnels(z)]
    )
    felled = shapely.contains_xy(clear, trees[:, 0], trees[:, 1]) & (trees[:, 5] < 0)
    trees = trees[~felled]
    print(f"{int(felled.sum())} trees felled round the landing grounds and in their approaches")

    # Roads, streets, tracks and bridges.
    for k, (rname, ln) in enumerate(city.roads):
        meta["roads"].append(
            {
                "id": f"road-{k}",
                "name": rname,
                "kind": "road",
                "width": 8.0,
                "points": np.round(np.asarray(ln.coords), 2).tolist(),
            }
        )
    for k, (kind_, ln, hw) in enumerate(city.streets):
        rk = STREETS.get(kind_)
        if rk is None:
            continue
        for j, seg in enumerate(_lines(ln)):
            meta["roads"].append(
                {
                    "id": f"{kind_}-{k}-{j}",
                    "name": None,
                    "kind": rk,
                    "width": round(2 * hw, 2),
                    "points": np.round(np.asarray(seg.coords), 2).tolist(),
                }
            )
    for k, (ln, hw) in enumerate(city.bridges):
        meta["roads"].append(
            {
                "id": f"bridge-{k}",
                "name": None,
                "kind": "street",
                "width": round(2 * hw, 2),
                "points": np.round(np.asarray(ln.coords), 2).tolist(),
            }
        )
    main = channels[0]
    meta["river"] = {
        "id": "river",
        "name": "the river",
        "width": round(float(np.mean(main.width)), 1),
        "depth": 3.0,
        "points": np.round(main.line[::4], 2).tolist(),
    }

    # Hedges, in segments of 20 m at most. Each boundary has its own character: a fence (no
    # hedge), or a hedge trimmed low, a typical one or an overgrown one, its height varying
    # along it; a longer one has a gateway. Round the landing grounds they are kept low.
    rng = np.random.default_rng(HEDGE_SEED)
    kept_low = unary_union([z.exterior.buffer(3.0) for _, z, _, _ in zones])
    hedges = []
    for ln in city.hedges:
        if ln.length < 4.0:
            continue
        r = rng.random()
        if r < 0.18:
            continue  # a post-and-wire fence
        lo, hi = (1.3, 1.9) if r < 0.45 else (1.9, 2.8) if r < 0.82 else (2.8, 4.0)
        if kept_low.contains(ln.interpolate(0.5, normalized=True)):
            lo, hi = 1.2, 1.6
        pieces = [ln]
        if ln.length > 60.0:  # a gateway, 5 m wide, near one end
            at = rng.uniform(10.0, 30.0)
            at = at if rng.random() < 0.5 else ln.length - at - 5.0
            pieces = [substring(ln, 0.0, at), substring(ln, at + 5.0, ln.length)]
        for piece in pieces:
            if piece.length < 2.0:
                continue
            n = max(math.ceil(piece.length / 20.0), 1)
            pts = [
                np.asarray(piece.interpolate(d).coords[0])
                for d in np.linspace(0, piece.length, n + 1)
            ]
            for a, b in itertools.pairwise(pts):
                height = float(rng.uniform(lo, hi))
                hedges.append([*a, *b, round(height, 2), round(0.7 + 0.45 * height, 2)])
    hedges = np.asarray(hedges, float).reshape(-1, 6)

    # Buildings: each part a box from the ground to its top, with its roof.
    boxes, bmeta = [], []
    for b in city.buildings:
        wall, roof = b.wall or TONES[b.kind][0], b.roof or TONES[b.kind][1]
        for k, p in enumerate(b.parts):
            c, th, length, width = _frame(p.footprint)
            if length < 0.3 or width < 0.3:
                continue
            q = np.asarray(p.footprint.exterior.coords)
            base = float(ground(np.append(q[:, 0], c[0]), np.append(q[:, 1], c[1])).min())
            g0 = float(ground(c[0], c[1])[0])
            top = g0 + p.z1 - base  # the part's top, above the lowest ground under it
            rise = p.rise if p.roof != "flat" else 0.0
            boxes.append(
                [c[0], c[1], length, width, th, base, top, top + rise, ROOFS.index(ROOF[p.roof]), 0]
            )
            bmeta.append(
                {
                    "kind": b.kind,
                    "name": b.name if k == 0 and b.name else None,
                    "colours": [wall, roof],
                }
            )

    # Structures: turbines, pylons, cooling towers, masts; and the wires.
    for k, t_ in enumerate(city.turbines):
        meta["structures"].append(
            {
                "id": f"turbine-{k + 1}",
                "name": f"wind turbine {k + 1}",
                "kind": "turbine",
                "n": t_["n"],
                "e": t_["e"],
                "base": float(ground(t_["n"], t_["e"])[0]),
                "height": t_["hub"] + t_["rotor"] / 2,
                "radius": 2.3,
                "heading": t_["facing"],
                "hub": t_["hub"],
                "rotor": t_["rotor"],
            }
        )
    for k, p in enumerate(city.pylons):
        meta["structures"].append(
            {
                "id": f"pylon-{k + 1}",
                "name": f"pylon {k + 1}",
                "kind": "pylon",
                "n": p["n"],
                "e": p["e"],
                "base": float(ground(p["n"], p["e"])[0]),
                "height": PEAK,
                "radius": FOOT,
                "heading": p["heading"],
                "hub": 0.0,
                "rotor": 0.0,
            }
        )
    for k, c_ in enumerate(city.cooling_towers):
        meta["structures"].append(
            {
                "id": f"cooling-tower-{k + 1}",
                "name": f"cooling tower {k + 1}",
                "kind": "cooling_tower",
                "n": c_["n"],
                "e": c_["e"],
                "base": float(ground(c_["n"], c_["e"])[0]),
                "height": c_["height"],
                "radius": c_["r_base"],
                "heading": 0.0,
                "hub": 0.0,
                "rotor": 0.0,
            }
        )
    for m in city.masts:
        z0 = float(ground(m["n"], m["e"])[0])
        meta["structures"].append(
            {
                "id": m["name"].replace(" ", "-"),
                "name": m["name"],
                "kind": "mast",
                "n": m["n"],
                "e": m["e"],
                "base": z0,
                "height": m["height"],
                "radius": 1.5,
                "heading": 0.0,
                "hub": 0.0,
                "rotor": 0.0,
            }
        )
        for a in m["anchors"]:
            za = float(ground(*a)[0])
            for lv in m["levels"]:
                pts = np.array([[m["n"], m["e"], z0 + lv], [a[0], a[1], za + 0.8]])
                meta["wires"].append({"kind": "stay", "points": np.round(pts, 2).tolist()})
    for w in city.wires:
        pts = np.asarray(w["pts"], float)
        ga, gb = float(ground(*w["a"])[0]), float(ground(*w["b"])[0])
        z = pts[:, 2] + ga + (gb - ga) * pts[:, 3]
        meta["wires"].append(
            {"kind": "conductor", "points": np.round(np.column_stack([pts[:, :2], z]), 2).tolist()}
        )

    # Landing grounds and places, for the survey and the gazetteer.
    for zid, poly, crop, zname in zones:
        pt = poly.representative_point()
        meta["zones"].append(
            {"id": zid, "at_ne_m": [round(pt.x, 2), round(pt.y, 2)], "crop": crop, "name": zname}
        )
    places = [("city", "the city", "town", [0.0, 0.0])]
    places += [
        (n.replace(" ", "-"), n, "village", np.round(c, 1).tolist()) for n, c in city.settlements
    ]
    for b in city.labels:
        places.append(
            (b.name.lower().replace(" ", "-"), b.name, "landmark", np.round(b.centre, 1).tolist())
        )
    plant = next((p for k, _, p in city.places if k == "plant"), None)
    if plant is not None:
        places.append(
            (
                "power-station",
                "the power station",
                "power station",
                np.round(plant.centroid.coords[0], 1).tolist(),
            )
        )
    sol = next((p for c, p, _ in city.fields if c == "solar"), None)
    if sol is not None:
        places.append(
            (
                "solar-farm",
                "the solar farm",
                "solar farm",
                np.round(sol.centroid.coords[0], 1).tolist(),
            )
        )
    if city.turbines:
        tc = np.mean([[t_["n"], t_["e"]] for t_ in city.turbines], axis=0)
        places.append(("wind-farm", "the wind farm", "wind farm", np.round(tc, 1).tolist()))
    for pid, pname, kind_, pt in places:
        meta["places"].append(
            {"id": pid, "name": pname, "kind": kind_, "point_ne_m": pt, "aliases": []}
        )
    meta["buildings"] = bmeta

    world_dir = ROOT / "configs" / "worlds"
    save_baked(
        world_dir / f"{name}.world.npz",
        dtm=dtm,
        trees=trees,
        hedges=hedges,
        buildings=np.asarray(boxes, float),
        meta=meta,
    )
    return {"datum": datum, "ops": ops, "sk": sk, "city": city}


def operations(name: str, datum: float, ops: dict) -> dict:
    """The site's operations: home at the airstrip, the geofence and ceilings, and every no-fly
    zone with its height."""
    a = ops["airstrip"]
    fence = np.asarray(ops["geofence"].exterior.coords)[:-1]
    no_fly = []
    for z in ops["zones"]:
        for k, p in enumerate(_polys(z["polygon"])):
            d = {
                "id": f"{z['name'].replace(' ', '-')}" + (f"-{k + 1}" if k else ""),
                "name": z.get("group", z["name"]),
                "reason": z.get("group", z["name"]),
                "polygons_ne_m": [
                    np.round(np.asarray(p.simplify(2.0).exterior.coords)[:-1], 1).tolist()
                ],
            }
            if z["ceiling_m"] is not None:
                d["ceiling_m"] = float(z["ceiling_m"])
                if z.get("datum_m") is not None:
                    d["ceiling_from"] = "centre"
                    d["centre_ground_m"] = round(float(z["datum_m"]) - datum, 2)
            no_fly.append(d)
    ceiling = float(ops["ceiling_m"]) - (datum if ops.get("ceiling_ref") == "datum" else 0.0)
    return {
        "home_ne_m": [round(a["centre"][0], 1), round(a["centre"][1], 1)],
        "geofence_ne_m": np.round(fence, 1).tolist(),
        "ceiling_agl_m": float(round(ceiling)),
        "ceiling_alt_m": float(round(ceiling)),
        "min_agl_m": 30.0,
        "no_fly": no_fly,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake a map sketch into a world the simulator flies.")
    ap.add_argument("name")
    a = ap.parse_args()
    got = bake(a.name)
    sk = got["sk"]
    spec = {
        "schema_version": 1,
        "id": a.name,
        "name": "Test map",
        "description": "The owner's test map, drawn on paper layer by layer (configs/sketches) "
        "and baked by tools/sketch_bake.py: a city on a river island, three villages, farms "
        "and woods, a wind farm, a power station, masts and an airstrip.",
        "seed": 5000,
        "bounds_ne_m": sk["bounds_ne_m"],
        "baked": f"{a.name}.world.npz",
    }
    world_yaml = ROOT / "configs" / "worlds" / f"{a.name}.yaml"
    world_yaml.write_text(
        "# The owner's test map (docs/07 §14), baked from the locked sketch (configs/"
        f"sketches/{a.name}.yaml)\n# by tools/sketch_bake.py. Edit the sketch and bake "
        "again; do not edit this.\n" + yaml.safe_dump(spec, sort_keys=False, width=100)
    )
    site_yaml = ROOT / "configs" / "sites" / f"{a.name}.yaml"
    site = yaml.safe_load(site_yaml.read_text())
    site["operations"] = operations(a.name, got["datum"], got["ops"])
    site["airspace_notes"] = (
        "Synthetic site, no real airspace. The operations come from the sketch: home at the "
        "airstrip, a geofence 150 m inside the map's edge, a flat 500 m ceiling, and the "
        "owner's no-fly zones with their heights."
    )
    site_yaml.write_text(
        "# The owner's test map (docs/07 §14). The operations block is written by\n"
        "# tools/sketch_bake.py.\n" + yaml.safe_dump(site, sort_keys=False, width=100)
    )
    print(world_yaml, site_yaml, ROOT / "configs" / "worlds" / f"{a.name}.world.npz", sep="\n")


if __name__ == "__main__":
    main()
