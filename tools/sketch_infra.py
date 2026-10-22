# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportOperatorIssue=false, reportReturnType=false, reportOptionalMemberAccess=false
# pyright: reportGeneralTypeIssues=false
# (Drawing tools over shapely, whose set operations are typed as returning a BaseGeometry; the
# code relies on them being polygons or lines at run time, which the checker cannot narrow.)
"""Telecom masts and a solar farm on the sketch (configs/sketches/<name>.yaml ``infra``).

A mast: a guyed lattice mast (a triangle of steel some 2 m across) on the top of a range of
hills, painted in seven bands of red and white for aircraft; held by stays from four levels
to three anchors 120° apart, about two-thirds of its height out; dishes and panels near
the top; a fenced compound with an equipment hut at its foot and a track to the nearest
road. The trees round its foot and its anchors are cleared; the stays pass over the rest.

The solar farm: a field of the right size near the city, level or facing south, away from
homes and the power lines, filled with rows of panels facing south (tilted 25°, 0.8 m off the ground
at the front, 2.5 m at the back, rows 7 m apart), inverter cabins among them, a fence round
the field inside its hedge and a track from its gate.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import shapely
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union

from sketch_city import Building, City, Part, _lines, _polys, _rect
from sketch_country import Ground
from sketch_land import SHADOW, _route
from sketch_power import _shade, _strip

RED, WHITE = "#cf4237", "#eeeeea"
PANEL = "#2c3b54"
TILT = math.radians(25.0)
PANEL_DEPTH = 3.6  # m, on the ground
PANEL_LOW, PANEL_HIGH = 0.8, 2.5
ROW_PITCH = 7.0


def _mast_site(m: dict, sk: dict, city: City, ground: Ground) -> np.ndarray | None:
    """The highest ground near the range's crest where a mast may stand."""
    rg = next(r for r in sk["terrain"]["ranges"] if r["name"] == m["range"])
    crest = LineString(rg["crest_ne_m"])
    height = float(m.get("height_m", 150.0))
    reach = 0.65 * height
    n0, n1, e0, e1 = ground.box
    homes = unary_union(
        [
            b.parts[0].footprint
            for b in city.buildings
            if b.kind in ("house", "old", "terrace", "block")
        ]
    )
    pylons = np.array([[p["n"], p["e"]] for p in city.pylons]) if city.pylons else np.zeros((0, 2))
    turbines = (
        np.array([[t["n"], t["e"]] for t in city.turbines]) if city.turbines else np.zeros((0, 2))
    )
    gn, ge = np.meshgrid(np.arange(n0, n1, 20.0), np.arange(e0, e1, 20.0), indexing="ij")
    p = np.stack([gn.ravel(), ge.ravel()], 1)
    margin = reach + 40
    p = p[
        (p[:, 0] > n0 + margin)
        & (p[:, 0] < n1 - margin)
        & (p[:, 1] > e0 + margin)
        & (p[:, 1] < e1 - margin)
    ]
    p = p[shapely.distance(shapely.points(p), crest) < rg["width_m"]]
    z = ground.at(ground.h, p)
    for k in np.argsort(-z):
        q = p[k]
        if len(turbines) and KDTree(turbines).query(q)[0] < 400:
            continue
        if len(pylons) and KDTree(pylons).query(q)[0] < 300:
            continue
        if homes.distance(Point(q)) < 250:
            continue
        return q
    return None


def _mast(m: dict, q: np.ndarray, rng) -> dict:
    height = float(m.get("height_m", 150.0))
    a0 = rng.uniform(0, 2 * math.pi / 3)
    anchors = [
        (
            q
            + 0.65
            * height
            * np.array([math.cos(a0 + k * 2 * math.pi / 3), math.sin(a0 + k * 2 * math.pi / 3)])
        ).tolist()
        for k in range(3)
    ]
    return {
        "name": m.get("name", "mast"),
        "n": float(q[0]),
        "e": float(q[1]),
        "height": height,
        "anchors": anchors,
        "levels": [round(f * height, 1) for f in (0.3, 0.55, 0.8, 0.97)],
        "turn": float(a0),
    }


def _solar_field(s: dict, city: City, ground: Ground):
    """The field for the solar farm: of about the size asked, near the city (``near_city_m``
    from its buildings), level or facing south, away from homes and from the power lines."""
    lo, hi = (a * 1e4 for a in s.get("size_ha", [6, 12]))
    want = 0.5 * (lo + hi)
    near_lo, near_hi = s.get("near_city_m", [150, 700])
    homes = unary_union(
        [
            b.parts[0].footprint
            for b in city.buildings
            if b.kind in ("house", "old", "terrace", "block")
        ]
    )
    town = unary_union(
        [
            b.parts[0].footprint
            for b in city.buildings
            if city.outline.contains(b.parts[0].footprint.centroid)
        ]
    )
    lines = (
        unary_union([LineString(np.asarray(w["pts"])[:, :2]) for w in city.wires])
        if city.wires
        else None
    )
    best = None
    for i, (crop, poly, _) in enumerate(city.fields):
        if not lo <= poly.area <= hi or crop in ("orchard",):
            continue
        d_town = poly.distance(town)
        if not near_lo <= d_town <= near_hi or poly.distance(homes) < s.get("from_homes_m", 120.0):
            continue
        if lines is not None and poly.distance(lines) < 80:
            continue
        c = np.asarray(poly.centroid.coords[0])
        q = np.asarray(poly.exterior.coords)
        z = ground.at(ground.h, q)
        slope = float(np.mean(ground.at(ground.slope, q)))
        # Facing south: the ground rising to the north (a south slope gets more sun).
        rise_n = float(np.polyfit(q[:, 0] - c[0], z, 1)[0]) if np.ptp(q[:, 0]) > 1 else 0.0
        score = (
            abs(poly.area - want) / want
            + 20 * slope
            - 30 * rise_n
            + poly.length**2 / poly.area / 40
            + d_town / 400
        )
        if best is None or score < best[0]:
            best = (score, i)
    return None if best is None else best[1]


def _solar(s: dict, poly: Polygon, city: City, ground: Ground, rng):
    """Rows of panels in the field, chunks of at most 40 m: (north of the front edge, east
    from, east to); and the cabins, fence and track."""
    inner = poly.buffer(-9)
    x0, y0, x1, y1 = inner.bounds
    rows = []
    for n in np.arange(x0 + 2, x1 - PANEL_DEPTH, ROW_PITCH):
        strip = LineString(
            [(n + PANEL_DEPTH / 2, y0 - 10), (n + PANEL_DEPTH / 2, y1 + 10)]
        ).intersection(inner.buffer(-PANEL_DEPTH / 2))
        for seg in _lines(strip):
            ea, eb = sorted(np.asarray(seg.coords)[[0, -1], 1])
            k = max(1, math.ceil((eb - ea) / 40))
            for j in range(k):
                a, b = (
                    ea + (eb - ea) * j / k,
                    ea + (eb - ea) * (j + 1) / k - (0.6 if j < k - 1 else 0),
                )
                if b - a > 4:
                    rows.append([float(n), float(a), float(b)])
    # Inverter cabins in a line up the middle, a gap in the rows for each.
    c = np.asarray(inner.centroid.coords[0])
    cabins = []
    for n in np.linspace(x0 + 30, x1 - 30, max(2, int((x1 - x0) / 110))):
        q = np.array([n, c[1]])
        if inner.contains(Point(q)):
            cabins.append(_rect(q, 0.0, 6, 3))
    if cabins:
        gaps = unary_union([cb.buffer(4) for cb in cabins])
        rows = [
            r
            for r in rows
            if not gaps.intersects(
                Polygon(
                    [
                        (r[0], r[1]),
                        (r[0], r[2]),
                        (r[0] + PANEL_DEPTH, r[2]),
                        (r[0] + PANEL_DEPTH, r[1]),
                    ]
                )
            )
        ]
    buildings = [Building([Part(cb, 0.0, 2.8)], "industrial") for cb in cabins]
    fence = poly.buffer(-4).exterior
    roads = unary_union(
        [ln for _, ln in city.roads]
        + [ln for k, ln, _ in city.streets if k in ("lane", "track", "access")]
    )
    gate = np.asarray(nearest_points(poly.buffer(-4).exterior, roads)[0].coords[0])
    to = np.asarray(nearest_points(Point(gate), roads)[1].coords[0])
    track = LineString([gate, to]) if np.hypot(*(to - gate)) > 2 else None
    return rows, buildings, fence, track


def build_infra(cfg: dict, sk: dict, city: City, h, ns, es) -> dict:
    ground = Ground(h, ns, es)
    rng = np.random.default_rng(cfg.get("seed", 1))
    masts, buildings, places, streets = [], [], [], []
    reach = unary_union(
        [ln for _, ln in city.roads]
        + [ln for k, ln, _ in city.streets if k in ("lane", "track", "service", "access")]
    )
    for m in cfg.get("masts", []):
        q = _mast_site(m, sk, city, ground)
        if q is None:
            continue
        mast = _mast(m, q, rng)
        masts.append(mast)
        th = mast["turn"]
        compound = _rect(q, th, 30, 24)
        places.append(("switchyard", "", compound))
        buildings.append(
            Building(
                [Part(_rect(q + np.array([math.cos(th), math.sin(th)]) * 9, th, 6, 4), 0.0, 3.0)],
                "industrial",
            )
        )
        for a in mast["anchors"]:
            buildings.append(
                Building([Part(_rect(np.asarray(a), th, 2.5, 2.5), 0.0, 1.0)], "industrial")
            )
        streets.append(("fence", compound.exterior, 0.4))
        to = np.asarray(nearest_points(Point(q), reach)[1].coords[0])
        streets.append(("track", _route(ground, q, to), 1.8))
    solar = None
    s = cfg.get("solar")
    if s:
        i = _solar_field(s, city, ground)
        if i is not None:
            poly = city.fields[i][1]
            rows, blds, fence, track = _solar(s, poly, city, ground, rng)
            buildings += blds
            streets.append(("fence", fence, 0.4))
            if track is not None:
                streets.append(("track", track, 1.8))
            solar = {"name": s.get("name", "solar farm"), "field": poly, "rows": rows}
    clear = unary_union(
        [
            *[Point(m["n"], m["e"]).buffer(22) for m in masts],
            *[Point(*a).buffer(6) for m in masts for a in m["anchors"]],
            *[ln.buffer(4) for k, ln, _ in streets if k == "track"],
            *([solar["field"]] if solar else []),
        ]
    )
    return {
        "masts": masts,
        "buildings": buildings,
        "places": places,
        "streets": streets,
        "solar": solar,
        "clear": clear,
    }


def apply_infra(city: City, state: dict) -> None:
    clear = state["clear"]
    shapely.prepare(clear)
    if len(city.forest):
        city.forest = city.forest[~shapely.contains_xy(clear, city.forest[:, 0], city.forest[:, 1])]
    if len(city.trees):
        city.trees = city.trees[~shapely.contains_xy(clear, city.trees[:, 0], city.trees[:, 1])]
    city.woods = [
        q
        for w in city.woods
        for q in (
            [w]
            if not clear.intersects(w)
            else _polys((w if w.is_valid else w.buffer(0)).difference(clear))
        )
        if q.area > 500
    ]
    solar = state["solar"]
    if solar is not None:
        f = solar["field"]
        city.fields = [
            ("solar" if p.intersection(f).area > 0.8 * p.area else c, p, o)
            for c, p, o in city.fields
        ]
        city.solar = solar["rows"]
    city.masts = state["masts"]
    city.buildings += state["buildings"]
    city.places += state["places"]
    city.streets += state["streets"]


def save_infra(state: dict, path) -> None:
    import gzip
    import json

    from sketch_city import _wkt, building_record

    s = state["solar"]
    data = {
        "masts": state["masts"],
        "buildings": [building_record(b) for b in state["buildings"]],
        "places": [[k, n, _wkt(p)] for k, n, p in state["places"]],
        "streets": [[k, _wkt(ln), hw] for k, ln, hw in state["streets"]],
        "solar": None
        if s is None
        else {
            "name": s["name"],
            "field": _wkt(s["field"]),
            "rows": np.round(s["rows"], 2).tolist(),
        },
        "clear": _wkt(state["clear"]),
    }
    with gzip.open(path, "wt") as f:
        json.dump(data, f, separators=(",", ":"))


def load_infra(path) -> dict:
    import gzip
    import json

    from sketch_city import building_from

    with gzip.open(path, "rt") as f:
        d = json.load(f)
    g = shapely.from_wkt
    s = d["solar"]
    return {
        "masts": d["masts"],
        "buildings": [building_from(b) for b in d["buildings"]],
        "places": [(k, n, g(w)) for k, n, w in d["places"]],
        "streets": [(k, g(w), hw) for k, w, hw in d["streets"]],
        "solar": None
        if s is None
        else {"name": s["name"], "field": g(s["field"]), "rows": s["rows"]},
        "clear": g(d["clear"]),
    }


# --- drawing ----------------------------------------------------------------------------


def mast_segments(m: dict) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
    """A guyed lattice mast as segments (from, to, width, colour), (north, east, height above
    its foot): three legs in red and white bands, zigzag bracing, aerials near the top."""
    H = m["height"]
    c = np.array([m["n"], m["e"]])
    legs = [
        c
        + 1.3
        * np.array(
            [math.cos(m["turn"] + k * 2 * math.pi / 3), math.sin(m["turn"] + k * 2 * math.pi / 3)]
        )
        for k in range(3)
    ]

    def band(z):
        return RED if int(z / (H / 7)) % 2 == 0 else WHITE

    segs = []
    zs = np.arange(0, H + 0.1, 7.5)
    for k in range(3):
        for z0, z1 in itertools.pairwise(zs):
            segs.append((np.r_[legs[k], z0], np.r_[legs[k], z1], 0.35, band(0.5 * (z0 + z1))))
            j = (k + 1) % 3
            zm = 0.5 * (z0 + z1)
            segs.append((np.r_[legs[k], z0], np.r_[legs[j], zm], 0.12, band(zm)))
            segs.append((np.r_[legs[j], zm], np.r_[legs[k], z1], 0.12, band(zm)))
    for k, z in enumerate((H - 12, H - 30, H - 45)):  # aerials and dishes
        a = m["turn"] + k * 2.1
        d = np.array([math.cos(a), math.sin(a)])
        segs.append((np.r_[c + d * 1.8, z - 1.5], np.r_[c + d * 1.8, z + 1.5], 1.2, "#b9bcc0"))
    segs.append((np.r_[c, H], np.r_[c, H + 4], 0.25, RED))  # the lightning rod and light
    return segs


def infra_items(city: City, view: np.ndarray, project, ground):
    """The masts (and their stays) and the solar panels, for a painter's drawing, as
    ``power_items``."""
    items, shadows = [], []

    def page(pts, z0):
        x, y = project(pts[:, 0], pts[:, 1], pts[:, 2] + z0)
        return np.column_stack([x, y])

    for m in city.masts:
        z0 = float(np.atleast_1d(ground(m["n"], m["e"]))[0])
        segs = mast_segments(m)
        items.append(
            [
                m["n"] + m["e"],
                [page(_strip(a, b, w, view), z0) for a, b, w, _ in segs],
                [col for *_, col in segs],
                np.array([m["n"], m["e"]]),
                z0,
                m["height"],
            ]
        )
        tip = np.array([m["n"], m["e"]]) + SHADOW * m["height"]
        shadows.append(
            page(
                np.array([[m["n"] - 1, m["e"] + 1, 0], [*tip, 0], [m["n"] + 1, m["e"] - 1, 0]]), z0
            )
        )
        for a in m["anchors"]:
            za = float(np.atleast_1d(ground(*a))[0])
            for lv in m["levels"]:
                p0 = np.array([m["n"], m["e"], z0 + lv])
                p1 = np.array([a[0], a[1], za + 0.8])
                for k in range(8):
                    q0, q1 = p0 + (p1 - p0) * k / 8, p0 + (p1 - p0) * (k + 1) / 8
                    mid = 0.5 * (q0 + q1)
                    items.append(
                        [
                            mid[0] + mid[1],
                            [page(_strip(q0, q1, 0.14, view), 0.0)],
                            ["#55595e"],
                            mid[:2],
                            0.0,
                            mid[2],
                        ]
                    )
    nrm = np.array([-math.sin(TILT), 0.0, math.cos(TILT)])  # the panels face south, tilted up
    col = _shade(PANEL, nrm)
    for n, a, b in getattr(city, "solar", []):
        z0 = float(np.atleast_1d(ground(n, 0.5 * (a + b)))[0])
        quad = np.array(
            [
                [n, a, PANEL_LOW],
                [n, b, PANEL_LOW],
                [n + PANEL_DEPTH, b, PANEL_HIGH],
                [n + PANEL_DEPTH, a, PANEL_HIGH],
            ]
        )
        items.append(
            [
                n + 0.5 * (a + b),
                [page(quad, z0)],
                [col],
                np.array([n, 0.5 * (a + b)]),
                z0,
                PANEL_HIGH,
            ]
        )
        g = quad[:, :2] + SHADOW * quad[:, 2:3]
        shadows.append(page(np.column_stack([np.vstack([quad[:2, :2], g[2:]]), np.zeros(4)]), z0))
    return items, shadows
