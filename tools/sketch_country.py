# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportOperatorIssue=false, reportReturnType=false, reportOptionalMemberAccess=false
# pyright: reportGeneralTypeIssues=false
# (Drawing tools over shapely, whose set operations are typed as returning a BaseGeometry; the
# code relies on them being polygons or lines at run time, which the checker cannot narrow.)
"""The countryside on the sketch (configs/sketches/<name>.yaml ``countryside``): villages along
the old roads, and farmsteads and single buildings in the open country, off the hills.

A village grows round its church where its road crosses flat, dry land a little way along
from where the sketch asks, clear of the city. A few lanes wander off the road, and houses
(some with a barn behind, as farms in a village are) take places along the road and lanes,
facing them, thickest near the church and thinning out. A green with trees faces the church.

Farmsteads are scattered over the open country, never closer than a few hundred metres to
each other, on low gentle ground away from the river, the roads, the city and the villages:
a farmhouse and barns round a yard (a silo on the larger ones), or a single cottage or barn,
aligned with the nearest road and joined to it by a track, with a windbreak of trees.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.ndimage import gaussian_filter, map_coordinates
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, substring, unary_union

from sketch_city import (
    PALETTE,
    Building,
    City,
    Part,
    _local,
    _ngon,
    _rect,
    _roofed,
    _wkt,
    building_from,
    building_record,
)

PALETTE.setdefault(
    "barn",
    (["#bfa184", "#a88b6c", "#c9b49a", "#9f7f63"], ["#7a7570", "#6d6f73", "#9a4f3c", "#8b5a45"]),
)


class Ground:
    """Heights and slopes of the sketch's terrain, and what counts as the low open country."""

    def __init__(self, h: np.ndarray, ns: np.ndarray, es: np.ndarray) -> None:
        self.h, self.n0, self.e0 = h, float(ns[0]), float(es[0])
        self.cell = float(ns[1] - ns[0])
        gy, gx = np.gradient(gaussian_filter(h, 3), self.cell)
        self.slope = np.hypot(gx, gy)
        self.plain = float(np.percentile(h, 40))
        self.box = (float(ns[0]), float(ns[-1]), float(es[0]), float(es[-1]))

    def at(self, grid: np.ndarray, q) -> np.ndarray:
        q = np.atleast_2d(np.asarray(q, float))
        return map_coordinates(
            grid,
            [(q[:, 0] - self.n0) / self.cell, (q[:, 1] - self.e0) / self.cell],
            order=1,
            mode="nearest",
        )

    def open_country(self, q, margin: float = 80.0) -> bool:
        """Low, gentle and on the map: not the hills."""
        n0, n1, e0, e1 = self.box
        if not (n0 + margin < q[0] < n1 - margin and e0 + margin < q[1] < e1 - margin):
            return False
        return bool(self.at(self.slope, q)[0] < 0.06 and self.at(self.h, q)[0] < self.plain + 45)

    def fits(self, p: Polygon) -> bool:
        """Its corners on the map and the ground under it gentle enough to build on."""
        q = np.asarray(p.exterior.coords)[:-1]
        n0, n1, e0, e1 = self.box
        if q[:, 0].min() < n0 or q[:, 0].max() > n1 or q[:, 1].min() < e0 or q[:, 1].max() > e1:
            return False
        z = self.at(self.h, q)
        return float(np.ptp(z)) / max(p.length / 4, 1.0) < 0.12


def _polys_of(g) -> list[Polygon]:
    return (
        [g]
        if isinstance(g, Polygon)
        else [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]
    )


def _point(ln: LineString, s: float) -> np.ndarray:
    return np.asarray(ln.interpolate(s).coords[0])


def _tangent(ln: LineString, s: float) -> np.ndarray:
    a, b = _point(ln, max(s - 8, 0)), _point(ln, min(s + 8, ln.length))
    return (b - a) / max(float(np.hypot(*(b - a))), 1e-9)


def _church(c, th, s: float = 1.0) -> list[Part]:
    """Nave and chancel, and a west tower with its spire (about 34 m)."""
    rect, _at = _local(c, th, s)
    return [
        Part(rect(-11, 13, -5.5, 5.5), 0, 9, "gable", 5),
        Part(rect(13, 19, -4, 4), 0, 7, "gable", 3.5),
        Part(rect(-18.5, -11, -3.8, 3.8), 0, 20),
        Part(rect(-18, -11.5, -3.3, 3.3), 20, 21, "hip", 13),
    ]


def _village(
    v: dict, city: City, ground: Ground, rng, avoid
) -> tuple[list[Building], list, list, list, np.ndarray] | None:
    road = dict(city.roads)[v["road"]]
    s0 = float(v.get("at_m", road.length / 2))
    radius = float(v.get("radius_m", 200.0))
    best = None
    for s in np.arange(max(s0 - 400, 60), min(s0 + 400, road.length - 60), 20.0):
        p = _point(road, s)
        if city.outline.distance(Point(p)) < radius + v.get("clear_m", 250.0):
            continue  # clear of the city: a village of its own
        cost = (
            100 * float(ground.at(ground.slope, p)[0])
            + abs(s - s0) / 150
            + 0.5 * max(0.0, float(ground.at(ground.h, p)[0]) - ground.plain - 15)
            + (200 if city.water.distance(Point(p)) < 70 else 0)
        )
        if best is None or cost < best[0]:
            best = (cost, s)
    if best is None:
        return None
    s = best[1]
    centre = _point(road, s)
    main = substring(road, max(s - 1.3 * radius, 0), min(s + 1.3 * radius, road.length))
    lanes = []

    def lane(q, a, length) -> LineString | None:
        pts = [q.copy()]
        while len(pts) * 8 < length:
            a += rng.normal(0, 0.07)
            q = q + 8 * np.array([math.cos(a), math.sin(a)])
            if city.water.distance(Point(q)) < 15 or float(ground.at(ground.slope, q)[0]) > 0.09:
                break
            pts.append(q.copy())
        return LineString(pts) if len(pts) > 5 else None

    for i in range(int(v.get("lanes", 5))):
        ss = s + rng.uniform(-0.45, 0.45) * radius
        t = _tangent(road, ss)
        side = 1 if i % 2 == 0 else -1
        a = math.atan2(t[1], t[0]) + side * (math.pi / 2 + rng.uniform(-0.5, 0.5))
        ln = lane(_point(road, ss), a, rng.uniform(0.6, 1.2) * radius)
        if ln is None:
            continue
        lanes.append(ln)
        if rng.random() < 0.5:  # a back lane off it
            sb = rng.uniform(0.3, 0.8) * ln.length
            tb = _tangent(ln, sb)
            ab = math.atan2(tb[1], tb[0]) + rng.choice([-1, 1]) * (
                math.pi / 2 + rng.uniform(-0.3, 0.3)
            )
            back = lane(_point(ln, sb), ab, rng.uniform(0.4, 0.7) * radius)
            if back is not None:
                lanes.append(back)
    keep_off = unary_union(
        [city.water.buffer(15), main.buffer(5 + 2), *[ln.buffer(3 + 2) for ln in lanes], avoid]
    )
    taken: list[Polygon] = []

    def ok(p: Polygon) -> bool:
        return (
            ground.fits(p)
            and not keep_off.intersects(p)
            and not any(p.distance(o) < 0.5 for o in taken)
        )

    buildings: list[Building] = []
    trees = []
    # The church by the road near the middle (every village has one), and a green with trees
    # across from it.
    green = None
    church = None
    for scale in (1.0, 0.85, 0.7):
        for k in range(80):
            ss = s + (k // 4) * 6 * (1 if k % 2 else -1)
            t = _tangent(road, ss)
            nrm = np.array([-t[1], t[0]]) * (1 if (k // 2) % 2 == 0 else -1)
            c = _point(road, ss) + nrm * (5 + 3 + 6 * scale + 2)
            parts = _church(c, math.atan2(t[1], t[0]), scale)
            foot = unary_union([p.footprint for p in parts]).convex_hull
            if ok(foot):
                church = (parts, foot, ss, nrm, t)
                break
        if church:
            break
    if church is not None:
        parts, foot, ss, nrm, t = church
        taken.append(foot)
        buildings.append(Building(parts, "landmark", f"{v['name']} church"))
        g = _rect(_point(road, ss) - nrm * (5 + 4 + 16), math.atan2(t[1], t[0]), 44, 30).difference(
            keep_off
        )
        if not g.is_empty:
            green = max(_polys_of(g), key=lambda q: q.area)
            taken.append(green)
            for _ in range(6):
                x0, y0, x1, y1 = green.bounds
                q = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
                if green.contains(Point(q)):
                    trees.append([*q, rng.uniform(3, 5), rng.uniform(7, 10)])
    # A big village's taller buildings, round its heart: flats, a hotel, a mill, a school
    # (four to seven storeys; no towers).
    tall = int(v.get("tall", 0))
    tries = 0
    while tall > 0 and tries < 3000:
        tries += 1
        ln = main if rng.random() < 0.5 or not lanes else lanes[int(rng.integers(len(lanes)))]
        sf = rng.uniform(0, ln.length)
        p = _point(ln, sf)
        if float(np.hypot(*(p - centre))) > 0.45 * radius:
            continue
        t = _tangent(ln, sf)
        nrm = np.array([-t[1], t[0]]) * rng.choice([-1, 1])
        w, d = rng.uniform(16, 30), rng.uniform(11, 16)
        q = _rect(
            p + nrm * ((5.0 if ln is main else 3.0) + rng.uniform(2, 5) + d / 2),
            math.atan2(t[1], t[0]),
            w,
            d,
        )
        if not ok(q):
            continue
        taken.append(q)
        if rng.random() < 0.7:
            buildings.append(
                Building([_roofed(q, rng.uniform(13, 21), "hip", rng.uniform(2, 3.5))], "block")
            )
        else:
            buildings.append(Building([Part(q, 0, rng.uniform(14, 22))], "modern"))
        tall -= 1
    # Houses along the road and lanes, thickest near the church: some take a free place on a
    # frontage, others join a row beside a house already there (terraced near the middle).
    fronts = [(main, 5.0, 2.0)] + [(ln, 3.0, 1.0) for ln in lanes]
    weights = np.array([w for _, _, w in fronts])
    target = int(v.get("buildings", 35))
    rows: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, float]
    ] = []  # centre, along, out, half width
    tries = 0
    while len(buildings) < target and tries < 40000:
        tries += 1
        w, d = rng.uniform(7, 12), rng.uniform(7, 10)
        if rows and rng.random() < 0.6:  # next in a row
            pc, t, nrm, half_w = rows[int(rng.integers(len(rows)))]
            near = math.exp(-((float(np.hypot(*(pc - centre))) / (0.5 * radius)) ** 2))
            gap = rng.uniform(0.6, 2.0) if rng.random() < near else rng.uniform(3, 7)
            sign = rng.choice([-1, 1])
            c = pc + t * sign * (half_w + gap + w / 2) + nrm * rng.uniform(-1, 1)
            th = math.atan2(t[1], t[0])
            p = c
        else:
            ln, half, _ = fronts[int(rng.choice(len(fronts), p=weights / weights.sum()))]
            sf = rng.uniform(0, ln.length)
            p = _point(ln, sf)
            t = _tangent(ln, sf)
            nrm = np.array([-t[1], t[0]]) * rng.choice([-1, 1])
            c = p + nrm * (half + rng.uniform(2, 6) + d / 2)
            th = math.atan2(t[1], t[0])
        if rng.random() > math.exp(-((float(np.hypot(*(p - centre))) / (0.6 * radius)) ** 2)):
            continue
        gable_on = rng.random() < 0.2  # its gable end to the road
        house = _rect(c, th + (math.pi / 2 if gable_on else 0.0), *((d, w) if gable_on else (w, d)))
        if not ok(house):
            continue
        taken.append(house)
        rows.append((c, t, nrm, (d if gable_on else w) / 2))
        storeys = rng.uniform(5.5, 8.5) if rng.random() < 0.7 else rng.uniform(8.5, 10.5)
        buildings.append(Building([_roofed(house, storeys, "gable", rng.uniform(3, 4))], "house"))
        if rng.random() < 0.2 and len(buildings) < target:  # a farm in the village: a barn behind
            bc = c + nrm * (d / 2 + rng.uniform(8, 14) + 5)
            barn = _rect(
                bc,
                th + math.pi / 2 + rng.uniform(-0.1, 0.1),
                rng.uniform(18, 26),
                rng.uniform(9, 12),
            )
            if ok(barn):
                taken.append(barn)
                buildings.append(
                    Building([_roofed(barn, rng.uniform(6.5, 8.5), "gable", 4.5)], "barn")
                )
        if rng.random() < 0.6:  # a tree in the back garden
            q = c + nrm * (d / 2 + rng.uniform(5, 10)) + t * rng.uniform(-5, 5)
            trees.append([*q, rng.uniform(2.5, 4.5), rng.uniform(6, 9)])
    streets = [("main", main, 4.5)] + [("lane", ln, 3.0) for ln in lanes]
    places = [("green", "", green)] if green is not None else []
    return buildings, streets, places, trees, centre


FARMS = {  # yard layouts in a local frame (u along the yard's axis): one per kind of holding
    "cottage": [("house", (-6, 6, -4.5, 4.5), 6.5, 3.0)],
    "barn": [("barn", (-12, 12, -5, 5), 7.5, 4.0)],
    "small": [("house", (-6, 6, -14, -6), 7.0, 3.5), ("barn", (-12, 12, 6, 17), 8.0, 4.5)],
    "medium": [
        ("house", (-6, 6, -14, -6), 7.0, 3.5),
        ("barn", (-14, 14, 6, 18), 8.5, 4.5),
        ("barn", (-22, -13, -8, 12), 7.0, 3.5),
        ("barn", (10, 18, -11, -3), 4.5, 2.0),
    ],
    "large": [
        ("house", (-6, 6, -16, -7), 7.5, 3.5),
        ("barn", (-16, 16, 6, 19), 9.0, 5.0),
        ("barn", (-25, -15, -10, 14), 7.5, 4.0),
        ("barn", (22, 46, 2, 15), 8.0, 4.0),
        ("silo", (17, 17, 11, 11), 13.0, 2.0),
    ],
}


@dataclass
class Group:
    """One village, string of roadside houses or farmstead: what it adds to the map, so that
    each can be locked, or grown again, on its own."""

    name: str
    kind: str  # village, ribbon or farm
    centre: np.ndarray
    radius: float
    buildings: list[Building] = field(default_factory=list)
    streets: list = field(default_factory=list)
    places: list = field(default_factory=list)
    trees: list = field(default_factory=list)  # [north, east, crown radius, crown height]


def _rng(cfg: dict, name: str, k: int = 0):
    """Each group's own random stream: growing one again leaves the others as they were."""
    return np.random.default_rng([cfg.get("seed", 1), zlib.crc32(name.encode()), k])


def _colour(g: Group, rng) -> None:
    for b in g.buildings:
        if b.kind in PALETTE and not b.wall:
            walls, roofs = PALETTE[b.kind]
            b.wall, b.roof = str(rng.choice(walls)), str(rng.choice(roofs))


def _ribbon(
    r: dict, city: City, ground: Ground, rng, groups: list[Group], keep_out
) -> Group | None:
    """Houses strung out along a road between the city and a village: thicker towards either
    end, sparse in the middle, some in twos and threes, the odd one with a barn."""
    road = dict(city.roads)[r["road"]]
    village = next((g for g in groups if g.name == r["to"]), None)
    if village is None:
        return None
    s_village = next(
        (
            s
            for s in np.arange(0, road.length, 10.0)
            if np.hypot(*(_point(road, s) - village.centre)) < 0.85 * village.radius
        ),
        None,
    )
    if s_village is None:
        return None
    # It starts where the city's own houses along the road give out (the town's fringe leaves
    # fields there), not where its outer bound is.
    along = [
        road.project(b.parts[0].footprint.centroid)
        for b in city.buildings
        if road.distance(b.parts[0].footprint.centroid) < 60
    ]
    s_city = max((s for s in along if s < s_village - 100), default=0.0) + 20
    if s_village - s_city < 100:
        return None
    built = unary_union([b.parts[0].footprint for b in city.buildings]).buffer(10)
    keep_off = unary_union([city.water.buffer(15), road.buffer(4.5 + 3), built, keep_out])
    shapely.prepare(keep_off)
    taken: list[Polygon] = []
    g = Group(r["name"], "ribbon", _point(road, (s_city + s_village) / 2), (s_village - s_city) / 2)
    target = int(r.get("buildings", 60))
    s = s_city + rng.uniform(5, 20)
    side = rng.choice([-1, 1])
    while s < s_village and len(g.buildings) < target:
        near_end = max(math.exp(-(s - s_city) / 180), math.exp(-(s_village - s) / 180))
        t = _tangent(road, s)
        th = math.atan2(t[1], t[0])
        side = -side if rng.random() < 0.6 else side
        nrm = np.array([-t[1], t[0]]) * side
        for k in range(
            1 if rng.random() > 0.25 else int(rng.integers(2, 5))
        ):  # one, or a row of a few
            w, d = rng.uniform(8, 12), rng.uniform(7, 10)
            c = _point(road, s + k * (w + rng.uniform(1, 4))) + nrm * (
                4.5 + rng.uniform(4, 10) + d / 2
            )
            house = _rect(c, th, w, d)
            if (
                not ground.fits(house)
                or keep_off.intersects(house)
                or any(house.distance(o) < 2 for o in taken)
            ):
                continue
            taken.append(house)
            g.buildings.append(
                Building(
                    [_roofed(house, rng.uniform(5.5, 8.5), "gable", rng.uniform(3, 4))], "house"
                )
            )
            if rng.random() < 0.15:  # a smallholding: a barn behind
                barn = _rect(
                    c + nrm * (d / 2 + rng.uniform(8, 14) + 5),
                    th + math.pi / 2,
                    rng.uniform(16, 24),
                    rng.uniform(8, 11),
                )
                if (
                    ground.fits(barn)
                    and not keep_off.intersects(barn)
                    and not any(barn.distance(o) < 2 for o in taken)
                ):
                    taken.append(barn)
                    g.buildings.append(
                        Building([_roofed(barn, rng.uniform(6, 8), "gable", 4)], "barn")
                    )
            if rng.random() < 0.6:
                q = c + nrm * (d / 2 + rng.uniform(5, 10)) + t * rng.uniform(-5, 5)
                g.trees.append([*q, rng.uniform(2.5, 4.5), rng.uniform(6, 9)])
        s += rng.uniform(12, 26) / (0.35 + 0.65 * near_end)
    return g


def _farms(cfg: dict, city: City, ground: Ground, road_union, avoid, lanes) -> list[Group]:
    """Farmsteads: darts thrown at the open country, kept where they are far enough apart."""
    rng = np.random.default_rng([cfg.get("seed", 1), 1])  # their own stream
    reach = unary_union([road_union, lanes])
    kinds = list(FARMS)
    probs = np.array(cfg.get("farm_mix", [0.15, 0.1, 0.3, 0.27, 0.18]), float)
    spacing = cfg.get("farm_spacing_m", 330.0)
    sites: list[np.ndarray] = []
    out: list[Group] = []
    keep_off = unary_union(
        [city.water.buffer(60), road_union.buffer(25), city.outline.buffer(250), avoid]
    )
    shapely.prepare(keep_off)
    n0, n1, e0, e1 = ground.box
    for _ in range(20000):
        if len(sites) >= cfg.get("farms", 30):
            break
        q = np.array([rng.uniform(n0, n1), rng.uniform(e0, e1)])
        if not ground.open_country(q, 120) or keep_off.contains(Point(q)):
            continue
        if any(float(np.hypot(*(q - o))) < spacing for o in sites):
            continue
        sites.append(q)
        g = Group(f"farm {len(sites):02d}", "farm", q, 60.0)
        r = np.asarray(nearest_points(Point(q), reach)[1].coords[0])
        th = (
            math.atan2(*(q - r)[::-1]) + math.pi / 2
            if np.hypot(*(q - r)) > 1
            else rng.uniform(0, math.pi)
        )
        th += rng.normal(0, 0.12)
        kind = kinds[int(rng.choice(len(kinds), p=probs / probs.sum()))]
        rect, at = _local(q, th, 1.0)
        for k, (u0, u1, v0, v1), z, rise in FARMS[kind]:
            if k == "silo":
                foot = _ngon(at(u0, v0), 3.2)
                b = Building([Part(foot, 0, z, "hip", rise)], "modern")
            else:
                foot = rect(u0, u1, v0, v1)
                b = Building([_roofed(foot, z * rng.uniform(0.9, 1.1), "gable", rise)], k)
            if (
                ground.fits(foot)
                and not city.water.buffer(25).intersects(foot)
                and not road_union.buffer(8).intersects(foot)
            ):
                g.buildings.append(b)
        if not g.buildings:
            continue
        # A track to the nearest road or lane, if it is near enough and stays dry.
        d = float(np.hypot(*(q - r)))
        if 30 < d < 700:
            nrm = np.array([-(r - q)[1], (r - q)[0]]) / d
            ctrl = (q + r) / 2 + nrm * d * rng.uniform(-0.15, 0.15)
            s = np.linspace(0, 1, 24)[:, None]
            track = LineString((1 - s) ** 2 * q + 2 * (1 - s) * s * ctrl + s**2 * r)
            if not track.intersects(city.water):
                g.streets.append(("track", track, 1.8))
        # A windbreak on one side of the yard.
        side = rng.choice([-1, 1])
        u = np.array([math.cos(th), math.sin(th)])
        vv = np.array([-u[1], u[0]])
        for k in range(rng.integers(3, 9)):
            p = q + vv * side * rng.uniform(26, 32) + u * (k * 8 - 24) + rng.normal(0, 1.5, 2)
            g.trees.append([*p, rng.uniform(3, 5), rng.uniform(7, 11)])
        _colour(g, _rng(cfg, g.name, 2))
        out.append(g)
    return out


def add_countryside(
    cfg: dict,
    city: City,
    h: np.ndarray,
    ns: np.ndarray,
    es: np.ndarray,
    locked: dict[str, Group] | None = None,
) -> list[Group]:
    """The villages, the houses strung along roads and the farmsteads, added to ``city``. A
    group in ``locked`` is used as it is; the rest are grown (each from its own random
    stream). A locked farm inside a newly grown village gives way to it. Returns the groups."""
    locked = locked or {}
    ground = Ground(h, ns, es)
    road_union = unary_union([ln for _, ln in city.roads])
    farms = [g for g in locked.values() if g.kind == "farm"]
    farm_foot = unary_union([b.parts[0].footprint for g in farms for b in g.buildings]).buffer(20)
    groups: list[Group] = []
    fresh: list[Group] = []
    avoid = Polygon()
    for v in cfg.get("villages", []):
        g = locked.get(v["name"])
        if g is None:
            got = _village(v, city, ground, _rng(cfg, v["name"]), avoid)
            if got is None:
                continue
            buildings, streets, places, ts, centre = got
            g = Group(
                v["name"],
                "village",
                centre,
                float(v.get("radius_m", 200.0)),
                buildings,
                streets,
                places,
                ts,
            )
            _colour(g, _rng(cfg, v["name"], 2))
            fresh.append(g)
        groups.append(g)
        avoid = unary_union([avoid, Point(g.centre).buffer(g.radius + 250)])
    villages_foot = unary_union([b.parts[0].footprint for g in groups for b in g.buildings]).buffer(
        25
    )
    for r in cfg.get("ribbons", []):
        g = locked.get(r["name"])
        if g is None:
            g = _ribbon(
                r,
                city,
                ground,
                _rng(cfg, r["name"]),
                groups,
                unary_union([farm_foot, villages_foot]),
            )
            if g is None:
                continue
            _colour(g, _rng(cfg, r["name"], 2))
            fresh.append(g)
        groups.append(g)
    if farms:
        clash = unary_union(
            [Point(g.centre).buffer(g.radius + 120) for g in fresh if g.kind == "village"]
            + [
                b.parts[0].footprint.buffer(40)
                for g in fresh
                if g.kind == "ribbon"
                for b in g.buildings
            ]
        )
        groups += [
            g for g in farms if not any(clash.intersects(b.parts[0].footprint) for b in g.buildings)
        ]
    else:
        lanes = unary_union([ln for g in groups for k, ln, _ in g.streets if k == "lane"])
        groups += _farms(cfg, city, ground, road_union, avoid, lanes)
    trees = []
    for g in groups:
        city.buildings += g.buildings
        city.streets += g.streets
        city.places += g.places
        trees += g.trees
        if g.kind == "village":
            city.settlements.append((g.name, g.centre))
    if trees:
        city.trees = np.vstack([city.trees, np.asarray(trees, float).reshape(-1, 4)])
    return groups


# --- locking ----------------------------------------------------------------------------


def save_countryside(groups: list[Group], path) -> None:
    """Write the groups, exactly as they are, to ``path`` (gzipped JSON)."""
    import gzip
    import json

    data = {
        "groups": [
            {
                "name": g.name,
                "kind": g.kind,
                "centre": np.round(np.asarray(g.centre, float), 2).tolist(),
                "radius": g.radius,
                "buildings": [building_record(b) for b in g.buildings],
                "streets": [[k, _wkt(ln), hw] for k, ln, hw in g.streets],
                "places": [[k, n, _wkt(p)] for k, n, p in g.places],
                "trees": np.round(np.asarray(g.trees, float).reshape(-1, 4), 2).tolist(),
            }
            for g in groups
        ]
    }
    with gzip.open(path, "wt") as f:
        json.dump(data, f, separators=(",", ":"))


def load_countryside(path) -> dict[str, Group]:
    """The groups saved by ``save_countryside``, by name, in their order."""
    import gzip
    import json

    with gzip.open(path, "rt") as f:
        d = json.load(f)
    g = shapely.from_wkt
    return {
        r["name"]: Group(
            r["name"],
            r["kind"],
            np.asarray(r["centre"], float),
            float(r["radius"]),
            [building_from(b) for b in r["buildings"]],
            [(k, g(w), hw) for k, w, hw in r["streets"]],
            [(k, n, g(w)) for k, n, w in r["places"]],
            [list(t) for t in r["trees"]],
        )
        for r in d["groups"]
    }
