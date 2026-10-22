# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportOperatorIssue=false, reportReturnType=false, reportOptionalMemberAccess=false
# pyright: reportGeneralTypeIssues=false
# (Drawing tools over shapely, whose set operations are typed as returning a BaseGeometry; the
# code relies on them being polygons or lines at run time, which the checker cannot narrow.)
"""A power station on the sketch (configs/sketches/<name>.yaml ``power``), and its lines.

The station: a nuclear plant with two reactors, laid out as France's inland stations are
(Golfech, Belleville): beside a river, two natural-draught cooling towers (the water comes
from the river and most goes back), behind them the two units (each a reactor building
under its dome, a fuel building, an auxiliary building, a stack, a turbine hall and its
transformers), then the switchyard where the lines leave; offices, workshops and a car park
by the gate, a pumping station on the bank, a fence round it all and a belt of trees
outside. It takes the flattest stretch of the bank between two villages that displaces the
fewest farm buildings, keeping well clear of homes and roads.

The lines: double-circuit 400 kV lines on lattice pylons (some 50 m high, three cross-arms
a side, an earth wire on top), one to a substation on the city's edge, one off the map to
the national grid. Each takes the cheapest way over the land (round homes, villages, woods
and the wind farm; short over water; the city line keeps to a strip along the river bank)
and is then straightened into a few long legs, with a
pylon every 300-350 m and at each turn. Conductors hang in catenaries from the arms (sagging
some 10 m), so between pylons they are lowest: about 15 m off the ground.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union

from sketch_city import LIGHT, Building, City, Part, _band_polygon, _lines, _ngon, _polys, _rect
from sketch_country import Ground
from sketch_land import SHADOW, _route

CONCRETE = "#dcd9d2"
STEEL = "#8e939a"
WIRE = "#4f5358"

# Pylon (a 400 kV double-circuit lattice tower, after the UK's L12): cross-arms (height,
# half-span), the earth-wire peak, the body's half-width at the foot and at the top arm.
ARMS = ((29.0, 10.5), (37.0, 11.5), (45.0, 9.5))
PEAK = 51.0
FOOT, NECK = 5.0, 1.6
INSULATOR = 4.0


@dataclass
class Frame:
    """The site's own axes: u along the river (downstream), v inland from the water's edge."""

    o: np.ndarray
    u: np.ndarray
    v: np.ndarray

    def at(self, a: float, b: float) -> np.ndarray:
        return self.o + a * self.u + b * self.v

    def rect(self, u0, u1, v0, v1) -> Polygon:
        return Polygon([self.at(u0, v0), self.at(u1, v0), self.at(u1, v1), self.at(u0, v1)])


def _site(cfg: dict, city: City, groups, ground: Ground, channels) -> tuple[Frame, Polygon] | None:
    """The best stretch of bank for the station between the two villages."""
    main = channels[0]
    line, width = np.asarray(main.line), np.asarray(main.width)
    vil = {g.name: np.asarray(g.centre) for g in groups if g.kind == "village"}
    a, b = (vil[n] for n in cfg["between"])
    mid = (a + b) / 2
    length, depth = cfg.get("size_m", [660, 520])
    farms = {
        id(bb) for g in groups if g.kind == "farm" for bb in g.buildings
    }  # (a farm may be bought out)
    homes = unary_union(
        [
            bb.parts[0].footprint
            for bb in city.buildings
            if bb.kind in ("house", "old", "terrace", "block") and id(bb) not in farms
        ]
    )
    roads = unary_union([ln for _, ln in city.roads])
    wet = city.water.buffer(12)
    wooded = unary_union(city.woods)
    rural = [bb.parts[0].footprint for g in groups if g.kind == "farm" for bb in g.buildings]
    n0, n1, e0, e1 = ground.box
    best = None
    for i in range(8, len(line) - 8, 3):
        p = line[i]
        if np.hypot(*(p - mid)) > 1500:
            continue
        t = line[i + 6] - line[i - 6]
        t = t / np.hypot(*t)
        nrm = np.array([-t[1], t[0]])
        if (mid - p) @ nrm < 0:
            nrm = -nrm
        f = Frame(p + nrm * width[i] / 2, t, nrm)
        site = f.rect(-length / 2, length / 2, 35, 35 + depth)
        x0, y0, x1, y1 = site.bounds
        if x0 < n0 + 60 or x1 > n1 - 60 or y0 < e0 + 60 or y1 > e1 - 60:
            continue
        if (
            site.intersects(wet)
            or site.buffer(40).intersects(roads)
            or site.distance(homes) < cfg.get("from_homes_m", 300)
        ):
            continue
        if site.buffer(150).intersects(city.outline):
            continue
        q = np.array(
            [
                np.asarray(site.exterior.interpolate(s, normalized=True).coords[0])
                for s in np.linspace(0, 1, 24)
            ]
        )
        flat = float(np.ptp(ground.at(ground.h, np.vstack([q, np.asarray(site.centroid.coords)]))))
        displaced = sum(1 for fp in rural if site.buffer(20).intersects(fp))
        score = (
            3.0 * displaced
            + flat
            + site.intersection(wooded).area / 4000
            + np.hypot(*(np.asarray(site.centroid.coords[0]) - mid)) / 300
        )
        if best is None or score < best[0]:
            best = (score, f, site)
    return None if best is None else (best[1], best[2])


def _band(c):
    return _band_polygon(np.asarray(c.line), np.asarray(c.width))


def _box(f: Frame, u0, u1, v0, v1, z, kind="industrial", name="") -> Building:
    return Building([Part(f.rect(u0, u1, v0, v1), 0.0, z)], kind, name)


def _plant(cfg: dict, f: Frame, site: Polygon):
    """The station's buildings, cooling towers, ground and roads in the site's frame (u
    along the river, v inland): the towers by the river, the two units close behind them
    sharing their services, the switchyard furthest inland, offices and workshops by the gate."""
    length, depth = cfg.get("size_m", [480, 380])
    hl, top = length / 2, 35 + depth
    ct_h = float(cfg.get("cooling_tower_m", 135.0))
    buildings: list[Building] = []
    towers: list[dict] = []
    labels: list[Building] = []
    for uc in (-120.0, 120.0):
        c = f.at(uc, 125)
        towers.append(
            {
                "n": float(c[0]),
                "e": float(c[1]),
                "height": ct_h,
                "r_base": 0.34 * ct_h,
                "r_throat": 0.215 * ct_h,
                "z_throat": 0.78 * ct_h,
            }
        )
    for k, (uc, side) in enumerate(((-70.0, -1.0), (70.0, 1.0))):
        vc = 240.0
        c = f.at(uc, vc)
        parts = [Part(_ngon(c, 19.0, 28), 0.0, 42.0)]
        for r, z0, z1 in (
            (18.0, 42, 47),
            (15.5, 47, 51),
            (12.0, 51, 54),
            (7.5, 54, 56.5),
            (3.0, 56.5, 58),
        ):
            parts.append(Part(_ngon(c, r, 28), z0, z1))
        reactor = Building(parts, "reactor", f"Reactor {k + 1}")
        buildings.append(reactor)
        if k == 0:
            labels.append(reactor)
        buildings.append(_box(f, uc - 13, uc + 13, vc - 44, vc - 21, 28.0))  # fuel building
        a0, a1 = sorted((uc + side * 22, uc + side * 46))
        buildings.append(_box(f, a0, a1, vc - 30, vc + 25, 24.0))  # auxiliary building
        buildings.append(_box(f, uc - 21, uc + 21, vc + 22, vc + 102, 34.0))  # turbine hall
        for du in (-12.0, 0.0, 12.0):  # its transformers
            buildings.append(
                _box(f, uc + du - 3.5, uc + du + 3.5, vc + 107, vc + 114, 7.0, "steel")
            )
        s = f.at(uc - side * 27, vc - 6)
        buildings.append(Building([Part(_ngon(s, 2.2, 12), 0.0, 65.0)], "industrial"))  # stack
    buildings.append(_box(f, -22, 22, 225, 275, 14.0))  # shared services
    buildings.append(_box(f, -20, 20, 45, 65, 11.0))  # pumping station on the bank
    buildings.append(_box(f, 140, 210, 325, 355, 13.0, "modern"))  # offices
    buildings.append(_box(f, 150, 210, 255, 300, 11.0))  # workshops
    buildings.append(_box(f, -210, -170, 50, 80, 9.0))  # stores
    # The switchyard's gantries: a portal (two posts and a beam) every 28 m.
    for u in np.arange(-160.0, 110.0, 28.0):
        buildings.append(
            Building(
                [
                    Part(f.rect(u - 0.6, u + 0.6, 368, 369.2), 0, 17),
                    Part(f.rect(u - 0.6, u + 0.6, 404.8, 406), 0, 17),
                    Part(f.rect(u - 0.6, u + 0.6, 368, 406), 15.5, 16.7),
                ],
                "steel",
            )
        )
    # Ground: grass inside the fence, concrete round the buildings, the switchyard, the car
    # park, the towers' basins; roads round the site and to the bank.
    roads = [
        LineString([f.at(-hl + 15, 185), f.at(hl - 15, 185)]),
        LineString([f.at(-hl + 15, top - 10), f.at(hl - 15, top - 10)]),
        LineString([f.at(-hl + 15, 50), f.at(-hl + 15, top - 10)]),
        LineString([f.at(hl - 15, 50), f.at(hl - 15, top - 10)]),
        LineString([f.at(0, 65), f.at(0, 185)]),
        LineString([f.at(180, top - 10), f.at(180, top + 5)]),
    ]
    footprints = unary_union([p.footprint for b in buildings for p in b.parts])
    apron = footprints.buffer(9, join_style="mitre").intersection(site)
    places = [
        ("plant", "", site),
        ("apron", "", apron),
        ("switchyard", "", f.rect(-175, 120, 360, 412)),
        ("car_park", "", f.rect(130, 225, 365, 410)),
    ]
    for t in towers:
        places.append(("basin", "", _ngon((t["n"], t["e"]), t["r_base"] + 4, 36)))
    streets = [("plant_road", ln, 3.5) for ln in roads]
    streets.append(("fence", site.exterior, 0.4))
    gate = f.at(180, top + 5)
    return buildings, towers, places, streets, labels, gate


def _substation(
    cfg: dict,
    city: City,
    ground: Ground,
    toward: np.ndarray,
    river: LineString,
    side: float,
    main_water,
):
    """A grid substation just outside the city, as near its centre as it can be: on the
    station's bank of the river and near it, 30-110 m from the town's buildings, outside its
    blocks, parks and streets, off water, out of woods and on level ground."""
    town = [
        b.parts[0].footprint
        for b in city.buildings
        if city.outline.contains(b.parts[0].footprint.centroid)
    ]
    town_u = unary_union(town)
    built = unary_union([b.parts[0].footprint for b in city.buildings]).buffer(30)
    fabric = unary_union(
        [
            *city.blocks,
            *[p for k, _, p in city.places if k in ("park", "garden", "green", "square", "pitch")],
            *[
                ln.buffer(hw + 4)
                for k, ln, hw in city.streets
                if k in ("street", "main", "quay", "lane")
            ],
            *[ln.buffer(8) for _, ln in city.roads],
        ]
    )
    shapely.prepare(built)
    shapely.prepare(fabric)
    wet = city.water.buffer(30)
    wooded = unary_union(city.woods)
    n0, n1, e0, e1 = ground.box
    gn, ge = np.arange(n0, n1, 10.0), np.arange(e0, e1, 10.0)
    nn, ee = np.meshgrid(gn, ge, indexing="ij")
    d_town = distance_transform_edt(~shapely.contains_xy(town_u, nn, ee)) * 10.0
    d_wet = distance_transform_edt(~shapely.contains_xy(main_water, nn, ee)) * 10.0
    ok = (d_town > 30) & (d_town < 110) & (d_wet > 40) & (d_wet < 300)
    cand = np.stack([nn[ok], ee[ok]], 1)
    centre = np.asarray(city.outline.centroid.coords[0])
    order = np.argsort(np.hypot(*(cand - centre).T))
    for k in order:
        c = cand[k]
        if _river_side(river, c) != side:
            continue
        th = math.atan2(*(toward - c)[::-1])
        comp = _rect(c, th, 90, 60)
        if (
            comp.intersects(built)
            or comp.intersects(fabric)
            or comp.intersects(wet)
            or comp.intersects(wooded)
        ):
            continue
        if float(np.ptp(ground.at(ground.h, np.asarray(comp.exterior.coords)))) > 3.0:
            continue
        break
    else:
        return None
    u = np.array([math.cos(th), math.sin(th)])
    v = np.array([-u[1], u[0]])
    blds = [Building([Part(_rect(c - u * 32 + v * 16, th, 14, 9), 0, 6)], "industrial")]
    for k in range(3):
        blds.append(Building([Part(_rect(c + v * (-15 + 15 * k), th, 8, 5), 0, 7)], "steel"))
    for a in np.arange(-24, 40, 16.0):  # gantries
        p0, p1 = c + u * a - v * 24, c + u * a + v * 24
        blds.append(
            Building(
                [
                    Part(_rect(p0, th, 1.2, 1.2), 0, 15),
                    Part(_rect(p1, th, 1.2, 1.2), 0, 15),
                    Part(_rect((p0 + p1) / 2, th, 1.2, 49), 13.5, 14.7),
                ],
                "steel",
            )
        )
    entry = c + u * 45  # where the line comes in
    return comp, blds, entry


def _river_side(river: LineString, p: np.ndarray) -> float:
    s = river.project(Point(p))
    a = np.asarray(river.interpolate(max(s - 10, 0)).coords[0])
    b = np.asarray(river.interpolate(min(s + 10, river.length)).coords[0])
    t, q = b - a, p - a
    return float(np.sign(t[0] * q[1] - t[1] * q[0]))


def _cost(city: City, ground: Ground, site: Polygon, substation, ras_cell: float = 10.0, bank=None):
    """What it costs a line to cross each 10 m of the map (1: open farmland). Along the
    bank of ``bank`` (a river's water): half that in a strip 40-120 m from it, and dear to
    cross."""
    n0, n1, e0, e1 = ground.box
    gn, ge = np.arange(n0, n1 + 1, ras_cell), np.arange(e0, e1 + 1, ras_cell)
    nn, ee = np.meshgrid(gn, ge, indexing="ij")
    built = shapely.contains_xy(
        unary_union([b.parts[0].footprint.buffer(2) for b in city.buildings]), nn, ee
    )
    near_built = distance_transform_edt(~built) * ras_cell
    cost = 1.0 + 60.0 * np.exp(-((near_built / 70.0) ** 2))
    cost += 6.0 * shapely.contains_xy(unary_union(city.woods), nn, ee)
    wet = shapely.contains_xy(city.water, nn, ee)
    if bank is not None:
        d_wet = distance_transform_edt(~shapely.contains_xy(bank, nn, ee)) * ras_cell
        cost *= np.where((d_wet > 35) & (d_wet < 100), 0.25, 1.0)
        cost += 2.0 * (d_wet <= 40)  # the riverside trees
        cost += 60.0 * wet
    else:
        cost += 4.0 * wet
    for t in city.turbines:
        cost += 100.0 * (np.hypot(nn - t["n"], ee - t["e"]) < 250)
    cost += 100.0 * shapely.contains_xy(site.buffer(20), nn, ee)
    if substation is not None:
        cost += 100.0 * shapely.contains_xy(substation.buffer(5), nn, ee)
    slope = gaussian_filter(ground.slope, 2)
    cost += (ground.at(slope, np.stack([nn.ravel(), ee.ravel()], 1)).reshape(nn.shape) / 0.1) ** 2
    return gn, ge, cost


def _cheapest(gn, ge, cost, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rows, cols = cost.shape
    idx = np.arange(rows * cols).reshape(rows, cols)
    cell = float(gn[1] - gn[0])
    src, dst, w = [], [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        i0, i1 = slice(0, rows - di), slice(di, rows)
        j0, j1 = (
            (slice(0, cols - dj), slice(dj, cols))
            if dj >= 0
            else (slice(-dj, cols), slice(0, cols + dj))
        )
        src.append(idx[i0, j0].ravel())
        dst.append(idx[i1, j1].ravel())
        w.append((cell * math.hypot(di, dj) * 0.5 * (cost[i0, j0] + cost[i1, j1])).ravel())
    graph = coo_matrix(
        (np.concatenate(w), (np.concatenate(src), np.concatenate(dst))), shape=(rows * cols,) * 2
    ).tocsr()

    def node(p):
        return int(np.clip(round((p[0] - gn[0]) / cell), 0, rows - 1)) * cols + int(
            np.clip(round((p[1] - ge[0]) / cell), 0, cols - 1)
        )

    s, g = node(a), node(b)
    _, pred = dijkstra(graph, directed=False, indices=s, return_predecessors=True)
    path = [g]
    while path[-1] != s and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    path = path[::-1]
    return np.array([[gn[k // cols], ge[k % cols]] for k in path])


def _taut(path: np.ndarray, gn, ge, cost, slack: float = 1.12) -> np.ndarray:
    """Pull a winding cheapest path taut into long straight legs: from each turn, straight to
    the furthest point of the path that a straight leg reaches for little more than the path
    costs, crossing nothing dearer than the path does."""
    cell = float(gn[1] - gn[0])

    def c_at(p):
        i = np.clip(np.round((p[:, 0] - gn[0]) / cell).astype(int), 0, len(gn) - 1)
        j = np.clip(np.round((p[:, 1] - ge[0]) / cell).astype(int), 0, len(ge) - 1)
        return cost[i, j]

    pc = c_at(path)
    seg = np.r_[0.0, np.hypot(*np.diff(path, axis=0).T) * 0.5 * (pc[1:] + pc[:-1])]
    acc = np.cumsum(seg)
    out, i = [path[0]], 0
    while i < len(path) - 1:
        best = i + 1
        for j in range(len(path) - 1, i + 1, -1):
            n = max(int(np.hypot(*(path[j] - path[i])) / (cell / 2)), 2)
            q = path[i] + (path[j] - path[i]) * np.linspace(0, 1, n)[:, None]
            cq = c_at(q)
            straight = float(np.hypot(*(path[j] - path[i])) * cq.mean())
            if cq.max() <= pc[i : j + 1].max() + 0.5 and straight <= slack * (acc[j] - acc[i]) + 5:
                best = j
                break
        out.append(path[best])
        i = best
    return np.array(out)


def _pylons(route: LineString, span: float, forbidden) -> list[dict]:
    """Pylons along a line: one at each turn and evenly between, ``span`` m apart at most,
    each moved along the line off forbidden ground (water, roads, buildings)."""
    pts = np.asarray(route.coords)
    out: list[np.ndarray] = []
    for a, b in itertools.pairwise(pts):
        n = max(1, math.ceil(np.hypot(*(b - a)) / span))
        for k in range(n):
            out.append(a + (b - a) * k / n)
    out.append(pts[-1])
    out = out[1:]  # the line starts at a gantry, not a pylon
    placed = []
    for p in out[:-1] if len(out) > 1 else out:
        if placed and np.hypot(*(p - placed[-1])) < 0.3 * span:
            continue  # a turn just past a pylon: that pylon takes it
        s0 = route.project(Point(p))
        for ds in (0, 10, -10, 20, -20, 30, -30, 45, -45, 60, -60):
            q = np.asarray(route.interpolate(min(max(s0 + ds, 0), route.length)).coords[0])
            if not forbidden.contains(Point(q)):
                placed.append(q)
                break
    pylons = []
    for k, p in enumerate(placed):
        a = placed[k - 1] if k > 0 else np.asarray(route.coords[0])
        b = placed[k + 1] if k + 1 < len(placed) else np.asarray(route.coords[-1])
        d1, d2 = p - a, b - p
        d = d1 / max(np.hypot(*d1), 1e-6) + d2 / max(np.hypot(*d2), 1e-6)
        pylons.append(
            {"n": float(p[0]), "e": float(p[1]), "heading": float(math.atan2(d[1], d[0]))}
        )
    return pylons


def _attach(p: dict, level: int, side: int) -> tuple[np.ndarray, float]:
    """Where a conductor hangs from a pylon (plan point, height above its foot)."""
    perp = np.array([-math.sin(p["heading"]), math.cos(p["heading"])])
    if level < 0:  # the earth wire, from the peak
        return np.array([p["n"], p["e"]]), PEAK
    z, half = ARMS[level]
    return np.array([p["n"], p["e"]]) + perp * side * half, z - INSULATOR


def _wires(
    start: tuple[np.ndarray, float, np.ndarray],
    pylons: list[dict],
    end: tuple[np.ndarray, float, np.ndarray],
) -> list[dict]:
    """Every conductor and the earth wire in every span, as catenaries (parabolas): points
    (north, east, height above the straight line between the two feet, fraction along)."""
    out = []
    stops = [("gantry", start)] + [("pylon", p) for p in pylons] + [("gantry", end)]
    for (ka, a), (kb, b) in itertools.pairwise(stops):
        pa_foot = np.array([a["n"], a["e"]]) if ka == "pylon" else a[0]
        pb_foot = np.array([b["n"], b["e"]]) if kb == "pylon" else b[0]
        span = float(np.hypot(*(pb_foot - pa_foot)))
        for level, side in [(-1, 0)] + [(lv, sd) for lv in range(3) for sd in (-1, 1)]:
            if level < 0 and (ka != "pylon" or kb != "pylon"):
                continue
            pa, za = _attach(a, level, side) if ka == "pylon" else (a[0] + a[2] * side * 7, a[1])
            pb, zb = _attach(b, level, side) if kb == "pylon" else (b[0] + b[2] * side * 7, b[1])
            sag = (0.022 if level < 0 else 0.032) * span
            s = np.linspace(0, 1, max(8, int(span / 25)))
            ne = pa[None] + (pb - pa)[None] * s[:, None]
            z = za + (zb - za) * s - 4 * sag * s * (1 - s)
            out.append(
                {
                    "a": pa_foot.tolist(),
                    "b": pb_foot.tolist(),
                    "pts": np.column_stack([ne, z, s]).round(2).tolist(),
                }
            )
    return out


def build_power(cfg: dict, city: City, groups, channels, h, ns, es) -> dict | None:
    """The station and its lines (everything ``apply_power`` adds to the map)."""
    ground = Ground(h, ns, es)
    got = _site(cfg, city, groups, ground, channels)
    if got is None:
        return None
    f, site = got
    buildings, towers, places, streets, labels, gate = _plant(cfg, f, site)
    rng = np.random.default_rng(cfg.get("seed", 1))
    # The access road from the gate to the nearest road or lane.
    reach = unary_union(
        [ln for _, ln in city.roads] + [ln for k, ln, _ in city.streets if k == "lane"]
    )
    to = np.asarray(nearest_points(Point(gate), reach)[1].coords[0])
    access = _route(ground, gate, to)
    streets.append(("access", access, 3.5))
    # A belt of trees outside the fence, on the land sides, with gaps.
    belt = (
        site.buffer(32)
        .difference(site.buffer(10))
        .difference(city.water.buffer(20))
        .difference(access.buffer(8))
    )
    belt = belt.difference(
        Polygon([f.at(-1e4, -50), f.at(1e4, -50), f.at(1e4, 60), f.at(-1e4, 60)])
    )
    x0, y0, x1, y1 = belt.bounds
    gn, ge = np.meshgrid(np.arange(x0, x1, 7.5), np.arange(y0, y1, 7.5), indexing="ij")
    p = np.stack([gn.ravel(), ge.ravel()], 1) + rng.uniform(-3, 3, (gn.size, 2))
    p = p[shapely.contains_xy(belt, p[:, 0], p[:, 1])]
    along = np.sin(
        np.arctan2(*(p - np.asarray(site.centroid.coords[0])).T[::-1]) * 7 + rng.uniform(0, 6)
    )
    p = p[(rng.random(len(p)) < 0.55 + 0.35 * along)]
    belt_trees = np.column_stack(
        [
            p,
            rng.uniform(3, 5, len(p)),
            rng.uniform(10, 19, len(p)),
            np.zeros(len(p)),
            rng.integers(0, 5, len(p)),
        ]
    )

    # The lines.
    lines_out, sub = [], None
    forbidden = unary_union(
        [
            city.water.buffer(12),
            unary_union([ln for _, ln in city.roads]).buffer(15),
            unary_union([b.parts[0].footprint for b in city.buildings]).buffer(12),
            site.buffer(15),
        ]
    )
    for k, ln in enumerate(cfg.get("lines", [])):
        start = f.at(
            -60 + 140 * k, 35 + cfg.get("size_m", [480, 380])[1] - 6
        )  # the switchyard's edge
        start_dir = f.v
        river = LineString(np.asarray(channels[0].line))
        main_water = unary_union(
            [_band(c) for c in channels if c.name in (channels[0].name, "arm")]
        )
        if ln.get("to") == "city":
            got = _substation(
                cfg,
                city,
                ground,
                np.asarray(site.centroid.coords[0]),
                river,
                _river_side(river, np.asarray(site.centroid.coords[0])),
                main_water,
            )
            if got is None:
                continue
            sub, sub_blds, entry = got
            buildings += sub_blds
            places.append(("switchyard", "", sub))
            target = entry
        else:
            target = np.asarray(ln["to_ne_m"], float)
        gn_, ge_, cost = _cost(
            city, ground, site, sub, bank=main_water if ln.get("along_river") else None
        )
        out_of_site = start + start_dir * 60
        path = _cheapest(gn_, ge_, cost, out_of_site, target)
        path = _taut(np.vstack([out_of_site, path[1:-1], target]), gn_, ge_, cost)
        route = LineString(np.vstack([start, path]))
        pylons = _pylons(route, float(cfg.get("span_m", 330.0)), forbidden)
        d_end = np.asarray(route.coords[-1]) - np.asarray(route.coords[-2])
        d_end = d_end / max(np.hypot(*d_end), 1e-6)
        perp = lambda d: np.array([-d[1], d[0]])  # noqa: E731
        wires = _wires(
            (start, 16.0, perp(start_dir)),
            pylons,
            (target, 16.0 if ln.get("to") == "city" else 37.0, perp(d_end)),
        )
        lines_out.append({"name": ln["name"], "route": route, "pylons": pylons, "wires": wires})

    # Felled and cleared: the site, the access road, the substation, round each pylon and
    # under each line.
    clear = unary_union(
        [
            site.buffer(10),
            access.buffer(5),
            *([sub.buffer(8)] if sub is not None else []),
            *[Point(p["n"], p["e"]).buffer(14) for ln in lines_out for p in ln["pylons"]],
        ]
    )
    corridor = unary_union([ln["route"].buffer(10) for ln in lines_out])
    tower_labels = [
        Building(
            [Part(_ngon((t["n"], t["e"]), t["r_base"], 24), 0, t["height"])],
            "cooling",
            "Cooling tower",
        )
        for t in towers[:1]
    ]
    return {
        "site": site,
        "buildings": buildings,
        "towers": towers,
        "places": places,
        "streets": streets,
        "labels": labels + tower_labels,
        "belt": belt_trees,
        "lines": lines_out,
        "clear": clear,
        "corridor": corridor,
    }


def apply_power(city: City, state: dict) -> None:
    """Put the station and its lines on the map: clear what was there, add what is new."""
    clear, corridor = state["clear"], state["corridor"]
    shapely.prepare(clear)
    keep = [
        b
        for b in city.buildings
        if not clear.intersects(b.parts[0].footprint) or b.kind not in ("house", "barn", "modern")
    ]
    city.buildings = keep
    if len(city.trees):
        city.trees = city.trees[~shapely.contains_xy(clear, city.trees[:, 0], city.trees[:, 1])]
    if len(city.forest):
        f = city.forest
        gone = shapely.contains_xy(clear, f[:, 0], f[:, 1]) | shapely.contains_xy(
            corridor, f[:, 0], f[:, 1]
        )
        city.forest = f[~gone]

    def cut(g):  # what is left of a field or wood the station takes some of
        return (
            [g]
            if not clear.intersects(g)
            else _polys((g if g.is_valid else g.buffer(0)).difference(clear))
        )

    city.fields = [(c, q, o) for c, p, o in city.fields for q in cut(p) if q.area > 1500]
    city.hedges = [
        seg
        for ln in city.hedges
        for seg in ([ln] if not clear.intersects(ln) else _lines(ln.difference(clear)))
    ]
    city.woods = [q for w in city.woods for q in cut(w) if q.area > 500]
    city.streets = [
        (k, seg, hw)
        for k, ln, hw in city.streets
        for seg in (
            _lines(ln.difference(state["site"].buffer(5))) if k in ("track", "lane") else [ln]
        )
    ]
    city.places += state["places"]
    city.streets += state["streets"]
    city.buildings += state["buildings"]
    city.labels += state["labels"]
    city.cooling_towers = state["towers"]
    city.pylons = [p for ln in state["lines"] for p in ln["pylons"]]
    city.wires = [w for ln in state["lines"] for w in ln["wires"]]
    if len(state["belt"]):
        city.forest = np.vstack([city.forest, state["belt"]])


# --- locking ----------------------------------------------------------------------------


def save_power(state: dict, path) -> None:
    import gzip
    import json

    from sketch_city import _wkt, building_record

    data = {
        "site": _wkt(state["site"]),
        "clear": _wkt(state["clear"]),
        "corridor": _wkt(state["corridor"]),
        "buildings": [building_record(b) for b in state["buildings"]],
        "labels": [building_record(b) for b in state["labels"]],
        "label_index": [
            next((i for i, b in enumerate(state["buildings"]) if b is lb), -1)
            for lb in state["labels"]
        ],
        "towers": state["towers"],
        "places": [[k, n, _wkt(p)] for k, n, p in state["places"]],
        "streets": [[k, _wkt(ln), hw] for k, ln, hw in state["streets"]],
        "belt": np.round(state["belt"], 2).tolist(),
        "lines": [
            {
                "name": ln["name"],
                "route": _wkt(ln["route"]),
                "pylons": ln["pylons"],
                "wires": ln["wires"],
            }
            for ln in state["lines"]
        ],
    }
    with gzip.open(path, "wt") as f:
        json.dump(data, f, separators=(",", ":"))


def load_power(path) -> dict:
    import gzip
    import json

    from sketch_city import building_from

    with gzip.open(path, "rt") as f:
        d = json.load(f)
    g = shapely.from_wkt
    buildings = [building_from(b) for b in d["buildings"]]
    labels = [
        buildings[i] if i >= 0 else building_from(b)
        for i, b in zip(d["label_index"], d["labels"], strict=True)
    ]
    return {
        "site": g(d["site"]),
        "clear": g(d["clear"]),
        "corridor": g(d["corridor"]),
        "buildings": buildings,
        "labels": labels,
        "towers": d["towers"],
        "places": [(k, n, g(w)) for k, n, w in d["places"]],
        "streets": [(k, g(w), hw) for k, w, hw in d["streets"]],
        "belt": np.asarray(d["belt"], float).reshape(-1, 6),
        "lines": [
            {
                "name": ln["name"],
                "route": g(ln["route"]),
                "pylons": ln["pylons"],
                "wires": ln["wires"],
            }
            for ln in d["lines"]
        ],
    }


# --- drawing ----------------------------------------------------------------------------


def _shade(base: str, n: np.ndarray) -> str:
    b = 0.55 + 0.45 * max(0.0, float(n @ LIGHT))
    rgb = [int(base[i : i + 2], 16) / 255 * b for i in (1, 3, 5)]
    return "#" + "".join(f"{round(min(float(v), 1.0) * 255):02x}" for v in rgb)


def tower_faces(t: dict, view: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """A natural-draught cooling tower (a hyperboloid shell, open at the top): its visible
    outside, far to near, then the opening and the inside of its far wall. Points are
    (north, east, height above its foot)."""
    H, rb, rt, zt = t["height"], t["r_base"], t["r_throat"], t["z_throat"]
    c = zt / math.sqrt((rb / rt) ** 2 - 1)

    def radius(z):
        return rt * np.sqrt(1 + ((z - zt) / c) ** 2)

    zs = np.linspace(0, H, 17)
    rs = radius(zs)
    ang = np.linspace(0, 2 * np.pi, 41)
    ctr = np.array([t["n"], t["e"]])
    out = []
    quads = []
    for k in range(len(zs) - 1):
        dr = (rs[k + 1] - rs[k]) / (zs[k + 1] - zs[k])
        for j in range(len(ang) - 1):
            a = 0.5 * (ang[j] + ang[j + 1])
            nrm = np.array([math.cos(a), math.sin(a), -dr])
            nrm /= np.linalg.norm(nrm)
            if nrm @ view <= 0:
                continue
            p = [
                np.array([*(ctr + r * np.array([math.cos(b), math.sin(b)])), z])
                for r, z, b in (
                    (rs[k], zs[k], ang[j]),
                    (rs[k], zs[k], ang[j + 1]),
                    (rs[k + 1], zs[k + 1], ang[j + 1]),
                    (rs[k + 1], zs[k + 1], ang[j]),
                )
            ]
            depth = float(np.array([math.cos(a), math.sin(a)]) @ view[:2])
            quads.append((depth, np.array(p), _shade(CONCRETE, nrm)))
    quads.sort(key=lambda q: q[0])  # the sides first, the front last
    out += [(p, col) for _, p, col in quads]
    top = np.array([[*(ctr + rs[-1] * np.array([math.cos(b), math.sin(b)])), H] for b in ang[:-1]])
    out.append((top, "#6c7075"))
    hv = view[:2] / max(np.linalg.norm(view[:2]), 1e-9)  # towards the viewer, level
    tan_e = view[2] / max(np.linalg.norm(view[:2]), 1e-9)
    far = []
    for b in ang:
        d = np.array([math.cos(b), math.sin(b)])
        cosf = -(d @ hv)  # 1 at the far side
        if cosf > 0:
            far.append((b, cosf))
    if far:
        rim = [
            np.array([*(ctr + rs[-1] * np.array([math.cos(b), math.sin(b)])), H]) for b, _ in far
        ]
        drop = [
            np.array(
                [
                    *(
                        ctr
                        + radius(H - 1.1 * rs[-1] * tan_e * cf)
                        * np.array([math.cos(b), math.sin(b)])
                    ),
                    H - 1.1 * rs[-1] * tan_e * cf,
                ]
            )
            for b, cf in far
        ]
        out.append((np.array(rim + drop[::-1]), "#b9b6b0"))
    return out


def plume(t: dict, wind_to: np.ndarray) -> list[tuple[np.ndarray, float, float]]:
    """The steam drifting off a cooling tower: (centre north, east, height; radius; opacity)."""
    out = []
    for k in range(6):
        c = np.array([t["n"], t["e"]]) + wind_to * (14 + 26 * k)
        out.append(
            (np.array([*c, t["height"] + 12 + 15 * k]), 26.0 + 7 * k, max(0.5 - 0.07 * k, 0.12))
        )
    return out


def pylon_segments(p: dict) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """A lattice pylon as segments (from, to, width), (north, east, height above its foot)."""
    c = np.array([p["n"], p["e"]])
    u = np.array([math.cos(p["heading"]), math.sin(p["heading"])])
    v = np.array([-u[1], u[0]])

    def corner(z, i):
        hw = (
            FOOT + (NECK - FOOT) * min(z / ARMS[-1][0], 1.0)
            if z <= ARMS[-1][0]
            else NECK * (1 - 0.7 * (z - ARMS[-1][0]) / (PEAK - ARMS[-1][0]))
        )
        su, sv = ((1, 1), (1, -1), (-1, -1), (-1, 1))[i]
        return np.array([*(c + u * su * hw + v * sv * hw), z])

    segs = []
    levels = [0.0, 7.0, 14.0, 21.0, ARMS[0][0], ARMS[1][0], ARMS[2][0]]
    for i in range(4):
        segs += [(corner(z0, i), corner(z1, i), 0.45) for z0, z1 in itertools.pairwise(levels)]
        segs.append((corner(ARMS[2][0], i), np.array([*c, PEAK]), 0.3))
        for z0, z1 in itertools.pairwise(levels):  # cross-bracing on each face
            j = (i + 1) % 4
            segs += [(corner(z0, i), corner(z1, j), 0.18), (corner(z0, j), corner(z1, i), 0.18)]
    for z, half in ARMS:  # the cross-arms: a frame each side, tapering to the tip
        for s in (-1, 1):
            tip = np.array([*(c + v * s * half), z])
            for i in range(4):
                segs.append((corner(z, i), tip, 0.22))
            segs.append((tip, tip - np.array([0, 0, INSULATOR]), 0.35))  # insulator string
    return segs


def _strip(a: np.ndarray, b: np.ndarray, w: float, view: np.ndarray) -> np.ndarray:
    side = np.cross(b - a, view)
    side = side / max(np.linalg.norm(side), 1e-9) * w / 2
    return np.array([a + side, b + side, b - side, a - side])


def power_items(city: City, view: np.ndarray, project, ground):
    """Everything of the station and its lines to stand on the ground, for a painter's
    drawing: ([depth, polygons, colours, (north, east), foot height, height]), and shadows
    (page polygons). ``project(n, e, z) → (x, y)``; ``ground(n, e) → z``."""
    items, shadows = [], []
    wind_to = np.array([math.cos(math.radians(45)), math.sin(math.radians(45))])

    def page(pts, z0):
        x, y = project(pts[:, 0], pts[:, 1], pts[:, 2] + z0)
        return np.column_stack([x, y])

    for t in city.cooling_towers:
        z0 = float(np.atleast_1d(ground(t["n"], t["e"]))[0])
        faces = tower_faces(t, view)
        items.append(
            [
                t["n"] + t["e"],
                [page(p, z0) for p, _ in faces],
                [c for _, c in faces],
                np.array([t["n"], t["e"]]),
                z0,
                t["height"],
            ]
        )
        ring = _ngon((t["n"], t["e"]), t["r_base"], 24)
        q = np.asarray(ring.exterior.coords)
        g = np.vstack([q, q + SHADOW * t["height"]])
        hull = np.asarray(shapely.MultiPoint(g).convex_hull.exterior.coords)
        shadows.append(page(np.column_stack([hull, np.zeros(len(hull))]), z0))
        for c3, r, a in plume(t, wind_to):
            x, y = project(np.array([c3[0]]), np.array([c3[1]]), np.array([c3[2] + z0]))
            ang = np.linspace(0, 2 * np.pi, 25)
            blob = np.column_stack([x[0] + r * np.cos(ang), y[0] + 0.8 * r * np.sin(ang)])
            items.append(
                [
                    t["n"] + t["e"] - 30,
                    [blob],
                    [f"#ffffff{int(a * 255):02x}"],
                    np.array([t["n"], t["e"]]),
                    z0,
                    c3[2] + r,
                ]
            )
    for p in city.pylons:
        z0 = float(np.atleast_1d(ground(p["n"], p["e"]))[0])
        segs = pylon_segments(p)
        polys = [page(_strip(a, b, w, view), z0) for a, b, w in segs]
        items.append(
            [p["n"] + p["e"], polys, [STEEL] * len(polys), np.array([p["n"], p["e"]]), z0, PEAK]
        )
        for a, b, w in segs[::3]:
            q = np.array([a[:2] + SHADOW * a[2], b[:2] + SHADOW * b[2]])
            shadows.append(
                page(
                    np.column_stack(
                        [
                            _strip(np.r_[q[0], 0], np.r_[q[1], 0], w * 1.5, np.array([0, 0, 1.0]))[
                                :, :2
                            ],
                            np.zeros(4),
                        ]
                    ),
                    z0,
                )
            )
    for w in city.wires:
        pts = np.asarray(w["pts"], float)
        ga, gb = (float(np.atleast_1d(ground(*w[k]))[0]) for k in ("a", "b"))
        z = pts[:, 2] + ga + (gb - ga) * pts[:, 3]
        p3 = np.column_stack([pts[:, :2], z])
        for k in range(len(p3) - 1):
            quad = _strip(p3[k], p3[k + 1], 0.25, view)
            mid = 0.5 * (p3[k] + p3[k + 1])
            items.append([mid[0] + mid[1], [page(quad, 0.0)], [WIRE], mid[:2], 0.0, mid[2]])
    return items, shadows
