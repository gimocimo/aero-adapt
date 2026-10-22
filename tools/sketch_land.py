# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportOperatorIssue=false, reportReturnType=false, reportOptionalMemberAccess=false
# pyright: reportGeneralTypeIssues=false
# (Drawing tools over shapely, whose set operations are typed as returning a BaseGeometry; the
# code relies on them being polygons or lines at run time, which the checker cannot narrow.)
"""The land between the settlements (configs/sketches/<name>.yaml ``land``): a wind farm, woods
on the hills, copses, and the fields of the farms and villages.

The wind farm: turbines picked, best first, from candidate sites on gentle ground in its
area, never closer to each other than about 3 rotor diameters across the wind and 5 along
it (so they do not stand in each other's wake), nor within 300 m of a home (barns and silos
may stand nearer); each on a crane pad, joined by gravel service tracks and one access track
to the nearest road, which keeps to gentle ground.

Woods: on each range's slopes where a smooth random field runs high, covering about the
share of the slopes the sketch asks for, with ragged edges; copses scattered over the farm
land. Hills are wooded where they are steep and poor; the valley is farmed.

Fields: the open land (not town, village, farmyard, wood, water, road or steep high
ground) is shared among the farms and villages, each taking the land nearest to it, and
each share is cut into fields: across its longer side, a little off the middle and a little
askew, again and again, until fields are the size enclosure fields are (paddocks near the
farm, larger fields beyond). Crops: grazing on the slopes and by the farm, wheat, barley,
oilseed rape, ploughland and stubble elsewhere. A hedge on every boundary, with a tree now
and then.
"""

from __future__ import annotations

import math
import zlib

import contourpy
import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points, split, unary_union, voronoi_diagram

from sketch_city import Building, City, Part, _frame, _lines, _ngon, _polys, _rect
from sketch_country import Ground

CROPS = {  # colour, share of the flat fields
    "wheat": ("#d4bd6a", 0.24),
    "barley": ("#e0d293", 0.15),
    "oilseed": ("#e8d64a", 0.09),
    "pasture": ("#8cc35e", 0.22),
    "meadow": ("#a9d376", 0.1),
    "ploughed": ("#9f7c58", 0.12),
    "stubble": ("#d8ca95", 0.08),
    "orchard": ("#a6cf7c", 0.0),  # chosen near the villages and farms, not at random
    "solar": ("#a9c88c", 0.0),  # grass under the panels
    "airstrip": ("#b4d48d", 0.0),  # mown grass
}
ARABLE = {"wheat", "barley", "oilseed", "ploughed", "stubble"}
WOOD = "#3f6f37"
CROWNS = ["#4c8a3c", "#5a9a45", "#3f7a33", "#679f4c", "#4a7f3a"]
GRAVEL = "#d3c6a4"


def _rng(cfg: dict, name: str):
    return np.random.default_rng([cfg.get("seed", 1), zlib.crc32(name.encode())])


def _smooth_noise(rng, shape, sigma: float) -> np.ndarray:
    z = gaussian_filter(rng.standard_normal(shape), sigma)
    return z / z.std()


def _polygons(mask: np.ndarray, ns: np.ndarray, es: np.ndarray, min_area: float) -> list[Polygon]:
    """The areas where ``mask`` is set (a grid over ``ns`` × ``es``), as polygons with holes."""
    gen = contourpy.contour_generator(
        es, ns, mask.astype(float), fill_type=contourpy.FillType.OuterOffset
    )
    out = []
    for pts, offs in zip(*gen.filled(0.5, 2.0), strict=True):
        rings = [pts[offs[i] : offs[i + 1]][:, ::-1] for i in range(len(offs) - 1)]
        p = Polygon(rings[0], [r for r in rings[1:] if len(r) > 3]).buffer(0)
        out += [q for q in _polys(p) if q.area >= min_area]
    return out


def _route(ground: Ground, a: np.ndarray, b: np.ndarray, pad: float = 300.0) -> LineString:
    """The gentlest way over the ground from ``a`` to ``b`` (a track keeps to easy grades)."""
    n0, n1 = min(a[0], b[0]) - pad, max(a[0], b[0]) + pad
    e0, e1 = min(a[1], b[1]) - pad, max(a[1], b[1]) + pad
    cell = 10.0
    gn, ge = np.arange(n0, n1, cell), np.arange(e0, e1, cell)
    nn, ee = np.meshgrid(gn, ge, indexing="ij")
    z = ground.at(ground.h, np.stack([nn.ravel(), ee.ravel()], 1)).reshape(nn.shape)
    rows, cols = z.shape
    idx = np.arange(rows * cols).reshape(rows, cols)
    src, dst, cost = [], [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        i0, i1 = slice(0, rows - di), slice(di, rows)
        j0, j1 = (
            (slice(0, cols - dj), slice(dj, cols))
            if dj >= 0
            else (slice(-dj, cols), slice(0, cols + dj))
        )
        length = cell * math.hypot(di, dj)
        grade = np.abs(z[i1, j1] - z[i0, j0]) / length
        src.append(idx[i0, j0].ravel())
        dst.append(idx[i1, j1].ravel())
        cost.append((length * (1 + (grade / 0.08) ** 2)).ravel())
    graph = coo_matrix(
        (np.concatenate(cost), (np.concatenate(src), np.concatenate(dst))), shape=(rows * cols,) * 2
    )

    def node(p):
        return int(np.clip(round((p[0] - n0) / cell), 0, rows - 1)) * cols + int(
            np.clip(round((p[1] - e0) / cell), 0, cols - 1)
        )

    start, goal = node(a), node(b)
    _, pred = dijkstra(graph.tocsr(), directed=False, indices=start, return_predecessors=True)
    path = [goal]
    while path[-1] != start and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    pts = np.array([[nn.ravel()[k], ee.ravel()[k]] for k in path[::-1]])
    return LineString(np.vstack([a, pts[1:-1], b])).simplify(4.0)


def _side(line: LineString, p: np.ndarray) -> float:
    """Which side of ``line`` the point is on (+1 or -1)."""
    s = line.project(Point(p))
    a = np.asarray(line.interpolate(max(s - 5, 0)).coords[0])
    b = np.asarray(line.interpolate(min(s + 5, line.length)).coords[0])
    t, q = b - a, p - a
    return float(np.sign(t[0] * q[1] - t[1] * q[0]))


def _turbine_sites(
    wf: dict, sk: dict, city: City, ground: Ground, rng
) -> tuple[np.ndarray, np.ndarray]:
    """Where a turbine may stand in the wind farm's area, and how good each place is."""
    (a0, b0), (a1, b1) = wf["area_ne_m"]
    rotor = float(wf.get("rotor_m", 92.0))
    homes = unary_union(
        [b.parts[0].footprint for b in city.buildings if b.kind == "house"]
    )  # not barns or silos
    keep_off = unary_union(
        [
            homes.buffer(wf.get("from_houses_m", 300.0)),
            city.outline.buffer(350),
            unary_union([ln for _, ln in city.roads]).buffer(60),
            city.water.buffer(60),
        ]
    )
    shapely.prepare(keep_off)
    g = np.stack(
        np.meshgrid(np.arange(a0, a1, 40.0), np.arange(b0, b1, 40.0), indexing="ij"), -1
    ).reshape(-1, 2)
    g = g + rng.uniform(-15, 15, g.shape)
    n0, n1, e0, e1 = ground.box
    margin = rotor / 2 + 30
    ok = (
        (g[:, 0] > n0 + margin)
        & (g[:, 0] < n1 - margin)
        & (g[:, 1] > e0 + margin)
        & (g[:, 1] < e1 - margin)
    )
    g = g[ok]
    slope = ground.at(gaussian_filter(ground.slope, 2), g)
    z = ground.at(ground.h, g)
    g = g[(slope < wf.get("max_slope", 0.15)) & (z < ground.plain + wf.get("max_rise_m", 120.0))]
    g = g[~shapely.contains_xy(keep_off, g[:, 0], g[:, 1])]
    g = g[ground.at(ground.h, g) > ground.plain + wf.get("min_rise_m", 0.0)]
    if wf.get("city_side_of"):  # on the slopes facing the city
        rg = next(r for r in sk["terrain"]["ranges"] if r["name"] == wf["city_side_of"])
        crest = LineString(rg["crest_ne_m"])
        ref = _side(crest, np.asarray(city.outline.centroid.coords[0]))
        g = g[np.array([_side(crest, p) == ref for p in g], bool)] if len(g) else g
    score = (
        ground.at(ground.h, g) / 200.0
        - 3.0 * ground.at(ground.slope, g)
        + rng.normal(0, 0.05, len(g))
    )
    return g, score


def _spaced(
    wf: dict, g: np.ndarray, score: np.ndarray, picked: list[np.ndarray], count: int
) -> list[np.ndarray]:
    """The best sites, in turn, that keep clear of the turbines already there: ``spacing_rotors``
    rotor diameters along the wind and across it (so none stands in another's wake)."""
    rotor = float(wf.get("rotor_m", 92.0))
    wind = math.radians(wf.get("wind_from_deg", 225.0))
    down = np.array([-math.cos(wind), -math.sin(wind)])  # the way the wind blows
    across = np.array([-down[1], down[0]])
    s_down, s_across = wf.get("spacing_rotors", [5.0, 3.3])
    new: list[np.ndarray] = []
    for k in np.argsort(-score):
        if len(picked) + len(new) >= count:
            break
        p = g[k]
        if all(
            ((p - q) @ down / (s_down * rotor)) ** 2 + ((p - q) @ across / (s_across * rotor)) ** 2
            >= 1
            for q in [*picked, *new]
        ):
            new.append(p)
    return new


def _turbine(wf: dict, p: np.ndarray, rng) -> dict:
    return {
        "n": float(p[0]),
        "e": float(p[1]),
        "hub": float(wf.get("hub_m", 85.0)),
        "rotor": float(wf.get("rotor_m", 92.0)),
        "facing": float(math.radians(wf.get("wind_from_deg", 225.0))),
        "phase": float(rng.uniform(0, 120)),
    }


def _pad(p: np.ndarray, tracks: list[LineString]) -> Polygon:
    """A turbine's crane pad, along the track that reaches it."""
    t = min(tracks, key=lambda ln: ln.distance(Point(p)))
    _, th, _, _ = _frame(t.buffer(1))
    return _rect(p, th, 40, 22)


def _wind_farm(wf: dict, sk: dict, city: City, ground: Ground, rng):
    """(turbines, pads, tracks, buildings) of the wind farm."""
    g, score = _turbine_sites(wf, sk, city, ground, rng)
    picked = _spaced(wf, g, score, [], int(wf.get("turbines", 12)))
    if not picked:
        return [], [], [], []
    turbines = [_turbine(wf, p, rng) for p in picked]
    pts = np.array(picked)
    # Service tracks: the shortest network joining the turbines, and one access track.
    d = np.hypot(*(pts[:, None] - pts[None]).transpose(2, 0, 1))
    mst = minimum_spanning_tree(d).tocoo()
    tracks = [LineString([pts[i], pts[j]]) for i, j in zip(mst.row, mst.col, strict=True)]
    roads = unary_union(
        [ln for _, ln in city.roads] + [ln for k, ln, _ in city.streets if k in ("lane", "track")]
    )
    first = min(range(len(pts)), key=lambda i: Point(pts[i]).distance(roads))
    to = np.asarray(nearest_points(Point(pts[first]), roads)[1].coords[0])
    tracks.append(_route(ground, pts[first], to))
    pads = [_pad(p, tracks) for p in pts]
    sub = _rect(
        to + (pts[first] - to) / max(float(np.hypot(*(pts[first] - to))), 1) * 40, 0.4, 14, 9
    )
    buildings = [Building([Part(sub, 0, 5.0)], "modern", "substation")]
    return turbines, pads, tracks, buildings


def _more_turbines(
    wf: dict, sk: dict, city: City, ground: Ground, rng, kept: list[dict], kept_tracks
):
    """Turbines added among those already standing (up to ``turbines`` in all), each with its
    pad and a track to the nearest turbine. (turbines, pads, tracks)."""
    g, score = _turbine_sites(wf, sk, city, ground, rng)
    have = [np.array([t["n"], t["e"]]) for t in kept]
    new = _spaced(wf, g, score, have, int(wf.get("turbines", len(kept))))
    turbines, tracks = [], []
    for p in new:
        q = min(have, key=lambda o: float(np.hypot(*(o - p))))
        tracks.append(LineString([q, p]))
        have.append(p)
        turbines.append(_turbine(wf, p, rng))
    pads = [_pad(np.array([t["n"], t["e"]]), [*kept_tracks, *tracks]) for t in turbines]
    return turbines, pads, tracks


def turbine_faces(t: dict, view: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """A wind turbine's visible faces (north, east, up above its foot), far to near: the
    tower (tapering), the nacelle, then the three blades on the rotor's upwind face."""
    from sketch_city import faces

    c = np.array([t["n"], t["e"]])
    hub, r = t["hub"], t["rotor"] / 2
    out = []
    for r0, z0, z1 in ((2.3, 0, hub * 0.5), (1.8, hub * 0.5, hub - 2)):
        for pts, shade, _ in faces(Part(_ngon(c, r0, 10), z0, z1), view):
            out.append((pts, _grey(0.95 * shade)))
    face = np.array([math.cos(t["facing"]), math.sin(t["facing"])])  # the rotor faces the wind
    nac = _rect(c - face * 2, t["facing"], 12, 4)
    for pts, shade, _ in faces(Part(nac, hub - 2, hub + 2), view):
        out.append((pts, _grey(0.92 * shade)))
    hub_c = np.array([*(c + face * 4.5), hub])
    lateral = np.array([-face[1], face[0], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    for k in range(3):
        a = math.radians(t["phase"] + 120 * k)
        along = math.cos(a) * up + math.sin(a) * lateral
        side = math.cos(a + math.pi / 2) * up + math.sin(a + math.pi / 2) * lateral
        blade = np.array(
            [
                hub_c + along * 1.5 + side * 1.6,
                hub_c + along * r + side * 0.4,
                hub_c + along * r - side * 0.4,
                hub_c + along * 1.5 - side * 1.2,
            ]
        )
        out.append((blade, "#f4f5f6"))
    return out


def _grey(v: float) -> str:
    x = int(np.clip(v, 0, 1) * 240)
    return f"#{x:02x}{x:02x}{min(x + 6, 255):02x}"


def _woods(spec: list[dict], sk: dict, ground: Ground, ns, es, rng, avoid) -> list[Polygon]:
    nn, ee = np.meshgrid(ns, es, indexing="ij")
    grid = np.stack([nn.ravel(), ee.ravel()], 1)
    rel = ground.h - ground.plain
    mask = np.zeros(ground.h.shape, bool)
    ranges = {r["name"]: r for r in sk["terrain"].get("ranges", [])}
    for w in spec:
        rg = ranges[w["range"]]
        crest = np.asarray(rg["crest_ne_m"], float)
        t = np.linspace(0, 1, 400)
        seg = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(crest, axis=0), axis=1)])
        line = np.stack([np.interp(t * seg[-1], seg, crest[:, i]) for i in (0, 1)], 1)
        near = KDTree(line).query(grid)[0].reshape(nn.shape) < 1.7 * rg["width_m"]
        cand = (
            near
            & (rel > w.get("above_m", 10.0))
            & (rel < w.get("below_m", 1e9))
            & (ground.slope < 0.7)
        )
        score = 0.65 * _smooth_noise(rng, nn.shape, 22) + 0.35 * _smooth_noise(rng, nn.shape, 7)
        if cand.any():
            mask |= cand & (score > np.quantile(score[cand], 1 - w.get("cover", 0.4)))
    mask = gaussian_filter(mask.astype(float), 1.2) > 0.5
    return [
        q
        for p in _polygons(mask, ns, es, 5000)
        for q in _polys(p.difference(avoid).simplify(3.0))
        if q.area > 5000
    ]


def _copses(count: int, ground: Ground, rng, avoid, free) -> list[Polygon]:
    """Small woods scattered over the farm land."""
    out: list[Polygon] = []
    n0, n1, e0, e1 = ground.box
    for _ in range(4000):
        if len(out) >= count:
            break
        q = np.array([rng.uniform(n0 + 150, n1 - 150), rng.uniform(e0 + 150, e1 - 150)])
        if not free.contains(Point(q)) or any(o.distance(Point(q)) < 300 for o in out):
            continue
        r = rng.uniform(45, 110)
        a = np.linspace(0, 2 * math.pi, 40, endpoint=False)
        wob = (
            1 + 0.25 * np.sin(3 * a + rng.uniform(0, 6)) + 0.15 * np.sin(5 * a + rng.uniform(0, 6))
        )
        blob = Polygon(q + (r * wob)[:, None] * np.stack([np.cos(a), np.sin(a)], 1)).difference(
            avoid
        )
        out += [p for p in _polys(blob) if p.area > 3000]
    return out


class Raster:
    """A 5 m grid over the map, to ask where things are quickly (masks, distances, noise)."""

    def __init__(self, box_ne, cell: float = 5.0) -> None:
        n0, n1, e0, e1 = box_ne
        self.cell = cell
        self.ns, self.es = np.arange(n0, n1 + cell / 2, cell), np.arange(e0, e1 + cell / 2, cell)
        self.nn, self.ee = np.meshgrid(self.ns, self.es, indexing="ij")

    def mask(self, geom) -> np.ndarray:
        if geom is None or geom.is_empty:
            return np.zeros(self.nn.shape, bool)
        return shapely.contains_xy(geom, self.nn, self.ee)

    def at(self, grid: np.ndarray, p: np.ndarray, order: int = 1) -> np.ndarray:
        p = np.atleast_2d(p)
        return map_coordinates(
            grid.astype(float),
            [(p[:, 0] - self.ns[0]) / self.cell, (p[:, 1] - self.es[0]) / self.cell],
            order=order,
            mode="nearest",
        )

    def noise(self, rng, sigma_m: float) -> np.ndarray:
        return _smooth_noise(rng, self.nn.shape, sigma_m / self.cell)

    def jittered(self, rng, spacing: float, keep: np.ndarray | None = None) -> np.ndarray:
        """Points on a jittered grid ``spacing`` m apart (where ``keep`` is set, if given)."""
        gn = np.arange(self.ns[0], self.ns[-1], spacing)
        ge = np.arange(self.es[0], self.es[-1], spacing)
        nn, ee = np.meshgrid(gn, ge, indexing="ij")
        p = np.stack([nn.ravel(), ee.ravel()], 1) + rng.uniform(-0.45, 0.45, (nn.size, 2)) * spacing
        return p if keep is None else p[self.at(keep, p, order=0) > 0.5]


def _tree_rows(kind, p, r, h, shade) -> np.ndarray:
    n = len(p)
    if n == 0:
        return np.zeros((0, 6))
    return np.column_stack(
        [
            p,
            np.broadcast_to(r, n),
            np.broadcast_to(h, n),
            np.full(n, kind),
            np.broadcast_to(shade, n),
        ]
    )


def _forest(cfg: dict, sk: dict, woods: list[Polygon], wooded, rng, ras: Raster):
    """Every tree in the woods: (north, east, crown radius, height, kind, shade); kind 0 a
    broadleaf (oak, beech, ash: round crowns, about 7 m apart), 1 a conifer (spruce, pine:
    narrow cones, planted 5.5 m apart in rows), 2 a shrub (hazel, hawthorn, holly). Conifers
    grow in stands, more of them on the ranges the sketch says (plantations on the high
    ground), few in the copses; every plantation has a broadleaf edge. Inside, the cover
    varies: thinner patches, and glades open to the sky. Returns (trees, glades, the woods'
    mask on the raster)."""
    if not woods:
        return np.zeros((0, 6)), [], np.zeros(ras.nn.shape, bool)
    from scipy.stats import norm

    wood = ras.mask(wooded)
    inside = distance_transform_edt(wood) * ras.cell  # metres in from the edge
    stands = ras.noise(rng, 30)  # stands some 120 m across
    angle = (ras.noise(rng, 45) > 0).astype(int) + 2 * (ras.noise(rng, 45) > 0)
    thin = ras.noise(rng, 40)
    glade_n = ras.noise(rng, 22)
    cover = np.clip(0.88 + 0.2 * thin, 0.35, 1.0)
    cover[thin < -1.25] = 0.3  # thin, scrubby patches
    glade = wood & (glade_n > 1.75) & (inside > 25)  # clearings, away from the edge
    cover[glade] = 0.0
    glades = _polygons(glade, ras.ns, ras.es, 300)

    ranges = {r["name"]: r for r in sk["terrain"].get("ranges", [])}
    specs = [
        (
            LineString(ranges[w["range"]]["crest_ne_m"]),
            ranges[w["range"]]["width_m"],
            w.get("conifer", 0.2),
        )
        for w in cfg.get("woods", [])
    ]

    def conifer_share(p):
        best = np.full(len(p), cfg.get("copse_conifer", 0.05))
        d_best = np.full(len(p), np.inf)
        for crest, width, share in specs:
            d = shapely.distance(shapely.points(p), crest) / width
            near = (d < 1.7) & (d < d_best)
            best[near], d_best[near] = share, d[near]
        return best

    # Broadleaves on a loose grid; in a plantation only along its edge.
    b = ras.jittered(rng, 7.0, wood)
    din = ras.at(inside, b)
    is_conifer_stand = ras.at(stands, b) < norm.ppf(conifer_share(b))
    keep = (~is_conifer_stand | (din < 9)) & (
        rng.random(len(b)) < ras.at(cover, b) * np.where(din < 10, 0.75, 1.0)
    )
    b, din = b[keep], din[keep]
    big = rng.random(len(b)) < 0.05  # the odd old giant
    edge = din < 10
    shrub = edge & (rng.random(len(b)) < 0.35)  # the woodland edge's shrub layer
    trees = [
        _tree_rows(
            0,
            b[~shrub],
            rng.uniform(3.4, 5.6, (~shrub).sum()) * np.where(big[~shrub], 1.4, 1.0),
            rng.uniform(13, 23, (~shrub).sum()) * np.where(big[~shrub], 1.15, 1.0),
            rng.integers(0, 5, (~shrub).sum()),
        ),
        _tree_rows(
            2,
            b[shrub],
            rng.uniform(1.8, 3.0, shrub.sum()),
            rng.uniform(4, 7, shrub.sum()),
            rng.integers(0, 3, shrub.sum()),
        ),
    ]
    # Conifers in rows, each stand's rows at its own angle; felled coupes left open.
    x0, y0, x1, y1 = wooded.bounds
    for k, ang in enumerate((0.3, 0.9, 1.5, 2.2)):
        u = np.array([math.cos(ang), math.sin(ang)])
        v = np.array([-u[1], u[0]])
        span = max(x1 - x0, y1 - y0) * 0.75
        mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        a, bb = np.meshgrid(np.arange(-span, span, 5.5), np.arange(-span, span, 5.5), indexing="ij")
        p = (
            mid
            + a.ravel()[:, None] * u
            + bb.ravel()[:, None] * v
            + rng.uniform(-0.7, 0.7, (a.size, 2))
        )
        p = p[(p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)]
        p = p[(ras.at(angle, p, order=0) == k) & (ras.at(wood, p, order=0) > 0.5)]
        p = p[(ras.at(stands, p) < norm.ppf(conifer_share(p))) & (ras.at(inside, p) >= 9)]
        p = p[rng.random(len(p)) < np.minimum(ras.at(cover, p) * 1.1, 1.0)]
        trees.append(
            _tree_rows(
                1,
                p,
                rng.uniform(2.1, 3.0, len(p)),
                rng.uniform(15, 27, len(p)),
                rng.integers(0, 3, len(p)),
            )
        )
    return np.vstack(trees), glades, wood


def _fringe(cfg: dict, rng, ras: Raster, wood: np.ndarray, grounds: dict) -> np.ndarray:
    """Woods do not end at a line: past the edge the trees thin out, smaller and scrubbier,
    over a width that varies along the edge (a few metres beside a ploughed field, up to
    some 120 m into rough grazing)."""
    outside = distance_transform_edt(~wood) * ras.cell
    width = np.clip(cfg.get("fringe_m", 50.0) + 38.0 * ras.noise(rng, 70), 6.0, 120.0)
    p = ras.jittered(rng, 6.0)
    d = ras.at(outside, p)
    p, d = p[(d > 0.5) & (d < 130)], d[(d > 0.5) & (d < 130)]
    w = ras.at(width, p)
    keep = rng.random(len(p)) < 0.85 * np.exp(-d / (0.42 * w)) * ras.at(grounds["welcome"], p)
    p, d, w = p[keep], d[keep], w[keep]
    shrub = rng.random(len(p)) < np.clip(0.2 + 0.7 * d / w, 0.2, 0.85)
    grow = np.clip(1.0 - 0.6 * d / w, 0.4, 1.0)  # smaller further out
    tb = ~shrub
    return np.vstack(
        [
            _tree_rows(
                0,
                p[tb],
                rng.uniform(2.6, 4.6, tb.sum()) * grow[tb],
                rng.uniform(8, 17, tb.sum()) * grow[tb],
                rng.choice([0, 1, 3, 3, 4], tb.sum()),
            ),
            _tree_rows(
                2,
                p[shrub],
                rng.uniform(1.5, 3.0, shrub.sum()),
                rng.uniform(3, 6.5, shrub.sum()),
                rng.integers(0, 3, shrub.sum()),
            ),
        ]
    )


def _scrub(cfg: dict, rng, ras: Raster, grounds: dict) -> np.ndarray:
    """Rough ground (steep, high or left over) grows scrub and the odd tree, in patches:
    some hillsides nearly bare, others thick with gorse, hawthorn and birch."""
    patch = ras.noise(rng, 160)
    density = np.clip(cfg.get("scrub", 0.05) + 0.16 * patch, 0.0, 0.35) * grounds["rough"]
    p = ras.jittered(rng, 8.0)
    p = p[rng.random(len(p)) < ras.at(density, p)]
    k = rng.random(len(p))
    shrub, conifer = k < 0.65, k > 0.95
    tree = ~shrub & ~conifer
    return np.vstack(
        [
            _tree_rows(
                2,
                p[shrub],
                rng.uniform(1.6, 3.2, shrub.sum()),
                rng.uniform(3, 6, shrub.sum()),
                rng.integers(0, 3, shrub.sum()),
            ),
            _tree_rows(
                0,
                p[tree],
                rng.uniform(2.4, 4.2, tree.sum()),
                rng.uniform(7, 14, tree.sum()),
                rng.choice([0, 3, 4], tree.sum()),
            ),
            _tree_rows(
                1,
                p[conifer],
                rng.uniform(1.6, 2.4, conifer.sum()),
                rng.uniform(6, 13, conifer.sum()),
                rng.integers(0, 3, conifer.sum()),
            ),
        ]
    )


def _riverside(channels, rng, ras: Raster, grounds: dict, town) -> np.ndarray:
    """Willows and alders along the banks out of town: in runs and clumps, with gaps."""
    along = ras.noise(rng, 90)
    out = []
    for c in channels:
        line, width = np.asarray(c.line), np.asarray(c.width)
        seg = np.hypot(*np.diff(line, axis=0).T)
        s = np.concatenate([[0], np.cumsum(seg)])
        for side in (-1, 1):
            t = rng.uniform(0, 8)
            while t < s[-1]:
                k = int(np.searchsorted(s, t)) - 1
                k = min(max(k, 0), len(line) - 2)
                d = (line[k + 1] - line[k]) / max(seg[k], 1e-6)
                nrm = np.array([-d[1], d[0]]) * side
                q = (
                    line[k]
                    + d * (t - s[k])
                    + nrm * (0.5 * width[k] + rng.uniform(2.5, 9.0 if width[k] > 15 else 6.0))
                )
                if (
                    not town.contains(Point(q))
                    and ras.at(grounds["bank"], q)[0] > 0.5
                    and rng.random() < np.clip(0.4 + 0.45 * ras.at(along, q)[0], 0.0, 0.95)
                ):
                    out.append(
                        [
                            *q,
                            rng.uniform(2.8, 5.0),
                            rng.uniform(8, 15),
                            3 if rng.random() < 0.55 else 0,
                            rng.integers(0, 3),
                        ]
                    )
                t += rng.uniform(5, 11)
    return np.asarray(out, float).reshape(-1, 6)


def _parkland(fields, rng, ras: Raster) -> np.ndarray:
    """Old trees standing out in some pastures (parkland), none in most."""
    park = ras.noise(rng, 300)
    out = []
    for crop, poly, _ in fields:
        if crop not in ("pasture", "meadow"):
            continue
        c = np.asarray(poly.representative_point().coords[0])
        lam = np.clip(0.15 + 1.6 * ras.at(park, c)[0], 0.0, 4.0) * poly.area / 4e4
        inner = poly.buffer(-10)
        if inner.is_empty:
            continue
        x0, y0, x1, y1 = inner.bounds
        placed: list[np.ndarray] = []
        for _ in range(int(rng.poisson(lam))):
            for _ in range(20):
                q = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
                if inner.contains(Point(q)) and all(np.hypot(*(q - o)) > 18 for o in placed):
                    placed.append(q)
                    out.append(
                        [*q, rng.uniform(5.0, 8.0), rng.uniform(13, 20), 0, rng.integers(0, 5)]
                    )
                    break
    return np.asarray(out, float).reshape(-1, 6)


def _orchard_trees(fields, rng) -> np.ndarray:
    out = []
    for crop, poly, _ in fields:
        if crop != "orchard":
            continue
        c, th, length, depth = _frame(poly)
        u, v = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
        inner = poly.buffer(-4)
        for a in np.arange(-length / 2, length / 2, 7.0):
            for b in np.arange(-depth / 2, depth / 2, 7.0):
                q = c + a * u + b * v + rng.normal(0, 0.4, 2)
                if inner.contains(Point(q)) and rng.random() > 0.06:
                    out.append(
                        [*q, rng.uniform(2.0, 2.9), rng.uniform(4.5, 6.5), 4, rng.integers(0, 3)]
                    )
    return np.asarray(out, float).reshape(-1, 6)


def _hedge_trees(hedges, rng, ras: Raster, wood_dist: np.ndarray) -> np.ndarray:
    """Each hedge has its own character: none, a few standards, trees every few tens of
    metres, a line of trees, or grown out into a tall, lumpy hedge. More of the thickly
    treed ones near the woods (hedgerows that link them)."""
    char = ras.noise(rng, 220)
    out = []
    for ln in hedges:
        if ln.length < 8:
            continue
        m = np.asarray(ln.interpolate(0.5, normalized=True).coords[0])
        c = (
            0.5
            + 0.3 * ras.at(char, m)[0]
            + 0.25 * rng.normal()
            + 0.3 * math.exp(-ras.at(wood_dist, m)[0] / 150)
        )
        if c < 0.2:
            continue  # a bare hedge
        if c < 1.05:
            gap = (rng.uniform(45, 95), rng.uniform(20, 38), rng.uniform(8, 13))[
                int(c >= 0.5) + int(c >= 0.82)
            ]
            big = c < 0.82
        else:
            gap, big = 0.0, False  # overgrown
        s = rng.uniform(2, gap if gap else 5)
        while s < ln.length:
            q = np.asarray(ln.interpolate(s).coords[0])
            if gap:
                r = rng.uniform(4.0, 6.5) if big else rng.uniform(3.0, 4.6)
                h = rng.uniform(12, 19) if big else rng.uniform(9, 14)
                out.append([*q, r, h, 0 if rng.random() < 0.85 else 3, rng.integers(0, 5)])
                if rng.random() < 0.15:  # a clump
                    for _ in range(rng.integers(1, 3)):
                        out.append(
                            [*(q + rng.normal(0, 3.5, 2)), r * 0.8, h * 0.85, 0, rng.integers(0, 5)]
                        )
                s += gap * rng.uniform(0.6, 1.4)
            else:
                out.append([*q, rng.uniform(1.8, 2.8), rng.uniform(4, 6.5), 2, rng.integers(0, 3)])
                if rng.random() < 0.12:
                    out.append(
                        [*q, rng.uniform(3.0, 4.5), rng.uniform(9, 14), 0, rng.integers(0, 5)]
                    )
                s += rng.uniform(3.2, 4.8)
    return np.asarray(out, float).reshape(-1, 6)


def _space_out(trees: np.ndarray, fixed: np.ndarray, gap: float = 2.8) -> np.ndarray:
    """Drop trees that would stand in another's trunk (closer than ``gap``)."""
    if len(trees) == 0:
        return trees
    keep = np.ones(len(trees), bool)
    if len(fixed):
        keep &= KDTree(fixed[:, :2]).query(trees[:, :2])[0] > gap
    for i, j in KDTree(trees[:, :2]).query_pairs(gap):
        if keep[i] and keep[j]:
            keep[max(i, j)] = False
    return trees[keep]


# Sunlight (from the west-south-west, high): a point z m up casts its shadow this far off.
SHADOW = np.array([0.331, 0.890])
BROADLEAF = ["#4f8a3c", "#5c9a46", "#4a7f36", "#67a14d", "#55893a"]  # 3, 4: lighter (birch, ash)
CONIFER = ["#2f5f34", "#355f2e", "#2b5531"]
SHRUB = ["#5f8a3e", "#6a9444", "#557d37"]
WILLOW = ["#8fae6a", "#9ab676", "#85a463"]
FRUIT = ["#6aa24c", "#78ad57", "#62994a"]
PALETTES = {0: BROADLEAF, 1: CONIFER, 2: SHRUB, 3: WILLOW, 4: FRUIT}  # by kind of tree


def crown_colour(kind: float, shade: float) -> str:
    pal = PALETTES.get(int(kind), BROADLEAF)
    return pal[int(shade) % len(pal)]


def tree_shapes(n, e, r, h, kind, shade, z0, sx, sy, z_up):
    """A tree drawn isometrically (page = sx (east − north), sy (north + east) + z_up height):
    (its shadow on the ground, [(polygon, colour), …] far to near). A broadleaf is a trunk
    and a lumpy crown lit from the west (a willow wider and paler, a fruit tree small); a
    conifer a trunk and a cone, sunlit on its west half; a shrub a low dome."""
    ring = np.linspace(0, 2 * np.pi, 13)
    cos, sin = np.cos(ring), np.sin(ring)
    x, g = sx * (e - n), sy * (n + e) + z_up * z0

    def page(pn, pe):  # plan → page, on the ground under the tree
        return np.column_stack([sx * (pe - pn), sy * (pn + pe) + z_up * z0])

    if kind == 2:  # a shrub: a low dome, no trunk to see
        base = np.array(_rgb_of(crown_colour(kind, shade)))
        y = g + z_up * 0.55 * h
        shapes = [
            (np.column_stack([x + r * cos, y + 0.8 * r * sin]), _hex_of(base * 0.72)),
            (
                np.column_stack([x - 0.2 * r + 0.7 * r * cos, y + 0.1 * r + 0.6 * r * sin]),
                _hex_of(base * 0.98),
            ),
        ]
        sc = np.array([n, e]) + SHADOW * 0.5 * h
        shadow = page(sc[0] + r * cos, sc[1] + r * sin)
        return shadow, shapes
    if kind != 1:  # a broadleaf: a willow wider and paler, a fruit tree small
        zc = h - 0.9 * r
        wide = 1.15 if kind == 3 else 1.0
        base = np.array(_rgb_of(crown_colour(kind, shade)))
        y = g + z_up * zc
        trunk = np.array(
            [[x - 0.35, g], [x + 0.35, g], [x + 0.35, y - 0.5 * r], [x - 0.35, y - 0.5 * r]]
        )
        shapes = [
            (trunk, "#5b4632"),
            (np.column_stack([x + wide * r * cos, y + 0.92 * r * sin]), _hex_of(base * 0.72)),
            (
                np.column_stack(
                    [x - 0.18 * r + 0.8 * wide * r * cos, y + 0.12 * r + 0.75 * r * sin]
                ),
                _hex_of(base * 0.95),
            ),
            (
                np.column_stack([x - 0.38 * r + 0.38 * r * cos, y + 0.38 * r + 0.34 * r * sin]),
                _hex_of(np.minimum(base * 1.2, 1)),
            ),
        ]
        sc = np.array([n, e]) + SHADOW * zc
        perp = np.array([-SHADOW[1], SHADOW[0]]) / np.hypot(*SHADOW)
        along = SHADOW / np.hypot(*SHADOW)
        shadow = page(
            sc[0] + r * (cos * along[0] * 1.3 + sin * perp[0]),
            sc[1] + r * (cos * along[1] * 1.3 + sin * perp[1]),
        )
    else:
        base = np.array(_rgb_of(CONIFER[int(shade) % len(CONIFER)]))
        yb, ya = g + z_up * 0.22 * h, g + z_up * h
        trunk = np.array([[x - 0.3, g], [x + 0.3, g], [x + 0.3, yb], [x - 0.3, yb]])
        shapes = [
            (trunk, "#4e3b2a"),
            (np.array([[x, ya], [x, yb - 0.3 * r], [x + r, yb]]), _hex_of(base * 0.7)),
            (
                np.array([[x, ya], [x - r, yb], [x, yb - 0.3 * r]]),
                _hex_of(np.minimum(base * 1.15, 1)),
            ),
        ]
        perp = np.array([-SHADOW[1], SHADOW[0]]) / np.hypot(*SHADOW)
        tip = np.array([n, e]) + SHADOW * h
        foot = np.array([n, e]) + SHADOW * 0.22 * h
        shadow = page(*np.array([foot + perp * r, tip, foot - perp * r]).T)
    return shadow, shapes


def _rgb_of(c: str) -> tuple[float, float, float]:
    return tuple(int(c[i : i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]


def _hex_of(rgb) -> str:
    return "#" + "".join(f"{round(float(np.clip(v, 0, 1)) * 255):02x}" for v in rgb)


def shadow_of(parts, sx, sy, z_up, z0) -> list[np.ndarray]:
    """The shadows on the ground of a building's (or turbine's) parts: each part's footprint
    and its top, cast off by the sun, and all between."""
    out = []
    for fp, lo, hi in parts:
        q = np.asarray(fp.exterior.coords)[:-1]
        pts = np.vstack([q + SHADOW * lo, q + SHADOW * hi])
        hull = shapely.MultiPoint(pts).convex_hull
        if hull.geom_type != "Polygon":
            continue
        c = np.asarray(hull.exterior.coords)
        out.append(
            np.column_stack([sx * (c[:, 1] - c[:, 0]), sy * (c[:, 0] + c[:, 1]) + z_up * z0])
        )
    return out


def turbine_shadow(t: dict, sx, sy, z_up, z0) -> list[np.ndarray]:
    """A turbine's shadow on the ground: its tower and its three blades."""
    out = shadow_of([(_ngon((t["n"], t["e"]), 2.3, 10), 0.0, t["hub"])], sx, sy, z_up, z0)
    for pts, col in turbine_faces(t, np.array([0.0, 0.0, 1.0])):
        if col == "#f4f5f6":  # a blade
            g = pts[:, :2] + SHADOW * pts[:, 2:3]
            out.append(
                np.column_stack([sx * (g[:, 1] - g[:, 0]), sy * (g[:, 0] + g[:, 1]) + z_up * z0])
            )
    return out


def _cut(poly: Polygon, centre: np.ndarray, sizes, rng, out: list[Polygon]) -> None:
    """Cut a holding into fields: across its longer side, a little off the middle and a
    little askew, again and again; small paddocks by the farm, growing to full-sized fields
    a few hundred metres out (``sizes``: paddock and field size ranges, m²)."""
    (plo, phi), (lo, hi) = sizes
    t = min(max((poly.centroid.distance(Point(centre)) - 60) / 300, 0.0), 1.0)
    if poly.area < rng.uniform(plo + t * (lo - plo), phi + t * (hi - phi)) * 1.4:
        if poly.area > 1500:
            out.append(poly)
        return
    c, th, length, depth = _frame(poly)
    u = np.array([math.cos(th), math.sin(th)])
    p = c + u * (rng.uniform(0.38, 0.62) - 0.5) * length
    a = th + math.pi / 2 + rng.normal(0, 0.12)
    d = np.array([math.cos(a), math.sin(a)]) * (length + depth)
    pieces = _polys(split(poly, LineString([p - d, p + d])))
    if len(pieces) < 2:
        out.append(poly)
        return
    for q in pieces:
        _cut(q, centre, sizes, rng, out)


def add_land(
    cfg: dict,
    sk: dict,
    city: City,
    groups,
    channels,
    h: np.ndarray,
    ns: np.ndarray,
    es: np.ndarray,
    locked: dict | None = None,
) -> dict:
    """Add the wind farm, woods, copses, fields (with their hedges) and trees to ``city``.
    With ``locked`` (from ``load_land``) the fields, woods, trees and the turbines already
    built are used as they are, and only turbines beyond those are added (felling trees for
    their pads and tracks); otherwise all is grown. Returns the wind farm as it now stands
    (for ``save_land``, which also writes the land as it now is)."""
    ground = Ground(h, ns, es)
    if locked is not None:
        return _add_locked_land(cfg, sk, city, ground, locked)
    roads = unary_union([ln for _, ln in city.roads])
    lanes = unary_union([ln for k, ln, _ in city.streets if k == "lane"])
    tracks = unary_union([ln for k, ln, _ in city.streets if k == "track"])
    settled = [g for g in groups if g.kind in ("village", "ribbon")]
    homes = unary_union([b.parts[0].footprint.buffer(40) for g in settled for b in g.buildings])
    homes = homes.buffer(30).buffer(-30)  # a village's houses and gardens, as one
    yards = unary_union(
        [b.parts[0].footprint.buffer(18) for g in groups if g.kind == "farm" for b in g.buildings]
    )
    water = city.water

    # The wind farm.
    wf = cfg.get("wind_farm")
    turbines, pads, service, extra = (
        _wind_farm(wf, sk, city, ground, _rng(cfg, wf["name"])) if wf else ([], [], [], [])
    )
    city.turbines = turbines
    city.places += [("pad", "", p) for p in pads]
    city.streets += [("service", ln, 2.5) for ln in service]
    city.buildings += extra
    farm_zone = unary_union(
        [Point(t["n"], t["e"]).buffer(60) for t in turbines] + [ln.buffer(8) for ln in service]
    )

    # Woods on the hills, and copses in the farm land.
    avoid = unary_union(
        [
            city.outline.buffer(40),
            homes,
            yards.buffer(25),
            roads.buffer(12),
            lanes.buffer(8),
            water.buffer(15),
            farm_zone,
        ]
    )
    rng = _rng(cfg, "woods")
    woods = _woods(cfg.get("woods", []), sk, ground, ns, es, rng, avoid)
    fc = cfg.get("fields", {})
    farmable = (ground.slope < fc.get("max_slope", 0.18)) & (
        ground.h < ground.plain + fc.get("max_rise_m", 140.0)
    )
    land = unary_union(_polygons(farmable, ns, es, 5000))
    free = land.difference(unary_union([avoid.buffer(80), unary_union(woods).buffer(100)]))
    woods += _copses(int(cfg.get("copses", 0)), ground, rng, avoid, free)
    city.woods = woods
    wooded = unary_union(woods)
    ras = Raster(ground.box)
    forest, city.glades, wood = _forest(cfg, sk, woods, wooded, rng, ras)

    # Fields: the open land shared among the farms and villages, each share cut into fields.
    rng = _rng(cfg, "fields")
    n0, n1, e0, e1 = ground.box
    town = unary_union(
        [
            *city.blocks,
            *[p for k, _, p in city.places if k != "pad"],
            *[ln.buffer(hw + 2) for k, ln, hw in city.streets if k in ("street", "main", "quay")],
            *[
                b.parts[0].footprint.buffer(22)
                for b in city.buildings
                if city.outline.contains(b.parts[0].footprint.centroid)
            ],
        ]
    )
    town = town.buffer(25).buffer(-25)  # the houses, their gardens and the streets between
    open_land = land.intersection(box(n0, e0, n1, e1)).difference(
        unary_union(
            [
                town,
                homes,
                yards,
                water.buffer(12),
                roads.buffer(7),
                lanes.buffer(5),
                tracks.buffer(2.5),
                wooded,
                unary_union([ln.buffer(4) for ln in service]) if service else Polygon(),
                unary_union(pads).buffer(3) if pads else Polygon(),
            ]
        )
    )
    owners = [
        (g.name, np.asarray(g.centre), g.kind) for g in groups if g.kind in ("farm", "village")
    ]
    cells = voronoi_diagram(
        shapely.MultiPoint([o[1] for o in owners]),
        envelope=box(n0 - 5000, e0 - 5000, n1 + 5000, e1 + 5000),
    )
    lo, hi = (a * 1e4 for a in fc.get("size_ha", [2.5, 10.0]))
    plo, phi = (a * 1e4 for a in fc.get("paddock_ha", [0.6, 2.0]))
    fields: list[tuple[str, Polygon, str]] = []
    for cell in cells.geoms:
        k = next((i for i, o in enumerate(owners) if cell.contains(Point(o[1]))), None)
        if k is None:
            continue
        name, centre, kind = owners[k]
        pieces: list[Polygon] = []
        for part in _polys(cell.intersection(open_land)):
            _cut(part, centre, ((plo, phi), (lo, hi)), rng, pieces)
        for p in pieces:
            c = np.asarray(p.representative_point().coords[0])
            steep = float(ground.at(ground.slope, c)[0]) > 0.08
            by_farm = float(np.hypot(*(c - centre))) < (200 if kind == "farm" else 350)
            if steep or (by_farm and rng.random() < 0.7):
                crop = "pasture" if rng.random() < 0.7 else "meadow"
            else:
                names = list(CROPS)
                crop = names[
                    int(rng.choice(len(names), p=np.array([CROPS[n][1] for n in names]) / 1.0))
                ]
            fields.append((crop, p.simplify(1.0), name))
    # A few small fields by the villages and farms are orchards.
    village_c = [np.asarray(g.centre) for g in groups if g.kind == "village"]
    farm_c = [np.asarray(g.centre) for g in groups if g.kind == "farm"]
    cand = [
        i
        for i, (crop, p, _) in enumerate(fields)
        if 3000 < p.area < 2.2e4
        and (
            any(p.distance(Point(c)) < 300 for c in village_c)
            or any(p.distance(Point(c)) < 120 for c in farm_c)
        )
    ]
    for i in rng.permutation(cand)[: int(cfg.get("orchards", 0))]:
        fields[i] = ("orchard", fields[i][1], fields[i][2])
    city.fields = fields
    hedges = _lines(unary_union([p.exterior for _, p, _ in fields]))
    city.hedges = hedges

    # The trees outside the woods. Where they may grow: not on the roads, in the yards or the
    # town; scrub and a wood's fringe on rough ground and (less) into grazing, never into
    # crops; along the banks out of town.
    rng = _rng(cfg, "trees")
    blocked = unary_union(
        [
            town,
            homes,
            yards,
            water.buffer(2.5),
            roads.buffer(6),
            lanes.buffer(4),
            tracks.buffer(2.5),
            unary_union([ln.buffer(4) for ln in service]) if service else Polygon(),
            unary_union(pads).buffer(4) if pads else Polygon(),
            farm_zone.buffer(20),
            unary_union([b.parts[0].footprint.buffer(5) for b in city.buildings]),
        ]
    )
    m_blocked = ras.mask(blocked)
    m_arable = ras.mask(unary_union([p for c, p, _ in fields if c in ARABLE or c == "orchard"]))
    m_grass = ras.mask(unary_union([p for c, p, _ in fields if c in ("pasture", "meadow")]))
    m_field = m_arable | m_grass
    grounds = {
        "welcome": np.where(m_blocked | m_arable, 0.0, np.where(m_grass, 0.45, 1.0)),
        "rough": (~m_blocked & ~m_field & ~wood).astype(float),
        "bank": (~m_blocked & ~m_arable).astype(float),
    }
    wood_dist = distance_transform_edt(~wood) * ras.cell
    outside = [
        _fringe(cfg, rng, ras, wood, grounds),
        _scrub(cfg, rng, ras, grounds),
        _riverside(channels, rng, ras, grounds, city.outline.buffer(40)),
        _parkland(fields, rng, ras),
        _orchard_trees(fields, rng),
        _hedge_trees(hedges, rng, ras, wood_dist),
    ]
    city.forest = np.vstack([forest, _space_out(np.vstack(outside), forest)])
    return {"wind": {"turbines": turbines, "pads": pads, "service": service, "buildings": extra}}


def _add_locked_land(cfg: dict, sk: dict, city: City, ground: Ground, locked: dict) -> dict:
    wind = locked["wind"]
    city.woods, city.glades = list(locked["woods"]), list(locked["glades"])
    city.fields, city.hedges = list(locked["fields"]), list(locked["hedges"])
    forest = locked["forest"]
    turbines, pads, service = list(wind["turbines"]), list(wind["pads"]), list(wind["service"])
    wf = cfg.get("wind_farm")
    if wf and int(wf.get("turbines", 0)) > len(turbines):
        more, more_pads, more_tracks = _more_turbines(
            wf, sk, city, ground, _rng(cfg, wf["name"] + " more"), turbines, service
        )
        # Felled for them: a clearing round each tower and its pad, a ride along each track.
        clearing = unary_union(
            [
                *[Point(t["n"], t["e"]).buffer(22) for t in more],
                *[p.buffer(6) for p in more_pads],
                *[ln.buffer(4.5) for ln in more_tracks],
            ]
        )
        city.woods = [g for w in city.woods for g in _polys(w.difference(clearing)) if g.area > 500]
        forest = forest[~shapely.contains_xy(clearing, forest[:, 0], forest[:, 1])]
        turbines, pads, service = turbines + more, pads + more_pads, service + more_tracks
    city.forest = forest
    city.turbines = turbines
    city.places += [("pad", "", p) for p in pads]
    city.streets += [("service", ln, 2.5) for ln in service]
    city.buildings += list(wind["buildings"])
    return {
        "wind": {
            "turbines": turbines,
            "pads": pads,
            "service": service,
            "buildings": list(wind["buildings"]),
        }
    }


def save_land(city: City, state: dict, path) -> None:
    """Write the land as it is now (fields, hedges, woods, glades, every tree) and the wind
    farm built so far, to ``path`` (gzipped JSON)."""
    import gzip
    import json

    from sketch_city import _wkt, building_record

    w = state["wind"]
    data = {
        "fields": [[c, _wkt(p), o] for c, p, o in city.fields],
        "hedges": [_wkt(ln) for ln in city.hedges],
        "woods": [_wkt(p) for p in city.woods],
        "glades": [_wkt(p) for p in city.glades],
        "forest": np.round(city.forest, 2).tolist(),
        "wind": {
            "turbines": w["turbines"],
            "pads": [_wkt(p) for p in w["pads"]],
            "service": [_wkt(ln) for ln in w["service"]],
            "buildings": [building_record(b) for b in w["buildings"]],
        },
    }
    with gzip.open(path, "wt") as f:
        json.dump(data, f, separators=(",", ":"))


def load_land(path) -> dict:
    import gzip
    import json

    from sketch_city import building_from

    with gzip.open(path, "rt") as f:
        d = json.load(f)
    g = shapely.from_wkt
    w = d["wind"]
    return {
        "fields": [(c, g(p), o) for c, p, o in d["fields"]],
        "hedges": [g(x) for x in d["hedges"]],
        "woods": [g(x) for x in d["woods"]],
        "glades": [g(x) for x in d["glades"]],
        "forest": np.asarray(d["forest"], float).reshape(-1, 6),
        "wind": {
            "turbines": w["turbines"],
            "pads": [g(x) for x in w["pads"]],
            "service": [g(x) for x in w["service"]],
            "buildings": [building_from(b) for b in w["buildings"]],
        },
    }
