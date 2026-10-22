# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportOperatorIssue=false, reportReturnType=false, reportOptionalMemberAccess=false
# pyright: reportGeneralTypeIssues=false
# (Drawing tools over shapely, whose set operations are typed as returning a BaseGeometry; the
# code relies on them being polygons or lines at run time, which the checker cannot narrow.)
"""A city on the sketch (configs/sketches/<name>.yaml ``city``), grown rather than planned.

1. Old roads met where they crossed the river at the island. Each takes the easiest way over
   the land to its end on the map's edge (gentle grades, few and short river crossings).
2. Streets grow out from the crossing, nearest first. Each follows a direction field: along
   the river near it, along the old roads near them, round the old centre, and a gentle
   general grain elsewhere, wavering most in the old town. A street runs both ways until it
   leaves the town, reaches the quay, or comes within half its spacing of a parallel street;
   cross streets start along it. Spacing is tight in the old town and wider further out
   (evenly spaced streamlines, as in Jobard and Lefer; street modelling as in Chen et al.).
   A few streets that reach the river carry on over it as new bridges.
3. Blocks are the land between the streets. Places claim some: the cathedral, castle, town
   hall, museum, a column, a stadium, parks, garden squares and paved squares; streets
   inside a park become its paths.
4. Buildings fill every other block by its character: tall narrow houses with pitched roofs
   in the old town, deeper perimeter blocks round courtyards further out, terraces, modern
   slabs, towers on podiums by the river on the south-east side, detached houses at the edge.
5. Trees: in parks and squares, along the main roads and quays, in courtyards and gardens.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, polylabel, unary_union

ROAD_HALF_M = 7.0  # half the width of an old main road in town
QUAY_HALF_M = 5.0
GRID_M = 10.0  # the direction field's grid

# Walls and roofs (or flat tops) by kind of building.
TONES = {
    "old": ("#efe4cf", "#b9684b"),
    "terrace": ("#e8d2b9", "#8e8b88"),
    "block": ("#eee7da", "#b3aea6"),
    "modern": ("#f4f4f1", "#dcdddd"),
    "tower": ("#ccd8e3", "#b7c5d3"),
    "house": ("#f2eadc", "#ad6650"),
    "landmark": ("#ead9b3", "#8f9daa"),
    "castle": ("#d2c9b7", "#aaa18f"),
    "barn": ("#bfa184", "#7a7570"),
    "industrial": ("#e6e4df", "#c4c6c9"),
    "reactor": ("#ebe9e4", "#d8d6d1"),
    "steel": ("#a9adb3", "#9a9ea4"),
    "antenna": ("#a7abb1", "#b3b7bc"),
    "windsock": ("#f08a24", "#f08a24"),
    "hangar": ("#c9cdd1", "#8f979e"),
    "beacon": ("#d23b30", "#e0463a"),
}
# Facades and roofs each kind is built in: stuccoed old houses under clay tiles, stone, brick
# and render under slate and zinc, brick terraces, white modern, glass towers.
PALETTE = {
    "old": (
        ["#f1e6cf", "#f2dcc0", "#ebd5c6", "#f3ecd9", "#e8dcc4", "#efe0b8"],
        ["#b9684b", "#a95a42", "#c27554", "#9e5643", "#b5735a"],
    ),
    "block": (
        ["#efe8da", "#e6ded0", "#dcbfa5", "#f4f2ec", "#d9d8d3", "#e9dcc2"],
        ["#a3a7ad", "#b3aea6", "#9a9894", "#bcb7ae", "#8e9298"],
    ),
    "terrace": (["#d6ad92", "#ddbb9f", "#e8d2b9", "#cf9f86"], ["#7f8288", "#8e8b88", "#75787d"]),
    "house": (["#f2eadc", "#f4efe6", "#eadfcb"], ["#ad6650", "#9a5a48", "#8e9095", "#b77a5c"]),
    "modern": (["#f4f4f1", "#e9ecee", "#dfe3e6"], ["#dcdddd", "#cfd3d6", "#e6e6e3"]),
    "tower": (["#ccd8e3", "#bfcfdc", "#d5dde4", "#b7c6d2"], ["#b7c5d3", "#aebdca"]),
}
PLACE_COLOURS = {
    "park": "#a9d08b",
    "garden": "#a3cd85",
    "green": "#b7da9c",
    "square": "#e6dfd1",
    "pitch": "#86c06a",
    "yard": "#cbd9b0",
    "pad": "#d3c6a4",
    "plant": "#b8cf98",
    "apron": "#d6d3cc",
    "switchyard": "#cdc6b4",
    "car_park": "#b7b6b1",
    "basin": "#86a6bd",
    "runway": "#d5eab2",
    "marker": "#f7f7f2",
}
GROUND = "#dcd7cd"  # yards, pavements
STREET = "#9b9ea3"
MAIN_ROAD = "#85888d"
TREE = "#5c9a46"
LIGHT = np.array([-0.24, -0.645, 0.725])  # sunlight from the west-south-west, high


@dataclass
class Part:
    """A prism from ``z0`` to ``z1`` m above the ground; its roof is flat, a gable (ridge
    along the longer sides, four-sided footprints) or a hip (a pyramid), rising ``rise`` m."""

    footprint: Polygon  # (north, east)
    z0: float
    z1: float
    roof: str = "flat"
    rise: float = 0.0


@dataclass
class Building:
    parts: list[Part]
    kind: str  # a key of TONES
    name: str = ""
    wall: str = ""  # its own colours (else its kind's)
    roof: str = ""

    @property
    def height(self) -> float:
        return max(p.z1 + p.rise for p in self.parts)

    @property
    def centre(self) -> np.ndarray:
        return np.asarray(self.parts[0].footprint.centroid.coords[0])


@dataclass
class City:
    name: str
    water: BaseGeometry  # (north, east)
    roads: list[tuple[str, LineString]]  # the old main roads, whole
    streets: list[tuple[str, LineString, float]]  # (kind, centreline, half width) in town
    bridges: list[tuple[LineString, float]]  # (line, half width)
    blocks: list[Polygon]  # built-up land between the streets
    places: list[tuple[str, str, Polygon]]  # (kind, name, area): parks, squares, greens
    paths: list[LineString]  # in parks and greens
    buildings: list[Building]
    trees: np.ndarray  # (k, 4): north, east, crown radius, crown height
    labels: list[Building] = field(default_factory=list)  # landmarks and the tallest tower
    outline: BaseGeometry = field(default_factory=Polygon)  # how far the town reaches
    settlements: list[tuple[str, np.ndarray]] = field(
        default_factory=list
    )  # villages: name, centre
    # The land round it (tools/sketch_land.py): woods and their crowns, fields, hedges, turbines.
    woods: list[Polygon] = field(default_factory=list)
    crowns: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # (unused: see forest)
    forest: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 6))
    )  # north, east, r, height, kind, shade
    glades: list[Polygon] = field(default_factory=list)  # clearings in the woods
    # The power station and its lines (tools/sketch_power.py).
    cooling_towers: list[dict] = field(default_factory=list)
    pylons: list[dict] = field(default_factory=list)
    wires: list[dict] = field(default_factory=list)
    # Telecom masts and a solar farm (tools/sketch_infra.py).
    masts: list[dict] = field(default_factory=list)
    solar: list = field(
        default_factory=list
    )  # panel rows: (north of the front edge, east from, east to)
    ops: dict = field(
        default_factory=dict
    )  # the airstrip, landing fields, no-fly zones (tools/sketch_ops.py)
    fields: list[tuple[str, Polygon, str]] = field(default_factory=list)  # crop, area, owner
    hedges: list[LineString] = field(default_factory=list)
    turbines: list[dict] = field(default_factory=list)


# --- geometry ---------------------------------------------------------------------------


def _band_polygon(line: np.ndarray, width: np.ndarray):
    d = np.gradient(line, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    nrm = np.stack([-d[:, 1], d[:, 0]], 1) * (0.5 * width)[:, None]
    return Polygon(np.vstack([line + nrm, (line - nrm)[::-1]])).buffer(0)


def _polys(g) -> list[Polygon]:
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    return [p for part in getattr(g, "geoms", []) for p in _polys(part)]


def _lines(g) -> list[LineString]:
    if g.is_empty:
        return []
    if isinstance(g, LineString):
        return [g]
    return [ln for part in getattr(g, "geoms", []) for ln in _lines(part)]


def _rect(c, theta: float, length: float, depth: float) -> Polygon:
    """A rectangle: centre (north, east), long axis at ``theta`` (radians from north)."""
    u = np.array([math.cos(theta), math.sin(theta)]) * length / 2
    v = np.array([-math.sin(theta), math.cos(theta)]) * depth / 2
    return Polygon([c + u + v, c - u + v, c - u - v, c + u - v])


def _ngon(c, r: float, k: int = 8) -> Polygon:
    a = np.linspace(0, 2 * math.pi, k, endpoint=False) + math.pi / k
    return Polygon(np.asarray(c) + r * np.stack([np.cos(a), np.sin(a)], 1))


def _frame(p) -> tuple[np.ndarray, float, float, float]:
    """(centre, long-axis angle, length, depth) of a shape's minimum rotated rectangle."""
    q = np.asarray(p.minimum_rotated_rectangle.exterior.coords)[:4]
    a, b = q[1] - q[0], q[2] - q[1]
    la, lb = float(np.hypot(*a)), float(np.hypot(*b))
    if la >= lb:
        return q.mean(0), math.atan2(a[1], a[0]), la, lb
    return q.mean(0), math.atan2(b[1], b[0]), lb, la


def _samples(lines, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``step`` m along the lines, and the lines' directions there."""
    pts, tans = [], []
    for ln in lines:
        s = np.linspace(0, ln.length, max(int(ln.length / step), 2))
        p = shapely.get_coordinates(shapely.line_interpolate_point(ln, s))
        pts.append(p)
        tans.append(np.gradient(p, axis=0))
    return np.vstack(pts), np.vstack(tans)


def _local(c, th: float, s: float):
    """Local frame → (north, east): u along ``th``, v to its left, scaled by ``s``."""
    u = np.array([math.cos(th), math.sin(th)])
    v = np.array([-u[1], u[0]])

    def rect(u0, u1, v0, v1) -> Polygon:
        return Polygon(
            [c + s * (a * u + b * v) for a, b in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
        )

    def at(a, b) -> np.ndarray:
        return c + s * (a * u + b * v)

    return rect, at


# --- the old roads ----------------------------------------------------------------------


def _roads(cfg, rng, h, ns, es, channels, core) -> list[tuple[str, LineString]]:
    """Least-cost routes from the crossing to each road's end on the map's edge."""
    cell = float(ns[1] - ns[0])
    rows, cols = h.shape
    nn, ee = np.meshgrid(ns, es, indexing="ij")
    pts = np.stack([nn.ravel(), ee.ravel()], 1)
    wet = np.zeros(rows * cols, bool)
    for c in channels:
        d, k = KDTree(c.line).query(pts)
        wet |= d < 0.5 * c.width[k] + 4
    wet = wet.reshape(rows, cols)
    wander = gaussian_filter(rng.standard_normal(h.shape), 15)
    wander = np.clip(1 + 0.25 * wander / wander.std(), 0.6, 1.6)  # no road is quite straight
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
        grade = np.abs(h[i1, j1] - h[i0, j0]) / length
        w = length * 0.5 * (wander[i0, j0] + wander[i1, j1]) * (1 + (grade / 0.05) ** 2)
        w = np.where(wet[i0, j0] | wet[i1, j1], w * 40, w)  # bridges are dear
        src.append(idx[i0, j0].ravel())
        dst.append(idx[i1, j1].ravel())
        cost.append(w.ravel())
    graph = coo_matrix(
        (np.concatenate(cost), (np.concatenate(src), np.concatenate(dst))), shape=(rows * cols,) * 2
    ).tocsr()
    start = int(np.argmin(np.abs(ns - core[0]))) * cols + int(np.argmin(np.abs(es - core[1])))
    _, pred = dijkstra(graph, directed=False, indices=start, return_predecessors=True)
    out = []
    for r in cfg.get("roads", []):
        n, e = r["to_ne_m"]
        k = int(np.argmin(np.abs(ns - n))) * cols + int(np.argmin(np.abs(es - e)))
        path = [k]
        while path[-1] != start and pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        p = pts[np.array(path[::-1])]
        p = np.vstack([np.asarray(core)[None], p[1:]])
        sm = np.stack([gaussian_filter1d(p[:, i], 4.0, mode="nearest") for i in (0, 1)], 1)
        sm[0], sm[-1] = p[0], p[-1]
        out.append((r["name"], LineString(sm).simplify(1.0)))
    return out


# --- the streets ------------------------------------------------------------------------


class Field:
    """Which way the streets run: along the river near it, along the old roads near them,
    radially round the old centre and with a general grain elsewhere (blended as tensors,
    so a direction and its opposite are the same), wavering most in the old town. Family 0
    follows it; family 1 crosses it."""

    def __init__(self, rng, channels, roads, core, window, grain_deg: float) -> None:
        self.w0 = window[0]
        self.n = round((window[1] - window[0]) / GRID_M) + 1
        g = self.w0 + GRID_M * np.arange(self.n)
        nn, ee = np.meshgrid(g, g, indexing="ij")
        self.nn, self.ee = nn, ee
        pts = np.stack([nn.ravel(), ee.ravel()], 1)
        cp = np.vstack([c.line for c in channels])
        ct = np.vstack([np.gradient(c.line, axis=0) for c in channels])
        cw = np.concatenate([c.width for c in channels])
        d, k = KDTree(cp).query(pts)
        edge = d - 0.5 * cw[k]
        self.edge = edge.reshape(self.n, self.n)  # metres from the water's edge
        rp, rt = _samples([ln for _, ln in roads], 5.0)
        dr, kr = KDTree(rp).query(pts)
        dc = np.hypot(*(pts - core).T)
        terms = [
            (1.6 * np.exp(-((np.maximum(edge, 0) / 180) ** 2)), np.arctan2(ct[k, 1], ct[k, 0])),
            (1.3 * np.exp(-((dr / 70) ** 2)), np.arctan2(rt[kr, 1], rt[kr, 0])),
            (
                0.6 * np.exp(-((dc / 450) ** 2)),
                np.arctan2(pts[:, 1] - core[1], pts[:, 0] - core[0]),
            ),
            (np.full_like(dc, 0.35), np.full_like(dc, math.radians(grain_deg))),
        ]
        c = sum(w * np.cos(2 * t) for w, t in terms)
        s = sum(w * np.sin(2 * t) for w, t in terms)
        wav = gaussian_filter(rng.standard_normal((self.n, self.n)), 5).ravel()
        th = 0.5 * np.arctan2(s, c) + (0.06 + 0.3 * np.exp(-((dc / 350) ** 2))) * wav / wav.std()
        self.c = np.cos(2 * th).reshape(self.n, self.n)
        self.s = np.sin(2 * th).reshape(self.n, self.n)

    def _cell(self, p) -> tuple[int, int, float, float]:
        fi, fj = (p[0] - self.w0) / GRID_M, (p[1] - self.w0) / GRID_M
        i, j = min(max(int(fi), 0), self.n - 2), min(max(int(fj), 0), self.n - 2)
        return i, j, min(max(fi - i, 0.0), 1.0), min(max(fj - j, 0.0), 1.0)

    def direction(self, p, fam: int) -> tuple[float, float]:
        i, j, a, b = self._cell(p)
        c, s = self.c, self.s
        cc = (c[i, j] * (1 - a) + c[i + 1, j] * a) * (1 - b) + (
            c[i, j + 1] * (1 - a) + c[i + 1, j + 1] * a
        ) * b
        ss = (s[i, j] * (1 - a) + s[i + 1, j] * a) * (1 - b) + (
            s[i, j + 1] * (1 - a) + s[i + 1, j + 1] * a
        ) * b
        t = 0.5 * math.atan2(ss, cc) + fam * math.pi / 2
        return math.cos(t), math.sin(t)

    def value(self, grid: np.ndarray, p) -> float:
        i, j, a, b = self._cell(p)
        return float(grid[i + int(a > 0.5), j + int(b > 0.5)])


class _Hash:
    """Points in square cells, to ask quickly whether any lies within a distance."""

    def __init__(self, cell: float = 16.0) -> None:
        self.cell = cell
        self.d: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)

    def add(self, p) -> None:
        self.d[(int(p[0] // self.cell), int(p[1] // self.cell))].append((p[0], p[1]))

    def near(self, p, r: float) -> bool:
        i, j, k, r2 = int(p[0] // self.cell), int(p[1] // self.cell), int(r // self.cell) + 1, r * r
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                for q in self.d.get((i + di, j + dj), ()):
                    if (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 < r2:
                        return True
        return False


def _streets(fld: Field, inside: np.ndarray, core, spacing, wet_m: float, guides, hashes):
    """Trace the streets (list of (family, points, (why the start stopped, why the end
    stopped)))."""
    step = 5.0

    def blocked(q) -> str:
        if not fld.value(inside, q):
            return "out"
        if fld.value(fld.edge, q) < wet_m:
            return "wet"
        return ""

    def trace(seed, fam):
        halves = []
        for sign in (1, -1):
            dx, dy = fld.direction(seed, fam)
            p, prev, pts, why, turned = seed, (sign * dx, sign * dy), [], "end", 0.0
            for _ in range(500):
                a = fld.direction(p, fam)
                a = a if a[0] * prev[0] + a[1] * prev[1] >= 0 else (-a[0], -a[1])
                m = (p[0] + a[0] * step / 2, p[1] + a[1] * step / 2)
                b = fld.direction(m, fam)
                b = b if b[0] * a[0] + b[1] * a[1] >= 0 else (-b[0], -b[1])
                q = (p[0] + b[0] * step, p[1] + b[1] * step)
                why = blocked(q)
                if why:
                    break
                if hashes[fam].near(q, 0.5 * spacing(q, fam)):
                    why = "close"
                    break
                turned += abs(prev[0] * b[1] - prev[1] * b[0])
                if turned > 2.2:  # no street curls back on itself
                    why = "turn"
                    break
                pts.append(q)
                p, prev = q, b
            halves.append((pts, why))
        return halves[1][0][::-1] + [seed] + halves[0][0], (halves[1][1], halves[0][1])

    heap: list = []
    counter = itertools.count()

    def push(p, fam: int) -> None:
        heapq.heappush(heap, (math.hypot(p[0] - core[0], p[1] - core[1]), next(counter), p, fam))

    push((float(core[0]), float(core[1])), 0)
    push((float(core[0]), float(core[1])), 1)
    for p, fam in guides:  # cross streets start along the old roads and quays
        push(p, 1 - fam)
    out, seen = [], set()
    while heap and len(out) < 4000:
        _, _, p, fam = heapq.heappop(heap)
        key = (fam, round(p[0] / 6), round(p[1] / 6))
        if key in seen:
            continue
        seen.add(key)
        if blocked(p) or hashes[fam].near(p, 0.9 * spacing(p, fam)):
            continue
        pts, why = trace(p, fam)
        if len(pts) < 7:
            continue
        for q in pts:
            hashes[fam].add(q)
        out.append((fam, pts, why))
        for k in range(0, len(pts), 4):
            q = pts[k]
            t = fld.direction(q, fam)
            s = spacing(q, fam)
            push((q[0] - t[1] * s, q[1] + t[0] * s), fam)
            push((q[0] + t[1] * s, q[1] - t[0] * s), fam)
            push(q, 1 - fam)
    return out


# --- buildings --------------------------------------------------------------------------


def _lots(
    block: Polygon, depth: float, frontage: tuple[float, float], rng
) -> tuple[list[Polygon], object]:
    """Cut a block's edge into lots ``depth`` deep and a random frontage wide; returns the lots
    and the courtyard left inside (perimeter blocks, as in most European cities)."""
    inner = block.buffer(-depth, join_style="mitre")
    ring = block.difference(inner) if inner.area > 40 else block
    ext = block.exterior
    length = ext.length
    cuts = []
    s = rng.uniform(0, frontage[1])
    while s < length - 2:
        p = np.asarray(ext.interpolate(s).coords[0])
        a = np.asarray(ext.interpolate(max(s - 1.5, 0)).coords[0])
        b = np.asarray(ext.interpolate(min(s + 1.5, length)).coords[0])
        t = (b - a) / max(float(np.hypot(*(b - a))), 1e-6)
        nrm = np.array([-t[1], t[0]])
        if not block.contains(Point(p + 1.5 * nrm)):
            nrm = -nrm
        cuts.append(LineString([p - nrm, p + nrm * (depth + 2)]))
        s += rng.uniform(*frontage)
    noded = unary_union([*_lines(ring.boundary), *cuts])
    lots = [
        q
        for q in polygonize(_lines(noded))
        if q.area > 18 and ring.contains(q.representative_point())
    ]
    return [q for q in lots if q.area / q.length**2 > 0.025], (
        inner if inner.area > 40 else Polygon()
    )


def _roofed(q: Polygon, z: float, roof: str, rise: float) -> Part:
    n = len(q.exterior.coords) - 1
    if roof == "gable" and n != 4:
        roof = "hip"
    return Part(q, 0.0, z, roof, rise if roof != "flat" else 0.0)


def _towers(block: Polygon, top: float, rng) -> list[Building]:
    """Towers on podiums: the block in one to three pieces, stepped back as they rise."""
    inner = block.buffer(-4, join_style="mitre")
    if inner.is_empty:
        return []
    c, th, length, _ = _frame(inner)
    k = max(1, min(3, int(length // 55)))
    u = np.array([math.cos(th), math.sin(th)])
    out = []
    for i in range(k):
        mid = c + u * length * ((i + 0.5) / k - 0.5)
        piece = inner.intersection(_rect(mid, th, length / k - 6, 400))
        for pc in _polys(piece):
            if pc.area < 350:
                continue
            pc_c, pth, pl, pd = _frame(pc)
            h = top * rng.uniform(0.75, 1.0)
            podium = rng.uniform(10, 22)
            tw = _rect(pc_c, pth, min(pl * 0.75, 48), min(pd * 0.75, 36)).intersection(pc)
            parts = [Part(pc, 0, podium)]
            if h < 45 or tw.area < 150:
                parts[0].z1 = max(podium, h * 0.6)
            elif h > 105:
                upper = _rect(pc_c, pth, min(pl * 0.55, 36), min(pd * 0.55, 27)).intersection(pc)
                parts += [Part(tw, podium, 0.68 * h), Part(upper, 0.68 * h, h)]
            else:
                parts.append(Part(tw, podium, h))
            out.append(Building(parts, "tower"))
    return out


def _slabs(block: Polygon, rng, lower: float = 1.0) -> list[Building]:
    """Modern slabs, parallel to the block's long side, in open ground."""
    inner = block.buffer(-7, join_style="mitre")
    if inner.is_empty:
        return []
    c, th, length, depth = _frame(inner)
    k = 1 if depth < 45 else 2
    v = np.array([-math.sin(th), math.cos(th)])
    out = []
    for i in range(k):
        mid = c + v * depth * 0.7 * ((i + 0.5) / k - 0.5)
        slab = _rect(mid, th, length * rng.uniform(0.55, 0.9), rng.uniform(12, 16)).intersection(
            inner
        )
        for pc in _polys(slab):
            if pc.area > 120:
                out.append(Building([Part(pc, 0, rng.uniform(18, 48) * lower)], "modern"))
    return out


def _houses(
    block: Polygon, frontage_ok, rng, apart: float = 1.0
) -> tuple[list[Building], list[np.ndarray]]:
    """Detached houses along the block's street edges, each with a tree or two behind."""
    ext = block.exterior
    out, trees = [], []
    s = rng.uniform(0, 10)
    taken = []
    while s < ext.length:
        p = np.asarray(ext.interpolate(s).coords[0])
        b = np.asarray(ext.interpolate(min(s + 2, ext.length)).coords[0])
        t = (b - p) / max(float(np.hypot(*(b - p))), 1e-6)
        nrm = np.array([-t[1], t[0]])
        if not block.contains(Point(p + 2 * nrm)):
            nrm = -nrm
        w, d = rng.uniform(8, 12), rng.uniform(8, 11)
        c = p + nrm * (6 + d / 2)
        q = _rect(c, math.atan2(t[1], t[0]), w, d)
        if (
            frontage_ok(p)
            and block.buffer(-1.5).contains(q)
            and not any(q.distance(o) < 4 for o in taken)
        ):
            taken.append(q)
            out.append(Building([_roofed(q, rng.uniform(6.5, 8.5), "hip", 2.5)], "house"))
            trees.append(
                np.array([*(c + nrm * (d / 2 + rng.uniform(5, 9))), rng.uniform(2.5, 4), 6.0])
            )
        s += rng.uniform(16, 24) * apart
    return out, trees


# --- landmarks --------------------------------------------------------------------------


def _cathedral(c, th, s, height):
    rect, _at = _local(c, th, s)
    return [
        Part(rect(-45, 35, -11, 11), 0, 24, "gable", 11),  # nave and choir
        Part(rect(0, 22, -27, 27), 0, 22, "gable", 10),  # transepts
        Part(rect(-52, -40, -15, -5), 0, 38, "hip", 6),  # west towers
        Part(rect(-52, -40, 5, 15), 0, 38, "hip", 6),
        Part(rect(4, 18, -7, 7), 0, 44 * s, "flat"),  # the crossing tower …
        Part(rect(7, 15, -4, 4), 44 * s, 47 * s, "hip", max(height - 47 * s, 5)),  # … and spire
    ]


def _castle(c, th, s, height):
    rect, at = _local(c, th, s)
    parts = [
        Part(rect(-32, 32, -30, -26), 0, 12),
        Part(rect(-32, 32, 26, 30), 0, 12),
        Part(rect(-32, -28, -26, 26), 0, 12),
        Part(rect(28, 32, -26, 26), 0, 12),
    ]
    for a, b in ((-30, -28), (-30, 28), (30, -28), (30, 28)):
        parts.append(Part(_ngon(at(a, b), 6.5 * s), 0, 17, "hip", 6))
    parts.append(Part(rect(-32, -24, -6, 6), 0, 16))  # gatehouse
    parts.append(Part(rect(-2, 18, -10, 10), 0, height))  # the keep
    return parts


def _town_hall(c, th, s, height):
    rect, _at = _local(c, th, s)
    return [
        Part(rect(-24, 24, -12, 12), 0, 20, "gable", 7),
        Part(rect(-4, 4, -16, -8), 0, height - 10, "flat"),
        Part(rect(-3, 3, -15, -9), height - 10, height - 9, "hip", 9),
    ]


def _dome(c, th, s, height):
    rect, _at = _local(c, th, s)
    parts = [Part(rect(-30, 30, -20, 20), 0, 20), Part(_ngon(c, 13 * s), 20, 30)]
    for r, z0, z1 in ((12, 30, 35), (10, 35, 39), (7, 39, 42), (4, 42, 44)):
        parts.append(Part(_ngon(c, r * s), z0, z1))
    parts.append(Part(_ngon(c, 2.2 * s), 44, 48, "hip", max(height - 48, 2)))
    return parts


def _column(c, th, s, height):
    rect, _at = _local(c, th, 1.0)
    return [
        Part(rect(-4.5, 4.5, -4.5, 4.5), 0, 4),
        Part(rect(-1.6, 1.6, -1.6, 1.6), 4, height - 5),
        Part(rect(-2.4, 2.4, -2.4, 2.4), height - 5, height - 3, "hip", 3),
    ]


def _stadium(c, th, s, height):
    _, at = _local(c, th, s)
    k = 32
    a = np.linspace(0, 2 * math.pi, k + 1)
    out_ = np.stack([95 * np.cos(a), 72 * np.sin(a)], 1)
    in_ = np.stack([60 * np.cos(a), 40 * np.sin(a)], 1)
    return [
        Part(Polygon([at(*out_[i]), at(*out_[i + 1]), at(*in_[i + 1]), at(*in_[i])]), 0, height)
        for i in range(k)
    ]


LANDMARKS = {
    "cathedral": (_cathedral, 75, 60),  # (parts, default height, radius of its site)
    "castle": (_castle, 28, 70),
    "town_hall": (_town_hall, 55, 45),
    "dome": (_dome, 52, 50),
    "column": (_column, 45, 35),
    "stadium": (_stadium, 24, 140),
}


# --- the city ---------------------------------------------------------------------------


def build_city(cfg: dict, channels, h: np.ndarray, ns: np.ndarray, es: np.ndarray) -> City:
    rng = np.random.default_rng(cfg.get("seed", 1))
    emb = cfg.get("embankment_m", 12.0)
    water = unary_union([_band_polygon(c.line, c.width) for c in channels])
    core = np.asarray(cfg["founded_ne_m"], float)
    centre = np.asarray(cfg.get("centre_ne_m", [0, 0]), float)
    roads = _roads(cfg, rng, h, ns, es, channels, core)

    # The town's extent: a wavering round edge (radius_m), beyond which it thins out over a
    # fringe into the country, further along the old roads.
    r0 = cfg.get("radius_m", 800.0)
    fringe = cfg.get("fringe", 0.3)  # as a fraction of the radius
    ang = np.linspace(-math.pi, math.pi, 361)
    ph = rng.uniform(0, 2 * math.pi, 5)
    radius = r0 * (
        1
        + 0.08 * np.sin(3 * ang + ph[0])
        + 0.06 * np.sin(5 * ang + ph[1])
        + 0.04 * np.sin(9 * ang + ph[2])
    )
    outer = (
        radius
        * (1 + fringe)
        * (1 + 0.07 * np.sin(4 * ang + ph[3]) + 0.05 * np.sin(7 * ang + ph[4]))
    )
    disc = Polygon(centre + outer[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1))
    reach = Point(centre).buffer(r0 * (1 + fringe) + cfg.get("ribbon_reach_m", 220.0))
    outline = unary_union(
        [disc, *[ln.buffer(cfg.get("ribbon_m", 90.0)).intersection(reach) for _, ln in roads]]
    )

    def rel_radius(p) -> float:  # 1 at the town's round edge
        a = math.atan2(p[1] - centre[1], p[0] - centre[0])
        return math.hypot(p[0] - centre[0], p[1] - centre[1]) / float(np.interp(a, ang, radius))

    win = r0 * (1 + fringe) * 1.13 + cfg.get("ribbon_reach_m", 220.0) + 60
    window = (float(min(centre) - win), float(max(centre) + win))
    fld = Field(rng, channels, roads, core, window, cfg.get("grain_deg", -45.0))
    inside = shapely.contains_xy(outline, fld.nn, fld.ee)

    def spacing(q, fam: int) -> float:
        t = min(max((math.hypot(q[0] - core[0], q[1] - core[1]) - 250) / 550, 0.0), 1.0)
        loose = 1 + 0.6 * min(
            max((rel_radius(q) - 0.95) / 0.35, 0.0), 1.0
        )  # looser beyond the edge
        s = (40 + 42 * t) * loose
        return s if fam == 0 else 1.55 * s

    # Quays along the water in the older town; the old roads in town. Streets keep their
    # spacing from them, and cross streets start along them.
    quay_r = cfg.get("quays_within_m", 650.0)
    quays = [
        ln
        for ln in _lines(
            water.buffer(emb + QUAY_HALF_M)
            .boundary.intersection(outline)
            .intersection(Point(core).buffer(quay_r))
        )
        if ln.length > 30
    ]
    town_roads = [seg for _, ln in roads for seg in _lines(ln.intersection(outline.buffer(40)))]
    hashes = (_Hash(), _Hash())
    guides = []
    for ln in [*town_roads, *quays]:
        p, t = _samples([ln], 5.0)
        for k, (q, d) in enumerate(zip(p, t, strict=True)):
            f = fld.direction(q, 0)
            fam = 0 if abs(f[0] * d[0] + f[1] * d[1]) / max(math.hypot(*d), 1e-9) > 0.7 else 1
            hashes[fam].add(q)
            if k % 6 == 0:
                guides.append(((float(q[0]), float(q[1])), fam))
    wet_m = emb + QUAY_HALF_M
    traced = _streets(fld, inside, core, spacing, wet_m, guides, hashes)

    # Loose ends: a street that stopped short of another runs on to meet it; one that
    # reached the river may carry on over it as a bridge.
    geoms = [LineString(pts) for _, pts, _ in traced]
    tree = shapely.STRtree(geoms + town_roads + quays)
    all_geoms = geoms + town_roads + quays
    bridge_lines = [seg for _, ln in roads for seg in _lines(ln.intersection(water))]
    extra = cfg.get("extra_bridges", 6)
    ends = []
    for i, (_, pts, why) in enumerate(traced):
        for which, w in ((0, why[0]), (1, why[1])):
            a, b = (pts[1], pts[0]) if which == 0 else (pts[-2], pts[-1])
            ends.append((math.hypot(b[0] - core[0], b[1] - core[1]), i, which, w, a, b))
    ends.sort()
    for _, i, which, w, a, b in ends:
        fam, pts, why = traced[i]
        d = np.array(b) - np.array(a)
        d /= max(float(np.hypot(*d)), 1e-9)
        if w in ("close", "turn", "end"):
            ray = LineString([b, np.array(b) + d * 0.75 * spacing(b, fam)])
            best = None
            for j in tree.query(ray):
                if j == i:
                    continue
                for q in shapely.get_coordinates(ray.intersection(all_geoms[j])):
                    dist = float(np.hypot(*(q - np.array(b))))
                    if dist > 1.0 and (best is None or dist < best[0]):
                        best = (dist, (float(q[0]), float(q[1])))
            if best is not None:
                pts.insert(0, best[1]) if which == 0 else pts.append(best[1])
        elif w == "wet" and extra > 0:
            p = np.array(b)
            wetted, land = False, None
            for k in range(1, 60):
                q = p + d * 4 * k
                e = fld.value(fld.edge, q)
                wetted |= e < 0
                if wetted and e > wet_m:
                    land = q
                    break
            if land is None or not fld.value(inside, land):
                continue
            mid = (p + land) / 2
            if any(bl.distance(Point(mid)) < 220 for bl in bridge_lines):
                continue
            if not (hashes[0].near(land, 25) or hashes[1].near(land, 25)):
                continue
            span = LineString([p, land])
            if water.intersection(span).length < 0.6 * span.length - 2 * wet_m:
                continue
            bridge_lines.append(span)
            extra -= 1
            pts.insert(0, tuple(land)) if which == 0 else pts.append(tuple(land))

    streets: list[tuple[str, LineString, float]] = []
    for fam, pts, _ in traced:
        ln = LineString(pts).simplify(0.6)
        old = float(np.mean([math.hypot(q[0] - core[0], q[1] - core[1]) for q in pts])) < 280
        edge = float(np.mean([rel_radius(q) for q in pts])) > 1.0  # lanes out at the edge
        streets.append(("street", ln, 3.0 if old or edge else (4.5 if fam == 0 else 4.0)))
    streets += [("quay", ln, QUAY_HALF_M) for ln in quays]
    streets += [("main", ln, ROAD_HALF_M) for ln in town_roads]
    bridges = []
    for ln in bridge_lines:
        for seg in _lines(ln.intersection(water.buffer(emb))):
            a, b = np.asarray(seg.coords[0]), np.asarray(seg.coords[-1])
            d = (b - a) / max(float(np.hypot(*(b - a))), 1e-6)
            half = ROAD_HALF_M if any(seg.distance(r) < 1 for r in town_roads) else 5.0
            bridges.append((LineString([a - 2 * d, b + 2 * d]), half))

    # Blocks: the land between the streets.
    paved = unary_union(
        [ln.buffer(hw) for _, ln, hw in streets] + [ln.buffer(hw) for ln, hw in bridges]
    )
    land = outline.intersection(box(window[0], window[0], window[1], window[1])).difference(
        water.buffer(emb)
    )
    blocks = [p.simplify(0.4) for p in _polys(land.difference(paved)) if p.area > 150]
    cent = np.array([p.centroid.coords[0] for p in blocks])
    claimed = np.zeros(len(blocks), bool)
    arm = next((c for c in channels if c.name == "arm"), None)
    island = Polygon()
    if arm is not None:
        _, (i0, i1) = KDTree(channels[0].line).query([arm.line[0], arm.line[-1]])
        island = Polygon(
            [*arm.line, *channels[0].line[min(i0, i1) : max(i0, i1) + 1][::-1]]
        ).buffer(0)

    def claim(near, radius: float) -> Polygon:
        """The unclaimed blocks round a point, joined over the streets between them."""
        d = np.hypot(*(cent - near).T)
        order = [i for i in np.argsort(d) if not claimed[i]]
        pick = [i for i in order if d[i] <= radius] or order[:1]
        claimed[pick] = True
        site = unary_union([blocks[i].buffer(7.5, join_style="mitre") for i in pick])
        return site.buffer(-7.5, join_style="mitre").intersection(land)

    places: list[tuple[str, str, Polygon]] = []
    buildings: list[Building] = []
    labels: list[Building] = []

    def place(fn, site: Polygon, kind: str, height: float):
        """The landmark's parts, as large as fits in the site (up to its full size)."""
        c = np.asarray(polylabel(site, 1.0).coords[0])
        _, axis, _, _ = _frame(site)
        for s in (1.0, 0.85, 0.7, 0.55):
            for th in (math.pi / 2, axis) if kind == "cathedral" else (axis, axis + math.pi / 2):
                parts = fn(c, th, s, height)
                if site.buffer(-2).contains(unary_union([p.footprint for p in parts])):
                    return parts, c, th
        return None

    for lm in cfg.get("landmarks", []):
        fn, default_h, site_r = LANDMARKS[lm["kind"]]
        if lm["kind"] == "castle" and "near_ne_m" not in lm and arm is not None:
            near = arm.line[0] + 0.3 * (core - arm.line[0])  # the island's upstream tip
        else:
            near = np.asarray(lm.get("near_ne_m", core), float)
        height = float(lm.get("height_m", default_h))
        d = np.hypot(*(cent - near).T)
        found = None
        for least in (0.85, 0.55):  # rather a nearby site where it stands full size
            for i in [i for i in np.argsort(d) if not claimed[i]][:15]:
                dd = np.hypot(*(cent - cent[i]).T)
                group = [
                    j
                    for j in np.argsort(dd)
                    if not claimed[j] and dd[j] <= lm.get("site_m", site_r)
                ] or [i]
                merged = unary_union([blocks[j].buffer(7.5, join_style="mitre") for j in group])
                merged = merged.buffer(-7.5, join_style="mitre").intersection(land)
                site = next(
                    (
                        g
                        for g in _polys(merged)
                        if g.buffer(1).contains(blocks[i].representative_point())
                    ),
                    None,
                )
                if site is None:
                    continue
                got = place(fn, site, lm["kind"], height)
                if (
                    got is not None
                    and _frame(unary_union([p.footprint for p in got[0]]))[2]
                    >= least
                    * _frame(unary_union([p.footprint for p in fn(np.zeros(2), 0.0, 1.0, height)]))[
                        2
                    ]
                ):
                    claimed[
                        [
                            j
                            for j in group
                            if site.buffer(1).contains(blocks[j].representative_point())
                        ]
                    ] = True
                    found = (site, *got)
                    break
            if found:
                break
        if found is None:
            continue
        site, parts, c, th = found
        kind = {"column": "square", "town_hall": "square", "dome": "square"}.get(
            lm["kind"], "green"
        )
        places.append((kind, lm["name"], site))
        if lm["kind"] == "stadium":
            places.append(("pitch", "", _rect(c, th, 100, 64)))
        b = Building(parts, "castle" if lm["kind"] == "castle" else "landmark", lm["name"])
        buildings.append(b)
        labels.append(b)
    for pk in cfg.get("parks", []):
        places.append(
            ("park", pk["name"], claim(np.asarray(pk["near_ne_m"], float), pk.get("radius_m", 120)))
        )
    # Garden squares in the inner town, and small paved squares in the old one.
    d_core = np.hypot(*(cent - core).T)
    areas = np.array([b.area for b in blocks])
    for kind, count, lo, hi, a0, a1 in (
        ("garden", cfg.get("garden_squares", 6), 280, 650, 1500, 7000),
        ("square", cfg.get("squares", 3), 60, 300, 350, 2500),
    ):
        pool = [
            i
            for i in rng.permutation(len(blocks))
            if not claimed[i] and lo < d_core[i] < hi and a0 < areas[i] < a1
        ]
        chosen: list[int] = []
        for i in pool:
            if len(chosen) >= count:
                break
            if all(np.hypot(*(cent[i] - cent[j])) > 180 for j in chosen) and not island.contains(
                Point(cent[i])
            ):
                chosen.append(i)
        for i in chosen:
            claimed[i] = True
            places.append((kind, "", blocks[i]))
    paths = []
    for kind, _, poly in places:
        if kind in ("park", "green"):
            inner = poly.buffer(-3)
            for _, ln, _ in streets:
                if ln.intersects(inner):
                    paths += _lines(ln.intersection(inner))

    # The tallest rise by the river on the south-east side: round a point upstream of the
    # crossing, highest in the middle.
    tw = cfg.get("towers", {})
    main = channels[0].line
    k0 = int(KDTree(main).query(core)[1])
    walked = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(main[: k0 + 1][::-1], axis=0).T))])
    up = main[: k0 + 1][::-1][int(np.searchsorted(walked, tw.get("upstream_m", 450)))]
    cbd_r = tw.get("radius_m", 230.0)
    tallest = float(tw.get("tallest_m", 190))

    trees: list[np.ndarray] = []
    urban: list[Polygon] = []  # blocks with paved yards (the rest are gardens and fields)
    frontage = paved.buffer(2.0)
    shapely.prepare(frontage)
    for i, blk in enumerate(blocks):
        if claimed[i]:
            continue
        c = cent[i]
        dc = float(d_core[i])
        rr = rel_radius(c)
        d_cbd = float(np.hypot(*(c - up)))
        on_island = island.contains(Point(c))
        if d_cbd < cbd_r and not on_island:
            top = tallest * (0.25 + 0.75 * math.exp(-((d_cbd / (0.55 * cbd_r)) ** 2)))
            buildings += _towers(blk, top, rng)
            urban.append(blk)
            continue
        # Out from the edge the town thins: more and more plots left as fields …
        if rr > 1.0 and rng.random() > max(1 - (rr - 1.0) / fringe, 0.08) ** 1.3:
            continue
        # … detached houses take over from terraces and blocks, ever further apart …
        if rng.random() < min(max((rr - 0.82) / 0.25, 0.0), 1.0):
            hs, ts = _houses(
                blk, lambda p: frontage.contains(Point(p)), rng, 1 + 1.5 * max(rr - 0.95, 0.0)
            )
            buildings += hs
            trees += ts
            continue
        urban.append(blk)
        lower = 1 - 0.35 * min(max((rr - 0.6) / 0.4, 0.0), 1.0)  # … and the rest lower
        if on_island or dc < 280:  # the old town: narrow, tall houses with pitched roofs
            lots, yard = _lots(blk, rng.uniform(10, 14), (6, 12), rng)
            for q in lots:
                buildings.append(
                    Building([_roofed(q, rng.uniform(10, 19), "gable", rng.uniform(3, 4.5))], "old")
                )
            if yard.area > 300:  # back buildings round small courtyards
                back, yard = _lots(yard.buffer(-3, join_style="mitre"), 8.0, (6, 10), rng)
                for q in back:
                    buildings.append(Building([_roofed(q, rng.uniform(8, 12), "gable", 3)], "old"))
        else:
            ter = 0.3 + 0.35 * min(max((rr - 0.7) / 0.3, 0.0), 1.0)
            style = (
                "block"
                if dc < 560
                else rng.choice(["block", "slabs", "terrace"], p=[0.8 - ter, 0.2, ter])
            )
            if style == "slabs":
                buildings += _slabs(blk, rng, lower)
                yard = blk.buffer(-20)
            elif style == "terrace":
                lots, yard = _lots(blk, 9.5, (5.5, 7.5), rng)
                for q in lots:
                    buildings.append(
                        Building(
                            [_roofed(q, rng.uniform(8, 10.5) * max(lower, 0.85), "gable", 3)],
                            "terrace",
                        )
                    )
            else:
                lots, yard = _lots(blk, rng.uniform(12, 16), (9, 22) if dc < 560 else (14, 30), rng)
                for q in lots:
                    if rng.random() < 0.04:
                        continue  # a gap: a yard, a garden, a car park
                    if rng.random() < 0.1:
                        buildings.append(Building([Part(q, 0, rng.uniform(26, 42))], "modern"))
                    else:
                        z = (rng.uniform(14, 24) if dc < 560 else rng.uniform(12, 21)) * lower
                        buildings.append(Building([Part(q, 0, z)], "block"))
        if not on_island and dc >= 280 and yard.area > 400:
            places.append(("yard", "", yard))  # courtyard gardens
        for g in _polys(yard.buffer(-3) if not yard.is_empty else yard):
            for _ in range(min(int(g.area / 260), 6)):
                p = np.array(g.representative_point().coords[0]) if g.area < 120 else None
                for _ in range(10):
                    x0, y0, x1, y1 = g.bounds
                    q = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
                    if g.contains(Point(q)):
                        p = q
                        break
                if p is not None:
                    trees.append(np.array([*p, rng.uniform(2.5, 4.5), rng.uniform(6, 9)]))

    # Beyond the edge, streets run only as far as the houses along them: no roads into empty
    # fields.
    edge = Polygon(centre + radius[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1))
    lived = unary_union(
        [edge, *[b.parts[0].footprint.buffer(35) for b in buildings if b.kind == "house"]]
    )
    kept = []
    for kind, ln, hw in streets:
        if kind != "street":
            kept.append((kind, ln, hw))
            continue
        kept += [(kind, seg, hw) for seg in _lines(ln.intersection(lived)) if seg.length > 25]
    streets[:] = kept

    # The tallest tower gets its name.
    towers = [b for b in buildings if b.kind == "tower"]
    if towers:
        top = max(towers, key=lambda b: b.height)
        scale = tallest / top.height
        for p in top.parts:
            p.z0 *= scale
            p.z1 *= scale
        top.name = tw.get("tallest_name", "")
        labels.append(top)

    # Trees in parks and greens (leaving open lawns), round squares, along the main roads
    # and quays.
    noise = gaussian_filter(rng.standard_normal(fld.nn.shape), 3)
    noise /= noise.std()
    built = unary_union(
        [p.footprint for b in buildings if b.kind in ("landmark", "castle") for p in b.parts]
    ).buffer(4)
    for kind, _, poly in places:
        x0, y0, x1, y1 = poly.bounds
        if kind in ("park", "green"):
            gx, gy = np.meshgrid(np.arange(x0, x1, 10.0), np.arange(y0, y1, 10.0), indexing="ij")
            p = np.stack([gx.ravel(), gy.ravel()], 1) + rng.uniform(-4, 4, (gx.size, 2))
            keep = shapely.contains_xy(poly.buffer(-3), p[:, 0], p[:, 1]) & ~shapely.contains_xy(
                built, p[:, 0], p[:, 1]
            )
            keep &= np.array([fld.value(noise, q) > (-0.2 if kind == "park" else 0.6) for q in p])
            if paths:
                near_path = shapely.distance(shapely.points(p), unary_union(paths)) < 3
                keep &= ~near_path
            for q in p[keep]:
                trees.append(np.array([*q, rng.uniform(3, 6), rng.uniform(7, 11)]))
        elif kind in ("garden", "square") and poly.area > 900:
            ring = poly.buffer(-4).exterior if not poly.buffer(-4).is_empty else None
            if ring is not None:
                for s in np.arange(0, ring.length, 10.0 if kind == "garden" else 14.0):
                    q = np.asarray(ring.interpolate(s).coords[0])
                    trees.append(np.array([*q, rng.uniform(3, 4.5), rng.uniform(7, 10)]))
    for kind, ln, hw in streets:
        if kind not in ("main", "quay"):
            continue
        for side in (-1, 1):
            off = ln.offset_curve(side * (hw - 1.5))
            for g in _lines(off):
                for s in np.arange(6, g.length, 12.0):
                    q = np.asarray(g.interpolate(s).coords[0])
                    if fld.value(fld.edge, q) > emb:
                        trees.append(np.array([*q, rng.uniform(2.5, 3.5), rng.uniform(7, 9)]))

    for b in buildings:
        if b.kind in PALETTE and not b.wall:
            walls, roofs = PALETTE[b.kind]
            b.wall, b.roof = str(rng.choice(walls)), str(rng.choice(roofs))
    return City(
        cfg.get("name", "the city"),
        water,
        roads,
        streets,
        bridges,
        urban,
        places,
        paths,
        buildings,
        np.array(trees) if trees else np.zeros((0, 4)),
        labels,
        outline,
    )


# --- drawing ----------------------------------------------------------------------------


def faces(part: Part, view: np.ndarray) -> list[tuple[np.ndarray, float, bool]]:
    """What a viewer (unit vector towards them) sees of a part: (corners (k, 3) as north,
    east, up; brightness; whether it is roof), walls first, nearest roof slope last."""
    q = np.asarray(part.footprint.exterior.coords)[:-1]
    if len(q) < 3:
        return []
    area = float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))
    sign = 1.0 if area > 0 else -1.0
    z0, z1 = part.z0, part.z1

    def bright(n: np.ndarray) -> float:
        return 0.55 + 0.45 * max(0.0, float(n @ LIGHT))

    walls = []
    for k in range(len(q)):
        a, b = q[k], q[(k + 1) % len(q)]
        d = b - a
        n = sign * np.array([d[1], -d[0], 0.0])
        n /= max(float(np.linalg.norm(n)), 1e-9)
        if n @ view > 0:
            walls.append((np.array([[*a, z0], [*b, z0], [*b, z1], [*a, z1]]), bright(n), False))
    roof: list[tuple[np.ndarray, float, bool, float]] = []
    c = q.mean(0)

    def add(pts: np.ndarray, is_roof: bool) -> None:
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        n /= max(float(np.linalg.norm(n)), 1e-9)
        mid = pts.mean(0)
        if n[2] < -1e-6 or (abs(n[2]) < 1e-6 and n[:2] @ (mid[:2] - c) < 0):
            n = -n
        if n @ view > 0:
            roof.append((pts, bright(n), is_roof, float(n @ view)))

    if part.roof == "gable" and len(q) == 4 and part.rise > 0:
        k = (
            0
            if np.hypot(*(q[1] - q[0])) + np.hypot(*(q[3] - q[2]))
            >= np.hypot(*(q[2] - q[1])) + np.hypot(*(q[0] - q[3]))
            else 1
        )
        v = [q[(k + i) % 4] for i in range(4)]
        zt = z1 + part.rise
        m1, m2 = (v[1] + v[2]) / 2, (v[3] + v[0]) / 2
        add(np.array([[*v[0], z1], [*v[1], z1], [*m1, zt], [*m2, zt]]), True)
        add(np.array([[*v[2], z1], [*v[3], z1], [*m2, zt], [*m1, zt]]), True)
        add(np.array([[*v[1], z1], [*v[2], z1], [*m1, zt]]), False)
        add(np.array([[*v[3], z1], [*v[0], z1], [*m2, zt]]), False)
    elif part.roof == "hip" and part.rise > 0:
        apex = np.array([*c, z1 + part.rise])
        for k in range(len(q)):
            add(np.array([[*q[k], z1], [*q[(k + 1) % len(q)], z1], apex]), True)
    else:
        roof.append(
            (np.column_stack([q, np.full(len(q), z1)]), bright(np.array([0, 0, 1.0])), True, 1.0)
        )
    roof.sort(key=lambda f: f[3])
    return walls + [(p, s, r) for p, s, r, _ in roof]


def building_faces(b: Building, view: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """A building's visible faces, in drawing order, with their colours."""
    wall, top = b.wall or TONES[b.kind][0], b.roof or TONES[b.kind][1]
    out = []
    parts = sorted(b.parts, key=lambda p: (p.z0, -float(np.sum(p.footprint.centroid.coords[0]))))
    for p in parts:
        for pts, shade, is_roof in faces(p, view):
            base = np.array(_rgb(top if is_roof else wall))
            out.append((pts, _hex(np.clip(base * shade, 0, 1))))
    return out


def _rgb(c: str) -> tuple[float, float, float]:
    return tuple(int(c[i : i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]


def _hex(rgb) -> str:
    return "#" + "".join(f"{round(x * 255):02x}" for x in rgb)


def _path(poly: Polygon, tf):
    from matplotlib.path import Path as MPath

    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        xy = np.asarray(ring.coords)
        x, y = tf(xy[:, 0], xy[:, 1])
        verts += list(zip(x, y, strict=True))
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY]
    return MPath(verts, codes)


def fill(ax, geoms, tf, z: float, **kw) -> None:
    """Fill shapely polygons (holes kept) through ``tf(north, east) → (x, y)``."""
    from matplotlib.collections import PathCollection

    paths = [_path(p, tf) for g in geoms for p in _polys(g)]
    if paths:
        ax.add_collection(PathCollection(paths, zorder=z, **kw), autolim=False)


def shade_overlay(h: np.ndarray, cell: float) -> np.ndarray:
    """The terrain's light and shade as a see-through layer (RGBA), to lay over what is drawn
    on the ground (fields, woods) so the hills still read."""
    from matplotlib.colors import LightSource

    i = LightSource(azdeg=315, altdeg=40).hillshade(h, vert_exag=1.5, dx=cell, dy=cell)
    d = i - math.sin(math.radians(40))
    rgba = np.zeros((*h.shape, 4))
    rgba[..., 3] = np.where(d < 0, np.clip(-d * 1.3, 0, 0.6), np.clip(d * 0.9, 0, 0.3))
    rgba[..., :3] = np.where(d[..., None] < 0, 0.0, 1.0)
    return rgba


def draw_land(ax, city: City, tf, detail: bool, squash: float = 1.0, crowns: bool = True) -> None:
    """Fields (with furrows when in detail), woods (with their trees' crowns, unless the trees
    are drawn standing), hedges, and the wind farm's pads and tracks. ``squash``: how much
    the view flattens a circle on the ground."""
    from matplotlib.collections import EllipseCollection, LineCollection

    from sketch_land import ARABLE, CROPS, GRAVEL, WOOD

    by_crop: dict[str, list] = {}
    for crop, poly, _ in city.fields:
        by_crop.setdefault(crop, []).append(poly)
    for crop, polys in by_crop.items():
        fill(ax, polys, tf, 3.0, facecolors=CROPS[crop][0], edgecolors="none")
    if detail and city.fields:
        segs = []
        for crop, poly, _ in city.fields:
            if crop not in ARABLE:
                continue
            c, th, length, depth = _frame(poly)
            u, v = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
            for off in np.arange(-depth / 2 + 7, depth / 2, 14.0):
                ln = LineString([c + v * off - u * length, c + v * off + u * length]).intersection(
                    poly
                )
                segs += [np.column_stack(tf(*np.asarray(g.coords).T)) for g in _lines(ln)]
        ax.add_collection(
            LineCollection(segs, colors=(0, 0, 0, 0.12), linewidths=0.35, zorder=3.02)
        )
    # The woods' floor, a little in from the edge (the trees themselves make the edge), and
    # the glades in them.
    floor = [g for w in city.woods for g in _polys(w.buffer(-7))]
    fill(ax, floor, tf, 3.05, facecolors=WOOD if crowns else "#3a6230", edgecolors="none")
    fill(ax, city.glades, tf, 3.055, facecolors="#a2c779", edgecolors="none")
    if crowns and len(city.forest):  # seen from above: each tree's crown
        from sketch_land import crown_colour

        c = city.forest
        x, y = tf(c[:, 0], c[:, 1])
        cols = [crown_colour(k, s) for k, s in zip(c[:, 4], c[:, 5], strict=True)]
        ax.add_collection(
            EllipseCollection(
                2 * c[:, 2],
                2 * c[:, 2] * squash,
                0,
                units="xy",
                offsets=np.column_stack([x, y]),
                offset_transform=ax.transData,
                facecolors=cols,
                edgecolors="none",
                zorder=3.06,
            )
        )
    if city.hedges:
        fill(
            ax,
            [unary_union([ln.buffer(1.6) for ln in city.hedges])],
            tf,
            3.1,
            facecolors="#3b6b32",
            edgecolors="none",
        )
    service = [ln.buffer(hw) for k, ln, hw in city.streets if k == "service"]
    if service:
        fill(ax, [unary_union(service)], tf, 3.2, facecolors=GRAVEL, edgecolors="none")


def draw_ground(ax, city: City, tf, detail: bool, squash: float = 1.0, crowns: bool = True) -> None:
    """The ground: the land round the town, then the town's yards, streets, squares, parks,
    paths and bridges."""
    from matplotlib.collections import LineCollection

    draw_land(ax, city, tf, detail, squash, crowns)
    roads = [np.column_stack(tf(*np.asarray(ln.coords).T)) for _, ln in city.roads]
    ax.add_collection(
        LineCollection(roads, colors=MAIN_ROAD, linewidths=2.2 if detail else 0.9, zorder=3.3)
    )
    tracks = [
        np.column_stack(tf(*np.asarray(ln.coords).T)) for k, ln, _ in city.streets if k == "track"
    ]
    if tracks:
        ax.add_collection(
            LineCollection(tracks, colors="#b59f78", linewidths=1.3 if detail else 0.5, zorder=3.3)
        )
    fill(ax, city.blocks, tf, 3.5, facecolors=GROUND, edgecolors="none")
    mains = [ln.buffer(hw) for k, ln, hw in city.streets if k == "main"]
    fences = [
        np.column_stack(tf(*np.asarray(ln.coords).T)) for k, ln, _ in city.streets if k == "fence"
    ]
    if fences:
        ax.add_collection(
            LineCollection(fences, colors="#55595e", linewidths=0.7 if detail else 0.3, zorder=4.2)
        )
    fill(
        ax,
        [
            unary_union(
                [
                    ln.buffer(hw)
                    for k, ln, hw in city.streets
                    if k not in ("main", "track", "service", "fence")
                ]
            )
        ],
        tf,
        3.6,
        facecolors=STREET,
        edgecolors="none",
    )
    fill(ax, [unary_union(mains)], tf, 3.7, facecolors=MAIN_ROAD, edgecolors="none")
    for kind, _, poly in city.places:
        fill(
            ax,
            [poly],
            tf,
            3.8 if kind != "pitch" else 3.9,
            facecolors=PLACE_COLOURS[kind],
            edgecolors="none",
        )
    segs = [np.column_stack(tf(*np.asarray(ln.coords).T)) for ln in city.paths]
    if segs:
        ax.add_collection(
            LineCollection(segs, colors="#efe7d6", linewidths=1.2 if detail else 0.4, zorder=3.95)
        )
    for ln, hw in city.bridges:
        fill(
            ax,
            [ln.buffer(hw + 1.5, cap_style="flat")],
            tf,
            4.0,
            facecolors="#55585c",
            edgecolors="none",
        )
        fill(
            ax, [ln.buffer(hw, cap_style="flat")], tf, 4.1, facecolors="#c9cbcf", edgecolors="none"
        )


def draw_city(ax, city: City, detail: bool, labels: bool = True, buildings: bool = True) -> None:
    """The city top down: the ground, buildings by kind (their roofs) and trees."""
    from matplotlib.collections import EllipseCollection, PolyCollection

    def tf(n, e):
        return e, n

    draw_ground(ax, city, tf, detail)
    verts, cols = [], []
    for b in city.buildings if buildings else []:
        for p in sorted(b.parts, key=lambda p: p.z0):
            xy = np.asarray(p.footprint.exterior.coords)
            verts.append(xy[:, ::-1])
            cols.append(b.roof or TONES[b.kind][1])
    ax.add_collection(
        PolyCollection(
            verts,
            facecolors=cols,
            edgecolors="#4a4540" if detail else "none",
            linewidths=0.2,
            zorder=6,
        )
    )
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    for ct in city.cooling_towers:  # seen from above: the shell, and its opening
        ax.add_patch(
            Circle((ct["e"], ct["n"]), ct["r_base"], fc="#dcd9d2", ec="#555", lw=0.4, zorder=8)
        )
        ax.add_patch(
            Circle((ct["e"], ct["n"]), ct["r_throat"] * 1.08, fc="#7a7e83", ec="none", zorder=8.1)
        )
    if city.wires:
        segs = [np.asarray(w["pts"])[:, [1, 0]] for w in city.wires]
        ax.add_collection(
            LineCollection(segs, colors="#4f5358", linewidths=0.35 if detail else 0.15, zorder=8.2)
        )
    for p in city.pylons:
        ax.plot(
            p["e"],
            p["n"],
            "s",
            ms=3 if detail else 1.4,
            mfc="none",
            mec="#3d4146",
            mew=0.6,
            zorder=8.3,
        )
    if city.solar:  # seen from above: each row of panels
        from sketch_infra import PANEL, PANEL_DEPTH

        rows = [
            np.array([[a, n], [b, n], [b, n + PANEL_DEPTH], [a, n + PANEL_DEPTH]])
            for n, a, b in city.solar
        ]
        ax.add_collection(PolyCollection(rows, facecolors=PANEL, edgecolors="none", zorder=6))
    for m in city.masts:
        guys = [np.array([[m["e"], m["n"]], [a[1], a[0]]]) for a in m["anchors"]]
        ax.add_collection(
            LineCollection(guys, colors="#55595e", linewidths=0.4 if detail else 0.2, zorder=8.2)
        )
        ax.plot(
            m["e"],
            m["n"],
            "^",
            ms=5 if detail else 2.5,
            color="#cf4237",
            mec="#5a1a14",
            mew=0.4,
            zorder=8.4,
        )
    for t in city.turbines:
        f = np.array([math.cos(t["facing"]), math.sin(t["facing"])])
        a = np.array([t["n"], t["e"]]) + f * 4 + np.array([-f[1], f[0]]) * t["rotor"] / 2
        b = np.array([t["n"], t["e"]]) + f * 4 - np.array([-f[1], f[0]]) * t["rotor"] / 2
        ax.plot(
            [a[1], b[1]],
            [a[0], b[0]],
            color="#f4f5f6",
            lw=1.6 if detail else 0.8,
            zorder=8,
            solid_capstyle="round",
            path_effects=None,
        )
        ax.plot(
            t["e"],
            t["n"],
            "o",
            ms=3 if detail else 1.5,
            color="#e9eaec",
            mec="#555",
            mew=0.4,
            zorder=8,
        )
    if len(city.trees):
        t = city.trees
        ax.add_collection(
            EllipseCollection(
                2 * t[:, 2],
                2 * t[:, 2],
                0,
                units="xy",
                offsets=t[:, [1, 0]],
                offset_transform=ax.transData,
                facecolors=TREE,
                edgecolors="#3d6e2c",
                linewidths=0.2 if detail else 0,
                zorder=7,
            )
        )
    if not labels:
        return
    for name, c in city.settlements:
        ax.text(
            c[1] + 60,
            c[0] + 60,
            name,
            fontsize=9 if detail else 7,
            fontweight="bold",
            zorder=9,
            color="#3a2a1a",
            bbox={"fc": "white", "ec": "none", "alpha": 0.7, "pad": 1},
            clip_on=True,
        )
    for b in city.labels:
        c = b.centre
        ax.annotate(
            f"{b.name} {b.height:.0f} m",
            xy=(c[1], c[0]),
            xytext=(14, 14),
            textcoords="offset points",
            fontsize=8 if detail else 6,
            zorder=9,
            bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1},
            arrowprops={"arrowstyle": "-", "color": "#333", "lw": 0.6},
        )
    if detail:
        from matplotlib.patches import Patch

        names = {
            "old": "old town",
            "terrace": "terraces",
            "block": "city blocks",
            "modern": "modern",
            "tower": "towers",
            "house": "houses",
            "landmark": "landmarks",
            "castle": "castle",
        }
        handles = [Patch(fc=TONES[k][1], ec="#4a4540", lw=0.4, label=v) for k, v in names.items()]
        handles += [
            Patch(fc=PLACE_COLOURS["park"], label="parks, greens"),
            Patch(fc=PLACE_COLOURS["square"], label="squares"),
        ]
        ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.85)


def draw_roads(ax, city: City, detail: bool, z: int = 4) -> None:
    """The old main roads, whole (for the map sheet)."""
    for _, ln in city.roads:
        x, y = ln.xy
        ax.plot(y, x, color="#8c8478", lw=3.6 if detail else 1.6, zorder=z, solid_capstyle="round")
        ax.plot(y, x, color="#f4efe6", lw=2.4 if detail else 0.9, zorder=z, solid_capstyle="round")


# --- edits to a locked city --------------------------------------------------------------


def raise_tower(city: City, name: str, height: float) -> None:
    """Make the named tower ``height`` m to its roof: everything above its podium stretched."""
    b = next(b for b in city.buildings if b.name == name)
    podium = b.parts[0].z1
    f = (height - podium) / (b.height - podium)
    for p in b.parts[1:]:
        p.z0, p.z1 = podium + (p.z0 - podium) * f, podium + (p.z1 - podium) * f


def add_antennas(city: City, rng) -> None:
    """Antennas on every tower's roof: two or three masts (6-18 m; the tallest tower one of
    20 m in the middle), each with a red light on top for aircraft."""
    for b in [b for b in city.buildings if b.kind == "tower"]:
        top = max(b.parts, key=lambda p: p.z1)
        roof = top.z1
        c, th, length, depth = _frame(top.footprint)
        u, v = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
        spots = [
            c + u * a * (length / 2 - 3) + v * s * (depth / 2 - 3)
            for a, s in ((1, 1), (-1, -1), (1, -1), (-1, 1))
        ]
        masts = [(c, 20.0, 1.4)] if b.name else []
        for k in rng.permutation(4)[: int(rng.integers(2, 4))]:
            masts.append((spots[k], float(rng.uniform(6, 18)), 0.8))
        for p, h, w in masts:
            city.buildings.append(Building([Part(_rect(p, th, w, w), roof, roof + h)], "antenna"))
            city.buildings.append(
                Building([Part(_rect(p, th, w + 0.4, w + 0.4), roof + h, roof + h + 0.8)], "beacon")
            )


def rebuild_arena(city: City, name: str = "Stadium") -> None:
    """The arena as a bowl round its pitch: a lower tier of stands (to 9 m), an upper tier (to
    22 m) and a roof over the upper tier's back (to 24 m), round a 90 x 56 m pitch that fits
    inside the lower tier; nothing grows in the bowl, and no path crosses it."""
    b = next(b for b in city.buildings if b.name == name)
    hull = unary_union([p.footprint for p in b.parts]).convex_hull
    c, th, length, width = _frame(hull)
    ra, rb = length / 2, width / 2  # the outer wall
    k = 48
    a = np.linspace(0, 2 * math.pi, k + 1)
    u, v = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])

    def ring(sa, sb):
        return [c + sa * math.cos(x) * u + sb * math.sin(x) * v for x in a]

    inner, mid, outer = ring(61, 43), ring(0.5 * (61 + ra), 0.5 * (43 + rb)), ring(ra, rb)
    roof_in = ring(0.5 * (61 + ra) + 3, 0.5 * (43 + rb) + 3)
    parts = []
    for i in range(k):
        parts.append(Part(Polygon([inner[i], inner[i + 1], mid[i + 1], mid[i]]), 0, 9))
        parts.append(Part(Polygon([mid[i], mid[i + 1], outer[i + 1], outer[i]]), 0, 22))
        parts.append(Part(Polygon([roof_in[i], roof_in[i + 1], outer[i + 1], outer[i]]), 22, 24))
    b.parts = parts
    bowl = Polygon(outer).buffer(4)
    city.places = [
        (kind, n, p) for kind, n, p in city.places if kind != "pitch" or not p.intersects(bowl)
    ]
    city.places.append(("pitch", "", _rect(c, th, 90, 56)))
    if len(city.trees):
        city.trees = city.trees[~shapely.contains_xy(bowl, city.trees[:, 0], city.trees[:, 1])]
    city.paths = [
        seg
        for ln in city.paths
        for seg in ([ln] if not ln.intersects(bowl) else _lines(ln.difference(bowl)))
    ]


# --- locking ----------------------------------------------------------------------------


def _wkt(g) -> str:
    return shapely.to_wkt(g, rounding_precision=2)


def building_record(b: Building) -> dict:
    return {
        "kind": b.kind,
        "name": b.name,
        "wall": b.wall,
        "roof": b.roof,
        "parts": [
            [_wkt(p.footprint), round(p.z0, 2), round(p.z1, 2), p.roof, round(p.rise, 2)]
            for p in b.parts
        ],
    }


def building_from(d: dict) -> Building:
    return Building(
        [Part(shapely.from_wkt(fp), z0, z1, roof, rise) for fp, z0, z1, roof, rise in d["parts"]],
        d["kind"],
        d["name"],
        d["wall"],
        d["roof"],
    )


def save_city(city: City, path) -> None:
    """Write the city, exactly as built, to ``path`` (gzipped JSON; geometry as WKT to cm)."""
    import gzip
    import json

    index = {id(b): i for i, b in enumerate(city.buildings)}
    wkt = _wkt
    data = {
        "name": city.name,
        "water": wkt(city.water),
        "outline": wkt(city.outline),
        "roads": [[n, wkt(ln)] for n, ln in city.roads],
        "streets": [[k, wkt(ln), hw] for k, ln, hw in city.streets],
        "bridges": [[wkt(ln), hw] for ln, hw in city.bridges],
        "blocks": [wkt(b) for b in city.blocks],
        "places": [[k, n, wkt(p)] for k, n, p in city.places],
        "paths": [wkt(ln) for ln in city.paths],
        "buildings": [building_record(b) for b in city.buildings],
        "trees": np.round(city.trees, 2).tolist(),
        "labels": [index[id(b)] for b in city.labels],
    }
    with gzip.open(path, "wt") as f:
        json.dump(data, f, separators=(",", ":"))


def load_city(path) -> City:
    """A city saved by ``save_city``."""
    import gzip
    import json

    with gzip.open(path, "rt") as f:
        d = json.load(f)
    g = shapely.from_wkt
    buildings = [building_from(b) for b in d["buildings"]]
    return City(
        d["name"],
        g(d["water"]),
        [(n, g(w)) for n, w in d["roads"]],
        [(k, g(w), hw) for k, w, hw in d["streets"]],
        [(g(w), hw) for w, hw in d["bridges"]],
        [g(w) for w in d["blocks"]],
        [(k, n, g(w)) for k, n, w in d["places"]],
        [g(w) for w in d["paths"]],
        buildings,
        np.asarray(d["trees"], float).reshape(-1, 4),
        [buildings[i] for i in d["labels"]],
        g(d["outline"]),
    )
