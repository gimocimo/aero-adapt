"""World generation (docs/07 §3): an authored :class:`WorldSpec` → every feature, exactly placed.

The output is deterministic (one seeded generator, consumed in a fixed order), so a world id
names one exact world: the simulator, the planner, the contingency's survey and the viewer
all read the same fields, hedges, trees and buildings.

Coordinates are site NE metres (north, east); heights are metres above the site datum, the
ground at the origin (home). Shapely geometries use x = north, y = east.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box
from shapely.ops import unary_union

from aeroadapt.core.polygons import principal_chords
from aeroadapt.world.schema import BuildingSpec, FieldSpec, WorldSpec

FIELD_MIN_AREA_M2 = 12_000.0  # smaller leftovers are rough grass, without hedges
ZONE_INSET_M = 15.0  # a safe zone keeps this far inside its field's hedges
FLATTEN_BLEND_M = 40.0
RIVER_BANK_M = 20.0  # the channel's banks blend into the valley floor over this width
VERGE_M = 3.0  # roads: verge each side (no fields, no hedges)
HEDGE_SEGMENT_M = 20.0
WOOD_TREE_SPACING_M = 7.5
TREE_BROADLEAF, TREE_CONIFER, TREE_WILLOW = 0, 1, 2
FUNNEL_LENGTH_M = 450.0  # treeless approach funnels off the ends of each landing ground …
FUNNEL_SPLAY = 0.15  # … widening by 15 % of the distance


# --------------------------------------------------------------------------- features
@dataclass(frozen=True)
class Dtm:
    """Terrain heights on a regular grid: rows along north from ``n0``, columns along east
    from ``e0``, ``cell`` metres apart."""

    n0: float
    e0: float
    cell: float
    h: np.ndarray  # (rows, cols) float64

    def at(self, n: np.ndarray | float, e: np.ndarray | float) -> np.ndarray:
        """Bilinear height (clamped at the edges)."""
        x = (np.asarray(n, float) - self.n0) / self.cell
        y = (np.asarray(e, float) - self.e0) / self.cell
        rows, cols = self.h.shape
        x = np.clip(x, 0.0, rows - 1.000001)
        y = np.clip(y, 0.0, cols - 1.000001)
        i, j = x.astype(int), y.astype(int)
        fx, fy = x - i, y - j
        h = self.h
        return (h[i, j] * (1 - fx) * (1 - fy) + h[i + 1, j] * fx * (1 - fy)
                + h[i, j + 1] * (1 - fx) * fy + h[i + 1, j + 1] * fx * fy)  # fmt: skip


@dataclass(frozen=True)
class Building:
    kind: str
    n: float
    e: float
    length: float
    width: float
    heading: float  # rad, of the long axis
    base: float  # ground height (lowest under the footprint)
    eaves: float  # above the base
    ridge: float  # above the base (spire top for a tower)
    roof: str  # gable | flat | spire
    variant: int  # colour scheme
    name: str | None = None

    def footprint(self) -> Polygon:
        return _rect(self.n, self.e, self.length, self.width, self.heading)


@dataclass(frozen=True)
class Hedge:
    line: LineString
    height: float
    width: float


@dataclass(frozen=True)
class Road:
    id: str
    name: str | None
    kind: str
    width: float
    line: LineString


@dataclass(frozen=True)
class River:
    id: str
    name: str
    width: float
    depth: float
    line: LineString


@dataclass(frozen=True)
class Wood:
    id: str
    name: str
    kind: str
    canopy: float
    polygon: Polygon


@dataclass(frozen=True)
class Plot:
    """A field (fenced by hedges) or another patch of land cover."""

    id: str
    kind: str  # field | rough | settlement | yard | green | airstrip
    cover: str  # wheat, pasture, …; for other kinds the kind itself
    polygon: Polygon
    zone: str | None = None  # safe zone id, when it is one
    name: str | None = None


@dataclass(frozen=True)
class Structure:
    id: str
    name: str
    kind: str  # mast | silo
    n: float
    e: float
    base: float
    height: float
    radius: float


@dataclass(frozen=True)
class Trees:
    n: np.ndarray
    e: np.ndarray
    base: np.ndarray  # ground height
    height: np.ndarray  # above the ground
    crown: np.ndarray  # crown radius
    kind: np.ndarray  # TREE_*
    wood: np.ndarray  # index into World.woods, -1 outside the woods

    def __len__(self) -> int:
        return len(self.n)


@dataclass(frozen=True)
class World:
    spec: WorldSpec
    dtm: Dtm
    river: River | None
    roads: list[Road]
    woods: list[Wood]
    plots: list[Plot]
    hedges: list[Hedge]
    trees: Trees
    buildings: list[Building]
    structures: list[Structure]
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        (n0, e0), (n1, e1) = self.spec.bounds_ne_m
        return n0, e0, n1, e1

    def zone_plots(self) -> list[Plot]:
        """Plots that are safe zones, in the order the world lists its zones."""
        by_zone = {p.zone: p for p in self.plots if p.zone is not None}
        return [by_zone[z.id] for z in self.spec.zones if z.id in by_zone]

    def obstacles(self):  # -> raster.Obstacles (imported lazily: numpy-heavy)
        if "obstacles" not in self._cache:
            from aeroadapt.world.raster import rasterise

            self._cache["obstacles"] = rasterise(self)
        return self._cache["obstacles"]

    def survey(self) -> list[dict]:
        """The safe zones as site entries (docs/07 §5)."""
        if "survey" not in self._cache:
            from aeroadapt.world.survey import survey_zones

            self._cache["survey"] = survey_zones(self)
        return self._cache["survey"]


# --------------------------------------------------------------------------- helpers
def _rect(n: float, e: float, length: float, width: float, heading: float) -> Polygon:
    u = np.array([math.cos(heading), math.sin(heading)])
    v = np.array([-u[1], u[0]])
    c = np.array([n, e])
    hl, hw = length / 2, width / 2
    pts = [c + hl * u + hw * v, c - hl * u + hw * v, c - hl * u - hw * v, c + hl * u - hw * v]
    return Polygon(pts)


def chaikin(pts: np.ndarray, n: int, *, closed: bool) -> np.ndarray:
    """Corner cutting: smooth polylines (ends kept) and outlines."""
    p = np.asarray(pts, float)
    for _ in range(n):
        if closed:
            q = np.roll(p, -1, axis=0)
            p = np.stack([0.75 * p + 0.25 * q, 0.25 * p + 0.75 * q], 1).reshape(-1, 2)
        else:
            a, b = p[:-1], p[1:]
            mid = np.stack([0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b], 1).reshape(-1, 2)
            p = np.vstack([p[:1], mid, p[-1:]])
    return p


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _parts(g) -> list[Polygon]:
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def _lines(g) -> list[LineString]:
    if g.is_empty:
        return []
    if isinstance(g, LineString):
        return [g]
    out: list[LineString] = []
    for p in getattr(g, "geoms", []):
        out += _lines(p)
    return out


def _segments(line: LineString, step: float) -> list[LineString]:
    """Split a line into pieces of at most ``step`` metres."""
    n = max(math.ceil(line.length / step), 1)
    d = np.linspace(0.0, line.length, n + 1)
    pts = [line.interpolate(x) for x in d]
    return [LineString([pts[k], pts[k + 1]]) for k in range(n)]


def _funnels(zone: Polygon) -> list[Polygon]:
    """The approach funnels off both ends of a landing ground's long axis."""
    ring = np.asarray(zone.exterior.coords)[:-1]
    centre, axis, hl, hw = principal_chords(ring)
    d = np.array([math.cos(axis), math.sin(axis)])
    r = np.array([-d[1], d[0]])
    out = []
    for s in (1.0, -1.0):
        a = centre + s * hl * d
        b = centre + s * (hl + FUNNEL_LENGTH_M) * d
        w0, w1 = hw + 20.0, hw + 20.0 + FUNNEL_SPLAY * FUNNEL_LENGTH_M
        out.append(Polygon([a + w0 * r, b + w1 * r, b - w1 * r, a - w0 * r]))
    return out


def _building(spec: BuildingSpec, dtm: Dtm, variant: int) -> Building:
    h = math.radians(spec.heading_deg)
    n, e = spec.centre_ne_m
    fp = np.asarray(_rect(n, e, spec.length_m, spec.width_m, h).exterior.coords)
    base = float(np.min(dtm.at(np.append(fp[:, 0], n), np.append(fp[:, 1], e))))
    return Building(spec.kind, n, e, spec.length_m, spec.width_m, h, base, spec.eaves_m,
                    spec.ridge_m, spec.roof, variant, spec.name)  # fmt: skip


# --------------------------------------------------------------------------- terrain
def _base_terrain(spec: WorldSpec, nn: np.ndarray, ee: np.ndarray, rng) -> np.ndarray:
    t = spec.terrain
    h = t.east_slope_per_km * ee / 1000.0
    for r in t.ridges:
        h = h + r.rise_m * _smoothstep((nn - r.from_n_m) / (r.to_n_m - r.from_n_m))
    for hill in t.hills:
        d2 = (nn - hill.centre_ne_m[0]) ** 2 + (ee - hill.centre_ne_m[1]) ** 2
        h = h + hill.height_m * np.exp(-d2 / hill.radius_m**2)
    if t.noise_m > 0:
        for _ in range(4):
            lam = rng.uniform(*t.noise_wavelength_m)
            th, ph = rng.uniform(0, 2 * math.pi, 2)
            h = h + 0.5 * t.noise_m * np.sin(
                2 * math.pi * (nn * math.cos(th) + ee * math.sin(th)) / lam + ph
            )
    return h


def _dtm(spec: WorldSpec, rng, flats: list[Polygon], river: River | None) -> Dtm:
    n0, e0, n1, e1 = (*spec.bounds_ne_m[0], *spec.bounds_ne_m[1])
    cell = spec.terrain.cell_m
    ns = np.arange(n0, n1 + 0.5 * cell, cell)
    es = np.arange(e0, e1 + 0.5 * cell, cell)
    nn, ee = np.meshgrid(ns, es, indexing="ij")
    h = _base_terrain(spec, nn, ee, rng)
    pts = shapely.points(nn.ravel(), ee.ravel())
    # Landing grounds are level: blend each to its mean height.
    for poly in flats:
        inside = shapely.contains_xy(poly, nn.ravel(), ee.ravel()).reshape(nn.shape)
        level = (
            float(h[inside].mean())
            if inside.any()
            else float(
                _base_terrain(spec, np.array(poly.centroid.x), np.array(poly.centroid.y), rng)
            )
        )
        d = shapely.distance(poly, pts).reshape(nn.shape)
        w = 1.0 - _smoothstep(d / FLATTEN_BLEND_M)
        h = h + w * (level - h)
    if river is not None:
        d = shapely.distance(river.line, pts).reshape(nn.shape)
        half = river.width / 2
        h = h - river.depth * (1.0 - _smoothstep((d - half) / RIVER_BANK_M))
    dtm = Dtm(float(n0), float(e0), cell, h)
    return Dtm(dtm.n0, dtm.e0, cell, h - float(dtm.at(0.0, 0.0)))  # the datum: ground at home


# --------------------------------------------------------------------------- generation
def generate(spec: WorldSpec) -> World:
    rng = np.random.default_rng(spec.seed)
    n0, e0, n1, e1 = (*spec.bounds_ne_m[0], *spec.bounds_ne_m[1])
    bounds = box(n0, e0, n1, e1)

    river = None
    if spec.river is not None:
        r = spec.river
        line = LineString(chaikin(np.asarray(r.points_ne_m), 3, closed=False))
        river = River(r.id, r.name, r.width_m, r.depth_m, line)
    roads = [
        Road(r.id, r.name, r.kind, r.width_m, LineString(chaikin(np.asarray(r.points_ne_m), 2,
                                                                 closed=False)))
        for r in spec.roads
    ]  # fmt: skip
    woods = [
        Wood(w.id, w.name, w.kind, w.canopy_m,
             Polygon(chaikin(np.asarray(w.polygon_ne_m), 2, closed=True)).buffer(0))
        for w in spec.woods
    ]  # fmt: skip

    # Built-up land: village, farmyards, the airstrip.
    plots: list[Plot] = []
    built: list[Polygon] = []
    if spec.village is not None:
        v = spec.village
        outline = Polygon(chaikin(np.asarray(v.outline_ne_m), 2, closed=True)).buffer(0)
        plots.append(Plot(v.id, "settlement", "settlement", outline, name=v.name))
        built.append(outline)
        if v.green_ne_m:
            plots.append(Plot(f"{v.id}-green", "green", "green", Polygon(v.green_ne_m),
                              name=f"{v.name} green"))  # fmt: skip
    for f in spec.farms:
        yard = Polygon(f.yard_ne_m)
        plots.append(Plot(f.id, "yard", "yard", yard, name=f.name))
        built.append(yard)
    strip = None
    if spec.airstrip is not None:
        a = spec.airstrip
        strip = _rect(*a.centre_ne_m, a.length_m, a.width_m, math.radians(a.heading_deg))
        built.append(strip)

    # Landing grounds drawn as fields of their own.
    drawn = {z.id: Polygon(z.polygon_ne_m) for z in spec.zones if z.polygon_ne_m is not None}
    for z in spec.zones:
        if z.id in drawn:
            plots.append(Plot(f"field-{z.id.lower()}", "field", z.crop or "pasture", drawn[z.id],
                              z.id, z.name))  # fmt: skip

    river_corridor = river.line.buffer(river.width / 2 + 8.0) if river else Polygon()
    road_corridors = unary_union([r.line.buffer(r.width / 2 + VERGE_M) for r in roads])
    wood_union = unary_union([w.polygon for w in woods])
    excluded = unary_union([river_corridor, road_corridors, wood_union, *built, *drawn.values()])

    # Fields: a jittered grid (rotated with the lie of the land) less everything above. A
    # world without a field spec is open grassland (hedges keep their default sizes).
    fs = spec.fields or FieldSpec(crops={"rough": 1.0})
    zone_at = {z.id: Point(z.at_ne_m) for z in spec.zones if z.at_ne_m is not None}
    zone_crop = {z.id: z.crop for z in spec.zones}
    us = vs = np.zeros(0)
    grid = np.zeros((0, 0, 2))
    crops: list[str] = []
    weights = np.zeros(0)
    if spec.fields is not None:
        rot = math.radians(fs.rotation_deg)
        R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
        reach = 1.5 * max(n1 - n0, e1 - e0) / 2 + max(fs.size_m)

        def lines_(scale: float) -> np.ndarray:
            xs, x = [-reach], -reach
            while x < reach:
                x += scale * rng.uniform(*fs.size_m)
                xs.append(x)
            return np.asarray(xs)

        us, vs = lines_(0.75), lines_(1.0)  # fields run longer east-west than north-south
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        grid = np.stack([uu, vv], -1) + rng.uniform(-fs.jitter_m, fs.jitter_m, (*uu.shape, 2))
        grid = grid @ R.T
        crops = list(fs.crops)
        weights = np.asarray([fs.crops[c] for c in crops], float)
        weights /= weights.sum()
    k = 0
    for i in range(len(us) - 1):
        for j in range(len(vs) - 1):
            cell = Polygon([grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]])
            if not cell.intersects(bounds):
                continue
            crop = crops[int(rng.choice(len(crops), p=weights))]
            for piece in _parts(cell.intersection(bounds).difference(excluded)):
                part = cast(Polygon, piece.simplify(0.5))
                if part.area < FIELD_MIN_AREA_M2:
                    plots.append(Plot(f"rough-{k}", "rough", "rough", part))
                    k += 1
                    continue
                zone = next((z for z, p in zone_at.items() if part.contains(p)), None)
                cover = (zone_crop.get(zone) or crop) if zone else crop
                plots.append(Plot(f"field-{k}", "field", cover, part, zone))
                k += 1
    if strip is not None:
        a = spec.airstrip
        assert a is not None
        zone = next((z for z, p in zone_at.items() if strip.contains(p)), None)
        plots.append(Plot(a.id, "airstrip", "airstrip", strip, zone, a.name))

    # Terrain, with the landing grounds levelled.
    flats = [p.polygon for p in plots if p.zone is not None]
    dtm = _dtm(spec, rng, flats, river)

    # Hedges: the field grid's edges and the road verges, outside everything built or wooded.
    edges = []
    for i in range(len(us)):
        for j in range(len(vs)):
            if j + 1 < len(vs):
                edges.append(LineString([grid[i, j], grid[i, j + 1]]))
            if i + 1 < len(us):
                edges.append(LineString([grid[i, j], grid[i + 1, j]]))
    for r in roads:
        hedged = next(s for s in spec.roads if s.id == r.id).hedged
        if hedged:
            off = r.width / 2 + VERGE_M - 0.5
            edges += _lines(r.line.offset_curve(off)) + _lines(r.line.offset_curve(-off))
    edges += [LineString(p.exterior.coords) for p in drawn.values()]
    no_hedge = unary_union([river_corridor, road_corridors.buffer(-1.0), wood_union.buffer(2.0),
                            *(b.buffer(2.0) for b in built),
                            *(p.buffer(-1.0) for p in drawn.values())])  # fmt: skip
    hedge_lines = _lines(MultiLineString(edges).intersection(bounds).difference(no_hedge))
    # No trees on the landing grounds' hedges or in their approach funnels.
    treeless = unary_union(
        [p.polygon.exterior.buffer(20.0) for p in plots if p.zone]
        + [f for p in plots if p.zone for f in _funnels(p.polygon)]
    )
    hedges: list[Hedge] = []
    tree_rows: list[tuple[float, float, float, float, int, int]] = []
    hh = fs.hedge_height_m
    for line in hedge_lines:
        if line.length < 5.0:
            continue
        height = float(rng.uniform(*hh))
        for seg in _segments(line, HEDGE_SEGMENT_M):
            hedges.append(Hedge(seg, height, fs.hedge_width_m))
        # Hedgerow trees, except on the trimmed hedges around the landing grounds.
        d = float(rng.uniform(0.2, 1.0)) * fs.hedge_tree_spacing_m
        while d < line.length:
            p = line.interpolate(d)
            if rng.uniform() < 0.7 and not treeless.contains(p):
                ht = float(rng.uniform(*fs.hedge_tree_height_m))
                tree_rows.append((p.x, p.y, ht, ht * rng.uniform(0.28, 0.36), TREE_BROADLEAF, -1))
            d += float(rng.uniform(0.6, 1.4)) * fs.hedge_tree_spacing_m

    # Willows along the river banks.
    if river is not None:
        for side in (-1.0, 1.0):
            for bank in _lines(river.line.offset_curve(side * (river.width / 2 + 4.0))):
                d = float(rng.uniform(5.0, 40.0))
                while d < bank.length:
                    p = bank.interpolate(d)
                    if (
                        rng.uniform() < 0.6
                        and not road_corridors.contains(p)
                        and not treeless.contains(p)
                    ):
                        ht = float(rng.uniform(8.0, 12.0))
                        tree_rows.append((p.x, p.y, ht, rng.uniform(3.5, 5.0), TREE_WILLOW, -1))
                    d += float(rng.uniform(25.0, 60.0))

    # Woods: trees on a jittered grid filling the outline (the canopy is solid to the sim).
    for wi, w in enumerate(woods):
        minx, miny, maxx, maxy = w.polygon.bounds
        s = WOOD_TREE_SPACING_M
        gx, gy = np.meshgrid(np.arange(minx, maxx, s), np.arange(miny, maxy, s), indexing="ij")
        gx = gx.ravel() + rng.uniform(-0.35 * s, 0.35 * s, gx.size)
        gy = gy.ravel() + rng.uniform(-0.35 * s, 0.35 * s, gy.size)
        keep = shapely.contains_xy(w.polygon.buffer(-1.0), gx, gy)
        conifer = w.kind == "conifer"
        for x, y in zip(gx[keep], gy[keep], strict=True):
            ht = w.canopy * float(rng.uniform(0.82, 1.0))
            r = float(rng.uniform(2.6, 3.3) if conifer else rng.uniform(3.6, 4.6))
            tree_rows.append((x, y, ht, r, TREE_CONIFER if conifer else TREE_BROADLEAF, wi))

    # Buildings: the authored ones, then houses along the village streets.
    buildings: list[Building] = []
    for f in spec.farms:
        buildings += [_building(b, dtm, int(rng.integers(0, 8))) for b in f.buildings]
    if spec.airstrip is not None:
        buildings += [_building(b, dtm, int(rng.integers(0, 8))) for b in spec.airstrip.buildings]
    if spec.village is not None:
        v = spec.village
        village = next(p.polygon for p in plots if p.kind == "settlement")
        buildings += [_building(b, dtm, int(rng.integers(0, 8))) for b in v.buildings]
        taken = [b.footprint().buffer(8.0 if b.kind in ("nave", "tower") else 3.0)
                 for b in buildings]  # fmt: skip
        if v.green_ne_m:
            taken.append(Polygon(v.green_ne_m))
        streets = [r for r in roads if r.id in v.streets]
        road_keep = unary_union([r.line.buffer(r.width / 2 + 2.0) for r in roads])
        for st in streets:
            for side in (-1.0, 1.0):
                d = float(rng.uniform(4.0, 12.0))
                while d < st.line.length - 4.0:
                    p0, p1 = st.line.interpolate(max(d - 1, 0)), st.line.interpolate(d + 1)
                    t = np.array([p1.x - p0.x, p1.y - p0.y])
                    t /= max(np.linalg.norm(t), 1e-9)
                    nrm = side * np.array([-t[1], t[0]])
                    length, width = float(rng.uniform(9.0, 14.0)), float(rng.uniform(6.5, 8.5))
                    back = st.width / 2 + float(rng.uniform(5.0, 9.0)) + width / 2
                    p = st.line.interpolate(d)
                    c = np.array([p.x, p.y]) + back * nrm
                    heading = math.atan2(t[1], t[0])
                    fp = _rect(c[0], c[1], length, width, heading)
                    ok = (village.contains(fp) and not fp.intersects(road_keep)
                          and not any(fp.intersects(q) for q in taken))  # fmt: skip
                    if ok:
                        bungalow = rng.uniform() < 0.15
                        eaves = 2.7 if bungalow else float(rng.uniform(4.8, 5.6))
                        pitch = math.radians(float(rng.uniform(35.0, 45.0)))
                        ridge = eaves + 0.5 * width * math.tan(pitch)
                        corners = np.asarray(fp.exterior.coords)
                        base = float(np.min(dtm.at(corners[:, 0], corners[:, 1])))
                        buildings.append(Building("house", float(c[0]), float(c[1]), length,
                                                  width, heading, base, eaves, ridge, "gable",
                                                  int(rng.integers(0, 8))))  # fmt: skip
                        taken.append(fp.buffer(3.0))
                        # A garden tree or two behind the house.
                        for _ in range(int(rng.integers(0, 3))):
                            q = (
                                c
                                + nrm * float(rng.uniform(8.0, 16.0))
                                + t * float(rng.uniform(-6.0, 6.0))
                            )
                            if (
                                village.contains(Point(q))
                                and not road_keep.contains(Point(q))
                                and not treeless.contains(Point(q))
                            ):
                                ht = float(rng.uniform(6.0, 11.0))
                                tree_rows.append((q[0], q[1], ht, ht * 0.32, TREE_BROADLEAF, -1))
                    d += float(rng.uniform(*v.house_spacing_m))

    structures = []
    for s in spec.structures:
        n, e = s.point_ne_m
        structures.append(Structure(s.id, s.name, s.kind, n, e, float(dtm.at(n, e)), s.height_m,
                                    s.radius_m))  # fmt: skip

    t = np.asarray(tree_rows, float).reshape(-1, 6)
    trees = Trees(t[:, 0], t[:, 1], dtm.at(t[:, 0], t[:, 1]), t[:, 2], t[:, 3],
                  t[:, 4].astype(np.int8), t[:, 5].astype(np.int16))  # fmt: skip
    return World(spec, dtm, river, roads, woods, plots, hedges, trees, buildings, structures)
