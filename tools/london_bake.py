# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
"""Bake London (docs/08): the City and Bankside, from Environment Agency LiDAR and
OpenStreetMap, into a world the simulator flies and its site.

Writes ``configs/worlds/london.world.npz`` and ``london.yaml``, and the operations block of
``configs/sites/london.yaml``. Run ``tools/london_fetch.py`` first (the data lives in the
git-ignored ``data/london/``). The world is a database derived from OpenStreetMap and stays
under the ODbL (``configs/worlds/LICENCE-london.md``).

- **Frame.** Local metres, true north, from the box's centre. OSM's WGS84 goes there directly
  (a tangent plane: millimetres over the box); the LiDAR's National Grid goes there through
  OSGB36 (Helmert) and the Transverse Mercator, fitted as a quadratic in (n, e). The Helmert
  step is good to a few metres, so the LiDAR is then slid onto OSM's footprints, by
  cross-correlation, to the offset where they agree best.
- **Ground.** The LiDAR DTM on a 2 m grid, the Thames and docks flattened to one level each
  (their beds 1.5 m below it, away from the banks).
- **Obstacles.** The LiDAR surface above that ground on a 1 m grid (the highest of the four
  LiDAR cells round each), so the simulator hits what was measured: buildings, trees,
  bridges, cranes and boats on the survey's day.
- **Buildings.** Each OSM footprint or building part (a building divided into parts is drawn
  as its parts), up to a height checked against the LiDAR: OSM's own height where the
  two agree, the LiDAR's otherwise. Roof shapes, colours and materials from OSM's tags.
- **The rest.** Roads by class, parks, squares and water as land cover, trees at the LiDAR
  canopy's tops, a landing ground at Potters Fields Park and the landmarks as places.

Usage::

    uv run python -I tools/london_bake.py
"""

from __future__ import annotations

import json
import math
import re
import zlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import shapely
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage, signal
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import linemerge, polygonize, unary_union

from aeroadapt.world.baked import ROOFS, save_baked
from aeroadapt.world.generate import Dtm
from aeroadapt.world.raster import QUANTUM_M, Obstacles

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "london"
NAME = "london"

# The LiDAR box on the National Grid (tools/london_fetch.py) and its pixels.
BOX_E0, BOX_N1 = 531_300.0, 181_900.0
# The world: local metres about the origin, true north, inside the box.
LAT0, LON0 = 51.51070, -0.08925  # the box's centre (E 532,694, N 180,816)
BOUNDS = (-1040.0, -1360.0, 1040.0, 1360.0)  # n0, e0, n1, e1
DTM_CELL = 2.0
OBS_CELL = 1.0
OBS_FLOOR_M = 1.0  # measured heights below this are the ground's own roughness
STOREY_M = 3.3

# --------------------------------------------------------------------------- geodesy
A_AIRY, B_AIRY = 6377563.396, 6356256.909
A_WGS, B_WGS = 6378137.0, 6356752.3142


def _tm_forward(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OSGB36 latitude and longitude (rad) → National Grid easting and northing (OS formulae)."""
    a, b, f0 = A_AIRY, B_AIRY, 0.9996012717
    lat0, lon0, n0, e0 = math.radians(49), math.radians(-2), -100_000.0, 400_000.0
    e2 = 1 - b * b / (a * a)
    n = (a - b) / (a + b)
    s, c, t = np.sin(lat), np.cos(lat), np.tan(lat)
    nu = a * f0 / np.sqrt(1 - e2 * s * s)
    rho = a * f0 * (1 - e2) / (1 - e2 * s * s) ** 1.5
    eta2 = nu / rho - 1
    dl, dp = lat - lat0, lat + lat0
    m = (
        b
        * f0
        * (
            (1 + n + 1.25 * n**2 + 1.25 * n**3) * dl
            - (3 * n + 3 * n**2 + 21 / 8 * n**3) * np.sin(dl) * np.cos(dp)
            + (15 / 8 * n**2 + 15 / 8 * n**3) * np.sin(2 * dl) * np.cos(2 * dp)
            - 35 / 24 * n**3 * np.sin(3 * dl) * np.cos(3 * dp)
        )
    )
    i = m + n0
    ii = nu / 2 * s * c
    iii = nu / 24 * s * c**3 * (5 - t * t + 9 * eta2)
    iiia = nu / 720 * s * c**5 * (61 - 58 * t * t + t**4)
    iv = nu * c
    v = nu / 6 * c**3 * (nu / rho - t * t)
    vi = nu / 120 * c**5 * (5 - 18 * t * t + t**4 + 14 * eta2 - 58 * t * t * eta2)
    dlon = lon - lon0
    return (
        e0 + iv * dlon + v * dlon**3 + vi * dlon**5,
        i + ii * dlon**2 + iii * dlon**4 + iiia * dlon**6,
    )


def _wgs84_to_osgb36(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The OS's 7-parameter Helmert (good to a few metres; the residual is measured below)."""
    e1 = 1 - B_WGS**2 / A_WGS**2
    s = np.sin(lat)
    nu = A_WGS / np.sqrt(1 - e1 * s * s)
    x, y, z = nu * np.cos(lat) * np.cos(lon), nu * np.cos(lat) * np.sin(lon), (1 - e1) * nu * s
    tx, ty, tz, sc = -446.448, 125.157, -542.060, 20.4894e-6
    rx, ry, rz = (math.radians(v / 3600) for v in (-0.1502, -0.2470, -0.8421))
    x2 = tx + (1 + sc) * x - rz * y + ry * z
    y2 = ty + rz * x + (1 + sc) * y - rx * z
    z2 = tz - ry * x + rx * y + (1 + sc) * z
    e2 = 1 - B_AIRY**2 / A_AIRY**2
    p = np.hypot(x2, y2)
    la = np.arctan2(z2, p * (1 - e2))
    for _ in range(8):
        la = np.arctan2(z2 + e2 * A_AIRY / np.sqrt(1 - e2 * np.sin(la) ** 2) * np.sin(la), p)
    return la, np.arctan2(y2, x2)


def _radii() -> tuple[float, float]:
    e2 = 1 - B_WGS**2 / A_WGS**2
    s = math.sin(math.radians(LAT0))
    return A_WGS * (1 - e2) / (1 - e2 * s * s) ** 1.5, A_WGS / math.sqrt(1 - e2 * s * s)


def local_from_wgs(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    m, nn = _radii()
    return (
        np.radians(np.asarray(lat) - LAT0) * m,
        np.radians(np.asarray(lon) - LON0) * nn * math.cos(math.radians(LAT0)),
    )


def wgs_from_local(n, e) -> tuple[np.ndarray, np.ndarray]:
    m, nn = _radii()
    return (
        LAT0 + np.degrees(np.asarray(n) / m),
        LON0 + np.degrees(np.asarray(e) / (nn * math.cos(math.radians(LAT0)))),
    )


class Grid:
    """Local (n, e) → National Grid (E, N), a quadratic fitted to the exact chain, plus the
    measured offset of the LiDAR from OSM."""

    def __init__(self) -> None:
        g = np.linspace(-1700, 1700, 15)
        nn, ee = (a.ravel() for a in np.meshgrid(g, g, indexing="ij"))
        la, lo = wgs_from_local(nn, ee)
        e_, n_ = _tm_forward(*_wgs84_to_osgb36(np.radians(la), np.radians(lo)))
        x = self._features(nn, ee)
        self.ce = np.linalg.lstsq(x, e_, rcond=None)[0]
        self.cn = np.linalg.lstsq(x, n_, rcond=None)[0]
        self.fit_m = float(np.hypot(x @ self.ce - e_, x @ self.cn - n_).max())
        self.shift = np.zeros(2)  # (dE, dN) added to reach the LiDAR

    @staticmethod
    def _features(n, e) -> np.ndarray:
        n, e = np.asarray(n, float).ravel(), np.asarray(e, float).ravel()
        return np.c_[np.ones_like(n), n, e, n * n, n * e, e * e]

    def bng(self, n, e) -> tuple[np.ndarray, np.ndarray]:
        shape = np.shape(n)
        x = self._features(n, e)
        return (x @ self.ce + self.shift[0]).reshape(shape), (x @ self.cn + self.shift[1]).reshape(
            shape
        )

    @property
    def rotation_deg(self) -> float:
        return math.degrees(math.atan2(self.ce[1], self.cn[1]))


def _px(E, N) -> tuple[np.ndarray, np.ndarray]:
    """National Grid → LiDAR pixel coordinates (column, row; pixel centres at integers)."""
    return np.asarray(E) - BOX_E0 - 0.5, BOX_N1 - np.asarray(N) - 0.5


def bilinear(img: np.ndarray, E, N) -> np.ndarray:
    x, y = _px(E, N)
    rows, cols = img.shape
    x = np.clip(x, 0, cols - 1.001)
    y = np.clip(y, 0, rows - 1.001)
    j, i = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - j, y - i
    return (
        img[i, j] * (1 - fx) * (1 - fy)
        + img[i, j + 1] * fx * (1 - fy)
        + img[i + 1, j] * (1 - fx) * fy
        + img[i + 1, j + 1] * fx * fy
    )


def highest(img: np.ndarray, E, N) -> np.ndarray:
    """The highest of the four LiDAR cells round each point (conservative)."""
    x, y = _px(E, N)
    rows, cols = img.shape
    j = np.clip(np.floor(x).astype(int), 0, cols - 2)
    i = np.clip(np.floor(y).astype(int), 0, rows - 2)
    return np.maximum.reduce([img[i, j], img[i, j + 1], img[i + 1, j], img[i + 1, j + 1]])


# --------------------------------------------------------------------------- OSM
def _ring(geom: list[dict]) -> np.ndarray:
    lat = np.array([p["lat"] for p in geom])
    lon = np.array([p["lon"] for p in geom])
    n, e = local_from_wgs(lat, lon)
    return np.c_[n, e]


def _valid(p) -> list[Polygon]:
    p = shapely.make_valid(p) if not p.is_valid else p
    if isinstance(p, Polygon):
        return [p] if p.area > 0 else []
    return [g for g in getattr(p, "geoms", []) if isinstance(g, Polygon) and g.area > 0]


def osm_areas(doc: dict) -> list[tuple[dict, Polygon, str]]:
    """Every closed way and multipolygon relation as (tags, polygon, osm id)."""
    out = []
    for el in doc["elements"]:
        tags = el.get("tags", {})
        if el["type"] == "way" and "geometry" in el:
            pts = _ring(el["geometry"])
            if len(pts) >= 4 and np.allclose(pts[0], pts[-1]):
                out += [(tags, p, f"w{el['id']}") for p in _valid(Polygon(pts))]
        elif el["type"] == "relation" and tags.get("type") in ("multipolygon", "building"):
            outer = [LineString(_ring(m["geometry"])) for m in el.get("members", [])
                     if m.get("role") == "outer" and m.get("geometry")]  # fmt: skip
            inner = [LineString(_ring(m["geometry"])) for m in el.get("members", [])
                     if m.get("role") == "inner" and m.get("geometry")]  # fmt: skip
            if not outer:
                continue
            shell = unary_union(list(polygonize(linemerge(outer))))
            holes = unary_union(list(polygonize(linemerge(inner)))) if inner else None
            shape = shell.difference(holes) if holes is not None else shell
            out += [(tags, p, f"r{el['id']}") for p in _valid(shape)]
    return out


def osm_lines(doc: dict) -> list[tuple[dict, LineString, str]]:
    out = []
    for el in doc["elements"]:
        if el["type"] == "way" and "geometry" in el and len(el["geometry"]) >= 2:
            out.append((el.get("tags", {}), LineString(_ring(el["geometry"])), f"w{el['id']}"))
    return out


_NUM = re.compile(r"^\s*~?([-\d.]+)\s*(m|ft|')?\s*$")


def _metres(v: str | None) -> float | None:
    if not v:
        return None
    m = _NUM.match(v.replace(",", "."))
    if not m:
        return None
    try:
        x = float(m.group(1))
    except ValueError:
        return None
    return x * 0.3048 if m.group(2) in ("ft", "'") else x


def _int(v: str | None) -> float | None:
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


_POINTS = {"N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5, "SE": 135,
           "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270,
           "WNW": 292.5, "NW": 315, "NNW": 337.5}  # fmt: skip


def _compass(v: str | None) -> float | None:
    """OSM's roof:direction (degrees or a compass point) in degrees, or None."""
    if not v:
        return None
    v = v.strip().upper()
    if v in _POINTS:
        return float(_POINTS[v])
    try:
        return float(v) % 360.0
    except ValueError:
        return None


# --------------------------------------------------------------------------- colours
CSS = {
    "white": "#f2f0ea", "black": "#2b2b2b", "grey": "#9a9a98", "gray": "#9a9a98",
    "lightgrey": "#c8c7c2", "lightgray": "#c8c7c2", "darkgrey": "#5f6064", "darkgray": "#5f6064",
    "silver": "#bfc2c4", "brown": "#7b5640", "red": "#9c3f32", "darkred": "#7a2f27",
    "maroon": "#6e2c25", "orange": "#c06a3a", "yellow": "#d6c36a", "beige": "#d9cdb0",
    "tan": "#c8b08a", "cream": "#e8dfc6", "ivory": "#ece7d6", "wheat": "#dccba0",
    "sandybrown": "#c99a6b", "sienna": "#8f5236", "blue": "#4a6a96", "lightblue": "#9db7cf",
    "navy": "#2e3d5c", "teal": "#3f7a78", "green": "#4f7a49", "darkgreen": "#33552f",
    "olive": "#7a7a45", "gold": "#c9a94a", "pink": "#c99a9a", "purple": "#6a4a78",
}  # fmt: skip
WALL_MATERIAL = {
    "glass": "#8ea3b5", "brick": "#9b5a45", "stone": "#d8d0bc", "concrete": "#b3b0a8",
    "metal": "#a5abb1", "steel": "#a5abb1", "plaster": "#e6e0d2", "render": "#e6e0d2",
    "wood": "#8a6a4a", "sandstone": "#d6c196", "limestone": "#e1d9c4", "granite": "#9a9894",
    "timber_framing": "#d9cfb6", "cement_block": "#b3b0a8", "tiles": "#a8645a",
}  # fmt: skip
ROOF_MATERIAL = {
    "slate": "#4a5058", "roof_tiles": "#9b4d38", "tile": "#9b4d38", "tiles": "#9b4d38",
    "metal": "#8c9298", "copper": "#6f9c88", "lead": "#7d8389", "glass": "#8ea3b5",
    "concrete": "#9a9890", "gravel": "#9a958a", "asphalt": "#5a5c60", "tar_paper": "#5a5c60",
    "grass": "#6f8f4f", "plants": "#6f8f4f", "stone": "#bdb6a4", "zinc": "#8c9298",
}  # fmt: skip
WALL_BY_KIND = {
    "tower": ["#8ea3b5", "#9fb0bf", "#7f93a6", "#a9b4bd"],
    "modern": ["#c9c4b8", "#a9a49a", "#d6d0c2", "#8f949a", "#c2b49a", "#b5aea2", "#9b9890",
               "#d9d3c8"],
    "old": ["#9b5a45", "#c4ad85", "#a8694f", "#d6cdb6", "#b08d6a", "#8a4f3e", "#cbbf9f", "#a3765a"],
    "landmark": ["#ddd5c1", "#d1c8b2", "#e3dccb", "#cfc5ad"],
    "nave": ["#ddd5c1", "#d1c8b2", "#e3dccb", "#cfc5ad"],
    "industrial": ["#8b8f93", "#9a9387", "#7f8488", "#a7a29a"],
    "canopy": ["#9aa0a6", "#8b9196", "#a5a9ad", "#7f8589"],
    "bridge": ["#8d9093", "#8d9093", "#8d9093", "#8d9093"],
}  # fmt: skip
ROOF_BY_KIND = {"old": ["#4a5058", "#9b4d38", "#5a5e66", "#8a4632"]}
ROOF_FLAT = ["#9a988f", "#8f8d86", "#a5a39b", "#86878a"]


def _colour(v: str | None) -> str | None:
    if not v:
        return None
    v = v.strip().lower().replace(" ", "")
    if re.fullmatch(r"#[0-9a-f]{6}", v):
        h = v
    elif re.fullmatch(r"#[0-9a-f]{3}", v):
        h = "#" + "".join(ch * 2 for ch in v[1:])
    else:
        h = CSS.get(v)
    if h is None:
        return None
    # A little towards grey: a tagged colour is the paint's, not the weathered building's.
    rgb = np.array([int(h[k : k + 2], 16) for k in (1, 3, 5)], float)
    rgb = 0.85 * rgb + 0.15 * 150
    return "#" + "".join(f"{round(x):02x}" for x in rgb)


# --------------------------------------------------------------------------- buildings
CHURCH = {"church", "cathedral", "chapel"}
STONE = {"castle", "civic", "public", "museum", "government", "townhall", "palace"}
HOMES = {"apartments", "residential", "house", "terrace", "detached", "semidetached_house",
         "dormitory", "hotel"}  # fmt: skip
INDUSTRY = {"industrial", "warehouse", "service", "garage", "garages", "shed", "train_station",
            "transportation", "construction", "parking"}  # fmt: skip
PITCHED = {"gabled", "hipped", "half-hipped", "mansard", "gambrel", "saltbox", "skillion",
           "round", "side_hipped", "double_saltbox", "quadruple_saltbox"}  # fmt: skip


def _kind(tags: dict, height: float, roof: str) -> str:
    b = tags.get("building") or tags.get("building:part") or "yes"
    if b in ("roof",) or tags.get("building:part") == "roof":
        return "canopy"
    if b in CHURCH or tags.get("amenity") == "place_of_worship":
        return "tower" if roof == "spire" else "nave"
    if b in STONE or "heritage" in tags or "historic" in tags:
        return "landmark"
    if b in INDUSTRY:
        return "industrial"
    glass = tags.get("building:material") == "glass"
    if (height >= 50 and roof == "flat") or (height >= 30 and glass):
        return "tower"
    if b in HOMES:
        return "old" if height < 22 else "modern"
    return "modern"


class Lidar:
    def __init__(self, grid: Grid, dsm: np.ndarray, datum: float) -> None:
        self.grid, self.dsm, self.datum = grid, dsm, datum

    def inside(self, poly: Polygon, erode: float = 0.75) -> np.ndarray:
        """LiDAR surface heights (above the datum) on a 1 m lattice inside a polygon."""
        core = poly.buffer(-erode)
        if core.is_empty or core.area < 3:
            core = poly
        n0, e0, n1, e1 = core.bounds
        nn, ee = np.meshgrid(
            np.arange(n0 + 0.5, n1, 1.0), np.arange(e0 + 0.5, e1, 1.0), indexing="ij"
        )
        k = shapely.contains_xy(core, nn, ee)
        if k.sum() < 3:
            c = core.representative_point()
            nn, ee, k = np.array([c.x]), np.array([c.y]), np.array([True])
        return bilinear(self.dsm, *self.grid.bng(nn[k], ee[k])) - self.datum


def buildings(areas, lidar: Lidar, dtm: Dtm, bounds_poly: Polygon) -> tuple[list, list, dict]:
    """Building boxes (save_baked's array rows), their meta, and counts for the report."""
    rows, metas = [], []
    stats = {"osm_height": 0, "lidar_height": 0, "levels": 0, "guessed": 0, "parts": 0,
             "outlines_dropped": 0, "gable_boxes": 0}  # fmt: skip
    parts = [(t, p, i) for t, p, i in areas if "building:part" in t]
    whole = [(t, p, i) for t, p, i in areas if "building" in t and "building:part" not in t]
    tree = shapely.STRtree([p for _, p, _ in parts]) if parts else None

    def keep(tags: dict) -> bool:
        if tags.get("building") in ("no",) or tags.get("location") == "underground":
            return False
        layer = _int(tags.get("layer"))
        return not (layer is not None and layer < 0 and "building:part" not in tags)

    # A part takes its building's character (a cathedral, stone, its colours) unless it has
    # its own: the building that contains it, the smallest if several do.
    outlines = shapely.STRtree([p for _, p, _ in whole]) if whole else None
    INHERIT = ("building", "amenity", "heritage", "historic", "building:material",
               "building:colour", "roof:colour", "roof:material")  # fmt: skip

    def parent(poly: Polygon) -> dict:
        if outlines is None:
            return {}
        hits = outlines.query(poly.representative_point(), predicate="intersects")
        if not len(hits):
            return {}
        k = min(hits, key=lambda k: whole[k][1].area)
        return {key: whole[k][0][key] for key in INHERIT if key in whole[k][0]}

    todo = []
    for tags, poly, oid in whole:
        if not keep(tags):
            continue
        if tree is not None:
            hits = tree.query(poly, predicate="intersects")
            if len(hits):
                # An outline with parts is drawn as its parts (OSM's Simple 3D Buildings).
                covered = unary_union([parts[k][1] for k in hits]).intersection(poly).area
                if covered > 0.3 * poly.area:
                    stats["outlines_dropped"] += 1
                    continue
        todo.append((tags, poly, oid))
    for tags, poly, oid in parts:
        if keep(tags) and tags.get("building:part") != "no":
            todo.append(({**parent(poly), **tags}, poly, oid))
            stats["parts"] += 1

    for tags, poly, oid in todo:
        for p in _valid(poly.intersection(bounds_poly)):
            p = p.simplify(0.25, preserve_topology=True)
            if p.area < 4.0:
                continue
            ring = np.asarray(p.exterior.coords)[:-1]
            base = float(np.min(dtm.at(ring[:, 0], ring[:, 1])))
            meas = lidar.inside(p)
            meas_top = float(np.percentile(meas, 90)) if len(meas) >= 3 else None
            near = lidar.inside(p.buffer(3.0), 0.0)
            meas_max = float(np.percentile(near, 99.5)) if len(near) >= 3 else None
            osm_h = _metres(tags.get("height"))
            levels = _int(tags.get("building:levels"))
            bottom = _metres(tags.get("min_height"))
            if bottom is None and _int(tags.get("building:min_level")) is not None:
                bottom = float(_int(tags.get("building:min_level"))) * STOREY_M  # type: ignore[arg-type]
            bottom = max(bottom or 0.0, 0.0)
            # OSM's height is its top: it agrees with the LiDAR if it is no higher than the
            # highest the LiDAR saw there (a tapering top is a sliver of cells) and, for a
            # whole building with a flat roof, no lower than the roof's typical height (else
            # OSM is out of date). A part may stand under another (a bridge's deck under its
            # walkways, a cathedral's floor under its dome): the LiDAR sees only the top one.
            tol = max(4.0, 0.15 * osm_h) if osm_h is not None else 0.0
            sloped = tags.get("roof:shape", "flat") != "flat" or "building:part" in tags
            if (
                osm_h is not None
                and meas_top is not None
                and meas_max is not None
                and base + osm_h <= meas_max + tol
                and (sloped or base + osm_h >= meas_top - tol)
            ):
                top, src = base + osm_h, "osm_height"
            elif meas_top is not None and meas_top - base > 2.0:
                top, src = meas_top, "lidar_height"
            elif osm_h is not None:
                top, src = base + osm_h, "osm_height"
            elif levels is not None:
                top, src = base + levels * STOREY_M + 1.0, "levels"
            else:
                top, src = base + 9.0, "guessed"
            stats[src] += 1
            if top - base < bottom + 1.0:
                continue
            shape = tags.get("roof:shape", "flat")
            roof_h = _metres(tags.get("roof:height"))
            if roof_h is None and _int(tags.get("roof:levels")) is not None:
                roof_h = float(_int(tags.get("roof:levels"))) * 2.8  # type: ignore[arg-type]
            rect = p.minimum_rotated_rectangle
            rc = np.asarray(rect.exterior.coords)[:4]
            s1, s2 = rc[1] - rc[0], rc[2] - rc[1]
            if np.hypot(*s1) >= np.hypot(*s2):
                length, width, axis = float(np.hypot(*s1)), float(np.hypot(*s2)), s1
            else:
                length, width, axis = float(np.hypot(*s2)), float(np.hypot(*s1)), s2
            heading = math.atan2(axis[1], axis[0])
            cn, ce = rect.centroid.x, rect.centroid.y
            height = top - base
            outline = True
            if shape in ("dome", "onion"):
                roof = "dome"
                eaves = top - (roof_h or 0.5 * min(length, width))
            elif shape in ("pyramidal", "cone", "spire", "pyramid"):
                roof = "spire"
                eaves = top - (roof_h or 0.6 * min(length, width))
            elif shape == "skillion" and _compass(tags.get("roof:direction")) is not None:
                roof = "skillion"
                eaves = top - (roof_h or min(0.4 * width, 4.0))
            elif shape in PITCHED:
                rh = roof_h or min(0.45 * width, 6.0)
                if p.area / max(rect.area, 1e-9) > 0.88 and len(ring) <= 8 and width > 3:
                    roof, eaves, outline = "gable", top - rh, False
                    stats["gable_boxes"] += 1
                else:
                    roof, eaves = "flat", top - 0.5 * rh
                    top = eaves
            else:
                roof, eaves = "flat", top
            eaves = max(eaves, base + bottom + 1.0)
            kind = _kind(tags, height, roof)
            variant = zlib.crc32(oid.encode()) % 8
            wall = (
                _colour(tags.get("building:colour"))
                or WALL_MATERIAL.get(tags.get("building:material", ""))
                or WALL_BY_KIND[kind][variant % len(WALL_BY_KIND[kind])]
            )
            roofc = (
                _colour(tags.get("roof:colour"))
                or ROOF_MATERIAL.get(tags.get("roof:material", ""))
                or (
                    ROOF_BY_KIND.get(kind, ROOF_FLAT)[variant % 4]
                    if roof != "flat"
                    else ROOF_FLAT[variant % 4]
                )
            )
            rows.append([cn, ce, length, width, heading, base, eaves - base, top - base,
                         ROOFS.index(roof), variant])  # fmt: skip
            # Names for the landmarks and the towers only: a real city names everything.
            name = tags.get("name")
            if name is not None and height < 80 and OSM_NAMES.get(name, name) not in LANDMARKS:
                name = None
            m = {"kind": kind, "name": name, "colours": [wall, roofc]}
            if roof == "skillion":
                m["slope"] = round(math.radians(_compass(tags.get("roof:direction"))), 4)  # type: ignore[arg-type]
            if outline:
                m["outline"] = np.round(ring, 2).tolist()
                m["holes"] = [np.round(np.asarray(h.coords)[:-1], 2).tolist() for h in p.interiors]
                m["bottom"] = round(bottom, 2)
            metas.append(m)
    return rows, metas, stats


# --------------------------------------------------------------------------- land cover, roads
GREEN = {("leisure", "park"): "park", ("leisure", "garden"): "garden",
         ("leisure", "common"): "park",
         ("leisure", "recreation_ground"): "park", ("landuse", "grass"): "park",
         ("landuse", "recreation_ground"): "park", ("landuse", "cemetery"): "park",
         ("landuse", "village_green"): "park", ("amenity", "grave_yard"): "park",
         ("landuse", "flowerbed"): "garden", ("leisure", "nature_reserve"): "park"}  # fmt: skip
PAVED = {("amenity", "parking"): "paved", ("place", "square"): "paved",
         ("highway", "pedestrian"): "paved", ("landuse", "railway"): "gravel",
         ("landuse", "construction"): "yard", ("railway", "platform"): "paved"}  # fmt: skip
ROAD_KIND = {"motorway": "road", "trunk": "road", "primary": "road", "secondary": "road",
             "tertiary": "road", "motorway_link": "road", "trunk_link": "road",
             "primary_link": "road", "secondary_link": "road", "tertiary_link": "road",
             "residential": "street", "unclassified": "street", "living_street": "street",
             "service": "street", "pedestrian": "lane"}  # fmt: skip
ROAD_WIDTH = {"motorway": 16, "trunk": 14, "primary": 12, "secondary": 10, "tertiary": 9,
              "residential": 7, "unclassified": 7, "living_street": 6, "service": 5,
              "pedestrian": 6}  # fmt: skip


def _water(tags: dict) -> bool:
    return (
        tags.get("natural") == "water"
        or tags.get("waterway") in ("riverbank", "dock")
        or (
            "water" in tags
            and tags.get("natural") in (None, "water")
            and tags.get("amenity") is None
        )
    )


def _polys(g) -> Iterable[Polygon]:
    if isinstance(g, Polygon):
        yield g
    elif isinstance(g, MultiPolygon):
        yield from g.geoms
    else:
        for x in getattr(g, "geoms", []):
            yield from _polys(x)


def _plot(pid: str, kind: str, cover: str, p: Polygon, zone=None, name=None) -> dict:
    return {
        "id": pid,
        "kind": kind,
        "cover": cover,
        "zone": zone,
        "name": name,
        "outer": np.round(np.asarray(p.exterior.coords)[:-1], 2).tolist(),
        "holes": [np.round(np.asarray(h.coords)[:-1], 2).tolist() for h in p.interiors],
    }


# --------------------------------------------------------------------------- the bake
def align(grid: Grid, areas, dsm: np.ndarray, dtm_img: np.ndarray) -> None:
    """Slide the LiDAR onto OSM's footprints: the offset that best matches buildings (OSM)
    with what stands 4 m clear of the ground (LiDAR), by cross-correlation at 0.5 m."""
    sc = 2
    n1, e0 = 800.0, -800.0  # the central 1.6 km square, dense with buildings
    n0 = -800.0
    size = int((n1 - n0) * sc)
    img = Image.new("1", (size, size), 0)
    dr = ImageDraw.Draw(img)
    for tags, p, _ in areas:
        if "building" in tags and "building:part" not in tags:
            ring = np.asarray(p.exterior.coords)
            dr.polygon(
                list(zip((ring[:, 1] - e0) * sc, (n1 - ring[:, 0]) * sc, strict=True)), fill=1
            )
    osm = np.asarray(img, np.float32)
    ii, jj = np.mgrid[0:size, 0:size]
    nn, ee = n1 - (ii + 0.5) / sc, e0 + (jj + 0.5) / sc
    E, N = grid.bng(nn, ee)
    lid = ((bilinear(dsm, E, N) - bilinear(dtm_img, E, N)) > 4.0).astype(np.float32)
    corr = signal.fftconvolve(lid, osm[::-1, ::-1], mode="same")
    i, j = np.unravel_index(np.argmax(corr), corr.shape)
    di, dj = i - size // 2, j - size // 2  # the LiDAR's pattern sits (di, dj) px from OSM's

    # Sub-pixel: a parabola through the peak and its neighbours.
    def frac(a, b, c):
        d = a - 2 * b + c
        return 0.0 if d == 0 else 0.5 * (a - c) / d

    fi = frac(corr[i - 1, j], corr[i, j], corr[i + 1, j])
    fj = frac(corr[i, j - 1], corr[i, j], corr[i, j + 1])
    # LiDAR at OSM's point p is found at p + offset: rows run south, columns east.
    dn, de = -(di + fi) / sc, (dj + fj) / sc
    # Move the sampling by the matching National Grid offset (rotation is under 2°: n ≈ N).
    grid.shift = np.array([de, dn])
    print(f"alignment: LiDAR offset from OSM dE {de:+.2f} m, dN {dn:+.2f} m")


def bake() -> dict:
    doc = json.loads((DATA / "osm.json").read_text())
    dsm = np.asarray(Image.open(DATA / "dsm_1m.tif"), np.float64)
    dtm_img = np.asarray(Image.open(DATA / "dtm_1m.tif"), np.float64)
    grid = Grid()
    print(f"frame: grid convergence {grid.rotation_deg:+.3f}°, fit {grid.fit_m * 1000:.1f} mm")
    areas = osm_areas(doc)
    align(grid, areas, dsm, dtm_img)
    bounds_poly = box(*BOUNDS)

    # Water: the Thames and the docks (one level each, the median of their LiDAR ground).
    water = unary_union([p for t, p, _ in areas if _water(t)]).intersection(bounds_poly)
    waters = [w for w in _polys(water) if w.area > 200]

    # Ground on a 2 m grid, its datum the ground at the origin.
    n0, e0, n1, e1 = BOUNDS
    ns = np.arange(n0, n1 + 1e-6, DTM_CELL)
    es = np.arange(e0, e1 + 1e-6, DTM_CELL)
    nn, ee = np.meshgrid(ns, es, indexing="ij")
    h = bilinear(dtm_img, *grid.bng(nn, ee))
    for w in waters:
        core = w.buffer(-12.0) if w.buffer(-12.0).area > 50 else w
        k = shapely.contains_xy(core, nn, ee)
        level = float(np.median(h[k])) if k.any() else float(np.median(h))
        h[shapely.contains_xy(w.buffer(1.0), nn, ee)] = level
        # The bed a little below the surface, away from the banks (the water is drawn level
        # with its lowest shore).
        h[shapely.contains_xy(w.buffer(-3.0), nn, ee)] = level - 1.5
        print(f"water {w.area / 1e4:.1f} ha at {level:+.2f} m OD")
    datum = float(bilinear(dtm_img, *grid.bng(0.0, 0.0)))
    dtm = Dtm(float(ns[0]), float(es[0]), DTM_CELL, h - datum)
    print(f"datum: the ground at the origin, {datum:.2f} m OD")

    # What stands on the ground, measured: the LiDAR surface above that ground, 1 m cells.
    on = np.arange(n0 + OBS_CELL / 2, n1, OBS_CELL)
    oe = np.arange(e0 + OBS_CELL / 2, e1, OBS_CELL)
    onn, oee = np.meshgrid(on, oe, indexing="ij")
    top = highest(dsm, *grid.bng(onn, oee)) - datum
    agl = top - dtm.at(onn, oee)
    agl[agl < OBS_FLOOR_M] = 0.0
    q = np.ceil(np.clip(agl, 0, 65535 * QUANTUM_M) / QUANTUM_M).astype(np.uint16)
    measured = Obstacles(float(on[0]), float(oe[0]), OBS_CELL, QUANTUM_M, q)
    print(f"measured obstacles: {q.shape[0]} × {q.shape[1]}, highest {q.max() * QUANTUM_M:.1f} m")

    # Buildings and bridges.
    lidar = Lidar(grid, dsm, datum)
    rows, bmeta, stats = buildings(areas, lidar, dtm, bounds_poly)
    bridges = 0
    built = shapely.STRtree([p for t, p, _ in areas if "building" in t or "building:part" in t])
    for tags, p, _ in areas:
        if tags.get("man_made") != "bridge":
            continue
        # A bridge mapped in building parts (Tower Bridge) is drawn as its parts.
        hits = built.query(p, predicate="intersects")
        if len(hits) and unary_union(built.geometries[hits]).intersection(p).area > 0.2 * p.area:
            continue
        for part in _valid(p.intersection(bounds_poly)):
            if part.area < 20:
                continue
            ring = np.asarray(part.simplify(0.3).exterior.coords)[:-1]
            if len(ring) < 3:
                continue
            base = float(np.min(dtm.at(ring[:, 0], ring[:, 1])))
            deck = float(np.percentile(lidar.inside(part, 1.0), 60))
            if deck - base < 2.0:
                continue
            rect = part.minimum_rotated_rectangle
            rows.append([rect.centroid.x, rect.centroid.y, 1.0, 1.0, 0.0, base, deck - base,
                         deck - base, ROOFS.index("flat"), 0])  # fmt: skip
            bmeta.append({
                "kind": "bridge", "name": tags.get("name"), "colours": ["#8d9093", "#5f6165"],
                "outline": np.round(ring, 2).tolist(), "holes": [],
                "bottom": round(max(deck - base - 2.5, 0.0), 2),
            })  # fmt: skip
            bridges += 1
    print(f"buildings: {len(rows) - bridges} ({stats}), bridges {bridges}")

    # Land cover: the city's paving, then squares and yards, parks and gardens, water.
    plots = [_plot("city", "settlement", "settlement", bounds_poly)]
    for tags, p, oid in areas:
        for (k, v), cover in PAVED.items():
            if tags.get(k) == v and (k != "highway" or tags.get("area") == "yes"):
                plots += [_plot(f"paved-{oid}-{i}", "paved", cover, q_)
                          for i, q_ in enumerate(_polys(p.intersection(bounds_poly)))
                          if q_.area > 20]  # fmt: skip
                break
    zone_poly = None
    for tags, p, oid in areas:
        for (k, v), cover in GREEN.items():
            if tags.get(k) == v:
                for i, q_ in enumerate(_polys(p.intersection(bounds_poly))):
                    if q_.area < 20:
                        continue
                    zone = None
                    if tags.get("name") == "Potters Fields Park" and zone_poly is None:
                        zone, zone_poly = "A", q_
                    plots.append(
                        _plot(f"green-{oid}-{i}", "green", cover, q_, zone, tags.get("name"))
                    )
                break
    for i, w in enumerate(waters):
        plots.append(_plot(f"water-{i}", "water", "water", w, None, None))
    if zone_poly is None:
        raise RuntimeError("Potters Fields Park is not in the OSM extract")

    roads = []
    for tags, line, oid in osm_lines(doc):
        hw = tags.get("highway")
        if hw not in ROAD_KIND or tags.get("area") == "yes":
            continue
        if tags.get("tunnel") or tags.get("covered") == "yes" or tags.get("bridge"):
            continue
        layer = _int(tags.get("layer"))
        if layer is not None and layer < 0:
            continue
        width = _metres(tags.get("width"))
        lanes = _int(tags.get("lanes"))
        if width is None:
            width = lanes * 3.2 + 1.0 if lanes else ROAD_WIDTH[hw.removesuffix("_link")]
        for k, seg in enumerate(_lines(line.intersection(bounds_poly))):
            if seg.length < 3:
                continue
            roads.append({
                "id": f"{oid}-{k}", "name": tags.get("name"), "kind": ROAD_KIND[hw],
                "width": round(float(np.clip(width, 3.0, 24.0)), 1),
                "points": np.round(np.asarray(seg.simplify(0.4).coords), 2).tolist(),
            })  # fmt: skip
    print(f"roads: {len(roads)}")

    # Railways above ground: on the ground, on viaducts (brick, solid to the ground) or on
    # bridges (a deck, over the river and the streets), their rails at the LiDAR's deck.
    rails = railways(doc, areas, waters, bounds_poly, dtm, lidar)
    print(f"railways: {len(rails)} lines, {sum(len(r['points']) for r in rails) * 4 / 1000:.1f} km")

    # Trees: the tops of the LiDAR canopy, away from buildings, bridges and water.
    trees = canopy_trees(measured, areas, waters, bounds_poly, dtm)
    print(f"trees: {len(trees)}")

    meta = {
        "plots": plots, "woods": [], "roads": roads, "structures": [], "wires": [],
        "river": thames_line(waters), "buildings": bmeta, "rails": rails,
        "zones": [{"id": "A", "at_ne_m": _pt(zone_poly), "crop": "park",
                   "name": "Potters Fields Park"}],
        "places": places(areas, doc, waters),
    }  # fmt: skip
    save_baked(
        ROOT / "configs" / "worlds" / f"{NAME}.world.npz",
        dtm=dtm,
        trees=np.asarray(trees, float).reshape(-1, 6),
        hedges=np.zeros((0, 6)),
        buildings=np.asarray(rows, float),
        meta=meta,
        measured=measured,
    )
    return {"datum": datum, "zone": zone_poly, "areas": areas}


def railways(doc, areas, waters, bounds_poly, dtm: Dtm, lidar: Lidar) -> list[dict]:
    """OSM's railways above ground, merged into long lines and resampled every 4 m."""
    ways = []
    for tags, line, _ in osm_lines(doc):
        if tags.get("railway") not in ("rail", "light_rail") or tags.get("tunnel"):
            continue
        layer = _int(tags.get("layer"))
        if layer is not None and layer < 0:
            continue
        ways.append((tags, line))
    if not ways:
        return []
    viaduct = unary_union([ln.buffer(1.5) for t, ln in ways if t.get("bridge") == "viaduct"])
    bridge = unary_union([ln.buffer(1.5) for t, ln in ways if t.get("bridge") == "yes"])
    roofed = unary_union([p for t, p, _ in areas if "building" in t])
    water = unary_union(waters)
    light = unary_union([ln.buffer(0.5) for t, ln in ways if t.get("railway") == "light_rail"])
    out = []
    for line in _lines(linemerge(unary_union([ln for _, ln in ways])).intersection(bounds_poly)):
        if line.length < 30:
            continue
        k = max(2, round(line.length / 4.0) + 1)
        pts = np.array([line.interpolate(d).coords[0] for d in np.linspace(0, line.length, k)])
        n, e = pts[:, 0], pts[:, 1]
        ground = dtm.at(n, e)
        on_viaduct = shapely.contains_xy(viaduct, n, e)
        on_bridge = shapely.contains_xy(bridge, n, e) | (
            on_viaduct & shapely.contains_xy(water, n, e)
        )
        up = on_viaduct | on_bridge
        if up.any():
            # The deck: the lowest LiDAR within 1.6 m (not a train on it), not under a station's
            # roof (there, between its ends), at least a metre over the ground.
            deck = np.full(k, np.nan)
            deck[up] = [float(lidar.inside(shapely.Point(a, b).buffer(1.6), 0.0).min())
                        for a, b in zip(n[up], e[up], strict=True)]  # fmt: skip
            deck[up & shapely.contains_xy(roofed, n, e)] = np.nan
            ok = np.isfinite(deck)
            if ok.any():
                d = np.r_[0.0, np.cumsum(np.hypot(np.diff(n), np.diff(e)))]
                deck = np.interp(d, d[ok], deck[ok])
            else:
                deck = ground + 8.0
            deck = np.where(up, np.maximum(deck, ground + 1.0), ground + 0.3)
        else:
            deck = ground + 0.3
        h = ndimage.median_filter(deck, size=5, mode="nearest")
        carried = np.where(on_bridge, 2, np.where(on_viaduct, 1, 0)).astype(int)
        # OSM calls long runs of brick viaduct bridges: over land, a bridge is a short span
        # (a street under it); a longer one is a viaduct.
        wet = shapely.contains_xy(water, n, e)
        run = 0
        for i in range(k + 1):
            if i < k and carried[i] == 2 and not wet[i]:
                run += 1
                continue
            if run * 4.0 > 40.0:
                carried[i - run : i] = 1
            run = 0
        carried[shapely.contains_xy(roofed, n, e)] = 3  # under a station's roof
        mid = pts[len(pts) // 2]
        kind = "light_rail" if shapely.contains_xy(light, mid[0], mid[1]) else "rail"
        out.append({
            "kind": kind,
            "points": np.round(np.c_[pts, h], 2).tolist(),
            "carried": carried.tolist(),
        })  # fmt: skip
    return out


def thames_line(waters) -> dict:
    """The Thames's centre line (west to east), for its name and the boats on it: the middle
    of the river across each line of constant east, smoothed."""
    river = max(waters, key=lambda w: w.area)
    pts, widths = [], []
    for e in np.arange(BOUNDS[1] + 10, BOUNDS[3] - 10, 20.0):
        cut = LineString([(BOUNDS[0], e), (BOUNDS[2], e)]).intersection(river)
        segs = [g for g in _lines(cut) if g.length > 30]
        if segs:
            g = max(segs, key=lambda g: g.length)
            (a, _), (b, _) = g.coords[0], g.coords[-1]
            pts.append(((a + b) / 2, e))
            widths.append(g.length)
    p = np.asarray(pts)
    k = np.ones(7) / 7
    smooth = np.c_[np.convolve(np.pad(p[:, 0], 3, mode="edge"), k, "valid"), p[:, 1]]
    return {
        "id": "thames",
        "name": "the Thames",
        "width": round(float(np.median(widths)), 1),
        "depth": 5.0,
        "points": np.round(smooth, 1).tolist(),
    }


def _lines(g) -> Iterable[LineString]:
    if isinstance(g, LineString):
        yield g
    else:
        for x in getattr(g, "geoms", []):
            yield from _lines(x)


def _pt(p: Polygon) -> list[float]:
    c = p.representative_point()
    return [round(c.x, 2), round(c.y, 2)]


def canopy_trees(measured: Obstacles, areas, waters, bounds_poly, dtm: Dtm) -> list[list[float]]:
    rows, cols = measured.agl.shape
    img = Image.new("1", (cols, rows), 0)
    dr = ImageDraw.Draw(img)
    n0, e0 = measured.n0 - measured.cell / 2, measured.e0 - measured.cell / 2
    blocked = [p for t, p, _ in areas if "building" in t or "building:part" in t
               or t.get("man_made") in ("bridge", "pier") or "railway" in t]  # fmt: skip
    blocked += list(waters)
    for p in blocked:
        for q_ in _polys(p.buffer(1.5).intersection(bounds_poly)):
            if q_.is_empty:
                continue
            ring = np.asarray(q_.exterior.coords)
            dr.polygon(list(zip(ring[:, 1] - e0, ring[:, 0] - n0, strict=True)), fill=1)
    off = np.asarray(img, bool)
    chm = measured.agl.astype(float) * measured.quantum
    chm[off] = 0.0
    smooth = ndimage.uniform_filter(chm, 3)
    peak = (smooth == ndimage.maximum_filter(smooth, size=7)) & (smooth >= 5.0) & (smooth <= 32.0)
    out = []
    for i, j in zip(*np.nonzero(peak), strict=True):
        ht = float(smooth[i, j])
        # A crown, not a lamp post: enough canopy near the top round the peak.
        win = chm[max(i - 6, 0) : i + 7, max(j - 6, 0) : j + 7]
        area = float((win >= 0.5 * ht).sum())
        if area < 12:
            continue
        crown = float(np.clip(math.sqrt(area / math.pi), 2.0, 7.5))
        n, e = measured.n0 + i * measured.cell, measured.e0 + j * measured.cell
        out.append([round(n, 2), round(e, 2), round(ht, 2), round(crown, 2), 0, -1])
    return out


LANDMARKS = {
    "St Paul's Cathedral": ("st-pauls", "landmark", ["St Paul's"]),
    "The Shard": ("the-shard", "landmark", ["the Shard"]),
    "Tower Bridge": ("tower-bridge", "bridge", []),
    "Tower of London": ("tower-of-london", "landmark", ["the Tower"]),
    "Tate Modern": ("tate-modern", "landmark", []),
    "30 St Mary Axe": ("the-gherkin", "landmark", ["the Gherkin"]),
    "The Leadenhall Building": ("leadenhall", "landmark", ["the Cheesegrater"]),
    "20 Fenchurch Street": ("walkie-talkie", "landmark", ["the Walkie-Talkie", "Sky Garden"]),
    "Monument to the Great Fire of London": ("monument", "landmark", ["the Monument"]),
    "Bank of England": ("bank-of-england", "landmark", []),
    "Guildhall": ("guildhall", "landmark", []),
    "London Bridge": ("london-bridge", "bridge", []),
    "Millennium Bridge": ("millennium-bridge", "bridge", ["the wobbly bridge"]),
    "Southwark Bridge": ("southwark-bridge", "bridge", []),
    "Blackfriars Bridge": ("blackfriars-bridge", "bridge", []),
    "Cannon Street": ("cannon-street", "station", ["Cannon Street station"]),
    "Liverpool Street": ("liverpool-street", "station", ["Liverpool Street station"]),
    "Borough Market": ("borough-market", "market", []),
    "Southwark Cathedral": ("southwark-cathedral", "landmark", []),
    "HMS Belfast": ("hms-belfast", "ship", []),
    "Shakespeare's Globe": ("the-globe", "landmark", ["the Globe"]),
    "Potters Fields Park": ("potters-fields", "park", []),
    "Finsbury Circus": ("finsbury-circus", "park", []),
    "22 Bishopsgate": ("22-bishopsgate", "landmark", []),
}


# OSM's names for some of them, and two that are only points (not fetched): their WGS84.
OSM_NAMES = {"Sky Garden": "20 Fenchurch Street", "The Walkie Talkie": "20 Fenchurch Street",
             "22 Bishopsgate Tower": "22 Bishopsgate"}  # fmt: skip
POINTS = {"Monument to the Great Fire of London": (51.51015, -0.08597),
          "Tower of London": (51.50810, -0.07594)}  # fmt: skip


def places(areas, doc, waters) -> list[dict]:
    found: dict[str, list] = {}
    for name, (lat, lon) in POINTS.items():
        n, e = local_from_wgs(lat, lon)
        found[name] = [shapely.Point(float(n), float(e))]
    for tags, p, _ in areas:
        for key in ("name", "alt_name", "loc_name"):
            name = OSM_NAMES.get(tags.get(key, ""), tags.get(key))
            if name in LANDMARKS and name not in POINTS:
                found.setdefault(name, []).append(p)
                break
    for el in doc["elements"]:
        name = el.get("tags", {}).get("name")
        if name in LANDMARKS and name not in found and el["type"] == "way" and "geometry" in el:
            found.setdefault(name, []).append(LineString(_ring(el["geometry"])))
    thames = max(waters, key=lambda w: w.area).representative_point()
    out = [{"id": "the-thames", "name": "the Thames", "kind": "river",
            "point_ne_m": [round(thames.x, 1), round(thames.y, 1)],
            "aliases": ["the river"]}]  # fmt: skip
    for name, geoms in found.items():
        pid, kind, aliases = LANDMARKS[name]
        g = unary_union(geoms)
        c = g.representative_point() if g.area > 0 else g.centroid
        if not box(*BOUNDS).contains(c):
            continue
        out.append({"id": pid, "name": name, "kind": kind,
                    "point_ne_m": [round(c.x, 1), round(c.y, 1)], "aliases": aliases})  # fmt: skip
    missing = sorted(set(LANDMARKS) - set(found))
    if missing:
        print(f"places not found in OSM: {missing}")
    return out


def operations(datum: float, zone: Polygon, areas) -> dict:
    """Home at Potters Fields Park; a geofence 60 m inside the edge; a 400 m ceiling (the
    Shard is 310 m); and no-fly zones invented for the exercise round the Tower and St Paul's."""
    n0, e0, n1, e1 = BOUNDS
    inset = 60.0
    fence = [[n1 - inset, e0 + inset], [n1 - inset, e1 - inset], [n0 + inset, e1 - inset],
             [n0 + inset, e0 + inset]]  # fmt: skip
    no_fly = []
    for pid, name, test in (
        ("tower-of-london", "the Tower of London",
         lambda t: t.get("name") == "Tower of London" or t.get("building") == "castle"),
        ("st-pauls", "St Paul's Cathedral", lambda t: t.get("name") == "St Paul's Cathedral"),
    ):  # fmt: skip
        g = unary_union([p for t, p, _ in areas if test(t)])
        if g.is_empty:
            raise RuntimeError(f"{name}: not in the OSM extract")
        hull = g.buffer(25.0).convex_hull.simplify(2.0)
        no_fly.append({
            "id": pid, "name": name, "reason": f"{name} (invented for the exercise)",
            "polygons_ne_m": [np.round(np.asarray(hull.exterior.coords)[:-1], 1).tolist()],
        })  # fmt: skip
    c = zone.representative_point()
    return {
        "home_ne_m": [round(c.x, 1), round(c.y, 1)],
        "geofence_ne_m": fence,
        "ceiling_agl_m": 400.0,
        "ceiling_alt_m": 400.0,
        "min_agl_m": 30.0,
        "no_fly": no_fly,
    }


def main() -> None:
    got = bake()
    spec = {
        "schema_version": 1,
        "id": NAME,
        "name": "London: the City and Bankside",
        "description": "Central London from Blackfriars to Tower Bridge, Liverpool Street to "
        "the Shard (docs/08): ground, buildings, trees and bridges from Environment Agency "
        "LiDAR and OpenStreetMap, baked by tools/london_bake.py.",
        "seed": 0,
        "bounds_ne_m": [[BOUNDS[0], BOUNDS[1]], [BOUNDS[2], BOUNDS[3]]],
        "baked": f"{NAME}.world.npz",
    }
    world_yaml = ROOT / "configs" / "worlds" / f"{NAME}.yaml"
    world_yaml.write_text(
        "# London (docs/08), baked by tools/london_bake.py from data/london\n"
        "# (tools/london_fetch.py).\n"
        "# Derived from OpenStreetMap (ODbL) and Environment Agency LiDAR (OGL): see\n"
        "# LICENCE-london.md. Bake again; do not edit this.\n"
        + yaml.safe_dump(spec, sort_keys=False, width=100)
    )
    site_yaml = ROOT / "configs" / "sites" / f"{NAME}.yaml"
    if not site_yaml.exists():
        raise SystemExit(f"{site_yaml} is missing (it carries the origin and magnetic field)")
    site = yaml.safe_load(site_yaml.read_text())
    site["operations"] = operations(got["datum"], got["zone"], got["areas"])
    site_yaml.write_text(
        "# London (docs/08). The operations block is written by tools/london_bake.py.\n"
        + yaml.safe_dump(site, sort_keys=False, width=100)
    )
    print(world_yaml, site_yaml, sep="\n")


if __name__ == "__main__":
    main()
