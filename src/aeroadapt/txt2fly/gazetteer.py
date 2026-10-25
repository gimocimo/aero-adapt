"""The site gazetteer (spec 06 §5): named places, no-fly areas and the geofence.

Places come from three sources, in this order of precedence:

1. the site's hand-made ``operations.places`` (home, …);
2. its safe zones, as fields ("field A");
3. the site's map, for features inside the geofence or near it: the OpenStreetMap extract
   ``<site>.osm.json`` (``tools/fetch_gazetteer.py``), or, on a world site, the world's own
   features (docs/07 §6).

OSM names are third-party text: they are data for the planner and, later, for the language
model, never instructions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from aeroadapt.scenarios.site import SITE_DIR, Site
from aeroadapt.txt2fly import geometry as geo
from aeroadapt.txt2fly.plan import (
    Beyond,
    Here,
    Home,
    LatLon,
    Nearest,
    Offset,
    PlaceRef,
    Point,
    Tallest,
    Target,
)

BEYOND_M = 300.0  # "the far side of X": this far past its edge
FIELD_WORDS = ("field", "landing field")  # a site's fields are its safe zones
NO_TOP_KINDS = ("home", "safe zone", "field", "airstrip", "river", "road", "street", "lane",
                "park", "green")  # fmt: skip

NEAR_GEOFENCE_M = 200.0  # OSM features this far outside the geofence are still listed


@dataclass(frozen=True)
class Place:
    id: str
    name: str
    kind: str
    point: np.ndarray  # (2,) north, east
    polygon: np.ndarray | None = None
    height_m: float | None = None
    source: str = "site"
    aliases: tuple[str, ...] = ()
    line: np.ndarray | None = None  # (k, 2) north, east: a river or a road (txt2fly.features)


ABOVE_ZONE_M = 30.0  # over an area closed below a height, a track keeps this far above it


@dataclass(frozen=True)
class NoFlyArea:
    """An area planned missions keep out of: at all heights, or (``ceiling_m``) up to a height
    above the ground under each point, or above ``base_m`` (a flat top over the ground at its
    centre: a turbine, a mast). A track may cross it ``ABOVE_ZONE_M`` above that height, and
    ``top_m`` is the highest it reaches within the tracking margin (heights above home)."""

    id: str
    name: str
    reason: str
    polygons: tuple[np.ndarray, ...]
    ceiling_m: float | None = None
    base_m: float | None = None
    top_m: float | None = None
    _set: geo.PolygonSet = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_set", geo.PolygonSet(self.polygons))

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        """Per point: distance to the area, negative inside it."""
        return self._set.signed_distance(pts)

    def closed_at(self, alt: float) -> bool:
        """Whether a track at ``alt`` (above home) must keep out of it everywhere."""
        return self.top_m is None or alt < self.top_m + ABOVE_ZONE_M

    def lowest_over(self, ground: np.ndarray) -> np.ndarray:
        """Per point, given the ground there: the lowest height (above home) a track may
        cross the area at (inf: closed at all heights)."""
        ground = np.asarray(ground, float)
        if self.ceiling_m is None:
            return np.full(ground.shape, np.inf)
        base = ground if self.base_m is None else np.full(ground.shape, self.base_m)
        return base + self.ceiling_m + ABOVE_ZONE_M

    @property
    def reaches(self) -> str:
        """How high it is closed, in words."""
        if self.ceiling_m is None:
            return "closed at all heights"
        if self.base_m is None:
            return f"closed up to {self.ceiling_m:g} m above the ground"
        return f"closed up to {self.top_m:.0f} m above home"


@dataclass(frozen=True)
class Resolved:
    """A target turned into geometry."""

    name: str
    point: np.ndarray
    polygon: np.ndarray | None = None
    height_m: float | None = None
    line: np.ndarray | None = None


class UnknownPlace(ValueError):
    pass


@dataclass(frozen=True)
class Gazetteer:
    site: Site
    places: dict[str, Place]
    no_fly: tuple[NoFlyArea, ...]
    geofence: np.ndarray
    home: np.ndarray
    attribution: str | None = None
    _ids: dict[str, str] = field(default_factory=dict, repr=False)

    def place(self, ident: str) -> Place:
        key = self._ids.get(ident.strip().lower())
        if key is None:
            raise UnknownPlace(f"there is no place called '{ident}' on this site")
        return self.places[key]

    def resolve(self, target: Target, here: np.ndarray) -> Resolved:
        """A plan target as a point (and outline, where it has one)."""
        if isinstance(target, PlaceRef):
            p = self.place(target.id)
            return Resolved(p.name, p.point, p.polygon, p.height_m, p.line)
        if isinstance(target, Here):
            return Resolved("the aircraft's position", np.asarray(here, float))
        if isinstance(target, Home):
            return Resolved("home", self.home)
        if isinstance(target, Offset):
            base = self._origin(target.origin, here)
            b = math.radians(target.bearing_deg)
            point = base.point + target.distance_m * np.array([math.cos(b), math.sin(b)])
            return Resolved(f"{target.distance_m:g} m from {base.name}", point)
        if isinstance(target, Nearest):
            base = self._origin(target.origin, here)
            among = target.among.strip().lower()
            kinds = {among} | ({"safe zone"} if among in FIELD_WORDS else set())  # the fields
            skip = None if target.origin in ("here", "home") else self.place(target.origin).id
            cands = [p for p in self.places.values() if p.kind.lower() in kinds and p.id != skip]
            if not cands:
                raise UnknownPlace(f"there is no {target.among} on this site")
            d = [float(np.linalg.norm(p.point - base.point)) for p in cands]
            p = cands[int(np.argmin(d) if target.kind == "nearest" else np.argmax(d))]
            return Resolved(p.name, p.point, p.polygon, p.height_m, p.line)
        if isinstance(target, Tallest):
            among = None if target.among is None else target.among.strip().lower()
            tops = [(top, p) for p in self.places.values()
                    if (among is None or p.kind.lower() == among)
                    and (top := self.heights(p)[1]) is not None]  # fmt: skip
            if not tops:
                raise UnknownPlace(f"there is no {target.among or 'place'} with a height here")
            p = max(tops, key=lambda tp: tp[0])[1]
            return Resolved(p.name, p.point, p.polygon, p.height_m, p.line)
        if isinstance(target, Beyond):
            base = self._origin(target.origin, here)
            x = self.place(target.id)
            u = x.point - base.point
            if float(np.linalg.norm(u)) < 1.0:
                raise ValueError(f"{base.name} is at {x.name}: it has no far side from there")
            u = u / float(np.linalg.norm(u))
            far = 0.0 if x.polygon is None else max(float(np.max((x.polygon - x.point) @ u)), 0.0)
            d = BEYOND_M if target.distance_m is None else float(target.distance_m)
            return Resolved(f"{d:g} m beyond {x.name}", x.point + (far + d) * u)
        if isinstance(target, Point):
            return Resolved("the point on the map", np.asarray(target.ne_m, float))
        if isinstance(target, LatLon):
            o = self.site.origin
            ned = o.geodetic().geodetic_to_ned(target.lat_deg, target.lon_deg, o.h_ellipsoid_m)
            return Resolved(f"{target.lat_deg:.5f}, {target.lon_deg:.5f}", np.asarray(ned[:2]))
        raise TypeError(target)

    def heights(self, p: Place) -> tuple[float, float | None]:
        """A place's ground and top, metres above home (the top: None where it means
        nothing: a field, a road, home)."""
        from aeroadapt.txt2fly.terrain import relief

        rel = relief(self.site)
        if rel is None:
            return 0.0, p.height_m
        ground = float(rel.ground(p.point[None])[0])
        if p.kind in NO_TOP_KINDS:
            return ground, None
        top = float(rel.top(p.point[None])[0])
        if p.height_m is not None:
            top = max(top, ground + p.height_m)
        return ground, top

    def _origin(self, origin: str, here: np.ndarray) -> Resolved:
        if origin in ("here", "home"):
            return self.resolve(Here() if origin == "here" else Home(), here)
        return self.resolve(PlaceRef(id=origin), here)


def _osm(site: Site) -> dict:
    """The site's map: its world's gazetteer, or its OpenStreetMap extract."""
    if site.world is not None:
        from aeroadapt.world import load_world
        from aeroadapt.world.places import world_gazetteer

        return world_gazetteer(load_world(site.world))
    path = SITE_DIR / f"{site.id}.osm.json"
    return json.loads(path.read_text()) if path.exists() else {"features": [], "areas": []}


def _zone_names(site: Site) -> dict[str, str]:
    """A world's names for its landing grounds ("Mill Meadow" for zone A)."""
    if site.world is None:
        return {}
    from aeroadapt.world import load_world_spec

    return {z.id: z.name for z in load_world_spec(site.world).zones if z.name}


def _load(site: Site) -> Gazetteer:
    ops = site.operations
    if ops is None:
        raise ValueError(f"site {site.id} has no operations block: planned missions need one")
    osm = _osm(site)
    fence = np.asarray(ops.geofence_ne_m, float)
    places: dict[str, Place] = {}
    for p in ops.places:
        places[p.id] = Place(p.id, p.name, p.kind, np.asarray(p.point_ne_m, float),
                             aliases=tuple(p.aliases))  # fmt: skip
    names = _zone_names(site)
    for z in site.safe_zones:
        poly = np.asarray(z.polygon_ne_m, float)
        pid = f"zone-{z.id.lower()}"
        aliases = (f"zone {z.id}", f"safe zone {z.id}", z.id)
        aliases += (names[z.id],) if z.id in names else ()
        places[pid] = Place(pid, f"field {z.id}", "safe zone", geo.centroid(poly), poly,
                            source="site safe zone", aliases=aliases)  # fmt: skip
    for f in osm["features"]:
        point = np.asarray(f["point_ne_m"], float)
        if not geo.inside(point, fence)[0] and geo.distance_to_boundary(point, fence)[0] > (
            NEAR_GEOFENCE_M
        ):
            continue
        poly = np.asarray(f["polygon_ne_m"], float) if "polygon_ne_m" in f else None
        source = "world" if site.world is not None else "OpenStreetMap"
        places.setdefault(f["id"], Place(f["id"], f["name"], f["kind"], point, poly,
                                         f.get("height_m"), source))  # fmt: skip
    areas = {a["id"]: a for a in osm.get("areas", [])}
    no_fly = []
    for z in ops.no_fly:
        area = z.osm_area or z.world_area
        if area is not None:
            if area not in areas:
                raise ValueError(f"no-fly {z.id}: no area {area} in the site's map")
            polys = areas[area]["polygons_ne_m"]
        else:
            polys = z.polygons_ne_m
        polys = tuple(np.asarray(p, float) for p in polys)
        base, top = _heights(site, z.ceiling_m, z.ceiling_from, z.centre_ground_m, polys)
        no_fly.append(NoFlyArea(z.id, z.name, z.reason, polys, z.ceiling_m, base, top))
    from aeroadapt.txt2fly.features import enrich

    places = enrich(site, places, tuple(no_fly))  # outlines and lines (docs/06 §17)
    ids: dict[str, str] = {}
    for p in places.values():  # ids first, then names and aliases (first come keeps them)
        ids.setdefault(p.id.lower(), p.id)
    for p in places.values():
        for name in (p.name, *p.aliases):
            ids.setdefault(name.lower(), p.id)
    return Gazetteer(site, places, tuple(no_fly), fence, np.asarray(ops.home_ne_m, float),
                     osm.get("attribution"), ids)  # fmt: skip


def _heights(
    site: Site,
    ceiling: float | None,
    ceiling_from: str,
    centre_ground: float | None,
    polys: tuple[np.ndarray, ...],
) -> tuple[float | None, float | None]:
    """A no-fly area's flat base (None: it follows the ground) and its top within the
    tracking margin, above home (None: all heights)."""
    from aeroadapt.txt2fly.terrain import ground_at

    if ceiling is None:
        return None, None
    pts = np.vstack(polys)
    if ceiling_from == "centre":
        if centre_ground is None:
            centre_ground = float(ground_at(site, geo.centroid(polys[0])[None])[0])
        return centre_ground, centre_ground + ceiling
    lo, hi = pts.min(axis=0), pts.max(axis=0)  # the ground under it, every 10 m
    axes = [np.arange(lo[k], hi[k] + 10.0, 10.0) for k in (0, 1)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
    inside = np.zeros(len(grid), bool)
    for p in polys:
        inside |= geo.inside(grid, p)
    return None, float(ground_at(site, np.vstack([pts, grid[inside]])).max()) + ceiling


_CACHE: dict[str, Gazetteer] = {}


def load_gazetteer(site: Site) -> Gazetteer:
    key = site.model_dump_json()
    if key not in _CACHE:
        _CACHE[key] = _load(site)
    return _CACHE[key]


def site_layers(site: Site) -> dict | None:
    """What the browser draws and offers: geofence, no-fly areas, places (None without an
    operations block)."""
    if site.operations is None:
        return None
    g = load_gazetteer(site)
    r = lambda a: np.round(np.asarray(a, float), 1).tolist()  # noqa: E731
    heights = {z.id: z for z in site.operations.no_fly}  # how high each reaches (None: all heights)
    return {
        "id": site.id,
        "geofence": r(g.geofence),
        "home": r(g.home),
        "no_fly": [
            {
                "id": a.id,
                "name": a.name,
                "reason": a.reason,
                "polygons": [r(p) for p in a.polygons],
                "ceiling_m": heights[a.id].ceiling_m,
                "ceiling_from": heights[a.id].ceiling_from,
            }
            for a in g.no_fly
        ],
        "places": [
            {"id": p.id, "name": p.name, "kind": p.kind, "point": r(p.point)}
            for p in g.places.values()
        ],
        "attribution": g.attribution,
    }
