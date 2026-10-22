"""Site definitions: geodetic origin, ground model, atmosphere offsets, Earth magnetic field,
pre-surveyed safe zones and, for planned missions (Txt2Fly), the operating area: geofence,
height limits, no-fly areas and named places (authored in YAML under configs/sites)."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aeroadapt.core.geodesy import GeodeticOrigin

SITE_DIR = Path(__file__).resolve().parents[3] / "configs" / "sites"
MAX_SAFE_ZONES = 24  # fixed so that contingency logic stays jit-able (spec §5.4)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Origin(_Strict):
    lat_deg: float = Field(ge=-90, le=90)
    lon_deg: float = Field(ge=-180, le=180)
    h_ellipsoid_m: float = Field(description="WGS84 ellipsoidal height of the ground (ADR-001)")

    def geodetic(self) -> GeodeticOrigin:
        return GeodeticOrigin(self.lat_deg, self.lon_deg, self.h_ellipsoid_m)


class GroundSurvey(_Strict):
    method: str
    date: date
    origin_tiles_surface_m: float
    origin_terrain_dtm_m: float
    within_300m_surface_m: dict[str, float]
    flat_area_note: str


class Isa(_Strict):
    temp_offset_k: float = 0.0
    qnh_hpa: float = Field(default=1013.25, gt=850, lt=1100)


class Magnetic(_Strict):
    model: str
    epoch_year: float
    declination_deg: float
    inclination_deg: float
    total_field_gauss: float = Field(gt=0.2, lt=0.7)
    field_ned_gauss: tuple[float, float, float]

    @model_validator(mode="after")
    def _vector_matches_angles(self) -> Magnetic:
        n, e, d = self.field_ned_gauss
        h = math.hypot(n, e)
        checks = {
            "declination": (math.degrees(math.atan2(e, n)), self.declination_deg, 0.05),
            "inclination": (math.degrees(math.atan2(d, h)), self.inclination_deg, 0.05),
            "total field": (math.sqrt(n * n + e * e + d * d), self.total_field_gauss, 1e-4),
        }
        for name, (from_vector, stated, tol) in checks.items():
            if abs(from_vector - stated) > tol:
                raise ValueError(f"magnetic {name} {stated} disagrees with field_ned_gauss")
        return self


class SafeZone(_Strict):
    id: str
    polygon_ne_m: list[tuple[float, float]] = Field(min_length=3)
    surface: str
    quality: float = Field(ge=0, le=1)
    approach_from_deg: list[float] = Field(default_factory=list)
    elevation_m: float = 0.0  # ground height above the site datum (level zones)
    # Displaced thresholds from the obstacle survey (world sites, docs/07 §5), for landing along
    # the zone's long axis either way: metres before the centre where the glide may first aim
    # at the ground (None: nothing on that approach).
    threshold_back_m: tuple[float | None, float | None] | None = None

    def contains(self, north_m: float, east_m: float) -> bool:
        """Point-in-polygon (even-odd ray casting) in the local NE plane."""
        inside = False
        pts = self.polygon_ne_m
        for (n1, e1), (n2, e2) in zip(pts, pts[1:] + pts[:1], strict=True):
            if (n1 > north_m) != (n2 > north_m):
                e_cross = e1 + (north_m - n1) * (e2 - e1) / (n2 - n1)
                if east_m < e_cross:
                    inside = not inside
        return inside


class NoFly(_Strict):
    """An area planned missions keep out of: polygons drawn here, an area of the site's
    OpenStreetMap extract (``<site>.osm.json``, by its id) or of its world (docs/07 §6)."""

    id: str
    name: str
    reason: str
    osm_area: str | None = None
    world_area: str | None = None
    polygons_ne_m: list[list[tuple[float, float]]] = Field(default_factory=list)
    # How high it reaches (None: all heights): metres above the ground under each point, or
    # (``ceiling_from: centre``) above the ground at ``centre_ne_m``, a flat top. The planner
    # keeps out of every area at all heights for now (docs/07 §14).
    ceiling_m: float | None = Field(default=None, gt=0)
    ceiling_from: Literal["ground", "centre"] = "ground"
    centre_ground_m: float | None = None  # the ground at its centre, for a flat top

    @model_validator(mode="after")
    def _one_source(self) -> NoFly:
        given = (
            (self.osm_area is not None) + (self.world_area is not None) + bool(self.polygons_ne_m)
        )
        if given != 1:
            raise ValueError(f"no-fly {self.id}: give one of osm_area, world_area, polygons_ne_m")
        return self


class Place(_Strict):
    """A named place for planned missions, beyond the safe zones and the OSM extract."""

    id: str
    name: str
    kind: str = "point"
    point_ne_m: tuple[float, float]
    aliases: list[str] = Field(default_factory=list)


class Operations(_Strict):
    """Where planned missions may fly (Txt2Fly spec §5). The geofence goes up with every
    planned mission (a breach switches the stack to RTL); the planner checks the rest."""

    home_ne_m: tuple[float, float]  # RTL loiter centre for planned missions
    geofence_ne_m: list[tuple[float, float]] = Field(min_length=3, max_length=16)
    ceiling_agl_m: float = Field(gt=0)
    # A flat ceiling besides, metres above the site datum: the altitude fence, and the
    # planner's roof wherever it is lower than ``ceiling_agl_m`` above the ground.
    ceiling_alt_m: float | None = Field(default=None, gt=0)
    min_agl_m: float = Field(gt=0)
    fence_alt_margin_m: float = Field(default=20.0, ge=0)  # altitude fence = ceiling + this
    no_fly: list[NoFly] = Field(default_factory=list)
    places: list[Place] = Field(default_factory=list)

    @model_validator(mode="after")
    def _heights(self) -> Operations:
        if self.min_agl_m >= self.ceiling_agl_m:
            raise ValueError("min_agl_m must be below ceiling_agl_m")
        return self


class Site(_Strict):
    schema_version: int = Field(ge=1, le=1)
    id: str
    name: str
    description: str
    origin: Origin
    geoid_undulation_m: float = Field(
        ge=-110, le=90, description="Ellipsoid → MSL separation (ISA and baro reference only)"
    )
    ground_model: Literal["flat", "world"]
    world: str | None = None  # configs/worlds/<id>.yaml when the ground model is "world"
    ground_survey: GroundSurvey | None = None
    isa: Isa
    magnetic: Magnetic
    airspace_notes: str
    safe_zones: list[SafeZone] = Field(default_factory=list, max_length=MAX_SAFE_ZONES)
    operations: Operations | None = None

    @model_validator(mode="after")
    def _ground(self) -> Site:
        if (self.ground_model == "world") != (self.world is not None):
            raise ValueError("a world site names its world, and only a world site does")
        if self.ground_model == "flat" and self.ground_survey is None:
            raise ValueError("a flat site records its ground survey")
        return self

    @property
    def ground_msl_m(self) -> float:
        return self.origin.h_ellipsoid_m - self.geoid_undulation_m


def load_site(name_or_path: str | Path) -> Site:
    """Load and validate a site by id (``configs/sites/<id>.yaml``) or by path. A world site's
    safe zones come from its world's survey (the YAML lists none)."""
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = SITE_DIR / f"{name_or_path}.yaml"
    with path.open() as f:
        raw = yaml.safe_load(f)
    site = Site.model_validate(raw)
    if site.world is None:
        return site
    if site.safe_zones:
        raise ValueError(f"site {site.id}: a world site's safe zones come from its world")
    from aeroadapt.world import load_world

    zones = [SafeZone.model_validate(z) for z in load_world(site.world).survey()]
    return site.model_copy(update={"safe_zones": zones})
