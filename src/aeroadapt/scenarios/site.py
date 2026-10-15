"""Site definitions: geodetic origin, ground model, atmosphere offsets, Earth magnetic field, and
pre-surveyed safe zones (authored in YAML under configs/sites)."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aeroadapt.core.geodesy import GeodeticOrigin

SITE_DIR = Path(__file__).resolve().parents[3] / "configs" / "sites"
MAX_SAFE_ZONES = 16  # fixed so that contingency logic stays jit-able (spec §5.4)


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


class Site(_Strict):
    schema_version: int = Field(ge=1, le=1)
    id: str
    name: str
    description: str
    origin: Origin
    geoid_undulation_m: float = Field(
        ge=-110, le=90, description="Ellipsoid → MSL separation (ISA and baro reference only)"
    )
    ground_model: Literal["flat"]
    ground_survey: GroundSurvey
    isa: Isa
    magnetic: Magnetic
    airspace_notes: str
    safe_zones: list[SafeZone] = Field(default_factory=list, max_length=MAX_SAFE_ZONES)

    @property
    def ground_msl_m(self) -> float:
        return self.origin.h_ellipsoid_m - self.geoid_undulation_m


def load_site(name_or_path: str | Path) -> Site:
    """Load and validate a site by id (``configs/sites/<id>.yaml``) or by path."""
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = SITE_DIR / f"{name_or_path}.yaml"
    with path.open() as f:
        return Site.model_validate(yaml.safe_load(f))
