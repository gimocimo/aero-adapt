"""Site configs: schema validation, magnetic-field self-consistency, frame round-trip."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from aeroadapt.scenarios import SITE_DIR, Site, load_site
from aeroadapt.scenarios.site import MAX_SAFE_ZONES


@pytest.mark.parametrize("path", sorted(SITE_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_site_config_validates(path) -> None:
    site = load_site(path)
    assert site.id == path.stem


def test_cambridge_west_dev_site() -> None:
    site = load_site("cambridge_west")
    # Ground height comes from the 3D-tiles surface over open field and must stay close to
    # the independent terrain model (ADR-001: ellipsoidal heights, UK geoid about +46 m).
    survey = site.ground_survey
    assert survey is not None
    assert abs(site.origin.h_ellipsoid_m - survey.origin_tiles_surface_m) < 0.1
    assert abs(survey.origin_terrain_dtm_m - survey.origin_tiles_surface_m) < 2.0
    # UK magnetic field in 2026: small easterly declination, steep inclination.
    assert 0.0 < site.magnetic.declination_deg < 3.0
    assert 65.0 < site.magnetic.inclination_deg < 70.0


def test_origin_frame_roundtrip() -> None:
    o = load_site("cambridge_west").origin.geodetic()
    lat, lon, h = o.ned_to_geodetic(np.array([[0.0, 0.0, 0.0], [1000.0, -500.0, -120.0]]))
    np.testing.assert_allclose(lat[0], 52.2075, atol=1e-12)
    np.testing.assert_allclose(lon[0], 0.0747, atol=1e-12)
    np.testing.assert_allclose(h, [63.9, 63.9 + 120.0], atol=0.2)  # flat-earth sagitta at 1 km


def test_rejects_inconsistent_magnetic_field() -> None:
    data = load_site("cambridge_west").model_dump(mode="json")
    data["magnetic"]["declination_deg"] = 5.0
    with pytest.raises(ValidationError, match="declination"):
        Site.model_validate(data)


def test_rejects_too_many_safe_zones() -> None:
    data = load_site("cambridge_west").model_dump(mode="json")
    zone = {"id": "z", "polygon_ne_m": [[0, 0], [0, 50], [50, 0]], "surface": "grass", "quality": 1}
    data["safe_zones"] = [dict(zone, id=f"z{i}") for i in range(MAX_SAFE_ZONES + 1)]
    with pytest.raises(ValidationError):
        Site.model_validate(data)


def test_safe_zone_contains() -> None:
    zones = load_site("cambridge_west").safe_zones
    assert len(zones) >= 4
    a = zones[0]
    assert a.contains(-150.0, -60.0)  # field interior
    assert not a.contains(200.0, 0.0)  # north of the field
    centroids = [np.mean(np.asarray(z.polygon_ne_m), axis=0) for z in zones]
    assert all(z.contains(*c) for z, c in zip(zones, centroids, strict=True))
