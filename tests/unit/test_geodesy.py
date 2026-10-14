"""V1 (part): WGS84 geodesy against pymap3d (an independent implementation)."""

from __future__ import annotations

import numpy as np
import pymap3d
import pytest

from aeroadapt.core.geodesy import GeodeticOrigin, ecef_to_geodetic, geodetic_to_ecef

# Candidate UK site origins (approximate), plus edge cases at other latitudes.
ORIGINS = [
    (50.8853, -0.2133, 190.0),  # South Downs
    (52.2075, 0.0600, 70.0),  # Cambridge fringe
    (53.3440, -1.7760, 230.0),  # Peak District edge
    (0.0, 0.0, 0.0),
    (-33.9, 151.2, 50.0),
]


@pytest.mark.parametrize("origin", ORIGINS)
def test_ned_matches_pymap3d_within_20km(origin: tuple[float, float, float]) -> None:
    lat0, lon0, h0 = origin
    o = GeodeticOrigin(lat0, lon0, h0)
    rng = np.random.default_rng(42)
    ned = rng.uniform([-20e3, -20e3, -600.0], [20e3, 20e3, 50.0], size=(500, 3))
    lat, lon, h = o.ned_to_geodetic(ned)
    lat_ref, lon_ref, h_ref = pymap3d.ned2geodetic(ned[:, 0], ned[:, 1], ned[:, 2], lat0, lon0, h0)
    # Compare as positions: convert both to ECEF and measure the distance.
    err = np.linalg.norm(
        geodetic_to_ecef(lat, lon, h) - geodetic_to_ecef(lat_ref, lon_ref, h_ref), axis=-1
    )
    assert err.max() <= 1e-3
    n, e, d = pymap3d.geodetic2ned(lat_ref, lon_ref, h_ref, lat0, lon0, h0)
    np.testing.assert_allclose(
        o.geodetic_to_ned(lat_ref, lon_ref, h_ref), np.stack([n, e, d], -1), atol=1e-3
    )


def test_ecef_roundtrip_precision() -> None:
    rng = np.random.default_rng(7)
    lat = rng.uniform(-80, 80, 2000)
    lon = rng.uniform(-180, 180, 2000)
    h = rng.uniform(-100, 10_000, 2000)
    lat2, lon2, h2 = ecef_to_geodetic(geodetic_to_ecef(lat, lon, h))
    np.testing.assert_allclose(lat2, lat, atol=1e-10)
    np.testing.assert_allclose(lon2, lon, atol=1e-10)
    np.testing.assert_allclose(h2, h, atol=1e-6)


def test_ecef_matches_pymap3d() -> None:
    xyz = geodetic_to_ecef(50.8853, -0.2133, 190.0)
    np.testing.assert_allclose(xyz, pymap3d.geodetic2ecef(50.8853, -0.2133, 190.0), atol=1e-6)


def test_origin_maps_to_zero_and_down_is_down() -> None:
    o = GeodeticOrigin(52.2075, 0.06, 70.0)
    np.testing.assert_allclose(o.geodetic_to_ned(52.2075, 0.06, 70.0), 0.0, atol=1e-9)
    # 100 m higher at the same lat/lon is 100 m "up", i.e. d = -100.
    np.testing.assert_allclose(o.geodetic_to_ned(52.2075, 0.06, 170.0), [0, 0, -100.0], atol=1e-6)
