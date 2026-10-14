# pyright: strict
"""WGS84 geodesy and the site-local NED frame (ADR-001).

This module is host-side NumPy in float64 on purpose. ECEF coordinates are ~6.4e6 m, so
float32 would quantise positions to ~0.5 m. The JAX simulator therefore works purely in the
local NED frame, and the imperative shell converts to latitude/longitude with this module.

Heights are WGS84 *ellipsoidal* heights. In the UK the geoid lies roughly 50 m above the
ellipsoid, so MSL heights must never be mixed in silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

F64 = NDArray[np.float64]


def geodetic_to_ecef(lat_deg: ArrayLike, lon_deg: ArrayLike, h_m: ArrayLike) -> F64:
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    h = np.asarray(h_m, dtype=np.float64)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(xyz: ArrayLike, iterations: int = 6) -> tuple[F64, F64, F64]:
    """ECEF → (lat_deg, lon_deg, h_m), by fixed-point iteration on latitude.

    The iteration converges quadratically away from the poles; six iterations reach
    float64 precision at UK latitudes, and this is verified in the tests.
    """
    p_xyz = np.asarray(xyz, dtype=np.float64)
    x, y, z = p_xyz[..., 0], p_xyz[..., 1], p_xyz[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    h = np.zeros_like(p)
    for _ in range(iterations):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    return np.rad2deg(lat), np.rad2deg(lon), h


def _r_ned_ecef(lat_deg: float, lon_deg: float) -> F64:
    """Rotation matrix taking ECEF vectors into the NED frame at (lat, lon)."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    return np.array(
        [
            [-sl * co, -sl * so, cl],
            [-so, co, 0.0],
            [-cl * co, -cl * so, -sl],
        ]
    )


@dataclass(frozen=True)
class GeodeticOrigin:
    """Origin of a site-local NED frame. ``h_ellipsoid_m`` is a WGS84 ellipsoidal height."""

    lat_deg: float
    lon_deg: float
    h_ellipsoid_m: float
    _ecef: F64 = field(init=False, repr=False, compare=False)
    _r_ned_ecef: F64 = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_ecef", geodetic_to_ecef(self.lat_deg, self.lon_deg, self.h_ellipsoid_m)
        )
        object.__setattr__(self, "_r_ned_ecef", _r_ned_ecef(self.lat_deg, self.lon_deg))

    def ecef_to_ned(self, xyz: ArrayLike) -> F64:
        d = np.asarray(xyz, dtype=np.float64) - self._ecef
        return d @ self._r_ned_ecef.T

    def ned_to_ecef(self, ned: ArrayLike) -> F64:
        return np.asarray(ned, dtype=np.float64) @ self._r_ned_ecef + self._ecef

    def geodetic_to_ned(self, lat_deg: ArrayLike, lon_deg: ArrayLike, h_m: ArrayLike) -> F64:
        return self.ecef_to_ned(geodetic_to_ecef(lat_deg, lon_deg, h_m))

    def ned_to_geodetic(self, ned: ArrayLike) -> tuple[F64, F64, F64]:
        return ecef_to_geodetic(self.ned_to_ecef(ned))
