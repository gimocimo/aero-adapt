"""Messages between a vehicle and the flight stack (ADR-004).

* :class:`SensorArrays` is the array form consumed by the stack's pure-JAX core. GNSS is in
  the site-local NED frame, which is float32-safe. The simulator produces it directly (for
  batched evaluation) and the shell produces it from a :class:`SensorFrame`.
* :class:`SensorFrame` and :class:`ActuatorCommand` are the host-level messages exchanged
  through the ``Vehicle`` interface. They use MAVLink-like semantics: ``time_us``, GNSS as
  WGS84 latitude/longitude/ellipsoidal height, and normalised servo-function commands.

The conversions between the two forms go through float64 geodesy (ADR-001).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from jax import Array

from aeroadapt.core.geodesy import GeodeticOrigin


class FreshFlags(NamedTuple):
    imu: Array
    baro: Array
    airspeed: Array
    gnss: Array
    mag: Array
    battery: Array
    esc: Array


class SensorArrays(NamedTuple):
    gyro: Array  # (3,) rad/s, body
    accel: Array  # (3,) m/s², specific force, body
    baro_pa: Array
    baro_temp_k: Array
    airspeed_dp_pa: Array  # raw differential pressure; IAS is derived by the navigator
    airspeed_temp_k: Array
    gnss_pos_ned: Array  # (3,) m, local NED
    gnss_vel_ned: Array  # (3,) m/s
    gnss_time_s: Array  # time of validity (latency already applied)
    mag: Array  # (3,) gauss, body
    battery_v: Array
    battery_a: Array
    battery_mah: Array
    esc_rpm: Array
    fresh: FreshFlags


@dataclass(frozen=True)
class SensorFrame:
    """One control period of sensor data, as a flight controller would see it."""

    time_us: int
    gyro_rps: np.ndarray  # (3,) body
    accel_mps2: np.ndarray  # (3,) specific force, body
    baro_pa: float
    baro_temp_k: float
    airspeed_dp_pa: float
    airspeed_temp_k: float
    gnss_lat_deg: float
    gnss_lon_deg: float
    gnss_alt_ellipsoid_m: float
    gnss_vel_ned_mps: np.ndarray  # (3,)
    gnss_time_us: int  # time of validity
    mag_gauss: np.ndarray  # (3,) body
    battery_v: float
    battery_a: float
    battery_mah: float
    esc_rpm: float
    fresh: dict[str, bool]


@dataclass(frozen=True)
class ActuatorCommand:
    """Normalised ArduPilot servo functions with aircraft-response signs (ADR-001)."""

    time_us: int
    aileron: float  # [-1, 1], +1 = roll right
    elevator: float  # [-1, 1], +1 = nose up
    rudder: float  # [-1, 1], +1 = nose right
    throttle: float  # [0, 1]

    def as_array(self) -> np.ndarray:
        return np.array([self.aileron, self.elevator, self.rudder, self.throttle])

    @staticmethod
    def from_array(time_us: int, cmd: np.ndarray) -> ActuatorCommand:
        a, e, r, t = (float(x) for x in np.asarray(cmd))
        return ActuatorCommand(time_us, a, e, r, t)


def frame_from_arrays(z: SensorArrays, time_us: int, origin: GeodeticOrigin) -> SensorFrame:
    """Array form (local NED GNSS) → host frame (WGS84 GNSS), in float64."""
    lat, lon, h = origin.ned_to_geodetic(np.asarray(z.gnss_pos_ned, dtype=np.float64))
    return SensorFrame(
        time_us=int(time_us),
        gyro_rps=np.asarray(z.gyro, dtype=np.float64),
        accel_mps2=np.asarray(z.accel, dtype=np.float64),
        baro_pa=float(z.baro_pa),
        baro_temp_k=float(z.baro_temp_k),
        airspeed_dp_pa=float(z.airspeed_dp_pa),
        airspeed_temp_k=float(z.airspeed_temp_k),
        gnss_lat_deg=float(lat),
        gnss_lon_deg=float(lon),
        gnss_alt_ellipsoid_m=float(h),
        gnss_vel_ned_mps=np.asarray(z.gnss_vel_ned, dtype=np.float64),
        gnss_time_us=round(float(z.gnss_time_s) * 1e6),
        mag_gauss=np.asarray(z.mag, dtype=np.float64),
        battery_v=float(z.battery_v),
        battery_a=float(z.battery_a),
        battery_mah=float(z.battery_mah),
        esc_rpm=float(z.esc_rpm),
        fresh={k: bool(v) for k, v in z.fresh._asdict().items()},
    )


def arrays_from_frame(f: SensorFrame, origin: GeodeticOrigin) -> SensorArrays:
    """Host frame → array form for the stack core (GNSS back to local NED, in float64)."""
    ned = origin.geodetic_to_ned(f.gnss_lat_deg, f.gnss_lon_deg, f.gnss_alt_ellipsoid_m)
    s = np.asarray
    # Host NumPy leaves (float64): the stack's jit boundary converts them.
    flags: dict[str, Any] = {k: s(v) for k, v in f.fresh.items()}
    fields: dict[str, Any] = dict(
        gyro=s(f.gyro_rps),
        accel=s(f.accel_mps2),
        baro_pa=s(f.baro_pa),
        baro_temp_k=s(f.baro_temp_k),
        airspeed_dp_pa=s(f.airspeed_dp_pa),
        airspeed_temp_k=s(f.airspeed_temp_k),
        gnss_pos_ned=s(ned),
        gnss_vel_ned=s(f.gnss_vel_ned_mps),
        gnss_time_s=s(f.gnss_time_us * 1e-6),
        mag=s(f.mag_gauss),
        battery_v=s(f.battery_v),
        battery_a=s(f.battery_a),
        battery_mah=s(f.battery_mah),
        esc_rpm=s(f.esc_rpm),
        fresh=FreshFlags(**flags),
    )
    return SensorArrays(**fields)
