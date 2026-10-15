"""Recording schema (ADR-005): column names, units and how each column is taken from the
closed-loop log. One table describes both ``frames.parquet`` (every control tick, float64)
and ``viz.arrow`` (the browser subset, 50 Hz, float32)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from aeroadapt.faultlab.slots import (
    ACTUATOR_KINDS,
    AIRFRAME_KINDS,
    ENERGY_KINDS,
    PROPULSION_KINDS,
    SENSOR_KINDS,
)
from aeroadapt.harness.closed_loop import LoopLog

BUNDLE_SCHEMA_VERSION = 1
VIZ_RATE_HZ = 50.0
FAULT_BITS = {
    "pitot": 1,
    "motor_cut": 2,
    "servo_jam_aileron": 4,
    "servo_jam_elevator": 8,
    "servo_jam_rudder": 16,
}
# faultlab v1 kinds (M2 spec §7): one bit per kind, from bit 5 upwards (31 bits in all).
FAULT_FAMILIES = {
    "sensor": SENSOR_KINDS[1:],
    "actuator": ACTUATOR_KINDS[1:],
    "propulsion": PROPULSION_KINDS[1:],
    "energy": ENERGY_KINDS[1:],
    "airframe": AIRFRAME_KINDS[1:],
}
_bit = 5
for _family, _kinds in FAULT_FAMILIES.items():
    for _kind in _kinds:
        FAULT_BITS[f"{_family}:{_kind}"] = 1 << _bit
        _bit += 1
assert _bit <= 31, "fault bits must fit an int32"

Getter = Callable[[LoopLog], np.ndarray]


@dataclass(frozen=True)
class Column:
    name: str
    unit: str
    get: Getter
    viz: bool = False  # also in viz.arrow
    integer: bool = False


def _f(field: str, i: int | None = None) -> Getter:
    def get(log: LoopLog) -> np.ndarray:
        x = np.asarray(getattr(log, field))
        return x if i is None else x[:, i]

    return get


def _sensor(field: str, i: int | None = None) -> Getter:
    def get(log: LoopLog) -> np.ndarray:
        x = np.asarray(getattr(log.sensors, field))
        return x if i is None else x[:, i]

    return get


def _fault_bits(log: LoopLog) -> np.ndarray:
    lab = log.labels
    bits = np.asarray(lab.pitot).astype(np.int32) * FAULT_BITS["pitot"]
    bits += np.asarray(lab.motor_cut).astype(np.int32) * FAULT_BITS["motor_cut"]
    jam = np.asarray(lab.servo_jam)
    for k, name in enumerate(("aileron", "elevator", "rudder")):
        bits += jam[:, k].astype(np.int32) * FAULT_BITS[f"servo_jam_{name}"]
    for family, kinds in FAULT_FAMILIES.items():
        active = np.asarray(getattr(lab, f"{family}_kind"))
        active = active.reshape(active.shape[0], -1)  # (steps, slots)
        for k, kind in enumerate(kinds, start=1):
            bits |= np.any(active == k, axis=1).astype(np.int32) * FAULT_BITS[f"{family}:{kind}"]
    return bits


def _reach_mask(log: LoopLog) -> np.ndarray:
    """Bit z set when safe zone z is reachable (height margin ≥ 0)."""
    m = np.asarray(log.zone_margin) >= 0.0
    return (m * (1 << np.arange(m.shape[1], dtype=np.int64))).sum(axis=1).astype(np.int32)


def _alt_agl(log: LoopLog) -> np.ndarray:
    return -np.asarray(log.pos_ned)[:, 2]


def _climb(log: LoopLog) -> np.ndarray:
    return -np.asarray(log.vel_ned)[:, 2]


def _groundspeed(log: LoopLog) -> np.ndarray:
    v = np.asarray(log.vel_ned)
    return np.hypot(v[:, 0], v[:, 1])


def _sensor_tas(log: LoopLog) -> np.ndarray:
    """TAS from the pitot differential pressure and baro density (what the stack sees)."""
    z = log.sensors
    rho = np.asarray(z.baro_pa) / (287.05287 * np.asarray(z.baro_temp_k))
    return np.sqrt(2.0 * np.maximum(np.asarray(z.airspeed_dp_pa), 0.0) / rho)


NED = ("n", "e", "d")
XYZ = ("x", "y", "z")
EULER = ("roll", "pitch", "yaw")
SURF = ("aileron", "elevator", "rudder")

COLUMNS: list[Column] = [
    # --- truth (privileged) ---------------------------------------------------------
    *(Column(f"truth.pos_{a}", "m", _f("pos_ned", i), viz=True) for i, a in enumerate(NED)),
    *(Column(f"truth.vel_{a}", "m/s", _f("vel_ned", i)) for i, a in enumerate(NED)),
    *(Column(f"truth.q{a}", "1", _f("quat", i), viz=True) for i, a in enumerate("wxyz")),
    *(Column(f"truth.{a}", "rad", _f("euler", i), viz=True) for i, a in enumerate(EULER)),
    *(Column(f"truth.omega_{a}", "rad/s", _f("omega", i)) for i, a in enumerate(XYZ)),
    Column("truth.alt_agl", "m", _alt_agl, viz=True),
    Column("truth.climb_rate", "m/s", _climb, viz=True),
    Column("truth.groundspeed", "m/s", _groundspeed, viz=True),
    Column("truth.tas", "m/s", _f("va"), viz=True),
    Column("truth.alpha", "rad", _f("alpha"), viz=True),
    Column("truth.beta", "rad", _f("beta"), viz=True),
    *(Column(f"truth.wind_{a}", "m/s", _f("wind_ned", i), viz=i < 2) for i, a in enumerate(NED)),
    Column("truth.thrust", "N", _f("thrust")),
    Column("truth.battery_v", "V", _f("battery_v")),
    Column("truth.soc", "1", _f("soc")),
    Column("truth.prop_rpm", "rpm", _f("prop_rpm")),
    *(Column(f"actual.{a}", "rad", _f("surfaces", i), viz=True) for i, a in enumerate(SURF)),
    Column("actual.throttle", "1", _f("throttle_actual"), viz=True),
    # --- sensors (as the stack received them) ---------------------------------------
    *(Column(f"sensors.gyro_{a}", "rad/s", _sensor("gyro", i)) for i, a in enumerate(XYZ)),
    *(Column(f"sensors.accel_{a}", "m/s^2", _sensor("accel", i)) for i, a in enumerate(XYZ)),
    Column("sensors.baro_pa", "Pa", _sensor("baro_pa")),
    Column("sensors.airspeed_dp_pa", "Pa", _sensor("airspeed_dp_pa")),
    Column("sensors.tas", "m/s", _sensor_tas, viz=True),
    *(Column(f"sensors.gnss_pos_{a}", "m", _sensor("gnss_pos_ned", i)) for i, a in enumerate(NED)),
    *(
        Column(f"sensors.gnss_vel_{a}", "m/s", _sensor("gnss_vel_ned", i))
        for i, a in enumerate(NED)
    ),
    *(Column(f"sensors.mag_{a}", "G", _sensor("mag", i)) for i, a in enumerate(XYZ)),
    Column("sensors.battery_v", "V", _sensor("battery_v")),
    Column("sensors.esc_rpm", "rpm", _sensor("esc_rpm")),
    # --- estimate --------------------------------------------------------------------
    *(Column(f"est.{a}", "rad", _f("est_euler", i)) for i, a in enumerate(EULER)),
    *(Column(f"est.pos_{a}", "m", _f("est_pos_ned", i)) for i, a in enumerate(NED)),
    *(Column(f"est.vel_{a}", "m/s", _f("est_vel_ned", i)) for i, a in enumerate(NED)),
    Column("est.ias", "m/s", _f("est_ias"), viz=True),
    Column("est.airspeed", "m/s", _f("est_airspeed"), viz=True),
    *(Column(f"est.wind_{a}", "m/s", _f("est_wind_ned", i)) for i, a in enumerate(NED[:2])),
    Column("est.airspeed_healthy", "bool", _f("airspeed_healthy"), viz=True, integer=True),
    # --- commands and demands --------------------------------------------------------
    *(Column(f"cmd.{a}", "1", _f("cmd", i)) for i, a in enumerate((*SURF, "throttle"))),
    Column("demands.roll", "rad", _f("roll_dem"), viz=True),
    Column("demands.pitch", "rad", _f("pitch_dem"), viz=True),
    Column("demands.alt_target", "m", _f("alt_target"), viz=True),
    Column("demands.speed_target", "m/s", _f("speed_target"), viz=True),
    Column("demands.alt_tecs", "m", _f("alt_dem")),
    Column("demands.speed_tecs", "m/s", _f("speed_dem")),
    Column("demands.lat_acc", "m/s^2", _f("lat_acc")),
    Column("demands.xtrack", "m", _f("xtrack")),
    # --- modes, contingency, labels ----------------------------------------------------
    Column("mode", "enum", _f("mode"), viz=True, integer=True),
    Column("wp_idx", "1", _f("wp_idx"), viz=True, integer=True),
    Column("cont.failed", "bool", _f("failed"), viz=True, integer=True),
    Column("cont.phase", "enum", _f("phase"), viz=True, integer=True),
    Column("cont.zone", "1", _f("zone"), viz=True, integer=True),
    Column("cont.reach_mask", "bits", _reach_mask, viz=True, integer=True),
    Column("cont.speed_to_fly", "m/s", _f("speed_to_fly")),
    Column("labels.fault_bits", "bits", _fault_bits, viz=True, integer=True),
    Column("done", "bool", _f("done"), integer=True),
    Column("termination", "enum", _f("termination"), integer=True),
]


def columns(log: LoopLog) -> dict[str, np.ndarray]:
    """All frame columns (host NumPy, float64 / int32), plus integer ``time_us``."""
    out: dict[str, np.ndarray] = {"time_us": np.round(np.asarray(log.t) * 1e6).astype(np.int64)}
    for c in COLUMNS:
        x = np.asarray(c.get(log))
        out[c.name] = x.astype(np.int32) if c.integer else x.astype(np.float64)
    return out


def units() -> dict[str, Any]:
    return {"time_us": "us", **{c.name: c.unit for c in COLUMNS}}
