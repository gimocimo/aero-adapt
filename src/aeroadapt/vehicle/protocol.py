"""The ``Vehicle`` interface (ADR-004) and its privileged extension (ADR-006)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

import numpy as np

from aeroadapt.core.geodesy import GeodeticOrigin
from aeroadapt.core.messages import ActuatorCommand, SensorFrame


class Termination(IntEnum):
    NONE = 0
    GROUND_CONTACT = 1
    OUT_OF_BOUNDS = 2
    NUMERICAL = 3
    TIME_LIMIT = 4


@dataclass(frozen=True)
class VehicleInfo:
    control_rate_hz: float
    airframe_id: str
    site_id: str
    origin: GeodeticOrigin


@dataclass(frozen=True)
class InitialCondition:
    """Start airborne in straight-and-level trim (spec §10.1 ``initial``)."""

    pos_ne_m: tuple[float, float] = (0.0, 0.0)
    alt_agl_m: float = 120.0
    heading_deg: float = 0.0
    airspeed_mps: float = 25.0
    soc: float = 1.0


@dataclass(frozen=True)
class StepResult:
    sensors: SensorFrame
    terminated: bool
    termination: Termination


@dataclass(frozen=True)
class TruthSnapshot:
    """Privileged ground truth for recording, evaluation and the oracle."""

    time_s: float
    pos_ned_m: np.ndarray
    vel_ned_mps: np.ndarray
    euler_rad: np.ndarray  # roll, pitch, yaw
    omega_rps: np.ndarray
    airspeed_mps: float
    alpha_rad: float
    beta_rad: float
    alt_agl_m: float
    wind_ned_mps: np.ndarray
    surfaces_rad: np.ndarray
    throttle: float
    thrust_n: float
    battery_v: float
    soc: float
    fault_labels: dict[str, np.ndarray]


@dataclass(frozen=True)
class ImpactRecord:
    time_s: float
    pos_ned_m: np.ndarray
    lat_deg: float
    lon_deg: float
    vel_ned_mps: np.ndarray
    sink_rate_mps: float
    groundspeed_mps: float
    roll_rad: float
    pitch_rad: float
    kinetic_energy_j: float
    touchdown_class: str  # "controlled" | "hard" | "crash" (provisional, spec §6.6)


def classify_touchdown(sink_rate: float, roll: float, pitch: float) -> str:
    """Provisional touchdown classes (spec §6.6); replaced by SORA-aligned metrics in M5."""
    deg = np.degrees
    if sink_rate <= 1.5 and abs(deg(roll)) <= 10 and -5 <= deg(pitch) <= 15:
        return "controlled"
    if sink_rate <= 3.0 and abs(deg(roll)) <= 25:
        return "hard"
    return "crash"


class Vehicle(Protocol):
    info: VehicleInfo

    def reset(self, initial: InitialCondition, seed: int) -> SensorFrame: ...
    def step(self, cmd: ActuatorCommand) -> StepResult: ...


class PrivilegedVehicle(Vehicle, Protocol):
    """Recorder / evaluator / oracle only. The flight stack never gets this view."""

    def truth(self) -> TruthSnapshot: ...
    def impact(self) -> ImpactRecord | None: ...
