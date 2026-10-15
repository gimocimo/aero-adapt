"""Fault specifications as authored in scenario YAML (M1 spec §8, M2 spec §7). Time triggers.

M1 faults: ``pitot_blockage``, ``servo_jam``, ``motor_cut``. faultlab v1 adds five families,
each with a ``kind``: ``sensor_fault``, ``actuator_fault``, ``propulsion_fault``,
``energy_fault`` and ``airframe_fault``. Magnitudes are in the units given per kind.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TimeTrigger(_Strict):
    at_time_s: float = Field(ge=0)
    # Sensor faults only: the fault clears at this time (interference stops, a GNSS fix
    # returns). Physical damage does not clear.
    until_s: float | None = None

    @model_validator(mode="after")
    def _ordered(self) -> TimeTrigger:
        if self.until_s is not None and self.until_s <= self.at_time_s:
            raise ValueError("until_s must be after at_time_s")
        return self


class PitotBlockage(_Strict):
    """Total-pressure port blocked.

    * ``drain_open``: the trapped total pressure leaks to static with time constant
      ``tau_s``, so indicated airspeed decays towards zero.
    * ``drain_blocked``: total pressure is trapped at its onset value, so the instrument
      behaves like an altimeter (indicated airspeed rises in a climb, falls in a descent).

    ``ramp_s`` blends from a healthy to a fully blocked reading (partial/gradual blockage).
    """

    type: Literal["pitot_blockage"]
    variant: Literal["drain_open", "drain_blocked"]
    trigger: TimeTrigger
    ramp_s: float = Field(default=0.0, ge=0)
    tau_s: float = Field(default=3.0, gt=0)


class ServoJam(_Strict):
    """A control surface stops following commands. The jam position defaults to the
    deflection at onset; the transition into the jam still obeys rate limit and lag.

    ``jam_position_deg`` is a *physical* deflection in Beard & McLain signs (ADR-001), not a
    normalised command.
    """

    type: Literal["servo_jam"]
    channel: Literal["aileron", "elevator", "rudder"]
    trigger: TimeTrigger
    jam_position_deg: float | None = None


class MotorCut(_Strict):
    """ESC desync or cut: no thrust, the propeller windmills, rpm decays, the battery
    supplies avionics current only."""

    type: Literal["motor_cut"]
    trigger: TimeTrigger


SensorKind = Literal[
    "airspeed_bias",  # m/s added to the indicated airspeed
    "airspeed_stuck",  # differential pressure frozen at onset (magnitude unused)
    "airspeed_drift",  # m/s per s
    "gyro_bias",  # rad/s on ``axis``
    "accel_bias",  # m/s² on ``axis``
    "imu_vibration",  # m/s² RMS on the accelerometers (gyros: 2 % of it in rad/s)
    "mag_interference",  # gauss on ``axis`` (body)
    "baro_drift",  # Pa per s
    "gnss_dropout",  # no fixes for ``magnitude`` seconds (0 = for the rest of the flight)
    "gnss_drift",  # m/s of position walk along ``axis`` (velocity unaffected)
    "gnss_spoof",  # m/s² walk-off along ``axis``, position and velocity consistent
]
ActuatorKind = Literal[
    "hard_over",  # surface driven to full travel; magnitude sign picks the side
    "floating",  # disconnected: trails with the local flow (magnitude unused)
    "slow",  # rate limit × magnitude (0-1)
    "reduced_throw",  # travel × magnitude (0-1)
    "backlash",  # dead zone of ``magnitude`` degrees in the linkage
    "reversed",  # mixer error: deflection sign inverted (magnitude unused)
]
PropulsionKind = Literal[
    "partial_loss",  # thrust and torque × (1 − magnitude)
    "prop_damage",  # thrust × (1 − magnitude) and vibration ∝ magnitude × rpm
    "thermal_derate",  # throttle ceiling falls to (1 − magnitude) over ``ramp_s``
]
EnergyKind = Literal[
    "cell_failure",  # one cell lost (magnitude unused)
    "abnormal_sag",  # internal resistance × magnitude
    "capacity_loss",  # capacity × (1 − magnitude)
]
AirframeKind = Literal[
    "cg_shift",  # centre of gravity moves aft by ``magnitude`` m (negative: forward)
    "payload_drop",  # mass and inertia × (1 − magnitude)
    "wing_damage",  # lift lost on one wing: fraction ``magnitude``, ``side`` ±1 (right = +1)
]


class SensorFault(_Strict):
    type: Literal["sensor_fault"]
    kind: SensorKind
    trigger: TimeTrigger
    magnitude: float = 0.0
    axis: int = Field(default=0, ge=0, le=2)
    ramp_s: float = Field(default=0.0, ge=0)


class ActuatorFault(_Strict):
    type: Literal["actuator_fault"]
    kind: ActuatorKind
    channel: Literal["aileron", "elevator", "rudder"]
    trigger: TimeTrigger
    magnitude: float = 0.0


class PropulsionFault(_Strict):
    type: Literal["propulsion_fault"]
    kind: PropulsionKind
    trigger: TimeTrigger
    magnitude: float = Field(default=0.3, ge=0, le=1)
    ramp_s: float = Field(default=0.0, ge=0)


class EnergyFault(_Strict):
    type: Literal["energy_fault"]
    kind: EnergyKind
    trigger: TimeTrigger
    magnitude: float = Field(default=0.3, ge=0)


class AirframeFault(_Strict):
    type: Literal["airframe_fault"]
    kind: AirframeKind
    trigger: TimeTrigger
    magnitude: float = 0.0
    side: Literal[-1, 1] = 1


FaultSpec = Annotated[
    PitotBlockage
    | ServoJam
    | MotorCut
    | SensorFault
    | ActuatorFault
    | PropulsionFault
    | EnergyFault
    | AirframeFault,
    Field(discriminator="type"),
]
