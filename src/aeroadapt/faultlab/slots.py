"""Fixed-shape fault slots (ADR-003).

Each fault type owns one slot of fixed shape, so a single compiled simulator can run
thousands of scenarios with *different* faults under ``vmap``. An inactive slot acts as an
identity transform via masks; nothing branches in Python on fault state. Triggers are
resolved to onset times here, on the host.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple, get_args

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from aeroadapt.faultlab.schema import (
    ActuatorFault,
    ActuatorKind,
    AirframeFault,
    AirframeKind,
    EnergyFault,
    EnergyKind,
    FaultSpec,
    MotorCut,
    PitotBlockage,
    PropulsionFault,
    PropulsionKind,
    SensorFault,
    SensorKind,
    ServoJam,
)

SURFACE_INDEX = {"aileron": 0, "elevator": 1, "rudder": 2}
# faultlab v1 kinds; index 0 is "none" in every family (M2 spec §7.1).
SENSOR_KINDS = ("none", *get_args(SensorKind))
ACTUATOR_KINDS = ("none", *get_args(ActuatorKind))
PROPULSION_KINDS = ("none", *get_args(PropulsionKind))
ENERGY_KINDS = ("none", *get_args(EnergyKind))
AIRFRAME_KINDS = ("none", *get_args(AirframeKind))
N_SENSOR_SLOTS = 2
N_ACTUATOR_SLOTS = 2


class PitotFault(NamedTuple):
    active: np.ndarray  # () bool
    drain_blocked: np.ndarray  # () bool; False = drain open
    onset_s: np.ndarray  # ()
    ramp_s: np.ndarray  # ()
    tau_s: np.ndarray  # () leak time constant (drain open)


class ServoJamFault(NamedTuple):
    active: np.ndarray  # (3,) bool, surfaces (aileron, elevator, rudder)
    onset_s: np.ndarray  # (3,)
    jam_rad: np.ndarray  # (3,) B&M-sign deflection, used when not at_onset
    at_onset: np.ndarray  # (3,) bool: jam at whatever deflection the surface has at onset


class MotorCutFault(NamedTuple):
    active: np.ndarray  # () bool
    onset_s: np.ndarray  # ()


class SensorFaultSlots(NamedTuple):
    """``N_SENSOR_SLOTS`` sensor faults (kind 0 = empty slot)."""

    kind: np.ndarray  # (S,) int32, index into SENSOR_KINDS
    onset_s: np.ndarray  # (S,)
    ramp_s: np.ndarray  # (S,)
    magnitude: np.ndarray  # (S,)
    axis: np.ndarray  # (S,) int32


class ActuatorFaultSlots(NamedTuple):
    kind: np.ndarray  # (A,) int32, index into ACTUATOR_KINDS
    onset_s: np.ndarray  # (A,)
    surface: np.ndarray  # (A,) int32
    magnitude: np.ndarray  # (A,)


class ScalarFaultSlot(NamedTuple):
    """One propulsion, energy or airframe fault."""

    kind: np.ndarray  # () int32
    onset_s: np.ndarray  # ()
    ramp_s: np.ndarray  # ()
    magnitude: np.ndarray  # ()
    side: np.ndarray  # () ±1 (wing damage)


class FaultConfig(NamedTuple):
    pitot: PitotFault
    servo_jam: ServoJamFault
    motor_cut: MotorCutFault
    sensors: SensorFaultSlots
    actuators: ActuatorFaultSlots
    propulsion: ScalarFaultSlot
    energy: ScalarFaultSlot
    airframe: ScalarFaultSlot


class FaultLabels(NamedTuple):
    """Ground truth emitted every step (diagnosis training data, ADR-003). The ``*_kind``
    labels are the active kind indices (0 before onset or for an empty slot)."""

    pitot: Array  # () bool
    servo_jam: Array  # (3,) bool
    motor_cut: Array  # () bool
    sensor_kind: Array  # (S,) int32
    actuator_kind: Array  # (A,) int32
    propulsion_kind: Array  # () int32
    energy_kind: Array  # () int32
    airframe_kind: Array  # () int32


def _empty_scalar() -> ScalarFaultSlot:
    z = np.array(0.0)
    return ScalarFaultSlot(np.array(0, np.int32), z, z, z, np.array(1.0))


def no_faults() -> FaultConfig:
    f, z = np.array(False), np.array(0.0)
    s, a = N_SENSOR_SLOTS, N_ACTUATOR_SLOTS
    return FaultConfig(
        pitot=PitotFault(f, f, z, z, np.array(3.0)),
        servo_jam=ServoJamFault(np.zeros(3, bool), np.zeros(3), np.zeros(3), np.zeros(3, bool)),
        motor_cut=MotorCutFault(f, z),
        sensors=SensorFaultSlots(
            np.zeros(s, np.int32), np.zeros(s), np.zeros(s), np.zeros(s), np.zeros(s, np.int32)
        ),
        actuators=ActuatorFaultSlots(
            np.zeros(a, np.int32), np.zeros(a), np.zeros(a, np.int32), np.zeros(a)
        ),
        propulsion=_empty_scalar(),
        energy=_empty_scalar(),
        airframe=_empty_scalar(),
    )


def build_fault_config(specs: Sequence[FaultSpec]) -> FaultConfig:
    cfg = no_faults()
    pitot, jam, cut = cfg.pitot, cfg.servo_jam, cfg.motor_cut
    jam_arrays = [a.copy() for a in jam]
    sens = [a.copy() for a in cfg.sensors]
    acts = [a.copy() for a in cfg.actuators]
    scalars = {"propulsion": cfg.propulsion, "energy": cfg.energy, "airframe": cfg.airframe}
    families = {
        PropulsionFault: ("propulsion", PROPULSION_KINDS),
        EnergyFault: ("energy", ENERGY_KINDS),
        AirframeFault: ("airframe", AIRFRAME_KINDS),
    }
    n_sens = n_act = 0
    for spec in specs:
        t0 = np.array(spec.trigger.at_time_s)
        if isinstance(spec, PitotBlockage):
            if bool(pitot.active):
                raise ValueError("only one pitot_blockage per scenario")
            pitot = PitotFault(
                np.array(True),
                np.array(spec.variant == "drain_blocked"),
                t0,
                np.array(spec.ramp_s),
                np.array(spec.tau_s),
            )
        elif isinstance(spec, ServoJam):
            i = SURFACE_INDEX[spec.channel]
            if jam_arrays[0][i]:
                raise ValueError(f"duplicate servo_jam on {spec.channel}")
            jam_arrays[0][i] = True
            jam_arrays[1][i] = spec.trigger.at_time_s
            jam_arrays[2][i] = math.radians(spec.jam_position_deg or 0.0)
            jam_arrays[3][i] = spec.jam_position_deg is None
        elif isinstance(spec, MotorCut):
            if bool(cut.active):
                raise ValueError("only one motor_cut per scenario")
            cut = MotorCutFault(np.array(True), t0)
        elif isinstance(spec, SensorFault):
            if n_sens == N_SENSOR_SLOTS:
                raise ValueError(f"at most {N_SENSOR_SLOTS} sensor faults per scenario")
            for arr, v in zip(
                sens,
                (SENSOR_KINDS.index(spec.kind), t0, spec.ramp_s, spec.magnitude, spec.axis),
                strict=True,
            ):
                arr[n_sens] = v
            n_sens += 1
        elif isinstance(spec, ActuatorFault):
            if n_act == N_ACTUATOR_SLOTS:
                raise ValueError(f"at most {N_ACTUATOR_SLOTS} actuator faults per scenario")
            for arr, v in zip(
                acts,
                (ACTUATOR_KINDS.index(spec.kind), t0, SURFACE_INDEX[spec.channel], spec.magnitude),
                strict=True,
            ):
                arr[n_act] = v
            n_act += 1
        else:
            name, kinds = families[type(spec)]
            if int(scalars[name].kind) != 0:
                raise ValueError(f"only one {name} fault per scenario")
            ramp = getattr(spec, "ramp_s", 0.0)
            side = getattr(spec, "side", 1)
            scalars[name] = ScalarFaultSlot(
                np.array(kinds.index(spec.kind), np.int32),
                t0,
                np.array(ramp),
                np.array(spec.magnitude),
                np.array(float(side)),
            )
    return FaultConfig(
        pitot,
        ServoJamFault(*jam_arrays),
        cut,
        SensorFaultSlots(*sens),
        ActuatorFaultSlots(*acts),
        scalars["propulsion"],
        scalars["energy"],
        scalars["airframe"],
    )


def active_kind(kind: ArrayLike, onset_s: ArrayLike, t: ArrayLike) -> Array:
    """The slot's kind index once ``t`` reaches its onset, else 0."""
    return jnp.where(jnp.asarray(t) >= onset_s, kind, 0)


def ramp(onset_s: ArrayLike, ramp_s: ArrayLike, t: ArrayLike) -> Array:
    """0 before onset, rising linearly to 1 over ``ramp_s`` (a step when ``ramp_s`` = 0)."""
    t, onset_s = jnp.asarray(t), jnp.asarray(onset_s)
    return jnp.where(
        t >= onset_s, jnp.clip((t - onset_s) / jnp.maximum(ramp_s, 1e-9), 0.0, 1.0), 0.0
    )


def fault_labels(cfg: FaultConfig, t: Array) -> FaultLabels:
    return FaultLabels(
        pitot=jnp.logical_and(cfg.pitot.active, t >= cfg.pitot.onset_s),
        servo_jam=jnp.logical_and(cfg.servo_jam.active, t >= cfg.servo_jam.onset_s),
        motor_cut=jnp.logical_and(cfg.motor_cut.active, t >= cfg.motor_cut.onset_s),
        sensor_kind=active_kind(cfg.sensors.kind, cfg.sensors.onset_s, t),
        actuator_kind=active_kind(cfg.actuators.kind, cfg.actuators.onset_s, t),
        propulsion_kind=active_kind(cfg.propulsion.kind, cfg.propulsion.onset_s, t),
        energy_kind=active_kind(cfg.energy.kind, cfg.energy.onset_s, t),
        airframe_kind=active_kind(cfg.airframe.kind, cfg.airframe.onset_s, t),
    )
