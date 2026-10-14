"""Fixed-shape fault slots (ADR-003).

Each fault type owns one slot of fixed shape, so a single compiled simulator can run
thousands of scenarios with *different* faults under ``vmap``. An inactive slot acts as an
identity transform via masks; nothing branches in Python on fault state. Triggers are
resolved to onset times here, on the host.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from aeroadapt.faultlab.schema import FaultSpec, MotorCut, PitotBlockage, ServoJam

SURFACE_INDEX = {"aileron": 0, "elevator": 1, "rudder": 2}


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


class FaultConfig(NamedTuple):
    pitot: PitotFault
    servo_jam: ServoJamFault
    motor_cut: MotorCutFault


class FaultLabels(NamedTuple):
    """Ground truth emitted every step (diagnosis training data, ADR-003)."""

    pitot: Array  # () bool
    servo_jam: Array  # (3,) bool
    motor_cut: Array  # () bool


def no_faults() -> FaultConfig:
    f, z = np.array(False), np.array(0.0)
    return FaultConfig(
        pitot=PitotFault(f, f, z, z, np.array(3.0)),
        servo_jam=ServoJamFault(np.zeros(3, bool), np.zeros(3), np.zeros(3), np.zeros(3, bool)),
        motor_cut=MotorCutFault(f, z),
    )


def build_fault_config(specs: Sequence[FaultSpec]) -> FaultConfig:
    cfg = no_faults()
    pitot, jam, cut = cfg.pitot, cfg.servo_jam, cfg.motor_cut
    jam_arrays = [a.copy() for a in jam]
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
    return FaultConfig(pitot, ServoJamFault(*jam_arrays), cut)


def fault_labels(cfg: FaultConfig, t: Array) -> FaultLabels:
    return FaultLabels(
        pitot=jnp.logical_and(cfg.pitot.active, t >= cfg.pitot.onset_s),
        servo_jam=jnp.logical_and(cfg.servo_jam.active, t >= cfg.servo_jam.onset_s),
        motor_cut=jnp.logical_and(cfg.motor_cut.active, t >= cfg.motor_cut.onset_s),
    )
