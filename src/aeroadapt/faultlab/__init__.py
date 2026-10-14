"""faultlab: fault specifications, fixed-shape fault slots and ground-truth labels (ADR-003).

Privileged: the flight stack must never import this package (ADR-006).
"""

from aeroadapt.faultlab.schema import FaultSpec, MotorCut, PitotBlockage, ServoJam, TimeTrigger
from aeroadapt.faultlab.slots import (
    SURFACE_INDEX,
    FaultConfig,
    FaultLabels,
    build_fault_config,
    fault_labels,
    no_faults,
)

__all__ = [
    "SURFACE_INDEX",
    "FaultConfig",
    "FaultLabels",
    "FaultSpec",
    "MotorCut",
    "PitotBlockage",
    "ServoJam",
    "TimeTrigger",
    "build_fault_config",
    "fault_labels",
    "no_faults",
]
