"""aerojax-UAV: a pure-functional JAX 6-DoF fixed-wing simulator (ADR-002)."""

from aeroadapt.sim.aerojax.environment import BM_PARITY, Environment
from aeroadapt.sim.aerojax.model import ForcesMoments, forces_moments, state_derivative, step
from aeroadapt.sim.aerojax.rollout import rollout
from aeroadapt.sim.aerojax.state import (
    RigidBodyState,
    SurfaceControls,
    controls_from_vector,
    state_from_vector,
    state_to_vector,
)

__all__ = [
    "BM_PARITY",
    "Environment",
    "ForcesMoments",
    "RigidBodyState",
    "SurfaceControls",
    "controls_from_vector",
    "forces_moments",
    "rollout",
    "state_derivative",
    "state_from_vector",
    "state_to_vector",
    "step",
]
