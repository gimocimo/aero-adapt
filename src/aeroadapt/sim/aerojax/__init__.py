"""aerojax-UAV: a pure-functional JAX 6-DoF fixed-wing simulator (ADR-002)."""

from aeroadapt.sim.aerojax.environment import BM_PARITY, Environment, EnvParams, make_env
from aeroadapt.sim.aerojax.model import ForcesMoments, forces_moments, state_derivative, step
from aeroadapt.sim.aerojax.rollout import rollout
from aeroadapt.sim.aerojax.sim import SimConfig, SimState, StepInfo, Truth, init_state, sim_step
from aeroadapt.sim.aerojax.state import (
    RigidBodyState,
    SurfaceControls,
    controls_from_vector,
    state_from_vector,
    state_to_vector,
)

__all__ = [
    "BM_PARITY",
    "EnvParams",
    "Environment",
    "ForcesMoments",
    "RigidBodyState",
    "SimConfig",
    "SimState",
    "StepInfo",
    "SurfaceControls",
    "Truth",
    "controls_from_vector",
    "forces_moments",
    "init_state",
    "make_env",
    "rollout",
    "sim_step",
    "state_derivative",
    "state_from_vector",
    "state_to_vector",
    "step",
]
