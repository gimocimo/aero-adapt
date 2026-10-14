"""Open-loop rollouts with ``lax.scan``. Batch them with ``jax.vmap`` over any argument."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from aeroadapt.airframes.params import AirframeParams
from aeroadapt.sim.aerojax.environment import Environment
from aeroadapt.sim.aerojax.model import step
from aeroadapt.sim.aerojax.state import RigidBodyState, SurfaceControls


def rollout(
    p: AirframeParams,
    env: Environment,
    s0: RigidBodyState,
    controls: SurfaceControls,
    winds_ned: Array,
    dt: float,
) -> RigidBodyState:
    """Integrate ``N`` steps. ``controls`` fields and ``winds_ned`` have leading axis ``N``.

    Returns the stacked states with leading axis ``N + 1``, including ``s0``.
    """

    def body(s: RigidBodyState, inputs: tuple[SurfaceControls, Array]):
        u, w = inputs
        s_next = step(p, env, s, u, w, dt)
        return s_next, s_next

    _, traj = jax.lax.scan(body, s0, (controls, winds_ned))
    return jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a[None], b]), s0, traj)
