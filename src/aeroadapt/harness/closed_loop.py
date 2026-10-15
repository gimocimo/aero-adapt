"""Batched closed loop: simulator + flight stack inside one ``lax.scan`` (spec §5.4).

The harness sits on the privileged side (ADR-006): it holds the simulator state and logs
truth next to the stack's estimates. The stack itself only ever receives ``SensorArrays``,
the same arrays the host-stepped ``AerojaxVehicle`` path converts to and from
``SensorFrame`` messages. ``closed_loop`` is pure, so it vmaps over seeds, faults and
environments.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from aeroadapt.airframes import Airframe, VehicleParams, to_vehicle_params
from aeroadapt.core import quaternion
from aeroadapt.core.messages import SensorArrays
from aeroadapt.faultlab import FaultConfig, no_faults
from aeroadapt.scenarios import Site
from aeroadapt.sim.aerojax import EnvParams, SimConfig, SimState, init_state, sim_step
from aeroadapt.stack.autopilot import MODE_AUTO, StackState, stack_init, stack_step
from aeroadapt.stack.config import StackParams, build_stack_params, load_stack_cfg
from aeroadapt.stack.types import Mission, Target
from aeroadapt.vehicle.aerojax_vehicle import trimmed_start
from aeroadapt.vehicle.protocol import InitialCondition

GuidedFn = Callable[[Array], Target]


class LoopLog(NamedTuple):
    """One row per control step (after the step)."""

    t: Array
    # truth
    pos_ned: Array
    vel_ned: Array
    euler: Array  # roll, pitch, yaw
    omega: Array
    va: Array
    alpha: Array
    beta: Array
    wind_ned: Array
    throttle_actual: Array
    motor_cut: Array  # fault label
    done: Array
    termination: Array
    # stack
    cmd: Array
    est_euler: Array
    est_pos_ned: Array
    est_vel_ned: Array
    est_airspeed: Array
    est_wind_ned: Array
    airspeed_healthy: Array
    roll_dem: Array
    pitch_dem: Array
    alt_dem: Array  # TECS (rate-limited) height demand
    speed_dem: Array
    xtrack: Array
    mode: Array
    wp_idx: Array
    failed: Array
    phase: Array
    zone: Array


@partial(jax.jit, static_argnames=("n_steps", "guided", "config"))
def closed_loop(
    vp: VehicleParams,
    env: EnvParams,
    faults: FaultConfig,
    sp: StackParams,
    mission: Mission,
    sim0: SimState,
    stack0: StackState,
    z0: SensorArrays,
    key: Array,
    *,
    n_steps: int,
    guided: GuidedFn | None = None,
    config: SimConfig = SimConfig(),  # noqa: B008
) -> tuple[SimState, StackState, LoopLog]:
    def body(carry: tuple[SimState, StackState, SensorArrays], k: Array):
        sim, st, z = carry
        st, out = stack_step(sp, st, z, mission, None if guided is None else guided(sim.t))
        sim, z, info = sim_step(vp, env, faults, sim, out.cmd, k, config)
        rb, tr = sim.rb, info.truth
        log = LoopLog(
            t=sim.t,
            pos_ned=rb.pos_ned,
            vel_ned=quaternion.body_to_ned(rb.quat, rb.vel_body),
            euler=jnp.stack(quaternion.to_euler(rb.quat)),
            omega=rb.omega,
            va=tr.va,
            alpha=tr.alpha,
            beta=tr.beta,
            wind_ned=tr.wind_ned,
            throttle_actual=tr.throttle,
            motor_cut=info.labels.motor_cut,
            done=sim.done,
            termination=sim.termination,
            cmd=out.cmd,
            est_euler=jnp.stack([out.est.roll, out.est.pitch, out.est.yaw]),
            est_pos_ned=out.est.pos_ned,
            est_vel_ned=out.est.vel_ned,
            est_airspeed=out.est.airspeed,
            est_wind_ned=out.est.wind_ned,
            airspeed_healthy=out.est.airspeed_healthy,
            roll_dem=out.roll_dem,
            pitch_dem=out.pitch_dem,
            alt_dem=st.tecs.hgt_dem,
            speed_dem=st.tecs.spd_dem,
            xtrack=out.xtrack,
            mode=out.mode,
            wp_idx=out.wp_idx,
            failed=out.cont.active,
            phase=out.cont.phase,
            zone=out.cont.zone,
        )
        return (sim, st, z), log

    keys = jax.random.split(key, n_steps)
    (sim, st, _), log = jax.lax.scan(body, (sim0, stack0, z0), keys)
    return sim, st, log


@dataclass(frozen=True)
class LoopSetup:
    """Everything ``closed_loop`` needs, built on the host from configs."""

    vp: VehicleParams
    env: EnvParams
    faults: FaultConfig
    sp: StackParams
    sim0: SimState
    stack0: StackState
    z0: SensorArrays
    config: SimConfig

    def run(
        self, mission: Mission, n_steps: int, key: Array, guided: GuidedFn | None = None
    ) -> tuple[SimState, StackState, LoopLog]:
        return closed_loop(
            self.vp,
            self.env,
            self.faults,
            self.sp,
            mission,
            self.sim0,
            self.stack0,
            self.z0,
            key,
            n_steps=n_steps,
            guided=guided,
            config=self.config,
        )


def setup_loop(
    airframe: Airframe,
    site: Site,
    env: EnvParams,
    initial: InitialCondition,
    *,
    faults: FaultConfig | None = None,
    stack: str | StackParams = "aerosonde_bm",
    mode: int = MODE_AUTO,
    seed: int = 0,
    config: SimConfig = SimConfig(),  # noqa: B008
) -> LoopSetup:
    vp = to_vehicle_params(airframe)
    fc = no_faults() if faults is None else faults
    sp = (
        build_stack_params(load_stack_cfg(stack), airframe, site, config.dt_control)
        if isinstance(stack, str)
        else stack
    )
    rb0, cmd0 = trimmed_start(airframe, env, initial)
    sim0, z0 = init_state(
        vp, env, fc, rb0, jnp.asarray(cmd0), jax.random.PRNGKey(seed), initial.soc, config
    )
    stack0 = stack_init(sp, z0, mode)
    vp, env, fc, sp = jax.device_put((vp, env, fc, sp))
    return LoopSetup(vp, env, fc, sp, sim0, stack0, z0, config)
