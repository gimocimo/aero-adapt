"""Batched closed loop: simulator + flight stack inside one ``lax.scan`` (spec §5.4).

The harness sits on the privileged side (ADR-006): it holds the simulator state and logs
truth next to the stack's estimates. The stack itself only ever receives ``SensorArrays``,
the same arrays the host-stepped ``AerojaxVehicle`` path converts to and from
``SensorFrame`` messages. ``closed_loop`` is pure, so it vmaps over seeds, faults and
environments.

The log is what the recorder writes (spec §10.3). ``log_every`` keeps every k-th step only:
campaigns of many flights read their outcomes from the final states and keep a sparse
trajectory for maps.
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
from aeroadapt.faultlab import FaultConfig, FaultLabels, no_faults
from aeroadapt.scenarios import Site
from aeroadapt.sim.aerojax import EnvParams, SimConfig, SimState, init_state, sim_step
from aeroadapt.stack.autopilot import MODE_AUTO, StackState, stack_init, stack_step
from aeroadapt.stack.config import StackParams, build_stack_params, load_stack_cfg
from aeroadapt.stack.types import Mission, Target
from aeroadapt.vehicle.aerojax_vehicle import trimmed_start
from aeroadapt.vehicle.protocol import InitialCondition

GuidedFn = Callable[[Array, StackParams], Target]  # (time, stack parameters) → target
Carry = tuple[SimState, StackState, SensorArrays]


class LoopLog(NamedTuple):
    """One row per logged control step (state after the step)."""

    t: Array
    # truth (privileged)
    pos_ned: Array
    vel_ned: Array
    quat: Array  # body → NED, scalar first
    euler: Array  # roll, pitch, yaw
    omega: Array
    va: Array
    alpha: Array
    beta: Array
    wind_ned: Array
    surfaces: Array  # (3,) actual deflections, rad (B&M signs)
    throttle_actual: Array
    thrust: Array
    battery_v: Array
    soc: Array
    prop_rpm: Array
    labels: FaultLabels
    motor_cut: Array  # = labels.motor_cut (kept for convenience)
    done: Array
    termination: Array
    # sensors (what the stack received at this step)
    sensors: SensorArrays
    # stack
    cmd: Array
    est_euler: Array
    est_pos_ned: Array
    est_vel_ned: Array
    est_ias: Array
    est_airspeed: Array  # used by the controllers: TAS, or synthetic if unhealthy
    est_wind_ned: Array
    airspeed_healthy: Array
    roll_dem: Array
    pitch_dem: Array
    alt_target: Array
    speed_target: Array
    alt_dem: Array  # TECS (rate-limited, lagged) height demand
    speed_dem: Array
    lat_acc: Array
    xtrack: Array
    mode: Array
    wp_idx: Array
    failed: Array
    phase: Array
    zone: Array
    zone_margin: Array  # (Z,) reachability height margin per safe zone
    speed_to_fly: Array


@partial(jax.jit, static_argnames=("n_steps", "guided", "config", "log_every"))
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
    log_every: int = 1,
) -> tuple[SimState, StackState, LoopLog]:
    """``n_steps`` control steps (a multiple of ``log_every``); logs every ``log_every``-th."""
    sim, st, _, log = closed_loop_carry(
        vp, env, faults, sp, mission, sim0, stack0, z0, key,
        n_steps=n_steps, guided=guided, config=config, log_every=log_every,
    )  # fmt: skip
    return sim, st, log


@partial(jax.jit, static_argnames=("n_steps", "guided", "config", "log_every"))
def closed_loop_carry(
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
    log_every: int = 1,
) -> tuple[SimState, StackState, SensorArrays, LoopLog]:
    """As :func:`closed_loop`, also returning the last sensor sample, so a flight can be
    continued chunk by chunk (live mode)."""
    if n_steps % log_every:
        raise ValueError("n_steps must be a multiple of log_every")

    def step(carry: Carry, k: Array) -> tuple[Carry, LoopLog]:
        sim, st, z_in = carry
        st, out = stack_step(sp, st, z_in, mission, None if guided is None else guided(sim.t, sp))
        sim, z, info = sim_step(vp, env, faults, sim, out.cmd, k, config)
        rb, tr = sim.rb, info.truth
        e, c = out.est, out.cont
        log = LoopLog(
            t=sim.t,
            pos_ned=rb.pos_ned,
            vel_ned=quaternion.body_to_ned(rb.quat, rb.vel_body),
            quat=rb.quat,
            euler=jnp.stack(quaternion.to_euler(rb.quat)),
            omega=rb.omega,
            va=tr.va,
            alpha=tr.alpha,
            beta=tr.beta,
            wind_ned=tr.wind_ned,
            surfaces=tr.surfaces,
            throttle_actual=tr.throttle,
            thrust=tr.thrust,
            battery_v=tr.battery_v,
            soc=tr.soc,
            prop_rpm=tr.prop_omega * (60.0 / (2.0 * jnp.pi)),
            labels=info.labels,
            motor_cut=info.labels.motor_cut,
            done=sim.done,
            termination=sim.termination,
            sensors=z_in,
            cmd=out.cmd,
            est_euler=jnp.stack([e.roll, e.pitch, e.yaw]),
            est_pos_ned=e.pos_ned,
            est_vel_ned=e.vel_ned,
            est_ias=e.ias,
            est_airspeed=e.airspeed,
            est_wind_ned=e.wind_ned,
            airspeed_healthy=e.airspeed_healthy,
            roll_dem=out.roll_dem,
            pitch_dem=out.pitch_dem,
            alt_target=out.target.alt,
            speed_target=out.target.speed,
            alt_dem=st.tecs.hgt_dem,
            speed_dem=st.tecs.spd_dem,
            lat_acc=out.lat_acc,
            xtrack=out.xtrack,
            mode=out.mode,
            wp_idx=out.wp_idx,
            failed=c.active,
            phase=c.phase,
            zone=c.zone,
            zone_margin=c.margin,
            speed_to_fly=c.speed_to_fly,
        )
        return (sim, st, z), log

    keys = jax.random.split(key, n_steps)
    carry0: Carry = (sim0, stack0, z0)
    if log_every == 1:
        (sim, st, z), log = jax.lax.scan(step, carry0, keys)
        return sim, st, z, log

    def chunk(carry: Carry, ks: Array) -> tuple[Carry, LoopLog]:
        carry, logs = jax.lax.scan(step, carry, ks)
        return carry, jax.tree.map(lambda x: x[-1], logs)

    (sim, st, z), log = jax.lax.scan(chunk, carry0, keys.reshape(-1, log_every, *keys.shape[1:]))
    return sim, st, z, log


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
        self,
        mission: Mission,
        n_steps: int,
        key: Array,
        guided: GuidedFn | None = None,
        log_every: int = 1,
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
            log_every=log_every,
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
