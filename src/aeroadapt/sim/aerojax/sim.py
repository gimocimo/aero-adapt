"""The assembled simulator: one pure ``sim_step`` per control period (ADR-002).

Per control step (default 100 Hz): ``substeps`` physics steps (default 2 x 5 ms), each one
advancing actuators → atmosphere and turbulence → RK4 rigid body → battery and propeller;
then ground-contact/termination checks; then the sensor models. Faults enter through fixed-
shape slots (ADR-003) and ground-truth labels are emitted every step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from aeroadapt.airframes.params import VehicleParams
from aeroadapt.core import quaternion
from aeroadapt.faultlab.slots import FaultConfig, FaultLabels, fault_labels
from aeroadapt.sim.aerojax.actuators import ActuatorState, actuator_step, init_actuators
from aeroadapt.sim.aerojax.aero import air_data
from aeroadapt.sim.aerojax.atmosphere import (
    DrydenState,
    dryden_stationary,
    dryden_step,
    isa,
    steady_wind,
)
from aeroadapt.sim.aerojax.battery import (
    BatteryState,
    battery_step,
    init_battery,
    terminal_voltage,
)
from aeroadapt.sim.aerojax.environment import Environment, EnvParams
from aeroadapt.sim.aerojax.model import forces_moments, step
from aeroadapt.sim.aerojax.propulsion import propeller
from aeroadapt.sim.aerojax.sensors import (
    SensorArrays,
    SensorState,
    SensorTruth,
    init_sensors,
    sample,
)
from aeroadapt.sim.aerojax.state import RigidBodyState, SurfaceControls

MOTOR_SPIN_DOWN_S = 0.3  # rpm decay time constant after a motor cut (spec §8.2)

# Termination codes
RUNNING, GROUND_CONTACT, OUT_OF_BOUNDS, NUMERICAL = 0, 1, 2, 3


@dataclass(frozen=True)
class SimConfig:
    """Static configuration (hashable, passed to ``jit`` as a static argument)."""

    dt_physics: float = 0.005
    substeps: int = 2
    contact_height_m: float = 0.15
    max_range_m: float = 8000.0

    @property
    def dt_control(self) -> float:
        return self.dt_physics * self.substeps

    @property
    def control_hz(self) -> float:
        return 1.0 / self.dt_control


class SimState(NamedTuple):
    t: Array  # () s
    k: Array  # () int32 control step index
    rb: RigidBodyState
    act: ActuatorState
    battery: BatteryState
    prop_omega: Array  # () rad/s
    turb: DrydenState
    gust_body: Array  # (3,) last body-axis gust
    sensors: SensorState
    done: Array  # () bool
    termination: Array  # () int32


class Truth(NamedTuple):
    """Privileged per-step truth for recording, evaluation and the oracle (ADR-006)."""

    va: Array
    alpha: Array
    beta: Array
    alt_agl: Array
    wind_ned: Array  # steady component at the aircraft
    gust_body: Array
    surfaces: Array  # (3,) actual deflections, B&M signs
    throttle: Array
    thrust: Array  # N, actual (windmill drag under a motor cut)
    battery_v: Array
    battery_a: Array
    soc: Array
    prop_omega: Array
    force_body: Array
    moment_body: Array
    rho: Array
    model_validity_ok: Array  # |α| ≤ 30° and |β| ≤ 20°: logged, not terminal


class StepInfo(NamedTuple):
    truth: Truth
    labels: FaultLabels


def _controls(act: ActuatorState) -> SurfaceControls:
    return SurfaceControls(act.surfaces[0], act.surfaces[1], act.surfaces[2], act.throttle)


def _atmosphere(env: EnvParams, rb: RigidBodyState):
    h_agl = -rb.pos_ned[2]
    temp, p_static, rho = isa(env.isa, env.isa.alt_origin_msl_m + h_agl)
    return h_agl, temp, p_static, rho, steady_wind(env.wind, h_agl)


def _physics_substep(
    vp: VehicleParams,
    env: EnvParams,
    faults: FaultConfig,
    s: SimState,
    cmd: Array,
    key: Array,
    dt: float,
) -> SimState:
    af = vp.airframe
    act = actuator_step(af, vp.actuators, faults.servo_jam, s.act, cmd, s.t, dt)
    h_agl, _, _, rho, wind = _atmosphere(env, s.rb)
    va0 = air_data(s.rb, wind, s.gust_body).va
    turb, gust = dryden_step(s.turb, env.turbulence, va0, h_agl, dt, key)

    v_batt = terminal_voltage(vp.battery, s.battery)
    cut = fault_labels(faults, s.t).motor_cut
    u = _controls(act)
    rb = step(af, Environment(rho, env.gravity), s.rb, u, wind, dt, gust, v_batt, cut)

    prop = propeller(af, rho, air_data(s.rb, wind, gust).va, act.throttle, v_batt)
    motor_draw = jnp.maximum(act.throttle * prop.motor_current, 0.0)  # no regeneration
    pack = vp.battery.avionics_current + jnp.where(cut, 0.0, motor_draw)
    omega = jnp.where(cut, s.prop_omega * jnp.exp(-dt / MOTOR_SPIN_DOWN_S), prop.omega)
    return s._replace(
        t=s.t + dt,
        rb=rb,
        act=act,
        battery=battery_step(vp.battery, s.battery, pack, dt),
        prop_omega=omega,
        turb=turb,
        gust_body=gust,
    )


def _sensor_truth(
    vp: VehicleParams, env: EnvParams, faults: FaultConfig, s: SimState
) -> tuple[SensorTruth, Truth]:
    af = vp.airframe
    h_agl, temp, p_static, rho, wind = _atmosphere(env, s.rb)
    cut = fault_labels(faults, s.t).motor_cut
    v_batt = terminal_voltage(vp.battery, s.battery)
    fm = forces_moments(
        af, Environment(rho, env.gravity), s.rb, _controls(s.act), wind, s.gust_body, v_batt, cut
    )
    gravity_body = quaternion.ned_to_body(s.rb.quat, jnp.array([0.0, 0.0, env.gravity]))
    st = SensorTruth(
        t=s.t,
        k=s.k,
        pos_ned=s.rb.pos_ned,
        vel_ned=quaternion.body_to_ned(s.rb.quat, s.rb.vel_body),
        quat=s.rb.quat,
        omega=s.rb.omega,
        specific_force=fm.force_body / af.mass - gravity_body,
        static_pa=p_static,
        temp_k=temp,
        rho=rho,
        u_air=fm.air.v_air_body[0],
        battery_v=v_batt,
        battery_a=s.battery.current,
        battery_mah=s.battery.consumed_mah,
        prop_omega=s.prop_omega,
        mag_ned=env.magnetic_ned_gauss,
    )
    truth = Truth(
        va=fm.air.va,
        alpha=fm.air.alpha,
        beta=fm.air.beta,
        alt_agl=h_agl,
        wind_ned=wind,
        gust_body=s.gust_body,
        surfaces=s.act.surfaces,
        throttle=s.act.throttle,
        thrust=fm.prop.thrust,
        battery_v=v_batt,
        battery_a=s.battery.current,
        soc=s.battery.soc,
        prop_omega=s.prop_omega,
        force_body=fm.force_body,
        moment_body=fm.moment_body,
        rho=rho,
        model_validity_ok=jnp.logical_and(
            jnp.abs(fm.air.alpha) <= jnp.deg2rad(30.0), jnp.abs(fm.air.beta) <= jnp.deg2rad(20.0)
        ),
    )
    return st, truth


def _termination(s: SimState, config: SimConfig) -> Array:
    finite = jnp.all(jnp.isfinite(jnp.concatenate([*s.rb])))
    ground = -s.rb.pos_ned[2] <= config.contact_height_m
    away = jnp.linalg.norm(s.rb.pos_ned[:2]) > config.max_range_m
    return jnp.where(
        ~finite, NUMERICAL, jnp.where(ground, GROUND_CONTACT, jnp.where(away, OUT_OF_BOUNDS, 0))
    ).astype(jnp.int32)


def sim_step(
    vp: VehicleParams,
    env: EnvParams,
    faults: FaultConfig,
    s: SimState,
    cmd: Array,
    key: Array,
    config: SimConfig = SimConfig(),  # noqa: B008  (frozen, hashable)
) -> tuple[SimState, SensorArrays, StepInfo]:
    """Advance one control period with normalised command ``cmd`` = (aileron, elevator,
    rudder, throttle) (ADR-001). Once terminated, the state is frozen at the terminal step."""
    keys = jax.random.split(key, config.substeps + 1)
    nxt = s
    for i in range(config.substeps):
        nxt = _physics_substep(vp, env, faults, nxt, cmd, keys[i], config.dt_physics)
    code = _termination(nxt, config)
    nxt = nxt._replace(k=s.k + 1, done=code != RUNNING, termination=code)
    # A numerical failure keeps the last finite state; any earlier termination freezes all.
    keep_old = jnp.logical_or(s.done, code == NUMERICAL)
    nxt = jax.tree_util.tree_map(lambda old, new: jnp.where(keep_old, old, new), s, nxt)._replace(
        done=jnp.logical_or(s.done, code != RUNNING),
        termination=jnp.where(s.done, s.termination, code),
    )
    st, truth = _sensor_truth(vp, env, faults, nxt)
    sens_state, sensors = sample(
        vp.sensors, faults.pitot, nxt.sensors, st, config.dt_control, config.control_hz, keys[-1]
    )
    nxt = nxt._replace(sensors=sens_state)
    return nxt, sensors, StepInfo(truth, fault_labels(faults, nxt.t))


def init_state(
    vp: VehicleParams,
    env: EnvParams,
    faults: FaultConfig,
    rb0: RigidBodyState,
    cmd0: Array,
    key: Array,
    soc0: float = 1.0,
    config: SimConfig = SimConfig(),  # noqa: B008
) -> tuple[SimState, SensorArrays]:
    """Initial state with actuators settled at ``cmd0``, developed turbulence, and a first
    (all-fresh) sensor sample."""
    k_turb, k_sens = jax.random.split(key)
    dtype = jnp.asarray(rb0.pos_ned).dtype
    # Strong, uniform dtypes: a weak-typed leaf (e.g. a quaternion built from Python floats)
    # would make the first jitted step compile twice.
    rb0 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=dtype), rb0)
    act = init_actuators(vp.airframe, vp.actuators, jnp.asarray(cmd0, dtype))
    h_agl, _, _, rho, wind = _atmosphere(env, rb0)
    va = air_data(rb0, wind).va
    battery = init_battery(jnp.asarray(soc0, dtype))
    prop = propeller(vp.airframe, rho, va, act.throttle, terminal_voltage(vp.battery, battery))
    zero3 = jnp.zeros(3, dtype)
    s = SimState(
        t=jnp.zeros((), dtype),
        k=jnp.zeros((), jnp.int32),
        rb=rb0,
        act=act,
        battery=battery,
        prop_omega=prop.omega,
        turb=dryden_stationary(env.turbulence, va, h_agl, k_turb, dtype),
        gust_body=zero3,
        sensors=None,  # type: ignore[arg-type]  # filled below
        done=jnp.zeros((), bool),
        termination=jnp.zeros((), jnp.int32),
    )
    st, _ = _sensor_truth(vp, env, faults, s)
    config_dt, config_hz = config.dt_control, config.control_hz
    sens_state, sensors = init_sensors(vp.sensors, faults.pitot, st, config_dt, config_hz, k_sens)
    return s._replace(sensors=sens_state), sensors
