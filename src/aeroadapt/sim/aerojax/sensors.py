"""Sensor models (spec §6.7), sampled once per control step from the true state.

Multi-rate: each sensor's rate becomes an integer decimation of the control rate. A sensor
that is not due repeats its last value with ``fresh = False`` (sample-and-hold), like
MAVLink/ArduPilot streams. GNSS positions are produced in the local NED frame (float32-safe);
the imperative shell converts them to latitude/longitude in float64 (ADR-001).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from aeroadapt.airframes.params import SensorParams
from aeroadapt.core import quaternion
from aeroadapt.core.messages import FreshFlags, SensorArrays
from aeroadapt.faultlab.slots import PitotFault
from aeroadapt.sim.aerojax import noise

GNSS_HISTORY = 32  # control steps of truth history (320 ms at 100 Hz) for GNSS latency


class SensorTruth(NamedTuple):
    """True quantities the sensor models need (computed by the simulator)."""

    t: Array
    k: Array  # control step index
    pos_ned: Array
    vel_ned: Array
    quat: Array
    omega: Array
    specific_force: Array  # (3,) body
    static_pa: Array
    temp_k: Array
    rho: Array
    u_air: Array  # axial (body-x) air-relative speed seen by the pitot
    battery_v: Array
    battery_a: Array
    battery_mah: Array
    prop_omega: Array
    mag_ned: Array  # (3,) Earth field, gauss


class SensorState(NamedTuple):
    gyro_bias: Array
    accel_bias: Array
    baro_drift: Array
    gnss_err: Array  # (3,) Gauss-Markov position error
    gnss_hist: Array  # (GNSS_HISTORY, 6) delayed truth: pos, vel
    gnss_head: Array  # () int32
    pitot_latched: Array  # () bool
    pitot_pt: Array  # () Pa, trapped/leaking total pressure under a pitot fault
    held: SensorArrays


def _decimation(rate_hz: Array, control_hz: float) -> Array:
    return jnp.maximum(jnp.round(control_hz / rate_hz), 1).astype(jnp.int32)


def _fresh(sp: SensorParams, k: Array, control_hz: float) -> FreshFlags:
    due = lambda rate: (k % _decimation(rate, control_hz)) == 0  # noqa: E731
    return FreshFlags(
        imu=due(sp.imu_rate),
        baro=due(sp.baro_rate),
        airspeed=due(sp.airspeed_rate),
        gnss=due(sp.gnss_rate),
        mag=due(sp.mag_rate),
        battery=due(sp.battery_rate),
        esc=due(sp.esc_rate),
    )


def pitot_dp(
    pf: PitotFault, s: SensorState, truth: SensorTruth, dt_c: float
) -> tuple[Array, Array, Array]:
    """True or faulted differential pressure → (dp, latched, total pressure state)."""
    dp_true = 0.5 * truth.rho * jnp.maximum(truth.u_air, 0.0) ** 2
    fault_now = jnp.logical_and(pf.active, truth.t >= pf.onset_s)
    newly = jnp.logical_and(fault_now, ~s.pitot_latched)
    pt = jnp.where(newly, truth.static_pa + dp_true, s.pitot_pt)
    leak = jnp.logical_and(jnp.logical_and(fault_now, ~newly), ~pf.drain_blocked)
    pt = jnp.where(leak, pt + (truth.static_pa - pt) * (1 - jnp.exp(-dt_c / pf.tau_s)), pt)
    blend = jnp.where(
        pf.ramp_s > 0, jnp.clip((truth.t - pf.onset_s) / jnp.maximum(pf.ramp_s, 1e-9), 0, 1), 1.0
    )
    dp_fault = pt - truth.static_pa
    dp = jnp.where(fault_now, (1 - blend) * dp_true + blend * dp_fault, dp_true)
    return dp, jnp.logical_or(s.pitot_latched, fault_now), pt


def sample(
    sp: SensorParams,
    pf: PitotFault,
    s: SensorState,
    truth: SensorTruth,
    dt_c: float,
    control_hz: float,
    key: Array,
) -> tuple[SensorState, SensorArrays]:
    ks = jax.random.split(key, 12)
    dtype = truth.pos_ned.dtype
    n3 = lambda i: noise.normal(ks[i], (3,), dtype)  # noqa: E731
    n1 = lambda i: noise.normal(ks[i], (), dtype)  # noqa: E731
    sq = jnp.sqrt(dt_c)
    fresh = _fresh(sp, truth.k, control_hz)

    # IMU: biases are random walks.
    gyro_bias = s.gyro_bias + sp.gyro_bias_rw * sq * n3(0)
    accel_bias = s.accel_bias + sp.accel_bias_rw * sq * n3(1)
    gyro = truth.omega + gyro_bias + sp.gyro_noise * n3(2)
    accel = truth.specific_force + accel_bias + sp.accel_noise * n3(3)

    # Barometer: static pressure with a slow drift.
    baro_drift = s.baro_drift + sp.baro_drift_rw * sq * n1(4)
    baro = truth.static_pa + baro_drift + sp.baro_noise * n1(5)

    # Pitot-static differential pressure (with blockage faults).
    dp, pitot_latched, pitot_pt = pitot_dp(pf, s, truth, dt_c)
    dp = dp + sp.airspeed_noise * n1(6)

    # GNSS: delayed truth + Gauss-Markov position error + white velocity noise.
    hist = s.gnss_hist.at[s.gnss_head].set(jnp.concatenate([truth.pos_ned, truth.vel_ned]))
    lat_steps = jnp.round(sp.gnss_latency * control_hz).astype(jnp.int32)
    delayed = hist[(s.gnss_head - lat_steps) % GNSS_HISTORY]
    a = jnp.exp(-dt_c / sp.gnss_tau)
    sigma = jnp.array([sp.gnss_sigma_h, sp.gnss_sigma_h, sp.gnss_sigma_v], dtype)
    gnss_err = a * s.gnss_err + sigma * jnp.sqrt(1 - a**2) * n3(7)
    gnss_pos = delayed[:3] + gnss_err
    gnss_vel = delayed[3:] + sp.gnss_vel_sigma * n3(8)

    mag = quaternion.ned_to_body(truth.quat, truth.mag_ned) + sp.mag_noise * n3(9)
    esc_rpm = truth.prop_omega * 60.0 / (2 * jnp.pi) * (1 + sp.esc_rpm_noise_frac * n1(10))
    bnoise = noise.normal(ks[11], (2,), dtype)

    new = SensorArrays(
        gyro=gyro,
        accel=accel,
        baro_pa=baro,
        baro_temp_k=truth.temp_k,
        airspeed_dp_pa=dp,
        airspeed_temp_k=truth.temp_k,
        gnss_pos_ned=gnss_pos,
        gnss_vel_ned=gnss_vel,
        gnss_time_s=truth.t - lat_steps * dt_c,
        mag=mag,
        battery_v=truth.battery_v + sp.battery_v_noise * bnoise[0],
        battery_a=truth.battery_a + sp.battery_i_noise * bnoise[1],
        battery_mah=truth.battery_mah,
        esc_rpm=esc_rpm,
        fresh=fresh,
    )
    h = s.held
    out = SensorArrays(
        gyro=jnp.where(fresh.imu, new.gyro, h.gyro),
        accel=jnp.where(fresh.imu, new.accel, h.accel),
        baro_pa=jnp.where(fresh.baro, new.baro_pa, h.baro_pa),
        baro_temp_k=jnp.where(fresh.baro, new.baro_temp_k, h.baro_temp_k),
        airspeed_dp_pa=jnp.where(fresh.airspeed, new.airspeed_dp_pa, h.airspeed_dp_pa),
        airspeed_temp_k=jnp.where(fresh.airspeed, new.airspeed_temp_k, h.airspeed_temp_k),
        gnss_pos_ned=jnp.where(fresh.gnss, new.gnss_pos_ned, h.gnss_pos_ned),
        gnss_vel_ned=jnp.where(fresh.gnss, new.gnss_vel_ned, h.gnss_vel_ned),
        gnss_time_s=jnp.where(fresh.gnss, new.gnss_time_s, h.gnss_time_s),
        mag=jnp.where(fresh.mag, new.mag, h.mag),
        battery_v=jnp.where(fresh.battery, new.battery_v, h.battery_v),
        battery_a=jnp.where(fresh.battery, new.battery_a, h.battery_a),
        battery_mah=jnp.where(fresh.battery, new.battery_mah, h.battery_mah),
        esc_rpm=jnp.where(fresh.esc, new.esc_rpm, h.esc_rpm),
        fresh=fresh,
    )
    state = SensorState(
        gyro_bias=gyro_bias,
        accel_bias=accel_bias,
        baro_drift=baro_drift,
        gnss_err=gnss_err,
        gnss_hist=hist,
        gnss_head=(s.gnss_head + 1) % GNSS_HISTORY,
        pitot_latched=pitot_latched,
        pitot_pt=pitot_pt,
        held=out,
    )
    return state, out


def init_sensors(
    sp: SensorParams,
    pf: PitotFault,
    truth: SensorTruth,
    dt_c: float,
    control_hz: float,
    key: Array,
) -> tuple[SensorState, SensorArrays]:
    """Sample turn-on biases and the stationary GNSS error, prime the GNSS history with the
    initial truth, then take a first sample (k = 0, so every sensor is fresh)."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    dtype = truth.pos_ned.dtype
    zeros3 = jnp.zeros(3, dtype)
    sigma = jnp.array([sp.gnss_sigma_h, sp.gnss_sigma_h, sp.gnss_sigma_v], dtype)
    z = jnp.zeros((), dtype)
    no = jnp.zeros((), bool)
    blank = SensorArrays(
        gyro=zeros3,
        accel=zeros3,
        baro_pa=z,
        baro_temp_k=z,
        airspeed_dp_pa=z,
        airspeed_temp_k=z,
        gnss_pos_ned=zeros3,
        gnss_vel_ned=zeros3,
        gnss_time_s=z,
        mag=zeros3,
        battery_v=z,
        battery_a=z,
        battery_mah=z,
        esc_rpm=z,
        fresh=FreshFlags(imu=no, baro=no, airspeed=no, gnss=no, mag=no, battery=no, esc=no),
    )
    s0 = SensorState(
        gyro_bias=sp.gyro_bias_init * noise.normal(k1, (3,), dtype),
        accel_bias=sp.accel_bias_init * noise.normal(k2, (3,), dtype),
        baro_drift=jnp.zeros((), dtype),
        gnss_err=sigma * noise.normal(k3, (3,), dtype),
        gnss_hist=jnp.broadcast_to(
            jnp.concatenate([truth.pos_ned, truth.vel_ned]), (GNSS_HISTORY, 6)
        ),
        gnss_head=jnp.zeros((), jnp.int32),
        pitot_latched=jnp.zeros((), bool),
        pitot_pt=truth.static_pa,
        held=blank,
    )
    return sample(sp, pf, s0, truth, dt_c, control_hz, k4)
