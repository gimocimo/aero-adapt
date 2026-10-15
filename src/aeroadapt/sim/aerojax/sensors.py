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
from aeroadapt.faultlab.slots import (
    N_SENSOR_SLOTS,
    PROPULSION_KINDS,
    SENSOR_KINDS,
    PitotFault,
    ScalarFaultSlot,
    SensorFaultSlots,
    active_kind,
    ramp,
)
from aeroadapt.sim.aerojax import noise

GNSS_HISTORY = 32  # control steps of truth history (320 ms at 100 Hz) for GNSS latency
RHO_INDICATED = 1.225  # indicated airspeed is referred to sea-level density
ACCEL_CLIP = 16 * 9.806_65  # IMU range: vibration saturates here
PROP_VIB_REF = 600.0  # rad/s: propeller speed at which damage vibration reaches its magnitude
_S = {name: i for i, name in enumerate(SENSOR_KINDS)}
_P = {name: i for i, name in enumerate(PROPULSION_KINDS)}


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
    fault_latched: Array  # (N_SENSOR_SLOTS,) bool: a stuck value has been captured
    fault_hold: Array  # (N_SENSOR_SLOTS,) the captured value (stuck airspeed: Pa)


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
    sf: SensorFaultSlots | None = None,
    prop: ScalarFaultSlot | None = None,
) -> tuple[SensorState, SensorArrays]:
    """``sf`` and ``prop`` carry faultlab v1 sensor faults and propeller-damage vibration;
    empty slots leave every value unchanged (bit for bit)."""
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
    fault_latched, fault_hold = s.fault_latched, s.fault_hold
    if sf is not None:
        new, fault_latched, fault_hold = _apply_sensor_faults(
            sf, prop, new, s, truth, jax.random.fold_in(key, 7)
        )
        fresh = new.fresh
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
        fault_latched=fault_latched,
        fault_hold=fault_hold,
    )
    return state, out


def _apply_sensor_faults(
    sf: SensorFaultSlots,
    prop: ScalarFaultSlot | None,
    z: SensorArrays,
    s: SensorState,
    truth: SensorTruth,
    key: Array,
) -> tuple[SensorArrays, Array, Array]:
    """faultlab v1 sensor faults on a fresh sample (M2 spec §7.1)."""
    t = truth.t
    dtype = z.accel.dtype
    k_acc, k_gyr, k_vib = jax.random.split(key, 3)
    gyro, accel, mag = z.gyro, z.accel, z.mag
    dp, baro = z.airspeed_dp_pa, z.baro_pa
    pos, vel, fresh = z.gnss_pos_ned, z.gnss_vel_ned, z.fresh
    latched, hold = s.fault_latched, s.fault_hold
    for i in range(N_SENSOR_SLOTS):
        kind = active_kind(sf.kind[i], sf.onset_s[i], t)
        r = ramp(sf.onset_s[i], sf.ramp_s[i], t)
        m = sf.magnitude[i]
        dt_on = jnp.maximum(t - sf.onset_s[i], 0.0)
        e = jnp.arange(3) == sf.axis[i]

        def is_(name: str, kind: Array = kind) -> Array:
            return kind == _S[name]

        # Airspeed: offsets are in m/s of indicated airspeed; "stuck" freezes the pressure.
        v_ind = jnp.sqrt(2.0 * jnp.maximum(dp, 0.0) / RHO_INDICATED)
        bias = jnp.where(
            is_("airspeed_bias"), m * r, jnp.where(is_("airspeed_drift"), m * dt_on, 0.0)
        )
        biased = 0.5 * RHO_INDICATED * jnp.maximum(v_ind + bias, 0.0) ** 2
        dp = jnp.where(is_("airspeed_bias") | is_("airspeed_drift"), biased, dp)
        newly = is_("airspeed_stuck") & ~latched[i]
        hold = hold.at[i].set(jnp.where(newly, dp, hold[i]))
        latched = latched.at[i].set(latched[i] | is_("airspeed_stuck"))
        dp = jnp.where(is_("airspeed_stuck"), hold[i], dp)
        # Inertial and magnetic.
        gyro = gyro + jnp.where(is_("gyro_bias") & e, m * r, 0.0)
        accel = accel + jnp.where(is_("accel_bias") & e, m * r, 0.0)
        vib = jnp.where(is_("imu_vibration"), m * r, 0.0)
        accel = accel + vib * noise.normal(jax.random.fold_in(k_acc, i), (3,), dtype)
        gyro = gyro + 0.02 * vib * noise.normal(jax.random.fold_in(k_gyr, i), (3,), dtype)
        mag = mag + jnp.where(is_("mag_interference") & e, m * r, 0.0)
        baro = baro + jnp.where(is_("baro_drift"), m * dt_on, 0.0)
        # GNSS.
        dropout = is_("gnss_dropout") & ((m <= 0.0) | (dt_on < m))
        fresh = fresh._replace(gnss=fresh.gnss & ~dropout)
        pos = pos + jnp.where(is_("gnss_drift") & e, m * dt_on, 0.0)
        pos = pos + jnp.where(is_("gnss_spoof") & e, 0.5 * m * dt_on**2, 0.0)
        vel = vel + jnp.where(is_("gnss_spoof") & e, m * dt_on, 0.0)
    if prop is not None:
        damaged = active_kind(prop.kind, prop.onset_s, t) == _P["prop_damage"]
        amp = jnp.where(
            damaged, 20.0 * prop.magnitude * (truth.prop_omega / PROP_VIB_REF) ** 2, 0.0
        )
        accel = accel + amp * noise.normal(k_vib, (3,), dtype)
    accel = jnp.where(jnp.abs(accel) > ACCEL_CLIP, jnp.sign(accel) * ACCEL_CLIP, accel)
    z = z._replace(
        gyro=gyro, accel=accel, mag=mag, airspeed_dp_pa=dp, baro_pa=baro,
        gnss_pos_ned=pos, gnss_vel_ned=vel, fresh=fresh,
    )  # fmt: skip
    return z, latched, hold


def init_sensors(
    sp: SensorParams,
    pf: PitotFault,
    truth: SensorTruth,
    dt_c: float,
    control_hz: float,
    key: Array,
    sf: SensorFaultSlots | None = None,
    prop: ScalarFaultSlot | None = None,
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
        fault_latched=jnp.zeros(N_SENSOR_SLOTS, bool),
        fault_hold=jnp.zeros(N_SENSOR_SLOTS, dtype),
    )
    return sample(sp, pf, s0, truth, dt_c, control_hz, k4, sf, prop)
