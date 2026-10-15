"""Navigator v0: a deliberately "DCM-class" estimator (spec §9.1). Pure JAX (ADR-002).

* **Attitude:** Mahony complementary filter with ArduPilot DCM's GNSS acceleration
  reference. The gyro is integrated and corrected so that the estimated specific force in
  NED, averaged over a sliding window of GNSS samples (~1 s), matches the one implied by
  the GNSS velocity change (Δv/Δt − g). The window trades 0.5 s of lag for 5× less noise
  than a single 0.2 s interval. This holds in any manoeuvre; the classic reference (accelerometer =
  gravity after ω × v_air compensation) is wrong in pull-ups, where the path rate is
  q − α̇, not q. Without fresh GNSS it falls back to that classic reference, with airspeed
  rate and centripetal compensation. Heading comes from the magnetometer only. A gyro-bias
  integrator closes the loop.
* **Position/velocity:** strapdown propagation in NED. GNSS is fused against a *delayed*
  copy of the estimate (it is valid ``latency`` ago); the correction is applied to the
  present and to the history. The barometer is the primary height source.
* **Air data:** IAS/TAS from differential pressure and baro density.
* **Wind:** w = v_ground − TAS·x̂_body (horizontal), low-pass filtered. It is frozen while
  the airspeed sensor is deemed unhealthy.
* **Airspeed health** (approximates ArduPlane's mismatch check): unhealthy when
  |TAS − |v_g − w|| exceeds a threshold for a while; then the synthetic airspeed is used.

Known, deliberate limitation (and a motivation for Sentinel): a *slowly* drifting airspeed
error is absorbed by the wind estimate and can escape the mismatch check, because wind and
airspeed error are confounded at constant heading.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from aeroadapt.core import quaternion
from aeroadapt.core.messages import SensorArrays
from aeroadapt.stack.config import NavParams

HISTORY = 32  # control steps of estimate history for GNSS latency
ALIGN_BOOST = 9.0  # fast-convergence phase after init: gains x (1 + 9·e^(-t/τ))
ALIGN_TAU_S = 2.0
TAS_LP_S = 0.2  # airspeed low-pass before differentiating
TAS_RATE_LP_S = 0.3  # airspeed-rate low-pass
GNSS_REF_TIMEOUT_S = 1.5  # fall back to the accelerometer reference after this long
GNSS_WIN = 6  # GNSS samples kept: the reference compares the newest with the oldest
WIND_TAU_START_FRAC = 0.15  # wind filter starts at 15% of its time constant …
WIND_BOOST_S = 20.0  # … relaxing to it with this time constant
R_AIR = 287.05287
RHO0 = 1.225
E3 = jnp.array([0.0, 0.0, 1.0])


class NavState(NamedTuple):
    t: Array  # time since initialisation
    quat: Array
    gyro_bias: Array
    pos: Array  # NED
    vel: Array  # NED
    hist: Array  # (HISTORY, 6) estimated pos, vel
    v_ins: Array  # (3,) uncorrected inertial velocity: ∫(R·f + g)dt (GNSS accel reference)
    vins_hist: Array  # (HISTORY, 3)
    gbuf_vel: Array  # (GNSS_WIN, 3) recent GNSS velocity samples …
    gbuf_t: Array  # (GNSS_WIN,) … their times of validity (-1 = empty) …
    gbuf_vins: Array  # (GNSS_WIN, 3) … and the inertial velocity at those times
    gbuf_head: Array  # next slot to write
    e_ref_ned: Array  # (3,) attitude error from the last GNSS interval, earth frame
    gps_age: Array  # s since the last GNSS-referenced error
    head: Array
    wind: Array  # (3,) NED, horizontal
    tas: Array  # last measured TAS
    tas_f: Array  # low-passed TAS
    tas_rate: Array  # low-passed dTAS/dt (longitudinal acceleration compensation)
    as_bad_t: Array  # seconds of continuous airspeed mismatch
    as_good_t: Array  # seconds of continuous agreement
    as_healthy: Array  # bool


class Estimate(NamedTuple):
    quat: Array
    roll: Array
    pitch: Array
    yaw: Array
    omega: Array  # bias-corrected body rates
    accel: Array  # specific force, body (raw)
    pos_ned: Array
    vel_ned: Array
    alt_agl: Array  # flat ground at the site origin (M1)
    climb_rate: Array
    ias: Array
    tas: Array
    airspeed: Array  # used by the controllers: TAS, or synthetic if unhealthy
    airspeed_healthy: Array
    wind_ned: Array
    groundspeed: Array
    course: Array


def baro_altitude(p: NavParams, pressure: Array) -> Array:
    """Height above the site ground from static pressure (ISA lapse, ground reference)."""
    return p.t_ground_k / 0.0065 * (1.0 - (pressure / p.p_ground_pa) ** (1 / 5.25588))


def _air_data(z: SensorArrays) -> tuple[Array, Array]:
    dp = jnp.maximum(z.airspeed_dp_pa, 0.0)
    rho = z.baro_pa / (R_AIR * z.baro_temp_k)
    return jnp.sqrt(2 * dp / RHO0), jnp.sqrt(2 * dp / rho)


def _mag_heading(p: NavParams, mag_body: Array, roll: Array, pitch: Array) -> Array:
    """Heading from the magnetometer, levelled with the estimated roll/pitch."""
    cr, sr, cp, sp = jnp.cos(roll), jnp.sin(roll), jnp.cos(pitch), jnp.sin(pitch)
    mx, my, mz = mag_body[0], mag_body[1], mag_body[2]
    xh = mx * cp + my * sr * sp + mz * cr * sp
    yh = my * cr - mz * sr
    declination = jnp.arctan2(p.mag_ned_unit[1], p.mag_ned_unit[0])
    return declination - jnp.arctan2(yh, xh)


def _wrap(a: Array) -> Array:
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def nav_init(p: NavParams, z: SensorArrays) -> NavState:
    """Align from one sample: roll/pitch from gravity, heading from the magnetometer."""
    f = z.accel
    roll = jnp.arctan2(-f[1], -f[2])
    pitch = jnp.arctan2(f[0], jnp.sqrt(f[1] ** 2 + f[2] ** 2))
    yaw = _mag_heading(p, z.mag, roll, pitch)
    _, tas = _air_data(z)
    pos = z.gnss_pos_ned.at[2].set(-baro_altitude(p, z.baro_pa))
    dtype = pos.dtype
    return NavState(
        t=jnp.zeros((), dtype),
        quat=quaternion.from_euler(roll, pitch, yaw).astype(dtype),
        gyro_bias=jnp.zeros(3, dtype),
        pos=pos,
        vel=z.gnss_vel_ned,
        hist=jnp.broadcast_to(jnp.concatenate([pos, z.gnss_vel_ned]), (HISTORY, 6)),
        v_ins=jnp.zeros(3, dtype),
        vins_hist=jnp.zeros((HISTORY, 3), dtype),
        gbuf_vel=jnp.zeros((GNSS_WIN, 3), dtype),
        gbuf_t=jnp.full((GNSS_WIN,), -1.0, dtype),
        gbuf_vins=jnp.zeros((GNSS_WIN, 3), dtype),
        gbuf_head=jnp.zeros((), jnp.int32),
        e_ref_ned=jnp.zeros(3, dtype),
        gps_age=jnp.full((), 1e3, dtype),
        head=jnp.zeros((), jnp.int32),
        wind=jnp.zeros(3, dtype),
        tas=tas,
        tas_f=tas,
        tas_rate=jnp.zeros((), dtype),
        as_bad_t=jnp.zeros((), dtype),
        as_good_t=jnp.zeros((), dtype),
        as_healthy=jnp.ones((), bool),
    )


def nav_step(p: NavParams, s: NavState, z: SensorArrays, dt: float) -> tuple[NavState, Estimate]:
    e3 = jnp.asarray(E3, s.pos.dtype)  # keep the state's precision (no float64 promotion)
    # ---------------------------------------------------------------- air data
    ias_m, tas_m = _air_data(z)
    tas = jnp.where(z.fresh.airspeed, tas_m, s.tas)

    # ---------------------------------------------------------------- attitude (Mahony)
    rot = quaternion.to_dcm(s.quat)  # body → NED
    omega = z.gyro - s.gyro_bias
    # Kinematic compensation with the *air* velocity: for a steady wind the inertial
    # acceleration is d(v_air)/dt + ω × v_air, so the wind drops out. The longitudinal part
    # d(v_air)/dt ≈ dTAS/dt matters: decelerating at 1 m/s² otherwise tilts the gravity
    # reference by 6° nose-up (the classic DCM pitch error, fixed in ArduPilot with GNSS).
    tas_f = s.tas_f + (dt / TAS_LP_S) * (tas - s.tas_f)
    tas_rate = s.tas_rate + (dt / TAS_RATE_LP_S) * ((tas_f - s.tas_f) / dt - s.tas_rate)
    v_air_body = jnp.array([1.0, 0.0, 0.0]) * jnp.maximum(tas, 0.0)
    a_long = jnp.array([1.0, 0.0, 0.0]) * jnp.where(s.as_healthy, tas_rate, 0.0)
    a_kin = a_long + jnp.cross(omega, v_air_body)
    g_vec = a_kin - z.accel  # ≈ gravity in body axes (down)
    g_norm = jnp.linalg.norm(g_vec)
    down_meas = g_vec / jnp.maximum(g_norm, 1e-6)
    down_est = rot.T @ e3
    gate = jnp.abs(g_norm - p.gravity) < p.accel_gate
    e_classic = jnp.where(gate, jnp.cross(down_meas, down_est), 0.0)

    # GNSS acceleration reference (AP_AHRS_DCM::drift_correction): over a window of GNSS
    # samples (all valid `latency` ago), compare the mean estimated specific force R·f with
    # the one the GNSS velocity change implies; rotate the former onto the latter.
    fresh_g = z.fresh.gnss
    vins_d = s.vins_hist[(s.head - p.gnss_latency_steps) % HISTORY]
    oldest = s.gbuf_head  # the slot about to be overwritten holds the oldest sample
    dt_g = z.gnss_time_s - s.gbuf_t[oldest]
    have = jnp.logical_and(fresh_g, jnp.logical_and(s.gbuf_t[oldest] >= 0, dt_g > 1e-3))
    ra = (vins_d - s.gbuf_vins[oldest]) / jnp.maximum(dt_g, 1e-3) - p.gravity * e3
    ga = (z.gnss_vel_ned - s.gbuf_vel[oldest]) / jnp.maximum(dt_g, 1e-3) - p.gravity * e3
    e_new = jnp.cross(ra / jnp.linalg.norm(ra), ga / jnp.linalg.norm(ga))
    e_new = e_new * jnp.array([1.0, 1.0, 0.0])  # roll/pitch only: heading is the mag's job
    e_ref_ned = jnp.where(have, e_new, s.e_ref_ned)
    gps_age = jnp.where(have, 0.0, s.gps_age + dt)
    slot = jnp.where(fresh_g, s.gbuf_head, GNSS_WIN)  # out of range = no write
    gbuf_vel = s.gbuf_vel.at[slot].set(z.gnss_vel_ned, mode="drop")
    gbuf_t = s.gbuf_t.at[slot].set(z.gnss_time_s, mode="drop")
    gbuf_vins = s.gbuf_vins.at[slot].set(vins_d, mode="drop")
    gbuf_head = jnp.where(fresh_g, (s.gbuf_head + 1) % GNSS_WIN, s.gbuf_head)
    use_gnss = gps_age < GNSS_REF_TIMEOUT_S
    e_acc = jnp.where(use_gnss, rot.T @ e_ref_ned, e_classic)

    roll, pitch, yaw = quaternion.to_euler(s.quat)
    psi_mag = _mag_heading(p, z.mag, roll, pitch)
    e_yaw = jnp.where(z.fresh.mag, _wrap(psi_mag - yaw), 0.0)
    e_mag = down_est * e_yaw * (p.mag_kp / p.att_kp)  # rotate about the NED down axis

    err = e_acc + e_mag
    # A single-sample alignment in turbulence can be off by tens of degrees; converge fast
    # first (as autopilots do after boot), then settle to the nominal, low-noise gains.
    boost = 1.0 + ALIGN_BOOST * jnp.exp(-s.t / ALIGN_TAU_S)
    gyro_bias = s.gyro_bias - p.att_ki * err * dt
    omega_corr = omega + p.att_kp * boost * err
    quat = quaternion.normalize(s.quat + quaternion.derivative(s.quat, omega_corr) * dt)
    rot = quaternion.to_dcm(quat)

    # ---------------------------------------------------------------- position / velocity
    acc_ned = rot @ z.accel + e3 * p.gravity
    v_ins = s.v_ins + acc_ned * dt
    vins_hist = s.vins_hist.at[s.head].set(v_ins)
    vel = s.vel + acc_ned * dt
    pos = s.pos + s.vel * dt + 0.5 * acc_ned * dt**2

    # GNSS against the delayed estimate (horizontal pos/vel, vertical velocity).
    delayed = s.hist[(s.head - p.gnss_latency_steps) % HISTORY]
    innov_p = (z.gnss_pos_ned - delayed[:3]) * jnp.array([1.0, 1.0, 0.0])
    innov_v = z.gnss_vel_ned - delayed[3:]
    g = z.fresh.gnss
    d_pos = jnp.where(g, p.gnss_pos_gain * innov_p, 0.0)
    d_vel = jnp.where(
        g, p.gnss_vel_gain * innov_v * jnp.array([1.0, 1.0, 0.5]) + p.gnss_pv_gain * innov_p, 0.0
    )
    # Barometer: primary height.
    innov_h = baro_altitude(p, z.baro_pa) - (-pos[2])
    b = z.fresh.baro
    d_pos = d_pos + jnp.where(b, -p.baro_gain * innov_h * e3, 0.0)
    d_vel = d_vel + jnp.where(b, -p.baro_vel_gain * innov_h * e3, 0.0)
    pos, vel = pos + d_pos, vel + d_vel
    hist = s.hist + jnp.concatenate([d_pos, d_vel])  # keep the history consistent
    hist = hist.at[s.head].set(jnp.concatenate([pos, vel]))
    head = (s.head + 1) % HISTORY

    # ---------------------------------------------------------------- wind & airspeed health
    x_body_ned = rot[:, 0]
    air_ned = tas * x_body_ned
    synthetic = jnp.linalg.norm((vel - s.wind)[:2])  # horizontal airspeed from wind triangle
    mismatch = jnp.abs(jnp.linalg.norm(air_ned[:2]) - synthetic)
    bad = mismatch > p.as_fail_mps
    as_bad_t = jnp.where(bad, s.as_bad_t + dt, 0.0)
    as_good_t = jnp.where(bad, 0.0, s.as_good_t + dt)
    healthy = jnp.where(s.as_healthy, as_bad_t < p.as_fail_s, as_good_t >= p.as_recover_s)
    wind_meas = (vel - air_ned) * jnp.array([1.0, 1.0, 0.0])
    # Fast initial convergence (the estimate starts at zero), then the nominal smoothing.
    wind_tau = p.wind_tau * (1.0 - (1.0 - WIND_TAU_START_FRAC) * jnp.exp(-s.t / WIND_BOOST_S))
    alpha_w = jnp.where(healthy, dt / wind_tau, 0.0)
    wind = s.wind + alpha_w * (wind_meas - s.wind)
    airspeed = jnp.where(healthy, tas, jnp.linalg.norm(vel - wind))

    roll, pitch, yaw = quaternion.to_euler(quat)
    state = NavState(
        t=s.t + dt,
        quat=quat,
        gyro_bias=gyro_bias,
        pos=pos,
        vel=vel,
        hist=hist,
        v_ins=v_ins,
        vins_hist=vins_hist,
        gbuf_vel=gbuf_vel,
        gbuf_t=gbuf_t,
        gbuf_vins=gbuf_vins,
        gbuf_head=gbuf_head,
        e_ref_ned=e_ref_ned,
        gps_age=gps_age,
        head=head,
        wind=wind,
        tas=tas,
        tas_f=tas_f,
        tas_rate=tas_rate,
        as_bad_t=as_bad_t,
        as_good_t=as_good_t,
        as_healthy=healthy,
    )
    est = Estimate(
        quat=quat,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        omega=z.gyro - gyro_bias,
        accel=z.accel,
        pos_ned=pos,
        vel_ned=vel,
        alt_agl=-pos[2],
        climb_rate=-vel[2],
        ias=ias_m,
        tas=tas,
        airspeed=airspeed,
        airspeed_healthy=healthy,
        wind_ned=wind,
        groundspeed=jnp.linalg.norm(vel[:2]),
        course=jnp.arctan2(vel[1], vel[0]),
    )
    return state, est
