"""JAX-friendly airframe parameters used by the simulator's functional core.

All containers are NamedTuples, so they are pytrees: they pass through ``jit`` and can be
batched with ``vmap`` across a fleet of airframes (M2). ``AirframeParams`` holds the rigid
body, aerodynamics and propulsion; ``VehicleParams`` adds actuators, battery and sensors.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from aeroadapt.airframes.schema import ALPHA_GRID_N, Airframe, SurfaceActuator


def alpha_grid() -> np.ndarray:
    """Angles of attack (rad) at which aero v2 tables are tabulated."""
    return np.linspace(-math.pi, math.pi, ALPHA_GRID_N)


class AeroTables(NamedTuple):
    """Aero v2 static coefficients and effectiveness factors, each of shape (ALPHA_GRID_N,).

    ``eta_ctrl`` scales the control-surface derivatives and ``eta_damp`` the rate-damping
    derivatives (1 in attached flow, falling past the stall).
    """

    C_L: np.ndarray
    C_D: np.ndarray
    C_m: np.ndarray
    eta_ctrl: np.ndarray
    eta_damp: np.ndarray


class AirframeParams(NamedTuple):
    # mass and inertia
    mass: float
    Jx: float
    Jy: float
    Jz: float
    Jxz: float
    # geometry
    S: float
    b: float
    c: float
    e_oswald: float
    AR: float
    # longitudinal
    C_L_0: float
    C_L_alpha: float
    C_L_q: float
    C_L_delta_e: float
    C_D_p: float
    C_D_q: float
    C_D_delta_e: float
    C_m_0: float
    C_m_alpha: float
    C_m_q: float
    C_m_delta_e: float
    # lateral
    C_Y_0: float
    C_Y_beta: float
    C_Y_p: float
    C_Y_r: float
    C_Y_delta_a: float
    C_Y_delta_r: float
    C_ell_0: float
    C_ell_beta: float
    C_ell_p: float
    C_ell_r: float
    C_ell_delta_a: float
    C_ell_delta_r: float
    C_n_0: float
    C_n_beta: float
    C_n_p: float
    C_n_r: float
    C_n_delta_a: float
    C_n_delta_r: float
    # stall blending
    M: float
    alpha0: float
    # propulsion
    D_prop: float
    C_T0: float
    C_T1: float
    C_T2: float
    C_Q0: float
    C_Q1: float
    C_Q2: float
    KV: float  # V·s/rad
    KQ: float  # N·m/A
    R_motor: float
    i0: float
    V_max: float
    C_windmill: float  # stopped/windmilling propeller drag coefficient on disc area
    ground_effect: float = 0.0  # 1: induced-drag reduction near the ground (0 leaves v1 exact)
    # aero v2: static tables; None selects the analytic B&M model (a static choice under jit)
    tables: AeroTables | None = None


class ActuatorParams(NamedTuple):
    """Surface channels are ordered (aileron, elevator, rudder); index 3 is the throttle."""

    max_deflection: np.ndarray  # (3,) rad
    trim: np.ndarray  # (3,) rad
    reversed: np.ndarray  # (3,) bool, mirrors ArduPilot SERVOn_REVERSED
    rate_limit: np.ndarray  # (3,) rad/s
    deadband: np.ndarray  # (3,) rad
    tau: np.ndarray  # (4,) s, first-order lag (surfaces + throttle)
    delay: np.ndarray  # (4,) s, transport delay (surfaces + throttle)


class BatteryParams(NamedTuple):
    n_cells: float
    capacity_ah: float
    r_internal: float  # ohm, pack
    avionics_current: float  # A
    ocv_soc: np.ndarray  # (K,) state-of-charge grid
    ocv_v_cell: np.ndarray  # (K,) open-circuit volts per cell


class SensorParams(NamedTuple):
    # rates in Hz; the simulator converts them to integer decimations of the control rate
    imu_rate: float
    gyro_noise: float
    gyro_bias_rw: float
    gyro_bias_init: float
    accel_noise: float
    accel_bias_rw: float
    accel_bias_init: float
    baro_rate: float
    baro_noise: float
    baro_drift_rw: float
    airspeed_rate: float
    airspeed_noise: float
    gnss_rate: float
    gnss_latency: float
    gnss_sigma_h: float
    gnss_sigma_v: float
    gnss_tau: float
    gnss_vel_sigma: float
    mag_rate: float
    mag_noise: float
    battery_rate: float
    battery_v_noise: float
    battery_i_noise: float
    esc_rate: float
    esc_rpm_noise_frac: float


class VehicleParams(NamedTuple):
    airframe: AirframeParams
    actuators: ActuatorParams
    battery: BatteryParams
    sensors: SensorParams


def bm_tables(p: AirframeParams) -> AeroTables:
    """The Beard & McLain static model (eqs. 4.9-4.11, C_m linear in α) sampled on the grid.

    Written in NumPy, separately from the JAX formulas, so that V16 compares two codings.
    """
    a = alpha_grid()
    ln_a, ln_b = -p.M * (a - p.alpha0), p.M * (a + p.alpha0)
    # σ = (1 + e^ln_a + e^ln_b) / ((1 + e^ln_a)(1 + e^ln_b)), with exponents clipped for float64
    ea, eb = np.exp(np.minimum(ln_a, 700.0)), np.exp(np.minimum(ln_b, 700.0))
    sigma = np.where(
        np.isfinite(ea * eb),
        (1.0 + ea + eb) / ((1.0 + ea) * (1.0 + eb)),
        1.0,
    )
    linear = p.C_L_0 + p.C_L_alpha * a
    flat_plate = 2.0 * np.sign(a) * np.sin(a) ** 2 * np.cos(a)
    ones = np.ones_like(a)
    return AeroTables(
        C_L=(1.0 - sigma) * linear + sigma * flat_plate,
        C_D=p.C_D_p + linear**2 / (math.pi * p.e_oswald * p.AR),
        C_m=p.C_m_0 + p.C_m_alpha * a,
        eta_ctrl=ones,
        eta_damp=ones.copy(),
    )


def to_params(af: Airframe) -> AirframeParams:
    mi, g, lo, la, st, pr = (
        af.mass_inertia,
        af.geometry,
        af.aero.longitudinal,
        af.aero.lateral,
        af.aero.stall,
        af.propulsion,
    )
    kv = pr.motor.KV_v_s_per_rad
    params = AirframeParams(
        mass=mi.mass_kg,
        Jx=mi.Jx_kgm2,
        Jy=mi.Jy_kgm2,
        Jz=mi.Jz_kgm2,
        Jxz=mi.Jxz_kgm2,
        S=g.wing_area_m2,
        b=g.span_m,
        c=g.chord_m,
        e_oswald=g.oswald_e,
        AR=g.aspect_ratio,
        **lo.model_dump(),
        **la.model_dump(),
        M=st.M,
        alpha0=st.alpha0_rad,
        D_prop=pr.prop_diameter_m,
        C_T0=pr.C_T[0],
        C_T1=pr.C_T[1],
        C_T2=pr.C_T[2],
        C_Q0=pr.C_Q[0],
        C_Q1=pr.C_Q[1],
        C_Q2=pr.C_Q[2],
        KV=kv,
        KQ=kv,
        R_motor=pr.motor.R_ohm,
        i0=pr.motor.i0_a,
        V_max=pr.battery.v_max,
        C_windmill=pr.windmill_cd,
        ground_effect=float(af.aero.ground_effect),
    )
    if af.aero.model == "bm_tabulated":
        params = params._replace(tables=bm_tables(params))
    elif af.aero.tables is not None:
        t = af.aero.tables
        params = params._replace(
            tables=AeroTables(*(np.asarray(getattr(t, k)) for k in AeroTables._fields))
        )
    return params


def to_vehicle_params(af: Airframe) -> VehicleParams:
    surfaces: list[SurfaceActuator] = [
        af.actuators.surfaces.aileron,
        af.actuators.surfaces.elevator,
        af.actuators.surfaces.rudder,
    ]
    thr = af.actuators.throttle
    rad = math.radians
    actuators = ActuatorParams(
        max_deflection=np.array([rad(c.max_deflection_deg) for c in surfaces]),
        trim=np.array([rad(c.trim_deg) for c in surfaces]),
        reversed=np.array([c.reversed for c in surfaces]),
        rate_limit=np.array([rad(c.rate_limit_dps) for c in surfaces]),
        deadband=np.array([rad(c.deadband_deg) for c in surfaces]),
        tau=np.array([*(c.tau_s for c in surfaces), thr.tau_s]),
        delay=np.array([*(c.delay_s for c in surfaces), thr.delay_s]),
    )
    b = af.propulsion.battery
    battery = BatteryParams(
        n_cells=float(b.n_cells),
        capacity_ah=b.capacity_ah,
        r_internal=b.r_internal_ohm,
        avionics_current=b.avionics_current_a,
        ocv_soc=np.array([x for x, _ in b.ocv_curve]),
        ocv_v_cell=np.array([v for _, v in b.ocv_curve]),
    )
    se = af.sensors
    sensors = SensorParams(
        imu_rate=se.imu.rate_hz,
        gyro_noise=se.imu.gyro_noise_rps,
        gyro_bias_rw=se.imu.gyro_bias_rw,
        gyro_bias_init=se.imu.gyro_bias_init_rps,
        accel_noise=se.imu.accel_noise_mps2,
        accel_bias_rw=se.imu.accel_bias_rw,
        accel_bias_init=se.imu.accel_bias_init_mps2,
        baro_rate=se.baro.rate_hz,
        baro_noise=se.baro.noise_pa,
        baro_drift_rw=se.baro.drift_rw_pa_per_sqrt_s,
        airspeed_rate=se.airspeed.rate_hz,
        airspeed_noise=se.airspeed.noise_pa,
        gnss_rate=se.gnss.rate_hz,
        gnss_latency=se.gnss.latency_s,
        gnss_sigma_h=se.gnss.pos_sigma_h_m,
        gnss_sigma_v=se.gnss.pos_sigma_v_m,
        gnss_tau=se.gnss.pos_tau_s,
        gnss_vel_sigma=se.gnss.vel_sigma_mps,
        mag_rate=se.mag.rate_hz,
        mag_noise=se.mag.noise_gauss,
        battery_rate=se.battery.rate_hz,
        battery_v_noise=se.battery.voltage_noise_v,
        battery_i_noise=se.battery.current_noise_a,
        esc_rate=se.esc.rate_hz,
        esc_rpm_noise_frac=se.esc.rpm_noise_frac,
    )
    return VehicleParams(to_params(af), actuators, battery, sensors)
