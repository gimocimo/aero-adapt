"""Flight-stack configuration: YAML schema → JAX parameter pytrees.

The stack may use *nominal* design data: the airframe YAML (glide polar, motor KV) and the
site config (barometric reference, magnetic field, safe zones). It may never use the
simulator or fault ground truth (ADR-006).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from aeroadapt.airframes import Airframe
from aeroadapt.scenarios.site import MAX_SAFE_ZONES, Site

STACK_DIR = Path(__file__).resolve().parents[3] / "configs" / "stack"
G0 = 9.80665
RHO0 = 1.225


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- YAML schema
class NavigatorCfg(_Strict):
    att_kp: float = Field(gt=0)
    att_ki: float = Field(ge=0)
    mag_kp: float = Field(ge=0)
    accel_gate_g: float = Field(gt=0)
    gnss_latency_s: float = Field(ge=0, le=0.3)
    gnss_pos_gain: float = Field(gt=0, le=1)
    gnss_vel_gain: float = Field(gt=0, le=1)
    gnss_pos_to_vel_gain: float = Field(ge=0)
    baro_gain: float = Field(gt=0, le=1)
    baro_vel_gain: float = Field(ge=0)
    wind_tau_s: float = Field(gt=0)
    airspeed_fail_mps: float = Field(gt=0)
    airspeed_fail_s: float = Field(gt=0)
    airspeed_recover_s: float = Field(gt=0)


class L1Cfg(_Strict):
    period_s: float = Field(gt=0)
    damping: float = Field(gt=0)
    wp_radius_m: float = Field(gt=0)
    roll_limit_deg: float = Field(gt=0, le=70)
    loiter_radius_m: float = Field(gt=0)
    # GLIDE_SLOPE_MIN: a mission leg changing altitude by at least this much spreads the
    # change along the leg; smaller changes are flown at once (0: always at once).
    glide_slope_min_m: float = Field(default=15.0, ge=0)


class TecsCfg(_Strict):
    time_const_s: float = Field(gt=0)
    thr_damp: float = Field(ge=0)
    ptch_damp: float = Field(ge=0)
    integ_gain: float = Field(ge=0)
    spd_weight: float = Field(ge=0, le=2)
    climb_max_mps: float = Field(gt=0)
    sink_min_mps: float = Field(gt=0)
    sink_max_mps: float = Field(gt=0)
    vert_acc_mps2: float = Field(gt=0)
    rll2thr: float = Field(ge=0)
    pitch_max_deg: float
    pitch_min_deg: float
    thr_cruise: float = Field(gt=0, lt=1)
    thr_slope: float  # throttle trim per m/s of speed demand (from the trim table)
    pitch_trim_deg: float
    pitch_trim_b_deg: float  # pitch trim = trim + b·((V_cruise/V)² − 1)
    airspeed_min_mps: float = Field(gt=0)
    airspeed_cruise_mps: float = Field(gt=0)
    airspeed_max_mps: float = Field(gt=0)


class AxisCfg(_Strict):
    tconst_s: float = Field(gt=0)
    rate_max_dps: float = Field(gt=0)
    ff: float = Field(ge=0)
    p: float = Field(ge=0)
    i: float = Field(ge=0)
    imax: float = Field(ge=0)


class YawCfg(_Strict):
    ff: float = Field(ge=0)
    damp: float = Field(ge=0)
    slip_i: float = Field(ge=0)
    imax: float = Field(ge=0)


class ServoTrimCfg(_Strict):
    """SERVOn_TRIM analogues (normalised), from ``tools/tune_stack.py``. The elevator trim is
    scheduled as a + b·n·(V_cruise/V)², the level-flight trim curve (n = load factor)."""

    aileron: float = Field(ge=-1, le=1)
    rudder: float = Field(ge=-1, le=1)
    elevator_a: float
    elevator_b: float


class AttitudeCfg(_Strict):
    roll: AxisCfg
    pitch: AxisCfg
    yaw: YawCfg
    scaler_min: float = Field(gt=0)
    scaler_max: float = Field(gt=0)
    servo_trim: ServoTrimCfg


class ContingencyCfg(_Strict):
    enabled: bool  # false = stock ArduPlane behaviour (baseline B0): no motor-failure logic
    esc_telemetry: bool
    rpm_fraction: float = Field(gt=0, lt=1)
    rpm_min_throttle: float = Field(ge=0, le=1)
    esc_persist_s: float = Field(gt=0)
    energy_tau_s: float = Field(gt=0)  # energy-rate residual low-pass
    energy_slack_mps: float = Field(gt=0)  # CUSUM slack k
    energy_threshold_m: float = Field(gt=0)  # CUSUM decision threshold h
    glide_speed_min_mps: float = Field(gt=0)
    headwind_speed_gain: float = Field(ge=0)
    h_pattern_m: float = Field(ge=0)
    h_margin_m: float = Field(ge=0)
    circle_radius_m: float = Field(gt=0)
    final_length_m: float = Field(gt=0)
    flare_alt_m: float = Field(gt=0)
    flare_sink_mps: float = Field(gt=0)
    flare_pitch_gain: float = Field(gt=0)
    land_roll_limit_deg: float = Field(gt=0)


class StackCfg(_Strict):
    schema_version: int = Field(ge=1, le=1)
    id: str
    airframe: str
    navigator: NavigatorCfg
    l1: L1Cfg
    tecs: TecsCfg
    attitude: AttitudeCfg
    contingency: ContingencyCfg


def load_stack_cfg(name_or_path: str | Path, airframe_id: str | None = None) -> StackCfg:
    """By id (``configs/stack/<id>.yaml``), by path, or from a fleet run's tuned stacks: a
    directory with ``stack_cfgs.jsonl`` (``aeroadapt fleet-accept``), for ``airframe_id``."""
    path = Path(name_or_path)
    for d in (path, STACK_DIR.parents[1] / path):
        if (d / "stack_cfgs.jsonl").exists():
            if airframe_id is None:
                raise ValueError(f"{d} holds tuned stacks: name the airframe")
            for line in (d / "stack_cfgs.jsonl").read_text().splitlines():
                row = json.loads(line)
                if row["airframe"] == airframe_id:
                    return StackCfg.model_validate(row["stack"])
            raise KeyError(f"no tuned stack for {airframe_id} in {d}")
    if path.suffix not in {".yaml", ".yml"}:
        path = STACK_DIR / f"{name_or_path}.yaml"
    with path.open() as f:
        return StackCfg.model_validate(yaml.safe_load(f))


# --------------------------------------------------------------------------- JAX pytrees
class NavParams(NamedTuple):
    att_kp: float
    att_ki: float
    mag_kp: float
    accel_gate: float  # m/s² band around g
    gnss_latency_steps: int
    gnss_pos_gain: float
    gnss_vel_gain: float
    gnss_pv_gain: float
    baro_gain: float
    baro_vel_gain: float
    wind_tau: float
    as_fail_mps: float
    as_fail_s: float
    as_recover_s: float
    p_ground_pa: float  # baro reference: static pressure at site ground level (pre-flight)
    t_ground_k: float
    mag_ned_unit: np.ndarray  # (3,) expected Earth-field direction at the site (WMM)
    gravity: float


class L1Params(NamedTuple):
    period: float
    damping: float
    wp_radius: float
    roll_limit: float  # rad
    loiter_radius: float
    glide_slope_min: float = 15.0  # m


class TecsParams(NamedTuple):
    time_const: float
    thr_damp: float
    ptch_damp: float
    integ_gain: float
    spd_weight: float
    climb_max: float
    sink_min: float
    sink_max: float
    vert_acc: float
    rll2thr: float  # (m/s)²/s of energy rate per unit (1/cos²φ − 1)
    pitch_max: float  # rad
    pitch_min: float  # rad
    thr_cruise: float
    thr_slope: float
    pitch_trim: float  # rad
    pitch_trim_b: float  # rad
    v_min: float
    v_cruise: float
    v_max: float


class AttParams(NamedTuple):
    roll_tau: float
    roll_rate_max: float
    roll_ff: float
    roll_p: float
    roll_i: float
    roll_imax: float
    pitch_tau: float
    pitch_rate_max: float
    pitch_ff: float
    pitch_p: float
    pitch_i: float
    pitch_imax: float
    yaw_ff: float
    yaw_damp: float
    yaw_slip_i: float
    yaw_imax: float
    v_cruise: float
    scaler_min: float
    scaler_max: float
    trim_ail: float
    trim_rud: float
    trim_ele_a: float
    trim_ele_b: float


class ContParams(NamedTuple):
    enabled: bool
    esc_enabled: bool
    rpm_fraction: float
    rpm_min_throttle: float
    esc_persist: float
    kv_rpm_per_volt: float  # nominal motor KV (airframe data)
    energy_tau: float
    energy_slack: float
    energy_threshold: float
    ld_max: float  # nominal glide polar (incl. windmilling drag)
    v_bg_rho0: float  # best-glide airspeed at ρ0
    v_glide_min: float
    headwind_gain: float
    h_pattern: float
    h_margin: float
    circle_radius: float
    final_length: float
    flare_alt: float
    flare_sink: float
    flare_pitch_gain: float
    land_roll_limit: float  # rad


class Zones(NamedTuple):
    """Pre-surveyed safe zones, padded to MAX_SAFE_ZONES (jit-able)."""

    valid: np.ndarray  # (Z,) bool
    centre: np.ndarray  # (Z, 2) NE
    axis: np.ndarray  # (Z,) rad, heading of the long axis (landing either way)
    half_length: np.ndarray  # (Z,)
    half_width: np.ndarray  # (Z,)
    quality: np.ndarray  # (Z,)


class StackParams(NamedTuple):
    nav: NavParams
    l1: L1Params
    tecs: TecsParams
    att: AttParams
    cont: ContParams
    zones: Zones
    dt: float  # control period


def glide_polar(af: Airframe) -> tuple[float, float]:
    """(L/D)max and best-glide airspeed at ρ0 from the nominal parabolic polar, including the
    windmilling propeller's drag (motor out)."""
    g, ae = af.geometry, af.aero.longitudinal
    disc = math.pi * af.propulsion.prop_diameter_m**2 / 4
    cd0 = ae.C_D_p + af.propulsion.windmill_cd * disc / g.wing_area_m2
    k = 1.0 / (math.pi * g.oswald_e * g.aspect_ratio)
    cl_bg = math.sqrt(cd0 / k)
    ld = 1.0 / (2.0 * math.sqrt(cd0 * k))
    v_bg = math.sqrt(2 * af.mass_inertia.mass_kg * G0 / (RHO0 * g.wing_area_m2 * cl_bg))
    return ld, v_bg


def _chord(pts: np.ndarray, c: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    """Parameters (t_min < 0 < t_max) where the line c + t·u leaves the convex polygon."""
    ts = []
    for a, b in zip(pts, np.roll(pts, -1, axis=0), strict=True):
        e = b - a
        den = u[0] * e[1] - u[1] * e[0]
        if abs(den) < 1e-12:
            continue
        w = a - c
        t = (w[0] * e[1] - w[1] * e[0]) / den  # along the line
        s = (w[0] * u[1] - w[1] * u[0]) / den  # along the edge
        if -1e-9 <= s <= 1 + 1e-9:
            ts.append(t)
    return min(ts), max(ts)


def zones_from_site(site: Site) -> Zones:
    """Each zone polygon → long axis (principal component), and the centre and half extents
    of the chords through the polygon along and across that axis: the run actually
    available on the centre line, not the (longer) vertex projection."""
    n = MAX_SAFE_ZONES
    valid = np.zeros(n, bool)
    centre = np.zeros((n, 2))
    axis = np.zeros(n)
    half_len = np.zeros(n)
    half_wid = np.zeros(n)
    quality = np.zeros(n)
    for i, z in enumerate(site.safe_zones):
        pts = np.asarray(z.polygon_ne_m, dtype=np.float64)
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c)
        major, minor = vt[0], vt[1]
        t0, t1 = _chord(pts, c, major)
        c = c + 0.5 * (t0 + t1) * major
        w0, w1 = _chord(pts, c, minor)
        valid[i] = True
        centre[i] = c + 0.5 * (w0 + w1) * minor
        axis[i] = math.atan2(major[1], major[0])
        half_len[i] = 0.5 * (t1 - t0)
        half_wid[i] = 0.5 * (w1 - w0)
        quality[i] = z.quality
    return Zones(valid, centre, axis, half_len, half_wid, quality)


def build_stack_params(cfg: StackCfg, airframe: Airframe, site: Site, dt: float) -> StackParams:
    nv, l1, te, at, co = cfg.navigator, cfg.l1, cfg.tecs, cfg.attitude, cfg.contingency
    # Pre-flight baro reference: ISA static pressure/temperature at the site's ground level.
    h0 = site.ground_msl_m
    t_std = 288.15 - 0.0065 * h0
    p_ground = site.isa.qnh_hpa * 100.0 * (t_std / 288.15) ** 5.25588
    mag = np.asarray(site.magnetic.field_ned_gauss)
    ld, v_bg = glide_polar(airframe)
    rad = math.radians
    return StackParams(
        nav=NavParams(
            att_kp=nv.att_kp,
            att_ki=nv.att_ki,
            mag_kp=nv.mag_kp,
            accel_gate=nv.accel_gate_g * G0,
            gnss_latency_steps=round(nv.gnss_latency_s / dt),
            gnss_pos_gain=nv.gnss_pos_gain,
            gnss_vel_gain=nv.gnss_vel_gain,
            gnss_pv_gain=nv.gnss_pos_to_vel_gain,
            baro_gain=nv.baro_gain,
            baro_vel_gain=nv.baro_vel_gain,
            wind_tau=nv.wind_tau_s,
            as_fail_mps=nv.airspeed_fail_mps,
            as_fail_s=nv.airspeed_fail_s,
            as_recover_s=nv.airspeed_recover_s,
            p_ground_pa=p_ground,
            t_ground_k=t_std + site.isa.temp_offset_k,
            mag_ned_unit=mag / np.linalg.norm(mag),
            gravity=G0,
        ),
        l1=L1Params(
            l1.period_s,
            l1.damping,
            l1.wp_radius_m,
            rad(l1.roll_limit_deg),
            l1.loiter_radius_m,
            l1.glide_slope_min_m,
        ),
        tecs=TecsParams(
            time_const=te.time_const_s,
            thr_damp=te.thr_damp,
            ptch_damp=te.ptch_damp,
            integ_gain=te.integ_gain,
            spd_weight=te.spd_weight,
            climb_max=te.climb_max_mps,
            sink_min=te.sink_min_mps,
            sink_max=te.sink_max_mps,
            vert_acc=te.vert_acc_mps2,
            rll2thr=te.rll2thr,
            pitch_max=rad(te.pitch_max_deg),
            pitch_min=rad(te.pitch_min_deg),
            thr_cruise=te.thr_cruise,
            thr_slope=te.thr_slope,
            pitch_trim=rad(te.pitch_trim_deg),
            pitch_trim_b=rad(te.pitch_trim_b_deg),
            v_min=te.airspeed_min_mps,
            v_cruise=te.airspeed_cruise_mps,
            v_max=te.airspeed_max_mps,
        ),
        att=AttParams(
            roll_tau=at.roll.tconst_s,
            roll_rate_max=rad(at.roll.rate_max_dps),
            roll_ff=at.roll.ff,
            roll_p=at.roll.p,
            roll_i=at.roll.i,
            roll_imax=at.roll.imax,
            pitch_tau=at.pitch.tconst_s,
            pitch_rate_max=rad(at.pitch.rate_max_dps),
            pitch_ff=at.pitch.ff,
            pitch_p=at.pitch.p,
            pitch_i=at.pitch.i,
            pitch_imax=at.pitch.imax,
            yaw_ff=at.yaw.ff,
            yaw_damp=at.yaw.damp,
            yaw_slip_i=at.yaw.slip_i,
            yaw_imax=at.yaw.imax,
            v_cruise=te.airspeed_cruise_mps,
            scaler_min=at.scaler_min,
            scaler_max=at.scaler_max,
            trim_ail=at.servo_trim.aileron,
            trim_rud=at.servo_trim.rudder,
            trim_ele_a=at.servo_trim.elevator_a,
            trim_ele_b=at.servo_trim.elevator_b,
        ),
        cont=ContParams(
            enabled=co.enabled,
            esc_enabled=co.esc_telemetry,
            rpm_fraction=co.rpm_fraction,
            rpm_min_throttle=co.rpm_min_throttle,
            esc_persist=co.esc_persist_s,
            kv_rpm_per_volt=airframe.propulsion.motor.KV_rpm_per_volt,
            energy_tau=co.energy_tau_s,
            energy_slack=co.energy_slack_mps,
            energy_threshold=co.energy_threshold_m,
            ld_max=ld,
            v_bg_rho0=v_bg,
            v_glide_min=co.glide_speed_min_mps,
            headwind_gain=co.headwind_speed_gain,
            h_pattern=co.h_pattern_m,
            h_margin=co.h_margin_m,
            circle_radius=co.circle_radius_m,
            final_length=co.final_length_m,
            flare_alt=co.flare_alt_m,
            flare_sink=co.flare_sink_mps,
            flare_pitch_gain=co.flare_pitch_gain,
            land_roll_limit=rad(co.land_roll_limit_deg),
        ),
        zones=zones_from_site(site),
        dt=dt,
    )
