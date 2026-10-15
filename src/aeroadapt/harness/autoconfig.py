"""Model-based auto-configuration of the stock stack (M2 spec §6).

The analogue of a manufacturer's ArduPlane parameter file: speeds, trim schedules and loop
gains computed from an airframe's nominal model, before any flight. It is design-time code
(like ``tools/tune_stack.py``), so it lives in the harness and may use the simulator's trim
and propulsion models; the stack still never reads the simulator at run time (ADR-006).

Loop gains follow ArduPlane's controller semantics:

- roll and pitch feed-forward = 1 / (steady body rate per unit command), from the roll
  damping and aileron power, and from the short-period steady state (C_mα, C_mq, C_mδe);
- P and I as fixed fractions of the feed-forward;
- yaw coordination from C_n,r / C_n,δr;
- TECS climb and sink limits from excess thrust and the glide polar; RLL2THR from the
  induced drag at cruise.

Three constants (``K_PITCH_FF``, ``K_YAW``, ``K_RLL2THR``) are calibrated once so that the
formulas reproduce the M1 Aerosonde gains that were tuned against V11 (tested); everything
else is physics. Airframes whose trim cannot be found over a usable speed range are reported
as unconfigurable, with the reason.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from aeroadapt.airframes import Airframe, AirframeParams, bm_tables, to_vehicle_params
from aeroadapt.scenarios.site import Site
from aeroadapt.sim.aerojax.atmosphere import isa
from aeroadapt.sim.aerojax.battery import open_circuit_voltage
from aeroadapt.sim.aerojax.environment import Environment
from aeroadapt.sim.aerojax.propulsion import propeller
from aeroadapt.sim.aerojax.trim import trim_level
from aeroadapt.stack.config import StackCfg, load_stack_cfg

G0 = 9.806_65
TRIM_ALT_M = 120.0
# Calibration: the formulas reproduce the V11-tuned Aerosonde at 25 m/s (see the tests).
K_PITCH_FF = 1.18
K_YAW = 1.0
K_RLL2THR = 3.36
LAG_REF_S = 0.045  # the Aerosonde's servo lag (20 ms delay + 20 ms lag + 10° at 300°/s)
SPEED_FACTORS = np.array([1.15, 1.25, 1.35, 1.5, 1.65, 1.8, 2.0, 2.25, 2.5, 2.8])  # × V_stall


@dataclass(frozen=True)
class AutoConfig:
    airframe: str
    cfg: StackCfg | None
    reason: str | None  # why the airframe could not be configured
    v_stall: float
    speeds: tuple[float, float, float]  # (min, cruise, max) m/s


def _cl_cd(p: AirframeParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = p.tables if p.tables is not None else bm_tables(p)
    alpha = np.linspace(-math.pi, math.pi, len(np.asarray(t.C_L)))
    return alpha, np.asarray(t.C_L), np.asarray(t.C_D)


def _polar(p: AirframeParams) -> tuple[float, float, float]:
    """(C_L,max for 0 < α ≤ 30°, C_D0, k) of the nominal model."""
    alpha, cl, cd = _cl_cd(p)
    pos = (alpha > 0.0) & (alpha <= math.radians(30.0))
    att = (alpha >= math.radians(-4.0)) & (alpha <= math.radians(8.0))
    k, cd0 = np.polyfit(cl[att] ** 2, cd[att], 1)
    return float(cl[pos].max()), max(float(cd0), 1e-3), max(float(k), 1e-3)


@jax.jit
def _trims(p: AirframeParams, rho: jax.Array, g: jax.Array, v_supply: jax.Array, va: jax.Array):
    env = Environment(rho, g)
    return jax.vmap(lambda pi, vs, v: trim_level(pi, env, vs, v))(p, v_supply, va)


def _stack(trees: Sequence[AirframeParams]) -> AirframeParams:
    return jax.tree.map(lambda *x: jnp.stack([jnp.asarray(v, jnp.float64) for v in x]), *trees)


def autoconfigure(
    airframes: Sequence[Airframe],
    site: Site,
    v_cruise: Sequence[float | None] | None = None,
) -> list[AutoConfig]:
    """Configure many airframes at once (trims are batched). ``v_cruise`` overrides the
    computed cruise speed per airframe (``None`` keeps it)."""
    ref = load_stack_cfg("aerosonde_bm")
    _, _, rho = isa(site_isa(site), site.ground_msl_m + TRIM_ALT_M)
    rho = float(rho)
    vps = [to_vehicle_params(a) for a in airframes]
    n = len(airframes)
    overrides = list(v_cruise) if v_cruise is not None else [None] * n

    v_full = np.array([float(open_circuit_voltage(v.battery, jnp.asarray(1.0))) for v in vps])
    polars = [_polar(v.airframe) for v in vps]
    masses = np.array([v.airframe.mass for v in vps])
    areas = np.array([v.airframe.S for v in vps])
    v_stall = np.sqrt(2 * masses * G0 / (rho * areas * np.array([c for c, _, _ in polars])))

    with jax.enable_x64(True):
        k = len(SPEED_FACTORS)
        params = _stack([v.airframe for v in vps for _ in range(k)])
        speeds = (v_stall[:, None] * SPEED_FACTORS[None, :]).ravel()
        z, res = _trims(params, rho, G0, jnp.repeat(jnp.asarray(v_full), k), jnp.asarray(speeds))
        z, res = np.asarray(z).reshape(n, k, 7), np.asarray(res).reshape(n, k)

    out = []
    for i, (af, vp) in enumerate(zip(airframes, vps, strict=True)):
        p, act = vp.airframe, vp.actuators
        sign = np.sign([p.C_ell_delta_a, p.C_m_delta_e, p.C_n_delta_r]) * np.where(
            np.asarray(act.reversed), -1.0, 1.0
        )
        travel = np.asarray(act.max_deflection)
        cmd = (z[i, :, 3:6] - np.asarray(act.trim)) / (sign * travel)  # normalised commands
        thr = z[i, :, 6]
        v = v_stall[i] * SPEED_FACTORS
        ok = (res[i] < 1e-6) & (thr >= 0.0) & (thr <= 0.9) & (np.abs(cmd[:, 1]) <= 0.8)
        ok &= v >= 1.25 * v_stall[i]
        _, cd0, kind = polars[i]
        w = masses[i] * G0
        v_md = math.sqrt(2 * w / (rho * areas[i])) * (kind / cd0) ** 0.25
        if ok.sum() < 3:
            out.append(
                AutoConfig(af.id, None, "trim: fewer than 3 usable speeds", v_stall[i], (0, 0, 0))
            )
            continue
        v_min = float(v[ok].min())
        v_hi = float(v[ok].max())
        vc = overrides[i] or round(2.0 * max(1.4 * v_stall[i], 1.15 * v_md)) / 2.0
        vc = float(min(max(vc, v_min * 1.1), v_hi))
        v_max = float(min(v_hi, 1.6 * vc))
        sel = ok & (v <= v_max + 1e-9)
        if sel.sum() < 3:
            out.append(
                AutoConfig(
                    af.id, None, "trim: speed range too narrow", v_stall[i], (v_min, vc, v_max)
                )
            )
            continue
        vs, c, th, theta = v[sel], cmd[sel], thr[sel], z[i, sel, 1]
        x = (vc / vs) ** 2
        ones = np.ones_like(x)
        ele_a, ele_b = np.linalg.lstsq(np.stack([ones, x], 1), c[:, 1], rcond=None)[0]
        thr_c, thr_slope = np.linalg.lstsq(np.stack([ones, vs - vc], 1), th, rcond=None)[0]
        pitch_c, pitch_b = np.linalg.lstsq(np.stack([ones, x - 1.0], 1), theta, rcond=None)[0]
        ail_trim = float(np.interp(vc, vs, c[:, 0]))
        rud_trim = float(np.interp(vc, vs, c[:, 2]))

        # Loop gains at cruise.
        qbar = 0.5 * rho * vc * vc
        p_per_cmd = abs(p.C_ell_delta_a) * travel[0] * (2 * vc / p.b) / abs(p.C_ell_p)
        a_term = p.C_m_alpha * p.mass * vc / (qbar * p.S * p.C_L_alpha) + p.C_m_q * p.c / (2 * vc)
        q_per_cmd = abs(p.C_m_delta_e / a_term) * travel[1]
        r_cmd = abs(p.C_n_r) * p.b / (2 * vc * abs(p.C_n_delta_r)) / travel[2]
        roll_ff = 1.0 / p_per_cmd
        pitch_ff = K_PITCH_FF / q_per_cmd
        yaw_ff = K_YAW * r_cmd
        # Servo lag (transport delay + first-order lag + time to slew 10° at the rate limit),
        # relative to the Aerosonde's (≈ 45 ms), for which the loop shapes were tuned.
        lag = (np.asarray(act.delay[:3]) + np.asarray(act.tau[:3])
               + math.radians(10.0) / np.asarray(act.rate_limit))  # fmt: skip
        slow = np.maximum(lag / LAG_REF_S, 1.0)

        # TECS limits from excess thrust and the glide polar at cruise.
        def thrust(
            throttle: float, vc: float = vc, p: AirframeParams = p, vf: float = v_full[i]
        ) -> float:
            return float(
                propeller(p, rho, jnp.asarray(vc), jnp.asarray(throttle), jnp.asarray(vf)).thrust
            )

        climb_cap = (thrust(1.0) - thrust(float(thr_c))) * vc / w
        cl_c = w / (qbar * p.S)
        disc = math.pi * p.D_prop**2 / 4
        glide_drag = qbar * (p.S * (cd0 + kind * cl_c**2) + p.C_windmill * disc)
        sink_glide = glide_drag * vc / w
        induced = qbar * p.S * kind * cl_c**2
        v_glide = max(1.3 * v_stall[i], v_min)
        turn30 = vc * vc / (G0 * math.tan(math.radians(30.0)))

        d = ref.model_dump()
        d["id"], d["airframe"] = f"{af.id}_auto", af.id
        d["navigator"]["airspeed_fail_mps"] = max(2.5, 0.2 * vc)
        d["l1"].update(
            wp_radius_m=float(np.clip(0.95 * vc * vc / G0, 15.0, 80.0)),
            loiter_radius_m=max(1.4 * turn30, 40.0),
        )
        d["tecs"].update(
            climb_max_mps=float(np.clip(0.6 * climb_cap, 1.0, 8.0)),
            sink_min_mps=float(np.clip(sink_glide, 0.5, 4.0)),
            sink_max_mps=float(np.clip(2.5 * sink_glide, 2.0, max(2.0, 0.3 * vc))),
            rll2thr=K_RLL2THR * induced * vc / p.mass,
            thr_cruise=float(np.clip(thr_c, 0.05, 0.95)),
            thr_slope=float(thr_slope),
            pitch_trim_deg=math.degrees(pitch_c),
            pitch_trim_b_deg=math.degrees(pitch_b),
            airspeed_min_mps=v_min,
            airspeed_cruise_mps=vc,
            airspeed_max_mps=v_max,
        )
        # Rate feedback scales down with √(lag ratio); the angle loops keep their time
        # constants within what the V11 rise-time limits allow (≤ 0.35 s).
        k = np.sqrt(slow)
        d["attitude"]["roll"].update(
            ff=roll_ff, p=0.25 * roll_ff / k[0], i=0.75 * roll_ff / k[0],
            rate_max_dps=min(75.0, 0.7 * math.degrees(p_per_cmd)),
        )  # fmt: skip
        d["attitude"]["pitch"].update(
            ff=pitch_ff, p=0.44 * pitch_ff / k[1], i=0.28 * pitch_ff / k[1],
            tconst_s=min(0.25 * k[1], 0.35), rate_max_dps=min(45.0, 0.5 * math.degrees(q_per_cmd)),
        )  # fmt: skip
        d["attitude"]["yaw"].update(ff=yaw_ff, damp=yaw_ff)
        d["attitude"]["servo_trim"] = {
            "aileron": float(np.clip(ail_trim, -1, 1)),
            "rudder": float(np.clip(rud_trim, -1, 1)),
            "elevator_a": float(ele_a),
            "elevator_b": float(ele_b),
        }
        d["contingency"].update(
            glide_speed_min_mps=v_glide,
            circle_radius_m=max(1.65 * v_glide**2 / (G0 * math.tan(math.radians(30.0))), 40.0),
            final_length_m=float(np.clip(12.0 * v_glide, 80.0, 250.0)),
            flare_alt_m=float(np.clip(0.16 * v_glide, 2.0, 3.0)),
        )
        out.append(
            AutoConfig(af.id, StackCfg.model_validate(d), None, v_stall[i], (v_min, vc, v_max))
        )
    return out


def site_isa(site: Site):
    from aeroadapt.sim.aerojax.atmosphere import IsaParams

    return IsaParams(site.isa.temp_offset_k, site.isa.qnh_hpa * 100.0, site.ground_msl_m)
