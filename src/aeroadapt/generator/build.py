"""Design → validated ``Airframe`` (M2 spec §4.1 steps 2-7, except the batched aerojax checks).

The order matters: a preliminary drag polar sets the cruise speed and the battery; the
neutral point sets the target centre of gravity; the full aerodynamic model is then computed
about that centre of gravity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aeroadapt.airframes import Airframe, load_airframe
from aeroadapt.generator.aero import SWEEP_DEG, AeroModel, _run, aero_model, neutral_point
from aeroadapt.generator.design import CLASSES, Design
from aeroadapt.generator.geometry import Layout, airplane, layout
from aeroadapt.generator.mass import G0, MassProperties, Rejected, mass_properties, stall_speed
from aeroadapt.generator.propulsion import LIPO_OCV, Propulsion, size_propulsion

GENERATOR_VERSION = "1"
RHO0 = 1.225


@dataclass(frozen=True)
class Candidate:
    design: Design
    airframe: Airframe
    meta: dict[str, Any] = field(default_factory=dict)


def airframe_id(d: Design) -> str:
    return f"gen-{d.cls}-{d.seed:05d}"


def _speeds(d: Design, lay: Layout) -> tuple[float, float, float, float]:
    """(V_cruise, C_L at cruise, α at cruise in degrees, cruise power estimate in W) from a
    preliminary polar."""
    mass = d.wing_loading * lay.S / G0
    v_guess = math.sqrt(2.0 * d.wing_loading / (RHO0 * 0.5))
    pre = _run(airplane(d, lay, (lay.x_qmac, 0.0, 0.0)), SWEEP_DEG, v_guess)
    att = (SWEEP_DEG >= -4.0) & (SWEEP_DEG <= 8.0)
    k, cd0 = np.polyfit(pre["CL"][att] ** 2, pre["CD"][att], 1)
    k, cd0 = max(float(k), 1e-3), max(float(cd0), 0.005)
    pos = (SWEEP_DEG > 0.0) & (SWEEP_DEG <= 30.0)
    cl_max = float(np.max(np.where(pos, pre["CL"], -np.inf)))
    if cl_max < 0.6:
        raise Rejected("aero: C_L,max below 0.6")
    v_stall = stall_speed(mass, lay.S, cl_max)
    v_md = math.sqrt(2.0 * d.wing_loading / RHO0) * (k / cd0) ** 0.25
    v_cruise = round(2.0 * max(1.4 * v_stall, 1.15 * v_md)) / 2.0
    cl = d.wing_loading / (0.5 * RHO0 * v_cruise**2)
    drag = 0.5 * RHO0 * v_cruise**2 * lay.S * (cd0 + k * cl * cl)
    power = drag * v_cruise / (0.6 * 0.85) + 10.0 * (0.3 + 0.06 * mass)
    alpha_cruise = float(np.interp(cl, pre["CL"][att], SWEEP_DEG[att]))
    return v_cruise, cl, alpha_cruise, power


def _actuators(rng: np.random.Generator) -> dict[str, Any]:
    def surface(max_deg: tuple[float, float]) -> dict[str, Any]:
        return {
            "max_deflection_deg": float(rng.uniform(*max_deg)),
            "rate_limit_dps": float(rng.uniform(150.0, 600.0)),
            "tau_s": float(rng.uniform(0.01, 0.04)),
            "delay_s": float(rng.uniform(0.01, 0.06)),
            "deadband_deg": float(rng.uniform(0.1, 0.5)),
            "trim_deg": 0.0,
            "reversed": False,
        }

    return {
        "surfaces": {
            "aileron": surface((20.0, 28.0)),
            "elevator": surface((22.0, 30.0)),
            "rudder": surface((20.0, 28.0)),
        },
        "throttle": {
            "tau_s": float(rng.uniform(0.03, 0.08)),
            "delay_s": float(rng.uniform(0.0, 0.02)),
        },
    }


def _sensors(rng: np.random.Generator) -> dict[str, Any]:
    """The M1 sensor suite with noise and bias levels scaled within datasheet spreads."""
    base = load_airframe("aerosonde_bm").sensors.model_dump()
    rates = {"rate_hz", "latency_s", "pos_tau_s"}
    out: dict[str, Any] = {}
    for name, block in base.items():
        out[name] = {
            k: (v if k in rates else float(v * rng.uniform(0.5, 2.0))) for k, v in block.items()
        }
    return out


def assemble(d: Design, lay: Layout, mp: MassProperties, am: AeroModel, pr: Propulsion) -> Airframe:
    rng = np.random.default_rng([99, CLASSES.index(d.cls), d.seed])
    der = am.derivatives
    lon_keys = (
        "C_L_0", "C_L_alpha", "C_L_q", "C_L_delta_e", "C_D_p", "C_D_q", "C_D_delta_e",
        "C_m_0", "C_m_alpha", "C_m_q", "C_m_delta_e",
    )  # fmt: skip
    lat_keys = [k for k in der if k not in lon_keys]
    longitudinal = {k: der[k] for k in lon_keys}
    longitudinal["C_D_p"] = max(longitudinal["C_D_p"], 0.005)
    t = am.tables
    return Airframe.model_validate(
        {
            "schema_version": 2,
            "id": airframe_id(d),
            "name": f"Generated {d.cls.replace('_', ' ')} #{d.seed}",
            "provenance": {
                "source": f"aeroadapt procedural airframe generator v{GENERATOR_VERSION}",
                "values_from": "AeroSandbox AeroBuildup + NeuralFoil (static tables, derivatives)",
                "notes": (
                    "Coefficients computed from geometry. Post-stall data, the propeller family "
                    "and the mass budget are approximate (M2 spec §4, §12)."
                ),
            },
            "mass_inertia": {
                "mass_kg": mp.mass,
                "Jx_kgm2": mp.Jx,
                "Jy_kgm2": mp.Jy,
                "Jz_kgm2": mp.Jz,
                "Jxz_kgm2": mp.Jxz,
            },
            "geometry": {
                "wing_area_m2": lay.S,
                "span_m": lay.span,
                "chord_m": lay.mac,
                "oswald_e": am.oswald_e,
            },
            "aero": {
                "model": "tabulated",
                "tables": {k: [float(x) for x in getattr(t, k)] for k in t._fields},
                "ground_effect": True,
                "longitudinal": longitudinal,
                "lateral": {k: der[k] for k in lat_keys},
                "stall": {"M": 50.0, "alpha0_rad": am.alpha_stall},
            },
            "propulsion": {
                "prop_diameter_m": pr.diameter,
                "C_T": list(pr.c_t),
                "C_Q": list(pr.c_q),
                "windmill_cd": 0.02,
                "motor": {
                    "KV_rpm_per_volt": pr.kv_rpm_per_volt,
                    "R_ohm": pr.r_motor,
                    "i0_a": pr.i0,
                },
                "battery": {
                    "n_cells": pr.n_cells,
                    "v_cell_v": 3.7,
                    "capacity_ah": pr.capacity_ah,
                    "r_internal_ohm": pr.r_internal,
                    "avionics_current_a": pr.avionics_current,
                    "ocv_curve": LIPO_OCV,
                },
            },
            "actuators": _actuators(rng),
            "sensors": _sensors(rng),
        }
    )


def build(d: Design) -> Candidate:
    """Raises :class:`Rejected` (with the reason) for infeasible designs."""
    lay = layout(d)
    v_cruise, cl_cruise, alpha_cruise, p_cruise = _speeds(d, lay)
    x_np = neutral_point(d, lay, v_cruise, alpha_cruise)
    mp = mass_properties(d, lay, x_np - d.static_margin * lay.mac, p_cruise)
    am = aero_model(d, lay, mp.cg, v_cruise, cl_cruise)
    pr = size_propulsion(d, mp.mass, mp.battery_energy_wh)
    af = assemble(d, lay, mp, am, pr)
    meta = {
        "id": af.id,
        "cls": d.cls,
        "seed": d.seed,
        "mass_kg": mp.mass,
        "span_m": lay.span,
        "wing_area_m2": lay.S,
        "mac_m": lay.mac,
        "wing_loading": mp.mass * G0 / lay.S,
        "x_np_m": x_np,
        "x_cg_m": float(mp.cg[0]),
        "static_margin": (x_np - float(mp.cg[0])) / lay.mac,
        "cl_max": am.cl_max,
        "alpha_stall_deg": math.degrees(am.alpha_stall),
        "v_stall": stall_speed(mp.mass, lay.S, am.cl_max),
        "v_cruise": v_cruise,
        "p_max_w": pr.p_max,
        "n_cells": pr.n_cells,
        "battery_wh": mp.battery_energy_wh,
        **{f"mass_{k}": v for k, v in mp.budget.items()},
    }
    return Candidate(d, af, meta)
