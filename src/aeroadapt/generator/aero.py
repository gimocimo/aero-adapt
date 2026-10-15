"""Aerodynamics from geometry with AeroSandbox (M2 spec §3.2 and §4.1 step 4).

- Static coefficients over 360° of α: AeroSandbox's ``AeroBuildup`` (NeuralFoil section
  polars with post-stall behaviour, fuselage and interference terms) about the centre of
  gravity, densely in attached flow and coarsely beyond, then PCHIP-interpolated onto the
  aero v2 grid.
- Stability and damping derivatives: ``AeroBuildup`` at the cruise angle of attack.
- Control derivatives: central differences of ±5° deflections at cruise.
- Effectiveness η(α): the elevator's pitching-moment effectiveness across α, relative to
  cruise. It is applied to every control and, as an approximation, to the damping
  derivatives (both are dominated by tail-surface lift slope).

Post-stall data is approximate (documented in the spec, §12).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import aerosandbox as asb
import numpy as np
from scipy.interpolate import PchipInterpolator

from aeroadapt.airframes.params import AeroTables, alpha_grid
from aeroadapt.generator.design import Design
from aeroadapt.generator.geometry import Layout, airplane

SWEEP_DEG = np.concatenate(
    [np.arange(-180.0, -30.0, 10.0), np.arange(-30.0, 40.0, 1.0), np.arange(40.0, 180.1, 10.0)]
)
DEFLECTION_DEG = 5.0


@dataclass(frozen=True)
class AeroModel:
    tables: AeroTables
    derivatives: dict[str, float]  # aerojax / B&M names, per radian
    cl_max: float
    alpha_stall: float  # rad, angle of maximum C_L
    cd0: float  # parasitic drag coefficient of the parabolic fit
    oswald_e: float
    cl0: float
    cl_alpha: float
    cm0: float
    cm_alpha: float


def _op(alpha_deg: np.ndarray | float, v: float) -> asb.OperatingPoint:
    a = np.atleast_1d(np.asarray(alpha_deg, dtype=float))
    return asb.OperatingPoint(velocity=np.full(a.shape, v), alpha=a, beta=np.zeros(a.shape))


def _scalar(x: object) -> float:
    return float(np.asarray(x, dtype=float).ravel()[0])


def _run(plane: asb.Airplane, alpha_deg: np.ndarray | float, v: float) -> dict[str, np.ndarray]:
    out = asb.AeroBuildup(airplane=plane, op_point=_op(alpha_deg, v)).run()
    return {
        k: np.atleast_1d(np.asarray(out[k], dtype=float))
        for k in ("CL", "CD", "CY", "Cl", "Cm", "Cn")
    }


def neutral_point(d: Design, lay: Layout, v: float, alpha_deg: float) -> float:
    """x of the stick-fixed neutral point (geometry axes) from C_mα and C_Lα at ``alpha_deg``."""
    ref = lay.x_qmac
    s = asb.AeroBuildup(
        airplane=airplane(d, lay, (ref, 0.0, 0.0)),
        op_point=asb.OperatingPoint(velocity=v, alpha=alpha_deg),
    ).run_with_stability_derivatives(alpha=True, beta=False, p=False, q=False, r=False)
    cla, cma = _scalar(s["CLa"]), _scalar(s["Cma"])
    return ref - cma / cla * lay.mac


def _to_grid(values: np.ndarray) -> np.ndarray:
    return PchipInterpolator(np.radians(SWEEP_DEG), values)(alpha_grid())


def aero_model(
    d: Design, lay: Layout, cg: np.ndarray, v_cruise: float, cl_cruise: float
) -> AeroModel:
    ref = (float(cg[0]), float(cg[1]), float(cg[2]))
    base = airplane(d, lay, ref)
    sweep = _run(base, SWEEP_DEG, v_cruise)

    # Elevator effectiveness across α (for η) and at cruise.
    up = _run(airplane(d, lay, ref, elevator=DEFLECTION_DEG), SWEEP_DEG, v_cruise)
    dn = _run(airplane(d, lay, ref, elevator=-DEFLECTION_DEG), SWEEP_DEG, v_cruise)
    two_d = 2.0 * math.radians(DEFLECTION_DEG)
    cm_de = (up["Cm"] - dn["Cm"]) / two_d

    # Attached-flow fits (−4°…8°) for the parabolic polar and the linear coefficients.
    att = (SWEEP_DEG >= -4.0) & (SWEEP_DEG <= 8.0)
    a_rad = np.radians(SWEEP_DEG[att])
    cl_alpha, cl0 = np.polyfit(a_rad, sweep["CL"][att], 1)
    cm_alpha, cm0 = np.polyfit(a_rad, sweep["Cm"][att], 1)
    k_ind, cd0 = np.polyfit(sweep["CL"][att] ** 2, sweep["CD"][att], 1)
    aspect = lay.span**2 / lay.S
    oswald = float(np.clip(1.0 / (math.pi * aspect * max(k_ind, 1e-4)), 0.3, 1.0))

    pos = (SWEEP_DEG > 0.0) & (SWEEP_DEG <= 30.0)
    i_max = int(np.argmax(np.where(pos, sweep["CL"], -np.inf)))
    cl_max, alpha_stall = float(sweep["CL"][i_max]), math.radians(float(SWEEP_DEG[i_max]))

    # Cruise point: the α giving the cruise lift coefficient.
    alpha_cruise = float(
        np.clip((cl_cruise - cl0) / cl_alpha, math.radians(-2.0), math.radians(10.0))
    )
    i_cr = int(np.argmin(np.abs(np.radians(SWEEP_DEG) - alpha_cruise)))
    eta = np.clip(cm_de / cm_de[i_cr], 0.05, 1.2)
    eta = np.where(np.abs(np.radians(SWEEP_DEG)) <= np.radians(90.0), eta, np.minimum(eta, 0.3))

    sd = asb.AeroBuildup(
        airplane=base,
        op_point=asb.OperatingPoint(velocity=v_cruise, alpha=math.degrees(alpha_cruise)),
    ).run_with_stability_derivatives()

    def f(k: str) -> float:
        return _scalar(sd[k])

    def ctrl(surface: str) -> dict[str, float]:
        p = _run(
            airplane(d, lay, ref, **{surface: DEFLECTION_DEG}), math.degrees(alpha_cruise), v_cruise
        )
        m = _run(
            airplane(d, lay, ref, **{surface: -DEFLECTION_DEG}),
            math.degrees(alpha_cruise),
            v_cruise,
        )
        return {k: float((p[k] - m[k])[0] / two_d) for k in p}

    ail, ele, rud = ctrl("aileron"), ctrl("elevator"), ctrl("rudder")
    derivs = {
        "C_L_0": float(cl0),
        "C_L_alpha": float(cl_alpha),
        "C_L_q": f("CLq"),
        "C_L_delta_e": ele["CL"],
        "C_D_p": float(cd0),
        "C_D_q": f("CDq"),
        "C_D_delta_e": ele["CD"],
        "C_m_0": float(cm0),
        "C_m_alpha": float(cm_alpha),
        "C_m_q": f("Cmq"),
        "C_m_delta_e": ele["Cm"],
        "C_Y_0": 0.0,
        "C_Y_beta": f("CYb"),
        "C_Y_p": f("CYp"),
        "C_Y_r": f("CYr"),
        "C_Y_delta_a": ail["CY"],
        "C_Y_delta_r": rud["CY"],
        "C_ell_0": 0.0,
        "C_ell_beta": f("Clb"),
        "C_ell_p": f("Clp"),
        "C_ell_r": f("Clr"),
        "C_ell_delta_a": ail["Cl"],
        "C_ell_delta_r": rud["Cl"],
        "C_n_0": 0.0,
        "C_n_beta": f("Cnb"),
        "C_n_p": f("Cnp"),
        "C_n_r": f("Cnr"),
        "C_n_delta_a": ail["Cn"],
        "C_n_delta_r": rud["Cn"],
    }
    eta_grid = np.clip(_to_grid(eta), 0.05, 1.2)
    tables = AeroTables(
        C_L=_to_grid(sweep["CL"]),
        C_D=np.maximum(_to_grid(sweep["CD"]), 0.5 * float(np.min(sweep["CD"]))),
        C_m=_to_grid(sweep["Cm"]),
        eta_ctrl=eta_grid,
        eta_damp=eta_grid.copy(),
    )
    return AeroModel(
        tables=tables,
        derivatives=derivs,
        cl_max=cl_max,
        alpha_stall=alpha_stall,
        cd0=float(cd0),
        oswald_e=oswald,
        cl0=float(cl0),
        cl_alpha=float(cl_alpha),
        cm0=float(cm0),
        cm_alpha=float(cm_alpha),
    )
