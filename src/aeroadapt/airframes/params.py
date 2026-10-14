"""Flat, JAX-friendly airframe parameters used by the simulator's functional core.

``AirframeParams`` is a NamedTuple, so it is a pytree: it passes through ``jit`` and can be
batched with ``vmap`` across a fleet of airframes (M2).
"""

from __future__ import annotations

from typing import NamedTuple

from aeroadapt.airframes.schema import Airframe


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
    return AirframeParams(
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
    )
