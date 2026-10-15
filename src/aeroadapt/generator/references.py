"""V18: the generator's aerodynamics against published values for two real airframes
(M2 spec §10).

- **Skywalker X8** (flying wing; held out of the training fleet): published coefficients from
  Gryte et al. (2018), via PyFly's parameter file (MIT). Its planform is reconstructed from the
  published reference area, span and mean chord (taper and sweep are estimates from the
  aircraft's published dimensions); MH45 reflexed sections, 3° washout, winglets, and elevons
  on the outer 60 % of the span. The centre of gravity is placed at the static margin implied
  by the published C_mα / C_Lα.
- **Aerosonde** (B&M): span, area and chord from the book; the book gives no tail geometry, so
  only wing-dominated derivatives (C_Lα, C_ℓp) are compared, and the rest are recorded.

VLM/buildup estimates typically differ from wind-tunnel and flight-identified values by
20-30 % on secondary derivatives; the criteria (C_Lα within 15 %, C_mα stable, C_ℓp within
25 %) and the report say so.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import aerosandbox as asb

from aeroadapt.airframes import load_airframe
from aeroadapt.generator.aero import _scalar
from aeroadapt.generator.design import Design
from aeroadapt.generator.geometry import airfoil, airplane, layout

X8_PUBLISHED = {
    "C_L_alpha": 4.0203,
    "C_m_alpha": -0.2524,
    "C_m_q": -1.3012,
    "C_L_q": 3.87,
    "C_ell_p": -0.4042,
    "C_ell_beta": -0.0849,
    "C_n_beta": 0.0283,
    "C_n_r": -0.0720,
    "C_Y_beta": -0.2239,
    "C_m_delta_e": -0.2292,
    "C_ell_delta_a": 0.1202,
}
X8_MASS, X8_S, X8_B, X8_C = 3.364, 0.75, 2.1, 0.357_142_857
COMPARED = {"C_L_alpha": 0.15, "C_ell_p": 0.25}  # derivative → relative tolerance


@dataclass(frozen=True)
class Comparison:
    airframe: str
    rows: list[tuple[str, float, float]]  # (name, published, generator)

    def relative_error(self, name: str) -> float:
        pub, gen = next((p, g) for n, p, g in self.rows if n == name)
        return abs(gen - pub) / abs(pub)

    @property
    def passed(self) -> bool:
        gen = {n: g for n, _, g in self.rows}
        return all(self.relative_error(k) <= tol for k, tol in COMPARED.items()) and (
            gen["C_m_alpha"] < 0
        )


def _x8(
    xyz_ref: tuple[float, float, float], aileron: float = 0.0, elevator: float = 0.0
) -> asb.Airplane:
    """Elevons: an elevator command is a symmetric deflection, an aileron one antisymmetric."""
    cr, ct, half = 0.52, 0.19, X8_B / 2.0
    tan_le = math.tan(math.radians(28.0))
    af = airfoil("mh45")
    sym = aileron == 0.0
    surf = asb.ControlSurface(
        name="elevon",
        symmetric=sym,
        hinge_point=0.75,
        deflection=elevator if sym else aileron,
    )

    def xsec(y: float, twist: float, surfaces: list[asb.ControlSurface]) -> asb.WingXSec:
        chord = cr + (ct - cr) * y / half
        return asb.WingXSec(
            xyz_le=[y * tan_le, y, 0.0],
            chord=chord,
            twist=twist,
            airfoil=af,
            control_surfaces=surfaces,
        )

    wing = asb.Wing(
        name="wing",
        symmetric=True,
        xsecs=[xsec(0.0, 0.0, []), xsec(0.4 * half, -0.6, [surf]), xsec(half, -3.0, [])],
    )
    tip_x = half * tan_le
    winglet = asb.Wing(
        name="winglet",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[tip_x, half, 0.0], chord=ct, airfoil=airfoil("naca0008")),
            asb.WingXSec(
                xyz_le=[tip_x + 0.08, half, 0.12], chord=0.09, airfoil=airfoil("naca0008")
            ),
        ],
    )
    body = asb.Fuselage(
        name="pod",
        xsecs=[asb.FuselageXSec(xyz_c=[x, 0.0, 0.0], radius=r) for x, r in
               ((-0.12, 0.02), (0.0, 0.07), (0.35, 0.07), (0.55, 0.03))],
    )  # fmt: skip
    return asb.Airplane(
        name="x8", xyz_ref=list(xyz_ref), wings=[wing, winglet], fuselages=[body],
        s_ref=X8_S, c_ref=X8_C, b_ref=X8_B,
    )  # fmt: skip


def _derivatives(make: Callable[[], asb.Airplane], v: float, alpha_deg: float) -> dict[str, float]:
    op = asb.OperatingPoint(velocity=v, alpha=alpha_deg)
    sd = asb.AeroBuildup(airplane=make(), op_point=op).run_with_stability_derivatives()
    d = {
        "C_L_alpha": _scalar(sd["CLa"]),
        "C_m_alpha": _scalar(sd["Cma"]),
        "C_m_q": _scalar(sd["Cmq"]),
        "C_L_q": _scalar(sd["CLq"]),
        "C_ell_p": _scalar(sd["Clp"]),
        "C_ell_beta": _scalar(sd["Clb"]),
        "C_n_beta": _scalar(sd["Cnb"]),
        "C_n_r": _scalar(sd["Cnr"]),
        "C_Y_beta": _scalar(sd["CYb"]),
    }
    return d


def x8_comparison(v: float = 18.0) -> Comparison:
    cl = 2.0 * X8_MASS * 9.81 / (1.225 * v * v * X8_S)
    alpha = math.degrees((cl - 0.0867) / X8_PUBLISHED["C_L_alpha"])
    # Neutral point about the root leading edge, then the CG at the published static margin.
    d0 = _derivatives(lambda: _x8((0.0, 0.0, 0.0)), v, alpha)
    x_np = -d0["C_m_alpha"] / d0["C_L_alpha"] * X8_C
    sm = -X8_PUBLISHED["C_m_alpha"] / X8_PUBLISHED["C_L_alpha"]
    ref = (x_np - sm * X8_C, 0.0, 0.0)
    d = _derivatives(lambda: _x8(ref), v, alpha)
    two = 2.0 * math.radians(5.0)

    op = asb.OperatingPoint(velocity=v, alpha=alpha)

    def moment(key: str, surface: str) -> float:
        plus = asb.AeroBuildup(airplane=_x8(ref, **{surface: 5.0}), op_point=op).run()
        minus = asb.AeroBuildup(airplane=_x8(ref, **{surface: -5.0}), op_point=op).run()
        return (_scalar(plus[key]) - _scalar(minus[key])) / two

    # Elevon sign conventions differ between the sources; compare magnitudes for controls.
    d["C_m_delta_e"] = -abs(moment("Cm", "elevator"))
    d["C_ell_delta_a"] = abs(moment("Cl", "aileron"))
    return Comparison("Skywalker X8", [(k, X8_PUBLISHED[k], d[k]) for k in X8_PUBLISHED])


def aerosonde_design() -> Design:
    """B&M's Aerosonde wing (rectangular, AR 15.2) with an assumed inverted-V-like tail."""
    af = load_airframe("aerosonde_bm").geometry
    return Design(
        cls="v_tail", seed=0, span=af.span_m, aspect_ratio=af.span_m**2 / af.wing_area_m2,
        taper=1.0, sweep_deg=0.0, dihedral_deg=2.0, wing_airfoil="naca2412",
        wing_incidence_deg=2.0, tail_airfoil="naca0010", h_volume=0.45, v_volume=0.03,
        h_aspect_ratio=4.5, v_aspect_ratio=1.5, aileron_chord=0.25, aileron_span=0.45,
        elevator_chord=0.3, rudder_chord=0.3, fuselage_length_ratio=0.6,
        fuselage_diameter_ratio=0.1, tail_arm_ratio=0.6, wing_loading=196.0,
        power_to_weight=150.0, endurance_min=60.0, static_margin=0.1, pusher=True,
        prop_diameter_ratio=0.175, pitch_ratio=0.6, structure_density=1.0,
    )  # fmt: skip


def aerosonde_comparison() -> Comparison:
    p = load_airframe("aerosonde_bm").aero
    published = {
        "C_L_alpha": p.longitudinal.C_L_alpha,
        "C_ell_p": p.lateral.C_ell_p,
        "C_m_alpha": p.longitudinal.C_m_alpha,
        "C_m_q": p.longitudinal.C_m_q,
        "C_L_q": p.longitudinal.C_L_q,
        "C_ell_beta": p.lateral.C_ell_beta,
        "C_n_beta": p.lateral.C_n_beta,
        "C_n_r": p.lateral.C_n_r,
    }
    d = aerosonde_design()
    lay = layout(d)
    ref = (lay.x_qmac + 0.1 * lay.mac, 0.0, 0.0)  # 10 % static margin-ish CG (the book's is ~49 %)
    gen = _derivatives(lambda: airplane(d, lay, ref), 25.0, 2.0)
    return Comparison("Aerosonde (B&M)", [(k, float(v), gen[k]) for k, v in published.items()])


def v18() -> list[Comparison]:
    return [aerosonde_comparison(), x8_comparison()]


if __name__ == "__main__":
    for c in v18():
        print(c.airframe, "passed" if c.passed else "FAILED")
        for name, pub, gen in c.rows:
            rel = (gen - pub) / abs(pub)
            print(f"  {name:14s} published {pub:8.4f}  generator {gen:8.4f}  ({rel:+.0%})")
