"""Write an aerojax airframe as a JSBSim aircraft definition (M2 spec §5).

The JSBSim model reproduces aerojax's aerodynamics term by term: static coefficients as
tables over the aero v2 grid (the B&M formulas sampled, for an analytic airframe), and the
damping and control derivatives scaled by the effectiveness tables. Forces are written in
body axes with lift and drag rotated by α only, as in B&M (stability axes). Everything else is
JSBSim's own: equations of motion, integrator, atmosphere, gravity and Earth model.

Control surfaces are driven through ``fcs/da-rad``, ``fcs/de-rad`` and ``fcs/dr-rad`` in B&M
signs; actuator dynamics stay in our layer, as with aerojax. Propulsion is not exported:
the vehicle shell computes thrust and propeller torque with aerojax's motor, propeller and
battery models and applies them as body-axis external reactions (``external_reactions/thrust``
in lbf and ``external_reactions/prop_torque`` in lbf·ft, both zero by default).
"""

from __future__ import annotations

import numpy as np

from aeroadapt.airframes.params import AirframeParams, alpha_grid, bm_tables

_ALPHA = "aero/alpha-rad"
_SURFACES = ("fcs/da-rad", "fcs/de-rad", "fcs/dr-rad")


def _table(name: str, values: np.ndarray) -> str:
    rows = "\n".join(
        f"            {a:.9f} {v:.12g}" for a, v in zip(alpha_grid(), values, strict=True)
    )
    return f"""    <function name="aero/coefficient/{name}">
      <table>
        <independentVar lookup="row">{_ALPHA}</independentVar>
        <tableData>
{rows}
        </tableData>
      </table>
    </function>"""


def _p(prop: str) -> str:
    return f"<property>{prop}</property>"


def _v(x: float) -> str:
    return f"<value>{x:.12g}</value>"


def _term(coef: float, *factors: str) -> str:
    return f"<product>{_v(coef)}{''.join(factors)}</product>"


def _rate(coef: float, half: str, rate: str) -> str:
    """A damping term, coef · (c or b)/(2V) · rate · η_damp."""
    return _term(coef, _p(half), _p(rate), _p("aero/coefficient/eta_damp"))


def _ctrl(coef: float, surface: str) -> str:
    return _term(coef, _p(surface), _p("aero/coefficient/eta_ctrl"))


def _sum_function(name: str, *terms: str) -> str:
    return f'    <function name="{name}">\n      <sum>{"".join(terms)}</sum>\n    </function>'


def airframe_xml(p: AirframeParams, name: str, description: str) -> str:
    t = p.tables if p.tables is not None else bm_tables(p)
    da, de, dr = _SURFACES
    ci2v, bi2v = "aero/ci2vel", "aero/bi2vel"
    pr, qr, rr = "velocities/p-rad_sec", "velocities/q-rad_sec", "velocities/r-rad_sec"
    beta = "aero/beta-rad"

    lift = _sum_function(
        "aero/coefficient/CL_total",
        _p("aero/coefficient/C_L"),
        _rate(p.C_L_q, ci2v, qr),
        _ctrl(p.C_L_delta_e, de),
    )
    drag_terms = [_p("aero/coefficient/C_D"), _rate(p.C_D_q, ci2v, qr), _ctrl(p.C_D_delta_e, de)]
    ground = ""
    if p.ground_effect:
        # McCormick: induced drag C_L²/(π e AR) scaled by φ = x²/(1 + x²), x = 16 h/b.
        x = (
            f"<product>{_v(16.0)}<quotient>{_p('position/h-agl-ft')}"
            f"{_p('metrics/bw-ft')}</quotient></product>"
        )
        x2 = f"<pow>{x}{_v(2.0)}</pow>"
        phi = f"<quotient>{x2}<sum>{_v(1.0)}{x2}</sum></quotient>"
        k = -p.ground_effect / (np.pi * p.e_oswald * p.AR)
        ground = (
            '    <function name="aero/coefficient/CD_ground">\n      <product>'
            f"{_v(k)}{_p('aero/coefficient/C_L')}{_p('aero/coefficient/C_L')}"
            f"<difference>{_v(1.0)}{phi}</difference></product>\n    </function>"
        )
        drag_terms.append(_p("aero/coefficient/CD_ground"))
    drag = _sum_function("aero/coefficient/CD_total", *drag_terms)
    side = _sum_function(
        "aero/coefficient/CY_total",
        _v(p.C_Y_0),
        _term(p.C_Y_beta, _p(beta)),
        _rate(p.C_Y_p, bi2v, pr),
        _rate(p.C_Y_r, bi2v, rr),
        _ctrl(p.C_Y_delta_a, da),
        _ctrl(p.C_Y_delta_r, dr),
    )
    roll = _sum_function(
        "aero/coefficient/Cl_total",
        _v(p.C_ell_0),
        _term(p.C_ell_beta, _p(beta)),
        _rate(p.C_ell_p, bi2v, pr),
        _rate(p.C_ell_r, bi2v, rr),
        _ctrl(p.C_ell_delta_a, da),
        _ctrl(p.C_ell_delta_r, dr),
    )
    pitch = _sum_function(
        "aero/coefficient/Cm_total",
        _p("aero/coefficient/C_m"),
        _rate(p.C_m_q, ci2v, qr),
        _ctrl(p.C_m_delta_e, de),
    )
    yaw = _sum_function(
        "aero/coefficient/Cn_total",
        _v(p.C_n_0),
        _term(p.C_n_beta, _p(beta)),
        _rate(p.C_n_p, bi2v, pr),
        _rate(p.C_n_r, bi2v, rr),
        _ctrl(p.C_n_delta_a, da),
        _ctrl(p.C_n_delta_r, dr),
    )
    qs = _p("aero/qbar-psf") + _p("metrics/Sw-sqft")
    cos_a, sin_a = f"<cos>{_p(_ALPHA)}</cos>", f"<sin>{_p(_ALPHA)}</sin>"
    cl, cd = _p("aero/coefficient/CL_total"), _p("aero/coefficient/CD_total")

    def axis(axis_name: str, body: str) -> str:
        return (
            f'    <axis name="{axis_name}">\n'
            f'      <function name="aero/force-moment/{axis_name}">{body}</function>\n'
            "    </axis>"
        )

    x_force = (
        f"<product>{qs}<sum><product>{_v(-1.0)}{cos_a}{cd}</product>"
        f"<product>{sin_a}{cl}</product></sum></product>"
    )
    z_force = (
        f"<product>{qs}<sum><product>{_v(-1.0)}{sin_a}{cd}</product>"
        f"<product>{_v(-1.0)}{cos_a}{cl}</product></sum></product>"
    )
    y_force = f"<product>{qs}{_p('aero/coefficient/CY_total')}</product>"
    l_moment = f"<product>{qs}{_p('metrics/bw-ft')}{_p('aero/coefficient/Cl_total')}</product>"
    m_moment = f"<product>{qs}{_p('metrics/cbarw-ft')}{_p('aero/coefficient/Cm_total')}</product>"
    n_moment = f"<product>{qs}{_p('metrics/bw-ft')}{_p('aero/coefficient/Cn_total')}</product>"

    tables = "\n".join(
        _table(k, getattr(t, k)) for k in ("C_L", "C_D", "C_m", "eta_ctrl", "eta_damp")
    )
    surfaces = "\n".join(f'    <property value="0">{s}</property>' for s in _SURFACES)
    origin = "<x>0</x><y>0</y><z>0</z>"
    return f"""<?xml version="1.0"?>
<!-- Generated by aeroadapt.sim.jsbsim.export from the aerojax airframe '{name}'. Do not edit. -->
<fdm_config name="{name}" version="2.0" release="BETA">
  <fileheader>
    <author>aeroadapt</author>
    <description>{description}</description>
  </fileheader>
  <metrics>
    <wingarea unit="M2">{p.S:.12g}</wingarea>
    <wingspan unit="M">{p.b:.12g}</wingspan>
    <chord unit="M">{p.c:.12g}</chord>
    <htailarea unit="M2">0</htailarea>
    <htailarm unit="M">0</htailarm>
    <vtailarea unit="M2">0</vtailarea>
    <vtailarm unit="M">0</vtailarm>
    <location name="AERORP" unit="M">{origin}</location>
    <location name="EYEPOINT" unit="M">{origin}</location>
    <location name="VRP" unit="M">{origin}</location>
  </metrics>
  <mass_balance>
    <ixx unit="KG*M2">{p.Jx:.12g}</ixx>
    <iyy unit="KG*M2">{p.Jy:.12g}</iyy>
    <izz unit="KG*M2">{p.Jz:.12g}</izz>
    <ixz unit="KG*M2">{p.Jxz:.12g}</ixz>
    <emptywt unit="KG">{p.mass:.12g}</emptywt>
    <location name="CG" unit="M">{origin}</location>
  </mass_balance>
  <ground_reactions/>
  <propulsion/>
  <external_reactions>
    <force name="thrust" frame="BODY" unit="LBS">
      <location unit="M">{origin}</location>
      <direction><x>1</x><y>0</y><z>0</z></direction>
    </force>
    <moment name="prop_torque" frame="BODY" unit="LBSFT">
      <direction><x>1</x><y>0</y><z>0</z></direction>
    </moment>
  </external_reactions>
  <flight_control name="FCS">
{surfaces}
  </flight_control>
  <aerodynamics>
{tables}
{ground}
{lift}
{drag}
{side}
{roll}
{pitch}
{yaw}
{axis("X", x_force)}
{axis("Y", y_force)}
{axis("Z", z_force)}
{axis("ROLL", l_moment)}
{axis("PITCH", m_moment)}
{axis("YAW", n_moment)}
  </aerodynamics>
</fdm_config>
"""
