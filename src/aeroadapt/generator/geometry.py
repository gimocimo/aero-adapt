"""Design → layout (all dimensions and stations) → AeroSandbox ``Airplane`` (M2 spec §4.1 step 2).

Geometry axes are AeroSandbox's: x aft from the nose, y to starboard, z up. Lengths in metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import aerosandbox as asb
import numpy as np

from aeroadapt.generator.design import Design


@dataclass(frozen=True)
class Layout:
    # wing
    S: float
    span: float
    root_chord: float
    tip_chord: float
    mac: float
    y_mac: float
    x_wing: float  # root leading edge
    z_wing: float
    tan_le: float  # leading-edge sweep
    x_qmac: float  # quarter point of the mean aerodynamic chord
    # horizontal tail (or V-tail panels: area of both panels, span tip to tip)
    S_h: float
    b_h: float
    c_h: float
    x_h: float  # leading edge at the root
    z_h: float
    # vertical tail(s): area per fin, height
    S_v: float
    h_v: float
    c_v: float
    x_v: float
    n_fins: int
    y_fins: float  # lateral station of each fin (0 for a single centreline fin)
    vtail_dihedral: float  # rad, V-tail only
    # fuselage / pod and booms
    length: float
    radius: float
    boom_radius: float
    tail_arm: float


def layout(d: Design) -> Layout:
    b = d.span
    S = b * b / d.aspect_ratio
    lam = d.taper
    cr = 2.0 * S / (b * (1.0 + lam))
    ct = lam * cr
    mac = 2.0 / 3.0 * cr * (1.0 + lam + lam * lam) / (1.0 + lam)
    y_mac = b / 6.0 * (1.0 + 2.0 * lam) / (1.0 + lam)
    tan_le = math.tan(math.radians(d.sweep_deg)) + (1.0 - lam) / (d.aspect_ratio * (1.0 + lam))
    L = d.fuselage_length_ratio * b
    radius = 0.5 * d.fuselage_diameter_ratio * L
    x_wing = {"twin_boom": 0.32, "glider": 0.24}.get(d.cls, 0.33 if d.pusher else 0.27) * L
    z_wing = 0.45 * radius if d.cls != "twin_boom" else 0.3 * radius
    x_qmac = x_wing + y_mac * tan_le + 0.25 * mac
    if d.cls == "twin_boom":
        tail_arm = d.tail_arm_ratio * 0.7 * b
    elif d.cls == "glider":
        tail_arm = d.tail_arm_ratio * 1.1 * L
    else:
        tail_arm = d.tail_arm_ratio * L

    S_h = d.h_volume * S * mac / tail_arm
    S_v = d.v_volume * S * b / tail_arm  # total vertical area
    n_fins, y_fins = (2, 0.18 * b) if d.cls == "twin_boom" else (1, 0.0)
    if d.cls == "v_tail":
        # Two panels whose projections supply the horizontal and vertical areas.
        dihedral = math.atan(math.sqrt(S_v / S_h))
        area = S_h + S_v
        b_h = math.sqrt(d.h_aspect_ratio * area)
        c_h = area / b_h
        S_h_out, S_v_out, h_v, c_v = area, 0.0, 0.0, 0.0
    else:
        dihedral = 0.0
        b_h = 2.0 * y_fins if d.cls == "twin_boom" else math.sqrt(d.h_aspect_ratio * S_h)
        c_h = S_h / b_h
        S_h_out = S_h
        S_v_out = S_v / n_fins
        h_v = math.sqrt(d.v_aspect_ratio * S_v_out)
        c_v = S_v_out / h_v
    x_h = x_qmac + tail_arm - 0.25 * c_h
    x_v = x_qmac + tail_arm - 0.25 * max(c_v, 1e-6) - 0.1 * c_h
    z_h = h_v * 0.95 if d.cls == "t_tail" else 0.0
    return Layout(
        S=S,
        span=b,
        root_chord=cr,
        tip_chord=ct,
        mac=mac,
        y_mac=y_mac,
        x_wing=x_wing,
        z_wing=z_wing,
        tan_le=tan_le,
        x_qmac=x_qmac,
        S_h=S_h_out,
        b_h=b_h,
        c_h=c_h,
        x_h=x_h,
        z_h=z_h,
        S_v=S_v_out,
        h_v=h_v,
        c_v=c_v,
        x_v=x_v,
        n_fins=n_fins,
        y_fins=y_fins,
        vtail_dihedral=dihedral,
        length=L,
        radius=radius,
        boom_radius=max(0.008, 0.006 * b),
        tail_arm=tail_arm,
    )


TRANSITION = 0.05  # forced transition at 5 % chord, upper and lower surfaces


class TrippedAirfoil(asb.Airfoil):
    """An airfoil whose NeuralFoil polars use forced transition near the leading edge.

    Small UAVs fly at chord Reynolds numbers of 1e5-5e5, where free transition gives
    laminar-separation kinks in the lift curve (strongest on thin tail sections). Rough
    surfaces, hinge gaps and atmospheric turbulence trip real boundary layers, so the fleet
    uses tripped polars: smooth lift curves, slightly higher drag.
    """

    def get_aero_from_neuralfoil(self, alpha, Re, **kwargs):  # type: ignore[override]
        kwargs.update(xtr_upper=TRANSITION, xtr_lower=TRANSITION)
        return super().get_aero_from_neuralfoil(alpha, Re, **kwargs)


@cache
def airfoil(name: str) -> asb.Airfoil:
    return TrippedAirfoil(name)


def _surface(
    name: str, chord_frac: float, deflection: float, symmetric: bool
) -> asb.ControlSurface:
    return asb.ControlSurface(
        name=name, symmetric=symmetric, hinge_point=1.0 - chord_frac, deflection=deflection
    )


def airplane(
    d: Design,
    lay: Layout,
    xyz_ref: tuple[float, float, float],
    aileron: float = 0.0,
    elevator: float = 0.0,
    rudder: float = 0.0,
) -> asb.Airplane:
    """The AeroSandbox model with control deflections in degrees (AeroSandbox signs: positive is
    trailing edge down for the wing and tailplane, and to port for a fin)."""
    wing_af, tail_af = airfoil(d.wing_airfoil), airfoil(d.tail_airfoil)
    half = 0.5 * lay.span
    y_ail = (1.0 - d.aileron_span) * half
    tan_dih = math.tan(math.radians(d.dihedral_deg))

    def wing_xsec(y: float, surfaces: list[asb.ControlSurface]) -> asb.WingXSec:
        chord = lay.root_chord + (lay.tip_chord - lay.root_chord) * y / half
        return asb.WingXSec(
            xyz_le=[lay.x_wing + y * lay.tan_le, y, lay.z_wing + y * tan_dih],
            chord=chord,
            twist=d.wing_incidence_deg,
            airfoil=wing_af,
            control_surfaces=surfaces,
        )

    wing = asb.Wing(
        name="wing",
        symmetric=True,
        xsecs=[
            wing_xsec(0.0, []),
            wing_xsec(y_ail, [_surface("aileron", d.aileron_chord, aileron, symmetric=False)]),
            wing_xsec(half, []),
        ],
    )
    wings = [wing]
    if d.cls == "v_tail":
        # Ruddervators: a symmetric deflection pitches, an antisymmetric one yaws. Only one of
        # the two is non-zero in any finite-difference evaluation.
        sym = rudder == 0.0
        defl = elevator if sym else rudder
        s = 0.5 * lay.b_h
        g = lay.vtail_dihedral
        wings.append(
            asb.Wing(
                name="vtail",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[lay.x_h, 0.0, 0.0],
                        chord=lay.c_h * 1.15,
                        airfoil=tail_af,
                        control_surfaces=[_surface("ruddervator", d.elevator_chord, defl, sym)],
                    ),
                    asb.WingXSec(
                        xyz_le=[lay.x_h + 0.3 * lay.c_h, s * math.cos(g), s * math.sin(g)],
                        chord=lay.c_h * 0.85,
                        airfoil=tail_af,
                    ),
                ],
            )
        )
    else:
        wings.append(
            asb.Wing(
                name="htail",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[lay.x_h, 0.0, lay.z_h],
                        chord=lay.c_h * 1.15,
                        airfoil=tail_af,
                        control_surfaces=[_surface("elevator", d.elevator_chord, elevator, True)],
                    ),
                    asb.WingXSec(
                        xyz_le=[lay.x_h + 0.15 * lay.c_h, 0.5 * lay.b_h, lay.z_h],
                        chord=lay.c_h * 0.85,
                        airfoil=tail_af,
                    ),
                ],
            )
        )
        fin_sweep = math.tan(math.radians(30.0))
        for i in range(lay.n_fins):
            y = 0.0 if lay.n_fins == 1 else (-1.0) ** i * lay.y_fins
            wings.append(
                asb.Wing(
                    name=f"fin{i}",
                    symmetric=False,
                    xsecs=[
                        asb.WingXSec(
                            xyz_le=[lay.x_v, y, 0.0],
                            chord=lay.c_v * 1.25,
                            airfoil=tail_af,
                            control_surfaces=[
                                _surface("rudder", d.rudder_chord, rudder, symmetric=True)
                            ],
                        ),
                        asb.WingXSec(
                            xyz_le=[lay.x_v + lay.h_v * fin_sweep, y, lay.h_v],
                            chord=lay.c_v * 0.75,
                            airfoil=tail_af,
                        ),
                    ],
                )
            )
    return asb.Airplane(
        name=f"{d.cls}-{d.seed}",
        xyz_ref=list(xyz_ref),
        wings=wings,
        fuselages=_fuselages(d, lay),
        s_ref=lay.S,
        c_ref=lay.mac,
        b_ref=lay.span,
    )


def _fuselages(d: Design, lay: Layout) -> list[asb.Fuselage]:
    L, R = lay.length, lay.radius
    if d.cls == "twin_boom":
        stations = [(0.0, 0.3), (0.15, 1.0), (0.8, 1.0), (1.0, 0.5)]
    else:
        stations = [(0.0, 0.25), (0.12, 1.0), (0.55, 1.0), (1.0, 0.35)]
    body = asb.Fuselage(
        name="fuselage",
        xsecs=[asb.FuselageXSec(xyz_c=[f * L, 0.0, 0.0], radius=r * R) for f, r in stations],
    )
    out = [body]
    tail_end = lay.x_h + lay.c_h * 1.15
    if d.cls == "twin_boom":
        for sign in (-1.0, 1.0):
            out.append(
                asb.Fuselage(
                    name="boom",
                    xsecs=[
                        asb.FuselageXSec(xyz_c=[x, sign * lay.y_fins, 0.0], radius=lay.boom_radius)
                        for x in (lay.x_wing, tail_end)
                    ],
                )
            )
    elif tail_end > L:
        # A tail boom from the end of the fuselage to the tail.
        out.append(
            asb.Fuselage(
                name="boom",
                xsecs=[
                    asb.FuselageXSec(xyz_c=[x, 0.0, 0.0], radius=max(lay.boom_radius, 0.35 * R))
                    for x in (L * 0.95, tail_end)
                ],
            )
        )
    return out


def wetted_fuselage_area(lay: Layout) -> float:
    return 2.0 * math.pi * lay.radius * lay.length * 0.8


def mesh(plane: asb.Airplane) -> tuple[np.ndarray, np.ndarray]:
    """Triangle mesh (points, faces) of the whole aircraft in geometry axes."""
    points, faces = plane.mesh_body(method="tri", thin_wings=False, stack_meshes=True)
    return np.asarray(points), np.asarray(faces)
