"""Mass budget, centre of gravity and inertia (M2 spec §4.1 step 3).

The total mass follows from the sampled wing loading. Structure, propulsion, battery and
avionics are sized from rules of thumb; the payload takes the remainder, and a design is
rejected if the remainder is negative. The battery and payload sit together in the fuselage
at the station that puts the centre of gravity at the target static margin.

Positions use geometry axes (x aft from the nose, y starboard, z up). The returned inertia
is about the centre of gravity in body axes (x forward, y starboard, z down).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aeroadapt.generator.design import Design
from aeroadapt.generator.geometry import Layout, wetted_fuselage_area

G0 = 9.806_65
SPECIFIC_ENERGY_WH_KG = 160.0  # LiPo pack, including wiring and packaging
MOTOR_W_PER_KG = 3_000.0  # motor, ESC and propeller together


class Rejected(ValueError):
    """The design is infeasible; the message names the reason (counted in fleet statistics)."""


@dataclass(frozen=True)
class MassProperties:
    mass: float
    cg: np.ndarray  # geometry axes, m
    Jx: float
    Jy: float
    Jz: float
    Jxz: float  # body-axis product of inertia, B&M sign convention (J = [[Jx, 0, -Jxz], ...])
    budget: dict[str, float]  # component masses, kg
    battery_energy_wh: float
    x_battery: float


@dataclass(frozen=True)
class _Part:
    mass: float
    pos: tuple[float, float, float]
    # principal second moments about its own centroid in geometry axes (xx, yy, zz)
    inertia: tuple[float, float, float] = (0.0, 0.0, 0.0)


def mass_properties(
    d: Design, lay: Layout, x_cg_target: float, cruise_power_w: float
) -> MassProperties:
    mass = d.wing_loading * lay.S / G0
    b, c, L, R = lay.span, lay.mac, lay.length, lay.radius
    # Structure: areal densities scale mildly with size and aspect ratio (bending loads).
    size = (b / 2.5) ** 0.5
    wing_m = d.structure_density * lay.S * size * (1.0 + 0.03 * (d.aspect_ratio - 8.0))
    tail_area = lay.S_h + lay.n_fins * lay.S_v
    tail_m = 0.55 * d.structure_density * tail_area
    fus_m = 0.9 * d.structure_density * wetted_fuselage_area(lay) * size
    boom_m = 0.0
    if d.cls == "twin_boom":
        boom_m = 2 * 0.25 * (lay.x_h - lay.x_wing) * size
    elif lay.x_h + lay.c_h > L:
        boom_m = 0.25 * (lay.x_h + lay.c_h - 0.95 * L) * size
    p_max = d.power_to_weight * mass
    prop_m = p_max / MOTOR_W_PER_KG
    energy = cruise_power_w * d.endurance_min / 60.0 / 0.85  # Wh, 15 % reserve
    batt_m = energy / SPECIFIC_ENERGY_WH_KG
    avionics_m = 0.12 + 0.02 * mass
    fixed = wing_m + tail_m + fus_m + boom_m + prop_m + batt_m + avionics_m
    payload_m = mass - fixed
    if payload_m < 0.03 * mass:
        raise Rejected("mass budget: no payload margin")

    # Station of each part (geometry axes).
    x_wing_c = lay.x_wing + lay.y_mac * lay.tan_le + 0.4 * lay.mac
    x_tail = lay.x_h + 0.5 * lay.c_h
    x_prop = 0.02 * L if not d.pusher else (0.98 * L)
    parts = [
        _Part(
            wing_m,
            (x_wing_c, 0.0, lay.z_wing),
            (wing_m * b * b / 12.0, wing_m * c * c / 12.0, wing_m * (b * b + c * c) / 12.0),
        ),
        _Part(
            tail_m,
            (x_tail, 0.0, 0.3 * max(lay.h_v, lay.b_h * 0.2)),
            (tail_m * lay.b_h**2 / 12.0, tail_m * lay.c_h**2 / 12.0, tail_m * lay.b_h**2 / 12.0),
        ),
        _Part(
            fus_m,
            (0.45 * L, 0.0, 0.0),
            (fus_m * R * R, fus_m * (R * R / 2 + L * L / 12), fus_m * (R * R / 2 + L * L / 12)),
        ),
        _Part(prop_m, (x_prop, 0.0, 0.0)),
        _Part(avionics_m, (0.3 * L, 0.0, 0.0)),
    ]
    if boom_m > 0:
        x0, x1 = lay.x_wing if d.cls == "twin_boom" else 0.95 * L, lay.x_h + lay.c_h
        lb = x1 - x0
        y_b = lay.y_fins if d.cls == "twin_boom" else 0.0
        iyy = boom_m * lb * lb / 12.0
        parts.append(_Part(boom_m, (0.5 * (x0 + x1), 0.0, 0.0), (boom_m * y_b * y_b, iyy, iyy)))

    # Battery and payload share one station chosen to place the CG at the target.
    movable = batt_m + payload_m
    moment = sum(p.mass * p.pos[0] for p in parts)
    x_batt = (mass * x_cg_target - moment) / movable
    if not 0.04 * L <= x_batt <= 0.75 * L:
        raise Rejected("balance: battery station outside the fuselage")
    parts.append(_Part(batt_m, (x_batt, 0.0, 0.0)))
    parts.append(_Part(payload_m, (x_batt, 0.0, -0.3 * R)))

    total = sum(p.mass for p in parts)
    cg = np.sum([p.mass * np.asarray(p.pos) for p in parts], axis=0) / total
    # Inertia tensor about the CG in geometry axes, then to body axes (x→−x, z→−z).
    j = np.zeros((3, 3))
    for p in parts:
        r = np.asarray(p.pos) - cg
        j += np.diag(p.inertia) + p.mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    flip = np.diag([-1.0, 1.0, -1.0])
    jb = flip @ j @ flip
    return MassProperties(
        mass=total,
        cg=cg,
        Jx=float(jb[0, 0]),
        Jy=float(jb[1, 1]),
        Jz=float(jb[2, 2]),
        Jxz=float(-jb[0, 2]),
        budget={
            "wing": wing_m,
            "tail": tail_m,
            "fuselage": fus_m,
            "booms": boom_m,
            "propulsion": prop_m,
            "battery": batt_m,
            "avionics": avionics_m,
            "payload": payload_m,
        },
        battery_energy_wh=energy,
        x_battery=x_batt,
    )


def stall_speed(mass: float, S: float, cl_max: float, rho: float = 1.225) -> float:
    return math.sqrt(2.0 * mass * G0 / (rho * S * cl_max))
