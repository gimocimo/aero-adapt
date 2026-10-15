"""Electric propulsion sizing (M2 spec §4.1 step 5).

The propeller family scales the Beard & McLain (McLain addendum) propeller by pitch ratio:
higher pitch moves the zero-thrust advance ratio out and raises the power coefficient.
The motor is sized so that full throttle absorbs about 75 % of the maximum electrical power
statically, with 8 % copper loss at that power. Approximate by design; documented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from aeroadapt.generator.design import Design

BASE_C_T = (0.09357, -0.06044, -0.1079)
BASE_C_Q = (0.005230, 0.004970, -0.01664)
BASE_PITCH_RATIO = 0.6
RHO0 = 1.225
LIPO_OCV = [
    [0.0, 3.30], [0.1, 3.60], [0.2, 3.70], [0.3, 3.74], [0.4, 3.77], [0.5, 3.80],
    [0.6, 3.85], [0.7, 3.90], [0.8, 3.98], [0.9, 4.08], [1.0, 4.20],
]  # fmt: skip


@dataclass(frozen=True)
class Propulsion:
    diameter: float
    c_t: tuple[float, float, float]
    c_q: tuple[float, float, float]
    kv_rpm_per_volt: float
    r_motor: float
    i0: float
    n_cells: int
    capacity_ah: float
    r_internal: float
    avionics_current: float
    p_max: float


def cells_for(p_max: float) -> int:
    for limit, cells in ((250.0, 3), (600.0, 4), (1_500.0, 6), (3_000.0, 8)):
        if p_max <= limit:
            return cells
    return 12


def size_propulsion(d: Design, mass: float, energy_wh: float) -> Propulsion:
    p_max = d.power_to_weight * mass
    n = cells_for(p_max)
    v_nom = 3.7 * n
    diameter = min(max(d.prop_diameter_ratio * d.span, 0.18), 0.66)
    s = BASE_PITCH_RATIO / d.pitch_ratio  # advance-ratio scale
    k_t = (d.pitch_ratio / BASE_PITCH_RATIO) ** 0.6
    k_q = (d.pitch_ratio / BASE_PITCH_RATIO) ** 1.4
    c_t = (k_t * BASE_C_T[0], k_t * BASE_C_T[1] * s, k_t * BASE_C_T[2] * s * s)
    c_q = (k_q * BASE_C_Q[0], k_q * BASE_C_Q[1] * s, k_q * BASE_C_Q[2] * s * s)
    # Static full-throttle shaft power ≈ 0.75 P_max: 2π ρ n³ D⁵ C_Q0 → rev/s.
    rev_s = (0.75 * p_max / (2.0 * math.pi * RHO0 * diameter**5 * c_q[0])) ** (1.0 / 3.0)
    kv = 60.0 * rev_s / (0.85 * v_nom)
    capacity = energy_wh / v_nom
    return Propulsion(
        diameter=diameter,
        c_t=c_t,
        c_q=c_q,
        kv_rpm_per_volt=kv,
        r_motor=0.08 * v_nom * v_nom / p_max,
        i0=0.03 * p_max / v_nom,
        n_cells=n,
        capacity_ah=capacity,
        r_internal=n * min(0.064 / capacity, 0.05),
        avionics_current=0.3 + 0.06 * mass,
        p_max=p_max,
    )
