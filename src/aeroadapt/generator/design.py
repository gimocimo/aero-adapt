"""Design space of the procedural airframe generator (M2 spec §4.1 step 1 and §4.2).

A :class:`Design` holds every sampled choice; everything else (geometry, mass, aerodynamics,
propulsion) is computed from it deterministically. Ranges are by configuration class and
follow small-UAV practice (tail volume coefficients, wing loadings and power-to-weight ratios
from Raymer and Gundlach, *Designing Unmanned Aircraft Systems*).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

ConfigClass = Literal["conventional", "t_tail", "v_tail", "twin_boom", "glider"]
CLASSES: tuple[ConfigClass, ...] = ("conventional", "t_tail", "v_tail", "twin_boom", "glider")

WING_AIRFOILS = ("naca2412", "naca4412", "naca4415", "clarky", "sd7062", "e205", "mh32")
GLIDER_AIRFOILS = ("sd7037", "e387", "ag35", "s3021", "mh32")
TAIL_AIRFOILS = ("naca0008", "naca0010", "naca0012")
FUSELAGE_LENGTH = {"glider": (0.45, 0.6), "twin_boom": (0.3, 0.45)}  # / span


@dataclass(frozen=True)
class Design:
    cls: ConfigClass
    seed: int
    # wing
    span: float  # m
    aspect_ratio: float
    taper: float  # tip / root chord
    sweep_deg: float  # quarter-chord sweep
    dihedral_deg: float
    wing_airfoil: str
    wing_incidence_deg: float
    # tails
    tail_airfoil: str
    h_volume: float  # horizontal tail volume coefficient V_h = S_h l_h / (S c)
    v_volume: float  # vertical tail volume coefficient V_v = S_v l_v / (S b)
    h_aspect_ratio: float
    v_aspect_ratio: float
    # control surfaces (fractions of local chord; aileron also of semi-span)
    aileron_chord: float
    aileron_span: float
    elevator_chord: float
    rudder_chord: float
    # fuselage
    fuselage_length_ratio: float  # fuselage length / span
    fuselage_diameter_ratio: float  # diameter / fuselage length
    tail_arm_ratio: float  # tail arm / fuselage length
    # sizing
    wing_loading: float  # N/m²
    power_to_weight: float  # W/kg, maximum electrical power
    endurance_min: float
    static_margin: float  # fraction of the mean aerodynamic chord
    pusher: bool
    prop_diameter_ratio: float  # propeller diameter / span
    pitch_ratio: float  # propeller pitch / diameter
    structure_density: float  # kg/m² of wing area (wing skin, spars), before scaling

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


MASS_RANGE_KG = (0.8, 15.0)  # doc 03 §2.1


def _wing_loading(
    rng: np.random.Generator, span: float, aspect_ratio: float, glider: bool
) -> float:
    """Wing loading within the class range, narrowed so that the mass stays in MASS_RANGE_KG."""
    area = span * span / aspect_ratio
    lo, hi = (20.0, 60.0) if glider else (40.0, 150.0)
    lo = max(lo, MASS_RANGE_KG[0] * 9.806_65 / area)
    hi = min(hi, MASS_RANGE_KG[1] * 9.806_65 / area)
    return _u(rng, lo, hi) if lo < hi else hi


def sample_design(cls: ConfigClass, seed: int) -> Design:
    """Draw one design of the given class. Deterministic in (cls, seed)."""
    rng = np.random.default_rng([CLASSES.index(cls), seed])
    glider = cls == "glider"
    span = _u(rng, 2.0, 4.0) if glider else _u(rng, 1.0, 4.0)
    if cls == "v_tail":
        span = _u(rng, 1.0, 3.0)
    aspect_ratio = _u(rng, 12.0, 20.0) if glider else _u(rng, 6.0, 11.0)
    return Design(
        cls=cls,
        seed=seed,
        span=span,
        aspect_ratio=aspect_ratio,
        taper=_u(rng, 0.4, 0.8) if glider else _u(rng, 0.45, 1.0),
        sweep_deg=_u(rng, 0.0, 3.0) if glider else _u(rng, 0.0, 8.0),
        dihedral_deg=_u(rng, 3.0, 7.0) if glider else _u(rng, 2.0, 7.0),
        wing_airfoil=str(rng.choice(GLIDER_AIRFOILS if glider else WING_AIRFOILS)),
        wing_incidence_deg=_u(rng, 0.0, 3.0),
        tail_airfoil=str(rng.choice(TAIL_AIRFOILS)),
        h_volume=_u(rng, 0.35, 0.6),
        v_volume=_u(rng, 0.025, 0.05) if not glider else _u(rng, 0.02, 0.035),
        h_aspect_ratio=_u(rng, 3.5, 5.5),
        v_aspect_ratio=_u(rng, 1.2, 2.0),
        aileron_chord=_u(rng, 0.2, 0.3),
        aileron_span=_u(rng, 0.35, 0.55),
        elevator_chord=_u(rng, 0.25, 0.4),
        rudder_chord=_u(rng, 0.25, 0.4),
        fuselage_length_ratio=(
            _u(rng, 0.45, 0.6)
            if glider
            else _u(rng, 0.3, 0.45)
            if cls == "twin_boom"
            else _u(rng, 0.55, 0.85)
        ),
        fuselage_diameter_ratio=_u(rng, 0.07, 0.12),
        tail_arm_ratio=_u(rng, 0.5, 0.65),
        wing_loading=_wing_loading(rng, span, aspect_ratio, glider),
        power_to_weight=_u(rng, 60.0, 110.0) if glider else _u(rng, 120.0, 250.0),
        endurance_min=_u(rng, 45.0, 120.0) if glider else _u(rng, 30.0, 90.0),
        static_margin=_u(rng, 0.06, 0.15),
        pusher=cls == "twin_boom" or (cls == "conventional" and bool(rng.random() < 0.25)),
        prop_diameter_ratio=_u(rng, 0.08, 0.13) if glider else _u(rng, 0.12, 0.2),
        pitch_ratio=_u(rng, 0.45, 0.8),
        structure_density=_u(rng, 0.35, 0.75) if glider else _u(rng, 0.6, 1.4),
    )
