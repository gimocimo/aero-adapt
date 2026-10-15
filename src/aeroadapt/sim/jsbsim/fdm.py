"""A JSBSim flight dynamics model in SI units and aerojax conventions (M2 spec §5).

JSBSim is an optional dependency (``uv sync --extra jsbsim``); it is imported on first use.
Aircraft definitions are looked up under ``tools/jsbsim/aircraft`` unless another aircraft
directory is given (the Rascal is fetched into a cache, never committed: ADR-006).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

FT = 0.3048  # m per ft
SLUG_PER_FT3 = 515.378_818  # kg/m³ per slug/ft³
JSBSIM_ROOT = Path(__file__).resolve().parents[4] / "tools" / "jsbsim"
SURFACE_PROPS = ("fcs/da-rad", "fcs/de-rad", "fcs/dr-rad")


class JSBSimFDM:
    """One JSBSim aircraft. Surfaces are set in radians with B&M signs (aileron, elevator,
    rudder), as aerojax uses internally; actuator dynamics are not part of the FDM."""

    def __init__(
        self,
        model: str,
        dt: float = 0.005,
        root: Path = JSBSIM_ROOT,
        aircraft_dir: Path | None = None,
    ) -> None:
        import jsbsim

        jsbsim.FGJSBBase().debug_lvl = 0
        self.fdm: Any = jsbsim.FGFDMExec(str(root))
        self.fdm.set_debug_level(0)
        if aircraft_dir is not None:
            self.fdm.set_aircraft_path(str(aircraft_dir))
        if not self.fdm.load_model(model):
            raise RuntimeError(f"JSBSim could not load {model!r}")
        self.fdm.set_dt(dt)
        self.dt = dt

    def __getitem__(self, prop: str) -> float:
        return float(self.fdm[prop])

    def __setitem__(self, prop: str, value: float) -> None:
        self.fdm[prop] = value

    def initialise(
        self,
        alt_msl_m: float,
        airspeed_mps: float,
        alpha: float = 0.0,
        gamma: float = 0.0,
        phi: float = 0.0,
        psi: float = 0.0,
        lat_deg: float = 0.0,
        lon_deg: float = 0.0,
        surfaces: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Wings-level-style initial condition from airspeed, α and γ (θ = α + γ when φ = 0),
        with zero sideslip and zero body rates. Surfaces are applied before the
        derivatives are evaluated, so accelerations read after this call are consistent."""
        f = self.fdm
        f["ic/lat-geod-deg"] = lat_deg
        f["ic/long-gc-deg"] = lon_deg
        f["ic/h-sl-ft"] = alt_msl_m / FT
        f["ic/vt-fps"] = airspeed_mps / FT
        f["ic/alpha-rad"] = alpha
        f["ic/beta-rad"] = 0.0
        f["ic/gamma-rad"] = gamma
        f["ic/phi-rad"] = phi
        f["ic/psi-true-rad"] = psi
        for name in ("ic/p-rad_sec", "ic/q-rad_sec", "ic/r-rad_sec"):
            f[name] = 0.0
        self.set_surfaces(*surfaces)
        if not f.run_ic():
            raise RuntimeError("JSBSim run_ic failed")
        self.set_surfaces(*surfaces)  # run_ic may reinitialise declared properties

    def set_surfaces(self, da: float, de: float, dr: float) -> None:
        for prop, value in zip(SURFACE_PROPS, (da, de, dr), strict=True):
            self.fdm[prop] = value

    def step(self) -> None:
        if not self.fdm.run():
            raise RuntimeError("JSBSim run failed")

    # ------------------------------------------------------------------ SI read-outs
    @property
    def rho(self) -> float:
        return self["atmosphere/rho-slugs_ft3"] * SLUG_PER_FT3

    @property
    def gravity(self) -> float:
        return self["accelerations/gravity-ft_sec2"] * FT

    @property
    def body_accel(self) -> tuple[float, float, float]:
        """(u̇, ẇ, q̇) in m/s² and rad/s²: the longitudinal trim residuals."""
        return (
            self["accelerations/udot-ft_sec2"] * FT,
            self["accelerations/wdot-ft_sec2"] * FT,
            self["accelerations/qdot-rad_sec2"],
        )

    @property
    def alpha(self) -> float:
        return self["aero/alpha-rad"]

    @property
    def theta(self) -> float:
        return self["attitude/theta-rad"]

    @property
    def euler(self) -> tuple[float, float, float]:
        """(φ, θ, ψ) in rad."""
        return (self["attitude/phi-rad"], self["attitude/theta-rad"], self["attitude/psi-rad"])

    @property
    def gamma(self) -> float:
        return math.radians(self["flight-path/gamma-deg"])

    @property
    def airspeed(self) -> float:
        return self["velocities/vt-fps"] * FT
