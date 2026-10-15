"""Scenario schema (spec §10.1): one YAML file fully describes a flight.

Everything a run depends on is here or referenced by id (site, airframe, stack parameter
set), so a scenario plus a git commit reproduces a recording bit for bit (V13).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aeroadapt.faultlab.schema import FaultSpec

SCENARIO_DIR = Path(__file__).resolve().parents[3] / "configs" / "scenarios"
MAX_WAYPOINTS = 16

# MIL-HDBK-1797 low-altitude Dryden: wind speed at 20 ft (kt) per intensity.
TURBULENCE_W20_KT = {"none": 0.0, "light": 15.0, "moderate": 30.0, "severe": 45.0}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InitialCfg(_Strict):
    """Start airborne in straight-and-level trim."""

    pos_ne_m: tuple[float, float] = (0.0, 0.0)
    alt_agl_m: float = Field(default=120.0, gt=0)
    heading_deg: float = 0.0
    airspeed_mps: float = Field(default=25.0, gt=0)
    soc: float = Field(default=1.0, gt=0, le=1)


class MissionCfg(_Strict):
    type: Literal["waypoints"] = "waypoints"
    alt_agl_m: float = Field(gt=0, description="default altitude for 2-D waypoints")
    waypoints_ne_m: list[tuple[float, float] | tuple[float, float, float]] = Field(
        min_length=1, max_length=MAX_WAYPOINTS
    )
    on_complete: Literal["loop", "rtl"] = "loop"
    speed_mps: float = Field(default=25.0, gt=0)
    home_ne_m: tuple[float, float] | None = Field(
        default=None, description="RTL loiter centre; defaults to the initial position"
    )

    def waypoints_3d(self) -> list[tuple[float, float, float]]:
        return [(w[0], w[1], w[2] if len(w) == 3 else self.alt_agl_m) for w in self.waypoints_ne_m]


class WindCfg(_Strict):
    from_deg: float = 0.0  # aviation convention: the direction the wind blows *from*
    speed_mps: float = Field(default=0.0, ge=0)
    shear_alpha: float = Field(default=0.0, ge=0, le=0.5, description="power-law exponent")


class TurbulenceCfg(_Strict):
    model: Literal["dryden_low_alt"] = "dryden_low_alt"
    intensity: Literal["none", "light", "moderate", "severe"] = "none"

    @property
    def w20_mps(self) -> float:
        return TURBULENCE_W20_KT[self.intensity] * 0.514444


class EnvironmentCfg(_Strict):
    wind: WindCfg = WindCfg()
    turbulence: TurbulenceCfg = TurbulenceCfg()
    temp_offset_k: float | None = None  # None: the site's value
    qnh_hpa: float | None = None


class StackChoice(_Strict):
    autopilot: Literal["stock"] = "stock"
    contingency: Literal["classical_v0", "none"] = "classical_v0"
    # Motor-failure detector: ESC telemetry (primary) or the total-energy-rate residual (for
    # airframes without ESC telemetry). None: as in the stack parameter set.
    motor_detector: Literal["esc", "energy"] | None = None
    estimator: Literal["navigator"] = "navigator"
    params: str | None = Field(default=None, description="configs/stack/<id>; default: airframe")


class Scenario(_Strict):
    schema_version: Literal[1]
    id: str
    description: str = ""
    site: str
    airframe: str
    seed: int = Field(default=0, ge=0)
    duration_s: float = Field(gt=0, le=3600)
    initial: InitialCfg = InitialCfg()
    mission: MissionCfg
    environment: EnvironmentCfg = EnvironmentCfg()
    faults: list[FaultSpec] = Field(default_factory=list)
    stack: StackChoice = StackChoice()

    @model_validator(mode="after")
    def _faults_within_flight(self) -> Scenario:
        for f in self.faults:
            if f.trigger.at_time_s >= self.duration_s:
                raise ValueError(f"{f.type} triggers after the end of the flight")
        return self


def load_scenario(name_or_path: str | Path) -> Scenario:
    """Load and validate a scenario by id (``configs/scenarios/<id>.yaml``) or by path."""
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = SCENARIO_DIR / f"{name_or_path}.yaml"
    with path.open() as f:
        return Scenario.model_validate(yaml.safe_load(f))
