"""Airframe definitions: YAML → validated schema → flat JAX parameters."""

from __future__ import annotations

from pathlib import Path

import yaml

from aeroadapt.airframes.params import (
    ALPHA_GRID_N,
    ActuatorParams,
    AeroTables,
    AirframeParams,
    BatteryParams,
    SensorParams,
    VehicleParams,
    alpha_grid,
    bm_tables,
    to_params,
    to_vehicle_params,
)
from aeroadapt.airframes.schema import Airframe

AIRFRAME_DIR = Path(__file__).resolve().parents[3] / "configs" / "airframes"

__all__ = [
    "AIRFRAME_DIR",
    "ALPHA_GRID_N",
    "ActuatorParams",
    "AeroTables",
    "Airframe",
    "AirframeParams",
    "BatteryParams",
    "SensorParams",
    "VehicleParams",
    "alpha_grid",
    "bm_tables",
    "load_airframe",
    "to_params",
    "to_vehicle_params",
]


def load_airframe(name_or_path: str | Path) -> Airframe:
    """Load and validate an airframe by id (``configs/airframes/<id>.yaml``) or by path."""
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = AIRFRAME_DIR / f"{name_or_path}.yaml"
    with path.open() as f:
        return Airframe.model_validate(yaml.safe_load(f))
