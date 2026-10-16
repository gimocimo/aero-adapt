"""Reading fleet bundles (M2 spec §4.3) without the generator's dependencies: the live server,
the bench and V20 load fleets; only building one needs the ``generator`` extra."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from aeroadapt.airframes.params import AeroTables
from aeroadapt.airframes.schema import Airframe

TABLE_FIELDS = AeroTables._fields
REPO = Path(__file__).resolve().parents[3]


def fleet_path(name_or_path: str | Path) -> Path:
    """A fleet bundle: a path (absolute, or relative to the repository) or a fleet id under
    ``runs/fleets``."""
    p = Path(name_or_path)
    for candidate in (p, REPO / p, REPO / "runs" / "fleets" / p):
        if (candidate / "airframes.parquet").exists():
            return candidate
    raise FileNotFoundError(f"no fleet bundle at {name_or_path}")


def _airframe(raw_json: str, tables: np.ndarray) -> Airframe:
    raw = json.loads(raw_json)
    t = tables.astype(np.float64)
    raw["aero"]["model"] = "tabulated"
    raw["aero"]["tables"] = {k: t[i].tolist() for i, k in enumerate(TABLE_FIELDS)}
    return Airframe.model_validate(raw)


def load_fleet(path: Path, ids: Sequence[str] | None = None) -> list[Airframe]:
    """Accepted airframes of a fleet bundle (all, or those in ``ids``), with their tables."""
    data = np.load(path / "tables.npz")
    by_id = dict(zip(data["ids"].tolist(), data["tables"], strict=True))
    filters = None if ids is None else [("id", "in", list(ids))]
    rows = pq.read_table(
        path / "airframes.parquet", columns=["id", "airframe"], filters=filters
    ).to_pylist()
    return [_airframe(r["airframe"], by_id[r["id"]]) for r in rows]


def load_fleet_airframe(path: Path, airframe_id: str) -> Airframe:
    """One airframe of a fleet bundle (a scenario that names a fleet airframe)."""
    rows = pq.read_table(
        path / "airframes.parquet", columns=["id", "airframe"], filters=[("id", "=", airframe_id)]
    ).to_pylist()
    if not rows:
        raise KeyError(f"{airframe_id} is not in the fleet at {path}")
    data = np.load(path / "tables.npz")
    k = int(np.flatnonzero(data["ids"] == airframe_id)[0])
    return _airframe(rows[0]["airframe"], data["tables"][k])
