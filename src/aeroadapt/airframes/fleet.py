"""Reading fleet bundles (M2 spec §4.3) without the generator's dependencies: the live server,
the bench and V20 load fleets; only building one needs the ``generator`` extra."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from aeroadapt.airframes.params import AeroTables
from aeroadapt.airframes.schema import Airframe

TABLE_FIELDS = AeroTables._fields


def load_fleet(path: Path) -> list[Airframe]:
    """Accepted airframes of a fleet bundle, with their tables restored."""
    data = np.load(path / "tables.npz")
    by_id = dict(zip(data["ids"].tolist(), data["tables"], strict=True))
    out = []
    for af_json in pq.read_table(path / "airframes.parquet").column("airframe").to_pylist():
        raw = json.loads(af_json)
        t = by_id[raw["id"]].astype(np.float64)
        raw["aero"]["model"] = "tabulated"
        raw["aero"]["tables"] = {k: t[i].tolist() for i, k in enumerate(TABLE_FIELDS)}
        out.append(Airframe.model_validate(raw))
    return out
