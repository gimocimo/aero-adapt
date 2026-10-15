"""Replay bundles (spec §10.3, ADR-005): a directory with

* ``manifest.json``: resolved scenario, site, provenance (config hashes, git commit), rates,
  units, enums;
* ``frames.parquet``: one row per control tick (float64; zstd);
* ``events.jsonl``: fault onsets, detections, zone selections, phases, modes, touchdown;
* ``viz.arrow``: the browser subset at 50 Hz in float32, as uncompressed Arrow IPC (the
  web server compresses on the wire).

Bundles are deterministic: no wall-clock time is recorded, so the same scenario on the same
machine and commit gives byte-identical files (V13).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

from aeroadapt.telemetry.schema import COLUMNS, VIZ_RATE_HZ

FILES = ("manifest.json", "frames.parquet", "events.jsonl", "viz.arrow")


def _table(cols: dict[str, np.ndarray]) -> pa.Table:
    return pa.table({k: pa.array(v) for k, v in cols.items()})


def viz_columns(cols: dict[str, np.ndarray], control_hz: float) -> dict[str, np.ndarray]:
    """Browser subset: every ``control_hz / 50``-th row, float32 (integers stay int32)."""
    step = max(round(control_hz / VIZ_RATE_HZ), 1)
    idx = np.arange(0, len(cols["time_us"]), step)
    if idx[-1] != len(cols["time_us"]) - 1:  # always keep the final (e.g. touchdown) row
        idx = np.append(idx, len(cols["time_us"]) - 1)
    out: dict[str, np.ndarray] = {"t": (cols["time_us"][idx] * 1e-6).astype(np.float32)}
    for c in COLUMNS:
        if c.viz:
            x = cols[c.name][idx]
            out[c.name] = x.astype(np.int32) if c.integer else x.astype(np.float32)
    return out


def write_bundle(
    out_dir: Path,
    manifest: dict[str, Any],
    cols: dict[str, np.ndarray],
    events: list[dict[str, Any]],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    control_hz = float(manifest["rates"]["control_hz"])
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    pq.write_table(_table(cols), out_dir / "frames.parquet", compression="zstd")
    with (out_dir / "events.jsonl").open("w") as f:
        for e in events:
            f.write(json.dumps(e, sort_keys=True) + "\n")
    feather.write_feather(
        _table(viz_columns(cols, control_hz)), out_dir / "viz.arrow", compression="uncompressed"
    )
    return out_dir


def read_bundle(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    """(manifest, frame columns, events)."""
    manifest = json.loads((path / "manifest.json").read_text())
    table = pq.read_table(path / "frames.parquet")
    cols = {name: table[name].to_numpy() for name in table.column_names}
    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    return manifest, cols, events
