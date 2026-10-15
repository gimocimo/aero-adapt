"""Fleet builds and fleet bundles (M2 spec §2 and §4.3).

A build cycles through the configuration classes with per-class seeds, builds candidates in
worker processes (AeroSandbox stage), filters each batch in aerojax (batched), and writes:

``runs/fleets/<fleet_id>/``
- ``manifest.json``: generator version, counts, yields and rejection reasons;
- ``designs.parquet``: every design tried, with its metadata, filter metrics and outcome;
- ``airframes.parquet``: accepted airframes as JSON (tables stored separately);
- ``tables.npz``: ``ids`` and aero v2 tables, shape (N, 5, ALPHA_GRID_N);
- ``gltf/<id>.glb``: a model of every accepted airframe.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from aeroadapt.airframes.fleet import TABLE_FIELDS
from aeroadapt.generator.build import GENERATOR_VERSION, Candidate, build
from aeroadapt.generator.design import CLASSES, ConfigClass, Design, sample_design
from aeroadapt.generator.mass import Rejected

REPO = Path(__file__).resolve().parents[3]
FLEET_DIR = REPO / "runs" / "fleets"


@dataclass(frozen=True)
class Outcome:
    design: Design
    candidate: Candidate | None
    reasons: tuple[str, ...]


def _build_one(d: Design) -> Outcome:
    try:
        return Outcome(d, build(d), ())
    except Rejected as e:
        return Outcome(d, None, (str(e),))
    except Exception as e:  # AeroSandbox or schema failure: counted, not fatal
        return Outcome(d, None, (f"error: {type(e).__name__}",))


def designs(classes: Sequence[ConfigClass], start: int = 0) -> Iterator[Design]:
    """Round-robin over classes; the k-th design of a class has seed ``start + k``."""
    k = start
    while True:
        for c in classes:
            yield sample_design(c, k)
        k += 1


def build_fleet(
    fleet_id: str,
    n_accepted: int,
    classes: Sequence[ConfigClass] = CLASSES,
    start: int = 0,
    workers: int | None = None,
    batch: int = 500,
    gltf: bool = True,
    out_root: Path = FLEET_DIR,
    progress: Callable[[str], None] = print,
) -> Path:
    from aeroadapt.generator.filters import check
    from aeroadapt.generator.gltf import airframe_glb

    out = out_root / fleet_id
    (out / "gltf").mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    source = designs(classes, start)
    rows: list[dict[str, Any]] = []
    accepted: list[Candidate] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        while len(accepted) < n_accepted:
            todo = [next(source) for _ in range(batch)]
            outcomes = list(pool.map(_build_one, todo, chunksize=4))
            built = [o.candidate for o in outcomes if o.candidate is not None]
            result = check([c.airframe for c in built], [c.meta for c in built]) if built else None
            j = 0
            for o in outcomes:
                if len(accepted) >= n_accepted:
                    break  # the quota is met: later designs in the batch are not counted
                row: dict[str, Any] = {
                    "id": f"gen-{o.design.cls}-{o.design.seed:05d}",
                    **o.design.to_dict(),
                }
                if o.candidate is None:
                    row.update(stage="build", accepted=False, reasons="; ".join(o.reasons))
                else:
                    assert result is not None
                    ok = bool(result.accepted[j])
                    row.update(o.candidate.meta)
                    row.update({k: float(v[j]) for k, v in result.metrics.items()})
                    row.update(stage="filter", accepted=ok, reasons="; ".join(result.reasons[j]))
                    if ok:
                        accepted.append(o.candidate)
                    j += 1
                rows.append(row)
            progress(f"{len(rows)} designs, {len(accepted)} accepted")
    if gltf:
        for c in accepted:
            cg = np.array([c.meta["x_cg_m"], 0.0, 0.0])
            (out / "gltf" / f"{c.airframe.id}.glb").write_bytes(airframe_glb(c.design, cg))
    _write_bundle(out, fleet_id, rows, accepted, classes)
    return out


def _write_bundle(
    out: Path,
    fleet_id: str,
    rows: list[dict[str, Any]],
    accepted: list[Candidate],
    classes: Sequence[ConfigClass],
) -> None:
    pq.write_table(pa.Table.from_pylist(rows), out / "designs.parquet", compression="zstd")
    ids = [c.airframe.id for c in accepted]
    tables = np.stack(
        [
            np.stack([np.asarray(getattr(c.airframe.aero.tables, k)) for k in TABLE_FIELDS])
            for c in accepted
        ]
    ).astype(np.float32)
    np.savez_compressed(out / "tables.npz", ids=np.array(ids), tables=tables)
    slim = [
        c.airframe.model_copy(
            update={
                "aero": c.airframe.aero.model_copy(update={"tables": None, "model": "bm_analytic"})
            }
        ).model_dump_json()
        for c in accepted
    ]
    pq.write_table(
        pa.table({"id": ids, "cls": [c.design.cls for c in accepted], "airframe": slim}),
        out / "airframes.parquet",
        compression="zstd",
    )
    reasons = reason_counts(rows)
    per_class = {
        c: {
            "tried": sum(r["cls"] == c for r in rows),
            "accepted": sum(a.design.cls == c for a in accepted),
        }
        for c in classes
    }
    manifest = {
        "fleet_id": fleet_id,
        "generator_version": GENERATOR_VERSION,
        "designs_tried": len(rows),
        "accepted": len(accepted),
        "per_class": per_class,
        "rejection_reasons": dict(reasons.most_common()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def reason_counts(rows: Sequence[dict[str, Any]]) -> Counter[str]:
    """Rejection reasons with numbers stripped ("trim fast: throttle 1.03" → "... throttle")."""
    return Counter(
        re.sub(r"\s*[-+]?\d+(\.\d+)?", "", r).strip()
        for row in rows
        if not row["accepted"]
        for r in str(row["reasons"]).split("; ")
        if r
    )


def rewrite_manifest(path: Path) -> None:
    """Recompute a bundle's rejection summary from ``designs.parquet``."""
    m = json.loads((path / "manifest.json").read_text())
    rows = pq.read_table(path / "designs.parquet").to_pylist()
    m["rejection_reasons"] = dict(reason_counts(rows).most_common())
    (path / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def subset_fleet(src: Path, ids: Sequence[str], dst: Path, stack_cfgs: Path | None = None) -> None:
    """A smaller bundle with the given accepted airframes (e.g. the committed sample fleet),
    optionally with their tuned stack configurations (a V20 run's ``stack_cfgs.jsonl``)."""
    keep = set(ids)
    (dst / "gltf").mkdir(parents=True, exist_ok=True)
    designs = pq.read_table(src / "designs.parquet").to_pylist()
    pq.write_table(
        pa.Table.from_pylist([r for r in designs if r["id"] in keep]),
        dst / "designs.parquet",
        compression="zstd",
    )
    afs = pq.read_table(src / "airframes.parquet").to_pylist()
    pq.write_table(
        pa.Table.from_pylist([r for r in afs if r["id"] in keep]),
        dst / "airframes.parquet",
        compression="zstd",
    )
    data = np.load(src / "tables.npz")
    mask = np.isin(data["ids"], list(keep))
    np.savez_compressed(dst / "tables.npz", ids=data["ids"][mask], tables=data["tables"][mask])
    for i in ids:
        shutil.copy(src / "gltf" / f"{i}.glb", dst / "gltf" / f"{i}.glb")
    if stack_cfgs is not None:
        lines = stack_cfgs.read_text().splitlines()
        rows = [ln for ln in lines if json.loads(ln)["airframe"] in keep]
        (dst / "stack_cfgs.jsonl").write_text("".join(f"{r}\n" for r in rows))
    m = json.loads((src / "manifest.json").read_text())
    m.update(
        fleet_id=dst.name,
        accepted=len(ids),
        subset_of=m["fleet_id"],
        note="A subset of an accepted fleet; counts and reasons describe the parent fleet.",
    )
    (dst / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
