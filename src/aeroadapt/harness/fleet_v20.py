"""A V20 run over a fleet (M2 spec §6): auto-configure, auto-tune and accept every airframe, in
fixed-size batches, and write the results with the tuned stack configurations.

``<out>/``
- ``summary.json``: pass rate, failure counts by check and per-class pass rates;
- ``results.parquet``: one row per airframe (passed, failed checks, every check's value and
  limit as JSON);
- ``stack_cfgs.jsonl``: the tuned ``StackCfg`` of every configurable airframe. Airframes that
  fail V20 are excluded from the benchmark fleet with the recorded reasons.

With ``parts``, each finished batch is also kept as ``<parts>/batch-<start>.json``, and a rerun
over the same airframes, seed and code (``fingerprint``) takes those batches from disk instead
of flying them again: an interrupted run of the whole fleet resumes where it stopped.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from aeroadapt.airframes import Airframe
from aeroadapt.harness.autoconfig import autoconfigure
from aeroadapt.harness.autotune import autotune
from aeroadapt.harness.fleet_acceptance import AirframeResult, run_v20
from aeroadapt.scenarios.site import Site
from aeroadapt.stack.config import StackCfg

PASS_RATE_TARGET = 0.90


@dataclass
class FleetV20:
    results: list[AirframeResult]
    cfgs: dict[str, StackCfg]  # tuned configurations, by airframe id
    resumed: int = 0  # batches taken from kept parts, not flown

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / max(len(self.results), 1)


def airframe_class(airframe_id: str) -> str:
    """The configuration class of a generated airframe (``gen-<class>-<seed>``)."""
    parts = airframe_id.split("-")
    return parts[1] if len(parts) == 3 and parts[0] == "gen" else "other"


PART_VERSION = 1


def _part_path(parts: Path, start: int) -> Path:
    return parts / f"batch-{start:05d}.json"


def save_part(path: Path, ids: list[str], seed: int, fingerprint: str, run: FleetV20) -> None:
    """One finished batch, written atomically (a killed run leaves no half-written part)."""
    data = {
        "version": PART_VERSION,
        "ids": ids,
        "seed": seed,
        "fingerprint": fingerprint,
        "results": [
            {"airframe": r.airframe, "v_cruise": r.v_cruise, "checks": r.checks}
            for r in run.results
        ],
        "cfgs": {k: c.model_dump(mode="json") for k, c in run.cfgs.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def load_part(path: Path, ids: list[str], seed: int, fingerprint: str) -> FleetV20 | None:
    """A kept batch, if it is for these airframes, this seed and this code; otherwise None."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if (
        data.get("version") != PART_VERSION
        or data.get("ids") != ids
        or data.get("seed") != seed
        or data.get("fingerprint") != fingerprint
    ):
        return None
    results = [
        AirframeResult(
            r["airframe"],
            float(r["v_cruise"]),
            {k: (float(v), float(lim), bool(ok)) for k, (v, lim, ok) in r["checks"].items()},
        )
        for r in data["results"]
    ]
    if sorted(r.airframe for r in results) != sorted(ids):
        return None
    cfgs = {k: StackCfg.model_validate(c) for k, c in data["cfgs"].items()}
    return FleetV20(results, cfgs)


def _fly(chunk: list[Airframe], site: Site, seed: int) -> FleetV20:
    """Configure, tune and accept one batch."""
    out = FleetV20([], {})
    acs = autoconfigure(chunk, site)
    ok = [(a, ac.cfg) for a, ac in zip(chunk, acs, strict=True) if ac.cfg is not None]
    for a, ac in zip(chunk, acs, strict=True):
        if ac.cfg is None:
            r = AirframeResult(a.id, float("nan"))
            r.add(f"auto-configuration: {ac.reason}", 0.0, 1.0, upper=False)
            out.results.append(r)
    if ok:
        fleet, cfgs = [a for a, _ in ok], [c for _, c in ok]
        tuned = autotune(fleet, cfgs, site)
        out.results.extend(run_v20(fleet, tuned, site, seed))
        out.cfgs.update({a.id: c for a, c in zip(fleet, tuned, strict=True)})
    return out


def _batch(job: tuple[list[Airframe], Site, int, Path | None, str]) -> FleetV20:
    """A worker's batch: flown, then kept (in the worker, so it survives the parent)."""
    chunk, site, seed, part, fingerprint = job
    run = _fly(chunk, site, seed)
    if part is not None:
        save_part(part, [a.id for a in chunk], seed, fingerprint, run)
    return run


def run_fleet_v20(
    airframes: Sequence[Airframe],
    site: Site,
    batch: int = 200,
    seed: int = 0,
    log: Callable[[str], None] = print,
    workers: int = 1,
    parts: Path | None = None,
    fingerprint: str = "",
) -> FleetV20:
    """``workers`` > 1 flies batches in parallel processes: JAX on the CPU keeps about two cores
    busy per process. Every airframe's noise comes from its id, so the results are the same.
    ``parts`` keeps each finished batch and resumes from those already kept (module docstring);
    ``fingerprint`` names the code and inputs, and a part with another one is flown again."""
    starts = list(range(0, len(airframes), batch))
    chunks = {s: list(airframes[s : s + batch]) for s in starts}
    out = FleetV20([], {})

    def take(part: FleetV20) -> None:
        out.results.extend(part.results)
        out.cfgs.update(part.cfgs)

    todo = []
    for s in starts:
        kept = None
        if parts is not None:
            kept = load_part(_part_path(parts, s), [a.id for a in chunks[s]], seed, fingerprint)
        if kept is None:
            todo.append(s)
        else:
            take(kept)
            out.resumed += 1
    if out.resumed:
        log(f"{out.resumed} of {len(starts)} batches taken from {parts}")

    def progress(extra: str = "") -> None:
        passed = sum(r.passed for r in out.results)
        log(f"{len(out.results)}/{len(airframes)}: {passed} passed{extra}")

    def part_of(s: int) -> Path | None:
        return None if parts is None else _part_path(parts, s)

    if workers > 1 and todo:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        jobs = [(chunks[s], site, seed, part_of(s), fingerprint) for s in todo]
        ctx = mp.get_context("spawn")  # JAX is not fork-safe
        with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
            for part in pool.map(_batch, jobs):
                take(part)
                progress()
    else:
        for s in todo:
            t0 = time.perf_counter()
            take(_batch((chunks[s], site, seed, part_of(s), fingerprint)))
            progress(f" ({time.perf_counter() - t0:.0f} s)")
    order = {a.id: i for i, a in enumerate(airframes)}
    out.results.sort(key=lambda r: order[r.airframe])
    return out


def summary(run: FleetV20) -> dict[str, Any]:
    by_class: dict[str, list[bool]] = {}
    for r in run.results:
        by_class.setdefault(airframe_class(r.airframe), []).append(r.passed)
    fails = Counter(k for r in run.results for k in r.failures())
    return {
        "airframes": len(run.results),
        "passed": sum(r.passed for r in run.results),
        "pass_rate": round(run.pass_rate, 4),
        "target": PASS_RATE_TARGET,
        "failures_by_check": dict(fails.most_common()),
        "pass_rate_by_class": {
            c: {"airframes": len(v), "pass_rate": round(sum(v) / len(v), 4)}
            for c, v in sorted(by_class.items())
        },
    }


def write_fleet_v20(out_dir: Path, run: FleetV20, meta: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = {**meta, **summary(run)}
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2) + "\n")
    rows = [
        {
            "airframe": r.airframe,
            "class": airframe_class(r.airframe),
            "v_cruise_mps": r.v_cruise,
            "passed": r.passed,
            "failures": r.failures(),
            "checks": json.dumps(r.checks),
        }
        for r in run.results
    ]
    pq.write_table(pa.Table.from_pylist(rows), out_dir / "results.parquet")
    with (out_dir / "stack_cfgs.jsonl").open("w") as f:
        for af_id, cfg in run.cfgs.items():
            f.write(json.dumps({"airframe": af_id, "stack": cfg.model_dump(mode="json")}) + "\n")
    return out_dir
