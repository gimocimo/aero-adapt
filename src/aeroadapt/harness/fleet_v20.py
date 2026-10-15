"""A V20 run over a fleet (M2 spec §6): auto-configure, auto-tune and accept every airframe, in
fixed-size batches, and write the results with the tuned stack configurations.

``<out>/``
- ``summary.json``: pass rate, failure counts by check and per-class pass rates;
- ``results.parquet``: one row per airframe (passed, failed checks, every check's value and
  limit as JSON);
- ``stack_cfgs.jsonl``: the tuned ``StackCfg`` of every configurable airframe. Airframes that
  fail V20 are excluded from the benchmark fleet with the recorded reasons.
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

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / max(len(self.results), 1)


def airframe_class(airframe_id: str) -> str:
    """The configuration class of a generated airframe (``gen-<class>-<seed>``)."""
    parts = airframe_id.split("-")
    return parts[1] if len(parts) == 3 and parts[0] == "gen" else "other"


def run_fleet_v20(
    airframes: Sequence[Airframe],
    site: Site,
    batch: int = 200,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> FleetV20:
    out = FleetV20([], {})
    for start in range(0, len(airframes), batch):
        t0 = time.perf_counter()
        chunk = list(airframes[start : start + batch])
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
        done = start + len(chunk)
        passed = sum(r.passed for r in out.results)
        log(f"{done}/{len(airframes)}: {passed} passed ({time.perf_counter() - t0:.0f} s)")
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
