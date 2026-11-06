"""A V20 run keeps each finished batch and resumes from it (``harness/fleet_v20.py``).

The flights are replaced by a stub that records which batches it was asked to fly, so these
tests check the bookkeeping only; the full tier's V20 tests fly the real thing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aeroadapt.harness import fleet_v20
from aeroadapt.harness.fleet_acceptance import AirframeResult
from aeroadapt.harness.fleet_v20 import FleetV20, load_part, run_fleet_v20, save_part
from aeroadapt.stack.config import StackCfg

SAMPLE_CFGS = (
    Path(__file__).resolve().parents[2] / "configs" / "fleets" / "sample" / "stack_cfgs.jsonl"
)


def _sample_cfg() -> StackCfg:
    line = SAMPLE_CFGS.read_text().splitlines()[0]
    return StackCfg.model_validate(json.loads(line)["stack"])


@pytest.fixture
def flown(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Replace the flights: each airframe passes one check whose value is its index."""
    calls: list[list[str]] = []
    cfg = _sample_cfg()

    def fake_fly(chunk: list[Any], site: Any, seed: int) -> FleetV20:
        calls.append([a.id for a in chunk])
        out = FleetV20([], {})
        for a in chunk:
            r = AirframeResult(a.id, 20.0)
            r.add("roll overshoot", float(a.index), 100.0)
            out.results.append(r)
            out.cfgs[a.id] = cfg
        return out

    monkeypatch.setattr(fleet_v20, "_fly", fake_fly)
    return calls


def _airframes(n: int) -> Any:
    return [SimpleNamespace(id=f"gen-glider-{i:05d}", index=i) for i in range(n)]


SITE: Any = SimpleNamespace()


def _run(parts: Path, n: int = 10, fingerprint: str = "a") -> FleetV20:
    return run_fleet_v20(
        _airframes(n), SITE, batch=4, parts=parts, fingerprint=fingerprint, log=lambda _: None
    )


def test_a_rerun_takes_every_kept_batch(tmp_path: Path, flown: list[list[str]]) -> None:
    first = _run(tmp_path)
    assert len(flown) == 3 and first.resumed == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "batch-00000.json",
        "batch-00004.json",
        "batch-00008.json",
    ]
    second = _run(tmp_path)
    assert len(flown) == 3 and second.resumed == 3  # nothing flown again
    assert [(r.airframe, r.checks) for r in second.results] == [
        (r.airframe, r.checks) for r in first.results
    ]
    assert {k: c.model_dump() for k, c in second.cfgs.items()} == {
        k: c.model_dump() for k, c in first.cfgs.items()
    }


def test_an_interrupted_run_flies_only_the_missing_batches(
    tmp_path: Path, flown: list[list[str]]
) -> None:
    _run(tmp_path)
    (tmp_path / "batch-00004.json").unlink()
    (tmp_path / "batch-00008.json").write_text('{"version": 1, "ids": [')  # half-written
    flown.clear()
    run = _run(tmp_path)
    assert flown == [
        [f"gen-glider-{i:05d}" for i in (4, 5, 6, 7)],
        ["gen-glider-00008", "gen-glider-00009"],
    ]
    assert run.resumed == 1
    assert [r.airframe for r in run.results] == [f"gen-glider-{i:05d}" for i in range(10)]


def test_other_code_or_other_airframes_fly_again(tmp_path: Path, flown: list[list[str]]) -> None:
    _run(tmp_path)
    flown.clear()
    assert _run(tmp_path, fingerprint="b").resumed == 0  # another commit or fleet
    assert len(flown) == 3
    flown.clear()
    run = _run(tmp_path, n=9, fingerprint="b")  # the last batch now holds one airframe
    assert run.resumed == 2 and flown == [["gen-glider-00008"]]


def test_a_part_round_trips_results_and_configurations(tmp_path: Path) -> None:
    cfg = _sample_cfg()
    ok = AirframeResult("gen-v_tail-00001", 18.5)
    ok.add("pitch overshoot", 0.1, 0.25)
    ok.add("roll steady-state error", math.inf, 1.0)
    failed = AirframeResult("gen-v_tail-00002", float("nan"))
    failed.add("auto-configuration: no trim", 0.0, 1.0, upper=False)
    run = FleetV20([ok, failed], {ok.airframe: cfg})
    ids = [ok.airframe, failed.airframe]
    path = tmp_path / "batch-00000.json"
    save_part(path, ids, 0, "fp", run)
    back = load_part(path, ids, 0, "fp")
    assert back is not None
    assert back.results[0].checks == ok.checks and back.results[0].v_cruise == 18.5
    assert math.isnan(back.results[1].v_cruise) and not back.results[1].passed
    assert back.cfgs[ok.airframe].model_dump() == cfg.model_dump()
    assert load_part(path, ids, 1, "fp") is None  # another seed
    assert load_part(path, ids[::-1], 0, "fp") is None  # another batch
    assert not list(tmp_path.glob("*.tmp"))
