"""V22 (M2 spec §10): bench reproducibility and sanity. The same spec and seed give an identical
outcome table; the classical contingency (B2) survives motor cuts at least as often as the
stock stack alone (B1). Runs on the committed sample fleet."""

from __future__ import annotations

import math
from dataclasses import asdict

import pytest

from aeroadapt.bench.run import run_bench, summary
from aeroadapt.bench.sampler import BenchSpec, FaultMix, load_bench_spec
from aeroadapt.bench.splits import split_of
from aeroadapt.bench.stats import bootstrap_mean, wilson


def test_splits_are_80_10_10_by_id() -> None:
    ids = [f"gen-conventional-{i:05d}" for i in range(20_000)]
    counts = {s: sum(split_of(i) == s for i in ids) for s in ("train", "val", "test")}
    assert counts["train"] == pytest.approx(16_000, rel=0.03)
    assert counts["val"] == pytest.approx(2_000, rel=0.08)
    assert counts["test"] == pytest.approx(2_000, rel=0.08)
    assert split_of("gen-v_tail-00031") == split_of("gen-v_tail-00031")


def test_intervals() -> None:
    p, lo, hi = wilson(8, 10)
    assert (
        p == 0.8 and lo == pytest.approx(0.490, abs=1e-3) and hi == pytest.approx(0.943, abs=1e-3)
    )
    assert all(math.isnan(x) for x in wilson(0, 0))
    mean, lo, hi = bootstrap_mean([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5 and lo < mean < hi


@pytest.mark.full
def test_v22_same_seed_same_table() -> None:
    spec = load_bench_spec("sample").model_copy(update={"n": 4, "duration_s": 60.0})
    a, b = run_bench(spec, log=lambda _: None), run_bench(spec, log=lambda _: None)
    assert [asdict(o) for o in a.outcomes] == [asdict(o) for o in b.outcomes]


@pytest.mark.full
def test_v22_contingency_helps_on_motor_cuts() -> None:
    spec: BenchSpec = load_bench_spec("sample").model_copy(
        update={
            "n": 6,
            "duration_s": 240.0,
            "faults": FaultMix(
                none=0.0, single=1.0, double=0.0, onset_s=(20.0, 40.0), kinds=["motor_cut"]
            ),
        }
    )
    s = summary(run_bench(spec, log=lambda _: None))["baselines"]
    b1, b2 = s["B1"]["motor_cut"], s["B2"]["motor_cut"]
    assert b1["survival"]["n"] == b2["survival"]["n"] == 6
    assert b2["detection"]["k"] == 6
    assert b2["survival"]["rate"] >= b1["survival"]["rate"]
