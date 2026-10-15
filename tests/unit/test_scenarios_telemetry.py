"""Scenario schema, resolution, events and replay bundles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from aeroadapt.scenarios.campaign import Variation, load_campaign, make_variants
from aeroadapt.scenarios.resolve import resolve
from aeroadapt.scenarios.runner import record, run_scenario
from aeroadapt.scenarios.schema import SCENARIO_DIR, Scenario, load_scenario
from aeroadapt.telemetry.recorder import read_bundle
from aeroadapt.telemetry.schema import COLUMNS


@pytest.mark.parametrize("path", sorted(SCENARIO_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_scenario_validates_and_resolves(path: Path) -> None:
    sc = load_scenario(path)
    assert sc.id == path.stem
    r = resolve(sc)
    assert r.n_steps == round(sc.duration_s * 100)
    assert bool(r.stack.cont.enabled) == (sc.stack.contingency == "classical_v0")


def test_fault_after_the_flight_is_rejected() -> None:
    d = load_scenario("m1-motorcut-classical").model_dump(mode="json")
    d["faults"][0]["trigger"]["at_time_s"] = d["duration_s"] + 1
    with pytest.raises(ValidationError):
        Scenario.model_validate(d)


def test_variants_are_reproducible_and_varied() -> None:
    t = load_scenario("m1-motorcut-classical")
    vary = Variation(fault_onset_s=(20, 200), wind_speed_mps=(0, 8), wind_from_deg=(0, 360))
    a, b = make_variants(t, 5, 3, vary), make_variants(t, 5, 3, vary)
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]
    onsets = {s.faults[0].trigger.at_time_s for s in a}
    assert len(onsets) == 5 and all(20 <= x <= 200 for x in onsets)
    assert len({s.seed for s in a}) == 5
    assert load_campaign("m1-mini").n == 50


def test_bundle_round_trip_and_events(tmp_path: Path) -> None:
    d = load_scenario("m1-motorcut-classical").model_dump(mode="json")
    d["duration_s"] = 64.0
    result = run_scenario(Scenario.model_validate(d))
    out = record(result, tmp_path / "bundle")
    manifest, cols, events = read_bundle(out)

    assert manifest["n_frames"] == len(cols["time_us"]) == 6400
    assert set(manifest["units"]) == {"time_us", *(c.name for c in COLUMNS)}
    assert manifest["site"]["safe_zones"][0]["id"] == "A"
    np.testing.assert_array_equal(np.diff(cols["time_us"]), 10_000)  # 100 Hz, exact
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "end"
    onset = next(e for e in events if e["type"] == "fault_onset")
    detect = next(e for e in events if e["type"] == "failure_detected")
    assert onset["payload"]["fault"] == "motor_cut"
    assert 0 < (detect["time_us"] - onset["time_us"]) * 1e-6 <= 1.0
    # Labels switch at the onset tick.
    k = int(np.argmax(cols["labels.fault_bits"] > 0))
    assert cols["time_us"][k] == onset["time_us"]
    # The browser subset: 50 Hz float32.
    viz = json.loads((out / "manifest.json").read_text())["viz_columns"]
    assert viz[0] == "t" and "sensors.tas" in viz and "truth.tas" in viz
