"""V20 (M2 spec §10, full tier): auto-configuration across the fleet.

The 1,000-airframe run needs the git-ignored fleet and takes about 40 minutes on a laptop
(``aeroadapt fleet-accept m2-fleet --n 1000``); the validation report reads its results.
Here, the committed sample fleet (V20-accepted airframes with the most margin) is
configured, tuned and accepted again from scratch, so a change that breaks the pipeline
fails the nightly tier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aeroadapt.airframes.fleet import load_fleet
from aeroadapt.harness.fleet_v20 import run_fleet_v20
from aeroadapt.scenarios import load_site

SAMPLE = Path(__file__).resolve().parents[2] / "configs" / "fleets" / "sample"


@pytest.mark.full
def test_sample_fleet_passes_v20_from_scratch() -> None:
    airframes = load_fleet(SAMPLE)
    run = run_fleet_v20(airframes, load_site("cambridge_west"), batch=len(airframes))
    failed = {r.airframe: r.failures() for r in run.results if not r.passed}
    assert not failed
    assert set(run.cfgs) == {a.id for a in airframes}


def test_airframe_keys_follow_the_id_not_the_position() -> None:
    from aeroadapt.harness.fleet_acceptance import airframe_keys

    a = np.asarray(airframe_keys(["x", "y", "z"], 0, 1))
    b = np.asarray(airframe_keys(["z", "x"], 0, 1))
    np.testing.assert_array_equal(a[0], b[1])
    np.testing.assert_array_equal(a[2], b[0])
    assert not np.array_equal(a[0], np.asarray(airframe_keys(["x"], 0, 0))[0])  # streams differ


@pytest.mark.full
def test_v20_does_not_depend_on_batching() -> None:
    """The same airframes, in another order and batch size: the same tuning and results."""
    airframes = load_fleet(SAMPLE)[:4]
    site = load_site("cambridge_west")
    a = run_fleet_v20(airframes, site, batch=4)
    b = run_fleet_v20(airframes[::-1], site, batch=2)
    assert {k: c.model_dump() for k, c in a.cfgs.items()} == {
        k: c.model_dump() for k, c in b.cfgs.items()
    }
    checks = {r.airframe: r.checks for r in a.results}
    for r in b.results:
        for name, (value, _, ok) in r.checks.items():
            assert ok == checks[r.airframe][name][2], (r.airframe, name)
            assert value == pytest.approx(checks[r.airframe][name][0], rel=1e-3, abs=1e-6)
