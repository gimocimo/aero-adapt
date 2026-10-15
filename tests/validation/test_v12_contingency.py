"""V12: classical contingency (spec §12), full tier.

The ESC detector (primary) and the mini-campaign are gated. The energy-residual backup is
checked for detection only: in light turbulence it cannot meet both the 4 s latency and the
zero-false-alarm criteria (documented in the validation report and the spec log).
"""

from __future__ import annotations

import numpy as np
import pytest

from aeroadapt.scenarios.campaign import failure_cause, summarise
from aeroadapt.scenarios.v12 import IN_ZONE_RATE_MIN, false_alarms, latency, mini_campaign

pytestmark = pytest.mark.full


def test_esc_detection_latency() -> None:
    result = latency("esc", n=100)
    assert np.all(np.isfinite(result.latencies_s))
    assert result.max_s <= result.limit_s


def test_esc_no_false_alarms_in_10_hours() -> None:
    assert false_alarms("esc", hours=10.0).false_alarms == 0


def test_energy_backup_detects_every_cut() -> None:
    result = latency("energy", n=100)
    assert np.all(np.isfinite(result.latencies_s))
    assert float(np.median(result.latencies_s)) <= result.limit_s


def test_mini_campaign() -> None:
    results, _ = mini_campaign()
    s = summarise(results)
    assert s["n"] == 50 and s["detected"] == 50
    assert s["reachable_at_detection"] >= 40  # the loop keeps most of the cuts within reach
    assert s["in_zone_rate"] >= IN_ZONE_RATE_MIN
    assert s["crashes"] == 0
    failures = [o for o in results if o.reachable_at_detection and not o.in_zone]
    assert all(failure_cause(o) is not None for o in failures)
