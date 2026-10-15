"""V12 (spec §12): classical contingency evaluation, shared by the full-tier test and the report.

* **Latency:** 100 motor cuts per detector (ESC ≤ 1.0 s, energy residual ≤ 4 s).
* **False alarms:** 10 h of fault-free flight in light turbulence per detector (zero allowed).
* **Mini-campaign:** 50 cuts anywhere on the mission loop in random wind; ≥ 80 % of the cases
  judged reachable at detection land inside the selected zone (controlled or hard), and
  every failure is listed with its cause.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from aeroadapt.harness.closed_loop import LoopLog
from aeroadapt.scenarios.campaign import (
    Outcome,
    Variation,
    load_campaign,
    make_variants,
    run_campaign,
)
from aeroadapt.scenarios.schema import Scenario, load_scenario

Detector = Literal["esc", "energy"]
LATENCY_LIMIT_S = {"esc": 1.0, "energy": 4.0}
IN_ZONE_RATE_MIN = 0.8


def _with(template: str, **changes: object) -> Scenario:
    d = load_scenario(template).model_dump(mode="json")
    for path, value in changes.items():
        node = d
        *parents, leaf = path.split("__")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return Scenario.model_validate(d)


@dataclass(frozen=True)
class LatencyResult:
    detector: Detector
    latencies_s: np.ndarray  # inf where missed
    limit_s: float

    @property
    def max_s(self) -> float:
        return float(self.latencies_s.max())

    @property
    def within_limit(self) -> float:
        return float(np.mean(self.latencies_s <= self.limit_s))


def latency(detector: Detector, n: int = 100, seed: int = 31) -> LatencyResult:
    """Cuts at a random time once the navigator has converged, in random wind."""
    t = _with("m1-motorcut-classical", duration_s=75.0, stack__motor_detector=detector)
    vary = Variation(fault_onset_s=(25.0, 60.0), wind_speed_mps=(0, 8), wind_from_deg=(0, 360))
    results, _ = run_campaign(make_variants(t, n, seed, vary))
    lat = np.array(
        [o.detect_latency_s if o.detect_latency_s is not None else np.inf for o in results]
    )
    return LatencyResult(detector, lat, LATENCY_LIMIT_S[detector])


@dataclass(frozen=True)
class FalseAlarmResult:
    detector: Detector
    hours: float
    false_alarms: int


def false_alarms(detector: Detector, hours: float = 10.0, seed: int = 41) -> FalseAlarmResult:
    """Fault-free 10-minute flights of the mission loop in light turbulence, random wind."""
    t = _with("m1-nominal", duration_s=600.0, stack__motor_detector=detector)
    n = round(hours * 6)
    vary = Variation(wind_speed_mps=(0, 8), wind_from_deg=(0, 360))
    results, _ = run_campaign(make_variants(t, n, seed, vary))
    return FalseAlarmResult(detector, n / 6.0, sum(o.detected for o in results))


def mini_campaign(name: str = "m1-mini") -> tuple[list[Outcome], LoopLog]:
    """(outcomes, stacked 1 Hz trajectories) of the V12 mini-campaign."""
    return run_campaign(load_campaign(name).scenarios())
