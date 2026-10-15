"""Golden flights (spec §13): four canonical scenarios whose metrics are stored and gated.

A change that moves a metric beyond its tolerance must regenerate the goldens explicitly
(``uv run python tools/update_goldens.py``) and justify it in the commit message.
Tolerances absorb cross-machine floating-point differences (V13: bitwise on one machine,
within tolerance across machines).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from aeroadapt.scenarios.resolve import ResolvedScenario, resolve
from aeroadapt.scenarios.runner import run_scenario
from aeroadapt.scenarios.schema import load_scenario

GOLDEN_SCENARIOS = (
    "m1-nominal",
    "m1-motorcut-classical",
    "m1-motorcut-stock",
    "m1-pitot-drain-blocked",
    "m1-servo-jam",
)

# metric → absolute tolerance (None: exact match)
TOLERANCES: dict[str, float | None] = {
    "flight_time_s": 3.0,
    "waypoints_reached": None,
    "alt_error_rms_m": 0.5,
    "tas_mean_mps": 0.3,
    "xtrack_rms_m": 1.0,
    "detect_latency_s": 0.05,
    "zone": None,
    "touchdown_class": None,
    "touchdown_n_m": 30.0,
    "touchdown_e_m": 30.0,
    "airspeed_unhealthy_at_s": 1.5,
    "max_airspeed_error_mps": 1.5,
    "terminated": None,
}


def metrics(
    scenario_id: str,
    transform: Callable[[ResolvedScenario], ResolvedScenario] | None = None,
) -> dict[str, Any]:
    """Golden metrics of a scenario; ``transform`` edits the resolved scenario first (V16 flies
    the goldens with tabulated aerodynamics this way)."""
    resolved = resolve(load_scenario(scenario_id))
    r = run_scenario(transform(resolved) if transform else resolved)
    c, ev = r.cols, r.events
    t = c["time_us"] * 1e-6
    m: dict[str, Any] = {
        "flight_time_s": round(float(t[-1]), 2),
        "terminated": bool(c["done"].any()),
        "waypoints_reached": sum(e["type"] == "waypoint" for e in ev),
    }
    onsets = [e["time_us"] for e in ev if e["type"] == "fault_onset"]
    # Tracking metrics before any fault: post-fault flight (glides, crashes) is too sensitive
    # to cross-machine rounding for tight tolerances.
    settled = (t > 30.0) & (t < (onsets[0] * 1e-6 if onsets else np.inf))
    if settled.sum() > 100:
        alt_err = c["truth.alt_agl"][settled] - c["demands.alt_tecs"][settled]
        m["alt_error_rms_m"] = round(float(np.sqrt(np.mean(alt_err**2))), 3)
        m["tas_mean_mps"] = round(float(np.mean(c["truth.tas"][settled])), 3)
        xt = c["demands.xtrack"][settled]
        m["xtrack_rms_m"] = round(float(np.sqrt(np.mean(xt**2))), 3)
    det = [e["time_us"] for e in ev if e["type"] == "failure_detected"]
    if onsets and det:
        m["detect_latency_s"] = round((det[0] - onsets[0]) * 1e-6, 3)
        m["zone"] = next(e["payload"]["zone"] for e in ev if e["type"] == "zone_selected")
    td = [e for e in ev if e["type"] == "touchdown"]
    if td:
        p = td[0]["payload"]
        m["touchdown_class"] = p["class"]
        m["touchdown_n_m"], m["touchdown_e_m"] = p["pos_ne_m"]
    bad = [e for e in ev if e["type"] == "airspeed_health" and not e["payload"]["healthy"]]
    if bad:
        m["airspeed_unhealthy_at_s"] = round(bad[0]["time_us"] * 1e-6, 2)
    if onsets:
        after = t >= onsets[0] * 1e-6
        err = np.abs(c["sensors.tas"][after] - c["truth.tas"][after])
        m["max_airspeed_error_mps"] = round(float(err.max()), 2)
    return m


def compare(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Differences beyond tolerance, as messages."""
    problems = []
    for k in sorted(set(expected) | set(actual)):
        if k not in expected or k not in actual:
            problems.append(f"{k}: expected {expected.get(k)!r}, got {actual.get(k)!r}")
            continue
        tol = TOLERANCES.get(k)
        e, a = expected[k], actual[k]
        if tol is None:
            if e != a:
                problems.append(f"{k}: expected {e!r}, got {a!r}")
        elif abs(float(a) - float(e)) > tol:
            problems.append(f"{k}: expected {e} ± {tol}, got {a}")
    return problems
