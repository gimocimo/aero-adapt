"""D4 (M2 spec): the held-out Rascal 110 flown in JSBSim by the stock stack, a short campaign.

The stack is the one auto-configured and auto-tuned on the Rascal's datasheet airframe
(``tools/rascal_airframe.py``, ``configs/fleets/rascal``); JSBSim flies the Rascal's own model
(``aeroadapt.sim.jsbsim.rascal``), which the stack never saw. Every flight is the M1 loop
(``configs/scenarios/rascal-nominal.yaml``) with its wind, turbulence and faults varied, and is
recorded as a replay bundle under ``runs/rascal/<flight>`` (Skyview: ``?bundle=…``).

Each nominal flight must complete without ground contact or loss of control; each motor cut
must be detected and end in a touchdown that is not a crash. ``runs/rascal/summary.json`` holds
the results, and the same flights in aerojax (on the datasheet model) for comparison.

Usage (the jsbsim extra; the Rascal is fetched into the user cache on first use)::

    uv run --extra jsbsim python tools/rascal_flights.py [--workers 4]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from aeroadapt.scenarios.schema import Scenario, load_scenario

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "rascal"

# (name, wind from deg, wind m/s, turbulence, motor cut at s or None, seed)
FLIGHTS: tuple[tuple[str, float, float, str, float | None, int], ...] = (
    ("calm", 250.0, 0.0, "none", None, 1),
    ("nominal", 250.0, 6.0, "light", None, 1001),
    ("wind-n", 0.0, 6.0, "light", None, 2),
    ("wind-e", 90.0, 6.0, "light", None, 3),
    ("wind-s", 180.0, 6.0, "light", None, 4),
    ("wind-w-8", 270.0, 8.0, "light", None, 5),
    ("moderate", 250.0, 6.0, "moderate", None, 6),
    ("moderate-2", 30.0, 5.0, "moderate", None, 7),
    ("motorcut-60", 250.0, 6.0, "light", 60.0, 1002),
    ("motorcut-110", 250.0, 6.0, "light", 110.0, 8),
    ("motorcut-170", 120.0, 4.0, "light", 170.0, 9),
    ("motorcut-230", 300.0, 5.0, "light", 230.0, 10),
)


def scenario(
    name: str, wind_from: float, wind: float, turb: str, cut: float | None, seed: int
) -> Scenario:
    base = load_scenario("rascal-nominal").model_dump(mode="json")
    base.update(id=f"rascal-{name}", seed=seed)
    env: dict[str, Any] = {"wind": {"from_deg": wind_from, "speed_mps": wind}}
    if turb != "none":
        env["turbulence"] = {"model": "dryden_low_alt", "intensity": turb}
    base["environment"] = env
    base["faults"] = [] if cut is None else [{"type": "motor_cut", "trigger": {"at_time_s": cut}}]
    return Scenario.model_validate(base)


def summarise(cols: dict[str, np.ndarray], events: list[dict[str, Any]]) -> dict[str, Any]:
    """Height, speed, bank and cross-track while flying under power (after 20 s, before any
    fault), the failure detection (delay after the cut, or a false alarm before it) and the
    touchdown."""
    t = cols["time_us"] * 1e-6
    faulted = np.cumsum(cols["labels.fault_bits"] != 0).astype(bool)
    flying = (t > 20.0) & ~faulted
    xt = cols["demands.xtrack"][flying & (t > 30.0)]
    impact = next((e["payload"] for e in events if e["type"] == "touchdown"), None)
    failed = cols["cont.failed"] > 0
    cut = int(np.argmax(faulted)) if faulted.any() else len(t)

    def band(x: np.ndarray) -> list[float] | None:
        return [float(x[flying].min()), float(x[flying].max())] if flying.any() else None

    return {
        "duration_s": float(t[-1]),
        "termination": int(cols["termination"][-1]),
        "waypoints": sum(e["type"] == "waypoint" for e in events),
        "alt_agl_m": band(cols["truth.alt_agl"]),
        "tas_mps": band(cols["truth.tas"]),
        "roll_max_deg": (
            float(np.degrees(np.abs(cols["truth.roll"][flying])).max()) if flying.any() else None
        ),
        "xtrack_rms_m": float(np.sqrt(np.mean(xt**2))) if xt.size else None,
        "false_alarm": bool(failed[:cut].any()),
        "failure_detected": bool(failed[cut:].any()),
        "detected_after_s": (
            float(t[cut + int(np.argmax(failed[cut:]))] - t[cut]) if failed[cut:].any() else None
        ),
        "touchdown": impact,
    }


def fly(spec: tuple[str, float, float, str, float | None, int]) -> dict[str, Any]:
    from aeroadapt.harness.host_flight import fly_vehicle
    from aeroadapt.scenarios.resolve import resolve
    from aeroadapt.scenarios.runner import record, run_scenario
    from aeroadapt.sim.jsbsim.rascal import CACHE_ROOT, DERIVED, MODEL_SIGNS
    from aeroadapt.vehicle.jsbsim_vehicle import JSBSimVehicle

    name = spec[0]
    r = resolve(scenario(*spec))
    vehicle = JSBSimVehicle(
        r.airframe, r.site, r.env, r.faults, r.config, model=DERIVED, root=CACHE_ROOT,
        model_signs=MODEL_SIGNS,
    )  # fmt: skip
    res = fly_vehicle(r, vehicle)
    record(res, OUT / name)
    ref = run_scenario(r)  # aerojax, on the datasheet model the stack was tuned on
    return {
        "flight": name,
        "wind_from_deg": spec[1],
        "wind_mps": spec[2],
        "turbulence": spec[3],
        "motor_cut_s": spec[4],
        "jsbsim": summarise(res.cols, res.events),
        "aerojax": summarise(ref.cols, ref.events),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="D4: the Rascal 110 in JSBSim, a short campaign")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from aeroadapt.sim.jsbsim.rascal import derived_model

    derived_model()  # fetch and write once, before the workers start
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(fly, FLIGHTS))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        j = row["jsbsim"]
        td = j["touchdown"]

        def f(x: float | None, fmt: str = ".1f") -> str:
            return "–" if x is None else format(x, fmt)

        alt, tas = j["alt_agl_m"] or [None, None], j["tas_mps"] or [None, None]
        print(
            f"{row['flight']:14s} {j['duration_s']:5.0f} s, termination {j['termination']}, "
            f"wp {j['waypoints']}, alt {f(alt[0], '.0f')}-{f(alt[1], '.0f')} m, "
            f"TAS {f(tas[0])}-{f(tas[1])}, roll {f(j['roll_max_deg'], '.0f')}°, "
            f"xtrack {f(j['xtrack_rms_m'])} m"
            + (", FALSE ALARM" if j["false_alarm"] else "")
            + (f", detected +{f(j['detected_after_s'])} s" if j["failure_detected"] else "")
            + (f", touchdown {td['class']} in zone {td['zone']}: {td['in_zone']}" if td else "")
        )
    print(OUT / "summary.json")


if __name__ == "__main__":
    main()
