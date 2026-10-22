"""Event extraction from a recorded flight (``events.jsonl``, ADR-005).

Events are derived after the run from the per-tick columns, so the flight core stays free of
host-side bookkeeping. Each event is ``{"time_us", "type", "payload"}``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from aeroadapt.scenarios.site import Site
from aeroadapt.stack.autopilot import MODE_NAMES
from aeroadapt.stack.contingency import PHASE_NAMES
from aeroadapt.telemetry.schema import FAULT_BITS
from aeroadapt.vehicle.protocol import Termination, classify_touchdown

Event = dict[str, Any]


def _edges(x: np.ndarray) -> np.ndarray:
    """Indices where ``x`` changes value (index of the new value)."""
    return np.flatnonzero(np.diff(x) != 0) + 1


def _ev(cols: dict[str, np.ndarray], k: int, type_: str, **payload: Any) -> Event:
    return {"time_us": int(cols["time_us"][k]), "type": type_, "payload": payload}


def touchdown_record(cols: dict[str, np.ndarray], k: int, site: Site, zone: int) -> dict[str, Any]:
    """Impact state at row ``k`` (the first terminated row)."""
    sink = float(cols["truth.vel_d"][k])
    roll, pitch = float(cols["truth.roll"][k]), float(cols["truth.pitch"][k])
    n, e = float(cols["truth.pos_n"][k]), float(cols["truth.pos_e"][k])
    in_zone = None
    if zone >= 0:
        in_zone = site.safe_zones[zone].contains(n, e)
    out = {
        "pos_ne_m": [round(n, 2), round(e, 2)],
        "sink_rate_mps": round(sink, 3),
        "groundspeed_mps": round(float(cols["truth.groundspeed"][k]), 3),
        "roll_deg": round(math.degrees(roll), 2),
        "pitch_deg": round(math.degrees(pitch), 2),
        "class": classify_touchdown(sink, roll, pitch),
        "zone": site.safe_zones[zone].id if zone >= 0 else None,
        "in_zone": in_zone,
    }
    if site.world is not None:
        # Into rising ground the ground closes faster than the sink rate: classify by the
        # speed towards the slope (sink plus the slope along the ground track).
        from aeroadapt.world import load_world

        dtm = load_world(site.world).dtm
        grad = np.array([dtm.at(n + 1.0, e) - dtm.at(n - 1.0, e),
                         dtm.at(n, e + 1.0) - dtm.at(n, e - 1.0)]) / 2.0  # fmt: skip
        v = np.array([float(cols["truth.vel_n"][k]), float(cols["truth.vel_e"][k])])
        closing = sink + float(grad @ v)
        out["surface"] = _world_at(site, n, e)
        out["slope_pct"] = round(100.0 * float(np.hypot(*grad)), 1)
        out["closing_speed_mps"] = round(closing, 3)
        out["class"] = classify_touchdown(max(sink, closing), roll, pitch)
    return out


def _world_at(site: Site, n: float, e: float) -> str:
    from aeroadapt.world import load_world
    from aeroadapt.world.places import what_is_at

    assert site.world is not None
    return what_is_at(load_world(site.world), n, e)


def extract_events(cols: dict[str, np.ndarray], site: Site) -> list[Event]:
    events: list[Event] = [_ev(cols, 0, "start", mode=MODE_NAMES[int(cols["mode"][0])])]

    bits = cols["labels.fault_bits"]
    for name, bit in FAULT_BITS.items():
        on = (bits & bit) > 0
        for k in np.flatnonzero(on[1:] & ~on[:-1]) + 1:
            events.append(_ev(cols, int(k), "fault_onset", fault=name))
        if on[0]:
            events.append(_ev(cols, 0, "fault_onset", fault=name))
        for k in np.flatnonzero(on[:-1] & ~on[1:]) + 1:  # sensor faults with a clear time
            events.append(_ev(cols, int(k), "fault_cleared", fault=name))

    failed = cols["cont.failed"]
    for k in np.flatnonzero(failed[1:] > failed[:-1]) + 1:
        events.append(_ev(cols, int(k), "failure_detected", detector="motor"))
    zone = cols["cont.zone"]
    for k in _edges(zone):
        z = int(zone[k])
        events.append(
            _ev(cols, int(k), "zone_selected", zone=site.safe_zones[z].id if z >= 0 else None)
        )
    phase = cols["cont.phase"]
    for k in _edges(phase):
        events.append(
            _ev(
                cols,
                int(k),
                "contingency_phase",
                phase=PHASE_NAMES[int(phase[k])],
                previous=PHASE_NAMES[int(phase[k - 1])],
            )
        )
    mode = cols["mode"]
    for k in _edges(mode):
        events.append(_ev(cols, int(k), "mode", mode=MODE_NAMES[int(mode[k])]))
    wp = cols["wp_idx"]
    for k in _edges(wp):
        events.append(_ev(cols, int(k), "waypoint", index=int(wp[k])))
    healthy = cols["est.airspeed_healthy"]
    for k in _edges(healthy):
        events.append(_ev(cols, int(k), "airspeed_health", healthy=bool(healthy[k])))

    done = cols["done"]
    if done.any():
        k = int(np.argmax(done > 0))
        term = Termination(int(cols["termination"][k]))
        if term == Termination.GROUND_CONTACT:
            events.append(
                _ev(cols, k, "touchdown", **touchdown_record(cols, k, site, int(zone[k])))
            )
        elif term == Termination.OBSTACLE and site.world is not None:
            n, e = float(cols["truth.pos_n"][k]), float(cols["truth.pos_e"][k])
            z = int(zone[k])
            events.append(_ev(cols, k, "collision", pos_ne_m=[round(n, 2), round(e, 2)],
                              height_m=round(float(cols["truth.hagl"][k]), 2),
                              speed_mps=round(float(cols["truth.groundspeed"][k]), 2),
                              hit=_world_at(site, n, e),
                              zone=site.safe_zones[z].id if z >= 0 else None))  # fmt: skip
        else:
            events.append(_ev(cols, k, "terminated", reason=term.name))
    events.append(_ev(cols, len(cols["time_us"]) - 1, "end"))
    events.sort(key=lambda e: (e["time_us"], e["type"]))
    return events
