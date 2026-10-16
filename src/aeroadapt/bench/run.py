"""Bench runner (M2 spec §8): every sampled scenario flown by each baseline, batched in one
``vmap`` over mixed airframes, weather, faults and missions.

- **B1:** the stock stack with no contingency (ArduPlane-like: it keeps flying the mission).
- **B2:** the stock stack plus the classical contingency v0, which acts only on a detected
  motor failure. For other faults B2 behaves as B1, unless its detector fires; the tables
  show that rather than hide it.

The two baselines of a scenario share the start state and the noise (turbulence, sensors),
so their difference is the contingency alone.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from aeroadapt.airframes import Airframe, to_params
from aeroadapt.airframes.fleet import REPO, fleet_path, load_fleet
from aeroadapt.bench.sampler import (
    BenchSpec,
    bench_airframes,
    fault_category,
    fault_kind,
    sample_scenarios,
    write_scenarios,
)
from aeroadapt.bench.stats import bootstrap_mean, wilson
from aeroadapt.core import quaternion
from aeroadapt.harness.closed_loop import closed_loop, setup_loop
from aeroadapt.scenarios.resolve import ResolvedScenario, resolve
from aeroadapt.scenarios.schema import Scenario
from aeroadapt.stack.autopilot import MODE_AUTO, stack_init
from aeroadapt.stack.config import StackCfg, load_stack_cfg
from aeroadapt.stack.contingency import FINAL, FLARE, STRAIGHT_IN
from aeroadapt.vehicle.protocol import Termination, classify_touchdown

LOG_EVERY = 10  # 10 Hz
BATCH = 64  # flights per vmap
LOC_ROLL_DEG, LOC_PITCH_DEG = 90.0, 60.0  # attitude beyond these is a loss of control
AIRSPEED_KINDS = {"pitot_blockage", "airspeed_bias", "airspeed_stuck", "airspeed_drift"}
PLANNED = (FINAL, FLARE, STRAIGHT_IN)  # contingency phases that end in a planned landing


@dataclass
class Outcome:
    scenario: str
    baseline: str
    airframe: str
    airframe_class: str
    wind_frac: float  # mean wind / cruise speed
    turbulence: str
    events: str  # e.g. "gust+thermals"
    fault_mode: str  # none | single | double
    faults: str  # kinds, "+"-joined
    category: str  # sensor | actuator | propulsion | energy | airframe | multiple | none
    first_onset_s: float | None
    survived: bool
    loss_of_control: bool
    ground_contact: bool
    planned_landing: bool
    touchdown: str | None  # controlled | hard | crash
    zone: int  # selected zone (-1: none)
    in_zone: bool
    mission_completed: bool  # one full lap
    motor_failure_detected: bool
    detect_latency_s: float | None  # motor cut → detection
    airspeed_flag_latency_s: float | None  # airspeed fault → flagged unreliable
    false_alarm: bool  # a detector fired on a fault-free flight
    flight_time_s: float


@dataclass
class BenchResult:
    spec: BenchSpec
    outcomes: list[Outcome]
    seconds: float


def _stack(trees: Sequence[Any]) -> Any:
    return jax.tree.map(lambda *x: jnp.stack([jnp.asarray(v) for v in x]), *trees)


def _wind_band(frac: float) -> str:
    return (
        "calm-ish (<25 %)"
        if frac < 0.25
        else "moderate (25-50 %)"
        if frac < 0.5
        else "strong (≥50 %)"
    )


def _category(faults: list[dict[str, Any]]) -> str:
    if not faults:
        return "none"
    return fault_category(faults[0]) if len(faults) == 1 else "multiple"


def _per_hour(clean: Sequence[Outcome], hours: float) -> float | None:
    return round(sum(o.false_alarm for o in clean) / hours, 3) if hours else None


def _outcome(
    sc: Scenario, r: ResolvedScenario, baseline: str, sim: Any, st: Any, log: Any
) -> Outcome:
    faults = [f.model_dump(mode="json") for f in sc.faults]
    kinds = [fault_kind(f) for f in faults]
    onsets = [f["trigger"]["at_time_s"] for f in faults]
    cut = [f["trigger"]["at_time_s"] for f in faults if f["type"] == "motor_cut"]
    air = [f["trigger"]["at_time_s"] for f in faults if fault_kind(f) in AIRSPEED_KINDS
           or f["type"] == "pitot_blockage"]  # fmt: skip
    t = np.asarray(log.t)
    euler = np.degrees(np.asarray(log.euler))
    loc = bool(
        (np.abs(euler[:, 0]) > LOC_ROLL_DEG).any() or (np.abs(euler[:, 1]) > LOC_PITCH_DEG).any()
    )
    term = Termination(int(sim.termination))
    ground = term == Termination.GROUND_CONTACT
    phase, zone = int(st.cont.phase), int(st.cont.zone)
    planned = ground and phase in PLANNED
    cls, in_zone = None, False
    if ground:
        q = sim.rb.quat
        vel = np.asarray(quaternion.body_to_ned(q, sim.rb.vel_body))
        roll, pitch, _ = (float(x) for x in quaternion.to_euler(q))
        cls = classify_touchdown(float(vel[2]), roll, pitch)
        n_, e_ = (float(x) for x in sim.rb.pos_ned[:2])
        in_zone = zone >= 0 and r.site.safe_zones[zone].contains(n_, e_)
    survived = not loc and (not ground or (planned and cls != "crash"))
    wp = np.asarray(log.wp_idx)
    detected = bool(st.cont.failed)
    latency = float(st.cont.t_detect) - cut[0] if detected and cut else None
    unhealthy = np.asarray(log.airspeed_healthy) < 0.5
    flag_latency = None
    if air:
        after = unhealthy & (t >= min(air))
        flag_latency = float(t[np.argmax(after)] - min(air)) if after.any() else None
    env = sc.environment
    ev = [k for k in ("gust", "shear") if getattr(env.events, k) is not None]
    ev += ["thermals"] if env.events.thermals else []
    return Outcome(
        scenario=sc.id,
        baseline=baseline,
        airframe=sc.airframe,
        airframe_class=sc.airframe.split("-")[1] if sc.airframe.startswith("gen-") else "other",
        wind_frac=round(env.wind.speed_mps / sc.mission.speed_mps, 3),
        turbulence=env.turbulence.intensity,
        events="+".join(ev) or "none",
        fault_mode=("none", "single", "double")[min(len(faults), 2)],
        faults="+".join(kinds) or "none",
        category=_category(faults),
        first_onset_s=min(onsets) if onsets else None,
        survived=survived,
        loss_of_control=loc,
        ground_contact=ground,
        planned_landing=planned,
        touchdown=cls,
        zone=zone,
        in_zone=in_zone,
        mission_completed=int((np.diff(wp) != 0).sum()) >= int(r.mission.n_wp),
        motor_failure_detected=detected,
        detect_latency_s=latency,
        airspeed_flag_latency_s=flag_latency,
        false_alarm=not faults and (detected or bool(unhealthy.any())),
        flight_time_s=float(sim.t),
    )  # fmt: skip


def _fly(flights: list[tuple[Scenario, ResolvedScenario, str, Any]], n: int) -> list[Outcome]:
    """One vmapped batch of (scenario, resolved, baseline, setup) flights."""
    cfg = flights[0][1].config
    st0 = [
        stack_init(f[3].sp if f[2] == "B2" else _no_cont(f[3].sp), f[3].z0, MODE_AUTO)
        for f in flights
    ]

    def one(vp, env, faults, sp, mission, sim0, s0, z0, key):
        return closed_loop(vp, env, faults, sp, mission, sim0, s0, z0, key,
                           n_steps=n, config=cfg, log_every=LOG_EVERY)  # fmt: skip

    sp = [f[3].sp if f[2] == "B2" else _no_cont(f[3].sp) for f in flights]
    keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(f[0].seed), 1) for f in flights])
    sims, sts, logs = jax.vmap(one)(
        _stack([f[3].vp for f in flights]), _stack([f[3].env for f in flights]),
        _stack([f[3].faults for f in flights]), _stack(sp), _stack([f[1].mission for f in flights]),
        _stack([f[3].sim0 for f in flights]), _stack(st0), _stack([f[3].z0 for f in flights]), keys,
    )  # fmt: skip
    sims, sts, logs = jax.tree.map(np.asarray, (sims, sts, logs))
    pick = lambda tree, i: jax.tree.map(lambda x: x[i], tree)  # noqa: E731
    return [
        _outcome(f[0], f[1], f[2], pick(sims, i), pick(sts, i), pick(logs, i))
        for i, f in enumerate(flights)
    ]


def _no_cont(sp: Any) -> Any:
    return sp._replace(cont=sp.cont._replace(enabled=False))


def run_bench(
    spec: BenchSpec, out_dir: Path | None = None, log: Callable[[str], None] = print
) -> BenchResult:
    t0 = time.perf_counter()
    pool = bench_airframes(spec)
    airframes: dict[str, Airframe] = {a.id: a for a in load_fleet(fleet_path(spec.fleet), pool)}
    stacks: dict[str, StackCfg] = {i: load_stack_cfg(REPO / spec.stacks, i) for i in pool}
    cruise = {i: stacks[i].tecs.airspeed_cruise_mps for i in pool}
    chord = {i: float(to_params(airframes[i]).c) for i in pool}
    scenarios = sample_scenarios(spec, cruise, chord)
    if out_dir is not None:
        write_scenarios(scenarios, out_dir / "scenarios")
    n = round(spec.duration_s / 0.01)
    n -= n % LOG_EVERY
    flights = []
    for sc in scenarios:
        r = resolve(sc, airframe=airframes[sc.airframe], stack_cfg=stacks[sc.airframe])
        su = setup_loop(r.airframe, r.site, r.env, r.initial, faults=r.faults, stack=r.stack,
                        seed=sc.seed, config=r.config)  # fmt: skip
        flights += [(sc, r, b, su) for b in spec.baselines]
    log(f"{len(scenarios)} scenarios over {len(airframes)} airframes prepared "
        f"({time.perf_counter() - t0:.0f} s)")  # fmt: skip
    outcomes: list[Outcome] = []
    for k in range(0, len(flights), BATCH):
        outcomes += _fly(flights[k : k + BATCH], n)
        log(
            f"{min(k + BATCH, len(flights))}/{len(flights)} flights "
            f"({time.perf_counter() - t0:.0f} s)"
        )
    return BenchResult(spec, outcomes, time.perf_counter() - t0)


# --------------------------------------------------------------------------- summary
def _rate(rows: Sequence[Outcome], key: str) -> dict[str, float]:
    k = sum(bool(getattr(o, key)) for o in rows)
    p, lo, hi = wilson(k, len(rows))
    return {"n": len(rows), "k": k, "rate": round(p, 4), "lo": round(lo, 4), "hi": round(hi, 4)}


def _by(rows: Sequence[Outcome], key: Callable[[Outcome], str], metric: str) -> dict[str, Any]:
    groups: dict[str, list[Outcome]] = {}
    for o in rows:
        groups.setdefault(key(o), []).append(o)
    return {g: _rate(v, metric) for g, v in sorted(groups.items())}


def summary(result: BenchResult) -> dict[str, Any]:
    out: dict[str, Any] = {
        "spec": result.spec.model_dump(mode="json"),
        "seconds": round(result.seconds),
        "baselines": {},
    }
    for b in result.spec.baselines:
        rows = [o for o in result.outcomes if o.baseline == b]
        cut = [o for o in rows if "motor_cut" in o.faults.split("+")]
        clean = [o for o in rows if o.fault_mode == "none"]
        lat = [o.detect_latency_s for o in cut if o.detect_latency_s is not None]
        flag = [o.airspeed_flag_latency_s for o in rows if o.airspeed_flag_latency_s is not None]
        hours = sum(o.flight_time_s for o in clean) / 3600
        mean, lo, hi = bootstrap_mean(lat)
        fmean, flo, fhi = bootstrap_mean(flag)
        out["baselines"][b] = {
            "flights": len(rows),
            "survival": _rate(rows, "survived"),
            "mission_completion": _rate(rows, "mission_completed"),
            "loss_of_control": _rate(rows, "loss_of_control"),
            "motor_cut": {
                "survival": _rate(cut, "survived"),
                "landed_in_zone": _rate(cut, "in_zone"),
                "touchdown": {
                    c: sum(o.touchdown == c for o in cut) for c in ("controlled", "hard", "crash")
                },
                "detection": _rate(cut, "motor_failure_detected"),
                "detect_latency_s": {"mean": mean, "lo": lo, "hi": hi, "n": len(lat)},
            },
            "airspeed_flag_latency_s": {"mean": fmean, "lo": flo, "hi": fhi, "n": len(flag)},
            "false_alarms": {
                **_rate(clean, "false_alarm"),
                "per_flight_hour": _per_hour(clean, hours),
            },
            "survival_by_category": _by(rows, lambda o: o.category, "survived"),
            "survival_by_class": _by(rows, lambda o: o.airframe_class, "survived"),
            "survival_by_wind": _by(rows, lambda o: _wind_band(o.wind_frac), "survived"),
            "survival_by_turbulence": _by(rows, lambda o: o.turbulence, "survived"),
        }
    if {"B1", "B2"} <= set(result.spec.baselines):
        pairs = {}
        for o in result.outcomes:
            pairs.setdefault(o.scenario, {})[o.baseline] = o
        diff = [p for p in pairs.values() if p["B1"].survived != p["B2"].survived]
        out["b2_vs_b1"] = {
            "scenarios_where_survival_differs": len(diff),
            "b2_better": sum(p["B2"].survived for p in diff),
            "b1_better": sum(p["B1"].survived for p in diff),
            "differing_fault_kinds": sorted({p["B1"].faults for p in diff}),
        }
    return out


def write_bench(out_dir: Path, result: BenchResult) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([asdict(o) for o in result.outcomes]), out_dir / "outcomes.parquet"
    )
    (out_dir / "summary.json").write_text(json.dumps(_finite(summary(result)), indent=2) + "\n")
    return out_dir


def _finite(x: Any) -> Any:
    """NaN (an empty group) → null, so summary.json is standard JSON."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v) for v in x]
    return None if isinstance(x, float) and math.isnan(x) else x
