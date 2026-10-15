"""Campaigns (spec §10.2): seeded variations of a template scenario, flown in batches.

Variants share the site, airframe and duration, so they share array shapes. Every per-run
input (initial state, environment, faults, mission, stack parameters, PRNG keys) is stacked
and the closed loop is ``vmap``-ed: one compilation, then tens of flights per second on a
laptop CPU. Outcomes come from the final states plus a sparse (1 Hz) trajectory.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field

from aeroadapt.core import quaternion
from aeroadapt.harness.closed_loop import LoopLog, LoopSetup, closed_loop, setup_loop
from aeroadapt.scenarios.resolve import ResolvedScenario, resolve
from aeroadapt.scenarios.schema import Scenario, load_scenario
from aeroadapt.vehicle.protocol import Termination, classify_touchdown

TRAJ_EVERY = 100  # sparse trajectory: every 100 control steps (1 Hz)
CAMPAIGN_DIR = Path(__file__).resolve().parents[3] / "configs" / "campaigns"


@dataclass(frozen=True)
class Variation:
    """Uniform ranges for the varied quantities (None: keep the template's value)."""

    fault_onset_s: tuple[float, float] | None = None
    wind_speed_mps: tuple[float, float] | None = None
    wind_from_deg: tuple[float, float] | None = None
    initial_offset_m: float | None = None  # uniform in a square of this half-size


def make_variants(template: Scenario, n: int, seed: int, vary: Variation) -> list[Scenario]:
    """``n`` scenarios drawn reproducibly from ``seed``; each gets its own run seed."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        d = template.model_dump(mode="json")
        d["id"] = f"{template.id}-{i:03d}"
        d["seed"] = int(rng.integers(2**31))
        wind = d["environment"]["wind"]
        if vary.wind_speed_mps is not None:
            wind["speed_mps"] = float(rng.uniform(*vary.wind_speed_mps))
        if vary.wind_from_deg is not None:
            wind["from_deg"] = float(rng.uniform(*vary.wind_from_deg))
        if vary.initial_offset_m is not None:
            off = rng.uniform(-vary.initial_offset_m, vary.initial_offset_m, 2)
            n0, e0 = d["initial"]["pos_ne_m"]
            d["initial"]["pos_ne_m"] = [n0 + float(off[0]), e0 + float(off[1])]
        if vary.fault_onset_s is not None:
            onset = float(rng.uniform(*vary.fault_onset_s))
            for f in d["faults"]:
                f["trigger"]["at_time_s"] = onset
        out.append(Scenario.model_validate(d))
    return out


@dataclass(frozen=True)
class Outcome:
    """One flight's result, from final states and the sparse log (truth: privileged)."""

    scenario_id: str
    fault_onset_s: float | None
    detected: bool
    detect_latency_s: float | None
    zone: int  # selected zone (-1: none, i.e. straight in / not detected)
    reachable_at_detection: bool  # a zone was selected when the failure was detected
    terminated: bool
    termination: str
    touchdown_ne_m: tuple[float, float] | None
    sink_rate_mps: float | None
    touchdown: str | None  # controlled / hard / crash
    in_zone: bool
    aim_error_m: tuple[float, float] | None  # touchdown vs aim point: (along course, lateral)
    flight_time_s: float


def _stack(trees: Sequence[Any]) -> Any:
    return jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *trees)


def _setup(r: ResolvedScenario) -> LoopSetup:
    return setup_loop(
        r.airframe,
        r.site,
        r.env,
        r.initial,
        faults=r.faults,
        stack=r.stack,
        seed=r.scenario.seed,
        config=r.config,
    )


def run_batch(resolved: Sequence[ResolvedScenario]) -> tuple[Any, Any, LoopLog]:
    """Fly a batch (one ``vmap``); returns stacked final sim/stack states and 1 Hz logs."""
    r0 = resolved[0]
    n = r0.n_steps - r0.n_steps % TRAJ_EVERY
    if any(r.n_steps != r0.n_steps or r.site.id != r0.site.id for r in resolved):
        raise ValueError("a batch needs one site and one duration")
    setups = [_setup(r) for r in resolved]
    su = setups[0]
    keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(r.scenario.seed), 1) for r in resolved])

    def one(env, faults, sp, mission, sim0, st0, z0, key):
        return closed_loop(
            su.vp,
            env,
            faults,
            sp,
            mission,
            sim0,
            st0,
            z0,
            key,
            n_steps=n,
            config=su.config,
            log_every=TRAJ_EVERY,
        )

    return jax.vmap(one)(
        _stack([s.env for s in setups]),
        _stack([s.faults for s in setups]),
        _stack([s.sp for s in setups]),
        _stack([r.mission for r in resolved]),
        _stack([s.sim0 for s in setups]),
        _stack([s.stack0 for s in setups]),
        _stack([s.z0 for s in setups]),
        keys,
    )


def outcomes(
    resolved: Sequence[ResolvedScenario], sims: Any, stacks: Any, logs: LoopLog
) -> list[Outcome]:
    sims, stacks, logs = jax.tree.map(np.asarray, (sims, stacks, logs))
    out = []
    for i, r in enumerate(resolved):
        sc, site = r.scenario, r.site
        onsets = [f.trigger.at_time_s for f in sc.faults if f.type == "motor_cut"]
        onset = onsets[0] if onsets else None
        cont = jax.tree.map(lambda x, i=i: x[i], stacks.cont)
        detected = bool(cont.failed)
        latency = float(cont.t_detect) - onset if detected and onset is not None else None
        failed_rows = np.flatnonzero(logs.failed[i])
        zone_at_detect = int(logs.zone[i][failed_rows[0]]) if len(failed_rows) else -1
        zone = int(cont.zone)
        done = bool(sims.done[i])
        term = Termination(int(sims.termination[i]))
        td_ne, sink, cls, in_zone, aim_err = None, None, None, False, None
        if term == Termination.GROUND_CONTACT:
            q = sims.rb.quat[i]
            vel = np.asarray(quaternion.body_to_ned(q, sims.rb.vel_body[i]))
            roll, pitch, _ = (float(x) for x in quaternion.to_euler(q))
            n_, e_ = (float(x) for x in sims.rb.pos_ned[i][:2])
            td_ne, sink = (n_, e_), float(vel[2])
            cls = classify_touchdown(sink, roll, pitch)
            in_zone = zone >= 0 and site.safe_zones[zone].contains(n_, e_)
            if zone >= 0:
                d = np.asarray(cont.land_dir, dtype=float)
                rel = np.array([n_, e_]) - np.asarray(cont.touchdown, dtype=float)
                aim_err = (float(rel @ d), float(d[0] * rel[1] - d[1] * rel[0]))
        out.append(
            Outcome(
                scenario_id=sc.id,
                fault_onset_s=onset,
                detected=detected,
                detect_latency_s=latency,
                zone=zone,
                reachable_at_detection=zone_at_detect >= 0,
                terminated=done,
                termination=term.name,
                touchdown_ne_m=td_ne,
                sink_rate_mps=sink,
                touchdown=cls,
                in_zone=in_zone,
                aim_error_m=aim_err,
                flight_time_s=float(sims.t[i]),
            )
        )
    return out


def run_campaign(
    scenarios: Sequence[Scenario], batch_size: int = 25
) -> tuple[list[Outcome], LoopLog]:
    """All outcomes, plus the stacked 1 Hz trajectories (leading axis: scenario)."""
    resolved = [resolve(s) for s in scenarios]
    results, logs = [], []
    for k in range(0, len(resolved), batch_size):
        chunk = resolved[k : k + batch_size]
        sims, stacks, log = run_batch(chunk)
        results += outcomes(chunk, sims, stacks, log)
        logs.append(jax.tree.map(np.asarray, log))
    return results, jax.tree.map(lambda *xs: np.concatenate(xs), *logs)


def summarise(results: Sequence[Outcome]) -> dict[str, Any]:
    reach = [o for o in results if o.reachable_at_detection]
    good = [o for o in reach if o.in_zone and o.touchdown in ("controlled", "hard")]
    lat = [o.detect_latency_s for o in results if o.detect_latency_s is not None]
    return {
        "n": len(results),
        "detected": sum(o.detected for o in results),
        "latency_max_s": max(lat) if lat else math.nan,
        "latency_mean_s": float(np.mean(lat)) if lat else math.nan,
        "reachable_at_detection": len(reach),
        "landed_in_zone": len(good),
        "in_zone_rate": len(good) / len(reach) if reach else math.nan,
        "crashes": sum(o.touchdown == "crash" for o in results),
    }


# --------------------------------------------------------------------------- campaign files
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VaryCfg(_Strict):
    fault_onset_s: tuple[float, float] | None = None
    wind_speed_mps: tuple[float, float] | None = None
    wind_from_deg: tuple[float, float] | None = None
    initial_offset_m: float | None = None


class CampaignSpec(_Strict):
    id: str
    template: str  # scenario id
    duration_s: float | None = Field(default=None, gt=0)
    n: int = Field(gt=0, le=10_000)
    seed: int = 0
    vary: VaryCfg = VaryCfg()

    def scenarios(self) -> list[Scenario]:
        t = load_scenario(self.template)
        if self.duration_s is not None:
            t = t.model_copy(update={"duration_s": self.duration_s})
        return make_variants(t, self.n, self.seed, Variation(**self.vary.model_dump()))


def load_campaign(name_or_path: str | Path) -> CampaignSpec:
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = CAMPAIGN_DIR / f"{name_or_path}.yaml"
    with path.open() as f:
        return CampaignSpec.model_validate(yaml.safe_load(f))


def failure_cause(o: Outcome) -> str | None:
    """Why a reachable case did not end in a good in-zone landing (V12: list every failure)."""
    if not o.reachable_at_detection:
        return None
    if not o.terminated:
        return "still airborne at the end of the run"
    if o.touchdown == "crash":
        return f"crash (sink {o.sink_rate_mps:.1f} m/s)"
    if not o.in_zone and o.aim_error_m is not None:
        along, lateral = o.aim_error_m
        if abs(lateral) > 40.0:
            return f"landed {abs(lateral):.0f} m to the side of the final course"
        return f"landed {abs(along):.0f} m {'short of' if along < 0 else 'past'} the aim point"
    if not o.in_zone:
        return "landed outside the zone"
    return None


def write_outputs(out_dir: Path, spec: CampaignSpec, results: Sequence[Outcome]) -> Path:
    """summary.json (headline numbers + failures) and outcomes.parquet (one row per run)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(o) for o in results]
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    for k in ("touchdown_ne_m", "aim_error_m"):
        cols[k] = [list(v) if v is not None else None for v in cols[k]]
    pq.write_table(pa.table(cols), out_dir / "outcomes.parquet", compression="zstd")
    summary = {
        "campaign": spec.model_dump(mode="json"),
        "summary": summarise(results),
        "failures": [
            {"scenario": o.scenario_id, "cause": c}
            for o in results
            if (c := failure_cause(o)) is not None
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return out_dir
