"""Confounded-pair families (doc 03 §3.2, M2 spec §7.3, V21c).

Each pair is a fault (arm ``a``) and a look-alike (arm ``b``: another fault, a weather
condition or a rigging error), flown from the same template scenario with the same seeds
(common random numbers: identical turbulence and sensor noise), next to a baseline arm with
neither. M2 checks three things over the pair's horizon after onset:

1. the M1 stack's simple monitors (airspeed health, motor-failure detection) respond the same
   way to both arms: both silent, or both flagging within a second of each other;
2. both arms produce the shared symptom, in the same direction and at a similar size (within
   a factor of 2 of each other, relative to the baseline);
3. the ground truth differs: arm ``a`` carries a fault label, and arm ``b`` a different one
   or none (weather and rigging errors are not faults).

These are the cases M4's diagnoser must resolve; M2 only shows that they are genuinely
ambiguous to the stock stack.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from aeroadapt.airframes import Airframe
from aeroadapt.harness.closed_loop import LoopLog, closed_loop, setup_loop
from aeroadapt.scenarios.resolve import resolve
from aeroadapt.scenarios.schema import Scenario, load_scenario

PAIR_DIR = Path(__file__).resolve().parents[3] / "configs" / "campaigns" / "confounded"
LOG_EVERY = 10  # 10 Hz
FLAG_TOLERANCE_S = 1.0
SYMPTOM_RATIO = 2.0

Symptom = Literal["ias_drop", "wind_shift", "throttle_rise", "elevator_shift", "roll_rate_rms"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArmCfg(_Strict):
    """Changes to the template: faults (triggered at the pair's onset unless they say
    otherwise), and patches deep-merged into the scenario's environment or the airframe."""

    faults: list[dict[str, Any]] = Field(default_factory=list)
    environment: dict[str, Any] = Field(default_factory=dict)
    airframe: dict[str, Any] = Field(default_factory=dict)


class PairCfg(_Strict):
    id: str
    description: str
    template: str
    onset_s: float = Field(gt=5)
    horizon_s: float = Field(gt=0)
    seeds: int = Field(default=8, ge=1)
    seed: int = 0
    symptom: Symptom
    min_symptom: float = Field(gt=0, description="smallest median symptom that counts as shown")
    a: ArmCfg
    b: ArmCfg


def load_pair(name_or_path: str | Path) -> PairCfg:
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = PAIR_DIR / f"{name_or_path}.yaml"
    return PairCfg.model_validate(yaml.safe_load(path.read_text()))


def pair_ids() -> list[str]:
    return sorted(p.stem for p in PAIR_DIR.glob("*.yaml"))


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def arm_scenario(cfg: PairCfg, arm: ArmCfg, seed: int) -> Scenario:
    d = load_scenario(cfg.template).model_dump(mode="json")
    d["id"] = f"{cfg.id}-{seed}"
    d["seed"] = seed
    d["duration_s"] = cfg.onset_s + cfg.horizon_s + 1.0
    d["environment"] = _merge(d["environment"], arm.environment)
    trigger = {"at_time_s": cfg.onset_s}
    d["faults"] = [*d["faults"], *({"trigger": trigger, **f} for f in arm.faults)]
    return Scenario.model_validate(d)


@dataclass
class PairResult:
    pair: str
    horizon_s: float
    flags_agree: list[bool]  # per seed: monitors respond alike to both arms
    flag_times: dict[str, list[tuple[float, float]]]  # monitor → per seed (arm a, arm b), s
    symptom_a: list[float]  # per seed, relative to the baseline arm
    symptom_b: list[float]
    truth_differs: bool  # arm a is labelled, and arm b's labels differ (every seed)
    min_symptom: float

    @property
    def symptom_ratio(self) -> float:
        a, b = float(np.median(self.symptom_a)), float(np.median(self.symptom_b))
        return a / b if b != 0 else math.inf

    @property
    def symptoms_alike(self) -> bool:
        a, b = float(np.median(self.symptom_a)), float(np.median(self.symptom_b))
        shown = min(abs(a), abs(b)) >= self.min_symptom
        return shown and 1 / SYMPTOM_RATIO <= self.symptom_ratio <= SYMPTOM_RATIO

    @property
    def passed(self) -> bool:
        return all(self.flags_agree) and self.symptoms_alike and self.truth_differs


def _first(flag: np.ndarray, t: np.ndarray) -> float:
    return float(t[np.argmax(flag)]) if flag.any() else math.inf


def _symptom(kind: Symptom, log: LoopLog, w: np.ndarray, late: np.ndarray) -> np.ndarray:
    """The symptom's metric over the window (per run): what the stack sees."""
    if kind == "ias_drop":  # reduced after differencing (see run_pair)
        return np.asarray(log.est_ias[:, w])
    if kind == "wind_shift":
        return np.asarray(
            log.est_wind_ned[:, late, :2].mean(axis=1)
        )  # (runs, 2): compared as vectors
    if kind == "throttle_rise":
        return np.asarray(log.cmd[:, w, 3].mean(axis=1))
    if kind == "elevator_shift":
        return np.asarray(log.cmd[:, late, 1].mean(axis=1))
    roll_rate = np.asarray(log.sensors.gyro[:, w, 0])
    return np.sqrt((roll_rate**2).mean(axis=1))  # roll_rate_rms


def run_pair(cfg: PairCfg) -> PairResult:
    arms = {"base": ArmCfg(), "a": cfg.a, "b": cfg.b}
    seeds = [cfg.seed + k for k in range(cfg.seeds)]
    setups, missions = [], []
    for arm in arms.values():
        for s in seeds:
            r = resolve(arm_scenario(cfg, arm, s))
            af = r.airframe
            if arm.airframe:  # a rigging error: the stack keeps the nominal configuration
                af = Airframe.model_validate(_merge(af.model_dump(mode="json"), arm.airframe))
            su = setup_loop(af, r.site, r.env, r.initial, faults=r.faults, stack=r.stack,
                            seed=s, config=r.config)  # fmt: skip
            setups.append(su)
            missions.append(r.mission)
    config = setups[0].config
    n = round((cfg.onset_s + cfg.horizon_s) / config.dt_control)

    def stack(trees: list[Any]) -> Any:
        return jax.tree.map(lambda *x: jnp.stack([jnp.asarray(v) for v in x]), *trees)

    def one(vp, env, f, sp, mission, sim0, st0, z0, key):
        return closed_loop(vp, env, f, sp, mission, sim0, st0, z0, key, n_steps=n,
                           config=config, log_every=LOG_EVERY)[2]  # fmt: skip

    keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(s), 1) for s in seeds] * 3)
    log = jax.vmap(one)(
        stack([su.vp for su in setups]), stack([su.env for su in setups]),
        stack([su.faults for su in setups]), stack([su.sp for su in setups]), stack(missions),
        stack([su.sim0 for su in setups]), stack([su.stack0 for su in setups]),
        stack([su.z0 for su in setups]), keys,
    )  # fmt: skip
    log = jax.tree.map(np.asarray, log)
    t = log.t[0]
    w = (t >= cfg.onset_s) & (t < cfg.onset_s + cfg.horizon_s)
    late = (t >= cfg.onset_s + cfg.horizon_s - 2.0) & w
    k = len(seeds)
    base, a, b = slice(0, k), slice(k, 2 * k), slice(2 * k, 3 * k)

    flag_times: dict[str, list[tuple[float, float]]] = {}
    agree = [True] * k
    monitors = {"airspeed unhealthy": ~log.airspeed_healthy, "motor failure": log.failed}
    for name, flag in monitors.items():
        times = []
        for i in range(k):
            ta = _first(flag[k + i][w], t[w])
            tb = _first(flag[2 * k + i][w], t[w])
            times.append((ta, tb))
            alike = (ta == tb == math.inf) or abs(ta - tb) <= FLAG_TOLERANCE_S
            agree[i] = agree[i] and alike
        flag_times[name] = times

    m = _symptom(cfg.symptom, log, w, late)
    if cfg.symptom == "ias_drop":  # the largest deficit against the baseline, sample by sample
        sa, sb = (m[a] - m[base]).min(axis=1), (m[b] - m[base]).min(axis=1)
    elif cfg.symptom == "wind_shift":  # signed by the direction of arm b's shift
        db = m[b] - m[base]
        unit = db.mean(axis=0) / max(float(np.linalg.norm(db.mean(axis=0))), 1e-9)
        sa, sb = (m[a] - m[base]) @ unit, db @ unit
    else:
        sa, sb = m[a] - m[base], m[b] - m[base]

    def truth(rows: slice) -> np.ndarray:
        """Per run: the largest value of every label over the window (kinds, flags)."""
        cols = [np.asarray(x)[rows][:, w].reshape(k, int(w.sum()), -1).max(axis=1)
                for x in log.labels]  # fmt: skip
        return np.concatenate(cols, axis=1).astype(np.int64)

    ta, tb = truth(a), truth(b)

    return PairResult(
        pair=cfg.id,
        horizon_s=cfg.horizon_s,
        flags_agree=agree,
        flag_times=flag_times,
        symptom_a=[float(x) for x in sa],
        symptom_b=[float(x) for x in sb],
        truth_differs=bool((ta.any(axis=1) & (ta != tb).any(axis=1)).all()),
        min_symptom=cfg.min_symptom,
    )
