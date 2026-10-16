"""Bench scenario sampler (M2 spec §8). From a spec and a seed it draws, per scenario:

- an airframe of the requested split that passed V20, flown with its tuned stack;
- a mission: a closed loop of 3-6 waypoints around the site, 80-150 m AGL, sized to take
  1.5-3 minutes per lap at the airframe's cruise speed;
- weather: mean wind 0-80 % of cruise speed from any direction, a turbulence level, and
  possibly a 1-cos gust, low-level shear and thermals;
- faults: none, one or two, uniform over the kinds, with magnitudes in each kind's range and
  onsets within the flight.

Every scenario is an ordinary scenario file (it names the fleet and the tuned stacks), so any
bench flight can be re-flown, recorded and replayed on its own.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from aeroadapt.airframes.fleet import REPO
from aeroadapt.bench.splits import Split, split_of
from aeroadapt.faultlab import build_fault_config
from aeroadapt.faultlab.schema import FaultSpec
from aeroadapt.scenarios.schema import TURBULENCE_W20_KT, Scenario

BENCH_DIR = REPO / "configs" / "bench"
_FAULT = TypeAdapter(FaultSpec)
G0 = 9.80665
Rng = np.random.Generator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MissionRanges(_Strict):
    waypoints: tuple[int, int] = (3, 6)
    alt_agl_m: tuple[float, float] = (80.0, 150.0)
    lap_s: tuple[float, float] = (90.0, 180.0)  # one lap at cruise speed
    centre_offset_m: float = 200.0


class WeatherRanges(_Strict):
    wind_frac: tuple[float, float] = (0.0, 0.8)  # of cruise speed
    # Turbulence follows the wind: the Dryden levels are defined by the wind 20 ft above the
    # ground (MIL-F-8785C: light 15 kt, moderate 30 kt, severe 45 kt), so the level is the one
    # nearest the sampled wind, at least light; or smooth air with this probability.
    smooth_air_p: float = 0.2
    gust_p: float = 0.3
    shear_p: float = 0.2
    thermal_p: float = 0.3


class FaultMix(_Strict):
    none: float = 1 / 3
    single: float = 1 / 3
    double: float = 1 / 3
    onset_s: tuple[float, float] = (30.0, 210.0)
    kinds: list[str] | None = None  # restrict to these kinds (default: all)


class BenchSpec(_Strict):
    id: str
    description: str = ""
    fleet: str  # fleet bundle (id under runs/fleets, or a path)
    stacks: str  # the fleet's V20 run: tuned stacks and which airframes passed
    split: Split | Literal["all"] = "test"
    site: str = "cambridge_west"
    n: int = Field(gt=0)
    seed: int = 0
    duration_s: float = Field(default=300.0, gt=0)
    mission: MissionRanges = MissionRanges()
    weather: WeatherRanges = WeatherRanges()
    faults: FaultMix = FaultMix()
    baselines: tuple[str, ...] = ("B1", "B2")


def load_bench_spec(name_or_path: str | Path) -> BenchSpec:
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = BENCH_DIR / f"{name_or_path}.yaml"
    return BenchSpec.model_validate(yaml.safe_load(path.read_text()))


# --------------------------------------------------------------------------- faults
def _sign(rng: Rng) -> float:
    return float(rng.choice([-1.0, 1.0]))


def _u(rng: Rng, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


_CHANNELS = ("aileron", "elevator", "rudder")
FaultSampler = Callable[[Rng, float], dict[str, Any]]  # (rng, mean chord in m) → spec


def _sensor(kind: str, lo: float, hi: float, signed: bool = True, axes: int = 0) -> FaultSampler:
    def draw(rng: Rng, _c: float) -> dict[str, Any]:
        m = _u(rng, lo, hi) * (_sign(rng) if signed else 1.0)
        spec: dict[str, Any] = {"type": "sensor_fault", "kind": kind, "magnitude": m}
        if axes:
            spec["axis"] = int(rng.integers(0, axes))
        return spec

    return draw


def _actuator(kind: str, lo: float = 0.0, hi: float = 0.0, sign: bool = False) -> FaultSampler:
    def draw(rng: Rng, _c: float) -> dict[str, Any]:
        m = _sign(rng) if sign else _u(rng, lo, hi)
        ch = str(rng.choice(_CHANNELS))
        return {"type": "actuator_fault", "kind": kind, "channel": ch, "magnitude": m}

    return draw


def _scalar(family: str, kind: str, lo: float, hi: float, **extra: Any) -> FaultSampler:
    def draw(rng: Rng, _c: float) -> dict[str, Any]:
        spec = {"type": f"{family}_fault", "kind": kind, "magnitude": _u(rng, lo, hi)}
        for k, v in extra.items():
            spec[k] = v(rng) if callable(v) else v
        return spec

    return draw


def _gnss_dropout(rng: Rng, _c: float) -> dict[str, Any]:
    # Half of them last for the rest of the flight (magnitude 0).
    m = 0.0 if rng.random() < 0.5 else _u(rng, 10.0, 60.0)
    return {"type": "sensor_fault", "kind": "gnss_dropout", "magnitude": m}


def _cg_shift(rng: Rng, chord: float) -> dict[str, Any]:
    # Relative to the airframe's size: -10 % to +15 % of the mean chord (aft positive).
    return {"type": "airframe_fault", "kind": "cg_shift", "magnitude": _u(rng, -0.10, 0.15) * chord}


# Every kind, with magnitudes in its own units (faultlab/schema.py).
FAULT_KINDS: dict[str, FaultSampler] = {
    "motor_cut": lambda rng, c: {"type": "motor_cut"},
    "pitot_blockage": lambda rng, c: {
        "type": "pitot_blockage",
        "variant": str(rng.choice(["drain_open", "drain_blocked"])),
    },
    "servo_jam": lambda rng, c: {"type": "servo_jam", "channel": str(rng.choice(_CHANNELS))},
    "airspeed_bias": _sensor("airspeed_bias", 2.0, 8.0),
    "airspeed_stuck": lambda rng, c: {"type": "sensor_fault", "kind": "airspeed_stuck"},
    "airspeed_drift": _sensor("airspeed_drift", 0.1, 0.8),
    "gyro_bias": _sensor("gyro_bias", 0.01, 0.08, axes=3),
    "accel_bias": _sensor("accel_bias", 0.3, 2.0, axes=3),
    "imu_vibration": _sensor("imu_vibration", 2.0, 10.0, signed=False),
    "mag_interference": _sensor("mag_interference", 0.1, 0.5, axes=3),
    "baro_drift": _sensor("baro_drift", 5.0, 30.0),
    "gnss_dropout": _gnss_dropout,
    "gnss_drift": _sensor("gnss_drift", 0.5, 3.0, axes=2),
    "gnss_spoof": _sensor("gnss_spoof", 0.05, 0.4, axes=2),
    "hard_over": _actuator("hard_over", sign=True),
    "floating": _actuator("floating"),
    "slow": _actuator("slow", 0.05, 0.5),
    "reduced_throw": _actuator("reduced_throw", 0.2, 0.7),
    "backlash": _actuator("backlash", 1.0, 5.0),
    "reversed": _actuator("reversed"),
    "partial_loss": _scalar("propulsion", "partial_loss", 0.2, 0.7),
    "prop_damage": _scalar("propulsion", "prop_damage", 0.1, 0.5),
    "thermal_derate": _scalar(
        "propulsion", "thermal_derate", 0.3, 0.8, ramp_s=lambda r: _u(r, 5.0, 30.0)
    ),
    "cell_failure": lambda rng, c: {"type": "energy_fault", "kind": "cell_failure"},
    "abnormal_sag": _scalar("energy", "abnormal_sag", 2.0, 8.0),
    "capacity_loss": _scalar("energy", "capacity_loss", 0.3, 0.8),
    "cg_shift": _cg_shift,
    "payload_drop": _scalar("airframe", "payload_drop", 0.1, 0.4),
    "wing_damage": _scalar(
        "airframe", "wing_damage", 0.05, 0.25, side=lambda r: int(r.choice([-1, 1]))
    ),
}


_M1_CATEGORY = {"pitot_blockage": "sensor", "servo_jam": "actuator", "motor_cut": "propulsion"}


def fault_category(spec: dict[str, Any]) -> str:
    """Family for the result tables: sensor, actuator, propulsion, energy or airframe."""
    t = str(spec["type"])
    return _M1_CATEGORY.get(t, t.removesuffix("_fault"))


def fault_kind(spec: dict[str, Any]) -> str:
    return str(spec.get("kind", spec["type"]))


# --------------------------------------------------------------------------- scenarios
def bench_airframes(spec: BenchSpec) -> list[str]:
    """The sampler's pool, sorted: airframes of the split that passed V20 (a sample bundle
    without V20 results carries only accepted airframes' stacks)."""
    stacks = REPO / spec.stacks
    if (stacks / "results.parquet").exists():
        rows = pq.read_table(stacks / "results.parquet").to_pylist()
        ids = [r["airframe"] for r in rows if r["passed"]]
    else:
        lines = (stacks / "stack_cfgs.jsonl").read_text().splitlines()
        ids = [json.loads(line)["airframe"] for line in lines]
    return sorted(i for i in ids if spec.split == "all" or split_of(i) == spec.split)


def _mission(rng: Rng, m: MissionRanges, v: float) -> tuple[list[list[float]], list[float], float]:
    n = int(rng.integers(m.waypoints[0], m.waypoints[1] + 1))
    perimeter = v * _u(rng, *m.lap_s)
    radius = perimeter / (2 * math.pi)
    # Legs at least four turn radii (45° bank) long, so every corner can be flown.
    turn = v * v / G0
    radius = max(radius, 4 * turn / (2 * math.sin(math.pi / n)))
    centre = rng.uniform(-m.centre_offset_m, m.centre_offset_m, 2)
    start = _u(rng, 0, 2 * math.pi)
    jitter = rng.uniform(-0.25, 0.25, n) * (2 * math.pi / n)
    ang = start + np.arange(n) * 2 * math.pi / n + jitter
    wps = [
        [round(float(centre[0] + radius * math.cos(a)), 1),
         round(float(centre[1] + radius * math.sin(a)), 1),
         round(_u(rng, *m.alt_agl_m), 1)]
        for a in ang
    ]  # fmt: skip
    # Start on the loop, half way along the last leg, heading for the first waypoint.
    a, b = np.array(wps[-1][:2]), np.array(wps[0][:2])
    pos = (a + b) / 2
    heading = math.degrees(math.atan2(b[1] - pos[1], b[0] - pos[0])) % 360
    return wps, [round(float(pos[0]), 1), round(float(pos[1]), 1)], heading


def _weather(rng: Rng, w: WeatherRanges, v: float, duration: float, centre: np.ndarray, r: float):
    env: dict[str, Any] = {
        "wind": {
            "speed_mps": round(v * _u(rng, *w.wind_frac), 2),
            "from_deg": round(_u(rng, 0, 360), 1),
        },
        "turbulence": {"model": "dryden_low_alt", "intensity": "none"},
    }
    if rng.random() >= w.smooth_air_p:
        speed = env["wind"]["speed_mps"]
        levels = {k: kt * 0.514444 for k, kt in TURBULENCE_W20_KT.items() if k != "none"}
        env["turbulence"]["intensity"] = min(levels, key=lambda k: abs(levels[k] - speed))
    events: dict[str, Any] = {}
    if rng.random() < w.gust_p:
        events["gust"] = {
            "at_time_s": round(_u(rng, 30, duration - 30), 1),
            "duration_s": round(_u(rng, 2, 8), 1),
            "speed_mps": round(v * _u(rng, 0.1, 0.4), 2),
            "from_deg": round(_u(rng, 0, 360), 1),
        }
    if rng.random() < w.shear_p:
        events["shear"] = {
            "height_m": round(_u(rng, 30, 100), 1),
            "speed_change_mps": round(v * _u(rng, -0.3, 0.3), 2),
            "from_deg": round(_u(rng, 0, 360), 1),
        }
    if rng.random() < w.thermal_p:
        events["thermals"] = [
            {
                "centre_ne_m": [round(float(c), 1) for c in centre + rng.uniform(-r, r, 2)],
                "radius_m": round(_u(rng, 40, 120), 1),
                "w_peak_mps": round(_u(rng, 1.5, 4.0) * (1 if rng.random() < 0.8 else -1), 2),
            }
            for _ in range(int(rng.integers(1, 3)))
        ]
    if events:
        env["events"] = events
    return env


def _faults(rng: Rng, f: FaultMix, chord: float) -> list[dict[str, Any]]:
    mode = rng.choice(["none", "single", "double"], p=[f.none, f.single, f.double])
    count = {"none": 0, "single": 1, "double": 2}[str(mode)]
    kinds = f.kinds or list(FAULT_KINDS)
    for _ in range(100):  # resample combinations the slots cannot hold
        chosen = rng.choice(kinds, size=count, replace=False) if count else []
        specs = []
        for k in chosen:
            onset = round(_u(rng, *f.onset_s), 2)
            specs.append({**FAULT_KINDS[str(k)](rng, chord), "trigger": {"at_time_s": onset}})
        try:
            build_fault_config([_FAULT.validate_python(s) for s in specs])
            return specs
        except ValueError:
            continue
    raise RuntimeError("could not draw a valid fault combination")


def sample_scenarios(spec: BenchSpec, cruise: dict[str, float], chord: dict[str, float]):
    """The bench's scenarios (as validated ``Scenario``s), in order."""
    pool = bench_airframes(spec)
    if not pool:
        raise ValueError(f"no V20-accepted {spec.split} airframes in {spec.stacks}")
    rng = np.random.default_rng(spec.seed)
    out = []
    for i in range(spec.n):
        af = str(rng.choice(pool))
        v = cruise[af]
        wps, start, heading = _mission(rng, spec.mission, v)
        centre = np.mean(np.array(wps)[:, :2], axis=0)
        radius = float(np.linalg.norm(np.array(wps[0][:2]) - centre))
        env = _weather(rng, spec.weather, v, spec.duration_s, centre, radius)
        faults = _faults(rng, spec.faults, chord[af])
        out.append(
            Scenario.model_validate(
                {
                    "schema_version": 1,
                    "id": f"{spec.id}-{i:04d}",
                    "description": f"bench {spec.id}, scenario {i}",
                    "site": spec.site,
                    "airframe": af,
                    "fleet": spec.fleet,
                    "seed": int(rng.integers(2**31)),
                    "duration_s": spec.duration_s,
                    "initial": {"pos_ne_m": start, "alt_agl_m": wps[0][2],
                                "heading_deg": round(heading, 1), "airspeed_mps": v},
                    "mission": {"alt_agl_m": wps[0][2], "waypoints_ne_m": wps,
                                "on_complete": "loop", "speed_mps": v},
                    "environment": env,
                    "faults": faults,
                    "stack": {"contingency": "classical_v0", "params": spec.stacks},
                }
            )
        )  # fmt: skip
    return out


def write_scenarios(scenarios: list[Scenario], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in scenarios:
        data = json.loads(s.model_dump_json(exclude_defaults=True))
        (out_dir / f"{s.id}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
