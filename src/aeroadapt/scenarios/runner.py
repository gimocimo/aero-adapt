"""Scenario runner (spec §10.2): scenario → closed-loop flight → replay bundle."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from pydantic import BaseModel

from aeroadapt.airframes import AIRFRAME_DIR
from aeroadapt.harness.closed_loop import LoopLog, setup_loop
from aeroadapt.scenarios.resolve import ResolvedScenario, resolve
from aeroadapt.scenarios.schema import Scenario
from aeroadapt.stack.autopilot import MODE_NAMES
from aeroadapt.stack.config import STACK_DIR
from aeroadapt.stack.contingency import PHASE_NAMES
from aeroadapt.telemetry.events import extract_events
from aeroadapt.telemetry.recorder import write_bundle
from aeroadapt.telemetry.schema import (
    BUNDLE_SCHEMA_VERSION,
    COLUMNS,
    FAULT_BITS,
    VIZ_RATE_HZ,
    columns,
    units,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TAIL_S = 1.0  # rows kept after termination (the state is frozen there)


@dataclass(frozen=True)
class RunResult:
    resolved: ResolvedScenario
    cols: dict[str, np.ndarray]
    events: list[dict[str, Any]]
    manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_sha(path: Path, model: BaseModel) -> str:
    """The config file's hash, or the model's JSON hash when it has no file (generated
    airframes, auto-configured stacks)."""
    if path.exists():
        return _sha256(path)
    return hashlib.sha256(model.model_dump_json().encode()).hexdigest()


def git_info() -> dict[str, Any]:
    """Commit and dirty flag of the working tree (provenance; empty outside a checkout)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {"commit": commit, "dirty": dirty}


def build_manifest(r: ResolvedScenario, cols: dict[str, np.ndarray]) -> dict[str, Any]:
    sc, site = r.scenario, r.site
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "scenario": sc.model_dump(mode="json"),
        "site": {
            "id": site.id,
            "name": site.name,
            "origin": site.origin.model_dump(mode="json"),
            "world": site.world,
            "safe_zones": [z.model_dump(mode="json") for z in site.safe_zones],
        },
        "mission": {
            "waypoints_ne_alt_m": [list(w) for w in sc.mission.waypoints_3d()],
            "on_complete": sc.mission.on_complete,
        },
        "provenance": {
            "airframe": {
                "id": r.airframe.id,
                "sha256": _config_sha(AIRFRAME_DIR / f"{r.airframe.id}.yaml", r.airframe),
            },
            "stack": {
                "id": r.stack_cfg.id,
                "sha256": _config_sha(
                    STACK_DIR / f"{sc.stack.params or sc.airframe}.yaml", r.stack_cfg
                ),
                "config": r.stack_cfg.model_dump(mode="json"),
            },
            "git": git_info(),
        },
        "seed": sc.seed,
        "rates": {
            "control_hz": r.config.control_hz,
            "physics_hz": 1.0 / r.config.dt_physics,
            "viz_hz": VIZ_RATE_HZ,
        },
        "n_frames": len(cols["time_us"]),
        "duration_s": float(cols["time_us"][-1]) * 1e-6,
        "units": units(),
        "viz_columns": ["t", *(c.name for c in COLUMNS if c.viz)],
        "enums": {"mode": list(MODE_NAMES), "phase": list(PHASE_NAMES), "fault_bits": FAULT_BITS},
    }


def _trim(log: LoopLog, control_hz: float) -> LoopLog:
    done = np.asarray(log.done)
    if not done.any():
        return log
    end = min(int(np.argmax(done)) + round(TAIL_S * control_hz) + 1, len(done))
    return jax.tree.map(lambda x: np.asarray(x)[:end], log)


def run_scenario(scenario: Scenario | ResolvedScenario) -> RunResult:
    r = scenario if isinstance(scenario, ResolvedScenario) else resolve(scenario)
    sc = r.scenario
    loop = setup_loop(
        r.airframe,
        r.site,
        r.env,
        r.initial,
        faults=r.faults,
        stack=r.stack,
        seed=sc.seed,
        config=r.config,
    )
    # The run key is derived from the scenario seed, independent of the initial-state key.
    key = jax.random.fold_in(jax.random.PRNGKey(sc.seed), 1)
    _, _, log = loop.run(r.mission, r.n_steps, key)
    log = _trim(jax.tree.map(np.asarray, log), r.config.control_hz)
    cols = columns(log)
    events = extract_events(cols, r.site)
    return RunResult(r, cols, events, build_manifest(r, cols))


def record(result: RunResult, out_dir: Path) -> Path:
    return write_bundle(out_dir, result.manifest, result.cols, result.events)
