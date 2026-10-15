"""A live flight (M2 spec §9): the same compiled closed loop as replays and campaigns, stepped
in short chunks so that faults can be injected while it flies.

Each chunk yields exactly the replay columns (``telemetry.schema.COLUMNS``), so Skyview has
one code path for replays and live sessions, and a session is recorded as an ordinary replay
bundle. Injected faults are appended to the scenario's fault list with their onset at the
next control step and recorded in the manifest (``live_commands``).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import numpy as np
from pydantic import TypeAdapter

from aeroadapt.airframes import Airframe
from aeroadapt.faultlab import build_fault_config
from aeroadapt.faultlab.schema import FaultSpec
from aeroadapt.harness.closed_loop import closed_loop_carry, setup_loop
from aeroadapt.scenarios.resolve import ResolvedScenario, resolve
from aeroadapt.scenarios.runner import build_manifest
from aeroadapt.scenarios.schema import Scenario
from aeroadapt.stack.config import StackCfg, build_stack_params
from aeroadapt.telemetry.events import extract_events
from aeroadapt.telemetry.recorder import write_bundle
from aeroadapt.telemetry.schema import COLUMNS, VIZ_RATE_HZ, columns

CHUNK_S = 0.1
_FAULT = TypeAdapter(FaultSpec)


class LiveSession:
    def __init__(
        self,
        scenario: Scenario,
        airframe: Airframe | None = None,
        stack_cfg: StackCfg | None = None,
    ) -> None:
        r = resolve(scenario)
        if airframe is not None:
            cfg = stack_cfg or r.stack_cfg
            r = replace(
                r,
                airframe=airframe,
                stack_cfg=cfg,
                stack=build_stack_params(cfg, airframe, r.site, r.config.dt_control),
            )
        self.resolved: ResolvedScenario = r
        sc = r.scenario
        su = setup_loop(r.airframe, r.site, r.env, r.initial, faults=r.faults, stack=r.stack,
                        seed=sc.seed, config=r.config)  # fmt: skip
        self._su = su
        self._sim, self._st, self._z = su.sim0, su.stack0, su.z0
        self._faults = su.faults
        self._key = jax.random.fold_in(jax.random.PRNGKey(sc.seed), 1)
        self._n = round(CHUNK_S / r.config.dt_control)
        self._viz_every = max(round(r.config.control_hz / VIZ_RATE_HZ), 1)
        self._rows = 0
        self.specs: list[Any] = list(sc.faults)
        self.commands: list[dict[str, Any]] = []
        self.chunks: list[dict[str, np.ndarray]] = []
        self._last_event_us = -1

    # ------------------------------------------------------------------ stepping
    @property
    def time_s(self) -> float:
        return float(self._sim.t)

    @property
    def done(self) -> bool:
        return bool(self._sim.done)

    def step(self) -> dict[str, np.ndarray]:
        """Fly one chunk; returns its 100 Hz replay columns."""
        r, su = self.resolved, self._su
        self._key, k = jax.random.split(self._key)
        self._sim, self._st, self._z, log = closed_loop_carry(
            su.vp, su.env, self._faults, su.sp, r.mission, self._sim, self._st, self._z, k,
            n_steps=self._n, config=r.config,
        )  # fmt: skip
        cols = columns(jax.tree.map(np.asarray, log))
        self.chunks.append(cols)
        return cols

    def viz(self, cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The browser subset of a chunk at 50 Hz (decimated on the global row index)."""
        n = len(cols["time_us"])
        keep = (self._rows + np.arange(n)) % self._viz_every == 0
        self._rows += n
        out: dict[str, np.ndarray] = {"t": (cols["time_us"][keep] * 1e-6).astype(np.float32)}
        for c in COLUMNS:
            if c.viz:
                x = cols[c.name][keep]
                out[c.name] = x.astype(np.int32) if c.integer else x.astype(np.float32)
        return out

    def new_events(self) -> list[dict[str, Any]]:
        """Events in the latest chunks not sent before (edges need the previous chunk)."""
        window = self.chunks[-2:]
        cols = {k: np.concatenate([c[k] for c in window]) for k in window[0]}
        # Anything at the window's first sample was sent with the previous chunk (or describes
        # the state at the start, not a change).
        first = int(cols["time_us"][0])
        events = [
            e
            for e in extract_events(cols, self.resolved.site)
            if e["type"] not in ("start", "end") and e["time_us"] > max(first, self._last_event_us)
        ]
        if events:
            self._last_event_us = max(e["time_us"] for e in events)
        return events

    # ------------------------------------------------------------------ commands
    def inject(self, spec: dict[str, Any]) -> str | None:
        """Add a fault starting at the next control step; returns an error message, if any."""
        onset = round(self.time_s + self.resolved.config.dt_control, 6)
        try:
            fault = _FAULT.validate_python({**spec, "trigger": {"at_time_s": onset}})
            config = build_fault_config([*self.specs, fault])
        except ValueError as e:
            return str(e).splitlines()[0]
        self.specs.append(fault)
        self._faults = jax.device_put(config)
        self.commands.append({"time_s": onset, "inject_fault": fault.model_dump(mode="json")})
        return None

    # ------------------------------------------------------------------ recording
    def manifest(self) -> dict[str, Any]:
        cols = self.all_columns() if self.chunks else {"time_us": np.zeros(1, np.int64)}
        m = build_manifest(self.resolved, cols)
        m["scenario"]["faults"] = [f.model_dump(mode="json") for f in self.specs]
        m["live_commands"] = self.commands
        return m

    def all_columns(self) -> dict[str, np.ndarray]:
        return {k: np.concatenate([c[k] for c in self.chunks]) for k in self.chunks[0]}

    def record(self, out_dir: Path) -> Path:
        cols = self.all_columns()
        return write_bundle(
            out_dir, self.manifest(), cols, extract_events(cols, self.resolved.site)
        )


def replay_commands(scenario: Scenario, commands: list[dict[str, Any]]) -> LiveSession:
    """Re-fly a live session offline from its command log (``manifest["live_commands"]``):
    same chunking, same keys, faults injected at the same steps, so the result is identical."""
    session = LiveSession(scenario)
    pending = sorted(commands, key=lambda c: c["time_s"])
    dt = session.resolved.config.dt_control
    end = max((c["time_s"] for c in pending), default=0.0)
    while pending or session.time_s < end:
        while pending and pending[0]["time_s"] <= session.time_s + dt + 1e-6:
            spec = dict(pending.pop(0)["inject_fault"])
            spec.pop("trigger", None)
            if (error := session.inject(spec)) is not None:
                raise ValueError(error)
        session.step()
    return session
