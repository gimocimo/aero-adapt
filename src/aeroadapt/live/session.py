"""A live flight (M2 spec §9): the same compiled closed loop as replays and campaigns, stepped
in short chunks so that faults can be injected while it flies.

Each chunk yields exactly the replay columns (``telemetry.schema.COLUMNS``), so Skyview has
one code path for replays and live sessions, and a session is recorded as an ordinary replay
bundle. Injected faults are appended to the scenario's fault list with their onset at the
next control step; a cleared sensor fault gets an ``until_s`` at the next control step. The
mean wind can be changed while flying (ramped over a few seconds). Every command is recorded
in the manifest (``live_commands``), from which the session re-flies offline.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import numpy as np
from pydantic import TypeAdapter, ValidationError

from aeroadapt.airframes import Airframe
from aeroadapt.core.frames import wind_from_to_ned
from aeroadapt.faultlab import build_fault_config
from aeroadapt.faultlab.schema import FaultSpec, SensorFault, TimeTrigger
from aeroadapt.faultlab.slots import N_SENSOR_SLOTS
from aeroadapt.harness.closed_loop import closed_loop_carry, setup_loop
from aeroadapt.scenarios.resolve import ResolvedScenario, resolve
from aeroadapt.scenarios.runner import build_manifest
from aeroadapt.scenarios.schema import Scenario
from aeroadapt.stack.config import StackCfg, build_stack_params
from aeroadapt.telemetry.events import extract_events
from aeroadapt.telemetry.recorder import write_bundle
from aeroadapt.telemetry.schema import COLUMNS, VIZ_RATE_HZ, columns

CHUNK_S = 0.1
WIND_RAMP_S = 5.0  # a wind change blends in over this time (a step would be a sharp gust)
MAX_WIND_MPS = 20.0
_FAULT = TypeAdapter(FaultSpec)


def _message(e: ValueError) -> str:
    if isinstance(e, ValidationError):
        err = e.errors()[0]
        where = ".".join(str(x) for x in err["loc"][1:])  # drop the union tag
        return f"{where}: {err['msg']}" if where else err["msg"]
    return str(e).splitlines()[0]


class LiveSession:
    def __init__(
        self,
        scenario: Scenario,
        airframe: Airframe | None = None,
        stack_cfg: StackCfg | None = None,
    ) -> None:
        r = resolve(scenario)
        if airframe is not None:
            # A fleet airframe flies the scenario with its own (tuned) configuration, from
            # trim at its own cruise speed, and flies the mission at that speed.
            cfg = stack_cfg or r.stack_cfg
            r = replace(
                r,
                airframe=airframe,
                stack_cfg=cfg,
                stack=build_stack_params(cfg, airframe, r.site, r.config.dt_control),
                initial=replace(r.initial, airspeed_mps=cfg.tecs.airspeed_cruise_mps),
                mission=r.mission._replace(
                    speed=jax.numpy.asarray(cfg.tecs.airspeed_cruise_mps, r.mission.speed.dtype)
                ),
            )
        self.resolved: ResolvedScenario = r
        sc = r.scenario
        su = setup_loop(r.airframe, r.site, r.env, r.initial, faults=r.faults, stack=r.stack,
                        seed=sc.seed, config=r.config)  # fmt: skip
        self._su = su
        self._env = su.env
        w0 = np.asarray(su.env.wind.steady_ned, dtype=np.float64)
        self._wind = (w0, w0, 0.0)  # (from, to, ramp start time)
        cfg = sc.environment.wind
        self._wind_set = {"speed_mps": float(cfg.speed_mps), "from_deg": float(cfg.from_deg)}
        self._sim, self._st, self._z = su.sim0, su.stack0, su.z0
        self._faults = su.faults
        self._key = jax.random.fold_in(jax.random.PRNGKey(sc.seed), 1)
        self._n = round(CHUNK_S / r.config.dt_control)
        self._viz_every = max(round(r.config.control_hz / VIZ_RATE_HZ), 1)
        self._rows = 0
        self.specs: list[Any] = list(sc.faults)
        # Indices into ``specs`` of the sensor faults, in slot order (slots are stable).
        self._sensor_slots = [i for i, f in enumerate(self.specs) if isinstance(f, SensorFault)]
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
        w_from, w_to, t0 = self._wind
        if not np.array_equal(w_from, w_to):  # ramping: this chunk flies the wind at its end
            u = min(max((self.time_s + CHUNK_S - t0) / WIND_RAMP_S, 0.0), 1.0)
            steady = np.asarray(w_from + u * (w_to - w_from), dtype=np.float32)
            wind = self._env.wind._replace(steady_ned=jax.numpy.asarray(steady))
            self._env = jax.device_put(self._env._replace(wind=wind))
            if u >= 1.0:
                self._wind = (w_to, w_to, t0)
        self._key, k = jax.random.split(self._key)
        self._sim, self._st, self._z, log = closed_loop_carry(
            su.vp, self._env, self._faults, su.sp, r.mission, self._sim, self._st, self._z, k,
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
    def _next_step(self) -> float:
        return round(self.time_s + self.resolved.config.dt_control, 6)

    def _apply(self, specs: list[Any], slots: list[int]) -> None:
        """Rebuild the fault configuration: every fault except sensor faults that have left
        their slot (they had cleared, and that part of the flight is flown)."""
        others = [f for f in specs if not isinstance(f, SensorFault)]
        config = build_fault_config([*others, *(specs[i] for i in slots)])
        self.specs, self._sensor_slots = specs, slots
        self._faults = jax.device_put(config)

    def _slots_with(self, fault: SensorFault, index: int) -> list[int]:
        """Sensor slots after adding a fault: a free slot, or one whose fault has cleared by
        the new onset."""
        slots = list(self._sensor_slots)
        if len(slots) < N_SENSOR_SLOTS:
            return [*slots, index]
        for j, i in enumerate(slots):
            until = self.specs[i].trigger.until_s
            if until is not None and until <= fault.trigger.at_time_s:
                slots[j] = index
                return slots
        raise ValueError(f"at most {N_SENSOR_SLOTS} sensor faults at once: clear one first")

    def inject(self, spec: dict[str, Any]) -> str | None:
        """Add a fault starting at the next control step; returns an error message, if any."""
        onset = self._next_step()
        try:
            fault = _FAULT.validate_python({**spec, "trigger": {"at_time_s": onset}})
            slots = self._sensor_slots
            if isinstance(fault, SensorFault):
                slots = self._slots_with(fault, len(self.specs))
            self._apply([*self.specs, fault], slots)
        except ValueError as e:
            return _message(e)
        self.commands.append({"time_s": onset, "inject_fault": fault.model_dump(mode="json")})
        return None

    def clear(self, index: int) -> str | None:
        """Clear an active sensor fault (by its index in ``specs``) at the next control step."""
        if not 0 <= index < len(self.specs):
            return f"no fault {index}"
        f = self.specs[index]
        if not isinstance(f, SensorFault):
            return f"{f.type} does not clear: physical damage persists"
        if f.trigger.until_s is not None:
            return "already cleared"
        # A fault cleared before it starts lasts one control step.
        dt = self.resolved.config.dt_control
        until = max(self._next_step(), round(f.trigger.at_time_s + dt, 6))
        trigger = TimeTrigger(at_time_s=f.trigger.at_time_s, until_s=until)
        specs = list(self.specs)
        specs[index] = f.model_copy(update={"trigger": trigger})
        self._apply(specs, self._sensor_slots)
        self.commands.append({"time_s": until, "clear_fault": index})
        return None

    def set_wind(self, speed_mps: float, from_deg: float) -> str | None:
        """Change the mean wind (aviation convention: the direction it blows from), blended in
        over ``WIND_RAMP_S`` from the next chunk."""
        if not 0.0 <= speed_mps <= MAX_WIND_MPS:
            return f"wind speed must be 0-{MAX_WIND_MPS:g} m/s"
        from_deg = from_deg % 360.0
        w_from, w_to, t0 = self._wind
        u = min(max((self.time_s - t0) / WIND_RAMP_S, 0.0), 1.0)
        current = w_from + u * (w_to - w_from)  # a change during a ramp starts from here
        target = np.asarray(wind_from_to_ned(from_deg, speed_mps), dtype=np.float64)
        self._wind = (current, target, self.time_s)
        self._wind_set = {"speed_mps": float(speed_mps), "from_deg": float(from_deg)}
        self.commands.append(
            {
                "time_s": self._next_step(),
                "set_wind": {"speed_mps": speed_mps, "from_deg": from_deg},
            }
        )
        return None

    @property
    def wind(self) -> dict[str, float]:
        """The wind set (reached after any ramp): speed and the direction it blows from (kept
        as set, so a calm keeps its direction)."""
        return dict(self._wind_set)

    def fault_list(self) -> list[dict[str, Any]]:
        """Every fault of the session, for the client (``index`` is what ``clear`` takes)."""
        return [
            {
                "index": i,
                "spec": f.model_dump(mode="json"),
                "clearable": isinstance(f, SensorFault) and f.trigger.until_s is None,
            }
            for i, f in enumerate(self.specs)
        ]

    # ------------------------------------------------------------------ recording
    def manifest(self) -> dict[str, Any]:
        cols = self.all_columns() if self.chunks else {"time_us": np.zeros(1, np.int64)}
        m = build_manifest(self.resolved, cols)
        m["scenario"]["faults"] = [f.model_dump(mode="json") for f in self.specs]
        m["scenario"]["airframe"] = self.resolved.airframe.id
        m["scenario"]["initial"]["airspeed_mps"] = self.resolved.initial.airspeed_mps
        m["live_commands"] = self.commands
        return m

    def all_columns(self) -> dict[str, np.ndarray]:
        return {k: np.concatenate([c[k] for c in self.chunks]) for k in self.chunks[0]}

    def record(self, out_dir: Path) -> Path:
        cols = self.all_columns()
        return write_bundle(
            out_dir, self.manifest(), cols, extract_events(cols, self.resolved.site)
        )


def replay_commands(
    scenario: Scenario,
    commands: list[dict[str, Any]],
    airframe: Airframe | None = None,
    stack_cfg: StackCfg | None = None,
) -> LiveSession:
    """Re-fly a live session offline from its command log (``manifest["live_commands"]``):
    same chunking, same keys, faults injected and cleared at the same steps, so the result is
    identical."""
    session = LiveSession(scenario, airframe, stack_cfg)
    pending = sorted(commands, key=lambda c: c["time_s"])
    dt = session.resolved.config.dt_control
    end = max((c["time_s"] for c in pending), default=0.0)
    while pending or session.time_s < end:
        while pending and pending[0]["time_s"] <= session.time_s + dt + 1e-6:
            c = pending.pop(0)
            if "clear_fault" in c:
                error = session.clear(int(c["clear_fault"]))
            elif "set_wind" in c:
                error = session.set_wind(**c["set_wind"])
            else:
                spec = dict(c["inject_fault"])
                spec.pop("trigger", None)
                error = session.inject(spec)
            if error is not None:
                raise ValueError(error)
        session.step()
    return session
