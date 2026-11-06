"""Scenarios flown through the host-stepped ``Vehicle`` interface, recorded as replay bundles.

The scenario runner flies aerojax inside one compiled loop. A backend that steps on the host
(JSBSim) exchanges ``SensorFrame`` and ``ActuatorCommand`` messages with the stack instead,
one control step at a time. This flies a resolved scenario that way, with the same stack
core, start (the wind estimate warm, as in every harness flight) and logging as
``closed_loop``, so the bundle has the same columns, events and manifest as any other and
Skyview replays it.

What a host backend does not expose is logged as the aerojax values for open, flat ground:
height above the ground and clearance are the height above the site's ground.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from aeroadapt.core import quaternion
from aeroadapt.core.messages import ActuatorCommand, arrays_from_frame
from aeroadapt.faultlab import FaultLabels
from aeroadapt.harness.closed_loop import LoopLog, warm_wind
from aeroadapt.scenarios.resolve import ResolvedScenario
from aeroadapt.scenarios.runner import TAIL_S, RunResult, build_manifest
from aeroadapt.stack.autopilot import MODE_AUTO, stack_init, stack_step
from aeroadapt.stack.config import StackParams
from aeroadapt.stack.types import Mission
from aeroadapt.telemetry.events import extract_events
from aeroadapt.telemetry.schema import columns
from aeroadapt.vehicle.protocol import PrivilegedVehicle

_STACK_FIELDS = (
    "cmd", "est_euler", "est_pos_ned", "est_vel_ned", "est_ias", "est_airspeed",
    "est_wind_ned", "airspeed_healthy", "roll_dem", "pitch_dem", "alt_target", "speed_target",
    "alt_dem", "speed_dem", "lat_acc", "xtrack", "mode", "wp_idx", "failed", "phase", "zone",
    "zone_margin", "speed_to_fly",
)  # fmt: skip


def _stack_step(sp: StackParams, st: Any, z: Any, mission: Mission) -> tuple[Any, dict]:
    st, out = stack_step(sp, st, z, mission, None)
    e, c = out.est, out.cont
    row = {
        "cmd": out.cmd,
        "est_euler": jnp.stack([e.roll, e.pitch, e.yaw]),
        "est_pos_ned": e.pos_ned,
        "est_vel_ned": e.vel_ned,
        "est_ias": e.ias,
        "est_airspeed": e.airspeed,
        "est_wind_ned": e.wind_ned,
        "airspeed_healthy": e.airspeed_healthy,
        "roll_dem": out.roll_dem,
        "pitch_dem": out.pitch_dem,
        "alt_target": out.target.alt,
        "speed_target": out.target.speed,
        "alt_dem": st.tecs.hgt_dem,
        "speed_dem": st.tecs.spd_dem,
        "lat_acc": out.lat_acc,
        "xtrack": out.xtrack,
        "mode": out.mode,
        "wp_idx": out.wp_idx,
        "failed": c.active,
        "phase": c.phase,
        "zone": c.zone,
        "zone_margin": c.margin,
        "speed_to_fly": c.speed_to_fly,
    }
    return st, row


_step = jax.jit(_stack_step)


def _device(z: Any) -> Any:
    """Host sensor arrays (float64) → device arrays in JAX's default precision, as the stack
    core takes them."""
    return jax.tree.map(jnp.asarray, z)


def fly_vehicle(r: ResolvedScenario, vehicle: PrivilegedVehicle) -> RunResult:
    """Fly ``r`` on ``vehicle`` (built for the same airframe, site, environment and faults)."""
    sc = r.scenario
    origin = vehicle.info.origin
    frame = vehicle.reset(r.initial, seed=sc.seed)
    z = _device(arrays_from_frame(frame, origin))
    st = warm_wind(stack_init(r.stack, z, MODE_AUTO), r.env, r.initial)
    mission = jax.device_put(r.mission)
    rows: list[dict[str, Any]] = []
    tail = round(TAIL_S * r.config.control_hz)
    end = r.n_steps
    for k in range(r.n_steps):
        st, out = _step(r.stack, st, z, mission)
        out = jax.device_get(out)
        res = vehicle.step(ActuatorCommand.from_array(frame.time_us, np.asarray(out["cmd"])))
        truth = vehicle.truth()
        rows.append({"truth": truth, "sensors": z, **out, "done": res.terminated,
                     "termination": int(res.termination)})  # fmt: skip
        if res.terminated and end == r.n_steps:
            end = min(k + 1 + tail, r.n_steps)
        if k + 1 >= end:
            break
        frame = res.sensors
        z = _device(arrays_from_frame(frame, origin))
    log = _log(rows)
    cols = columns(log)
    return RunResult(r, cols, extract_events(cols, r.site), build_manifest(r, cols))


def _log(rows: list[dict[str, Any]]) -> LoopLog:
    def stack(get: Callable[[dict[str, Any]], Any]) -> np.ndarray:
        return np.stack([np.asarray(get(row)) for row in rows])

    def truth(name: str) -> np.ndarray:
        return stack(lambda r: getattr(r["truth"], name))

    euler = truth("euler_rad")
    agl = truth("alt_agl_m")
    per_label: dict[str, Any] = {
        k: stack(lambda r, k=k: r["truth"].fault_labels[k]) for k in FaultLabels._fields
    }
    labels = FaultLabels(**per_label)
    fields: dict[str, Any] = {
        "t": truth("time_s"),
        "pos_ned": truth("pos_ned_m"),
        "vel_ned": truth("vel_ned_mps"),
        "quat": np.stack([np.asarray(quaternion.from_euler(*e)) for e in euler]),
        "euler": euler,
        "omega": truth("omega_rps"),
        "hagl": agl,
        "clearance": agl,
        "va": truth("airspeed_mps"),
        "alpha": truth("alpha_rad"),
        "beta": truth("beta_rad"),
        "wind_ned": truth("wind_ned_mps"),
        "surfaces": truth("surfaces_rad"),
        "throttle_actual": truth("throttle"),
        "thrust": truth("thrust_n"),
        "battery_v": truth("battery_v"),
        "soc": truth("soc"),
        "prop_rpm": truth("prop_rpm"),
        "labels": labels,
        "motor_cut": labels.motor_cut,
        "done": stack(lambda r: r["done"]),
        "termination": stack(lambda r: r["termination"]),
        "sensors": jax.tree.map(
            lambda *x: np.stack([np.asarray(v) for v in x]), *[r["sensors"] for r in rows]
        ),
        **{f: stack(lambda r, f=f: r[f]) for f in _STACK_FIELDS},
    }
    return LoopLog(**fields)
