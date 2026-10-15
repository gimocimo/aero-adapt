"""``FlightStack``: the imperative shell around the pure stack core (spec §5.3).

It converts each WGS84 :class:`SensorFrame` into local-NED arrays (float64 geodesy, ADR-001),
runs the jitted ``stack_step`` and returns an :class:`ActuatorCommand` plus telemetry. It
consumes only the messages a real autopilot would get: this is the path a vehicle
backend that is not JAX (JSBSim, ArduPilot SITL) will drive in M2+.

Host↔device traffic is one packed vector in and one out per step: per-leaf transfers of
the sensor and telemetry pytrees cost more than the stack itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from aeroadapt.core.geodesy import GeodeticOrigin
from aeroadapt.core.messages import (
    ActuatorCommand,
    FreshFlags,
    SensorArrays,
    SensorFrame,
    arrays_from_frame,
)
from aeroadapt.stack.autopilot import MODE_AUTO, MODE_NAMES, StackState, stack_init, stack_step
from aeroadapt.stack.config import StackParams
from aeroadapt.stack.contingency import PHASE_NAMES
from aeroadapt.stack.types import Mission

_FLAGS = FreshFlags._fields


def pack_sensors(z: SensorArrays) -> np.ndarray:
    """Host-side: sensor arrays → one flat float64 vector (fixed order)."""
    return np.concatenate([
        z.gyro, z.accel,
        [z.baro_pa, z.baro_temp_k, z.airspeed_dp_pa, z.airspeed_temp_k],
        z.gnss_pos_ned, z.gnss_vel_ned, [z.gnss_time_s], z.mag,
        [z.battery_v, z.battery_a, z.battery_mah, z.esc_rpm],
        [float(getattr(z.fresh, f)) for f in _FLAGS],
    ]).astype(np.float64)  # fmt: skip


def unpack_sensors(v: Array) -> SensorArrays:
    """Device-side inverse of :func:`pack_sensors`."""
    return SensorArrays(
        gyro=v[0:3], accel=v[3:6], baro_pa=v[6], baro_temp_k=v[7], airspeed_dp_pa=v[8],
        airspeed_temp_k=v[9], gnss_pos_ned=v[10:13], gnss_vel_ned=v[13:16], gnss_time_s=v[16],
        mag=v[17:20], battery_v=v[20], battery_a=v[21], battery_mah=v[22], esc_rpm=v[23],
        fresh=FreshFlags(*(v[24 + i] > 0.5 for i in range(len(_FLAGS)))),
    )  # fmt: skip


@dataclass(frozen=True)
class StackTelemetry:
    """Host-side summary of one stack step."""

    time_us: int
    mode: str
    phase: str  # contingency phase (NOMINAL unless a failure is latched)
    failed: bool
    zone: int  # selected safe zone (-1 = none)
    wp_idx: int
    euler_rad: np.ndarray  # estimated roll, pitch, yaw
    pos_ned_m: np.ndarray  # estimated
    vel_ned_mps: np.ndarray
    alt_agl_m: float
    airspeed_mps: float
    airspeed_healthy: bool
    wind_ne_mps: np.ndarray
    roll_dem_rad: float
    pitch_dem_rad: float
    alt_target_m: float
    speed_target_mps: float


class FlightStack:
    def __init__(self, params: StackParams, origin: GeodeticOrigin) -> None:
        self._origin = origin
        # JAX's default float (float32 unless x64 is enabled), matching the parameters.
        dtype = jnp.zeros(()).dtype
        # Device-resident parameters (closing over host arrays re-transfers them every call).
        sp = jax.device_put(params)

        def init(v: Array, mode: int) -> StackState:
            return stack_init(sp, unpack_sensors(v.astype(dtype)), mode)

        def step(s: StackState, v: Array, mission: Mission) -> tuple[StackState, Array]:
            s, o = stack_step(sp, s, unpack_sensors(v.astype(dtype)), mission)
            e, c = o.est, o.cont
            tel = jnp.concatenate([
                o.cmd,
                jnp.stack([o.mode, o.wp_idx, c.phase, c.zone, c.active]).astype(o.cmd.dtype),
                jnp.stack([e.roll, e.pitch, e.yaw]), e.pos_ned, e.vel_ned,
                jnp.stack([e.alt_agl, e.airspeed, e.airspeed_healthy.astype(o.cmd.dtype)]),
                e.wind_ned[:2],
                jnp.stack([o.roll_dem, o.pitch_dem, o.target.alt, o.target.speed]),
            ])  # fmt: skip
            return s, tel

        self._init = jax.jit(init, static_argnums=1)
        self._step = jax.jit(step)
        self._state: StackState | None = None
        self._mission: Mission | None = None

    def _vector(self, frame: SensorFrame) -> np.ndarray:
        return pack_sensors(arrays_from_frame(frame, self._origin))

    def reset(self, frame: SensorFrame, mission: Mission, mode: int = MODE_AUTO) -> None:
        """Initialise (align) from the first sensor frame and load the mission."""
        self._mission = jax.device_put(mission)
        self._state = self._init(self._vector(frame), mode)

    def step(self, frame: SensorFrame) -> tuple[ActuatorCommand, StackTelemetry]:
        if self._state is None or self._mission is None:
            raise RuntimeError("reset() first")
        self._state, tel_dev = self._step(self._state, self._vector(frame), self._mission)
        t = np.asarray(jax.device_get(tel_dev), dtype=np.float64)
        cmd = ActuatorCommand.from_array(frame.time_us, t[0:4])
        return cmd, StackTelemetry(
            time_us=frame.time_us,
            mode=MODE_NAMES[int(t[4])],
            phase=PHASE_NAMES[int(t[6])],
            failed=bool(t[8] > 0.5),
            zone=int(t[7]),
            wp_idx=int(t[5]),
            euler_rad=t[9:12],
            pos_ned_m=t[12:15],
            vel_ned_mps=t[15:18],
            alt_agl_m=float(t[18]),
            airspeed_mps=float(t[19]),
            airspeed_healthy=bool(t[20] > 0.5),
            wind_ne_mps=t[21:23],
            roll_dem_rad=float(t[23]),
            pitch_dem_rad=float(t[24]),
            alt_target_m=float(t[25]),
            speed_target_mps=float(t[26]),
        )
