"""``AerojaxVehicle``: the JAX simulator behind the ``Vehicle`` interface (imperative shell).

It converts between the simulator's local-NED arrays and WGS84 :class:`SensorFrame` messages
in float64 (ADR-001). It also closes over the static parameters so that each host-stepped
call only moves the state.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from aeroadapt.airframes import Airframe, to_vehicle_params
from aeroadapt.core import quaternion
from aeroadapt.core.messages import ActuatorCommand, SensorFrame, frame_from_arrays
from aeroadapt.faultlab import FaultConfig, fault_labels, no_faults
from aeroadapt.scenarios import Site
from aeroadapt.sim.aerojax import EnvParams, RigidBodyState, SimConfig, init_state, sim_step
from aeroadapt.sim.aerojax.actuators import target_to_command
from aeroadapt.sim.aerojax.atmosphere import isa
from aeroadapt.sim.aerojax.battery import open_circuit_voltage
from aeroadapt.sim.aerojax.environment import Environment
from aeroadapt.sim.aerojax.sim import SimState, StepInfo
from aeroadapt.sim.aerojax.trim import trim
from aeroadapt.vehicle.protocol import (
    ImpactRecord,
    InitialCondition,
    StepResult,
    Termination,
    TruthSnapshot,
    VehicleInfo,
    classify_touchdown,
)


def trimmed_start(
    airframe: Airframe, env: EnvParams, initial: InitialCondition
) -> tuple[RigidBodyState, np.ndarray]:
    """Straight-and-level trim at the requested airspeed and height. Returns the state
    (rotated to the requested heading and placed at the requested position) and the
    normalised command that holds it."""
    vp = to_vehicle_params(airframe)
    _, _, rho = isa(env.isa, env.isa.alt_origin_msl_m + initial.alt_agl_m)
    v_supply = float(open_circuit_voltage(vp.battery, jnp.asarray(initial.soc)))
    tr = trim(
        vp.airframe,
        Environment(float(rho), env.gravity),
        initial.airspeed_mps,
        h=initial.alt_agl_m,
        v_supply=v_supply,
    )
    if not tr.success:
        raise ValueError(f"no trim for {initial}")
    q = quaternion.from_euler(tr.phi, tr.theta, math.radians(initial.heading_deg))
    state = RigidBodyState(
        pos_ned=jnp.asarray([*initial.pos_ne_m, -initial.alt_agl_m]),
        vel_body=jnp.asarray(tr.state.vel_body),
        quat=jnp.asarray(q),
        omega=jnp.zeros(3),
    )
    cmd = np.asarray(target_to_command(vp.airframe, vp.actuators, jnp.asarray(tr.controls)))
    return state, cmd


class AerojaxVehicle:
    def __init__(
        self,
        airframe: Airframe,
        site: Site,
        env: EnvParams,
        faults: FaultConfig | None = None,
        config: SimConfig = SimConfig(),  # noqa: B008
    ) -> None:
        self._airframe = airframe
        self._vp = to_vehicle_params(airframe)
        self._env = env
        self._faults = no_faults() if faults is None else faults
        self._config = config
        self._origin = site.origin.geodetic()
        self.info = VehicleInfo(config.control_hz, airframe.id, site.id, self._origin)
        # Put the parameters on the device once. Closing over host NumPy arrays makes JAX
        # transfer them on every call (≈20x slower host-stepping).
        vp, env, faults_cfg = jax.device_put((self._vp, env, self._faults))

        def _step(s: SimState, cmd: jax.Array, key: jax.Array):
            k, key = jax.random.split(key)
            s, sensors, info = sim_step(vp, env, faults_cfg, s, cmd, k, config)
            return s, sensors, info, key

        self._step = jax.jit(_step)
        self._state: SimState | None = None
        self._info: StepInfo | None = None
        self._key = jax.random.PRNGKey(0)
        self.trim_command: np.ndarray | None = None

    # ------------------------------------------------------------------ Vehicle
    def reset(self, initial: InitialCondition, seed: int) -> SensorFrame:
        rb0, cmd0 = trimmed_start(self._airframe, self._env, initial)
        self.trim_command = cmd0
        k0, self._key = jax.random.split(jax.random.PRNGKey(seed))
        self._state, sensors = init_state(
            self._vp, self._env, self._faults, rb0, jnp.asarray(cmd0), k0, initial.soc, self._config
        )
        self._info = None
        return frame_from_arrays(jax.device_get(sensors), self._time_us(), self._origin)

    def step(self, cmd: ActuatorCommand) -> StepResult:
        if self._state is None:
            raise RuntimeError("reset() first")
        state, sensors, self._info, self._key = self._step(self._state, cmd.as_array(), self._key)
        self._state = state
        # One device→host transfer for everything the frame needs (keeps the shell fast).
        z, t, done, code = jax.device_get((sensors, state.t, state.done, state.termination))
        frame = frame_from_arrays(z, round(float(t) * 1e6), self._origin)
        return StepResult(frame, bool(done), Termination(int(code)))

    # ------------------------------------------------------------------ privileged
    @property
    def state(self) -> SimState:
        assert self._state is not None
        return self._state

    def truth(self) -> TruthSnapshot:
        s = self.state
        rb = s.rb
        euler = np.array([float(x) for x in quaternion.to_euler(rb.quat)])
        vel_ned = np.asarray(quaternion.body_to_ned(rb.quat, rb.vel_body))
        labels = fault_labels(self._faults, s.t)
        t = self._info.truth if self._info is not None else None
        return TruthSnapshot(
            time_s=float(s.t),
            pos_ned_m=np.asarray(rb.pos_ned),
            vel_ned_mps=vel_ned,
            euler_rad=euler,
            omega_rps=np.asarray(rb.omega),
            airspeed_mps=float(t.va) if t is not None else float("nan"),
            alpha_rad=float(t.alpha) if t is not None else float("nan"),
            beta_rad=float(t.beta) if t is not None else float("nan"),
            alt_agl_m=float(-rb.pos_ned[2]),
            wind_ned_mps=np.asarray(t.wind_ned) if t is not None else np.zeros(3),
            surfaces_rad=np.asarray(s.act.surfaces),
            throttle=float(s.act.throttle),
            thrust_n=float(t.thrust) if t is not None else float("nan"),
            battery_v=float(t.battery_v) if t is not None else float("nan"),
            soc=float(s.battery.soc),
            fault_labels={k: np.asarray(v) for k, v in labels._asdict().items()},
        )

    def impact(self) -> ImpactRecord | None:
        s = self.state
        if int(s.termination) != Termination.GROUND_CONTACT:
            return None
        rb = s.rb
        vel = np.asarray(quaternion.body_to_ned(rb.quat, rb.vel_body))
        roll, pitch, _ = (float(x) for x in quaternion.to_euler(rb.quat))
        lat, lon, _ = self._origin.ned_to_geodetic(np.asarray(rb.pos_ned, dtype=np.float64))
        sink = float(vel[2])
        return ImpactRecord(
            time_s=float(s.t),
            pos_ned_m=np.asarray(rb.pos_ned),
            lat_deg=float(lat),
            lon_deg=float(lon),
            vel_ned_mps=vel,
            sink_rate_mps=sink,
            groundspeed_mps=float(np.hypot(vel[0], vel[1])),
            roll_rad=roll,
            pitch_rad=pitch,
            kinetic_energy_j=0.5 * self._vp.airframe.mass * float(vel @ vel),
            touchdown_class=classify_touchdown(sink, roll, pitch),
        )

    def _time_us(self) -> int:
        return round(float(self.state.t) * 1e6)
