"""``JSBSimVehicle``: JSBSim behind the ``Vehicle`` interface (M2 spec §5).

JSBSim integrates the rigid body with its own equations of motion, integrators, WGS84 gravity,
rotating Earth, atmosphere, wind and Milspec turbulence. Everything that is not airframe
dynamics stays in our layer and reuses the aerojax models, so the two backends differ only in
the flight dynamics:

- actuators (delay, lag, rate limit, deadband) and their faults;
- motor, propeller and battery, applied to JSBSim as body-axis external reactions, with the
  propulsion and energy faults (a motor cut leaves only the windmilling-propeller drag);
- sensors and the sensor faults, sampled from JSBSim's truth.

faultlab v1 airframe faults (CG shift, payload drop, wing damage) change the rigid body
itself and are aerojax-only for now.

Heights: JSBSim is initialised with its altitude above sea level equal to our MSL altitude,
and the terrain at the site's ground MSL height. ISA quantities therefore line up with
aerojax, and the local frame's down axis is minus JSBSim's height above ground.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aeroadapt.airframes import Airframe, to_vehicle_params
from aeroadapt.core import quaternion
from aeroadapt.core.messages import ActuatorCommand, SensorFrame, frame_from_arrays
from aeroadapt.faultlab import FaultConfig, fault_labels, no_faults
from aeroadapt.scenarios import Site
from aeroadapt.sim.aerojax import EnvParams, SimConfig
from aeroadapt.sim.aerojax.actuators import (
    ActuatorState,
    actuator_step,
    init_actuators,
)
from aeroadapt.sim.aerojax.battery import (
    BatteryState,
    battery_step,
    init_battery,
    terminal_voltage,
)
from aeroadapt.sim.aerojax.propulsion import propeller, windmill_thrust
from aeroadapt.sim.aerojax.sensors import SensorState, SensorTruth, init_sensors, sample
from aeroadapt.sim.aerojax.sim import MOTOR_SPIN_DOWN_S, faulted_plant
from aeroadapt.sim.jsbsim.fdm import FT, JSBSIM_ROOT, SLUG_PER_FT3, JSBSimFDM
from aeroadapt.vehicle.aerojax_vehicle import trimmed_start
from aeroadapt.vehicle.protocol import (
    ImpactRecord,
    InitialCondition,
    StepResult,
    Termination,
    TruthSnapshot,
    VehicleInfo,
    classify_touchdown,
)

N_PER_LBF = 4.448_221_615
NM_PER_LBFFT = 1.355_817_948
TURB_MILSPEC = 3  # atmosphere/turb-type: MIL-F-8785C / MIL-HDBK-1797 Dryden


class JSBSimVehicle:
    def __init__(
        self,
        airframe: Airframe,
        site: Site,
        env: EnvParams,
        faults: FaultConfig | None = None,
        config: SimConfig = SimConfig(),  # noqa: B008
        model: str | None = None,
        root: Path = JSBSIM_ROOT,
    ) -> None:
        self._airframe = airframe
        self._vp = vp = to_vehicle_params(airframe)
        self._env = env
        self._faults = faults_cfg = no_faults() if faults is None else faults
        self._config = config
        self._site = site
        self._origin = site.origin.geodetic()
        self._fdm = JSBSimFDM(model or airframe.id, dt=config.dt_physics, root=root)
        self.info = VehicleInfo(config.control_hz, airframe.id, site.id, self._origin)
        dt = config.dt_physics
        vp, env_d, faults_d = jax.device_put((vp, env, faults_cfg))

        def pre_substep(
            act: ActuatorState,
            batt: BatteryState,
            prop_omega: jax.Array,
            cmd: jax.Array,
            t: jax.Array,
            rho: jax.Array,
            va: jax.Array,
            flow: jax.Array,
        ):
            """Actuators, then propulsion at the current airspeed, then the battery (the
            aerojax physics-substep order)."""
            af = vp.airframe
            _, bp, phys, ceiling = faulted_plant(vp, faults_d, t)
            act = actuator_step(
                af, vp.actuators, faults_d.servo_jam, act, cmd, t, dt, faults_d.actuators, flow
            )
            act = act._replace(throttle=jnp.minimum(act.throttle, ceiling))
            v_batt = terminal_voltage(bp, batt)
            cut = fault_labels(faults_d, t).motor_cut
            prop = propeller(af, rho, va, act.throttle, v_batt)
            thrust = jnp.where(cut, windmill_thrust(af, rho, va), prop.thrust * phys.thrust_scale)
            torque = jnp.where(cut, 0.0, prop.torque * phys.thrust_scale)
            draw = jnp.maximum(act.throttle * prop.motor_current, 0.0)
            pack = bp.avionics_current + jnp.where(cut, 0.0, draw)
            omega = jnp.where(cut, prop_omega * jnp.exp(-dt / MOTOR_SPIN_DOWN_S), prop.omega)
            batt = battery_step(bp, batt, pack, dt)
            return act, batt, omega, thrust, torque

        def sense(sens: SensorState, st: SensorTruth, key: jax.Array):
            return sample(
                vp.sensors,
                faults_d.pitot,
                sens,
                st,
                config.dt_control,
                config.control_hz,
                key,
                faults_d.sensors,
                faults_d.propulsion,
            )

        self._pre = jax.jit(pre_substep)
        self._sense = jax.jit(sense)
        self._vterm = jax.jit(lambda b, t: terminal_voltage(faulted_plant(vp, faults_d, t)[1], b))
        self._env_d = env_d
        self._key = jax.random.PRNGKey(0)
        self._t = 0.0
        self._k = 0
        self._done = False
        self._termination = Termination.NONE
        self._act: ActuatorState | None = None
        self._batt: BatteryState | None = None
        self._prop_omega = jnp.asarray(0.0)
        self._sens: SensorState | None = None
        self._thrust = 0.0
        self.trim_command: np.ndarray | None = None

    # ------------------------------------------------------------------ Vehicle
    def reset(self, initial: InitialCondition, seed: int) -> SensorFrame:
        """Start from aerojax's trim for the same airframe (the stack's view of trim);
        JSBSim then settles from it."""
        rb0, cmd0 = trimmed_start(self._airframe, self._env, initial)
        self.trim_command = cmd0
        e = self._env
        f = self._fdm
        lat, lon, _ = self._origin.ned_to_geodetic(
            np.array([initial.pos_ne_m[0], initial.pos_ne_m[1], 0.0])
        )
        f["atmosphere/delta-T"] = e.isa.temp_offset_k * 1.8  # Rankine
        f["position/terrain-elevation-asl-ft"] = e.isa.alt_origin_msl_m / FT
        wind = np.asarray(e.wind.steady_ned, dtype=float) / FT
        f["atmosphere/wind-north-fps"], f["atmosphere/wind-east-fps"] = wind[0], wind[1]
        f["atmosphere/wind-down-fps"] = wind[2]
        if float(e.turbulence.w20_mps) > 0:
            f["atmosphere/turb-type"] = TURB_MILSPEC
            f["atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps"] = (
                float(e.turbulence.w20_mps) / FT
            )
            f["atmosphere/turbulence/milspec/severity"] = 3
        f["simulation/randomseed"] = seed
        phi, theta, psi = (float(x) for x in quaternion.to_euler(rb0.quat))
        vel = np.asarray(rb0.vel_body, dtype=float)
        alpha = math.atan2(vel[2], vel[0])
        f.initialise(
            e.isa.alt_origin_msl_m + initial.alt_agl_m,
            float(np.linalg.norm(vel)),
            alpha=alpha,
            gamma=theta - alpha,
            phi=phi,
            psi=psi,
            lat_deg=float(lat),
            lon_deg=float(lon),
        )
        # JSBSim's initial airspeed is relative to the air mass, as in aerojax's trim.
        vp = self._vp
        self._act = init_actuators(vp.airframe, vp.actuators, jnp.asarray(cmd0))
        self._batt = init_battery(jnp.asarray(initial.soc))
        self._prop_omega = jnp.asarray(0.0)
        self._t, self._k, self._done = 0.0, 0, False
        self._termination = Termination.NONE
        k_sens, self._key = jax.random.split(jax.random.PRNGKey(seed))
        self._apply_actuators()
        vp_d = jax.device_put(vp)
        self._sens, z = init_sensors(
            vp_d.sensors,
            self._faults.pitot,
            self._sensor_truth(),
            self._config.dt_control,
            self._config.control_hz,
            k_sens,
            self._faults.sensors,
            self._faults.propulsion,
        )
        return frame_from_arrays(jax.device_get(z), 0, self._origin)

    def step(self, cmd: ActuatorCommand) -> StepResult:
        if self._act is None or self._batt is None or self._sens is None:
            raise RuntimeError("reset() first")
        if not self._done:
            c = jnp.asarray(cmd.as_array())
            for _ in range(self._config.substeps):
                f = self._fdm
                v = max(f.airspeed, 1.0)
                flow = jnp.asarray(
                    [
                        f.alpha,
                        f["aero/beta-rad"],
                        f["velocities/p-rad_sec"] * self._vp.airframe.b / (2 * v),
                    ]
                )
                act, batt, omega, thrust, torque = self._pre(
                    self._act, self._batt, self._prop_omega, c, self._t, f.rho, f.airspeed, flow
                )
                self._act, self._batt, self._prop_omega = act, batt, omega
                self._thrust = float(thrust)
                f["external_reactions/thrust/magnitude"] = self._thrust / N_PER_LBF
                # The propeller spins about +x_body, so the airframe feels -torque (ADR-001).
                f["external_reactions/prop_torque/magnitude-lbsft"] = -float(torque) / NM_PER_LBFFT
                self._apply_actuators()
                f.step()
                self._t += self._config.dt_physics
            self._k += 1
            self._check_termination()
        k, self._key = jax.random.split(self._key)
        self._sens, z = self._sense(self._sens, self._sensor_truth(), k)
        frame = frame_from_arrays(jax.device_get(z), round(self._t * 1e6), self._origin)
        return StepResult(frame, self._done, self._termination)

    # ------------------------------------------------------------------ internals
    def _apply_actuators(self) -> None:
        assert self._act is not None
        da, de, dr = (float(x) for x in np.asarray(self._act.surfaces))
        self._fdm.set_surfaces(da, de, dr)

    def _alt_agl(self) -> float:
        return self._fdm["position/h-agl-ft"] * FT

    def _pos_ned(self) -> np.ndarray:
        f = self._fdm
        ned = self._origin.geodetic_to_ned(
            f["position/lat-geod-deg"], f["position/long-gc-deg"], self._origin.h_ellipsoid_m
        )
        return np.array([ned[0], ned[1], -self._alt_agl()])

    def _vel_ned(self) -> np.ndarray:
        f = self._fdm
        return FT * np.array(
            [f["velocities/v-north-fps"], f["velocities/v-east-fps"], f["velocities/v-down-fps"]]
        )

    def _omega(self) -> np.ndarray:
        f = self._fdm
        return np.array(
            [f["velocities/p-rad_sec"], f["velocities/q-rad_sec"], f["velocities/r-rad_sec"]]
        )

    def _quat(self) -> np.ndarray:
        return np.asarray(quaternion.from_euler(*self._fdm.euler))

    def _sensor_truth(self) -> SensorTruth:
        assert self._batt is not None
        f = self._fdm
        mass = f["inertia/mass-slugs"]
        force = np.array(
            [f["forces/fbx-total-lbs"], f["forces/fby-total-lbs"], f["forces/fbz-total-lbs"]]
        )
        return SensorTruth(
            t=jnp.asarray(self._t),
            k=jnp.asarray(self._k, jnp.int32),
            pos_ned=jnp.asarray(self._pos_ned()),
            vel_ned=jnp.asarray(self._vel_ned()),
            quat=jnp.asarray(self._quat()),
            omega=jnp.asarray(self._omega()),
            specific_force=jnp.asarray(force / mass * FT),  # lbf/slug = ft/s²
            static_pa=jnp.asarray(f["atmosphere/P-psf"] * 47.880_258_98),
            temp_k=jnp.asarray(f["atmosphere/T-R"] / 1.8),
            rho=jnp.asarray(f["atmosphere/rho-slugs_ft3"] * SLUG_PER_FT3),
            u_air=jnp.asarray(f["velocities/u-aero-fps"] * FT),
            battery_v=self._vterm(self._batt, jnp.asarray(self._t)),
            battery_a=self._batt.current,
            battery_mah=self._batt.consumed_mah,
            prop_omega=self._prop_omega,
            mag_ned=jnp.asarray(self._env.magnetic_ned_gauss),
        )

    def _check_termination(self) -> None:
        pos = self._pos_ned()
        state = np.concatenate([pos, self._vel_ned(), self._omega()])
        if not np.all(np.isfinite(state)):
            code = Termination.NUMERICAL
        elif -pos[2] <= self._config.contact_height_m:
            code = Termination.GROUND_CONTACT
        elif np.hypot(pos[0], pos[1]) > self._config.max_range_m:
            code = Termination.OUT_OF_BOUNDS
        else:
            return
        self._done, self._termination = True, code

    # ------------------------------------------------------------------ privileged
    def truth(self) -> TruthSnapshot:
        assert self._act is not None and self._batt is not None
        f = self._fdm
        wind = FT * np.array(
            [
                f["atmosphere/total-wind-north-fps"],
                f["atmosphere/total-wind-east-fps"],
                f["atmosphere/total-wind-down-fps"],
            ]
        )
        labels = fault_labels(self._faults, jnp.asarray(self._t))
        return TruthSnapshot(
            time_s=self._t,
            pos_ned_m=self._pos_ned(),
            vel_ned_mps=self._vel_ned(),
            euler_rad=np.array(f.euler),
            omega_rps=self._omega(),
            airspeed_mps=f.airspeed,
            alpha_rad=f.alpha,
            beta_rad=f["aero/beta-rad"],
            alt_agl_m=self._alt_agl(),
            wind_ned_mps=wind,
            surfaces_rad=np.asarray(self._act.surfaces),
            throttle=float(self._act.throttle),
            thrust_n=self._thrust,
            battery_v=float(self._vterm(self._batt, jnp.asarray(self._t))),
            soc=float(self._batt.soc),
            fault_labels={k: np.asarray(v) for k, v in labels._asdict().items()},
        )

    def impact(self) -> ImpactRecord | None:
        if self._termination != Termination.GROUND_CONTACT:
            return None
        vel = self._vel_ned()
        roll, pitch, _ = self._fdm.euler
        pos = self._pos_ned()
        lat, lon, _ = self._origin.ned_to_geodetic(pos)
        sink = float(vel[2])
        return ImpactRecord(
            time_s=self._t,
            pos_ned_m=pos,
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
