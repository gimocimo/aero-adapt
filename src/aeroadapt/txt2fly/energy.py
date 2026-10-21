"""Battery use of a planned mission (spec 06 §6): an estimate, labelled as one.

The pack current at a given airspeed comes from the airframe's nominal model at trim (the
same batched level trims auto-configuration uses), plus the avionics; climbing adds the
potential energy at an assumed 50 % propulsive efficiency. Like auto-configuration this is
design-time knowledge of the aircraft (a manufacturer's endurance chart), not simulator
state. Turns, gusts and battery sag are not modelled; the 30 % reserve covers them.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from aeroadapt.airframes import Airframe, to_vehicle_params
from aeroadapt.scenarios.site import Site
from aeroadapt.sim.aerojax.atmosphere import IsaParams, isa
from aeroadapt.sim.aerojax.battery import open_circuit_voltage
from aeroadapt.sim.aerojax.environment import Environment
from aeroadapt.sim.aerojax.propulsion import propeller
from aeroadapt.sim.aerojax.trim import trim_level

G0 = 9.806_65
TRIM_ALT_M = 120.0
CLIMB_EFFICIENCY = 0.5  # electrical → potential energy while climbing
RESERVE = 0.30  # of capacity, kept for landing and for the estimate's errors


@dataclass(frozen=True)
class EnergyModel:
    speeds: np.ndarray  # m/s
    amps: np.ndarray  # pack current in level flight at each speed
    volts: float  # nominal pack voltage
    mass: float
    capacity_mah: float

    def current(self, airspeed: np.ndarray, climb_rate: np.ndarray) -> np.ndarray:
        """Pack current (A) along a track."""
        level = np.interp(airspeed, self.speeds, self.amps)
        climb = self.mass * G0 * np.maximum(climb_rate, 0.0) / (CLIMB_EFFICIENCY * self.volts)
        return level + climb

    def used_mah(self, t: np.ndarray, airspeed: np.ndarray, climb_rate: np.ndarray) -> np.ndarray:
        """Cumulative charge drawn along a track sampled at times ``t``."""
        amps = self.current(airspeed, climb_rate)
        dt = np.diff(t, prepend=t[0])
        return np.cumsum(amps * dt) / 3.6


_CACHE: dict[tuple[str, str, float, float], EnergyModel] = {}


def energy_model(airframe: Airframe, site: Site, v_min: float, v_max: float) -> EnergyModel:
    key = (airframe.model_dump_json(), site.id, v_min, v_max)
    if key in _CACHE:
        return _CACHE[key]
    vp = to_vehicle_params(airframe)
    p, b = vp.airframe, vp.battery
    isa_p = IsaParams(site.isa.temp_offset_k, site.isa.qnh_hpa * 100.0, site.ground_msl_m)
    _, _, rho = isa(isa_p, site.ground_msl_m + TRIM_ALT_M)
    volts = float(open_circuit_voltage(b, jnp.asarray(0.7)))
    speeds = np.linspace(v_min, v_max, 6)
    env = Environment(rho, G0)
    with jax.enable_x64(True):
        z, _ = jax.vmap(lambda v: trim_level(p, env, jnp.asarray(volts), v))(jnp.asarray(speeds))
        thr = jnp.clip(z[:, 6], 0.0, 1.0)
        out = jax.vmap(lambda v, t: propeller(p, rho, v, t, jnp.asarray(volts)))(
            jnp.asarray(speeds), thr
        )
        amps = np.asarray(thr * jnp.maximum(out.motor_current, 0.0)) + b.avionics_current
    model = EnergyModel(speeds, amps, volts, float(p.mass), 1000.0 * float(b.capacity_ah))
    _CACHE[key] = model
    return model
