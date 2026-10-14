"""Battery v0 (spec §6.4): open-circuit voltage from SOC, internal resistance, coulomb
counting. The terminal voltage uses the previous step's current (one-step lag), which
resolves the motor–battery algebraic loop."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from aeroadapt.airframes.params import BatteryParams


class BatteryState(NamedTuple):
    soc: Array  # () state of charge in [0, 1]
    current: Array  # () A drawn from the pack during the last step
    consumed_mah: Array  # ()


def open_circuit_voltage(b: BatteryParams, soc: Array) -> Array:
    return b.n_cells * jnp.interp(jnp.clip(soc, 0.0, 1.0), b.ocv_soc, b.ocv_v_cell)


def terminal_voltage(b: BatteryParams, s: BatteryState) -> Array:
    return open_circuit_voltage(b, s.soc) - s.current * b.r_internal


def init_battery(soc: Array) -> BatteryState:
    soc = jnp.asarray(soc)
    return BatteryState(soc, jnp.zeros_like(soc), jnp.zeros_like(soc))


def battery_step(b: BatteryParams, s: BatteryState, current: Array, dt: float) -> BatteryState:
    """Integrate a pack current (A, positive = discharge) over ``dt``."""
    charge_ah = current * dt / 3600.0
    return BatteryState(
        soc=jnp.clip(s.soc - charge_ah / b.capacity_ah, 0.0, 1.0),
        current=current,
        consumed_mah=s.consumed_mah + 1000.0 * charge_ah,
    )
