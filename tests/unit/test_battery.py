"""Battery v0: OCV curve, internal-resistance sag, coulomb counting."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from aeroadapt.airframes import VehicleParams
from aeroadapt.sim.aerojax.battery import (
    BatteryState,
    battery_step,
    init_battery,
    open_circuit_voltage,
    terminal_voltage,
)


def test_open_circuit_voltage_endpoints(vehicle: VehicleParams) -> None:
    b = vehicle.battery
    assert float(open_circuit_voltage(b, jnp.asarray(1.0))) == pytest.approx(12 * 4.20)
    assert float(open_circuit_voltage(b, jnp.asarray(0.0))) == pytest.approx(12 * 3.30)
    assert float(open_circuit_voltage(b, jnp.asarray(0.55))) == pytest.approx(12 * 3.825)


def test_sag_and_coulomb_counting(vehicle: VehicleParams) -> None:
    b = vehicle.battery
    s = init_battery(jnp.asarray(0.8))
    s = battery_step(b, s, jnp.asarray(10.0), 360.0)  # 10 A for 6 minutes = 1 Ah
    assert float(s.consumed_mah) == pytest.approx(1000.0)
    assert float(s.soc) == pytest.approx(0.8 - 1.0 / b.capacity_ah)
    sag = float(open_circuit_voltage(b, s.soc) - terminal_voltage(b, s))
    assert sag == pytest.approx(10.0 * b.r_internal)


def test_soc_is_clamped(vehicle: VehicleParams) -> None:
    s = BatteryState(jnp.asarray(0.001), jnp.asarray(0.0), jnp.asarray(0.0))
    s = battery_step(vehicle.battery, s, jnp.asarray(100.0), 3600.0)
    assert float(s.soc) == 0.0
