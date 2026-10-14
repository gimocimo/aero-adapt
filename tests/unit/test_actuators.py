"""Actuator pipeline: servo-map signs, delay, rate limit, lag, deadband (spec §6.3)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aeroadapt.airframes import VehicleParams
from aeroadapt.faultlab import no_faults
from aeroadapt.sim.aerojax.actuators import (
    actuator_step,
    command_to_target,
    init_actuators,
    target_to_command,
)

DT = 0.005


def _run(vp: VehicleParams, cmd0, cmds, jam=None):
    jam = no_faults().servo_jam if jam is None else jam
    s = init_actuators(vp.airframe, vp.actuators, jnp.asarray(cmd0))

    def body(s, x):
        k, c = x
        s = actuator_step(vp.airframe, vp.actuators, jam, s, c, k * DT, DT)
        return s, (s.surfaces, s.throttle)

    ks = jnp.arange(len(cmds))
    _, (surf, thr) = jax.lax.scan(body, s, (ks, jnp.asarray(cmds)))
    return np.asarray(surf), np.asarray(thr)


def test_aircraft_response_signs(vehicle: VehicleParams) -> None:
    af = vehicle.airframe
    t = np.asarray(command_to_target(af, vehicle.actuators, jnp.array([1.0, 1.0, 1.0, 0.5])))
    # Positive commands must produce positive roll, pitch and yaw moments (ADR-001).
    assert np.sign(t[0] * af.C_ell_delta_a) > 0
    assert np.sign(t[1] * af.C_m_delta_e) > 0
    assert np.sign(t[2] * af.C_n_delta_r) > 0
    assert t[3] == pytest.approx(0.5)
    back = target_to_command(af, vehicle.actuators, jnp.asarray(t))
    np.testing.assert_allclose(back, [1.0, 1.0, 1.0, 0.5], atol=1e-12)


def test_transport_delay_is_exact(vehicle: VehicleParams) -> None:
    n_delay = round(float(vehicle.actuators.delay[0]) / DT)  # 20 ms → 4 steps
    cmds = np.tile([0.2, 0.0, 0.0, 0.5], (20, 1))
    surf, _ = _run(vehicle, [0.0, 0.0, 0.0, 0.5], cmds)
    # The first n_delay steps still see the old (zero) target; motion starts after the delay.
    np.testing.assert_array_equal(surf[:n_delay, 0], 0.0)
    assert abs(surf[n_delay, 0]) > 0


def test_rate_limit_and_lag(vehicle: VehicleParams) -> None:
    act = vehicle.actuators
    # Full-scale step: per-step change never exceeds rate · dt.
    surf, _ = _run(vehicle, [0.0] * 3 + [0.5], np.tile([1.0, 0.0, 0.0, 0.5], (100, 1)))
    assert np.max(np.abs(np.diff(surf[:, 0]))) <= float(act.rate_limit[0]) * DT + 1e-12
    np.testing.assert_allclose(
        surf[-1, 0], float(act.max_deflection[0]), atol=float(act.deadband[0])
    )
    # Small step (not rate limited): first-order lag reaches 1 - e^-1 of the step one τ later.
    small = 2.0 / 25.0  # 2 degrees of a 25-degree throw
    surf, _ = _run(vehicle, [0.0] * 3 + [0.5], np.tile([small, 0.0, 0.0, 0.5], (40, 1)))
    n_delay = round(float(act.delay[0]) / DT)
    n_tau = round(float(act.tau[0]) / DT)
    target = small * float(act.max_deflection[0]) - float(act.deadband[0]) / 2
    assert surf[n_delay + n_tau - 1, 0] == pytest.approx(target * (1 - np.exp(-1)), rel=1e-9)


def test_deadband_bounds_steady_state_error(vehicle: VehicleParams) -> None:
    act = vehicle.actuators
    cmds = np.tile([0.37, -0.21, 0.11, 0.5], (200, 1))
    surf, _ = _run(vehicle, [0.0] * 3 + [0.5], cmds)
    target = np.asarray(command_to_target(vehicle.airframe, act, jnp.asarray(cmds[0])))[:3]
    assert np.all(np.abs(surf[-1] - target) <= np.asarray(act.deadband) / 2 + 1e-12)
