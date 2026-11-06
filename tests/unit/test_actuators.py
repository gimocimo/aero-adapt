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


@pytest.mark.parametrize("airframe", ["aerosonde_bm", "gen-twin_boom-00058"])
@pytest.mark.parametrize("reversed_servos", [False, True])
def test_a_floating_surface_destabilises_in_either_sign_convention(
    airframe: str, reversed_servos: bool
) -> None:
    """A disconnected surface trails with its local flow and always destabilises: the
    ailerons reinforce the roll rate, the elevator pitches up at positive α, the rudder yaws
    against the sideslip. B&M's Aerosonde and the generator's airframes deflect ailerons and
    rudder in opposite conventions (signs of C_ℓ,δa and C_n,δr), and a broken linkage ignores
    the servo's reversal."""
    from pathlib import Path

    from pydantic import TypeAdapter

    from aeroadapt.airframes import load_airframe, to_vehicle_params
    from aeroadapt.airframes.fleet import load_fleet
    from aeroadapt.faultlab import build_fault_config
    from aeroadapt.faultlab.schema import FaultSpec
    from aeroadapt.sim.aerojax.actuators import FLOAT_TRAIL

    if airframe.startswith("gen-"):
        sample = Path(__file__).resolve().parents[2] / "configs" / "fleets" / "sample"
        af = next(a for a in load_fleet(sample) if a.id == airframe)
    else:
        af = load_airframe(airframe)
    vp = to_vehicle_params(af)
    vp = vp._replace(actuators=vp.actuators._replace(reversed=np.full(3, reversed_servos)))
    p = vp.airframe
    flow = jnp.array([0.05, 0.04, 0.03])  # α, β, p·b/2V
    jam = no_faults().servo_jam
    moments = {}
    for i, channel in enumerate(("aileron", "elevator", "rudder")):
        spec = TypeAdapter(FaultSpec).validate_python(
            {"type": "actuator_fault", "kind": "floating", "channel": channel,
             "trigger": {"at_time_s": 0.0}}
        )  # fmt: skip
        faults = build_fault_config([spec]).actuators
        s = init_actuators(p, vp.actuators, jnp.array([0.0, 0.0, 0.0, 0.5]))

        def body(s, k, faults=faults):
            cmd = jnp.array([0.0, 0.0, 0.0, 0.5])
            return actuator_step(p, vp.actuators, jam, s, cmd, k * DT, DT, faults, flow), None

        s, _ = jax.lax.scan(body, s, jnp.arange(400))  # 2 s, many servo lags
        local = float(np.asarray(flow)[[2, 0, 1]][i])
        deflection = float(s.surfaces[i])
        assert abs(deflection) == pytest.approx(FLOAT_TRAIL * local, abs=np.radians(0.5))
        power = (p.C_ell_delta_a, p.C_m_delta_e, p.C_n_delta_r)[i]
        moments[channel] = float(power) * deflection
    assert moments["aileron"] > 0  # with p > 0: less roll damping
    assert moments["elevator"] > 0  # nose up at α > 0: less stick-free stability
    assert moments["rudder"] < 0  # nose left with β > 0: less weathercock stability
