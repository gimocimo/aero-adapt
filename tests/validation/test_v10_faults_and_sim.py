"""V10 faults in the full simulator, plus simulator integration checks: trim hold, control
semantics, termination, determinism (V13) and the recompilation guard (ADR-002)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import TypeAdapter

from aeroadapt.airframes import VehicleParams
from aeroadapt.faultlab import FaultConfig, FaultSpec, build_fault_config, no_faults
from aeroadapt.sim.aerojax import EnvParams, SimConfig, init_state, sim_step
from aeroadapt.sim.aerojax.sim import GROUND_CONTACT

CFG = SimConfig()
SPECS = TypeAdapter(list[FaultSpec])


def _fly(vehicle, env, faults: FaultConfig, cruise_trim, cmds: np.ndarray, seed: int = 0):
    s0, _ = init_state(
        vehicle,
        env,
        faults,
        cruise_trim.state,
        cruise_trim.cmd,
        jax.random.PRNGKey(seed),
        config=CFG,
    )

    def body(s, x):
        cmd, key = x
        s, sens, info = sim_step(vehicle, env, faults, s, cmd, key, CFG)
        return s, (s, sens, info)

    keys = jax.random.split(jax.random.PRNGKey(seed + 1), len(cmds))
    _, out = jax.jit(lambda s: jax.lax.scan(body, s, (jnp.asarray(cmds), keys)))(s0)
    return out


def _hold(cruise_trim, n: int) -> np.ndarray:
    return np.tile(np.asarray(cruise_trim.cmd), (n, 1))


def test_open_loop_trim_hold(vehicle: VehicleParams, cambridge_env: EnvParams, cruise_trim) -> None:
    states, _, info = _fly(
        vehicle, cambridge_env, no_faults(), cruise_trim, _hold(cruise_trim, 1000)
    )
    assert np.max(np.abs(np.asarray(info.truth.va) - 25.0)) < 0.5
    assert np.max(np.abs(np.asarray(info.truth.alt_agl) - 120.0)) < 3.0
    assert not bool(states.done[-1])
    assert np.asarray(info.truth.model_validity_ok).all()


@pytest.mark.parametrize(("axis", "rate_index"), [(0, 0), (1, 1), (2, 2)])
def test_positive_commands_give_positive_rates(
    vehicle, cambridge_env, cruise_trim, axis: int, rate_index: int
) -> None:
    cmds = _hold(cruise_trim, 40)
    cmds[:, axis] += 0.2
    states, _, _ = _fly(vehicle, cambridge_env, no_faults(), cruise_trim, cmds)
    assert float(states.rb.omega[-1, rate_index]) > 0.05


def test_motor_cut(vehicle, cambridge_env, cruise_trim) -> None:
    faults = build_fault_config(
        SPECS.validate_python([{"type": "motor_cut", "trigger": {"at_time_s": 1.0}}])
    )
    states, _, info = _fly(vehicle, cambridge_env, faults, cruise_trim, _hold(cruise_trim, 300))
    t = np.asarray(states.t)
    after = t > 1.0 + CFG.dt_control  # first full step after onset
    thrust = np.asarray(info.truth.thrust)
    assert thrust[t <= 1.0].min() > 5.0 and thrust[after].max() <= 0.0
    np.testing.assert_array_equal(np.asarray(info.labels.motor_cut), t >= 1.0)
    rpm0 = np.asarray(states.prop_omega)[t <= 1.0][-1]
    assert np.asarray(states.prop_omega)[t >= 2.5].max() < 0.05 * rpm0
    np.testing.assert_allclose(
        np.asarray(info.truth.battery_a)[t > 1.1], vehicle.battery.avionics_current
    )


def test_servo_jam_holds_surface(vehicle, cambridge_env, cruise_trim) -> None:
    faults = build_fault_config(
        SPECS.validate_python(
            [{"type": "servo_jam", "channel": "elevator", "trigger": {"at_time_s": 0.5}}]
        )
    )
    cmds = _hold(cruise_trim, 150)
    cmds[60:, 1] = -0.8  # command a big nose-down after the jam
    states, _, info = _fly(vehicle, cambridge_env, faults, cruise_trim, cmds)
    t = np.asarray(states.t)
    elev = np.asarray(info.truth.surfaces)[:, 1]
    at_onset = elev[t >= 0.5][0]
    db = float(vehicle.actuators.deadband[1])
    assert np.max(np.abs(elev[t >= 0.5] - at_onset)) <= db


def test_ground_contact_terminates_and_freezes(vehicle, cambridge_env, cruise_trim) -> None:
    cmds = _hold(cruise_trim, 1500)
    cmds[:, 1] = -0.6  # sustained nose-down
    cmds[:, 3] = 0.0
    states, _, _ = _fly(vehicle, cambridge_env, no_faults(), cruise_trim, cmds)
    done = np.asarray(states.done)
    assert done.any()
    first = int(np.argmax(done))
    assert int(states.termination[-1]) == GROUND_CONTACT
    pos = np.asarray(states.rb.pos_ned)
    np.testing.assert_array_equal(pos[first:], np.broadcast_to(pos[first], pos[first:].shape))
    assert float(-states.rb.pos_ned[first, 2]) <= CFG.contact_height_m


def test_v13_determinism(vehicle, cambridge_env, cruise_trim) -> None:
    a = _fly(
        vehicle,
        cambridge_env._replace(turbulence=cambridge_env.turbulence._replace(w20_mps=7.7)),
        no_faults(),
        cruise_trim,
        _hold(cruise_trim, 200),
        seed=3,
    )
    b = _fly(
        vehicle,
        cambridge_env._replace(turbulence=cambridge_env.turbulence._replace(w20_mps=7.7)),
        no_faults(),
        cruise_trim,
        _hold(cruise_trim, 200),
        seed=3,
    )
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_no_recompilation_across_parameter_values(vehicle, cambridge_env, cruise_trim) -> None:
    step = jax.jit(sim_step, static_argnames="config")
    s, _ = init_state(
        vehicle,
        cambridge_env,
        no_faults(),
        cruise_trim.state,
        cruise_trim.cmd,
        jax.random.PRNGKey(0),
        config=CFG,
    )
    key = jax.random.PRNGKey(1)
    step(vehicle, cambridge_env, no_faults(), s, cruise_trim.cmd, key, config=CFG)
    heavier = vehicle._replace(airframe=vehicle.airframe._replace(mass=12.5))
    faults = build_fault_config(
        SPECS.validate_python([{"type": "motor_cut", "trigger": {"at_time_s": 3.0}}])
    )
    step(heavier, cambridge_env, faults, s, cruise_trim.cmd, key, config=CFG)
    # Private JAX API, used deliberately: the compiled-variant count must stay at one.
    assert step._cache_size() == 1  # pyright: ignore[reportAttributeAccessIssue]
