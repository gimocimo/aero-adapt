"""V3: N-version parity between aerojax (JAX) and the independent NumPy reference.

Fast tier: 1,000 random tuples and 3 short trajectories.
Full tier (``-m full``): 10,000 tuples and 20 trajectories of 60 s.

Tolerances (spec §12, float64): forces/moments/propulsion rtol 1e-10; trajectories 1e-8.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aeroadapt.airframes import AirframeParams
from aeroadapt.sim.aerojax import (
    BM_PARITY,
    controls_from_vector,
    forces_moments,
    rollout,
    state_derivative,
    state_from_vector,
    state_to_vector,
)
from reference.bm_numpy import model as ref

DT = 0.005


def _sample(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random (state, control, wind) tuples covering the normal flight envelope (Va >= 10)."""
    rng = np.random.default_rng(seed)
    xs, us, ws = [], [], []
    while len(xs) < n:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        x = np.concatenate(
            [
                rng.uniform(-2000, 2000, 3),
                [rng.uniform(12, 35), rng.uniform(-4, 4), rng.uniform(-4, 4)],
                q,
                rng.uniform(-1, 1, 3),
            ]
        )
        w = rng.uniform(-8, 8, 3)
        va, _, _ = ref.air_data(x, w)
        if va < 10.0:
            continue
        u = np.concatenate([rng.uniform(-0.35, 0.35, 3), [rng.uniform(0, 1)]])
        xs.append(x)
        us.append(u)
        ws.append(w)
    return np.array(xs), np.array(us), np.array(ws)


_fm_batched = jax.jit(
    jax.vmap(
        lambda p, x, u, w: forces_moments(
            p, BM_PARITY, state_from_vector(x), controls_from_vector(u), w
        ),
        in_axes=(None, 0, 0, 0),
    )
)
_xdot_batched = jax.jit(
    jax.vmap(
        lambda p, x, u, w: state_to_vector(
            state_derivative(p, BM_PARITY, state_from_vector(x), controls_from_vector(u), w)
        ),
        in_axes=(None, 0, 0, 0),
    )
)


def _check_pointwise(params: AirframeParams, ref_params: ref.Params, n: int, seed: int) -> None:
    xs, us, ws = _sample(n, seed)
    fm = _fm_batched(params, xs, us, ws)
    xdot = _xdot_batched(params, xs, us, ws)

    ref_f, ref_m, ref_t, ref_q, ref_om, ref_xdot = [], [], [], [], [], []
    for x, u, w in zip(xs, us, ws, strict=True):
        f, m = ref.forces_moments(ref_params, x, u, w)
        va, _, _ = ref.air_data(x, w)
        t, q, om = ref.propulsion(ref_params, va, u[3])
        ref_f.append(f)
        ref_m.append(m)
        ref_t.append(t)
        ref_q.append(q)
        ref_om.append(om)
        ref_xdot.append(ref.derivatives(ref_params, x, u, w))

    np.testing.assert_allclose(fm.force_body, np.array(ref_f), rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(fm.moment_body, np.array(ref_m), rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(fm.prop.thrust, ref_t, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(fm.prop.torque, ref_q, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(fm.prop.omega, ref_om, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(xdot, np.array(ref_xdot), rtol=1e-10, atol=1e-9)


def test_pointwise_parity_fast(params: AirframeParams, ref_params: ref.Params) -> None:
    _check_pointwise(params, ref_params, n=1_000, seed=0)


@pytest.mark.full
def test_pointwise_parity_full(params: AirframeParams, ref_params: ref.Params) -> None:
    _check_pointwise(params, ref_params, n=10_000, seed=1)


def _smooth_inputs(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Band-limited control inputs around a cruise setting, plus a slowly varying wind."""
    rng = np.random.default_rng(seed)
    t = DT * np.arange(n)
    nominal = np.array([0.0, -0.12, 0.0, 0.6])
    amp = np.array([0.03, 0.04, 0.02, 0.15])
    freqs = rng.uniform(0.1, 0.8, size=(4, 3))
    phases = rng.uniform(0, 2 * np.pi, size=(4, 3))
    u = nominal + amp * np.stack(
        [np.sin(2 * np.pi * freqs[i][:, None] * t + phases[i][:, None]).mean(0) for i in range(4)],
        axis=-1,
    )
    u[:, 3] = np.clip(u[:, 3], 0.0, 1.0)
    w_base = rng.uniform(-5, 5, 3)
    w = w_base + 1.5 * np.sin(2 * np.pi * 0.05 * t[:, None] + rng.uniform(0, 6, 3))
    return u, w


_rollout = jax.jit(
    lambda p, x0, us, ws: rollout(
        p, BM_PARITY, state_from_vector(x0), controls_from_vector(us), ws, DT
    )
)


def _check_trajectories(params: AirframeParams, ref_params: ref.Params, seeds, seconds: float):
    n = round(seconds / DT)
    for seed in seeds:
        u, w = _smooth_inputs(n, seed)
        x0 = np.array([0.0, 0.0, -100.0, 25.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        traj = np.asarray(state_to_vector(_rollout(params, jnp.asarray(x0), u, w)))
        xs_ref = ref.simulate(ref_params, x0, u, w, DT)
        assert np.all(np.isfinite(traj))
        np.testing.assert_allclose(traj, xs_ref, rtol=1e-9, atol=1e-8, err_msg=f"seed {seed}")


def test_trajectory_parity_fast(params: AirframeParams, ref_params: ref.Params) -> None:
    _check_trajectories(params, ref_params, seeds=range(3), seconds=20.0)


@pytest.mark.full
def test_trajectory_parity_full(params: AirframeParams, ref_params: ref.Params) -> None:
    _check_trajectories(params, ref_params, seeds=range(100, 120), seconds=60.0)
