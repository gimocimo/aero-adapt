"""Aerodynamic model numerics: the log-sum-exp stall blending is exact and float32-safe."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from aeroadapt.airframes import AirframeParams
from aeroadapt.sim.aerojax.aero import lift_coefficient, stall_sigma


def _sigma_direct(m: float, a0: float, alpha: float) -> float:
    num = 1 + math.exp(-m * (alpha - a0)) + math.exp(m * (alpha + a0))
    den = (1 + math.exp(-m * (alpha - a0))) * (1 + math.exp(m * (alpha + a0)))
    return num / den


def test_lse_sigma_equals_direct_formula_in_float64(params: AirframeParams) -> None:
    alpha = np.linspace(-math.pi, math.pi, 2001)
    direct = np.array([_sigma_direct(params.M, params.alpha0, a) for a in alpha])
    np.testing.assert_allclose(
        stall_sigma(params, jnp.asarray(alpha)), direct, rtol=1e-12, atol=1e-300
    )


def test_sigma_is_finite_and_bounded_in_float32(params: AirframeParams) -> None:
    alpha = jnp.linspace(-math.pi, math.pi, 4001, dtype=jnp.float32)
    s = stall_sigma(params, alpha)
    assert s.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(s)))
    assert bool(jnp.all((s >= 0) & (s <= 1)))
    assert bool(jnp.all(jnp.isfinite(lift_coefficient(params, alpha))))


def test_sigma_limits(params: AirframeParams) -> None:
    # ~0 (attached flow) at small α, ~1 (flat plate) well past α0 in either direction.
    assert float(stall_sigma(params, jnp.array(0.0))) < 1e-9
    assert float(stall_sigma(params, jnp.array(params.alpha0 + 0.3))) > 0.999
    assert float(stall_sigma(params, jnp.array(-params.alpha0 - 0.3))) > 0.999
