"""V8: Dryden turbulence statistics (spec §12).

Fast tier: 16 seeds × 5 min, σ within ±15 %. Full tier: 64 seeds × 10 min, σ within ±5 %,
and the empirical PSD within ±1.5 dB of the analytic Dryden spectrum over the significant
band (0.5/τ to 5/τ).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.signal import welch

from aeroadapt.sim.aerojax.atmosphere import (
    TurbulenceParams,
    dryden_scales,
    dryden_stationary,
    dryden_step,
)

DT = 0.005
VA = 25.0
W20 = 15 * 0.514444  # "light" turbulence


def _simulate(h_agl: float, seeds: int, seconds: float) -> np.ndarray:
    turb = TurbulenceParams(W20)
    n = round(seconds / DT)

    def one(key):
        k0, k1 = jax.random.split(key)
        s0 = dryden_stationary(turb, jnp.asarray(VA), jnp.asarray(h_agl), k0, jnp.float64)

        def body(s, k):
            s, g = dryden_step(s, turb, jnp.asarray(VA), jnp.asarray(h_agl), DT, k)
            return s, g

        _, gust = jax.lax.scan(body, s0, jax.random.split(k1, n))
        return gust

    keys = jax.random.split(jax.random.PRNGKey(int(h_agl)), seeds)
    return np.asarray(jax.jit(jax.vmap(one))(keys))  # (seeds, n, 3)


def _check_sigma(h_agl: float, seeds: int, seconds: float, rel: float) -> np.ndarray:
    g = _simulate(h_agl, seeds, seconds)
    sigma_uv, sigma_w, _, _ = (
        float(x) for x in dryden_scales(jnp.asarray(W20), jnp.asarray(h_agl))
    )
    std = g.reshape(-1, 3).std(axis=0)
    np.testing.assert_allclose(std, [sigma_uv, sigma_uv, sigma_w], rtol=rel)
    return g


@pytest.mark.parametrize("h_agl", [50.0, 120.0])
def test_v8_sigma_fast(h_agl: float) -> None:
    _check_sigma(h_agl, seeds=16, seconds=300.0, rel=0.15)


@pytest.mark.full
@pytest.mark.parametrize("h_agl", [50.0, 120.0])
def test_v8_sigma_and_psd_full(h_agl: float) -> None:
    g = _check_sigma(h_agl, seeds=64, seconds=600.0, rel=0.05)
    sigma_uv, sigma_w, l_uv, l_w = (
        float(x) for x in dryden_scales(jnp.asarray(W20), jnp.asarray(h_agl))
    )
    f, p = welch(g, fs=1 / DT, nperseg=2**16, axis=1)
    p = p.mean(axis=0)  # average the seeds: (freqs, 3)
    w = 2 * np.pi * f
    spectra = {
        0: (l_uv, lambda w, s, L: s**2 * 2 * L / (np.pi * VA) / (1 + (L * w / VA) ** 2), sigma_uv),
        1: (
            l_uv,
            lambda w, s, L: (
                s**2 * L / (np.pi * VA) * (1 + 3 * (L * w / VA) ** 2) / (1 + (L * w / VA) ** 2) ** 2
            ),
            sigma_uv,
        ),
        2: (
            l_w,
            lambda w, s, L: (
                s**2 * L / (np.pi * VA) * (1 + 3 * (L * w / VA) ** 2) / (1 + (L * w / VA) ** 2) ** 2
            ),
            sigma_w,
        ),
    }
    for ch, (length, phi, sigma) in spectra.items():
        tau = length / VA
        band = (w >= 0.5 / tau) & (w <= 5.0 / tau)
        empirical = p[band, ch] / (2 * np.pi)  # per-Hz one-sided → per rad/s
        db = 10 * np.log10(empirical / phi(w[band], sigma, length))
        assert np.max(np.abs(db)) <= 1.5, f"channel {ch}: max |Δ| {np.max(np.abs(db)):.2f} dB"
