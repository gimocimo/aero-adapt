"""ISA, wind profile and Dryden scale lengths/intensities."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aeroadapt.sim.aerojax.atmosphere import IsaParams, WindParams, dryden_scales, isa, steady_wind

STD = IsaParams(temp_offset_k=0.0, qnh_pa=101_325.0, alt_origin_msl_m=0.0)


@pytest.mark.parametrize(
    ("h", "temp", "pressure", "rho"),
    [  # ICAO standard atmosphere table values
        (0.0, 288.15, 101_325.0, 1.2250),
        (1000.0, 281.65, 89_874.6, 1.1117),
        (2000.0, 275.15, 79_495.2, 1.0066),
    ],
)
def test_isa_matches_standard_table(h: float, temp: float, pressure: float, rho: float) -> None:
    t, p, r = isa(STD, jnp.asarray(h))
    assert float(t) == pytest.approx(temp, abs=1e-6)
    assert float(p) == pytest.approx(pressure, rel=2e-5)
    assert float(r) == pytest.approx(rho, rel=2e-4)


def test_isa_deviation_changes_density_not_pressure() -> None:
    _, p0, r0 = isa(STD, jnp.asarray(500.0))
    _, p1, r1 = isa(STD._replace(temp_offset_k=15.0), jnp.asarray(500.0))
    assert float(p1) == pytest.approx(float(p0))
    assert float(r1) < float(r0)


def test_boundary_layer_profile() -> None:
    w = WindParams(jnp.array([-10.0, 0.0, 1.0]), bl_alpha=1 / 7, bl_ref_m=50.0)
    np.testing.assert_allclose(steady_wind(w, jnp.asarray(120.0)), [-10.0, 0.0, 1.0])
    low = steady_wind(w, jnp.asarray(5.0))
    assert float(low[0]) == pytest.approx(-10.0 * (5 / 50) ** (1 / 7))
    assert float(low[2]) == pytest.approx(1.0)  # vertical component untouched
    off = steady_wind(w._replace(bl_alpha=0.0), jnp.asarray(5.0))
    np.testing.assert_allclose(off, [-10.0, 0.0, 1.0])


def test_dryden_low_altitude_scales() -> None:
    # MIL-HDBK-1797 low altitude at 120 m (393.7 ft), W20 = 10 m/s.
    sigma_uv, sigma_w, l_uv, l_w = dryden_scales(jnp.asarray(10.0), jnp.asarray(120.0))
    k = 0.177 + 0.000823 * 120.0 / 0.3048
    assert float(sigma_w) == pytest.approx(1.0)
    assert float(sigma_uv) == pytest.approx(1.0 / k**0.4)
    assert float(l_w) == pytest.approx(120.0)
    assert float(l_uv) == pytest.approx(120.0 / k**1.2)
    # Below 10 ft the model is clamped.
    _, _, _, l_w_low = dryden_scales(jnp.asarray(10.0), jnp.asarray(1.0))
    assert float(l_w_low) == pytest.approx(10 * 0.3048)
