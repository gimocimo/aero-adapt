"""V1 (part): frame conversions and wind conventions."""

from __future__ import annotations

import jax
import numpy as np

from aeroadapt.core import frames


def test_ned_enu_roundtrip_and_mapping() -> None:
    v = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
    np.testing.assert_allclose(frames.enu_to_ned(frames.ned_to_enu(v)), v, atol=0)
    np.testing.assert_allclose(frames.ned_to_enu(np.array([1.0, 2.0, 3.0])), [2.0, 1.0, -3.0])


def test_frd_to_flu() -> None:
    np.testing.assert_allclose(frames.frd_to_flu(np.array([1.0, 2.0, 3.0])), [1.0, -2.0, -3.0])


def test_wind_from_convention() -> None:
    # A northerly wind (from 0°) moves air towards the south.
    np.testing.assert_allclose(frames.wind_from_to_ned(0.0, 10.0), [-10.0, 0.0, 0.0], atol=1e-12)
    # A westerly wind (from 270°) moves air towards the east.
    np.testing.assert_allclose(frames.wind_from_to_ned(270.0, 5.0), [0.0, 5.0, 0.0], atol=1e-12)


def test_wind_roundtrip() -> None:
    from_deg = np.linspace(0.0, 359.0, 60)
    speed = np.linspace(0.5, 20.0, 60)
    back_dir, back_speed = frames.ned_to_wind_from(frames.wind_from_to_ned(from_deg, speed))
    np.testing.assert_allclose(back_dir, from_deg, atol=1e-9)
    np.testing.assert_allclose(back_speed, speed, atol=1e-12)
