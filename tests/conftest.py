"""Shared fixtures. Tests run with JAX float64 enabled; float32 is tested explicitly."""

from __future__ import annotations

import jax
import pytest

from aeroadapt.airframes import AIRFRAME_DIR, Airframe, AirframeParams, load_airframe, to_params
from aeroadapt.sim.aerojax import BM_PARITY
from reference.bm_numpy import model as ref


def pytest_configure(config: pytest.Config) -> None:
    jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def aerosonde() -> Airframe:
    return load_airframe("aerosonde_bm")


@pytest.fixture(scope="session")
def params(aerosonde: Airframe) -> AirframeParams:
    return to_params(aerosonde)


@pytest.fixture(scope="session")
def ref_params() -> ref.Params:
    raw = ref.load_params(AIRFRAME_DIR / "aerosonde_bm.yaml")
    return ref.with_environment(raw, rho=BM_PARITY.rho, gravity=BM_PARITY.gravity)
