"""Shared fixtures. Tests run with JAX float64 enabled; float32 is tested explicitly."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from aeroadapt.airframes import (
    AIRFRAME_DIR,
    Airframe,
    AirframeParams,
    VehicleParams,
    load_airframe,
    to_params,
    to_vehicle_params,
)
from aeroadapt.scenarios import load_site
from aeroadapt.sim.aerojax import BM_PARITY, Environment, EnvParams, RigidBodyState, make_env
from aeroadapt.sim.aerojax.actuators import target_to_command
from aeroadapt.sim.aerojax.atmosphere import isa
from aeroadapt.sim.aerojax.battery import open_circuit_voltage
from aeroadapt.sim.aerojax.trim import trim
from reference.bm_numpy import model as ref


class CruiseTrim(NamedTuple):
    state: RigidBodyState
    cmd: Array  # normalised (aileron, elevator, rudder, throttle)


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
    return ref.with_environment(raw, rho=float(BM_PARITY.rho), gravity=float(BM_PARITY.gravity))


# --------------------------------------------------------------------------- full simulator
@pytest.fixture(scope="session")
def vehicle(aerosonde: Airframe) -> VehicleParams:
    return to_vehicle_params(aerosonde)


@pytest.fixture(scope="session")
def cambridge_env() -> EnvParams:
    site = load_site("cambridge_west")
    return make_env(
        alt_origin_msl_m=site.ground_msl_m,
        magnetic_ned_gauss=site.magnetic.field_ned_gauss,
    )


@pytest.fixture(scope="session")
def cruise_trim(vehicle: VehicleParams, cambridge_env: EnvParams) -> CruiseTrim:
    """Straight-and-level 25 m/s at 120 m AGL over Cambridge, full battery."""
    _, _, rho = isa(cambridge_env.isa, cambridge_env.isa.alt_origin_msl_m + 120.0)
    v_full = float(open_circuit_voltage(vehicle.battery, jnp.asarray(1.0)))
    env = Environment(float(rho), cambridge_env.gravity)
    tr = trim(vehicle.airframe, env, 25.0, h=120.0, v_supply=v_full)
    assert tr.success
    cmd = target_to_command(vehicle.airframe, vehicle.actuators, jnp.asarray([*tr.controls]))
    return CruiseTrim(jax.tree_util.tree_map(jnp.asarray, tr.state), cmd)
