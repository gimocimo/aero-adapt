"""Navigator v0 accuracy in closed loop (spec §9.1): the stock stack flies a mission in wind
and light turbulence, and the estimate is scored against truth after convergence."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax.typing import ArrayLike

from aeroadapt.airframes import Airframe
from aeroadapt.harness import LoopLog, setup_loop
from aeroadapt.harness.acceptance import LIGHT_W20_MPS
from aeroadapt.scenarios import load_site
from aeroadapt.sim.aerojax import make_env
from aeroadapt.stack.types import make_mission
from aeroadapt.vehicle.protocol import InitialCondition

SETTLE_S = 30.0


def _wrap(a: ArrayLike) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _rms(x: ArrayLike) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x)))))


@pytest.fixture(scope="module")
def flight(aerosonde: Airframe) -> LoopLog:
    site = load_site("cambridge_west")
    env = make_env(
        alt_origin_msl_m=site.ground_msl_m,
        magnetic_ned_gauss=site.magnetic.field_ned_gauss,
        wind_from_deg=250.0,
        wind_speed_mps=6.0,
        w20_mps=LIGHT_W20_MPS,
    )
    loop = setup_loop(aerosonde, site, env, InitialCondition(), seed=7)
    mission = make_mission([(700, 0, 120), (700, 700, 140), (0, 700, 120), (0, 0, 120)])
    _, _, log = loop.run(mission, 15_000, jax.random.PRNGKey(7))  # 150 s, with turns
    return jax.tree.map(np.asarray, log)


def test_attitude(flight: LoopLog) -> None:
    m = flight.t > SETTLE_S
    err = np.degrees(_wrap(flight.est_euler[m] - flight.euler[m]))
    assert np.abs(flight.euler[m, 0]).max() > np.radians(25)  # the mission does turn
    assert _rms(err[:, 0]) < 2.0
    assert _rms(err[:, 1]) < 2.0
    assert _rms(err[:, 2]) < 3.0


def test_position_velocity_height(flight: LoopLog, aerosonde: Airframe) -> None:
    """Horizontal position: the GNSS error is Gauss-Markov with a 300 s correlation time, a
    bias over this flight that no filter can observe. So the offset is bounded by the GNSS
    error model and the navigator's own error is the scatter around it."""
    m = flight.t > SETTLE_S
    err = flight.est_pos_ned[m, :2] - flight.pos_ned[m, :2]
    offset = err.mean(axis=0)
    sigma_h = aerosonde.sensors.gnss.pos_sigma_h_m
    assert np.linalg.norm(offset) < 3.0 * np.sqrt(2.0) * sigma_h
    assert _rms(np.linalg.norm(err - offset, axis=1)) < 1.0
    assert _rms(flight.est_pos_ned[m, 2] - flight.pos_ned[m, 2]) < 0.5
    assert _rms(np.linalg.norm(flight.est_vel_ned[m] - flight.vel_ned[m], axis=1)) < 0.5


def test_air_data_and_wind(flight: LoopLog) -> None:
    m = flight.t > SETTLE_S
    assert _rms(flight.est_airspeed[m] - flight.va[m]) < 0.3
    assert flight.airspeed_healthy.all()
    late = flight.t > flight.t[-1] - 20.0
    wind_err = np.linalg.norm(
        flight.est_wind_ned[late, :2].mean(0) - flight.wind_ned[late, :2].mean(0)
    )
    assert wind_err < 1.5
