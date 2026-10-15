"""``JSBSimVehicle``: the stock stack flies the Aerosonde in JSBSim through the same host-stepped
``Vehicle`` interface as aerojax (M2 spec D3)."""

from __future__ import annotations

import numpy as np
import pytest

from aeroadapt.airframes import Airframe
from aeroadapt.scenarios import load_site
from aeroadapt.sim.aerojax import make_env
from aeroadapt.stack.config import build_stack_params, load_stack_cfg
from aeroadapt.stack.flight_stack import FlightStack
from aeroadapt.stack.types import make_mission
from aeroadapt.vehicle.protocol import InitialCondition

pytest.importorskip("jsbsim")

from aeroadapt.vehicle.jsbsim_vehicle import JSBSimVehicle


def test_stock_stack_flies_the_aerosonde_in_jsbsim(aerosonde: Airframe) -> None:
    site = load_site("cambridge_west")
    env = make_env(
        alt_origin_msl_m=site.ground_msl_m,
        magnetic_ned_gauss=site.magnetic.field_ned_gauss,
        wind_from_deg=250.0,
        wind_speed_mps=5.0,
    )
    vehicle = JSBSimVehicle(aerosonde, site, env)
    params = build_stack_params(
        load_stack_cfg("aerosonde_bm"), aerosonde, site, 1.0 / vehicle.info.control_rate_hz
    )
    stack = FlightStack(params, vehicle.info.origin)
    frame = vehicle.reset(InitialCondition(), seed=4)
    stack.reset(frame, make_mission([(600, 0, 120), (600, 600, 120)]))
    tel = None
    for _ in range(2_500):  # 25 s: first leg and the turn at the waypoint
        cmd, tel = stack.step(frame)
        res = vehicle.step(cmd)
        assert not res.terminated
        frame = res.sensors
    assert tel is not None
    truth = vehicle.truth()
    assert tel.mode == "AUTO" and tel.phase == "NOMINAL" and tel.wp_idx == 1
    assert abs(truth.alt_agl_m - 120.0) < 5.0
    assert abs(truth.airspeed_mps - 25.0) < 2.0
    assert np.linalg.norm(tel.pos_ned_m[:2] - truth.pos_ned_m[:2]) < 3.0
    assert np.degrees(np.abs(tel.euler_rad[:2] - truth.euler_rad[:2])).max() < 2.0
    assert truth.thrust_n > 0.0 and truth.soc < 1.0
