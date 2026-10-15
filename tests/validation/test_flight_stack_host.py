"""Host-stepped integration: ``AerojaxVehicle`` + ``FlightStack`` exchanging only
``SensorFrame`` / ``ActuatorCommand`` messages (the path non-JAX backends will use)."""

from __future__ import annotations

import numpy as np

from aeroadapt.airframes import Airframe
from aeroadapt.scenarios import load_site
from aeroadapt.sim.aerojax import make_env
from aeroadapt.stack.config import build_stack_params, load_stack_cfg
from aeroadapt.stack.flight_stack import FlightStack
from aeroadapt.stack.types import make_mission
from aeroadapt.vehicle.aerojax_vehicle import AerojaxVehicle
from aeroadapt.vehicle.protocol import InitialCondition


def test_host_stepped_flight(aerosonde: Airframe) -> None:
    site = load_site("cambridge_west")
    env = make_env(
        alt_origin_msl_m=site.ground_msl_m,
        magnetic_ned_gauss=site.magnetic.field_ned_gauss,
        wind_from_deg=250.0,
        wind_speed_mps=5.0,
    )
    vehicle = AerojaxVehicle(aerosonde, site, env)
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
    assert np.linalg.norm(tel.pos_ned_m[:2] - truth.pos_ned_m[:2]) < 3.0
    assert np.degrees(np.abs(tel.euler_rad[:2] - truth.euler_rad[:2])).max() < 2.0
