"""The ``AerojaxVehicle`` adapter (ADR-004) and its privileged extension (ADR-006)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aeroadapt.airframes import Airframe
from aeroadapt.core.messages import ActuatorCommand
from aeroadapt.scenarios import load_site
from aeroadapt.sim.aerojax import EnvParams
from aeroadapt.vehicle.aerojax_vehicle import AerojaxVehicle
from aeroadapt.vehicle.protocol import InitialCondition, Termination, classify_touchdown


@pytest.fixture(scope="module")
def vehicle(aerosonde: Airframe, cambridge_env: EnvParams) -> AerojaxVehicle:
    return AerojaxVehicle(aerosonde, load_site("cambridge_west"), cambridge_env)


def test_reset_places_the_aircraft_and_reports_wgs84(vehicle: AerojaxVehicle) -> None:
    initial = InitialCondition(pos_ne_m=(100.0, -50.0), alt_agl_m=80.0, heading_deg=90.0)
    frame = vehicle.reset(initial, seed=1)
    truth = vehicle.truth()
    np.testing.assert_allclose(truth.pos_ned_m, [100.0, -50.0, -80.0], atol=1e-6)
    assert math.degrees(truth.euler_rad[2]) == pytest.approx(90.0, abs=1e-6)
    origin = vehicle.info.origin
    # GNSS is reported in WGS84 near the expected point (latency and noise: tens of metres).
    ned = origin.geodetic_to_ned(frame.gnss_lat_deg, frame.gnss_lon_deg, frame.gnss_alt_ellipsoid_m)
    assert np.linalg.norm(ned - truth.pos_ned_m) < 15.0
    assert vehicle.trim_command is not None


def test_trim_command_holds_level_flight(vehicle: AerojaxVehicle) -> None:
    vehicle.reset(InitialCondition(), seed=2)
    assert vehicle.trim_command is not None
    cmd = vehicle.trim_command
    for k in range(200):  # 2 s
        res = vehicle.step(ActuatorCommand.from_array(k * 10_000, cmd))
        assert not res.terminated
    truth = vehicle.truth()
    assert abs(truth.alt_agl_m - 120.0) < 1.0
    assert abs(truth.airspeed_mps - 25.0) < 0.5
    assert np.all(np.abs(np.degrees(truth.euler_rad[:2])) < [1.0, 4.0])


def test_ground_contact_terminates_with_an_impact_record(vehicle: AerojaxVehicle) -> None:
    vehicle.reset(InitialCondition(alt_agl_m=8.0), seed=3)
    assert vehicle.impact() is None
    res = None
    for k in range(500):
        res = vehicle.step(ActuatorCommand(k * 10_000, 0.0, -1.0, 0.0, 0.0))  # nose down
        if res.terminated:
            break
    assert res is not None and res.terminated
    assert res.termination == Termination.GROUND_CONTACT
    impact = vehicle.impact()
    assert impact is not None
    assert impact.sink_rate_mps > 3.0
    assert impact.touchdown_class == "crash"


@pytest.mark.parametrize(
    ("sink", "roll_deg", "pitch_deg", "expected"),
    [
        (0.5, 0.0, 4.0, "controlled"),
        (1.5, -10.0, -5.0, "controlled"),
        (2.0, 0.0, 4.0, "hard"),
        (1.0, 20.0, 4.0, "hard"),
        (3.5, 0.0, 4.0, "crash"),
        (1.0, 30.0, 4.0, "crash"),
        (1.0, 0.0, -20.0, "hard"),  # pitch only limits the controlled class
    ],
)
def test_touchdown_classes(sink: float, roll_deg: float, pitch_deg: float, expected: str) -> None:
    assert classify_touchdown(sink, math.radians(roll_deg), math.radians(pitch_deg)) == expected
