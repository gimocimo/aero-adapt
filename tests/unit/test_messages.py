"""Message conversions between the array world (local NED) and the host world (WGS84)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from aeroadapt.core.messages import (
    ActuatorCommand,
    FreshFlags,
    SensorArrays,
    arrays_from_frame,
    frame_from_arrays,
)
from aeroadapt.scenarios import load_site
from aeroadapt.stack.flight_stack import pack_sensors, unpack_sensors


def _arrays(rng: np.random.Generator) -> SensorArrays:
    v3 = lambda s: rng.normal(0.0, s, 3)  # noqa: E731
    flags: list[Any] = [np.asarray(bool(b)) for b in rng.integers(0, 2, 7)]
    fields: dict[str, Any] = dict(
        gyro=v3(0.1),
        accel=v3(5.0),
        baro_pa=np.asarray(99_800.0),
        baro_temp_k=np.asarray(287.0),
        airspeed_dp_pa=np.asarray(380.0),
        airspeed_temp_k=np.asarray(287.5),
        gnss_pos_ned=np.array([1234.5, -876.25, -120.0]),
        gnss_vel_ned=v3(10.0),
        gnss_time_s=np.asarray(12.34),
        mag=v3(0.3),
        battery_v=np.asarray(48.2),
        battery_a=np.asarray(9.1),
        battery_mah=np.asarray(812.0),
        esc_rpm=np.asarray(4880.0),
        fresh=FreshFlags(*flags),
    )
    return SensorArrays(**fields)


def test_actuator_command_round_trip() -> None:
    cmd = ActuatorCommand(1_000, 0.1, -0.2, 0.3, 0.7)
    back = ActuatorCommand.from_array(1_000, cmd.as_array())
    assert back == cmd


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_frame_round_trip_is_lossless(seed: int) -> None:
    """NED → WGS84 frame → NED keeps millimetre precision (float64 geodesy, ADR-001)."""
    origin = load_site("cambridge_west").origin.geodetic()
    z = _arrays(np.random.default_rng(seed))
    back = arrays_from_frame(frame_from_arrays(z, 12_340_000, origin), origin)
    np.testing.assert_allclose(back.gnss_pos_ned, z.gnss_pos_ned, atol=1e-6)
    np.testing.assert_allclose(back.gyro, z.gyro)
    np.testing.assert_allclose(back.gnss_time_s, z.gnss_time_s, atol=1e-6)
    assert [bool(f) for f in back.fresh] == [bool(f) for f in z.fresh]


def test_sensor_packing_round_trip() -> None:
    """The flight stack's single-vector transport carries every field and flag."""
    z = _arrays(np.random.default_rng(3))
    back = unpack_sensors(pack_sensors(z))  # type: ignore[arg-type]
    for name in SensorArrays._fields:
        if name == "fresh":
            assert [bool(x) for x in back.fresh] == [bool(x) for x in z.fresh]
        else:
            np.testing.assert_allclose(getattr(back, name), getattr(z, name), err_msg=name)
