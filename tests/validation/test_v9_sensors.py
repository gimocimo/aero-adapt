"""V9: sensor models (spec §12): noise statistics, multi-rate hold, GNSS latency, pitot physics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aeroadapt.airframes import VehicleParams
from aeroadapt.faultlab import FaultSpec, build_fault_config, no_faults
from aeroadapt.sim.aerojax.atmosphere import IsaParams, isa
from aeroadapt.sim.aerojax.sensors import SensorTruth, init_sensors, sample

DT_C, HZ = 0.01, 100.0
STD = IsaParams(0.0, 101_325.0, 0.0)


def _truth(k: int, alt: float = 120.0, u_air: float = 25.0, pos=None) -> SensorTruth:
    temp, p, rho = isa(STD, jnp.asarray(alt))
    return SensorTruth(
        t=jnp.asarray(k * DT_C),
        k=jnp.asarray(k, jnp.int32),
        pos_ned=jnp.asarray([0.0, 0.0, -alt] if pos is None else pos),
        vel_ned=jnp.zeros(3),
        quat=jnp.array([1.0, 0, 0, 0]),
        omega=jnp.array([0.1, -0.2, 0.3]),
        specific_force=jnp.array([0.0, 0.0, -9.80665]),
        static_pa=p,
        temp_k=temp,
        rho=rho,
        u_air=jnp.asarray(u_air),
        battery_v=jnp.asarray(48.0),
        battery_a=jnp.asarray(8.0),
        battery_mah=jnp.asarray(0.0),
        prop_omega=jnp.asarray(600.0),
        mag_ned=jnp.array([0.19171, 0.00411, 0.45455]),
    )


def _run(sp, pf, truths: list[SensorTruth], seed=0):
    key0, key = jax.random.split(jax.random.PRNGKey(seed))
    s, first = init_sensors(sp, pf, truths[0], DT_C, HZ, key0)
    outs = [first]
    step = jax.jit(lambda s, tr, k: sample(sp, pf, s, tr, DT_C, HZ, k))
    for tr, k in zip(truths[1:], jax.random.split(key, len(truths) - 1), strict=True):
        s, o = step(s, tr, k)
        outs.append(o)
    return jax.tree_util.tree_map(lambda *x: np.stack(x), *outs)


def test_noise_statistics(vehicle: VehicleParams) -> None:
    sp = vehicle.sensors._replace(
        gyro_bias_rw=0.0,
        accel_bias_rw=0.0,
        baro_drift_rw=0.0,
        gyro_bias_init=0.0,
        accel_bias_init=0.0,
    )
    out = _run(sp, no_faults().pitot, [_truth(k) for k in range(4000)])
    np.testing.assert_allclose(out.gyro.std(axis=0), sp.gyro_noise, rtol=0.05)
    np.testing.assert_allclose(out.accel.std(axis=0), sp.accel_noise, rtol=0.05)
    np.testing.assert_allclose(out.mag.std(axis=0), sp.mag_noise, rtol=0.05)
    np.testing.assert_allclose(out.gyro.mean(axis=0), [0.1, -0.2, 0.3], atol=5 * sp.gyro_noise / 60)
    baro = out.baro_pa[out.fresh.baro]  # independent samples only
    assert baro.std() == pytest.approx(sp.baro_noise, rel=0.1)


def test_multirate_freshness_and_hold(vehicle: VehicleParams) -> None:
    out = _run(vehicle.sensors, no_faults().pitot, [_truth(k) for k in range(100)])
    assert out.fresh.imu.all()
    np.testing.assert_array_equal(np.flatnonzero(out.fresh.gnss), np.arange(0, 100, 20))
    np.testing.assert_array_equal(np.flatnonzero(out.fresh.baro), np.arange(0, 100, 5))
    stale = ~out.fresh.gnss
    np.testing.assert_array_equal(
        out.gnss_pos_ned[stale][:19], np.repeat(out.gnss_pos_ned[:1], 19, 0)
    )


def test_gnss_latency_is_exact(vehicle: VehicleParams) -> None:
    sp = vehicle.sensors._replace(gnss_sigma_h=0.0, gnss_sigma_v=0.0, gnss_vel_sigma=0.0)
    truths = [_truth(k, pos=[float(k), 0.0, -120.0]) for k in range(100)]
    out = _run(sp, no_faults().pitot, truths)
    lat = round(sp.gnss_latency * HZ)  # 12 control steps
    fresh = np.flatnonzero(out.fresh.gnss)
    for k in fresh[fresh >= lat]:
        assert out.gnss_pos_ned[k, 0] == pytest.approx(k - lat, abs=1e-12)
        assert out.gnss_time_s[k] == pytest.approx((k - lat) * DT_C, abs=1e-12)


def _pitot_fault(variant: str, onset: float, **kw) -> FaultSpec:
    from pydantic import TypeAdapter

    return TypeAdapter(FaultSpec).validate_python(
        {"type": "pitot_blockage", "variant": variant, "trigger": {"at_time_s": onset}, **kw}
    )


def test_pitot_drain_blocked_reads_like_an_altimeter(vehicle: VehicleParams) -> None:
    sp = vehicle.sensors._replace(airspeed_noise=0.0)
    pf = build_fault_config([_pitot_fault("drain_blocked", onset=0.5)]).pitot
    # Level for 1 s (fault at 0.5 s), then a 50 m climb at constant true airspeed.
    alts = [120.0] * 100 + list(np.linspace(120.0, 170.0, 500)) + [170.0] * 50
    out = _run(sp, pf, [_truth(k, alt=a) for k, a in enumerate(alts)])
    _, p_lo, _ = isa(STD, jnp.asarray(120.0))
    _, p_hi, _ = isa(STD, jnp.asarray(170.0))
    expected_rise = float(p_lo - p_hi)
    rise = out.airspeed_dp_pa[-1] - out.airspeed_dp_pa[99]
    assert rise > 0  # indicated airspeed increases in a climb
    assert rise == pytest.approx(expected_rise, rel=0.01)


def test_pitot_drain_open_decays_with_tau(vehicle: VehicleParams) -> None:
    sp = vehicle.sensors._replace(airspeed_noise=0.0)
    pf = build_fault_config([_pitot_fault("drain_open", onset=0.0, tau_s=3.0)]).pitot
    out = _run(sp, pf, [_truth(k) for k in range(1001)])
    dp0 = out.airspeed_dp_pa[0]
    fresh = np.flatnonzero(out.fresh.airspeed)
    t = fresh * DT_C
    tau_fit = -(np.polyfit(t[1:], np.log(out.airspeed_dp_pa[fresh[1:]] / dp0), 1)[0] ** -1)
    assert tau_fit == pytest.approx(3.0, rel=0.05)


def test_pitot_ramp_blends(vehicle: VehicleParams) -> None:
    sp = vehicle.sensors._replace(airspeed_noise=0.0)
    pf = build_fault_config([_pitot_fault("drain_open", onset=0.0, ramp_s=1.0, tau_s=0.01)]).pitot
    out = _run(sp, pf, [_truth(k) for k in range(101)])
    dp0 = out.airspeed_dp_pa[0]
    assert out.airspeed_dp_pa[50] == pytest.approx(0.5 * dp0, rel=0.02)  # half-way, leaked out
    assert abs(out.airspeed_dp_pa[100]) < 1e-6 * dp0
