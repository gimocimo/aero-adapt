"""Simulator throughput benchmark (V15).

Measures, in float32 (the intended training precision):

* batched open-loop rollouts with ``vmap`` + ``lax.scan`` at several batch sizes, reported
  as control steps and physics steps per second;
* host-stepped latency: one jitted ``sim_step`` called from Python per control period,
  which is the M1 imperative-shell mode. Reported as a real-time factor.

Usage: ``uv run python tools/bench.py [--batches 1,64,1024] [--steps 1000]``
"""

from __future__ import annotations

import argparse
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

from aeroadapt.airframes import load_airframe, to_vehicle_params
from aeroadapt.faultlab import no_faults
from aeroadapt.scenarios import load_site
from aeroadapt.sim.aerojax import Environment, SimConfig, init_state, make_env, sim_step
from aeroadapt.sim.aerojax.actuators import target_to_command
from aeroadapt.sim.aerojax.atmosphere import isa
from aeroadapt.sim.aerojax.battery import open_circuit_voltage
from aeroadapt.sim.aerojax.trim import trim


def setup():
    vp = to_vehicle_params(load_airframe("aerosonde_bm"))
    site = load_site("cambridge_west")
    env = make_env(
        alt_origin_msl_m=site.origin.h_ellipsoid_m - 46.0,
        magnetic_ned_gauss=site.magnetic.field_ned_gauss,
        wind_from_deg=250.0,
        wind_speed_mps=5.0,
        w20_mps=15 * 0.514444,
    )
    _, _, rho = isa(env.isa, env.isa.alt_origin_msl_m + 120.0)
    v_full = float(open_circuit_voltage(vp.battery, jnp.asarray(1.0)))
    tr = trim(vp.airframe, Environment(float(rho), env.gravity), 25.0, h=120.0, v_supply=v_full)
    cmd = target_to_command(vp.airframe, vp.actuators, jnp.asarray([*tr.controls]))
    rb0 = jax.tree_util.tree_map(jnp.asarray, tr.state)
    return vp, env, no_faults(), rb0, jnp.asarray(cmd)


def bench_batched(vp, env, faults, rb0, cmd, batch: int, steps: int, cfg: SimConfig):
    def run(keys):
        def one(key):
            k0, k1 = jax.random.split(key)
            s, _ = init_state(vp, env, faults, rb0, cmd, k0, config=cfg)

            def body(s, k):
                s, _, info = sim_step(vp, env, faults, s, cmd, k, cfg)
                return s, info.truth.alt_agl

            return jax.lax.scan(body, s, jax.random.split(k1, steps))[1]

        return jax.vmap(one)(keys)

    fn = jax.jit(run)
    keys = jax.random.split(jax.random.PRNGKey(0), batch)
    t0 = time.perf_counter()
    jax.block_until_ready(fn(keys))
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(jax.random.split(jax.random.PRNGKey(1), batch)))
    run_s = time.perf_counter() - t0
    assert np.isfinite(np.asarray(out)).all()
    return compile_s, batch * steps / run_s


def bench_host_stepped(vp, env, faults, rb0, cmd, steps: int, cfg: SimConfig):
    step = jax.jit(sim_step, static_argnames="config")
    s, _ = init_state(vp, env, faults, rb0, cmd, jax.random.PRNGKey(0), config=cfg)
    keys = jax.random.split(jax.random.PRNGKey(1), steps + 1)
    s, *_ = step(vp, env, faults, s, cmd, keys[0], config=cfg)  # compile
    jax.block_until_ready(s)
    t0 = time.perf_counter()
    for k in keys[1:]:
        s, *_ = step(vp, env, faults, s, cmd, k, config=cfg)
    jax.block_until_ready(s)
    per_step = (time.perf_counter() - t0) / steps
    return per_step, cfg.dt_control / per_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,64,1024")
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()
    cfg = SimConfig()
    vp, env, faults, rb0, cmd = setup()
    dev = jax.devices()[0]
    print(
        f"device: {dev.platform} ({dev.device_kind}) · {platform.machine()} · jax {jax.__version__}"
        f" · float{'64' if jax.config.jax_enable_x64 else '32'}"
    )
    physics_hz = 1 / cfg.dt_physics
    rates = f"physics {physics_hz:.0f} Hz x {cfg.substeps} substeps"
    print(f"{rates} → control {cfg.control_hz:.0f} Hz")
    print(f"{'batch':>6} {'compile s':>10} {'ctrl steps/s':>14} {'physics steps/s':>16}")
    for b in (int(x) for x in args.batches.split(",")):
        compile_s, sps = bench_batched(vp, env, faults, rb0, cmd, b, args.steps, cfg)
        print(f"{b:>6} {compile_s:>10.1f} {sps:>14,.0f} {sps * cfg.substeps:>16,.0f}")
    per_step, rtf = bench_host_stepped(vp, env, faults, rb0, cmd, 2000, cfg)
    print(f"host-stepped: {per_step * 1e6:.0f} µs per control step → real-time factor {rtf:.0f}x")


if __name__ == "__main__":
    main()
