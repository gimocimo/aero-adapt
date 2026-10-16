"""Fleet throughput (V15 for M2, D9): a heterogeneous fleet with faults, batched.

Every lane flies one of the committed sample fleet's airframes (cycled) with its own mean
wind, turbulence level and fault (onset during the run), in float32. Two numbers per batch
size, both in control steps per second (one step = 10 ms of flight, 5 physics substeps):

- **simulator:** ``sim_step`` holding the trim command, which is what a reinforcement-learning
  environment step costs (the policy supplies the command);
- **closed loop:** simulator plus the stock flight stack (navigator, guidance, TECS, attitude
  loops, contingency) on a box mission.

Usage: ``uv run python tools/bench_fleet.py [--batches 64,256,1024] [--steps 1000]``. The GPU
run uses the same script (``notebooks/gpu_throughput.ipynb``).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import numpy as np
from pydantic import TypeAdapter

from aeroadapt.airframes import to_vehicle_params
from aeroadapt.airframes.fleet import load_fleet
from aeroadapt.bench.sampler import FAULT_KINDS
from aeroadapt.faultlab import build_fault_config
from aeroadapt.faultlab.schema import FaultSpec
from aeroadapt.harness.closed_loop import closed_loop
from aeroadapt.harness.fleet_acceptance import ALT0, _initial, _stack, airframe_keys
from aeroadapt.scenarios import load_site
from aeroadapt.sim.aerojax import SimConfig, init_state, make_env, sim_step
from aeroadapt.sim.aerojax.actuators import target_to_command
from aeroadapt.stack.autopilot import MODE_AUTO, stack_init
from aeroadapt.stack.config import StackCfg, build_stack_params
from aeroadapt.stack.types import make_mission

SAMPLE = Path(__file__).resolve().parents[1] / "configs" / "fleets" / "sample"
LEVELS = (0.0, 15.0, 30.0)  # turbulence W20 (kt): none, light, moderate


def lanes(batch: int, steps: int, seed: int = 0):
    """Per-lane parameters, environments, faults, start states and stacks."""
    afs = load_fleet(SAMPLE)
    cfgs = {
        (row := json.loads(line))["airframe"]: StackCfg.model_validate(row["stack"])
        for line in (SAMPLE / "stack_cfgs.jsonl").read_text().splitlines()
    }
    site = load_site("cambridge_west")
    cfg = SimConfig()
    rng = np.random.default_rng(seed)
    vps = [to_vehicle_params(a) for a in afs]
    v0 = np.array([cfgs[a.id].tecs.airspeed_cruise_mps for a in afs])
    calm = make_env(
        alt_origin_msl_m=site.ground_msl_m, magnetic_ned_gauss=site.magnetic.field_ned_gauss
    )
    sim1, _, _ = _initial(vps, calm, v0, airframe_keys([a.id for a in afs], seed, 0))
    pick = [i % len(afs) for i in range(batch)]
    adapter = TypeAdapter(FaultSpec)
    envs, faults, sps, missions = [], [], [], []
    for i in pick:
        a, v = afs[i], float(v0[i])
        envs.append(make_env(
            alt_origin_msl_m=site.ground_msl_m, magnetic_ned_gauss=site.magnetic.field_ned_gauss,
            wind_from_deg=float(rng.uniform(0, 360)), wind_speed_mps=float(rng.uniform(0, 0.4 * v)),
            w20_mps=float(rng.choice(LEVELS)) * 0.514444,
        ))  # fmt: skip
        kind = str(rng.choice(list(FAULT_KINDS)))
        spec = FAULT_KINDS[kind](rng, 0.2)
        spec["trigger"] = {"at_time_s": float(rng.uniform(0.2, 0.8)) * steps * cfg.dt_control}
        faults.append(build_fault_config([adapter.validate_python(spec)]))
        sps.append(build_stack_params(cfgs[a.id], a, site, cfg.dt_control))
        s = 60.0 * v
        missions.append(make_mission([(s, 0, ALT0), (s, s, ALT0), (0, s, ALT0), (0, 0, ALT0)]))
    take = lambda tree: jax.tree.map(lambda x: x[np.array(pick)], tree)  # noqa: E731
    vp_b = _stack([vps[i] for i in pick])
    return cfg, vp_b, _stack(envs), _stack(faults), _stack(sps), _stack(missions), take(sim1)


def timed(fn, *args) -> float:
    jax.block_until_ready(fn(*args))  # compile
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return time.perf_counter() - t0


def bench(batch: int, steps: int) -> tuple[float, float]:
    cfg, vp, env, faults, sp, mission, sim1 = lanes(batch, steps)
    keys = jax.random.split(jax.random.PRNGKey(1), batch)

    def start(vp, env, f, s, key):
        """Trim command and a start state with this lane's environment and fault."""
        cmd = target_to_command(vp.airframe, vp.actuators, s.act.buffer[0])
        sim0, z0 = init_state(vp, env, f, s.rb, cmd, key, config=cfg)
        return cmd, sim0, z0

    @jax.jit
    def open_loop(vp, env, faults, sim1, keys):
        def one(vp, env, f, s, key):
            cmd, s, _ = start(vp, env, f, s, key)

            def body(s, k):
                return sim_step(vp, env, f, s, cmd, k, cfg)[0], None

            return jax.lax.scan(body, s, jax.random.split(key, steps))[0].t

        return jax.vmap(one)(vp, env, faults, sim1, keys)

    @jax.jit
    def closed(vp, env, faults, sp, mission, sim1, keys):
        def one(vp, env, f, sp, mission, s, key):
            _, s0, z0 = start(vp, env, f, s, key)
            st0 = stack_init(sp, z0, MODE_AUTO)
            return closed_loop(vp, env, f, sp, mission, s0, st0, z0, key, n_steps=steps,
                               config=cfg, log_every=steps)[0].t  # fmt: skip

        return jax.vmap(one)(vp, env, faults, sp, mission, sim1, keys)

    t_open = timed(open_loop, vp, env, faults, sim1, keys)
    t_closed = timed(closed, vp, env, faults, sp, mission, sim1, keys)
    return batch * steps / t_open, batch * steps / t_closed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--batches", default="64,256,1024")
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()
    dev = jax.devices()[0]
    bits = "64" if jax.config.jax_enable_x64 else "32"
    print(f"device: {dev.platform} ({dev.device_kind}) · {platform.machine()} · "
          f"jax {jax.__version__} · float{bits}")  # fmt: skip
    print("heterogeneous fleet (10 airframes cycled) · per-lane wind, turbulence and one fault")
    print(f"{'batch':>7} {'simulator steps/s':>18} {'closed-loop steps/s':>20}")
    for b in (int(x) for x in args.batches.split(",")):
        sim, loop = bench(b, args.steps)
        print(f"{b:>7} {sim:>18,.0f} {loop:>20,.0f}", flush=True)


if __name__ == "__main__":
    main()
