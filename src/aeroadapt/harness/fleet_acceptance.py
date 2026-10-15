"""V20: auto-configured stock stack acceptance across a fleet (M2 spec §6 and §10).

A batched, reduced V11: every airframe flies with its own auto-configured parameters, all
airframes of a batch in one ``vmap``. Calm air: roll step, pitch step, altitude step, speed
step, limit cycles and L1 tracking on a box mission. Light turbulence: L1 tracking and servo
saturation on the same box. Dimensional limits scale with the cruise speed (the M1 V11
values are for the Aerosonde at 25 m/s); times and percentages are the V11 limits.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from aeroadapt.airframes import Airframe, VehicleParams, to_vehicle_params
from aeroadapt.core import quaternion
from aeroadapt.faultlab import no_faults
from aeroadapt.harness.acceptance import band_rms, leg_xtrack, settled_mask, step_metrics
from aeroadapt.harness.closed_loop import LoopLog, closed_loop
from aeroadapt.scenarios.site import Site
from aeroadapt.sim.aerojax import SimConfig, init_state, make_env
from aeroadapt.sim.aerojax.actuators import target_to_command
from aeroadapt.sim.aerojax.atmosphere import isa
from aeroadapt.sim.aerojax.battery import open_circuit_voltage
from aeroadapt.sim.aerojax.environment import Environment, EnvParams
from aeroadapt.sim.aerojax.state import RigidBodyState
from aeroadapt.sim.aerojax.trim import trim_level
from aeroadapt.stack.autopilot import MODE_AUTO, MODE_GUIDED, stack_init
from aeroadapt.stack.config import StackCfg, StackParams, build_stack_params
from aeroadapt.stack.types import (
    LAT_DIRECT,
    VERT_DIRECT,
    VERT_TECS,
    Target,
    make_mission,
    target,
)

T_STEP = 20.0
ALT0 = 120.0
V_REF = 25.0  # the V11 reference airframe's cruise speed


def alt_step(v0: float | Array) -> float | Array:
    """The V11 altitude step (20 m at 25 m/s) scaled with V² (the same energy change relative
    to the kinetic energy), at least 5 m."""
    return jnp.maximum(20.0 * (v0 / V_REF) ** 2, 5.0)


LIGHT_W20_MPS = 15 * 0.514444
CONFIG = SimConfig()


@dataclass(frozen=True)
class FleetGuided:
    """Per-airframe guided schedules: targets come from each airframe's stack parameters."""

    kind: str  # roll | pitch | alt | speed | level

    def __call__(self, t: Array, sp: StackParams) -> Target:
        after = t >= T_STEP
        v0 = sp.tecs.v_cruise
        level = dict(lateral=LAT_DIRECT, roll_dem=0.0, roll_limit=math.radians(45.0))
        tecs = dict(vertical=VERT_TECS, alt=ALT0, speed=v0)
        if self.kind == "roll":
            return target(
                **{**level, "roll_dem": jnp.where(after, math.radians(30.0), 0.0)}, **tecs
            )
        if self.kind == "pitch":
            pd = sp.tecs.pitch_trim + jnp.where(after, math.radians(5.0), 0.0)
            return target(**level, vertical=VERT_DIRECT, pitch_dem=pd, throttle=sp.tecs.thr_cruise)
        if self.kind == "alt":
            return target(**level, **{**tecs, "alt": ALT0 + jnp.where(after, alt_step(v0), 0.0)})
        if self.kind == "speed":
            return target(**level, **{**tecs, "speed": v0 * jnp.where(after, 1.2, 1.0)})
        return target(**level, **tecs)


@dataclass
class AirframeResult:
    airframe: str
    v_cruise: float
    checks: dict[str, tuple[float, float, bool]] = field(default_factory=dict)  # value, limit, ok

    @property
    def passed(self) -> bool:
        return all(ok for _, _, ok in self.checks.values())

    def failures(self) -> list[str]:
        return [k for k, (_, _, ok) in self.checks.items() if not ok]

    def add(self, name: str, value: float, limit: float, upper: bool = True) -> None:
        ok = bool(value <= limit) if upper else bool(value >= limit)
        self.checks[name] = (float(value), float(limit), ok and math.isfinite(value))


def airframe_keys(ids: Sequence[str], seed: int, stream: int) -> Array:
    """One PRNG key per airframe, from its id: an airframe meets the same turbulence and sensor
    noise however a fleet is batched, so runs are reproducible and comparisons paired."""
    base = jax.random.fold_in(jax.random.PRNGKey(seed), stream)
    return jnp.stack([jax.random.fold_in(base, zlib.crc32(i.encode()) & 0x7FFFFFFF) for i in ids])


def _stack(trees: Sequence) -> object:
    return jax.tree.map(lambda *x: jnp.stack([jnp.asarray(v) for v in x]), *trees)


@jax.jit
def _trim_batch(p, rho, g, v_supply, va):
    env = Environment(rho, g)
    return jax.vmap(lambda pi, vs, v: trim_level(pi, env, vs, v))(p, v_supply, va)


def _initial(vps: Sequence[VehicleParams], env: EnvParams, v0: np.ndarray, keys: Array):
    """Batched straight-and-level start at ALT0 heading north (trim at the site density)."""
    _, _, rho = isa(env.isa, env.isa.alt_origin_msl_m + ALT0)
    v_full = jnp.asarray([float(open_circuit_voltage(v.battery, jnp.asarray(1.0))) for v in vps])
    with jax.enable_x64(True):
        p = jax.tree.map(lambda x: jnp.asarray(x, jnp.float64), _stack([v.airframe for v in vps]))
        z, res = _trim_batch(p, float(rho), float(env.gravity), v_full.astype(jnp.float64),
                             jnp.asarray(v0, jnp.float64))  # fmt: skip
        z, res = np.asarray(z), np.asarray(res)
    states, cmds = [], []
    for i, vp in enumerate(vps):
        a, th = z[i, 0], z[i, 1]
        vel = v0[i] * np.array([math.cos(a), 0.0, math.sin(a)])
        q = quaternion.from_euler(z[i, 2], th, 0.0)
        states.append(
            RigidBodyState(jnp.array([0.0, 0.0, -ALT0]), jnp.asarray(vel), q, jnp.zeros(3))
        )
        cmds.append(target_to_command(vp.airframe, vp.actuators, jnp.asarray(z[i, 3:7])))

    def one(vp, rb, cmd, key):
        return init_state(vp, env, no_faults(), rb, cmd, key, 1.0, CONFIG)

    sim0, z0 = jax.vmap(one)(_stack(vps), _stack(states), jnp.stack(cmds), keys)
    return sim0, z0, res


def _fly(
    vp, env, sp, mission, sim0, z0, mode: int, n: int, guided, every: int, keys: Array
) -> LoopLog:
    st0 = jax.vmap(lambda s, z: stack_init(s, z, mode))(sp, z0)

    def one(vp, sp, mission, sim0, st0, z0, key):
        return closed_loop(
            vp, env, no_faults(), sp, mission, sim0, st0, z0, key,
            n_steps=n, guided=guided, config=CONFIG, log_every=every,
        )[2]  # fmt: skip

    return jax.tree.map(np.asarray, jax.vmap(one)(vp, sp, mission, sim0, st0, z0, keys))


def run_v20(
    airframes: Sequence[Airframe], cfgs: Sequence[StackCfg], site: Site, seed: int = 0
) -> list[AirframeResult]:
    n = len(airframes)
    vps = [to_vehicle_params(a) for a in airframes]
    sps = [
        build_stack_params(c, a, site, CONFIG.dt_control)
        for c, a in zip(cfgs, airframes, strict=True)
    ]
    v0 = np.array([c.tecs.airspeed_cruise_mps for c in cfgs])
    results = [AirframeResult(a.id, float(v)) for a, v in zip(airframes, v0, strict=True)]

    def env_for(turbulent: bool) -> EnvParams:
        return make_env(
            alt_origin_msl_m=site.ground_msl_m,
            magnetic_ned_gauss=site.magnetic.field_ned_gauss,
            wind_from_deg=250.0,
            wind_speed_mps=4.0 if turbulent else 0.0,
            w20_mps=LIGHT_W20_MPS if turbulent else 0.0,
        )

    vp_b, sp_b = _stack(vps), _stack(sps)
    ids = [a.id for a in airframes]
    init_keys, fly_keys = airframe_keys(ids, seed, 0), airframe_keys(ids, seed, 1)
    dummy = _stack([make_mission([(60.0 * v * 10, 0.0, ALT0)]) for v in v0])
    # L1 box: four legs of one minute each at cruise speed.
    boxes = _stack(
        [make_mission([(s, 0, ALT0), (s, s, ALT0), (0, s, ALT0), (0, 0, ALT0)]) for s in 60.0 * v0]
    )
    box_seconds = float(4 * 60.0 * 1.15 + 20.0)
    for turbulent in (False, True):
        env = env_for(turbulent)
        sim0, z0, res = _initial(vps, env, v0, init_keys)
        tag = "turb" if turbulent else "calm"
        for i in range(n):
            results[i].add(f"start trim residual ({tag})", float(res[i]), 1e-6)

        def fly(
            kind: str, seconds: float, every: int = 2, env: EnvParams = env, sim0=sim0, z0=z0
        ) -> LoopLog:
            g = FleetGuided(kind) if kind != "l1" else None
            mission = boxes if kind == "l1" else dummy
            mode = MODE_AUTO if kind == "l1" else MODE_GUIDED
            return _fly(
                vp_b,
                env,
                sp_b,
                mission,
                sim0,
                z0,
                mode,
                round(seconds / CONFIG.dt_control),
                g,
                every,
                fly_keys,
            )

        if not turbulent:
            for kind, secs in (("roll", T_STEP + 8), ("pitch", T_STEP + 4), ("alt", T_STEP + 30),
                               ("speed", T_STEP + 25), ("level", 40.0)):  # fmt: skip
                lg = fly(kind, secs)
                for i in range(n):
                    _score(results[i], kind, jax.tree.map(lambda x, i=i: x[i], lg), sps[i], v0[i])
        lg = fly("l1", box_seconds, every=10)
        for i in range(n):
            one = jax.tree.map(lambda x, i=i: x[i], lg)
            mask = settled_mask(one, 25.0)
            square = jax.tree.map(lambda x, i=i: x[i], boxes)
            xt = leg_xtrack(one, square, np.zeros(2))[mask]
            rms = float(np.sqrt(np.mean(xt**2))) if xt.size else float("inf")
            # The autopilot can only track as well as its position fix: the limits scale with
            # the airframe's GNSS accuracy (σ = 1 m for the V11 reference).
            gnss = max(1.0, float(vps[i].sensors.gnss_sigma_h))
            results[i].add(f"L1 cross-track RMS ({tag})", rms, (5.0 if turbulent else 2.0) * gnss)
            sat = np.any(np.abs(one.cmd[:, :3]) >= 0.98, axis=1)
            results[i].add(f"servo saturation ({tag})", 100 * float(np.mean(sat)), 5.0)
            results[i].add(f"no ground contact ({tag})", float(one.done.any()), 0.0)
    return results


def _score(r: AirframeResult, kind: str, m: LoopLog, sp: StackParams, v0: float) -> None:
    if bool(m.done.any()):
        r.add(f"{kind}: flight completed", 0.0, 1.0, upper=False)
        return
    if kind == "roll":
        rise, ovs, sse = step_metrics(m.t, m.euler[:, 0], math.radians(30.0))
        turn = m.t > T_STEP + 3.0
        r.add("roll step rise 10-90%", rise, 1.0)
        r.add("roll step overshoot", 100 * ovs, 10.0)
        r.add("roll step steady-state error", math.degrees(sse), 1.0)
        r.add("|beta| in 30 deg turn", math.degrees(np.abs(m.beta[turn]).max()), 3.0)
    elif kind == "pitch":
        rise, ovs, _ = step_metrics(m.t, m.euler[:, 1], float(sp.tecs.pitch_trim) + math.radians(5))
        r.add("pitch step rise 10-90%", rise, 0.8)
        r.add("pitch step overshoot", 100 * ovs, 15.0)
    elif kind == "alt":
        h = -m.pos_ned[:, 2]
        dh = float(alt_step(v0))
        late = m.t >= T_STEP + 20.0
        r.add("alt step error after 20 s", np.abs(h[late] - ALT0 - dh).max(), max(0.5, 0.1 * dh))
        r.add("alt step overshoot", max(float(h.max()) - ALT0 - dh, 0.0), max(0.75, 0.15 * dh))
        r.add("alt step airspeed deviation", np.abs(m.va - v0).max(), 0.08 * v0)
    elif kind == "speed":
        late = m.t >= T_STEP + 15.0
        r.add(
            "speed step error after 15 s", np.abs(m.va[late] - 1.2 * v0).max(), max(0.3, 0.02 * v0)
        )
        r.add("speed step altitude deviation", np.abs(-m.pos_ned[:, 2] - ALT0).max(), 3.0)
    elif kind == "level":
        w = m.t > 10.0
        dt = float(m.t[1] - m.t[0])
        osc = max(band_rms(m.euler[w, 0], dt), band_rms(m.euler[w, 1], dt))
        r.add("limit cycle 0.3-5 Hz attitude RMS", math.degrees(osc), 0.25)
