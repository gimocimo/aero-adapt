"""Simulation-based tuning of the attitude loops (an offline analogue of ArduPlane AUTOTUNE).

Starting from the model-based configuration (``autoconfig``), each airframe flies doublets
with a small grid of gain variants, all batched in one ``vmap``, and keeps the variant that
best follows a reference first-order response:

- roll: a ±20° bank doublet; variants scale the feed-forward, the rate P/I and the time
  constant;
- yaw: the same doublet; variants scale the coordination feed-forward and damper, scored
  on sideslip;
- pitch: a ±4° doublet around trim at fixed throttle; variants as for roll.

The tuning manoeuvres (doublets, model following) differ from the V20 acceptance tests
(steps, L1 tracking, turbulence, altitude and speed changes), so V20 is not graded on the
data the tuner optimised.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from aeroadapt.airframes import Airframe, to_vehicle_params
from aeroadapt.harness.fleet_acceptance import ALT0, CONFIG, _fly, _initial, _stack
from aeroadapt.scenarios.site import Site
from aeroadapt.sim.aerojax import make_env
from aeroadapt.stack.autopilot import MODE_GUIDED
from aeroadapt.stack.config import StackCfg, StackParams, build_stack_params
from aeroadapt.stack.types import LAT_DIRECT, VERT_DIRECT, VERT_TECS, Target, make_mission, target

T0 = 15.0
HALF = 3.0
YAW_HALF = 6.0  # yaw is scored on a slower doublet: mostly steady banked flight
ROLL_AMP = math.radians(20.0)
PITCH_AMP = math.radians(4.0)
TAU_REF = {"roll": 0.4, "pitch": 0.3}
ROLL_GRID = list(itertools.product((0.8, 1.0, 1.25), (0.5, 1.0, 1.6), (0.3, 0.4)))  # ff, p, tc
PITCH_GRID = list(itertools.product((0.8, 1.0, 1.25), (0.5, 1.0, 1.6), (0.22, 0.3)))
YAW_GRID = (0.6, 0.8, 1.0, 1.25, 1.6)
CHUNK_RUNS = 900  # closed loops per vmap (memory)


def _doublet(t: Array, amp: float) -> Array:
    return jnp.where(
        (t >= T0) & (t < T0 + HALF),
        amp,
        jnp.where((t >= T0 + HALF) & (t < T0 + 2 * HALF), -amp, 0.0),
    )


@dataclass(frozen=True)
class Doublet:
    axis: str  # roll | pitch

    def __call__(self, t: Array, sp: StackParams) -> Target:
        level = dict(lateral=LAT_DIRECT, roll_limit=math.radians(45.0))
        if self.axis == "roll":
            return target(
                **level,
                roll_dem=_doublet(t, ROLL_AMP),
                vertical=VERT_TECS,
                alt=ALT0,
                speed=sp.tecs.v_cruise,
            )
        pd = sp.tecs.pitch_trim + _doublet(t, PITCH_AMP)
        return target(
            **level, roll_dem=0.0, vertical=VERT_DIRECT, pitch_dem=pd, throttle=sp.tecs.thr_cruise
        )


def _reference(t: np.ndarray, amp: float, tau: float) -> np.ndarray:
    """First-order response (time constant ``tau``) to the doublet command."""
    cmd = np.where(
        (t >= T0) & (t < T0 + HALF),
        amp,
        np.where((t >= T0 + HALF) & (t < T0 + 2 * HALF), -amp, 0.0),
    )
    y = np.zeros_like(t)
    for k in range(1, len(t)):
        y[k] = y[k - 1] + (cmd[k - 1] - y[k - 1]) * (1 - math.exp(-(t[k] - t[k - 1]) / tau))
    return y


def _variant(
    sp: StackParams, axis: str, a: float, b: float = 1.0, c: float | None = None
) -> StackParams:
    at = sp.att
    if axis == "roll":
        at = at._replace(roll_ff=at.roll_ff * a, roll_p=at.roll_p * b, roll_i=at.roll_i * b,
                         roll_tau=at.roll_tau if c is None else c)  # fmt: skip
    elif axis == "pitch":
        at = at._replace(pitch_ff=at.pitch_ff * a, pitch_p=at.pitch_p * b, pitch_i=at.pitch_i * b,
                         pitch_tau=at.pitch_tau if c is None else c)  # fmt: skip
    else:
        at = at._replace(yaw_ff=at.yaw_ff * a, yaw_damp=at.yaw_damp * a)
    return sp._replace(att=at)


def _evaluate(vps, sps: list[StackParams], env, sim0, z0, idx: np.ndarray, axis: str) -> np.ndarray:
    """Cost of each run (airframe index ``idx[r]`` with stack ``sps[r]``)."""
    costs = np.empty(len(sps))
    half = YAW_HALF if axis == "yaw" else HALF
    n = round((T0 + 2 * half + 4.0) / CONFIG.dt_control)
    guided = Doublet(axis)
    for s in range(0, len(sps), CHUNK_RUNS):
        sl = slice(s, s + CHUNK_RUNS)
        ii = idx[sl]
        take = lambda tree, ii=ii: jax.tree.map(lambda x: x[ii], tree)  # noqa: E731
        mission = _stack([make_mission([(5000.0, 0.0, ALT0)])] * len(ii))
        lg = _fly(_stack([vps[i] for i in ii]), env, _stack(sps[sl]), mission, take(sim0), take(z0),
                  MODE_GUIDED, n, guided, 5, 0)  # fmt: skip
        t = np.asarray(lg.t[0])
        w = t >= T0 - 1.0
        for k in range(len(ii)):
            if bool(lg.done[k].any()):
                costs[s + k] = np.inf
                continue
            if axis == "yaw":
                # Mostly the steady part of each bank (its last 3 s), plus the transients.
                b = np.degrees(np.abs(lg.beta[k]))
                steady = ((t >= T0 + YAW_HALF - 3.0) & (t < T0 + YAW_HALF)) | (
                    (t >= T0 + 2 * YAW_HALF - 3.0) & (t < T0 + 2 * YAW_HALF)
                )
                costs[s + k] = float(b[steady].mean() + 0.2 * b[w].max())
                continue
            j = 0 if axis == "roll" else 1
            y = lg.euler[k][:, j] - (0.0 if axis == "roll" else float(sps[s + k].tecs.pitch_trim))
            amp = ROLL_AMP if axis == "roll" else PITCH_AMP
            ref = _reference(t, amp, TAU_REF[axis])
            err = np.degrees(y[w] - ref[w])
            costs[s + k] = float(np.sqrt(np.mean(err**2)))
    return costs


def autotune(airframes: Sequence[Airframe], cfgs: Sequence[StackCfg], site: Site) -> list[StackCfg]:
    vps = [to_vehicle_params(a) for a in airframes]
    sps = [
        build_stack_params(c, a, site, CONFIG.dt_control)
        for c, a in zip(cfgs, airframes, strict=True)
    ]
    env = make_env(
        alt_origin_msl_m=site.ground_msl_m, magnetic_ned_gauss=site.magnetic.field_ned_gauss
    )
    v0 = np.array([c.tecs.airspeed_cruise_mps for c in cfgs])
    sim0, z0, _ = _initial(vps, env, v0, 0)
    n = len(airframes)
    chosen = {i: {} for i in range(n)}
    for axis, grid in (
        ("roll", ROLL_GRID),
        ("yaw", [(g,) for g in YAW_GRID]),
        ("pitch", PITCH_GRID),
    ):
        runs, idx = [], []
        for i in range(n):
            for g in grid:
                runs.append(_variant(sps[i], axis, *g))
                idx.append(i)
        costs = _evaluate(vps, runs, env, sim0, z0, np.array(idx), axis).reshape(n, len(grid))
        for i in range(n):
            best = grid[int(np.argmin(costs[i]))] if np.isfinite(costs[i]).any() else grid[0]
            chosen[i][axis] = best
            sps[i] = _variant(sps[i], axis, *best)
    out = []
    for i, c in enumerate(cfgs):
        d = c.model_dump()
        ff, p, tc = chosen[i]["roll"]
        r = d["attitude"]["roll"]
        r.update(ff=r["ff"] * ff, p=r["p"] * p, i=r["i"] * p, tconst_s=tc)
        ff, p, tc = chosen[i]["pitch"]
        q = d["attitude"]["pitch"]
        q.update(ff=q["ff"] * ff, p=q["p"] * p, i=q["i"] * p, tconst_s=tc)
        (k,) = chosen[i]["yaw"]
        y = d["attitude"]["yaw"]
        y.update(ff=y["ff"] * k, damp=y["damp"] * k)
        out.append(StackCfg.model_validate(d))
    return out
