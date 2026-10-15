"""Simulation-based tuning of the stock stack (an offline analogue of ArduPlane AUTOTUNE).

Starting from the model-based configuration (``autoconfig``), each airframe flies a tuning
manoeuvre per axis with a small grid of parameter variants, all batched in one ``vmap``, and
keeps the variant with the lowest cost:

- roll: a ±20° bank doublet, scored on following a first-order reference (τ = 0.3 s) plus
  overshoot; variants scale the feed-forward and the rate P/I and set the time constant;
- yaw: a slow ±20° bank doublet (6 s each way), scored mostly on steady-bank sideslip;
  variants scale the coordination feed-forward and damper and set the sideslip integrator;
- pitch: a ±4° doublet around trim at fixed throttle, as for roll (τ = 0.3 s);
- TECS: a height doublet, scored on airspeed excursion and height tracking; variants set the
  time constant and the speed weighting.

The axes are tuned in that order, each with the previous choices in place. The manoeuvres
(doublets, model following) differ from the V20 acceptance tests (steps, L1 tracking,
turbulence, altitude and speed steps), so V20 is not graded on what the tuner optimised.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from aeroadapt.airframes import Airframe, VehicleParams, to_vehicle_params
from aeroadapt.harness.closed_loop import LoopLog
from aeroadapt.harness.fleet_acceptance import ALT0, CONFIG, _fly, _initial, _stack, alt_step
from aeroadapt.scenarios.site import Site
from aeroadapt.sim.aerojax import EnvParams, make_env
from aeroadapt.stack.autopilot import MODE_GUIDED
from aeroadapt.stack.config import StackCfg, StackParams, build_stack_params
from aeroadapt.stack.types import LAT_DIRECT, VERT_DIRECT, VERT_TECS, Target, make_mission, target

T0 = 15.0
ROLL_AMP = math.radians(20.0)
PITCH_AMP = math.radians(4.0)
TAU_REF = 0.3
CHUNK_RUNS = 900  # closed loops per vmap (memory)


@dataclass(frozen=True)
class Manoeuvre:
    """A guided doublet on one axis (static under jit; amplitudes come from the stack)."""

    axis: str  # roll | yaw | pitch | tecs
    half: float  # seconds per half of the doublet

    def __call__(self, t: Array, sp: StackParams) -> Target:
        first = (t >= T0) & (t < T0 + self.half)
        second = (t >= T0 + self.half) & (t < T0 + 2 * self.half)
        sign = jnp.where(first, 1.0, jnp.where(second, -1.0, 0.0))
        level = dict(lateral=LAT_DIRECT, roll_limit=math.radians(45.0))
        v0 = sp.tecs.v_cruise
        if self.axis in ("roll", "yaw"):
            return target(**level, roll_dem=sign * ROLL_AMP, vertical=VERT_TECS, alt=ALT0, speed=v0)
        if self.axis == "pitch":
            pd = sp.tecs.pitch_trim + sign * PITCH_AMP
            return target(
                **level,
                roll_dem=0.0,
                vertical=VERT_DIRECT,
                pitch_dem=pd,
                throttle=sp.tecs.thr_cruise,
            )
        dh = 0.5 * alt_step(v0) * jnp.maximum(sign, 0.0)  # up, then back to the start height
        return target(**level, roll_dem=0.0, vertical=VERT_TECS, alt=ALT0 + dh, speed=v0)


def _doublet(t: np.ndarray, amp: float, half: float) -> np.ndarray:
    first = (t >= T0) & (t < T0 + half)
    second = (t >= T0 + half) & (t < T0 + 2 * half)
    return np.where(first, amp, np.where(second, -amp, 0.0))


def _reference(t: np.ndarray, amp: float, half: float, tau: float) -> np.ndarray:
    """First-order response (time constant ``tau``) to the doublet command."""
    cmd = _doublet(t, amp, half)
    y = np.zeros_like(t)
    for k in range(1, len(t)):
        y[k] = y[k - 1] + (cmd[k - 1] - y[k - 1]) * (1 - math.exp(-(t[k] - t[k - 1]) / tau))
    return y


def _cost_attitude(
    log: LoopLog, k: int, t: np.ndarray, sp: StackParams, axis: str, half: float
) -> float:
    j = 0 if axis == "roll" else 1
    amp = ROLL_AMP if axis == "roll" else PITCH_AMP
    y = log.euler[k][:, j] - (0.0 if axis == "roll" else float(sp.tecs.pitch_trim))
    w = t >= T0 - 1.0
    err = np.degrees(y[w] - _reference(t, amp, half, TAU_REF)[w])
    over = max(0.0, float(np.degrees(np.abs(y[w]).max() - amp)))
    return float(np.sqrt(np.mean(err**2)) + 0.5 * over)


def _cost_yaw(log: LoopLog, k: int, t: np.ndarray, half: float) -> float:
    b = np.degrees(np.abs(log.beta[k]))
    first = (t >= T0 + half - 3.0) & (t < T0 + half)
    second = (t >= T0 + 2 * half - 3.0) & (t < T0 + 2 * half)
    return float(b[first | second].mean() + 0.2 * b[t >= T0 - 1.0].max())


def _cost_tecs(log: LoopLog, k: int, t: np.ndarray, sp: StackParams) -> float:
    v0 = float(sp.tecs.v_cruise)
    w = t >= T0
    dv = float(np.abs(log.va[k][w] - v0).max()) / v0
    dh = float(np.sqrt(np.mean((-log.pos_ned[k][w, 2] - log.alt_target[k][w]) ** 2)))
    return 100.0 * dv + 0.3 * dh


# Parameters per tuning key: (StackParams paths, config paths, "scale" or "set").
_PATHS: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
    "roll_ff": (("att.roll_ff",), ("attitude.roll.ff",), "scale"),
    "roll_pi": (("att.roll_p", "att.roll_i"), ("attitude.roll.p", "attitude.roll.i"), "scale"),
    "roll_tau": (("att.roll_tau",), ("attitude.roll.tconst_s",), "set"),
    "pitch_ff": (("att.pitch_ff",), ("attitude.pitch.ff",), "scale"),
    "pitch_pi": (("att.pitch_p", "att.pitch_i"), ("attitude.pitch.p", "attitude.pitch.i"), "scale"),
    "pitch_tau": (("att.pitch_tau",), ("attitude.pitch.tconst_s",), "set"),
    "yaw_ff": (("att.yaw_ff", "att.yaw_damp"), ("attitude.yaw.ff", "attitude.yaw.damp"), "scale"),
    "yaw_slip": (("att.yaw_slip_i",), ("attitude.yaw.slip_i",), "set"),
    "tecs_tc": (("tecs.time_const",), ("tecs.time_const_s",), "set"),
    "tecs_spd": (("tecs.spd_weight",), ("tecs.spd_weight",), "set"),
}
# Axis → (seconds per doublet half, grid per key).
AXES: dict[str, tuple[float, dict[str, tuple[float, ...]]]] = {
    "roll": (3.0, {"roll_ff": (0.8, 1.0, 1.25), "roll_pi": (0.5, 1.0, 1.6),
                   "roll_tau": (0.25, 0.3, 0.4)}),
    "yaw": (6.0, {"yaw_ff": (0.6, 0.8, 1.0, 1.25), "yaw_slip": (0.02, 0.08, 0.2)}),
    "pitch": (3.0, {"pitch_ff": (0.8, 1.0, 1.25), "pitch_pi": (0.5, 1.0, 1.6),
                    "pitch_tau": (0.22, 0.3)}),
    "tecs": (15.0, {"tecs_tc": (3.0, 5.0, 8.0), "tecs_spd": (1.0, 1.5)}),
}  # fmt: skip


def _apply(sp: StackParams, choice: dict[str, float]) -> StackParams:
    for key, value in choice.items():
        paths, _, mode = _PATHS[key]
        for path in paths:
            group, field = path.split(".")
            sub = getattr(sp, group)
            new = getattr(sub, field) * value if mode == "scale" else value
            sp = sp._replace(**{group: sub._replace(**{field: new})})
    return sp


def _apply_cfg(cfg: dict[str, Any], choice: dict[str, float]) -> None:
    for key, value in choice.items():
        _, paths, mode = _PATHS[key]
        for path in paths:
            *head, field = path.split(".")
            node = cfg
            for h in head:
                node = node[h]
            node[field] = node[field] * value if mode == "scale" else value


def _evaluate(
    vps: Sequence[VehicleParams],
    sps: list[StackParams],
    env: EnvParams,
    sim0: Any,
    z0: Any,
    idx: np.ndarray,
    axis: str,
) -> np.ndarray:
    """Cost of each run (airframe ``idx[r]`` flying with stack ``sps[r]``)."""
    half = AXES[axis][0]
    n = round((T0 + 2 * half + 4.0) / CONFIG.dt_control)
    guided = Manoeuvre(axis, half)
    score: Callable[[LoopLog, int, np.ndarray, StackParams], float]
    if axis == "yaw":
        score = lambda lg, k, t, sp: _cost_yaw(lg, k, t, half)  # noqa: E731
    elif axis == "tecs":
        score = _cost_tecs
    else:
        score = lambda lg, k, t, sp: _cost_attitude(lg, k, t, sp, axis, half)  # noqa: E731
    costs = np.empty(len(sps))
    for s in range(0, len(sps), CHUNK_RUNS):
        sl = slice(s, s + CHUNK_RUNS)
        ii = idx[sl]
        take = lambda tree, ii=ii: jax.tree.map(lambda x: x[ii], tree)  # noqa: E731
        mission = _stack([make_mission([(5000.0, 0.0, ALT0)])] * len(ii))
        vp_b = _stack([vps[i] for i in ii])
        lg = _fly(vp_b, env, _stack(sps[sl]), mission, take(sim0), take(z0), MODE_GUIDED, n,
                  guided, 5, 0)  # fmt: skip
        t = np.asarray(lg.t[0])
        for k in range(len(ii)):
            costs[s + k] = np.inf if bool(lg.done[k].any()) else score(lg, k, t, sps[s + k])
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
    chosen: list[dict[str, float]] = [{} for _ in range(n)]
    for axis, (_, grid) in AXES.items():
        keys = list(grid)
        combos = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*grid.values())]
        runs, idx = [], []
        for i in range(n):
            for c in combos:
                runs.append(_apply(sps[i], c))
                idx.append(i)
        costs = _evaluate(vps, runs, env, sim0, z0, np.array(idx), axis).reshape(n, len(combos))
        for i in range(n):
            if np.isfinite(costs[i]).any():
                best = combos[int(np.argmin(costs[i]))]
            else:  # every variant crashed: keep the model-based values
                best = {k: 1.0 for k in keys if _PATHS[k][2] == "scale"}
            chosen[i].update(best)
            sps[i] = _apply(sps[i], best)
    out = []
    for i, c in enumerate(cfgs):
        d = c.model_dump()
        _apply_cfg(d, chosen[i])
        d["attitude"]["yaw"]["imax"] = max(d["attitude"]["yaw"]["imax"], 0.3)
        out.append(StackCfg.model_validate(d))
    return out
