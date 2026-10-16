"""Simulation-based tuning of the stock stack (an offline analogue of ArduPlane AUTOTUNE).

Starting from the model-based configuration (``autoconfig``), each airframe flies a tuning
manoeuvre per axis with a small grid of parameter variants, all batched in one ``vmap``. Each
variant is scored against the axis's requirements, measured on the tuning manoeuvre on two
independent noise streams, and the variant with the largest worst-case margin is kept (the
worst ratio of value to limit; a small tracking-error term breaks ties). This is
requirements-driven tuning, as flight-control engineers tune to handling-qualities limits; a
weighted sum of costs instead trades one requirement against another unseen.

- roll: a ±20° bank doublet, then wings level. Rise time, overshoot, the error at the end of
  each half, and any oscillation left while holding level (as AUTOTUNE backs gains off when
  it detects oscillation: with servo deadband and lag, high gains limit-cycle). Variants
  scale the feed-forward and the rate P/I and set the time constant.
- yaw: a slow ±20° bank doublet (6 s each way): steady-bank and peak sideslip. Variants
  scale the coordination feed-forward and damper and set the sideslip integrator and the
  aileron-to-rudder mix (KFF_RDDRMIX) against adverse yaw.
- pitch: a ±4° doublet around trim at fixed throttle: rise time, overshoot, oscillation.
- TECS: a height doublet: airspeed excursion and height overshoot. Variants set the time
  constant, the speed weighting (2 = speed priority, ArduPilot's setting for gliders) and
  the climb-rate limit.

The limits are the V11 ones where V11 has one. The axes are tuned in that order, each with
the previous choices in place. The manoeuvres (doublets at other amplitudes, calm air) differ
from the V20 acceptance flights (steps, L1 tracking, turbulence, altitude and speed steps),
so V20 verifies rather than repeats what the tuner saw.
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
from aeroadapt.harness.acceptance import band_rms, step_metrics
from aeroadapt.harness.closed_loop import LoopLog
from aeroadapt.harness.fleet_acceptance import (
    ALT0,
    CONFIG,
    _fly,
    _initial,
    _stack,
    airframe_keys,
    alt_step,
)
from aeroadapt.scenarios.site import Site
from aeroadapt.sim.aerojax import EnvParams, make_env
from aeroadapt.stack.autopilot import MODE_GUIDED
from aeroadapt.stack.config import StackCfg, StackParams, build_stack_params
from aeroadapt.stack.types import LAT_DIRECT, VERT_DIRECT, VERT_TECS, Target, make_mission, target

T0 = 15.0
ROLL_AMP = math.radians(20.0)
PITCH_AMP = math.radians(4.0)
TAU_REF = 0.3
HOLD_S = 8.0  # level flight after the doublet
TIE_BREAK = 0.01  # cost per degree (or metre) of tracking RMS, below any requirement ratio
# Requirements (V11 limits where V11 has one): rise time (s), overshoot (fraction), error at
# the end of a doublet half (deg), residual 0.3-5 Hz oscillation (deg RMS).
LIMITS = {
    "roll": {"rise": 1.0, "overshoot": 0.10, "end_error": 1.0, "cycle": 0.25},
    "pitch": {"rise": 0.8, "overshoot": 0.15, "cycle": 0.25},
}
BETA_LIMIT_DEG = 3.0
TECS_SPEED_LIMIT = 0.08  # airspeed excursion / cruise speed
TECS_OVERSHOOT_LIMIT = 0.15  # height overshoot / height step
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
    first = t < T0 + half  # the first half is a step from level to +amp
    rise, ovs, _ = step_metrics(t[first], y[first], amp, t0=T0)
    ends = ((t >= T0 + half - 0.5) & (t < T0 + half)) | (
        (t >= T0 + 2 * half - 0.5) & (t < T0 + 2 * half)
    )
    err = np.degrees(y - _doublet(t, amp, half))
    hold = t >= T0 + 2 * half + 2.0
    value = {
        "rise": rise,
        "overshoot": ovs,
        "end_error": float(np.abs(err[ends]).mean()),
        "cycle": math.degrees(band_rms(y[hold], float(t[1] - t[0]))),
    }
    worst = max(value[q] / lim for q, lim in LIMITS[axis].items())
    w = t >= T0 - 1.0
    track = np.degrees(y[w] - _reference(t, amp, half, TAU_REF)[w])
    return float(worst + TIE_BREAK * np.sqrt(np.mean(track**2)))


def _cost_yaw(log: LoopLog, k: int, t: np.ndarray, half: float) -> float:
    b = np.degrees(np.abs(log.beta[k]))
    steady = ((t >= T0 + half - 3.0) & (t < T0 + half)) | (
        (t >= T0 + 2 * half - 3.0) & (t < T0 + 2 * half)
    )
    turning = ((t >= T0 + 2.0) & (t < T0 + half)) | ((t >= T0 + half + 2.0) & (t < T0 + 2 * half))
    worst = max(float(b[steady].mean()), float(b[turning].max())) / BETA_LIMIT_DEG
    return float(worst + TIE_BREAK * b[t >= T0 - 1.0].mean())


def _cost_tecs(log: LoopLog, k: int, t: np.ndarray, sp: StackParams) -> float:
    v0 = float(sp.tecs.v_cruise)
    w = t >= T0
    h = -log.pos_ned[k][:, 2]
    dh_step = 0.5 * float(alt_step(v0))
    up = (t >= T0) & (t < T0 + AXES["tecs"][0])
    dv = float(np.abs(log.va[k][w] - v0).max()) / v0
    over = max(float(h[up].max()) - ALT0 - dh_step, 0.0) / dh_step
    worst = max(dv / TECS_SPEED_LIMIT, over / TECS_OVERSHOOT_LIMIT)
    track = float(np.sqrt(np.mean((h[w] - log.alt_target[k][w]) ** 2)))
    return worst + TIE_BREAK * track


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
    "yaw_mix": (("att.yaw_mix",), ("attitude.yaw.rudder_mix",), "set"),
    "tecs_tc": (("tecs.time_const",), ("tecs.time_const_s",), "set"),
    "tecs_spd": (("tecs.spd_weight",), ("tecs.spd_weight",), "set"),
    "tecs_climb": (("tecs.climb_max",), ("tecs.climb_max_mps",), "scale"),
}
# Axis → (seconds per doublet half, grid per key).
# Rate gains go down to 0.35x: with servo deadband and lag, high gains limit-cycle. Halves of
# 4 s let the attitude settle, so the end-of-half error shows steady-state error.
AXES: dict[str, tuple[float, dict[str, tuple[float, ...]]]] = {
    "roll": (4.0, {"roll_ff": (0.8, 1.0, 1.25), "roll_pi": (0.35, 0.5, 1.0, 1.6),
                   "roll_tau": (0.25, 0.3, 0.4)}),
    "yaw": (6.0, {"yaw_ff": (0.6, 0.8, 1.0, 1.25, 1.6), "yaw_slip": (0.02, 0.08, 0.2),
                  "yaw_mix": (0.0, 0.25, 0.5)}),
    "pitch": (4.0, {"pitch_ff": (0.8, 1.0, 1.25), "pitch_pi": (0.35, 0.5, 1.0, 1.6),
                    "pitch_tau": (0.22, 0.3, 0.4)}),
    "tecs": (15.0, {"tecs_tc": (3.0, 5.0, 8.0), "tecs_spd": (1.0, 1.5, 2.0),
                    "tecs_climb": (0.4, 0.6, 1.0)}),
}  # fmt: skip
# Each variant flies on this many independent noise streams and is scored on its worst: a
# variant that meets the requirements only on a lucky draw of turbulence and sensor noise is
# not kept.
NOISE_STREAMS = 2


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
    keys: Array,
) -> np.ndarray:
    """Cost of each run (airframe ``idx[r]`` flying with stack ``sps[r]``). Every variant of an
    airframe flies with the airframe's key: the variants are compared on the same noise."""
    half = AXES[axis][0]
    n = round((T0 + 2 * half + HOLD_S) / CONFIG.dt_control)
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
                  guided, 5, keys[ii])  # fmt: skip
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
    ids = [a.id for a in airframes]
    sim0, z0, _ = _initial(vps, env, v0, airframe_keys(ids, 0, 0))
    streams = [airframe_keys(ids, 0, 1 + k) for k in range(NOISE_STREAMS)]
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
        costs = np.max(
            [_evaluate(vps, runs, env, sim0, z0, np.array(idx), axis, k) for k in streams], axis=0
        )
        costs = costs.reshape(n, len(combos))
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
