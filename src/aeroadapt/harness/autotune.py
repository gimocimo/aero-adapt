"""Simulation-based tuning of the stock stack (an offline analogue of ArduPlane AUTOTUNE).

Starting from the model-based configuration (``autoconfig``), each airframe flies a tuning
manoeuvre per axis with a small grid of parameter variants, all batched in one ``vmap``. Each
variant is scored against the axis's requirements, measured on the tuning manoeuvre on two
noise streams, and the variant with the largest worst-case margin is kept (the worst ratio of
value to limit; a small tracking-error term breaks ties). This is requirements-driven tuning,
as flight-control engineers tune to handling-qualities limits; a weighted sum of costs instead
trades one requirement against another unseen.

Each noise stream is a separate power-up: its own sensor turn-on biases (the accelerometer's,
σ up to 0.1 m/s² per axis, set the sideslip the rudder integrator holds and tilt the attitude
estimate) and its own sensor noise and GNSS error (the tuner flies in calm air). The streams
are drawn from ``TUNE_SEED``, apart from V20's, so V20 flies power-ups the tuner never saw.

- roll: a ±20° bank doublet, then wings level. Rise time, overshoot, the error at the end of
  each half, and any oscillation left while holding level (as AUTOTUNE backs gains off when
  it detects oscillation: with servo deadband and lag, high gains limit-cycle). Variants
  scale the feed-forward and the rate P/I and set the time constant.
- yaw: a slow ±30° bank doublet (6 s each way, V20's bank): steady-bank and peak sideslip.
  Variants scale the coordination feed-forward and damper and set the sideslip integrator
  (including none, ArduPlane's default YAW2SRV_INT = 0) and the aileron-to-rudder mix
  (KFF_RDDRMIX) against adverse yaw. The integrator nulls the *measured* lateral
  acceleration, so with an accelerometer bias it holds a sideslip of its own; without it, the
  sideslip an imperfect coordination leaves grows with the turn rate, hence V20's bank.
- roll, sustained: the slow ±30° doublet again, with the roll requirements and the error over
  the last 2 s of each half. Variants raise the rate I and P on their own. The first roll stage
  scales them together, and keeps them low where they limit-cycle; there the bank then
  wanders in a sustained turn, as the integrator is too slow to hold it.
- pitch: a ±4° doublet around trim at fixed throttle: rise time, overshoot, oscillation.
- pitch, in the loop: the pitch doublet and the TECS flight below together, each with its
  requirements. The TECS flight closes the whole loop (TECS, pitch loop, servo deadband),
  where the integrators hunt round the deadband; the pitch doublet, at fixed throttle, cannot
  see it. Variants raise the pitch rate P (damping) and lower its I. (A slower TECS time
  constant also stops it, at the cost of slower height and speed control.)
- TECS: a height step of the V11 size (scaled with V²), then back: airspeed excursion over
  the whole flight, overshoot, the height error 20-30 s after each step (V20's window), and
  any attitude oscillation in level flight before and after it. And a +20 % speed step, then
  back: the height deviation (V11: 3 m) and the speed error 15-25 s after each step. Variants
  set the time constant, the speed weighting (2 = speed priority, ArduPilot's setting for
  gliders) and the climb-rate limit.

The limits are the V11 ones where V11 has one. The stages are tuned in that order, each with
the previous choices in place; a stage's variant is scored on the worst of its manoeuvres.
The tuner shares several of V20's manoeuvres, with V20's limits: the TECS stage flies V20's
altitude step, speed step and limit-cycle window, and the yaw and sustained-roll stages V20's
30° bank. It does not fly the pitch step at V20's size, L1 tracking or turbulence. V20
re-flies the shared manoeuvres on power-ups and noise the tuner never saw (``TUNE_SEED``), so
it checks that the tuning generalises to new sensor draws, not to new manoeuvres.

Known limitation: the tracking tie-break is added per flight before the worst flight is taken.
The TECS flight's term (its height against the raw step target, which includes the climb ramp)
is a near-constant 0.01-0.06, so in the two stages with two flights it can tip the choice
between variants whose worst ratios differ by less than that. Combining the worst ratios first
and the tie-breaks after would fix it; it changes every tuned stack, so it waits for the next
tuner change (M3).
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
TUNE_SEED = 1_000_003  # the tuner's power-ups and noise, apart from V20's (seed 0, 1, …)
ROLL_AMP = math.radians(20.0)
BANK_AMP = math.radians(30.0)  # V20's turn
PITCH_AMP = math.radians(4.0)
TAU_REF = 0.3
HOLD_S = 8.0  # level flight after the doublet …
HOLD_TECS_S = 15.0  # … longer after the height doublet, to measure oscillation in level flight
LEVEL_FROM_S = 8.0  # level flight before a doublet, once the navigator has converged
TIE_BREAK = 0.01  # cost per degree (or metre) of tracking RMS (see the known limitation)
# Requirements (V11 limits where V11 has one): rise time (s), overshoot (fraction), error at
# the end of a doublet half (deg), residual 0.3-5 Hz oscillation (deg RMS).
LIMITS = {
    "roll": {"rise": 1.0, "overshoot": 0.10, "end_error": 1.0, "cycle": 0.25},
    "pitch": {"rise": 0.8, "overshoot": 0.15, "cycle": 0.25},
}
BETA_LIMIT_DEG = 3.0
TECS_SPEED_LIMIT = 0.08  # airspeed excursion / cruise speed
# Height overshoot and the error 20-30 s after each step (V20's window): fractions of the step,
# with V20's floors (m).
TECS_OVERSHOOT = (0.15, 0.75)
TECS_SETTLE = (0.10, 0.5)
SETTLE_S = 20.0
SETTLED_S = 10.0
SPEED_STEP = 0.2  # of cruise, V11's speed step …
SPEED_ALT_LIMIT_M = 3.0  # … the height it may lose or gain …
SPEED_SETTLE = (0.02, 0.3)  # … and its speed error 15-25 s after the step (fraction, floor m/s)
SPEED_SETTLE_S = 15.0
CYCLE_LIMIT_DEG = 0.25  # attitude RMS in 0.3-5 Hz, level flight (V11 "no limit cycles")
CHUNK_RUNS = 900  # closed loops per vmap (memory)


@dataclass(frozen=True)
class Manoeuvre:
    """A guided doublet on one axis (static under jit; amplitudes come from the stack)."""

    axis: str  # roll (a bank doublet) | pitch | tecs (height) | speed
    half: float  # seconds per half of the doublet
    amp: float = ROLL_AMP  # roll: the bank (rad)

    def __call__(self, t: Array, sp: StackParams) -> Target:
        first = (t >= T0) & (t < T0 + self.half)
        second = (t >= T0 + self.half) & (t < T0 + 2 * self.half)
        sign = jnp.where(first, 1.0, jnp.where(second, -1.0, 0.0))
        level = dict(lateral=LAT_DIRECT, roll_limit=math.radians(45.0))
        v0 = sp.tecs.v_cruise
        if self.axis == "roll":
            return target(**level, roll_dem=sign * self.amp, vertical=VERT_TECS, alt=ALT0, speed=v0)
        if self.axis == "pitch":
            pd = sp.tecs.pitch_trim + sign * PITCH_AMP
            return target(
                **level,
                roll_dem=0.0,
                vertical=VERT_DIRECT,
                pitch_dem=pd,
                throttle=sp.tecs.thr_cruise,
            )
        up = jnp.maximum(sign, 0.0)  # up, then back to the start
        if self.axis == "speed":
            speed = v0 * (1.0 + SPEED_STEP * up)
            return target(**level, roll_dem=0.0, vertical=VERT_TECS, alt=ALT0, speed=speed)
        dh = alt_step(v0) * up
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


def _cycle(log: LoopLog, k: int, t: np.ndarray, w: np.ndarray) -> float:
    """Attitude oscillation (deg RMS in 0.3-5 Hz, the worse of roll and pitch) over ``w``."""
    dt = float(t[1] - t[0])
    return math.degrees(max(band_rms(log.euler[k][w, j], dt) for j in (0, 1)))


def _cost_attitude(log: LoopLog, k: int, t: np.ndarray, sp: StackParams, flight: Flight) -> float:
    axis, half, end_s = flight.score, flight.half, flight.end_s
    j = 0 if axis == "roll" else 1
    amp = flight.amp if axis == "roll" else PITCH_AMP
    y = log.euler[k][:, j] - (0.0 if axis == "roll" else float(sp.tecs.pitch_trim))
    first = t < T0 + half  # the first half is a step from level to +amp
    rise, ovs, _ = step_metrics(t[first], y[first], amp, t0=T0)
    ends = ((t >= T0 + half - end_s) & (t < T0 + half)) | (
        (t >= T0 + 2 * half - end_s) & (t < T0 + 2 * half)
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


def _cost_tecs(log: LoopLog, k: int, t: np.ndarray, sp: StackParams, half: float) -> float:
    v0 = float(sp.tecs.v_cruise)
    w = t >= T0
    h = -log.pos_ned[k][:, 2] - ALT0
    step = float(alt_step(v0))
    up = (t >= T0) & (t < T0 + half)
    down = (t >= T0 + half) & (t < T0 + 2 * half)
    dv = float(np.abs(log.va[k] - v0).max()) / v0  # the whole flight, as V20
    over = max(float(h[up].max()) - step, float(-h[down].min()), 0.0)
    end_up = (t >= T0 + SETTLE_S) & up
    end_down = (t >= T0 + half + SETTLE_S) & down
    settle = max(float(np.abs(h[end_up] - step).max()), float(np.abs(h[end_down]).max()))
    before = (t >= LEVEL_FROM_S) & (t < T0)
    cycle = max(_cycle(log, k, t, before), _cycle(log, k, t, t >= T0 + 2 * half + 3.0))
    worst = max(
        dv / TECS_SPEED_LIMIT,
        over / max(TECS_OVERSHOOT[1], TECS_OVERSHOOT[0] * step),
        settle / max(TECS_SETTLE[1], TECS_SETTLE[0] * step),
        cycle / CYCLE_LIMIT_DEG,
    )
    track = float(np.sqrt(np.mean((h[w] + ALT0 - log.alt_target[k][w]) ** 2)))
    return worst + TIE_BREAK * track


def _cost_speed(log: LoopLog, k: int, t: np.ndarray, sp: StackParams, half: float) -> float:
    v0 = float(sp.tecs.v_cruise)
    h = -log.pos_ned[k][:, 2] - ALT0
    va = log.va[k]
    dh = float(np.abs(h[t >= T0]).max())
    up = (t >= T0 + SPEED_SETTLE_S) & (t < T0 + half)
    down = (t >= T0 + half + SPEED_SETTLE_S) & (t < T0 + 2 * half)
    err = max(
        float(np.abs(va[up] - (1.0 + SPEED_STEP) * v0).max()), float(np.abs(va[down] - v0).max())
    )
    worst = max(dh / SPEED_ALT_LIMIT_M, err / max(SPEED_SETTLE[1], SPEED_SETTLE[0] * v0))
    w = t >= T0
    return worst + TIE_BREAK * float(np.sqrt(np.mean(h[w] ** 2)))


# Parameters per tuning key: (StackParams paths, config paths, "scale" or "set").
_PATHS: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
    "roll_ff": (("att.roll_ff",), ("attitude.roll.ff",), "scale"),
    "roll_pi": (("att.roll_p", "att.roll_i"), ("attitude.roll.p", "attitude.roll.i"), "scale"),
    "roll_tau": (("att.roll_tau",), ("attitude.roll.tconst_s",), "set"),
    "roll_i": (("att.roll_i",), ("attitude.roll.i",), "scale"),
    "roll_p": (("att.roll_p",), ("attitude.roll.p",), "scale"),
    "pitch_i": (("att.pitch_i",), ("attitude.pitch.i",), "scale"),
    "pitch_p": (("att.pitch_p",), ("attitude.pitch.p",), "scale"),
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


@dataclass(frozen=True)
class Flight:
    axis: str  # the manoeuvre (roll | pitch | tecs | speed) …
    score: str  # … and the requirements (roll | yaw | pitch | tecs | speed)
    half: float  # seconds per doublet half
    end_s: float = 0.5  # attitude: the end-of-half window for the tracking error
    amp: float = ROLL_AMP  # roll: the bank (rad)


@dataclass(frozen=True)
class Stage:
    name: str
    flights: tuple[Flight, ...]  # a variant is scored on the worst of them
    grid: dict[str, tuple[float, ...]]


# Rate gains go down to 0.35x: with servo deadband and lag, high gains limit-cycle. Halves of
# 4 s let the attitude settle, so the end-of-half error shows steady-state error.
ROLL = Flight("roll", "roll", 4.0)
PITCH = Flight("pitch", "pitch", 4.0)
TECS = Flight("tecs", "tecs", SETTLE_S + SETTLED_S)
SPEED = Flight("speed", "speed", SPEED_SETTLE_S + 10.0)
STAGES = (
    Stage("roll", (ROLL,),
          {"roll_ff": (0.8, 1.0, 1.25), "roll_pi": (0.35, 0.5, 1.0, 1.6),
           "roll_tau": (0.25, 0.3, 0.4)}),
    Stage("yaw", (Flight("roll", "yaw", 6.0, amp=BANK_AMP),),
          {"yaw_ff": (0.6, 0.8, 1.0, 1.25, 1.6), "yaw_slip": (0.0, 0.02, 0.08, 0.2),
           "yaw_mix": (0.0, 0.25, 0.5)}),
    Stage("roll, sustained", (Flight("roll", "roll", 6.0, 2.0, BANK_AMP),),
          {"roll_i": (1.0, 2.0, 3.0), "roll_p": (1.0, 2.0)}),
    Stage("pitch", (PITCH,),
          {"pitch_ff": (0.8, 1.0, 1.25), "pitch_pi": (0.35, 0.5, 1.0, 1.6),
           "pitch_tau": (0.22, 0.3, 0.4)}),
    Stage("pitch, in the loop", (PITCH, TECS),
          {"pitch_p": (1.0, 1.5, 2.0), "pitch_i": (0.5, 1.0)}),
    Stage("tecs", (TECS, SPEED),
          {"tecs_tc": (3.0, 5.0, 8.0), "tecs_spd": (1.0, 1.5, 2.0),
           "tecs_climb": (0.4, 0.6, 1.0)}),
)  # fmt: skip
# Each variant flies on this many noise streams (power-ups) and is scored on its worst: a
# variant that meets the requirements only on a lucky draw of sensor biases and noise is not
# kept.
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
    flight: Flight,
    keys: Array,
) -> np.ndarray:
    """Cost of each run (airframe ``idx[r]`` flying with stack ``sps[r]``). Every variant of an
    airframe flies with the airframe's key: the variants are compared on the same noise."""
    half, axis = flight.half, flight.axis
    hold = HOLD_TECS_S if axis == "tecs" else HOLD_S
    n = round((T0 + 2 * half + hold) / CONFIG.dt_control)
    guided = Manoeuvre(axis, half, flight.amp)
    score: Callable[[LoopLog, int, np.ndarray, StackParams], float]
    if flight.score == "yaw":
        score = lambda lg, k, t, sp: _cost_yaw(lg, k, t, half)  # noqa: E731
    elif flight.score == "tecs":
        score = lambda lg, k, t, sp: _cost_tecs(lg, k, t, sp, half)  # noqa: E731
    elif flight.score == "speed":
        score = lambda lg, k, t, sp: _cost_speed(lg, k, t, sp, half)  # noqa: E731
    else:
        score = lambda lg, k, t, sp: _cost_attitude(lg, k, t, sp, flight)  # noqa: E731
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
    # The climb knob scales the demand limit only: the throttle keeps its scaling, written out.
    cfgs = [
        c.model_copy(
            update={
                "tecs": c.tecs.model_copy(
                    update={"climb_full_thr_mps": c.tecs.climb_full_thr_mps or c.tecs.climb_max_mps}
                )
            }
        )
        for c in cfgs
    ]
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
    streams = []
    for k in range(NOISE_STREAMS):
        sim0, z0, _ = _initial(vps, env, v0, airframe_keys(ids, TUNE_SEED, 2 * k))
        streams.append((sim0, z0, airframe_keys(ids, TUNE_SEED, 2 * k + 1)))
    n = len(airframes)
    chosen: list[dict[str, float]] = [{} for _ in range(n)]
    for stage in STAGES:
        grid = stage.grid
        keys = list(grid)
        combos = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*grid.values())]
        runs, idx = [], []
        for i in range(n):
            for c in combos:
                runs.append(_apply(sps[i], c))
                idx.append(i)
        costs = np.max(
            [
                _evaluate(vps, runs, env, sim0, z0, np.array(idx), f, fly_keys)
                for f in stage.flights
                for sim0, z0, fly_keys in streams
            ],
            axis=0,
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
        out.append(StackCfg.model_validate(d))
    return out
