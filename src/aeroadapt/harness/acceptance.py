"""V11 autopilot acceptance (spec §13): step and tracking tests with fixed criteria.

Every test flies the full closed loop (simulator + navigator + stock stack) and is scored on
*truth*. In calm air a single run is used. In light turbulence each step test is scored on
the ensemble mean of several seeds: the mean over seeds is the step response, the spread
is the disturbance response. Tracking, saturation and limit-cycle metrics use every run.

``run_v11`` returns a list of :class:`Check`; the validation tests and the tuning loop both
consume it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from aeroadapt.airframes import Airframe, load_airframe
from aeroadapt.harness.closed_loop import LoopLog, LoopSetup, closed_loop, setup_loop
from aeroadapt.scenarios import Site, load_site
from aeroadapt.sim.aerojax import EnvParams, make_env
from aeroadapt.stack.autopilot import MODE_AUTO, MODE_GUIDED
from aeroadapt.stack.config import StackParams, build_stack_params, load_stack_cfg
from aeroadapt.stack.types import (
    LAT_DIRECT,
    LAT_LOITER,
    VERT_DIRECT,
    VERT_TECS,
    Mission,
    Target,
    make_mission,
    target,
)
from aeroadapt.vehicle.protocol import InitialCondition

# Steps start after the navigator has converged (attitude, gyro bias, wind): an aircraft is
# not step-tested seconds after power-up. At 3 s a wind-dependent 1.5° roll bias remains.
T_STEP = 20.0
ALT0, V0 = 120.0, 25.0
LIGHT_W20_MPS = 15 * 0.514444  # MIL-HDBK-1797 light turbulence: W20 = 15 kt
LIGHT_WIND_MPS = 5.0
SATURATION = 0.98
TESTS = ("roll", "pitch", "alt", "speed", "l1", "loiter", "level")


@dataclass(frozen=True)
class Check:
    name: str
    value: float
    limit: float
    unit: str
    upper: bool = True  # value must be ≤ limit (else ≥)

    @property
    def ok(self) -> bool:
        return bool(self.value <= self.limit) if self.upper else bool(self.value >= self.limit)

    def __str__(self) -> str:
        op = "≤" if self.upper else "≥"
        mark = "PASS" if self.ok else "FAIL"
        return f"{mark}  {self.name:<44} {self.value:9.3f} {op} {self.limit:g} {self.unit}"


@dataclass(frozen=True)
class Guided:
    """Hashable guided-mode schedule (a static argument of the jitted loop)."""

    kind: str  # roll | pitch | alt | speed | level | loiter
    pitch_trim: float = 0.0
    throttle: float = 0.0

    def __call__(self, t: Array, sp: StackParams | None = None) -> Target:
        after = t >= T_STEP
        level = dict(lateral=LAT_DIRECT, roll_dem=0.0, roll_limit=math.radians(45.0))
        tecs = dict(vertical=VERT_TECS, alt=ALT0, speed=V0)
        if self.kind == "roll":
            return target(
                **{**level, "roll_dem": jnp.where(after, math.radians(30.0), 0.0)}, **tecs
            )
        if self.kind == "pitch":
            pd = self.pitch_trim + jnp.where(after, math.radians(5.0), 0.0)
            return target(**level, vertical=VERT_DIRECT, pitch_dem=pd, throttle=self.throttle)
        if self.kind == "alt":
            return target(**level, **{**tecs, "alt": ALT0 + jnp.where(after, 20.0, 0.0)})
        if self.kind == "speed":
            return target(**level, **{**tecs, "speed": V0 + jnp.where(after, 5.0, 0.0)})
        if self.kind == "loiter":
            return target(
                lateral=LAT_LOITER,
                centre=jnp.array([400.0, 400.0]),
                radius=150.0,
                direction=1.0,
                roll_limit=math.radians(45.0),
                **tecs,
            )
        return target(**level, **tecs)


# --------------------------------------------------------------------------- metrics
def step_metrics(
    t: ArrayLike, y: ArrayLike, y_cmd: float, t0: float = T_STEP
) -> tuple[float, float, float]:
    """(10–90% rise time, overshoot fraction, |steady-state error|) of a step at ``t0``.

    Rise and overshoot are relative to the response's own initial and settled values (the
    last two seconds); the steady-state error is the settled value against the command."""
    t, y = np.asarray(t), np.asarray(y)
    y0 = float(np.mean(y[(t > t0 - 1.0) & (t <= t0)]))
    y_ss = float(np.mean(y[t > t[-1] - 2.0]))
    m = t > t0
    frac = (y[m] - y0) / (y_ss - y0)
    tt = t[m]
    i10 = np.argmax(frac >= 0.1) if np.any(frac >= 0.1) else len(tt) - 1
    i90 = np.argmax(frac >= 0.9) if np.any(frac >= 0.9) else len(tt) - 1
    return float(tt[i90] - tt[i10]), float(max(frac.max() - 1.0, 0.0)), abs(y_ss - y_cmd)


def leg_xtrack(log: LoopLog, mission: Mission, start: np.ndarray) -> np.ndarray:
    """Truth cross-track error to the leg being flown at each step."""
    wps = np.asarray(mission.wps)[:, :2]
    n = int(mission.n_wp)
    idx = np.asarray(log.wp_idx)
    pos = np.asarray(log.pos_ned)[:, :2]
    switched = np.cumsum(np.r_[0, np.diff(idx) != 0]) > 0
    a = np.where(switched[:, None], wps[(idx - 1) % n], start[None, :])
    b = wps[idx]
    ab = (b - a) / np.linalg.norm(b - a, axis=1, keepdims=True)
    rel = pos - a
    return rel[:, 0] * ab[:, 1] - rel[:, 1] * ab[:, 0]


def settled_mask(log: LoopLog, after_switch_s: float) -> np.ndarray:
    """Steps at least ``after_switch_s`` after the last waypoint switch (and the start)."""
    t, idx = np.asarray(log.t), np.asarray(log.wp_idx)
    t_switch = np.maximum.accumulate(np.where(np.r_[True, np.diff(idx) != 0], t, -np.inf))
    return t - np.maximum(t_switch, 0.0) >= after_switch_s


# --------------------------------------------------------------------------- runner
@dataclass
class V11Runner:
    airframe: Airframe
    site: Site
    sp: StackParams
    turbulent: bool
    seeds: tuple[int, ...]

    def env(self) -> EnvParams:
        return make_env(
            alt_origin_msl_m=self.site.ground_msl_m,
            magnetic_ned_gauss=self.site.magnetic.field_ned_gauss,
            wind_from_deg=250.0,
            wind_speed_mps=LIGHT_WIND_MPS if self.turbulent else 0.0,
            w20_mps=LIGHT_W20_MPS if self.turbulent else 0.0,
        )

    def _setups(self, mode: int) -> list[LoopSetup]:
        return [
            setup_loop(
                self.airframe,
                self.site,
                self.env(),
                InitialCondition(alt_agl_m=ALT0),
                stack=self.sp,
                mode=mode,
                seed=s,
            )
            for s in self.seeds
        ]

    def fly(self, mission: Mission, seconds: float, guided: Guided | None, mode: int) -> LoopLog:
        """Runs every seed (vmapped); returns logs with a leading seed axis."""
        setups = self._setups(mode)
        su = setups[0]
        stack = lambda *xs: jnp.stack(xs)  # noqa: E731
        sim0 = jax.tree.map(stack, *[s.sim0 for s in setups])
        st0 = jax.tree.map(stack, *[s.stack0 for s in setups])
        z0 = jax.tree.map(stack, *[s.z0 for s in setups])
        keys = jnp.stack([jax.random.PRNGKey(1000 + s) for s in self.seeds])
        n = round(seconds / su.config.dt_control)

        def one(sim0, st0, z0, key):
            return closed_loop(
                su.vp,
                su.env,
                su.faults,
                su.sp,
                mission,
                sim0,
                st0,
                z0,
                key,
                n_steps=n,
                guided=guided,
                config=su.config,
            )[2]

        return jax.tree.map(np.asarray, jax.vmap(one)(sim0, st0, z0, keys))


def _mean(log: LoopLog) -> LoopLog:
    return jax.tree.map(lambda x: x.mean(0) if x.dtype.kind == "f" else x[0], log)


def run_v11(
    sp: StackParams | None = None,
    *,
    turbulent: bool = False,
    seeds: tuple[int, ...] = (0,),
    airframe: str = "aerosonde_bm",
    site: str = "cambridge_west",
    only: tuple[str, ...] = TESTS,
) -> list[Check]:
    """Run the V11 tests named in ``only`` (default: all) and return their checks."""
    af, st = load_airframe(airframe), load_site(site)
    if sp is None:
        sp = build_stack_params(load_stack_cfg(airframe), af, st, 0.01)
    r = V11Runner(af, st, sp, turbulent, seeds)
    tag = "light turb." if turbulent else "calm"
    checks: list[Check] = []
    for name in only:
        if name == "level" and turbulent:
            continue
        checks += _TESTS[name](r, tag)
    return checks


_DUMMY = make_mission([(1000.0, 0.0, ALT0)])


def _roll(r: V11Runner, tag: str) -> list[Check]:
    """Roll 0 → 30°, and coordination in the steady 30° turn."""
    m = _mean(r.fly(_DUMMY, T_STEP + 8.0, Guided("roll"), MODE_GUIDED))
    rise, ovs, sse = step_metrics(m.t, m.euler[:, 0], math.radians(30.0))
    turn = m.t > T_STEP + 3.0
    return [
        Check(f"roll step rise 10-90% ({tag})", rise, 1.0, "s"),
        Check(f"roll step overshoot ({tag})", 100 * ovs, 10.0, "%"),
        Check(f"roll step steady-state error ({tag})", math.degrees(sse), 1.0, "deg"),
        Check(
            f"|beta| in 30 deg turn ({tag})", math.degrees(np.abs(m.beta[turn]).max()), 3.0, "deg"
        ),
    ]


def _pitch(r: V11Runner, tag: str) -> list[Check]:
    """Pitch trim → trim + 5° at fixed (trim) throttle."""
    pitch_trim, thr = float(r.sp.tecs.pitch_trim), float(r.sp.tecs.thr_cruise)
    m = _mean(r.fly(_DUMMY, T_STEP + 4.0, Guided("pitch", pitch_trim, thr), MODE_GUIDED))
    rise, ovs, _ = step_metrics(m.t, m.euler[:, 1], pitch_trim + math.radians(5.0))
    return [
        Check(f"pitch step rise 10-90% ({tag})", rise, 0.8, "s"),
        Check(f"pitch step overshoot ({tag})", 100 * ovs, 15.0, "%"),
    ]


def _alt(r: V11Runner, tag: str) -> list[Check]:
    """Altitude +20 m (TECS)."""
    m = _mean(r.fly(_DUMMY, T_STEP + 30.0, Guided("alt"), MODE_GUIDED))
    h = -m.pos_ned[:, 2]
    late = m.t >= T_STEP + 20.0
    return [
        Check(f"alt step error after 20 s ({tag})", np.abs(h[late] - ALT0 - 20).max(), 2.0, "m"),
        Check(f"alt step overshoot ({tag})", max(float(h.max()) - ALT0 - 20, 0.0), 3.0, "m"),
        Check(f"alt step airspeed deviation ({tag})", np.abs(m.va - V0).max(), 2.0, "m/s"),
    ]


def _speed(r: V11Runner, tag: str) -> list[Check]:
    """Airspeed +5 m/s (TECS)."""
    m = _mean(r.fly(_DUMMY, T_STEP + 25.0, Guided("speed"), MODE_GUIDED))
    late = m.t >= T_STEP + 15.0
    return [
        Check(
            f"speed step error after 15 s ({tag})", np.abs(m.va[late] - V0 - 5).max(), 0.5, "m/s"
        ),
        Check(
            f"speed step altitude deviation ({tag})",
            np.abs(-m.pos_ned[:, 2] - ALT0).max(),
            3.0,
            "m",
        ),
    ]


def _l1(r: V11Runner, tag: str) -> list[Check]:
    """L1 on a 1.5 km square (settled legs), and servo saturation over the whole flight."""
    side = 1500.0
    square = make_mission([(side, 0, ALT0), (side, side, ALT0), (0, side, ALT0), (0, 0, ALT0)])
    lg = r.fly(square, 300.0, None, MODE_AUTO)
    xt, sat = [], []
    for k in range(len(r.seeds)):
        one = jax.tree.map(lambda x, k=k: x[k], lg)
        mask = settled_mask(one, 25.0)
        xt.append(leg_xtrack(one, square, np.zeros(2))[mask])
        sat.append(np.any(np.abs(one.cmd[:, :3]) >= SATURATION, axis=1))
    xt_rms = float(np.sqrt(np.mean(np.concatenate(xt) ** 2)))
    return [
        Check(f"L1 cross-track RMS ({tag})", xt_rms, 5.0 if r.turbulent else 2.0, "m"),
        Check(f"servo saturation, mission ({tag})", 100 * float(np.mean(sat)), 5.0, "%"),
    ]


def _loiter(r: V11Runner, tag: str) -> list[Check]:
    """Loiter radius after capture."""
    lg = r.fly(_DUMMY, 150.0, Guided("loiter"), MODE_GUIDED)
    errs = []
    for k in range(len(r.seeds)):
        p = lg.pos_ned[k][lg.t[k] > 60.0, :2]
        errs.append(np.linalg.norm(p - np.array([400.0, 400.0]), axis=1) - 150.0)
    rms = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
    return [Check(f"loiter radius RMS error ({tag})", rms, 5.0, "m")]


def band_rms(y: ArrayLike, dt: float, f_lo: float = 0.3, f_hi: float = 5.0) -> float:
    """RMS of ``y`` within [f_lo, f_hi] Hz (FFT band-pass of the detrended signal)."""
    y = np.asarray(y)
    y = y - np.polyval(np.polyfit(np.arange(len(y)), y, 1), np.arange(len(y)))
    spec = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), dt)
    spec[(f < f_lo) | (f > f_hi)] = 0.0
    return float(np.sqrt(np.mean(np.fft.irfft(spec, len(y)) ** 2)))


def _level(r: V11Runner, tag: str) -> list[Check]:
    """No limit cycles: in calm level flight there is no sustained attitude oscillation in
    the band where loop limit cycles live (0.3-5 Hz). Slower estimator wander and the TECS
    phugoid control are excluded by the band."""
    m = _mean(r.fly(_DUMMY, 40.0, Guided("level"), MODE_GUIDED))
    w = m.t > 10.0
    dt = float(m.t[1] - m.t[0])
    osc = max(band_rms(m.euler[w, 0], dt), band_rms(m.euler[w, 1], dt))
    return [Check(f"limit cycle: 0.3-5 Hz attitude RMS ({tag})", math.degrees(osc), 0.25, "deg")]


_TESTS = {
    "roll": _roll,
    "pitch": _pitch,
    "alt": _alt,
    "speed": _speed,
    "l1": _l1,
    "loiter": _loiter,
    "level": _level,
}
