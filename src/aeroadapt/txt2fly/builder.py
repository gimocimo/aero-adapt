"""Plan → mission (spec 06 §6): the request realised as closely as the rules allow.

A plan says what the operator wants. The builder searches for the closest way to fly it with
the aircraft flying, in the wind blowing, inside the site's rules:

* **Each step has options:** the requested values first, then nearby ones. A circle the
  aircraft cannot hold becomes the tightest it can; a height outside the band is brought
  inside it; a circle or pass moves (centre, crossing point, direction) to keep its turns
  clear of no-fly areas; a place inside a no-fly area is replaced by the nearest point clear
  of it.
* **Routes go round no-fly areas** on the way to a step, between a patrol's places and on
  the way home (``routing.py``).
* **Options are judged by flying them.** A point-mass aircraft flies each candidate with the
  stack's own item sequencer and L1 guidance, its bank following the roll loop's time
  constant and rate limit, in the current wind; all candidates of a step fly at once
  (``vmap``). A beam search keeps the best few partial plans: clear of every no-fly area
  first, then closest to the request, then quickest.
* **Every change is a note** in the plan the operator confirms. The checker then judges the
  final mission on its own full preview.

With ``adapt=False`` the plan is taken literally (no changes, no routes): the checker's tests
use it to see each rule catch its violation.

The tightest circle keeps a 5° bank margin for tracking corrections, at the ground speed
downwind:  r_min = (V + W)² / (g · tan(φ_lim − 5°)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from aeroadapt.airframes import Airframe
from aeroadapt.stack.autopilot import MODE_AUTO, MODE_RTL, _mission_step, ap_init
from aeroadapt.stack.config import StackCfg, StackParams, build_stack_params
from aeroadapt.stack.guidance import l1_loiter, l1_waypoint, roll_from_lat_acc
from aeroadapt.stack.types import (
    ITEM_FLYOVER,
    ITEM_JUMP,
    ITEM_LOITER_TIME,
    ITEM_LOITER_TURNS,
    ITEM_LOITER_UNLIM,
    ITEM_RTL,
    ITEM_WAYPOINT,
    LAT_LOITER,
    MAX_ITEMS,
    Item,
    Mission,
    mission_from_items,
)
from aeroadapt.txt2fly import geometry as geo
from aeroadapt.txt2fly.energy import RESERVE, EnergyModel, energy_model
from aeroadapt.txt2fly.gazetteer import Gazetteer, Resolved, UnknownPlace
from aeroadapt.txt2fly.plan import (
    Goto,
    Hold,
    Orbit,
    PassOver,
    Patrol,
    Plan,
    ReturnHome,
    SetSpeed,
)
from aeroadapt.txt2fly.routing import ClearanceMap, clearance_map
from aeroadapt.txt2fly.terrain import CLEARANCE_M, ceil5, climb_out, landmark, relief

G0 = 9.806_65
BANK_MARGIN_DEG = 5.0
RUN_MIN_M = 100.0  # shortest straight run before a pass
TRACK_MARGIN_M = 30.0  # the checker's margin round no-fly areas (measured by V26)
GRID_SLACK_M = 4.0  # the clearance map's interpolation error, added when planning with it
PREVIEW_DT = 0.1
PREVIEW_STEPS_S = (300.0, 1500.0)  # a short preview first, the long one if the plan needs it
PREVIEW_S = PREVIEW_STEPS_S[-1]
ALT_TAU_S = 5.0  # height response of the preview aircraft (rate-limited by TECS's limits)
BEAM = 3  # partial plans kept from step to step
FINALISTS = 6  # complete plans flown in full, in order, until one keeps clear
EVAL_EXTRA_M = 10.0  # the search keeps this much more clearance than the checker asks
MAX_OPTIONS = 12  # options per step tried in flight
LANES = 16  # candidate previews flown per batch (one compiled shape)
COMPASS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")


def r_min(airspeed: float, wind: float, roll_limit_deg: float) -> float:
    """Tightest circle over the ground that leaves a 5° bank margin (downwind point)."""
    bank = math.radians(roll_limit_deg - BANK_MARGIN_DEG)
    return (airspeed + wind) ** 2 / (G0 * math.tan(bank))


def bank_for(radius: float, airspeed: float, wind: float) -> float:
    """Bank (deg) a circle of ``radius`` needs at its downwind point."""
    return math.degrees(math.atan((airspeed + wind) ** 2 / (G0 * radius)))


@dataclass(frozen=True)
class Aircraft:
    """What the planner knows, as a ground station does: the aircraft's configuration and its
    state from telemetry (navigator estimates, battery monitor)."""

    stack_cfg: StackCfg
    airframe: Airframe
    pos_ne: tuple[float, float]
    alt_agl: float
    course_deg: float
    wind_ne: tuple[float, float] = (0.0, 0.0)  # the air's velocity (towards), m/s
    used_mah: float = 0.0

    @property
    def wind_speed(self) -> float:
        return float(np.hypot(*self.wind_ne))


@dataclass
class StepGeom:
    """What a plan step became: its items and the geometry the checker explains."""

    index: int
    do: str
    label: str
    items: tuple[int, int] = (0, 0)  # [first, end) into the item list
    target: Resolved | None = None
    centre: np.ndarray | None = None
    radius: float | None = None
    alt: float | None = None
    speed: float | None = None
    line: np.ndarray | None = None  # (2, 2) a pass's run-in and run-out points


@dataclass
class Preview:
    t: np.ndarray
    ne: np.ndarray  # (K, 2)
    alt: np.ndarray
    item: np.ndarray
    mode: np.ndarray
    captured: np.ndarray  # on a loiter circle
    bank_deg: np.ndarray
    airspeed: np.ndarray
    climb: np.ndarray
    used_mah: np.ndarray  # cumulative, from the start of the plan


@dataclass
class Energy:
    available_mah: float  # above the reserve
    needed_mah: float | None  # to the end of the plan (None: open-ended)
    minutes: float | None  # duration of the plan, or time until the reserve if open-ended
    laps: int | None = None  # patrols: laps that fit (until_reserve) or asked for
    lap_mah: float | None = None


@dataclass
class Built:
    plan: Plan
    aircraft: Aircraft
    items: list[Item]
    mission: Mission | None
    steps: list[StepGeom]
    preview: Preview | None
    energy: Energy | None
    rmin: dict[float, float]  # tightest circle per airspeed used
    endless: bool  # the plan never ends by itself
    notes: list[str] = field(default_factory=list)  # what differs from the request, and why
    errors: list[str] = field(default_factory=list)  # K1: plans that cannot be built


def _unit(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([math.cos(r), math.sin(r)])


def _ceil10(x: float) -> float:
    return math.ceil(x / 10.0) * 10.0


def _compass(v: np.ndarray) -> str:
    deg = (math.degrees(math.atan2(v[1], v[0])) + 360.0) % 360.0
    return COMPASS[int((deg + 22.5) // 45) % 8]


def _speeds(cfg: StackCfg) -> dict[str, float]:
    t = cfg.tecs
    return {
        "slow": round(0.5 * (t.airspeed_min_mps + t.airspeed_cruise_mps), 1),
        "cruise": t.airspeed_cruise_mps,
        "fast": round(0.5 * (t.airspeed_cruise_mps + t.airspeed_max_mps), 1),
    }


_SP: dict[tuple[str, str, str], StackParams] = {}


def stack_params(ac: Aircraft, gaz: Gazetteer) -> StackParams:
    key = (ac.stack_cfg.model_dump_json(), ac.airframe.id, gaz.site.id)
    if key not in _SP:
        _SP[key] = build_stack_params(ac.stack_cfg, ac.airframe, gaz.site, 0.01)
    return _SP[key]


def _energy(ac: Aircraft, gaz: Gazetteer) -> EnergyModel:
    t = ac.stack_cfg.tecs
    return energy_model(ac.airframe, gaz.site, t.airspeed_min_mps, t.airspeed_max_mps)


# --------------------------------------------------------------------------- the search
@dataclass
class _State:
    """Where the aircraft is when a step begins."""

    pos: np.ndarray
    course: float  # deg
    alt: float
    speed: float


@dataclass
class _Option:
    """One way to fly a step. JUMP targets count from the option's first item."""

    items: list[Item]
    entry: np.ndarray | None  # where the aircraft heads first (None: no flight, e.g. speed)
    exit: np.ndarray | None  # where it leaves
    cost: float  # how far this departs from the request (0: as asked)
    notes: list[str]
    geom: StepGeom
    alt: float | None = None
    speed: float | None = None  # a speed step's new airspeed
    endless: str | None = None
    until_reserve: bool = False
    eval_items: list[Item] | None = None  # shorter stand-ins for evaluation (one turn, …)


@dataclass
class _Node:
    legs: list[tuple[list[np.ndarray], _Option]]  # (route waypoints, option) per step
    state: _State
    cost: float = 0.0
    clear: float = math.inf  # least clearance flown so far (preview)
    feasible: bool = True
    endless: str | None = None
    t: float = 0.0


class _Ctx:
    def __init__(
        self, plan: Plan, ac: Aircraft, gaz: Gazetteer, adapt: bool, room: float = 1.0
    ) -> None:
        self.plan, self.ac, self.gaz, self.adapt = plan, ac, gaz, adapt
        self.room = room  # the room to turn kept round no-fly areas, in turn radii (× usual)
        self.cfg = ac.stack_cfg
        self.cruise = self.cfg.tecs.airspeed_cruise_mps
        self.cm: ClearanceMap = clearance_map(gaz)
        self.rmin: dict[float, float] = {}
        ops = gaz.site.operations
        assert ops is not None
        self.ops = ops

    # ------------------------------------------------------------------ helpers
    def r_min(self, speed: float) -> float:
        if speed not in self.rmin:
            self.rmin[speed] = r_min(speed, self.ac.wind_speed, self.cfg.l1.roll_limit_deg)
        return self.rmin[speed]

    def route_need(self, speed: float) -> float:
        """Clearance a route keeps: the tracking margin and room to turn at its corners."""
        return TRACK_MARGIN_M + GRID_SLACK_M + 0.6 * self.room * self.r_min(speed)

    def item(self, kind: int, ne=(0.0, 0.0), alt=0.0, speed=0.0, **kw) -> Item:
        spd = 0.0 if speed in (0.0, self.cruise) else speed
        return Item(kind, float(ne[0]), float(ne[1]), float(alt), speed=spd, **kw)

    def where(self, p: np.ndarray, name: str, what: str) -> str:
        """Why a place cannot be used as it is: within a no-fly area, outside the operating
        area, or too close to one of them for ``what``."""
        if not geo.inside(p, self.gaz.geofence)[0]:
            return f"{name} is outside the operating area (geofence)"
        area = self.area_at(p)
        if float(self.cm.at(p[None])[0]) < 0:
            return f"{name} is within {area}"
        return f"{name} is too close to {area} for {what}"

    def area_at(self, p: np.ndarray) -> str:
        """What a point is too close to: the nearest no-fly area, or the geofence."""
        fence = -float(geo.signed_distance(p[None], self.gaz.geofence)[0])
        best, name = fence, "the edge of the operating area (geofence)"
        for a in self.gaz.no_fly:
            d = float(a.signed_distance(p[None])[0])
            if d < best:
                best, name = d, f"{a.name} (no-fly)"
        return name

    def alt(self, requested: float | None, state: _State, label: str) -> tuple[float, list[str]]:
        a = float(round(state.alt)) if requested is None else float(requested)
        if not self.adapt:
            return a, []
        lo, hi = self.ops.min_agl_m, self.ops.ceiling_agl_m
        if a > hi:
            return hi, [f"{label}: at {hi:g} m, the ceiling here (asked {a:g} m)"]
        if a < lo:
            return lo, [f"{label}: at {lo:g} m, the lowest allowed here (asked {a:g} m)"]
        return a, []

    def clear_point(
        self, p: np.ndarray, name: str, need: float, what: str
    ) -> tuple[np.ndarray, list[str], float]:
        """``p``, or the nearest point with ``need`` of clearance if p is too close to a
        no-fly area: (point, notes, cost)."""
        if not self.adapt or float(self.cm.at(p[None])[0]) >= need:
            return p, [], 0.0
        q = self.cm.nearest_clear(p, need)
        if q is None:
            return p, [], 0.0
        d = float(np.linalg.norm(q - p))
        note = (
            f"{self.where(p, name, 'that')}: {what} the nearest place clear of it, "
            f"{d:.0f} m {_compass(q - p)} of {name}"
        )
        return q, [note], 1.0 + d / 200.0

    def crossings(
        self, r: Resolved, speed: float, verb: str = "passes over"
    ) -> list[tuple[np.ndarray, float, str | None]]:
        """Where to visit a place: (point, cost, note). Its middle, unless it is in a no-fly
        area; points inside it (20 m in from its edge) clear of no-fly areas by half a turn
        plus the margin, the nearest to the middle and the clearest; and if the place has no
        such point (it lies in, beside or beyond a no-fly area or the geofence), the nearest
        clear point outside it."""
        rm = self.r_min(speed)
        need = TRACK_MARGIN_M + GRID_SLACK_M + 0.5 * self.room * rm
        if not self.adapt:
            return [(r.point, 0.0, None)]
        out: list[tuple[np.ndarray, float, str | None]] = []
        reach = self.route_need(speed)  # every point visited must be reachable on a route
        centre = float(self.cm.at(r.point[None])[0])
        if centre >= TRACK_MARGIN_M + GRID_SLACK_M and self.cm.at_reachable(r.point, reach)[0]:
            out.append((r.point, 0.0, None))
        if r.polygon is not None:
            lo, hi = r.polygon.min(axis=0), r.polygon.max(axis=0)
            grid = np.stack(
                np.meshgrid(np.arange(lo[0], hi[0], 10.0), np.arange(lo[1], hi[1], 10.0)), -1
            ).reshape(-1, 2)
            grid = grid[geo.signed_distance(grid, r.polygon) <= -20.0]
            if len(grid):
                clear = self.cm.at(grid)
                size = float(np.max(np.linalg.norm(r.polygon - r.point, axis=1)))
                dist = np.linalg.norm(grid - r.point, axis=1)
                picks = []
                for room in (0.5, 1.0):  # half a turn of room, then a whole turn
                    need_r = TRACK_MARGIN_M + GRID_SLACK_M + room * self.room * rm
                    ok = (clear >= need_r) & self.cm.at_reachable(grid, max(need_r, reach))
                    if ok.any():
                        picks.append(int(np.flatnonzero(ok)[np.argmin(dist[ok])]))
                if picks:
                    picks.append(int(np.argmax(clear)))
                for k in dict.fromkeys(picks):
                    if dist[k] > 15.0 or not out:
                        note = (
                            f"{verb} {r.name} away from its middle, to keep the turns clear of "
                            f"{self.area_at(grid[k])}"
                        )
                        out.append((grid[k], float(0.5 * dist[k] / max(size, 1.0)), note))
        if not any(c < 0.5 for _, c, _ in out) or centre < need:
            # The nearest clear places outside it, with more and more room to turn.
            seen: list[np.ndarray] = []
            for room in (0.5, 1.0, 1.5):
                q = self.cm.nearest_clear(
                    r.point, max(TRACK_MARGIN_M + GRID_SLACK_M + room * self.room * rm, reach)
                )
                if q is None or (r.polygon is not None and geo.inside(q, r.polygon)[0]):
                    continue
                if any(np.linalg.norm(q - w) < 20.0 for w in seen):
                    continue
                seen.append(q)
                d = float(np.linalg.norm(q - r.point))
                note = (
                    f"{self.where(r.point, r.name, 'that')}: {verb} the nearest place clear of "
                    f"it, {d:.0f} m {_compass(q - r.point)} of {r.name}"
                )
                out.append((q, 1.0 + d / 200.0, note))
        out.sort(key=lambda x: x[1])
        return out or [(r.point, 0.0, None)]

    # ------------------------------------------------------------------ options per step
    def options(self, k: int, s, state: _State) -> list[_Option]:
        speed = state.speed
        g = StepGeom(k, s.do, s.do, speed=speed)
        if isinstance(s, SetSpeed):
            t = self.cfg.tecs
            v = _speeds(self.cfg)[s.speed] if isinstance(s.speed, str) else float(s.speed)
            notes = []
            if self.adapt and not t.airspeed_min_mps <= v <= t.airspeed_max_mps:
                w = min(max(v, t.airspeed_min_mps), t.airspeed_max_mps)
                notes.append(f"airspeed {w:g} m/s, the nearest this aircraft can fly (asked {v:g})")
                v = w
            g.label, g.speed = f"airspeed {v:g} m/s", v
            return [_Option([], None, None, 0.0, notes, g, speed=v)]
        if isinstance(s, ReturnHome):
            g.label = "return home"
            home = self.gaz.home
            return [
                _Option(
                    [self.item(ITEM_RTL)],
                    home,
                    home,
                    0.0,
                    [],
                    g,
                    endless="the aircraft has returned home",
                )
            ]
        if isinstance(s, Goto):
            r = self.gaz.resolve(s.target, np.asarray(self.ac.pos_ne, float))
            alt, notes = self.alt(s.alt_m, state, f"go to {r.name}")
            g.label, g.target, g.alt = f"go to {r.name}", r, alt
            return [
                _Option([self.item(ITEM_WAYPOINT, p, alt, speed)], p, p, cost,
                        notes + ([note] if note else []), g, alt=alt)
                for p, cost, note in self.crossings(r, speed, "goes to")
            ]  # fmt: skip
        if isinstance(s, PassOver):
            return self.pass_options(s, state, g)
        if isinstance(s, Orbit | Hold):
            return self.circle_options(s, state, g)
        if isinstance(s, Patrol):
            return self.patrol_options(s, state, g)
        raise TypeError(s)

    def pass_options(self, s: PassOver, state: _State, g: StepGeom) -> list[_Option]:
        speed = state.speed
        r = self.gaz.resolve(s.target, np.asarray(self.ac.pos_ne, float))
        alt, alt_notes = self.alt(s.alt_m, state, f"pass over {r.name}")
        g.label, g.target, g.alt = f"pass over {r.name}", r, alt
        rm = self.r_min(speed)
        approach = r.point - state.pos
        if np.linalg.norm(approach) < 1.0:
            approach = _unit(state.course)
        approach = approach / np.linalg.norm(approach)
        notes0: list[str] = []
        if s.heading == "into_wind" and self.ac.wind_speed >= 1.0:
            dirs = [(-np.asarray(self.ac.wind_ne, float) / self.ac.wind_speed, 0.0, None)]
        elif isinstance(s.heading, int | float):
            dirs = [(_unit(float(s.heading)), 0.0, None)]
        else:
            if s.heading == "into_wind":
                notes0.append(f"calm air: the pass over {r.name} follows its long axis")
            if r.polygon is not None:
                ax = geo.long_axis(r.polygon)[0]
                ax = ax if ax @ approach >= 0 else -ax
                across = np.array([-ax[1], ax[0]])
                note = f"the pass runs across {r.name}, not along it, to keep its turns clear"
                slant = f"the pass crosses {r.name} at a slant, to keep its turns clear"
                diag = [(ax + across) / math.sqrt(2), (ax - across) / math.sqrt(2)]
                dirs = [
                    (ax, 0.0, None),
                    (-ax, 0.1, None),
                    (across, 0.3, note),
                    (-across, 0.3, note),
                    *((v, 0.25, slant) for d in diag for v in (d, -d)),
                ]
            else:
                dirs = [(approach, 0.0, None)] + [(_unit(a), 0.2, None) for a in range(0, 360, 45)]
            if not self.adapt:
                dirs = dirs[:1]
        out = []
        for p, c_cost, c_note in self.crossings(r, speed):
            for u, d_cost, d_note in dirs:
                half_in = half_out = 0.0
                if r.polygon is not None and geo.inside(p, r.polygon)[0]:  # over the place
                    proj = (r.polygon - p) @ u
                    half_in, half_out = max(float(-proj.min()), 0.0), max(float(proj.max()), 0.0)
                # Runs: one turn radius in (L1 lines up within it) and half out; then shorter,
                # where room is short (V26 bounds what the turns then do).
                # Over a large place, a pass over part of it (about the crossing point).
                part = (f"the pass crosses part of {r.name}, to keep clear"
                        if max(half_in, half_out) > 150.0 else None)  # fmt: skip
                for run_in, run_out, l_cost, whole in (
                    (max(rm, RUN_MIN_M), max(0.5 * rm, 50.0), 0.0, True),
                    (0.6 * max(rm, RUN_MIN_M), 0.25 * rm, 0.15, True),
                    *([(max(rm, RUN_MIN_M), max(0.5 * rm, 50.0), 0.3, False)] if part else []),
                ):
                    h_in, h_out = (half_in, half_out) if whole else (0.0, 0.0)
                    a = p - (h_in + run_in) * u
                    b = p + (h_out + run_out) * u
                    items = [
                        self.item(ITEM_WAYPOINT, a, alt, speed),
                        self.item(ITEM_FLYOVER, p, alt, speed),
                        self.item(ITEM_WAYPOINT, b, alt, speed),
                    ]
                    notes = [*alt_notes, *notes0, *(n for n in (c_note, d_note) if n)]
                    if not whole and part:
                        notes.append(part)
                    geom = replace(g, line=np.array([a, b]))
                    cost = c_cost + d_cost + l_cost
                    out.append(_Option(items, a, b, cost, notes, geom, alt=alt))
        return self.prune(out, [(o.entry, o.exit) for o in out], rm)

    def circle_options(self, s: Orbit | Hold, state: _State, g: StepGeom) -> list[_Option]:
        speed = state.speed
        rm = self.r_min(speed)
        if isinstance(s, Orbit):
            r = self.gaz.resolve(s.target, np.asarray(self.ac.pos_ne, float))
            alt, notes = self.alt(s.alt_m, state, f"circle {r.name}")
            asked, turns, secs = s.radius_m, s.turns, s.duration_s
            sign = 1.0 if s.direction == "cw" else -1.0
            what = "circle"
        else:
            r = Resolved("the aircraft's position", state.pos.copy())
            alt, notes = self.alt(None, state, "hold")
            asked, turns, secs, sign, what = s.radius_m, None, None, 1.0, "hold"
        if turns is not None and secs is not None:
            notes.append(f"circling {r.name}: {turns:g} turns (the time is ignored)")
        usual = max(self.cfg.l1.loiter_radius_m, rm)
        tight = _ceil10(rm)
        # (radius, cost, note): what was asked (or the usual), then larger, then the tightest.
        if asked is None:
            r0 = _ceil10(usual)
            radii = [(r0, 0.0, f"{what} {r.name}: a {r0:g} m circle (this aircraft's usual)")]
        elif self.adapt and asked < rm:
            r0 = tight
            air = "in this wind" if self.ac.wind_speed >= 1.0 else "in calm air"
            radii = [(r0, 0.0, (
                f"{what} {r.name}: a {asked:.0f} m circle would need "
                f"{bank_for(asked, speed, self.ac.wind_speed):.0f}° of bank at {speed:g} m/s "
                f"{air} (this aircraft banks up to {self.cfg.l1.roll_limit_deg:g}°); "
                f"circles at {r0:g} m, the tightest it can"
            ))]  # fmt: skip
        else:
            r0 = float(round(asked))
            radii: list[tuple[float, float, str | None]] = [(r0, 0.0, None)]
        if self.adapt:
            r1 = _ceil10(1.25 * r0)
            radii.append((r1, 0.4, f"{what} {r.name}: a {r1:g} m circle, to keep the turns clear"))
            if tight < r0:  # tighter (still over the place) before moving the centre
                was = "" if asked is None else f" (asked {asked:.0f} m)"
                note = f"{what} {r.name}: a {tight:g} m circle{was}, the tightest, to keep it clear"
                radii.append((tight, 0.2 if asked is None else 0.3, note))
        g.label, g.target, g.alt = f"{what} {r.name}", r, alt
        need = TRACK_MARGIN_M + GRID_SLACK_M + 0.25 * self.room * rm
        cands: list[tuple[np.ndarray, float, float, list[str]]] = []  # centre, radius, cost, notes
        ring = np.stack(
            [
                np.cos(np.linspace(0, 2 * np.pi, 24, endpoint=False)),
                np.sin(np.linspace(0, 2 * np.pi, 24, endpoint=False)),
            ],
            1,
        )
        for rad, r_cost, r_note in radii:
            offsets = [(np.zeros(2), 0.0)]
            if self.adapt:
                offsets += [(f * rad * _unit(a), f) for f in (0.35, 0.7) for a in range(0, 360, 45)]
            for off, f in offsets:
                c = r.point + off
                if self.adapt and float(self.cm.at(c + rad * ring).min()) < need:
                    continue
                note = (
                    []
                    if f == 0
                    else [
                        f"{what} {r.name}: centred {np.linalg.norm(off):.0f} m "
                        f"{_compass(off)} of it (still over it), to keep clear of "
                        f"{self.area_at(r.point - rad * off / np.linalg.norm(off))}"
                    ]
                )
                cands.append((c, rad, r_cost + 0.5 * f, ([r_note] if r_note else []) + note))
        if not cands:  # the place is in or by a no-fly area: the nearest circle that fits
            c = self.cm.nearest_clear(r.point, r0 + need)
            if c is not None:
                d = float(np.linalg.norm(c - r.point))
                note = (
                    f"{self.where(r.point, r.name, f'a {r0:g} m circle round it')}: circles the "
                    f"nearest place clear of it, centred {d:.0f} m {_compass(c - r.point)} of "
                    f"{r.name}"
                )
                cands.append((c, r0, 1.0 + d / 200.0, [n for n in (radii[0][2], note) if n]))
            else:
                cands.append((r.point, r0, 0.0, [n for n in (radii[0][2],) if n]))
        signs: list[tuple[float, float, str | None]] = [(sign, 0.0, None)]
        if self.adapt:
            flip = "anticlockwise" if sign > 0 else "clockwise"
            signs.append((-sign, 0.3, f"{what} {r.name}: {flip}, to keep the turns clear"))
        out = []
        for c, rad, cost, cnotes in cands:
            for sg, s_cost, s_note in signs:
                if turns is not None:
                    kind, p1, ev = ITEM_LOITER_TURNS, float(turns), min(float(turns), 1.0)
                elif secs is not None:
                    kind, p1 = ITEM_LOITER_TIME, float(secs)
                    ev = min(float(secs), 2 * math.pi * rad / speed)
                else:
                    kind, p1, ev = ITEM_LOITER_UNLIM, 0.0, 0.0
                it = self.item(kind, c, alt, speed, radius=sg * rad, p1=p1)
                geom = replace(g, centre=c, radius=rad)
                endless = (
                    f"circling {r.name} never ends (until the battery reserve)"
                    if kind == ITEM_LOITER_UNLIM
                    else None
                )
                out.append(
                    _Option(
                        [it],
                        c,
                        c,
                        cost + s_cost,
                        notes + cnotes + ([s_note] if s_note else []),
                        geom,
                        alt=alt,
                        endless=endless,
                        eval_items=[it._replace(p1=ev)],
                    )
                )
        out.sort(key=lambda o: o.cost)
        return out[:MAX_OPTIONS]

    def patrol_options(self, s: Patrol, state: _State, g: StepGeom) -> list[_Option]:
        speed = state.speed
        if len(s.targets) < 2:
            raise ValueError("a patrol needs at least two places")
        places = [self.gaz.resolve(t, np.asarray(self.ac.pos_ne, float)) for t in s.targets]
        alt, alt_notes = self.alt(s.alt_m, state, "the patrol")
        g.label, g.alt = f"patrol {', '.join(p.name for p in places)}", alt
        asked_kind = ITEM_FLYOVER if s.visit == "pass_over" else ITEM_WAYPOINT
        cross = [self.crossings(p, speed) for p in places]
        base = [0] * len(places)
        variants: list = [(base, False, 0.0)]
        if self.adapt:
            variants.append((base, True, 0.3))
            for level in range(1, max(len(c) for c in cross)):  # every place moved at once
                v = [min(level, len(c) - 1) for c in cross]
                variants += [(v, False, 0.1 * level), (v, True, 0.1 * level + 0.3)]
            for i, c in enumerate(cross):  # one place moved
                for j in range(1, len(c)):
                    v = list(base)
                    v[i] = j
                    variants += [(v, False, 0.1), (v, True, 0.4)]
        need = self.route_need(speed)
        out = []
        if self.adapt and asked_kind == ITEM_FLYOVER:  # fly-by turns inside the corners
            variants += [(v, r, c + 0.2, ITEM_WAYPOINT) for v, r, c in variants]
        variants = [(*v, asked_kind) if len(v) == 3 else v for v in variants]
        for pick, rev, v_cost, kind in variants:
            order = list(range(len(places)))[::-1] if rev else list(range(len(places)))
            if rev:  # the same loop the other way round, starting at the same place
                order = [order[-1], *order[:-1]]
            pts = [cross[i][pick[i]][0] for i in order]
            notes = list(alt_notes) + [n for i in order if (n := cross[i][pick[i]][2])]
            cost = v_cost + sum(cross[i][pick[i]][1] for i in order)
            if rev:
                notes.append("patrols the places the other way round, to keep the turns clear")
            if kind != asked_kind:
                notes.append("passes by each place (turning early) rather than over its middle, "
                             "to keep the turns clear")  # fmt: skip
            items: list[Item] = []
            ok = True
            laps = s.laps
            closes = laps is None or laps > 1
            for a, b in zip(pts, pts[1:] + ([pts[0]] if closes else []), strict=False):
                items.append(self.item(kind, a, alt, speed))
                way = self.cm.route(a, b, need) if self.adapt else []
                if way is None:
                    ok = False
                    break
                items += [self.item(ITEM_WAYPOINT, w, alt, speed) for w in way]
            if not ok:
                continue
            if not closes:
                items.append(self.item(kind, pts[-1], alt, speed))
            exit_ = np.array([items[-1].north, items[-1].east])
            eval_items = list(items)
            if laps is None:
                items.append(self.item(ITEM_JUMP, p1=0, p2=-1))
                eval_items.append(self.item(ITEM_JUMP, p1=0, p2=1))
            elif laps > 1:
                items.append(self.item(ITEM_JUMP, p1=0, p2=laps - 1))
                eval_items.append(self.item(ITEM_JUMP, p1=0, p2=min(laps - 1, 1)))
            out.append(
                _Option(
                    items,
                    pts[0],
                    exit_,
                    cost,
                    notes,
                    replace(g),
                    alt=alt,
                    until_reserve=laps is None,
                    eval_items=eval_items,
                )
            )
        out.sort(key=lambda o: o.cost)
        if not out:
            raise ValueError("the patrol's places cannot be joined clear of the no-fly areas")
        return out[: 2 * MAX_OPTIONS]

    def shape(
        self, items: list[Item], start: np.ndarray
    ) -> tuple[list[Item], list[int], list[str]]:
        """Tidy, then make every turn's side certain (``tidy`` and ``fix_reversals``); returns
        the items, where each old item now begins, and the sides chosen for reversals."""
        tidy, w1 = self.tidy(items, start)
        out, w2, sides = self.fix_reversals(tidy, start)
        w2.append(len(out))
        return out, [w2[k] for k in w1], sides

    def tidy(self, items: list[Item], start: np.ndarray) -> tuple[list[Item], list[int]]:
        """Drop route waypoints closer to the point before than 1.5 acceptance radii (the
        line replacing them still clear): a cluster of waypoints the aircraft passes within a
        second makes its sequence, and so its path, sensitive to small differences. Visits,
        circles, jump targets and the last waypoint before a jump stay."""
        keep_always = {int(it.p1) for it in items if it.kind == ITEM_JUMP}
        keep_always |= {k - 1 for k, it in enumerate(items) if it.kind == ITEM_JUMP}
        spacing = 1.5 * self.cfg.l1.wp_radius_m
        out: list[Item] = []
        where: list[int] = []
        prev = np.asarray(start, float)
        for k, it in enumerate(items):
            where.append(len(out))
            if it.kind == ITEM_WAYPOINT and k not in keep_always:
                p = np.array([it.north, it.east])
                nxt = next((items[j] for j in range(k + 1, len(items))
                            if items[j].kind in (ITEM_WAYPOINT, ITEM_FLYOVER)), None)  # fmt: skip
                if nxt is not None and np.linalg.norm(p - prev) < spacing:
                    q = np.array([nxt.north, nxt.east])
                    if self.cm.line(prev, q) >= TRACK_MARGIN_M + GRID_SLACK_M:
                        continue  # dropped: `where` points at the next item kept
            out.append(it)
            if it.kind not in (ITEM_JUMP, ITEM_RTL):  # a position the aircraft flies from
                prev = np.array([it.north, it.east])
        out = [it._replace(p1=float(where[int(it.p1)])) if it.kind == ITEM_JUMP else it
               for it in out]  # fmt: skip
        return out, where

    def fix_reversals(
        self, items: list[Item], start: np.ndarray
    ) -> tuple[list[Item], list[int], list[str]]:
        """Make every turn's side certain, so the aircraft and the preview take the same one:

        * **joining a circle** (a loiter, or RTL's circle over home), a waypoint first where the
          path meets the circle tangentially, already going its way round;
        * **doubling back** (a turn of more than 135°) at a waypoint, a turn point after it on
          the side with more clearance (a left and a right turn are otherwise nearly as short).

        Returns the items, where each old item now begins, and the sides chosen."""
        home = self.gaz.home
        r_home = self.cfg.l1.loiter_radius_m
        loiters = (ITEM_LOITER_TURNS, ITEM_LOITER_TIME, ITEM_LOITER_UNLIM)

        def circle(it: Item) -> tuple[np.ndarray, float, float]:
            if it.kind == ITEM_RTL:
                return home, r_home, 1.0  # the stack's RTL circles home clockwise
            return np.array([it.north, it.east]), abs(it.radius), (1.0 if it.radius >= 0 else -1.0)

        def at(k: int) -> np.ndarray:
            it = items[k]
            return (
                circle(it)[0]
                if it.kind == ITEM_RTL or it.kind in loiters
                else np.array([it.north, it.east])
            )

        def heading_to(i: int) -> np.ndarray:
            """Where the aircraft goes after item i (a jump with laps left: its target)."""
            j = i + 1
            while j < len(items):
                it = items[j]
                if it.kind == ITEM_JUMP:
                    if it.p2 != 0:
                        return at(int(it.p1))
                    j += 1
                    continue
                return at(j)
            return home

        out: list[Item] = []
        where: list[int] = []
        sides: list[str] = []
        prev = np.asarray(start, float)
        first = next((it for it in items if it.kind not in (ITEM_JUMP, ITEM_RTL)), None)
        prev_alt, prev_speed = (first.alt, first.speed) if first is not None else (0.0, 0.0)
        for i, it in enumerate(items):
            where.append(len(out))
            if it.kind in loiters or it.kind == ITEM_RTL:
                c, r, d = circle(it)
                v = prev - c
                dist = float(np.linalg.norm(v))
                if dist > r + 20.0:  # tangent point where the path meets the circle going its way
                    alpha = math.acos(r / dist)
                    base = math.atan2(v[1], v[0])
                    best = None
                    for sgn in (1.0, -1.0):
                        th = base + sgn * alpha
                        j = c + r * np.array([math.cos(th), math.sin(th)])
                        along = d * np.array([-math.sin(th), math.cos(th)])  # its way round
                        score = float((j - prev) @ along)
                        if best is None or score > best[0]:
                            best = (score, j)
                    assert best is not None
                    alt = it.alt if it.kind != ITEM_RTL else prev_alt
                    spd = it.speed if it.kind != ITEM_RTL else prev_speed
                    out.append(self.item(ITEM_WAYPOINT, best[1], alt, spd))
                out.append(it)
                prev = c
                if it.kind != ITEM_RTL:  # the way on from a circle keeps its height and speed
                    prev_alt, prev_speed = it.alt, it.speed
                continue
            out.append(it)
            if it.kind not in (ITEM_WAYPOINT, ITEM_FLYOVER):
                continue
            prev_alt, prev_speed = it.alt, it.speed
            p = np.array([it.north, it.east])
            u_in, u_out = p - prev, heading_to(i) - p
            prev = p
            if np.linalg.norm(u_in) < 1.0 or np.linalg.norm(u_out) < 1.0:
                continue
            u_in, u_out = u_in / np.linalg.norm(u_in), u_out / np.linalg.norm(u_out)
            if u_in @ u_out > -0.7:  # not a reversal
                continue
            rm = self.r_min(it.speed or self.cruise)
            best = None
            for side, sign in (("right", 1.0), ("left", -1.0)):
                normal = sign * np.array([-u_in[1], u_in[0]])
                turn = p + 0.6 * rm * u_in + 1.5 * rm * normal
                centre = p + 0.5 * rm * u_in + rm * normal
                room = float(min(self.cm.at(turn[None])[0], self.cm.at(centre[None])[0] - rm))
                if best is None or room > best[0]:
                    best = (room, side, turn)
            assert best is not None
            out.append(it._replace(kind=ITEM_WAYPOINT, north=float(best[2][0]),
                                   east=float(best[2][1])))  # fmt: skip
            sides.append(best[1])
            prev = best[2]
        out = [it._replace(p1=float(where[int(it.p1)])) if it.kind == ITEM_JUMP else it
               for it in out]  # fmt: skip
        return out, where, sides

    def prune(self, opts: list[_Option], ends, rm: float) -> list[_Option]:
        """Options whose ends (and the line between them) are clear, by cost; then the
        clearest of the others."""
        if not self.adapt:
            return opts[:1]
        need_end = TRACK_MARGIN_M + GRID_SLACK_M + 0.5 * self.room * rm
        scored = []
        for o, (a, b) in zip(opts, ends, strict=True):
            line = self.cm.line(a, b)
            end = float(self.cm.at(np.array([a, b])).min())
            scored.append((line >= TRACK_MARGIN_M and end >= need_end, min(line, end), o))
        good = sorted((x for x in scored if x[0]), key=lambda x: x[2].cost)
        rest = sorted((x for x in scored if not x[0]), key=lambda x: -x[1])
        # The clear ones by cost, then the clearest others: the flown preview decides.
        return [x[2] for x in good + rest][:MAX_OPTIONS]

    # ------------------------------------------------------------------ assembly
    def leg_items(
        self, way: list[np.ndarray], opt: _Option, items: list[Item], speed: float, base: int
    ) -> list[Item]:
        """Route waypoints, then the option's items with their JUMPs made absolute."""
        alt = opt.alt if opt.alt is not None else (items[0].alt if items else 0.0)
        out = [self.item(ITEM_WAYPOINT, w, alt, speed) for w in way]
        start = base + len(out)
        out += [it._replace(p1=it.p1 + start) if it.kind == ITEM_JUMP else it for it in items]
        return out

    def home_route(self, exit_: np.ndarray, speed: float, alt: float) -> list[Item] | None:
        way = self.cm.route(exit_, self.gaz.home, self.route_need(speed)) if self.adapt else []
        if way is None:
            return None
        return [self.item(ITEM_WAYPOINT, w, alt, speed) for w in way]

    def mission(self, items: list[Item], home_alt: float) -> Mission:
        model = _energy(self.ac, self.gaz)
        home = (
            float(self.gaz.home[0]),
            float(self.gaz.home[1]),
            float(np.clip(home_alt, self.ops.min_agl_m, self.ops.ceiling_agl_m)),
        )
        return mission_from_items(
            items,
            speed=self.cruise,
            home=home,
            loop=False,
            fence=[tuple(v) for v in self.gaz.geofence],
            fence_alt=self.ops.ceiling_alt_m
            if self.ops.ceiling_alt_m is not None
            else self.ops.ceiling_agl_m + self.ops.fence_alt_margin_m,
            rtl_mah=model.capacity_mah * (1.0 - RESERVE),
        )


# --------------------------------------------------------------------------- previews
def _fly_one(sp: StackParams, mission: Mission, pos0, psi0, alt0, wind, n: int):
    """The preview aircraft: a point mass in coordinated turns at the commanded airspeed. Its
    bank follows the guidance's demand with the roll loop's time constant and rate limit, but
    stops 5° short of the roll limit (the margin r_min keeps), so its turns are a little wider
    than the aircraft's and a real turn stays inside the one judged. Its height moves towards
    the target within TECS's limits."""
    ap0 = ap_init(MODE_AUTO, pos0, alt0)
    climb_max, sink_max = sp.tecs.climb_max, sp.tecs.sink_max
    tau, rate = sp.att.roll_tau, sp.att.roll_rate_max

    def body(c, _):
        pos, psi, alt, ap, t, v, bank = c
        vel = v * jnp.stack([jnp.cos(psi), jnp.sin(psi)]) + wind
        ap, tgt = _mission_step(sp, ap, pos, mission, vel=vel, t=t, height=alt)
        wp = l1_waypoint(sp.l1, pos, vel, tgt.a, tgt.b)
        lo = l1_loiter(sp.l1, pos, vel, tgt.centre, tgt.radius, tgt.direction)
        lat = jnp.where(tgt.lateral == LAT_LOITER, lo.lat_acc, wp.lat_acc)
        demand = roll_from_lat_acc(lat, tgt.roll_limit - math.radians(BANK_MARGIN_DEG), G0)
        phi = bank + jnp.clip((demand - bank) / tau, -rate, rate) * PREVIEW_DT
        v = tgt.speed
        climb = jnp.clip((tgt.alt - alt) / ALT_TAU_S, -sink_max, climb_max)
        out = (pos, alt, ap.wp_idx, ap.mode, ap.captured, phi, v, climb)
        pos = pos + vel * PREVIEW_DT
        psi = psi + G0 * jnp.tan(phi) / v * PREVIEW_DT
        alt = alt + climb * PREVIEW_DT
        return (pos, psi, alt, ap, t + PREVIEW_DT, v, phi), out

    z = jnp.zeros_like(alt0)
    c0 = (pos0, psi0, alt0, ap0, z, mission.speed, z)
    return jax.lax.scan(body, c0, None, length=n)[1]


_fly = partial(jax.jit, static_argnames=("n",))(_fly_one)


@partial(jax.jit, static_argnames=("n",))
def _fly_many(sp, missions, pos0, psi0, alt0, wind, n: int):
    return jax.vmap(lambda m, p, s, a: _fly_one(sp, m, p, s, a, wind, n))(
        missions, pos0, psi0, alt0
    )


def _f(x):
    return jnp.asarray(x, jnp.result_type(float))  # the mission's float (float64 with x64)


def _evaluate(ctx: _Ctx, lanes: list[tuple[list[Item], _State, float, int]]) -> list[dict]:
    """Fly candidate missions at once from their start states. Each lane is (items, start,
    home height, how many items at the end are the way home). Per lane: whether it comes
    within the tracking margin of a no-fly area, its least clearance, and the state where the
    step ends (when the aircraft moves on to the way home)."""
    sp = stack_params(ctx.ac, ctx.gaz)
    n = int(PREVIEW_STEPS_S[0] / PREVIEW_DT)
    results: list[dict] = []
    for k in range(0, len(lanes), LANES):
        chunk = lanes[k : k + LANES]
        pad = chunk + [chunk[0]] * (LANES - len(chunk))
        ms = [ctx.mission(items, alt) for items, _, alt, _ in pad]
        stacked = jax.tree.map(lambda *x: jnp.stack(x), *ms)
        with jax.default_device(jax.devices("cpu")[0]):
            out = _fly_many(
                sp,
                stacked,
                _f([st.pos for _, st, _, _ in pad]),
                _f([math.radians(st.course) for _, st, _, _ in pad]),
                _f([st.alt for _, st, _, _ in pad]),
                _f(ctx.ac.wind_ne),
                n,
            )
        pos, alt, item, mode = (np.asarray(x) for x in out[:4])
        for j, (items, st, _, tail) in enumerate(chunk):
            ne = pos[j]
            d = ctx.cm.at(ne) - GRID_SLACK_M
            hit = d < TRACK_MARGIN_M + EVAL_EXTRA_M
            if hit[0]:
                out_ = np.nonzero(~hit)[0]
                hit[: out_[0] if len(out_) else len(hit)] = False
            # The step ends when the aircraft moves past its items (to the route home / RTL).
            first_after = np.nonzero((item[j] >= len(items) - tail) | (mode[j] == MODE_RTL))[0]
            if len(first_after):
                e = max(int(first_after[0]), 1)
                v = ne[e] - ne[e - 1]
                course = (
                    math.degrees(math.atan2(v[1], v[0])) if np.linalg.norm(v) > 0 else st.course
                )
                end = _State(ne[e].astype(float), course, float(alt[j][e]), st.speed)
            else:  # not finished within the preview: hand over from where the step leaves
                last = [it for it in items[: len(items) - tail] if it.kind != ITEM_JUMP][-2:]
                ends = [np.array([it.north, it.east]) for it in last]
                v = ends[-1] - ends[0] if len(ends) == 2 else _unit(st.course)
                course = math.degrees(math.atan2(v[1], v[0]))
                end, e = _State(ends[-1], course, float(last[-1].alt), st.speed), len(ne) - 1
            results.append(
                {
                    "conflict": bool(hit.any()),
                    "clear": float(d.min()),
                    "end": end,
                    "t": e * PREVIEW_DT,
                }
            )
    return results


def _preview(ctx: _Ctx, items: list[Item], mission: Mission) -> Preview:
    """The final preview from the aircraft's state, cut where the plan's pattern is complete
    (one circle over home after RTL, two laps of a patrol, one turn of an endless circle). A
    short run first; the full horizon only when the plan needs it."""
    pv, complete = _preview_n(ctx, items, mission, int(PREVIEW_STEPS_S[0] / PREVIEW_DT))
    for seconds in PREVIEW_STEPS_S[1:]:
        if complete:
            break
        pv, complete = _preview_n(ctx, items, mission, int(seconds / PREVIEW_DT))
    return pv


def _preview_n(ctx: _Ctx, items: list[Item], mission: Mission, n: int) -> tuple[Preview, bool]:
    sp = stack_params(ctx.ac, ctx.gaz)
    ac = ctx.ac
    with jax.default_device(jax.devices("cpu")[0]):
        out = _fly(
            sp,
            mission,
            _f(ac.pos_ne),
            _f(math.radians(ac.course_deg)),
            _f(ac.alt_agl),
            _f(ac.wind_ne),
            n=n,
        )
    pos, alt, item, mode, captured, phi, v, climb = (np.asarray(x) for x in out)
    t = np.arange(n) * PREVIEW_DT
    end = n
    kinds = np.array([it.kind for it in items])
    rtl = np.nonzero(mode == MODE_RTL)[0]
    if len(rtl):  # one circle over home
        loiter_r = ctx.cfg.l1.loiter_radius_m
        end = min(n, rtl[0] + int(2 * math.pi * loiter_r / ctx.cruise / PREVIEW_DT) + 1)
    elif (kinds == ITEM_JUMP).any():  # two laps
        j = int(np.nonzero(kinds == ITEM_JUMP)[0][0])
        back = np.nonzero((item[1:] == int(items[j].p1)) & (item[:-1] > item[1:]))[0]
        if len(back) >= 2:
            end = min(n, back[1] + 2)  # up to the first sample of the third lap
    elif (kinds == ITEM_LOITER_UNLIM).any():  # one turn on the circle
        cap = np.nonzero(captured & (kinds[np.minimum(item, len(kinds) - 1)] == ITEM_LOITER_UNLIM))[
            0
        ]
        if len(cap):
            r = abs(items[int(item[cap[0]])].radius)
            end = min(n, cap[0] + int(2 * math.pi * r / float(v[cap[0]]) / PREVIEW_DT) + 1)
    used = _energy(ac, ctx.gaz).used_mah(t, v, climb)
    sl = slice(0, end)
    pv = Preview(
        t[sl],
        pos[sl],
        alt[sl],
        item[sl],
        mode[sl],
        captured[sl],
        np.degrees(phi[sl]),
        v[sl],
        climb[sl],
        used[sl],
    )
    return pv, end < n


@dataclass
class _Laps:
    to_lap2_s: float  # from now to the end of the first lap (the way out included)
    to_lap2_mah: float
    lap_s: float  # one lap (the second, measured in the preview)
    lap_mah: float
    home_s: float  # from the farthest point of the pattern back home
    home_mah: float


def _lap_stats(ctx: _Ctx, items: list[Item], jump: int, pv: Preview) -> _Laps | None:
    first = int(items[jump].p1)
    at = pv.item == first
    starts = np.nonzero(at & np.concatenate([[True], ~at[:-1]]))[0]
    if len(starts) < 3:
        return None
    i, j = starts[1], starts[2]
    far = float(np.max(np.linalg.norm(pv.ne - ctx.gaz.home, axis=1)))
    home_s = far / max(ctx.cruise - ctx.ac.wind_speed, 1.0)
    amps = float(_energy(ctx.ac, ctx.gaz).current(np.array([ctx.cruise]), np.zeros(1))[0])
    return _Laps(
        float(pv.t[i]),
        float(pv.used_mah[i]),
        float(pv.t[j] - pv.t[i]),
        float(pv.used_mah[j] - pv.used_mah[i]),
        home_s,
        home_s * amps / 3.6,
    )


# --------------------------------------------------------------------------- build
def _search(ctx: _Ctx) -> tuple[list[_Node], list[str]]:
    """Beam search over the steps' options; returns the best complete plans (best first) and
    any errors."""
    ac = ctx.ac
    start = _State(
        np.asarray(ac.pos_ne, float), ac.course_deg, float(round(ac.alt_agl)), ctx.cruise
    )
    nodes = [_Node([], start)]
    errors: list[str] = []
    first_flight = True
    for k, step in enumerate(ctx.plan.steps):
        if nodes[0].endless is not None:
            errors.append(f"step {k + 1} would never be flown: {nodes[0].endless}")
            break
        children: list[_Node] = []
        lanes: list[tuple[list[Item], _State, float, int]] = []
        meta: list[tuple[_Node, _Option, list[np.ndarray], float]] = []
        try:
            for node in nodes:
                opts = ctx.options(k, step, node.state)
                for opt in opts:
                    if opt.entry is None:  # no flight (an airspeed change)
                        st = replace(node.state, speed=opt.speed or node.state.speed)
                        children.append(
                            _Node(
                                [*node.legs, ([], opt)],
                                st,
                                node.cost + opt.cost,
                                node.clear,
                                node.feasible,
                                node.endless,
                                node.t,
                            )
                        )
                        continue
                    speed = node.state.speed
                    need = ctx.route_need(speed)
                    way = ctx.cm.route(node.state.pos, opt.entry, need) if ctx.adapt else []
                    if way is None:
                        continue
                    starts = [[]]
                    if first_flight and ctx.adapt:  # which way to turn first
                        rm = ctx.r_min(speed)
                        for turn in (-70.0, 70.0, -130.0, 130.0):  # gently, then hard
                            starts.append(
                                [node.state.pos + 2.5 * rm * _unit(node.state.course + turn)]
                            )
                    for dep in starts:
                        wps = [*dep, *way]
                        body = opt.eval_items or opt.items
                        alt = opt.alt if opt.alt is not None else node.state.alt
                        items = ctx.leg_items(wps, opt, body, speed, 0)
                        home_at = len(items)
                        if opt.endless is None and opt.exit is not None:  # then home too
                            home = ctx.home_route(opt.exit, speed, alt) or []
                            items = items + home + [ctx.item(ITEM_RTL)]
                        if ctx.adapt:
                            items, where, _ = ctx.shape(items, node.state.pos)
                            home_at = where[home_at] if home_at < len(where) else len(items)
                        tail = len(items) - home_at
                        direct = float(np.linalg.norm(opt.entry - node.state.pos))
                        path = sum(
                            float(np.linalg.norm(b - a))
                            for a, b in zip([node.state.pos, *wps], [*wps, opt.entry], strict=True)
                        )
                        detour = 0.2 * max(path / max(direct, 1.0) - 1.0, 0.0) + 0.1 * len(dep)
                        lanes.append((items, node.state, alt, tail))
                        meta.append((node, opt, wps, detour))
        except (UnknownPlace, ValueError) as e:
            errors.append(f"step {k + 1}: {e}")
            break
        if lanes:
            for (node, opt, wps, detour), res in zip(meta, _evaluate(ctx, lanes), strict=True):
                feasible = node.feasible and not res["conflict"]
                # The next step starts where this one ends, at its planned height.
                end = replace(res["end"], alt=opt.alt if opt.alt is not None else node.state.alt)
                children.append(
                    _Node(
                        [*node.legs, (wps, opt)],
                        end,
                        node.cost + opt.cost + detour,
                        min(node.clear, res["clear"]),
                        feasible,
                        opt.endless,
                        node.t + res["t"],
                    )
                )
            first_flight = False
        if not children:
            errors.append(f"step {k + 1}: no way to fly it clear of the no-fly areas")
            break
        children.sort(key=lambda c: (not c.feasible, c.cost if c.feasible else -c.clear, c.t))
        last = k == len(ctx.plan.steps) - 1
        nodes = children[: FINALISTS if last else BEAM]
    return nodes, errors


def _final_clear(ctx: _Ctx, pv: Preview) -> float:
    """Least exact clearance of a full preview from no-fly areas and the geofence's edge, as
    the checker measures it (an area the aircraft starts in counts once it has left it)."""
    least = math.inf
    edges = [lambda q: -geo.signed_distance(q, ctx.gaz.geofence)]
    edges += [a.signed_distance for a in ctx.gaz.no_fly]
    for dist in edges:
        d = dist(pv.ne)
        near = d < TRACK_MARGIN_M
        start = 0
        if near[0]:
            out = np.nonzero(~near)[0]
            start = int(out[0]) if len(out) else len(d)
        if start < len(d):
            least = min(least, float(d[start:].min()))
    return least


def _lift(
    ctx: _Ctx, items: list[Item], pv: Preview, home_alt: float, steps: list[StepGeom]
) -> tuple[list[Item], float, Preview, list[str]]:
    """On a world site, raise the legs that would pass too low over the ground or what stands
    on it (docs/07 §7): each to the lowest multiple of 5 m that keeps ``min_agl`` above the
    ground and 30 m above every obstacle, within the tracking margin. A leg flown on the
    climb from a lower one may need that one raised too: up to four passes."""
    rel = relief(ctx.gaz.site)
    if rel is None or not ctx.adapt:
        return items, home_alt, pv, []
    ops = ctx.ops
    items = list(items)
    raised: dict[int, tuple[float, np.ndarray]] = {}  # item → (new height, binding point)
    for _ in range(4):
        floor = rel.floor(pv.ne, ops.min_agl_m)
        short = pv.alt < floor - 0.5
        short[: climb_out(pv.alt, floor)] = False  # the climb from a start already too low
        if not short.any():
            break
        rtl = pv.mode == MODE_RTL
        for k in np.unique(pv.item[short & ~rtl]):
            leg = short & (pv.item == k) & ~rtl
            j = int(np.argmax(np.where(leg, floor, -np.inf)))
            need = ceil5(float(floor[j]))
            k = int(k)
            # Already high enough: the climb from the leg before starts too low.
            k = k if items[k].alt < need - 0.5 or k == 0 else k - 1
            while k > 0 and items[k].kind in (ITEM_JUMP, ITEM_RTL):
                k -= 1
            if items[k].alt < need:
                items[k] = items[k]._replace(alt=need)
                raised[k] = (need, pv.ne[j])
        if (short & rtl).any():
            j = int(np.argmax(np.where(short & rtl, floor, -np.inf)))
            if home_alt < ceil5(float(floor[j])):
                home_alt = ceil5(float(floor[j]))
                raised[-1] = (home_alt, pv.ne[j])
        pv = _preview(ctx, items, ctx.mission(items, home_alt))
    by_label: dict[str, tuple[float, np.ndarray]] = {}
    for k, (alt, p) in sorted(raised.items()):
        g = next((g for g in steps if g.items[0] <= k < g.items[1]), None)
        label = "the way home" if k < 0 else (g.label if g is not None else "the route")
        if g is not None:
            g.alt = max(g.alt or 0.0, alt)
        if label not in by_label or alt > by_label[label][0]:
            by_label[label] = (alt, p)
    notes = []
    for label, (alt, p) in by_label.items():
        what = landmark(ctx.gaz, p)
        if rel.by_obstacle(p[None], ops.min_agl_m)[0]:
            why = f"{CLEARANCE_M:g} m above {what or 'the trees and buildings'}"
        else:
            why = f"{ops.min_agl_m:g} m above the rising ground" + (f" at {what}" if what else "")
        notes.append(f"{label}: at {alt:g} m (heights are above home) to keep {why}")
    return items, home_alt, pv, notes


CLIMB_SEARCH_M = 400.0  # how far from the aircraft a climb circle may be
CLIMB_RATE_SHARE = 0.6  # of TECS's best climb rate, for the first guess at the turns needed


def _near(gaz: Gazetteer, p: np.ndarray) -> str:
    """A place to name a point by: the nearest one within 300 m (streets left out)."""
    best, name = 300.0, None
    for place in gaz.places.values():
        if place.kind in ("road", "street", "lane", "home"):
            continue
        d = float(np.linalg.norm(place.point - p))
        if d < best:
            best, name = d, place.name
    return f"near {name}" if name else "where it is"


def _late_climb(ctx: _Ctx, items: list[Item], pv: Preview, home_alt: float) -> np.ndarray:
    """Where the track passes too low only because the climb to its leg's height comes late:
    the leg itself is high enough there (what ``_lift`` cannot fix by raising a leg)."""
    rel = relief(ctx.gaz.site)
    assert rel is not None
    floor = rel.floor(pv.ne, ctx.ops.min_agl_m)
    low = pv.alt < floor - 1.0
    low[: climb_out(pv.alt, floor)] = False
    flat = (ITEM_JUMP, ITEM_RTL)
    leg = np.array([home_alt if items[int(k)].kind in flat else items[int(k)].alt for k in pv.item])
    return low & (leg >= floor - 0.5)


def _climb_first(
    ctx: _Ctx, items: list[Item], pv: Preview, home_alt: float
) -> tuple[list[Item], Preview, int, str] | None:
    """On a world site, when the track passes too low because the aircraft climbs while it
    travels (a long climb from a low start, past tall buildings), climb first: circle where the
    ground and what stands on it are lowest, nearby and clear of no-fly areas, up to the height
    the plan needs, then go. Returns the items (all shifted by the climb's), the preview, the
    shift and a note; None if no climb is needed or none helps."""
    rel = relief(ctx.gaz.site)
    if rel is None or not ctx.adapt or not items:
        return None
    late = _late_climb(ctx, items, pv, home_alt)
    if not late.any():
        return None
    flat = (ITEM_JUMP, ITEM_RTL)
    last = int(pv.item[np.nonzero(late)[0][-1]])
    height = max(it.alt for it in items[: last + 1] if it.kind not in flat)
    first = next((it for it in items if it.kind not in flat), None)
    if first is None:
        return None
    cruise = ctx.cruise
    radius = _ceil10(max(ctx.cfg.l1.loiter_radius_m, ctx.r_min(cruise)))
    need = ctx.route_need(cruise)
    pos = np.asarray(ctx.ac.pos_ne, float)
    # Candidate centres round the aircraft, best first: the lowest floor round the circle.
    span = np.arange(-CLIMB_SEARCH_M, CLIMB_SEARCH_M + 1.0, 50.0)
    grid = np.stack(np.meshgrid(span, span, indexing="ij"), -1).reshape(-1, 2) + pos
    grid = grid[np.linalg.norm(grid - pos, axis=1) <= CLIMB_SEARCH_M]
    grid = grid[ctx.cm.at(grid) >= radius + need]
    if not len(grid):
        return None
    theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    ring = np.stack([np.cos(theta), np.sin(theta)], -1) * radius
    worst = np.array([rel.floor(c + ring, ctx.ops.min_agl_m).max() for c in grid])
    order = np.argsort(worst + 0.05 * np.linalg.norm(grid - pos, axis=1))
    turn_s = 2.0 * math.pi * radius / cruise
    rate = CLIMB_RATE_SHARE * ctx.cfg.tecs.climb_max_mps
    for c in grid[order[:3]]:
        way = ctx.cm.route(c, np.array([first.north, first.east]), need)
        if way is None:
            continue
        turns = max(1, math.ceil((height - ctx.ac.alt_agl) / rate / turn_s))
        for _ in range(4):
            climb = [ctx.item(ITEM_LOITER_TURNS, c, height, radius=radius, p1=float(turns))]
            climb += [ctx.item(ITEM_WAYPOINT, w, height) for w in way]
            shift = len(climb)
            new = climb + [it._replace(p1=it.p1 + shift) if it.kind == ITEM_JUMP else it
                           for it in items]  # fmt: skip
            pv2 = _preview(ctx, new, ctx.mission(new, home_alt))
            late2 = _late_climb(ctx, new, pv2, home_alt)
            if not late2.any():
                where = _near(ctx.gaz, c)
                note = (f"climbs to {height:g} m first (heights are above home), circling "
                        f"{where}, so the way there keeps {CLEARANCE_M:g} m above the trees and "
                        "buildings")  # fmt: skip
                return new, pv2, shift, note
            if (pv2.item[late2] < shift).any():
                break  # too low while circling: another centre
            turns += 1  # left the circle before it was high enough
    return None


ROOMS = (1.0, 2.0, 3.0)  # searches with more and more room to turn, while none keeps clear


def build(plan: Plan, ac: Aircraft, gaz: Gazetteer, *, adapt: bool = True) -> Built:
    fallback: tuple[float, Built] | None = None
    for room in ROOMS if adapt else ROOMS[:1]:
        ctx = _Ctx(plan, ac, gaz, adapt, room)
        nodes, errors = _search(ctx)
        if errors and fallback is not None:  # more room found nothing: keep the closest so far
            continue
        if errors or not adapt:
            return _assemble(ctx, nodes[0], errors)
        for node in nodes:  # the best first; the next if the full flight comes too close
            b = _assemble(ctx, node, [])
            if b.errors or b.preview is None:
                fallback = fallback or (-math.inf, b)
                continue
            clear = _final_clear(ctx, b.preview)
            if clear >= TRACK_MARGIN_M:
                return b
            if fallback is None or clear > fallback[0]:
                fallback = (clear, b)
    assert fallback is not None
    return fallback[1]


def _assemble(ctx: _Ctx, best: _Node, errors: list[str]) -> Built:
    plan, ac, gaz = ctx.plan, ctx.ac, ctx.gaz
    errors = list(errors)
    items: list[Item] = []
    steps: list[StepGeom] = []
    notes: list[str] = []
    jump = None
    until_reserve = False
    speed = ctx.cruise
    last_exit, last_alt = None, float(round(ac.alt_agl))
    for wps, opt in best.legs:
        if opt.speed is not None:
            speed = opt.speed
        start = len(items)
        leg = ctx.leg_items(wps, opt, opt.items, speed, start)
        for i, it in enumerate(leg):
            if it.kind == ITEM_JUMP:
                jump = start + i
                until_reserve = opt.until_reserve
        items += leg
        steps.append(replace(opt.geom, items=(start, len(items))))
        if wps and opt.entry is not None:
            areas = sorted({ctx.area_at(p) for p in wps})
            notes.append(f"{opt.geom.label}: goes round {' and '.join(areas)} on the way")
        notes += opt.notes
        if opt.exit is not None:
            last_exit, last_alt = opt.exit, opt.alt if opt.alt is not None else last_alt
    endless = best.endless is not None
    if not errors and not items:
        errors.append("the plan has nothing to fly")
    if not errors and not endless and last_exit is not None:
        home = ctx.home_route(last_exit, speed, last_alt)
        if home:
            notes.append(
                f"goes round {ctx.area_at(np.array([home[0].north, home[0].east]))} on the way home"
            )
        items += (home or []) + [ctx.item(ITEM_RTL)]
    if ctx.adapt and items:
        items, where, sides = ctx.shape(items, np.asarray(ac.pos_ne, float))
        where.append(len(items))
        steps = [replace(g, items=(where[g.items[0]], where[g.items[1]])) for g in steps]
        if jump is not None:
            jump = where[jump]
        notes += [f"turns {side} where the route doubles back, on the side with room"
                  for side in dict.fromkeys(sides)]  # fmt: skip
    if len(items) > MAX_ITEMS:
        errors.append(
            f"the plan needs {len(items)} mission items; the autopilot holds "
            f"{MAX_ITEMS}: split it in two"
        )
    notes = list(dict.fromkeys(notes))
    built = Built(plan, ac, items, None, steps, None, None, ctx.rmin, endless, notes, errors)
    if errors:
        return built
    model = _energy(ac, gaz)
    available = model.capacity_mah * (1.0 - RESERVE) - ac.used_mah
    mission = ctx.mission(items, last_alt)
    pv = _preview(ctx, items, mission)
    items, last_alt, pv, lifted = _lift(ctx, items, pv, last_alt, steps)
    if lifted:
        notes += lifted
        mission = ctx.mission(items, last_alt)
    climb = _climb_first(ctx, items, pv, last_alt)
    if climb is not None:
        items, pv, shift, note = climb
        steps = [replace(g, items=(g.items[0] + shift, g.items[1] + shift)) for g in steps]
        built.steps = steps
        jump = None if jump is None else jump + shift
        notes.append(note)
        mission = ctx.mission(items, last_alt)
    laps = lap_mah = None
    st: _Laps | None = None
    if jump is not None and (st := _lap_stats(ctx, items, jump, pv)) is not None:
        lap_mah = st.lap_mah
        if until_reserve:
            spare = available - st.to_lap2_mah - st.home_mah
            laps = 1 + int(spare // st.lap_mah) if spare >= 0 else 0
            if laps >= 1:
                items[jump] = items[jump]._replace(p2=float(laps - 1))
                mission = ctx.mission(items, last_alt)
                pv = _preview(ctx, items, mission)
                notes.append(f"patrol: {laps} laps before the battery reserve (estimated)")
        else:
            laps = int(items[jump].p2) + 1
    returned = bool((pv.mode == MODE_RTL).any())
    if endless and items[-1].kind == ITEM_LOITER_UNLIM:
        needed = None  # until the battery reserve, then home (the stack's failsafe)
        rate = float(model.current(np.array([ctx.cruise]), np.zeros(1))[0]) / 3.6
        spare = max(available - float(pv.used_mah[-1]), 0.0)
        minutes = (float(pv.t[-1]) + spare / rate) / 60.0
    elif returned or laps is None:
        needed, minutes = float(pv.used_mah[-1]), float(pv.t[-1]) / 60.0
        if not returned:
            notes.append(
                f"the plan is longer than the {PREVIEW_S / 60:.0f} min preview: "
                "the battery estimate covers only that part"
            )
    else:  # the preview stops after two laps: add the rest and the way home
        assert lap_mah is not None and st is not None
        more = max(laps - 2, 0)
        needed = float(pv.used_mah[-1]) + more * lap_mah + st.home_mah
        minutes = (float(pv.t[-1]) + more * st.lap_s + st.home_s) / 60.0
    built.items, built.mission, built.preview, built.notes = items, mission, pv, notes
    built.energy = Energy(available, needed, minutes, laps, lap_mah)
    return built
