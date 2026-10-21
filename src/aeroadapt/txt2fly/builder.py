"""Plan → mission (spec 06 §6): airframe- and wind-aware, deterministic.

The builder turns a plan's steps into mission v2 items for the aircraft flying and the wind
blowing, then *previews* the mission: a point-mass aircraft flies it with the stack's own
item sequencer and L1 guidance, banking up to the roll limit in the current wind. The checker
and the map both use that preview, so turns, overshoots and loiter captures are where the
stock autopilot would put them, not where a straight-line sketch would. (The real aircraft
also has roll dynamics, gusts and sensor error; V26 measures the gap.)

The tightest circle the plan may ask for keeps a 5° bank margin for tracking corrections, at
the ground speed downwind:  r_min = (V + W)² / (g · tan(φ_lim − 5°)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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

G0 = 9.806_65
BANK_MARGIN_DEG = 5.0
RUN_MIN_M = 100.0  # shortest straight run before and after a pass
TRACK_MARGIN_M = 30.0  # the checker's margin round no-fly areas (kept here to plan with)
MAX_LEVEL = 4  # crossing-point moves tried per place
LEVEL_CAP = 4 * (MAX_LEVEL + 1)  # … times the four directions of a pass
MAX_REPAIRS = 10  # distinct attempts previewed
PREVIEW_DT = 0.1
PREVIEW_STEPS_S = (300.0, 1500.0)  # a short preview first, the long one if the plan needs it
PREVIEW_S = PREVIEW_STEPS_S[-1]
ALT_TAU_S = 5.0  # height response of the preview aircraft (rate-limited by TECS's limits)


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
    notes: list[str] = field(default_factory=list)  # readings the builder chose
    errors: list[str] = field(default_factory=list)  # K1: plans that cannot be built


def _unit(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([math.cos(r), math.sin(r)])


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


class _Builder:
    def __init__(
        self, plan: Plan, ac: Aircraft, gaz: Gazetteer, levels: dict[tuple[int, int], int]
    ) -> None:
        self.plan, self.ac, self.gaz = plan, ac, gaz
        self.levels = levels  # per crossing (step, place): how far to move it from no-fly areas
        self.visit_of: dict[int, tuple[int, int]] = {}  # item → its crossing
        self.k = 0  # the step being built
        self.cfg = ac.stack_cfg
        self.cruise = self.cfg.tecs.airspeed_cruise_mps
        self.items: list[Item] = []
        self.steps: list[StepGeom] = []
        self.notes: list[str] = []
        self.errors: list[str] = []
        self.rmin: dict[float, float] = {}
        self.cursor = np.asarray(ac.pos_ne, float)
        self.course = _unit(ac.course_deg)
        self.alt = float(round(ac.alt_agl))
        self.speed = self.cruise
        self.endless: str | None = None  # why the plan never ends
        self.jump: int | None = None  # index of a patrol's JUMP item
        self.until_reserve = False  # … whose laps are sized to the battery

    # ------------------------------------------------------------------ helpers
    def _r_min(self, speed: float) -> float:
        if speed not in self.rmin:
            self.rmin[speed] = r_min(speed, self.ac.wind_speed, self.cfg.l1.roll_limit_deg)
        return self.rmin[speed]

    def _item(self, kind: int, ne=(0.0, 0.0), alt=0.0, visit=None, **kw) -> int:
        spd = 0.0 if self.speed == self.cruise else self.speed
        self.items.append(Item(kind, float(ne[0]), float(ne[1]), float(alt), speed=spd, **kw))
        if visit is not None:
            self.visit_of[len(self.items) - 1] = visit
        return len(self.items) - 1

    def _alt(self, alt_m: float | None) -> float:
        if alt_m is not None:
            self.alt = float(alt_m)
        return self.alt

    def _resolve(self, target) -> Resolved:
        return self.gaz.resolve(target, np.asarray(self.ac.pos_ne, float))

    # ------------------------------------------------------------------ steps
    def _clearances(self, pts: np.ndarray) -> np.ndarray:
        """Per point: distance to the nearest no-fly area or the geofence's edge."""
        d = -geo.signed_distance(pts, self.gaz.geofence)
        for area in self.gaz.no_fly:
            d = np.minimum(d, area.signed_distance(pts))
        return d

    def _clearance(self, pts: np.ndarray) -> float:
        return float(self._clearances(pts).min())

    def _over(self, r: Resolved, key: tuple[int, int], level: int | None = None) -> np.ndarray:
        """Where to cross a place: its centre, unless the preview showed the turns there come too
        close to a no-fly area; then (level n) the point inside it, at least 20 m in from its
        edge, nearest the centre and (0.5 + 0.5 n) turn radii plus the margin clear of every
        no-fly area, or at the last level the clearest point."""
        level = self.levels.get(key, 0) if level is None else level
        if r.polygon is None or level == 0:
            return r.point
        lo, hi = r.polygon.min(axis=0), r.polygon.max(axis=0)
        grid = np.stack(np.meshgrid(np.arange(lo[0], hi[0], 10.0), np.arange(lo[1], hi[1], 10.0)),
                        -1).reshape(-1, 2)  # fmt: skip
        grid = grid[geo.signed_distance(grid, r.polygon) <= -20.0]
        if not len(grid):
            return r.point
        clear = self._clearances(grid)
        need = (0.5 + 0.5 * level) * self._r_min(self.speed) + TRACK_MARGIN_M
        ok = clear >= need
        if level < MAX_LEVEL and ok.any():
            best = grid[ok][int(np.argmin(np.linalg.norm(grid[ok] - r.point, axis=1)))]
        else:
            best = grid[int(np.argmax(clear))]
        self.notes.append(f"crosses {r.name} away from its middle, to keep the turns clear of "
                          f"{self._nearest_no_fly(best)}")  # fmt: skip
        return best

    def _nearest_no_fly(self, pt: np.ndarray) -> str:
        areas = sorted(self.gaz.no_fly, key=lambda a: float(a.signed_distance(pt[None])[0]))
        return areas[0].name if areas else "the geofence"

    def _pass(self, r: Resolved, heading, alt: float, key: tuple[int, int]) -> np.ndarray:
        """Run-in, fly-over and run-out items for a pass over ``r``; returns the line.

        Along a free axis (``along``) over a field, the candidates are its long axis either
        way, then across it either way: the first whose run-in and run-out points are a turn's
        width clear of the no-fly areas and the geofence is taken (the nearer run-in first), or
        if none is, the clearest."""
        level = self.levels.get(key, 0)
        free = heading == "along" or (heading == "into_wind" and self.ac.wind_speed < 1.0)
        n_dirs = 4 if free and r.polygon is not None else 1
        # Each repair of a pass first tries the next direction, then moves the crossing.
        p = self._over(r, key, level // n_dirs)
        approach = p - self.cursor
        if np.linalg.norm(approach) < 1.0:
            approach = self.course
        approach = approach / np.linalg.norm(approach)
        rm = self._r_min(self.speed)
        if heading == "into_wind" and self.ac.wind_speed >= 1.0:
            options = [(-np.asarray(self.ac.wind_ne, float) / self.ac.wind_speed, 0)]
        elif isinstance(heading, int | float):
            options = [(_unit(float(heading)), 0)]
        else:
            if heading == "into_wind":
                self.notes.append(f"calm air: the pass over {r.name} follows its long axis")
            if r.polygon is not None:
                ax = geo.long_axis(r.polygon)[0]
                across = np.array([-ax[1], ax[0]])
                options = [(ax, 0), (-ax, 0), (across, 1), (-across, 1)]
            else:
                options = [(approach, 0)]

        def ends(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            half = 0.0
            if r.polygon is not None:
                proj = (r.polygon - p) @ v
                half = float(max(proj.max(), -proj.min()))
            run_in = half + max(rm, RUN_MIN_M)  # room to line up
            run_out = half + max(0.5 * rm, 0.5 * RUN_MIN_M)  # wings level past the far edge
            return p - run_in * v, p + run_out * v

        def score(o: tuple[np.ndarray, int]) -> tuple[int, int, float]:
            """Options whose line is clear and whose ends are a turn's width clear first (along
            before across, nearer run-in first); then clear lines with the clearest ends; a
            line through a no-fly area last."""
            a, b = ends(o[0])
            line = self._clearance(np.linspace(a, b, max(int(np.linalg.norm(b - a) // 10), 2)))
            c = self._clearance(np.array([a, b]))
            if line < TRACK_MARGIN_M:
                return (2, 0, -line)
            if c < rm:
                return (1, 0, -c)
            return (0, o[1], float(np.linalg.norm(a - self.cursor)))

        ranked = sorted(options, key=score)
        u, rank = ranked[level % len(ranked)]
        if score((u, rank))[0] == 2:  # never a line through a no-fly area by choice
            u, rank = ranked[0]
        if rank == 1:
            self.notes.append(
                f"the pass runs across {r.name}: along it, the turns would come too close to "
                "a no-fly area or the geofence"
            )
        a, b = ends(u)
        self._item(ITEM_WAYPOINT, a, alt, visit=key)
        self._item(ITEM_FLYOVER, p, alt, visit=key)
        self._item(ITEM_WAYPOINT, b, alt, visit=key)
        self.cursor, self.course = b, u
        return np.array([a, b])

    def _radius(self, radius_m: float | None, what: str) -> float:
        if radius_m is not None:
            return float(radius_m)
        r = max(self.cfg.l1.loiter_radius_m, self._r_min(self.speed))
        r = math.ceil(r / 10.0) * 10.0
        self.notes.append(f"{what}: a {r:g} m circle (this aircraft's usual)")
        return r

    def step(self, k: int, s) -> None:
        if self.endless is not None:
            self.errors.append(f"step {k + 1} would never be flown: {self.endless}")
            return
        first = len(self.items)
        g = StepGeom(k, s.do, s.do)
        if isinstance(s, SetSpeed):
            v = _speeds(self.cfg)[s.speed] if isinstance(s.speed, str) else float(s.speed)
            self.speed = v
            g.label, g.speed = f"airspeed {v:g} m/s", v
        elif isinstance(s, Goto):
            r = self._resolve(s.target)
            alt = self._alt(s.alt_m)
            self._item(ITEM_WAYPOINT, r.point, alt)
            self.course = r.point - self.cursor
            self.cursor = r.point
            g.label, g.target, g.alt, g.speed = f"go to {r.name}", r, alt, self.speed
        elif isinstance(s, PassOver):
            r = self._resolve(s.target)
            alt = self._alt(s.alt_m)
            g.line = self._pass(r, s.heading, alt, (k, 0))
            g.label, g.target, g.alt, g.speed = f"pass over {r.name}", r, alt, self.speed
        elif isinstance(s, Orbit | Hold):
            if isinstance(s, Orbit):
                r = self._resolve(s.target)
                alt = self._alt(s.alt_m)
                radius = self._radius(s.radius_m, f"circling {r.name}")
                sign = 1.0 if s.direction == "cw" else -1.0
                turns, secs = s.turns, s.duration_s
            else:
                r = Resolved("the aircraft's position", np.asarray(self.ac.pos_ne, float))
                alt = self.alt
                radius = self._radius(s.radius_m, "holding")
                sign, turns, secs = 1.0, None, None
            if turns is not None and secs is not None:
                self.notes.append(f"circling {r.name}: {turns:g} turns (the time is ignored)")
            if turns is not None:
                self._item(ITEM_LOITER_TURNS, r.point, alt, radius=sign * radius, p1=turns)
            elif secs is not None:
                self._item(ITEM_LOITER_TIME, r.point, alt, radius=sign * radius, p1=secs)
            else:
                self._item(ITEM_LOITER_UNLIM, r.point, alt, radius=sign * radius)
                self.endless = f"circling {r.name} never ends (until the battery reserve)"
            self.cursor = r.point
            what = "hold" if isinstance(s, Hold) else "circle"
            g.label, g.target, g.alt, g.speed = f"{what} {r.name}", r, alt, self.speed
            g.centre, g.radius = r.point, radius
        elif isinstance(s, Patrol):
            if len(s.targets) < 2:
                self.errors.append(f"step {k + 1}: a patrol needs at least two places")
                return
            alt = self._alt(s.alt_m)
            names = []
            for j, t in enumerate(s.targets):
                r = self._resolve(t)
                names.append(r.name)
                # The route between the places sets each pass's direction: over the middle
                # of each (fly-over), or by it (fly-by, turning early).
                kind = ITEM_FLYOVER if s.visit == "pass_over" else ITEM_WAYPOINT
                over = self._over(r, (k, j))
                self._item(kind, over, alt, visit=(k, j))
                self.cursor = over
            if s.laps is None:  # sized to the battery after a first preview
                self.jump = self._item(ITEM_JUMP, p1=first, p2=-1)
                self.until_reserve = True
            elif s.laps > 1:
                self.jump = self._item(ITEM_JUMP, p1=first, p2=s.laps - 1)
            g.label, g.alt, g.speed = f"patrol {', '.join(names)}", alt, self.speed
        elif isinstance(s, ReturnHome):
            self._item(ITEM_RTL)
            self.endless = "the aircraft has returned home"
            g.label = "return home"
        g.items = (first, len(self.items))
        self.steps.append(g)


def _mission(b: _Builder, items: list[Item]) -> Mission:
    ops = b.gaz.site.operations
    assert ops is not None
    home_alt = float(np.clip(b.alt, ops.min_agl_m, ops.ceiling_agl_m))
    model = _energy(b.ac, b.gaz)
    reserve = RESERVE * model.capacity_mah
    return mission_from_items(
        items,
        speed=b.cruise,
        home=(float(b.gaz.home[0]), float(b.gaz.home[1]), home_alt),
        loop=False,
        fence=[tuple(v) for v in b.gaz.geofence],
        fence_alt=ops.ceiling_agl_m + ops.fence_alt_margin_m,
        rtl_mah=model.capacity_mah - reserve,
    )


def _energy(ac: Aircraft, gaz: Gazetteer) -> EnergyModel:
    t = ac.stack_cfg.tecs
    return energy_model(ac.airframe, gaz.site, t.airspeed_min_mps, t.airspeed_max_mps)


@partial(jax.jit, static_argnames=("n",))
def _fly(sp: StackParams, mission: Mission, pos0, psi0, alt0, wind, used0, n: int):
    """The preview aircraft: a point mass in coordinated turns at the commanded airspeed. Its
    bank follows the guidance's demand (up to the roll limit) with the roll loop's time
    constant and rate limit; its height moves towards the target within TECS's limits."""
    ap0 = ap_init(MODE_AUTO, pos0, alt0)
    climb_max, sink_max = sp.tecs.climb_max, sp.tecs.sink_max
    tau, rate = sp.att.roll_tau, sp.att.roll_rate_max

    def body(c, _):
        pos, psi, alt, ap, t, v, bank = c
        vel = v * jnp.stack([jnp.cos(psi), jnp.sin(psi)]) + wind
        ap, tgt = _mission_step(sp, ap, pos, mission, vel=vel, t=t, height=alt, used_mah=used0)
        wp = l1_waypoint(sp.l1, pos, vel, tgt.a, tgt.b)
        lo = l1_loiter(sp.l1, pos, vel, tgt.centre, tgt.radius, tgt.direction)
        lat = jnp.where(tgt.lateral == LAT_LOITER, lo.lat_acc, wp.lat_acc)
        demand = roll_from_lat_acc(lat, tgt.roll_limit, G0)
        phi = bank + jnp.clip((demand - bank) / tau, -rate, rate) * PREVIEW_DT
        v = tgt.speed
        climb = jnp.clip((tgt.alt - alt) / ALT_TAU_S, -sink_max, climb_max)
        out = (pos, alt, ap.wp_idx, ap.mode, ap.captured, phi, v, climb)
        pos = pos + vel * PREVIEW_DT
        psi = psi + G0 * jnp.tan(phi) / v * PREVIEW_DT
        alt = alt + climb * PREVIEW_DT
        return (pos, psi, alt, ap, t + PREVIEW_DT, v, phi), out

    v0 = mission.speed
    c0 = (pos0, psi0, alt0, ap0, jnp.zeros_like(alt0), v0, jnp.zeros_like(alt0))
    return jax.lax.scan(body, c0, None, length=n)[1]


def _preview(b: _Builder, mission: Mission) -> Preview:
    """The preview, cut where the plan's pattern is complete (one circle over home after
    RTL, two laps of a patrol, one turn of an endless circle). A short run first; the full
    horizon only when the plan needs it (two compiled lengths)."""
    pv, complete = _preview_n(b, mission, int(PREVIEW_STEPS_S[0] / PREVIEW_DT))
    for seconds in PREVIEW_STEPS_S[1:]:
        if complete:
            break
        pv, complete = _preview_n(b, mission, int(seconds / PREVIEW_DT))
    return pv


def _preview_n(b: _Builder, mission: Mission, n: int) -> tuple[Preview, bool]:
    sp = stack_params(b.ac, b.gaz)
    f = jnp.result_type(float)  # the mission's float (float64 when x64 is on)
    with jax.default_device(jax.devices("cpu")[0]):
        out = _fly(
            sp, mission, jnp.asarray(b.ac.pos_ne, f),
            jnp.asarray(math.radians(b.ac.course_deg), f), jnp.asarray(b.ac.alt_agl, f),
            jnp.asarray(b.ac.wind_ne, f), jnp.asarray(0.0, f), n,
        )  # fmt: skip
    pos, alt, item, mode, captured, phi, v, climb = (np.asarray(x) for x in out)
    t = np.arange(n) * PREVIEW_DT
    end = n
    loiter_r = b.cfg.l1.loiter_radius_m
    kinds = np.array([it.kind for it in b.items])
    rtl = np.nonzero(mode == MODE_RTL)[0]
    if len(rtl):  # one circle over home
        end = min(n, rtl[0] + int(2 * math.pi * loiter_r / b.cruise / PREVIEW_DT) + 1)
    elif (kinds == ITEM_JUMP).any():  # two laps
        j = int(np.nonzero(kinds == ITEM_JUMP)[0][0])
        back = np.nonzero((item[1:] == int(b.items[j].p1)) & (item[:-1] > item[1:]))[0]
        if len(back) >= 2:
            end = min(n, back[1] + 2)  # up to the first sample of the third lap
    elif (kinds == ITEM_LOITER_UNLIM).any():  # one turn on the circle
        cap = np.nonzero(captured & (kinds[item] == ITEM_LOITER_UNLIM))[0]
        if len(cap):
            r = abs(b.items[int(item[cap[0]])].radius)
            end = min(n, cap[0] + int(2 * math.pi * r / float(v[cap[0]]) / PREVIEW_DT) + 1)
    model = _energy(b.ac, b.gaz)
    used = model.used_mah(t, v, climb)
    sl = slice(0, end)
    pv = Preview(t[sl], pos[sl], alt[sl], item[sl], mode[sl], captured[sl],
                 np.degrees(phi[sl]), v[sl], climb[sl], used[sl])  # fmt: skip
    return pv, end < n


@dataclass
class _Laps:
    to_lap2_s: float  # from now to the end of the first lap (the way out included)
    to_lap2_mah: float
    lap_s: float  # one lap (the second, measured in the preview)
    lap_mah: float
    home_s: float  # from the farthest point of the pattern back home
    home_mah: float


def _lap_stats(b: _Builder, pv: Preview) -> _Laps | None:
    assert b.jump is not None
    first = int(b.items[b.jump].p1)
    at = pv.item == first
    starts = np.nonzero(at & np.concatenate([[True], ~at[:-1]]))[0]
    if len(starts) < 3:
        return None
    i, j = starts[1], starts[2]
    far = float(np.max(np.linalg.norm(pv.ne - b.gaz.home, axis=1)))
    home_s = far / max(b.cruise - b.ac.wind_speed, 1.0)
    amps = float(_energy(b.ac, b.gaz).current(np.array([b.cruise]), np.zeros(1))[0])
    return _Laps(float(pv.t[i]), float(pv.used_mah[i]), float(pv.t[j] - pv.t[i]),
                 float(pv.used_mah[j] - pv.used_mah[i]), home_s, home_s * amps / 3.6)  # fmt: skip


def _conflict(b: _Builder, pv: Preview) -> tuple[int | None, float]:
    """The first preview sample within the tracking margin of a no-fly area or the geofence's
    edge (after leaving any it started in), or None; and the least clearance after that."""
    d = -geo.signed_distance(pv.ne, b.gaz.geofence)
    for area in b.gaz.no_fly:
        d = np.minimum(d, area.signed_distance(pv.ne))
    hit = d < TRACK_MARGIN_M
    start = 0
    if hit[0]:
        out = np.nonzero(~hit)[0]
        start = int(out[0]) if len(out) else len(hit)
        hit[:start] = False
    least = float(d[start:].min()) if start < len(d) else float("inf")
    return (int(np.argmax(hit)) if hit.any() else None), least


def _assemble(plan: Plan, ac: Aircraft, gaz: Gazetteer, levels) -> _Builder:
    b = _Builder(plan, ac, gaz, levels)
    for k, s in enumerate(plan.steps):
        b.k = k
        try:
            b.step(k, s)
        except UnknownPlace as e:
            b.errors.append(f"step {k + 1}: {e}")
    if not b.items and not b.errors:
        b.errors.append("the plan has nothing to fly")
    if len(b.items) > MAX_ITEMS:
        b.errors.append(
            f"the plan needs {len(b.items)} mission items; the autopilot holds {MAX_ITEMS}: "
            "split it in two"
        )
    return b


def build(plan: Plan, ac: Aircraft, gaz: Gazetteer) -> Built:
    """Build, preview, and where the preview comes too close to a no-fly area after crossing
    a place, move that crossing further from it and build again (a few times at most): the
    checker then judges the final plan."""
    levels: dict[tuple[int, int], int] = {}
    best: tuple[float, _Builder] | None = None
    seen: set[tuple] = set()
    key: tuple[int, int] | None = None
    b = _assemble(plan, ac, gaz, levels)
    for _ in range(LEVEL_CAP * max(len(plan.steps), 1)):
        if b.errors:
            best = None
            break
        sig = tuple(b.items)
        if sig not in seen:  # (a repair that changed nothing takes the next step at once)
            seen.add(sig)
            if len(seen) > MAX_REPAIRS:
                break
            pv = _preview(b, _mission(b, b.items))
            i, least = _conflict(b, pv)
            if best is None or least > best[0]:
                best = (least, b)
            if i is None:
                break
            # The overshoot comes after a crossing, while flying to the next item: blame the
            # item flown before the current one (or, still on the way out, the first).
            before = np.nonzero(pv.item[:i] != pv.item[i])[0]
            culprit = int(pv.item[before[-1]]) if len(before) else int(pv.item[i])
            key = b.visit_of.get(culprit)
        if key is None or levels.get(key, 0) >= LEVEL_CAP:
            break
        levels[key] = levels.get(key, 0) + 1
        b = _assemble(plan, ac, gaz, levels)
    if best is not None:  # the attempt that kept clearest (the checker judges it)
        b = best[1]
    built = Built(plan, ac, b.items, None, b.steps, None, None, b.rmin, b.endless is not None,
                  b.notes, b.errors)  # fmt: skip
    if b.errors:
        return built
    model = _energy(ac, gaz)
    available = model.capacity_mah * (1.0 - RESERVE) - ac.used_mah
    mission = _mission(b, b.items)
    pv = _preview(b, mission)
    laps = lap_mah = None
    st: _Laps | None = None
    if b.jump is not None and (st := _lap_stats(b, pv)) is not None:
        lap_mah = st.lap_mah
        if b.until_reserve:
            spare = available - st.to_lap2_mah - st.home_mah
            laps = 1 + int(spare // st.lap_mah) if spare >= 0 else 0
            if laps >= 1:
                b.items[b.jump] = b.items[b.jump]._replace(p2=float(laps - 1))
                mission = _mission(b, b.items)
                pv = _preview(b, mission)
                b.notes.append(f"patrol: {laps} laps before the battery reserve (estimated)")
        else:
            laps = int(b.items[b.jump].p2) + 1
    returned = bool((pv.mode == MODE_RTL).any())
    if b.endless is not None and b.items[-1].kind == ITEM_LOITER_UNLIM:
        needed = None  # until the battery reserve, then home (the stack's failsafe)
        rate = float(model.current(np.array([b.cruise]), np.zeros(1))[0]) / 3.6
        spare = max(available - float(pv.used_mah[-1]), 0.0)
        minutes = (float(pv.t[-1]) + spare / rate) / 60.0
    elif returned or laps is None:
        needed, minutes = float(pv.used_mah[-1]), float(pv.t[-1]) / 60.0
        if not returned:
            b.notes.append(
                f"the plan is longer than the {PREVIEW_S / 60:.0f} min preview: "
                "the battery estimate covers only that part"
            )
    else:  # the preview stops after two laps: add the rest and the way home
        assert lap_mah is not None and st is not None
        more = max(laps - 2, 0)
        needed = float(pv.used_mah[-1]) + more * lap_mah + st.home_mah
        minutes = (float(pv.t[-1]) + more * st.lap_s + st.home_s) / 60.0
    built.items, built.mission, built.preview = b.items, mission, pv
    built.energy = Energy(available, needed, minutes, laps, lap_mah)
    return built
