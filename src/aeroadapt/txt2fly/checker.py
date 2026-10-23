"""The checker (spec 06 §7): may this plan fly, and if not, why, in plain words.

Every rule returns a pass or a fail with a reason and, where one exists, the nearest plan that
would pass. Hard rules cannot be overridden by the plan, by text or by a model; they change
only in the site and stack configuration. The one soft rule (K7, glide reach) needs the
operator's acknowledgement in the confirm step.

The rules read the builder's preview (the mission flown by a point-mass aircraft with the
stack's guidance in the current wind), widened by a tracking margin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from aeroadapt.sim.aerojax.atmosphere import IsaParams, isa
from aeroadapt.stack.contingency import assess_zones
from aeroadapt.txt2fly import geometry as geo
from aeroadapt.txt2fly.builder import Built, StepGeom, bank_for, r_min, stack_params
from aeroadapt.txt2fly.energy import RESERVE
from aeroadapt.txt2fly.gazetteer import Gazetteer
from aeroadapt.txt2fly.terrain import CLEARANCE_M, ceil5, climb_out, landmark, relief

TRACK_MARGIN_M = 30.0  # the flown track may stray this far from the preview
GLIDE_STEP_M = 50.0

RULES = {
    "K1": "Plan",
    "K2": "Turns",
    "K3": "No-fly areas and geofence",
    "K4": "Height",
    "K5": "Airspeed",
    "K6": "Battery",
    "K7": "Glide reach",
    "K8": "The way there",
}


@dataclass
class Check:
    rule: str
    ok: bool
    reason: str
    hard: bool = True
    suggestion: str | None = None
    where: list[list[float]] = field(default_factory=list)  # NE points to show on the map

    @property
    def title(self) -> str:
        return RULES[self.rule]


@dataclass
class Report:
    built: Built
    checks: list[Check]

    @property
    def ok(self) -> bool:
        """Every hard rule passes (soft warnings still need an acknowledgement)."""
        return all(c.ok for c in self.checks if c.hard)

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and not c.hard]


def _ceil10(x: float) -> float:
    return math.ceil(x / 10.0) * 10.0


def _step_of(built: Built, item: int) -> StepGeom | None:
    for g in built.steps:
        if g.items[0] <= item < g.items[1]:
            return g
    return None


def _where(ne: np.ndarray, mask: np.ndarray, every: int = 8) -> list[list[float]]:
    pts = ne[mask][::every]
    return [[round(float(a), 1), round(float(b), 1)] for a, b in pts]


def _join_end(built: Built) -> int:
    """Index of the first preview sample past the way to the first item."""
    pv = built.preview
    assert pv is not None
    moved = np.nonzero((pv.item != pv.item[0]) | pv.captured)[0]
    return int(moved[0]) if len(moved) else len(pv.t)


def _k2_turns(built: Built) -> Check:
    ac = built.aircraft
    lim = ac.stack_cfg.l1.roll_limit_deg
    w = ac.wind_speed
    air = "in this wind" if w >= 1.0 else "in calm air"
    for g in built.steps:
        if g.radius is None or g.speed is None:
            continue
        rm = r_min(g.speed, w, lim)
        if g.radius < rm:
            best = _ceil10(rm)
            name = g.target.name if g.target is not None else "here"
            return Check(
                "K2",
                False,
                f"Circling {name}: a {g.radius:g} m circle needs "
                f"{bank_for(g.radius, g.speed, w):.0f}° of bank at {g.speed:g} m/s {air}; this "
                f"aircraft is limited to {lim:g}°. The tightest circle now is about {best:g} m.",
                suggestion=f"circle at {best:g} m",
            )
    return Check("K2", True, "Every circle is within the aircraft's bank limit in this wind.")


def _k3_k8_areas(built: Built, gaz: Gazetteer) -> tuple[Check, Check]:
    pv = built.preview
    assert pv is not None
    ne = pv.ne
    join = _join_end(built)
    hits: list[tuple[str, np.ndarray]] = []  # (what, mask over the preview)
    for area in gaz.no_fly:
        hit = area.signed_distance(ne) < TRACK_MARGIN_M
        if hit[0]:  # starting inside or at the edge: leaving it is allowed
            out = np.nonzero(~hit)[0]
            hit[: out[0] if len(out) else len(hit)] = False
        hits.append((f"{area.name} (no-fly: {area.reason})", hit))
    outside = geo.signed_distance(ne, gaz.geofence) > -TRACK_MARGIN_M
    if outside[0]:
        inside = np.nonzero(~outside)[0]
        outside[: inside[0] if len(inside) else len(outside)] = False
    hits.append(("the edge of the operating area (geofence)", outside))
    early = np.arange(len(ne)) < join

    def first(part: np.ndarray) -> tuple[int, str, np.ndarray] | None:
        found = [(int(np.argmax(m & part)), what, m & part) for what, m in hits if (m & part).any()]
        return min(found, key=lambda f: f[0]) if found else None

    k8, k3 = first(early), first(~early)
    if k8 is not None:
        _, what, mask = k8
        c8 = Check(
            "K8",
            False,
            f"The way there comes within {TRACK_MARGIN_M:g} m of {what}.",
            where=_where(ne, mask),
        )
    else:
        c8 = Check("K8", True, "The way from the aircraft to the plan is clear.")
    if k3 is not None:
        i, what, mask = k3
        g = _step_of(built, int(pv.item[i]))
        part = g.label if g is not None else "the return home"
        c3 = Check(
            "K3",
            False,
            f"'{part[:1].upper() + part[1:]}' would come within {TRACK_MARGIN_M:g} m of {what}.",
            where=_where(ne, mask),
        )
    else:
        c3 = Check("K3", True, "The route stays clear of the no-fly areas and inside the geofence.")
    return c3, c8


def _k4_height(built: Built, gaz: Gazetteer) -> Check:
    ops = gaz.site.operations
    assert ops is not None
    rel = relief(gaz.site)
    for g in built.steps:
        if g.alt is None:
            continue
        if g.alt > ops.ceiling_agl_m:
            return Check(
                "K4",
                False,
                f"{ops.ceiling_agl_m:g} m is the ceiling here ({g.label} asks for {g.alt:g} m).",
                suggestion=f"fly at {ops.ceiling_agl_m:g} m",
            )
        if g.alt < ops.min_agl_m:
            return Check(
                "K4",
                False,
                f"{ops.min_agl_m:g} m is the lowest height allowed here "
                f"({g.label} asks for {g.alt:g} m).",
                suggestion=f"fly at {ops.min_agl_m:g} m",
            )
        # (On a world site the whole track is checked against every obstacle, below.)
        h = g.target.height_m if g.target is not None and rel is None else None
        if h is not None and g.target is not None and g.alt < h + CLEARANCE_M:
            return Check(
                "K4",
                False,
                f"{g.alt:g} m is only {g.alt - h:.0f} m above the top of "
                f"{g.target.name} ({h:g} m tall); keep {CLEARANCE_M:g} m clear.",
                suggestion=f"fly at {math.ceil(h + CLEARANCE_M):g} m",
            )
    pv = built.preview
    if rel is None or pv is None:
        return Check(
            "K4",
            True,
            f"Every height is between {ops.min_agl_m:g} m and the {ops.ceiling_agl_m:g} m ceiling.",
        )
    # A world site (docs/07 §7): the whole track, over the ground and what stands on it.
    assert pv is not None
    floor = rel.floor(pv.ne, ops.min_agl_m)
    roof = rel.ceiling(pv.ne, ops.ceiling_agl_m)
    if ops.ceiling_alt_m is not None:
        roof = np.minimum(roof, ops.ceiling_alt_m)
    low, high = pv.alt < floor - 1.0, pv.alt > roof + 1.0
    low[: climb_out(pv.alt, floor)] = False  # the climb from a start already too low
    if low.any():
        k = int(np.argmax(floor - pv.alt))
        g = _step_of(built, int(pv.item[k]))
        part = g.label if g is not None else "the return home"
        what = landmark(gaz, pv.ne[k])
        leg = pv.item == pv.item[k]
        fix = f"fly at {ceil5(float(floor[leg].max())):g} m"
        if rel.by_obstacle(pv.ne[k][None], ops.min_agl_m)[0]:
            top = float(rel.hi_surface[rel._idx(pv.ne[k])][0])
            return Check(
                "K4",
                False,
                f"'{part}' would pass {pv.alt[k] - top:.0f} m above "
                f"{what or 'the trees and buildings'}; keep {CLEARANCE_M:g} m clear.",
                suggestion=fix,
                where=_where(pv.ne, low),
            )
        ground = float(rel.ground(pv.ne[k])[0])
        return Check(
            "K4",
            False,
            f"'{part}' would pass {pv.alt[k] - ground:.0f} m above the rising ground"
            f"{f' at {what}' if what else ''}; {ops.min_agl_m:g} m is the lowest allowed.",
            suggestion=fix,
            where=_where(pv.ne, low),
        )
    if high.any():
        k = int(np.argmax(pv.alt - roof))
        g = _step_of(built, int(pv.item[k]))
        part = g.label if g is not None else "the return home"
        return Check(
            "K4",
            False,
            f"'{part}' would be {pv.alt[k] - roof[k] + ops.ceiling_agl_m:.0f} m above the low "
            f"ground there; {ops.ceiling_agl_m:g} m is the ceiling.",
            suggestion=f"fly at {math.floor(float(roof[pv.item == pv.item[k]].min())):g} m",
            where=_where(pv.ne, high),
        )
    return Check(
        "K4",
        True,
        f"The whole route keeps {ops.min_agl_m:g} m to {ops.ceiling_agl_m:g} m above the ground, "
        f"and {CLEARANCE_M:g} m above trees and buildings.",
    )


def _k5_speed(built: Built) -> Check:
    t = built.aircraft.stack_cfg.tecs
    for g in built.steps:
        if g.speed is None:
            continue
        if g.speed < t.airspeed_min_mps:
            return Check(
                "K5",
                False,
                f"This aircraft cannot fly slower than {t.airspeed_min_mps:.1f} m/s.",
                suggestion=f"fly at {t.airspeed_min_mps:.1f} m/s",
            )
        if g.speed > t.airspeed_max_mps:
            return Check(
                "K5",
                False,
                f"This aircraft cannot fly faster than {t.airspeed_max_mps:.1f} m/s.",
                suggestion=f"fly at {t.airspeed_max_mps:.1f} m/s",
            )
    return Check("K5", True, "Every airspeed is within the aircraft's range.")


def _k6_battery(built: Built) -> Check:
    e = built.energy
    assert e is not None
    capacity = e.available_mah / (1.0 - RESERVE)  # (with nothing used yet)
    keep = f"the {RESERVE * 100:.0f} % reserve is kept"
    if e.needed_mah is None:
        return Check(
            "K6",
            True,
            f"It circles until the battery reaches the {RESERVE * 100:.0f} % reserve "
            f"(about {e.minutes:.0f} min, estimated), then returns home.",
        )
    if e.laps is not None and e.laps < 1:
        return Check("K6", False, f"There is not enough battery for one lap with {keep}.")
    pct = 100.0 * e.needed_mah / capacity
    if e.needed_mah > e.available_mah:
        return Check(
            "K6",
            False,
            f"This plan needs about {pct:.0f} % of the battery; only "
            f"{100 * e.available_mah / capacity:.0f} % can be used ({keep}).",
        )
    laps = f"{e.laps} laps, " if e.laps else ""
    return Check(
        "K6",
        True,
        f"It uses about {pct:.0f} % of the battery ({laps}about "
        f"{e.minutes:.0f} min, estimated); {keep}.",
    )


def _k7_glide(built: Built, gaz: Gazetteer) -> Check:
    pv = built.preview
    assert pv is not None
    sp = stack_params(built.aircraft, gaz)
    seg = np.linalg.norm(np.diff(pv.ne, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    idx = np.searchsorted(dist, np.arange(0.0, dist[-1] + 1e-6, GLIDE_STEP_M))
    idx = np.minimum(idx, len(pv.t) - 1)
    site = gaz.site
    p = IsaParams(site.isa.temp_offset_k, site.isa.qnh_hpa * 100.0, site.ground_msl_m)
    rho = isa(p, site.ground_msl_m + pv.alt[idx])[2]
    f = jnp.result_type(float)
    wind = jnp.asarray(built.aircraft.wind_ne, f)
    margin = jax.vmap(lambda pos, h, r: assess_zones(sp.cont, sp.zones, pos, h, wind, r)[0])(
        jnp.asarray(pv.ne[idx], f),
        jnp.asarray(pv.alt[idx], f),
        jnp.asarray(rho, f),
    )
    covered = np.asarray(jnp.max(margin, axis=1)) >= 0.0
    if covered.all():
        return Check(
            "K7", True, "From every point of the route a safe field is within glide.", hard=False
        )
    first = idx[int(np.argmin(covered))]
    g = _step_of(built, int(pv.item[first]))
    part = g.label if g is not None else "the return home"
    frac = 100.0 * (1.0 - covered.mean())
    mask = np.zeros(len(pv.t), bool)
    mask[idx[~covered]] = True
    return Check(
        "K7",
        False,
        f"If the motor failed during '{part}' (and on {frac:.0f} % of the route), no safe field "
        "would be within glide.",
        hard=False,
        suggestion="fly higher, or nearer the safe fields",
        where=_where(pv.ne, mask, every=1),
    )


def check(built: Built, gaz: Gazetteer) -> Report:
    if built.errors:
        return Report(built, [Check("K1", False, "; ".join(built.errors))])
    n = len(built.items)
    k1 = Check("K1", True, f"Every place was found; the mission has {n} item{'s' * (n != 1)}.")
    k3, k8 = _k3_k8_areas(built, gaz)
    checks = [
        k1,
        _k2_turns(built),
        k3,
        _k4_height(built, gaz),
        _k5_speed(built),
        _k6_battery(built),
        _k7_glide(built, gaz),
        k8,
    ]
    return Report(built, checks)


def describe(report: Report) -> str:
    """The report as text (the CLI)."""
    b = report.built
    ac = b.aircraft
    wind = ac.wind_speed
    frm = (math.degrees(math.atan2(-ac.wind_ne[1], -ac.wind_ne[0])) + 360.0) % 360.0
    lines = [f"Plan: {b.plan.summary or '(no summary)'}"]
    lines += [f"  assumes: {a}" for a in b.plan.assumptions]
    lines.append(
        f"Aircraft: {ac.airframe.id} at {ac.stack_cfg.tecs.airspeed_cruise_mps:g} m/s, roll "
        f"limit {ac.stack_cfg.l1.roll_limit_deg:g}°; wind "
        + (f"{wind:.1f} m/s from {frm:03.0f}°" if wind >= 0.5 else "calm")
    )
    lines.append("Steps:")
    for g in b.steps:
        extra = []
        if g.radius is not None:
            extra.append(f"{g.radius:g} m circle")
        if g.alt is not None:
            extra.append(f"{g.alt:g} m")
        lines.append(f"  {g.index + 1}. {g.label}" + (f" ({', '.join(extra)})" if extra else ""))
    lines += [f"  note: {n}" for n in b.notes]
    lines.append("Checks:")
    for c in report.checks:
        mark = "✓" if c.ok else ("⚠" if not c.hard else "✗")
        lines.append(f"  {mark} {c.rule} {c.title}: {c.reason}")
        if c.suggestion and not c.ok:
            lines.append(f"      suggestion: {c.suggestion}")
    if report.ok and not report.warnings:
        verdict = "can fly"
    elif report.ok:
        verdict = "can fly once the warnings are acknowledged"
    else:
        verdict = "cannot fly: " + ", ".join(c.rule for c in report.checks if c.hard and not c.ok)
    lines.append(f"Verdict: {verdict}")
    return "\n".join(lines)


def report_json(report: Report, every_s: float = 1.0) -> dict:
    """A report for the browser: verdict, steps, checks, the preview path and the items."""
    from aeroadapt.stack.types import ITEM_NAMES

    b = report.built
    preview: list[list[float]] = []
    if b.preview is not None:
        k = max(round(every_s / float(b.preview.t[1] - b.preview.t[0])), 1)
        pv = b.preview
        preview = [[round(float(n), 1), round(float(e), 1), round(float(h), 1)]
                   for (n, e), h in zip(pv.ne[::k], pv.alt[::k], strict=True)]  # fmt: skip
    e = b.energy
    return {
        "ok": report.ok,
        "warnings": [c.rule for c in report.warnings],
        "summary": b.plan.summary,
        "assumptions": list(b.plan.assumptions),
        "notes": list(b.notes),
        "steps": [
            {"label": g.label, "alt": g.alt, "radius": g.radius, "speed": g.speed} for g in b.steps
        ],
        "checks": [
            {
                "rule": c.rule,
                "title": c.title,
                "ok": c.ok,
                "hard": c.hard,
                "reason": c.reason,
                "suggestion": c.suggestion,
                "where": c.where,
            }
            for c in report.checks
        ],
        "preview": preview,
        "items": [
            {
                "kind": ITEM_NAMES[it.kind],
                "north": it.north,
                "east": it.east,
                "alt": it.alt,
                "radius": it.radius,
                "p1": it.p1,
                "p2": it.p2,
                "speed": it.speed,
            }
            for it in b.items
        ],
        "energy": None
        if e is None
        else {
            "needed_mah": e.needed_mah,
            "available_mah": e.available_mah,
            "minutes": e.minutes,
            "laps": e.laps,
        },
        "plan": b.plan.model_dump(mode="json"),
    }
