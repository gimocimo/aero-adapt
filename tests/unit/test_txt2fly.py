"""Txt2Fly deterministic core (spec 06 §4-§7): geometry, plans, gazetteer, builder, checker."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from aeroadapt.airframes import Airframe, load_airframe
from aeroadapt.scenarios import load_site
from aeroadapt.stack.config import load_stack_cfg
from aeroadapt.stack.types import (
    ITEM_FLYOVER,
    ITEM_JUMP,
    ITEM_LOITER_TIME,
    ITEM_LOITER_TURNS,
    ITEM_RTL,
    ITEM_WAYPOINT,
    mission_from_dict,
    mission_to_dict,
)
from aeroadapt.txt2fly import geometry as geo
from aeroadapt.txt2fly.builder import Aircraft, bank_for, build, r_min
from aeroadapt.txt2fly.checker import check, report_json
from aeroadapt.txt2fly.gazetteer import Gazetteer, UnknownPlace, load_gazetteer, site_layers
from aeroadapt.txt2fly.plan import Home, Offset, PlaceRef, Plan, load_plan

PLANS = Path(__file__).resolve().parents[2] / "configs" / "txt2fly" / "plans"
WEST = (-6.0 * math.cos(math.radians(250.0)), -6.0 * math.sin(math.radians(250.0)))


@pytest.fixture(scope="module")
def gaz() -> Gazetteer:
    return load_gazetteer(load_site("cambridge_west"))


@pytest.fixture(scope="module")
def ac(aerosonde: Airframe) -> Aircraft:
    return Aircraft(load_stack_cfg("aerosonde_bm"), aerosonde, (-150.0, -200.0), 100.0, 0.0, WEST)


def _plan(*steps: dict, **kw) -> Plan:
    return Plan.model_validate({"steps": list(steps), **kw})


def _place(i: str) -> dict:
    return {"kind": "place", "id": i}


# --------------------------------------------------------------------------- geometry
def test_offset_convex_contains_the_grown_polygon() -> None:
    rng = np.random.default_rng(1)
    hull = geo.convex_hull(rng.uniform(-100, 100, (30, 2)))
    grown = geo.offset_convex(hull, 25.0)
    edges = zip(hull, np.roll(hull, -1, 0), strict=True)
    ring = np.vstack([np.linspace(a, b, 20) for a, b in edges])
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    disc = 24.9 * np.stack([np.cos(angles), np.sin(angles)], 1)
    around = (ring[:, None] + disc[None]).reshape(-1, 2)
    assert geo.inside(around, grown).all()


def test_simplify_stays_within_its_tolerance() -> None:
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    ring = np.stack([100 * np.cos(t), 60 * np.sin(t) + 5 * np.sin(7 * t)], 1)
    poly, tol = geo.simplify(ring, 16)
    assert len(poly) <= 16
    assert geo.distance_to_boundary(ring, poly).max() <= tol + 1e-9


def test_polygon_set_is_exact_outside_and_signs_inside() -> None:
    rng = np.random.default_rng(2)
    centres = rng.uniform(-300, 300, (12, 2))
    polys = [geo.convex_hull(rng.uniform(-50, 50, (8, 2)) + c) for c in centres]
    pts = rng.uniform(-500, 500, (2000, 2))
    ref = np.min([geo.signed_distance(pts, p) for p in polys], axis=0)
    got = geo.PolygonSet(polys).signed_distance(pts)
    out = ref >= 0
    np.testing.assert_allclose(got[out], ref[out], atol=1e-9)
    assert (np.sign(got) == np.sign(ref)).all()


def test_long_axis_of_a_rectangle() -> None:
    rect = np.array([[0.0, 0.0], [400.0, 100.0], [375.0, 200.0], [-25.0, 100.0]])  # tilted
    u, half = geo.long_axis(rect)
    assert abs(abs(u @ np.array([0.970, 0.243])) - 1.0) < 1e-2
    assert half == pytest.approx(0.5 * math.hypot(400, 100), rel=0.05)


# --------------------------------------------------------------------------- plans, places
def test_example_plans_load() -> None:
    files = sorted(PLANS.glob("*.yaml"))
    assert len(files) >= 4
    for f in files:
        assert load_plan(f).steps


def test_places_resolve_by_id_name_and_alias(gaz: Gazetteer) -> None:
    a = gaz.place("zone-a")
    assert gaz.place("field A") is a and gaz.place("A") is a and gaz.place("safe zone a") is a
    church = gaz.place("Saint Peter (Coton Parish Church)")
    assert church.kind == "church" and church.source == "OpenStreetMap"
    with pytest.raises(UnknownPlace, match="the old mill"):
        gaz.place("the old mill")
    here = np.array([0.0, 0.0])
    off = gaz.resolve(Offset(origin="home", bearing_deg=90.0, distance_m=100.0), here)
    np.testing.assert_allclose(off.point, gaz.home + np.array([0.0, 100.0]), atol=1e-9)
    np.testing.assert_allclose(gaz.resolve(Home(), here).point, gaz.home)


def test_no_fly_areas_cover_the_village_and_motorway(gaz: Gazetteer) -> None:
    church = gaz.place("Saint Peter (Coton Parish Church)").point
    by_id = {a.id: a for a in gaz.no_fly}
    assert by_id["coton"].signed_distance(church[None])[0] < 0
    assert by_id["m11"].signed_distance(np.array([[-499.0, 82.0]]))[0] < 0  # on the motorway
    assert all(a.signed_distance(gaz.home[None])[0] > 100 for a in gaz.no_fly)
    layers = site_layers(gaz.site)
    assert layers is not None and {a["id"] for a in layers["no_fly"]} == set(by_id)
    assert "OpenStreetMap" in (layers["attribution"] or "")


def test_turn_radius_worked_examples() -> None:
    """Spec 06 §6's table: the tightest circle (5° bank margin) and the bank a 30 m circle
    needs at its downwind point."""
    assert r_min(25.0, 0.0, 45.0) == pytest.approx(76, abs=0.5)
    assert r_min(25.0, 6.0, 45.0) == pytest.approx(117, abs=0.5)
    assert r_min(13.0, 6.0, 45.0) == pytest.approx(44, abs=0.5)
    assert bank_for(30.0, 25.0, 6.0) == pytest.approx(73, abs=0.5)
    assert bank_for(30.0, 13.0, 0.0) == pytest.approx(30, abs=0.5)


# --------------------------------------------------------------------------- builder
def test_steps_become_items(gaz: Gazetteer, ac: Aircraft) -> None:
    plan = _plan(
        {"do": "goto", "target": _place("field B"), "alt_m": 90},
        {"do": "pass_over", "target": _place("field A")},
        {"do": "orbit", "target": _place("field C"), "turns": 2, "direction": "ccw"},
        {"do": "orbit", "target": _place("field F"), "radius_m": 150, "duration_s": 60},
        {"do": "return_home"},
    )
    b = build(plan, ac, gaz, adapt=False)  # literally: one item layout to check
    assert not b.errors
    kinds = [it.kind for it in b.items]
    assert kinds == [ITEM_WAYPOINT, ITEM_WAYPOINT, ITEM_FLYOVER, ITEM_WAYPOINT,
                     ITEM_LOITER_TURNS, ITEM_LOITER_TIME, ITEM_RTL]  # fmt: skip
    assert all(it.alt == 90.0 for it in b.items[:6])  # the height carries over
    c = b.items[4]
    assert c.radius < 0 and c.p1 == 2.0  # anticlockwise; usual radius, rounded up to 10 m
    assert abs(c.radius) == math.ceil(max(150.0, r_min(25.0, 6.0, 45.0)) / 10) * 10
    assert b.items[5].p1 == 60.0 and b.items[5].radius == 150.0
    run_in, over, run_out = (np.array([it.north, it.east]) for it in b.items[1:4])
    assert np.linalg.norm(run_in - over) > np.linalg.norm(run_out - over)  # longer run-in
    m = b.mission
    assert m is not None and int(m.n_fence) == 4 and float(m.fence_alt) == 140.0
    assert float(m.rtl_mah) == pytest.approx(0.7 * 16000.0)
    again = mission_from_dict(mission_to_dict(m))
    for x, y in zip(again, m, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_patrol_laps_and_battery_sizing(gaz: Gazetteer, ac: Aircraft) -> None:
    targets = [_place("field A"), _place("field B"), _place("field C")]
    three = build(_plan({"do": "patrol", "targets": targets, "laps": 3}), ac, gaz, adapt=False)
    jump = next(it for it in three.items if it.kind == ITEM_JUMP)
    assert jump.p1 == 0 and jump.p2 == 2
    assert three.energy is not None and three.energy.laps == 3
    until = build(_plan({"do": "patrol", "targets": targets}), ac, gaz)
    e = until.energy
    assert e is not None and e.laps is not None and e.laps > 3
    jump = next(it for it in until.items if it.kind == ITEM_JUMP)
    assert until.items[int(jump.p1)].kind in (ITEM_FLYOVER, ITEM_WAYPOINT)
    assert jump.p2 == e.laps - 1 and until.items[-1].kind == ITEM_RTL
    assert e.needed_mah is not None and e.needed_mah <= e.available_mah
    assert any("laps before the battery reserve" in n for n in until.notes)


def test_plans_that_cannot_be_built(gaz: Gazetteer, ac: Aircraft) -> None:
    endless = _plan({"do": "hold"}, {"do": "goto", "target": _place("field A")})
    assert "never be flown" in build(endless, ac, gaz).errors[0]
    unknown = _plan({"do": "goto", "target": _place("the old mill")})
    assert "the old mill" in build(unknown, ac, gaz).errors[0]
    many = _plan(*[{"do": "pass_over", "target": _place(z)} for z in "ABCFABCFABCF"])
    assert "split it in two" in build(many, ac, gaz).errors[0]


# --------------------------------------------------------------------------- checker
def _failed(r) -> set[str]:
    return {c.rule for c in r.checks if not c.ok}


def test_the_church_at_30_m_is_explained(gaz: Gazetteer, ac: Aircraft) -> None:
    """Taken literally, the checker refuses it and says why; the planner flies the closest
    allowed circle and says what it changed."""
    r = check(build(load_plan(PLANS / "church_30m.yaml"), ac, gaz, adapt=False), gaz)
    assert not r.ok
    k2 = next(c for c in r.checks if c.rule == "K2")
    assert "needs 73° of bank at 25 m/s in this wind" in k2.reason
    assert "limited to 45°" in k2.reason and k2.suggestion == "circle at 120 m"
    assert {"K2", "K8"} <= _failed(r)  # the village is also no-fly
    assert "Coton village" in next(c for c in r.checks if c.rule == "K8").reason
    js = report_json(r)
    assert js["ok"] is False and js["preview"] and js["plan"]["steps"][0]["do"] == "orbit"
    a = check(build(load_plan(PLANS / "church_30m.yaml"), ac, gaz), gaz)
    assert a.ok
    notes = " ".join(a.built.notes)
    assert "would need 73° of bank" in notes and "circles at 120 m, the tightest it can" in notes
    assert "is within Coton village (no-fly): circles the nearest place clear of it" in notes


def test_good_plans_pass(gaz: Gazetteer, ac: Aircraft) -> None:
    for name in ("patrol_abc", "fly_by_field_a"):
        r = check(build(load_plan(PLANS / f"{name}.yaml"), ac, gaz), gaz)
        assert r.ok and not r.warnings, (name, [c.reason for c in r.checks if not c.ok])


def test_each_rule_catches_its_violation(gaz: Gazetteer, ac: Aircraft) -> None:
    """Plans taken literally: every rule catches its violation."""

    def failed(*steps: dict) -> set[str]:
        return _failed(check(build(_plan(*steps), ac, gaz, adapt=False), gaz))

    assert "K4" in failed({"do": "pass_over", "target": _place("field A"), "alt_m": 150})
    assert "K4" in failed({"do": "pass_over", "target": _place("field A"), "alt_m": 25})
    assert "K5" in failed({"do": "set_speed", "speed": 45.0},
                          {"do": "pass_over", "target": _place("field A")})  # fmt: skip
    assert "K6" in failed({"do": "patrol", "targets": [_place("field A"), _place("field C")],
                           "laps": 400})  # fmt: skip
    # A wide circle round Rowan's Wood grazes the village: the circle itself fails (K3).
    wood = {"do": "orbit", "target": _place("Rowan's Wood"), "radius_m": 150, "turns": 1}
    assert "K3" in failed(wood)


def test_glide_reach_is_a_warning(gaz: Gazetteer, ac: Aircraft) -> None:
    """Far out at 40 m, no field is within glide: K7 warns, and the plan still passes."""
    far = {"do": "goto", "target": {"kind": "point", "ne_m": [-1150.0, -1150.0]}, "alt_m": 40}
    r = check(build(_plan(far), ac, gaz), gaz)
    k7 = next(c for c in r.checks if c.rule == "K7")
    assert not k7.ok and not k7.hard and k7.where
    assert [c.rule for c in r.warnings] == ["K7"]


def test_unknown_place_is_k1(gaz: Gazetteer, ac: Aircraft) -> None:
    plan = _plan({"do": "goto", "target": PlaceRef(id="nowhere").model_dump()})
    r = check(build(plan, ac, gaz), gaz)
    assert [c.rule for c in r.checks] == ["K1"] and not r.ok


def test_airframe_matters(gaz: Gazetteer) -> None:
    """The same 30 m circle: refused for the Aerosonde on turns, fine for a slow aircraft."""
    glider = load_airframe("aerosonde_bm").model_copy()  # stands in: only the stack differs
    slow = load_stack_cfg("aerosonde_bm")
    t = slow.tecs.model_copy(update={"airspeed_min_mps": 9.0, "airspeed_cruise_mps": 12.0,
                                     "airspeed_max_mps": 18.0})  # fmt: skip
    slow = slow.model_copy(update={"tecs": t})
    ac = Aircraft(slow, glider, (-150.0, -200.0), 100.0, 0.0, (0.0, 0.0))
    circle = {"do": "orbit", "target": _place("field B"), "radius_m": 30, "turns": 1}
    r = check(build(_plan(circle), ac, gaz), gaz)
    assert next(c for c in r.checks if c.rule == "K2").ok


def test_requests_are_adapted_and_explained(gaz: Gazetteer, ac: Aircraft) -> None:
    """The planner flies the closest allowed version of a request and says what changed;
    only what nothing can fix (here the battery) is refused."""

    def adapted(*steps: dict) -> tuple[bool, str]:
        r = check(build(_plan(*steps), ac, gaz), gaz)
        return r.ok, " ".join(r.built.notes)

    ok, notes = adapted({"do": "pass_over", "target": _place("field A"), "alt_m": 150})
    assert ok and "at 120 m, the ceiling here (asked 150 m)" in notes
    ok, notes = adapted({"do": "set_speed", "speed": 45.0},
                        {"do": "pass_over", "target": _place("field A")})  # fmt: skip
    assert ok and "the nearest this aircraft can fly (asked 45)" in notes
    wood = {"do": "orbit", "target": _place("Rowan's Wood"), "radius_m": 150, "turns": 1}
    ok, notes = adapted(wood)
    assert ok and "Rowan's Wood" in notes
    many = {"do": "patrol", "targets": [_place("field A"), _place("field C")], "laps": 400}
    assert not adapted(many)[0]
    beyond = {"do": "goto", "target": _place("Saint Peter (Coton Parish Church)")}
    ok, notes = adapted(beyond, {"do": "return_home"})
    assert ok and "goes to the nearest place clear of it" in notes
