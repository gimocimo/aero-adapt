"""Txt2Fly and the map (docs/06 §17): zone heights, the area and line steps, the spatial
grader, the review, and the map picture."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point as SPoint
from shapely.geometry import Polygon

from aeroadapt.airframes import Airframe
from aeroadapt.harness.txt2fly_spatial import encloses, outcome
from aeroadapt.scenarios import load_site
from aeroadapt.scenarios.site import SITE_DIR
from aeroadapt.stack.config import load_stack_cfg
from aeroadapt.stack.types import ITEM_LOITER_UNLIM
from aeroadapt.txt2fly import areas
from aeroadapt.txt2fly import translate as tr
from aeroadapt.txt2fly.builder import Aircraft, _eased, _longest_run, build
from aeroadapt.txt2fly.checker import check
from aeroadapt.txt2fly.gazetteer import Gazetteer, load_gazetteer
from aeroadapt.txt2fly.plan import Plan

REPO = Path(__file__).resolve().parents[2]
WSW = (-6.0 * math.cos(math.radians(250.0)), -6.0 * math.sin(math.radians(250.0)))


@pytest.fixture(scope="module")
def testmap() -> Gazetteer:
    return load_gazetteer(load_site("testmap"))


def _ac(af: Airframe, pos: tuple[float, float], alt: float = 150.0, course: float = 0.0):
    return Aircraft(load_stack_cfg("aerosonde_bm"), af, pos, alt, course, WSW)


def _plan(*steps: dict) -> Plan:
    return Plan.model_validate({"steps": list(steps)})


def _place(i: str) -> dict:
    return {"kind": "place", "id": i}


# --------------------------------------------------------------------------- zone heights
def test_zones_know_how_high_they_reach(testmap: Gazetteer) -> None:
    by = {}
    for a in testmap.no_fly:
        by.setdefault(a.id, a)
    city, line, turbine = by["city"], by["power-line"], by["turbine-12"]
    assert by["city-centre"].top_m is None and by["city-centre"].closed_at(499.0)
    # The city is closed up to 100 m above its ground, which rises to about 26 m.
    assert city.base_m is None and city.top_m == pytest.approx(126.0, abs=2.0)
    assert city.closed_at(150.0) and not city.closed_at(160.0)
    assert line.top_m is not None and line.top_m < 105.0 and not line.closed_at(150.0)
    assert line.lowest_over(np.array([0.0, 10.0])) == pytest.approx([130.0, 140.0])
    # A turbine's cylinder has a flat top over the ground at its centre.
    assert turbine.base_m == pytest.approx(45.22) and turbine.top_m == pytest.approx(245.22)
    assert turbine.lowest_over(np.array([0.0, 99.0])) == pytest.approx([275.22, 275.22])


def test_k3_counts_an_area_closed_up_to_a_height_only_when_too_low(
    testmap: Gazetteer, aerosonde: Airframe
) -> None:
    """Straight across the power lines (closed up to 100 m above the ground): at 110 m that
    is too low, at 160 m it is clear above them."""
    west = {"do": "goto", "target": {"kind": "point", "ne_m": [1100.0, -1500.0]}}
    for alt, ok in ((110.0, False), (160.0, True)):
        ac = _ac(aerosonde, (1100.0, -600.0), alt, 270.0)
        r = check(build(_plan({**west, "alt_m": alt}), ac, testmap, adapt=False), testmap)
        k8 = next(c for c in r.checks if c.rule == "K8")  # all of it is the way there
        assert k8.ok is ok, k8.reason
        if not ok:
            assert "power lines" in k8.reason and "below the" in k8.reason


# --------------------------------------------------------------------------- the steps
def test_around_the_city_encloses_it_over_the_power_lines(
    testmap: Gazetteer, aerosonde: Airframe
) -> None:
    """The owner's case: "circle the whole city" goes round all of it, crossing the power
    lines (open above 130 m) rather than stopping at them."""
    ac = _ac(aerosonde, (-1095.0, 488.0))
    r = check(build(_plan({"do": "around", "target": _place("city")}), ac, testmap), testmap)
    assert r.ok, [c.reason for c in r.checks if not c.ok and c.hard]
    assert outcome({"encloses": "city"}, r, testmap) is None
    assert any(n.startswith("passes over power lines") for n in r.built.notes)


def test_an_orbit_round_a_no_fly_area_widens_rather_than_moves(aerosonde: Airframe) -> None:
    london = load_gazetteer(load_site("london"))
    tower = london.place("tower-of-london")
    assert tower.polygon is not None
    ac = _ac(aerosonde, (-300.0, 0.0), 120.0)
    r = check(build(_plan({"do": "orbit", "target": _place("tower-of-london")}), ac, london),
              london)  # fmt: skip
    assert r.ok, [c.reason for c in r.checks if not c.ok and c.hard]
    assert any("rather than moved off it" in n for n in r.built.notes)
    circle = next(it for it in r.built.items if it.kind == ITEM_LOITER_UNLIM)
    centre = Polygon(tower.polygon).centroid
    assert math.hypot(circle.north - centre.x, circle.east - centre.y) < 60.0
    assert outcome({"encloses": "tower-of-london"}, r, london) is None


def test_a_loop_that_cannot_go_all_the_way_round_says_why(
    testmap: Gazetteer, aerosonde: Airframe
) -> None:
    """The wind farm's eastern turbines stand on the geofence: no loop goes round it, and the
    builder says so and which edge can be flown; following its edge flies that part."""
    ac = _ac(aerosonde, (-1095.0, 488.0))
    b = build(_plan({"do": "around", "target": _place("wind-farm")}), ac, testmap)
    assert b.errors and "no loop goes all the way round the wind farm" in b.errors[0]
    assert "geofence" in b.errors[0] and "can fly along its" in b.errors[0]
    edge = check(build(_plan({"do": "follow", "target": _place("wind-farm"), "alt_m": 350}), ac,
                       testmap), testmap)  # fmt: skip
    assert edge.ok, [c.reason for c in edge.checks if not c.ok and c.hard]
    assert any("only part of the way round" in n for n in edge.built.notes)
    assert edge.built.cost > 0.0 and tr.review_text(edge) is None  # noted, not reviewed
    expect = {"follows": "wind-farm", "share": 0.6, "within_m": 250}
    assert outcome(expect, edge, testmap) is None


def test_follow_a_river_and_survey_a_field(testmap: Gazetteer, aerosonde: Airframe) -> None:
    ac = _ac(aerosonde, (-1095.0, 488.0))
    river = check(build(_plan({"do": "follow", "target": _place("river")}), ac, testmap), testmap)
    assert outcome({"follows": "river"}, river, testmap) is None
    wren = load_gazetteer(load_site("wrenfold_vale"))
    ac = _ac(aerosonde, (0.0, 0.0), 100.0)
    survey = check(build(_plan({"do": "survey", "target": _place("zone-b")}), ac, wren), wren)
    assert survey.ok, [c.reason for c in survey.checks if not c.ok and c.hard]
    assert outcome({"covers": "zone-b"}, survey, wren) is None


# --------------------------------------------------------------------------- shapes
def test_loops_runs_and_easing() -> None:
    square = Polygon([(0, 0), (0, 400), (400, 400), (400, 0)])
    loop = areas.loop_round(square, 100.0, 150.0, 16)
    assert len(loop) <= 16
    assert min(square.distance(SPoint(*p)) for p in loop) >= 100.0 - 1e-6
    assert areas.clockwise(np.array([[0.0, 0.0], [0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0]]))
    np.testing.assert_array_equal(_longest_run(np.array([1, 1, 0, 1, 1, 1, 0, 1], bool)),
                                  [3, 4, 5])  # fmt: skip
    np.testing.assert_array_equal(_longest_run(np.array([1, 0, 0, 1, 1], bool)), [3, 4, 0])
    assert _longest_run(np.zeros(4, bool)) is None
    eased = _eased(np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 20.0)
    np.testing.assert_allclose(eased, [80, 90, 100, 90, 80, 70, 60, 70])  # 1 in 2, round


def test_the_grader_measures_what_is_flown() -> None:
    area = Polygon([(-100, -100), (-100, 100), (100, 100), (100, -100)])
    t = np.linspace(0, 2 * np.pi, 200)
    ring = np.stack([300 * np.cos(t), 300 * np.sin(t)], 1)
    assert encloses(area, ring, 1200.0)[0] is None
    assert "of the way round" in (encloses(area, ring[:120], 1200.0)[0] or "")
    assert "flies over it" in (encloses(area, ring * 0.2, 1200.0)[0] or "")


# --------------------------------------------------------------------------- the review
class _Scripted:
    """A translator that answers from a script, and can revise (as the API's can)."""

    name = "scripted"

    def __init__(self, *replies: dict) -> None:
        self.replies = list(replies)
        self.reviews: list[str] = []

    async def translate(self, ctx: tr.Context, text: str) -> tr.Translation:
        raw = self.replies.pop(0)
        return tr.Translation(tr.to_reply(raw), raw, self.name, latency_s=1.0,
                              turn=tr.Turn(text, json.dumps(raw)))  # fmt: skip

    async def revise(self, ctx: tr.Context, text: str, first: tr.Translation,
                     review: str) -> tr.Translation:  # fmt: skip
        self.reviews.append(review)
        return await self.translate(ctx, text)


def test_the_model_sees_what_was_made_of_its_plan(testmap: Gazetteer, aerosonde: Airframe) -> None:
    ac = _ac(aerosonde, (-1095.0, 488.0))
    ctx = tr.Context(testmap, ac)

    async def run_check(plan: Plan):
        return check(build(plan, ac, testmap), testmap)

    def plan(*steps: dict) -> dict:
        return {"reply": {"kind": "plan", "plan": {"steps": list(steps)}}}

    ask = {"reply": {"kind": "clarify", "question": "Its edge instead?", "options": []}}
    t = _Scripted(plan({"do": "around", "target": _place("wind-farm")}), ask)
    got, report, _ = asyncio.run(tr.translate_checked(t, ctx, "circle the wind farm", run_check))
    assert isinstance(got.reply, tr.Clarify) and report is None
    assert "cannot be built" in t.reviews[0] and "no loop goes all the way round" in t.reviews[0]
    assert got.review and got.first is not None and got.latency_s == 2.0
    # Built as asked: no second request.
    t = _Scripted(plan({"do": "goto", "target": _place("zone-a")}, {"do": "hold"}))
    got, report, _ = asyncio.run(tr.translate_checked(t, ctx, "go to field A", run_check))
    assert report is not None and not t.reviews and got.review is None


# --------------------------------------------------------------------------- the map
def test_the_map_pictures_are_up_to_date() -> None:
    """Each site's map picture was drawn from its data as it is now (re-draw with
    ``uv run python tools/site_map.py``)."""
    spec = importlib.util.spec_from_file_location("site_map", REPO / "tools" / "site_map.py")
    assert spec is not None and spec.loader is not None
    site_map = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(site_map)
    metas = sorted(SITE_DIR.glob("*.map.json"))
    assert {m.name.split(".")[0] for m in metas} >= {"wrenfold_vale", "testmap", "london"}
    for meta in metas:
        site = meta.name.split(".")[0]
        assert json.loads(meta.read_text())["hash"] == site_map.map_hash(site), site
        assert tr.site_map(site) is not None
