"""Synthetic worlds (docs/07 §9): V29 the simulator touches what the world holds; V30 forced
landings over Wrenfold's hedges and trees; V31 planned flights keep clear of the ground and of
everything standing on it."""

from __future__ import annotations

import collections

import jax
import numpy as np
import pytest

from aeroadapt.harness.closed_loop import setup_loop
from aeroadapt.live.session import LiveSession
from aeroadapt.scenarios.resolve import resolve
from aeroadapt.scenarios.runner import run_scenario
from aeroadapt.scenarios.schema import Scenario, load_scenario
from aeroadapt.stack.types import make_mission
from aeroadapt.telemetry.events import extract_events
from aeroadapt.telemetry.schema import columns
from aeroadapt.txt2fly.plan import load_plan
from aeroadapt.vehicle.protocol import InitialCondition, Termination
from aeroadapt.world import load_world

PLANS = "configs/txt2fly/plans/wrenfold_vale"


def _fly_level(start: tuple[float, float], alt: float, heading: float, to: tuple[float, float],
               seconds: float):  # fmt: skip
    """The stock stack holding ``alt`` (above home) towards ``to``; (events, columns)."""
    r = resolve(load_scenario("wrenfold-nominal"))
    ini = InitialCondition(pos_ne_m=start, alt_agl_m=alt, heading_deg=heading)
    loop = setup_loop(r.airframe, r.site, r.env, ini, stack=r.stack, seed=7, config=r.config)
    mission = make_mission([(*to, alt)], home=(*to, alt))
    _, _, log = loop.run(mission, round(seconds * r.config.control_hz), jax.random.PRNGKey(1))
    cols = columns(jax.tree.map(np.asarray, log))
    return extract_events(cols, r.site), cols


# --------------------------------------------------------------------------- V29 (fast)
def test_v29_flying_into_the_church_tower_is_a_collision() -> None:
    tower = next(b for b in load_world("wrenfold").buildings if b.kind == "tower")
    events, cols = _fly_level((tower.n, tower.e - 120.0), tower.base + 20.0, 90.0,
                              (tower.n, tower.e + 400.0), 8.0)  # fmt: skip
    hit = [e for e in events if e["type"] == "collision"]
    assert hit and hit[0]["payload"]["hit"] == "St Michael's Church tower"
    k = int(np.argmax(cols["done"] > 0))
    assert cols["termination"][k] == Termination.OBSTACLE
    assert abs(cols["truth.pos_e"][k] - tower.e) < 6.0  # at the tower, not before it


def test_v29_flying_into_rising_ground_is_a_crash() -> None:
    """Level at 25 m above home towards the south ridge over open fields: the ground rises
    to meet it, and the impact is classed by how fast the slope closes, not the sink rate."""
    events, cols = _fly_level((-700.0, -100.0), 25.0, 180.0, (-1500.0, -100.0), 60.0)
    k = int(np.argmax(cols["done"] > 0))
    assert cols["done"].any() and cols["termination"][k] == Termination.GROUND_CONTACT
    ground = load_world("wrenfold").dtm.at(cols["truth.pos_n"][k], cols["truth.pos_e"][k])
    assert ground > 15.0 and cols["truth.hagl"][k] < 0.2
    td = next(e["payload"] for e in events if e["type"] == "touchdown")
    assert td["sink_rate_mps"] < 1.5 < td["closing_speed_mps"] and td["class"] != "controlled"
    assert td["surface"].startswith("a field")


def test_v29_motor_cut_lands_in_a_surveyed_field() -> None:
    r = run_scenario(load_scenario("wrenfold-motorcut"))
    td = [e for e in r.events if e["type"] == "touchdown"]
    assert td and td[0]["payload"]["in_zone"] and td[0]["payload"]["zone"] == "E"
    assert "Hollow Pasture" in td[0]["payload"]["surface"]
    assert float(np.min(r.cols["truth.clearance"][r.cols["cont.phase"] == 0])) > 50.0


# --------------------------------------------------------------------------- V30 (full)
@pytest.mark.full
def test_v30_forced_landings_over_wrenfold() -> None:
    """Motor cuts at six points of the circuit in four winds (24 flights): the classical
    contingency lands in a surveyed field in at least 75 % of them, and hits a hedge, tree or
    building in at most 10 % (docs/07 §9 records the run)."""
    base = load_scenario("wrenfold-motorcut").model_dump(mode="json")
    out = collections.Counter()
    for wind_from in (250.0, 70.0, 160.0, 340.0):
        for cut in (40.0, 75.0, 110.0, 150.0, 190.0, 230.0):
            d = {**base, "faults": [{"type": "motor_cut", "trigger": {"at_time_s": cut}}]}
            wind = {**base["environment"]["wind"], "from_deg": wind_from}
            d["environment"] = {**base["environment"], "wind": wind}
            r = run_scenario(Scenario.model_validate(d))
            end = next(e for e in r.events if e["type"] in ("touchdown", "collision", "terminated"))
            if end["type"] == "touchdown":
                out["in zone" if end["payload"]["in_zone"] else "outside"] += 1
            else:
                out[end["type"]] += 1
                assert end["type"] == "collision" and end["payload"]["hit"]
    assert sum(out.values()) == 24
    assert out["in zone"] >= 18, out
    assert out["collision"] <= 2, out


# --------------------------------------------------------------------------- V31
def _session() -> LiveSession:
    return LiveSession(load_scenario("txt2fly-demo"))


def test_v31_low_plans_are_lifted_over_the_trees() -> None:
    s = _session()
    for _ in range(10):
        s.step()
    err, prop = s.propose(load_plan(f"{PLANS}/hanger_wood_low.yaml"))
    assert err is None and prop is not None and prop["ok"], prop and prop["checks"]
    notes = " ".join(prop["notes"])
    assert "to keep 30 m above Hanger Wood" in notes
    k4 = next(c for c in prop["checks"] if c["rule"] == "K4")
    assert k4["ok"] and "30 m above trees and buildings" in k4["reason"]


def test_v31_literal_low_plans_fail_k4() -> None:
    from aeroadapt.txt2fly.builder import build
    from aeroadapt.txt2fly.checker import check
    from aeroadapt.txt2fly.gazetteer import load_gazetteer

    s = _session()
    s.step()
    gaz = load_gazetteer(s.resolved.site)
    built = build(load_plan(f"{PLANS}/hanger_wood_low.yaml"), s.aircraft(), gaz, adapt=False)
    k4 = next(c for c in check(built, gaz).checks if c.rule == "K4")
    assert not k4.ok and "Hanger Wood" in k4.reason and k4.suggestion


@pytest.mark.full
@pytest.mark.parametrize("name", ["hanger_wood_low", "patrol_abc", "church_30m"])
def test_v31_flown_plans_keep_clear_of_trees_and_ground(name: str) -> None:
    """Flown by the full simulator and stock stack: at least 20 m above any obstacle and
    35 m above the ground all the way (planned: 30 m and 40 m)."""
    s = _session()
    for _ in range(20):
        s.step()
    err, prop = s.propose(load_plan(f"{PLANS}/{name}.yaml"))
    assert err is None and prop is not None and prop["ok"]
    assert s.confirm(prop["id"], prop["warnings"]) is None
    t0 = s.time_s
    for _ in range(round(min(prop["energy"]["minutes"] or 6.0, 6.0) * 600)):
        s.step()
    cols = s.all_columns()
    keep = cols["time_us"] * 1e-6 > t0 + 1.0
    assert float(cols["truth.clearance"][keep].min()) > 20.0
    assert float(cols["truth.hagl"][keep].min()) > 35.0
