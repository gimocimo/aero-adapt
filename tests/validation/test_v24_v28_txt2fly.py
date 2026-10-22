"""Txt2Fly validation (spec 06 §12): V24 the turn-radius rule against flown circles, V25 the
checker's soundness on random plans, V26 flown plans against their previews, V28 live plans.

Flights are the full simulator and stock stack (the live session's closed loop), starting from
the state a plan was checked in; the planner only ever sees the navigator's estimates."""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aeroadapt.airframes.fleet import load_fleet_airframe
from aeroadapt.harness.closed_loop import closed_loop
from aeroadapt.live.server import SAMPLE_FLEET
from aeroadapt.live.session import LiveSession, replay_commands
from aeroadapt.scenarios.schema import Scenario, load_scenario
from aeroadapt.stack.autopilot import MODE_AUTO, MODE_RTL, ap_init
from aeroadapt.stack.config import load_stack_cfg
from aeroadapt.stack.types import ITEM_LOITER_UNLIM, Item, mission_from_items, mission_to_dict
from aeroadapt.txt2fly import geometry as geo
from aeroadapt.txt2fly.builder import TRACK_MARGIN_M, Built, build, r_min
from aeroadapt.txt2fly.checker import check
from aeroadapt.txt2fly.gazetteer import Gazetteer, load_gazetteer
from aeroadapt.txt2fly.plan import Plan, load_plan

PLANS = Path(__file__).resolve().parents[2] / "configs" / "txt2fly" / "plans" / "cambridge_west"
AIRFRAMES = ("aerosonde_bm", "gen-twin_boom-00058", "gen-glider-00191")


def _session(
    airframe: str = "aerosonde_bm", wind: tuple[float, float] = (6.0, 250.0)
) -> LiveSession:
    d = load_scenario("txt2fly-cambridge").model_dump(mode="json")
    d["environment"]["wind"].update(speed_mps=wind[0], from_deg=wind[1])
    sc = Scenario.model_validate(d)
    if airframe == "aerosonde_bm":
        return LiveSession(sc)
    return LiveSession(sc, load_fleet_airframe(SAMPLE_FLEET, airframe),
                       load_stack_cfg(SAMPLE_FLEET, airframe))  # fmt: skip


def _fly(s: LiveSession, seconds: float) -> None:
    for _ in range(round(seconds / 0.1)):
        s.step()


def _truth(s: LiveSession, since: float) -> tuple[np.ndarray, np.ndarray]:
    cols = s.all_columns()
    keep = cols["time_us"] * 1e-6 >= since
    ne = np.stack([cols["truth.pos_n"], cols["truth.pos_e"]], 1)[keep]
    return ne, cols["truth.alt_agl"][keep]


def _clearance(gaz: Gazetteer, ne: np.ndarray) -> np.ndarray:
    d = -geo.signed_distance(ne, gaz.geofence)
    for a in gaz.no_fly:
        d = np.minimum(d, a.signed_distance(ne))
    return d


def _off_preview(ne: np.ndarray, preview: np.ndarray) -> np.ndarray:
    """Distance from each flown point to the preview's polyline."""
    a, b = preview[:-1], preview[1:]
    ab = b - a
    ab2 = np.maximum((ab * ab).sum(1), 1e-9)
    out = np.empty(len(ne))
    for k in range(0, len(ne), 256):
        p = ne[k : k + 256, None, :]
        t = np.clip(((p - a) * ab).sum(-1) / ab2, 0.0, 1.0)
        out[k : k + 256] = np.linalg.norm(p - (a + t[..., None] * ab), axis=-1).min(1)
    return out


# --------------------------------------------------------------------------- V28 (fast)
def test_v28_live_plan_applies_at_the_next_chunk_and_replays() -> None:
    s = _session()
    _fly(s, 3.0)
    err, prop = s.propose(load_plan(PLANS / "patrol_abc.yaml"))
    assert err is None and prop is not None and prop["ok"], prop and prop["checks"]
    _fly(s, 1.0)  # the operator reads it; confirming rebuilds from the state then
    t_confirm = s.time_s
    assert s.confirm(prop["id"], prop["warnings"]) is None
    assert s.confirm(prop["id"]) == f"no proposal {prop['id']}"
    _fly(s, 0.1)  # the plan starts with the next chunk
    entry = s.commands[-1]
    assert entry["time_s"] == pytest.approx(t_confirm) and "set_mission" in entry
    assert entry["set_mission"]["plan"]["steps"][0]["do"] == "patrol"
    assert s.active_plan is not None and int(s._st.ap.mode) == MODE_AUTO
    cols = s.all_columns()
    t = cols["time_us"] * 1e-6
    assert (cols["wp_idx"][t > t_confirm] == 0).all()  # flying to the first item
    _fly(s, 60.0)
    r = replay_commands(s.resolved.scenario, s.commands)
    while r.time_s < s.time_s - 1e-6:
        r.step()
    a, b = s.all_columns(), r.all_columns()
    assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)


def test_v28_warnings_need_acknowledging() -> None:
    s = _session()
    _fly(s, 1.0)
    far = {"steps": [{"do": "goto", "target": {"kind": "point", "ne_m": [-1150.0, -1150.0]},
                      "alt_m": 40}]}  # fmt: skip
    err, prop = s.propose(far)
    assert err is None and prop is not None and prop["ok"] and prop["warnings"] == ["K7"]
    assert s.confirm(prop["id"]) == "acknowledge K7 first"
    err, prop = s.propose(far)
    assert prop is not None and s.confirm(prop["id"], ["K7"]) is None
    s.step()
    assert s.commands[-1]["set_mission"]["acknowledged"] == ["K7"]


def test_v28_bad_plans_are_refused_with_reasons() -> None:
    s = _session()
    err, _ = s.propose({"steps": [{"do": "fly_upside_down"}]})
    assert err is not None
    places = [{"kind": "place", "id": "field A"}, {"kind": "place", "id": "field C"}]
    _, prop = s.propose({"steps": [{"do": "patrol", "targets": places, "laps": 400}]})
    assert prop is not None and not prop["ok"]  # nothing fixes a battery too small
    assert "the plan no longer passes" in (s.confirm(prop["id"]) or "")


# --------------------------------------------------------------------------- V24 (full)
@cache
def _circle_radius(airframe: str, wind: float, radius: float) -> tuple[float, float]:
    """Fly an endless circle over field A; returns the commanded speed and the 95th
    percentile of the flown distance from its centre over two turns after a turn to settle."""
    s = _session(airframe, (wind, 250.0))
    _fly(s, 2.0)
    centre = (-166.0, -61.0)
    m = mission_from_items([Item(ITEM_LOITER_UNLIM, *centre, 100.0, radius=radius)],
                           speed=s.resolved.stack_cfg.tecs.airspeed_cruise_mps)  # fmt: skip
    s.set_mission({"mission": mission_to_dict(m)})
    v = s.resolved.stack_cfg.tecs.airspeed_cruise_mps
    turn = 2 * math.pi * max(radius, r_min(v, wind, 45.0)) / v
    _fly(s, 30.0 + turn)
    t0 = s.time_s
    _fly(s, 2 * turn)
    ne, _ = _truth(s, t0)
    return v, float(np.percentile(np.linalg.norm(ne - np.array(centre), axis=1), 95))


@pytest.mark.full
@pytest.mark.parametrize("airframe", AIRFRAMES)
@pytest.mark.parametrize("wind", [0.0, 6.0])
def test_v24_tightest_circle_is_flown(airframe: str, wind: float) -> None:
    """At r_min the circle is held (within 15 %); at half of it the aircraft cannot hold the
    circle and flies wider: the rule describes what the stock autopilot can do."""
    cfg = (load_stack_cfg(airframe) if airframe == "aerosonde_bm"
           else load_stack_cfg(SAMPLE_FLEET, airframe))  # fmt: skip
    rm = r_min(cfg.tecs.airspeed_cruise_mps, wind, cfg.l1.roll_limit_deg)
    _, flown = _circle_radius(airframe, wind, rm)
    assert flown == pytest.approx(rm, rel=0.15)
    _, tight = _circle_radius(airframe, wind, 0.5 * rm)
    assert tight > 0.6 * rm


# --------------------------------------------------------------------------- V26 (full)
@pytest.mark.full
@pytest.mark.parametrize("airframe", AIRFRAMES)
@pytest.mark.parametrize("wind", [(0.0, 0.0), (6.0, 250.0)])
def test_v26_flown_plans_stay_near_their_previews(airframe: str, wind) -> None:
    flown = 0
    for name in ("patrol_abc", "fly_by_field_a", "circle_field_c"):
        s = _session(airframe, wind)
        _fly(s, 5.0)
        gaz = load_gazetteer(s.resolved.site)
        plan = load_plan(PLANS / f"{name}.yaml")
        err, prop = s.propose(plan)
        assert err is None and prop is not None
        if not prop["ok"]:
            continue  # refused (with reasons): nothing flies
        report = check(build(plan, s.aircraft(), gaz), gaz)
        preview = report.built.preview
        assert preview is not None and s.confirm(prop["id"], prop["warnings"]) is None
        t0 = s.time_s
        _fly(s, min(float(preview.t[-1]), 300.0))
        ne, alt = _truth(s, t0)
        off = _off_preview(ne, preview.ne)
        # The preview banks 5° short of the limit, so its turns are a little wider than the
        # aircraft's: what matters is that nothing flown comes near a no-fly area.
        assert off.max() <= 3 * TRACK_MARGIN_M, (name, off.max())
        assert _clearance(gaz, ne).min() > 0.0, name
        assert alt.max() < 140.0
        flown += 1
    assert flown >= 1


# --------------------------------------------------------------------------- V25 (full)
def _random_plan(rng: np.random.Generator, gaz: Gazetteer) -> Plan:
    places = sorted(p.id for p in gaz.places.values()
                    if p.kind in ("safe zone", "field", "wood", "home", "farm"))  # fmt: skip
    pick = lambda: {"kind": "place", "id": str(rng.choice(places))}  # noqa: E731
    steps = []
    for _ in range(int(rng.integers(1, 3))):
        alt = [None, 60.0, 80.0, 100.0, 120.0][int(rng.integers(5))]
        kind = str(rng.choice(["goto", "pass_over", "orbit", "patrol"]))
        if kind == "orbit":
            steps.append({"do": "orbit", "target": pick(), "alt_m": alt, "turns": 1,
                          "radius_m": float(rng.uniform(60.0, 250.0))})  # fmt: skip
        elif kind == "patrol":
            targets = [pick() for _ in range(int(rng.integers(2, 4)))]
            steps.append({"do": "patrol", "targets": targets, "alt_m": alt, "laps": 1})
        else:
            steps.append({"do": kind, "target": pick(), "alt_m": alt})
    steps.append({"do": "return_home"})
    return Plan.model_validate({"steps": steps})


@pytest.mark.full
def test_v25_accepted_random_plans_never_intrude_and_violations_are_caught() -> None:
    s = _session()
    _fly(s, 5.0)
    gaz = load_gazetteer(s.resolved.site)
    ac = s.aircraft()
    rng = np.random.default_rng(25)
    accepted: list[Built] = []
    n = 200
    for _ in range(n):
        plan = _random_plan(rng, gaz)
        r = check(build(plan, ac, gaz), gaz)
        if r.ok:
            accepted.append(r.built)
            # Injected violations, one at a time, are each caught by their rule.
            high = plan.model_copy(update={"steps": [
                st.model_copy(update={"alt_m": 150.0}) if hasattr(st, "alt_m") else st
                for st in plan.steps]})  # fmt: skip
            literal = check(build(high, ac, gaz, adapt=False), gaz)  # as asked, unadapted
            assert "K4" in {c.rule for c in literal.checks if not c.ok}
            church = Plan.model_validate({"steps": [
                {"do": "goto", "target": {"kind": "place",
                                          "id": "Saint Peter (Coton Parish Church)"}},
                *[st.model_dump() for st in plan.steps]]})  # fmt: skip
            literal = check(build(church, ac, gaz, adapt=False), gaz)
            failed = {c.rule for c in literal.checks if not c.ok}
            assert failed & {"K3", "K8"}
    assert len(accepted) >= 20, f"only {len(accepted)} of {n} random plans were accepted"

    # Fly every accepted plan at once (vmap over missions) from the state they were checked in.
    su, r = s._su, s.resolved
    same = [jax.tree.map(lambda v, o: jnp.asarray(v, o.dtype), b.mission, r.mission)
            for b in accepted]  # fmt: skip
    batched = jax.tree.map(lambda *x: jnp.stack(x), *same)
    nav = s._st.nav
    st0 = s._st._replace(ap=ap_init(MODE_AUTO, nav.pos[:2], -nav.pos[2]))
    steps = 30000  # 300 s
    fly = jax.vmap(lambda m: closed_loop(su.vp, s._env, s._faults, su.sp, m, s._sim, st0, s._z,
                                         jax.random.PRNGKey(7), n_steps=steps, config=r.config,
                                         log_every=50)[2])  # fmt: skip
    log = fly(batched)
    pos = np.asarray(log.pos_ned)  # (plans, samples, 3) truth
    mode = np.asarray(log.mode)
    for k, b in enumerate(accepted):
        ne = pos[k, :, :2]
        assert _clearance(gaz, ne).min() > 0.0, (k, b.plan.model_dump())
        assert (-pos[k, :, 2]).max() < 140.0
    assert (mode == MODE_RTL).any(axis=1).mean() > 0.8  # most plans reach home in 300 s
