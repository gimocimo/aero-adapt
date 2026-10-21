"""Mission v2 (Txt2Fly spec §8): item sequencing, loiters, jumps, RTL and the geofence."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from aeroadapt.airframes import Airframe
from aeroadapt.scenarios import load_site
from aeroadapt.stack.autopilot import MODE_AUTO, MODE_RTL, _mission_step, ap_init, inside_polygon
from aeroadapt.stack.config import StackParams, build_stack_params, load_stack_cfg
from aeroadapt.stack.types import (
    ITEM_FLYOVER,
    ITEM_JUMP,
    ITEM_LOITER_TIME,
    ITEM_LOITER_TURNS,
    ITEM_LOITER_UNLIM,
    ITEM_RTL,
    ITEM_WAYPOINT,
    LAT_LOITER,
    LAT_WAYPOINT,
    MAX_FENCE,
    Item,
    mission_from_items,
)


@pytest.fixture(scope="module")
def sp(aerosonde: Airframe) -> StackParams:
    return build_stack_params(
        load_stack_cfg("aerosonde_bm"), aerosonde, load_site("cambridge_west"), 0.01
    )


def _ap():
    return ap_init(MODE_AUTO, jnp.zeros(2), jnp.asarray(120.0))


def _step(sp, ap, mission, pos, vel=(0.0, 0.0), t=0.0, height=120.0):
    return _mission_step(sp, ap, jnp.asarray(pos), mission, vel=jnp.asarray(vel),
                         t=jnp.asarray(t), height=jnp.asarray(height))  # fmt: skip


def test_fly_over_switches_only_on_passing(sp: StackParams) -> None:
    for kind, switched in ((ITEM_WAYPOINT, True), (ITEM_FLYOVER, False)):
        m = mission_from_items([Item(kind, 500.0, 0.0, 120.0), Item(ITEM_WAYPOINT, 500, 500, 120)])
        ap, _ = _step(sp, _ap(), m, (480.0, 0.0))  # inside the 60 m acceptance radius
        assert (int(ap.wp_idx) == 1) is switched, kind
    ap, _ = _step(sp, _ap(), m, (505.0, 2.0))  # past it
    assert int(ap.wp_idx) == 1
    np.testing.assert_allclose(ap.leg_a, [500.0, 0.0])  # the next leg starts at the point


def _circle(sp, mission, *, radius, direction, start_deg, step_deg, n, t0=0.0, dt=1.0, ap=None):
    """Fly ``n`` positions around the item-0 centre (at the origin); returns the states."""
    ap = _ap() if ap is None else ap
    states = []
    for j in range(n):
        th = math.radians(start_deg + direction * step_deg * j)
        pos = (radius * math.cos(th), radius * math.sin(th))
        vel = (-direction * math.sin(th) * 20.0, direction * math.cos(th) * 20.0)
        ap, tgt = _step(sp, ap, mission, pos, vel, t=t0 + dt * j)
        states.append((ap, tgt))
    return states


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_loiter_turns_counts_from_capture_and_exits_towards_the_next_item(
    sp: StackParams, direction: float
) -> None:
    m = mission_from_items(
        [Item(ITEM_LOITER_TURNS, 0.0, 0.0, 150.0, radius=100.0 * direction, p1=1.0),
         Item(ITEM_WAYPOINT, 0.0, 1000.0, 150.0)]  # due east of the circle
    )  # fmt: skip
    # Start at the north point: heading east if clockwise, west if anticlockwise.
    states = _circle(sp, m, radius=100.0, direction=direction, start_deg=0.0, step_deg=5.0, n=150)
    idx = [int(a.wp_idx) for a, _ in states]
    first, tgt = states[0]
    assert bool(first.captured) and int(tgt.lateral) == LAT_LOITER
    assert float(tgt.radius) == 100.0 and float(tgt.direction) == direction
    assert float(tgt.alt) == 150.0
    done = idx.index(1)
    turned = 5.0 * done
    # One full turn, then on until the course points at the next item (within 30°): straight
    # away when clockwise, half a turn more when anticlockwise.
    assert (360.0 <= turned < 420.0) if direction > 0 else (480.0 <= turned < 600.0)
    ap, tgt = states[done]
    assert int(tgt.lateral) == LAT_WAYPOINT and not bool(ap.captured)
    th = math.radians(direction * turned)
    pos = np.array([100 * math.cos(th), 100 * math.sin(th)])
    np.testing.assert_allclose(ap.leg_a, pos, atol=1e-3)
    course = np.array([-direction * math.sin(th), direction * math.cos(th)])
    to_next = np.array([0.0, 1000.0]) - pos
    assert course @ to_next / np.linalg.norm(to_next) >= math.cos(math.radians(30.0))


def test_loiter_counts_only_once_on_the_circle(sp: StackParams) -> None:
    m = mission_from_items([Item(ITEM_LOITER_TURNS, 0, 0, 120, radius=100.0, p1=1.0),
                            Item(ITEM_WAYPOINT, 0, 1000, 120)])  # fmt: skip
    ap, _ = _step(sp, _ap(), m, (400.0, 0.0), (-20.0, 0.0))  # approaching from the north
    assert not bool(ap.captured) and float(ap.turned) == 0.0
    ap, _ = _step(sp, ap, m, (105.0, 0.0), (0.0, 20.0))  # within 10 % of the radius
    assert bool(ap.captured)


def test_loiter_time_and_unlimited(sp: StackParams) -> None:
    timed = mission_from_items([Item(ITEM_LOITER_TIME, 0, 0, 120, radius=80.0, p1=30.0),
                                Item(ITEM_WAYPOINT, 0, 1000, 120)])  # fmt: skip
    states = _circle(sp, timed, radius=80.0, direction=1.0, start_deg=0.0, step_deg=10.0,
                     n=60, t0=100.0, dt=1.0)  # fmt: skip
    idx = [int(a.wp_idx) for a, _ in states]
    k = idx.index(1)
    assert k >= 30 and float(states[0][0].t_capture) == 100.0
    forever = mission_from_items([Item(ITEM_LOITER_UNLIM, 0, 0, 120, radius=80.0)])
    states = _circle(sp, forever, radius=80.0, direction=1.0, start_deg=0.0, step_deg=10.0, n=200)
    assert all(int(a.wp_idx) == 0 and int(a.mode) == MODE_AUTO for a, _ in states)


def test_jump_repeats_then_rtl(sp: StackParams) -> None:
    a, b = (500.0, 0.0), (500.0, 500.0)
    m = mission_from_items(
        [Item(ITEM_WAYPOINT, *a, 120.0), Item(ITEM_WAYPOINT, *b, 120.0),
         Item(ITEM_JUMP, p1=0, p2=2)],  # the A-B loop three times in all
        loop=False,
    )  # fmt: skip
    ap, seq = _ap(), []
    for _ in range(3):
        for p in (a, b):
            ap, _ = _step(sp, ap, m, p)
            seq.append((int(ap.wp_idx), int(ap.mode)))
    assert seq[:-1] == [(1, MODE_AUTO), (0, MODE_AUTO)] * 2 + [(1, MODE_AUTO)]
    assert seq[-1][1] == MODE_RTL and list(np.asarray(ap.jumps)[:3]) == [0, 0, 2]
    forever = m._replace(param=m.param.at[2, 2].set(-1.0))
    ap = _ap()
    for _ in range(10):
        for p in (a, b):
            ap, _ = _step(sp, ap, forever, p)
    assert int(ap.mode) == MODE_AUTO and int(ap.wp_idx) == 0


def test_rtl_item_and_item_speed(sp: StackParams) -> None:
    m = mission_from_items(
        [Item(ITEM_WAYPOINT, 500.0, 0.0, 120.0, speed=18.0), Item(ITEM_RTL)], speed=25.0
    )
    ap, tgt = _step(sp, _ap(), m, (0.0, 0.0))
    assert float(tgt.speed) == 18.0
    ap, tgt = _step(sp, ap, m, (505.0, 0.0))
    assert int(ap.mode) == MODE_RTL and int(tgt.lateral) == LAT_LOITER and float(tgt.speed) == 25.0


def test_geofence_breach_switches_to_rtl(sp: StackParams) -> None:
    fence = [(1000.0, -1000.0), (1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, -1000.0)]
    m = mission_from_items([Item(ITEM_WAYPOINT, 3000.0, 0.0, 120.0)], fence=fence, fence_alt=150.0)
    ap, _ = _step(sp, _ap(), m, (900.0, 0.0))
    assert int(ap.mode) == MODE_AUTO
    ap, _ = _step(sp, _ap(), m, (1001.0, 0.0))
    assert int(ap.mode) == MODE_RTL
    ap, _ = _step(sp, _ap(), m, (0.0, 0.0), height=151.0)
    assert int(ap.mode) == MODE_RTL
    ap, _ = _mission_step(sp, _ap(), jnp.asarray([1001.0, 0.0]), m)  # no height: no fence check
    assert int(ap.mode) == MODE_AUTO


def _inside_reference(p: np.ndarray, poly: np.ndarray) -> bool:
    inside = False
    for (an, ae), (bn, be) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        if (ae > p[1]) != (be > p[1]) and p[0] < an + (p[1] - ae) * (bn - an) / (be - ae):
            inside = not inside
    return inside


def test_inside_polygon_matches_a_reference() -> None:
    rng = np.random.default_rng(3)
    poly = np.array([[0, 0], [400, 100], [300, 400], [150, 200], [-100, 350], [-50, 50]], float)
    verts = np.zeros((MAX_FENCE, 2))
    verts[: len(poly)] = poly
    for p in rng.uniform(-200, 500, size=(400, 2)):
        got = bool(inside_polygon(jnp.asarray(p), jnp.asarray(verts), jnp.asarray(len(poly))))
        assert got == _inside_reference(p, poly), p


def test_battery_reserve_switches_to_rtl(sp: StackParams) -> None:
    """ArduPlane's BATT_FS_LOW_ACT = RTL, on the battery monitor's consumed charge."""
    m = mission_from_items(
        [Item(ITEM_LOITER_UNLIM, 0.0, 0.0, 120.0, radius=150.0)], rtl_mah=11200.0
    )
    ap, _ = _mission_step(sp, _ap(), jnp.asarray([150.0, 0.0]), m, used_mah=jnp.asarray(11199.0))
    assert int(ap.mode) == MODE_AUTO
    ap, tgt = _mission_step(sp, _ap(), jnp.asarray([150.0, 0.0]), m, used_mah=jnp.asarray(11200.0))
    assert int(ap.mode) == MODE_RTL and int(tgt.lateral) == LAT_LOITER
    legacy = mission_from_items([Item(ITEM_WAYPOINT, 900.0, 0.0, 120.0)])  # rtl_mah 0: off
    ap, _ = _mission_step(sp, _ap(), jnp.asarray([0.0, 0.0]), legacy, used_mah=jnp.asarray(1e6))
    assert int(ap.mode) == MODE_AUTO


def test_loiter_ends_at_once_when_the_next_item_is_inside_the_circle(sp: StackParams) -> None:
    """ArduPlane's verify_loiter_heading: the course never points at a point inside the
    circle, so the loiter ends once its turns are done."""
    m = mission_from_items([Item(ITEM_LOITER_TURNS, 0, 0, 120, radius=150.0, p1=1.0),
                            Item(ITEM_WAYPOINT, 60.0, 40.0, 120.0)])  # fmt: skip
    states = _circle(sp, m, radius=150.0, direction=1.0, start_deg=0.0, step_deg=5.0, n=90)
    idx = [int(a.wp_idx) for a, _ in states]
    assert 1 in idx and 360.0 <= 5.0 * idx.index(1) < 370.0
    # … and towards home when the next item is RTL.
    rtl = mission_from_items([Item(ITEM_LOITER_TURNS, 0, 0, 120, radius=150.0, p1=1.0),
                              Item(ITEM_RTL)], home=(0.0, 1000.0, 120.0))  # fmt: skip
    states = _circle(sp, rtl, radius=150.0, direction=1.0, start_deg=0.0, step_deg=5.0, n=90)
    modes = [int(a.mode) for a, _ in states]
    assert MODE_RTL in modes and 360.0 <= 5.0 * modes.index(MODE_RTL) < 420.0
