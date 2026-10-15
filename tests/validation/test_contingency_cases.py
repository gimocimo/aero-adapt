"""Classical contingency v0 on hand-made motor-cut cases (M1 week-3 exit criterion; the
statistical V12 campaign comes in week 4).

The aircraft flies a square mission north-east of the Coton fields in light turbulence; the
motor is cut on the first leg, with the safe zones within reach, in several winds. Plus two
cuts where no zone is reachable (straight ahead into wind).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
import pytest

from aeroadapt.airframes import Airframe
from aeroadapt.faultlab import build_fault_config
from aeroadapt.faultlab.schema import MotorCut, TimeTrigger
from aeroadapt.harness import setup_loop
from aeroadapt.harness.acceptance import LIGHT_W20_MPS
from aeroadapt.scenarios import load_site
from aeroadapt.sim.aerojax import make_env
from aeroadapt.stack.types import make_mission
from aeroadapt.vehicle.protocol import InitialCondition, classify_touchdown

pytestmark = pytest.mark.full

SQUARE = [(800, 0), (800, 800), (0, 800), (0, 0)]


@dataclass(frozen=True)
class Case:
    t_cut: float
    wind_from: float
    wind: float
    alt: float = 120.0
    mission: tuple[tuple[float, float], ...] = tuple(SQUARE)
    heading: float = 0.0


IN_REACH = [
    Case(20, 0, 0),
    Case(20, 250, 6),
    Case(20, 70, 6),
    Case(20, 180, 5),
    Case(30, 250, 6, alt=150, mission=((-1200, -1200), (0, 0)), heading=225),
]
MARGINAL = [Case(45, 70, 6)]  # ~900 m out: on the reachability boundary, either outcome is fine
MARGINAL = [Case(45, 70, 6)]  # ~900 m out: on the reachability boundary, either outcome is fine
OUT_OF_REACH = [
    Case(75, 250, 6),
    Case(40, 250, 6, alt=80, mission=((2500, 0), (2500, 2500))),
]


@dataclass(frozen=True)
class Outcome:
    latency: float
    zone: int
    in_zone: bool
    touchdown: str


def _fly(aerosonde: Airframe, case: Case, seed: int) -> Outcome:
    site = load_site("cambridge_west")
    env = make_env(
        alt_origin_msl_m=site.ground_msl_m,
        magnetic_ned_gauss=site.magnetic.field_ned_gauss,
        wind_from_deg=case.wind_from,
        wind_speed_mps=case.wind,
        w20_mps=LIGHT_W20_MPS,
    )
    faults = build_fault_config(
        [MotorCut(type="motor_cut", trigger=TimeTrigger(at_time_s=case.t_cut))]
    )
    loop = setup_loop(
        aerosonde,
        site,
        env,
        InitialCondition(alt_agl_m=case.alt, heading_deg=case.heading),
        faults=faults,
        seed=seed,
    )
    mission = make_mission([(n, e, case.alt) for n, e in case.mission])
    _, _, log = loop.run(mission, 30_000, jax.random.PRNGKey(100 + seed))
    log = jax.tree.map(np.asarray, log)
    assert log.done.any(), "still airborne after 300 s"
    k = int(np.argmax(log.done))
    v, e = log.vel_ned[k], log.euler[k]
    zone = int(log.zone[k])
    n, e_ = (float(x) for x in log.pos_ned[k, :2])
    in_zone = zone >= 0 and site.safe_zones[zone].contains(n, e_)
    return Outcome(
        latency=float(log.t[int(np.argmax(log.failed))] - case.t_cut),
        zone=zone,
        in_zone=in_zone,
        touchdown=classify_touchdown(float(v[2]), float(e[0]), float(e[1])),
    )


@pytest.fixture(scope="module")
def outcomes(aerosonde: Airframe) -> list[Outcome]:
    return [_fly(aerosonde, c, i) for i, c in enumerate(IN_REACH + OUT_OF_REACH + MARGINAL)]


def test_detection_latency(outcomes: list[Outcome]) -> None:
    assert max(o.latency for o in outcomes) <= 1.0  # V12: ESC ≤ 1.0 s


def test_lands_in_zone_when_in_reach(outcomes: list[Outcome]) -> None:
    reach = outcomes[: len(IN_REACH)]
    assert all(o.zone >= 0 for o in reach)
    assert sum(o.in_zone for o in reach) >= len(reach) - 1, reach


def test_no_crashes(outcomes: list[Outcome]) -> None:
    assert all(o.touchdown in ("controlled", "hard") for o in outcomes), outcomes


def test_out_of_reach_glides_straight_in(outcomes: list[Outcome]) -> None:
    n = len(IN_REACH)
    assert all(o.zone == -1 for o in outcomes[n : n + len(OUT_OF_REACH)])
