"""V11: stock autopilot acceptance (spec §13). The criteria live in
``aeroadapt.harness.acceptance`` so tuning and tests share one definition."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from aeroadapt.airframes import Airframe
from aeroadapt.harness import setup_loop
from aeroadapt.harness.acceptance import TESTS, leg_xtrack, run_v11, settled_mask
from aeroadapt.scenarios import load_site
from aeroadapt.sim.aerojax import make_env
from aeroadapt.stack.types import make_mission
from aeroadapt.vehicle.protocol import InitialCondition


def test_closed_loop_smoke(aerosonde: Airframe) -> None:
    """Fast tier: the stock stack flies a mission leg and a turn in calm air."""
    site = load_site("cambridge_west")
    env = make_env(
        alt_origin_msl_m=site.ground_msl_m, magnetic_ned_gauss=site.magnetic.field_ned_gauss
    )
    loop = setup_loop(aerosonde, site, env, InitialCondition())
    mission = make_mission([(900, 0, 120), (900, 900, 120)])
    _, _, log = loop.run(mission, 5_500, jax.random.PRNGKey(0))  # 55 s
    log = jax.tree.map(np.asarray, log)
    assert not log.done.any()
    assert int(log.wp_idx[-1]) == 1  # reached the first waypoint and turned
    assert np.abs(-log.pos_ned[:, 2] - 120.0).max() < 6.0
    assert np.abs(log.va - 25.0).max() < 3.0
    xt = leg_xtrack(log, mission, np.zeros(2))[settled_mask(log, 25.0)]
    assert np.sqrt(np.mean(xt**2)) < 3.0


@pytest.mark.full
@pytest.mark.parametrize("test", TESTS)
def test_v11_calm(test: str) -> None:
    failures = [str(c) for c in run_v11(only=(test,)) if not c.ok]
    assert not failures, "\n".join(failures)


@pytest.mark.full
@pytest.mark.parametrize("test", [t for t in TESTS if t != "level"])
def test_v11_light_turbulence(test: str) -> None:
    checks = run_v11(turbulent=True, seeds=tuple(range(16)), only=(test,))
    failures = [str(c) for c in checks if not c.ok]
    assert not failures, "\n".join(failures)
