"""V16 and V17 (M2 spec §10): aero v2 tables.

V16: the Beard & McLain model sampled onto the tables reproduces the analytic model (aero v1)
in the normal flight envelope, pointwise and in closed loop (golden flights).
V17: N-version parity of the tabulated model, with random tables, against the NumPy reference.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aeroadapt.airframes import (
    ALPHA_GRID_N,
    AeroTables,
    Airframe,
    AirframeParams,
    alpha_grid,
    bm_tables,
)
from aeroadapt.scenarios.goldens import GOLDEN_SCENARIOS, compare, metrics
from aeroadapt.sim.aerojax import (
    BM_PARITY,
    controls_from_vector,
    forces_moments,
    rollout,
    state_from_vector,
    state_to_vector,
)
from aeroadapt.sim.aerojax.aero import drag_coefficient, lift_coefficient
from aeroadapt.sim.aerojax.aero_tables import table_lookup
from reference.bm_numpy import model as ref
from tests.regression.test_golden_flights import GOLDENS
from tests.validation.test_v3_nversion import _sample, _smooth_inputs

DT = 0.005
FAST_GOLDENS = ("m1-nominal", "m1-motorcut-classical")


def tabulated(af: Airframe) -> Airframe:
    return af.model_copy(update={"aero": af.aero.model_copy(update={"model": "bm_tabulated"})})


def random_tables(seed: int) -> AeroTables:
    """Smooth but arbitrary tables: a few random harmonics in α for every coefficient."""
    rng = np.random.default_rng(seed)
    a = alpha_grid()

    def series(scale: float, offset: float = 0.0) -> np.ndarray:
        k = np.arange(1, 6)
        amp = rng.normal(size=(2, 5)) * scale / k
        return offset + amp[0] @ np.sin(np.outer(k, a)) + amp[1] @ np.cos(np.outer(k, a))

    return AeroTables(
        C_L=series(0.8),
        C_D=np.abs(series(0.3)) + 0.03,
        C_m=series(0.2),
        eta_ctrl=np.clip(series(0.2, 0.8), 0.1, 1.0),
        eta_damp=np.clip(series(0.2, 0.8), 0.1, 1.0),
    )


def test_lookup_is_exact_at_nodes_and_linear_between() -> None:
    a = alpha_grid()
    table = np.random.default_rng(0).normal(size=ALPHA_GRID_N)
    np.testing.assert_allclose(table_lookup(jnp.asarray(table), jnp.asarray(a)), table, atol=1e-12)
    mid = 0.5 * (a[:-1] + a[1:])
    np.testing.assert_allclose(
        table_lookup(jnp.asarray(table), jnp.asarray(mid)),
        0.5 * (table[:-1] + table[1:]),
        atol=1e-12,
    )
    # A fleet axis on the tables vmaps with one shape.
    fleet = jnp.stack([jnp.asarray(table), 2.0 * jnp.asarray(table)])
    out = jax.vmap(table_lookup, in_axes=(0, None))(fleet, jnp.asarray(0.1))
    assert out.shape == (2,) and abs(float(out[1]) - 2.0 * float(out[0])) < 1e-12


def test_v16_bm_tables_match_the_analytic_coefficients(params: AirframeParams) -> None:
    t = bm_tables(params)
    alpha = np.radians(np.random.default_rng(1).uniform(-15.0, 20.0, 2_000))
    pairs = (
        (t.C_L, lift_coefficient(params, jnp.asarray(alpha))),
        (t.C_D, drag_coefficient(params, jnp.asarray(alpha))),
        (t.C_m, params.C_m_0 + params.C_m_alpha * alpha),
    )
    for table, exact in pairs:
        err = np.abs(np.asarray(table_lookup(jnp.asarray(table), jnp.asarray(alpha))) - exact)
        assert err.max() <= 1e-3, err.max()


def test_v16_bm_tables_forces_match_aero_v1(params: AirframeParams) -> None:
    p_tab = params._replace(tables=bm_tables(params))
    xs, us, ws = _sample(1_000, seed=2)

    def fm(p: AirframeParams):
        f = jax.vmap(
            lambda x, u, w: forces_moments(
                p, BM_PARITY, state_from_vector(x), controls_from_vector(u), w
            )
        )
        return jax.jit(f)(xs, us, ws)

    a, b = fm(params), fm(p_tab)
    # The V16 envelope is α in [-15°, 20°]. Near B&M's sharp stall blend (27°) the 0.5° grid is
    # coarser than the sigmoid, which is why real stall data is tabulated instead.
    alpha = np.degrees(np.asarray(a.air.alpha))
    ok = (alpha >= -15.0) & (alpha <= 20.0)
    assert ok.sum() > 500
    # Relative to the aerodynamic scale q̄·S (and q̄·S·b for moments) at each sample.
    qs = 0.5 * float(BM_PARITY.rho) * np.asarray(a.air.va)[ok] ** 2 * params.S
    df = np.abs(np.asarray(a.force_body) - np.asarray(b.force_body))[ok] / qs[:, None]
    dm = np.abs(np.asarray(a.moment_body) - np.asarray(b.moment_body))[ok]
    dm = dm / (qs * params.b)[:, None]
    assert df.max() <= 1e-3 and dm.max() <= 1e-3, (df.max(), dm.max())


@pytest.mark.parametrize(
    "scenario_id",
    [
        pytest.param(s, marks=() if s in FAST_GOLDENS else pytest.mark.full)
        for s in GOLDEN_SCENARIOS
    ],
)
def test_v16_golden_flights_with_tabulated_aero(scenario_id: str) -> None:
    def tab(r):
        return dataclasses.replace(r, airframe=tabulated(r.airframe))

    problems = compare(GOLDENS[scenario_id], metrics(scenario_id, transform=tab))
    assert not problems, "\n".join(problems)


def _ref_with_tables(ref_params: ref.Params, t: AeroTables) -> ref.Params:
    return {**ref_params, "tables": {k: np.asarray(v) for k, v in t._asdict().items()}}


def test_v17_tabulated_pointwise_parity(params: AirframeParams, ref_params: ref.Params) -> None:
    for seed in (3, 4):
        t = random_tables(seed)
        p_tab, rp = params._replace(tables=t), _ref_with_tables(ref_params, t)
        xs, us, ws = _sample(500, seed=seed)
        fm = jax.jit(
            jax.vmap(
                lambda x, u, w, p=p_tab: forces_moments(
                    p, BM_PARITY, state_from_vector(x), controls_from_vector(u), w
                )
            )
        )(xs, us, ws)
        ref_fm = [ref.forces_moments(rp, x, u, w) for x, u, w in zip(xs, us, ws, strict=True)]
        np.testing.assert_allclose(
            fm.force_body, np.array([f for f, _ in ref_fm]), rtol=1e-10, atol=1e-9
        )
        np.testing.assert_allclose(
            fm.moment_body, np.array([m for _, m in ref_fm]), rtol=1e-10, atol=1e-9
        )


def test_v17_tabulated_trajectory_parity(params: AirframeParams, ref_params: ref.Params) -> None:
    t = bm_tables(params)
    p_tab, rp = params._replace(tables=t), _ref_with_tables(ref_params, t)
    n = round(20.0 / DT)
    u, w = _smooth_inputs(n, seed=5)
    x0 = np.array([0.0, 0.0, -100.0, 25.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    traj = jax.jit(
        lambda x0, us, ws: rollout(
            p_tab, BM_PARITY, state_from_vector(x0), controls_from_vector(us), ws, DT
        )
    )(jnp.asarray(x0), u, w)
    xs_ref = ref.simulate(rp, x0, u, w, DT)
    np.testing.assert_allclose(np.asarray(state_to_vector(traj)), xs_ref, rtol=1e-9, atol=1e-8)


def test_v17_ground_effect_parity(params: AirframeParams, ref_params: ref.Params) -> None:
    """Ground effect (tabulated airframe, wing 0-6 m above the ground) against the reference."""
    t = random_tables(9)
    p_tab = params._replace(tables=t, ground_effect=1.0)
    rp = {**_ref_with_tables(ref_params, t), "ground_effect": 1.0}
    xs, us, ws = _sample(300, seed=10)
    xs[:, 2] = -np.random.default_rng(11).uniform(0.0, 6.0, len(xs))
    fm = jax.jit(
        jax.vmap(
            lambda x, u, w: forces_moments(
                p_tab, BM_PARITY, state_from_vector(x), controls_from_vector(u), w
            )
        )
    )(xs, us, ws)
    ref_f = np.array(
        [ref.forces_moments(rp, x, u, w)[0] for x, u, w in zip(xs, us, ws, strict=True)]
    )
    np.testing.assert_allclose(fm.force_body, ref_f, rtol=1e-10, atol=1e-9)
    # And it matters: the same states without ground effect have more drag.
    p_off = p_tab._replace(ground_effect=0.0)
    off = jax.jit(
        jax.vmap(
            lambda x, u, w: forces_moments(
                p_off, BM_PARITY, state_from_vector(x), controls_from_vector(u), w
            )
        )
    )(xs, us, ws)
    assert float(jnp.max(jnp.abs(off.force_body - fm.force_body))) > 0.1
