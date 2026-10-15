"""Procedural airframe generator (M2 spec §4): determinism, physical consistency, glTF axes,
batched aerojax filters and the fleet bundle round trip."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aerosandbox")

import trimesh

from aeroadapt.airframes import ALPHA_GRID_N, to_params
from aeroadapt.generator.build import Candidate, build
from aeroadapt.generator.design import CLASSES, sample_design
from aeroadapt.generator.filters import check
from aeroadapt.generator.fleet import build_fleet, load_fleet
from aeroadapt.generator.gltf import airframe_glb


@pytest.fixture(scope="module")
def candidates() -> list[Candidate]:
    return [build(sample_design(cls, 3)) for cls in CLASSES]


def test_designs_are_deterministic_and_in_range() -> None:
    for cls in CLASSES:
        a, b = sample_design(cls, 7), sample_design(cls, 7)
        assert a == b and a != sample_design(cls, 8)
        assert 1.0 <= a.span <= 4.0 and 0.06 <= a.static_margin <= 0.15


def test_candidates_are_complete_and_consistent(candidates: list[Candidate]) -> None:
    for c in candidates:
        af, m = c.airframe, c.meta
        assert af.aero.model == "tabulated" and af.aero.tables is not None
        assert len(af.aero.tables.C_L) == ALPHA_GRID_N
        p = to_params(af)
        assert p.tables is not None and np.all(np.isfinite(p.tables.C_D))
        assert math.isclose(
            sum(v for k, v in m.items() if k.startswith("mass_") and k != "mass_kg"),
            m["mass_kg"],
            rel_tol=1e-9,
        )
        assert m["static_margin"] == pytest.approx(c.design.static_margin, abs=1e-9)
        assert 0.6 < m["cl_max"] < 2.2 and m["v_stall"] < m["v_cruise"]
        # Control derivatives have the signs of a conventional aircraft's response.
        lo = af.aero.longitudinal
        assert lo.C_L_alpha > 3.0 and lo.C_m_alpha < 0.0 and lo.C_m_q < 0.0


def test_gltf_is_in_the_viewer_frame(candidates: list[Candidate]) -> None:
    """glTF axes: +Z forward, +Y up, +X left (the calibration model's convention)."""
    c = next(c for c in candidates if c.design.cls == "conventional")
    glb = airframe_glb(c.design, np.array([c.meta["x_cg_m"], 0.0, 0.0]))
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    assert isinstance(scene, trimesh.Scene)
    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    v = np.concatenate([m.vertices for m in meshes])  # nodes have identity transforms
    span_axis = np.ptp(v, axis=0).argmax()
    assert span_axis == 0  # the wing spans X
    top = v[v[:, 1].argmax()]
    assert top[2] < 0.0  # the fin tip is aft of the CG
    nose = v[v[:, 2].argmax()]
    assert nose[2] > 0.1 * c.design.span * c.design.fuselage_length_ratio


def test_filters_agree_with_the_generator(candidates: list[Candidate]) -> None:
    r = check([c.airframe for c in candidates], [c.meta for c in candidates])
    assert r.accepted.dtype == bool and len(r.reasons) == len(candidates)
    sm = r.metrics["static_margin_aerojax"]
    target = np.array([c.meta["static_margin"] for c in candidates])
    # aerojax's static margin from the tables matches the generator's placement.
    np.testing.assert_allclose(sm, target, atol=0.03)
    assert np.all(r.metrics["trim_throttle_cruise"] > 0.2)


def test_fleet_bundle_round_trip(tmp_path: Path) -> None:
    out = build_fleet(
        "t", 4, start=100, workers=2, batch=10, out_root=tmp_path, progress=lambda _: None
    )
    fleet = load_fleet(out)
    assert len(fleet) == 4 and len({a.id for a in fleet}) == 4
    assert all((out / "gltf" / f"{a.id}.glb").exists() for a in fleet)
    # V19: every stored airframe (float32 tables) passes the filters again.
    import pyarrow.parquet as pq

    rows = {r["id"]: r for r in pq.read_table(out / "designs.parquet").to_pylist()}
    r = check(fleet, [rows[a.id] for a in fleet])
    assert r.accepted.all(), r.reasons
