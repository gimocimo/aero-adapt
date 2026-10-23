"""Synthetic worlds (docs/07): generation, the obstacle raster, the safe-zone survey, the
world site, and the simulator's terrain lookups agreeing with the world."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
import shapely

from aeroadapt.scenarios.resolve import site_terrain
from aeroadapt.scenarios.site import load_site
from aeroadapt.sim.aerojax.terrain import flat_terrain, ground_height, obstacle_height
from aeroadapt.world import World, generate, load_world, load_world_spec
from aeroadapt.world.generate import ZONE_INSET_M
from aeroadapt.world.places import what_is_at, world_gazetteer
from aeroadapt.world.survey import APPROACH_CLEARANCE_M, APPROACH_SLOPE


@pytest.fixture(scope="module")
def world() -> World:
    return load_world("wrenfold")


def test_generation_is_deterministic(world: World) -> None:
    again = generate(load_world_spec("wrenfold"))
    np.testing.assert_array_equal(again.dtm.h, world.dtm.h)
    np.testing.assert_array_equal(again.trees.n, world.trees.n)
    np.testing.assert_array_equal(again.trees.height, world.trees.height)
    assert [b.footprint().wkt for b in again.buildings] == [
        b.footprint().wkt for b in world.buildings
    ]
    assert [h.line.wkt for h in again.hedges] == [h.line.wkt for h in world.hedges]
    np.testing.assert_array_equal(again.obstacles().agl, world.obstacles().agl)


def test_a_country_valley(world: World) -> None:
    """The ground at home is the datum; the relief and the features are what docs/07 says."""
    assert float(world.dtm.at(0.0, 0.0)) == pytest.approx(0.0, abs=1e-9)
    assert 35.0 < float(world.dtm.h.max()) < 50.0
    assert -6.0 < float(world.dtm.h.min()) < 0.0  # the river's bed
    assert len(world.buildings) > 60  # the village's houses, the church, the farms
    assert len(world.trees) > 20_000
    assert sum(p.kind == "field" for p in world.plots) > 100
    assert len(world.hedges) > 1000


def test_landing_grounds_are_level(world: World) -> None:
    for plot in world.zone_plots():
        inner = plot.polygon.buffer(-5.0)
        x0, y0, x1, y1 = inner.bounds
        nn, ee = np.meshgrid(np.arange(x0, x1, 10.0), np.arange(y0, y1, 10.0), indexing="ij")
        inside = shapely.contains_xy(inner, nn, ee)
        h = world.dtm.at(nn[inside], ee[inside])
        assert float(h.max() - h.min()) < 0.05, plot.zone


def test_obstacle_raster_covers_every_feature(world: World) -> None:
    """Conservative: every point of a feature lies in a cell at least as tall as it."""
    obs = world.obstacles()
    tol = obs.quantum + 0.6  # rounding, and the ground's slope across a cell
    t = world.trees
    lone = t.wood < 0
    top = t.base[lone] + t.height[lone] - world.dtm.at(t.n[lone], t.e[lone])
    assert np.all(obs.at(t.n[lone], t.e[lone]) >= top - tol)
    for h in world.hedges[::7]:
        p = h.line.interpolate(0.5, normalized=True)
        assert float(obs.at(p.x, p.y)) >= h.height - tol
    for b in world.buildings:
        assert float(obs.at(b.n, b.e)) >= b.base + b.ridge - float(world.dtm.at(b.n, b.e)) - tol
    tower = next(b for b in world.buildings if b.kind == "tower")
    assert float(obs.at(tower.n, tower.e)) == pytest.approx(30.0, abs=1.0)
    for w in world.woods:
        c = w.polygon.representative_point()
        assert float(obs.at(c.x, c.y)) >= w.canopy - tol
    # … and open fields are open.
    for plot in world.zone_plots():
        c = plot.polygon.centroid
        assert float(obs.at(c.x, c.y)) == 0.0


def test_the_survey_keeps_every_approach_clear(world: World) -> None:
    """For each zone and landing direction, the approach surface from the displaced threshold
    passes at least the clearance above everything within the corridor's core."""
    obs = world.obstacles()
    zones = world.survey()
    assert [z["id"] for z in zones] == ["S", "A", "B", "C", "D", "E", "F", "G", "H"]
    from aeroadapt.core.polygons import principal_chords

    for z in zones:
        centre, axis, hl, _ = principal_chords(np.asarray(z["polygon_ne_m"]))
        d1 = np.array([math.cos(axis), math.sin(axis)])
        for sign, thr in zip((1.0, -1.0), z["threshold_back_m"], strict=True):
            d = sign * d1
            if thr is None:
                continue
            s = np.arange(-hl - 600.0, -thr, 2.0)  # before the threshold, along d
            p = centre[None, :] + s[:, None] * d[None, :]
            top = world.dtm.at(p[:, 0], p[:, 1]) + obs.at(p[:, 0], p[:, 1]) - z["elevation_m"]
            surface = (-thr - s) / APPROACH_SLOPE
            stands = top > 0.25
            assert np.all(surface[stands] >= top[stands] + APPROACH_CLEARANCE_M - 0.3), z["id"]
        # The zone is its field inset by the hedge margin.
        assert hl > 120.0
    assert ZONE_INSET_M == 15.0


def test_world_site_takes_its_zones_from_the_survey() -> None:
    site = load_site("wrenfold_vale")
    assert site.world == "wrenfold" and len(site.safe_zones) == 9
    a = next(z for z in site.safe_zones if z.id == "A")
    assert "Mill Meadow" in a.surface and a.elevation_m > 0.0
    assert a.threshold_back_m is not None


def test_gazetteer_entries(world: World) -> None:
    g = world_gazetteer(world)
    names = {f["name"] for f in g["features"]}
    assert {"Wrenfold", "St Michael's Church", "Hanger Wood", "Hollow Farm", "the radio mast",
            "Mill Lane bridge"} <= names  # fmt: skip
    church = next(f for f in g["features"] if f["kind"] == "church")
    assert church["height_m"] == pytest.approx(30.0, abs=1.5)
    assert {a["id"] for a in g["areas"]} == {
        "settlement-wrenfold", "yard-hollow-farm", "yard-ridge-farm", "structure-mast"
    }  # fmt: skip


def test_what_is_at(world: World) -> None:
    tower = next(b for b in world.buildings if b.kind == "tower")
    assert what_is_at(world, tower.n, tower.e).startswith("St Michael's Church")
    wood = next(w for w in world.woods if w.id == "hanger-wood").polygon.representative_point()
    assert what_is_at(world, wood.x, wood.y) == "Hanger Wood (trees)"
    assert what_is_at(world, 0.0, 0.0) == "Wrenfold airstrip (zone S)"


def test_simulator_lookups_match_the_world(world: World) -> None:
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1550.0, 1550.0, (2000, 2))
    t = site_terrain(load_site("wrenfold_vale"))
    assert t is not None
    g = np.asarray(ground_height(t, jnp.asarray(pts)))
    np.testing.assert_allclose(g, world.dtm.at(pts[:, 0], pts[:, 1]), atol=1e-3)
    o = np.asarray(obstacle_height(t, jnp.asarray(pts)))
    np.testing.assert_allclose(o, world.obstacles().at(pts[:, 0], pts[:, 1]), atol=1e-6)


def test_flat_terrain_is_exactly_zero() -> None:
    t = flat_terrain()
    pts = jnp.asarray([[0.0, 0.0], [1234.5, -987.6], [-1e4, 3e3]], jnp.float32)
    assert np.all(np.asarray(ground_height(t, pts)) == 0.0)
    assert np.all(np.asarray(obstacle_height(t, pts)) == 0.0)


def test_the_test_map_baked() -> None:
    """The owner's test map, baked from the locked sketch (docs/07 §14): read whole, with the
    tall things the simulator must see (taller than the old 8-bit raster could hold)."""
    w = load_world("testmap")
    assert w.spec.baked == "testmap.world.npz"
    assert w.bounds == (-2500.0, -2500.0, 2500.0, 2500.0)
    assert float(w.dtm.at(0.0, 0.0)) == pytest.approx(
        0.0, abs=1e-6
    )  # the datum: the ground at the centre
    assert len(w.buildings) > 5000 and len(w.trees) > 90_000 and len(w.wires) > 50
    kinds = {s.kind for s in w.structures}
    assert {"turbine", "pylon", "cooling_tower", "mast"} <= kinds
    obs = w.obstacles()
    assert obs.agl.dtype == np.uint16
    spire = next(b for b in w.buildings if b.name == "The Spire")
    assert float(obs.at(spire.n, spire.e)) > 240.0  # 250 m to its roof
    turbine = next(s for s in w.structures if s.kind == "turbine")
    # The rotor is a wall across the wind up to the blade tips (through the tower), the
    # nacelle's length off it is clear.
    assert float(obs.at(turbine.n, turbine.e)) == pytest.approx(turbine.height, abs=0.5)
    across = np.array([-math.sin(turbine.heading), math.cos(turbine.heading)]) * turbine.rotor * 0.4
    assert float(obs.at(turbine.n + across[0], turbine.e + across[1])) > turbine.height - 15.0
    along = np.array([math.cos(turbine.heading), math.sin(turbine.heading)]) * 6.0
    assert float(obs.at(turbine.n + along[0], turbine.e + along[1])) < 1.0
    mast = next(s for s in w.structures if s.kind == "mast")
    # (The summit falls away a little within the mast's 2 m cell.)
    assert mast.height <= float(obs.at(mast.n, mast.e)) < mast.height + 3.0
    site = load_site("testmap")
    assert [z.id for z in site.safe_zones][:2] == ["A", "L1"] and len(site.safe_zones) == 23
    ops = site.operations
    assert (
        ops is not None
        and ops.ceiling_alt_m == 500.0
        and ops.home_ne_m == pytest.approx(site.safe_zones[0].polygon_ne_m[0], abs=400)
    )
    centre = next(a for a in ops.no_fly if a.name.startswith("city centre"))
    assert centre.ceiling_m is None  # closed at all heights
    cylinders = [a for a in ops.no_fly if a.name == "wind turbines"]
    assert len(cylinders) == 22 and all(a.ceiling_from == "centre" for a in cylinders)


def test_footprint_buildings_and_measured_obstacles(tmp_path) -> None:
    """A world from survey data (docs/08): buildings on their own footprints (flat, skillion
    and domed roofs, a part clear of the ground), obstacles measured on the ground merged in
    at their own cell size, and both through a baked file and back."""
    from aeroadapt.world.baked import ROOFS, load_baked, save_baked
    from aeroadapt.world.generate import Dtm
    from aeroadapt.world.raster import QUANTUM_M, Obstacles

    spec = load_world_spec("testmap").model_copy(
        update={"bounds_ne_m": ((-50.0, -50.0), (50.0, 50.0)), "baked": None}
    )
    dtm = Dtm(-50.0, -50.0, 2.0, np.zeros((51, 51)))
    square = [[-10.0, -10.0], [-10.0, 10.0], [10.0, 10.0], [10.0, -10.0]]
    shifted = [[n + 30.0, e] for n, e in square]
    measured = np.zeros((100, 100), np.uint16)
    measured[0, 0] = int(7.0 / QUANTUM_M)  # a crane in the south-west corner
    rows = [
        # flat, 30 m; a skillion falling to the north from 40 m to 10 m; a dome
        [0, 0, 20, 20, 0, 0, 30, 30, ROOFS.index("flat"), 0],
        [30, 0, 20, 20, 0, 0, 10, 40, ROOFS.index("skillion"), 0],
        [0, 30, 20, 20, 0, 0, 20, 35, ROOFS.index("dome"), 0],
    ]
    domed = [[n, e + 30.0] for n, e in square]
    meta = {
        "plots": [], "woods": [], "roads": [], "structures": [], "wires": [], "zones": [],
        "places": [], "river": None,
        "buildings": [
            {"kind": "modern", "name": None, "colours": None, "outline": square, "holes": [],
             "bottom": 0.0},
            {"kind": "tower", "name": None, "colours": None, "outline": shifted, "holes": [],
             "bottom": 0.0, "slope": 0.0},
            {"kind": "landmark", "name": None, "colours": None, "outline": domed, "holes": [],
             "bottom": 5.0},
        ],
    }  # fmt: skip
    path = save_baked(
        tmp_path / "survey.world.npz",
        dtm=dtm,
        trees=np.zeros((0, 6)),
        hedges=np.zeros((0, 6)),
        buildings=np.asarray(rows, float),
        meta=meta,
        measured=Obstacles(-49.5, -49.5, 1.0, QUANTUM_M, measured),
    )
    w = load_baked(spec.model_copy(update={"baked": str(path)}))
    flat, skillion, dome = w.buildings
    assert flat.outline is not None and flat.footprint().area == pytest.approx(400.0)
    assert skillion.roof == "skillion" and dome.roof == "dome" and dome.bottom == 5.0
    obs = w.obstacles()
    assert obs.cell == 1.0 and obs.agl.shape == (100, 100)
    assert float(obs.at(0.0, 0.0)) == pytest.approx(30.0)
    assert float(obs.at(-49.5, -49.5)) == pytest.approx(7.0)  # measured, merged in
    # The skillion: high at its southern edge, low at its northern one (conservatively).
    assert float(obs.at(21.0, 0.0)) > 37.0 and 10.0 <= float(obs.at(39.5, 0.0)) < 13.0
    # The dome: its top over the centre, falling to its eaves at the footprint's corners.
    assert float(obs.at(0.0, 30.0)) == pytest.approx(35.0, abs=0.5)
    assert 20.0 <= float(obs.at(9.5, 39.5)) < 30.0
    assert float(obs.at(0.0, 15.0)) == 0.0  # between the buildings


def test_london_baked() -> None:
    """London (docs/08): the City and Bankside from LiDAR and OpenStreetMap, the obstacles
    measured on a 1 m grid, the landmarks at their heights, home at Potters Fields Park."""
    w = load_world("london")
    assert w.bounds == (-1040.0, -1360.0, 1040.0, 1360.0)
    assert float(w.dtm.at(0.0, 0.0)) == pytest.approx(0.0, abs=1e-6)
    assert len(w.buildings) > 6000 and len(w.trees) > 5000 and len(w.roads) > 3000
    assert sum(b.outline is not None for b in w.buildings) > 5500
    assert w.measured is not None and w.measured.cell == 1.0
    obs = w.obstacles()
    assert obs.cell == 1.0 and obs.agl.shape == (2080, 2720) and obs.agl.dtype == np.uint16
    places = {p.id: p for p in w.spec.places}
    assert {"the-shard", "st-pauls", "tower-bridge", "the-gherkin", "the-thames"} <= set(places)
    # The Shard: 310 m in OSM, as tall in the drawing; the LiDAR sees its top near 290 m.
    shard = max(w.buildings, key=lambda b: b.ridge)
    assert shard.ridge == pytest.approx(310.0, abs=1.0)  # above its base
    sn, se = places["the-shard"].point_ne_m
    near = obs.agl[
        int(sn - obs.n0) - 30 : int(sn - obs.n0) + 30, int(se - obs.e0) - 30 : int(se - obs.e0) + 30
    ]
    assert 270.0 < float(near.max()) * obs.quantum < 320.0
    # The Thames is level, and open water has nothing on it.
    tn, te = places["the-thames"].point_ne_m
    assert float(obs.at(tn, te)) < 2.0
    # Railways: viaducts into London Bridge, bridges over the river, the stations' roofs.
    carried = np.concatenate([r.carried for r in w.rails])
    assert len(w.rails) > 100 and set(np.unique(carried)) == {0, 1, 2, 3}
    deck = np.concatenate([r.points[r.carried == 1] for r in w.rails])
    assert 5.0 < float(np.median(deck[:, 2] - w.dtm.at(deck[:, 0], deck[:, 1]))) < 15.0
    site = load_site("london")
    assert [z.id for z in site.safe_zones] == ["A"]
    ops = site.operations
    assert ops is not None and ops.ceiling_alt_m == 400.0
    assert {a.id for a in ops.no_fly} == {"tower-of-london", "st-pauls"}
