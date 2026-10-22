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
