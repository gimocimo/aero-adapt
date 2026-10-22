# 08. London: a real city to fly through

**Status:** proposed, 22 Oct 2026. The owner chose the area (the City and Bankside) and the vehicle (a small fixed-wing). Nothing is downloaded or built yet.

## 0. Why

The test map (docs/07 §14) is invented. The second world is real: central London, reproduced from measured data, small enough to curate, and dense enough that a small UAV flies between buildings rather than over fields. It is a demo, and a harder test of everything the stack does near obstacles.

## 1. The area

A box on the National Grid (EPSG:27700), 2.8 × 2.2 km:

- **Easting** 531,300 to 534,100; **northing** 179,700 to 181,900.
- **In latitude and longitude** (WGS84, to a few metres): 51.5010 to 51.5201 N, 0.1098 to 0.0686 W.
- **West:** Blackfriars Bridge. **East:** the Tower, Tower Bridge and St Katharine Docks. **North:** Liverpool Street. **South:** Bankside, Borough Market, London Bridge station and the Shard.
- **Inside:** St Paul's, the City's towers (22 Bishopsgate, the Gherkin, the Scalpel, the Walkie-Talkie, the Leadenhall Building), Tate Modern, the Monument, the river crossings from Blackfriars to London Bridge, and Tower Bridge.

The world's frame is local metres from the box's centre (E 532,700, N 180,800), turned about 1.5° from the grid so that its north is true north.

## 2. Data

All of it is open. None of it is committed raw: it lives in `data/london/`, which is git-ignored.

| What | Source | Licence | Download |
|---|---|---|---|
| Surface heights: roofs, trees, bridges | Environment Agency, LIDAR Composite DSM, 1 m, last return | Open Government Licence | WCS `GetCoverage` for the box only: one GeoTIFF, 2,800 × 2,200 float32, about 25 MB |
| Ground heights | Environment Agency, LIDAR Composite DTM, 1 m | Open Government Licence | the same, about 25 MB |
| Footprints, building parts, roads, the river, parks, trees, bridges | OpenStreetMap, through the Overpass API | ODbL | one query for the box with a 50 m margin: about 25,000 features (some 5,100 buildings and 3,100 building parts), about 20 MB of JSON |

The LiDAR comes from the Environment Agency's WCS services, so it arrives already cut to the box on its native 1 m grid, with no 5 km tiles to stitch. Its vertical accuracy is ±15 cm (RMSE) per survey. The composite mixes surveys from different years, so recent towers may be missing or half-built.

## 3. The pipeline

Two tools, as for the test map:

- **`tools/london_fetch.py`** fetches the three files above into `data/london/`, and refuses to fetch anything else.
- **`tools/london_bake.py`** writes `configs/worlds/london.world.npz`, `configs/worlds/london.yaml` and the site `configs/sites/london.yaml`:
  - **Frame.** OSM's latitudes and longitudes go to the National Grid with a Helmert shift and the Transverse Mercator (good to about 5 m). The rest is measured: the building footprints are slid over the LiDAR's roofs to the offset where they match best, so OSM and LiDAR agree to well under a metre.
  - **Terrain.** The DTM, resampled to a 2 m grid.
  - **Buildings.** Each OSM footprint, or building part where a building has them, becomes a prism up to a high percentile of the LiDAR's heights inside it. The roof is flat or pitched by how the heights spread inside the footprint. OSM's own heights are used only to check.
  - **Trees.** From the canopy: LiDAR height above the ground, outside buildings, with a crown at each local top.
  - **The rest.** The Thames as water. Roads by class, with their widths. Parks, churchyards and squares as land cover.
- **Landmarks.** St Paul's dome, Tower Bridge's towers and the Shard come out of the building parts as stacks of prisms. Hand-made models, if wanted, come later and are drawn only (§7).

## 4. The simulator

- **A 1 m obstacle grid**, set per world (it is 2 m everywhere now). At 1 m, the box is 6.2 million cells of uint16, 12 MB.
- **What the aircraft can hit is the measured surface:** the greater of the LiDAR (surface minus ground) and the baked buildings. That includes trees, street furniture tall enough to show, and whatever cranes stood on the day of the survey.
- **Bridges are solid down to the water.** The grid holds one height per cell, so nothing can be flown under: not the bridges, not Tower Bridge's walkways. That is conservative; flying under them is not the point.

## 5. The vehicle

A new airframe: a small fixed-wing of about 1 m span and 1 kg, cruising at about 12 m/s.

- **Turning.** At 45° of bank, it turns on a radius of about 15 m.
- **What it can fly:** above the roofs; along the river (about 250 m wide); straight along the wider streets; through the gaps in the City's cluster of towers. It turns at junctions and open spaces.
- **What it cannot:** weave down narrow streets; hover. That needs a multirotor, which is a different project.

## 6. Operations (to decide)

- **Launch and recovery.** Both are fictional: a fixed-wing needs an open place to launch from and come down on, and the City has very few. The candidates are Potters Fields Park, by Tower Bridge, and a rooftop.
- **Contingency.** The Thames is the only large open space. A landing on water is not modelled (docs/07 §10), so for now a forced landing there counts as a ditching.
- **No-fly zones.** As on the test map, written by the bake, and invented for the exercise (round the Tower and St Paul's, for example). Real drone flights in central London need permissions this world does not model.

## 7. Skyview

- London in the cinematic look (docs/07 §15). Facades follow OSM's building type and height: offices and glass for the towers, stone for churches and listed buildings, homes and shops elsewhere. OSM's colours are used where it has them.
- `?map=real` already shows Google's photorealistic 3D tiles over the same box, for comparison. Those may be shown, but their licence does not allow copying their geometry into the simulator.

## 8. Fidelity

**Measured:**

- the ground;
- every building's footprint and height;
- where the roofs and trees are, to a metre across and about 15 cm in height.

**Not measured:**

- facades: windows and stone are patterns;
- roof detail finer than a metre;
- buildings put up after their survey;
- airflow between buildings (no channelling, no downdraughts);
- GPS among tall buildings (no multipath yet; the fault lab could add it).

The world is a model of London on its survey dates, not a live one.

## 9. Licence

The code stays Apache-2.0. The baked world is a database derived from OpenStreetMap, so if it is committed it carries its own licence: ODbL, with "© OpenStreetMap contributors" and the Environment Agency's attribution, in a notice beside it (as `ardupilot/` will carry GPL-3.0, ADR-006).

## 10. Steps

1. Fetch the data (needs the owner's go-ahead).
2. Bake the terrain, buildings and river; look at it in Skyview against Google's tiles.
3. The 1 m obstacle grid, and the simulator on it.
4. The small airframe.
5. Operations and a first scenario: a tour down the river and round the City's towers.
6. Landmarks and finish.
