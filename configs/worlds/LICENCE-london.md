# The London world: licence and attribution

`london.world.npz`, `london.yaml` and the operations block of `../sites/london.yaml` make up a database derived from two sources (docs/08). The code that made them, `tools/london_fetch.py` and `tools/london_bake.py`, is under the project's Apache-2.0 licence. The derived database is not: it is under the **Open Database License 1.0** (ODbL), as its OpenStreetMap source requires.

- **OpenStreetMap.** © OpenStreetMap contributors. Building footprints and parts, heights, roof shapes and colours, roads, water, parks and land use, and names. Licensed under the ODbL 1.0: https://opendatacommons.org/licenses/odbl/1-0/ and https://www.openstreetmap.org/copyright.
- **Environment Agency LiDAR.** LIDAR Composite DSM (last return) and DTM, 1 m. © Environment Agency copyright and/or database right. Licensed under the Open Government Licence v3.0: https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/. The ground, the measured obstacle grid, building heights checked against it, and the trees come from it.

Anyone may use, share and adapt this database under the ODbL: attribute both sources as above, and share an adapted database under the ODbL too.

The source files themselves (`data/london/`, fetched by `tools/london_fetch.py`) are not in the repository. `data/london/manifest.json` records where each came from, its SHA-256 and when it was fetched.
