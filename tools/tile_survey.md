# UK site tile survey (M1 week 1, ~2 h)

**Goal:** choose the UK development site, and later the hero site, by the quality of Google Photorealistic 3D Tiles coverage and by the landing-decision interest of the area (spec §11.1).

## Setup

1. Create a Cesium ion account.
2. Add **Google Photorealistic 3D Tiles** (asset `2275207`) to your account from the Asset Depot.
3. Create an access token.
4. Configure and start the viewer:

   ```bash
   cp web/skyview/.env.example web/skyview/.env.local
   # paste the token into VITE_CESIUM_ION_TOKEN, then:
   make web-dev
   ```

## Procedure (for each candidate in `web/skyview/src/candidates.ts`)

1. Select the site. Wait for the status line to report that the tiles have loaded.
2. Note the **terrain DTM** height (Cesium World Terrain) and the **3D-tiles surface** height.
   - The site config's `h_ellipsoid_m` must match the **rendered tiles surface over open ground**, because that is what the viewer shows. At Coton the terrain model sat about 5 m above the tiles, so a terrain-based value would make landings hover visibly.
   - Put the origin on an open field, not on a building or trees. Survey a grid (as in the Cambridge example below) to confirm the area is flat enough for M1's flat-ground model.
3. Run the **calibration checklist** once, at any one site.
4. Using the chase and free cameras at 120 m AGL, take screenshots towards N, E, S and W.
5. Score the site from 1 to 5 on each criterion below.

| Criterion | What good looks like |
|---|---|
| Tile fidelity | Crisp buildings, trees and field boundaries at 120 m AGL; no melted geometry |
| Landing options | Several distinct fields within about 1.5 km, of varied size and orientation |
| Ground-risk interest | A town edge, a major road and power lines within view, so trade-offs are visible |
| Visual drama | Terrain relief, water, coastline |
| Airspace realism | Outside aerodrome flight restriction zones and restricted areas (check the CAA's current guidance) |

## Results

| Site | Ground h_ell (m) | Fidelity | Landing | Ground risk | Drama | Airspace | Notes |
|---|---|---|---|---|---|---|---|
| South Downs, Devil's Dyke | 249.09 | | | | | | |
| Cambridge west fringe, Coton | 72.00 | | | | | | |
| Bath north fringe, Lansdown | 167.06 (point is below the plateau; refine the coordinate) | | | | | | |
| Peak District edge, Hope Valley | 236.04 | | | | | | |

## Decision (14 Oct 2026): `cambridge_west`

**Chosen for:** the strongest tile fidelity, a mix of housing, trees and several fields, and the most visual interest. The site config is [`configs/sites/cambridge_west.yaml`](../configs/sites/cambridge_west.yaml).

**Origin.** The survey moved the origin from Coton village (a residential street, lon 0.0600) to the large arable field about 1 km east of it (52.2075, 0.0747). This field sits between Coton, the woodland strip, the M11 (~200 m east) and the West Cambridge campus.

**Ground height.** A 150 m grid within ±600 m gives:

| Measurement | Value |
|---|---|
| Tiles surface at the origin | 63.92 m (ellipsoidal) |
| Terrain DTM at the origin | 64.72 m |
| Tiles surface within 300 m | 63.2–64.4 m |
| Spikes (woodland and buildings to the north) | 5–28 m above the field |

**Terrain.** A wider 500 m grid shows the ground rising 20–50 m towards Madingley in the west. M1 safe zones must therefore stay in the flat area east of Coton; M5 adds a DEM.

**Airspace.** About 7.8 km west of Cambridge City Airport and away from its runway extended centrelines, so by our estimate it is outside its flight restriction zone. Bourn airfield lies about 8 km west. Verify on the CAA's official map before any real-world use.

**Recommendation from the spec:** if you know which club field you would fly from in months 9–12, add it as a candidate. A simulated twin of the real test site makes the strongest story.
