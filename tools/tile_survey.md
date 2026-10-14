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
2. Record the **ground height (ellipsoidal)** shown in the panel. It goes into the site config as `h_ellipsoid_m`.
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
| South Downs, Devil's Dyke | | | | | | | |
| Cambridge west fringe, Coton | | | | | | | |
| Bath north fringe, Lansdown | | | | | | | |
| Peak District edge, Hope Valley | | | | | | | |

**Recommendation from the spec:** if you know which club field you would fly from in months 9–12, add it as a candidate. A simulated twin of the real test site makes the strongest story.
