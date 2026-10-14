# ADR-001: Frames, units and sign conventions

**Status:** Accepted, 14 Oct 2026

## Context

Sign and frame errors are the most common source of simulator bugs. They are also the hardest to see, because a wrong sign often still produces plausible-looking flight. AeroAdapt spans a JAX simulator, a flight stack, ArduPilot (later) and a CesiumJS viewer. Each of these has its own native conventions.

## Decision

| Item | Convention |
|---|---|
| Units | SI internally. Angles are radians internally; degrees appear only in YAML/UI fields with a `_deg` suffix. |
| Time | `time_us`: integer microseconds since scenario start (MAVLink-style). Physics uses float seconds. |
| World frame | NED local tangent plane at the site origin. Flat-earth dynamics are valid within a 10 km radius. |
| Geodetic | WGS84 with **ellipsoidal** heights. Conversions use exact ECEF in **float64 NumPy** (`aeroadapt.core.geodesy`). |
| Body frame | FRD (forward, right, down) |
| Attitude | Hamilton quaternion, scalar-first `[e0, e1, e2, e3]`, body → NED. Euler angles are ZYX (ψ, θ, φ), used for display and controllers only. |
| Wind | `wind_ned` is the air-mass velocity (the direction the air moves *towards*). Aviation "from" direction is converted only at the boundary. |
| Surface deflections (simulator) | Beard & McLain signs: δe > 0 is trailing edge down, so C_m,δe < 0. δa > 0 gives a positive rolling moment. Throttle δt ∈ [0, 1]. |
| Commands (stack ↔ vehicle) | Normalised ArduPilot servo functions with *aircraft-response* signs: `aileron` +1 = roll right, `elevator` +1 = nose up, `rudder` +1 = nose right. A per-airframe ServoMap converts these to physical deflections. |
| Propeller | Rotates positively about +x_body. The airframe feels the reaction torque −Q_p. |
| Viewer (CesiumJS) | Model frame FLU (Cesium's glTF correction maps glTF +Z forward / +Y up to +X / +Z). Local world frame ENU. The single conversion is `web/skyview/src/cesiumOrientation.ts`. |

**Two cross-checks keep these conventions honest:**

- **Python ↔ TypeScript:** `tools/export_frame_goldens.py` writes goldens from `aeroadapt.core`, and `web/skyview/src/frames.test.ts` checks the TypeScript implementation against them.
- **Visual calibration:** the nav-light calibration model (`tools/make_gltf.py`: red port, green starboard, yellow nose, blue fin) is checked visually at headings 0/90/180/270°, pitch ±30° and roll ±45°.

## Consequences

- **Float64 for absolute positions.** The JAX simulator never handles latitude/longitude, because in float32 ECEF positions quantise to about 0.5 m. Sensor models produce GNSS positions in local NED, and the imperative shell converts them to geodetic coordinates in float64.
- **UK height gotcha.** Google 3D Tiles use ellipsoidal heights, and the UK geoid is about 50 m above the ellipsoid. Site configs therefore store the ground's ellipsoidal height, sampled from the tiles.
- **One place to fix the viewer.** Any viewer orientation bug has a single place to look and a test that pins it down.
