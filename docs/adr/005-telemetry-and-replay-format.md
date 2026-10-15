# ADR-005: Telemetry and replay format

*Status: Accepted, implemented (M1 week 4) · 15 Oct 2026*

## Context

Every flight, whether a single scenario, a campaign member, the demo, or (from M2) a live session,
must be:

- replayable in the browser (Skyview);
- analysable offline (metrics, reports, learning data);
- reproducible (V13).

The browser needs a compact, typed subset at display rate. Analysis needs everything at the control
rate, including truth, sensors and estimates side by side. Fault ground truth must travel with every
recording (D4), and the stack must never see it (ADR-006).

## Decision

A flight is recorded as a **replay bundle**: a directory with four files.

| File | Format | Contents |
|---|---|---|
| `manifest.json` | JSON, sorted keys | Schema version; the fully resolved scenario; the site (origin, safe zones); mission; provenance (airframe and stack-config SHA-256, stack config, git commit and dirty flag); seed; rates; per-column units; enums (modes, phases, fault bits); the list of viz columns |
| `frames.parquet` | Parquet, zstd, float64 | One row per control tick (100 Hz): `time_us`, `truth.*`, `actual.*`, `sensors.*`, `est.*`, `cmd.*`, `demands.*`, `mode`, `wp_idx`, `cont.*`, `labels.fault_bits`, `done`, `termination` |
| `events.jsonl` | JSON Lines | `{time_us, type, payload}`: fault onsets, detection, zone selection, contingency phases, modes, waypoints, airspeed health, touchdown (with class and in-zone flag), termination |
| `viz.arrow` | Arrow IPC file, uncompressed, float32 / int32 | The browser subset at 50 Hz (~40 columns), always including the final row |

Details:

- **One schema table.** `aeroadapt.telemetry.schema.COLUMNS` defines every column once: its name,
  unit, how it is taken from the closed-loop log, and whether it is in the viz subset. Parquet, Arrow
  and the manifest's units all come from it.
- **Truth and estimates side by side.** The HUD's "indicated vs true airspeed" is
  `est.ias` / `sensors.tas` against `truth.tas`. Fault labels are a bit field (`labels.fault_bits`).
- **Events are derived after the run** from the columns, so the pure flight core carries no host-side
  bookkeeping.
- **Determinism.** No wall-clock time is written anywhere. On one machine and commit, a scenario
  reproduces byte-identical files (V13, tested). Across machines, the golden-flight metrics must agree
  within tolerance.
- **Compression at the transport layer for Arrow.** Arrow JS does not decode compressed IPC buffers
  in every build, so `viz.arrow` is uncompressed and the web server compresses on the wire. A
  10-minute flight is about 4.5 MB of Arrow (about 1 MB gzipped) and loads in well under the 2 s
  budget.
- **Campaigns** keep a 1 Hz trajectory per run plus an outcomes table (`outcomes.parquet`,
  `summary.json`). Recording 100 Hz bundles for every campaign member would cost about 1.5 GB per
  50 flights.

## Consequences

- Skyview reads typed arrays directly (no JSON parsing in the playback path). It interpolates
  positions linearly and attitudes with quaternion slerp between 50 Hz samples.
- Adding a column means one entry in `COLUMNS`. Renaming or removing one bumps
  `BUNDLE_SCHEMA_VERSION`.
- Bundles are large compared with the viz subset (≈9 MB of Parquet per 5 minutes, dominated by
  incompressible sensor noise). They are written to `runs/` (git-ignored). `aeroadapt export-viz`
  copies only the three browser files into Skyview.
- The live protocol (spec §10.4, M2) will stream the same viz columns over a WebSocket, so the viewer
  has one code path for replays and live sessions.
