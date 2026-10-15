# AeroAdapt-UAV

**An autonomous pilot for fixed-wing UAVs that adapts to aircraft it has never flown, works out what broke when something fails, keeps the aircraft flying, and puts it down where nobody gets hurt.**

AeroAdapt is built simulation-first. It is validated rigorously and compared against strong classical baselines. The aim is to move from simulation to a real aircraft once the simulated system is proven.

> **Status:** Milestone M1 (vertical slice) complete: a fixed-wing UAV flies a mission over the real Coton fields west of Cambridge in the browser, and when its motor fails, a classical contingency glides it to a surveyed field.
>
> - **Simulator:** a JAX 6-DoF fixed-wing simulator with actuators, battery, turbulence, sensors and fault injection. It matches an independent NumPy implementation to machine precision.
> - **Flight stack:** a DCM-class navigator and an ArduPlane-style autopilot (our reimplementation of ArduPlane's cruise-flight control laws). They meet every V11 acceptance criterion in calm air and light turbulence.
> - **Motor-failure contingency** (50 random motor cuts):
>   - **91%** of the cases judged reachable land inside the selected field;
>   - zero crashes;
>   - ESC detection in 0.73 s median.
> - **Recordings and replay:** reproducible replay bundles, played back in **Skyview**, a CesiumJS player over Google Photorealistic 3D Tiles. It shows a HUD with indicated vs true airspeed, live safe-zone reachability, and a ghost aircraft for stock-vs-contingency comparisons.
> - **Evidence:** [validation report](docs/validation/m1_report.md) · [M1 write-up](docs/demo/m1.md), including its [fidelity and limits](docs/demo/m1.md#fidelity-and-limits): a textbook aircraft model, flat simulated ground (the 3D map is visual only), and no comparison with real ArduPlane yet.
>
> Next: M2 (JSBSim backend, live server, the airframe fleet).

![Skyview: the classical contingency flaring over field A after a motor cut](docs/media/skyview_final.jpg)

## Plan and design documents

| Doc | What |
|---|---|
| [01 — State of the art](docs/01_sota_review_and_research_direction.md) | Literature review, open problems, research framing |
| [02 — Vision](docs/02_aeroadapt_vision_architecture_roadmap.md) | Original vision and evaluation philosophy (now superseded by 03) |
| [03 — AeroAdapt-UAV plan](docs/03_uav_architecture_and_six_month_roadmap.md) | **Current plan**: architecture, fleet, benchmarks, six-month roadmap |
| [04 — M1 technical spec](docs/04_m1_technical_spec.md) | Interfaces, models, validation plan and work plan for M1 |
| [ADRs](docs/adr/) | Architecture decision records |

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Node 22 (for the viewer).

```bash
make setup      # Python env (uv) + web dependencies
make check      # lint, type check, import contracts, fast test tier
make validate   # full validation tier (slow)
```

Fly a scenario and replay it in Skyview:

```bash
uv run aeroadapt run m1-motorcut-classical          # → runs/m1-motorcut-classical (replay bundle)
uv run aeroadapt run m1-motorcut-stock
uv run aeroadapt export-viz runs/m1-motorcut-classical
uv run aeroadapt export-viz runs/m1-motorcut-stock
cp web/skyview/.env.example web/skyview/.env.local   # optional: add a Cesium ion token for 3D tiles
make web-dev   # open http://localhost:5173/?replay=m1-motorcut-classical&ghost=m1-motorcut-stock
```

Other commands: `aeroadapt campaign m1-mini` (the 50-cut V12 mini-campaign), `make report` (validation report), `make demo` (demo video from seeds), `aeroadapt bench`.

## Repository layout

| Path | Contents |
|---|---|
| `src/aeroadapt/core` | Frames, quaternions, WGS84 geodesy (ADR-001) |
| `src/aeroadapt/airframes` | Airframe schema and loader |
| `src/aeroadapt/sim/aerojax` | JAX 6-DoF fixed-wing simulator, written as a pure functional core (ADR-002): actuators, battery, atmosphere and turbulence, sensors, trim |
| `src/aeroadapt/faultlab` | Fault specifications, fixed-shape fault slots and ground-truth labels (ADR-003) |
| `src/aeroadapt/scenarios` | Sites, scenario schema and resolver, runner, campaigns, V12 evaluation, golden flights |
| `src/aeroadapt/telemetry` | Recording schema, events and replay bundles (ADR-005) |
| `src/aeroadapt/stack` | Flight stack: navigator, L1, TECS, attitude loops, modes and mission, classical motor-failure contingency, and the `FlightStack` host shell. It runs on sensor data only and must not import the simulator or a vehicle backend (ADR-006). |
| `src/aeroadapt/vehicle` | The `Vehicle` interface (ADR-004) and the `AerojaxVehicle` adapter, with privileged truth access (ADR-006) |
| `src/aeroadapt/harness` | Closed loop (simulator + stack) in one `lax.scan`, vmap-able. V11 acceptance tests. |
| `src/aeroadapt/cli.py` | `aeroadapt run / campaign / validate / report / bench / export-viz` |
| `validation/` | Validation report builder (outside the package: it uses the reference implementation) |
| `reference/bm_numpy` | Independent NumPy reference implementation, used only for N-version validation |
| `configs/` | Airframes, sites, stack parameters, scenarios, campaigns |
| `web/skyview` | Skyview replay player (CesiumJS, React, Arrow) |

## Validation philosophy

Every model is backed by evidence. The simulator is checked three ways:

- **Against an independent reference implementation**, which must agree to machine precision.
- **Against physical invariants and analytic approximations.**
- **Against an independent simulator** (JSBSim, planned for M2).

See the [M1 spec, §12](docs/04_m1_technical_spec.md).

## Licence

Apache-2.0, see [LICENSE](LICENSE). Third-party data attributions are listed in [NOTICE](NOTICE).
