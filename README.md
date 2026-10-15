# AeroAdapt-UAV

**An autonomous pilot for fixed-wing UAVs that adapts to aircraft it has never flown, works out what broke when something fails, keeps the aircraft flying, and puts it down where nobody gets hurt.**

AeroAdapt is built simulation-first. It is validated rigorously and compared against strong classical baselines. The aim is to move from simulation to a real aircraft once the simulated system is proven.

> **Status:** Milestone M1 (vertical slice), week 3 done.
>
> - **Simulator:** actuators, battery, ISA and turbulence, sensors, fault injection, trim and validation are in place.
> - **Flight stack:** a DCM-class navigator and an ArduPlane-equivalent autopilot (L1, TECS, attitude loops, AUTO/RTL/GUIDED modes) meet every V11 acceptance criterion, in calm air and in light turbulence.
> - **Motor failure:** the classical contingency detects a motor cut in 0.7 s. It glides a circuit to a pre-surveyed field and lands in it on the hand-made test cases.
>
> Next: scenarios, the runner, the recorder, the Skyview replay and the statistical V12 campaign.

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

Run the Cesium spike viewer:

```bash
cp web/skyview/.env.example web/skyview/.env.local   # then add your Cesium ion token
make web-dev
```

## Repository layout

| Path | Contents |
|---|---|
| `src/aeroadapt/core` | Frames, quaternions, WGS84 geodesy (ADR-001) |
| `src/aeroadapt/airframes` | Airframe schema and loader |
| `src/aeroadapt/sim/aerojax` | JAX 6-DoF fixed-wing simulator, written as a pure functional core (ADR-002): actuators, battery, atmosphere and turbulence, sensors, trim |
| `src/aeroadapt/faultlab` | Fault specifications, fixed-shape fault slots and ground-truth labels (ADR-003) |
| `src/aeroadapt/scenarios` | Site definitions (and, from week 4, scenarios) |
| `src/aeroadapt/stack` | Flight stack: navigator, L1, TECS, attitude loops, modes and mission, classical motor-failure contingency, and the `FlightStack` host shell. It runs on sensor data only and must not import the simulator or a vehicle backend (ADR-006). |
| `src/aeroadapt/vehicle` | The `Vehicle` interface (ADR-004) and the `AerojaxVehicle` adapter, with privileged truth access (ADR-006) |
| `src/aeroadapt/harness` | Closed loop (simulator + stack) in one `lax.scan`, vmap-able. V11 acceptance tests. |
| `reference/bm_numpy` | Independent NumPy reference implementation, used only for N-version validation |
| `configs/` | Airframe, site and scenario definitions |
| `web/skyview` | Browser viewer (CesiumJS) |

## Validation philosophy

Every model is backed by evidence. The simulator is checked three ways:

- **Against an independent reference implementation**, which must agree to machine precision.
- **Against physical invariants and analytic approximations.**
- **Against an independent simulator** (JSBSim).

See the [M1 spec, §12](docs/04_m1_technical_spec.md).

## Licence

Apache-2.0, see [LICENSE](LICENSE). Third-party data attributions are listed in [NOTICE](NOTICE).
