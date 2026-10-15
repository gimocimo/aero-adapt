# M2 Technical Specification: The Fleet

*Status: draft · 15 Oct 2026 · Parent plan: [03 — AeroAdapt-UAV](03_uav_architecture_and_six_month_roadmap.md) §7.2 · Previous milestone: [04 — M1](04_m1_technical_spec.md)*

**Goal of M2.** Replace "one textbook aircraft" with **a fleet of thousands of physically plausible
fixed-wing UAVs** whose aerodynamics are computed from their geometry. The stock autopilot must be
able to fly any of them. The fleet must run batched with ~30 fault types and harsher weather, and
first baseline numbers must come from a 1,000-scenario benchmark. Two items carry over from M1:

- **JSBSim:** a second, independent simulator (V14);
- **live mode:** a server and a browser fault panel, so faults can be injected while you watch.

**Weeks 5–8, about 30 h/week (~120 h).**

**Out of scope for M2** (later milestones):
- RL and any learned component (M3);
- EKF, Sentinel and the Self-model (M4);
- geodata, terrain-aware ground and the ground-risk model (M5);
- ArduPilot SITL (M6);
- ground-roll and crash dynamics (flights still end at ground contact; belly-landing impact metrics
  arrive in M5).

---

## 0. Key findings that change the plan

1. **Fidelity is the honest weak point of M1** ([m1.md, Fidelity and limits](demo/m1.md#fidelity-and-limits)).
   M2 targets the parts that matter most for a small UAV:
   - **Coefficients from geometry.** A vortex-lattice and component-buildup tool computes them, so
     they are not just copied from a textbook.
   - **A realistic stall.** It comes from airfoil section polars, replacing B&M's late (27°) sigmoid
     blend.
   - **Ground effect** (Should).
   - **Propeller slipstream over the tail** (Could).

   CFD in the loop is not an option (10⁷–10⁸× slower). Detailed tools produce *coefficients*; a fast
   6-DoF model flies with them, which is industry practice. To carry generated stall and post-stall
   data, aerojax gets a **tabulated aerodynamic model (aero v2)**. The B&M model becomes one special
   case of it.
2. **The stock autopilot must fly every generated airframe.** Doc 03 assumes it can, but M1's gains
   were hand-tuned for one aircraft. M2 adds **model-based auto-configuration**:
   - trim schedules and loop gains computed from each airframe's model, the way a manufacturer
     configures an ArduPlane before AUTOTUNE;
   - an automated acceptance flight per airframe (V20).

   Airframes it cannot fly acceptably are reported and excluded from the baseline fleet; they are not
   hidden.
3. **Licences.**
   - **The Rascal 110 JSBSim model** lives in the ArduPilot repository (GPL-3.0, derived from
     FlightGear). Following ADR-006, it is **fetched at a pinned commit into a local cache and never
     committed**.
   - **JSBSim** (LGPL-2.1), **AeroSandbox** and **NeuralFoil** (MIT) are optional dependency extras.
     So is CasADi (LGPL-3.0, pulled in by AeroSandbox).
   - **The Skywalker X8 coefficients** come from PyFly (MIT, from Gryte et al. 2018) and are used
     with citation.
4. **No CUDA GPU on the development machine.** The M2 throughput criterion needs one GPU-hour once:
   - the benchmark is packaged as a notebook and a CLI command;
   - CPU numbers are recorded locally on every report;
   - a GPU number is recorded when a GPU run is available (open question 1).

Verified on 15 Oct 2026, on the development Mac (arm64, Python 3.12):
- `jsbsim` 1.3.2 installs from PyPI and ships 60+ aircraft, but not the Rascal.
- AeroSandbox 4.2.10 computes stability derivatives for a conventional layout in 0.06 s (VLM) and
  0.17 s (AeroBuildup).
- NeuralFoil returns post-stall section polars.

---

## 1. Deliverables and Definition of Done

M2 is **done** when all of the following hold.

| # | Criterion | Evidence | Priority |
|---|---|---|---|
| D1 | CI is green on `main` (fast tier, web, Playwright); the nightly full tier is green | GitHub Actions | Must |
| D2 | **Aero v2.** Tabulated aerodynamics in aerojax and in the NumPy reference: N-version parity (V17). The B&M Aerosonde as tables reproduces aero v1 (V16). The M1 golden flights still pass. | V16, V17 | Must |
| D3 | **JSBSim backend.** A `JSBSimVehicle` behind the `Vehicle` interface. V14 passes on the Aerosonde re-expressed in JSBSim XML. | V14 | Must |
| D4 | **Rascal 110 in JSBSim.** The stock stack flies the M1 mission on the held-out Rascal. | Replay + short campaign | Should |
| D5 | **Airframe generator.** 10,000 accepted airframes across four configuration classes, each with a glTF model. Yield, distributions and filter statistics are reported. Aerosonde and X8 derivatives regenerated from geometry fall within published tolerance bands. | V18, V19 | Must |
| D6 | **Auto-configuration.** ≥90% of a 1,000-airframe sample passes the V11-lite acceptance flight under the stock stack. Failures are characterised. | V20 | Must |
| D7 | **faultlab v1.** ≥24 fault types (target 30) plus discrete gusts, wind shear and thermals. Ground-truth labels for all. The five confounded-pair families of doc 03 §3.2 are defined as campaigns. | V21 | Must (24) / Should (30) |
| D8 | **Bench v0.** 1,000 scenarios (fleet × faults × weather) with B1 (stock) and B2 (classical contingency) results, with confidence intervals. One command reproduces it. | V22, report section | Must |
| D9 | **Throughput.** ≥1M environment steps/s on one GPU, with a heterogeneous fleet and faults batched. CPU numbers recorded. | V15 (updated) | Must (CPU) / Should (GPU, depends on open question 1) |
| D10 | **Live mode.** `aeroadapt serve` plus Skyview live mode. A fault panel injects any faultlab fault at any moment; also airframe and scenario picker, pause, speed and reset. Every live session is recorded as a replay bundle. | V23; Playwright test | Must |
| D11 | **Demo and write-up.** "10,000 aircraft": a fleet-view visual, a live fault-injection clip and the bench table. `docs/demo/m2.md`, plus the validation report extended with the M2 sections. | Video + docs | Must |

Priority labels are as in M1:
- **Must:** needed for the Definition of Done.
- **Should:** planned; moves to M3 week 1 if the schedule slips.
- **Could:** only if there is time.

---

## 2. Repository additions

| Path | Contents |
|---|---|
| `src/aeroadapt/sim/aerojax/aero_tables.py` | Aero v2: fixed-grid tables, interpolation, effectiveness factors, ground effect |
| `src/aeroadapt/airframes/` | Schema v2 (`aero.model: bm_analytic \| tabulated`), fleet bundle loader, batched `AirframeParams` |
| `src/aeroadapt/generator/` | **Offline** airframe generator: sampling, geometry, AeroSandbox aero, polars, mass, propulsion, actuators, filters, glTF. *Never imported by the simulator or the stack* (new import-linter contract). |
| `src/aeroadapt/vehicle/jsbsim_vehicle.py` | `JSBSimVehicle` (host-stepped), with the same sensor and fault layer as aerojax |
| `src/aeroadapt/stack/autoconfig.py` | Model-based trim schedules and gains → `StackConfig` |
| `src/aeroadapt/faultlab/` | v1 slots: sensor, actuator, propulsion, energy, airframe, link; environment events |
| `src/aeroadapt/bench/` | Scenario sampler, splits, baselines B1/B2, metrics, intervals |
| `src/aeroadapt/live/` | WebSocket server (msgpack), session recorder, command handling |
| `tools/jsbsim/` | `aerosonde_bm.xml` (ours, Apache-2.0), `fetch_rascal.py` (pinned ArduPilot commit → cache) |
| `reference/bm_numpy/` | Extended with independent table interpolation (V17) |
| `web/skyview/src/live/` | Live client, ring buffer, fault panel, airframe picker, fleet view |

Generated data goes to `runs/` (git-ignored):
- **Fleets:** `runs/fleets/<fleet_id>/` holds `manifest.json`, `airframes.npz` (stacked parameters and
  tables), `meta.parquet` (geometry, class, filter results) and `gltf/<id>.glb`.
- **Committed sample:** a 12-airframe sample fleet, for tests and Skyview fixtures.

Optional extras in `pyproject.toml`:
- `generator`: `aerosandbox`, `neuralfoil`, `trimesh`;
- `jsbsim`: `jsbsim`;
- `live`: `websockets`, `msgpack`.

CI installs `jsbsim` and `live`. The generator runs offline and is tested in CI on the committed
geometry fixtures, with the extra installed.

---

## 3. Aero v2: tabulated aerodynamics

### 3.1 Model

The force and moment structure of B&M ch. 4 is kept (stability axes for lift and drag, body axes for
the rest). The changes are in where the coefficients come from:

| Term | Aero v1 (B&M) | Aero v2 |
|---|---|---|
| Static longitudinal | C_L = blend(linear, flat plate; σ(α)); C_D = parasitic + induced from the *linear* C_L; C_m linear in α | **Tables** C_L(α), C_D(α), C_m(α) on a fixed grid, covering 360° |
| Dynamic and control derivatives | Constants | Constants × an **effectiveness table** η(α) (1 in attached flow, falling past stall) |
| Lateral static | Constants (C_Yβ, C_lβ, C_nβ) | Constants (α-dependence: Could) |
| Ground effect | None | Should: induced-drag and lift factor in h/b (McCormick); a flag per airframe |
| Slipstream on the tail | None | Could: momentum-theory dynamic-pressure increase on elevator and rudder terms |

**Grid.**
- **Range:** a uniform 0.5° grid from −180° to +180° (721 points). Indexing is arithmetic, with no
  search, which matters on a GPU.
- **Memory:** 4 tables × 721 × float32 ≈ 11.5 kB per airframe, so 10,000 airframes is about 115 MB.
- **Shape:** fixed for every airframe, so a heterogeneous fleet `vmap`s without padding.

**The B&M model as tables.** `bm_analytic` stays the default for `aerosonde_bm` and for the reference
parity of M1. `tabulated` is any table set, including the B&M formulas sampled on the grid (V16).

### 3.2 Where tables come from

| Source | C_L, C_D, C_m tables | Derivatives |
|---|---|---|
| B&M Aerosonde | Sampled from the B&M formulas | B&M constants |
| Generator (§4) | AeroBuildup over α; post-stall from NeuralFoil section polars and flat-plate theory, blended at the wing's stall angle | AeroBuildup / VLM stability derivatives; control derivatives by finite differences of deflected surfaces |
| X8 | Gryte et al. 2018 polynomial fits sampled in attached flow; flat-plate blend past stall (documented as approximate) | Published values |

---

## 4. Procedural airframe generator

### 4.1 Pipeline (offline, deterministic per seed)

1. **Sample a design** within the class ranges of §4.2: span, aspect ratio, taper, sweep, dihedral,
   airfoils, tail volume coefficients, control-surface chord fractions, fuselage length, wing loading
   target, and propulsion layout (tractor or pusher).
2. **Lay out the geometry** as an AeroSandbox `Airplane`: wing, tails (conventional, T, V or
   twin-boom) and fuselage.
3. **Mass and inertia.**
   - Component budget: structure (areal density × wetted area), battery, motor, ESC, avionics and
     payload.
   - The wing is placed to hit a static-margin target (5–15% of the mean aerodynamic chord) using the
     computed neutral point.
   - The inertia tensor is assembled from components (wing and tail panels as plates, fuselage as a
     shell, point masses).
4. **Aerodynamics** at the cruise Reynolds number:
   - stability and control derivatives (AeroBuildup, with VLM as a cross-check on a sample);
   - static tables (§3.2);
   - effectiveness η(α).
5. **Propulsion sizing.**
   - Rules of thumb: power-to-weight by class, propeller diameter from span, motor KV and cell count
     chosen so that cruise throttle falls at 30–70%.
   - Propeller C_T(J) and C_Q(J) from a parametric family scaled by pitch-to-diameter ratio.
   - The M1 battery model is reused (state of charge, internal resistance, voltage sag).
6. **Actuators** randomised per airframe:
   - rate 150–600°/s;
   - **delay 10–60 ms** (doc 03 §3.1);
   - time constant, deadband, **backlash (new)** and travel ±20–30°.
7. **Sensors:** the M1 suite, with noise and bias drawn within datasheet ranges.
8. **Validity filters,** re-checked with **aerojax's own trim and linearisation**, independent of
   AeroSandbox:
   - static margin 3–25%;
   - trimmable from 1.2·V_stall to V_max with |δe| ≤ 80% of travel;
   - C_nβ > 0 and C_lβ < 0;
   - short period and Dutch roll damped;
   - spiral time-to-double > 10 s;
   - roll rate ≥ 60°/s at cruise;
   - thrust-to-weight ≥ 0.4;
   - stall speed 7–20 m/s;
   - wing loading 20–150 N/m².
9. **Export:** a fleet bundle (§2) and a glTF mesh from the same geometry.

### 4.2 Classes

| Class | Span | Mass | Notes | Split |
|---|---|---|---|---|
| Conventional | 1–4 m | 0.8–15 kg | Tractor or pusher | Train / val / test |
| T-tail | 1–4 m | 1–15 kg | | Train / val / test |
| V-tail | 1–3 m | 0.8–8 kg | Ruddervators mixed into elevator and rudder commands | Train / val / test (cut if behind) |
| Twin-boom pusher | 1.5–4 m | 2–15 kg | | Train / val / test |
| Glider | 2–4 m | 1–6 kg | AR 12–20, small motor | Train / val / test (cut if behind) |
| Flying wing | — | — | **Held out entirely**: only X8 and Zagi as references | Test only |

Splits are by airframe ID hash: 80/10/10. Held-out classes and reference models (Rascal, X8) never
appear in training.

### 4.3 Throughput and caching

At about 1–2 s per airframe on one core, with 8 worker processes, 10,000 accepted airframes take
roughly an hour, depending on yield. The fleet bundle is cached by
(generator version, seed range, config hash).

---

## 5. JSBSim backend

- **`JSBSimVehicle`** wraps `jsbsim.FGFDMExec`.
  - **Rates:** 100 Hz control and the JSBSim integration rate at 200 Hz (RK4-class; recorded).
  - **Same interface:** sensors (pure-JAX sensor models, evaluated host-side from JSBSim truth),
    commands, faults (sensor and actuator faults in our layer; motor cut through JSBSim's engine
    model) and privileged truth.
  - **Ground:** flat at the site elevation, as in aerojax.
- **`tools/jsbsim/aerosonde_bm.xml`** re-expresses the B&M Aerosonde (aero v1 equations as JSBSim
  functions; density matched; β = 0 for V14).
- **Rascal 110.**
  - `fetch_rascal.py` downloads `Tools/autotest/aircraft/Rascal` from ArduPilot at a pinned commit
    into `~/.cache/aeroadapt/jsbsim/`.
  - Its piston engine means the motor-failure detector uses an rpm model learned from a calibration
    flight. It does not use the electric KV model, which is a realistic configuration step.
- **Stack configuration for the Rascal** uses only "datasheet" information (mass, span, wing area,
  inertia, cruise speed and generic aero assumptions) through `autoconfig`. It never uses the JSBSim
  aero tables, because the Rascal is held out.

---

## 6. Stack auto-configuration

`autoconfig(airframe_model) → StackConfig`. It is the analogue of a manufacturer's parameter file;
ArduPlane's in-flight AUTOTUNE is a Could here, because M3's RL policy targets the same need.

| Parameter group | Method |
|---|---|
| Speed limits | V_stall from C_L,max; ARSPD_FBW_MIN = 1.3·V_stall; cruise from minimum power required; V_max from the throttle and structural limit |
| Trim schedules | aerojax trim at 6–8 speeds → throttle(V), pitch(V) and elevator(V), fitted to the M1 forms (`thr_slope`, `pitch_trim_b`, the elevator trim curve) |
| Roll loop | Roll-mode time constant τ_p = −I_x / (q̄Sb²C_lp / 2V) and aileron rate gain → FF, P, I for the ArduPlane-style rate loop at the target time constant (0.5 s) |
| Pitch loop | Short-period approximation from C_mα, C_mq, C_mδe → FF, P, I; same target |
| Yaw | Damper gain from C_nr and C_nδr; turn coordination from the kinematic relation |
| TECS | climb_max and sink_min from excess power along the trim schedule; time constant scaled with the phugoid period |
| L1 | Period scaled from the roll response and cruise speed (ArduPlane's 17 s default at Aerosonde scale) |
| Contingency | Glide polar (best glide speed, minimum sink) from the trim model; approach and flare speeds from V_stall |

**V20 acceptance (V11-lite):**
- roll and pitch steps;
- altitude and speed steps;
- L1 cross-track error on a box mission;
- servo saturation.

Runs are batched over the fleet in calm air and in light turbulence. The thresholds are the M1 V11
thresholds scaled by airframe size and speed where they are dimensional.

**Batching.** `StackParams` leaves gain a fleet axis, so stack and simulator `vmap` together over
heterogeneous airframes.

---

## 7. faultlab v1

### 7.1 Slot types (fixed shape, ADR-003)

The fixed number of simultaneous faults per scenario is:

| Slot | How many |
|---|---|
| Sensor | 2 |
| Actuator | 2 |
| Propulsion | 1 |
| Energy | 1 |
| Airframe | 1 |
| Link | 1 |

Each slot has an onset, an optional ramp time, a magnitude and a kind enum. Labels are extended from
`fault_bits` to `(bits, per-slot kind, magnitude, onset)`.

| Category | Kinds (count) |
|---|---|
| Air data | Pitot blockage (total, with or without drain: M1), partial blockage (gradual), airspeed bias, stuck, drift (5) |
| Inertial | Gyro bias step, accelerometer bias step, IMU vibration (noise + clipping), magnetometer interference, barometer drift (5) |
| GNSS | Dropout, slow drift, **spoofing walk-off (consistent position and velocity)** (3) |
| Actuators | Stuck at position (M1), hard-over, floating, slow (rate limit), reduced throw, backlash, **reversed** (7) |
| Propulsion | Motor cut (M1), partial thrust loss, propeller damage (thrust loss + vibration), thermal derate (4) |
| Energy | Cell failure, abnormal sag (resistance step), capacity loss (3) |
| Airframe | CG shift, payload drop (mass and inertia step), wing damage (lift, drag and roll asymmetry) (3) |
| Link | RC / telemetry link loss (a flag to the stack's failsafe logic) (1) |

That is 31 kinds, against a Must of ≥24.

### 7.2 Environment events (scenario elements, not faults)

- **Discrete gust:** 1−cos, MIL-F-8785C shape, with length and amplitude.
- **Wind shear:** a linear gradient below a height; a downburst-lite outflow profile.
- **Thermals and sink:** the Allen (2006) updraft model, positioned in the site frame.
- **Strong wind:** up to 80% of cruise speed.
- **Turbulence:** moderate and severe Dryden levels, added to M1's light.

### 7.3 Confounded-pair families

These are defined as campaign YAMLs and used for diagnosis in M4. M2 only checks that each pair is
indistinguishable to the M1 stack's simple monitors for its stated horizon (V21c).
- pitot blockage ↔ real airspeed loss;
- GNSS walk-off ↔ wind change;
- battery sag ↔ propeller damage;
- CG shift ↔ elevator offset;
- floating surface ↔ turbulence.

---

## 8. Bench v0

- **Scenario sampler.** Given a split and a seed, it draws:
  - an airframe;
  - a mission (a random closed loop within the site, altitude 80–150 m AGL);
  - weather (wind 0–80% of cruise speed, turbulence level, events);
  - faults (none, single, or two; uniform over kinds) and their onsets.

  It writes scenario YAMLs referencing the fleet bundle.
- **Baselines:**
  - **B1:** stock stack, no contingency;
  - **B2:** stock stack plus the classical contingency v0, which only acts on motor failure. For
    other faults B2 = B1, and that is reported, not hidden.
- **Metrics (doc 03 §5.3, M2 subset):**
  - survival (no loss of control and no ground contact before a planned landing);
  - mission completion;
  - for motor cuts: landed in zone and touchdown class;
  - detection latency where a detector exists;
  - false alarms on fault-free scenarios.
- **Statistics:** Wilson intervals for rates and bootstrap intervals for continuous metrics. Results
  are broken down by fault category, airframe class and weather band.
- **Output:**
  - `runs/bench/<id>/outcomes.parquet`, `summary.json` and a report section;
  - `aeroadapt bench-run <spec>` reproduces it;
  - the same seed gives an identical table (V22).

---

## 9. Live mode

- **Server (`aeroadapt serve`).**
  - It runs the host-stepped `FlightStack` and `Vehicle` (aerojax or JSBSim) paced to 0.25–4× real
    time. M1 measured a 21× real-time capacity on one CPU core.
  - It streams the M1 viz column set at 50 Hz over WebSocket + msgpack: `hello` (manifest), `frame`,
    `event`.
  - It records a replay bundle for every session.
  - It binds to localhost only.
- **Commands:** `inject_fault{spec}` (any §7 kind, effective at the next control step),
  `clear_fault{slot}`, `set_speed{x}`, `pause`, `resume`, `reset{scenario, airframe}`.
- **Skyview live mode** (`?live=ws://localhost:8765`).
  - Frames go into a ring buffer feeding the same `SceneController`, HUD and event log as replays.
  - A **fault panel**: one button per fault kind, with magnitude controls.
  - An **airframe picker** that loads each airframe's glTF.
  - A **ghost** can replay the same scenario from a recorded bundle for comparison.
- **Fleet view** (Should): the 1 Hz trajectories of a campaign or bench run drawn as many small
  aircraft. This is the "10,000 aircraft" visual.

---

## 10. Validation additions (continuing M1's numbering)

| ID | What | Criterion | Tier |
|---|---|---|---|
| V14 | **JSBSim cross-check** (from M1): `aerosonde_bm.xml` power-off glide trim and elevator-doublet response | Trim α within 0.1° and γ within 0.05°; pitch-response RMS difference ≤ 0.3° over 10 s | full |
| V15 | Throughput (updated): heterogeneous fleet + faults, CPU and GPU | Recorded; GPU ≥ 1M env steps/s (D9) | bench |
| V16 | Aero v2 vs v1: B&M formulas sampled as tables | max \|ΔC_L\|, \|ΔC_D\|, \|ΔC_m\| ≤ 1e-3 for α ∈ [−15°, 20°]; M1 golden flights within tolerance | fast |
| V17 | N-version parity for tables: NumPy reference interpolation (written independently) vs aerojax, 1,000 random states with random tables | Forces and moments relative error ≤ 1e-10 (float64) | fast |
| V18 | Generator vs published values: Aerosonde (B&M geometry) and X8 (Gryte et al. 2018) regenerated from geometry | C_Lα within 15%, C_mα sign and stability, C_lp within 25%; others recorded with bands (VLM-vs-wind-tunnel discrepancies of 20–30% are typical, and the report says so) | full |
| V19 | Generator validity and diversity | 100% of accepted airframes re-pass the filters in aerojax; yield and parameter distributions reported; no duplicate airframes | full |
| V20 | Auto-configuration acceptance (V11-lite) on 1,000 sampled airframes, calm and light turbulence | ≥ 90% pass; failures classified | full |
| V21 | faultlab v1: (a) each kind produces its intended observable effect; (b) no effect before onset; (c) the confounded pairs are indistinguishable to M1 monitors within their horizon | All kinds | fast (a, b) / full (c) |
| V22 | Bench reproducibility and sanity | Identical outcome table from the same seed; B2 ≥ B1 on motor-cut survival | full |
| V23 | Live mode: pacing jitter, injection latency, session bundle equal to an offline rerun of the same command log | Jitter p99 ≤ 20 ms at 1×; injection effective at the next control step; bundle byte-identical | fast |

---

## 11. Work plan (weeks 5–8, about 30 h/week)

| Week | Work (hours) | Exit criterion |
|---|---|---|
| **5 · Second simulator and aero v2** | Aero v2 tables + effectiveness + schema v2 (6) · Reference extension + V16/V17 (4) · `JSBSimVehicle` + `aerosonde_bm.xml` + V14 (10) · Rascal fetch + datasheet config + first flight (Should, 4) · Ground effect (Should, 2) · Log and docs (2) | V14, V16, V17 green; the Aerosonde flies in both simulators |
| **6 · The generator** | Sampling, geometry, mass and inertia (8) · AeroSandbox derivatives, tables, polars (8) · Propulsion sizing, actuators, sensors (4) · Filters re-checked in aerojax (4) · glTF export (2) · V18 Aerosonde/X8 + fleet build + V19 (4) | 10k-airframe fleet bundle; V18/V19 in the report |
| **7 · Flying the fleet; faults; live** | `autoconfig` + batched `StackParams` + V20 (9) · faultlab v1 slots + environment events + V21 (11) · Live server + Skyview live mode + fault panel + airframe picker + V23 (10) | ≥90% V20 pass; ≥24 fault kinds; live fault injection works end to end |
| **8 · Bench, throughput, proof** | Bench sampler + B1/B2 + metrics + V22 (10) · Fleet throughput (CPU) + GPU notebook (4) · Fleet view (Should, 4) · Validation report M2 sections (4) · Demo video + `docs/demo/m2.md` + post draft (6) · Log (2) | **Definition of Done (§1)** |

**If behind**, slip in this order:
1. Glider and V-tail classes.
2. Fleet view.
3. Rascal (D4).
4. faultlab from 30 kinds down to 24.
5. Ground effect.

Never cut aero v2 parity (V16/V17), V14, V20, the bench or live fault injection.

---

## 12. M2 risks

| Risk | Mitigation |
|---|---|
| Generated derivatives are inaccurate (VLM and buildup limits, low Reynolds) | V18 against two published airframes; tolerance bands stated honestly; filters in aerojax rather than AeroSandbox |
| Post-stall data is approximate | Documented; effectiveness tables keep control authority plausible; M2 does not claim stall-recovery fidelity |
| Low generator yield or narrow diversity | Report yield per filter; tune sampling ranges, not filters |
| Auto-configuration fails on a slice of the fleet | Report the failures by class and cause; exclude from the B1 fleet with the reason recorded |
| JSBSim unit and axis conventions | V14 on a known model before the Rascal; conversion tested like ADR-001 |
| GPU access | Notebook + CLI; CPU numbers always reported; GPU number recorded when available |
| Scope (this is a large milestone) | Must/Should/Could; slip order above |

---

## 13. Open questions (non-blocking until week 8)

1. **GPU for the throughput run.** Is a free Google Colab or Kaggle T4 session acceptable (you run a
   prepared notebook)? Or would you rather rent a cloud GPU for about an hour (roughly £1–3; needs an
   account)?
2. **Where the M2 post goes** (doc 03: each milestone ends with a public post). A draft will be in
   `docs/blog/`.

---

## Sources

- AeroSandbox and NeuralFoil: P. Sharpe, MIT licence, https://github.com/peterdsharpe/AeroSandbox
- JSBSim 1.3.2, LGPL-2.1, https://github.com/JSBSim-Team/jsbsim
- Rascal 110 JSBSim model: ArduPilot `Tools/autotest/aircraft/Rascal` (GPL-3.0, derived from FlightGear)
- K. Gryte et al., "Aerodynamic modeling of the Skywalker X8 Fixed-Wing Unmanned Aerial Vehicle", ICUAS 2018; parameter file via PyFly (MIT), https://github.com/eivindeb/pyfly
- MIL-F-8785C and MIL-HDBK-1797 (gust and turbulence models)
- M. J. Allen, "Updraft Model for Development of Autonomous Soaring Uninhabited Air Vehicles", AIAA 2006-1510
- B. W. McCormick, *Aerodynamics, Aeronautics and Flight Mechanics* (ground effect)

---

## 14. Implementation log (living section)

*Appended weekly: results, deviations, and the Definition of Done status.*

### Week 5 (15 Oct 2026)

**Built**

- **Aero v2** (§3):
  - Static C_L, C_D and C_m tabulated over 360° of α on a uniform 0.5° grid (721 points),
    with effectiveness tables scaling the damping and control derivatives.
  - `AirframeParams.tables = None` keeps the analytic B&M model with *unchanged expressions*,
    so the M1 goldens and V13 are untouched.
  - `aero.model: bm_tabulated` samples the B&M formulas onto the grid (in NumPy, separately
    from the JAX formulas).
  - Lookups are index arithmetic, so a heterogeneous fleet `vmap`s with one shape.
- **Ground effect** (Should, done):
  - McCormick's induced-drag factor, selected by `aero.ground_effect`. With the flag at 0 the
    correction is exactly zero.
  - Off for the textbook Aerosonde; it will be on for the generated fleet.
  - The lift-curve-slope increase is neglected (documented).
- **JSBSim backend** (`jsbsim` optional extra, LGPL, installed in CI):
  - `sim/jsbsim/export.py` writes any aerojax airframe as a JSBSim model:
    - body-axis forces, with lift and drag rotated by α only, as in B&M;
    - tables and effectiveness factors;
    - ground effect;
    - external reactions for thrust and propeller torque.
  - `tools/jsbsim/aircraft/aerosonde_bm` is generated, and a test checks that it is current.
  - `JSBSimFDM` is an SI wrapper around JSBSim.
  - `JSBSimVehicle` puts JSBSim behind the `Vehicle` interface:
    - JSBSim provides the rigid body, gravity, Earth, atmosphere, wind and Milspec turbulence;
    - actuators, motor, propeller, battery, sensors and faults reuse the aerojax models.
- **Rascal 110:** `sim/jsbsim/rascal.py` fetches it from ArduPilot at a pinned commit into
  `~/.cache/aeroadapt/jsbsim` (GPL, never committed).
  - The model as fetched: 5.9 kg; a 2.2 kW JSBSim electric engine with an 18×8 propeller; its
    own control system with asymmetric elevator travel (−0.35/+0.30 rad).
  - Its surface signs map directly onto B&M signs.
- **Report builder:** now writes `docs/validation/m2_report.md` (M1 sections recomputed, plus
  V14 and V16/V17). `m1_report.md` is the frozen M1 artefact.

**Results**

| Check | Result | Criterion |
|---|---|---|
| V14 power-off glide trim at 22.5 m/s (ρ and g matched) | α 4.567° vs 4.558° (Δ 0.009°); γ −4.471° vs −4.475° (Δ 0.005°) | Δα ≤ 0.1°, Δγ ≤ 0.05° ✅ |
| V14 elevator doublet ±2°, 10 s | 0.051° RMS in pitch | ≤ 0.3° ✅ |
| V14 aileron doublet ±2°, 10 s (recorded) | 0.31° RMS in roll on a 23° response; 0.27° in heading | ≤ 1° ✅ (sign conventions and the product of inertia confirmed) |
| Exporter vs aerojax, all six axes, 40 random conditions (random tables, β, rates, ground effect) | within 2e-5 of q̄S (q̄Sb for moments) | ✅ |
| V16 tables vs analytic, α ∈ [−15°, 20°] | C_L 9.6e-5, C_D 1.4e-5, C_m 2e-15 | ≤ 1e-3 ✅ |
| V16 golden flights on tabulated aero | all 5 within golden tolerances | ✅ |
| V17 reference parity (random tables; 20 s trajectory; ground effect at 0–6 m) | ≤ 1e-10 relative | ✅ |
| Stock stack in JSBSim (Aerosonde, 5 m/s wind, 90 s box mission) | altitude ±2 m, navigator–truth position error ≤ 2 m, 6.9× real time (aerojax host-stepped: 13.6×) | D3 ✅ |

**Deviations**

1. **The Rascal's first flight moves to week 7.** It needs a stack configuration and a
   "datasheet" airframe that must not come from its JSBSim aerodynamic tables (it is held out).
   - Week 6's generator, which estimates coefficients from geometry, provides that airframe.
   - Week 7's auto-configuration provides the stack configuration.
   - Hand-tuning it now would be thrown away.
2. **V14 runs in the fast tier**, not the full tier: it takes about 2 s.
3. **JSBSim integrators.** JSBSim's defaults are used (rectangular Euler for the rotational
   states, Adams–Bashforth 2/3 for the translational ones) at 200 Hz, to keep the simulator
   independent. The roll-response difference in V14 is consistent with that.
4. **`JSBSimVehicle` initial state.** It starts from aerojax's trim of the same airframe (the
   stack's view of trim), and JSBSim settles from there within the first second.

**Definition of Done status (M2)**

| # | Status |
|---|---|
| D1 CI green | ✅ (week 5 commits) |
| D2 Aero v2 + V16/V17 | ✅ |
| D3 JSBSim backend + V14 | ✅ |
| D4 Rascal flies (Should) | ⏳ week 7 (deviation 1) |
| D5–D11 | ⏳ weeks 6–8 |
