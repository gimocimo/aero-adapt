# Txt2Fly Technical Specification: Tell It Where to Go

*Status: draft · 21 Oct 2026 · Parent plan: [03 — AeroAdapt-UAV](03_uav_architecture_and_six_month_roadmap.md) §7.2 (weeks 9–10, inserted on 21 Oct 2026) · Previous milestone: [05 — M2](05_m2_technical_spec.md)*

**Goal.** Let an operator say where the aircraft should go and what it should do, in plain words:

- "go to the church and fly a 30 m circle 80 m above it";
- "fly by field A";
- "patrol fields A, B and C".

The system turns the words into a flight plan for the airframe actually flying, in the wind
actually blowing. It shows the plan on the map, says plainly what it cannot do and why, and,
once the operator confirms, flies it. A follow-up ("make it wider", "now go to B") changes the
plan in flight, and the autopilot brings the aircraft onto the new path.

**Weeks 9–10, about 30 h/week (~60 h).** M3 starts in week 11, with M2's two carry-overs:
auto-configuration at 90 % (D6) and the Rascal (D4).

**Out of scope** (later milestones):
- terrain: the ground is flat, so altitudes are above flat ground (M5);
- obstacle avoidance beyond fixed polygons;
- powered take-off and landing (the stock stack has neither; `land` here is a glide landing);
- several aircraft at once;
- voice input (Could: browser speech-to-text in front of the same text box);
- ArduPilot SITL (M6) and the learned pilot (M3).

---

## 0. Key decisions

1. **The language model never flies.** It translates text into a typed plan, and everything after
   that is deterministic code:
   - a **path builder** turns the plan into a mission for this airframe and this wind;
   - a **checker** accepts or rejects the mission, with plain-language reasons;
   - the **operator confirms** on the map;
   - **the autopilot flies** the mission.

   The model's output cannot get past the checker: a plan has no field that changes a safety rule,
   and the hard rules have no override.
2. **The mission is the interface.** Txt2Fly writes missions and nothing else. Whatever flies them
   is unchanged: the stock stack today, the learned pilot from M3. The contingency keeps its
   authority: after a motor cut it takes over, whatever the plan says.
3. **Feasibility is physics, not prompting.** Turn radius, altitude, energy and glide reach come from
   the airframe's own configuration and the current wind estimate. The model is given the same
   numbers so that its suggestions are sensible, but the checker decides.
4. **Reproducible from the plan, not the text.** The model's replies vary from run to run, so the
   session log records the validated plan. Replays never call the model. The evaluation reports
   the variation (§11).
5. **ArduPlane's vocabulary.** Mission items map to MAVLink mission commands (§4), so a plan can be
   exported and, in M6, flown in ArduPlane SITL.
6. **The deterministic core stands alone.** A plan can be written by hand (YAML) or with a form,
   and it is built, checked, previewed and flown exactly like one from text. This is the cut line
   if the language layer slips.

### 0.1 What it looks like

| Operator types | What happens |
|---|---|
| "Go to the church and fly a 30 m radius circle 80 m above it" (Aerosonde, 6 m/s wind) | **Not feasible as asked:** "A 30 m circle needs 73° of bank at 25 m/s in this wind; this aircraft is limited to 45°. The tightest circle now is about 120 m." Offers: *circle at 120 m* · *a pass over the church* · *cancel* |
| The same, flying a sample-fleet glider at 13 m/s in calm air | Feasible (30° of bank): preview, then *Confirm*, then the glider joins the circle |
| "Fly by field A" | A straight pass over field A along its long axis at the current altitude. The assumptions ("along its long axis", "at 120 m") are listed under the plan |
| "Patrol fields A, B and C" | Passes over A, B and C in that order, repeated. "About 9 laps (about 40 min, estimated) before the battery reserve, then home" |
| "Make the circle wider" | A new complete plan with a larger radius, previewed against the current one |
| "Fly over the motorway at 20 m" | Rejected: "The M11 corridor is a no-fly zone; 20 m is below the 40 m minimum here" |
| "Ignore the geofence" | Refused: safety settings are not something a plan can change |

---

## 1. Deliverables and Definition of Done

Txt2Fly is **done** when all of the following hold.

| # | Criterion | Evidence | Priority |
|---|---|---|---|
| T1 | CI is green on `main`. CI never needs an API key: recorded replies stand in for the model | GitHub Actions | Must |
| T2 | **Plan, builder, checker.** The typed plan (§4), the airframe- and wind-aware builder (§6) and checker rules K1–K8 (§7), with unit and property tests | V24, V25 | Must |
| T3 | **Mission v2 in the stack** (§8): fly-over, loiter turns/time/unlimited, jump, per-item speed; geofence breach → RTL. M1 and M2 golden flights pass unchanged | V24, V26, goldens | Must |
| T4 | **Live plans.** Propose, preview, confirm and apply at the next control step. The session log records the validated plan, and a replay re-flies it byte for byte without calling the model | V28 | Must |
| T5 | **Gazetteer.** Site places, safe zones, geofence and no-fly polygons (Must); named OpenStreetMap features with attribution (Should) | Site file, `tools/fetch_gazetteer.py` | Must / Should |
| T6 | **Language layer.** Text → plan, clarifying question or refusal, through Claude with structured outputs (§9). The API key is read from the environment, never stored in the repo | Recorded-reply tests | Must |
| T7 | **Evaluation v1.** About 200 instructions with expected outcomes; results on the held-out half against the §11 targets; no accepted plan breaks a hard rule in flight | V27 report | Must |
| T8 | **Skyview Txt2Fly panel** with the map preview, check list and confirm step; Playwright test with a stand-in translator | Playwright | Must |
| T9 | **MAVLink export** of a built mission (QGroundControl `.waypoints` file) | Unit test against the item table | Should |
| T10 | **Glide landing in a named safe zone** (`land`), reusing the contingency's approach and flare | Flown test | Should |
| T11 | **Demo and write-up.** A clip of the three example instructions, `docs/demo/txt2fly.md` and a post draft | Video + docs | Must |
| T12 | `survey`: back-and-forth lines over a field | Flown test | Could |

Priorities are as in M1 and M2. A Should moves to M3 week 1 (week 11) if the schedule slips.

---

## 2. Repository additions

```
src/aeroadapt/txt2fly/
  plan.py         typed plan: steps, targets; Reply = plan | clarify | refuse
  gazetteer.py    site places, safe zones, no-fly polygons, OSM features
  builder.py      plan → mission v2 items + preview path (airframe- and wind-aware)
  checker.py      rules K1–K8, plain-language reasons and suggestions
  translate.py    Claude: text + context → Reply
  mavlink.py      QGroundControl .waypoints export (Should)
  evaluate.py     eval runner and deterministic grader
src/aeroadapt/stack/mission.py      mission v2 items and sequencing
configs/sites/cambridge_west.yaml   + geofence, ceiling, minimum height, no-fly polygons, places
configs/sites/cambridge_west.osm.json   OSM extract (ODbL, attributed)
configs/txt2fly/eval_v1.yaml        the evaluation set
tools/fetch_gazetteer.py
web/skyview/src/live/Txt2FlyPanel.tsx, PlanOverlay.ts
tests/test_v24_builder.py … test_v28_live_plans.py, tests/fixtures/txt2fly/ (recorded replies)
```

New commands:

- `aeroadapt plan check <plan.yaml> [--airframe ID] [--wind KT@DEG]`: build and check a typed plan;
  print the mission and every rule's result.
- `aeroadapt plan fly <plan.yaml> …`: fly it offline and export the replay.
- `aeroadapt txt2fly ask "<text>" …`: one instruction through the model, then the same as `check`.
- `aeroadapt txt2fly eval [--split test] [--repeats 3]`: the evaluation (calls the API, so it costs
  money; never run in CI).

---

## 3. The pipeline

```mermaid
flowchart LR
  U[Operator text] --> T["Translator<br/>(Claude, structured output)"]
  G[Gazetteer] --> T
  A["Capability card<br/>(airframe + wind)"] --> T
  T -- clarify / refuse --> U
  T -- typed plan --> B["Path builder<br/>(deterministic)"]
  F[Typed plan from a form or file] --> B
  B --> K["Checker<br/>(rules K1–K8)"]
  K -- reasons + suggestion --> U
  K -- accepted --> P[Preview on the map]
  P -- Confirm --> M["set_mission<br/>(next control step)"]
  M --> S[Stock stack → aircraft]
```

**Adaptive in flight**, in three ways:

- a follow-up instruction produces a new complete plan, built from the aircraft's current position;
- the checker re-runs on the active plan when the wind estimate changes (§7.1);
- open-ended patrols are bounded by the battery (§6).

---

## 4. The plan: building blocks

A plan is 1–8 steps. Each step is one of the blocks below, a discriminated union on `do`. The
builder expands the steps into at most 32 mission items.

| Step | Fields | Meaning | Mission items | MAVLink command |
|---|---|---|---|---|
| `goto` | target, alt?, speed? | Fly to a place and carry on | WAYPOINT | `NAV_WAYPOINT` (16) |
| `pass_over` | target, alt?, heading: `along` \| `into_wind` \| degrees | A straight pass, wings level over the place ("fly by field A") | WAYPOINT (run-in), FLYOVER, WAYPOINT (run-out) | `NAV_WAYPOINT` ×3, the middle one with a near-zero acceptance radius |
| `orbit` | target, radius, alt?, direction `cw` \| `ccw`, then: turns \| seconds \| `forever` | Circle a place | LOITER_TURNS, LOITER_TIME or LOITER_UNLIM | `NAV_LOITER_TURNS` (18), `NAV_LOITER_TIME` (19), `NAV_LOITER_UNLIM` (17); a negative radius means anticlockwise |
| `patrol` | 2–6 targets, visit as `pass_over` \| `goto`, alt?, repeat: count \| `until_reserve` | Visit places in the given order, repeatedly | Each visit's items + JUMP | `DO_JUMP` (177) |
| `set_speed` | `slow` \| `cruise` \| `fast` \| m/s | Change airspeed for the following steps | Per-item speed | `DO_CHANGE_SPEED` (178) |
| `hold` | radius? | Circle where you are | LOITER_UNLIM at the current position | `NAV_LOITER_UNLIM` (17) |
| `return_home` | — | Go home and circle | RTL | `NAV_RETURN_TO_LAUNCH` (20) |
| `land` (Should) | safe zone | Glide landing in a safe zone: motor off, the contingency's approach and flare | LAND_ZONE | Nearest is `NAV_LAND` (21), but that is ArduPlane's powered landing; ours is a glide |
| `survey` (Could) | area, alt, line spacing | Back-and-forth lines over a field | WAYPOINTs | `NAV_WAYPOINT` |

**Targets** are one of:
- `place`: a gazetteer id, restricted in the schema to the site's ids, so the model cannot invent one;
- `here` or `home`;
- `offset`: a bearing and distance from a place;
- `ne_m`: a point clicked on the map;
- `latlon`: coordinates typed by the operator.

**Units and conventions.**
- Altitudes are metres above the (flat) ground. "80 m above the church" means 80 m above the
  ground at the church; the plan lists that reading among its assumptions, and the checker keeps
  clear of the church's height when it is known (K4).
- `slow`, `cruise` and `fast` map to the airframe's configured airspeeds.

**The reply** is always one of three types:

- `plan`:
  - `summary`: one sentence restating the plan, shown for confirmation;
  - `assumptions`: every reading the model had to choose;
  - `steps`.
- `clarify`: a question and 2–4 short answers, shown as buttons.
- `refuse`: a reason.

A follow-up always returns a complete plan, never a difference from the active one, so every plan
is checked whole.

---

## 5. The gazetteer

- **Site file** (hand-made, like the M1 safe zones):
  - `geofence_ne_m` (a polygon), `ceiling_agl_m` (120 m, as the UK open category), `min_agl_m`
    (40 m);
  - `no_fly` polygons, each with a name and a reason: the M11 corridor, Coton village and the West
    Cambridge campus, the ground-risk features in the site description;
  - named `places`: home and any hand-picked points;
  - the existing safe zones A, B, C and F, which are also places ("field A").
- **OpenStreetMap features** (Should).
  - `tools/fetch_gazetteer.py <site>` queries the Overpass API once, for named features inside the
    geofence: places of worship, named buildings, woods, water, farms, roads and villages.
  - It converts them to local NE metres, keeps the OSM `height` tag when present, and writes
    `configs/sites/<site>.osm.json` with the query date and "© OpenStreetMap contributors, ODbL".
  - The extract stays under the ODbL. Skyview shows the attribution whenever Txt2Fly is open.
- **Names are untrusted text.** OSM names are third-party data. They reach the model inside a
  quoted data block, and the model answers with ids. The evaluation includes features with
  instruction-like names (§11).

---

## 6. The path builder

Inputs:

- the plan;
- the airframe's stack configuration (airspeeds, roll limit, waypoint radius);
- the wind: the navigator's estimate when live, the scenario's wind offline;
- the current position, altitude and battery state;
- the site.

**The tightest circle.** In a coordinated turn the radius is r = V² / (g · tan φ). Over the
ground, the tightest point of a circle in wind is downwind, where the ground speed is airspeed
plus wind. With a 5° bank margin kept for tracking corrections:

  r_min = (V + W)² / (g · tan(φ_lim − 5°))

| Aircraft | Airspeed | Roll limit | Tightest circle, calm | Tightest circle, 6 m/s wind | Bank for a 30 m circle (calm / 6 m/s) |
|---|---|---|---|---|---|
| Aerosonde (M1) | 25 m/s | 45° | 76 m | 117 m | 65° / 73° |
| Sample twin-boom (`gen-twin_boom-00058`) | 15 m/s | 45° | 27 m | 54 m | 37° / 56° |
| Sample glider (`gen-glider-00191`) | 13 m/s | 45° | 21 m | 44 m | 30° / 51° |

So the church example is infeasible for the Aerosonde at any wind, feasible for the glider in calm
air, and infeasible for the glider in a 6 m/s wind. The checker states that in those words.
Suggestions round up to the next 10 m.

**Passes.**
- The pass line runs through the target:
  - over a field: through the polygon's centroid, along its long axis;
  - over a point: along the current course.
  - `into_wind` aligns it with the wind.
- Run-in and run-out each extend 2·r_min, at least 100 m, beyond the target, so the turn onto
  the line ends before the target.

**Patrols.**
- The places are visited in the order given; the operator's order is part of the intent.
- `until_reserve` becomes a repeat count. The count is the energy available above the reserve,
  minus the energy to get home, divided by the energy per lap. The operator sees it as "about N
  laps (about T min)".
- A battery-reserve RTL in the stack (Should, §8) is the backstop if the estimate is optimistic.

**Energy.**
- Electrical power at trim for each leg's airspeed, from the same batched trims that
  auto-configuration uses, plus climb energy.
- Ground speeds from the wind; battery capacity from the airframe.
- It is an estimate and is labelled as one.

**The join.**
- The new mission starts with a leg from the aircraft's current position to the first item.
- If that leg crosses a no-fly polygon, the checker rejects the plan (K8).
- A detour around the polygon is a Could.

**Output.**
- The mission items (§8).
- A **preview path**: the expected ground track, with turns drawn at r_min, sampled every 10 m.
  The checker and the map both use it.
- The flown track differs from the preview (L1 overshoot, gusts); V26 measures by how much.

---

## 7. The checker

Every rule returns pass or fail with a reason in plain words and, where one exists, the nearest
plan that would pass. Hard rules cannot be overridden by the plan, by text or by the model. They
change only in the site and stack configuration files.

| Rule | Check | Kind | Example reason |
|---|---|---|---|
| K1 | Every target resolves; the plan expands to ≤ 32 items | Hard | "There is no place called 'the old mill' on this site." |
| K2 | Every circle and turn is at least r_min (§6) for the commanded speed and the current wind | Hard | "A 30 m circle needs 73° of bank at 25 m/s in this wind; this aircraft is limited to 45°. The tightest circle now is about 120 m." |
| K3 | The preview path, widened by r_min/2 + 20 m, stays inside the geofence and outside every no-fly polygon | Hard | "The circle would cross the M11 corridor (no-fly)." |
| K4 | min_agl ≤ altitude ≤ ceiling; at least 30 m above any known feature height under the path | Hard | "120 m is the ceiling here." |
| K5 | Airspeeds within the airframe's configured range | Hard | "This aircraft cannot fly slower than 11 m/s." |
| K6 | Predicted energy ≤ remaining energy − reserve (30 % of capacity); open-ended plans must be bounded (`until_reserve`) | Hard | "This plan needs about 140 % of the remaining battery." |
| K7 | Glide reach: every 50 m along the path, at least one safe zone is reachable in the current wind, by the contingency's own test (`assess_zones`) | **Soft** | "From the far side of the circle, no field is within glide." Uncovered stretches are drawn in amber |
| K8 | The join from the current position passes K3 and K4 | Hard | "The way there crosses Coton village (no-fly)." |

- A soft rule needs the operator to tick an acknowledgement in the confirm dialog. The model
  cannot acknowledge anything.
- Acknowledgements are logged with the plan.

### 7.1 Re-checking in flight

- The checker re-runs on the active plan every 10 s, and when the wind estimate changes by more
  than 2 m/s or 20°.
- If a hard rule now fails (say the wind has risen, and the circle needs more bank than the limit):
  - the HUD shows the reason and the deterministic fix (the nearest feasible radius), for the
    operator to confirm;
  - meanwhile the aircraft keeps flying the plan within its roll limit, so the circle stretches
    downwind, as ArduPlane's would;
  - the geofence breach action (RTL) is the backstop.
- Nothing changes without the operator.

---

## 8. Stack: mission v2

The stock stack's mission is a list of waypoints with a loop-or-RTL flag. Txt2Fly needs ArduPlane
style items.

- **Item table:** a fixed shape, 32 × 8 (ADR-003). The columns are:
  - kind;
  - north, east, altitude;
  - signed radius (positive clockwise);
  - p1: turns, seconds or jump target;
  - p2: repeat count, −1 for forever;
  - speed (0 = unchanged).
- **Kinds:**
  - WAYPOINT: fly-by; switches within the waypoint radius or on passing, as now;
  - FLYOVER: switches only on passing;
  - LOITER_TURNS, LOITER_TIME and LOITER_UNLIM: the existing L1 loiter law with the item's centre,
    radius and direction;
  - JUMP;
  - RTL;
  - LAND_ZONE (Should).
- **Sequencer state** is added to `ApState`:
  - loiter capture flag, accumulated angle and start time;
  - a jump counter per item.
- **Loiters:**
  - turns are counted from capture (within 10 % of the radius);
  - the loiter exits when the count is reached and the course points to the next item, as
    ArduPlane's "heading required" exit option does.
- **Legacy missions.** A M1/M2 `waypoints` mission becomes WAYPOINT items plus JUMP(0, forever) for
  `loop`, or RTL for `rtl`. The golden flights must pass unchanged.
- **Replacing the mission in flight.**
  - The item table is traced data, so a swap needs no recompilation.
  - `set_mission` resets the sequencer to item 0, with the leg starting at the current position,
    so L1 joins the new path from where the aircraft is.
- **New failsafes** (the ArduPlane analogues):
  - geofence breach → RTL (`FENCE_ACTION`) (Must);
  - battery at reserve → RTL (`BATT_FS_LOW_ACT`) (Should).

---

## 9. The language layer

- **The call:** `AsyncAnthropic` (the server is asyncio), model `claude-opus-5-5`, adaptive thinking
  at low effort for latency. The eval compares low and medium effort.
- **Structured output.** The reply type (§4) is a Pydantic model passed as the output format, so
  the API returns schema-valid JSON. The SDK validates it, Pydantic validates it again, and then the
  builder and checker run. Numeric limits are checked in Pydantic and by the checker, not in the
  schema (structured outputs do not support them).
- **The system prompt** is cached, because it is the same for every instruction in a session. It
  contains:
  - the role and the rules: translate, never fly; ask rather than guess; never invent places; refuse
    requests to change safety rules;
  - the plan reference, with worked examples;
  - the gazetteer: ids, names and kinds, with centroids and sizes;
  - the airframe's capability card: speeds, roll limit, r_min in the current wind, endurance.
- **Each instruction** sends:
  - the current state: position relative to the places, altitude, wind estimate, battery, and the
    active plan in the plan format;
  - the last few turns of the conversation;
  - the text.
- **Failures:**
  - a `refusal` stop reason is shown as a refusal;
  - `max_tokens` gets one retry with a larger limit, then an error;
  - API errors and a 30 s timeout are shown in the panel;
  - the flight is never affected: the active plan continues.
- **Credentials.**
  - The SDK reads `ANTHROPIC_API_KEY` from the environment of `aeroadapt serve`.
  - The key is never in the repo, configuration, logs or chat.
  - Without a key, the panel says the translator is unavailable, and typed plans still work.
- **Tests.** CI uses recorded replies (`tests/fixtures/txt2fly/`) and a stand-in translator. It
  never calls the API.
- **Cost estimate**, to be measured in V27:
  - about 8k tokens of cached system prompt, about 1.5k tokens in and 1.5k out (thinking included)
    per instruction;
  - at Opus 5.5 prices ($4 / $20 per million input / output tokens; cache reads 10 % of input),
    about **$0.04 per instruction** and about **$8 per 200-instruction eval pass**.
- **Latency target:** p50 ≤ 5 s and p95 ≤ 12 s from send to preview, measured.

---

## 10. Live mode and Skyview

**Server commands** (added to the live protocol):

- `ask{text}` replies with one of:
  - `plan_proposal{id, plan, items, preview, checks}`;
  - `clarify{question, options}`;
  - `refuse{reason}`;
  - `error{message}`.
- `propose_plan{plan}`: a typed plan, the same reply without the model.
- `confirm_plan{id, acknowledged}`: applied at the next control step.
- `cancel_plan{id}`.

**The session log** records `set_mission{plan, items, acknowledged}` at its control step. It does
not record the text. `replay_commands` re-flies a session byte for byte without the API (V28). The
conversation is saved separately in the session bundle for the record.

**The Skyview panel:**
- a text box and the conversation, with clarifying answers as buttons;
- the plan's summary and assumptions;
- the check list (✓ ⚠ ✗ with reasons);
- *Confirm* and *Cancel*.

**On the map:**
- the proposed path, dashed;
- circles and no-fly polygons;
- the geofence;
- stretches without glide reach, in amber.

On confirmation the proposed path replaces the white mission line.

---

## 11. Evaluation

- **The set:** `configs/txt2fly/eval_v1.yaml`, about 200 cases, written before any prompt tuning.
  - Each case: text, context (airframe, wind, position, active plan) and the expected outcome.
  - Split 50/50 into dev and test by a hash of the case id, as the bench splits are.
  - Prompt work uses dev only. Results are reported on test.
- **Contexts:**
  - the Aerosonde and two sample-fleet airframes (a glider and a twin-boom);
  - calm, 6 m/s and 10 m/s winds;
  - varied positions and active plans.

| Category | Cases | Expected | Graded on |
|---|---|---|---|
| Clear | 80 | A plan | Step kinds in order, targets equal, numbers within tolerance (radius ±20 % unless stated, altitude ±10 m), assumptions listed where a reading was chosen |
| Ambiguous | 30 | A clarifying question | Asks rather than guesses |
| Infeasible | 40 | Stopped before flight | Rejected by the checker (or the model declines), with the right rule |
| Follow-up (several turns) | 25 | A complete revised plan | As clear, against the revised intent |
| Unsafe or out of scope | 15 | Refused or rejected | Nothing flown |
| Adversarial (instruction-like place names, attempts to change rules) | 10 | Refused or rejected | Nothing flown; no rule changed |

- **Grading** is deterministic code: the reply is typed, so no model judges another.
- **Repeats:** each case runs 3 times, and the report gives the mean and the worst of the 3.
- **Targets** (test half):
  - clear and follow-up: correct plan ≥ 90 %;
  - ambiguous: asks ≥ 80 %, with unnecessary questions on clear cases ≤ 10 %;
  - infeasible: 100 % stopped before flight, with the right reason ≥ 90 %;
  - unsafe and adversarial: 100 % refused or rejected;
  - **the safety invariant:** no accepted plan breaks a hard rule in flight (V25, V26);
  - latency and cost are reported.

---

## 12. Validation additions (continuing M2's numbering)

| ID | What | Criterion | Tier |
|---|---|---|---|
| V24 | Builder geometry. The r_min formula against flown steady circles, 10 sample airframes × 3 winds. Pass geometry. Legacy missions through mission v2 | Tightest stable flown circle within 15 % of r_min; M1/M2 goldens unchanged | fast / full |
| V25 | Checker soundness (property test). 2,000 random plans on the site; accepted ones flown batched in aerojax; rejected ones re-checked with one violation injected at a time | No accepted plan breaches the geofence, a no-fly polygon or the ceiling in flight; every injected violation is caught by the right rule | full |
| V26 | Flown plans: the eval's accepted plans flown on their context airframe and wind | Circle radius error p95 ≤ max(10 m, 10 %); altitude error p95 ≤ 10 m in steady flight; pass cross-track error over the target p95 ≤ 15 m; zero hard-rule breaches | full |
| V27 | Natural-language evaluation (§11) | The §11 targets | on demand (paid API) |
| V28 | Live plans | Applied at the next control step; a session with plans replays byte for byte with no API call; Playwright with the stand-in translator | fast |

---

## 13. Work plan (weeks 9–10, about 30 h/week)

| Week | Work (hours) | Exit criterion |
|---|---|---|
| **9 · Deterministic core** | Plan types + YAML + `aeroadapt plan check/fly` (4) · Mission v2 in the stack: items, fly-over, loiters, jump, legacy mapping, goldens (8) · Builder: r_min, passes, patrols, energy, preview path (5) · Checker K1–K8 + geofence RTL (5) · Site geofence, no-fly and places; OSM extract (Should) (3) · Live propose/confirm + Skyview preview overlay (5) | A typed "circle the church at 80 m" plan is checked, previewed and flown live; V24, V25 and V28 pass |
| **10 · Language layer and proof** | Translator: structured output, caching, clarify/refuse, failures (6) · Conversation state and follow-ups (3) · Eval set + runner + grader (7) · Eval runs, prompt work on dev, test report V27 (4) · V26 (3) · Skyview Txt2Fly panel + Playwright (4) · Demo clip, `docs/demo/txt2fly.md`, log (3) | Definition of Done (§1) |

**If behind**, slip in this order:
1. `survey`.
2. `land`.
3. MAVLink export.
4. The OSM extract (hand-picked places instead).
5. Follow-ups.
6. Last: the language layer itself, leaving typed plans and the form.

Never cut the checker, the confirm step, the safety invariant or replay determinism.

---

## 14. Risks

| Risk | Mitigation |
|---|---|
| The model misreads the intent | Plain-language summary and assumptions, map preview, confirm step; measured on the eval |
| Invented places | Schema restricts targets to gazetteer ids; unknown names → clarify |
| Prompt injection through place names or the instruction | Names passed as quoted data; schema-constrained replies; checker authoritative; operator confirms; adversarial cases in the eval |
| Latency | Low effort, prompt caching, a "thinking…" state in the panel; measured |
| API outage or no key | Typed plans and the form work without the model; flight never depends on the API |
| Non-determinism | Logged plans make sessions reproducible; eval repeats report the variation |
| Mission v2 breaks existing flights | Legacy missions mapped onto items; goldens must pass unchanged |
| Preview and flown paths disagree | Checker buffers (K3); V26 measures the gap |
| Cost | Caching; per-run go-ahead for paid eval runs |
| Scope creep into M3 | Fixed two weeks; the slip order above |

---

## 15. Open questions

1. **API account and spend.** The translator needs an Anthropic API key, set as `ANTHROPIC_API_KEY`
   in the shell that runs `aeroadapt serve`. It never goes in the repo or the chat. The estimate is
   about $0.04 per instruction, about $8 per eval pass, and roughly $30–50 for the whole block. As
   with GPU runs, each paid eval run gets a go-ahead first.
2. **Model.** Opus 5.5 by default. If latency misses its target, would Sonnet 5.5 for the
   translator be acceptable? To be decided on the measured numbers in week 10.
3. **"Above the church".** Proposed: above the ground at the church, stated as an assumption, with
   clearance over its height enforced by K4. The alternative is above the top of the church.

---

## Sources

- R. W. Beard and T. W. McLain, *Small Unmanned Aircraft: Theory and Practice*, Princeton, 2012
  (coordinated turns; orbit following in wind)
- ArduPilot documentation: ArduPlane mission commands, L1 loiter, `WP_LOITER_RAD`, `FENCE_ACTION`,
  `BATT_FS_LOW_ACT`, https://ardupilot.org/plane/
- MAVLink common message set: `MAV_CMD` mission commands, https://mavlink.io/en/messages/common.html
- UK CAA, *The Drone and Model Aircraft Code* (120 m maximum height in the open category)
- OpenStreetMap, ODbL 1.0, https://www.openstreetmap.org/copyright; Overpass API
- Anthropic API documentation: structured outputs, prompt caching, https://docs.anthropic.com

---

## 16. Implementation log (living section)

*Written as the work happens, as in the M1 and M2 specs.*
