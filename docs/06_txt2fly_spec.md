# Txt2Fly Technical Specification: Tell It Where to Go

*Status: draft · 21 Oct 2026 · Parent plan: [03 — AeroAdapt-UAV](03_uav_architecture_and_six_month_roadmap.md) §7.2 (weeks 9 and 11, inserted on 21 Oct 2026; week 10 became the synthetic world, [07](07_world_and_skyview_spec.md)) · Previous milestone: [05 — M2](05_m2_technical_spec.md)*

**Goal.** Let an operator say where the aircraft should go and what it should do, in plain words:

- "go to the church and fly a 30 m circle 80 m above it";
- "fly by field A";
- "patrol fields A, B and C".

The system turns the words into a flight plan for the airframe actually flying, in the wind
actually blowing. It shows the plan on the map, says plainly what it cannot do and why, and,
once the operator confirms, flies it. A follow-up ("make it wider", "now go to B") changes the
plan in flight, and the autopilot brings the aircraft onto the new path.

**Weeks 9 and 11, about 30 h/week (~60 h).** Week 10 went to the synthetic world and the
four-panel Skyview ([07](07_world_and_skyview_spec.md)). M3 starts in week 12, with M2's two
carry-overs: auto-configuration at 90 % (D6) and the Rascal (D4).

**Out of scope** (later milestones):
- terrain following: heights are above home. On a world site ([07](07_world_and_skyview_spec.md) §7)
  the planner keeps the whole track clear of the ground and of trees and buildings, raising legs
  where needed; on a flat site the ground is flat;
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
3b. **Always a flyable plan** (the owner's request, 21 Oct 2026). The builder treats the request
   as the goal. It searches for the closest version that keeps every rule and says what it
   changed: a tighter circle the aircraft can hold, a height inside the band, a route round a
   no-fly area, the nearest allowed place to one inside it. Hard rules are never relaxed;
   only what nothing can fix (a battery too small for the plan) is refused.
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

- **Site file** (the operator's choices, like the M1 safe zones):
  - `geofence_ne_m` (a polygon), `ceiling_agl_m` (120 m, as the UK open category), `min_agl_m`
    (40 m);
  - `no_fly` areas, each with a name and a reason: the M11 corridor, Coton village and the
    West Cambridge campus, the ground-risk features in the site description. They are areas
    of the OSM extract chosen by id: the motorway 50 m either side, the built-up area 30 m
    around it, the campus as mapped, each merged into one outline (or drawn by hand);
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
- The run-in extends one turn radius (at least 100 m) beyond the target's near edge and the
  run-out half that beyond its far edge: L1 is on the line before the target (V26).

**Patrols.**
- The places are visited in the order given; the operator's order is part of the intent.
- Each visit is a fly-over (or, with `visit: goto`, a fly-by) of the place on the route from
  the one before, so the pattern stays compact.
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
- A **preview**: the mission flown by a point-mass aircraft with the stack's own item
  sequencer and L1 guidance, in the current wind. Its bank follows the guidance's demand with
  the roll loop's time constant and rate limit, and its height moves within TECS's limits.
  The checker and the map both use it.
- The flown track differs from the preview (gusts, navigation error, the full dynamics); V26
  measures by how much.

**Realising the request** (week 9, revised the same day so that a request always yields a
flyable plan where one exists):

- **Each step has options**, the request first and then nearby versions:
  - circles: the radius asked (or the tightest the aircraft can hold, if tighter was asked),
    a larger and a tighter one, the centre moved while the place stays inside the circle, the
    other way round;
  - passes: along the place either way, across it, at a slant; the middle, or a point inside
    the place with room to turn; full runs, shorter runs, or a pass over part of a large place;
  - patrols: the order given or reversed, visits over each place or by it (turning early),
    places moved for room;
  - heights and airspeeds outside the allowed band are brought inside it;
  - a place inside a no-fly area or beyond the geofence: the nearest reachable place clear
    of it, with more and more room to turn.
- **Routes** go round no-fly areas (A* on the site's clearance map, pulled taut), on the way
  to a step, between a patrol's places and on the way home.
- **Turns are made certain.** A waypoint is added where a path meets a circle (a loiter, or RTL's
  circle over home), already going its way round; a turn point fixes the side where the route
  doubles back; waypoints closer together than 1.5 acceptance radii are merged. A left and a
  right turn are otherwise nearly as short, and the aircraft and the preview could take
  different ones (V25 found tracks 300 m apart before this).
- **The preview is conservative:** it banks 5° short of the roll limit, so its turns are a little
  wider than the aircraft's.
- **Beam search.** For each step all options fly at once in the preview (`vmap`), from where the
  previous step left the aircraft; the best three partial plans go on. Clear of every no-fly
  area by the margin plus 10 m comes first, then closest to the request, then quickest. The six
  best complete plans are then flown in full, in order, until one keeps clear. If none does,
  the search runs again asking for twice, then three times, the room to turn.
- **Every change is a note** the operator reads before confirming ("circle field C: centred 84 m
  north-west of it (still over it), to keep clear of M11 motorway (no-fly)").
- `build(..., adapt=False)` takes a plan literally; the checker's tests use it to show every
  rule catching its violation.

---

## 7. The checker

Every rule returns pass or fail with a reason in plain words and, where one exists, the nearest
plan that would pass. Hard rules cannot be overridden by the plan, by text or by the model. They
change only in the site and stack configuration files.

| Rule | Check | Kind | Example reason |
|---|---|---|---|
| K1 | Every target resolves; the plan expands to ≤ 32 items | Hard | "There is no place called 'the old mill' on this site." |
| K2 | Every circle and turn is at least r_min (§6) for the commanded speed and the current wind | Hard | "A 30 m circle needs 73° of bank at 25 m/s in this wind; this aircraft is limited to 45°. The tightest circle now is about 120 m." |
| K3 | The preview stays 30 m (the tracking margin) inside the geofence and away from every no-fly area | Hard | "The circle would cross the M11 corridor (no-fly)." |
| K4 | min_agl ≤ altitude ≤ ceiling; at least 30 m above any known feature height under the path. On a world site, along the whole preview: at least min_agl above the ground, at most the ceiling above it, 30 m above every tree and building (docs/07 §7) | Hard | "'Circle Hanger Wood' would pass 12 m above Hanger Wood; keep 30 m clear." |
| K5 | Airspeeds within the airframe's configured range | Hard | "This aircraft cannot fly slower than 11 m/s." |
| K6 | Predicted energy ≤ remaining energy − reserve (30 % of capacity); open-ended plans must be bounded (`until_reserve`) | Hard | "This plan needs about 140 % of the remaining battery." |
| K7 | Glide reach: every 50 m along the path, at least one safe zone is reachable in the current wind, by the contingency's own test (`assess_zones`) | **Soft** | "From the far side of the circle, no field is within glide." Uncovered stretches are drawn in amber |
| K8 | The join from the current position passes K3 and K4 | Hard | "The way there crosses Coton village (no-fly)." |

- The 30 m tracking margin rests on measurement: in V26 the flown track strayed at most 24 m
  from its preview (the Aerosonde in a 6 m/s wind; at most 14 m for the fleet aircraft).
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

- **The model: Claude Haiku 5.5** (`claude-haiku-5-5`), the owner's choice (21 Oct 2026), because
  latency comes first: commands should feel immediate.
  - It is called through `AsyncAnthropic`, since the server is asyncio.
  - Thinking is off by default; the API allows that at low effort. The eval also runs adaptive
    thinking at low effort, and one Opus 5.5 pass as a reference for how much a larger model would
    add (§11).
- **Designed for a small model.** The model's job is narrow:
  1. choose the step types;
  2. pick places from an enum;
  3. copy numbers and directions from the text;
  4. ask when the text leaves something open.

  Geometry, physics and safety all live in code. Relative positions ("200 m north of the church")
  are resolved by the builder, not by the model, and the cached prompt carries worked examples of
  every step type.
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
  - about 8k tokens of cached system prompt, about 1.5k tokens in and 0.5k out per instruction;
  - at Haiku 5.5 prices ($0.10 / $0.50 per million input / output tokens for prompts under 100k
    tokens; cache reads 10 % of input), about **$0.0005 per instruction**;
  - about **$0.30 for the eval** (200 cases × 3 repeats);
  - the one Opus 5.5 reference pass is about $8 and gets a go-ahead first.
- **Latency target:** p50 ≤ 2 s and p95 ≤ 4 s from send to preview, measured. Confirming is one
  key press (Enter).

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
- **Configurations:**
  - Haiku 5.5 with thinking off (the default);
  - Haiku 5.5 with adaptive thinking at low effort;
  - one Opus 5.5 pass, as the reference.

  The report gives accuracy and latency for each, so the cost of a smaller model is measured, not
  guessed.
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

## 13. Work plan (weeks 9 and 11, about 30 h/week)

| Week | Work (hours) | Exit criterion |
|---|---|---|
| **9 · Deterministic core** | Plan types + YAML + `aeroadapt plan check/fly` (4) · Mission v2 in the stack: items, fly-over, loiters, jump, legacy mapping, goldens (8) · Builder: r_min, passes, patrols, energy, preview path (5) · Checker K1–K8 + geofence RTL (5) · Site geofence, no-fly and places; OSM extract (Should) (3) · Live propose/confirm + Skyview preview overlay (5) | A typed "circle the church at 80 m" plan is checked, previewed and flown live; V24, V25 and V28 pass |
| **11 · Language layer and proof** (was week 10) | Translator: structured output, caching, clarify/refuse, failures (6) · Conversation state and follow-ups (3) · Eval set + runner + grader (7) · Eval runs, prompt work on dev, test report V27 (4) · V26 (3) · Skyview Txt2Fly panel + Playwright (4) · Demo clip, `docs/demo/txt2fly.md`, log (3) | Definition of Done (§1) |

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
| A small model (Haiku) misreads more often than a large one | Narrow job and tight schema (enums for places and step types); worked examples in the prompt; geometry in code; the Opus reference pass measures the gap |
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

1. **API account.** The translator needs an Anthropic API key, set as `ANTHROPIC_API_KEY` in the
   shell that runs `aeroadapt serve`. It never goes in the repo or the chat. With Haiku 5.5 the
   spend is small (about $0.30 per eval). The Opus reference pass (about $8) gets a go-ahead first.
2. **Model** (resolved 21 Oct 2026). Haiku 5.5, for latency. If it misses the accuracy targets, the
   eval shows by how much and on which categories, and moving up a model is then the owner's choice.
3. **"Above the church"** (resolved 22 Oct 2026 for now). Plan heights are above home, the frame
   the autopilot flies; on a world site the planner keeps 40 m above the ground and 30 m above
   the church (and everything else) along the track, and says when it had to fly higher. "80 m
   above the church" is the model's to turn into a height above home (the gazetteer gives the
   church's ground and height). Heights above the ground below need terrain following, a later
   step ([07](07_world_and_skyview_spec.md) §12).

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

### Week 9 (21 Oct 2026): the deterministic core

**Built**

- **Mission v2 in the stack** (`stack/types.py`, `stack/autopilot.py`):
  - an item table (32 × 8): WAYPOINT (fly-by), FLYOVER, LOITER_TURNS / LOITER_TIME /
    LOITER_UNLIM (turns and time count from capture; the loiter ends heading for the next
    item), JUMP (with a repeat count), RTL, and a per-item airspeed;
  - failsafes: leaving the geofence (16 vertices) or climbing above its ceiling → RTL; the
    battery monitor's consumed charge reaching the reserve → RTL (the Should from §8);
  - `make_mission` builds the M1/M2 waypoint missions as before; `ap_init` starts a new
    mission from where the aircraft is; missions serialise for session logs.
- **Plans** (`txt2fly/plan.py`): the §4 steps and targets; example plans in
  `configs/txt2fly/plans/`.
- **The site's operating area** (`configs/sites/cambridge_west.yaml`, new `operations` block):
  - a geofence over the farmland west of the M11 (2.1 × 1.6 km), a 120 m ceiling, a 40 m
    minimum, and home over field A (so the RTL circle stays clear of the M11);
  - no-fly areas from the OpenStreetMap extract (`cambridge_west.osm.json`): the M11 with
    50 m either side, Coton's built-up area with 30 m around it, houses on Madingley Road,
    the West Cambridge campus;
  - 48 places: the four safe zones, 20 named fields, woods, farms, the church and others.
- **`tools/fetch_gazetteer.py`**: one Overpass query; named features to local metres; no-fly
  candidates merged into outlines with shapely (for this tool only).
- **Builder and checker** (`txt2fly/builder.py`, `checker.py`, `energy.py`): §6 and §7, with
  the preview and the adaptation described there; plain-language reasons and suggestions.
- **Live mode:** `propose_plan`, `confirm_plan` (rebuilt and re-checked from the state at that
  moment, then flown from the next chunk), `cancel_plan`; the session log records the mission
  a plan became, so a replay never rebuilds anything. The server compiles the preview at
  start-up.
- **Skyview:** a Plan tab next to Faults (example plans, an editor, the verdict, the steps,
  the builder's notes, the check list, warnings to acknowledge, Confirm or Enter); on the map,
  the no-fly areas and geofence, the proposal's track (dashed) with the points where checks
  fail, and the plan being flown in place of the mission line; the OSM attribution.
- **CLI:** `aeroadapt plan check <plan>` and `aeroadapt plan fly <plan>` (a replay bundle).
- **Tests:** unit tests for the geometry, gazetteer, builder and every rule; mission items;
  V24, V25, V26 (full tier) and V28; a Playwright test of the plan flow; import contracts (the
  planner never imports faultlab; the stack never imports the planner).

**Results**

- **Existing flights unchanged:** the M1 golden flights pass. Flown before and after, the
  nominal and servo-jam flights are bit-identical; the motor-cut flight differs by at most
  1.3e-10 in float64 (XLA fuses the larger graph differently; in float32 a one-ulp difference
  appears after a phase change and grows).
- **Energy:** the model gives 8.9 A for the Aerosonde at 25 m/s; the simulator draws 8.7 A over
  the M1 box mission, turns included.
- **V26, preview against flight** (Aerosonde, a twin-boom and a glider from the sample fleet;
  calm and 6 m/s; the three example plans): the flown track strays at most 24 m from the
  preview (Aerosonde), 14 m for the fleet aircraft; nothing flown comes near a no-fly area.
  With an instant-bank preview it was 44 m, which is why the preview has roll dynamics.
- **V24:** at r_min the three aircraft hold the circle within 15 %, calm and in 6 m/s; at half
  of it they fly wider.
- **V25:** 200 random plans; 68 accepted (refusals: K3 110, K8 51, K2 19; a plan can fail
  several). All 68 flown at once for 300 s: the closest any came to a no-fly area or the
  geofence's edge was 16 m, and the highest was 124.4 m (the fence is at 140 m). Every
  injected violation (a ceiling breach, a detour over the village) was caught by its rule.
- **V28:** a confirmed plan starts with the next chunk; the session replays byte for byte;
  unacknowledged warnings and plans that no longer pass are refused.
- **Speed:** a plan is built and checked in 0.2–0.5 s on average, about 1.2 s at worst, once
  the preview is compiled.
- **The Aerosonde over this site:** from the 24 points of its waiting circle, the patrol of
  fields A, B and C passes in both test winds; the fly-by of field A passes from all 24 in an
  easterly but only from 12 in a 6 m/s westerly. From the other 12 the aircraft is heading
  for the M11, and its 117 m turns cannot avoid the buffer; the checker says so. The fleet
  aircraft (13–15 m/s) fit the fields easily.

**Deviations from the spec**

1. **The preview flies the stack's guidance** instead of drawing turns at r_min (§6). K3's
   widening of r_min/2 + 20 m became a fixed 30 m margin, set from V26.
2. **The builder adapts** pass directions and crossing points to the site (§6), which the spec
   did not plan; the site is tight for a 25 m/s aircraft.
3. **Patrols fly over the places on the route** between them instead of making long passes.
4. **Pass run-ins are one turn radius**, not two: L1 is on the line in time (V26).
5. **No-fly areas come from OpenStreetMap** rather than being drawn by hand; the site file
   chooses them.
6. **Not yet done:** MAVLink export (T9, Should), the glide landing (T10, Should) and the
   survey (T12, Could). The language layer, the evaluation and the demo are week 10.

**Definition of Done status (after week 9)**

| # | Status |
|---|---|
| T1 CI | ✅ when the week-9 push is green |
| T2 Plan, builder, checker | ✅ (V24, V25) |
| T3 Mission v2 | ✅ (goldens, V24, V26) |
| T4 Live plans | ✅ (V28) |
| T5 Gazetteer | ✅ site and OSM |
| T6 Language layer | ⏳ week 10 |
| T7 Evaluation | ⏳ week 10 |
| T8 Skyview panel | ✅ for typed plans; the text box is week 10 |
| T9 MAVLink export | ⏳ (Should) |
| T10 Glide landing | ⏳ (Should) |
| T11 Demo and write-up | ⏳ week 10 |
| T12 Survey | ⏳ (Could) |

### Week 9, follow-up (21 Oct 2026): always a flyable plan

The owner tried the live panel and found that most proposals were refused: the builder took
requests literally (a 30 m circle, a height above the ceiling, a field beside the M11) and
adapted little. It now realises each request as closely as the rules allow (§6, *Realising the
request*), and says what it changed.

**Built**

- `txt2fly/routing.py`: the site's clearance map (10 m grid, within 3.5 m of the exact
  distances) and routes round no-fly areas (A*, pulled taut; only reachable places count,
  since a pocket between the M11 and the campus is clear but cut off).
- `txt2fly/builder.py` rewritten: options per step, routes, a batched beam search (`vmap`),
  finalists flown in full, a retry with more room to turn, and the turn-certainty pass
  (circle joins, reversal turn points, merged waypoints).
- The stack: a loiter whose next item lies inside its circle now ends once its turns are done
  (as ArduPlane's `verify_loiter_heading`); before, the course never pointed at the item and
  the loiter never ended. The planner's preview found it.
- Skyview: the planner's notes under "How it will fly it".

**Results**

- **Requests that come back flyable:** 300 of 300 random requests (Aerosonde and a fleet
  twin-boom, calm and 6 m/s, from points of the waiting circle; places anywhere in the
  gazetteer, including inside no-fly areas and beyond the geofence). Before: 33 of 60.
- **The panel's examples:** 360 of 360 (five plans × 24 start points × three winds up to
  10 m/s).
- **In flight (V25):** the 200 random plans, flown in the full simulator, keep at least 29 m
  from every no-fly area and the geofence's edge. On the way there, V25 found three ways a
  flight could leave its preview by up to 300 m, all now closed:
  1. near-reversals (the aircraft and the preview turning opposite ways);
  2. joining a circle the wrong way round;
  3. clusters of waypoints passed within a second.

  The flown track now stays within 70 m of its preview (41 m at the 90th percentile), the
  preview's turns being slightly wider by design.
- **Speed:** about 1 s per request on average, 2.2 s at the 95th percentile, 7 s at worst
  (a retry with more room).

**Deviations from the spec**

- §6: the builder adapts the request instead of the checker refusing it; the checker is
  unchanged and still judges the final mission. `build(..., adapt=False)` keeps the literal
  reading for tests.
- §6: the preview banks 5° short of the roll limit (conservative), and V26 now checks
  clearance in flight rather than the distance from the preview.

### Week 10 (22 Oct 2026): the synthetic world reaches the planner

Week 10 went to the synthetic world and the four-panel Skyview ([07](07_world_and_skyview_spec.md));
the language layer moves to week 11. For Txt2Fly:

- `txt2fly-demo` now flies over **Wrenfold Vale**, whose gazetteer and no-fly areas (the
  village, the farmyards, the radio mast) come from the world. The Cambridge demo is kept as
  `txt2fly-cambridge`, and V24–V28 run there unchanged. Example plans live per site,
  `configs/txt2fly/plans/<site>/`; Skyview's example buttons follow the site.
- **Heights over terrain.** The builder raises legs that would pass under 40 m above the ground
  or 30 m above trees and buildings (within the 30 m margin), to the lowest multiple of 5 m that
  clears, and explains it ("circle Hanger Wood: at 80 m (heights are above home) to keep 30 m
  above Hanger Wood"). K4 checks the whole preview on a world site. V31: a circle of Hanger
  Wood asked at 50 m is lifted and passes; built literally it fails K4 with the wood named;
  flown, three plans keep more than 20 m above every obstacle and 35 m above the ground.
- **Bug fixed.** A turn point the builder inserts before a return-home circle took height 0 m
  when the leg before it was a circle, so the aircraft dived on the way home (unnoticed on flat
  Cambridge). It now keeps the circle's height.

