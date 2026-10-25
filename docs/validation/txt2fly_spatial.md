# Txt2Fly V32: does it understand the map?

*Spec: [06 — Txt2Fly](../06_txt2fly_spec.md) §17. Set: `configs/txt2fly/eval_spatial.yaml` (49
cases on four sites, split into dev and test by a hash of the id, as V27's). Graded by code on
what would be flown: the built plan's preview against the map's shapes. Each case three
times. Claude Haiku 5.5, thinking off, effort low, unless said otherwise.*

The owner's report (24 Oct 2026): "circle the whole city" drew a small circle beside the city,
and "the system doesn't really seem to understand the map". The cases were written that day,
before any change, from five kinds of request: round an area, along a line, over an area,
places by their relation to others, and routes. Five more must be asked about.

## The baseline (24 Oct 2026)

The system as it was (commit 7c9a75c): places as points with their heights, steps for circles,
passes, patrols and gotos, a planner that keeps out of every no-fly area at all heights.

| Category | Cases | Mean | Worst of the 3 | Target |
|---|---|---|---|---|
| around | 13 | 0% | 0% | ≥ 85% |
| along | 10 | 10% | 10% | ≥ 80% |
| survey | 6 | 33% | 33% | ≥ 80% |
| relative | 10 | 10% | 10% | ≥ 85% |
| route | 5 | 53% | 40% | ≥ 85% |
| ambiguous | 5 | 100% | 100% | ≥ 80% |

Latency p50 1.67 s, p95 2.64 s; $0.032 in all. (Graded on the first labels; the corrected ones,
below, are for requests no plan of that system could meet either.)

What the failures showed:

- **"Circle the whole city" was a question or a small circle.** The model had no step for
  "round the outside of an area", and an `orbit` of the city circled its middle at the usual
  160 m, which the builder then moved clear of the city-centre zone.
- **The city could not be gone round at all.** On the test map the power line runs from the
  city's edge to past the geofence. It is closed only up to 100 m above the ground, but the
  planner kept out of every zone at all heights, so no loop round the city existed.
- **Lines were followed as points.** "Follow the river" became a `goto` to the river's middle.
- **Relations were guessed.** "The field closest to Hollow Farm" and "the wood furthest from
  the airstrip" were answered by eye from the list of positions, and mostly wrongly.

## What changed

All of it is in [docs/06 §17](../06_txt2fly_spec.md). In short:

1. **What the model is told:**
   - each area's extent and size, each line's length and course;
   - each no-fly area's extent and how high it is closed;
   - a map picture, as a setting (see below).
2. **What a plan can say:**
   - `around` (enclose an area), `follow` (a line, or an area's edge) and `survey`;
   - `nearest`, `furthest` and `beyond` targets, measured by the gazetteer;
   - the geometry in code.
3. **Keeping the meaning:**
   - a circle that does not fit is widened round the place before it is moved off it;
   - a loop that cannot go all the way round says why and what can be flown;
   - zone heights are used, so plans cross the power lines at 150 m.
4. **The review:** a plan that cannot be built, or fails a hard check, goes back to the model
   once, with what would be flown and why.

## Tuning (24–25 Oct 2026)

Dev half first, then the test half once, then both together as the set is small. Every
failure was read.

| Run | around | along | survey | relative | route | ambiguous | p50 / p95 |
|---|---|---|---|---|---|---|---|
| dev 1: steps, shapes, heights, map picture, review of every adaptation | 53% | 100% | 67% | 33% | 100% | 100% | 4.7 / 10.3 s |
| dev 2: exact names, cheaper wider circles | 100% | 83% | 67% | 25% | 83% | 100% | 4.2 / 7.7 s |
| dev 3: relation targets in code | 100% | 75% | 67% | 75% | 100% | 100% | 4.2 / 6.8 s |
| test 1 | 96% | 67% | 100% | 44% | 100% | 100% | 4.2 / 9.1 s |
| all, after test 1's general fixes | 87% | 100% | 83% | 43% | 100% | 100% | 4.1 / 7.8 s |
| all, "Not a question" rules, landing fields | 97% | 97% | 100% | 67% | 100% | 100% | 4.2 / 8.8 s |

Every run in this table sent the map picture (then the default) and reviewed every adaptation
above a cost; their latency is mostly the picture's (see below).

What the failures showed, and what changed (general rules and code, not fixes for one case):

- **A loop must bend, not cut across.** A loop corner too close to another no-fly area was
  dropped, and the route round it doubled back ("goes 0.16 of the way round"). *Code:* the
  corner moves out from the area, or in towards it, whichever is shorter.
- **The way in to a circle round a no-fly area went to its centre**, inside the area. *Code:*
  the route joins the circle at its point nearest the aircraft.
- **An endless circle's preview stopped short of a full turn in wind** (2πr ÷ airspeed). *Code:*
  it stops after a full turn of bearing round the centre.
- **Lines and roads near the geofence were cut inside**, because every point kept a route's
  room to turn round. *Code:* along a line, the margin and room for gentle bends; every clear
  stretch is flown, with routes between.
- **Routes had too many legs** (25 and 29 waypoints round the city). *Code:* searched with
  30 m more room first; a route leaving a tight spot keeps the clearance the spot has.
- **Survey lines between turbines were dropped whole.** *Code:* each line's clear stretches,
  with routes between; only the outer ends keep room to turn.
- **Haiku guessed distances and bearings** (field G as "closest to Hollow Farm"; the far side
  of the city at bearing 0 or 170 instead of 336). *Code:* `nearest`, `furthest`, `beyond`.
- **Needless questions**: "Long Field" (a field's other name), "the woods nearest the wind
  farm", "the landing field furthest from home". *Rules:* a name or other name that matches
  exactly is that place; words that single one out (nearest, north of, the eastern one) are
  not a question; the landing fields are the safe zones. *Schema:* "field" and "landing field"
  are kinds the model can name.
- **After a loop proved impossible, "circle the village" became a question.** *Rule:* asked to
  "circle" it, circle over its middle; asked for "the whole" of it, ask.

### Labels corrected (with the reason, in the file)

- **around-003, -007, -009, -013** ("circle around the wind farm", "a big loop round the
  western woods", "all the way round Hanger Wood", "the whole of Coton village"): the wind
  farm's eastern turbines stand on the geofence, the western woods run 150 m past it, Hanger
  Wood lies mostly north of it, and Coton runs out past it and almost touches the M11. No loop
  can go all the way round any of them. Now: a question, or a plan along the edge that can be
  flown.
- **rel-005** ("the field closest to Hollow Farm"): field B is closest by its middle (359 m),
  field E by its edge (85 m). Both accepted.
- **survey-006** ("search the eastern woods back and forth"): the world names two woods "the
  eastern woods" (the bake adds "(10 ha)" to the smaller one to tell them apart). Asking is
  right too.

**Method note.** The test half was read once, after test 1; the fixes that followed are general
(code, rules, the schema) and the labels above, each with its reason. The test half is no longer
fully held out, so the final runs below are on the whole set.

## Two things that did not work

**The map picture.** Sent with every request (cached), it was measured three ways:

| Measured | Without the picture | With it |
|---|---|---|
| Spatial set, all categories (25 Oct, same code) | around 97 %, along 93 %, survey 100 %, relative 77 %, route 93 % | 100 %, 100 %, 100 %, 77 %, 100 % |
| Spatial latency p50 / p95 | 1.91 / 4.03 s | 3.93 / 7.80 s |
| V27's test half, final code | every target met | adversarial 75 % (target 100 %) |
| V27 latency p50 / p95 | 1.98 / 2.81 s | 2.83 / 4.67 s |

On the spatial set it is ahead by four attempts in about a hundred, within run-to-run noise
(the same setting moved by more between runs). It costs about 2 s a request, though the image
is cached (about 1,400 tokens read from the cache each time). On V27's requests, Haiku trusted
its few labels over the full list:
- it refused the notice board (3 of 3) and asked about "Middle Field" (2 of 3), which is listed
  but not labelled;
- in an earlier ablation, it refused "the London Eye" rather than asking.

A prompt paragraph about the picture ("names come from the list") made Haiku refuse unknown
places even with no picture sent. The picture is therefore a setting (`haiku-map`, also for
`aeroadapt serve --translator`), off by default.

**Reviewing every adaptation.** The first review loop sent the model every plan the builder
had changed beyond a cost (a widened circle, a moved one, a loop instead of a circle). Haiku
then asked the operator about changes that keep the meaning: "the builder widened it to 380 m:
shall I?", and "the circle round the church becomes a loop round the village: is that
acceptable?". On V27's test half, clear instructions fell to 82 % (needless questions 15 %) and
follow-ups to 61 %. The review now runs only when a plan cannot be built or fails a hard check,
which is where the model can mend something: a loop that cannot go round becomes a circle over
the middle, or a question; a height the checker suggests is taken. What the builder changed to
keep a plan within the rules is in its notes, for the operator, as before.

## V27 kept (25 Oct 2026)

The new rules, steps and schema were checked against V27's 200 cases, tuned on the dev half
only. The test half was read once more, after the regression above, and the changes that
followed are general:
- copy numbers as given, even outside a limit;
- names that read as instructions are still places;
- following anything that moves is out of scope;
- a proposed but unconfirmed plan is changed like an active one;
- "much wider" is × 2;
- a distance and direction from the aircraft is an offset from here;
- the grader counts a translation as right when its first answer was, and the review changed
  only what a check asked for.

Code: a reply that is not a valid plan goes back once with the validation error (a patrol with
no places, about 1 in 10 for one phrasing, after the target object grew a field).

| Category | Target | Haiku 5.5, thinking off (test half) |
|---|---|---|
| clear | ≥ 90 % | **98 %** (worst 98 %), needless questions 2 % |
| ambiguous | ≥ 80 % | **90 %** |
| infeasible | 100 % (right reason ≥ 90 %) | **100 %** (93 %) |
| follow-up | ≥ 90 % | **100 %** |
| unsafe | 100 % | **100 %**, all refused |
| adversarial | 100 % | **100 %** |
| latency p50 / p95 | 2 s / 4 s | **1.98 s / 2.81 s** |
| cost | | $0.101, $0.00031 per instruction |

## The final run (25 Oct 2026)

The whole set (the test half is no longer held out: see the method note), three repeats.

| Category | Cases | Target | Haiku 5.5, thinking off (the default) | Haiku 5.5, adaptive thinking |
|---|---|---|---|---|
| around | 13 | ≥ 85 % | **100 %** | **100 %** |
| along | 10 | ≥ 80 % | **100 %** | **100 %** |
| survey | 6 | ≥ 80 % | **100 %** | **100 %** |
| relative | 10 | ≥ 85 % | 67 % (worst 50 %) | **100 %** |
| route | 5 | ≥ 85 % | **87 %** (worst 60 %) | **87 %** |
| ambiguous | 5 | ≥ 80 % | **100 %** | **80 %** |
| latency p50 / p95 | | | 2.11 s / 4.38 s | 2.60 s / 5.06 s |
| cost | | | $0.055 | $0.064 |

From the baseline: around 0 → 100 %, along 10 → 100 %, survey 33 → 100 %, relative 10 → 67 %
(100 % with thinking), route 53 → 87 %. "Circle the whole city" now flies a loop round the city
(crossing the power lines about 50 m above the height they are closed to), and "circle the city"
flies a 1,020 m circle round it.

**What still fails** (every failure read):

- **Relative, without thinking: questions where the code would measure.** Haiku asks "which
  woods? several are near the wind farm" (3 of 3), "which bridge?" for "the bridge nearest the
  Shard" (3 of 3), "which field?" for "the field closest to Hollow Farm" (2 of 3). The plan
  would have a `nearest` target, which the gazetteer resolves exactly. Once it took the
  tallest "tower" instead of the tallest place, missing the Shard (a "landmark"). With thinking,
  every relative case passes.
- **"Go to Cuckoo Wood by way of the hangar"** became "go to the hangar and wait" (1 of 3; 2 of
  3 with thinking), dropping the destination, though the prompt has the same example.
- **"Fly to the wind farm, then the power station, then home"** stopped at the wind farm once.
- **"Fly along the lane"** (Wrenfold): with thinking, Mill Lane each time, the one place of kind
  "lane" (the rule says so); but Church Lane is a lane too in all but its map tag, and the set
  says ask. The rule and the label disagree; the label stands.
- **Not graded but seen:** at 150 m, following the wind farm's edge fails K4 where the ridge
  climbs faster than the aircraft; the checker says so and suggests 315 m, and at 350 m it
  passes. The eastern woods' survey covers 46 % of the wood (needed 60 %), its lines broken by
  turbine zones closed up to 234–400 m; flying it above them is the way, not yet offered.

**The default stays Haiku without thinking** (the owner's choice, for latency): it meets every
V32 target but relative positions, and V27's. With adaptive thinking it meets them all on V32
(ambiguous at the target exactly) for about 0.5 s more at the median; its V27 results (23 Oct)
also met every target. Switching is a one-word change (`--translator haiku-think`), and the
owner's call.

**25 Oct 2026: the owner switched the default to `haiku-think`.** `serve` (auto) and
`aeroadapt txt2fly ask|eval` now use Haiku 5.5 with adaptive thinking; `--translator haiku` is
the faster setting without it.
