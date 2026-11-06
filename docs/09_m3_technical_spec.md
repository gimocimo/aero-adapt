# M3 Technical Specification: One Brain, Many Airframes

*Status: decisions settled (§0), after a three-reviewer check · 6 Nov 2026 · Parent plan: [03 — AeroAdapt-UAV](03_uav_architecture_and_six_month_roadmap.md) §4.2, §5 and §7.2 · Previous milestone: [05 — M2](05_m2_technical_spec.md)*

**Goal of M3.** One frozen learned flight controller flies airframes it never trained on, in an
independent simulator too, and recovers from a hard-over where one is physically recoverable.
The milestone headline (doc 03 §7.2) is "the same frozen policy flies six airframes it never
trained on, including recovery from a hard-over". The evidence is an A/B report with
confidence intervals on the generalisation splits G2–G4, against the stock autopilot with its
classical contingency (B1) and a classical expert controller (B2: INDI with adaptive
allocation). The names follow doc 03 §5.4; the M2 bench code uses B1 and B2 differently (§1).

**Weeks 12–17 in doc 03 (amended 6 Nov 2026), about 30 h/week.** Week 12 went to M2's carry-overs, both done and
logged in [05 §14](05_m2_technical_spec.md): M2 D6 (auto-configuration: V20 at 95.1 %, from
85.5 % at the end of M2) and M2 D4 (the Rascal flies the M1 mission in JSBSim from its
datasheet configuration). (Txt2Fly's map work is also
logged as week 12, 24–25 Oct, in [06 §14](06_txt2fly_spec.md).) Costed line by line, the plan
is about 159 h [D] (§11), against about 120 h in the four weeks doc 03 first gave M3. The owner
chose one more week (§0.6): five weeks, about 150 h, with the cut list closing the rest.

This spec builds on a literature review (six topic reports and a synthesis, 6 Nov 2026), in
[`docs/research/m3/`](research/m3/README.md). Claims marked **[U]** come from secondary summaries or memory
and are checked before they are quoted in a report. **[D]** marks derivations and estimates
made for this plan, by the review or in this spec (the hours in §11 included), which need a
numerical check.

**Out of scope for M3** (later milestones):
- Sentinel, the learned fault diagnoser, and the EKF (M4). Sensor faults (pitot, GNSS,
  magnetometer) are not in L1's training or headline: §3.6.
- The self-model L3 (M4), Captain and Pathfinder (M5). L1's auxiliary head (§3.4) is a first
  step towards L3, not L3.
- Doc 03's B2 also has EKF-based fault detection and isolation and ground-risk-aware site
  selection. M3's B2 is its inner loops only (§5); the rest belongs to M4 and M5.
- Doc 03 §4.2's third L1 baseline, L1 adaptive control. It is not built in M3, so M3's A/B
  covers B1 and B2 only. It moves to M6's campaign as a Could (§13, question 7), and doc 03
  §4.2 is amended.
- ArduPilot SITL, the hero flight, the Flight Test Center (M6).
- Handling-qualities fits (equivalent systems against dynamically or Froude-scaled limits;
  unscaled MIL-F-8785C limits do not apply to small UAVs, Foster 2004): M6.
- Spins and deep stall. The post-stall tables are approximate, and M2 does not claim
  stall-recovery fidelity (M2 §12; 05 §14 week 6, known approximations).

---

## 0. Decisions (settled 6 Nov 2026)

These shape the milestone. On 6 Nov 2026 the owner approved the GPU cap with a checkpoint,
chose RunPod as the provider, and asked for the recommendations to be taken on the rest. Each
item opens with its decision; §13 lists the answers in one place.

1. **Split the control surfaces before any learning.** *Decided: split, in week 13.*
   - aerojax lumps the surfaces into three channels (aileron, elevator, rudder:
     `sim/aerojax/actuators.py`, `faultlab/slots.py` `SURFACE_INDEX`). With three surfaces for
     three moments, allocation is a matrix inverse with clipping: nothing can be re-allocated.
   - A full aileron hard-over is probably unrecoverable on most airframes with lumped ailerons
     [D]. On the Rascal the rudder, through sideslip and dihedral, cancels only about 24 % of
     it; on the Aerosonde about 33° of sideslip would be needed (linear, quasi-steady
     estimates). With split ailerons, the other aileron and the rudder can fight it.
   - Without the split, "adaptive allocation" and the hard-over headline are both empty, and so
     is the "jammed aileron re-allocated to rudder and the other aileron" of doc 03 §4.2, which
     Flight 0 relies on (§6.1).
   - Cost: about 20 h in week 13, 3–4 days (§11). The three channels are hard-coded in about two
     dozen modules (§9), and ADR-004 is amended. Each pair's existing lumped derivatives are
     pinned (§2), so the old model is reproduced exactly through a fixed mixer and M1 and M2
     results stand (V33).
   - Fallback (Gate A): keep three channels. D2 is then waived and recorded as not done
     (fallback taken), D3 becomes "effectiveness-aware inversion with fault-aware bounds",
     stated plainly, and M3 can still be done on the other deliverables; the report says so. The headline fault
     becomes a rudder hard-over on airframes with a rudder. The X8 and the Zagi have no rudder,
     and a single-elevon jam cannot be expressed in three channels, so they fly criterion (i)
     of §8.4 only, and the headline names them.
2. **G3 as two case studies, not a class.** *Decided: the X8 and the Zagi.* Doc 03 holds out "the whole
   flying-wing class". The generator has no flying-wing class, and M2 found the buildup method
   under-predicts a flying wing's directional derivatives (M2 §14, V18). M3's G3 is the X8
   (Gryte et al. 2018, through PyFly's parameter file) and the Zagi (Beard & McLain). Neither
   exists as a flyable airframe yet: building both, with B1 support for a rudderless airframe,
   is about 12 h (§7, §11). If the Zagi is not built by the pre-registration tag, the
   pre-registration names a third G2 airframe instead. Procedural flying wings and a full
   leave-one-class-out rotation move to M6. The Zagi's coefficients are checked against Beard &
   McLain's appendix if the owner has the book, otherwise against the authors' published files
   [U]; with neither, cut 3 (§11) is taken early.
3. **The headline's hard-over, chosen by rule, per airframe (§8.4).** *Decided: as below.* A trim-reachability
   oracle (V41) picks the first of a single-aileron hard-over, a rudder hard-over and a
   half-travel elevon or ruddervator jam that the airframe can still trim against with a 10 %
   roll margin. The single-aileron candidate is skipped on the Aerosonde and the Rascal, whose
   aileron data lack the symmetric term (§2). A half-travel jam is reported as a partial jam,
   not a hard-over. The headline then reads "... including recovery from a hard-over on every
   airframe where one is physically recoverable", and the report gives the counts and names the
   partial jams.
4. **Pre-registration.** *Decided: yes, tagged in week 14.* Splits, hypotheses, scenario seeds, the six airframes and
   their hard-overs are committed and tagged before the teacher first trains on the fleet
   (ladder rung 2) and before B2 is tuned, in week 14 (§8.1, §11). The tag needs the X8, the
   Zagi, the oracle and V20 on the whole fleet first; all four are now costed.
5. **GPU budget.** *Decided by the owner: a cap of $400 for M3, on RunPod, with a checkpoint.*
   - The estimate is about 12–57 GPU-hours [D, pending D9], about $17–130 at Sep–Oct 2026 A100
     and H100 marketplace prices of about $1.4–2.3/h [U: secondary trackers, which moved by more
     than 20 % between August and October]. The cap leaves room for a GPU that only meets M2's
     1M-control-steps/s target, which is 0.5M policy steps/s for the simulator alone: if the
     networks and the update halve that, the 103B steps take about 114 GPU-hours, $160–260
     [D]. It also covers idle time and retries (§3.8).
   - The checkpoint: the first spend (about $5–15) runs M2's GPU notebook, which closes M2 D9,
     and then times the full training step (D9). The plan is re-costed from the measured
     throughput, and if it comes to more than about $300, the owner decides before any further
     spend.
   - The owner tops up the RunPod account's credit; with automatic top-up off, the balance
     also enforces the cap. If the cap binds: L1 keeps five seeds (ten only with money to
     spare, §8.3), the trained ablations are cut to A4 and A5 (D7 amended, as in §12's GPU row),
     and leave-one-class-out goes (cut 2).
6. **The schedule (§11).** *Decided: (a), five weeks, weeks 13–17; doc 03 §7.2 amended on
   6 Nov 2026, so the plan is 30 weeks.* Cut 1 is decided at Gate B from the hours actually
   spent; cuts 2–4 come first. Costed line by line, the plan is about 159 h
   [D]; four weeks hold 120 h. The whole cut list frees about 17 h, so cuts alone cannot close
   the gap without dropping a Must.
   - **(a) One more week (chosen).** M3 runs to week 17 (weeks 13–17, about 150 h). M4–M6
     move one week later and the plan becomes 30 weeks (doc 03 §7.2 amended, as on 21 and
     22 Oct). About 9 h remain over. Cuts 2–4 free about 7 h, so (a) also needs cut 1 (L2 to
     scripted recovery at Gate B, about 10 h) unless at least 2 h of the week 13–14 work is
     done in week 12. Under (a), cuts 2–4 come first.
   - **(b) Four weeks.** L2 becomes scripted recovery, and the Zagi, leave-one-class-out and the
     linearised margins are dropped, all now (about 17 h). Skyview's allocation view and the
     blog move to M4's first week (about 8 h; D8 amended). About 14 h of overrun remain, so a
     "partially met" verdict is likely.
   - Either way, the cut order is the single list in §11.
7. **Survival counts recovered upsets.** *Decided: yes.* The primary endpoint, survival, means no
   unrecovered loss of control and no uncontrolled ground contact (§8.3). This departs from
   M2's bench, which fails a flight on any excursion beyond |φ| 90° or |θ| 60° on its 10 Hz
   log. Under the bench's rule, every L2 recovery from beyond 90° of bank would count as a
   failure. The bench's definition is reported alongside, so M2 and M3 tables stay comparable.
8. **Baseline names follow doc 03.** *Decided: yes, in code too, in week 13.* B1 is the stock stack with the classical
   contingency v0, and B2 is INDI with the shared allocator. The M2 bench's "B1" (no
   contingency) becomes B0 and its "B2" becomes B1, in code too (§1).

---

## 1. Deliverables and Definition of Done

M3 is **done** when all of the following hold.

| # | Criterion | Evidence | Priority |
|---|---|---|---|
| D1 | CI is green on `main` (fast tier, web, Playwright); the nightly full tier is green | GitHub Actions | Must |
| D2 | **Split surfaces.** Up to eight independently actuated surfaces per airframe, each with its own derivatives and faults. Through the stock stack's fixed mixer, every M1 and M2 validation result is reproduced. | V33 | Must |
| D3 | **Adaptive allocation.** A shared weighted-least-squares allocator with online effectiveness estimation and a reversed/dead-surface test, in pure JAX, used by L1 and B2. | V34, V35 | Must |
| D4 | **L1 adaptive policy.** A privileged teacher distilled into a history-only GRU student of under 50k parameters, at 50 Hz, trained on the fleet with faults, wind and randomised delay. Five training seeds. | V36, V37 | Must |
| D5 | **B2 classical expert.** INDI with the same allocator, fixed and adaptive, tuned on G1-val only with a budget comparable to L1's sweeps. | V38 | Must |
| D6 | **L2 recovery.** An RL upset-recovery policy with hand-back to L1, compared with scripted recovery. Cut rule in §6.6. | V39 | Should (the cut is scripted recovery) |
| D7 | **The A/B report.** Pre-registered hypotheses P1–P6 (§8.2) on G2, G2-x, G3 and G4 with confidence intervals, the ablations A1–A5, robustness sweeps, a failure gallery, and the headline verdict (§8.4). One command reproduces every number. | V40, V41, V42, report | Must |
| D8 | **Demo and write-up.** The six-airframe montage, an allocation view in Skyview, `docs/demo/m3.md` and a blog draft. | Video + docs | Must |
| D9 | **GPU throughput (closes M2 D9).** M2's criterion: at least 1M simulator steps/s (10 ms control steps, as V15 counts them) on one GPU with a heterogeneous fleet and faults (`notebooks/gpu_throughput.ipynb`). Alongside it, the full training step (environment, allocator, networks and PPO update) in policy steps/s, which chooses the GPU type (§3.8). Both recorded in V15. | V15 | Must |

Priority labels are as in M1 and M2: **Must** is needed for the Definition of Done; **Should**
is planned and moves to M4 if the schedule slips; **Could** only if there is time.

**Names.** M3 uses doc 03's baseline names. The M2 bench code labels them differently:

| M3 name | What it is | M2 bench label (`bench/run.py`, `bench/sampler.py`) |
|---|---|---|
| B0 | The stock stack with no contingency (ArduPlane-like: it keeps flying the mission) | "B1" |
| B1 | The stock stack with the classical contingency v0 (doc 03's B1, as far as M3 needs it) | "B2" |
| B2 | INDI with the shared allocator, fixed or adaptive (doc 03's classical expert, inner loops only, §5) | — |

The bench labels are renamed in code in week 13 (§11). M2's documents keep their labels; V33
reproduces bench v0's table under the mapping above.

**Amendments that follow from the rules elsewhere.** If Gate A takes the fallback (§0.1), D2
is waived and recorded as not done (fallback taken), D3 becomes "effectiveness-aware inversion
with fault-aware bounds", and M3 is done on the other deliverables. If A3 is cut (§12), D7's ablations read A1, A2, A4 and A5. Under §0.6 (b), D6 is
scripted recovery and D8 leaves the allocation view and the blog to M4.

---

## 2. Split surfaces (week 13)

- **The array.** A fixed shape, N_MAX = 8, with a per-airframe mask: aileron left and right,
  elevator (or ruddervator, or elevon) left and right, rudder, flap left and right, spare.
- **Aerodynamics.** Each surface has a 6-vector of derivatives per radian (C_L, C_D, C_Y, C_ℓ,
  C_m, C_n). Each is scaled by aero v2's control-effectiveness table η_ctrl(α), as every
  control term is today (`sim/aerojax/aero.py`), and the surfaces superpose linearly. For a
  left/right pair, each side's vector is half the pair's symmetric combination plus or minus
  half its antisymmetric one.
  - The existing lumped derivatives are one or both combinations: ailerons give the
    antisymmetric one, the elevator the symmetric one, elevons and ruddervators both (as δa and
    δe, or δr and δe). They are pinned, and each pair's contribution is computed as
    S·(δL + δR)/2 + A·(δL − δR)/2, so under the fixed mixer the missing combination cancels and
    the three-channel model is reproduced exactly for dedicated pairs (ailerons, a conventional
    elevator). For mixed pairs (ruddervators, elevons), M2 limits travel, rate and deadband per
    virtual channel; replays of M1 and M2 results keep those per-channel actuator dynamics ahead
    of the mixer, and per-surface limits apply only to the new controllers (V33).
  - The missing combination (the ailerons' symmetric, flap-like effect; a split elevator's roll
    and yaw) comes, for generator airframes, from one more ±5° run per pair on the existing
    mirrored geometry with AeroSandbox's `symmetric` flag flipped. No one-sided geometry is
    needed: the generator and `generator/references.py` deflect both sides of a pair together
    today, never one side. It is one extra run per pair for the 10,000 airframes, at most about
    an hour of CPU at M2's generator rate [D].
  - Where no data exist (the Aerosonde from B&M, the Rascal from `Rascal.xml`), the missing
    combination is zero; see the Rascal below.
  - Under the symmetric mixer the missing combination cancels, so the stock stack's plant, V19
    and V20 are unchanged. It acts only when one side of a pair is faulted or the allocator
    deflects the two sides differently.
  - The optional |δ| drag term is new and is off in V33. Elevon travel falls out of the
    per-surface limits.
- **Actuators and faults.** Delay, lag, rate limit and deadband per surface. faultlab's actuator
  slots address a surface index 0–7 instead of three channels. A lumped-channel fault on a
  dedicated pair (ailerons, a conventional elevator) maps to both surfaces. On a mixed pair
  (ruddervators) it has no per-surface equivalent, so M2 replays keep a channel-level fault
  ahead of the mixer, and V33 checks this on bench v0's v_tail scenarios. The floating-surface
  trail takes its sign from each surface's own force (`float_signs`, 4ffb8d6 and 6f94db2).
- **The command.** `ActuatorCommand` carries the masked surface array. This amends ADR-004,
  whose command channels are aileron, elevator, rudder and throttle, with mixing in the
  per-airframe ServoMap. The stock stack (B0, B1) keeps its (aileron, elevator, rudder) servo
  functions through a fixed 8×3 mixer. ADR-004 assigns elevon and V-tail mixing to the
  per-airframe ServoMap, but none is implemented yet (V-tails fly on lumped δe/δr derivatives);
  ArduPlane's fixed elevon and V-tail mixing is believed to work this way [U]. The telemetry bundle
  records per-surface columns (bundle schema version 2, ADR-005). The airframe schema's rudder
  becomes optional under the mask: the X8 and the Zagi have none.
- **Acceptance (V33).** Through the mixer, the M1 golden flights, V3/V16/V17 parity, V19 and
  V20 on the committed sample and bench v0's table (M2's B1/B2, now B0/B1) are reproduced; the
  V-suite is re-run.
- **The Rascal.** Only the left aileron enters its C_ℓ,δa and C_n,δa in `Rascal.xml`. The
  derived model (`sim/jsbsim/rascal.py`) is rewired to 0.5·(left − right), a wiring change with
  the coefficients untouched and recorded.
  - Like B&M's Aerosonde, whose aileron data are only C_Y,δa, C_ℓ,δa and C_n,δa, the Rascal has
    no symmetric aileron effect. That flatters re-allocation against a single-aileron fault, so
    single-aileron faults on the Rascal and the Aerosonde are exploratory and outside the
    headline (§8.4).
  - Elevon airframes do not have this gap: an elevon pair's two combinations are the published
    C_m,δe and C_ℓ,δa, so a single-elevon fault is represented to linear order. Which other
    elevon terms (C_L,δe, C_D,δe, C_Y,δa, C_n,δa) PyFly's file and B&M give is checked in V42;
    any missing one is zero and recorded.
- **Option (Could).** Dual elevator servos on about 30 % of conventional airframes, for pitch
  redundancy.

---

## 3. L1: the adaptive flight policy

### 3.1 Interfaces

- **What it replaces.** L1 replaces B1's TECS, attitude and rate loops. Guidance (L1
  navigation), mission logic and the classical motor-cut contingency are shared by every M3
  controller (B1, B2 and L1). B2 keeps TECS (§5). An A/B difference against B1 or B2 therefore
  includes the height and energy loop, not only the inner loops. No planned arm isolates the
  inner loops alone, and the report says so.
- **References.** φ_ref comes from L1 navigation for every controller. L1's γ_ref comes from an
  altitude-shaping law, γ_ref = clip(k_h·(h_ref − h)/V, ±γ_max). B1 and B2 keep TECS's own
  height demand, rate-limited and with a 2 s lag (`stack/tecs.py`). V_ref comes from the
  mission or the contingency.
- **Rates.** Policy at 50 Hz (each action held for two control steps), allocator and estimator
  at 100 Hz, physics at 200 Hz. One policy step is two control steps is four physics steps:
  every throughput figure is quoted in policy steps per second unless it says otherwise.

### 3.2 Action and observation

- **Actions (4).** a ∈ [−1, 1]³ through tanh, scaled to angular-acceleration demands
  ν_i = ν_max,i·a_i about the three axes, relative to the trim deflections (a = 0 is trim on the
  nominal airframe), plus throttle sent directly. ν_max is the nominal control angular
  acceleration at cruise, computed by auto-configuration. One output range then means the same
  rotational response from 0.8 to 15 kg; it is also INDI's virtual control, so B2 shares the
  allocator.
- **Disclosure.** ν_max, the trim deflections and the nominal effectiveness come from
  auto-configuration: manufacturer-level configuration, the same information budget as B1.
  "Same frozen policy" means the same weights; the per-airframe scalings are computed, not
  tuned.
- **Student observation (about 34).** Gyro rates, gravity in body axes from the attitude
  estimate, accelerometer specific force (9); measured airspeed over cruise, dynamic pressure
  over cruise, altitude rate (3); tracking errors and references (5); the previous action (4);
  the allocator's attainable authority per axis and direction and its saturation residual (9);
  a nominal descriptor from auto-configuration, ablated in A7 (4).
- **Excluded:** α, β, true airspeed and the true effectiveness.
  - Privileged quantities stay on the harness side of ADR-006. In the batched environment, the
    teacher and the critic read them from the simulator state and the fault slots.
  - The student's observation is built from `SensorArrays` and stack outputs by a module inside
    `aeroadapt.stack` (`stack/observation.py`). The existing import-linter contract
    `stack-has-no-privileged-access` (the stack must not import `sim` or `faultlab`) then
    enforces the boundary in CI, and a unit test checks that the builder's inputs hold no
    privileged field.
  - `TruthSnapshot` remains the host-side path for recording and evaluation.

### 3.3 The teacher

- **Privileged inputs describe effects, not causes (about 30):** per-axis effectiveness at the
  current dynamic pressure relative to nominal; per-surface health (1 healthy, 0 dead or jammed,
  −1 reversed, fractions for reduced throw); the moment of a jam's offset; non-dimensional
  damping (C_ℓp, C_mq, C_nr) and static stability (C_mα, C_nβ, C_ℓβ); V/V_stall and thrust
  available against required; servo lag and delay; mean wind in body axes, turbulence
  intensity, true α and β. Describing a fault by its effect also prepares for G5 (unseen fault
  classes).
- **Architecture.** An encoder MLP 30→64→12 conditions every hidden layer of an MLP actor
  256-256-128 (ELU) through FiLM; about 125k parameters (the teacher has no size limit). FiLM or
  similar conditioning beat concatenation for fixed-wing actuator faults in Marquis & Farhood
  (arXiv 2604.03392) [U].
- **No leakage.** No future information (upcoming gusts). A fault is revealed only at or after
  onset: after a sampled detectability delay of 0.1–0.5 s, or once the surface is commanded
  away from a jam inside its range.
  - This delayed reveal is the review's own proposal for faults that cannot be seen until
    excited (a jam inside the range, a reversed surface in unexcited cruise). It is untested;
    its effect is measured by the teacher–student action gap per fault kind (§12).
  - Weihs et al. (ADVISOR, NeurIPS 2021) describe the imitation gap itself. Their remedy,
    adaptive weighting of the imitation and RL losses, is a further fallback.
- **Critic** (never deployed): an MLP 512-256-128 on the observation, the raw privileged
  parameters, the true state, the fault's kind, severity and time since onset, and the
  allocator's internal state.

### 3.4 The student

- **Dense(34→64, ELU) → GRU(64) → Dense(64→64, ELU) → Dense(64→4)** with a state-independent
  log-std: about 31k deployable parameters. In Miki et al. (Science Robotics 2022), a GRU belief
  encoder beat an MLP encoder in every case tested (perceptive legged locomotion). RAPTOR flies
  ten real quadrotors with a 2,084-parameter recurrent policy (arXiv 2509.11481), so a GRU-32
  student is a cheap ablation (A8).
- **Auxiliary head** (Dense(64→30), dropped at export): it predicts the teacher's normalised
  effect vector of §3.3. It is interpretable in Skyview ("believed aileron effectiveness 40 %")
  and a first step towards the self-model L3. RMA-style latent regression is the fallback.

### 3.5 Training

**Distillation in three stages, with a gate.**
1. **PPO teacher**, about 2B policy steps per seed.
2. **DAgger in the same batched environment.** The student flies; with probability β it executes
   the teacher's action, β decaying from 1 to 0 over the first 20 %. Loss: squared error of the
   pre-tanh means plus 0.5 × the auxiliary head's error. Truncated backpropagation over 256-step
   windows from the stored GRU state with a 32-step burn-in; minibatches over environments,
   never over time. About 0.6B steps. Offline behaviour cloning on logged teacher rollouts is
   not used: it suffers covariate shift. RMA (RSS 2021) trains its adaptation module on
   on-policy data, and Lee et al. (2020) are recalled to distil DAgger-style on the student's
   own rollouts [U].
3. **Gate C.** On G1-val (1,000 scenarios, §8.3), student survival within 2 percentage points of
   the teacher's and tracking RMS within 15 % (point estimates): stop. Otherwise fine-tune the
   whole student with recurrent PPO, with the critic initialised from the teacher's and a
   behaviour-cloning term towards the teacher decaying from 1 to 0 over about 0.4B steps. This
   extends A-RMA's third phase (Kumar et al., IROS 2022), which fine-tunes only the base policy
   with model-free RL, keeps the adaptation module frozen and uses no imitation term. If that
   fails too: single-stage asymmetric recurrent PPO with the auxiliary head and no teacher,
   after DreamWaQ (Nahrendra et al., ICRA 2023), which trains an asymmetric actor-critic with a
   jointly trained context estimator.

**Reward and termination.** Bounded positive kernels k(e, σ) = exp(−(e/σ)²):
r = 0.35·k(e_φ, 5°) + 0.25·k(e_γ, 2°) + 0.30·k(e_V/V_ref, 0.05) + 0.10·k(a_y, 1 m/s²)
− 0.05·‖Δa‖² − 0.5·𝟙[V < 1.2·V_s] − 0.5·𝟙[n_z out of limits],
plus the CAPS smoothness regulariser (Mysore et al., ICRA 2021). Airspeed outweighs flight path,
so the policy protects speed when thrust is short, as TECS does.
- Termination without bootstrap at ground contact, loss of control or a non-finite state.
- Loss of control uses the bench's thresholds, |φ| > 90° or |θ| > 60° (`LOC_ROLL_DEG` and
  `LOC_PITCH_DEG` in `bench/run.py`). The bench applies them after the flight, to its 10 Hz
  log; L1's training applies them at every step. L2's training does not terminate on them (§6).
- Truncation at 60 s ± 20 % with a bootstrap from the final observation (Pardo et al., ICML
  2018).

**PPO starting values** (JAX, large batch):

| | Teacher | Student fine-tune | L2 |
|---|---|---|---|
| Environments | 8,192 (sweep 4k–16k) | 8,192 | 4,096 |
| Unroll | 32 steps | 64 steps from the stored state | 32 steps |
| Epochs × minibatches | 4 × 8 | 4 × 8, over environments | 4 × 4 |
| Learning rate | 3e-4, adaptive on KL 0.01, in [1e-5, 1e-3] | 1e-4 | 3e-4 adaptive |
| Clip; gradient norm | 0.2; 1.0 | 0.2; 1.0 | 0.2; 1.0 |
| γ; GAE λ | 0.99 (sweep 0.995); 0.95 | same | 0.99; 0.95 |
| Entropy | 0.001 | 0 | 0.001 |

Separate critic, value coefficient 0.5–1.0, no value clipping; per-minibatch advantage
normalisation; a Welford observation normaliser clipped to ±5, saved with each checkpoint and
frozen at evaluation; tanh-squashed Gaussian with the pre-tanh sample stored, log-std from
ln 0.5 with a floor of 0.05. Spare GPU capacity goes into seeds, not bigger batches: PPO
saturates as the batch grows (SAPG, ICML 2024).

### 3.6 Randomisation and curriculum

| Factor | Range |
|---|---|
| Airframe | resampled every episode from the training split of the V20-accepted fleet |
| Aerodynamic derivatives | ×(1 ± 0.2), each, per episode |
| Mass; CG; inertia; thrust | ±20 %; ±5 % of MAC; ±30 %; ±20 % |
| Servo lag | 10–60 ms (the fleet's 10–40 ms, widened) |
| Total transport delay | 10–100 ms, plus a random extra control step (capped at 150 ms) |
| Rate limit; deadband | ×(1 ± 0.4); 0–1° |
| Sensors | noise ×0.5–3; gyro and accelerometer bias; airspeed lag 0–0.2 s |
| Wind | mean 0–12 m/s (at most 0.6 × cruise); Dryden light to moderate; gusts up to 5 m/s; shear; thermals. Wind of 0.6–0.8 × cruise is evaluated as a separate stratum (§8.3); wind above 80 % of cruise (G7) stays held out. |
| Topology, towards G3 | rudder removed 10 %; rudder effectiveness ×0–0.5 15 %; C_nβ ×0.3–1 15 %; ailerons dead with split elevators (tailerons) 5 % |
| Allocator's effectiveness source | oracle 0.2, nominal 0.3, online estimate 0.5, with ±30 % column error |
| Faults | 60 % of episodes carry one actuator, propulsion, energy or airframe fault, onset 2–40 s |

- **Sensor faults are excluded from L1.** A teacher that sees true airspeed while the student's
  pitot is blocked opens a large imitation gap, and a pitot fault also corrupts the allocator's
  dynamic-pressure scaling. They are M4's (synthetic airspeed and Sentinel), and the report says
  so.
- **The ladder**, compared with B1 at every rung: one nominal airframe in calm air; the whole
  training fleet; wind and turbulence; single faults family by family; the full delay and noise
  ranges; allocator-source and topology randomisation. Then automatic sampling: each (fault kind
  × class) cell in proportion to its current failure rate, with a 20 % uniform floor.
- **Starts.** 20 % of resets begin inside L2's engage boundary (|φ| ≤ 60°, θ in [−30°, 40°], V
  from 0.9 to 1.5 × cruise), so L1 never meets an unfamiliar state at hand-back.

### 3.7 The trainer

- A lean PPO trainer in the PureJaxRL style in `src/aeroadapt/learn/` (about 600–800 lines),
  validated first on a reference task (gymnax Pendulum) against published curves (V36); a
  jitted chunk of about 20 updates under `lax.scan`, with a thin Python loop for logging,
  evaluation and checkpoints.
- Environment `reset(key, pool_index)` and `step(state, action, key)` are pure functions; the
  fleet is a pytree on the device. A reset gathers from an offline pool of trimmed states (about
  8 trims per airframe, in float64): trim never runs inside the jitted loop.
- Unit tests for GAE with termination and truncation inside and at the edge of a rollout, for the
  recurrent state's reset at episode boundaries, and for distinct random keys across lanes.
- Asynchronous checkpoints about every 10 minutes (parameters, optimiser, normaliser, random
  keys, curriculum, config hash and git commit); a checkpoint on SIGTERM; the compilation cache
  on a network volume. `jax_default_matmul_precision='highest'` on A100/H100, then V3 parity and
  the JSBSim cross-check are re-run on the GPU before training.
- Monitored: approximate KL, clip fraction, value explained variance, terminations by cause,
  return per class, steps per second.

### 3.8 Compute

| Item | Policy steps |
|---|---|
| Ladder and debugging | about 20B |
| L1, 5 seeds × 3B (teacher 2B, distillation 0.6B, fine-tune 0.4B) | 15B (30B with ten seeds, §8.3) |
| Ablations A3–A5, 3 seeds each | about 27B |
| Leave-one-class-out, 2 folds × 3 seeds | about 18B |
| L2, 5 seeds × 0.5B | 2.5B |
| Sweeps | about 5B |
| **Total** | **about 88B (about 103B with ten L1 seeds)** |

- At an assumed 0.5–2M policy steps per second on one rented GPU [D, pending D9], that is
  about 12–57 GPU-hours. The assumption needs the simulator alone at 1–4M control steps/s or
  more, before any network or update cost [D]; that is above M2's 1M target (D9).
- Today's CPU numbers (V15, laptop, float32, batch 1,024) are about 0.28M policy steps/s for the
  simulator alone (552k control steps/s) and 0.11M with the stock stack in the loop (225k
  control steps/s). At batch 4,096 they are 0.43M and 0.15M (05 §14: 851k and 303k control
  steps/s).
- A 3B-step seed therefore needs at least 2–3 h of laptop CPU for the simulator alone. One seed
  per night is possible only if the allocator, the networks and the update cost less than about
  3× the simulator [D].
- The first GPU spend runs M2's notebook (`notebooks/gpu_throughput.ipynb`, with a dummy policy
  of the student's size in the loop). It closes M2 D9 and checks GPU parity.
- Once the environment and the trainer run (§11), a benchmark of the full training step at 8k,
  32k and 128k environments on a few GPU types chooses the GPU by cost per billion policy steps.
- The provider is RunPod (the owner's choice, 6 Nov 2026), so the benchmark compares the GPU
  types RunPod rents, and the cap is $400 with a checkpoint after the benchmark (§0.5).
- Debugging time, not GPU money, is the binding constraint.

---

## 4. The adaptive allocation layer

- **One module, `src/aeroadapt/stack/allocation.py`,** pure JAX, no simulator imports; L1 and B2
  share it, so the A/B compares controllers, not allocators.
- **Problem.** min ‖W_u(u − u_p)‖² + γ‖W_v(B̂u − ν)‖² over lo ≤ u ≤ hi, in normalised units,
  with γ = 1e3 (swept 1e2–1e4) and W_v = (1, 1, 0.2) for roll, pitch and yaw. The normal
  equations square the problem's condition number, and at γ of 1e5–1e6 the result is more than
  float32 can hold. Non-incremental for L1 (B̂ = q̄·B̃̂, u_p = trim); incremental for INDI (Δu
  about the filtered actuator position).
- **Solver.** Härkegård's primal active-set method (IEEE CDC 2002), unrolled as a fixed 10
  iterations with a converged mask (a `while_loop` would run every lane as long as the slowest),
  masked normal equations by an 8×8 Cholesky with 1e-6·I, warm-started from the previous step.
  Every iterate is feasible, so a truncated solve is still admissible; the iteration count is
  logged. Tested against `scipy.optimize.lsq_linear` on random problems with pinned surfaces
  (V34). The redistributed pseudo-inverse is the ablation and fallback.
- **Bounds carry the faults; the solver never changes.** Rate limits every step.
  - A detected jam, hard-over or floating surface: its column is zeroed and its command pinned.
    There is no servo position sensor, so in the non-incremental mode the jam's offset is a
    disturbance the policy cancels through its history.
  - A reversed surface: zero until confirmed, then flipped. A wrong-sign column is positive
    feedback: by the review's linear analysis it diverges within a few filter time constants,
    well under a second [D], and Schenk & Ducard (arXiv 2607.16471) found a sign error the most
    destabilising INDI mismatch.
- **Online effectiveness (V34).** Recursive least squares on increments, normalised by dynamic
  pressure, with Smeur's synchronisation of the actuator model and angular-acceleration filters
  (JGCD 2016), forgetting 0.995 (variable down to 0.95 on large residuals), column scales
  projected to [0, 3] × nominal with the sign locked. A per-surface three-hypothesis test
  (nominal, dead, reversed) over a 0.3 s window switches at a posterior of 0.95. Null-space
  dither of 0.5–1° at 2–5 Hz separates left and right columns where redundancy and margin
  allow.
- **What the policy sees:** per-axis attainable authority and the saturation residual (§3.2);
  never the true effectiveness.

---

## 5. B2: the classical expert

- **Structure.** φ_ref from L1 navigation; B1's TECS unchanged for θ_ref and throttle; INDI
  rate and attitude loops give ν; the shared allocator in incremental mode gives the surfaces.
  B2 replaces B1's attitude and rate loops; L1 replaces TECS as well. The report states the
  asymmetry (§3.1). In M3, B2 is doc 03's classical expert reduced to its inner loops; its
  EKF-based fault detection and isolation and its site selection belong to M4 and M5.
- **Control law.** ν = K_ω(ω_ref − ω_f) + ω̇_ref, ω_ref from a second-order reference model with
  pseudo-control hedging; Δu = allocation(ν − ω̇_f), u = u_f + Δu (the increment from the filtered
  actuator position, not the last command); turn coordination r_ref = (g/V)·sin φ·cos θ plus
  lateral-acceleration feedback.
- **Filters and gains.** A second-order Butterworth at 12 Hz (swept 6–20 Hz); a starting rule
  K_ω ≈ 0.75/(τ̂ + d̂ + 0.225/f_c) [D] (about 13 rad/s on the Aerosonde, 6 rad/s on the slowest
  servos), K_att ≈ K_ω/4, with only global multipliers tuned.
- **Variants.** B2-fixed (effectiveness from auto-configuration) and B2-adaptive (L1's
  estimator and test, the same code).
- **Fairness.** Only nominal servo, inertia and derivative values from auto-configuration; the
  same delay and noise randomisation as L1; 6–8 global hyperparameters tuned on G1-val by a
  batched grid or CMA-ES with a budget comparable to L1's sweeps, frozen before any held-out
  evaluation (V38).
- **Expectations, not announcements.**
  - Robust hybrid INDI and an H∞-synthesised, gain-scheduled two-degree-of-freedom PID came out
    virtually equivalent in robustness and performance (Pollack et al., AST 2024). A hand- or
    auto-tuned PID such as B1 is not that baseline.
  - On the Aerosonde, the roll-mode time constant (about 46 ms) is close to its servo lag plus
    delay (about 40 ms) [D], and the fleet's slower servos are worse, so INDI may fall short of
    its manned-aircraft reputation.
  - Large L1 gains are credible where INDI is structurally weak (reversed surfaces, slow servos)
    or the structure is new (G3).

---

## 6. L2: upset recovery (Should)

1. **Scope.** Upsets aerojax can represent: bank beyond 60° to inverted; nose-low spirals (θ to
   −60°, V to V_ne); nose-high low-speed states; stall entry and exit, on post-stall tables M2
   does not validate (M2 §12), so stall results are exploratory; with or without an active
   actuator fault. No spins or deep stall. After the freeze, every L2 result is also reported
   on the Rascal in JSBSim (G4); before it, L2 is checked in JSBSim only on an exported
   training airframe (§8.1). Every fixed-wing RL recovery result found in the review is
   simulation-only; so is this.
2. **Upsets.** Half from a grid (φ 60–180°, θ −60° to 45°, rates to 120°/s, V 0.8–1.6 × cruise,
   150–400 m above ground); half harvested at the moment the engage predicate fires in flights
   with faults on training airframes: B1's and the teacher's flights until the Gate-C student
   exists, then the student's. Half carry an actuator fault.
3. **Method.** Single-stage asymmetric recurrent PPO (privileged critic, the auxiliary head, the
   same GRU-64 and allocator with roll and pitch weighted equally). L2 owns the throttle. It
   trains against the Gate-C student.
4. **Reward.** A height-loss penalty; two-phase potential shaping (first kill the rates, then
   level the wings, recover the flight path and the speed; after Sayyed et al., arXiv
   2601.06439); penalties on load factor, V_ne, α beyond stall and a secondary upset; +10 on
   success (the hand-back predicate held for 1 s), −10 at ground contact, truncation at 20 s.
   No termination on the 90°/60° rule.
5. **Engage and hand back** on onboard estimates only. Engage on any of |φ| > 60°, θ < −30°,
   θ > 40°, ‖ω‖ > 90°/s, or V < 1.1·V_s while descending; hand back at |φ| ≤ 15°, all rates
   ≤ 15°/s, γ ≥ 0 and 1.2·V_s ≤ V ≤ 0.9·V_ne, held for 1 s, after at least 1 s engaged. L1's GRU
   runs throughout, seeing L2's actions as its previous action. 5 % of the student's training
   episodes (DAgger and fine-tune) start under a foreign controller, scripted recovery (§11),
   so this is familiar. The teacher has no memory and needs only the 20 % starts near L2's engage boundary (§3.6).
6. **Cut rule (Gate D).** Scripted recovery: unload, roll wings level with roll priority, power,
   a bounded pull, hand back; on INDI and the allocator; tuned on G1-val. L2 ships only if, on
   the fixed G1-val upset grid (its size fixed in the pre-registration), the paired 90 % lower
   bound of the success difference (L2 − scripted, airframe-cluster bootstrap) is above 0.
   Otherwise scripted recovery ships as M3's L2, RL recovery moves to M6, and P4 is withdrawn
   (§8.2). If L2's seeds are not done by Friday, Gate D and the weights freeze move to the
   following Monday.

---

## 7. New airframes

- **The X8**, a flyable airframe built from Gryte et al. (2018) through PyFly's parameter file,
  which `generator/references.py` already cites. `references.py` holds only the 11 derivatives
  V18 compares, with mass, area, span and chord. The build adds:
  - the remaining force and moment coefficients and the inertia tensor, from PyFly's file [U
    until read];
  - attached-flow tables from the published polynomial fits, with a flat-plate blend past stall
    (planned in M2 §3.2 and never built);
  - propulsion: PyFly's model if it has one [U], otherwise the generator's sizing from the
    published mass;
  - actuators and sensors, assumed, since the elevon sizes and servos are not published (M2 §14,
    week 6);
  - per-elevon columns: each elevon is half the published C_m,δe plus or minus half C_ℓ,δa
    (§2), with whatever other elevon terms the file has.
  - It has no rudder. The schema's rudder becomes optional (§2), and B1's auto-configuration
    runs without the rudder mix and yaw damper (`stack/attitude.py`, `stack/config.py`). B1
    must pass V20's acceptance flights (V11-lite) on it before the tag (V42).
- **The Zagi** from Beard & McLain's appendix, on the same path. Its coefficients are to be
  checked against the book [U] (V42).
- **The Aerosonde** keeps its role: independent provenance, never in training. Its aileron data
  lack the symmetric term (§2).
- **The Rascal** (M2 D4): its datasheet airframe and tuned B1 stack (`configs/fleets/rascal`,
  built by the generator from the kit's published geometry; to be committed) and its own JSBSim
  model.
  - For G4, {L1, B1} × {the datasheet Rascal in aerojax, the Rascal's own model in JSBSim}.
    Against G2, the aerojax column measures the new-airframe gap. The JSBSim column against the
    aerojax one measures the independent model and the independent simulator together.
  - An optional third column, the datasheet Rascal exported to JSBSim (`sim/jsbsim/export.py`),
    separates the simulator from the model (Could).

---

## 8. Evaluation

### 8.1 Pre-registration

`docs/validation/m3_prereg.md` is committed with a signed tag before the teacher first trains on
the fleet (ladder rung 2) and before B2 is tuned, in week 14 (§11). It is amended only by
dated, tagged addenda before the freeze. It holds the hypotheses with endpoints and margins;
the splits with airframe-id lists and hashes; scenario seeds and counts; each controller's
information budget; the analysis plan and the checkpoint rule; the headline airframes and their
hard-overs; the cut rules and their effect on the hypotheses (§8.2).

- **The ledger.** A test-access ledger records commit, weights hash, date and split for every
  run on G2, G2-x, G3, G4 or the Aerosonde.
- **Before the freeze,** only G1-val is used for selection and tuning. The held-out airframes are
  flown only by the stock stack (B0 and B1, whose per-airframe auto-configuration is part of
  their definition) in regression and acceptance tests (V33, V35, V42 and the full-fleet V20),
  and trimmed by the oracle (§8.4). No L1, L2 or B2 run touches them, and every access is in
  the ledger.
- **The pre-freeze JSBSim smoke test** flies a training airframe exported to JSBSim
  (`sim/jsbsim/export.py`), not the Rascal.

### 8.2 Hypotheses (Holm-adjusted across the six; everything else exploratory)

- **P1.** G2 fault-scenario survival: L1 with allocation above B1 (the stock stack with the
  contingency).
- **P2.** G3 fault survival: L1 above B1 on the X8 and on the Zagi, as case studies. Each is a
  single-airframe claim; P2 holds only if both hold.
- **P3.** G4, the Rascal in JSBSim: nominal tracking non-inferior to B1 (proposed margins +1.0°
  RMS in roll, +0.5 m/s RMS in airspeed), and higher survival under rudder and whole-channel
  faults. P3 holds only if all three components hold.
- **P4.** L2 against scripted recovery on a fixed upset grid on G2 airframes: success (height
  loss secondary).
- **P5.** G2 fault survival: L1 against B2-adaptive, two-sided. Losing this is a legitimate
  finding, and is reported as one.
- **P6.** G2-x fault survival: L1 above B1. This is the extrapolation in size and mass that doc
  03's G2 includes.

Rules, fixed in the pre-registration:
- P2 and P3 are "this airframe" claims, tested over scenarios within each airframe (§8.3). They
  are intersection–union tests: each one's p-value is the largest of its components'.
- If L2 is cut (at Gate B or Gate D), P4 is withdrawn and Holm runs over the remaining five.
- If the Zagi is not built by the tag (or is cut), P2 is the X8 alone (G3 is one case study),
  and the family stays at six.
- If A5 becomes primary at Gate B, every hypothesis and headline criterion that names L1 (P1,
  P2, P3, P5, P6 and §8.4) tests A5, and A5 is trained with five seeds (two added to its three,
  from L1's budget in §3.8).
- Either switch is recorded by a dated, tagged addendum before the freeze.

### 8.3 Splits, scenarios, metrics and statistics

| Split | Content | Scenarios (proposed) | Role |
|---|---|---|---|
| G1-val | the hash validation split | 1,000 | selection, tuning and the gates only |
| G2 | the hash test split, minus extremes (G2-x) | about 2,200, at most 3 per airframe [D], fixed from the real count after the full-fleet V20 | interpolation |
| G2-x | outside the 5th–95th percentile of mass or wing loading, removed from training before the teacher starts | 1,000 | extrapolation (P6) |
| G3 | X8 and Zagi | 300 per airframe | case studies |
| Aerosonde | B&M in aerojax, never in training | 300 | one of the six |
| G4 | the Rascal in JSBSim, with the aerojax column (§7) | 200 × 120 s | case study |
| LOCO | glider and V-tail folds, 3 seeds each | 1,000 per fold | class-level evidence |

- **V20 on the whole fleet.** Training, G1-val, G2 and G2-x all need V20 on the whole
  10,000-airframe fleet. M2 ran it on the first 1,000; their 855 accepted airframes include 80
  in the test split, the pool bench v0 used (05 §14, week 8). With five worker processes
  (51 min per 1,000 airframes in week 12), the whole fleet takes about 9 h of laptop CPU. The
  owner chose the laptop, on mains power, from 6 Nov 2026 (§13, question 4), with a few
  GPU-hours [D] as the fallback. The run keeps each finished batch, so an interruption loses
  at most the batches in flight, and its first 1,000 airframes must reproduce M2's run
  (`runs/v20/m2-fleet`) exactly. It must finish before the tag.
- **Scenarios.**
  - 30 % nominal (wind and turbulence only), 70 % single in-scope faults stratified by fault
    family × class; 300 s in aerojax, onsets 10–200 s.
  - Mean wind 0–0.6 × cruise, the training range. Wind of 0.6–0.8 × cruise is a separate,
    exploratory extrapolation stratum; above 0.8 is G7 (not in M3). The bench sampler's
    `wind_frac` is set accordingly; bench v0 keeps 0–0.8 for V33.
  - Every controller flies the same 100 Hz harness step with the same per-step keys
    (`jax.random.split(key, n_steps)` in `harness/closed_loop.py`, as today), so all meet
    identical gusts and sensor noise. L1's 50 Hz action hold does not change the keying. In
    JSBSim, paired flights share the scenario's seed (JSBSim's random seed and the sensor key,
    `vehicle/jsbsim_vehicle.py`).
  - Training seeds are disjoint from scenario seeds.
  - Each fault scenario is labelled recoverable or not by the trim-reachability oracle, and
    survival is reported on both sets.
- **Survival (the primary endpoint, §0.7):** no unrecovered loss of control and no uncontrolled
  ground contact; a planned contingency landing that is not a crash counts, as in the bench.
  Loss of control has three tiers:
  1. upset entered (L2's engage predicate);
  2. loss of control by the bench's 90°/60° rule on the 10 Hz log;
  3. unrecovered: after a tier-2 excursion, the hand-back predicate (§6.5) is not regained
     before ground contact or the end of the flight.

  Survival by the bench's definition (no tier 2 and no uncontrolled ground contact,
  `bench/run.py`) is the secondary endpoint, so M2 and M3 tables stay comparable.
- **Other metrics (one fixed set).**
  - Tracking on a step and doublet battery.
  - The height-loss recovery profile: the fraction of upsets recovered within h metres, failures
    never reaching the curve; never a mean over survivors.
  - Envelope exceedances; surface rate, saturation time and a smoothness metric.
  - An empirical delay margin: injected delay swept past the training range until loss of
    control or a limit cycle. The sweep compiles with a 64-step actuator delay buffer (320 ms)
    and lifts the schema's 150 ms cap; training keeps 32 steps (`sim/aerojax/actuators.py`).
  - Linearised gain and phase margins at trim on the six airframes; estimator convergence time;
    inference latency.
- **Statistics.** Two separate statements: "this frozen policy" (over scenarios, clustered by
  airframe) and "this training recipe" (over seeds).
  - Paired survival over many airframes (P1, P4, P5, P6): a cluster bootstrap of the paired
    difference (10k resamples; airframes, then scenarios within them) with an airframe-level
    sign-flip permutation test; exact McNemar and Newcombe's paired interval as secondary.
    Clustered data analysed as independent can exceed a 45 % false-positive rate (Saravanan et
    al.); the design effect is reported.
  - Single airframes (P2, P3 and §8.4): with one or two airframes there is nothing to cluster
    over (two airframes give four sign patterns, so a one-sided p of 0.25 at best). They are
    paired over scenarios within each airframe: exact McNemar with Newcombe's interval for
    survival, and for P3's tracking a one-sided paired test against each margin.
  - Continuous metrics: paired differences with failures ranked worst. The airframe-cluster
    bootstrap interval is the primary interval; Wilcoxon signed-rank with the Hodges-Lehmann
    estimate is secondary.
  - Recipe claims: rliable's interquartile mean over per-stratum survival per seed, stratified
    bootstrap intervals, performance profiles and probability of improvement (Agarwal et al.,
    NeurIPS 2021), plus a permutation test over seeds.
  - Aerospace form where it fits: "unrecovered loss of control ≤ 1 % at 95 % confidence" needs
    0 failures in 299 independent scenarios. Scenarios cluster by airframe, so the bound is
    given per airframe (0 of the Aerosonde's 300 gives ≤ 1.0 %; 0 of G4's 200 gives only
    ≤ 1.5 %) or with the design effect.
- **Seeds and checkpoints.**
  - Five seeds for L1 (never cut) and for L2; three for each of A3–A5 and each LOCO fold.
  - The first three L1 seeds are a pilot. If the seed-to-seed standard deviation of the
    teachers' G1-val survival exceeds 2 points, five more L1 seeds are trained (15B policy
    steps, inside §3.8's upper total). The pilot is judged on the teachers because five more
    seeds must start before Gate C to finish before the freeze.
  - The final checkpoint of every seed is used. The headline policy is the median seed by
    G1-val (with ten seeds, the sixth-best by G1-val survival, the lower of the middle two),
    declared in advance. All seeds are reported, with
    the weights hashes published before any test evaluation.

### 8.4 The headline

- **The six airframes, by rule:** two G2 airframes by hash draw (one from G2, one from G2-x), the
  Aerosonde, the X8, the Zagi (or a third G2 airframe by hash if the Zagi is not built by the
  tag), and the Rascal in JSBSim.
- **The fault per airframe, by rule.** Candidates in order:
  1. a single-aileron hard-over at full travel, skipped on airframes whose aileron data lack
     the symmetric term (the Aerosonde and the Rascal, §2);
  2. a rudder hard-over at full travel;
  3. a half-travel elevon or ruddervator jam (a partial jam).

  The first candidate passes if the trim-reachability oracle (the float64 trim with the faulted
  surface pinned, V41) finds wings-level trim with a roll margin of at least 10 %. The margin is
  the attainable roll moment in each direction about the faulted trim, as a fraction of the
  nominal airframe's maximum roll moment at the same q̄.
  - For the Rascal the oracle trims the aerojax datasheet Rascal (`configs/fleets/rascal`), and
    a JSBSim trim confirms the choice before the tag (logged in the ledger).
  - Under linear superposition, cancelling a full-travel single-aileron hard-over takes the
    healthy aileron's full opposite travel. The margin towards the jam must then come from the
    rudder through sideslip and dihedral, so the oracle may often reject candidate 1 [D].
  - The X8 and the Zagi have no ailerons or rudder separate from their elevons, so their only
    candidate is a partial jam.
- **Per airframe:**
  - (i) Nominal: 50 of 50 flights without loss of control (tier 2) in light to moderate
    turbulence, and the upper 95 % bound of the paired (L1 − B1) RMS difference over the 50
    flights within +1.0° in roll and +0.5 m/s in airspeed (the proposed P3 margins, for all
    six).
  - (ii) For an airframe with a qualifying fault: 50 variants (onset 10–30 s in cruise, wind
    0–min(8 m/s, 0.6 × cruise)), at least 45 of 50 surviving (a Wilson lower bound of about 79 %) and a median
    height loss of at most 30 m, with B1 and B2 alongside.
- **Verdict.** Met when all six airframes meet criterion (i), every airframe with a qualifying
  fault meets criterion (ii), and P1 holds after the Holm adjustment. Otherwise "partially
  met", naming the failing airframes.
  - An airframe with no qualifying fault is named with the oracle's reason.
  - The report states how many of the six carry a full-travel hard-over, how many a partial
    jam, and how many none; the headline's wording follows those counts.
  - If tracking is given up (§12), the verdict is "partially met".

### 8.5 Ablations (pre-registered)

| ID | Variant | Training |
|---|---|---|
| A1 | the frozen student with an oracle allocator | none |
| A2 | the frozen student with nominal effectiveness, estimator off | none |
| A3 | a raw-surface student: 8 masked surfaces and throttle out; the nominal effectiveness columns and the mask in | 3 seeds |
| A4 | a history-free MLP, randomised (robust but sluggish, per Ferede et al., ICRA 2025); never cut: it is the memory check of §12 | 3 seeds |
| A5 | L1 over INDI: total angular-acceleration demands into the incremental allocator | 3 seeds; the fallback primary |
| A6–A8 | no auxiliary head; no descriptor; GRU-32 | Could |

The allocator should win on instantaneous redundancy (split-surface jams, G3's channel
structure); the policy on dynamic cross-axis use (rudder, sideslip, dihedral roll), which a
static allocator cannot represent. The review found no fixed-wing head-to-head of a policy
through an allocator against one on raw surfaces (its searches ran out of budget), and the
multirotor evidence is mixed, so the result is reported either way.

### 8.6 Fidelity statements for the report

- Every result is simulation to simulation (aerojax to JSBSim).
- B1 is ArduPlane-equivalent, not ArduPlane, until M6's SITL subset.
- Linear superposition of control derivatives overstates authority at large deflections; aerojax,
  the allocator and the JSBSim Rascal share this, so their agreement does not validate hard-over
  recovery.
- Single-aileron faults on the Aerosonde and the Rascal lack the symmetric aileron term (§2)
  and are exploratory.
- Stall and post-stall results rest on approximate tables (M2 §12); M2 does not claim
  stall-recovery fidelity.
- The X8 flies on published coefficients with assumed actuators; no spins are modelled.
- Novelty claims ("the first cross-airframe, cross-class zero-shot fixed-wing RL controller of
  which the author is aware") wait for a read of Chowdhury & Keshmiri (IEEE TAES 2024) and a
  fresh 2025–26 search.

---

## 9. Repository additions

Most rows change existing modules: the three channels are hard-coded in about two dozen
modules under `src/` (`grep -lw aileron` lists 24).

| Path | Contents |
|---|---|
| `src/aeroadapt/core/messages.py`, `docs/adr/004-vehicle-interface.md` | `ActuatorCommand` as a masked per-surface array; the ADR-004 amendment |
| `src/aeroadapt/airframes/schema.py`, `params.py` | per-surface derivatives and actuators; an optional rudder |
| `src/aeroadapt/generator/aero.py`, `geometry.py` | the missing symmetric or antisymmetric run per surface pair (§2) |
| `src/aeroadapt/sim/aerojax/aero.py`, `sim.py`, `actuators.py`; `faultlab/` | split surfaces (N_MAX = 8, mask), per-surface faults and floating signs; the larger delay buffer for the margin sweep |
| `src/aeroadapt/sim/jsbsim/fdm.py`, `v14.py`, `export.py`; `vehicle/` | per-surface commands in both backends |
| `src/aeroadapt/sim/jsbsim/rascal.py` (being modified for M2 D4), `tools/rascal_airframe.py` (M2 D4, uncommitted) | the Rascal's per-surface wiring |
| `src/aeroadapt/stack/attitude.py`, `autopilot.py`, `config.py`; `harness/autoconfig.py`, `autotune.py` | the stock stack's fixed mixer; B1 without a rudder |
| `src/aeroadapt/telemetry/schema.py` | per-surface columns, bundle schema version 2 (ADR-005) |
| `src/aeroadapt/stack/allocation.py` | WLS active-set allocator, bounds, effectiveness estimator, reversed/dead test |
| `src/aeroadapt/stack/observation.py` | the student's observation, from `SensorArrays` and stack outputs only (§3.2) |
| `src/aeroadapt/stack/indi.py` | B2: INDI rate and attitude loops on the shared allocator |
| `src/aeroadapt/stack/recovery.py` | scripted recovery on INDI and the allocator (§6.6) |
| `src/aeroadapt/learn/` | environment (reset pool, auto-reset, randomisation, keys), PPO trainer, teacher, student, DAgger, checkpoints |
| `src/aeroadapt/policies/` | L1 and L2 inference for the stack (pure JAX, no training imports) |
| `src/aeroadapt/bench/run.py`, `sampler.py` | baselines relabelled B0 and B1; the survival tiers (§8.3); M3's wind ranges |
| `src/aeroadapt/bench/stats.py` (extended; it holds Wilson and the percentile bootstrap) | cluster bootstrap, sign-flip permutation, McNemar, Newcombe, Hodges-Lehmann, rliable wrappers |
| `src/aeroadapt/bench/oracle.py` | the trim-reachability oracle (V41) |
| `configs/airframes/` | the X8 and the Zagi |
| `docs/validation/m3_prereg.md` | the pre-registration |
| `notebooks/gpu_throughput.ipynb` (extended) and a new training runner | M2 D9 and the full training step; the GPU training runs |
| `docs/research/m3/` | the literature review (§13, question 8) |

**Import-linter.** A forbidden contract covers a package and all its submodules, so the
inference code lives outside `learn/`.
- New: `aeroadapt.stack` and `aeroadapt.policies` must not import `aeroadapt.learn`.
- `aeroadapt.policies` joins the source modules of `stack-has-no-privileged-access` and
  `stack-is-backend-agnostic`.
- `aeroadapt.learn` joins the source modules of `generator-is-offline`.

---

## 10. Validation additions (continuing the numbering)

| ID | What | Criterion | Tier |
|---|---|---|---|
| V33 | Split surfaces reproduce the three-channel model through the stock stack's mixer | M1 goldens, V3/V16/V17, V19 and V20 (sample), and bench v0's table (M2 labels B1/B2, now B0/B1) unchanged | fast / full |
| V34 | Allocator: active-set against `lsq_linear` on random problems with pinned surfaces; estimator convergence; the reversed/dead test's detection time | on lanes that converge within 10 iterations, within 1e-6 of `lsq_linear` on the same stacked problem, regulariser included (float64); at least 99 % of lanes converge, and every lane is feasible; detection ≤ 0.5 s on a fixed set of 1,000 G1-val reversed and dead onsets | fast |
| V35 | Allocator in the loop: B1 through the allocator with nominal effectiveness flies the M1 missions (the stock stack only, §8.1) | V11 criteria | full |
| V36 | Trainer: GAE and recurrent-state tests; a reference task (Pendulum) reaches its published return | unit tests; within 10 % | fast / full |
| V37 | L1 student: Gate C (§3.5); no privileged input reachable from the student's observation (the import-linter contract and a unit test) | Gate C's statistics recorded against its thresholds; the thresholds met, or the fallback taken | full |
| V38 | B2-fixed and B2-adaptive fly V20's acceptance flights (V11-lite) on a fixed sample of G1-val airframes, before the freeze and before any held-out evaluation | V20's criteria (≥ 90 % pass); B1's pass rate on the same airframes alongside; no held-out airframe used | full |
| V39 | L2 against scripted recovery on G1-val (Gate D) | Gate D's statistic recorded against its threshold (§6.6); the cut rule applied | full |
| V40 | The A/B report reproduces from one command, with the pre-registration's hash | identical tables from the same seeds | full |
| V41 | Trim-reachability oracle | (a) with no fault, it reproduces M1's V4 trims on the Aerosonde and auto-configuration's cruise trims on the sample fleet (float64); (b) a surface pinned at its nominal trim deflection returns the nominal trim and margin; (c) on a sample of cases it agrees with a brute-force grid search over the free surfaces (an independent implementation). The review's [D] cases (the Rascal's 24 %, the Aerosonde's 33° of sideslip) are recomputed and recorded. All before the tag. | fast |
| V42 | The X8 and the Zagi | coefficients match PyFly's file and B&M's appendix value by value; float64 trim; B1 passes V20's acceptance flights (V11-lite); all before the tag | fast / full |

---

## 11. Work plan (weeks 12–17) and gates

All hours are estimates [D]. Week 12 holds M2's carry-overs and is not counted in M3's hours.
The table is option (a) of §0.6, five weeks after week 12, which the owner chose on 6 Nov 2026;
option (b) is kept below for the record.

| Week | Work (hours) | Total | Exit |
|---|---|---|---|
| **12 · Carry-overs** | M2 D6 auto-configuration at 90 % · M2 D4 the Rascal in JSBSim · this spec · M2's GPU notebook if time allows | — | M2 D6 and M2 D4 done and logged in 05 §14 |
| **13 · Surfaces and allocator** | the full-fleet V20, queued as soon as M2 D6 closes, in week 12 if possible (1; about 9 h of CPU) · baseline labels B0/B1 in code, the ADR-004 amendment (1) · split surfaces across the modules of §9, per-surface faults, the mixer, telemetry schema v2, V33 (20) · per-side columns and the fleet's extra deflection runs (3) · Rascal per-surface wiring (2) · `allocation.py` and V34 (6) | 33 | **Gate A** |
| **14 · Airframes, oracle, pre-registration, trainer** | the X8 and the Zagi, V42 (10) · B1 without a rudder (2) · trim-reachability oracle, V41, the hard-overs, the Rascal's confirmed by a JSBSim trim (4) · splits frozen, pre-registration v1 tagged (3) · PPO trainer on a reference task, V36 (8) · M2's GPU notebook and GPU parity (2) | 29 | pre-registration tagged; full-fleet V20 done; V36 green |
| **15 · Environment, estimator, teacher, INDI** | environment, randomisation, keys, GAE and recurrent tests (6) · estimator, test and dither (5) · full-training-step benchmark (1) · teacher ladder rungs 1–4 on a GPU (8) · B2 fixed and adaptive, tuned on G1-val, V38 (8) · `bench/stats.py` extensions (3) | 31 | **Gate B** |
| **16 · Student, ablations, L2** | scripted recovery (4) · rungs 5–6, teacher seeds (3) · DAgger, Gate C (Wednesday), stage 3 if needed (10) · A3–A5 and two LOCO folds launched (3) · L2 v0 (10) · JSBSim smoke test on an exported training airframe, 10-minute drift test (4) | 34 | **Gate D** (Friday, or the Monday after); **all weights frozen**, hashes in the ledger |
| **17 · Evaluation, report, demo** | the campaign on G2, G2-x, G3, G4 and the Aerosonde; sweeps and margins (8) · analysis and failure gallery (10) · Skyview six-airframe montage and allocation view (8) · `docs/demo/m3.md` and blog (6) | 32 | **Definition of Done (§1)** |

**Gates.**
- **Gate A** (Friday, week 13): V33 passes. Otherwise the fallback (§0.1).
- **Gate B** (Friday, week 15): on the 1,000 G1-val scenarios, the teacher's fault survival
  exceeds B1's by at least 3 points (point estimate), with the paired airframe-cluster bootstrap
  90 % lower bound of the difference above 0; and nominal roll and airspeed RMS within +1.0° and
  +0.5 m/s of B1 (upper 90 % bound; the P3 margins). Otherwise A5 becomes primary and L2 is cut
  now (§8.2's rules).
- **Gate C** (Wednesday, week 16): §3.5.
- **Gate D** (Friday, week 16): §6.6.

**The overrun.** The table holds about 159 h [D]. The first draft listed 118 h; the new lines are
the PPO trainer and V36 (8), the X8 and the Zagi with B1 support (12), scripted recovery (4),
the GPU notebook, GPU parity and the full-step benchmark (3), the per-side columns (3), the
oracle's checks (1), the full-fleet V20 (1, plus about 9 h of CPU), the relabelling and the ADR
(1), the split surfaces raised from 14 to 20 h after counting the modules (6), and the demo
page (2). The 3-seed pilot costs compute, not hours.
- **(a) Five weeks (chosen)** hold about 150 h: about 9 h over. Cuts 2–4 free about 7 h; cut 1 frees
  about 10 h if it is made at Gate B or earlier (after Gate D, L2 v0's hours are already
  spent).
- **(b) Four weeks** hold 120 h. Cutting L2, LOCO, the Zagi and the linearised margins now frees
  about 17 h, and moving the allocation view and the blog to M4 frees about 8 h more. About
  134 h remain, still about 14 h over. Under (b), weeks 13–16 take the table's work in the same
  order, with the pre-registration still in week 14.
- Anything finished in week 12 beyond the carry-overs (the GPU notebook, the start of the X8)
  comes off the total. If M2 D6 or M2 D4 is not done by the end of week 12, its remainder comes
  out of week 13 and the Zagi is cut first.

**If behind**, cut in this order (hours freed [D]):
1. L2 becomes scripted recovery (doc 03's own cut): about 10 h if decided at Gate B or earlier.
   P4 is withdrawn.
2. Leave-one-class-out from two folds to none: about 1 h and 18B policy steps.
3. The Zagi: about 4 h. The sixth airframe becomes a third G2 airframe.
4. Linearised margins: about 2 h.

A6–A8 are Could and carry no hours, so cutting them frees nothing.

Never cut the split surfaces (or the explicit fallback), the pre-registration, B2, five seeds
for L1, A4 (the memory check, §12), or the G4 Rascal.

---

## 12. M3 risks

| Risk | Early warning | Mitigation | Fallback |
|---|---|---|---|
| The split-surface refactor overruns or breaks parity | V33 failing mid-week 13 | the mixer and the pinned combinations keep backward compatibility; tests first | three channels; D2 and D3 amended; a rudder hard-over headline (§0.1) |
| The GPU path is slow or broken | M2's notebook; GPU parity; the full-step benchmark | 'highest' precision; compile cache; benchmark the whole training step | a second provider; CPU overnight for the ladder and, if a seed fits in a night [D], single seeds; trained ablations cut to A4 and A5 (D7 amended); if L1's five seeds cannot be trained, the schedule choice (§0.6) is reopened |
| The teacher learns an "average airframe" | per-class returns diverge | normalised actions; log-scaled privileged inputs; bounded rewards; the ladder | per-class teachers distilled into one student; A5 primary |
| Imitation gap on faults unobservable until excited | action gap by fault kind | the delayed reveal (untested, §3.3); the auxiliary head; stage 3 | an ADVISOR-style adaptive imitation weight; single-stage training; those faults reported separately |
| The two adaptive loops fight | oscillation at onset with the estimator on | mixed effectiveness sources; separated time scales | the estimator restricted to the dead/reversed test |
| A reversed surface diverges before detection | reversed-surface survival near 0 for L1 and B2 | zero column until confirmed; 0.3 s test; L2 engages | a stated limitation, outside the headline |
| Simulator shortcutting; G4 collapses | the JSBSim smoke test (week 16): an exported training airframe flies worse in JSBSim than in aerojax | nuisance randomisation | the gap reported; every pre-registered controller, A5 included, flies G4, and none is chosen from G4 results |
| G3 fails; it cannot be checked before the freeze | proxies only: rudderless training airframes | topology randomisation (§3.6) | the negative result reported; procedural flying wings in M6 |
| The headline fault is physically unrecoverable | the oracle (V41) | the oracle runs before the tag | per-airframe wording (§8.4) |
| Over-wide randomisation makes L1 sluggish | nominal RMS on G1-val above the margin | moderate ranges plus history | narrower ranges; claim survival, not tracking, and the headline is "partially met" (§8.4) |
| Recurrent PPO bugs: memory unused | student no better than A4 | unit tests; the A4 comparison (A4 is never cut) | frame stacking |
| Chattering surfaces | smoothness far above B1 | CAPS plus the rate penalty | a fixed low-pass on ν, in the loop |
| Reward hacking (diving for authority, post-stall exploits) | height loss, α excursions | envelope penalties; height in the reward; JSBSim check | a restricted L2 envelope |
| Test-set reuse or a curated headline | none: prevented by design | pre-registration, ledger, the pre-freeze access rule (§8.1), hash draws, the median seed | — |
| M2's carry-overs or the new airframes are late | the week-12 exit; V42 not green by the tag | the costed lines in §11 | the Zagi cut first; a third G2 airframe takes its place |
| The full-fleet V20 is late | not running by the start of week 13 | queued as soon as M2 D6 closes; finished batches kept, so a restart resumes | a few GPU-hours [D]; the tag waits for it |
| Schedule: about 159 h of work for 150 h in five weeks [D] | the week-12 exit; Gate A; Gate B | the costed plan (§11); five weeks chosen (§0.6) | the cut list in order, cut 1 at Gate B if the hours spent call for it |

---

## 13. Open questions, answered 6 Nov 2026

The owner decided question 3 and asked for the recommendations to be taken on the rest.

1. **The split surfaces** (§0.1): go ahead, in week 13; the three-channel fallback stays at
   Gate A.
2. **G3 as two case studies** (§0.2): yes, the X8 and the Zagi; procedural flying wings move to
   M6. *Still open:* whether the owner has Beard & McLain's book, to check the Zagi's
   coefficients against its appendix [U] (otherwise the authors' published files; with neither,
   cut 3 early).
3. **The GPU cap** (§0.5): $400 for M3, on RunPod, with a checkpoint: after M2's GPU notebook
   and the full-training-step benchmark the plan is re-costed, and above about $300 the owner
   decides before further spend.
4. **V20 on the whole fleet**: on the laptop's CPU, about 9 h with five workers, on mains
   power, from 6 Nov 2026; a few GPU-hours [D] only if that fails. The tuner is not changed
   first: B1 stays as M2 delivered it, so the run's first 1,000 airframes reproduce M2's D6 run
   exactly, and the tie-break limitation (05 §14, week 12) stays recorded.
5. **The schedule** (§0.6): five weeks, (a); doc 03 §7.2 amended, the plan now 30 weeks.
6. **Survival and names** (§0.7, §0.8): recovered upsets count as survival, with the bench's
   definition reported alongside; the bench's baselines become B0 and B1, in code too.
7. **Doc 03 §4.2's L1-adaptive-control baseline**: moved to M6's evaluation campaign as a Could
   (doc 03 §4.2 amended). It is the standard adaptive-control comparison, but M3 has no hours
   for a third baseline, and B2's adaptive INDI already gives the classical adaptive reference.
8. **The literature review**: committed under [`docs/research/m3/`](research/m3/README.md), the
   synthesis and the six topic reports. The spec check's and the revision's working files stay
   local.

---

## Sources

The full reference list, with the claims still to check, is in
[`docs/research/m3/synthesis.md`](research/m3/synthesis.md) (6 Nov 2026). The ones the design
rests on:

- Kumar et al., RMA: Rapid Motor Adaptation for Legged Robots, RSS 2021, https://arxiv.org/abs/2107.04034; Adapting Rapid Motor Adaptation for Bipedal Robots, IROS 2022, https://arxiv.org/abs/2205.15299
- Lee et al., Learning quadrupedal locomotion over challenging terrain, Science Robotics 2020, https://arxiv.org/abs/2010.11251; Miki et al., Science Robotics 2022, https://arxiv.org/abs/2201.08117
- Nahrendra et al., DreamWaQ, ICRA 2023, https://arxiv.org/abs/2301.10602
- Eschmann, Albani, Loianno, RAPTOR: A Foundation Policy for Quadrotor Control, arXiv 2509.11481; Learning to Fly in Seconds, IEEE RA-L 2024, https://arxiv.org/abs/2311.13081
- Bøhn et al., PPO attitude control of fixed-wing UAVs, ICUAS 2019, https://arxiv.org/abs/1911.05478; field experiments, IEEE TNNLS 2023, https://arxiv.org/abs/2111.04153
- Marquis & Farhood, Hypernetwork-Conditioned RL for Fixed-Wing Aircraft under Actuator Failures, arXiv 2026, https://arxiv.org/abs/2604.03392
- Mysore et al., CAPS, ICRA 2021, https://arxiv.org/abs/2012.06644; Pardo et al., Time Limits in RL, ICML 2018, https://arxiv.org/abs/1712.00378
- PureJaxRL, https://github.com/luchris429/purejaxrl; Singla et al., SAPG, ICML 2024, https://arxiv.org/abs/2407.20230
- Härkegård, Efficient active set algorithms for constrained least squares in aircraft control allocation, IEEE CDC 2002; Johansen & Fossen, Control allocation: a survey, Automatica 2013
- Smeur, Chu, de Croon, Adaptive INDI for attitude control of micro air vehicles, JGCD 2016
- Schenk & Ducard, Linear stability analysis of an INDI pitch-rate controller under model mismatch, arXiv 2026, https://arxiv.org/abs/2607.16471
- Pollack, Theodoulis, van Kampen, Commonalities between robust hybrid INDI and PID, AST 2024
- Agarwal et al., Deep RL at the Edge of the Statistical Precipice, NeurIPS 2021 (rliable)
- Ferede et al., One Net to Rule Them All, ICRA 2025, https://arxiv.org/abs/2504.21586
- Weihs et al., ADVISOR, NeurIPS 2021, https://arxiv.org/abs/2007.12173
- Foster, Dynamic Stability and Handling Qualities of Small UAVs, BYU MSc 2004, https://scholarsarchive.byu.edu/etd/219

---

## 14. Implementation log (living section)

### Week 12 (6 Nov 2026)

- M2's carry-overs are done and logged in [05 §14](05_m2_technical_spec.md): M2 D6, V20 at
  95.1 % (from 85.5 %), and M2 D4, the Rascal in JSBSim (8 of 8 nominal flights, 4 of 4 motor
  cuts detected and landed).
- 4ffb8d6: a floating surface now trails the destabilising way in either sign convention
  (dated corrections in 05). 6f94db2: the trail is keyed on each surface's own force, and
  confounded pair 5 is recorded as a deviation in 05. §2's per-surface floating signs build on
  them.
- This spec drafted, then revised after a three-reviewer check (72 issues on repository facts,
  internal consistency and literature). The revision costs the plan line by line (§11) and
  puts the schedule choice to the owner (§0.6).
- Decisions (evening): the owner approved the GPU cap of $400 with a checkpoint and chose
  RunPod, and asked for the recommendations to be taken on the rest (§0, §13). Five weeks, so
  doc 03 §7.2 is amended to 30 weeks; the L1-adaptive-control baseline moves to M6 as a Could.
  The literature review is committed under `docs/research/m3/`. `aeroadapt fleet-accept` now
  keeps each finished batch and resumes from them, for the full-fleet V20 on the laptop.
