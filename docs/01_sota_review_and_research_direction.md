# aero-adapt: State of the Art, Open Problems, and a Proposed Research Direction

*Draft for discussion, 13 Oct 2026. Literature checked through Oct 2026. Many 2025–26 preprints were read at abstract level only; claims that rest only on an abstract are marked †.*

---

## 0. Executive summary

**The field so far.** For multirotors, fast adaptation to *parametric* variation within a known vehicle family is now largely solved.

- A single recurrent policy with 2,084 parameters (RAPTOR, 2025) flies ten real quadrotors, from 32 g to 2.4 kg, zero-shot.
- Meta-learned residual adaptation (Neural-Fly) handles strong wind.
- Learned fault-tolerant controllers survive the loss of a single rotor on real hardware.

So a project framed as "an adaptive RL controller that generalises across more X" is incremental unless X is qualitatively new.

**What is not solved, and what actually brings aircraft down, is adaptation when the *cause* of an anomaly is ambiguous.** Consider Birgenair 301, Aeroperú 603, XL Airways 888T, AF447 and Lion Air 610 / Ethiopian 302. The common pattern in these accidents is not missing control authority or adaptivity. In each case the system, whether crew or automation, treated a faulty or obstructed sensor as a real change of state.

Today's learned adaptive controllers compress their history into one latent with no assigned meaning. They then adapt to *whatever best explains the data*, so by construction they are exposed to the same failure.

I found no work that tests state-of-the-art learned adaptive flight controllers against *confounded* faults. By confounded I mean cases where sensor, actuator, aerodynamic and disturbance hypotheses fit the data equally well but demand opposite responses. That I found nothing is not proof that nothing exists; Phase 0 is designed to confirm the gap.

**Recommended direction: *Adaptation under Ambiguity*.** The question is: when should an aircraft hedge, when should it experiment on itself to find out what broke, and when should it admit that none of its hypotheses fit? The project has three linked contributions.

1. **Phenomenon and benchmark.** Show and characterise misattribution failures in state-of-the-art adaptive controllers, using pairs of confounded faults that are mined automatically.
2. **Theory: a *probe-or-hedge frontier*.** Whether actively disambiguating is admissible depends on two things. One is how fast a probe separates the competing hypotheses. The other is each hypothesis's time-to-unrecoverability. The prediction is that fast multirotor faults mostly sit in the "hedge" regime, and slower fixed-wing air-data faults in the "probe" regime.
3. **Method and hardware demonstration.** The method has three parts:
   - a physics-structured, factored world model with a frozen, amortised, multimodal posterior over which mechanism has failed;
   - a learned recoverability critic, conditioned on that belief, which decides when probing is admissible;
   - a calibrated misspecification monitor that hands control to a runtime-assurance fallback.

**Why both audiences should care.**

- **ML:** this is a general problem: control when the cause is non-identifiable and confounded. It touches Bayes-adaptive RL, simulation-based inference and causal world models.
- **Aerospace:** it mirrors pilot training for unreliable airspeed:
  - first fly a pitch-and-power setting that is safe under every hypothesis (a *hedge*);
  - then cross-check and troubleshoot (a *probe*);
  - then commit.

  The method produces explicit hypotheses, which helps explainability. Its weights stay frozen, which fits offline learning assurance. It also slots into a runtime-assurance architecture.

**De-risking.** Phase 0 takes about six weeks and needs only laptop-scale simulation. It answers the kill question:

> Does an end-to-end in-context policy, trained with sensor faults in its randomisation, already hedge and disambiguate nearly as well as an oracle that knows the true fault?

If it does, pivot to Direction D2 (belief-conditioned recoverability), which reuses most of the Phase 0 infrastructure.

---

## 1. Sharpening the research question

The core question is: can an AI learn unknown aerodynamics, adapt to unexpected conditions, and safely fly complex manoeuvres without task-specific retraining? It bundles four sub-problems of very different maturity:

| Sub-question | Status (Oct 2026) | What makes it hard |
|---|---|---|
| Learn unknown aerodynamics | Largely solved *within the support of the training data* (residual models, meta-learned bases) | Extrapolating to mechanisms absent from training |
| Adapt to unexpected conditions | Solved for parametric shifts and smooth disturbances. Open for confounded faults and structural damage. | **Attribution**, identifiability, time budget |
| …safely | Open | After a fault the safe set is unknown and shrinking; risk estimates are uncalibrated off-distribution |
| …without task-specific retraining | Mostly a non-problem for tracking, since any feasible reference counts as a "task". Open for tasking that respects the current flight envelope. | Knowing which manoeuvres are still feasible *now* |

"Unexpected conditions" also needs unpacking. The classes differ in timescale and, critically, in *identifiability*:

| Class | Examples | Identifiable from onboard data alone? | Time to catastrophe | Coverage by current methods |
|---|---|---|---|---|
| Parametric | mass, inertia, motor constants, battery sag | Yes, with mild excitation | seconds or more | **Good** (RAPTOR, Extreme Adaptation) |
| External disturbance | steady wind, gusts, ground effect, downwash | Partly; confounded with aero-model error | seconds | **Good** for smooth wind (Neural-Fly); weaker for realistic turbulence |
| Actuator degradation | loss of effectiveness, saturation, delay | Usually, if the actuator allocation is excited | under a second (multirotor) to seconds (fixed-wing) | **Moderate** (learned FTC, INDI-FTC) |
| Actuator loss or jam | rotor loss, stuck control surface | Yes | under a second for multirotors | **Moderate** (specialised policies) |
| Structural or aero damage | wing damage, chipped propeller, CG shift | Partly; inertia and aero change together | varies | **Poor** (few learning works; database approaches) |
| Sensor faults | bias, scale error, frozen value, saturation, dropout, blockage, spoofing | **Often not**, without redundancy or deliberate excitation | depends on what the controller *does* with the bad data | **Poor** (Learn-to-Recover 2020, simulation only; ad hoc fixes such as MonoRace's accelerometer patch) |
| Compound or confounded | sensor + actuator; damage + wind | **Frequently ambiguous over short horizons** | — | **Essentially none** |

The open science is in the bottom three rows.

---

## 2. State of the art: critical review

### 2.1 Learned adaptive control (multirotors)

- **Neural-Fly** (O'Connell et al., *Science Robotics* 2022).
  - **Method:** domain-adversarially invariant meta-learning learns a wind-invariant basis from about 12 minutes of flight data. Online, a composite adaptation law updates the linear coefficients of the last layer.
  - **Results:** in the Caltech wind tunnel, at wind speeds up to 12.1 m/s, it reported average tracking improvements of 66% over a nonlinear tracking controller, 42% over L1 adaptive control and 35% over INDI.
  - **Assumptions:** the residual dynamics lie in the span of the learned basis; adaptation is linear in the parameters; state feedback is trustworthy.
- **RMA-style latent "extrinsics".** Rapid Motor Adaptation (Kumar et al., RSS 2021) infers a latent from the recent history of sensor readings and actions. This led to a single near-hover controller for very different quadcopters (Zhang et al., ICRA 2023), and then to *Extreme Adaptation* (Zhang et al., T-RO 2025). These works are evaluated on held-out *parameter ranges*.
- **DATT** (Huang et al., CoRL 2023) combines RL with L1-adaptive feed-forward to handle infeasible references and wind.
- **RAPTOR** (Eschmann, Albani, Loianno, 2025/26) is the most important recent result.
  - **Method:** "meta-imitation": 1,000 RL teacher policies are distilled into one recurrent student with 2,084 parameters.
  - **Result:** the student adapts in context, within milliseconds, to ten real quadrotors it never saw in training. It copes with wind, being poked, and propeller swaps.
  - **Takeaway:** a tiny recurrent state is *sufficient* for parametric adaptation.
- **Recent extensions.**
  - *Embodiment-conditioned generalists* (arXiv 2606.10857†, and a library-based initialisation in 2602.15533†) extend this to multirotor layouts other than the standard X.
  - **Learned INDI** (PMLR 2026) replaces INDI's dependence on measured acceleration with a neural residual.

**Critique.**

- Every generalisation claim above is over *parameters of a fixed structural family, with trustworthy sensors*. "Out of distribution" means parameter extrapolation, not a new failure mechanism.
- Adaptation is *unattributed*. The latent has no assigned meaning, so there is no way to ask "what does the controller think happened?", let alone check the answer. That blocks any safety case.
- When two explanations predict the same data but call for opposite actions, the policy commits to whichever one the training prior favoured.

### 2.2 Classical baselines that ML papers underweight

- **INDI** (incremental nonlinear dynamic inversion; Smeur et al.; Tal & Karaman 2018).
  - **How it works:** it cancels unmodelled dynamics using *measured* accelerations, and needs only a model of control effectiveness.
  - **Strength:** this makes it remarkably robust to model error. INDI-based fault-tolerant control has flown quadrotors after the loss of one rotor, and even of two opposing rotors (Sun et al., T-RO 2021).
  - **Weakness:** the mirror image of that strength. INDI *trusts its sensors*.
- **L1 adaptive control** separates how fast the controller adapts from how robust it is. It was flight-tested on NASA's AirSTAR Generic Transport Model, a subscale jet, around 2009–2011.
- **History matters for credibility.**
  - Neural-network adaptive flight control flew on NASA's NF-15B in the 2000s (the IFCS programme).
  - NASA's Propulsion-Controlled Aircraft programme landed an MD-11 using only the throttles (1990s).
  - The Rohrs et al. (1985) counterexample showed classical adaptive controllers going unstable in the presence of unmodelled dynamics.

  What blocked adoption was assurance, and the interaction between pilots and adapting controllers, not performance. Aerospace reviewers will read a paper that ignores this as naïve.
- **Consequences for this project.**
  - Baselines must include tuned INDI (and INDI-FTC) and L1, not just PID or NMPC.
  - Neural-Fly's 35% gain over INDI is a realistic margin *within* a vehicle family. Beating INDI under *structural* novelty is unproven.
  - Under *sensor* faults INDI may well be *worse* than learned methods. That would be an interesting finding in itself.

### 2.3 RL and world models in flight

- **Racing and agile flight.**
  - **Swift** (Kaufmann et al., *Nature* 2023) beat human champion drone racers.
  - **Song et al.** (*Science Robotics* 2023) argue that RL beats optimal control in racing because it optimises the task objective directly and is robust to unmodelled effects.
  - **MonoRace** (2026) reached champion level using a single camera. It is notable for one hand-engineered patch: when the accelerometer saturates, a model-based acceleration predictor takes over. That is hard-coded *sensor-fault attribution* inside a learned stack. Practitioners hit this problem and solve it ad hoc.
- **World models.**
  - *Dream to Fly* (Romero et al., 2025) uses DreamerV3 to go straight from camera pixels to commands for racing. In flight, world models are used today for sample efficiency and perception, not for adaptation or safety.
  - Generalist world-model agents (DreamerV3, TD-MPC2) generalise across tasks in simulation. But their single, opaque latent is hard to calibrate off-distribution (ensembles become overconfident under shift; Ovadia et al. 2019), and it is opaque to safety arguments.
- **Fixed-wing and crewed aircraft.**
  - Recent RL work includes spin recovery with PPO on an F-18 HARV model (2026†) and upset recovery with SAC for jet trainers (2026†). A PPO generalist across several fixed-wing aircraft (2026†) also exists.
  - DARPA's AlphaDogfight (2020) and the AI-flown sorties of the X-62A VISTA (2023–24) flew under runtime-assurance supervision.
  - Fixed-wing learning lags multirotors by roughly 3–5 years, and mostly does not address adaptation or fault ambiguity.

### 2.4 Fault-tolerant control (FTC)

- **Classical active FTC** combines fault detection and isolation (FDI) with reconfiguration of the controller.
  - **FDI techniques:** analytical redundancy, banks of observers, and multiple-model methods (MMAE and IMM).
  - **Strength:** the hypotheses are explicit.
  - **Weaknesses:** it needs an enumerated list of faults, each with its own model. Integrating FDI with the controller is a long-acknowledged sore point.
  - **Active diagnosis:** it has long been recognised that some faults in redundant actuators cannot be detected from normal inputs without deliberate excitation. That observation motivated *active* fault diagnosis.
- **Learning-based FTC.**
  - *Learn-to-Recover* (ICRA 2020) trains an RL policy against actuator and sensor faults and attacks, without explicit diagnosis, **in simulation only**.
  - A Selector-Controller passive FTC handles any single-rotor failure and was flown on real hardware (2025).
  - Other examples are transformer-based online adaptation for FTC (2025†) and meta-RL for severe rotor loss (2022).
  - Active fault diagnosis by RL or planning exists outside flight. Examples are cart-pole (2026†) and counterfactual diagnosis of perception faults using MCTS (2025).
- **Gap:** I found no learned flight controller that reasons explicitly over sensor, actuator and aero hypotheses and tells them apart in closed loop.

### 2.5 Uncertainty and safety

- **Safety from reachability.**
  - Hamilton–Jacobi (HJ) reachability and control barrier functions (CBFs) are the standard tools. DeepReach approximates HJ reachability with neural networks in high dimension.
  - **Parameter-conditioned reachable sets** (Borquez, Nakamura, Bansal, ICRA 2023) treat changing parameters as extra "virtual" states. Safety assurances can then be updated online by lookup.
  - Their limits are that they condition on a single parameter value rather than a belief, cannot handle changes that are not parameterised, and give no formal guarantee for the neural approximation. That last point was partly addressed later with scenario optimisation and conformal verification.
- **Belief-aware safety.** The closest conceptual ancestors of belief-conditioned recoverability are:
  - Fisac et al. (2017), a general safety framework for learning-based control that refines its HJ safety analysis with Bayesian updates;
  - Bansal, Bajcsy et al. (2019/20), HJ reachability over a joint space of state and model belief, used for predicting human motion.
- **UNISafe** (Seo, Nakamura, Bajcsy, 2025).
  - **Method:** it uses the world model's epistemic uncertainty and an out-of-distribution threshold calibrated by conformal prediction. It then runs reachability over the joint space of latent state and uncertainty.
  - **Limits:** it is demonstrated on manipulation, with guarantees at the component level only. The authors themselves call for a system-level failure monitor.
- **Safe dual control** (learning about the system while acting on it safely) is an active area directly adjacent to this proposal:
  - Lew et al. (2020/22): meta-learned features with Bayesian adaptation of the last layer, and exploration under chance constraints;
  - Hu, Isele, Bae, Fisac (IJRR 2024): dual control that accounts for a safety shield;
  - Davydov et al. (RA-L 2025): actively gathering information while driving at the limits of handling;
  - Panagou group (2025–26): gatekeeper-based safe dual control of a quadrotor under *parametric* uncertainty.

  None of these covers:
  - discrete hypotheses about *which mechanism* failed, when those hypotheses are confounded (sensor vs actuator);
  - fast, open-loop-unstable regimes;
  - learned recoverability;
  - an open set of hypotheses.
- **Conformal prediction under shift.**
  - Adaptive conformal inference gives long-run coverage under arbitrary shift.
  - Robust conformal runtime verification assumes the shift is bounded in f-divergence (2023/24).
  - Recent work studies "endogenous" shift, where the controller itself changes the data distribution (2025).
  - Robust conformal CBF/CLF controllers use a shift budget (2026†).

  **The key mismatch:** all of these assume the shift is bounded or gradual, and a step fault violates that by construction. What is needed is fast *change detection* with false-alarm control that is valid at any stopping time (sequential tests, e-processes), not coverage.

### 2.6 Benchmarks and datasets

- **Simulators.**
  - Established platforms: gym-pybullet-drones, safe-control-gym, OmniDrones, Aerial Gym, RotorPy.
  - **AdaptiveQuadBench** (Zhang, Tao, Cheng, Hovakimyan, Mueller; Oct 2025) adds wind, payload shifts, rotor faults and latency, together with a library of controllers.
  - **MetaTune** (2026) trains in a differentiable JAX simulator and transfers zero-shot to PX4-Gazebo.
  - **JSBSim** is the standard open-source flight-dynamics model for fixed-wing aircraft.
- **Real fault data.**
  - **ALFA** (CMU, IJRR 2021): 47 processed real fixed-wing flights, with 23 engine failures and 24 control-surface faults across 7 types.
  - **RflyMAD** (Beihang, IJRR 2025): 5,629 multicopter cases, 11 fault types, including 497 real flights.
  - **UAV-SEAD** (2026†): anomalies in state estimation.
- **Critique.**
  1. No benchmark holds out entire *mechanisms* for testing.
  2. None contains *confounded* faults.
  3. None scores *attribution* or how well *predicted risk is calibrated*.
  4. The real fault datasets are open-loop logs, so they can evaluate detectors but not controllers.

### 2.7 Certification reality

- **Current guidance and standards.**
  - **EASA:** the AI Concept Paper Issue 2 (March 2024) covers Level 1 and Level 2 machine learning. Issue 3, now in consultation, extends coverage to RL and symbolic AI.
  - **SAE G-34 / EUROCAE WG-114:** the process standard ARP6983 / ED-324 is planned for around Q4 2026. Its first issue reportedly targets *supervised, offline* machine learning.
  - **ASTM F3269** defines runtime-assurance architectures in the style of Simplex.
- **Implication.** Updating network weights in flight has no near-term certification path.
  - A policy that adapts *in context* with frozen weights is a fixed function of its input history. In principle it falls under offline learning assurance. *Verify the exact Concept Paper wording before relying on this.*
  - Combined with runtime assurance, this is the architecture industry could plausibly certify.
  - A method that keeps its weights frozen *and* outputs explicit, checkable hypotheses will transfer to industry far more easily than one that does not.

---

## 3. Assumptions worth challenging

1. **"Adaptation means inferring a context vector."**
   - A single latent conflates causes. Adapting without attribution is dangerous exactly when two causes demand opposite responses. Examples:
     - a faulty angle-of-attack (AoA) vane vs a genuinely high AoA;
     - a blocked pitot tube vs a genuine overspeed;
     - a gyro scale error vs a motor losing effectiveness.
   - *Design consequence:* factor the world model into separate channels for aero, actuators, sensors and disturbances. Competing explanations can then be represented at all.
2. **"Broader domain randomisation means better generalisation."**
   - Randomisation buys *interpolation within its support*, and evaluations rarely hold out whole *classes* of mechanism.
   - An implicit belief inside an RNN also bakes in the *training prior* over faults. Real fault frequencies differ, and an implicit belief offers no handle for correcting that.
   - *Consequence:* evaluate on splits that leave one mechanism out, test combinations of faults, and shift the fault prior.
3. **"Ensemble disagreement is the uncertainty that matters."**
   - The quantities that matter for decisions are *P(unrecoverable within T | history)* and *"is my hypothesis class even right?"*.
   - Ensembles become overconfident under shift. Standard conformal guarantees break under step faults and under shift that the closed loop itself causes.
   - *Consequence:* calibrate risk predictions directly, and detect misspecification with change detection that is valid at any stopping time.
4. **"Task generality needs a world model and a planner."**
   - In flight, tracking any feasible reference is already task-general. The real constraint is knowing *which references are still feasible* once something has broken.
   - *Consequence:* the most valuable output of a flight world model is not a prediction to plan with, but the **set of states that can still be recovered after the fault**.
5. **"Learning beats classical robust control."**
   - Within a vehicle family, gains over well-tuned INDI or L1 are tens of percent, not orders of magnitude. Many ML papers compare against weak baselines.
   - *Consequence:* include INDI, INDI-FTC, L1 and an IMM/MMAE bank, or controls reviewers will reject the paper.
6. **"Online learning, meaning weight updates, is needed to adapt."**
   - In-context adaptation with frozen weights already adapts within milliseconds (RAPTOR). It avoids the instabilities seen in classical adaptive control (Rohrs et al.). It is also the only form of adaptation with a plausible certification story.
   - *Consequence:* adapt through inference, not gradient steps.
7. **"Safety means never leaving a fixed safe set."**
   - After severe damage, the safe set for the nominal mission may be *empty*. Safety then becomes graceful degradation: choosing what to give up, such as yaw authority, altitude, or the runway.
   - *Consequence:* treat safety as a decision over a recoverable set that is shrinking and uncertain.

---

## 4. Fundamental open problems (ranked by importance × tractability)

| # | Problem | Why it is fundamental | Closest partial answers |
|---|---|---|---|
| **P1** | **Attribution when the cause is not identifiable.** Several mechanism hypotheses fit the data but call for conflicting actions. | Inference alone cannot resolve it; it needs decisions *about* information. It is the dominant pattern in automation-related accidents. | Classical active FDI (enumerated models); Learn-to-Recover (no attribution) |
| **P2** | **Safe active identification in fast, unstable regimes.** When is it safe to excite the vehicle to learn what broke? | This is the dual-control problem (Feldbaum, 1960s). It is intractable in general and still open with learned models and tight time budgets. | Safe dual control under parametric uncertainty (Panagou; Lew; Hu & Fisac) |
| **P3** | **Knowing that none of the hypotheses fit** (an open set, or a misspecified model), with a calibrated false-alarm rate that holds in closed loop. | Every learned model is wrong somewhere. The question is whether you know *when*. | UNISafe (latent OOD); robust and adaptive conformal methods (which assume bounded shift) |
| **P4** | **Online recoverability under a belief.** What can still be reached safely, given *uncertain* post-fault dynamics? | It underpins graceful degradation and the choice of emergency landing site. | Parameter-conditioned HJ (single parameter value); Delft safe-envelope databases (enumerated damage) |
| **P5** | **Generalisation across flight regimes**: multirotor, fixed-wing and VTOL. | It would be a genuine "flight foundation model". | Generalists exist only within one vehicle family |
| **P6** | **Evaluation methodology**: held-out mechanisms, confounded faults, calibration metrics. | Without these, generalisation claims cannot be falsified. | AdaptiveQuadBench (no held-out mechanisms, no confounding) |

---

## 5. Candidate research directions, compared

| Direction | Core claim | Novelty | Impact (ML / aero) | Feasibility (small team, 12–18 mo) | Risk of being scooped |
|---|---|---|---|---|---|
| **D1 Adaptation under Ambiguity** (P1+P2+P3) | State-of-the-art adaptive controllers misattribute confounded faults. A probe-or-hedge frontier governs when a vehicle should experiment on itself. Structured beliefs plus probing gated by recoverability fix this. | **High** | High / **High** | Medium–High | Medium (adjacent groups are active) |
| **D2 Belief-conditioned recoverability** (P4) | Amortised reachability conditioned on a *belief* about the dynamics enables graceful degradation and emergency-landing choice in real time. | Medium–High | Medium / **High** | Medium (reachability in high dimension is hard) | Medium (Bansal and Bajcsy groups) |
| **D3 Closed-loop competence monitors** (P3) | Sequential tests that remain valid at any stopping time certify when an adaptive controller's model still holds, and trigger runtime assurance when it does not. | Medium | Medium / High | **High** | Medium |
| **D4 Cross-regime flight foundation policy** (P5) | One policy for multirotor, fixed-wing and VTOL. | Medium (mostly scaling) | Medium–High / Medium | Low–Medium (compute) | **High** (the Loianno, Scaramuzza and Mueller labs are close) |
| **D5 Benchmark only** (P6) | A benchmark with held-out mechanisms and confounded faults. | Low–Medium | Medium / Medium | **High** | Low |

**Recommendation.**

- Make **D1** the spine of the project.
- **D2 and D3 are not alternatives.** They form D1's safety layer: probing must be gated by recoverability, and the hypothesis set must be guarded by a misspecification monitor.
- **D5** falls out as a by-product.
- **Avoid D4 head-on.** It is a compute race against well-resourced labs, and it yields little scientific insight per GPU-hour.

---

## 6. Recommended direction: Adaptation under Ambiguity

### 6.1 Thesis

> Adapting to *what the data says* is not the same as adapting to *what happened*. When faults are confounded, safe flight needs a controller that does four things. It represents competing explanations. It acts robustly across them when it can. It disambiguates actively when it must, and only when that is recoverable. And it recognises when no explanation fits.

### 6.2 Formalisation (sketch)

**System and belief.**

- **Dynamics:** `x_{t+1} = f(x_t, A_{θa}(u_t); θ_f) + w_t`, where `A_{θa}` is the actuator map and `θ_f` the aerodynamic parameters.
- **Observations:** `y_t = S_{θs}(x_t) + v_t`, where `S_{θs}` is the sensor model.
- **Hypotheses:** a hypothesis is `h = (m, θ)`. Here `m ∈ M ∪ {⊥}` is the mechanism class: actuator, sensor, aero, disturbance, or a combination. `⊥` means "none of the modelled mechanisms".
- **Belief:** `b_t(h) = p(h | y_{0:t}, u_{0:t−1})`.

**Ambiguity and conflict.**

- **Ambiguity set:** `H_ε(t) = {h : b_t(h) ≥ ε}`.
- **Safe actions:** `A_safe(h, x)` is the set of actions from which recovery remains feasible under hypothesis `h`. It is given by a recoverability value `V_h(x)`.
- **Conflict:** occurs when `⋂_{h∈H_ε} A_safe(h, x) = ∅`, that is, when no single action is safe under every plausible hypothesis.

**Elementary bound.** For any action `a`:

`P(unsafe | a, b_t) ≥ Σ_{h: a∉A_safe(h)} b_t(h)`

- A controller that acts on a point estimate `ĥ` takes risk roughly equal to the posterior mass of every hypothesis its action is not safe for.
- A controller that is aware of its belief can minimise that mass. That is a **hedge**.
- When the conflict cannot be hedged away, the only way to reduce risk is to *change the belief*. That is a **probe**.

**Time budget.**

- **Time to unrecoverability:** each hypothesis has one, `τ_h(x, a)`: how long the current or hedging action can be held before `V_h < 0`.
- **Information from a probe:** a probe lasting `τ_p` yields an expected separation between hypotheses of `I(h; y_{t:t+τ_p})`.
- **Admissibility:** probing is admissible only if `V_h ≥ 0` holds for every `h ∈ H_ε` throughout `τ_p + τ_commit`.

**The probe-or-hedge frontier.** These quantities define a frontier in the space of (separation rate × min_h τ_h). It gives a crisp prediction that does not depend on the platform:

- multirotor actuator and sensor faults, where `τ` is under a second, should mostly sit in the *hedge* region;
- fixed-wing air-data faults, where `τ` is seconds to minutes, should sit in the *probe* region. That matches human procedures for unreliable airspeed.

**Misspecification.** A sequential test (an e-process, or a CUSUM-type statistic) runs on the residuals of the posterior predictive. Its false-alarm rate is controlled at any stopping time. When it flags `⊥`, control passes to the runtime-assurance fallback.

### 6.3 Falsifiable hypotheses

- **H1 (the phenomenon).** Learned adaptive controllers trained on standard randomisation will misattribute confounded faults. This covers in-context RNN or transformer policies, RMA-style latent policies and Neural-Fly-style controllers. Their crash rates will be far above those of an oracle that knows the fault class. INDI will be robust to some sensor faults and *worse* on others.
- **H2 (the crux).** Add sensor faults to the training randomisation of an end-to-end in-context policy. On *conflicting* fault pairs, this will narrow the gap to a belief-aware oracle but not close it. The policy will also degrade under prior shift, when fault frequencies at test time differ from training. **This is the kill test: if H2 is false, pivot.**
- **H3 (structure helps).** Explicit factored hypotheses, plus probing gated by recoverability, will close at least half of the gap to the oracle on conflicting pairs. The cost in nominal performance will be no more than a few percent.
- **H4 (the frontier).** The empirically optimal behaviour will switch between hedging and probing where the analysis of time-to-unrecoverability against information rate predicts. This should hold on both multirotor and fixed-wing.
- **H5 (open set).** The monitor will detect held-out mechanisms with a bounded false-alarm rate. The fallback will improve survival compared with flying without the monitor.

### 6.4 Method sketch (no implementation yet)

1. **Physics-structured, factored world model.** It has five channels:
   - a known 6-DoF rigid-body core;
   - an actuator channel: effectiveness, stuck state and delay for each actuator;
   - an aero channel: parametric terms plus a learned residual conditioned on a latent;
   - a sensor channel: bias, scale error, frozen value, dropout and blockage for each sensor;
   - a disturbance channel: a wind process.

   The factorisation *is* the causal structure that makes competing explanations representable.
2. **Amortised multimodal posterior.**
   - **Architecture:** a sequence encoder outputs a mixture over mechanism classes, with a normalising flow over each class's continuous parameters.
   - **Training:** simulation-based inference (neural posterior estimation) on very large numbers of randomised rollouts.
   - **Deployment:** weights are frozen. Because the priors are explicit, they can be corrected at test time, in the spirit of PriorGuide (2025).
3. **Recoverability critic `V(x, b)`, conditioned on the belief.**
   - **Training:** offline, either DeepReach-style or as an RL safety critic, with a robust variant that takes the worst case over `H_ε`.
   - **Claims:** it is calibrated and verified statistically (scenario optimisation or conformal methods). It is not claimed to be exact.
4. **Controller.**
   - **Planner:** sampling-based MPC in belief space, over particles representing the hypotheses. The objective is task cost plus λ × information gain, subject to recoverability under every `h ∈ H_ε`.
   - **Hedging:** when no admissible probe exists, the controller hedges.
   - **Deployment:** the MPC is distilled into a compact, belief-conditioned policy for embedded hardware, again with frozen weights.
5. **Runtime assurance.** If the monitor flags `⊥`, or `V` drops below a margin, control switches to a robust fallback. The fallback is INDI plus an attitude-and-thrust hold, the analogue of a pilot's "pitch and power".

*Honest note:* every component has precedent. The contribution is:

- the phenomenon;
- the frontier theory;
- active attribution over *confounded mechanisms*, gated by recoverability;
- the evidence on hardware.

### 6.5 Closest prior work and what D1 adds

| Prior work | What it does | What D1 adds |
|---|---|---|
| RAPTOR / Extreme Adaptation | In-context adaptation to parameter changes | Confounded mechanisms, explicit hypotheses, safety gating |
| Learn-to-Recover (2020) | RL policy robust to sensor and actuator faults | Attribution, probing, recoverability, hardware, held-out evaluation |
| Classical active FDI (MMAE/IMM) | Explicit banks of hypotheses | Learned residuals, open-set detection, probing gated by recoverability and integrated with control |
| Safe dual control (Lew; Hu & Fisac; Panagou) | Safe information gathering about *parameters* | *Discrete, confounded mechanisms*; fast unstable regime; learned recoverability; open set |
| UNISafe / parameter-conditioned HJ | Reachability that accounts for uncertainty | A belief over mechanisms, active disambiguation, application to flight |

### 6.6 What separates a groundbreaking result from an incremental one

**Groundbreaking**, if it delivers three things:

1. a clean demonstration that the field's dominant way of adapting has a systematic, safety-critical blind spot;
2. a *general principle*, the probe-or-hedge frontier, that predicts behaviour across vehicle classes and agrees with long-standing airmanship;
3. evidence on hardware.

**Incremental**, if it reduces to "we added a fault classifier to an RL controller". Avoid that by leading with H1 and H4, not with the architecture.

---

## 7. Benchmark and evaluation protocol

**Platforms.**

- **P1:** a vectorised quadrotor simulator in JAX, cross-validated against RotorPy and AdaptiveQuadBench.
- **P2:** fixed-wing in JSBSim, used to test the frontier.
- **P3:** Crazyflie-class hardware.
- **P4:** real fault logs, used offline to evaluate attribution only (ALFA, RflyMAD).

**Fault library.**

- **Multirotor.**
  - *Actuators:* loss of effectiveness; stuck or dead motor; motor delay; chipped propeller (changed thrust and torque coefficients, plus vibration).
  - *Airframe:* shifts in CG, mass and inertia.
  - *Sensors:* gyro and accelerometer bias, scale error and saturation; barometer step and drift; bias, latency and dropout in position/GNSS.
  - *Disturbances:* steady wind, Dryden turbulence, gust fronts, ground effect.
- **Fixed-wing.**
  - *Control surfaces and engine:* loss of effectiveness, stuck surface, floating surface; engine failure.
  - *Airframe:* asymmetric damage, modelled by modified aero coefficients; icing, modelled by a changed lift curve.
  - *Air data and IMU:* blocked pitot total or static pressure; frozen AoA vane; IMU faults.

**Splits.**

| Split | What it tests |
|---|---|
| S1 | Parameters within the training distribution |
| S2 | Parameter extrapolation |
| S3 | One mechanism class left out of training |
| S4 | Combinations: train on single faults, test on pairs |
| **S5** | **Mined conflicting pairs.** An automatic search finds hypothesis pairs whose observation sequences are ε-close over a horizon T, but whose safe-action sets are disjoint. This "ambiguity mining" is a contribution in itself. |
| S6 | Shifted fault prior: fault frequencies at test time differ from training |
| S7 | Hardware flights and real logs |

**Metrics.**

- **Survival / crash rate** (primary), with Wilson confidence intervals.
- **Recovery quality:** time to stabilise; tracking RMSE after recovery; rate of constraint violations.
- **Attribution:** the **rate of confident but wrong attributions**, and the time taken to reach the correct attribution.
- **Risk calibration:** Brier score and ECE of the predicted P(unrecoverable within T).
- **Monitor:** detection delay against false-alarm rate.
- **Cost:** loss of nominal performance; onboard latency and memory.

**Baselines (all required).**

- tuned geometric or PID control;
- INDI and INDI-FTC;
- L1 adaptive control;
- a Neural-Fly-style controller;
- an RMA- or Extreme-Adaptation-style latent policy;
- a RAPTOR-style in-context policy;
- **the same in-context policy retrained on exactly the same fault distribution as our method** (the key control);
- an IMM/MMAE bank built on the same fault set;
- a DreamerV3 or TD-MPC2 agent;
- an **oracle** that knows the true fault (the upper bound).

**Rigour.**

- held-out splits pre-registered before results are seen;
- at least 10 seeds per method;
- results stratified by ambiguity score;
- release of the code, the fault library and the pair-mining generator.

---

## 8. Phased plan with kill criteria

| Phase | Time | Work | Exit or kill criterion | Output |
|---|---|---|---|---|
| **0** | weeks 1–6 | JAX simulator with fault injection; INDI, L1 and an in-context RNN baseline; oracle; ambiguity mining; test H1 and H2 | **Kill D1 if** the in-context policy trained with sensor faults comes within about 10% of the oracle on S5 *and* is robust on S6. Then pivot to D2 or D3, reusing the infrastructure. | Internal report; possibly a workshop paper |
| 1 | months 2–5 | Frontier theory; factored world model and amortised posterior; hedging controller | H3 partly supported in simulation | Paper 1 (CoRL / ICRA / L4DC) |
| 2 | months 5–9 | Recoverability critic; gated probing; misspecification monitor; frontier test on fixed-wing | H4 and H5 | Paper 2 (RSS / NeurIPS) |
| 3 | months 9–15 | Hardware: sensor faults injected in firmware, physically damaged propellers, wind from fans; benchmark release | Survival gains reproduced on hardware | Benchmark (NeurIPS Datasets & Benchmarks); journal paper (T-RO, or *Science Robotics* if the hardware results are strong) |

---

## 9. Risks and honest weaknesses

- **"RNN hidden states already are beliefs."** True in principle, and that is exactly why H2 is the kill test. Even if the RNN closes the gap, explicit beliefs still buy three things: verifiability, robustness to prior shift, and explanations. That is a weaker angle, but still publishable.
- **"This is just 1990s active FDI."**
  - *Mitigation:* compare directly against an IMM/MMAE bank.
  - *Defensible novelty:* the open set of hypotheses, learned residuals, probing gated by recoverability, and the frontier theory.
- **Mined ambiguities may be artefacts of the simulator.**
  - Anchor them in physically documented cases, such as air-data faults, or a gyro scale error vs motor loss of effectiveness.
  - Confirm them on hardware.
- **Recoverability estimates in high dimension are approximate.** Claim calibrated statistical estimates, not guarantees, and keep runtime assurance behind them.
- **Compute.** Belief-space MPC will not run on a microcontroller, so distil it into a small policy.
- **Scope creep.** Do multirotor first. Use fixed-wing only to test the frontier.
- **Being scooped.** Adjacent groups (Panagou, Fisac, Bajcsy, Bansal) are active. Move fast on Phase 0 and post a preprint early.

---

## 10. Decisions needed

1. **Resources.**
   - *GPU:* Phase 0 runs fine on a laptop CPU with JAX; Phases 1–2 want one GPU.
   - *Hardware budget:* a Crazyflie plus a positioning system costs roughly £1–3k.
2. **Context and timeline.**
   - Is this a PhD chapter or thesis (2–3 years), or a 6–12-month project?
   - Will you work solo, or with a lab or supervisor?
3. **Platform emphasis.** Start with multirotors (fast and cheap), or put more weight on fixed-wing (more pull from aerospace, harder hardware)?
4. **Appetite for theory.** How far should the frontier result go towards formal statements, rather than empirical characterisation?

---

## References (selected; † = read at abstract level only)

**Learned adaptive control and generalist policies**
- O'Connell et al., *Neural-Fly enables rapid learning for agile flight in strong winds*, Science Robotics 2022. [arXiv:2205.06908](https://arxiv.org/abs/2205.06908)
- Eschmann, Albani, Loianno, *RAPTOR: A Foundation Policy for Quadrotor Control*, 2025/26. [arXiv:2509.11481](https://arxiv.org/abs/2509.11481)
- Zhang et al., *A Learning-based Quadcopter Controller with Extreme Adaptation*, T-RO 2025. [arXiv:2409.12949](https://arxiv.org/abs/2409.12949)
- Zhang et al., *Learning a Single Near-hover Position Controller for Vastly Different Quadcopters*, ICRA 2023. [arXiv:2209.09232](https://arxiv.org/abs/2209.09232)
- *Embodiment-conditioned Generalist Control for Multirotor Aerial Robots* †. [arXiv:2606.10857](https://arxiv.org/abs/2606.10857)
- Library-based reuse of multirotor policies (RA-L 2026) †. [arXiv:2602.15533](https://arxiv.org/abs/2602.15533)
- *Learned Incremental Nonlinear Dynamic Inversion for Quadrotors with and without Slung Payloads*, PMLR 2026. [link](https://proceedings.mlr.press/v331/cobo-briesewitz26a.html)
- Tal & Karaman, *Accurate Tracking of Aggressive Quadrotor Trajectories using INDI and Differential Flatness*. [arXiv:1809.04048](https://arxiv.org/abs/1809.04048)
- Also cited: Kumar et al., *RMA: Rapid Motor Adaptation*, RSS 2021; Huang et al., *DATT*, CoRL 2023; Sun et al., *Incremental Nonlinear Fault-Tolerant Control of a Quadrotor with Complete Loss of Two Opposing Rotors*, T-RO 2021.

**RL, world models, fixed-wing**
- Romero et al., *Dream to Fly: Model-Based RL for Vision-Based Drone Flight*, 2025. [arXiv:2501.14377](https://arxiv.org/abs/2501.14377)
- *MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI*, 2026. [arXiv:2601.15222](https://arxiv.org/abs/2601.15222)
- Spin recovery with PPO, 2026 †. [arXiv:2601.06439](https://arxiv.org/abs/2601.06439)
- Aircraft upset recovery with RL, 2026 †. [arXiv:2604.24355](https://arxiv.org/abs/2604.24355)
- Single controller for multiple aircraft by deep RL, 2026 †. [link](https://mrforum.com/product/9781644904251-3/)
- Also cited: Kaufmann et al., *Champion-level drone racing using deep RL*, Nature 2023; Song et al., *Reaching the limit in autonomous racing*, Science Robotics 2023; DreamerV3; TD-MPC2.

**Fault tolerance and diagnosis**
- *Learning-Based Passive Fault-Tolerant Control of a Quadrotor with Rotor Failure*, 2025. [arXiv:2503.02649](https://arxiv.org/abs/2503.02649)
- *RL-based Fault-Tolerant Control for Quadrotor with Online Transformer Adaptation*, 2025 †. [arXiv:2505.08223](https://arxiv.org/abs/2505.08223)
- *Learn-to-Recover*, ICRA 2020. [PDF](https://friends.cs.purdue.edu/pubs/ICRA20_LearnToRecover.pdf)
- *A Counterfactual Reasoning Framework for Fault Diagnosis in Robot Perception Systems*, 2025. [arXiv:2509.18460](https://arxiv.org/abs/2509.18460)

**Safety, uncertainty, dual control**
- Borquez, Nakamura, Bansal, *Parameter-Conditioned Reachable Sets for Updating Safety Assurances Online*, ICRA 2023. [arXiv:2209.14976](https://arxiv.org/abs/2209.14976)
- Seo, Nakamura, Bajcsy, *Uncertainty-aware Latent Safety Filters (UNISafe)*, 2025. [arXiv:2505.00779](https://arxiv.org/abs/2505.00779)
- Fisac et al., *A General Safety Framework for Learning-Based Control in Uncertain Robotic Systems*, 2017. [arXiv:1705.01292](https://arxiv.org/abs/1705.01292)
- Bansal, Bajcsy et al., *An HJ Reachability-Based Framework for Predicting and Analyzing Human Motion for Safe Planning*. [arXiv:1910.13369](https://arxiv.org/abs/1910.13369)
- Lew et al., *Safe Active Dynamics Learning and Control*. [arXiv:2008.11700](https://arxiv.org/abs/2008.11700)
- Naveed, Agrawal, Panagou, *A Formal Gatekeeper Framework for Safe Dual Control with Active Exploration*, 2025. [arXiv:2510.06351](https://arxiv.org/abs/2510.06351)
- *Trajectory Planning for Safe Dual Control with Active Exploration*, 2026. [arXiv:2604.15507](https://arxiv.org/abs/2604.15507)
- Zhao et al., *Robust Conformal Prediction for STL Runtime Verification under Distribution Shift*. [arXiv:2311.09482](https://arxiv.org/abs/2311.09482)
- Binny & Dixit, *Who Moved My Distribution?*, 2025. [arXiv:2511.11567](https://arxiv.org/abs/2511.11567)
- *Robust Conformal CBF and CLF Controllers via Iterative Policy Updates*, 2026 †. [arXiv:2606.15366](https://arxiv.org/abs/2606.15366)
- *PriorGuide: Test-Time Prior Adaptation for Simulation-Based Inference*, 2025. [arXiv:2510.13763](https://arxiv.org/abs/2510.13763)
- Ramos et al., *BayesSim*. [arXiv:1906.01728](https://arxiv.org/abs/1906.01728)

**Benchmarks and datasets**
- Zhang et al., *A Simulation Evaluation Suite for Robust Adaptive Quadcopter Control (AdaptiveQuadBench)*, 2025. [arXiv:2510.03471](https://arxiv.org/abs/2510.03471)
- *MetaTune: Adjoint-based Meta-tuning via Robotic Differentiable Dynamics*, 2026. [arXiv:2603.27313](https://arxiv.org/abs/2603.27313)
- Keipour, Mousaei, Scherer, *ALFA: A Dataset for UAV Fault and Anomaly Detection*. [arXiv:1907.06268](https://arxiv.org/abs/1907.06268)
- *RflyMAD: A Dataset for Multicopter Fault Detection and Health Assessment*. [arXiv:2311.11340](https://arxiv.org/abs/2311.11340)
- *UAV-SEAD: State Estimation Anomaly Dataset for UAVs* †. [arXiv:2602.13900](https://arxiv.org/abs/2602.13900)

**Certification and safety statistics**
- EASA Artificial Intelligence Concept Paper, Issue 2. [EASA](https://www.easa.europa.eu/en/document-library/general-publications/easa-artificial-intelligence-concept-paper-issue-2)
- Status of ARP6983 / ED-324. [StandICT](https://standict.eu/standards-repository/eurocae-wg114-sae-g34-artificial-intelligence-aviation-as6983-process-standard)
- IATA 2025 safety performance: no LOC-I accidents in 2025, the second such year after 2020. ICAO data puts LOC-I at about one third of fatal accidents within its high-risk categories. [summary](https://www.travelextra.ie/iata-reports-eight-fatal-accidents-and-394-fatalities-in-2025/)
- IATA presentation on LOC-I to ICAO, 2021: 53 LOC-I accidents from 2011 to mid-2021, 49 of them fatal, with 1,858 fatalities. [PDF](https://www.icao.int/ESAF/Documents/meetings/2021/LOC-I%20and%20UPRT%20Workshop%2010-11%20November%202021/Presentations/Day%202/IATA-LOC-I%20Presentation%20to%20ICAO%20AFI%20Workshop%20Nov11_2021_Final.pdf)
- Zhang, de Visser, Chu (TU Delft): database-driven online estimation of the safe flight envelope for damaged aircraft.
