# M3 technical-spec recommendations: L1 adaptive policy, adaptive allocation, L2 recovery, INDI baseline and evaluation

This synthesises six research reports, dated 6 Nov 2026. I ran nothing. I read the repo facts from source: docs/03 §4.2, §5, §6.1 and §7.2; docs/validation/m2_report.md V15; the actuator, faultlab and generator modules.

Tags used throughout:
- **[U]**: the recommendation rests on a literature claim not confirmed this session (from memory, a secondary summary or a single snippet).
- **[D]**: it rests on a report author's own derivation or estimate, which needs a numerical check before it is quoted.

## 0. Six decisions that shape everything

1. **Split the control surfaces before any RL.** Three reports read the code independently and found the same blocker:
   - aerojax has three lumped channels: `actuators.py` holds `surfaces (3,)`, and `faultlab/slots.py` has `SURFACE_INDEX = {aileron, elevator, rudder}`.
   - With three surfaces for three moments, B is square and weighted-least-squares (WLS) allocation reduces to B⁻¹ with clipping. Nothing can be re-allocated.
   - A full aileron or elevator hard-over is physically unrecoverable on most airframes. On the Rascal, rudder through dihedral cancels about 24% of a full aileron hard-over, leaving about 70°/s of roll [D]. On the Aerosonde about 33° of sideslip would be needed [D].
   - Without the split, the "adaptive allocation" claim and the hard-over headline are both empty, and so is Flight 0's "jammed aileron re-allocated to rudder and the other aileron". Budget 3–5 days in week 13.
2. **L1 outputs normalised control angular-acceleration demands plus throttle into one shared WLS allocator** with online effectiveness estimation (non-incremental mode). "RL over INDI" is the pre-registered ablation and the fallback primary: the same policy feeding the incremental mode of the same allocator.
3. **The teacher-student recipe.**
   - One privileged teacher, conditioned through FiLM on a compact privileged vector that describes the fault's effects.
   - DAgger distillation into a GRU student of about 31k parameters with an auxiliary head.
   - PPO fine-tuning only if a gate fails.
   - Asymmetric (privileged) critics everywhere.
4. **L2 is trained in one stage and is the first thing cut.** It uses asymmetric recurrent PPO with no teacher. Choose the headline hard-over so that L1 plus allocation handles it without L2; the headline then does not depend on L2.
5. **Pre-register before any test-split run.**
   - Choose the six airframes and each airframe's hard-over by rule, using a trim-reachability oracle and a hash draw.
   - Frozen-policy CIs come from a bootstrap that resamples airframes. Recipe CIs come from seeds, using rliable.
6. **Be honest about the schedule.** What follows is about 150 h of work for about 120 h available. Pre-declared cuts:
   - L2 becomes scripted recovery (the plan's own cut).
   - Leave-one-class-out (LOCO) drops to 2 folds.
   - G3 becomes the X8 and Zagi as case studies; procedural flying wings come after M3.
   - Handling-quality fits move to M6.
   - The L1-adaptive-control baseline in docs/03 §4.2 is dropped.

---

## 1. The L1 policy

### 1.1 Scope and interfaces
- **What it replaces.** L1 replaces B1's TECS, attitude and rate loops. Guidance (L1 nav), mission logic and the classical motor-cut contingency (`stack/contingency.py`; 83% → 86%) stay identical for every controller. A/B differences then come only from the inner loops.
- **References.** φ_ref comes from L1 nav. γ_ref comes from one shared altitude-shaping law used by every controller: γ_ref = clip(k_h·(h_ref − h)/V, ±γ_max). V_ref comes from the mission or the contingency.
- **Rates.** Policy at 50 Hz, with each action held for two control ticks. Allocator and estimator at 100 Hz. Physics at 200 Hz. The 50 Hz rate matches servo update rates; Bøhn et al. flew a real X8 with RL against ArduPlane PID (TNNLS 2023).
- **Units.** 1 policy step = 2 control steps = 4 physics steps. Quote every throughput and budget in policy steps per second.

### 1.2 Action space (4 outputs)
- **Moments.** a ∈ [−1, 1]³ through tanh, then ν_i = ν_max,i · a_i.
  - ν_max,i is the nominal control angular acceleration about axis i at V_cruise, computed by autoconfig: Σ_j |B̃_ij|·δ_max,j·q̄_cruise.
  - ν is the angular acceleration the surfaces add relative to the autoconfig trim deflections (u_p = u_trim), so a = 0 is trim on the nominal airframe.
- **Thrust.** Throttle in [0, 1], sent directly, not through the allocator.
- **Why angular acceleration and not raw moments.** One output range then means the same rotational response from 0.8 to 15 kg. This is the scale-aware thrust-plus-torque idea (arXiv 2503.09622) and the abstraction result of Kaufmann et al. (ICRA 2022). It is also INDI's virtual-control convention, so B2 can share the allocator.
- **Moment-coefficient demands** (the alternative raised in report 2) are not worth an M3 ablation.
- **What to disclose.** ν_max, u_trim and the nominal B̃ come from autoconfig, i.e. manufacturer-level configuration, the same information budget as B1. "Same frozen policy" means same weights. The per-airframe scalings are computed, not tuned.

### 1.3 Student observation (about 34 dimensions)
Bøhn et al. (ICUAS 2019) found a small observation plus history mattered most.

| Group | Contents | Dims |
|---|---|---|
| Inertial | gyro p, q, r; gravity vector in body axes (from the attitude estimate); accelerometer specific force | 9 |
| Air data | measured airspeed / V_cruise; q̄ / q̄_cruise; altitude rate | 3 |
| References | e_φ, e_γ, e_V/V_ref, φ_ref, γ_ref | 5 |
| Previous action | a_{t−1} (3) and throttle | 4 |
| Allocator feedback | attainable authority ν_k± (6); saturation residual ν − ν_achieved (3) | 9 |
| Nominal descriptor (ablated, A7) | V_stall, V_cruise, span, mass from autoconfig | 4 |

- **Excluded:** α, β, true airspeed and the true B.
- **Privileged boundary.** Privileged quantities travel only on the existing ADR-006 path (TruthSnapshot). Add a test asserting that the student's observation builder cannot read it.

### 1.4 Teacher privileged inputs: effects, not causes
About 30 dimensions, log-scaled or non-dimensional (report 1):
- **Effectiveness.** Per-axis effectiveness at the current q̄ relative to nominal (3). Per-surface health multipliers: 1 healthy, 0 dead or jammed, −1 reversed, fractions for reduced throw (8). The moment from a jam offset (3).
- **Damping.** C_lp, C_mq, C_nr scaled by V/b or V/c̄ (3).
- **Static stability.** C_mα, C_nβ, C_lβ (3).
- **Energy.** V/V_stall; thrust available against thrust required (2).
- **Actuators.** Servo τ and delay (2).
- **Air.** Mean wind in body axes and turbulence intensity (4); true α and β (2).

Architecture:
- An encoder MLP 30→64→z (12 dims).
- FiLM modulation of every actor hidden layer. Marquis & Farhood (arXiv 2604.03392) found FiLM or LoRA conditioning beat concatenation for fixed-wing actuator faults. Their numbers come from a secondary summary [U].
- Teacher actor: the current-step observation plus FiLM(z) feeding an MLP 256-256-128 with ELU, about 125k parameters. The teacher has no size limit.
- Describing faults by their effect on effectiveness rather than by fault ID also prepares for G5 (unseen fault classes).

Leakage rules:
- No future information, such as upcoming gusts.
- Reveal a fault only at or after onset, with a sampled detectability delay of 0.1–0.5 s, or once the surface has been commanded away from its jam position.
- This cheaply closes the worst imitation gaps: jams inside the deflection range, and reversed surfaces in unexcited cruise (Weihs et al., ADVISOR, NeurIPS 2021).

### 1.5 Critic
- A separate MLP (512, 256, 128). Its inputs: the observation, the full raw privileged parameters (log-scaled), the true state including α and β, wind, fault kind (one-hot), severity and time since onset, and the allocator's internal state (B̂, trace of P).
- In Learning to Fly in Seconds (RA-L 2024), removing the asymmetric critic cut real-flight success from 10/10 to 6/10. That ablation figure is from a secondary summary [U].
- The critic is never deployed, so it does not count against the 50k budget.

### 1.6 Student architecture
- **Layers:** Dense(34→64, ELU) → GRU(64) → Dense(64→64, ELU) → Dense(64→4), plus a state-independent log-std.
- **Size:** about 31k parameters deployable, plus an auxiliary head Dense(64→30) of about 2k that is dropped at export.
- **Evidence:**
  - Miki et al. (Science Robotics 2022) found the GRU beat the MLP in every case.
  - RAPTOR (arXiv 2509.11481) flies ten real quadrotors with a 2,084-parameter recurrent policy. A GRU-32 variant (about 16k) is therefore a cheap ablation (A8).
  - A GRU-96 would push the total past about 50k, so 64 is the width.
- **Out of scope for M3:** transformers and TCNs.
- **Auxiliary target.** The head predicts the normalised physical effect vector of §1.4, not z. This is my judgement. It is interpretable ("believed aileron effectiveness 40%") for Skyview, and it is a precursor of the L3 self-model. RMA-style z regression is the fallback.

### 1.7 Distillation in three stages, with gates
1. **Stage 1, PPO teacher.** About 2B policy steps per seed.
2. **Stage 2, DAgger in the same vmapped environment.**
   - The student flies. With probability β it executes the teacher's action, with β decaying from 1 to 0 over the first 20%.
   - Loss = ‖μ_student − μ_teacher‖² on pre-tanh means in normalised action space, plus 0.5·‖ê − e‖² on the auxiliary head.
   - Truncated BPTT over 256-step windows (5.1 s), starting from the stored carry with a 32-step burn-in. Build minibatches over environments, never over timesteps. About 0.6B steps.
   - Do not use offline behaviour cloning on logged teacher rollouts: it causes covariate shift. Giral et al. (arXiv 2411.02975) distilled offline. Lee et al. (Science Robotics 2020) and RMA (RSS 2021) use on-policy data instead.
3. **Gate C.** On G1-val, the student's survival must be within 2 pp of the teacher's and its tracking RMS within 15%. If so, stop here.
4. **Stage 3, only if Gate C fails.** This is the A-RMA step, which Kumar et al. (IROS 2022) call critical for reliable real-world performance.
   - Fine-tune the whole student with recurrent PPO. Initialise the critic from the teacher's critic.
   - lr 1e-4, 64-step segments with stored carry.
   - A behaviour-cloning term towards the teacher, with weight decaying from 1 to 0 over about 0.4B steps.
5. **If stage 3 also fails,** fall back to DreamWaQ-style single-stage asymmetric recurrent PPO with the auxiliary head and no teacher (Nahrendra et al., ICRA 2023). Its reported 95% vs 82% survival comparison is unverified [U].

Monitoring:
- Log the teacher-student action gap per fault kind and per time since onset.
- Test hidden-state drift on 10-minute flights; training episodes are only 60 s.

The recipe does not depend on RMA's internals [U]: an 8-D z, a 1-D CNN over about 50 steps, and an MSE latent loss.

### 1.8 Reward and termination
Use positive, bounded kernels, so that crashing never ends a stream of costs:

r = 0.35·k(e_φ, 5°) + 0.25·k(e_γ, 2°) + 0.30·k(e_V/V_ref, 0.05) + 0.10·k(a_y, 1 m/s²) − 0.05·‖a_t − a_{t−1}‖² − 0.5·𝟙[V_true < 1.2·V_s] − 0.5·𝟙[n_z outside limits]

where k(e, σ) = exp(−(e/σ)²).

- **Speed priority.** Airspeed outweighs flight path, so the policy protects speed when thrust is short, as TECS does.
- **Smoothness.** Add the CAPS temporal and spatial regulariser to the loss (Mysore et al., ICRA 2021), with λ about 0.01–0.1 [U on the values]. Olivares et al. (ICINCO 2024) found an action-variation penalty alone did not smooth PPO.
- **Privileged truth.** The reward may use it; it is computed inside the simulator.
- **Termination (no bootstrap):** ground or terrain contact; loss of control (|φ| > 90°, |θ| > 60°, as in `bench/run.py`); a non-finite state.
- **Truncation:** at 60 s with ±20% jitter. Bootstrap from V(final observation, final privileged observation), keeping both in `info` before the auto-reset (Pardo et al., ICML 2018).

### 1.9 PPO hyperparameters (starting values for the JAX trainer)

| | Teacher | Student fine-tune | L2 |
|---|---|---|---|
| Environments | 8,192 (sweep 4k / 16k) | 8,192 | 4,096 |
| Unroll | 32 steps (0.64 s) | 64 steps plus stored carry | 32 steps |
| Samples per update | 262k | 524k | 131k |
| Epochs × minibatches | 4 × 8 (about 32k samples each) | 4 × 8, minibatches over environments | 4 × 4 |
| Learning rate | 3e-4; adaptive KL with target 0.01, ×/÷1.5, bounded to [1e-5, 1e-3] | 1e-4 | 3e-4 adaptive |
| Clip; max gradient norm | 0.2; 1.0 | 0.2; 1.0 | 0.2; 1.0 |
| γ; GAE λ | 0.99 (sweep 0.995 for phugoid and energy); 0.95 | same | 0.99; 0.95 |
| Entropy | 0.001 (0–0.003) | 0 | 0.001 |
| Value | coefficient 0.5–1.0; separate critic; no value clipping | same | same |
| Normalisation | per-minibatch advantage normalisation; Welford observation normaliser clipped to ±5, saved with the checkpoint and frozen at evaluation | same | same |
| Policy distribution | tanh-squashed Gaussian; store the pre-tanh sample; log-std starts at ln 0.5, floored at 0.05 | same | same |

Sources:
- The large-batch structure follows Rudin et al. (CoRL 2021). Its exact values (4,096 environments, 24 steps, 5 epochs, ELU 512-256-128) are [U]. The rsl_rl defaults (adaptive schedule, desired_kl 0.01) are confirmed in its docs.
- Andrychowicz et al. (ICLR 2021) and "The 37 implementation details of PPO" (ICLR blog 2022) are the defaults checklist. Their specific values are [U].
- SAPG (ICML 2024) shows PPO saturates as the batch grows. Spend spare GPU capacity on vmapped seeds, not on bigger batches.

### 1.10 Randomisation ranges for this fleet

| Factor | Range |
|---|---|
| Airframe | Resample every episode from about 5,800 training airframes [D: about 8,550 accepted × 0.8, minus G2-x]. Embodiment-scaling results favour many distinct airframes (Ai et al., CoRL 2025). |
| Aero derivatives | ×(1 ± 0.2) each, per episode |
| Mass; CG; inertia; thrust | ±20%; ±5% of MAC (this moves C_mα); ±30%; ±20% |
| Servo time constant τ | 10–60 ms (the fleet's 10–40 ms, widened) |
| Total transport delay (sensor + compute + servo) | 10–100 ms, plus a random 0 or 1 control tick before the action applies; the buffer caps it at 150 ms. L2F got 0/10 real flights without delay modelling [U on the figure]; Bøhn calls actuation delay crucial. |
| Rate limit; deadband or backlash | ×(1 ± 0.4); 0–1° |
| IMU and airspeed | noise ×0.5–3; gyro and accelerometer bias; airspeed lag 0–0.2 s; integrator jitter, against simulator shortcutting |
| Wind | mean 0–12 m/s (at most 0.6·V_cruise); Dryden light to moderate; 1-cos gusts up to 5 m/s; shear; thermals. G7 (wind above 80% of cruise) stays held out. |
| Topology, for G3 (§2.7) | rudder removed 10%; rudder effectiveness ×0–0.5 15%; C_nβ ×0.3–1 15%; tailerons 5% |
| Allocator B̂ source (§2.5) | oracle 0.2 / nominal 0.3 / online estimate 0.5; ±30% column-scale error |
| Faults | 60% of episodes carry one in-scope fault, onset 2–40 s |

- **Fault scope for M3.** All actuator kinds (hard_over, floating, slow, reduced_throw, backlash, reversed, jam), plus propulsion, energy and airframe kinds.
- **Sensor faults** (pitot, GNSS, magnetometer) are out of L1 training and the headline; only noise, bias and latency are randomised. A teacher that sees true airspeed while the student's pitot is blocked opens a large imitation gap, and pitot faults also corrupt the q̄ scaling of the allocator. They belong to M4. State the exclusion in the report.
- **Do not widen much further.** Ferede et al. (ICRA 2025) found wider randomisation without adaptation makes policies robust but slower.

### 1.11 Curriculum
Compare against B1 at every rung:
1. One nominal airframe in calm air. It should learn in minutes.
2. The whole training fleet.
3. Add wind, turbulence, gusts and shear.
4. Add single faults, family by family.
5. Add the full delay and noise ranges.
6. Add allocator-source mixing and topology randomisation.

Then switch to automatic sampling: pick each (fault kind × class) cell with probability proportional to its current failure rate, with a 20% uniform floor.

- **References.** Half are random steps, ramps and doublets; half come from the guidance stack flying random missions.
- **Initial conditions.** 20% of resets start inside the region up to L2's engage boundary: |φ| ≤ 60°, θ ∈ [−30°, 40°], V from 0.9 to 1.5·V_cruise. L1 then never meets an out-of-distribution state at hand-back.

### 1.12 Trainer engineering
**Code structure**
- Write a lean trainer in the PureJaxRL style in `src/aeroadapt/learn/` (about 600–800 lines), rather than adopting Brax's trainer.
- Validate it on gymnax Pendulum against published curves first.
- Use the Anakin layout: a jitted `train_chunk` runs about 20 updates under `lax.scan`, and a thin Python loop handles logging, evaluation, checkpoints and pre-emption.

**Environment**
- `reset(key, pool_idx)` and `step(state, action, key)` are pure functions. The fleet is a pytree on the device.
- A reset is a gather from an offline pool of trimmed states (x64 trim, about 8 trims per airframe). Never run trim inside jit: `jnp.where` auto-reset evaluates the reset for every lane on every step.
- Do not reuse Brax's AutoResetWrapper. From memory it resets to a cached first state, which would collapse diversity [U].

**Tests and checks**
- GAE with termination and with truncation, both mid-rollout and at the rollout edge.
- Carry-reset alignment at episode boundaries.
- PRNG keys distinct across lanes.

**GPU numerics and operations**
- Set `jax_default_matmul_precision='highest'` on A100 and H100; TF32 by default is [U]. Then re-run N-version parity and the JSBSim cross-check on the GPU before training.
- Orbax async checkpoints about every 10 min. Each holds parameters, optimiser state, normaliser, RNG key data, curriculum state, config hash and git SHA.
- Keep the persistent compilation cache on a network volume. Flush a checkpoint on SIGTERM. Use a pinned image with jax 0.11.2, as in V15.

**Monitor:** approximate KL (about 0.01), clip fraction (0.05–0.3), value explained variance (above 0.8), terminations by cause, return per class, NaN resets, steps per second.

### 1.13 Compute estimate and GPU plan
**Today.** V15 on the laptop CPU, heterogeneous fleet with faults at batch 1024: 552k simulator steps/s and 225k closed-loop steps/s. The GPU figure (D9) is still pending.
**Benchmarks.** NeuralPlane reports about 5M aircraft-steps/s; FALCON-S reports 19.5M env-steps/s.
**Assumption.** Training throughput (environment, allocator, network and PPO together) of 0.5–2M policy steps/s on one rented GPU [D, pending D9].

| Item | Policy steps |
|---|---|
| Ladder and debugging | about 20B |
| L1 main, 5 seeds × 3B (teacher 2B + distillation 0.6B + fine-tune 0.4B) | 15B (30B with 10 seeds) |
| Trained ablations A3–A5, 3 seeds each | about 27B |
| LOCO, 2 folds × 3 seeds | about 18B |
| L2, 5 seeds × 0.5B | 2.5B |
| Sweeps (γ, entropy, lr, number of environments) | about 5B |
| **Total** | **about 90–105B, i.e. 15–55 GPU-hours** |

- **Prices** (secondary trackers, which moved more than 20% between August and October): A100 80GB about $1.4–1.6/h on RunPod; H100 about $1.5–2.3/h on marketplaces. That puts the core compute at about $40–140. Cap M3 at about $400 including idle time and retries.
- **Benchmark first** (about $5–15). Time the full training step at 8k, 32k and 128k environments on an RTX 4090, L40S, A100 and H100. Choose by dollars per billion policy steps, not by simulator-only steps/s.
- **Fill the GPU** with 5–10 vmapped seeds once steps/s flattens with the number of environments.
- **CPU fallback.** One seed overnight (about 4–8 h [D]). Enough for the ladder, not for seeds and ablations.
- Debugging time, not GPU dollars, is the binding constraint.

---

## 2. The adaptive allocation layer

### 2.0 Prerequisite: split surfaces (week 13)
- **Surface array.** A fixed-shape array, N_MAX = 8, with a per-airframe mask: [ail_L, ail_R, elev_L or ruddervator_L or elevon_L, elev_R or ruddervator_R or elevon_R, rudder, flap_L, flap_R, spare].
- **Aerodynamics.** Each surface gets a 6-vector of derivatives per radian, [C_L, C_D, C_Y, C_l, C_m, C_n], generated one side at a time in AeroSandbox; `references.py` already deflects the X8's elevons by ±5°. Superpose linearly, with an optional |δ| drag term.
- **Elevons.** Their shared travel then falls out of the per-surface limits.
- **Faults.** Widen the fault slot index to 0–7.
- **B1 unchanged.** `ActuatorCommand` carries the surface array, and B1 keeps its (a, e, r) commands through a fixed 8×3 mixer, which is realistic ArduPlane fixed mixing.
- **Acceptance.** The M1/M2 parity tests reproduce the old three-channel model exactly through the mixer, and the V-suite is re-run.
- **Option.** Dual elevator servos on about 30% of conventional airframes, for pitch redundancy.

### 2.1 Formulation and solver
**Formulation**
- One module, `src/aeroadapt/stack/allocation.py`: pure JAX, no simulator imports. L1 and B2 both use it, so the A/B tests the policy and not the allocator.
- Solve min ‖W_u(u − u_p)‖² + γ‖W_v(B̂u − ν)‖² subject to lo ≤ u ≤ hi, in normalised units (u ∈ [−1, 1] per surface, ν divided by ν_max).
- Non-incremental mode for L1: B̂ = q̄·B̃̂ and u_p = u_trim. Incremental mode for INDI: Δu around the filtered actuator position u_f, with demand ν − ω̇_f.
- **γ = 1e3 by default; sweep 1e2–1e4.** Report 5 suggested 1e4–1e6. I side with report 4: the normal equations square the condition number, which float32 cannot afford at 1e5–1e6.
- W_v = (roll 1, pitch 1, yaw 0.2), W_u = 1. Thrust is not allocated.

**Solver**
- Härkegård's primal active-set method always reaches the optimum of the box-constrained least-squares problem, at a cost similar to the cascaded generalised inverse (CDC 2002). Petersen & Bodson (TCST 2006) found WLS active-set the most efficient of the methods they compared.
- Unroll it as `lax.fori_loop` with K = 10 iterations and a converged mask. Do not use `while_loop`: under vmap every lane runs as long as the slowest one.
- Masked normal equations, solved by an 8×8 Cholesky with 1e-6·I added. Warm-start u and the working set from the previous control step, carried in the environment state.
- Every iterate is feasible, so a truncated solve is still admissible. Log the iteration count and alert whenever K is reached.
- Unit-test against `scipy.optimize.lsq_linear` on random B, random bounds and pinned surfaces.
- The typical warm-started count of 1–3 iterations is [U].

**Ablation and fallback.** The redistributed pseudo-inverse. It can stop short of demands that are attainable (Durham, JGCD 1993; Bodson, JGCD 2002). Skip FISTA and ADMM.

### 2.2 Bounds and fault handling (all through bounds and columns; the solver never changes)
- **Every step:** lo = max(u_min, u_prev − ρ·dt); hi = min(u_max, u_prev + ρ·dt).
- **Jammed, hard-over or floating, once detected:** zero column j and pin u_j at its last command.
  - No servo-position sensor exists (`core/messages.py`), so the jam position is not known onboard.
  - In non-incremental mode the jam's bias moment is a disturbance the policy cancels through its history. The teacher's privileged vector contains it.
  - In incremental (INDI) mode, Δu_j ∈ [0, 0], and the measured angular acceleration absorbs the bias automatically.
- **Reversed:** effectiveness zero until confirmed, then flip the sign. Never run a wrong-sign column: that is positive feedback, which diverges in under a second.
- **Reduced throw:** the estimator shrinks the column scale.
- **Slow servo:** shrink ρ only if the slowdown is known.
- **Backlash:** nothing in the allocator.

### 2.3 Online effectiveness estimation
**Regression**
- Normalise by dynamic pressure, B = q̄·B̃, so the estimate is nearly constant.
- Recursive least squares (RLS) on increments: y = Δω̇_f (3), φ = q̄·Δu_f (n). One shared covariance matrix; forgetting factor λ = 0.995, about 2 s of memory at 100 Hz.
- ω̇_f is the backward difference of the gyro passed through a second-order Butterworth low-pass at 12 Hz. u_f is the command passed through the nominal servo model (delay plus lag) and then through the identical filter. This is Smeur's synchronisation (JGCD 2016).

**Guards**
- Skip updates when ‖φ‖ is under a noise threshold.
- Cap the trace of P.
- Variable forgetting: λ_t = λ_min + (λ_max − λ_min)·exp(−e²/σ²), with λ_min = 0.95. This is a heuristic in the spirit of AA-INDI (Atmaca & de Visser, SciTech 2026), whose exact form is [U].
- Project each column scale to [0, 3]× nominal with the sign locked. Declare a surface dead below 0.25×.

**Detecting reversed and dead surfaces**
- Run a per-surface three-hypothesis test (nominal / zero / flipped) on residual log-likelihoods over a sliding 0.3 s window. Switch when the posterior exceeds 0.95, then reset that column's covariance.
- Its 0.2–0.5 s budget comes from the runaway time constant of about 1/(|r|·ω_f) [D].

**Excitation.** Antisymmetric mixing makes the left and right columns collinear, so RLS cannot separate them. Add null-space dither of 0.5–1° at 2–5 Hz, only when redundancy exists and the margin allows it. Report the estimator's convergence time.

### 2.4 What the policy sees from the allocator
- Per-axis attainable authority ν_k±. The cheap form is Σ_j max(B̂_kj·lo_j, B̂_kj·hi_j), ignoring coupling between axes: O(n) instead of six extra WLS solves.
- Achieved ν = B̂·u_prev, and the saturation residual.
- These give anti-windup and an embodiment descriptor computed onboard. Never feed the true B.

### 2.5 Trained together or separately: decision
- **The allocator is not learned.** It runs inside the RL environment from rung 6, and through distillation and fine-tuning, so the policy learns to coexist with a second adaptive loop.
- **Mix the source of B̂ per episode:** oracle (0.2), nominal autoconfig (0.3) or online RLS (0.5). Add ±30% column-scale error and transient wrong-zero columns at fault onset.
- **At evaluation the student runs with RLS only.**
- **Time scales.** Allocator tests about 0.3 s, RLS memory about 2 s; the policy's GRU adapts over 1–5 s.
- The teacher's privileged vector carries the true effectiveness, so the teacher does not depend on the estimator.

### 2.6 The raw-surface ablation and the ablation set (pre-registered)

| ID | Variant | Training cost |
|---|---|---|
| A1 | Same frozen student, oracle allocator | none (switch at evaluation) |
| A2 | Same frozen student, nominal B̂, estimator off | none |
| A3 | Raw-surface student: outputs 8 masked surfaces plus throttle; its observation adds the normalised nominal B̃ columns (24) and the mask (8), after Konstantaropoulos et al. (arXiv 2606.10857) | 3 seeds |
| A4 | History-free MLP, domain-randomised (Ferede et al.'s trade-off: robust but sluggish); also detects a GRU that ignores its memory | 3 seeds |
| A5 | L1 over INDI: the policy outputs total ω̇ demands into the incremental mode | 3 seeds; also the fallback primary |
| A6–A8 | No auxiliary head; no autoconfig descriptor; GRU-32 | if time |

- **Expected outcome.** The allocator should win on instantaneous redundancy: split-surface jams and the change of channel structure in G3. The policy should win on dynamic cross-axis use (rudder, then sideslip, then dihedral roll), which a static allocator cannot represent (report 4).
- **The result may be negative.** No fixed-wing head-to-head exists, and the multirotor evidence is mixed: Kaufmann et al. (ICRA 2022) favour abstraction, while RAPTOR and Zhang et al. (T-RO 2025) show direct-actuator policies generalising. Report it either way.

### 2.7 Preparing for G3 without leaking the class
- On training airframes, remove the rudder, weaken it, or reduce C_nβ (rates in §1.10).
- On airframes with split elevators, kill the ailerons in 5% of episodes so roll comes from differential elevator (tailerons).
- A B with no usable yaw column, and roll-pitch coupling through shared surfaces, are then interpolation; flying-wing aerodynamics stay unseen.
- In the report, separate what is new aerodynamics in G3 from what is new topology.

### 2.8 The Rascal (G4)
- Only `left-aileron-pos` enters C_lδa and C_nδa in Rascal.xml. Rewire these to 0.5·(left − right) in the derived model: a wiring change only, coefficients untouched, recorded in the docstring and the spec. The adapter must write each surface's position.
- The Rascal models no effect from symmetric aileron deflection, which flatters re-allocation. G4's primary endpoints therefore use nominal flight plus rudder or whole-channel faults. Split-aileron faults on the Rascal are exploratory, with that caveat.

---

## 3. L2 recovery

### 3.1 Scope
- **In scope:** upsets aerojax can represent. Bank beyond 60° up to inverted; nose-low spirals (θ down to −60°, V up to V_ne); nose-high, low-speed states; stall entry and exit. Each with or without an active actuator fault.
- **Out of scope: spins and deep stall.** The Beard & McLain post-stall blending is not validated for them.
- **Cross-check** every L2 win on the Rascal in JSBSim.
- **Simulation only.** Every fixed-wing RL recovery paper found is simulation-only (Grillo et al., SciTech 2024; Sayyed et al., arXiv 2601.06439; Demir et al., arXiv 2604.24355). Say so.

### 3.2 Upset distribution
- **50% synthetic grid:** φ 60–180°, θ −60° to +45°, body rates up to 120°/s, V from 0.8 to 1.6·V_cruise (below V_ne), starting 150–400 m above ground.
- **50% harvested:** the state at the instant L2's engage predicate fires in B1 and L1 rollouts on G1-train fault scenarios. B1's roughly 17% of non-survivals are the richest source.
- **Faults:** 50% of episodes carry an in-scope actuator fault.

### 3.3 Method
This is my judgement, against report 1's suggestion to distil L2 the same way as L1.
- **Single-stage asymmetric recurrent PPO:** a privileged critic, an auxiliary privileged-prediction head, the same GRU-64 architecture, and the same allocator with equal roll and pitch weights (1, 1, 0.1).
- **Why:** episodes are short (20 s at most), the allocator's test has usually flagged the fault before L2 engages, and a second teacher-student pipeline costs about a week.
- **L2 owns the throttle.** Delaying full throttle by 0.75 s raised altitude loss by up to 30% (Grillo et al.).
- **Fallback:** distil L2 as in §1.7.
- **Allocation weights.** Report 4 suggested pitch-unload priority and report 5 roll priority. Equal weights let the policy sequence the phases itself; a pitch-priority ablation runs if time allows.

### 3.4 Reward
- **Height loss:** a per-step penalty −k_h·max(0, −ḣ)·dt/h_ref.
- **Two-phase potential shaping** (Sayyed et al.):
  - Φ1 = −‖ω‖/ω_ref, until ‖ω‖ < 20°/s.
  - Then Φ2 = −(|φ|/30° + max(0, −γ)/5° + |V − V_target|/(0.2·V_cruise)).
- **Penalties:** n_z beyond the airframe's limit; V above V_ne; α past stall plus a margin (using privileged truth); a secondary upset, i.e. re-crossing the engage predicate after recovering.
- **Success:** terminal, with a +10 bonus, when the hand-back predicate holds for 1 s.
- **Ground contact:** terminal, −10.
- **Timeout at 20 s:** truncation, with bootstrap.

### 3.5 Curriculum
- Severity levels are bins over (|φ|, θ, ‖ω‖, fault severity).
- Advance when success exceeds 80%. Keep 20% replay from easier levels.
- Within a level, sample cells in proportion to their failure rate.
- Checkpoint the curriculum state.

### 3.6 Engagement and hand-back
All quantities come from onboard estimates.
- **Engage L2 when** any of: |φ| > 60°; θ < −30°; θ > 40°; ‖ω‖ > 90°/s; or V < 1.1·V_s with γ < 0.
- **Hand back when** |φ| ≤ 15°, |p|, |q|, |r| ≤ 15°/s, γ ≥ 0 and 1.2·V_s ≤ V ≤ 0.9·V_ne, held for 1 s.
- **Hysteresis and dwell.** The gap between the two predicates provides the hysteresis; L2 stays engaged for at least 1 s.
- **"Recovered"** in the metrics uses the same predicate, pre-registered.
- **L1's GRU runs throughout.** It sees the real observations, with L2's executed actions as its "previous action". To make that familiar, train L1 with 5% of episodes in which a scripted or L2 controller flies the first 2–5 s.

### 3.7 Baseline and metrics
- **Baseline:** scripted recovery on INDI + WLS. Unload to 0–0.5 g, roll wings level with roll priority, apply power, pull at a bounded n_z, then hand back. Tuned on G1 only.
- **Metrics:**
  - success;
  - height loss from engagement and from fault onset, as a recovery profile (§5.5);
  - time to recover;
  - peak n_z, α and V;
  - secondary-upset rate;
  - hand-back success, i.e. no re-engagement within 10 s.

### 3.8 Cut rule
If L2 does not beat scripted recovery on G1-val (paired) by Gate D, ship scripted recovery as M3's L2, as the plan allows, and move L2 to M6.

---

## 4. The INDI + allocation baseline (B2-control)

### 4.1 Structure
- φ_ref from L1 nav; B1's TECS, unchanged, supplies θ_ref and throttle.
- INDI rate and attitude loops produce ν; the shared allocator in incremental mode produces the surfaces.
- So B2 replaces only B1's PID attitude and rate loops. L1 replaces TECS as well. Both are complete controllers fed the same guidance references; state this asymmetry.

### 4.2 Control law
- ν = K_ω(ω_ref − ω_f) + ω̇_ref.
- ω_ref = K_att·(attitude_ref − attitude), with limits, from a second-order reference model with pseudo-control hedging so saturation does not wind up.
- Δu = WLS_incremental(ν − ω̇_f); u_cmd = u_f + Δu. Take the increment from u_f, not from the previous command (Smeur et al., JGCD 2016).
- Turn coordination: r_ref = (g/V)·sinφ·cosθ, plus a_y feedback.

### 4.3 Filters
- A second-order Butterworth at f_c = 12 Hz, with a group delay of about 19 ms [D]. Sweep {6, 8, 12, 16, 20} Hz. Paparazzi's default is 8 Hz at much higher loop rates (confirmed).
- The internal actuator model uses nominal τ̂ and d̂, with the same structure as `actuators.py`.
- Smeur's ω_n ≈ 50 rad/s, ζ ≈ 0.55 and 512 Hz are [U] and are not used.

### 4.4 Effectiveness
- Ĝ(q̄) = J⁻¹·q̄·S·diag(b, c̄, b)·C_δ per surface, including J_xz, from autoconfig.
- Compute q̄ from V̂ clamped to [0.9·V_s, 1.1·V_max]. An under-reading pitot shrinks Ĝ, which pushes the effectiveness ratio r = g/ĝ above 2 and makes the loop oscillate.
- **Stability.** For the synchronised scalar loop, r ∈ (0, 2) is sufficient for stability, and a reversed surface (r < 0) puts an unstable pole at +|r|·ω_f [D]. This agrees with Schenk & Ducard (arXiv 2607.16471), who found a sign error the most destabilising mismatch.
- Use report 5's worked Aerosonde matrix as a unit test [D].

### 4.5 Gains
- Starting rule [D]: K_ω ≈ 0.75/(τ̂ + d̂ + 0.225/f_c), for about 45–50° of phase margin.
  - Aerosonde (τ = 20 ms, d = 20 ms, f_c = 12 Hz): about 13 rad/s.
  - Slowest fleet servos (40 + 60 ms): about 6 rad/s.
- K_att ≈ K_ω/4.
- Tune only the global multipliers on top.

### 4.6 Variants
- **B2-fixed:** Ĝ from autoconfig.
- **B2-adaptive:** the same estimator, hypothesis test and projection as L1's allocator. Same code.
- Smeur's LMS update is skipped.

### 4.7 Information budget and tuning
- **Information.** Only nominal τ̂, d̂, derivatives and inertia from autoconfig; never the true per-scenario delay. B2 gets the same delay and noise randomisation as L1.
- **Tuning.** Tune 6–8 global hyperparameters on G1 with a batched JAX grid or CMA-ES: f_c, gain multipliers, γ, W_v yaw, λ and the dead-zone. Log the budget and keep it comparable to L1's sweep budget, then freeze before G2–G4. This follows Patterson et al. (JMLR 2024), and NASA's practice of tuning seven MRAC variants with one metrics-based method (Campbell et al., AIAA 2010).
- **Optional stretch in week 16: "B2-ID" on the Rascal.** Identify Ĝ and the servo delay from 2–3 s of doublets by RLS and cross-correlation (Blaha et al., arXiv 2406.11723). This is the fairest classical counterpart to zero-shot RL.

### 4.8 Pre-registered robustness sweeps
**Sweeps (applied to L1 too)**
- Ĝ scale {0.3, 0.5, 1, 2, 3}.
- Delay mismatch of 0 to +3 control steps.
- τ mismatch ±50%.
- Gyro noise ×{1, 2, 4}.

**Pre-registered hypotheses**
- *Strong:* loss of effectiveness, reduced throw, CG shift, wind.
- *Neutral:* motor cut (TECS's job).
- *Weak:* reversed surfaces without the detector, slow servos and rate limits (synchronisation mismatch), vibration, pitot faults.
- Constant gyro bias does not affect ω̇_f.

**Expectations**
- Do not pre-announce "INDI beats PID". Robust hybrid INDI and H∞-tuned PID came out virtually equivalent (Pollack et al., AST 2024).
- Time-scale separation is marginal on small UAVs: the Aerosonde roll mode is about 46 ms against about 40 ms of servo lag plus delay [D]. INDI may fall short of its manned-aircraft reputation. Steffensen et al.'s actuator-aware form (JGCD 2023) is the fix, but it is outside M3.
- Learned-over-INDI gains in the literature are tens of percent within a vehicle family: Neural-Fly was 35% better than INDI, and Ferede et al. (arXiv 2311.16948) found a 0.17 s gap in real flight. Large L1 gains are credible only where INDI is structurally weak (reversed surfaces, slow servos) or under structural novelty (G3).

**Diagnostics reported with every A/B:** share of time saturated; the realised r (from the privileged view); actuator total variation per second; oscillation flags.

---

## 5. Evaluation protocol

### 5.1 Pre-registration
**Where and when.** Commit `docs/validation/m3_prereg.md` with a signed tag by the end of week 13, before any training reaches test data. Amend it only with dated, tagged addenda, and only before the freeze.

**Contents**
- Hypotheses with endpoints and margins.
- Split definitions, with airframe-ID lists and hashes.
- Scenario seeds and counts.
- Controllers and their information budgets.
- The analysis plan and the checkpoint rule.
- The headline airframes and each one's hard-over, chosen by rule.
- The cut rules.

**Test-access ledger.** Record commit, weights hash, date and split for every G2, G2-x, G3 or G4 run. Before the freeze, only G1-val is used.

**Primary hypotheses** (Holm-adjusted across all five; everything else is exploratory):
- **P1.** G2 fault-scenario survival: L1 + allocation better than B1 + contingency.
- **P2.** G3 (X8, Zagi) fault survival: L1 better than B1, as case studies.
- **P3.** G4 Rascal in JSBSim: nominal tracking non-inferior to B1, with proposed margins of +1.0° RMS in roll and +0.5 m/s RMS in airspeed; and better survival under rudder and whole-channel faults.
- **P4.** L2 against scripted recovery on a fixed upset grid flown on G2 airframes: success (height loss is secondary).
- **P5.** G2 fault survival: L1 against B2-adaptive, two-sided. Losing this is a legitimate finding (H1 in docs/01).

### 5.2 Splits

| Split | Content | Airframes | Scenarios | Role |
|---|---|---|---|---|
| G1-val | hash validation split | about 850 | 1,000 | selection and tuning only |
| G2 | hash test split, minus extremes | about 850 [D: needs V20 run on the full 10k fleet; b0's pool gives about 85 test airframes] | 2,500, at most 3 per airframe | interpolation |
| G2-x | outside the 5th–95th percentile of mass or wing loading, removed from training before teacher training starts | about 15% of the fleet | 1,000 | extrapolation |
| G3 | X8 (published coefficients) and Zagi (Beard & McLain appendix coefficients, from memory [U]; not built yet) | 2 | 300 each | case studies, not a class claim |
| Aerosonde | Beard & McLain model in aerojax; independent provenance, never in training | 1 | 300 | one of the six |
| G4 | Rascal in JSBSim, plus a 2×2 with Rascal in aerojax (`configs/fleets/rascal`) to separate the simulator gap from the airframe gap | 1 | 200 × 120 s | case study |
| LOCO | glider and v_tail folds (the most distinct dynamics and topology), 3 seeds each | class | 1,000 per fold | evidence at class level |

- **Distance diagnostic.** For each test airframe, compute the k-nearest-neighbour distance to the training fleet in a non-dimensional descriptor space: C_lp, C_mα, C_nβ, control power, roll-mode τ, short-period ω_n, V_s, W/S. Plot survival and height loss against that distance with Wilson bands. G2 alone only shows interpolation, so lean on G2-x, G3 and G4.
- **Deviation from docs/03.** G3 was to include procedural flying wings. The generator has no flying-wing class (`generator/design.py`), and docs/05 notes the buildup method under-predicts C_nβ, C_nr and C_Yβ. Phrase G3 as two case studies in M3.

### 5.3 Controllers
- B1 + contingency.
- B2-fixed and B2-adaptive.
- L1 student, 5 seeds.
- L1 teacher, an upper bound that splits the gap into distillation loss and generalisation loss.
- Ablations A1–A5.
- Scripted recovery and L2.
- The trim-reachability oracle, which supplies labels.

### 5.4 Scenario design
- **Mix.** 30% nominal (wind and turbulence only); 70% single in-scope faults, stratified by fault family × class.
- **Flights.** 300 s in aerojax, with onsets between 10 and 200 s.
- **Common random numbers.** Key every noise and gust stream by (scenario id, simulated time), not by step or call count, so a 50 Hz and a 100 Hz controller meet identical gusts. Keep training seeds disjoint from scenario seeds.
- **Recoverable set.** Label each fault scenario as recoverable if the trim-reachability oracle (§5.8) succeeds. Report survival on all scenarios and on the recoverable set.

### 5.5 Metrics
Use one fixed set for every controller.
- **Outcome.** Survival (the existing definition), with loss of control in three tiers:
  - upset entered (thresholds in the style of AC 120-111, [U]);
  - LOC (the existing 90°/60° rule, plus a multi-envelope check in the style of Wilborn and Foster as a sensitivity analysis [U on the envelope details]);
  - unrecovered LOC.
- **Tracking.** A step and doublet battery. Per channel: RMS, normalised mean absolute error, rise time, 5% settling time, overshoot.
- **Height loss as a recovery profile.** The fraction of upsets recovered within h metres, for h from 0 to 400 m, with failures never reaching the curve. Never a mean over survivors.
- **Envelope exceedances:** V < 1.1·V_s, n_z, α.
- **Actuators.** RMS surface rate, % of time saturated, the CAPS smoothness metric Sm, motor energy.
- **Robustness.**
  - An empirical delay margin: sweep injected delay past the training range (beyond 100 ms, up to the buffer cap) until LOC or a limit cycle, in the style of AdaptiveQuadBench's when2fail (arXiv 2510.03471).
  - Sweeps of effectiveness gain and of noise.
  - Linearised gain and phase margins at trim on the six airframes: JAX linearisation of plant + GRU state + allocator, a credibility artefact in the style of Bøhn et al.
- **Engineering.** Estimator convergence time; inference latency.
- **Deferred.** Handling-quality equivalent-system fits move to M6. Never use unscaled MIL-F-8785C limits (Foster, BYU 2004).

### 5.6 Statistics
**Two separate uncertainty statements**
- (a) "This frozen policy": over scenarios, clustered by airframe.
- (b) "This training recipe": over seeds.

**Paired survival**
- Primary: a cluster bootstrap CI of the paired difference (10k resamples; resample airframes, then scenarios within each), plus a sign-flip permutation p-value at airframe level.
- Secondary: exact McNemar on the discordant pairs, and Newcombe's paired CI (Statistics in Medicine 1998). Worked example: 60 vs 30 discordant pairs gives χ² = 10, p ≈ 0.002.
- Clustered data analysed as independent can exceed a 45% false-positive rate (Saravanan et al., arXiv 2007.07797). Report the design effect.

**Single rates.** Wilson intervals: about ±2.2 pp at n = 1,000 and ±1.0 pp at n = 5,000, for p = 0.85. Use the cluster bootstrap whenever the data are clustered.

**Continuous metrics.** Paired differences with failures ranked worst (a crash counts as infinite height loss). Wilcoxon signed-rank with the Hodges-Lehmann estimate, plus a cluster bootstrap CI.

**Recipe claims**
- rliable: IQM over per-stratum survival rates per seed (strata are fault family × class), stratified-bootstrap CIs, performance profiles and probability of improvement (Agarwal et al., NeurIPS 2021).
- Plus Welch's t or an exact permutation test over seeds; the bootstrap under-covers at small N (Colas et al., arXiv 1806.08295).
- Never apply IQM to individual binary scenarios.

**Statement in aerospace form.** For example, "unrecovered LOC ≤ 1% at 95% confidence" needs 0 failures in 299 scenarios (rule of three; Hanson & Beard's framing, NASA/TP-2010-216447).

**Stretch: equal-budget falsification.** Give L1, B2 and B1 the same CMA-ES search over onset, severity, wind, gust phase and initial state, and report the worst cases found (GARTEUR clearance, LNCIS 283; Corso et al., JAIR 2021).

### 5.7 Seeds and checkpoints
- **Seed counts.** 5 seeds for every trained method. 10 for the headline L1 recipe if a 3-seed pilot shows a seed-to-seed SD of G1-val survival above 2 pp. Power at α = 0.05 and 0.8: Cohen's d = 1 needs about 17 seeds per arm, d = 1.5 about 8, d = 2 about 6.
- **Checkpoints.** Use the final checkpoint of every seed. The headline policy is the median seed by G1-val, declared in advance.
- **Disclosure.** Report all seeds and the headline seed's rank. Publish the weights hashes before any test evaluation.

### 5.8 When the headline counts as met
The proposal, to be pre-registered.

**The six airframes, chosen by rule**
- Two from G2 by hash draw: one interpolated, one from G2-x.
- The Aerosonde.
- The X8.
- The Zagi if it is built; otherwise a third G2 airframe by hash.
- The Rascal in JSBSim.

**The hard-over per airframe, chosen by rule.** Take the first option in this order for which the trim-reachability oracle finds wings-level trim with at least 10% residual authority in both roll directions:
1. single-aileron hard-over at full travel;
2. rudder hard-over;
3. a partial elevon or ruddervator jam at half travel.

Full elevon and ruddervator hard-overs are probably unrecoverable on tailless and V-tail airframes [D], so the X8 will probably get a partial jam. The oracle reuses the x64 trim code with the faulted surface pinned. The headline then reads: "...including recovery from a hard-over on every airframe where one is physically recoverable".

**Criteria per airframe**
- **(i) Nominal.** 50 of 50 flights without LOC in light to moderate turbulence, and tracking inside the non-inferiority margin of B1.
- **(ii) Hard-over.** 50 variants (onset at 10–30 s in cruise, wind 0–8 m/s): at least 45 of 50 survive, which gives a Wilson lower bound of about 79%. Median height loss at most 30 m (proposed threshold). Show B1 and B2 alongside.

**Verdict.** Met = 6 of 6 airframes on both criteria, and P1 significant after Holm. Anything less is "partially met", with the failing airframes named.

### 5.9 Fidelity statements for the report
- Every result is simulation-to-simulation (aerojax to JSBSim).
- B1 is ArduPlane-equivalent, not ArduPlane, until the M6 SITL subset.
- Linear superposition of control derivatives overstates authority at large deflections. aerojax, the allocator and the JSBSim Rascal all share this, so their agreement does not validate hard-over recovery.
- The X8 flies on published coefficients. No spins are modelled.

**Novelty claim.** "The first cross-airframe and cross-class zero-shot fixed-wing RL controller of which the author is aware", and "RL through a fixed-wing WLS allocator with online effectiveness estimation". Both are conditional on reading Chowdhury & Keshmiri (IEEE TAES 2024), which may be cross-airframe fixed-wing RL work, and on a fresh 2025–26 search. The search budget ran out in every report.

---

## 6. Biggest risks, mitigations and fallbacks

| # | Risk | Early warning | Mitigation | Fallback |
|---|---|---|---|---|
| 1 | The split-surface refactor overruns or breaks parity | parity still failing mid-week 13 | mixer-based backward compatibility; write the tests first | keep 3 channels; headline hard-over becomes a rudder hard-over; downgrade the allocation claim to "effectiveness-aware inversion with fault-aware bounds", stated plainly |
| 2 | The GPU path is slow or broken (compile time, TF32, memory) | D9 numbers; GPU parity fails | precision 'highest'; persistent compile cache; benchmark the full training step | CPU overnight for single seeds; 3 seeds; ablations cut to A1, A2, A5 |
| 3 | The teacher collapses to an "average airframe" (sluggish on agile airframes, unstable on slow ones) | per-class returns diverge; rung 2 stalls | normalised actions; log-scaled privileged inputs; bounded per-term rewards; ladder | 5 per-class teachers distilled into one student (RAPTOR-lite), or A5 as primary |
| 4 | Imitation gap on faults that stay unobservable until excited | action gap by fault kind; student far below teacher on reversed surfaces or jams | detectability-delayed reveal; auxiliary head; stage-3 fine-tune | DreamWaQ-style single stage; report those faults separately |
| 5 | The two adaptive loops fight | oscillation at onset only with the estimator on; A1 far better than the student | mixing the source of B̂; separated time scales; authority fed to the policy | estimator restricted to the dead/reversed tests, with no continuous rescaling |
| 6 | A reversed surface diverges before it is detected | reversed-surface survival near 0 for both L1 and B2 | zero column until confirmed; 0.3 s test; L2 engages | known limitation, excluded from the headline |
| 7 | Simulator shortcutting; G4 collapses | 2×2 shows Rascal good in aerojax, bad in JSBSim | nuisance randomisation; weekly JSBSim smoke test | A5 for G4; report the simulator gap honestly |
| 8 | G3 (X8) fails; it cannot be checked before the freeze | only proxies: rudderless training airframes | topology randomisation | report the negative result; procedural flying wings and full LOCO in M6 |
| 9 | Over-wide randomisation makes the policy sluggish and it loses to tuned PID on tracking (Ferede et al.) | nominal RMS on G1-val above the margin | moderate ranges plus history; honest non-inferiority margins | narrow the ranges; claim fault survival, not tracking |
| 10 | Recurrent PPO, GAE or carry bugs: training looks fine but memory goes unused | student no better than A4; 10-min drift | unit tests; A4 comparison | TCN or frame stacking |
| 11 | Chattering surfaces | Sm and surface-rate RMS far above B1 | CAPS plus the rate penalty | fixed low-pass on ν, trained with it in the loop |
| 12 | Reward hacking: diving for authority; exploiting post-stall aero (L2 most exposed) | height loss and α excursions | envelope penalties; height in the reward; JSBSim check | restrict L2's envelope |
| 13 | Test-set reuse or a curated headline | none (prevent it by design) | pre-registration, ledger, hash draws, median seed | none needed |
| 14 | The hard-over is physically unrecoverable | the oracle says so | reachability oracle before announcing anything | wording per airframe |
| 15 | Pitot faults corrupt the q̄ scaling | not applicable in M3 | sensor faults excluded; q̄ clamp | M4 synthetic airspeed; compound cases in G6 |
| 16 | Schedule (about 150 h of work for about 120 h) | Gate B missed | gates and the cut list | L2 cut, or a one-week slip, decided at Gate B and not drifted into |

---

## 7. Week-by-week plan, weeks 12–16

**Week 12 (spent on D6 and D4).**
- If any slack remains: run the D9 GPU notebook with a dummy policy in the loop (about $5–15), set the precision flag and check GPU parity.
- Queue the full-fleet V20 acceptance run as a CPU job once the current CPU job finishes.
- Draft the pre-registration skeleton.
- Confirm the split-surface scope.

**Week 13: surfaces, allocator, pre-registration (about 30 h, overfull).**
- Split surfaces (N_MAX = 8), per-surface faults, one-sided derivatives, the B1 mixer, parity and the V-suite (14 h).
- Rascal rewiring and per-surface adapter (2 h).
- `allocation.py`: active-set WLS, bounds, tests against `lsq_linear` (6 h).
- Trim-reachability oracle; hard-overs per airframe chosen by rule (3 h).
- Freeze the splits (remove G2-x from training, LOCO folds, six-airframe draw); commit and tag pre-registration v1 (3 h).
- Trainer skeleton validated on Pendulum (overflows into week 14).
- **Gate A (Friday):** parity passes. Otherwise take fallback 1.

**Week 14: environment, estimator, teacher, INDI.**
- Estimator, hypothesis test and dither (5 h).
- Environment: reset pool, auto-reset, truncation bootstrap, randomisation, CRN keys, GAE and carry tests (6 h).
- Teacher ladder rungs 1–4 on GPU (8 h).
- B2-fixed and B2-adaptive on the shared allocator, tuned on G1 (8 h).
- Additions to `bench/stats.py`: cluster bootstrap, sign-flip permutation, McNemar, Newcombe, Hodges-Lehmann, rliable wrappers (3 h).
- **Gate B (Friday):** the teacher on G1-val beats B1's fault survival by at least 3 pp and is non-inferior on nominal tracking. If not, make A5 the primary and cut L2 now.

**Week 15: student, ablations, L2.**
- Rungs 5–6; teacher seeds launched on Monday (3 h).
- DAgger distillation with the auxiliary head; Gate C on Wednesday; stage 3 if needed (10 h).
- Launch A3–A5 and the two LOCO folds (3 h).
- L2 v0: upset sampler, reward, curriculum, engage and hand-back, L1 episodes with foreign actions (10 h).
- JSBSim G4 smoke test and 10-minute drift test (4 h).
- **Gate D (Friday):** L2 against scripted recovery; cut if it does not win.
- **Freeze all weights on Friday** and record the hashes in the ledger.

**Week 16: evaluation, report, demo.**
- Campaign on G2, G2-x, G3, G4 (with the 2×2) and the Aerosonde; sweeps, delay margins and linearised margins (8 h, mostly compute).
- Analysis following the pre-registered tables, plus a failure gallery (10 h).
- Skyview: the six-airframe montage and an allocation view showing B̂ columns with confidence, bounds and jam markers (8 h).
- Blog post (4 h).
- **Done when** the A/B report with CIs on G2–G4 exists and one command reproduces every number.

**Deferred or stretch:** falsification search; B2-ID; handling-quality fits; procedural flying wings and full LOCO rotation; A6–A8; the L1-adaptive-control baseline.

---

## Claims to verify before quoting them in docs

1. **RMA:** 8-D z, a 1-D CNN over about 50 steps, MSE latent loss, about 1.2B samples.
2. **Lee et al. 2020:** the student TCN, the combined loss, DAgger use, the curriculum.
3. **RAPTOR:** GRU width about 16; its inputs; non-adaptive per-platform teachers; DAgger or behaviour cloning; the sampling ranges.
4. **Learning to Fly in Seconds:** whether the algorithm is TD3; the 0/10, 5/10 and 6/10 ablation figures (secondary summary).
5. **Secondary-summary numbers:** Marquis & Farhood's path errors; DreamWaQ's 95.23% vs 82.37%; Sayyed's 6,000 episodes.
6. **PPO defaults:** Rudin et al.'s hyperparameters; Andrychowicz et al.'s and the "37 details" specific values; CAPS λ values.
7. **Brax and JAX behaviour:** AutoResetWrapper's cached first state; Brax batch_size semantics; JAX TF32 default precision.
8. **Allocation and INDI details:** Härkegård's warm-start iteration counts; Paparazzi's `wls_alloc` defaults; AA-INDI's forgetting-factor form; Smeur 2016's filter values.
9. **Own derivations [D]:** the Rascal 24% and 70°/s figures; the Aerosonde Ĝ matrix and the 1.4% and 33° figures; the 46 ms roll mode; r ∈ (0, 2); the gain rule; elevon and ruddervator full hard-overs being unrecoverable; all compute estimates; about 85 test airframes in b0.
10. **Possibly unreliable paper:** arXiv 2610.37470 (a real-flight PPO interception on the X-UAV Talon, and the 30%-randomisation degradation). The search results conflict.
11. **Read before any novelty claim:** Chowdhury & Keshmiri, IEEE TAES 2024.
12. **Zagi coefficients** in Beard & McLain's appendix.
13. **Standards and statistics:** Wilborn-Foster's envelopes and 3-of-5 rule; AC 120-111's numeric thresholds; MIL-F-8785C Level boundaries; Colas et al.'s "at least 15 seeds" (from slides).
14. **Cloud GPU prices.**

## References (title, venue and year, URL)

**Teacher-student and adaptation**
- Kumar et al., RMA: Rapid Motor Adaptation for Legged Robots, RSS 2021, https://arxiv.org/abs/2107.04034
- Kumar et al., Adapting Rapid Motor Adaptation for Bipedal Robots, IROS 2022, https://arxiv.org/abs/2205.15299
- Lee et al., Learning quadrupedal locomotion over challenging terrain, Science Robotics 2020, https://arxiv.org/abs/2010.11251
- Miki et al., Learning robust perceptive locomotion for quadrupedal robots in the wild, Science Robotics 2022, https://arxiv.org/abs/2201.08117
- Eschmann, Albani, Loianno, RAPTOR: A Foundation Policy for Quadrotor Control, arXiv 2509.11481 (Science Robotics), https://arxiv.org/abs/2509.11481
- Eschmann et al., Learning to Fly in Seconds, IEEE RA-L 2024, https://arxiv.org/abs/2311.13081
- Zhang et al., A Learning-based Quadcopter Controller with Extreme Adaptation, IEEE T-RO 2025, https://arxiv.org/abs/2409.12949
- Zhang et al., Learning a Single Near-hover Position Controller for Vastly Different Quadcopters, ICRA 2023, https://arxiv.org/abs/2209.09232
- Ferede et al., One Net to Rule Them All, ICRA 2025, https://arxiv.org/abs/2504.21586
- Dynamics-Invariant Quadrotor Control using Scale-Aware Deep RL, arXiv 2025, https://arxiv.org/abs/2503.09622
- Ai et al., Towards Embodiment Scaling Laws in Robot Locomotion, CoRL 2025, https://arxiv.org/abs/2505.05753
- Bohlinger et al., One Policy to Run Them All, CoRL 2024, https://arxiv.org/abs/2409.06366
- Weihs et al., ADVISOR, NeurIPS 2021, https://arxiv.org/abs/2007.12173
- Shenfeld et al., TGRL, ICML 2023, https://arxiv.org/abs/2307.03186
- Messikommer et al., Student-Informed Teacher Training, ICLR 2025, https://arxiv.org/abs/2412.09149
- Nahrendra et al., DreamWaQ, ICRA 2023, https://arxiv.org/abs/2301.10602

**Fixed-wing RL and recovery**
- Bøhn et al., PPO attitude control of fixed-wing UAVs, ICUAS 2019, https://arxiv.org/abs/1911.05478
- Bøhn et al., Data-Efficient Deep RL for Attitude Control of Fixed-Wing UAVs: Field Experiments, IEEE TNNLS 2023, https://arxiv.org/abs/2111.04153
- Giral et al., Transformer-Based Fault-Tolerant Control for Fixed-Wing UAVs, arXiv 2024, https://arxiv.org/abs/2411.02975
- Marquis & Farhood, Hypernetwork-Conditioned RL for Fixed-Wing Aircraft under Actuator Failures, arXiv 2026, https://arxiv.org/abs/2604.03392
- Marquis et al., Adversarial RL for Robust Control of Fixed-Wing Aircraft under Model Uncertainty, arXiv 2025, https://arxiv.org/abs/2510.16650
- Wada et al., Sim-to-Real Transfer for Fixed-Wing UAV: Pitch Control, IEEE RA-L 2022, https://ieeexplore.ieee.org/document/9882333/
- Olivares et al., Model-Free versus Model-Based RL for Fixed-Wing UAV Attitude Control, ICINCO 2024, https://arxiv.org/abs/2409.17896
- Mysore et al., Regularizing Action Policies for Smooth Control with RL (CAPS), ICRA 2021, https://arxiv.org/abs/2012.06644
- Dally & van Kampen, Soft Actor-Critic Deep RL for Fault Tolerant Flight Control, AIAA SciTech 2022, https://arxiv.org/abs/2202.09262
- Grillo et al., Optimal Stall Recovery via Deep RL, AIAA SciTech 2024, https://gabo-tor.github.io/publication/grillo-optimal-2024
- Sayyed et al., Deep RL Control Design for Aircraft Recovery from Loss-of-Control, arXiv 2026, https://arxiv.org/abs/2601.06439
- Demir et al., An Aircraft Upset Recovery System with RL, arXiv 2026, https://arxiv.org/abs/2604.24355
- Chowdhury & Keshmiri, Interchangeable RL Flight Controller for Fixed-Wing UASs, IEEE TAES 2024, https://www.researchgate.net/publication/377280809
- Xue et al., NeuralPlane, NeurIPS 2024 Datasets & Benchmarks, https://proceedings.neurips.cc/paper_files/paper/2024/hash/af97b61e9fef8f55e32a2602af364d8c-Abstract.html
- FALCON-S, arXiv 2026, https://arxiv.org/abs/2609.06046

**PPO in JAX**
- PureJaxRL, https://github.com/luchris429/purejaxrl
- Rutherford et al., JaxMARL, NeurIPS 2024 D&B, https://arxiv.org/abs/2311.10090
- Matthews et al., Craftax, ICML 2024, https://arxiv.org/abs/2402.16801
- Rudin et al., Learning to Walk in Minutes Using Massively Parallel Deep RL, CoRL 2021, https://arxiv.org/abs/2109.11978
- Singla et al., SAPG, ICML 2024, https://arxiv.org/abs/2407.20230
- Hilton et al., Batch size-invariance for policy optimization, NeurIPS 2022, https://arxiv.org/abs/2110.00641
- Andrychowicz et al., What Matters for On-Policy Deep Actor-Critic Methods?, ICLR 2021, https://arxiv.org/abs/2006.05990
- Huang et al., The 37 Implementation Details of PPO, ICLR Blog Track 2022, https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details
- Pardo et al., Time Limits in RL, ICML 2018, https://arxiv.org/abs/1712.00378
- Lu et al., Structured State Space Models for In-Context RL (S5), NeurIPS 2023, https://arxiv.org/abs/2303.03982
- Zakka et al., MuJoCo Playground, RSS 2025, https://arxiv.org/abs/2502.08844
- JAX GPU performance tips, https://docs.jax.dev/en/latest/gpu_performance_tips.html

**Allocation and INDI**
- Johansen & Fossen, Control allocation: a survey, Automatica 2013
- Härkegård, Efficient active set algorithms for constrained least squares in aircraft control allocation, IEEE CDC 2002
- Petersen & Bodson, Constrained quadratic programming techniques for control allocation, IEEE TCST 2006, https://pubweb.eng.utah.edu/~bodson/pdf/Constrained%20Quadratic%20Programming%20Techniques%20for%20Control%20Allocation.pdf
- Bodson, Evaluation of optimization methods for control allocation, JGCD 2002
- Durham, Constrained control allocation, JGCD 1993
- Härkegård, Dynamic control allocation using constrained quadratic programming, JGCD 2004
- Smeur, Chu, de Croon, Adaptive INDI for attitude control of micro air vehicles, JGCD 2016
- Paparazzi stabilization_indi docs, https://paparazzi-uav.readthedocs.io/en/latest/modules/stabilization_indi.html
- Atmaca & de Visser, AA-INDI, AIAA SciTech 2026
- Kaufmann, Bauersfeld, Scaramuzza, A benchmark comparison of learned control policies for agile quadrotor flight, ICRA 2022, https://arxiv.org/abs/2202.10796
- Konstantaropoulos et al., Embodiment-conditioned generalist control for multirotor aerial robots, arXiv 2026, https://arxiv.org/abs/2606.10857
- Steffensen et al., NDI with Actuator Dynamics: An Incremental Control Perspective, JGCD 2023, https://arxiv.org/abs/2201.09805
- van 't Veld et al., Stability and robustness of INDI, AIAA 2018
- Wang et al., Stability Analysis for INDI Control, JGCD 2019
- Schenk & Ducard, Linear stability analysis of an INDI pitch-rate controller under model mismatch, arXiv 2026, https://arxiv.org/abs/2607.16471
- Pollack, Theodoulis, van Kampen, Commonalities between robust hybrid INDI and PID, AST 2024
- Ferede et al., End-to-end RL vs INDI inner loop, arXiv 2023, https://arxiv.org/abs/2311.16948
- O'Connell et al., Neural-Fly, Science Robotics 2022, https://arxiv.org/abs/2205.06908
- Blaha et al., Control of Unknown Quadrotors from a Single Throw, arXiv 2024, https://arxiv.org/abs/2406.11723

**Evaluation**
- Agarwal et al., Deep RL at the Edge of the Statistical Precipice, NeurIPS 2021, https://arxiv.org/abs/2108.13264
- Colas et al., How Many Random Seeds?, arXiv 2018, https://arxiv.org/abs/1806.08295
- Patterson et al., Empirical Design in RL, JMLR 2024, https://jmlr.org/papers/v25/23-0183.html
- Saravanan et al., The hierarchical bootstrap for multi-level data, arXiv 2020, https://arxiv.org/abs/2007.07797
- Newcombe, Improved CIs for the difference between binomial proportions based on paired data, Statistics in Medicine 1998
- Hanson & Beard, Monte Carlo simulation for launch vehicle requirements verification, NASA/TP-2010-216447, https://ntrs.nasa.gov/citations/20100038453
- Fielding et al. (eds), Advanced Techniques for Clearance of Flight Control Laws, Springer LNCIS 283, 2002
- Corso et al., A Survey of Algorithms for Black-Box Safety Validation, JAIR 2021, https://jair.org/index.php/jair/article/view/12716
- Dorobantu, Seiler, Balas, Time-Delay Margin Analysis for Adaptive Control, JGCD 2012
- Zhang et al., A Simulation Evaluation Suite for Robust Adaptive Quadcopter Control (AdaptiveQuadBench), arXiv 2025, https://arxiv.org/abs/2510.03471
- FAA AC 120-111, https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-111.pdf
- Foster, Dynamic Stability and Handling Qualities of Small UAVs, BYU MSc 2004, https://scholarsarchive.byu.edu/etd/219

**Repo files read (no simulations run):**
- docs/03_uav_architecture_and_six_month_roadmap.md
- docs/validation/m2_report.md
- src/aeroadapt/sim/aerojax/actuators.py
- src/aeroadapt/faultlab/schema.py
- src/aeroadapt/generator/design.py
- src/aeroadapt/generator/references.py
- configs/fleets/rascal/