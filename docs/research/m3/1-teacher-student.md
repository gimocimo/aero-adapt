# Report 1: Teacher–student and privileged-information policies

*Part of the M3 literature review, 6 Nov 2026 ([README](README.md)). The brief and the findings are as the research agent returned them; local paths are made repository-relative. Where this report and the spec differ, [docs/09](../../09_m3_technical_spec.md) holds.*

## Brief

Teacher-student / privileged-information policies for zero-shot adaptation across embodiments and dynamics (RMA, Lee 2020, RAPTOR, Learning to Fly in Seconds, Swift, fixed-wing equivalents) and what they imply for M3 L1/L2 in aero-adapt. Note: the shared web-search budget ran out partway through. Claims marked confirmed=false come from my own knowledge of the papers and were not checked this session. Repo context comes from docs/03 §4.2, §5 and §7.2, plus src/aeroadapt/airframes/params.py, faultlab/schema.py and vehicle/protocol.py. Nothing was run.

## Key findings (18 of 21 confirmed in the session)

### 1.1 · confirmed

RMA splits adaptation into (1) a base policy plus an environment-factor encoder trained jointly with PPO on privileged parameters and (2) an adaptation module trained by supervised regression of the latent from the state-action history, on on-policy data gathered while the robot is driven by the estimated latent (a DAgger-like setup). The two run at different rates.

- *Evidence:* Unitree A1 in RaiSim, proprioception only. The privileged vector e_t is 17-D: payload mass and position (3), motor strength (12), friction (1), terrain height (1). Adaptation module at 10 Hz, base policy at 100 Hz. A secondary summary gives training/test ranges as: friction [0.05,4.5]/[0.04,6.0], motor strength [0.90,1.10]/[0.88,1.22], payload 0-6/0-7 kg. The test ranges are deliberately wider than the training ranges.
- *Source:* Kumar, Fu, Pathak, Malik, 'RMA: Rapid Motor Adaptation for Legged Robots', RSS 2021, https://arxiv.org/abs/2107.04034 ; supplementary https://ashish-kmr.github.io/rma-legged-robots/rma-locomotion-supplementary.pdf

### 1.2 · **not confirmed** (memory or secondary)

RMA internals from memory: an 8-D extrinsics latent z, an adaptation module that is a 1-D CNN over the last ~50 state-action pairs (~0.5 s at 100 Hz), a 3-layer MLP base policy, and roughly 1.2B PPO samples. The latent loss is MSE(z_hat, z); no action-imitation loss is used.

- *Evidence:* Not visible in any search snippet. Check Section 3 of the arXiv PDF before quoting these numbers.
- *Source:* RMA, RSS 2021, https://arxiv.org/pdf/2107.04034

### 1.3 · confirmed

Regressing the latent alone leaves an estimation error that the base policy never saw in training. A-RMA adds a third phase: fine-tune the base policy with PPO while the adaptation module is frozen. The authors call this critical for reliable real-world performance.

- *Evidence:* Cassie biped, zero-shot real deployment. Phases 1-2 are as in RMA; phase 3 is PPO fine-tuning of the base policy on top of the estimated extrinsics.
- *Source:* Kumar, Li, Zeng, Pathak, Sreenath, Malik, 'Adapting Rapid Motor Adaptation for Bipedal Robots', IROS 2022, https://arxiv.org/abs/2205.15299

### 1.4 · confirmed

Lee et al. made privileged learning the enabling step for blind rough-terrain locomotion. Training the proprioceptive policy directly with RL failed because the supervisory signal was too sparse. A privileged teacher, distilled into a history-based student, worked and transferred zero-shot to two generations of ANYmal.

- *Evidence:* Lee, Hwangbo, Wellhausen, Koltun, Hutter. Proprioception only at deployment; zero-shot sim-to-real into natural environments.
- *Source:* 'Learning quadrupedal locomotion over challenging terrain', Science Robotics 5(47), 2020, DOI 10.1126/scirobotics.abc5986, https://arxiv.org/abs/2010.11251

### 1.5 · **not confirmed** (memory or secondary)

Lee et al. details from memory: the teacher MLP sees terrain heights around the feet, contact states and forces, friction and external disturbance. The student is a TCN over a long proprioceptive history. It is trained DAgger-style on its own rollouts, with the teacher labelling, and the loss combines action imitation with regression of the teacher's latent. An adaptive terrain curriculum (particle-filter style) keeps terrains at medium difficulty.

- *Evidence:* Not visible in snippets. Check the methods section.
- *Source:* Lee et al., Science Robotics 2020, https://arxiv.org/abs/2010.11251

### 1.6 · confirmed

In the perceptive follow-up, a recurrent (GRU) belief encoder beat an MLP encoder in every case tested. The student's belief state is trained so that privileged ground truth can be reconstructed from it.

- *Evidence:* Supplementary comparison reported as 'GRU worked better than MLP in all cases'. A teacher trained with RL on privileged data is distilled into a student with a belief encoder. The exact loss terms and the DAgger usage were not visible.
- *Source:* Miki, Lee, Hwangbo, Wellhausen, Koltun, Hutter, 'Learning robust perceptive locomotion for quadrupedal robots in the wild', Science Robotics 2022, https://arxiv.org/abs/2201.08117

### 1.7 · confirmed

RAPTOR: one tiny recurrent policy (3 layers, 2,084 parameters, GRU-based) flies ten real quadrotors it never saw in training, from 32 g to 2.4 kg, adapting in context within milliseconds. It was obtained by 'meta-imitation learning'. 1,000 quadrotors were sampled from a generative distribution, an RL teacher was trained for each, and all 1,000 teachers were distilled into one student.

- *Evidence:* The ten platforms differ in motors, frames, propellers and flight controllers. The paper also reports coping with wind, being poked and propeller swaps. One secondary review says the paper has no DAgger-style covariate-shift analysis; I could not verify this.
- *Source:* Eschmann, Albani, Loianno, 'RAPTOR: A Foundation Policy for Quadrotor Control', arXiv 2509.11481 (v2 Apr 2026); Science Robotics, DOI 10.1126/scirobotics.aec1481, https://arxiv.org/abs/2509.11481

### 1.8 · **not confirmed** (memory or secondary)

RAPTOR details from memory, unverified. The GRU hidden width is about 16. The input is the full state plus the previous action and the output is motor commands. The teachers are non-adaptive per-quadrotor specialists, not latent-conditioned policies, so there is no explicit latent bottleneck. The method exists partly because recurrent RL directly over the whole distribution is hard to train.

- *Evidence:* A 22-in→16 dense, GRU(16), 16→4 out stack gives about 2.07k parameters, consistent with 2,084. The sampling ranges (thrust-to-weight, inertia, motor time constants) were not visible.
- *Source:* RAPTOR, https://arxiv.org/pdf/2509.11481

### 1.9 · confirmed

Learning to Fly in Seconds: an asymmetric actor-critic (privileged critic), an action history in the actor's input, motor-delay modelling and a curriculum together give direct-RPM Crazyflie control after 18 s of training on a laptop. The ablations show delay modelling and action history are not optional.

- *Evidence:* Reported ablation (via a secondary summary): without motor-delay modelling 0/10 real flights succeed; without the action history 5/10 instead of 10/10; without the asymmetric actor-critic 6/10 instead of 10/10. The simulator runs about 5 months of flight per second on a laptop GPU. From memory, the algorithm is off-policy (TD3) in RLtools; this is unconfirmed.
- *Source:* Eschmann, Albani, Loianno, 'Learning to Fly in Seconds', IEEE RA-L 9(7), 2024, https://arxiv.org/abs/2311.13081

### 1.10 · confirmed

Swift (champion-level racing) is not a teacher-student method. It is one embodiment, specialised by fitting residual models from real flights and fine-tuning the policy in the augmented simulator. It is the opposite end of the generality axis from RAPTOR.

- *Evidence:* PPO in simulation. The perception residual is modelled with Gaussian processes and the dynamics residual with k-NN regression, both fitted from real data. Fine-tuning used 2×10^7 interactions and was iterated (policy-0 → policy-1 → policy-2); only the control-policy weights were updated, with the gate detector frozen. A secondary summary gives 15/25 wins against three champions. The 2×128 MLP size comes from a third-party repo and is unconfirmed.
- *Source:* Kaufmann, Bauersfeld, Loquercio, Müller, Koltun, Scaramuzza, 'Champion-level drone racing using deep reinforcement learning', Nature 620:982-987, 2023, https://www.nature.com/articles/s41586-023-06419-4

### 1.11 · confirmed

RMA has been transferred to quadrotors spanning large changes in mass and geometry. Learned adaptation extrapolated well beyond the training ranges in simulation, but only along smooth parametric families.

- *Evidence:* Zhang et al. 2023: an adaptation module over the state-action history feeds a 50 Hz high-level controller, with a 500 Hz low-level loop. The target drones differ by at least 10× in mass and planar area, real tests cover a 4.5× mass ratio, and disturbances reach one third of vehicle mass. Extreme Adaptation (T-RO 2025): imitation learning plus RL, with a latent inferred from the sensor-action history. In simulation it generalises to ranges up to 16× broader than training. Real tests cover a 3.7× mass ratio and propeller constants differing by more than 100×, with off-centre payloads and motor failures.
- *Source:* Zhang, Loquercio, Wu, Kumar, Malik, Mueller, 'Learning a Single Near-hover Position Controller for Vastly Different Quadcopters', arXiv 2209.09232 (ICRA 2023), https://arxiv.org/abs/2209.09232 ; Zhang et al., 'A Learning-based Quadcopter Controller with Extreme Adaptation', T-RO 2025, https://arxiv.org/abs/2409.12949

### 1.12 · confirmed

A history-free, domain-randomised generalist controller has to trade speed for robustness. Without randomisation the policy fails to transfer; more randomisation makes it robust but slower.

- *Evidence:* A single state-only network drives the motors of two different real racing quads (3-inch and 5-inch). Randomisation levels of 0/10/20/30% were compared: 0% fails sim-to-real, and wider randomisation makes the policy slower than per-platform models. A separate fixed-wing interception paper found that up to 30% parameter randomisation also degraded performance.
- *Source:* Ferede, Blaha, Lucassen, De Wagter, de Croon, 'One Net to Rule Them All: Domain Randomization in Quadcopter Racing Across Different Platforms', ICRA 2025, https://arxiv.org/abs/2504.21586 ; 'Autonomous fixed-wing interception using reinforcement learning', https://arxiv.org/html/2610.37470

### 1.13 · confirmed

A platform-agnostic action space works well with a history-based dynamics latent. A quadrotor policy that outputs collective thrust and body torques, combined with an encoder that learns platform characteristics from history, removes platform-specific tuning. This is the direct analogue of aero-adapt's virtual-command-plus-allocation design.

- *Evidence:* The policy outputs combined thrust and body torques (CTBT). An MLP dynamics encoder maps history to an 8-D latent, trained with PPO. The details come from secondary summaries.
- *Source:* 'Dynamics-Invariant Quadrotor Control using Scale-Aware Deep Reinforcement Learning', arXiv 2503.09622, 2025, https://arxiv.org/abs/2503.09622

### 1.14 · confirmed

For generalising to unseen embodiments, adding more distinct procedurally generated embodiments matters more than adding more data on a few. One policy over many morphologies transfers zero-shot better than multi-task baselines.

- *Evidence:* Embodiment scaling: about 1,000 procedural embodiments (humanoids, quadrupeds, hexapods); scaling the number of embodiments beats scaling data on a fixed small set; zero-shot transfer to a real Go2 and H1; flat terrain only. URMA: one policy over 16 embodiments of 3 morphologies, real zero-shot on one unseen quadruped. No failure analysis was visible.
- *Source:* Ai, Dai, Bohlinger et al., 'Towards Embodiment Scaling Laws in Robot Locomotion', CoRL 2025, https://arxiv.org/abs/2505.05753 ; Bohlinger et al., 'One Policy to Run Them All', CoRL 2024, https://arxiv.org/abs/2409.06366

### 1.15 · confirmed

Fixed-wing deep-RL attitude control has been flight-tested on a flying wing (Skywalker X8) and matched ArduPlane's PID attitude controller. The authors call large actuation delay in training and diversified simulated dynamics crucial for transfer. In earlier simulation work, a small observation vector with a few past time steps mattered most.

- *Evidence:* TNNLS 2023: about 3 min of flight data needed, performance comparable to the ArduPlane PID, no online learning, linear analysis of the learned controller; one summary says SAC was used. ICUAS 2019: PPO on the X8 model; keeping the observation small but including several previous time steps was the most important factor for training; generalised to unseen severe wind and turbulence. This is single-airframe work, not cross-airframe.
- *Source:* Bøhn, Coates, Reinhardt, Johansen, 'Data-Efficient Deep RL for Attitude Control of Fixed-Wing UAVs: Field Experiments', IEEE TNNLS 35(3), 2023, https://arxiv.org/abs/2111.04153 ; Bøhn, Coates, Moe, Johansen, ICUAS 2019, https://arxiv.org/abs/1911.05478

### 1.16 · confirmed

The closest published fixed-wing teacher-student work trains a privileged DreamerV3 teacher, logs its data, and distils it offline into a transformer student with a context window. The student flies surfaces and throttle directly from altitude, heading and airspeed references, and survives semi-wing damage where the industry baseline crashes. All of it is in simulation.

- *Evidence:* Fault-tolerant control is framed as a POMDP. The teacher uses privileged information, with physical-parameter randomisation at reset and during the episode. Lyapunov-inspired terms are added to offline-RL training. No inner attitude loop and no separate fault detector are used. Network sizes and metrics were not visible.
- *Source:* Giral, Gómez, Vinuesa, Le Clainche, 'Transformer-Based Fault-Tolerant Control for Fixed-Wing UAVs Using Knowledge Distillation and In-Context Adaptation', arXiv 2411.02975 (journal version in Eng. Appl. of AI, 2026), https://arxiv.org/abs/2411.02975

### 1.17 · confirmed

How the privileged fault vector enters the network matters. For a fixed-wing path-following policy conditioned on actuator-fault parameters, FiLM or LoRA hypernetwork conditioning generalised better than a plain MLP with the parameters concatenated to its input, including to unseen time-varying faults.

- *Evidence:* PPO, trained jointly with the hypernetwork, in high-fidelity 6-DoF simulation only. Reported worst-case static rudder-fault path error is about 22 m for FiLM+HC and LoRA against 36.8 m for the MLP, with a much larger gap under flutter faults. The numbers come from a secondary summary.
- *Source:* Marquis, Farhood, 'Hypernetwork-Conditioned RL for Robust Control of Fixed-Wing Aircraft under Actuator Failures', arXiv 2604.03392, 2026, https://arxiv.org/abs/2604.03392

### 1.18 · confirmed

I found no paper that tests a single fixed-wing policy zero-shot across different airframes or configuration classes. Existing fixed-wing RL work is single-airframe robustness (wind, aero uncertainty, adversarial RL, actuator faults). The M3 headline would therefore be novel, as far as these searches reached.

- *Evidence:* Several searches came back empty for cross-airframe fixed-wing results. Nearest neighbours: a Bristol wind-tunnel pitch-control study where only the domain-randomised controller transferred, adversarial RL under aero-model uncertainty (arXiv 2510.16650), and NeuralPlane, a GPU-parallel fixed-wing simulator with a 0.2 s step at 10^6 parallel environments and no faults or cross-airframe split. I could not check an 'Interchangeable RL flight controller for fixed-wing UASs' paper before the budget ran out.
- *Source:* Bristol: https://research-information.bris.ac.uk/ws/portalfiles/portal/339244813/ ; NeuralPlane NeurIPS 2024 D&B: https://proceedings.neurips.cc/paper_files/paper/2024/hash/af97b61e9fef8f55e32a2602af364d8c-Abstract.html ; https://arxiv.org/abs/2510.16650

### 1.19 · confirmed

The 'imitation gap' is the main theoretical failure mode of privileged distillation. Where the teacher's privileged information calls for different actions in states the student cannot tell apart, a cloned student averages them. Remedies mix imitation with reward-driven RL, or make the teacher aware of the student.

- *Evidence:* ADVISOR adaptively weights the imitation and RL losses. TGRL scales teacher guidance against a counterfactual reward-only policy. Student-Informed Teacher Training (ICLR 2025) penalises the teacher for actions the student cannot reproduce and adds an alignment step. DreamWaQ avoids a teacher altogether: an asymmetric actor-critic with a jointly trained context estimator, running under 1 ms on board. Its secondary-reported 95.23% survival against 82.37% for an RMA-style baseline is unverified.
- *Source:* Weihs et al., 'Bridging the Imitation Gap by Adaptive Insubordination', NeurIPS 2021, https://arxiv.org/abs/2007.12173 ; Shenfeld et al., 'TGRL', ICML 2023, https://arxiv.org/abs/2307.03186 ; Messikommer et al., 'Student-Informed Teacher Training', ICLR 2025, https://arxiv.org/abs/2412.09149 ; Nahrendra et al., 'DreamWaQ', ICRA 2023, https://arxiv.org/abs/2301.10602

### 1.20 · confirmed

Transformers over the observation-action history can adapt in context on a real full-size humanoid. The paper notes that combining teacher imitation with RL helped. The context is short; my memory says about 16 steps, which is unconfirmed.

- *Evidence:* A causal transformer over the proprioception and action history, trained by large-scale RL in randomised simulation, deployed zero-shot on Digit. A secondary summary reports zero falls over a week of outdoor tests.
- *Source:* Radosavovic, Xiao, Zhang, Darrell, Malik, Sreenath, 'Real-world humanoid locomotion with reinforcement learning', Science Robotics 2024, DOI 10.1126/scirobotics.adi9579, https://arxiv.org/abs/2303.03381

### 1.21 · confirmed

An interpretable alternative to a black-box latent exists. Neural-Fly learns a shared basis offline (DAIML) and adapts only the linear coefficients online through a composite adaptation law with an exponential-stability guarantee. This is relevant to the allocation layer's online effectiveness estimation.

- *Evidence:* 12 min of flight data for training; tested in a wind tunnel up to 12.1 m/s; about 66% better than a nonlinear tracking controller and 42% better than L1 adaptive control. Code and data are public.
- *Source:* O'Connell, Shi, Shi, Azizzadenesheli, Anandkumar, Yue, Chung, 'Neural-Fly enables rapid learning for agile flight in strong winds', Science Robotics 7(66), 2022, https://arxiv.org/abs/2205.06908

## Recommendations for M3

1. Teacher input: the effect, not the cause. Give the teacher a compact, non-dimensional, control-relevant privileged vector rather than the ~45 raw entries of AirframeParams plus fault one-hots. Suggested contents (16-40 dims): per-axis control effectiveness as angular acceleration per unit virtual command at current dynamic pressure (e.g. q̄·S·b·C_ell_δa/Jx); the per-surface effectiveness multipliers left by faults (1 healthy, 0 jammed, -1 reversed, plus a jam offset moment); damping (C_ell_p, C_m_q, C_n_r scaled by V/b or c); static stability (C_m_α, C_n_β, C_ell_β); V_stall and V_trim relative to the current airspeed; thrust available versus required; actuator tau and delay; mean wind and turbulence intensity. Encode it with a 2-layer MLP into z of 8-16 dims, and feed z through FiLM rather than concatenation (Marquis & Farhood). Describing faults by their effect on the effectiveness matrix rather than by fault ID also prepares for G5 (unseen fault classes).
2. Critic: always asymmetric. Give it the true state, privileged vector, wind and fault onset (L2F: removing it cut real success from 10/10 to 6/10). Keep the actor and critic as separate networks so the critic's size does not count against the 50k budget.
3. Distillation recipe, in three stages. (1) PPO teacher with privileged z. (2) DAgger distillation on the student's own rollouts: execute the teacher's action with probability β decaying from 1 to 0 over the first ~20% of distillation, then pure student. Loss = ||a_student − a_teacher||² in normalised virtual-command space + λ·||z_hat − z||² with λ≈0.1-1, as an auxiliary head on the GRU state (Lee/Miki style). (3) A-RMA-style PPO fine-tuning of the whole student for the last 10-20% of compute, with a decaying KL or BC term to the teacher. Stage 3 closes the imitation gap and lets the student learn excitation behaviour the teacher never needed.
4. Ablations in the A/B report: (a) RMA-pure (latent regression only, frozen base); (b) single-stage asymmetric recurrent PPO with an auxiliary privileged-prediction head (DreamWaQ-style, no teacher); (c) an oracle-z student, i.e. the base policy given true z, to separate estimation error from policy error; (d) a history-free domain-randomised MLP (the Ferede trade-off: expect robust but sluggish). Report teacher(G1), student(G1), student(G2/G3/G4) so the gap splits into distillation loss and generalisation loss.
5. Student architecture: input MLP (obs ~30-40 dims → 64), one GRU layer with 64 hidden units, then a 64→64→4 head. That is about 25k parameters, under the 50k budget, with room for a 4-16-dim auxiliary z head. RAPTOR shows 2k parameters suffice for quadrotors, so a 32-wide GRU is a cheap extra ablation. The student's observations: the estimated state the stock autopilot uses (for a fair A/B), the reference (attitude, airspeed and flight-path targets), the previous action, and optionally the configured nominal parameters from autoconfig D6 (cruise and min airspeed, span, mass), since a real ArduPlane also has these. Track a TCN with a 100-step (2 s) receptive field as the alternative. Skip transformers for L1 in M3.
6. History and BPTT: run the policy at 50 Hz, matching servo update rates and ArduPlane's rate loops, with randomised extra delay. Train with truncated BPTT over 64-128-step segments (1.3-2.6 s) plus a burn-in of 32 steps, with episodes 30-120 s long so the hidden state must stay stable far beyond the BPTT window. Store the initial hidden state per sequence in the PPO buffer and zero it correctly at resets. In JAX this is the most common source of silent bugs. Test on 10-minute flights for hidden-state drift.
7. Randomise delay and actuator dynamics on purpose. L2F got 0/10 real flights without motor-delay modelling, and Bøhn et al. call actuation delay crucial. Sample transport delay 0-60 ms (0-3 steps at 50 Hz), servo tau 0.02-0.15 s, rate limits ±40%, deadband, IMU rate and noise ×0.5-3, and an integrator step jitter so the policy cannot fit aerojax-specific timing. This matters most for G4 (JSBSim Rascal).
8. Embodiment randomisation: the 10,000-airframe fleet is the main source of variety. Embodiment-scaling results favour many distinct airframes over many steps per airframe, so resample the airframe every episode across thousands of airframes. On top, add moderate per-episode perturbations of ±15-25% on aero derivatives, ±20% mass, ±5% MAC CG shift (which moves C_m_α), ±30% inertia and ±20% thrust, to cover the AeroSandbox-to-JSBSim model gap. Avoid going much wider: Ferede et al. and the interception paper found that wide randomisation without adaptation makes policies conservative.
9. Make the virtual commands non-dimensional so one policy spans 0.8-15 kg. Have L1 output angular-acceleration demands (rad/s², or as fractions of a nominal authority) plus a normalised thrust demand, as in the scale-aware CTBT quadrotor work. The allocator turns these into moments with its inertia estimate, then into surfaces with q̄-scaled effectiveness, which is INDI's virtual-control convention and lets the INDI baseline share the allocator. Train L1 with the real allocator, including its online effectiveness estimator, inside the environment so the policy learns to coexist with a second adaptive loop. Add an oracle-allocator ablation (true effectiveness) to attribute failures.
10. OOD diagnostics: compute every held-out airframe's distance from the training fleet in the privileged parameter space (Mahalanobis or k-NN distance), and plot success and altitude lost against that distance with Wilson 95% CIs. Expect G3 (flying wings: elevons, low C_n_β, no rudder) to be the hardest. The allocation layer is what insulates the policy from the change in channel structure, so report G3 with and without it (the raw-surface ablation in §4.2).
11. Pre-register the six headline airframes before training finishes, chosen to span G2 (two, including an extrapolated size or mass), G3 (two flying wings) and G4 (the Rascal plus one other independent model). Pre-register the hard-over scenario too (surface, onset, severity), so the claim 'the same frozen policy flies six airframes it never trained on' is not open to cherry-picking.
12. Imitation-gap monitor: log the teacher-student action gap per fault kind and per time since onset. Faults that are unobservable until excited (a jam inside the current deflection range, a reversed surface during steady cruise) will show a large gap. Mitigations, cheapest first: (i) reveal the fault in the teacher's privileged vector only after a detectability delay, sampled 0.1-1 s, or only once the surface has been commanded away from its jam position; (ii) stage-3 RL fine-tuning; (iii) a TGRL/ADVISOR-style adaptive imitation weight.
13. Scope of sensor faults: in M3, train L1 against sensor noise, bias and latency randomisation, but leave pitot, GNSS and magnetometer faults to M4 (Navigator and Sentinel). A teacher that sees the true airspeed while the student's pitot is blocked opens a large imitation gap and confounds the L1 A/B. State this exclusion in the A/B report.
14. L2 (recovery): train a separate privileged teacher over upset states × actuator faults, starting with mild upsets and widening the curriculum by success rate, Lee-style, and distil it the same way. To avoid switching transients, train L1 from initial conditions sampled from L2's hand-back distribution (bank ±60°, pitch ±30°, off-trim airspeed), and hand back on a state-based region-of-attraction check rather than a timer.
15. Compute budget: at ≥1M aerojax steps/s on one GPU, 1B steps takes about 17 min, so plan roughly 2-5B teacher steps, 0.5-1B distillation steps and 0.5B fine-tuning steps per seed, and 5 seeds per method. Raw compute is not the bottleneck: iteration speed and evaluation discipline are. Spend spare GPU on more seeds and more airframes, not longer runs.
16. Positioning for the write-up: frame M3 as the first cross-airframe and cross-class zero-shot fixed-wing result of which the author is aware. Cite Bøhn et al. (single X8, flight-tested), Giral et al. (single airframe, transformer distillation, simulation only) and Marquis & Farhood (fault conditioning, simulation only). Say plainly that it is evaluated sim-to-sim (aerojax to JSBSim), not flight-tested.

## Risks and pitfalls

1. Latent extrapolation. On a class-level change (flying wing) the history encoder can produce z_hat outside the training manifold, and the base policy then does something undefined. Extreme Adaptation's 16× extrapolation was along a smooth parametric family; a change in configuration class is not that.
2. The imitation gap from unobservable privileged information (jams inside the deflection range, reversed surfaces in steady flight, wind versus airspeed-sensor confounds). The student learns an averaged action. This is the H1 hypothesis in docs/01 resurfacing inside L1.
3. Pure behaviour cloning on teacher rollouts causes covariate shift: the student drifts into states the teacher never visited. Use DAgger with student-driven rollouts; RMA's on-policy data serves the same purpose.
4. Two adaptive loops interacting: the GRU's implicit adaptation and the allocator's online effectiveness estimator can fight, the classic adaptive-control coupling issue (Rohrs et al.), especially at hard-over onset. Mitigation: train with the estimator in the loop, separate their time-scales (fast allocator estimator, slower policy?), and run the oracle-allocator ablation.
5. Simulator shortcutting: a history policy can identify the airframe from simulator-specific artefacts (noise statistics, integrator timing, servo model shape) instead of from physics, and then fail on JSBSim. Randomise these nuisance factors. G4 is the honest check.
6. Missing or under-randomised actuation delay makes transfer fail outright (L2F: 0/10). The JSBSim Rascal's actuator and FCS modelling differs from aerojax's.
7. Over-wide domain randomisation without adequate history makes the policy conservative and sluggish (Ferede et al.; the 30% interceptor result). The tracking error and altitude lost could then lose to per-airframe-tuned PID even when survival improves.
8. GRU hidden-state drift or saturation on flights far longer than the training episodes. Also recurrent-PPO bookkeeping bugs in JAX (stale hidden states after resets, minibatches that cut sequences) that train 'fine' yet produce a policy that never really uses its memory. Test by comparing against a history-free ablation.
9. G2 can look easy because unseen generator airframes share the generator's structure and priors. The G2 result alone does not support 'generalises to new aircraft'. Lean on G3 and G4 and report OOD distance.
10. Teacher leakage: if the privileged vector includes future information (an upcoming gust from the wind field, or a fault before onset), the teacher learns anticipatory actions no student can copy. Give only current values, and reveal faults at or after onset.
11. Reward hacking under faults, e.g. diving to gain airspeed and control authority at the cost of altitude. Score altitude lost and envelope exceedances, not tracking error alone.
12. Headline cherry-picking risk: choose the six airframes and the hard-over case before results come in, and report all of them with CIs.
13. Overclaiming fidelity (per the project's fidelity rule, docs/09 §8.6): every result is simulation-to-simulation. Say so, and note that no fixed-wing cross-airframe RL has been flight-tested in the literature found.

## Open questions

1. Unverified numbers worth checking in the PDFs before they go into docs: RMA's latent dimension, history length and adaptation-module architecture (I believe 8-D z, ~50 steps, 1-D CNN); Lee 2020's student TCN receptive field and loss terms; RAPTOR's GRU width, quadrotor sampling ranges, teacher algorithm and whether distillation is DAgger or offline BC; L2F's algorithm (TD3?) and action-history length; Swift's MLP size and total PPO budget; Radosavovic's context length.
2. Which effective history length do fixed-wing dynamics need? Roll effectiveness should be identifiable in ~0.5-1 s of manoeuvring, but pitch-axis and thrust/drag properties couple to the phugoid (tens of seconds). Is a 2-4 s GRU context enough for L1's scope, or should slow quantities (drag polar, thrust available) come from the L3 self-model?
3. Should L1 be a stand-alone policy or a residual on INDI plus allocation (docs/02 mentions both)? A residual eases RL and gives a safety floor, but muddies the 'learned versus classical' A/B.
4. RAPTOR-style many-specialist distillation versus a single latent-conditioned teacher: with fault × wind variety per airframe, is one privileged PPO teacher good enough everywhere, or do some regions (flying wings near stall, high-aspect gliders) need specialist teachers?
5. Does the student get the configured nominal descriptor (autoconfig D6 parameters)? It is realistic, since ArduPlane has them, but it weakens the 'history-only' claim. It probably needs to be reported both ways.
6. How should the policy and the allocator split adaptation responsibility? Does the allocator estimate only per-surface effectiveness and sign, leaving damping and stability changes to the policy? Does a reversed-surface detection inside the allocator make the policy's job easier or create a switching transient?
7. Is the flying-wing class (G3) inside the allocator's design envelope, i.e. elevon mixing and no rudder? If the generator's flying wings never appear in training at all, can the allocation layer alone bridge the change in channel structure?
8. The 'Interchangeable Reinforcement-Learning Flight Controller for Fixed-Wing UASs' paper (ResearchGate 377280809) may be relevant prior work on cross-airframe fixed-wing RL. The search budget ran out before it could be checked, and it should be read before any novelty claim.
