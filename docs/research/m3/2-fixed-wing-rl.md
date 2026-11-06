# Report 2: Deep RL for fixed-wing control, fault tolerance and upset recovery

*Part of the M3 literature review, 6 Nov 2026 ([README](README.md)). The brief and the findings are as the research agent returned them; local paths are made repository-relative. Where this report and the spec differ, [docs/09](../../09_m3_technical_spec.md) holds.*

## Brief

Deep RL for fixed-wing UAV low-level control, fault-tolerant control and upset recovery: what has been done, what transferred to hardware, and what it means for aero-adapt M3 (L1 adaptive policy + allocation, L2 recovery, A/B on G2/G3/G4). NOTE: the shared web-search budget (200 calls per turn) ran out during this run. Findings marked confirmed=false come from my own knowledge or from a single conflicting snippet, and need checking against the full texts. I did not open any full-text PDFs: the task said to rely on search results, so network sizes, step counts and exact delay values for most papers are not confirmed.

## Key findings (27 of 30 confirmed in the session)

### 2.1 · confirmed

Bøhn et al. (2019): PPO attitude control of the Skywalker X8 flying wing in simulation. The policy stabilises from a large set of initial conditions to roll/pitch/airspeed references, converges in more cases than PID with comparable performance, and generalises to unseen wind and severe turbulence.

- *Evidence:* The abstract says the most important factor for training progress was limiting the number of observation variables and including several previous time steps of them, i.e. a small observation plus a short history. Simulator: PyFly (Python 6-DoF with Dryden turbulence). Its X8 parameters come from wind-tunnel data, and it ships a PID tuned to the X8. Network sizes, step counts and the exact reward were not visible in the search results.
- *Source:* Bøhn, Coates, Moe, Johansen, 'Deep Reinforcement Learning Attitude Control of Fixed-Wing UAVs Using Proximal Policy Optimization', ICUAS 2019, pp. 523-533, DOI 10.1109/ICUAS.2019.8798254, https://arxiv.org/abs/1911.05478 ; PyFly: https://github.com/eivindeb/pyfly

### 2.2 · confirmed

Bøhn et al. (TNNLS 2023) is the main fixed-wing RL sim-to-real result. A policy trained in simulation flew a real Skywalker X8 with performance comparable to ArduPlane's PID attitude controller, with no online learning. Learning with significant actuation delay and diversified (randomised) simulated dynamics was crucial for transfer.

- *Evidence:* Algorithm: SAC, chosen for data efficiency. It learned from as little as 3 minutes of flight data. Hardware: a Cube Orange running ArduPlane, with the ArduPlane PID attitude controller as the in-flight baseline in the same stack. The paper adds a linear analysis of the learned controller to explain its behaviour. Exact delay ranges and loop rate were not confirmed. A companion paper (ICUAS 2022) describes an embedded platform for demanding low-level controllers with direct actuator control, flown alongside a standard autopilot for safety.
- *Source:* Bøhn, Coates, Reinhardt, Johansen, 'Data-Efficient Deep Reinforcement Learning for Attitude Control of Fixed-Wing UAVs: Field Experiments', IEEE TNNLS 35(3):3168-3180, 2023/24, DOI 10.1109/TNNLS.2023.3263430, https://arxiv.org/abs/2111.04153 ; Coates, Reinhardt, Gryte, Johansen, 'Toward Nonlinear Flight Control for Fixed-Wing UAVs: System Architecture, Field Experiments, and Lessons Learned', ICUAS 2022, https://hdl.handle.net/11250/3052343

### 2.3 · confirmed

The X8 in aero-adapt's G3 (flying-wing hold-out) is the same airframe on which RL has been flown for real. That gives M3 an external anchor: on the X8, an RL policy matched ArduPlane PID on nominal attitude tracking.

- *Evidence:* Repo: docs/05 §2 lists the flying wing as 'Held out entirely: only X8 and Zagi as references', with the X8 flown on its published coefficients. Bøhn's sim and flights use the X8 too.
- *Source:* docs/05_m2_technical_spec.md (line ~211, ~600); Bøhn et al. 2019/2023 above

### 2.4 · confirmed

Wind-tunnel sim-to-real for fixed-wing pitch (Bristol). A policy trained on a linear model failed on the reality gap. High-fidelity modelling and domain randomisation both transferred, but only the domain-randomised controller kept its performance when conditions changed. A companion study varied training delay offsets of 15, 150 and 250 ms.

- *Evidence:* 1-DoF pitch rig in a wind tunnel, elevator control. Three training regimes compared. Quantitative tracking numbers were not in the snippets.
- *Source:* Wada, Araujo-Estrada, Windsor, 'Sim-to-Real Transfer for Fixed-Wing Uncrewed Aerial Vehicle: Pitch Control by High-Fidelity Modelling and Domain Randomization', IEEE RA-L 7(4):11735-11742, 2022, https://ieeexplore.ieee.org/document/9882333/ ; Wada et al., 'Unmanned Aerial Vehicle Pitch Control under Delay Using Deep RL with Continuous Action in Wind Tunnel Test', Aerospace 2021, https://doi.org/10.3390/aerospace8090258

### 2.5 · confirmed

Model-free versus model-based RL for fixed-wing attitude in wind. TD-MPC beat PID, PPO and SAC on hard nominal attitude references. Under turbulence the gap closes: PPO and TD-MPC are about equal to PID, and SAC is clearly worse. TD-MPC's learned model generalised poorly under random gusts.

- *Evidence:* The paper adds actuation fluctuation as a metric. An action-variation penalty (AVP) reduced fluctuation for SAC and TD-MPC but not for PPO. Simulation only.
- *Source:* Olivares, Fournier, Vasishta, Marzat, 'Model-Free versus Model-Based Reinforcement Learning for Fixed-Wing UAV Attitude Control Under Varying Wind Conditions', ICINCO 2024, https://arxiv.org/abs/2409.17896

### 2.6 · confirmed

Koch/Mancuso GymFC (quadrotor, the attitude-RL precedent): PPO beat DDPG and TRPO, and beat PID on rise time and overall error for inner-loop attitude control. A follow-up (Mysore et al.) produced smooth PPO attitude controllers that flew real quadrotors.

- *Evidence:* Algorithms compared: DDPG, TRPO, PPO in GymFC. Follow-up arXiv 2012.06656 reports PPO agents tracking real-flight rotation rates faster than a tuned PID, in simulation and on hardware. The CAPS smoothness regulariser (temporal plus spatial Lipschitz terms; Mysore et al., ICRA 2021) is from my knowledge, not confirmed this session.
- *Source:* Koch, Mancuso, West, Bestavros, 'Reinforcement Learning for UAV Attitude Control', ACM TCPS 2019, https://arxiv.org/abs/1804.04154 ; Mysore et al., 'How to Train your Quadrotor: A Framework for Consistently Smooth and Responsive Flight Control via RL', https://arxiv.org/abs/2012.06656

### 2.7 · confirmed

JSBSim-based RL environments: gym-jsbsim (Gor-Ren, Bath), and its forks jsbgym and gymnasium-jsbsim. They offer HeadingControl and TurnHeadingControl tasks on the Cessna 172P. Agents learned altitude and heading hold by commanding control surfaces directly with continuous actions.

- *Evidence:* Environment IDs look like JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0, with shaped and unshaped reward variants. The search did not confirm which algorithm the Bath study used (I believe PPO) or its sample counts.
- *Source:* gym-jsbsim (Gor-Ren), https://pypi.org/project/gym-jsbsim/ ; jsbgym, https://pypi.org/project/jsbgym/ ; gymnasium-jsbsim, https://pypi.org/project/gymnasium-jsbsim/ ; Bath, 'Autonomous control of simulated fixed wing aircraft using deep RL', https://researchportal.bath.ac.uk/en/publications/autonomous-control-of-simulated-fixed-wing-aircraft-using-deep-re/

### 2.8 · confirmed

Fault-tolerant RL on a business jet (TU Delft, PH-LAB Cessna Citation model). An offline-trained cascaded SAC controller survived six failure cases: rudder jammed at -15 deg, aileron effectiveness down 70%, structural failure, icing and an aft CG shift. It flew a coordinated 40 deg bank climbing turn with 2.64% normalised MAE.

- *Evidence:* It was also robust to biased sensor noise, atmospheric disturbances and varied initial conditions and reference shapes. A follow-up hybrid SAC-IDHP controller (online adaptive critic) improved nMAE over SAC alone by 0.74% (nominal), 5.46% (longitudinal failures) and 0.82% (lateral failures). Earlier IDHP work (Heyer et al. 2020) learned near-optimal control online in simulation. I found no report of an RL controller actually flown on PH-LAB. Code: github kdally/fault-tolerant-flight-control-drl.
- *Source:* Dally & van Kampen, 'Soft Actor-Critic Deep RL for Fault Tolerant Flight Control', AIAA SciTech 2022, https://arxiv.org/abs/2202.09262 ; Teirlinck & van Kampen, AIAA SciTech 2024, https://repository.tudelft.nl/record/uuid:dc63cae7-4289-47c7-889e-253f7abd7c72 ; Heyer, Kroezen, van Kampen, AIAA SciTech 2020, https://pure.tudelft.nl/ws/portalfiles/portal/68362912/6.2020_1844.pdf

### 2.9 · confirmed

The closest prior art to L1's teacher-student plan for fixed-wing faults is a transformer controller. An RL teacher with privileged environment information was distilled into a partially observed student that adapts in context, with no separate fault-detection layer.

- *Evidence:* Failures tested: jammed rudder, broken aileron, semi-wing damage. In the semi-wing-damage case a conventional FCS loses control and crashes, while the student stabilises and keeps tracking. The authors claim it beats an industry-standard FCS and state-of-the-art RL. Student data: teacher trajectories logged as (partial observation, action, reward), i.e. offline distillation rather than DAgger. Simulation only. Teacher-versus-student numbers were not visible in the search results.
- *Source:* Giral, Gómez, Vinuesa, Le Clainche, 'Transformer-Based Fault-Tolerant Control for Fixed-Wing UAVs Using Knowledge Distillation and In-Context Adaptation', arXiv 2411.02975 (2024), https://arxiv.org/abs/2411.02975

### 2.10 · confirmed

Conditioning a small-UAS path-following RL policy on an actuator-fault descriptor through a hypernetwork (FiLM/LoRA-style modulation) beat plain MLP policies and generalised to time-varying failure modes it never saw in training. Separately, adversarial RL that perturbs the aerodynamic coefficients (rate-bounded) beat training with stochastic model uncertainty.

- *Evidence:* 6-DoF high-fidelity simulation (Virginia Tech). A third-party summary puts the MLP's path deviation near 40 m under rudder flutter; that figure is lower confidence. The fault vector is privileged, so this is the teacher half of a teacher-student scheme. The RARL paper also discusses action/observation choice and reward shaping for fixed-wing path following.
- *Source:* Marquis & Farhood, 'Hypernetwork-Conditioned RL for Robust Control of Fixed-Wing Aircraft under Actuator Failures', arXiv 2604.03392 (Apr 2026), https://arxiv.org/abs/2604.03392 ; Marquis, Wilhelm, Muniraj, Farhood, 'Adversarial RL for Robust Control of Fixed-Wing Aircraft under Model Uncertainty', arXiv 2510.16650 (Oct 2025), https://arxiv.org/abs/2510.16650

### 2.11 · confirmed

Further simulation-only fixed-wing fault-tolerant RL: a deep RL plus evolutionary-strategies attitude controller benchmarked on a saturated aileron, a rudder jammed at 15 deg and a partial elevator loss.

- *Evidence:* ICGNC 2024 draft from Tsinghua. No hardware.
- *Source:* Zongo & Li, ICGNC 2024 draft, https://alex-zongo.github.io//files/icgnc2024_paper_draft3.pdf

### 2.12 · confirmed

Stall recovery with PPO, framed as minimum altitude loss. PPO matched value iteration on a reduced-order problem, then solved the full general-aviation problem, and its policies agree with published GA stall-recovery guidance. Throttle timing matters: delaying full throttle by only 0.75 s increased altitude loss by up to 30%.

- *Evidence:* Actions are control surfaces plus throttle. Objective: total altitude loss. Simulation only.
- *Source:* Grillo, Torre, Bunge, 'Optimal Stall Recovery via Deep Reinforcement Learning for a General Aviation Aircraft', AIAA SciTech 2024, https://gabo-tor.github.io/publication/grillo-optimal-2024

### 2.13 · confirmed

Spin and loss-of-control recovery with PPO on an F-18/HARV 6-DoF model with nonlinear aerodynamics, actuator saturation and rate limits. The reward is potential-based in two phases: first penalise large angular rates (arrest the rotation), then enforce trimmed flight. The policy generalised to unseen upset initial conditions and compared favourably with sliding-mode control.

- *Evidence:* A first-pass search summary said training took about 6,000 episodes; a second search could not confirm that number. Safety-constrained learning is listed as future work. Simulation only.
- *Source:* Sayyed, Konar, Sinha, 'Deep RL based Control Design for Aircraft Recovery from Loss-of-Control Scenario', arXiv 2601.06439 (Jan 2026), https://arxiv.org/abs/2601.06439

### 2.14 · confirmed

SAC-based upset (stall and spin) recovery for a jet trainer, trained on a high-fidelity model with stall and spin start states sampled over a discretised state space, then validated on JSBSim's F-16. The authors claim faster roll and pitch correction and less negative-g exposure than classical methods.

- *Evidence:* Results are self-reported and come mostly through AI-generated secondary summaries. Domain experts reviewed the behaviour qualitatively. Simulation only.
- *Source:* Demir et al., 'An Aircraft Upset Recovery System with Reinforcement Learning', arXiv 2604.24355 (Apr 2026), https://arxiv.org/abs/2604.24355

### 2.15 · confirmed

Across every RL upset-recovery paper I found for fixed-wing aircraft, none reports a flight test or formal safety verification. Recovery RL for fixed-wing is a simulation-only literature.

- *Evidence:* Seen across the search summaries for arXiv 2601.06439, AIAA 2024 Grillo and arXiv 2604.24355. A related quadrotor fall-recovery paper (arXiv 2606.16513) uses a recurrent policy with an asymmetric actor-critic and an INDI inner loop that tracks the policy output. That is the hybrid pattern L2 could copy.
- *Source:* As above; 'Agile Fall Recovery for Quadrotors with Bidirectional Thrust via RL', https://arxiv.org/abs/2606.16513

### 2.16 · confirmed

RAPTOR shows how small an adaptive student can be. A 3-layer recurrent policy with 2,084 parameters, distilled from 1,000 per-quadrotor RL teachers by Meta-Imitation Learning, adapts zero-shot within milliseconds to 10 real quadrotors from 32 g to 2.4 kg.

- *Evidence:* Control is framed as a Bayes-adaptive POMDP with unobserved dynamics parameters, and adaptation is in-context through the recurrent hidden layer. One teacher per platform, not one teacher conditioned on parameters.
- *Source:* Eschmann, Albani, Loianno, 'RAPTOR: A Foundation Policy for Quadrotor Control', arXiv 2509.11481 (v2 Apr 2026), https://arxiv.org/abs/2509.11481

### 2.17 · confirmed

An RMA-style learned quadrotor controller adapts across a range 16 times wider than its training distribution in simulation. On hardware it handled quadrotors differing 3.68x in mass and 3.1x in arm length, propeller constants varying more than 100x, off-centre payloads and motor failures.

- *Evidence:* T-RO 2025. Code: github.com/muellerlab/xadapt_ctrl.
- *Source:* Zhang, Loquercio, Tang, Wang, Malik, Mueller, 'A Learning-based Quadcopter Controller with Extreme Adaptation', IEEE T-RO 2025, https://arxiv.org/abs/2409.12949

### 2.18 · confirmed

RMA pattern details: an RL base policy runs at 100 Hz, and a supervised adaptation module runs asynchronously at 10 Hz. The module regresses the privileged 'extrinsics' latent from recent state-action history, trained against the true extrinsics available in simulation.

- *Evidence:* Running asynchronously was critical for the low-compute A1 robot. The 50-step history window comes from secondary sources.
- *Source:* Kumar, Fu, Pathak, Malik, 'RMA: Rapid Motor Adaptation for Legged Robots', RSS 2021, https://arxiv.org/abs/2107.04034

### 2.19 · confirmed

'Learning to Fly in Seconds': sim-to-real direct RPM control of a Crazyflie after 18 s of training on a consumer laptop, using an asymmetric actor-critic (privileged critic), a curriculum and an action history to cope with motor delay. The simulator runs about 5 months of flight per second on a laptop GPU.

- *Evidence:* RA-L 2024, vol. 9, pp. 6336-6343. Code: github.com/arplaboratory/learning-to-fly.
- *Source:* Eschmann, Albani, Loianno, 'Learning to Fly in Seconds', IEEE RA-L 2024, https://arxiv.org/abs/2311.13081

### 2.20 · confirmed

Throughput reference points for massively parallel fixed-wing simulation. NeuralPlane takes 0.2 s per step at 1e6 parallel aircraft (about 5M steps/s) and ships hierarchical RL-plus-classical templates. FALCON-S (NVIDIA Warp) reports 19.46M env-steps/s at 16,384 environments, and a PPO altitude policy doing 150M steps in under 5 minutes. A pure-JAX quadrotor stack (Crazyflow) compiles the whole PPO loop on the GPU, using 2,048 environments and 2M timesteps for tracking.

- *Evidence:* aero-adapt's aerojax runs about 0.85M steps/s on a laptop CPU, and its GPU number (D9) is pending. On CPU, 1e9 steps would take about 20 h.
- *Source:* Xue et al., 'NeuralPlane', NeurIPS 2024 Datasets & Benchmarks, https://proceedings.neurips.cc/paper_files/paper/2024/hash/af97b61e9fef8f55e32a2602af364d8c-Abstract.html ; FALCON-S, arXiv 2609.06046, https://arxiv.org/abs/2609.06046 ; Crazyflow, arXiv 2606.01478, https://arxiv.org/abs/2606.01478

### 2.21 · confirmed

Real-world fixed-wing learning: a 2 m glider learned thermal soaring by RL in the field, with the controller adjusting bank and pitch. The learned strategy used vertical wind acceleration and roll-wise torque as cues, and climbs reached about 700 m.

- *Evidence:* Training pooled the glider's flight experience over several days. This is a high-level policy over attitude set-points, not low-level surface control.
- *Source:* Reddy, Wong-Ng, Celani, Sejnowski, Vergassola, 'Glider soaring via reinforcement learning in the field', Nature 562:236-239, 2018, DOI 10.1038/s41586-018-0533-0

### 2.22 · confirmed

DARPA ACE: an RL agent from Shield AI flew within-visual-range dogfights on the X-62A VISTA, a modified F-16, against a manned F-16 from September 2023, with a safety pilot aboard. Reporting says the agent had to be given explicit safety rules, because otherwise it would push the aircraft to and beyond its limits.

- *Evidence:* This is tactical decision-making on top of the aircraft's own flight-control system, not low-level RL control. The runtime-assurance mechanism was not confirmed in the search.
- *Source:* Shield AI, 'Autonomy for the World: X-62 VISTA', https://shield.ai/autonomy-for-the-world-x-62-vista/ ; The Debrief on the ACE programme, https://thedebrief.org/darpas-groundbreaking-ace-program-and-x-62a-becomes-first-ai-controlled-jet-to-dogfight-against-manned-f-16-in-real-world/

### 2.23 · confirmed

NASA Learn-to-Fly is the classical counterpart to online effectiveness estimation. Real-time system identification (equation error in the frequency domain, recursive Fourier transform, multivariate orthogonal functions) fed adaptive NDI or L1 adaptive control with switched reference models. It was flight-tested on a subscale fixed-wing aircraft (110 cc engine, about 50 kn cruise).

- *Evidence:* NTRS reports by Morelli, Snyder, Grauer and colleagues, 2014-2021.
- *Source:* NASA NTRS, e.g. https://ntrs.nasa.gov/citations/20190027207 ; 'L1 Adaptive Control with Switched Reference Models: Application to Learn-to-Fly', https://arxiv.org/abs/2108.08462

### 2.24 · confirmed

I found no fixed-wing paper that trains an RL policy to output virtual moment commands through a WLS or INDI allocator with online effectiveness estimation for actuator-fault reallocation. The nearest work: a QP allocator with a slow RL layer that tunes weights only in the null space (EuroGNC 2026, ICE aircraft); learned nonlinear allocation for INDI (arXiv 2604.06337); and model-based fault-tolerant allocation that assumes the fault is known (arXiv 2510.04853).

- *Evidence:* This is a gap, so aero-adapt's L1-plus-allocation design is plausibly novel for fixed-wing aircraft.
- *Source:* EuroGNC 2026 paper 033, https://eurognc.ceas.org/archive/EuroGNC2026/pdf/CEAS-GNC-2026-033.pdf ; https://arxiv.org/abs/2604.06337 ; https://arxiv.org/abs/2510.04853

### 2.25 · confirmed

Reward design: in a 2026 comparison of PPO, SAC and TD3 on a kinematic fixed-wing simulator, waypoint observations and control-rate penalties were essential for stable control, and goal-only rewards produced unstable behaviour.

- *Evidence:* Journal of Intelligent & Robotic Systems, 2026. Only the snippet was seen.
- *Source:* https://link.springer.com/article/10.1007/s10846-026-02408-7

### 2.26 · confirmed

Residual RL on a fixed-wing UAS: domain randomisation plus observation stacking was used to learn a residual augmentation that compensates matched uncertainties. The University of Kansas Flight Research Lab lists the paper under flight tests.

- *Evidence:* IEEE TAES 2024. The details of the flight result were not confirmed.
- *Source:* Chowdhury & Keshmiri, 'Interchangeable Reinforcement-Learning Flight Controller for Fixed-Wing UASs', IEEE TAES 2024, https://www.researchgate.net/publication/377280809

### 2.27 · **not confirmed** (memory or secondary)

A preprint reports a PPO policy flown on a real X-UAV Talon (Pixhawk 6C, ArduPilot, Stable-Baselines3) for fixed-wing interception, mapping filtered vision and flight-controller data to actuator commands.

- *Evidence:* This appeared in a single search summary (arXiv 2610.37470). A follow-up title search returned a different interception paper (arXiv 2407.06909, by Giral et al., using DreamerV3, TQC and SAC, with no PPO). Treat as unverified.
- *Source:* https://arxiv.org/html/2610.37470 (unverified)

### 2.28 · **not confirmed** (memory or secondary)

Massively parallel PPO recipe (legged robots, which carries over to JAX fleets): thousands of environments with short rollouts (about 24 steps per environment per update), 5 epochs, 4 minibatches, an adaptive-KL learning rate around 1e-3, gamma 0.99, lambda 0.95, clip 0.2. Locomotion trained in minutes on one GPU.

- *Evidence:* From memory of Rudin et al. 2021. Not confirmed this session.
- *Source:* Rudin, Hoeller, Reist, Hutter, 'Learning to Walk in Minutes Using Massively Parallel Deep RL', CoRL 2021, https://arxiv.org/abs/2109.11978

### 2.29 · **not confirmed** (memory or secondary)

Clarke & Hwang (Purdue) used deep RL for aerobatic manoeuvres of an agile fixed-wing aircraft in simulation. Bristol (Clarke and others; Fletcher's thesis) used RL, including NAF, for high-alpha aerobatic and post-stall manoeuvres.

- *Evidence:* The title search did not find the Purdue paper. The Bristol work exists in the university's portal. Details unconfirmed.
- *Source:* Clarke & Hwang, AIAA SciTech 2020 (unverified); Bristol, https://research-information.bris.ac.uk/en/publications/reinforcement-learning-derived-high-alpha-aerobatic-manoeuvres-fo/

### 2.30 · confirmed

Repo fact with design consequences: aerojax's actuator and fault model has three lumped surface channels (aileron, elevator, rudder) plus throttle. There are no separate left/right ailerons, elevons or ruddervators. The control step is 100 Hz (5 ms physics x 2 substeps), servo transport delay is capped at 150 ms (32-step ring buffer at 200 Hz), and the actuator fault kinds are hard_over, floating, slow, reduced_throw, backlash and reversed, per channel.

- *Evidence:* faultlab/schema.py: channel is Literal['aileron','elevator','rudder']. sim/aerojax/actuators.py: surfaces has shape (3,), DELAY_BUFFER = 32. sim/aerojax/sim.py: dt_physics = 0.005, substeps = 2. generator/design.py: CLASSES = conventional, t_tail, v_tail, twin_boom, glider (no flying wing).
- *Source:* src/aeroadapt/faultlab/schema.py ; src/aeroadapt/sim/aerojax/actuators.py ; src/aeroadapt/sim/aerojax/sim.py ; src/aeroadapt/generator/design.py

## Recommendations for M3

1. Split the actuators before building allocation. With three lumped channels (aileron, elevator, rudder) the 3-moment WLS allocation is a square 3x3 inversion with no null space. A 'jammed aileron' cannot be re-allocated to 'the other aileron', and the virtual-command-versus-raw-surface ablation will show almost nothing. Model the physical surfaces separately: left and right aileron, left and right elevator or ruddervator, rudder, optional flaps; elevons for the X8 and Zagi. Apply faults per physical surface, and build the effectiveness matrix B (3 moments x n surfaces) per airframe from the AeroSandbox derivatives.
2. Make the L1 action space airframe-invariant. Across a 0.8-15 kg fleet, raw moment demands in N·m vary by orders of magnitude. Have the policy output normalised moment-coefficient demands (or angular-acceleration demands as a fraction of estimated available authority) plus throttle. The allocator converts these with the measured dynamic pressure and the online B estimate. Keep the roll/pitch/airspeed (or flight-path) references that the existing L1-guidance and TECS outer loops produce, as Bøhn did. This replaces the inner attitude and energy loops, not guidance.
3. Keep the student's observation small, plus a short history (Bøhn's main training lesson): gyro rates, attitude (gravity vector in body axes), measured airspeed (possibly faulted), reference errors, previous allocator outputs and throttle, and dynamic pressure. Feed it to a GRU (hidden 64-128; about 20-30k parameters in total, well under the 50k budget; RAPTOR shows about 2k can be enough). Do not give the student angle of attack or sideslip, since small UAVs rarely measure them. The teacher and the critic get privileged inputs: normalised airframe parameters (mass, inertia, damping and control derivatives, stall speed), fault kind, magnitude and onset, the true B, wind, and true AoA and sideslip.
4. Training recipe: run PPO on the GPU with jit-compiled aerojax (D9 first). Suggested settings: 4k-16k environments, 32-64-step rollouts, gamma 0.99 at 50 Hz, GAE lambda 0.95, clip 0.2, adaptive-KL learning rate, observation and reward normalisation, privileged critic. Budget 1-5e9 teacher steps. NeuralPlane (about 5M steps/s) and FALCON-S (about 19M steps/s) suggest 1e9 steps takes roughly 1-2 h per GPU with PPO overhead, versus about 20 h on the CPU at 0.85M steps/s. Run 5 seeds from the start; seed variance is part of the A/B.
5. Distil teacher to student with DAgger-style on-policy imitation: the student flies and the teacher labels, using a beta-mixture schedule. Do not use pure offline behaviour cloning on teacher rollouts (Giral et al. did offline distillation). Optionally add an RMA-style auxiliary loss so the GRU state regresses the teacher's privileged latent. Then fine-tune the student with PPO and the asymmetric critic, which closes the imitation gap where the teacher used information the history cannot reveal. Report teacher, student and stock PID together; the teacher-student gap is the adaptation cost.
6. Rates and delays: run L1 at 50 Hz with the action held over two 100 Hz control steps, and the allocator at 100 Hz. Randomise total loop delay (sensor + compute + MAVLink + servo) over about 10-100 ms, servo lag, rate limits, gyro noise and bias, airspeed noise and lag, and ±20-30% on inertia, damping and control derivatives on top of the procedural fleet. Bøhn found 'significant actuation delay and diversified dynamics' crucial for the real X8, and Wada found domain randomisation, not fidelity, kept performance when conditions changed. Add policy-period jitter.
7. Reward for L1: a sum of bounded per-term tracking rewards (exp or clipped-abs kernels on roll, pitch or flight-path, and airspeed errors, normalised by airframe-agnostic scales). Add a small penalty on allocator-output rate (or a CAPS-style temporal and spatial smoothness regulariser; Olivares found an action-variation penalty did not smooth PPO), envelope penalties (load factor, airspeed below 1.2 V_stall, bank limit) and a modest per-step survival term. Terminate on ground contact or envelope exit, but keep the crash penalty small so it does not swamp tracking. Report Olivares-style actuator fluctuation alongside tracking error.
8. Curriculum for L1: first nominal airframes in calm air across a wide spread of initial conditions (Bøhn's breadth of initial conditions), then add wind and turbulence, then faults by family. Use an automatic curriculum that up-weights each (fault kind, airframe class) cell by its current failure rate. Sample references both randomly (steps and ramps across the envelope) and from the closed-loop guidance stack, so the training distribution matches deployment.
9. Prepare the policy for the G3 flying wings without leaking the class: in training, randomly zero or reduce rudder effectiveness, or remove the rudder, on conventional, T-tail and twin-boom airframes, and randomise directional stability downwards. The policy then learns to fly with rank-deficient yaw authority, and the allocator learns to handle a B with no usable yaw column. Flying wings stay unseen, but 'no yaw control' does not.
10. Online effectiveness estimation: run a recursive least-squares estimator of B (or a Learn-to-Fly-style orthogonal-function / recursive-Fourier estimator) driven by gyro-derived angular acceleration and commanded deflections. Test a sign flip on each surface column explicitly, to catch 'reversed' within about 0.5 s. Feed the estimated B, or its confidence, to the student as an input. This keeps the fault explicit and inspectable, as §4.2 intends, and provides a classical comparator for L3/L4 later.
11. Design L2 around the evidence: (a) reward altitude loss directly (Grillo) with Sayyed's two-phase potential shaping: first arrest angular rates, then reach trim with positive climb and airspeed in band. (b) L2 must own throttle, since throttle timing alone changed altitude loss by up to 30%. (c) Sample start states from where the stock B1 stack actually fails under faults (the roughly 17% non-survival set), plus a reverse curriculum from mild bank and pitch to inverted and steep nose-low spirals. (d) Hand back to L1 on a predicate with hysteresis (wings within 15 deg, positive flight-path angle, airspeed above 1.3 V_s, held for 1-2 s), and keep L1's GRU running throughout so its hidden state is warm at hand-back. (e) Train L1 on L2's terminal-state distribution so the switch never sees an out-of-distribution state.
12. Restrict L2 claims to upsets that aerojax's aerodynamics can represent. Beard & McLain-style post-stall blending is not validated for spins or deep stall. Limit M3's L2 curriculum to bank above 90 deg, inverted, nose-low spirals and stall entry and exit, leaving spins out, and cross-check every L2 'win' on the JSBSim Rascal in G4. Use 'scripted recovery' (the cut-if-behind) as the baseline, and state that these are simulation-only results; no fixed-wing RL recovery paper has flight data.
13. Make the hard-over headline physically fair. With the current lumped aileron channel, a full-travel hard-over leaves only rudder-induced roll (through dihedral), and many airframes cannot recover. Once surfaces are split, use a single-aileron hard-over. Label each hard-over scenario 'recoverable' with a trim or reachability check (can wings-level steady flight be trimmed with that surface at its stop?) and report survival normalised to the recoverable set, as the oracle-normalised mission metric does.
14. A/B protocol: L1 student versus B1 (PID+TECS tuned per airframe) versus INDI+WLS versus a raw-surface ablation (embodiment descriptor in, surfaces out), on identical paired scenarios over G2, G3 (X8 and Zagi) and G4 (Rascal in JSBSim). Use 5 seeds, Wilson 95% intervals for survival, and paired bootstrap intervals for continuous metrics (tracking error, altitude lost, time to stabilise, actuator fluctuation). Expect about-equal nominal tracking against tuned PID (Bøhn on the X8; Olivares in turbulence) and frame the headline on fault survival and unseen airframes. Put the X8 results next to Bøhn's real-flight finding (RL comparable to ArduPlane PID) as an external sanity anchor.
15. Add Bøhn's linear-analysis credibility artefact: linearise the closed loop (student plus allocator plus airframe) around trim on each G2-G4 airframe, and report gain and phase margins and bandwidth next to B1's. This is cheap and answers 'is the network stable?' for hiring-audience readers.
16. For G4, put exactly the same sensor, delay and actuator wrappers around the JSBSim Rascal as around aerojax, and measure its effective servo and FCS delay. Use JSBSim single-environment runs as a fast cross-simulator smoke test during development, not only at the end.
17. Keep the INDI baseline honest: synchronise the filters on the gyro-derived angular acceleration and on actuator feedback, which matters most with 10-100 ms delays. Give INDI the same online B estimate as L1's allocator, so the A/B isolates the learned policy rather than the allocation.

## Risks and pitfalls

1. Allocation does nothing with lumped surfaces: with three surface channels plus throttle against three moments there is no redundancy, so re-allocation and the allocation ablation would be uninformative, and 'recovering from a hard-over aileron' may be physically impossible for many airframes.
2. Action and reward scaling across the fleet: inertia and control power span orders of magnitude between 0.8 and 15 kg. Unnormalised moment actions or reward terms give PPO a policy that is sluggish on agile airframes and unstable on sluggish ones (the 'average airframe' collapse). The privileged teacher, coefficient-space actions and per-term bounded rewards mitigate this.
3. Imitation gap: the teacher can use privileged information the student cannot infer from a short history. Examples: a reversed rudder in straight, unexcited flight; a pitot fault that looks like wind. DAgger then teaches the student to act on information it does not have. Mitigations: PPO fine-tuning of the student, noisy or limited privileged inputs to the teacher, and reporting the teacher-student gap.
4. Low excitation defeats both RLS effectiveness estimation and history-based adaptation. In smooth cruise, B estimates drift, and a sign flip may go undetected until a manoeuvre. Plan small probing inputs, or use estimator confidence gating.
5. High-frequency or bang-bang actions: PPO policies chatter, and on real servos that means heat, wear and current draw. Olivares found an action-variation penalty did not fix PPO. Use CAPS-style regularisation or a parameterised low-pass on the output, and report actuator fluctuation.
6. Reward hacking: survival bonuses plus crash termination can teach 'hold high airspeed or dive to stay alive' rather than track. Altitude-loss rewards can teach exploiting unrealistic post-stall lift in a simple aero model. L2 is the most exposed, because aerojax's high-alpha and spin aerodynamics are not validated.
7. Overclaiming: all fixed-wing RL upset-recovery and fault-tolerant control results in the literature are simulation-only. The only real-flight RL attitude result on a small fixed-wing aircraft (Bøhn, on the X8) is nominal attitude tracking, about equal to ArduPlane PID. M3 claims must say 'in simulation', and the B1 baseline is ArduPlane-equivalent, not ArduPlane itself (consistent with the project's no-overclaim rule).
8. Delay mismatch: training without realistic, randomised delay was the documented cause of transfer failure for the real X8. Too much delay produces sluggish, low-gain policies. Measure the hardware MAVLink and servo path later, and keep the 150 ms schema cap in mind.
9. G3 is a large jump. The training fleet has no tailless, elevon-only, weak-directional-stability airframes, so a raw-surface policy is very likely to fail outright. Even virtual commands can fail if the policy has learned to rely on yaw authority.
10. Seed variance and cherry-picking: single-seed RL results are not credible. Five seeds with paired scenarios are needed, and the training seed must be separate from the scenario seed.
11. Compute risk: at 0.85M steps/s on CPU, a 1-5e9-step teacher plus distillation will not fit week 12-16 wall-clock without the pending GPU run (D9). A JAX recompilation or memory blow-up from per-airframe parameter batches can eat days. Benchmark GPU throughput with policy inference in the loop first.
12. GRU hidden state across mode switches: resetting or freezing L1's state while L2 or the Simplex fallback flies means re-adapting from scratch at hand-back, just when the aircraft is fragile.
13. Distribution shift between random references and closed-loop guidance references (L1 nav plus TECS): a policy trained only on random steps can misbehave on the smooth, coupled commands the real stack produces.
14. Search caveats: exact hyperparameters, network sizes, step counts and delay values for the Bøhn papers, Dally/van Kampen and Sayyed et al. (including the 6,000-episode figure) were not confirmed from full texts. The X-UAV Talon PPO flight-test preprint (arXiv 2610.37470) is unverified.

## Open questions

1. Should L1 output normalised moment-coefficient demands (my recommendation), angular-acceleration demands (INDI-like, which blurs the A/B against B2), or body-rate commands to a classical rate loop (most transferable to ArduPlane, but a less ambitious claim)?
2. Will aerojax gain per-surface actuators (left/right aileron, elevons, ruddervators) and per-surface faults in week 12? This decides whether the adaptive-allocation story can be shown at all, and it affects faultlab, the generator, the Rascal adapter and Skyview.
3. Which six held-out airframes make the headline: for example two G2 size/mass extremes, X8 and Zagi (G3), Rascal (G4), plus one more? Which hard-over (channel, side, magnitude) counts as 'recoverable' under a trim/reachability check?
4. Can aerojax's post-stall aerodynamics support meaningful L2 training, or should M3's L2 be limited to non-spin upsets, with spins deferred until there are high-alpha and rotary-balance tables (or JSBSim models with high-alpha data)?
5. What end-to-end loop delay should be expected on the eventual hardware path (companion computer to ArduPlane over MAVLink servo override, or a native module)? Bøhn's ICUAS 2022 architecture paper would answer this but was not read in full.
6. Is a single parameter-conditioned teacher enough across 10,000 airframes and 26 fault kinds, or do teachers per class or per fault family (RAPTOR's 1,000-teacher approach) distil better?
7. How fast must reversed-surface detection be for survival (likely under 1 s in roll)? Does the GRU student detect it faster than RLS sign tests?
8. Exact training details from the primary sources that need a full-text read: Bøhn 2019 (PPO network sizes, steps, reward), Bøhn 2023 (delay ranges, rate, SAC settings, control outputs on the X8), Sayyed 2026 (episode count, reward terms), Giral 2024 (teacher-versus-student numbers), and whether TU Delft ever flew an RL controller on PH-LAB.
9. Did the web-search budget cut off relevant 2025-2026 fixed-wing real-flight RL work? A follow-up search pass on '2025-2026 fixed-wing RL flight test', 'RL control allocation fixed-wing' and 'Kang et al. Drones 2026 JSBSim F-16 PPO' is advisable.
