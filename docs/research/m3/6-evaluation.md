# Report 6: Evaluating the M3 controllers

*Part of the M3 literature review, 6 Nov 2026 ([README](README.md)). The brief and the findings are as the research agent returned them; local paths are made repository-relative. Where this report and the spec differ, [docs/09](../../09_m3_technical_spec.md) holds.*

## Brief

How to evaluate the M3 RL controllers rigorously: rliable statistics (IQM, stratified bootstrap, performance profiles), paired comparisons on identical scenarios, number of seeds, pre-registered held-out splits, leave-one-class-out for new airframe types, how robotics and aerospace papers report tracking error, recovery height loss and loss of control, and what makes an A/B result credible to aerospace reviewers (Monte Carlo campaigns, worst-case search, delay margins, handling-qualities criteria scaled for small UAVs). Repo read (no simulations run): src/aeroadapt/bench/stats.py, src/aeroadapt/bench/splits.py, src/aeroadapt/bench/run.py, configs/bench/b0.yaml, docs/03 §4.2/§5/§7.2, docs/05 §4.2. Web search budget ran out before a few figures could be checked; those are marked unconfirmed.

## Key findings (20 of 21 confirmed in the session)

### 6.1 · confirmed

A handful of runs gives unreliable point estimates. Report interval estimates instead: stratified-bootstrap CIs, the interquartile mean (IQM), performance profiles and probability of improvement. All of these are in the rliable library.

- *Evidence:* IQM is the mean of the middle 50% of the pooled run x task scores. It resists outliers but is statistically more efficient than the median. The stratified bootstrap resamples runs with replacement within each task, and is meant for small run counts. Performance profiles show the empirical tail distribution of scores over all runs and tasks, with 95% stratified-bootstrap bands. The area under a profile equals the mean, and where it crosses y=0.5 is the median. The README example uses probability_of_improvement with 2,000 bootstrap repetitions. The paper's re-analysis of published Atari 100k claims, and its NeurIPS 2021 outstanding-paper award, I recall from memory.
- *Source:* Agarwal, Schwarzer, Castro, Courville, Bellemare, 'Deep RL at the Edge of the Statistical Precipice', NeurIPS 2021, https://arxiv.org/abs/2108.13264 ; rliable https://github.com/google-research/rliable ; Google Research blog https://research.google/blog/rliable-towards-reliable-evaluation-reporting-in-reinforcement-learning/

### 6.2 · confirmed

Variance across seeds is big enough that comparisons need many seeds and proper significance tests, not mean plus or minus standard error from a few runs.

- *Evidence:* Henderson et al. ran 5 seeds per configuration (the same seed set across experiments), plotted mean and standard error, tried 2-sample t-tests, Kolmogorov-Smirnov tests and the bootstrap, and warned that these tests may not suit RL. From memory, unconfirmed here: they showed two disjoint groups of 5 seeds of the same algorithm producing visibly different learning curves.
- *Source:* Henderson et al., 'Deep Reinforcement Learning that Matters', AAAI 2018, https://arxiv.org/abs/1709.06560

### 6.3 · confirmed

Choose the seed count from a power analysis. Welch's t-test (no equal-variance assumption) is the recommended default, and bootstrap and other tests can exceed their nominal false-positive rate at small N.

- *Evidence:* Colas et al. compared tests empirically on false-positive rate and power as a function of seed count and effect size, including violated normality and equal-variance assumptions. Code: flowersteam/rl-difference-testing. A Sigaud lecture deck citing Colas 2019 recommends Welch and at least 15 seeds where possible; that is guidance from the slides, not a verified number from the paper. Standard power arithmetic (mine): a two-sided Welch/t test at alpha=0.05 with power 0.8 needs about 17 seeds per arm for Cohen's d=1, about 8 for d=1.5 and about 6 for d=2.
- *Source:* Colas, Sigaud, Oudeyer, 'How Many Random Seeds?', arXiv 2018, https://arxiv.org/abs/1806.08295 ; 'A Hitchhiker's Guide to Statistical Comparisons of RL Algorithms', ICLR 2019 RML workshop, https://arxiv.org/abs/1904.06979 ; slides https://master-dac.isir.upmc.fr/arob/cours/stats.pdf

### 6.4 · confirmed

Empirical-design guidance for RL: state each metric's statistical assumptions, characterise variation across runs, treat multi-agent comparisons specially, build fair baselines, and control hyperparameter tuning and experimenter bias. Jordan et al. add performance percentiles, which carry uncertainty (tuning included) into the aggregate score.

- *Evidence:* Patterson, Neumann, White and White, JMLR 25(318):1-63, 2024. Its topics include hypothesis testing, comparing multiple agents, building baselines, hyperparameters and experimenter bias. Jordan et al. (ICML 2020) propose performance percentiles, a weighted average of each algorithm's CDF, plus performance bound propagation and a game-theoretic aggregate.
- *Source:* Patterson et al., 'Empirical Design in Reinforcement Learning', JMLR 2024, https://jmlr.org/papers/v25/23-0183.html ; Jordan et al., 'Evaluating the Performance of RL Algorithms', ICML 2020, https://arxiv.org/abs/2006.16958

### 6.5 · confirmed

Nested data (scenarios within airframes within classes, policies within seeds) analysed as if independent gives badly anti-conservative intervals. A hierarchical bootstrap that resamples level by level fixes this.

- *Evidence:* Saravanan et al. report that traditional tests on nested data can exceed a 45% false-positive rate at a nominal 5% alpha. The hierarchical bootstrap stays within the nominal rate and keeps more power than averaging up to the top level. It performed comparably to linear mixed models in simulation and better on two real datasets.
- *Source:* Saravanan, Berman, Sober, 'Application of the hierarchical bootstrap to multi-level data in neuroscience', NBDT 3(5), 2020, https://arxiv.org/abs/2007.07797

### 6.6 · confirmed

The right tool for a paired binary outcome on identical scenarios (survived or not under controller A and controller B) is McNemar's test on the discordant pairs plus a paired-difference CI such as Newcombe's Wilson-based method. Simple Wald and 'exact' intervals perform poorly.

- *Evidence:* Newcombe calls earlier methods for paired proportions 'highly unsatisfactory'. His interval combines Wilson intervals on the discordant proportions with a correlation adjustment. It is implemented in R's biostatUZH::confIntPairedProportion. Worked example (my arithmetic): 60 scenarios where only L1 survives versus 30 where only B1 survives gives McNemar chi-squared = (60-30)^2/90 = 10, p of about 0.002. Power depends only on the discordant count, not on total n.
- *Source:* Newcombe, 'Improved confidence intervals for the difference between binomial proportions based on paired data', Statistics in Medicine 17:2635-2650, 1998, https://www.site.uottawa.ca/~nat/Courses/csi5388/Newcombe.1998.pdf

### 6.7 · confirmed

What the repo has now: Wilson intervals for rates, a percentile bootstrap for means, and an airframe split of 80/10/10 by hash of the airframe ID. B1 and B2 share start state and noise, and run.py counts discordant scenarios (b2_vs_b1) but runs no paired test and computes no paired CI. Loss of control is |roll|>90 deg or |pitch|>60 deg. Flying wings have no procedural members, only X8 and Zagi. The b0 test pool comes from a V20 acceptance run on 1,000 airframes, so it holds roughly 100 test-split airframes.

- *Evidence:* From the code. stats.py has wilson() and bootstrap_mean(). splits.py: split_of() puts sha256 mod 10 = 9 in test and 8 in val. run.py: LOC_ROLL_DEG=90, LOC_PITCH_DEG=60, and summary()['b2_vs_b1'] holds only counts. b0.yaml: 'fleet-accept m2-fleet --n 1000', split: test, n: 1000. docs/05 §4.2: 'Flying wing ... Held out entirely: only X8 and Zagi as references'. At about 85% acceptance, b0's 1,000 scenarios sit on roughly 85 distinct airframes, about 12 scenarios each (my estimate, not checked by running anything).
- *Source:* Repo: src/aeroadapt/bench/stats.py, src/aeroadapt/bench/splits.py, src/aeroadapt/bench/run.py, configs/bench/b0.yaml, docs/05_m2_technical_spec.md

### 6.8 · confirmed

Aerospace Monte Carlo verification phrases a requirement as 'met in at least y% of cases with stated consumer risk' and derives the number of runs from order or binomial statistics. Aerospace reviewers recognise this framing.

- *Evidence:* The NASA MSFC framing is: 'show that x is met for at least y% of cases, with 10% consumer risk', where consumer risk is the chance a finite-sample verification passes when the requirement is not actually met. NASA/TP-2010-216447 has an appendix deriving the run count. My arithmetic: zero failures in n runs demonstrates 99.73% at 90% confidence when n >= ln(0.10)/ln(0.9973), about 852. 'LOC rate < 1% at 95%' with zero failures needs n >= 299 (rule of three).
- *Source:* Hanson & Beard, 'Applying Monte Carlo Simulation to Launch Vehicle Design and Requirements Verification', AIAA GNC 2010-8433 / J. Spacecraft & Rockets 2012 ; NASA/TP-2010-216447, https://ntrs.nasa.gov/citations/20100038453

### 6.9 · confirmed

Flight-control-law clearance searches for the worst case as well as averaging. Grid or random sampling can miss interior worst cases, and the hard part is asserting that no weakness exists. The black-box safety-validation literature formalises falsification, most-likely failure and failure-probability estimation (importance sampling).

- *Evidence:* GARTEUR FM(AG11), 1999-2002, compared mu-analysis, nu-gap, polynomial, bifurcation/continuation and optimisation-based clearance; the results are in Springer LNCIS 283. Menon, Bates and Postlethwaite used differential evolution plus local search because worst-case parameter combinations need not lie at the extremes. Corso et al. (JAIR 72:377-428, 2021) survey optimisation, path planning, RL and importance-sampling methods for these tasks, with aircraft collision-avoidance case studies.
- *Source:* Fielding, Varga, Bennani, Selier (eds), 'Advanced Techniques for Clearance of Flight Control Laws', Springer LNCIS 283, 2002, https://link.springer.com/doi/10.1007/3-540-45864-6_21 ; Varga chapter https://elib.dlr.de/11884/1/varga_lncis02b.pdf ; Corso et al., JAIR 2021, https://jair.org/index.php/jair/article/view/12716

### 6.10 · confirmed

For adaptive and nonlinear controllers, aerospace practice uses time-delay margin as the accepted robustness metric in place of phase margin. Monte Carlo gives an upper bound on it and analysis a lower bound. When the papers found were written, no adaptive flight-control system had been FAA-certified.

- *Evidence:* Dorobantu, Seiler and Balas (JGCD 2012) compute sum-of-squares lower bounds and Monte Carlo upper bounds on time-delay margin. NASA work describes brute-force Monte Carlo as the only V&V technique for some adaptive designs. L1 adaptive control claims a time-delay margin bounded away from zero and was flight-tested on AirSTAR GTM. Separately, a NASA study fitted 7 MRAC variants into an identical dynamic-inversion architecture, tuned all of them with one metrics-based method, then tested sensitivity to time delay, flight condition, model uncertainty and cross-coupling. That is a template for a fair comparison.
- *Source:* Dorobantu, Seiler, Balas, 'Time-Delay Margin Analysis for Adaptive Control', JGCD 2012, https://dept.aem.umn.edu/~SeilerControl/Papers/2012/DorobantuEtAl_12JGCD_TimeDelayMarginAnalysisForAdaptiveControl.pdf ; Campbell, Kaneshige, Nguyen, Krishnakumar, 'Implementation and Evaluation of Multiple Adaptive Control Technologies for a Generic Transport Aircraft Simulation', AIAA Infotech 2010 (NTRS 20100034875)

### 6.11 · confirmed

AdaptiveQuadBench, the multirotor analogue of this benchmark, standardises a 'when2fail' stress test (raise a disturbance until a success threshold breaks), delay-margin analysis and position RMSE across MPC, L1-MPC, geometric INDI and L1 controllers.

- *Evidence:* Built on RotorPy, with disturbances covering wind, payload shifts, rotor faults and control latency; code is public. Secondary summaries (not verified against the paper) say L1-MPC and INDI compensate actuator faults within 0.1 s.
- *Source:* Zhang, Tao, Cheng, Hovakimyan, Mueller, 'A Simulation Evaluation Suite for Robust Adaptive Quadcopter Control', arXiv 2510.03471, 2025, https://arxiv.org/abs/2510.03471

### 6.12 · confirmed

Credible learned-adaptive-control papers report per-condition tracking error against strong adaptive baselines, INDI and L1 included, not only against PID.

- *Evidence:* Neural-Fly reports average tracking-error reductions of 66% versus a nonlinear tracking controller, 42% versus L1 adaptive and 35% versus INDI, in winds up to 12.1 m/s in the Caltech Real Weather Wind Tunnel. Its ablations include transfer from a different drone and a constant-adaptation variant, which performed like L1 and INDI.
- *Source:* O'Connell et al., 'Neural-Fly enables rapid learning for agile flight in strong winds', Science Robotics 2022, doi 10.1126/scirobotics.abm6597, https://arxiv.org/abs/2205.06908

### 6.13 · confirmed

RAPTOR, the pattern behind L1's teacher-to-student distillation, evaluated one tiny recurrent policy zero-shot on 10 real held-out quadrotors. It showed breadth well, but a third-party review found its quantitative benchmark reporting unclear. Separately, Zhang et al. report generalisation as an extrapolation factor beyond the training range.

- *Evidence:* RAPTOR: 2,084 parameters, 3 layers, one recurrent hidden layer. Meta-imitation learning: 1,000 sampled quadrotors, an RL teacher for each, distilled into one student. Tested on 10 real quadrotors from 32 g to 2.4 kg, including tracking, wind and poking. Zhang et al.: in simulation, an adaptation range up to 16x wider than training; on hardware, about 3.7x mass and more than 100x propeller-constant variation, plus motor failures and off-centre payloads.
- *Source:* Eschmann, Albani, Loianno, 'RAPTOR: A Foundation Policy for Quadrotor Control', arXiv 2509.11481, 2025, https://arxiv.org/abs/2509.11481 ; Zhang, Loquercio et al., 'A Learning-based Quadcopter Controller with Extreme Adaptation', IEEE T-RO 2025, https://arxiv.org/abs/2409.12949

### 6.14 · confirmed

Papers on generalising across robot bodies (embodiment generalisation) usually show zero-shot transfer to a few held-out morphologies, without class-level hold-outs or confidence intervals. A leave-one-class-out evaluation with CIs would be above the field's usual bar.

- *Evidence:* SMP (ICML 2020): modular shared policies, zero-shot to unseen planar morphologies. GenLoco (CoRL 2022, PMLR 205:1893-1903): morphology randomisation, transfer to novel simulated and real quadrupeds. URMA (CoRL 2024): trained on 16 robots, zero-shot to 2 simulated and 3 real robots; URMAv2 scales to 50. The judgement that the field's bar is low is mine, from these abstracts.
- *Source:* Huang, Mordatch, Pathak, ICML 2020, https://proceedings.mlr.press/v119/huang20d.html ; Feng et al., GenLoco, CoRL 2022, https://proceedings.mlr.press/v205/feng23a.html ; Bohlinger et al., 'One Policy to Run Them All', CoRL 2024, https://arxiv.org/abs/2409.06366

### 6.15 · confirmed

In the fixed-wing RL precedents, PPO attitude control matches tuned PID while converging in more cases, and fault-tolerance papers report normalised mean absolute error (nMAE) per tracked channel and per failure case. Typical improvements are small.

- *Evidence:* Bøhn et al. (ICUAS 2019): PPO roll/pitch/airspeed control converged in more cases than PID with comparable performance and generalised to unseen severe wind and turbulence; a restricted observation plus several past timesteps mattered most. Dally & van Kampen (SciTech 2022): SAC on a Cessna Citation 500 model identified from flight data, robust to biased sensor noise, disturbances and varied initial conditions. Teirlinck & van Kampen (SciTech 2024): the SAC-IDHP hybrid improves nMAE over SAC by 0.74% nominal, 5.46% longitudinal-failure and 0.82% lateral-failure.
- *Source:* Bøhn, Coates, Moe, Johansen, ICUAS 2019, https://arxiv.org/abs/1911.05478 ; Dally & van Kampen, AIAA SciTech 2022, https://arxiv.org/abs/2202.09262 ; Teirlinck & van Kampen, AIAA SciTech 2024 (TU Delft repository)

### 6.16 · confirmed

RL upset-recovery papers optimise and report altitude loss, angular-rate arrest and angle of attack against scripted or sliding-mode baselines. There is no standard altitude-loss benchmark.

- *Evidence:* Grillo, Torre, Bunge (SciTech 2024): PPO stall recovery that minimises total altitude loss; delaying full throttle by 0.75 s raised altitude loss by up to 30%. Kim, Oh, Seo, Kim (JGCD 40(4):1074-1081, 2017): RL flat-spin recovery for a UAV, reportedly with separate rate-recovery and altitude-recovery models (unverified). Sayyed et al. (arXiv 2601.06439): PPO spin recovery on F-18/HARV against sliding-mode control, 6,000 episodes, generalising to unseen upset initialisations. Demir et al. (arXiv 2604.24355): SAC for PARS (pilot-activated recovery).
- *Source:* Grillo et al., 'Optimal Stall Recovery via Deep RL for a General Aviation Aircraft', AIAA SciTech 2024, https://gabo-tor.github.io/publication/grillo-optimal-2024 ; Kim et al., JGCD 2017, doi 10.2514/1.G001739 ; https://arxiv.org/abs/2601.06439 ; https://www.alphaxiv.org/abs/2604.24355

### 6.17 · **not confirmed** (memory or secondary)

Loss of control (LOC) has a quantitative, multi-envelope definition: Wilborn and Foster's 'quantitative loss-of-control' (QLC) metrics, envelopes in two-dimensional parameter spaces. FAA AC 120-111 defines an upset as unintentionally exceeding the parameters normally seen in line operations.

- *Evidence:* Confirmed: QLC metrics are 2-D envelopes, used by NASA to define safe envelopes and to time LOC onset and severity. AC 120-111 has been active since 14 Apr 2015 and leaves stalls to AC 120-109. From memory, unconfirmed: there are five envelopes (alpha-beta aerodynamics, bank-pitch attitude, airspeed-load-factor structural, dynamic pitch control, dynamic roll control), LOC is declared when 3 or more are exceeded, and the commonly cited upset thresholds are pitch >25 deg nose-up, >10 deg nose-down, or bank >45 deg.
- *Source:* Wilborn & Foster, 'Defining Commercial Transport Loss-of-Control: A Quantitative Approach', AIAA 2004-4811 (cited in NASA https://ntrs.nasa.gov/api/citations/20090030517/downloads/20090030517.pdf) ; FAA AC 120-111, https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-111.pdf

### 6.18 · confirmed

MIL-F-8785C and MIL-HDBK-1797 handling-qualities limits do not apply unscaled to small UAVs. Small airframes have higher modal frequencies, and either dynamically scaled or Froude-scaled criteria are needed.

- *Evidence:* Foster (BYU, 2004): small UAVs fell outside the MIL-F-8785C short-period frequency range even though test pilots rated them acceptable. He proposed a dynamically scaled short-period criterion and checked it by flight-testing 3 small UAVs to the point of poor handling. Ivler et al. (VFS 2022/2024) Froude-scale UAS test results to full scale for comparison with ADS-33E / MIL-DTL-32742 criteria, and Froude-scaled disturbance-rejection-bandwidth criteria predicted Level 1 in flight. Not confirmed in this session: the numerical Level boundaries (for example roll-mode time constant and equivalent time delay of 0.10/0.20/0.25 s for Levels 1/2/3, from memory).
- *Source:* Foster, 'Dynamic Stability and Handling Qualities of Small Unmanned-Aerial-Vehicles', MSc thesis, BYU 2004, https://scholarsarchive.byu.edu/etd/219 ; Ivler et al., VFS 2022, https://gcp-web.sjsu.edu/researchfoundation/docs/VFS_2022_Ivler.PDF

### 6.19 · confirmed

RL controllers need actuator-smoothness reporting. CAPS defines an FFT-based smoothness metric Sm, and in hardware tests smooth policies drew far less power.

- *Evidence:* Sm is the amplitude-weighted mean normalised frequency of the control signal; the authors note peak frequency alone is a poor measure. In quadrotor flight tests the CAPS agent drew 4.86 +/- 2.32 A versus 22.87 A for Neuroflight.
- *Source:* Mysore, Mabsout, Mancuso, Saenko, 'Regularizing Action Policies for Smooth Control with RL', ICRA 2021, https://arxiv.org/abs/2012.06644

### 6.20 · confirmed

Regulators frame machine-learning assurance around a defined Operational Domain, completeness and representativeness of the data, and independent test data. Pre-registered splits map naturally onto this. In ML, pre-registration (protocol reviewed before results exist) was piloted at NeurIPS.

- *Evidence:* The EASA AI Concept Paper Issue 2 (2024) covers Level 1 and 2 machine learning, a W-shaped learning-assurance process, OD at system level, and ODD at the AI/ML-constituent level. NeurIPS 2020 and 2021 workshops reviewed papers on their experimental protocol before results; a later paper notes little uptake.
- *Source:* EASA AI Concept Paper Issue 2, https://www.easa.europa.eu/en/document-library/general-publications/easa-artificial-intelligence-concept-paper-issue-2 ; NeurIPS 2020 pre-registration workshop https://neurips.cc/virtual/2020/workshop/16158 ; 2021 https://neurips.cc/virtual/2021/workshop/21885

### 6.21 · confirmed

NeuralPlane, the closest fixed-wing RL benchmark, focuses on GPU-parallel simulation and tooling and has no fault or held-out-airframe protocol. That leaves room for a statistically rigorous fixed-wing generalisation benchmark.

- *Evidence:* NeurIPS 2024 Datasets & Benchmarks: about 0.2 s per step for 10^6 parallel aircraft, with code templates, evaluation and visualisation tools, and hierarchical RL plus classical frameworks. The search did not surface its evaluation protocol.
- *Source:* Xue et al., 'NeuralPlane', NeurIPS 2024 D&B, https://proceedings.neurips.cc/paper_files/paper/2024/hash/af97b61e9fef8f55e32a2602af364d8c-Abstract.html

## Recommendations for M3

1. Pre-register before the first test-split run. Write docs/09_m3_eval_prereg.md, commit it and git-tag it (signed). It should list: 4 primary hypotheses with endpoints, exact airframe-ID lists and hashes per split, scenario seeds, n per split, the analysis plan, the checkpoint-selection rule and the success thresholds. Suggested primaries: P1, survival without unrecovered LOC on G2 fault scenarios, L1+allocation vs B1, paired superiority. P2, the same on G3. P3, Rascal-in-JSBSim (G4) tracking nMAE, non-inferior to B1 on nominal flights (declared margin, e.g. +1 deg RMS roll) and superior under actuator faults. P4, L2 recovery success and height loss vs the scripted recovery on a fixed upset grid. Apply Holm-Bonferroni across P1-P4 and label everything else exploratory. Keep a test-access ledger (commit, policy-weights hash, date) of every G2/G3/G4 run.
2. Report two uncertainty statements and keep them separate. (a) 'This frozen policy': CIs over scenarios, cluster-bootstrapped by airframe. This is what the 'same frozen policy flies six airframes' headline needs. (b) 'This training recipe': CIs over training seeds, using rliable IQM, a stratified bootstrap over seeds within strata (fault category x airframe class), performance profiles of per-stratum survival rates, and probability of improvement. Use 10 seeds for the main L1 recipe (cheap in JAX on rented GPUs), which detects roughly Cohen's d of 1.3 at power 0.8, and at least 5 for L2. Add a Welch t-test or an exact permutation test over seeds alongside the bootstrap, since the bootstrap under-covers at small N.
3. Extend the paired design that B1/B2 already use (shared start state and noise) to B1, INDI+allocation, L1+allocation, scripted recovery and L2, so every controller flies the identical scenario list. Key turbulence and sensor-noise streams by (scenario, simulated time), not by step index or controller call count, so 50 Hz and 100 Hz controllers see the same disturbances. Add to src/aeroadapt/bench/stats.py: mcnemar_exact (binomial on discordant pairs), newcombe_paired_ci, cluster_bootstrap (resample airframes, then scenarios within each), and paired-difference bootstrap plus Wilcoxon signed-rank or Hodges-Lehmann for continuous metrics. Turn summary()['b2_vs_b1'] into a generic pairwise table with effect size, CI and Holm-adjusted p.
4. Fix the effective sample size of G2. Run V20 acceptance on the whole 10k fleet so the test split holds about 850 accepted airframes instead of about 85. Then sample 2,000-5,000 scenarios with at most 3-5 per airframe and report the design effect (1+(m-1)\*rho). Reference Wilson half-widths at p=0.85: +/-2.2 pp at n=1,000, +/-1.0 pp at n=5,000, +/-5 pp at n=200. Evaluation compute is small: 300 s at 100 Hz is 30k steps a flight, so 2,000 scenarios x 5 controllers is about 3e8 steps, minutes to an hour on CPU depending on stack and policy cost. Airframe count and test-set hygiene are the constraint, not compute.
5. Add an extrapolation split, G2-x. The hash split only tests interpolation, so hold out the top and bottom 5-10% by mass, span, wing loading and inertia ratios, and train without them. Also plot survival and tracking error against distance to the nearest training airframe in a non-dimensional dynamics-descriptor space (Cl_p, Cm_alpha, Cn_beta, control power, roll-mode time constant, short-period frequency). Report it as 'x times outside the training range', as Zhang et al. (2024) do.
6. Make G3 a real class-level test. It currently has 1-2 flying wings (X8, with Zagi planned). Generate at least 100 test-only procedural flying wings with corrected directional derivatives (docs/05 notes that the buildup method under-predicts Cn_beta, Cn_r and CY_beta) and an elevon-only effector set routed through the allocator. Report X8 and Zagi separately as reference models. Until then, phrase G3 as 'two case studies', not class generalisation.
7. Rotate leave-one-class-out (LOCO) across the five training classes: 5 extra L1 trainings with 3 seeds each, about 15 runs, within the approved GPU budget. For each held-out class, report the generalisation gap: performance on class C when trained without C, minus performance on C when trained with all classes. This controls for how hard each class is intrinsically. Normalise by B1's per-class survival. Expect T-tail or V-tail to transfer almost free from conventional, so also report the gap against the dynamics-descriptor distance, not only class labels.
8. Separate the simulator gap from the airframe gap on G4 with a 2x2: Rascal in aerojax vs Rascal in JSBSim, and a training-fleet airframe in aerojax vs the same airframe in JSBSim if one can be exported. Rascal is n=1 airframe, so present scenario-level CIs as a case study and make clear the stock-baseline numbers are ArduPlane-equivalent, not real ArduPlane SITL, until the M6 SITL check.
9. Use one fixed set of metrics for every controller. Tracking: RMS and nMAE per channel (roll, pitch, airspeed, flight-path angle or altitude) on a fixed step and doublet battery, plus rise time, 5% settling time and overshoot. Actuators: RMS surface rate, % time at rate or position saturation, CAPS Sm smoothness, and motor energy. Robustness: an empirical time-delay margin (sweep injected delay past the training range until LOC or limit-cycle, in AdaptiveQuadBench when2fail style), a control-effectiveness gain sweep, a noise sweep, and local gain and phase margins from JAX linearisation of the closed loop (plant + GRU state + allocator) at trim across the fleet. Handling qualities: fit low-order equivalent systems (LOES) to closed-loop roll and pitch responses and report short-period frequency and damping, roll-mode time constant and equivalent delay, compared descriptively with Froude- or dynamically-scaled MIL-F-8785C / MIL-HDBK-1797 boundaries. Do not claim pass/fail against unscaled limits.
10. Define loss of control in three tiers. (1) Upset entered: AC 120-111-style attitude and airspeed thresholds. (2) LOC: the existing |roll|>90 / |pitch|>60 rule, plus a Wilborn-Foster multi-envelope QLC check as a sensitivity analysis. (3) Unrecovered LOC: ground contact, or no recovery within T seconds or H metres. For L2, define recovery precisely before testing (for example |phi|<10 deg, |p|,|q|,|r|<10 deg/s, gamma>=0, 1.1 Vs < V < Vne, held for 2 s). Report success rate, height loss from fault onset and from L1-to-L2 hand-over, time to recover, peak n_z, peak alpha, peak airspeed, secondary-upset rate and hand-back success.
11. Report height loss as a 'recovery profile', not as a mean over survivors: the fraction of upsets recovered within h metres, for h from 0 to 400 m, with failures never reaching the curve and 95% bands. This avoids survivorship bias and is the per-scenario analogue of an rliable performance profile. Add an oracle-normalised height loss: for a subset of about 100 upsets, solve a minimum-height-loss recovery by trajectory optimisation with privileged dynamics in JAX and report the RL-to-oracle ratio.
12. Go beyond random Monte Carlo. Give every controller the same falsification budget (CMA-ES or cross-entropy importance sampling over fault onset time, severity, wind, gust phase and initial state), report the worst cases found, and estimate the importance-sampled failure probability in the rare tail (Corso et al. 2021; GARTEUR clearance). State the Monte Carlo claim in the aerospace form, for example 'unrecovered LOC <= 1% at 95% confidence'; zero failures in 299 scenarios meets that.
13. Keep the baselines at parity and document it. Autotune B1 per airframe on test airframes too (this favours the baseline; say so). Give INDI the same online effectiveness estimator and the same information as L1's allocator, with no privileged truth for anyone at test time. Give INDI gains and L1 hyperparameters an equal tuning budget, logged (Patterson et al. 2024). Tune the scripted recovery on the training fleet only.
14. Select checkpoints on G1-val only, by the pre-declared rule (for example final checkpoint, or median seed by val score, never the best seed on test). Publish the weights hash before any test evaluation, report every seed, and state where the demo policy ranks among them.
15. Choose the six headline airframes by a pre-declared hash draw before evaluation (for example 3 from G2, 2 flying wings, and Rascal in JSBSim), not by curating. Show per-airframe tracking and recovery numbers with CIs, including the hard-over recovery height loss on each, next to the aggregate G2/G3/G4 tables.
16. Build the report page from the pre-registered tables: n_scenarios, n_airframes and n_seeds per split; rates with Wilson and cluster-bootstrap CIs; paired differences with Newcombe CIs and Holm-adjusted McNemar p-values; rliable plots; recovery profiles; delay-margin curves; a failure gallery linked to replays; and raw parquet. One command should reproduce every number.

## Risks and pitfalls

1. Test-set reuse. Iterating on L1 while repeatedly checking G2, G3 or G4 turns them into validation sets (the garden of forking paths). Without a frozen pre-registration and an access ledger, aerospace and ML reviewers will discount every number.
2. Overclaiming 'unseen airframes'. The sha256 80/10/10 split draws test airframes from the same generator distribution, so it measures interpolation. Extrapolation needs G2-x and the distance-based curve.
3. G3 statistics. One or two flying wings cannot support a class-level claim. A scenario-level CI conditional on X8 says nothing about flying wings in general. X8 results are also confounded by model provenance: published coefficients, not the generator's buildup aero.
4. Clustering ignored. About 12 scenarios per airframe on about 85 airframes, treated as 1,000 independent trials, gives Wilson CIs that are too narrow; nested-data analyses can exceed 45% false positives (Saravanan 2020). Cluster by airframe, and by seed for recipe claims.
5. Survivorship bias. Height loss, time-to-recover or tracking RMS computed only over surviving flights can make a controller that crashes the hard cases look better. Use recovery profiles, rank-based paired tests with failures ranked worst, or report survival and conditional metrics together.
6. Misapplying rliable. IQM and performance profiles are defined over run x task scores. Applying IQM to individual binary scenario outcomes is meaningless. Bootstrap CIs over 3-5 seeds under-cover. A single-seed policy supports no 'method' claim.
7. Common random numbers silently breaking. If noise is indexed by step count or controller call count, a 50 Hz policy and a 100 Hz stack see different gusts, and the paired comparison loses power or becomes biased.
8. Multiple comparisons. 26 fault kinds x 5 classes x several splits x 4-5 controllers produce hundreds of contrasts, so some will be 'significant' by chance unless primaries are pre-declared and the rest labelled exploratory.
9. Baseline strength. An under-tuned INDI, or an INDI denied the effectiveness estimate L1's allocator gets, makes the A/B unconvincing. Conversely, B1's per-airframe autotune on test airframes is an advantage that should be stated, not hidden.
10. Delay margin made true by construction. If the injected-delay sweep stays inside the training-time delay randomisation range, the 'margin' only measures the training distribution. Sweep beyond it and report the breaking point.
11. RL exploiting simulator artefacts: bang-bang or chattering surfaces, post-stall aero regions with weak modelling, missing actuator rate or bandwidth limits. Without CAPS-style smoothness and actuator-rate metrics, and the JSBSim cross-check, a sim result may not survive contact with hardware.
12. Gaming the loss-of-control definition. A policy can avoid the 90/60 attitude thresholds while flying erratically or overstressing the airframe. Add load-factor, alpha and airspeed envelopes (QLC-style) as a sensitivity check.
13. Handling-qualities misuse. MIL-F-8785C / MIL-HDBK-1797 Level boundaries were written for piloted full-scale aircraft. Applied unscaled to a 1-15 kg UAV, they mislabel good closed-loop responses (Foster 2004). Present them as scaled, descriptive equivalent-system parameters.
14. A curated headline. Picking the six demo airframes after seeing results, or demoing the best seed, would be found out by any careful reviewer. Use a pre-declared hash draw and the median or pre-declared seed.
15. LOCO compute and schedule. 5 folds x 3 seeds is about 15 extra PPO runs on top of 10 headline seeds. Budget the GPU time early, or cut LOCO to 2-3 informative folds (glider, twin-boom), as a pre-declared plan.
16. Fidelity framing. Every result is sim-to-sim: aerojax to JSBSim. Per the project's own fidelity rule, do not call B1 'ArduPlane' until the SITL subset check exists, and state plainly that no hardware results exist.

## Open questions

1. Is the M3 headline about the frozen policy or about the training recipe? This decides whether CIs run over scenarios (cluster bootstrap by airframe) or over seeds (rliable), and both are probably needed.
2. Will procedural flying wings be generated for G3, and is the aero buildup good enough for flying-wing directional dynamics (Cn_beta, Cn_r) for them to count as test airframes? Or does G3 stay X8 + Zagi, phrased as case studies?
3. How many distinct accepted G2 test airframes exist after V20? The b0 config suggests about 85 from a 1,000-airframe acceptance run. Is a full 10k V20 run affordable once the long CPU job finishes?
4. What is the exact, pre-declared L2 recovery criterion (attitude, rate, flight-path and airspeed thresholds and dwell time), and does height loss count from fault onset or from L1-to-L2 hand-over?
5. Does INDI+allocation get the same online effectiveness estimate as L1's allocator, the true effectiveness (an oracle variant), or a geometry-derived model? Each needs declaring, and possibly reporting more than one.
6. How large is seed-to-seed variance of distilled GRU students in this setting? A 3-seed pilot on G1-val would set the seed count for the confirmatory runs (10 if d is about 1.3, fewer if larger).
7. Should the M3 report include a falsification or worst-case search with equal budget per controller, or leave that for M6? It is the strongest answer to the aerospace 'averages hide tails' critique.
8. Not confirmed in this session (the web search budget ran out): the numerical MIL-F-8785C / MIL-HDBK-1797 Level boundaries (roll-mode time constant, Dutch-roll damping, equivalent time delay of 0.10/0.20/0.25 s); the AC 120-111 numeric upset thresholds (25/10/45 deg); Wilborn-Foster's five envelopes and the 3-of-5 LOC rule; MIL-F-9490D's 6 dB / 45 deg stability margins; Colas et al.'s exact Welch false-positive rates; and Chan et al. 2020 ('Measuring the Reliability of RL Algorithms', ICLR 2020; CVaR across runs and dispersion metrics), recalled but not checked. Verify these against the primary documents before quoting them in docs or the report.
