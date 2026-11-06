# Report 4: Control allocation with actuator faults

*Part of the M3 literature review, 6 Nov 2026 ([README](README.md)). The brief and the findings are as the research agent returned them; local paths are made repository-relative. Where this report and the spec differ, [docs/09](../../09_m3_technical_spec.md) holds.*

## Brief

Control allocation for fixed-wing UAVs with actuator faults (WLS, redistributed pseudo-inverse, active set; online effectiveness estimation; jammed, floating and reversed surfaces; rate and position limits) and how a learned virtual-moment policy should interact with an allocator, sized for a JAX implementation at 100 Hz in aero-adapt M3

## Key findings (20 of 20 confirmed in the session)

### 4.1 · confirmed

REPO (blocking for the M3 plan): aerojax has only three surface channels (one aileron, one elevator, one rudder) plus throttle. The effectiveness matrix B is therefore square (3x3, nearly diagonal), and WLS allocation reduces to inverting it. The plan in docs/03 §4.2 and Flight 0 (a jammed aileron 're-allocated to rudder and the other aileron') cannot happen in the simulator as built.

- *Evidence:* src/aeroadapt/sim/aerojax/actuators.py: ActuatorState.surfaces is (3,), commented '(aileron, elevator, rudder)'. src/aeroadapt/faultlab/slots.py: SURFACE_INDEX = {aileron:0, elevator:1, rudder:2}, and ServoJamFault fields are (3,). src/aeroadapt/airframes/params.py: max_deflection, rate_limit and deadband are (3,). No servo-position measurement exists in SensorArrays (core/messages.py), so jams and floats can only be inferred from IMU moment residuals. Actuator delay can be up to 150 ms (schema.py delay_s le=0.15). The physics rate is 200 Hz (32-step buffer = 160 ms).
- *Source:* aero-adapt repo, src/aeroadapt/sim/aerojax/actuators.py, faultlab/slots.py, airframes/params.py, airframes/schema.py, core/messages.py

### 4.2 · confirmed

REPO: in the held-out Rascal 110 (G4, JSBSim), only the left-aileron position enters the aerodynamics. Splitting the ailerons would leave the right one aerodynamically dead unless the derived model is rewired, and a symmetric aileron deflection has no modelled effect at all.

- *Evidence:* ~/.cache/aeroadapt/jsbsim/aircraft/Rascal/Rascal.xml: Clda = qbar*S*b*fcs/left-aileron-pos-rad*0.13 (Mach 0), Cnda = -0.03*left-aileron-pos-rad, Cldr = 0.01, Clb = -0.10, Clp = -0.40, Cnb = 0.12, Cndr = -0.05, Cmde = -0.5 (the file notes it 'was -1.1'). Span 9.17 ft (2.80 m). Our adapter (sim/jsbsim/rascal.py) writes right = -left from a single fcs/da-rad.
- *Source:* ArduPilot Tools/autotest/aircraft/Rascal/Rascal.xml at commit b032bcb (cached), and src/aeroadapt/sim/jsbsim/rascal.py

### 4.3 · confirmed

DERIVED (my calculation from the Rascal coefficients): a full-throw aileron hard-over is not recoverable on a single-aileron-channel conventional airframe. The rudder, acting through sideslip and dihedral, can cancel only about a quarter of the roll moment. A full elevator hard-over is likewise unrecoverable. A rudder hard-over is recoverable with the ailerons. This matters directly for the M3 headline 'including recovery from a hard-over'.

- *Evidence:* Quasi-steady yaw balance: Cnb*beta + Cndr*dr = 0, so beta/dr = 0.05/0.12 = 0.42. Rudder roll per rad = Cldr + Clb*0.42 = 0.01 - 0.042 = -0.032. Assuming equal 0.35 rad throws (the Rascal FCS normalises both with 0.35 rad), the rudder gives 0.011 against a full-aileron 0.13*0.35 = 0.0455, i.e. about 24%. The residual steady roll rate is p*b/2V = (0.0455-0.011)/0.40 = 0.086, so at V = 20 m/s and b = 2.8 m, p is about 1.2 rad/s (about 70 deg/s): a continuous roll. This is a linear, quasi-steady estimate and ignores lag and nonlinearity.
- *Source:* Own analysis using Rascal.xml coefficients (above)

### 4.4 · confirmed

Johansen & Fossen's survey sets the reference framing. Control allocation is the middle of a three-level hierarchy (motion controller, then allocation, then actuator control), so the high-level controller can be designed without detailed knowledge of the effectors. Optimisation-based allocation is the strong approach for position and rate saturation, operational constraints and secondary objectives. Methods split into linear and nonlinear-model classes.

- *Evidence:* Abstract and scope from indexed copies; about 860 citations.
- *Source:* T. A. Johansen, T. I. Fossen, 'Control allocation - A survey', Automatica 49(5):1087-1103, 2013, doi:10.1016/j.automatica.2013.01.035 (https://dblp.uni-trier.de/rec/journals/automatica/JohansenF13.html)

### 4.5 · confirmed

Härkegård's primal active-set algorithm for weighted-least-squares (WLS) allocation always reaches the exact optimum of the box-constrained least-squares problem, and runs in about the same time as the cascaded generalised inverse (CGI) and the fixed-point method. The slides' example uses 8 actuators and 3 moments, with position and rate limits. Petersen & Bodson later note that the WLS variant was the most efficient of Härkegård's active-set methods.

- *Evidence:* Abstract: 'always find the optimal control distribution'; 'timing requirements are in the same range as for two previously proposed solvers'. Slides: 8 actuators x 3 moments. The often-quoted typical count of 1-3 iterations when warm-started from the previous working set comes from my memory of the thesis and paper; I could not confirm it from the search results.
- *Source:* O. Härkegård, 'Efficient active set algorithms for solving constrained least squares problems in aircraft control allocation', Proc. 41st IEEE CDC, pp. 1295-1300, 2002 (https://core.ac.uk/works/21014643); J. Petersen, M. Bodson, 'Constrained quadratic programming techniques for control allocation', IEEE TCST 14(1), 2006 (https://pubweb.eng.utah.edu/~bodson/pdf/Constrained%20Quadratic%20Programming%20Techniques%20for%20Control%20Allocation.pdf)

### 4.6 · confirmed

Bodson's comparison found that full optimisation (direct allocation, mixed LP) cuts the moment error only modestly on average compared with cheap methods such as the redistributed pseudo-inverse and the fixed-point method, but sometimes by a lot. The large differences arise at saturation and in degraded configurations, which is exactly the fault regime aero-adapt targets.

- *Evidence:* Abstract: performance gains 'small on the average, but sometimes significant', judged by desired-vs-achieved moment error across aircraft models with different actuator counts.
- *Source:* M. Bodson, 'Evaluation of optimization methods for control allocation', JGCD 25(4):703-711, 2002, doi:10.2514/2.4937 (https://my.eng.utah.edu/~bodson/code/Evaluation%20of%20optimization%20methods.pdf)

### 4.7 · confirmed

Durham showed that no single generalised inverse (a pseudo-inverse, for example) can reach the whole attainable moment set (AMS) without violating a constraint somewhere. Direct allocation does reach the full AMS and preserves the direction of the demand, but it costs more computation.

- *Evidence:* Abstract/summary statements for Durham 1993; a NASA follow-up says it extends the Moore-Penrose reachable volume to the whole AMS.
- *Source:* W. C. Durham, 'Constrained control allocation', JGCD 16(4), 1993, doi:10.2514/3.21072; NASA NTRS 20200002585 (https://ntrs.nasa.gov/api/citations/20200002585/downloads/20200002585.pdf)

### 4.8 · confirmed

Paparazzi's INDI flies an active-set WLS allocator (wls_alloc). Its parameters are gamma_sq (the preference for meeting the demand over staying near the preferred u_p), imax (the iteration cap) and Wv, Wu. It returns the number of iterations used, or -1 on failure. Smeur et al. prioritise roll and pitch over thrust, and thrust over yaw, with example weights roll 1000, pitch 1000, thrust 100, yaw 1.

- *Evidence:* Generated API docs (v5.12 to v7.0) and the IMAV 2017 slides and abstract. I could not confirm the default gamma_sq or imax values.
- *Source:* Paparazzi UAS wls_alloc.c docs (https://docs.paparazziuav.org/v5.18/wls__alloc_8c.html); E. J. J. Smeur, D. C. Höppener, C. De Wagter, 'Prioritized control allocation for quadrotors subject to saturation', IMAV 2017, pp. 37-43 (https://www.imavs.org/2017/content/uploads/2017/10/imav_ewoud.pdf)

### 4.9 · confirmed

The worst-case iteration count is what matters for a hard real-time budget, and also for vmap batching, where a batch runs as long as its slowest member. Blaha's TU Delft thesis cut worst-case active-set timings by 65% on a hexarotor and flight-tested the result. An open-source Rust port of the WLS active-set solver (wls-alloc) exists and can serve as a reference.

- *Evidence:* Thesis abstract: 'worst-case timings for a 6-rotor multicopter UAV can be improved by 65% over a current Active-Set solver'; the docs.rs crate wls-alloc 0.3.1 has a solver.rs.
- *Source:* T. M. Blaha, 'Computationally efficient control allocation using active-set algorithms', MSc thesis, TU Delft, 2023 (https://repository.tudelft.nl/record/uuid:bffb47bf-5864-4b18-921b-588b3a664866); https://docs.rs/crate/wls-alloc/0.3.1/source/src/solver.rs

### 4.10 · confirmed

Härkegård's dynamic allocation puts rate limits inside the QP and adds a term penalising change from the previous command. Without active constraints this behaves like a first-order filter, u(t) = F u(t-T) + G v(t), with frequency characteristics chosen per actuator. Rate limits enter at each step as tightened box bounds.

- *Evidence:* Abstract: 'automatic redistribution of the control effort when one actuator saturates in position or in rate'; the filter form appears in the paper.
- *Source:* O. Härkegård, 'Dynamic control allocation using constrained quadratic programming', JGCD 27(6), 2004, doi:10.2514/1.11607

### 4.11 · confirmed

Adaptive allocation with unknown effector parameters can be done with Lyapunov-based dynamic update laws that track the optimal allocation over time, instead of solving the optimisation fresh at every step. The paper gives uniform global or local asymptotic stability results for cascaded set-stable systems.

- *Evidence:* Abstract of the Automatica paper.
- *Source:* J. Tjønnås, T. A. Johansen, 'Adaptive control allocation', Automatica 44:2754-2765, 2008 (https://www.sintef.no/en/publications/publication/360349)

### 4.12 · confirmed

Online estimation of control effectiveness inside INDI has been flown. Smeur et al. used LMS on micro air vehicles and handled the delays in deriving angular acceleration from gyro rates. Later TU Delft fixed-wing and flying-wing work switched to variable-forgetting-factor RLS (VFF-RLS), because it converges faster after faults and can raise its adaptation rate when residuals grow. One fixed-wing LMS study found that adverse CG shifts can break INDI's time-scale-separation assumption. In another study, adaptation alone (without allocation) sometimes destabilised the aircraft for severe failures.

- *Evidence:* Smeur 2016: LMS, convergence given enough excitation (stated in a follow-up paper). Atmaca & de Visser, SciTech 2026: VFF-RLS on the Flying-V, better than INDI and adaptive INDI under sudden actuator failures (simulation only). Smit, Pollack & van Kampen 2022: CG shift breaks time-scale separation. Higgs 2015 (Wits): allocation was needed for severe failures. The exact update gains and filter cut-offs used in these papers are not confirmed.
- *Source:* E. J. J. Smeur, Q. P. Chu, G. C. H. E. de Croon, 'Adaptive incremental nonlinear dynamic inversion for attitude control of micro air vehicles', JGCD 39(3), 2016 / AIAA GNC 2016, doi:10.2514/6.2016-1390 (https://research.tudelft.nl/en/publications/adaptive-incremental-nonlinear-dynamic-inversion-for-attitude-con); D. Atmaca, C. C. de Visser, AA-INDI, AIAA SciTech 2026, doi:10.2514/6.2026-1743 (https://repository.tudelft.nl/record/uuid:0101b2d8-19bd-4f86-a5dd-aceaf36cfd04); Higgs, Wits MSc 2015 (https://wiredspace.wits.ac.za/items/d7252d34-0b25-473f-a94e-ccf3ed66026e)

### 4.13 · confirmed

For jammed effectors, NASA Dryden compared allocation-based reconfiguration with a robust servomechanism design. Allocation gave more uniform performance at somewhat higher computational cost: the redundant surfaces are redistributed to cancel the jammed surface's moment.

- *Evidence:* NTRS summary snippet.
- *Source:* J. J. Burken, P. Lu, Z. Wu, C. Bahm, 'Two reconfigurable flight-control design methods: robust servomechanism and control allocation', JGCD 24(3), 2001 (NTRS: https://ntrs.nasa.gov/api/citations/20010063955/downloads/20010063955.pdf). The journal citation details are from memory.

### 4.14 · confirmed

Small-UAV fault-tolerant practice (Ducard) treats allocation as a modular layer with explicit, cheap laws that sits after fault detection and isolation (FDI). With FDI in place, the controller does not need to be redesigned. When an aileron fails, feeding the degraded performance (for example a lower bank limit) back to guidance measurably improves mission safety. This matches the docs/03 idea of 'max bank 25 deg (was 45 deg)' being passed to Pathfinder.

- *Evidence:* The book has a Control Allocation chapter between FDI and nonlinear control. Ducard, Geering & Dumitrescu 2006: explicit allocation laws for small processors. Ducard, Kulling & Geering 2007 (ACC): reconfiguring guidance after an aileron failure improves safety in simulation.
- *Source:* G. J. J. Ducard, 'Fault-tolerant Flight Control and Guidance Systems: Practical Methods for Small Unmanned Aerial Vehicles', Springer 2009, doi:10.1007/978-1-84882-561-1; Ducard, Kulling, Geering, ACC 2007

### 4.15 · confirmed

Multirotor evidence on abstraction level: a quadrotor policy that commands collective thrust and body rates (CTBR) and relies on a low-level controller and mixer transfers from simulation to reality more robustly than one that outputs individual rotor thrusts (SRT). This is the closest published evidence that sending a learned policy's output through a lower layer helps transfer.

- *Evidence:* Abstract of the ICRA 2022 benchmark; a CTBR deep-RL policy also flew a real quadrotor above 45 km/h. I could not confirm the paper's detailed robustness numbers.
- *Source:* E. Kaufmann, L. Bauersfeld, D. Scaramuzza, 'A benchmark comparison of learned control policies for agile quadrotor flight', ICRA 2022, pp. 10504-10510, arXiv:2202.10796 (https://arxiv.org/abs/2202.10796)

### 4.16 · confirmed

Counter-evidence: policies that output motor commands directly can still generalise across very different airframes if they adapt from history. These results come from multirotors, where the actuator layout stays the same (four rotors) and only scales and signs vary.

- *Evidence:* Zhang et al. (T-RO 2025): motor-level outputs with a latent adaptation module; in simulation it handles parameter ranges 16 times wider than training; on real quadcopters, 3.7 times the mass and propeller constants differing by more than 100 times; motor failures handled. They argue adaptation must happen at motor level for extreme variation. RAPTOR (2025): an end-to-end policy with 2084 parameters in three layers, recurrent and adapting in context, trained by meta-imitation from 1000 per-quadrotor RL teachers; zero-shot on 10 real quadrotors from 32 g to 2.4 kg, differing in motors, frames, propellers and flight controllers.
- *Source:* D. Zhang, A. Loquercio, J. Tang, T.-H. Wang, J. Malik, M. W. Mueller, 'A learning-based quadcopter controller with extreme adaptation', IEEE T-RO 2025, arXiv:2409.12949 (https://arxiv.org/abs/2409.12949); J. Eschmann, D. Albani, G. Loianno, 'RAPTOR: A foundation policy for quadrotor control', arXiv:2509.11481, 2025 (https://arxiv.org/abs/2509.11481)

### 4.17 · confirmed

When the actuator layout itself changes (a 'geometry-agnostic' policy), recent work either conditions the policy on a normalised allocation matrix, or splits it into a wrench-output policy plus a separate allocation layer. Both report real-world zero-shot transfer. I found no fixed-wing head-to-head test of 'through an allocator' against 'raw surfaces plus a descriptor', so the planned ablation would itself be new.

- *Evidence:* Konstantaropoulos et al. 2026: a single PPO policy conditioned on the mass- and inertia-normalised allocation matrix (motor thrusts to body accelerations), outputting motor commands; trained in about 5 minutes on an RTX 3090 (NVIDIA Warp); zero-shot to three real hexarotors (planar, partly symmetric non-planar, asymmetric non-planar). Rehberg et al. RA-L 2026: an RL policy outputs a wrench and an allocation network trained against a QP expert maps it to motors, so each part can be verified separately; reusing policies from a library saves up to 73.5% of environment interactions. Pi et al. 2021: a single policy outputting thrust T and moment M, mapped to a quadrotor and a hexrotor, flown in simulation and real tests.
- *Source:* O. Konstantaropoulos, ..., K. Alexis, 'Embodiment-conditioned generalist control for multirotor aerial robots', arXiv:2606.10857, 2026 (https://arxiv.org/abs/2606.10857); W. Rehberg, M. Kulkarni, P. Weiss, K. Alexis, arXiv:2602.15533, RA-L 2026 (https://arxiv.org/abs/2602.15533); C.-H. Pi et al., Sensors 21(13):4560, 2021 (https://www.mdpi.com/1424-8220/21/13/4560)

### 4.18 · confirmed

Fixed-wing deep RL without an allocator can tolerate some actuator faults when trained offline. A cascaded SAC controller on a Cessna Citation model stayed stable and completed the manoeuvre with the rudder jammed at -15 deg and with aileron effectiveness cut by 70%. It did not demonstrate re-allocating around an aileron hard-over.

- *Evidence:* Normalised mean absolute error 2.64% on a coordinated 40-deg-bank climbing turn; six failure cases including the rudder jam, -70% aileron effectiveness, structural damage, icing and an aft CG shift.
- *Source:* K. Dally, E.-J. van Kampen, 'Soft actor-critic deep reinforcement learning for fault tolerant flight control', AIAA SciTech 2022, arXiv:2202.09262 (https://arxiv.org/abs/2202.09262)

### 4.19 · confirmed

Two newer lines of work tackle nonlinear effectiveness. One learns a compact SINDy model of control effectiveness with analytic derivatives, runs it inside a nonlinear allocator that accounts for actuator dynamics, and refreshes the model when prediction residuals grow (for example after an actuator failure). The other uses a neural network to approximate the moment-to-effector map directly. Both are relevant if the linear-derivative assumption breaks at hard-over deflections.

- *Evidence:* Demir et al. report accuracy close to a full onboard nonlinear model at much lower computational cost (abstract only). Khan et al. use an ANN allocator.
- *Source:* U. Demir, A. Ahmad, W. Fichter, arXiv:2606.13794, 2026 (https://arxiv.org/abs/2606.13794); H. Z. I. Khan et al., 'Nonlinear control allocation: a learning based approach', arXiv:2201.06180 (https://arxiv.org/abs/2201.06180)

### 4.20 · confirmed

Hobbyist autopilots do not do optimal or adaptive allocation. PX4's control_allocator builds the effectiveness matrix at runtime and offers two methods (CA_METHOD): 0, pseudo-inverse with output clipping; 1, pseudo-inverse with sequential desaturation (saturated actuators removed one by one and the pseudo-inverse recomputed). As far as I know (not confirmed), ArduPlane uses fixed mixers (elevon, V-tail) and estimates no effectiveness online. Online-estimating WLS is therefore beyond what stock autopilots do. It is a fair B2 classical baseline, but would need a custom implementation on hardware.

- *Evidence:* PX4 community and technical notes describe CA_METHOD and the desaturation recursion; the official PX4 documentation did not appear in the search results.
- *Source:* https://tlab-uav.github.io/tech-details/docs/hardware/px4-firmware/px4-dynamic-control-allocation ; https://discuss.px4.io/t/control-allocation-roll-pitch-torque-effectiveness-actual-meaning/37070

## Recommendations for M3

1. FIRST, before any RL: give aerojax split surfaces, or the allocation layer has nothing to allocate. Use a fixed-shape surface array, N_MAX = 8, plus a per-airframe mask: [ail_L, ail_R, elev_L (or ruddervator_L / elevon_L), elev_R (or ruddervator_R / elevon_R), rudder, flap_L, flap_R, spare]. Give each surface a 6-vector of derivatives [C_L, C_D, C_Y, C_l, C_m, C_n] per unit deflection and superpose linearly: dC = sum_i c_i*delta_i, with an optional |delta| drag term so asymmetric deflections produce yaw. Generate one-sided derivatives in AeroSandbox by deflecting each side alone (references.py already does +/-5 deg for X8 elevons). Fault slots already carry a surface index, so widen it to 0..7. Extend ActuatorCommand to a surfaces array, keep the old (a, e, r) through a fixed mixer M (8x3) so the stock B1 autopilot is unchanged, and re-run the M1 parity tests with the old 3-channel model recovered exactly by that mixer. Estimate: 3-5 days.
2. Rewire the derived Rascal for G4: compute Clda and Cnda from a new property 0.5*(left - right) in place of left-aileron-pos-rad. This is a wiring change only, with coefficients untouched; document it in the derived-model docstring and the M3 spec. State openly that the Rascal models no effect from symmetric aileron deflection (no lift or pitch change), so the 'use the other aileron' trick costs nothing in G4. Report split-aileron faults on the Rascal with that caveat, or keep G4 fault cases to whole-channel faults.
3. Choose the headline hard-over with the oracle before announcing it. On the current 3-channel model, a full-throw aileron or elevator hard-over is physically unrecoverable: on the Rascal the rudder cancels only about 24% of a full aileron hard-over, leaving about 70 deg/s of roll. Candidates that are both recoverable and impressive: (a) a rudder hard-over (cross-control with the ailerons, crab or sideslip flight, landing in wind); (b) once split, a single-aileron hard-over (the other aileron runs to its limit in the same direction, used symmetrically, the elevator absorbs the pitch and lift change, and the rudder plus dihedral supply the missing roll direction, giving a one-sided turn capability); (c) an aileron jam at about 1/3 to 2/3 of full throw (Flight 0's +6 deg). Run B5 (the oracle) on each before fixing the headline wording.
4. Allocator formulation (shared by L1 and the B2 INDI baseline, so the A/B tests the policy and not the allocator). Solve min_u ||W_u(u - u_p)||^2 + gamma*||W_v(B u - nu)||^2 subject to lo <= u <= hi. This is a box-constrained least-squares problem with A = [sqrt(gamma)*W_v*B ; W_u] and b = [sqrt(gamma)*W_v*nu ; W_u*u_p]. Work in normalised units: u in [-1,1] per surface, and nu divided by a per-axis scale such as 20, 10, 5 rad/s^2, so ||W_v B|| is about 1. Then gamma = 1e3 to 1e4 is enough for near-strict priority and keeps float32 well conditioned. Do not copy gamma ~ 1e5-1e6 into a float32 GPU solve. W_v in normal flight: roll 1, pitch 1, yaw 0.1 to 0.3, with thrust handled separately unless throttle has a real torque arm. In L2 recovery mode: pitch (unload) 1, roll 0.5, yaw 0.05. W_u = 1 with u_p = trim, or a small preference for the previous u.
5. JAX active-set implementation (primal, after Härkegård 2002). Keep a free-set mask f in {0,1}^n and a sign vector s in {-1,0,+1}. Each iteration: (1) d = b - A u. (2) Solve the masked normal equations (A^T A * (f f^T) + diag(1-f) + 1e-6 I) p = f * (A^T d) with an n x n Cholesky; the W_u block keeps A^T A positive definite, so this never fails. (3) If u + p is feasible, set u <- u + p and form the multipliers lam = s * (A^T (A u - b)); if every lam >= 0 then STOP, otherwise free the bound with the most negative lam. (4) Otherwise take the largest step that stays feasible, alpha = min over free i with p_i != 0 of (bound_i - u_i)/p_i, set u <- u + alpha p, and fix the blocking bound. Use lax.fori_loop with K = n_free_max + 2 (10 for n = 8) and a converged flag that freezes the state; avoid lax.while_loop, because under vmap the batch runs until its slowest member finishes. Every iterate is feasible, so a truncated solve is still admissible. Warm-start u and the working set from the previous control step, and store them in the policy-environment carry. Cost is about 10 Cholesky solves of size 8 per step, negligible beside the aerodynamics, even with 10k environments at 100 Hz. Unit-test against scipy.optimize.lsq_linear(A, b, bounds=(lo, hi)), which solves the same bounded-variable least-squares problem, on randomised B, bounds and nu, including hard-over bounds.
6. Use a redistributed pseudo-inverse or cascaded generalised inverse only as a quick fallback or an ablation. Compute u = u_p + W^-1 B^T (B W^-1 B^T)^-1 (nu - B u_p), clip whatever saturates, freeze it, subtract its moment from nu, and repeat on the remaining free columns, at most n iterations. It can stop short of attainable demands (Durham 1993, Bodson 2002). Skip FISTA, coordinate descent and ADMM box-QP solvers: at a condition number around 1e3 they need hundreds of iterations to reach active-set accuracy.
7. Bounds per control step (100 Hz, dt = 0.01 s): lo = max(u_min, u_prev - rho*dt), hi = min(u_max, u_prev + rho*dt). In INDI incremental form, shift both by the filtered actuator position u_f. Fault handling, all through bounds and columns so the solver never changes. Jammed or hard-over at an estimated delta_j: lo_j = hi_j = delta_j. The solver then cancels the jam's moment B_j*delta_j automatically; in INDI form, set Delta_u bounds to [0,0] and the measured angular acceleration absorbs the bias. Floating: set column B_j to 0, exclude u_j, and leave its flow-driven moment (aerojax: -0.6 x the local flow angle) to INDI and the policy as a state-dependent disturbance. Reversed: B_j <- -B_j once confirmed, with zero effectiveness until then, never a wrong-sign column. Reduced throw: shrink the box. Slow: rho <- m*rho. Backlash: widen the deadband in the actuator model used for u_f.
8. Effectiveness estimator, run at the control rate. Normalise by dynamic pressure so the estimate is the near-constant B_tilde in B = qbar * B_tilde, where B_tilde = J^-1 S diag(b, cbar, b) C_delta. Use incremental regression on synchronised signals: y = Delta(omega_dot_f) (3) and phi = qbar * Delta(u_f) (n). Get omega_dot_f by differentiating the gyro through a second-order Butterworth low-pass at about 8-15 Hz, and pass u_f through the identical filter, plus the known delay and lag (Smeur 2016). Use RLS with one shared covariance P (n x n) for all three rows: K = P phi / (lambda + phi^T P phi); Theta <- Theta + (y - Theta phi) K^T; P <- (P - K phi^T P)/lambda. Start with lambda = 0.995 (about 2 s memory at 100 Hz). Add three guards: (i) a dead-zone that skips updates when ||phi|| is below a noise threshold; (ii) cap trace(P) to stop windup in steady cruise; (iii) a variable forgetting factor, lambda_t = lambda_min + (lambda_max - lambda_min)*exp(-e^2/sigma^2) with lambda_min about 0.95. The variable-forgetting rule is a heuristic in the spirit of AA-INDI; its exact form there is unconfirmed. Kalman variant: treat vec(B_tilde) as a random walk with Q, and inflate Q and P for column j when the Sentinel (M4), or the M3 interim residual test, raises the probability of a fault on surface j. That is the clean place to connect the Sentinel's beliefs.
9. Detecting reversed and dead surfaces needs a hypothesis test, not just RLS drift: a wrong-sign column under closed loop diverges in under a second. Over a sliding window of about 0.2-0.4 s, compute the residual r_h = y - B_h phi for each surface j under three hypotheses (nominal, zero, flipped) and accumulate log-likelihoods. Switch when the posterior exceeds 0.95, then reset that column's covariance. During the window, hand over to L2 or a conservative mode. This is the 'gentle surface exercise' probe from docs/03 §4.4.
10. Excitation and identifiability. With split surfaces under a fixed antisymmetric mixer (delta_L = -delta_R), the left and right columns are perfectly collinear and RLS cannot separate them. Coordinated turns also correlate the aileron and rudder. Add null-space dither: a small perturbation z in the null space of B_hat (B_hat z = 0), about 0.5-1 deg at 2-5 Hz, only when redundancy exists and the recoverability margin allows. Report estimator convergence time in the A/B.
11. Policy action and observation design for L1 and L2. Output a demanded angular acceleration nu = nu_scale * tanh(a) (3 axes) plus throttle, not moments: the same output then means the same rotational response on every airframe, and it plugs straight into INDI. Feed back what the allocator achieved: nu_ach_prev = B_hat u_prev, the per-axis saturation residual nu - nu_ach, and per-axis attainable bounds nu_k+ and nu_k- (6 numbers, from 6 extra WLS solves with demand +/-M e_k, cheap in JAX). These give the policy an embodiment and authority descriptor it can compute onboard, and prevent integrator-like windup at the edge of the attainable set. Give the privileged teacher the true B, the bounds and the fault; give the GRU student only quantities computable onboard (history plus the allocator outputs above). Do not give the student the true B.
12. Train with the allocator in the loop and its errors randomised, so the policy does not overfit to perfect allocation. Per episode, sample the allocator's B from oracle (p = 0.2), nominal from autoconfig (0.3) or online RLS (0.5). Add +/-30% column-scale error and occasional transient wrong-zero columns at fault onset. Run the RLS transients inside training: two adaptive loops (the GRU's in-context adaptation and RLS) can otherwise fight each other.
13. Randomise actuator topology in training so G3 tests a new aerodynamic class rather than a new topology. Randomly delete or fault channels: missing rudder, merged elevator, ailerons dead and elevons only, per-surface derivative scaling. The X8's elevons-only, rudderless case (yaw available only through adverse elevon yaw) is then interpolation for the policy. In the report, say which parts of G3 are new aerodynamics and which are a new topology.
14. Ablations and A/B set (paired scenarios, Wilson confidence intervals, at least 5 seeds): (1) L1 through WLS with an oracle B; (2) through WLS with RLS; (3) through WLS with the nominal B and no estimation; (4) a raw-surface policy on the N_MAX = 8 masked surfaces, given a normalised per-surface effectiveness descriptor (following Konstantaropoulos 2026); (5) the L1 output plus a small bounded residual on the surfaces (to capture nonlinear or dynamic effects the allocator misses); (6) B2 INDI with the same allocator and estimator. Expect the allocator to win on instantaneous redundancy (split surfaces, jams), and the policy layer to win on dynamic cross-axis use (rudder, then sideslip, then dihedral roll), which a static allocator cannot represent. Show both.
15. Add a Skyview 'allocation view' (docs/03 §6.2) fed from the estimator state: B_hat columns with confidence (diagonal of P), bounds, a jam marker and the attainable-set bounds. It is cheap once the estimator exists and makes the fault handling inspectable. Log the iteration count of every allocation to catch any case that hits K.

## Risks and pitfalls

1. As built, the 'adaptive allocation' claim is empty: with 3 surfaces for 3 moments, B is square and WLS is just B^-1 with clipping. Split surfaces must land first, or the M3 write-up will overclaim (see the 'explain fidelity plainly' memory).
2. The G4 Rascal aerodynamics read only the left aileron and give symmetric deflection no effect. Split-aileron results there are either impossible without rewiring or optimistic after it.
3. Hard-over headline: a full-throw aileron or elevator hard-over is unrecoverable on single-channel airframes (about 70 deg/s residual roll on the Rascal by my estimate). Choose and verify the headline fault with the oracle first.
4. Linear superposition of control derivatives overstates authority at large deflections (separation and drag rise at 20-30 deg). The allocator, aerojax and the JSBSim Rascal all share this optimism, so sim-to-sim agreement does not validate hard-over recovery. Flag it as a fidelity limit.
5. Wrong-sign allocation (an unconfirmed reversed surface) is positive feedback and diverges within about 1 s. RLS with lambda = 0.995 is far too slow, so hypothesis testing and an L2 hand-over are required.
6. Identifiability: collinear columns (antisymmetric mixing, coordinated turns) and closed-loop correlation bias RLS. In steady cruise, low excitation winds up the covariance, and the next manoeuvre then produces a burst of estimator error. Dead-zones, trace caps and dither are needed.
7. Dynamic-pressure normalisation depends on airspeed, and pitot blockage is a headline fault. A blocked pitot corrupts qbar, and so the scaling of B_hat. Use the Navigator's synthetic airspeed (M4) or groundspeed plus wind, or fall back to estimating B unnormalised with faster forgetting. Test pitot faults combined with actuator faults (G6).
8. Actuator delay up to 150 ms and first-order lag: an INDI-style incremental allocation must use the filtered or predicted actual surface position, not the command. With mismatched synchronising filters, INDI loses its robustness. The policy also needs enough history in its GRU to see delayed effects.
9. Numerical issues: float32 with large gamma (1e5-1e6) squares the condition number in the normal equations. Use gamma of 1e3-1e4 in normalised units, or QR/f64 for the allocator alone.
10. Under vmap, lax.while_loop runs every lane to the slowest one's iteration count; use a fixed fori_loop with a converged mask.
11. Two adaptive loops (the GRU in-context adaptation and the RLS in the allocator) can interact or oscillate if the policy never saw estimator transients in training.
12. If the policy cannot see allocation saturation, it behaves like an integrator in windup (it keeps demanding unattainable nu). Feed back the achieved nu and the attainable bounds.
13. Teacher-student leakage: if the student gets B_hat from a privileged oracle path in evaluation but RLS onboard, the G2-G4 numbers are inflated. Keep observation provenance explicit, as ADR-006 already does for the privileged boundary.
14. Fairness of the A/B: B2 INDI must use the same allocator, estimator and fault logic. Otherwise the comparison measures the allocators, not the learned policy.
15. No published fixed-wing evidence shows that learning through an allocator transfers better across configurations; the multirotor evidence is mixed (Kaufmann 2022 supports abstraction; RAPTOR and Zhang 2025 show direct-actuator policies generalising). The result may come out negative; pre-register the ablation.

## Open questions

1. Should the policy output angular-acceleration demands (rad/s^2, invariant in response) or nondimensional moment-coefficient demands (invariant in dynamic pressure)? The evidence is indirect; an ablation is cheap once the allocator exists.
2. Does online effectiveness estimation add anything over a GRU student with a nominal-B allocator, given that the GRU already adapts in context? It may matter only for inspectability and the Skyview allocation view, which is still a legitimate reason to keep it.
3. How much faster and more reliable is fault re-configuration when it is driven by the Sentinel's fault posterior (M4) than by RLS and a residual test alone (M3)? Should M3 ship a simple 3-hypothesis per-surface test as a stand-in?
4. Should aerojax model floating surfaces as hinge-moment dynamics (a second-order surface free to rotate) instead of the static -0.6 x local-flow trail? This changes stick-free stability, especially for a floating elevator.
5. Add an optional, faultable servo-position or servo-current sensor? Many small UAV servos lack one, but where present it makes jam isolation almost trivial. It should probably be an airframe option, off by default.
6. Whether the L2 recovery policy should use the same allocator with different W_v priorities (pitch unload first), or output raw surfaces given how aggressive recoveries are. A head-to-head on hard-over scenarios would answer it.
7. Unconfirmed numbers I could not verify through search: Härkegård's typical warm-started iteration counts; Paparazzi's default gamma_sq and imax; Smeur 2016's LMS gains and filter cut-offs; the exact form of AA-INDI's variable forgetting factor; and whether ArduPlane has any online allocation. These should be checked in the PDFs (the search budget for this run was used up).
