# Report 5: INDI as the classical baseline

*Part of the M3 literature review, 6 Nov 2026 ([README](README.md)). The brief and the findings are as the research agent returned them; local paths are made repository-relative. Where this report and the spec differ, [docs/09](../../09_m3_technical_spec.md) holds.*

## Brief

INDI (incremental nonlinear dynamic inversion) for fixed-wing UAVs as the M3 classical baseline (B2-control): the literature, its robustness limits, how it compares with PID and RL, and a design for aerojax at 100 Hz

## Key findings (19 of 21 confirmed in the session)

### 5.1 · confirmed

Grondman et al. (2018) is a flight test on a manned passenger jet, not a small UAV. INDI control laws flew on a Cessna Citation II and did better than classical NDI.

- *Evidence:* From the abstract only: INDI and NDI control laws were compared in flight on the Citation II (TU Delft/DLR PH-LAB with an experimental fly-by-wire system), and INDI performed better. I could not read the internal details (sampling rate, filters, how measured surface positions were synchronised). It is the research paper inside Grondman's MSc thesis 'Dynamic Inversion Flight Control Law Design for Fixed-Wing Aircraft'.
- *Source:* Grondman, Looye, Kuchar, Chu, van Kampen, 'Design and flight testing of incremental nonlinear dynamic inversion based control laws for a passenger aircraft', AIAA GNC (SciTech) 2018, doi:10.2514/6.2018-0385; https://repository.tudelft.nl/record/uuid:964e1942-5c83-45e7-8ace-507a33ea3146

### 5.2 · confirmed

Smeur, Chu and de Croon (2016) set the standard small-UAV INDI recipe. Angular acceleration comes from a second-order filter on the gyro; the same filter is applied to the actuator states to cancel its delay (synchronisation); the control effectiveness is adapted online with LMS; a G2 term covers propeller spin-up torque. It was flown on a Parrot Bebop running Paparazzi, and the code is open source.

- *Evidence:* JGCD 39(3):450–461, doi:10.2514/1.G001490. Contributions listed in the abstract: delay correction when deriving angular acceleration from rate, adaptive INDI estimating control effectiveness online, propeller momentum in the controller, real flight tests. Specific numbers I remember but could NOT confirm: a 512 Hz loop, and a second-order filter with ωn≈50 rad/s and ζ≈0.55. Paparazzi source that I did confirm: the angular-acceleration filter is a 2nd-order Butterworth with default FILT_CUTOFF 8 Hz (v5.14/v5.18); the newer stabilization_indi has per-axis rate cutoffs of 20 Hz and an estimation cutoff of 4 or 8 Hz depending on version; G1 (effectiveness per actuator per axis), G2 (actuator-derivative effectiveness on yaw); the adaptive update uses a gain mu and clamps estimates to a small positive floor; WLS allocation is an option.
- *Source:* Smeur, Chu, de Croon, 'Adaptive Incremental Nonlinear Dynamic Inversion for Attitude Control of Micro Air Vehicles', JGCD 2016, https://research.tudelft.nl/en/publications/adaptive-incremental-nonlinear-dynamic-inversion-for-attitude-con-2/ ; Paparazzi docs https://paparazzi-uav.readthedocs.io/en/latest/modules/stabilization_indi.html and https://docs.paparazziuav.org/v5.18/stabilization__indi__simple_8c_source.html

### 5.3 · confirmed

Cascaded INDI (an INDI outer loop on linear acceleration over an INDI attitude loop) beat PID at gust rejection in a wind tunnel, and it does not need frequent position updates.

- *Evidence:* Quadrotor in a wind tunnel at 10 m/s; the paper also compares linearised and nonlinear calculations of the thrust-vector increment. Control Engineering Practice 73:79–90 (2018).
- *Source:* Smeur, de Croon, Chu, 'Cascaded incremental nonlinear dynamic inversion for MAV disturbance rejection', CEP 2018, arXiv:1701.07254, https://arxiv.org/abs/1701.07254

### 5.4 · confirmed

Smeur and colleagues extended INDI with control allocation to hybrid aircraft. On a tailsitter, unmodelled forces and moments were compensated by the incremental structure, and effectiveness was found from test flights rather than a wind tunnel. Later work uses weighted least squares (WLS) allocation directly inside cascaded INDI.

- *Evidence:* Cyclone tailsitter, two cascaded INDI loops (attitude and velocity), outdoor flights, JGCD 43(2), 2020. The search did not show whether the 2020 paper itself uses WLS. Ma, Smeur, de Croon (IEEE RA-L 2025) use cascaded WLS-based INDI on a tilt-rotor tailsitter with elevons: outdoor waypoint tracking at 16 m/s, and hover–forward transitions without actuator saturation. XINCA (Karssies & de Wagter, IJMAV 2022) extends incremental nonlinear control allocation to the outer loop of a quadplane, solved with an active-set method.
- *Source:* Smeur, Bronz, de Croon, 'Incremental Control and Guidance of Hybrid Aircraft Applied to a Tailsitter UAV', JGCD 2020, arXiv:1802.00714; Ma, Smeur, de Croon, arXiv:2503.02158 (RA-L 2025); Karssies & de Wagter, IJMAV 2022, doi:10.1177/17568293211070825, https://pure.tudelft.nl/ws/portalfiles/portal/110997639/17568293211070825.pdf

### 5.5 · confirmed

The Smeur/Remes line on allocation and online effectiveness estimation: Blaha, Smeur and Remes surveyed optimal control allocation for aerial vehicles and found a trend towards linearised incremental schemes. The same group identifies the whole INDI effectiveness matrix online with recursive least squares (RLS) from a brief excitation.

- *Evidence:* Survey: Actuators 12(7):282, 2023. 'Control of Unknown Quadrotors from a Single Throw': 52 parameters (effectiveness plus motor model) fitted by RLS from rotor-speed feedback and measured angular-rate derivatives, then an INDI attitude recovery after 450 ms of excitation. 'Flying a quadrotor with unknown actuators and sensor configuration' (IMAV 2024) adds identification of IMU position and rotation and of motor thrust directions, using sequential open-loop excitation of each motor.
- *Source:* Blaha, Smeur, Remes, 'A Survey of Optimal Control Allocation for Aerial Vehicle Control', Actuators 2023, https://www.mdpi.com/2076-0825/12/7/282 ; arXiv:2406.11723 ; arXiv:2409.01080

### 5.6 · confirmed

The standard solver for bounded WLS allocation is Härkegård's active-set method, which always finds the optimal solution at a cost similar to the redistributed pseudo-inverse. Paparazzi's WLS allocation descends from it.

- *Evidence:* Proc. 41st IEEE CDC, pp. 1295–1300, 2002, doi:10.1109/CDC.2002.1184694. Petersen & Bodson found WLS active-set the most efficient of the methods they compared. That Paparazzi's wls_alloc was ported from Härkegård's QCAT, and its default weights (e.g. Wv≈{1000,1000,1,100}, γ≈1e5), are from my memory and NOT confirmed.
- *Source:* Härkegård, 'Efficient active set algorithms for solving constrained least squares problems in aircraft control allocation', CDC 2002; Petersen & Bodson, https://pubweb.eng.utah.edu/~bodson/pdf/Constrained%20Quadratic%20Programming%20Techniques%20for%20Control%20Allocation.pdf

### 5.7 · confirmed

With servo dynamics, INDI approximates a sensor-based NDI law (one that models the actuator explicitly) arbitrarily well, but only when the actuators are fast. When actuator bandwidth is low relative to the state dynamics, the NDI-with-actuator form is better: it compensates state-derivative terms, has well-defined error dynamics, and tracks a reference model exactly.

- *Evidence:* Derived for first-order linear actuators and validated on the roll motion of a fixed-wing aircraft. JGCD 46(4):709–717, 2023. Related: Steffensen, Ginnell, Holzapfel (Actuators 2024) flight-tested an incremental differential-PI law (iDPI, derived from INDI) on a subscale fixed-wing, with frequency-domain identification of servo and linkage dynamics.
- *Source:* Steffensen, Steinert, Smeur, 'Nonlinear Dynamic Inversion with Actuator Dynamics: An Incremental Control Perspective', JGCD 2023, arXiv:2201.09805 ; Steffensen et al., Actuators 13(4):130, 2024, https://mdpi.com/2076-0825/13/4/130

### 5.8 · **not confirmed** (memory or secondary)

For small UAVs, the time-scale separation INDI assumes is marginal on the roll axis. On the repo's own Aerosonde, the roll mode is about as fast as the servo lag plus delay, so the Steffensen/Sieberling corrections matter.

- *Evidence:* Own calculation from configs/airframes/aerosonde_bm.yaml (C_lp=-0.51, b=2.8956 m, S=0.55 m², Jx=0.8244 kg m², V=25 m/s): L_p/Jx = q̄Sb²C_lp/(2V Jx) ≈ -21.8 1/s, so the roll-mode time constant is about 46 ms. The servos have τ=20 ms and 20 ms delay, and the procedural fleet samples τ 10–40 ms and delay 10–60 ms (generator/build.py). Linear analysis says the unrejected residual is a·(1−H)·ω, i.e. damping left uncancelled above the filter bandwidth: it costs tracking accuracy, not stability.
- *Source:* Own derivation plus repo inspection (configs/airframes/aerosonde_bm.yaml; src/aeroadapt/generator/build.py)

### 5.9 · confirmed

Two methods remove the angular-acceleration estimation lag: a linear predictive filter (Sieberling 2010) and a hybrid model/sensor estimate built with a complementary filter (Kumtepe 2022). Angular accelerometers (Keijzer 2019, flight-tested on the Citation) cut sensor delay and improved robustness to model mismatch in simulation.

- *Evidence:* Sieberling, Chu, Mulder, JGCD 33(6):1732–1742 (2010): a predictive filter estimates angular acceleration from rates and references, which makes INDI work with conventional IMUs. Kumtepe, Pollack, van Kampen, AIAA SciTech 2022-1597: complementary blend of model and sensor, F-16 attitude law, tested against sensor delay and aero uncertainty. Keijzer et al., AIAA 2019-0129: flight tests on the Citation II with angular-accelerometer feedback.
- *Source:* https://repository.tudelft.nl/record/uuid:cf9119d3-8bbf-489b-b734-f25c4cd1402a ; https://research.tudelft.nl/files/144858203/6.2022_1597.pdf ; https://repository.tudelft.nl/record/uuid:e4814409-24df-4a4f-9e9a-c03400e78ad7

### 5.10 · confirmed

Discrete INDI sampled faster than 50 Hz (Ts < 0.02 s) has large stability margins, but it degrades with actuator-measurement bias, rate noise and measurement delay. Real-time delay identification plus pseudo-control hedging (PCH) restores synchronisation and stops oscillation.

- *Evidence:* AIAA GNC 2018-1127. The analytic stability analysis gives the Ts < 0.02 s result. Simulation shows degradation from actuator-measurement bias, rate noise and delays. Remedy: latency-based delay identification to align rate and actuator-measurement delays, plus PCH.
- *Source:* van 't Veld, van Kampen, Chu, 'Stability and Robustness Analysis and Improvements for Incremental Nonlinear Dynamic Inversion Control', AIAA 2018, https://repository.tudelft.nl/record/uuid:53f5c0b0-9e3a-4bec-9471-9ce1aa1c3ba8

### 5.11 · confirmed

INDI's weak spot is singular perturbations: transport delay, unmodelled actuator or structural dynamics. Quantitatively, an error in the effectiveness matrix turns the INDI loop into a neutral delay equation with a computable time-delay margin.

- *Evidence:* Pollack & van Kampen (JGCD 46(9):1785–1798, 2023; SciTech 2022-1395) used structured-singular-value (μ) analysis on an F-16 pitch-rate law; basic INDI has a relatively small margin against singular perturbations, and they propose robust inversion-loop augmentations. Huang, Zhang, Pool, Stroosma, Chu (JGCD 2022, doi:10.2514/1.G006024) give the first quantified delay margin and effectiveness-uncertainty tolerance for INDI. A TU Delft report notes INDI has been observed robust to effectiveness mismatch up to about 50% without a general theory.
- *Source:* https://research.tudelft.nl/en/publications/robust-stability-and-performance-analysis-of-incremental-dynamic--2/ ; https://repository.tudelft.nl/record/uuid:6ea56b4f-6e50-4296-899c-d79bdca6f4fc ; https://repository.tudelft.nl/file/File_cf6553e3-d731-400c-8c38-8473719971bc

### 5.12 · confirmed

A sign error in control effectiveness (a reversed surface or mixer error) is the most destabilising mismatch for INDI. Lag and inertia mismatch are fairly benign. This means a reversed-surface fault needs explicit detection in the classical baseline.

- *Evidence:* Schenk & Ducard (2026): closed-form fifth-order loop of an INDI pitch-rate/elevon channel on a tilt-rotor VTOL, with Routh–Hurwitz over mismatch parameters. Effectiveness mismatch, especially a sign error in the allocation, destabilises most; actuator lag and inertia mismatch are fairly benign at nominal gain. Separately, a 2022 IFAC paper found that conventional INDI fails the sufficient stability conditions under severe damage.
- *Source:* Schenk & Ducard, 'Linear Stability Analysis of an INDI Pitch-Rate Controller under Model Mismatch for a Tilt-Rotor VTOL UAV', arXiv:2607.16471 ; 'Predictor-based Adaptive INDI for Fault-Tolerant Flight Control', https://research.tudelft.nl/en/publications/predictor-based-adaptive-incremental-nonlinear-dynamic-inversion-

### 5.13 · **not confirmed** (memory or secondary)

Own linear analysis (scalar, synchronised, r = g_true/ĝ): INDI tracks exactly at DC for any r > 0. A small-gain argument guarantees stability of the incremental loop for 0 < r < 2. Overestimating ĝ is safe but slow; underestimating it makes the loop aggressive; r < 0 runs away. In MIMO the condition is ||I − G Ĝ⁺||·||H A||∞ < 1.

- *Evidence:* With u = u_f + (ν − ω̇_f)/ĝ, u_f = H·A·u, plant ω̇ = d + g·A·u: ω̇ = d + rA(ν − Hd)/(1 − (1−r)HA). At DC (H=A=1) this gives ω̇ = ν for any r ≠ 0. H (Butterworth) and A (lag·delay) have peak gain ≤ 1, so |1−r| < 1 is sufficient. With a first-order H = ωf/(s+ωf) and ideal actuators, the incremental pole is at s = −r·ωf, so a reversed surface (r<0) gives an unstable pole at +|r|ωf. The reversal therefore has to be detected within a few 1/ωf. This matches the Schenk & Ducard finding.
- *Source:* Own derivation (cross-check against Wang et al., JGCD 2019; van 't Veld et al. 2018)

### 5.14 · confirmed

Wang et al. reformulated INDI without the time-scale-separation assumption and gave Lyapunov/perturbation-theory stability results for any relative degree. This is the citation to use for 'INDI is not just a heuristic'.

- *Evidence:* JGCD 42(5):1116–1129, 2019. Robustness to regular and singular perturbations; Monte Carlo on aircraft command tracking with disturbances and model uncertainty.
- *Source:* Wang, van Kampen, Chu, Lu, 'Stability Analysis for Incremental Nonlinear Dynamic Inversion Control', JGCD 2019, https://pure.tudelft.nl/ws/portalfiles/portal/104464714/1.G003791.pdf

### 5.15 · confirmed

Against a robustly synthesised PID, INDI's robustness advantage largely disappears. Hybrid INDI and an H∞-tuned two-degree-of-freedom gain-scheduled PID are 'virtually equivalent' in robustness and performance. The real advantage of INDI is modular robust design in an implicit model-following structure.

- *Evidence:* Non-smooth multi-objective H∞ synthesis with mixed parametric and dynamic uncertainty; near-identical robustness in both time-invariant (LTI) and time-varying (LTV) settings. Aerospace Science and Technology 152:109377 (2024).
- *Source:* Pollack, Theodoulis, van Kampen, 'Commonalities between robust hybrid incremental nonlinear dynamic inversion and PID flight control law design', AST 2024, https://pure.tudelft.nl/ws/portalfiles/portal/214067341/1-s2.0-S127096382400508X-main.pdf

### 5.16 · confirmed

INDI's main empirical strength is absorbing unknown moments from damage, aero changes and wind. Its fault-tolerant control has flown a quadrotor that lost two opposing rotors, and INDI trajectory control tolerated actuator faults plus model uncertainty on fixed-wing aircraft.

- *Evidence:* Sun, Wang, Chu, de Visser (IEEE T-RO 37(1):116–130, 2020): flight tests in an open-jet wind tunnel; better than LQR in windy, high-speed flight. Lu, van Kampen, de Visser, Chu (CEP 57:126–141, 2016): an INDI trajectory controller follows the reference with model uncertainties and actuator faults (the abstract does not say whether stuck-surface cases are included). Rota et al. (ICUAS 2026, simulation): INDI and NDI with a disturbance observer give identical inputs in nominal conditions, but INDI is more robust under model mismatch and combined stresses, and needs less wrench energy.
- *Source:* arXiv:2002.07837 ; https://repository.tudelft.nl/record/uuid:051aee7e-458b-43c6-9a65-9818b67ec279 ; arXiv:2605.05825

### 5.17 · confirmed

In head-to-head studies, learned controllers beat INDI by tens of percent within a vehicle family. End-to-end RL is faster but more sensitive to modelling error than an RL-over-INDI stack. Published fixed-wing RL is compared with PID, not INDI.

- *Evidence:* Ferede, De Wagter, Izzo, de Croon (arXiv:2311.16948): end-to-end RL beat a network that commands thrust and body rates into an INDI inner loop by 1.39 s in simulation but only 0.17 s in real flight; the authors attribute the shrinking gap to end-to-end sensitivity to model error versus INDI's robustness. Neural-Fly (O'Connell et al., Science Robotics 2022, as cited in the repo's docs/01): 35% better tracking than INDI in a wind tunnel up to 12.1 m/s. Bøhn et al.: a deep-RL fixed-wing attitude controller flown on a real UAV matched ArduPlane's PID. Olivares et al. (ICINCO 2024): TD-MPC beat PID and model-free RL in wind, with no faults. Dally & van Kampen (SciTech 2022): SAC fault-tolerant control on the Citation (jammed rudder, 70% aileron-effectiveness loss); I did not find an INDI head-to-head. Hachem, Roos, Miquel, Bronz (arXiv:2501.07223): robust-control augmentation of INDI improved gust rejection by more than 50%.
- *Source:* arXiv:2311.16948 ; arXiv:2409.17896 ; arXiv:2202.09262 ; arXiv:2501.07223 ; repo docs/01_sota_review_and_research_direction.md

### 5.18 · confirmed

For online effectiveness estimation under faults, recent TU Delft fixed-wing and flying-wing work replaces Smeur's LMS with RLS using a variable forgetting factor (VFF-RLS), because it adapts faster after a fault. The estimate may not converge to the true effectiveness, but adapting still helps control.

- *Evidence:* Atmaca & van Kampen (AIAA 2025, Flying-V): two-step estimation, EKF for states and VFF-RLS for parameters. Atmaca, de Visser, van Kampen (AIAA 2026): adds an IMU-fault differentiator and analyses analytically how actuator and IMU faults affect INDI. Ul Haq, Atmaca, van Kampen (AIAA 2026): adaptive INDI fault-tolerant control on a flying wing, where the true effectiveness is not always recovered. Hafner et al. (IEEE CCTA 2023): RLS adaptive allocation with extended INDI on a hexacopter. These came from search-result summaries, not full texts.
- *Source:* Search results for TU Delft repository entries (e.g. https://repository.tudelft.nl/file/File_96f8edb1-c562-4b4e-a03f-aeb9696641a0) ; https://portal.fis.tum.de/en/publications/least-squares-based-adaptive-control-allocation/

### 5.19 · confirmed

There is fixed-wing and flying-wing UAV INDI flight-test evidence, but it is thin and mostly paywalled or non-English.

- *Evidence:* An 'improved INDI' for 6-DoF fixed-wing UAV waypoint tracking, flight-tested (ASCE J. Aerospace Engineering 35(6), 2022). INDI plus a tracking differentiator to replace numerical differentiation, wide-envelope flying-wing attitude control, simulation and flight with noisy measurements (Chen et al., Acta Aeronautica et Astronautica Sinica 47(15), 2026). van Ekeren's 2016 TU Delft MSc: INDI and incremental backstepping flight-tested on the FASER subscale UAV. Tal & Karaman: flatness plus INDI on a tailsitter flying wing with extensive flight tests (arXiv:2207.13218; I could not confirm the journal). An INDI tracking controller was used in dynamic-soaring flight tests in strong wind (arXiv:2512.06610). Pfeifle & Fichter's fixed-wing INDI in the aerodynamic frame for wind compensation is known only second-hand.
- *Source:* https://ascelibrary.com/doi/10.1061/%28ASCE%29AS.1943-5525.0001495 ; https://www.sciopen.com/article/10.7527/S1000-6893.2026.33119 ; https://repository.tudelft.nl/file/File_769ce192-e1ef-49de-aab0-7f875a1913c9 ; arXiv:2207.13218 ; arXiv:2512.06610

### 5.20 · confirmed

Repo fact: aerojax models three lumped surface channels (aileron, elevator, rudder) plus throttle, with Beard & McLain derivatives (δe only enters C_m). Allocation is therefore a square 3×3 inversion with no surface redundancy. A full aileron or elevator hard-over is physically unrecoverable on most airframes; only a rudder hard-over is realistically recoverable.

- *Evidence:* src/aeroadapt/sim/aerojax/actuators.py: 'surfaces: (3,) … (aileron, elevator, rudder)'; faultlab hard_over drives a channel to full travel. My calculation for the Aerosonde at 25 m/s (q̄=383 Pa, Jxz included) gives G per unit normalised command (rad/s², B&M signs) ≈ [[55.2, 0, −0.76], [0, −18.3, 0], [2.1, 0, −10.5]]. The rudder's direct roll authority (0.76) is about 1.4% of the aileron's (55). Cancelling an aileron hard-over through dihedral would need β ≈ 0.57 rad (33°). The X8 flying wing (G3) uses elevons as lumped a/e channels, has no rudder (yaw is uncontrolled, so B has rank 2), and has no shared elevon travel limit. Real elevons satisfy |δe|+|δa| ≤ δmax, so the lumped model overstates authority.
- *Source:* Repo inspection: src/aeroadapt/sim/aerojax/actuators.py, src/aeroadapt/faultlab/schema.py, configs/airframes/aerosonde_bm.yaml, src/aeroadapt/generator/references.py; own numbers

### 5.21 · confirmed

Repo fact: the timing budget suits INDI. Physics runs at 200 Hz and control at 100 Hz (Ts = 0.01 s, inside the 0.02 s bound above). The IMU runs at 100 Hz with gyro noise 0.005 rad/s. The stack gets no servo-position feedback, so INDI must use an internal actuator model, as Smeur does for motors without RPM feedback.

- *Evidence:* SimConfig dt_physics=0.005 s, substeps=2. Aerosonde IMU rate_hz 100, gyro_noise_rps 0.005. A raw finite-difference ω̇ would have σ ≈ 0.71 rad/s², reduced substantially by the filter. The vehicle protocol puts surfaces_rad only in TruthSnapshot (privileged; ADR-006). The actuator delay buffer allows up to 150 ms.
- *Source:* Repo inspection: src/aeroadapt/sim/aerojax/sim.py, src/aeroadapt/vehicle/protocol.py, configs/airframes/aerosonde_bm.yaml

## Recommendations for M3

1. Build one shared allocation module (e.g. src/aeroadapt/stack/allocation.py: pure JAX, no sim imports, per ADR-006). It should hold the WLS allocator, the effectiveness model Ĝ(q̄) and an optional RLS estimator, used by both the INDI baseline and L1. Then the A/B isolates the brain from the allocator. Suggested L1 action space: per-axis angular-acceleration demand ν plus thrust, fed into the same INDI+WLS layer (RL over INDI, as in Ferede et al.). Add two ablations: L1 emitting surface-moment demands into a non-incremental WLS, and raw surfaces with an embodiment descriptor.
2. INDI inner law at 100 Hz: u_cmd = u_f + Δu, where Δu = argmin γ‖W_v(Ĝ Δu − (ν − ω̇_f))‖² + ‖W_u(u_f + Δu − u_pref)‖², subject to the position limits and the rate limit × dt. Take the increment from u_f, the filtered modelled actuator position, not from the previous command; this is the core of Smeur's synchronisation.
3. Filters: H(z) is a 2nd-order Butterworth low-pass, applied identically to the bias-corrected gyro and to the modelled actuator positions. Start at f_c = 12 Hz (group delay ≈ 0.225/f_c ≈ 19 ms) and sweep 6, 8, 12, 16 and 20 Hz; Paparazzi defaults to 8 Hz at much higher loop rates. Take ω̇_f as the backward difference of the filtered rate. The internal actuator model per channel copies the structure in actuators.py: delay of round(d̂/dt) steps, first-order lag with α = 1 − exp(−dt/τ̂), rate limit and clip, using the nominal τ̂ and d̂ from the airframe config.
4. Effectiveness: autoconfig computes Ĝ(q̄) = J⁻¹ q̄ S diag(b, c̄, b) C_δ diag(δ_max·sign) from the nominal derivatives, including Jxz (the 'manufacturer parameter file' rule B1 already follows). Schedule on q̄̂ = ½ρV̂², with V̂ clamped to [0.9 V_stall, 1.1 V_max]. The lower clamp matters: an under-reading pitot shrinks Ĝ, so r = g/ĝ rises past 2 and the loop oscillates; an over-reading pitot only makes it sluggish. Reuse the worked Aerosonde matrix in the findings as a unit test.
5. Gains, as starting points derived from a phase margin of about 45–50° on L(s) = K·e^(−sd)/(s(τs+1)): rate gain K_ω,i ≈ 0.75/(τ̂_i + d̂_i), adding 0.225/f_c if the P term uses the filtered rate. For the Aerosonde (τ = 20 ms, d = 20 ms) that gives about 18 rad/s on the raw gyro or about 13 on the filtered rate; for the slowest fleet servos (40 ms + 60 ms) about 7. Attitude gains ≈ K_ω/4 with rate limits. Feed-forward ω̇_ref from a second-order reference model with pseudo-control hedging so saturation does not wind up. Turn coordination: r_ref = (g/V)·sinφ·cosθ, plus feedback on lateral acceleration or sideslip estimated from a_y.
6. WLS solver in JAX: Härkegård's active set, unrolled for a fixed n_u + 2 iterations with masks so it vmaps. γ between 1e4 and 1e6 (sweep). W_v: roll = pitch = 1 and yaw 0.1–0.3 nominally; in upset mode roll ≫ pitch (bank first). W_u = I, u_pref = trim. If an effectiveness column is zeroed (jam) or the matrix is rank-deficient (the X8 has no rudder), the W_u term keeps the problem well-posed, so no special case is needed.
7. Adaptive variant (B2-adaptive) alongside fixed-Ĝ INDI: RLS per axis on increments Δω̇_f ≈ Ĝ·Δu_f, forgetting factor λ = 0.995–0.999 at 100 Hz (about 2–10 s memory). Update only when ‖Δu_f‖ is large enough, with covariance bounds. Project each column to [0.25×, 3×] nominal with the sign locked. Add a separate reversal detector: a sustained negative correlation between Δu_f and Δω̇_f on the primary axis, or command and rate error diverging together, flips the sign. Per my derivation the runaway time constant is about 1/(|r|·ω_f), so the detector must act within about 0.2–0.5 s. Keep Smeur's LMS (G ← G − μ(GΔu_f − Δω̇_f)Δu_fᵀ) as a cheap ablation.
8. Pre-register INDI robustness sweeps before the A/B: Ĝ scale in {0.3, 0.5, 1, 2, 3}; delay mismatch of 0 to ±3 control steps; τ mismatch ±50%; f_c sweep; gyro-noise multipliers. Expected outcome: stable for effectiveness ratio r in (0, 2) when synchronised, degrading fastest under delay mismatch (Pollack, van 't Veld). This makes a credibility figure, and it protects the B2 numbers against reviewers.
9. Fair tuning protocol: tune INDI's 6–8 global hyperparameters (f_c, gain multipliers, γ, W_v, λ) on the G1 training fleet only, with a batched JAX grid or CMA-ES, then freeze them before G2–G4. The per-airframe values come only from autoconfig formulas, as with B1. Give INDI the nominal servo τ and delay, not the randomised true values, and apply the same delay and noise randomisation to INDI as to L1.
10. G4 (Rascal in JSBSim): report two INDI variants. One takes Ĝ from the Rascal's published or JSBSim aero data via autoconfig; the other identifies Ĝ and the servo delay from 2–3 s of doublet excitation with RLS and cross-correlation (Blaha et al., van 't Veld). The second is the fairest classical counterpart to zero-shot RL.
11. Fault-matrix hypotheses to pre-register for INDI. Strongest on loss of effectiveness, reduced throw, CG shift and damage, and wind (INDI absorbs unknown moments). Neutral on motor cut (TECS's job). Weak on reversed surfaces without the detector, slow-servo or rate-limit faults (sync mismatch, a singular perturbation), vibration or gyro noise (derivative amplification) and pitot faults (Ĝ scheduling). Constant gyro bias does not affect ω̇_f. These give L1 concrete places to win, and an honest place to lose (H1 in docs/01).
12. Before the M3 A/B, and before committing to the 'recovery from a hard-over' headline, split the actuator model into per-surface effectors: left/right aileron, left/right elevator or elevon, ruddervators, rudder. Give each its own column in B (half the lumped C_lδa each, adverse yaw split), and enforce shared elevon travel for flying wings. Otherwise allocation is a 3×3 inversion with no redundancy, and aileron or elevator hard-overs cannot be recovered. If the split is cut, make the headline a rudder hard-over, or an aileron stuck at an offset or of partial magnitude, and say so.
13. Classical counterpart to L2: INDI with a scripted recovery (unload to about 0–0.5 g, roll wings-level with roll priority in W_v, pull at a bounded load factor, then hand back). INDI suits this because its inner loop does not need the aero model at large α or β. Effectiveness loss near stall gives r < 1 (sluggish but stable); aileron reversal near stall gives r < 0 (diverges). Check whether aerojax's linear-derivative aero can represent either before claiming upset results.
14. Report INDI-specific diagnostics with every A/B: share of time saturated, the realised effectiveness ratio r (truth versus Ĝ, from the privileged view), actuator activity (total variation per second), and oscillation flags. These explain why INDI fails where it does, which is what makes the RL gains credible.

## Risks and pitfalls

1. Lumped surface channels (repo fact) make the 'adaptive allocation re-routes around a jammed aileron' story and the hard-over headline physically impossible for aileron or elevator faults. Any controller, RL included, will fail them. Without split surfaces, the A/B on these faults measures nothing.
2. Flying wings (G3, X8/Zagi) have no rudder, so B has rank 2 and yaw is uncontrolled. Lumped elevon channels with independent limits overstate authority, and the coupled roll/pitch saturation is where allocation matters most. If this is not modelled, G3 flatters both INDI and RL.
3. Time-scale separation is marginal for small UAVs: the Aerosonde roll mode is about 46 ms against 40 ms of servo lag plus delay, and up to 100 ms on slow fleet servos. Plain INDI stays stable but does not follow the reference model exactly. Expect INDI to look worse than its manned-aircraft reputation unless the hybrid model term or the Steffensen form is added.
4. Synchronisation depends on knowing the servo delay. aerojax randomises it from 10–60 ms per airframe. Giving INDI the true per-scenario delay is an information leak that favours B2; giving it nothing makes it oscillate. Fix and document the information budget, which I suggest should be nominal config values only.
5. A reversed surface destabilises INDI exponentially (sign error: Schenk & Ducard 2026, and my derivation). Without a detector, B2 crashes on every reversed-surface scenario and the RL looks artificially good. With a detector tuned on the test faults, B2 is over-fitted. Tune it on G1 only.
6. Pitot faults feed straight into Ĝ(q̄). An under-reading pitot pushes r above 2 and causes oscillation, so B2 needs a q̄ floor or synthetic airspeed. Without it, sensor-fault results conflate the diagnosis problem with INDI's scheduling.
7. Literature caveat: robust PID ≈ hybrid INDI (Pollack et al. 2024). The B1 stock PID is autotuned per airframe, so INDI may not clearly beat B1 on nominal or wind tracking. Do not pre-announce 'INDI > PID'; expect INDI's gains to show only on effectiveness and damage faults.
8. Learned-over-INDI gains in the literature are tens of percent within a family (Neural-Fly 35%; Ferede et al. 0.17 s in real flight). Large L1 gains over a properly tuned B2 would look suspicious unless they come from faults INDI cannot structurally handle (reversal, slow servos, sensor faults). Claims about gains under structural novelty (G3) have little precedent.
9. The WLS active-set solver must be unrolled with fixed iterations to vmap. A wrong termination or a missing check that the solution is feasible gives silent sub-optimal allocations. Unit-test against a reference QP solver, such as scipy lsq_linear with bounds, on random problems.
10. An INDI P term on the raw gyro with a high gain amplifies vibration noise (aerojax adds 0.02×vib to the gyro). On the filtered rate it adds about 19 ms of delay. Pick one deliberately and keep it the same across airframes.
11. Not confirmed in this session (search budget exhausted; I could not open full texts): Smeur 2016's exact filter values (ωn≈50 rad/s, ζ≈0.55, 512 Hz); Grondman's sampling, filter and synchronisation details; Paparazzi's WLS default weights and γ; the journal venue of Tal & Karaman; whether Smeur 2020 itself uses WLS; Sun et al. (T-RO 2022) on whether an INDI inner loop is decisive for NMPC and flatness-based control; the 'Learned INDI' PMLR 2026 paper cited in docs/01. The gain rules and stability conditions above are my own derivations and should be checked numerically before they are quoted.

## Open questions

1. Will aerojax get per-surface effectors (split ailerons and elevons, shared elevon travel) before the M3 A/B, or is the hard-over headline re-scoped to the rudder or to a partial or offset aileron?
2. What information budget does B2 get: nominal servo τ and delay plus nominal derivatives (the autoconfig rule), or self-identification from an excitation manoeuvre? Should G4/Rascal use JSBSim aero data or online identification?
3. Should L1 output angular-acceleration demands into the shared INDI+WLS (RL over INDI, inheriting INDI's robustness) or surface-moment demands into a non-incremental WLS? Which one carries the headline claim, and which is the ablation?
4. Does aerojax's linear-derivative (Beard & McLain) aero represent loss of control effectiveness and aileron reversal near or after stall well enough for L2 upset-recovery claims and for INDI's r<0 failure mode?
5. Should B2 include a reversed-surface detector and classical FDI from M4, or should M3's B2 be pure INDI-WLS, with fault detection and isolation added in M4, so as not to borrow from the M4 schedule?
6. How should B2 hyperparameters be tuned under a compute budget comparable to the L1 training compute, so that reviewers accept the comparison as fair?
7. Is servo-position feedback worth modelling as an optional sensor (some digital servos report position)? It would let INDI use measured u0 and make jams observable, in line with real fly-by-wire practice (Grondman, Citation).
8. Add a hybrid INDI (model plus sensor complementary filter, Kumtepe 2022) or the Steffensen NDI-with-actuator form as a third classical variant, given the marginal time-scale separation on small UAVs?
