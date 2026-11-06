# Report 3: Massively parallel PPO in JAX

*Part of the M3 literature review, 6 Nov 2026 ([README](README.md)). The brief and the findings are as the research agent returned them; local paths are made repository-relative. Where this report and the spec differ, [docs/09](../../09_m3_technical_spec.md) holds.*

## Brief

Massively parallel PPO in JAX for aero-adapt M3 (L1 adaptive policy, L2 recovery): libraries, published throughput, large-batch hyperparameters, recurrent PPO, termination vs truncation, checkpointing, common bugs, and running cheaply on rented GPUs. Repo context read: docs/03 §4.2, §5 and §7.2; docs/validation/m2_report.md V15; src/aeroadapt/sim/aerojax/{sim,trim}.py; src/aeroadapt/core/quaternion.py. No simulations were run. Web search hit the session's shared limit partway through, so claims marked confirmed=false come from memory and need checking in the named source.

## Key findings (21 of 23 confirmed in the session)

### 3.1 · confirmed

PureJaxRL's speed comes from putting the environment and the PPO learner inside one jit and vmapping both over environments and over whole training runs (seeds or hyperparameters). The headline 1000x and 4000x figures count many vectorised runs against a CPU-environment PyTorch baseline. They are not per-run speedups.

- *Evidence:* PureJaxRL README: over 1000x faster than standard PyTorch RL implementations when running many agents in parallel, and 2048 PPO agents train in half the time of one PyTorch PPO agent on one GPU. The JaxMARL paper quotes PureJaxRL as about 10x per run (same-GPU env and learner versus a CPU env) and 4000x when runs are vectorised. There is no separate PureJaxRL paper: the repo cites DPO (NeurIPS 2022), and JOSS-style citations call it Lu 2023.
- *Source:* PureJaxRL, github.com/luchris429/purejaxrl; Rutherford et al., JaxMARL, NeurIPS 2024 D&B, arXiv 2311.10090

### 3.2 · confirmed

JaxMARL gives about 14x faster wall-clock per run than existing approaches, and up to 12,500x when many training runs are vectorised. Its IPPO implementation is PureJaxRL-style.

- *Evidence:* The arXiv v5 abstract (Nov 2024) says about 14x faster, and up to 12,500x with vectorised runs. Earlier versions phrased 12,500x as a per-run figure, so quote v5. Algorithms: IPPO, MAPPO, QMIX, VDN, IQL.
- *Source:* Rutherford et al., JaxMARL, NeurIPS 2024 Datasets & Benchmarks, https://arxiv.org/abs/2311.10090

### 3.3 · confirmed

Craftax published wall-clock figures for end-to-end JAX PPO on a single GPU: 1 billion environment interactions in under an hour. The PPO-RNN baseline in Craftax_Baselines is a copy of PureJaxRL's ppo_rnn.py.

- *Evidence:* Abstract: PPO with 1B interactions finishes in under an hour on one GPU and reaches about 90% of optimal reward on Craftax-Classic. Craftax-Classic is up to 250x faster than the Python Crafter.
- *Source:* Matthews et al., Craftax, ICML 2024 (PMLR 235), https://arxiv.org/abs/2402.16801; github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py

### 3.4 · confirmed

Brax PPO (a JAX PPO compiled together with the physics) reaches millions of simulator steps/s on one accelerator, and Ant trains to a good score in about 10 s versus about 3 h with MuJoCo on a workstation.

- *Evidence:* Google Research blog (Jul 2021): Ant reaches a score of about 4000 in about 10 s versus about 3 h, a claimed 100-1000x speedup. The scaling plots used a 4x2 TPU v3. The paper abstract says millions of steps/s on one accelerator for Ant. Exact Humanoid minutes were not retrieved.
- *Source:* Freeman et al., Brax, NeurIPS 2021 D&B, https://arxiv.org/abs/2106.13281; https://research.google/blog/speeding-up-reinforcement-learning-with-a-new-physics-simulation-engine/

### 3.5 · confirmed

MuJoCo Playground (MJX + Brax PPO) trains most tasks in minutes on one GPU, around 10 min on a single RTX 4090 for some tasks. Its technical report has a PPO steps/s table across RTX 4090, A100 and H100. The DM Control PPO config is 2048 envs, unroll 30, batch_size 1024 (trajectories per minibatch), 32 minibatches, 16 updates per batch, reward_scaling 10, and discount 0.995 on some tasks.

- *Evidence:* Report: ten minutes on a single RTX 4090 for one setup, 30 min on two 4090s for another, and wall-clock for LeapCubeReorient across 4090, A100 and H100. I could not retrieve the throughput table's values. The DMC config numbers come from a secondary SLM-Lab comparison of the official config. From memory (unconfirmed): DMC lr 1e-3 and entropy 1e-2. Go1 joystick: 8192 envs, batch 256, unroll 20, 32 minibatches, 4 updates, gamma 0.97, lr 3e-4, entropy 1e-2, max_grad_norm 1.0, policy and value nets (512,256,128), and a privileged_state critic input (asymmetric actor-critic).
- *Source:* Zakka et al., MuJoCo Playground, RSS 2025, https://arxiv.org/abs/2502.08844 and https://playground.mujoco.org/assets/playground_technical_report.pdf; SLM-Lab doc https://huggingface.co/datasets/SLM-Lab/benchmark/blob/main/docs/phase5_spec_research.md

### 3.6 · **not confirmed** (memory or secondary)

In Brax PPO, env steps per update = batch_size x num_minibatches x unroll_length, and one minibatch = batch_size trajectories of unroll_length. With the DMC config that is about 983k env steps per update: each of 2048 envs collects 16 unrolls of 30 steps, followed by 16 epochs of 32 minibatches of 30,720 samples.

- *Evidence:* The SLM-Lab doc confirms that batch_size means trajectories per minibatch. The rest is my reading of brax/training/agents/ppo/train.py from memory. Getting this wrong when porting configs changes the effective batch by 10-30x.
- *Source:* github.com/google/brax brax/training/agents/ppo/train.py (from memory); SLM-Lab doc above

### 3.7 · confirmed

Massively parallel PPO for legged robots (Rudin et al.) set the large-batch recipe. The batch is B = n_envs x n_steps, with few steps per env. Too few steps fails to converge because GAE needs multi-step returns. Minibatches of tens of thousands of samples stabilise learning without adding wall-clock time. Everything stays on the GPU.

- *Evidence:* All four points above are confirmed from the paper text. From memory (unconfirmed): 4096 envs, 24 steps per env, 5 epochs, 4 minibatches, clip 0.2, gamma 0.99, lambda 0.95, entropy 0.01, an adaptive-KL learning rate with target 0.01, ELU nets (512,256,128), flat terrain in under 4 min and rough terrain in about 20 min on one GPU. The rsl_rl docs confirm num_mini_batches default 4, desired_kl 0.01 and schedule 'adaptive'.
- *Source:* Rudin et al., Learning to Walk in Minutes Using Massively Parallel Deep RL, CoRL 2021, https://arxiv.org/abs/2109.11978; rsl_rl configuration guide https://leggedrobotics.github.io/rsl_rl/guide/configuration.html; Isaac Lab rsl_rl rl_cfg source

### 3.8 · confirmed

PPO's asymptotic performance stops improving as parallel environments are added. Past a point, extra GPU capacity is better spent on separate seeds, hyperparameters or split policies (SAPG) than on a bigger batch.

- *Evidence:* SAPG Fig. 2: PPO plateaus as batch size grows, because sampled actions cluster near the mean and the data becomes redundant. SAPG splits envs into blocks (leader and followers) and combines them by importance sampling. It succeeds on dexterous tasks where PPO gets zero success. ICML 2024 oral.
- *Source:* Singla, Agarwal, Pathak, SAPG, ICML 2024 (PMLR 235), https://arxiv.org/abs/2407.20230

### 3.9 · confirmed

PPO can be made batch size-invariant by decoupling the proximal policy from the behaviour policy (PPO-EWMA). Changes in batch size are then compensated by formulaically adjusting the Adam step size, the EWMA decay and advantage normalisation. This is the principled route for changing num_envs without re-tuning.

- *Evidence:* Batch size-invariance is achieved by adjusting Adam's step size, beta_prox and advantage normalisation; the method also makes more efficient use of stale data.
- *Source:* Hilton, Cobbe, Schulman, Batch size-invariance for policy optimization, NeurIPS 2022, https://arxiv.org/abs/2110.00641

### 3.10 · confirmed

The large on-policy ablation study (Andrychowicz et al.) gives the default choices for continuous-control PPO.

- *Evidence:* Confirmed: more than 50 choices and more than 250,000 agents on 5 continuous-control tasks, with interactions such as lr x minibatch size. Specific recommendations from memory (unconfirmed): always normalise observations; separate actor and critic networks; PPO clip about 0.25; GAE lambda 0.9; last policy layer initialised 100x smaller; softplus std with a negative offset (initial std about 0.5); Adam lr 3e-4; tanh activations for shallow nets; no PPO-style value clipping.
- *Source:* Andrychowicz et al., What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study, ICLR 2021, https://arxiv.org/abs/2006.05990

### 3.11 · confirmed

The '37 implementation details' post is the checklist for continuous-action and LSTM PPO correctness.

- *Evidence:* Confirmed: 13 core, 9 continuous-action, 5 LSTM and 9 Atari details. From memory (unconfirmed): state-independent log-std; clip sampled actions to bounds but store the unclipped sample for log-prob; obs normalisation with clip ±10; reward scaling by running return std with clip ±10; orthogonal init (policy head 0.01, value head 1.0); Adam eps 1e-5; linear lr annealing; grad-norm clip 0.5. LSTM: reset the state at episode boundaries and build minibatches over environments, never over individual timesteps.
- *Source:* Huang et al., The 37 Implementation Details of PPO, ICLR Blog Track 2022, https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details; github.com/vwxyzjn/ppo-implementation-details

### 3.12 · confirmed

Hitting a time limit is not a terminal state. Use partial-episode bootstrapping (bootstrap from the value of the final state at a timeout, not at a real termination). Brax's GAE handles truncation by masking: it zeroes the TD error and the advantage on truncated steps. PureJaxRL's GAE treats any done (including a timeout) as terminal.

- *Evidence:* Pardo et al.: ignoring the distinction causes state aliasing and instability. With PEB, Hopper trained on 200-step episodes kept hopping for about 1e6 steps. The Brax compute_gae structure (delta \*= 1 - truncation; acc masked the same way) is confirmed via a third-party port. PureJaxRL's done = terminal is from memory. Gymnasium's rule: no bootstrap on termination, bootstrap on truncation.
- *Source:* Pardo et al., Time Limits in RL, ICML 2018 (PMLR 80), https://arxiv.org/abs/1712.00378; Brax compute_gae (github.com/google/brax, brax/training/agents/ppo/losses.py); skrl PPO docs (time_limit_bootstrap flag)

### 3.13 · confirmed

The standard JAX recurrent PPO is PureJaxRL's ScannedRNN. A GRUCell inside nn.scan gets the episode done flags packed with the embedded observation. At flagged steps, jnp.where swaps the carry for a freshly initialised zero state before the cell runs. initialize_carry uses a dummy key because the init is zeros.

- *Evidence:* Confirmed via the Craftax_Baselines copy and the DeepWiki summary of purejaxrl/ppo_rnn.py. I did not confirm the exact alignment (that the flag is the previous step's done, 'last_done', so the reset happens on the first obs of the new episode). Check the original before porting.
- *Source:* github.com/luchris429/purejaxrl (ppo_rnn.py); https://deepwiki.com/luchris429/purejaxrl/3.1.2-ppo-with-rnn

### 3.14 · confirmed

For long recurrent sequences on GPU, an S5 state-space layer with a resettable associative operator keeps parallel-scan speed while allowing resets at episode boundaries inside a trajectory.

- *Evidence:* S5 with resets ran more than 5x faster than RNNs on POMDPs, with strong meta-RL on randomly projected continuous-control tasks. Code is at luchris429/s5rl (also cited as popjaxrl).
- *Source:* Lu et al., Structured State Space Models for In-Context RL, NeurIPS 2023, https://arxiv.org/abs/2303.03982; github.com/luchris429/popjaxrl

### 3.15 · confirmed

rejax and Stoix/Mava (Anakin architecture) are maintained pure-JAX alternatives. rejax exposes whole training algorithms to jit, vmap and pmap, so vmapping over seeds or hyperparameters is a one-liner. Anakin runs env and learner fully compiled on the accelerator.

- *Evidence:* rejax README: PPO, SAC, DQN, PQN, IQN, TD3; an example trains 300 SAC agents at once; speedup plots against CleanRL. I could not confirm that a rejax paper exists. Stoix is a community project adapted from Mava, not an InstaDeep product, and offers Anakin (JAX envs) and Sebulba (Envpool or Gymnasium). I did not confirm a recurrent PPO in Stoix. Gymnax (Lange 2022) supplies JAX classic-control, MinAtar and bsuite test environments (not re-checked).
- *Source:* github.com/keraJLi/rejax; pypi.org/project/rejax; Hessel et al., Podracer architectures, arXiv 2104.06272 (2021); github.com/instadeepai/Mava; Stoix (github.com/EdanToledo/Stoix)

### 3.16 · confirmed

Discovered Policy Optimisation is a closed-form drift function, found by meta-learning in the Mirror Learning space and evolved with PureJaxRL tooling. It is a near drop-in replacement for PPO's clipped objective and matches or beats PPO on Brax.

- *Evidence:* LPO is the meta-learned drift. Analysing it showed 'rollback' and optimism for low-reward actions, which led to the closed-form DPO. It is state of the art on Brax with transfer to unseen settings, and better than PPO on MinAtar. From memory (unconfirmed): drift is ReLU((r-1)A - alpha\*tanh((r-1)A/alpha)) for A>=0 and ReLU(log(r)A - beta\*tanh(log(r)A/beta)) for A<0, with alpha=2 and beta=0.6. PureJaxRL ships dpo_continuous_action.py.
- *Source:* Lu et al., Discovered Policy Optimisation, NeurIPS 2022, https://arxiv.org/abs/2210.05639

### 3.17 · confirmed

Off-policy algorithms built for massive parallelism are now credible PPO alternatives on wall-clock. FastTD3 and FastSAC beat PPO on humanoids in IsaacLab and MuJoCo Playground. PQN (LayerNorm DQN with no replay buffer or target network, run on vectorised envs) matches PPO-RNN on Craftax.

- *Evidence:* FastTD3: many HumanoidBench tasks in under 3 h on one A100, faster than PPO in IsaacLab and Playground, with the largest gap on rough terrain with domain randomisation; defaults tuned for one A100 80GB. A follow-up: FastSAC/FastTD3 sim-to-real humanoid in about 15 min on one RTX 4090. PQN: up to 50x faster than DQN, competitive with Rainbow on Atari, PPO-RNN on Craftax and QMix on SMAX; code at mttga/purejaxql.
- *Source:* Seo et al., FastTD3, arXiv 2505.22642 (2025); Gallici et al., Simplifying Deep TD Learning, ICLR 2025, https://arxiv.org/abs/2407.04811

### 3.18 · confirmed

The closest precedent for the L1 teacher-to-student plan is RAPTOR: one RL teacher per platform, distilled into one tiny recurrent student that adapts zero-shot to unseen real quadrotors.

- *Evidence:* 1,000 sampled quadrotors, each with its own RL teacher, distilled into a 3-layer student of 2,084 parameters with a recurrent hidden layer. It adapts within milliseconds and was tested on 10 real quadrotors from 32 g to 2.4 kg with different motors, frames, propeller counts and flight controllers. The same group's 'Learning to Fly in Seconds': trained in 18 s on a laptop, asymmetric actor-critic, curriculum, and a simulator running about 5 months of flight per second on a laptop GPU. The algorithm (TD3) is not confirmed.
- *Source:* Eschmann, Albani, Loianno, RAPTOR: A Foundation Policy for Quadrotor Control, arXiv 2509.11481 (2025, rev. 2026); Learning to Fly in Seconds, IEEE RA-L 2024, https://arxiv.org/abs/2311.13081

### 3.19 · confirmed

NeuralPlane, the GPU-parallel fixed-wing RL platform, reports about 0.2 s per simulation step at 10^6 parallel aircraft (about 5M aircraft-steps/s) and uses PPO for heading and control tasks.

- *Evidence:* Abstract: 0.2 s per step at 10^6 scale; hierarchical RL plus classical control templates. A garbled appendix snippet suggests a recurrent hidden size of 128 (unverified). I did not retrieve the GPU model or the PPO hyperparameters.
- *Source:* Xue et al., NeuralPlane, NeurIPS 2024 D&B, https://proceedings.neurips.cc/paper_files/paper/2024/hash/af97b61e9fef8f55e32a2602af364d8c-Abstract.html; github.com/xuecy22/NeuralPlane

### 3.20 · confirmed

aerojax currently runs float32 at 100 Hz control with 2 x 5 ms RK4 physics substeps. On the laptop CPU, a heterogeneous fleet with faults reaches 552k simulator steps/s and 225k closed-loop steps/s at batch 1024. The GPU figure is still pending. Trim runs in x64 with Python-level maths, so it belongs in an offline reset pool, not inside a jitted auto-reset.

- *Evidence:* docs/validation/m2_report.md V15 (jax 0.11.2, arm64): batch 64/256/1024 gives 444k/511k/552k simulator steps/s and 210k/228k/225k closed-loop steps/s. sim.py: substeps=2. trim.py: `with jax.enable_x64(True)` and `math.inf`/`math.cos` on Python floats. quaternion.py rotates with jnp.einsum, which is relevant to the TF32 pitfall below.
- *Source:* docs/validation/m2_report.md; src/aeroadapt/sim/aerojax/sim.py; src/aeroadapt/sim/aerojax/trim.py; src/aeroadapt/core/quaternion.py

### 3.21 · confirmed

Cloud GPU prices, Sep-Oct 2026 snapshots from secondary trackers: H100 about $1.5-2.3/h on marketplaces and $2.5-4/h for stable on-demand; A100 80GB $1.39-1.59/h on RunPod; L4 about $0.43/h on RunPod. Hyperscalers cost 2-4x more.

- *Evidence:* Vast.ai H100 $1.49-2.27/h, varying daily. RunPod Community H100 PCIe $1.99; RunPod Secure H100 SXM went from $3.29 to $3.99 between Aug and Oct. Lambda H100 SXM $3.99 per GPU-hour. RunPod A100 80GB $1.39 (PCIe) and $1.59 (SXM); Lambda A100 $2.06-2.79; Vast A100 spot floor about $0.39. L4 $0.43 on RunPod versus $1.15 on GCP. Azure H100 up to about $7-12/h. RTX 4090/5090 and L40S rental prices were not retrieved.
- *Source:* Secondary trackers: getdeploying.com H100 vs L4; intuitionlabs.ai H100 rental comparison; jarvislabs.ai A100 price 2026; tech-insider.org RunPod vs Lambda vs Vast 2026; keenable.ai H100 effective price reports

### 3.22 · confirmed

JAX GPU operations: XLA preallocates GPU memory (set XLA_PYTHON_CLIENT_MEM_FRACTION to 0.8-0.9). The latency-hiding scheduler is on by default at O1 in recent toolchains. Profile-guided latency estimation (PGLE) exists. Orbax CheckpointManager handles max_to_keep and save_interval_steps, saves asynchronously (call wait_until_finished before exit) and restores against an abstract pytree template.

- *Evidence:* From the JAX GPU performance tips page, NVIDIA's JAX-Toolbox guide and Orbax checkpoint 101. Unconfirmed (search limit reached): JAX's default f32 matmul precision on A100/H100 allows TF32, so set jax_default_matmul_precision='highest' for physics code. Raw typed PRNG keys need jax.random.key_data to serialise.
- *Source:* https://docs.jax.dev/en/latest/gpu_performance_tips.html; https://docs.nvidia.com/jax-toolbox/performance-profiling/gpu-performance; https://orbax.readthedocs.io/en/latest/guides/checkpoint/orbax_checkpoint_101.html

### 3.23 · **not confirmed** (memory or secondary)

Brax's AutoResetWrapper resets each env to the same cached first state on every done, rather than drawing a new random initial state. Diversity then comes only from the initial batch, refreshed via num_resets_per_eval or a separate domain-randomisation wrapper.

- *Evidence:* From memory of brax/envs/wrappers/training.py (first_pipeline_state/first_obs swapped in with jnp.where). I could not re-verify this because the search limit was reached. If true, porting Brax's wrapper unchanged would sharply cut fleet x fault x wind diversity.
- *Source:* github.com/google/brax brax/envs/wrappers/training.py (unverified this session)

## Recommendations for M3

1. Write a lean, PureJaxRL-style PPO in the repo (about 500-800 lines: an `aeroadapt.learn` package) rather than adopting Brax's trainer. L1 needs things Brax PPO does not provide cleanly: a GRU or recurrent student, an asymmetric privileged critic, truncation-correct GAE with final-observation bootstrapping, a device-resident reset pool, and curriculum state. Use PureJaxRL ppo_continuous_action.py and ppo_rnn.py, Brax losses.py and rejax as references. Validate the trainer on a known task (gymnax Pendulum or a Brax/Playground env) against published curves before touching aerojax.
2. Use the Anakin structure: one jitted `train_chunk(state) -> (state, metrics)` that runs K PPO updates (for example 10-50) with lax.scan, and a thin Python loop around it for logging, evaluation, checkpointing and pre-emption handling. Do not compile the whole run as one scan (pure PureJaxRL style): you lose mid-run checkpoints and logs on spot instances. Return metrics from the chunk instead of calling float() or printing inside the loop.
3. Environment API: pure `reset(key, pool_idx)` and `step(env_state, virtual_cmd, key)`. The adaptive-allocation estimator state, the actuator and fault state, and the Dryden filters all live in env_state, and the per-env airframe parameters are a gathered slice of a stacked fleet pytree on the device. Auto-reset draws a fresh random entry from a precomputed reset pool of trimmed initial conditions (airframe x speed x altitude x fault x wind seed), built offline with the existing x64 trim. Do not reset to a cached first state, and do not run trim inside jit. jnp.where auto-reset evaluates the reset branch for every env on every step, so reset must cost a gather and nothing more.
4. Rates and horizon: run the policy at 50 Hz (each action held for two 100 Hz control ticks, i.e. four 200 Hz physics steps), with allocation at every tick. Start with gamma=0.99 at 50 Hz (about a 2 s horizon) and lambda=0.95, then sweep gamma to 0.995 (about 4 s) because airspeed and height energy exchange and the phugoid are slow (periods of several seconds or more). Use 32-step rollouts (0.64 s) and go to 64 only if value explained-variance stays under 0.8. Make episodes 30-60 s with per-env jittered time limits so envs do not all truncate on the same update.
5. Starting teacher configuration: num_envs 8192 (sweep 4096/16384); unroll 32, giving 262k samples per update; 4 epochs x 8 minibatches of about 32k samples; Adam lr 3e-4 with an adaptive-KL schedule (target KL 0.01; divide or multiply lr by 1.5 when KL exceeds 2x or falls below 0.5x the target; bound lr to [1e-5, 1e-3]); clip 0.2; max_grad_norm 1.0; entropy 0-0.003; value coefficient 0.5-1.0 with a separate critic and no value clipping at first; per-minibatch advantage normalisation; Welford obs normalisation clipped to ±5, frozen at evaluation and saved with the checkpoint; rewards scaled so per-step reward is O(0.1-1). Teacher actor: MLP (256,256,128) ELU. Critic: (512,256,128) fed privileged inputs (true airframe parameters, log-scaled; fault kind as one-hot plus severity; true wind; true state).
6. Actions: use a state-independent log-std initialised to about ln(0.5) in normalised action units, with a floor (std ≥ 0.05). Either tanh-squash and store the pre-tanh sample for log-prob (Brax-style, so atanh is never taken of ±1), or sample a Gaussian, clip, and store the unclipped sample (CleanRL-style). Normalise the virtual commands per airframe (for example, angular-acceleration demands scaled by each airframe's nominal control authority, thrust as a fraction of maximum), so that one output range means the same thing from 0.8 to 15 kg. That normalisation is what makes the G2 and G3 transfer possible. Add action-rate or smoothness penalties (CAPS-style temporal and spatial terms) and report surface-rate RMS beside the PID baseline.
7. Termination and truncation: terminate with no bootstrap on ground or terrain impact, structural or envelope limits, and non-finite state (also replace the state from the reset pool). Truncate on the time limit and bootstrap from V(final_obs, final_privileged_obs): the auto-reset wrapper must keep the pre-reset observation in info. Brax-style masking of truncated steps is an acceptable simpler alternative with 1,500-3,000-step episodes. Write unit tests of GAE on hand-built sequences with a termination mid-rollout, a truncation mid-rollout, and both at the rollout edge. If rewards are mostly negative tracking costs, add an alive bonus or a crash penalty larger than the expected remaining cost, or the policy learns that crashing ends the costs.
8. GRU student (target under 50k parameters): input MLP 64, then GRU 64-96, then a linear head. That is about 20-40k parameters with roughly 40-60 inputs (sensors, previous virtual command, command targets). Train it by DAgger-style distillation: the student flies in the same vmapped env and the teacher labels actions, with MSE on actions and optionally on the teacher's latent (RMA). Use BPTT through lax.scan over 5-10 s windows (250-500 steps at 50 Hz). Reset the carry with jnp.where(is_first_step, init, carry) before the cell, build minibatches over envs (whole sequences) and never over timesteps, and store the carry at each segment start. Use jax.checkpoint (remat) on the scanned cell if memory binds (16k envs x 500 steps of activations is several GB). Fine-tune with recurrent PPO and the privileged critic only if the distillation gap to the teacher exceeds a few percent on G1. Consider S5 with resets only if windows grow past about 500 steps and the GRU scan dominates time.
9. L2 recovery: same trainer. Use a curriculum over initial upset state and fault severity, with the curriculum level checkpointed. Hand-back to L1 is a termination with a bonus, not a truncation. Upsets drive aero tables into extrapolation and produce NaNs, so put jnp.isfinite guards plus nan_to_num on observations in the env step, count NaN resets per update and alert when the count rises.
10. First GPU session (about $5-15): run the pending throughput notebook, then benchmark the full training step (env + allocation + policy + PPO update) at num_envs {2k, 8k, 32k, 128k} on L4, RTX 4090/5090 or L40S, A100 80GB and H100, about 30 min each. Choose hardware by dollars per billion training steps, not by sim-only steps/s. Physics in float32 with table lookups is bound by FP32 throughput and memory bandwidth, not tensor cores, so consumer Ada/Blackwell cards may beat an A100 per dollar. Pay for an H100 only if it is more than 2x an A100.
11. Precision on A100/H100: set jax.config.update('jax_default_matmul_precision','highest'), or pass precision=HIGHEST to the einsum and @ calls in core/quaternion.py and atmosphere.py. Then rerun the existing N-version parity and JSBSim cross-check on the GPU before any training. Stay in float32 and do not use bf16 for networks this small.
12. Fill the GPU with seeds: once the steps/s and return curve flattens with num_envs (the SAPG saturation result), vmap 5-10 seeds or a small hyperparameter grid in one process (the PureJaxRL/rejax pattern). That delivers the plan's ≥5 seeds with Wilson CIs at near-zero extra cost and avoids batches so large that PPO wastes them.
13. Checkpointing and operations: Orbax CheckpointManager (async; max_to_keep=3 plus the best-by-evaluation) every about 10 min to a network volume or object storage. Each checkpoint holds actor and critic params, optimiser state, obs-normaliser stats, RNG key data, update counter, curriculum state, config hash, git SHA, and JAX/CUDA versions. Put JAX's persistent compilation cache on the volume so a pre-empted spot instance does not recompile for minutes. Add a SIGTERM handler that flushes a checkpoint. Use a pinned Docker image (jax[cuda12] at the version the CPU results came from). Export the frozen policy as numpy weights with the normaliser folded in, and test parity on recorded inputs.
14. Sanity ladder, with the PID baseline compared at every rung: (1) one nominal airframe, no wind, no faults, attitude tracking, which should learn in minutes; (2) the whole training fleet; (3) plus Dryden, gusts and shear; (4) plus single faults; (5) plus randomised delay and noise. Monitor approx-KL (about 0.01), clip fraction (0.05-0.3), value explained-variance (>0.8), policy std, grad norm, terminations by cause, return per airframe class, NaN resets and SPS.
15. Budget expectation (an estimate pending the GPU benchmark): at 1-3M training steps/s, a 1B-step teacher run takes about 6-17 min, under $1-2 at A100 marketplace rates. All of M3 (teachers, students, L2, 5-10 seeds, ablations, sweeps, retries) plausibly fits in about $100-500. Debugging time is the binding constraint, not GPU dollars, so favour fast iteration (small configs on an L4 or 4090) over big single runs.
16. Optional ablations if time allows: DPO in place of the PPO clip (a few lines of change); PPO-EWMA if you change num_envs a lot. Keep FastTD3 or PQN as a fallback only if PPO plateaus on the fault-heavy distribution, since FastTD3 shows better wall-clock than PPO on humanoids.

## Risks and pitfalls

1. Treating truncation as termination (PureJaxRL's default done handling) biases value estimates near time limits. It also bootstraps from the reset observation if masks are wrong. This matters most for energy-state (airspeed and altitude) objectives with gamma ≥ 0.99.
2. Reset diversity and cost: a Brax-style auto-reset to a cached first state (memory-based claim) collapses fleet x fault x wind diversity. Any heavy work in reset, such as trim or scenario sampling, runs for every env on every step under jnp.where.
3. Synchronised episodes: all vmapped envs starting and truncating together put periodic non-stationarity into PPO batches. Jitter the episode lengths and the initial step counters.
4. PRNG misuse: reusing or not splitting keys across envs, substeps or seeds gives identical turbulence or faults in many lanes, which silently shrinks the effective batch.
5. Recurrent PPO bugs: shuffling timesteps instead of envs into minibatches; off-by-one done alignment (the carry resets a step late or leaks across episodes); not resetting the carry at evaluation; starting a minibatch sequence from the wrong stored carry.
6. Action-distribution bugs: log-prob computed on clipped or squashed actions (atanh(±1) gives inf or NaN); tanh-Gaussian entropy has no closed form (Brax estimates it by sampling with the log-det-Jacobian); a high entropy coefficient gives bang-bang, jittery surfaces that look poor next to PID and stress servo models.
7. The obs normaliser is the silent checkpoint killer: stats not saved, updated during evaluation, applied twice, or differing between teacher and student. Privileged parameters spanning orders of magnitude (mass 0.8-15 kg, inertias) need log-scaling before normalisation.
8. Possible TF32 on A100/H100 for f32 einsum/matmul (default precision, unconfirmed this session) would degrade the quaternion rotations in core/quaternion.py and the Dryden filter, and break CPU-GPU parity.
9. NaN or inf from upsets, deep stall or inverted flight (L2 especially) spreads through the batch into the parameters within one update. Guard and terminate per env. Use jax_debug_nans only for debugging, because it is slow.
10. PPO will exploit simulator artefacts, such as aero-table extrapolation at high alpha or beta, or allocation-estimator loopholes. G4 (Rascal in JSBSim) and the existing JSBSim cross-checks are the guard, so run them early, not only at the end of M3.
11. Large-batch saturation: more envs is not better past a point (SAPG). Scaling num_envs without retuning lr, epochs and minibatches can hurt (Hilton et al.).
12. Memory for student BPTT: 16k envs x 500-step windows x (obs + hidden) plus GRU activations can exceed 40 GB. Use remat, fewer envs or shorter windows. Remember XLA preallocates about 75% of GPU memory by default.
13. Compile time: scanned rollouts with RK4 substeps and allocation can take minutes to compile. Without a persistent compilation cache, every spot pre-emption pays that again. Changing Python-level config or shapes triggers recompiles.
14. Host-device syncs from per-step logging or Python control flow on traced values can cut throughput by 10x or more.
15. Misreading published throughput: the 1000x/4000x/12,500x headlines compare vectorised multi-run JAX against slow CPU-env baselines. Quote aero-adapt's own measured training steps/s and $/run in the M3 report.
16. Spot or marketplace instances (Vast, RunPod Community) get pre-empted and vary in reliability. Prices moved more than 20% between Aug and Oct 2026, so re-check before budgeting.
17. Search coverage gap: the session's web-search limit was reached before several details could be checked: the Brax AutoResetWrapper behaviour, JAX TF32 defaults, Rudin's exact hyperparameters and timings, the Playground throughput table values, and the exact Andrychowicz and '37 details' recommendations. Those are marked confirmed=false or 'from memory' and should be checked in source before being cited in docs.

## Open questions

1. What closed-loop training throughput (env + allocation + GRU or MLP policy + PPO update) does aerojax reach on each GPU type at 8k-128k envs? This is the pending notebook and decides the hardware and num_envs.
2. Does JAX 0.11 on A100/H100 run the vmapped 3x3 einsum in quaternion.py and the @ in atmosphere.py under TF32 by default, and does that break the existing parity tolerances?
3. Does current Brax's AutoResetWrapper still reset to a cached first state? This matters if any Brax wrapper is reused.
4. What discount and horizon suit fixed-wing tracking at 50 Hz: 0.99 (2 s) versus 0.995 (4 s) versus 0.998? Phugoid and TECS-like energy dynamics argue for longer; variance argues for shorter. It needs a sweep.
5. Teacher design: one privileged teacher across the whole fleet, or per-class or per-airframe teachers distilled RAPTOR-style (RAPTOR used 1,000 per-platform teachers)? Which distils better into a 50k-parameter GRU for G3 (flying wing held out)?
6. Should the allocation layer's online effectiveness estimator be inside the RL environment (the policy sees its effects), or run with privileged truth during teacher training and estimated during student training? This affects what 'history-only' means for the student.
7. Is distillation alone enough for the student, or does it need a recurrent-PPO fine-tune with the asymmetric critic to close the teacher-student gap under hard-over faults?
8. Which smoothness regulariser (action-rate penalty, CAPS, low-pass in allocation) keeps surface activity comparable to tuned PID without hurting fault recovery?
9. For L2, how should hand-back to L1 and the 'recovered' condition be defined as terminations, and what upset-state curriculum schedule avoids the policy learning only easy upsets?
10. Is PPO's sample cost acceptable for the fault-heavy distribution, or would FastTD3 or PQN, which beat PPO on wall-clock elsewhere, be worth a spike?
11. What are current rental prices for RTX 4090/5090 and L40S? Not retrieved, and possibly the best $/step for this FP32 physics workload.
