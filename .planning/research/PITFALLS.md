# Domain Pitfalls

**Domain:** Atari Imitation Learning Benchmark — FTRL vs DAgger empirical study
**Researched:** 2026-03-19
**Confidence:** HIGH (based on direct codebase inspection + verified library documentation)

---

## Critical Pitfalls

These cause silent failures, invalid results, or complete rewrites.

---

### Pitfall 1: Atari Observation Space Mismatch Between Expert and Learner

**What goes wrong:** The HuggingFace-loaded PPO expert policy was trained with SB3's `make_atari_env`, which applies `AtariWrapper` (grayscale, 84x84, channel-first) plus `VecFrameStack` (stacking 4 frames, producing shape `(4, 84, 84)` in `uint8`). If the training environment for DAgger/FTRL uses a different wrapper stack — even subtly different (e.g., RGB instead of grayscale, or 84x84 without grayscale newaxis, or frame stack outside the `VecEnv`) — `SimpleDAggerTrainer.__init__` will raise a `ValueError: Mismatched observation space` or, worse, silently accept a compatible-shaped but semantically different space.

**Why it happens:** The `imitation` library's environment ingredient (`environment.py`) does not automatically apply Atari wrappers. It calls `util.make_vec_env` with the `gym_id` but adds only `RolloutInfoWrapper`. Atari-specific preprocessing (grayscale, resize, frame skip, frame stack) must be added explicitly via `post_wrappers` or `env_make_kwargs`. The HuggingFace expert was trained with the full SB3 Atari stack. Any mismatch produces a policy that receives observations it was never trained on.

**Consequences:**
- If shapes match by accident (e.g., both 84x84x4 but one is HWC, one is CHW), the learner trains on wrong input ordering — policy will produce near-random actions but no error is raised.
- If shapes differ, `SimpleDAggerTrainer` raises on construction, which is at least visible.
- Normalized scores will be wrong: the trained agent scores near-random but the "expert" score in the same run was correctly computed, giving a misleading normalized value close to 0.

**Prevention:**
- Use `stable_baselines3.common.env_util.make_atari_env` as the base, not generic `make_vec_env`, for Atari games. This applies `AtariWrapper` and correct preprocessing automatically.
- Verify observation space after environment creation: `assert venv.observation_space == expert_policy.observation_space`.
- Load the expert policy first, then construct the environment to match its `observation_space`, not the other way around.
- For HuggingFace RL Zoo models, the expected obs space is `Box(0, 255, shape=(84, 84, 4), dtype=uint8)` (channel-last, stacked in last axis via `VecFrameStack`).

**Warning signs:**
- `ValueError: Mismatched observation space` during `SimpleDAggerTrainer` construction.
- Learner reward near random baseline from the first round despite collecting expert demonstrations.
- `expert_stats` show high return but `imit_stats` show near-random scores immediately.

**Phase:** Atari environment setup (before any algorithm runs).

---

### Pitfall 2: Double Frame-Skipping from Wrong Gym ID

**What goes wrong:** Using `PongNoFrameskip-v4` (or `*NoFrameskip-v4`) as the `gym_id` while also applying `AtariWrapper` (which applies its own 4x frame skip) results in 4x frame skip as intended. However, if someone uses `Pong-v4` (non-NoFrameskip ALE variant, which already has a built-in 2-4x frame skip) together with `AtariWrapper`, the agent effectively skips 8-16 frames per action — far slower dynamics than the expert was trained on.

**Why it happens:** The gym `Pong-v4` applies a random (2-4) frame skip at the ALE level. `AtariWrapper` documentation explicitly warns: "Use this wrapper only with Atari v4 without frame skip: `env_id = "*NoFrameskip-v4"`." But Sacred configs default to `seals/CartPole-v0` and don't enforce this constraint for Atari.

**Consequences:** Expert performance is degraded (it was trained on 4-skip dynamics), learner never matches expert behavior, normalized scores are artifically low for all methods equally, making the FTL vs FTRL comparison appear to show both methods fail.

**Prevention:**
- Always use `*NoFrameskip-v4` gym IDs for Atari (e.g., `ALE/Pong-v5` or `PongNoFrameskip-v4`).
- Verify via `env.unwrapped.game` and check the `frameskip` attribute on the ALE environment.
- Document the exact `gym_id` used for each game in Sacred config and double-check against the HuggingFace model repo name, which encodes the env ID.

**Warning signs:** Episode lengths are half what they should be. Expert achieves lower return than expected baseline. Game appears to play at "fast-forward" pace in rendered episodes.

**Phase:** Atari environment setup.

---

### Pitfall 3: FTRL Regularization Coefficient Lambda — Wrong Scale Destroys Learning

**What goes wrong:** FTRL (Proposition 4.1, Eq. 6 in Lavington et al.) adds an L2 penalty anchored to the previous iterate. The coefficient `lambda` (regularization strength) controls the tradeoff between fitting new data and staying close to the previous policy. Setting `lambda` too large makes FTRL behave like BC on the first round's data (ignores new data). Setting it too small makes FTRL identical to DAgger (FTL), eliminating the regularization benefit and making the comparison meaningless.

**Why it happens:** The existing `LpRegularizer` in `imitation/regularization/regularizers.py` computes the L2 norm of *all model weights*, not the L2 distance from the previous iterate. The FTRL formulation from the paper requires a **proximal** term: `lambda/2 * ||theta - theta_{t-1}||^2`. This is not the same as weight decay. Naively reusing `LpRegularizer` will penalize large weights globally, not deviation from the anchor point — producing a different algorithm than the paper's FTRL.

**Consequences:**
- Lambda at wrong scale: learning plateaus immediately (too large) or shows no improvement over DAgger (too small or semantically wrong regularizer).
- Using `LpRegularizer` as-is instead of a proximal term: algorithm is no longer FTRL per the paper, invalidating the comparison.
- Results will not match Figure 4 in Lavington et al., making the study non-reproducible.

**Prevention:**
- Implement FTRL's regularization as a **proximal term**: after each round's BC update, store a frozen copy of policy weights `theta_prev`. During the next round's BC loss, add `lambda/2 * sum(||p - p_prev||^2 for p, p_prev in zip(policy.parameters(), frozen_params))`. This is distinct from `LpRegularizer`.
- Do not reuse `WeightDecayRegularizer` or `LpRegularizer` for FTRL — they are not proximal regularizers.
- Start lambda search at the scale used in Lavington et al.'s code (examine their `Atari-Experiments/atari_ex.py`). Typical useful range for neural net IL is `lambda in [1e-4, 1e-1]` relative to the BC loss scale.
- Log the ratio `proximal_term / bc_loss` each round. If it exceeds 10x, lambda is too large. If it is consistently below 0.01x, lambda is likely too small.

**Warning signs:**
- FTRL normalized score curve is identical to DAgger curve across all games (lambda too small or wrong regularizer type).
- FTRL normalized score stays flat while DAgger improves (lambda too large).
- Proximal term is much larger than BC loss in logged values.

**Phase:** FTRL implementation.

---

### Pitfall 4: Normalized Score is Undefined When Expert and Random Have Near-Zero Difference

**What goes wrong:** The normalization formula `(score - random) / (expert - random)` produces `nan` or `inf` when `expert_score ≈ random_score`. On some Atari games (e.g., early in training, or games where a random policy accidentally scores high), the denominator is near zero. The existing `sacred_output_to_markdown_summary.py` computes this per-seed without guarding against zero denominators.

**Why it happens:** The current normalization code (line 107 in `sacred_output_to_markdown_summary.py`) divides directly: `(score - random_score) / (expert_score - random_score)`. For games like Pong where a random agent scores around -21 and expert scores ~+21, the denominator is 42 — fine. But for games with high random-policy variance or where the HuggingFace expert for a game hasn't fully converged, the denominator can be unexpectedly small.

**Consequences:** A `nan` in `accumulated_normalized_scores` silently propagates through `rly.get_interval_estimates`, producing `nan` aggregate metrics. The figure generation script produces figures with missing or blank curves with no error message.

**Prevention:**
- Add a guard: if `abs(expert_score - random_score) < threshold` (e.g., 1.0), log a warning and either skip that game or use a separate normalization approach.
- Pre-validate all (game, expert_score, random_score) pairs before running the full benchmark.
- For random baselines: the `get_random_agent_score` function fetches from `HumanCompatibleAI/random-{env}` on HuggingFace. Verify these datasets exist for all 10+ games before the full run.

**Warning signs:** `nan` in normalized score columns in the markdown summary. `rly` returns `nan` confidence intervals. Missing curves in figures.

**Phase:** Normalization pipeline validation (before full benchmark run).

---

### Pitfall 5: Random Baseline Dataset Missing on HuggingFace for Chosen Games

**What goes wrong:** `get_random_agent_score` in `sacred_output_to_markdown_summary.py` loads from `HumanCompatibleAI/random-{EnvironmentName(env)}`. The `EnvironmentName` wrapper transforms the gym ID into a canonical form (e.g., `seals/Pong-v0` becomes `seals-Pong-v0`). If a random-policy dataset for a game is not hosted on HuggingFace under this naming scheme, the function raises a `datasets.exceptions.DatasetNotFoundError` during figure generation — after all training has completed.

**Why it happens:** The dataset naming is implicit: the `EnvironmentName` transformation is not documented anywhere in the codebase, and the mapping from `gym_id` to dataset name is done at analysis time. A game selection made early (e.g., using `seals/Atlantis-v0`) that lacks a HuggingFace random dataset causes figure generation to fail silently or crash.

**Consequences:** Hours of GPU training complete successfully but the analysis pipeline fails. All normalized scores for a game or the entire benchmark are unavailable.

**Prevention:**
- Before finalizing the game list, programmatically verify all required HuggingFace datasets exist: attempt `datasets.load_dataset(f"HumanCompatibleAI/random-{EnvironmentName(env)}")` for each game in the list and log the result.
- Alternatively, collect your own random-policy baselines by running 30+ episodes with `random` policy type and computing `rollout_stats`, then caching locally — this removes the HuggingFace dependency entirely and is more reproducible.
- Prefer seals-wrapped games (e.g., `seals/Pong-v0`) if HumanCompatibleAI datasets use that naming scheme; otherwise use raw ALE names consistently.

**Warning signs:** `DatasetNotFoundError` during `sacred_output_to_markdown_summary.py`. Absence of the game's dataset in the HumanCompatibleAI HuggingFace organization.

**Phase:** Game selection and normalization pipeline validation.

---

## Moderate Pitfalls

These cause incorrect results or wasted compute but not total failure.

---

### Pitfall 6: DAgger Demo Dataset Grows Unboundedly — Disk and Memory Exhaustion

**What goes wrong:** `DAggerTrainer._all_demos` accumulates all trajectories from all rounds in memory (a Python list of `Trajectory` objects). For Atari, each observation is `(84, 84, 4) uint8` — about 28KB per timestep. With 30 rounds × 5000 timesteps per round × 4 parallel envs, the in-memory demo buffer is approximately 16GB before round 30. Additionally, each trajectory is saved to disk as `.npz` files under `scratch_dir/demos/round-XXX/`, and `_load_all_demos` reloads all previous rounds' demos every round (lines 400-407 in `dagger.py`).

**Why it happens:** The `_load_all_demos` method re-reads all demos from all rounds every time demos are loaded (line 400: `for round_num in range(self._last_loaded_round + 1, self.round_num + 1)`). For long Atari runs this is not merely slow — it fills RAM. The codebase notes this explicitly in CONCERNS.md under "Trajectory Buffer Memory."

**Consequences:** OOM crash mid-run (typically around round 15-25 for full Atari runs). Loss of all training progress because `DAggerTrainer` lacks checkpointing (confirmed missing feature in CONCERNS.md). Server disk can also fill up with `.npz` files if scratch dirs aren't managed.

**Prevention:**
- Implement a rolling window over recent rounds rather than accumulating all demos. Keep only the last N rounds of data (N=10 is a reasonable tradeoff).
- Set `scratch_dir` to a path with monitored disk quota and add a disk usage check before each round.
- For the FTRL study, consider whether full data accumulation is required by the algorithm (DAgger originally uses all historical data; check if Lavington et al. use a window or full accumulation).
- Monitor peak RSS memory during the smoke-test run and extrapolate to full benchmark scale before committing to a full run.

**Warning signs:** RSS memory grows linearly per round. `round-XXX` directories accumulate rapidly. `MemoryError` during `_load_all_demos`. Disk quota warnings.

**Phase:** DAgger/FTRL implementation and smoke-test validation.

---

### Pitfall 7: Beta Schedule Terminates Too Early — DAgger Degenerates to BC After Round 15

**What goes wrong:** The default `LinearBetaSchedule(rampdown_rounds=15)` (dagger.py line 355) sets beta=0 from round 15 onward. This means from round 15 the DAgger trainer collects trajectories using only the learner policy, never the expert. For Atari, where a poorly-initialized policy produces low-diversity states early on, the transition to pure learner rollouts at round 15 may be premature — the policy hasn't yet visited enough of the state space to benefit from expert labels at those states.

**Why it happens:** The default rampdown is tuned for low-dimensional environments (CartPole, etc.). Atari requires more rounds to reach competent policy behavior, meaning the DAgger property (visiting states under the learner's distribution and querying the expert) is lost before the learner is competent enough for its rollouts to be informative.

**Consequences:** DAgger curves plateau early and underperform their theoretical maximum. The FTL vs FTRL comparison may look unfavorable to both compared to the paper's results, because the data distribution used for training degrades.

**Prevention:**
- Set `rampdown_rounds` to at least the total number of DAgger rounds (e.g., if running 30 rounds, set `rampdown_rounds=30` so beta decays uniformly over the entire run).
- Or use `ExponentialBetaSchedule(decay_probability=0.95)` for a slower decay.
- Match the beta schedule used in Lavington et al.'s Atari experiments.
- Log `dagger/round_num` and beta value each round to verify the schedule is as intended.

**Warning signs:** Normalized score improvement stops around round 15 even though training continues. Log shows beta=0 long before the final round.

**Phase:** DAgger/FTRL training configuration.

---

### Pitfall 8: Multi-GPU Experiment Management — CUDA_VISIBLE_DEVICES Not Set Per Process

**What goes wrong:** Running 4 parallel Sacred experiments (one per GPU) via tmux without explicitly setting `CUDA_VISIBLE_DEVICES` causes all processes to default to GPU 0. All 4 jobs compete for VRAM on GPU 0, causing OOM errors, while GPUs 1-3 sit idle. PyTorch's `device="auto"` selects GPU 0 when multiple GPUs are available and `CUDA_VISIBLE_DEVICES` is not set.

**Why it happens:** The project runs experiments via tmux sessions (remote execution on CC-server). The `parallel.py` script uses Ray Tune for parallelism, but if experiments are launched manually in tmux panes rather than through Ray, each process will call `device="auto"` or `utils.get_device("auto")` which picks `cuda:0` by default.

**Consequences:** 3 out of 4 GPUs sit idle. GPU 0 OOMs on Atari (large CNN + 4-frame observation stack). All 4 jobs fail with `RuntimeError: CUDA out of memory`.

**Prevention:**
- When launching experiments in tmux, prepend `CUDA_VISIBLE_DEVICES=N` to each command, where N is 0, 1, 2, 3 for the four panes.
- For Ray Tune parallelism: set `resources_per_trial={"gpu": 1}` and `ray.init(num_gpus=4)` so Ray handles GPU assignment automatically.
- Verify GPU assignment at start of each job with a quick `nvidia-smi` check or log `torch.cuda.current_device()`.
- Create a launch script that assigns one GPU per game/algorithm pair rather than relying on default device selection.

**Warning signs:** `nvidia-smi` shows GPU 0 at 100% utilization and GPUs 1-3 at 0%. `RuntimeError: CUDA out of memory` on GPU 0.

**Phase:** Multi-GPU parallel experiment runner.

---

### Pitfall 9: Sacred File Storage Observer Writes to Same Directory — Run ID Collision

**What goes wrong:** When multiple Sacred experiments write to the same `observer_path` (e.g., `output/sacred/train_imitation`), Sacred's `FileStorageObserver` assigns sequential integer run IDs (1, 2, 3...). If two experiments start near-simultaneously from different tmux panes, they can claim the same run ID and overwrite each other's `config.json`, `run.json`, and `cout.txt` files. This data loss is silent — Sacred does not warn about concurrent writes to the same directory.

**Why it happens:** The Sacred `FileStorageObserver` uses a simple integer counter based on the highest existing directory number plus one. Under concurrent access, two processes can read the same "next ID" before either creates its directory. Additionally, `parallel.py`'s known bug requires `CAPTURE_MODE = "sys"` (documented at line 205), but if this isn't set before `ray.init`, the default `"fd"` mode causes Sacred to fail silently, losing all experiment logs.

**Consequences:** Experiment results overwritten. Post-hoc analysis tools (`sacred_output_to_csv.py`, `sacred_output_to_markdown_summary.py`) parse incorrect or mixed data. Reproducibility is destroyed.

**Prevention:**
- Use separate `observer_path` directories per experiment: e.g., `output/sacred/dagger_PongNoFrameskip`, `output/sacred/ftrl_PongNoFrameskip`.
- Always set `sacred.SETTINGS.CAPTURE_MODE = "sys"` before running parallel experiments (this is already done in `parallel.py`'s Ray wrapper, but not in direct tmux runs).
- For tmux-based runs (not via Ray), run one experiment at a time per observer directory, or use unique run directories: pass `--file_storage=output/sacred/run_{algo}_{game}_{seed}` as a Sacred CLI argument.
- After a full run completes, verify `find output/sacred -name run.json | wc -l` equals the expected number of runs.

**Warning signs:** Duplicate or missing run IDs in the Sacred output directory. `run.json` timestamps from different algorithms collide. `sacred_output_to_csv.py` reports fewer runs than expected.

**Phase:** Multi-GPU parallel experiment runner and Sacred logging setup.

---

### Pitfall 10: Expert Policy Action Labels Use Deterministic Mode — Not Matching Paper Setup

**What goes wrong:** `SimpleDAggerTrainer.train` calls `rollout.generate_trajectories` with `deterministic_policy=True` (line 673 in dagger.py). This means the expert always outputs the argmax action. For Atari PPO experts from HuggingFace, which were trained with entropy regularization and have non-trivial action distributions, querying deterministically may produce suboptimal expert labels compared to sampling from the distribution — particularly in stochastic game states.

**Why it happens:** The `deterministic_policy=True` default is appropriate for many settings but may not match how Lavington et al. generated expert labels. If the paper used stochastic expert labels, the FTL/FTRL comparison is not faithful.

**Consequences:** Expert trajectories have lower variance than optimal, which makes DAgger's dataset slightly less informative for learning policies that handle stochastic situations. The effect is small but consistent across games and could bias the FTL vs FTRL comparison if one benefits more from stochastic vs deterministic labels.

**Prevention:**
- Check Lavington et al.'s code (`Atari-Experiments/atari_ex.py`) to confirm whether expert labels are deterministic or stochastic.
- Use the same setting consistently across all algorithms (BC, DAgger, FTRL) and document it.
- This is a minor source of bias, not a catastrophic pitfall, but should be explicitly chosen and logged.

**Warning signs:** Expert return in `dagger/mean_episode_reward` log is lower than the expert score from `expert_stats` (which measures expert performance in a separate eval).

**Phase:** DAgger/FTRL training — verify before first full run.

---

## Minor Pitfalls

### Pitfall 11: Smoke-Test Skips Real Atari Game — Uses CartPole Instead

**What goes wrong:** The Sacred environment ingredient defaults to `seals/CartPole-v0`. If the smoke test config doesn't explicitly override `environment.gym_id` to an Atari game, the smoke test passes on CartPole but does not validate Atari preprocessing, expert loading, or frame stacking.

**Prevention:** Always specify an Atari game in the smoke test config (e.g., `PongNoFrameskip-v4`), even with reduced timesteps. The smoke test exists to catch Atari-specific failures before wasting GPU hours on a full run.

**Phase:** Smoke-test configuration.

---

### Pitfall 12: `expert_stats` Missing From Sacred Result When Expert Lacks Reward

**What goes wrong:** `_collect_stats` in `train_imitation.py` only computes `expert_stats` if all trajectories have reward information (`_all_trajectories_have_reward` check). HuggingFace-loaded expert policies generate trajectories through `rollout.generate_trajectories` which does include rewards, but if the environment is wrapped differently (e.g., `RolloutInfoWrapper` absent), reward may be stripped. When `expert_stats` is absent, `sacred_output_to_markdown_summary.py` will `KeyError` on `run["result"]["expert_stats"]["monitor_return_mean"]`.

**Prevention:** Verify that expert rollout trajectories always contain reward by running `assert all(isinstance(t, TrajectoryWithRew) for t in expert_trajs)` after loading. Ensure the evaluation environment includes `RolloutInfoWrapper`.

**Phase:** Evaluation pipeline.

---

### Pitfall 13: Ray Version Pinned to `2.0.0` May Conflict With Python 3.10+

**What goes wrong:** `setup.py` pins `ray[debug,tune]~=2.0.0`. Ray 2.0.0 was released in 2022 and has known compatibility issues with Python 3.10+ and newer numpy. If the CC-server Python environment uses Python 3.10 or higher, Ray initialization may fail with import errors or deprecation warnings that silently break experiment management.

**Prevention:** Verify Python version on CC-server (`python --version`) before setting up the virtualenv. If Python >= 3.10, upgrade Ray to 2.x latest in the project's requirements rather than using the pinned version.

**Phase:** Server environment setup.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|----------------|------------|
| Atari env setup | Observation space mismatch (channel order, frame stack location) | Use `make_atari_env` as base; assert space equality against expert |
| Atari env setup | Double frame-skipping from wrong gym ID | Enforce `*NoFrameskip-v4` naming in all configs |
| Game selection | Missing random baseline on HuggingFace | Pre-validate all 10+ games before full run |
| FTRL implementation | Proximal term vs weight decay confusion | Implement as `||theta - theta_prev||^2`, not `||theta||^2` |
| FTRL implementation | Lambda scale destroys learning | Log `proximal_term / bc_loss` ratio each round; validate on smoke test |
| DAgger training config | Beta rampdown too fast | Set `rampdown_rounds` = total rounds; log beta per round |
| DAgger data accumulation | OOM from unbounded `_all_demos` list | Monitor RAM per round; implement rolling window if needed |
| Multi-GPU runner | All processes claim GPU 0 | Set `CUDA_VISIBLE_DEVICES=N` per tmux pane or use Ray GPU assignment |
| Sacred parallel logging | Run ID collision and CAPTURE_MODE bug | Separate observer dirs per experiment; set capture mode to `"sys"` |
| Normalization pipeline | `nan` from near-zero denominator | Pre-check `expert_score - random_score > 1.0` for all games |
| Smoke test | CartPole passes but Atari fails | Force Atari game in smoke test config |
| Server setup | Ray 2.0.0 / Python 3.10 conflict | Check Python version; upgrade Ray if needed |

---

## Sources

- Direct inspection of `src/imitation/algorithms/dagger.py` (lines 355, 400-407, 592-597, 673)
- Direct inspection of `src/imitation/scripts/parallel.py` (lines 202-205 — CAPTURE_MODE bug workaround)
- Direct inspection of `benchmarking/sacred_output_to_markdown_summary.py` (lines 105-108 — normalization formula)
- Direct inspection of `.planning/codebase/CONCERNS.md` (DAgger Expert Data Sources Limitation, Trajectory Buffer Memory, Sacred Capture Mode Issue, Missing Checkpointing in BC and DAgger)
- [SB3 Atari Wrappers Documentation](https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html) — "Use this wrapper only with Atari v4 without frame skip" (HIGH confidence)
- [Lavington et al. 2022, arXiv 2208.00088](https://arxiv.org/abs/2208.00088) — FTRL formulation (HIGH confidence — primary paper reference)
- [Lavington et al. GitHub](https://github.com/WilderLavington/Improved-Policy-Optimization-for-Online-Imitation-Learning) — reference Atari experiment code (MEDIUM confidence — code inspected only at directory level)
- [sb3/ppo-BreakoutNoFrameskip-v4 HuggingFace model card](https://huggingface.co/sb3/ppo-BreakoutNoFrameskip-v4) — confirms SB3 Atari preprocessing stack used for HuggingFace experts (HIGH confidence)
- [DAgger original paper (Ross et al. 2011)](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf) — data accumulation design (HIGH confidence)
