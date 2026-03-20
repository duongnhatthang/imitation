# Project Research Summary

**Project:** DAgger vs FTRL Empirical Study — Atari Milestone
**Domain:** Online imitation learning — empirical benchmark on Atari environments
**Researched:** 2026-03-19
**Confidence:** MEDIUM-HIGH (stack and architecture HIGH; expert coverage and random-baseline datasets MEDIUM)

## Executive Summary

This project implements FTRL (Follow-the-Regularized-Leader) as a variant of DAgger and benchmarks it against BC and DAgger (FTL) on a 7–10 game Atari suite, following Lavington et al. 2022. The existing `imitation` library already provides BC, DAgger, Sacred experiment tracking, multi-GPU benchmarking infrastructure, rliable-based IQM metrics, and HuggingFace expert loading — the entire scaffold exists. The work reduces to: (1) implementing a ~90-line FTRL algorithm extension, (2) adding Sacred named configs for Atari games, and (3) running 300+ experiments across 4 GPUs.

The recommended approach keeps the change surgical: a new `FTRLLossCalculator` adds a proximal L2 term anchored to the previous round's policy snapshot, injected into the existing `BC` class via a constructor argument. `FTRLTrainer` extends `SimpleDAggerTrainer` with a ~60-line override of `extend_and_update`. Nothing in the DAgger rollout collection, beta schedule, or Sacred infrastructure changes. The 4-GPU CC-server runs jobs via `CUDA_VISIBLE_DEVICES` + GNU parallel inside tmux — no Ray cluster needed.

The primary risk is not algorithmic: it is infrastructure correctness before the full benchmark run. Three issues can silently corrupt results without raising errors — observation space mismatches between the HuggingFace expert and the training environment, missing HuggingFace random-baseline datasets for chosen games, and Sacred run-ID collisions under concurrent writes. All three must be validated during the smoke-test phase before committing GPU-hours to the full benchmark.

## Key Findings

### Recommended Stack

The existing stack (PyTorch, SB3 ~2.2.1, Sacred, gymnasium ~0.29, seals 0.2.1) requires no major additions. The `seals[atari]` extra is already pinned and registers 7 games as `seals/{Game}-v5` with `MaskScoreWrapper` and fixed-length episodes — the correct format for fair IL comparison. Expert policies come from the `sb3` HuggingFace organization (not `HumanCompatibleAI`, which covers only MuJoCo). The gymnasium pin at ~0.29 must not be upgraded; ale-py and seals both depend on it. GNU parallel (system package) replaces the sequential benchmark runner for 4x throughput with zero daemon overhead.

**Core technologies:**
- `seals[atari] ~0.2.1`: Atari environment wrappers — already pinned; provides `MaskScoreWrapper` and `AutoResetWrapper` for fair IL evaluation
- `sb3` HuggingFace org: Pre-trained PPO expert policies — standard for the benchmark; Breakout, Pong, SpaceInvaders confirmed; BeamRider, Enduro, Qbert coverage needs manual verification
- GNU parallel (system): Multi-GPU job dispatch — `slot()-1` maps parallel slots to GPU IDs; no daemon, no scheduler interference
- `huggingface_sb3 ~3.0`: Already in project; downloads SB3 `.zip` policy files from HF Hub
- gymnasium ~0.29: Locked — do not upgrade to 1.x; seals 0.2.1 is incompatible with gymnasium 1.x

### Expected Features

**Must have (table stakes):**
- FTRL implementation matching Lavington et al. Eq. 6 exactly — core contribution; proximal L2 term `lambda/2 * ||theta - theta_prev||^2`, not weight decay
- 7–10 Atari game suite with consistent ALE preprocessing — multi-game results required for credibility
- BC and DAgger baselines with Atari Sacred named configs — required comparison points
- 10 seeds per configuration — existing benchmark standard; single-seed results are not credible
- Normalized scores `(score - random) / (expert - random)` — already implemented; cross-game comparability
- Random baseline scores per game — required denominator for normalization; verify HumanCompatibleAI datasets exist before finalizing game list
- Sacred experiment tracking with per-run log files — already in use; must add Atari named configs
- Smoke-test configuration on a real Atari game — catch silent failures before full GPU run

**Should have (differentiators):**
- IQM aggregate metric with 95% stratified bootstrap CI — rliable already integrated; modern best practice per Agarwal et al. 2021
- Probability of Improvement (FTRL vs DAgger) — directly answers the study's core question; existing script, plug in Atari results
- Per-round normalized score curves — shows convergence behavior matching Lavington et al. Figure 4
- Per-game score table (CSV output) — reveals which game types FTRL helps or hurts

**Defer (post-benchmark):**
- Hyperparameter sensitivity analysis on FTRL lambda — adds 5-10x compute; reference only if reviewers ask
- Alt-FTRL and AdaFTRL variants — out of scope per PROJECT.md
- Performance profiles (tau-curves) — add only if paper target requires it
- Probability of Improvement computation — runs in minutes after benchmark completes; not on the critical path

### Architecture Approach

The architecture is three cleanly separated components with explicit ownership boundaries. Component 1 is the FTRL algorithm layer (`FTRLTrainer` + `FTRLLossCalculator`), which owns algorithm correctness and nothing else. Component 2 is the Atari experiment infrastructure — Sacred named configs per game, a bash orchestration script, and a single-run entry point. Component 3 is the evaluation and figure pipeline — Sacred output collection, normalization, and curve generation. The key structural insight is that `FTRLLossCalculator` replaces `BehaviorCloningLossCalculator` as a constructor-injected collaborator inside `BC`; BC never needs to know FTRL exists. This requires one backward-compatible two-line change to `BC.__init__` to accept an optional external `loss_calculator` argument.

**Major components:**
1. `FTRLTrainer` (extends `SimpleDAggerTrainer`, ~60 lines) — manages anchor snapshot at round boundaries; overrides `extend_and_update` only
2. `FTRLLossCalculator` (~30 lines) — computes `neglogp + ent_loss + l2_loss + ftrl_lambda/2 * ||theta - anchor||^2`; injected via `BC(loss_calculator=...)`
3. Atari experiment infra — Sacred named configs per game, `run_atari_benchmark.sh` with `CUDA_VISIBLE_DEVICES` round-robin GPU assignment
4. Evaluation pipeline — `sacred_output_to_csv.py` (existing) + normalization + `plot_curves.py` producing figures matching Lavington et al. Figure 4

### Critical Pitfalls

1. **Observation space mismatch between HuggingFace expert and training env** — Use `make_atari_env` as the environment base (not generic `make_vec_env`); assert `venv.observation_space == expert_policy.observation_space` before any training. HuggingFace experts expect `Box(0, 255, (84, 84, 4), uint8)`. A shape-compatible but semantically wrong space (CHW vs HWC) produces near-random scores with no error raised.

2. **FTRL proximal term implemented as weight decay instead of anchor distance** — Do not reuse `LpRegularizer` or `WeightDecayRegularizer`. The FTRL term is `||theta - theta_prev||^2`, not `||theta||^2`. Using the wrong regularizer makes the algorithm differ from the paper, invalidating the comparison. Log `proximal_term / bc_loss` each round as a sanity check.

3. **Random baseline datasets missing on HuggingFace for chosen games** — `get_random_agent_score` fetches from `HumanCompatibleAI/random-{EnvironmentName(env)}`. If a dataset is missing, figure generation crashes after all training completes. Pre-validate all chosen games programmatically before the full run; fall back to collecting local random rollouts if needed.

4. **DAgger demo buffer causes OOM mid-run** — `_all_demos` accumulates all rounds in memory. At 30 rounds × 5000 timesteps × `(84,84,4)` observations, this exceeds 16GB RAM. Implement a rolling window (keep last N rounds) and monitor RSS during smoke test before committing to the full benchmark.

5. **Sacred run-ID collision under concurrent writes** — Two tmux processes writing to the same `observer_path` can claim the same run ID and silently overwrite each other's results. Use per-experiment observer directories: `output/sacred/{algo}_{game}_{seed}` and set `CAPTURE_MODE = "sys"`.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: FTRL Algorithm Implementation
**Rationale:** Everything else depends on a correct FTRL implementation. Algorithm correctness must be validated independently of Atari infrastructure costs. A CartPole or seals/CartPole smoke test confirms the proximal term works before any GPU-hours are spent.
**Delivers:** `src/imitation/algorithms/ftrl.py`, `FTRLLossCalculator`, `FTRLTrainer`, unit tests in `tests/test_ftrl.py`, and the two-line `BC.__init__` change
**Addresses:** Core FTRL feature (table stakes), proximal term correctness
**Avoids:** Pitfall 2 (proximal term vs weight decay), Pitfall 3 lambda scale issues

### Phase 2: Atari Environment Setup and Expert Verification
**Rationale:** Observation space correctness (Pitfall 1) and random baseline availability (Pitfall 3/5) are silent failure modes that corrupt all downstream results. They must be validated before Sacred configs or the full runner are written. Game selection is blocked on confirming which games have both HuggingFace PPO experts (`sb3` org) and `HumanCompatibleAI` random-score datasets.
**Delivers:** Finalized 7–10 game list, verified expert downloads for all games, confirmed or locally collected random baseline scores, `make_atari_env`-based environment construction with observation space assertions
**Uses:** `seals[atari] ~0.2.1`, `sb3` HF org, `huggingface_sb3 ~3.0`
**Avoids:** Pitfall 1 (obs space mismatch), Pitfall 2 (double frame skip), Pitfall 5 (missing random baseline datasets), Pitfall 4 (NaN normalization)

### Phase 3: Sacred Named Configs and Smoke Test
**Rationale:** With a verified FTRL implementation and confirmed game list, add Sacred named configs for all three algorithms across all games. The smoke test must use a real Atari game (not CartPole) to catch Atari-specific wrappers, expert loading, and normalization pipeline bugs. This is the last checkpoint before paying full compute costs.
**Delivers:** `src/imitation/scripts/config/atari_games.py` (named configs for BC, DAgger, FTRL per game), `experiments/run_smoke_test.sh` (2 games, 1 seed, 3 rounds), verified Sacred output structure and normalization output
**Avoids:** Pitfall 11 (smoke test uses CartPole not Atari), Pitfall 7 (beta schedule too short), Pitfall 12 (expert_stats missing from Sacred result)

### Phase 4: Multi-GPU Experiment Runner
**Rationale:** With validated configs and a passing smoke test, write the orchestration script. Explicit `CUDA_VISIBLE_DEVICES` assignment, per-experiment Sacred observer directories, and tmux session management must be in place before the full 300+ run benchmark.
**Delivers:** `experiments/run_atari_benchmark.sh` with `CUDA_VISIBLE_DEVICES` round-robin GPU assignment, GNU parallel job dispatch, per-run tee logging, separate Sacred observer paths per experiment
**Uses:** GNU parallel (system), tmux, CUDA_VISIBLE_DEVICES
**Avoids:** Pitfall 8 (all processes claim GPU 0), Pitfall 9 (Sacred run-ID collision), Pitfall 6 (OOM from unbounded demo buffer — validate rolling window here)

### Phase 5: Full Benchmark Run and Analysis
**Rationale:** Execute 300+ runs (3 algorithms × 7–10 games × 10 seeds). No algorithmic risk at this phase; moderate pipeline engineering complexity in aligning Sacred outputs across seeds and handling missing or failed runs.
**Delivers:** Complete Sacred output directory, CSV of normalized scores, IQM aggregate metrics with 95% CI, Probability of Improvement (FTRL vs DAgger), per-round learning curves, figures matching Lavington et al. Figure 4 style
**Implements:** Evaluation pipeline component (Component 3)
**Avoids:** Pitfall 4 (NaN normalization denominators — pre-validated in Phase 2)

### Phase Ordering Rationale

- Phase 1 before Phase 2: FTRL correctness is validated cheaply on CartPole; don't pay Atari compute costs to discover algorithm bugs
- Phase 2 before Phase 3: Game selection (and therefore all Sacred configs) depends on confirming which games have both expert policies and random baselines — this is a research question, not an implementation task
- Phase 3 smoke test before Phase 4 runner: The smoke test is the last cheap validation gate; the runner executes at scale and failures are expensive
- Phase 4 before Phase 5: GPU assignment and Sacred isolation must be correct before 300 concurrent writes begin

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Confirm whether `BC.__init__` currently exposes `loss_calculator` as a constructor argument, or whether `FTRLTrainer` must patch it post-construction — read `bc.py` source to decide which two-line change is needed
- **Phase 2:** Manually verify `sb3` org HuggingFace coverage for BeamRider, Enduro, and Qbert (only Breakout, Pong, SpaceInvaders confirmed); verify `HumanCompatibleAI` random-score dataset naming scheme for Atari games
- **Phase 3:** Check Lavington et al.'s code for the exact DAgger rounds count, beta schedule, and whether expert labels are deterministic or stochastic — these must match to reproduce Figure 4

Phases with standard patterns (skip research-phase):
- **Phase 4:** GNU parallel GPU assignment with `slot()-1` is a well-documented pattern; `CUDA_VISIBLE_DEVICES` per-process is standard
- **Phase 5:** Sacred-to-CSV pipeline and rliable IQM are already implemented in the codebase; no new patterns needed

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | seals source confirmed; gymnasium lock verified; GNU parallel GPU pattern verified; only ale-py exact version needs on-server confirmation |
| Features | HIGH | Codebase inspection confirmed normalization formula, rliable integration, Sacred schema; FTRL formulation from PROJECT.md reference |
| Architecture | HIGH | Direct source inspection of `dagger.py`, `bc.py`, `regularizers.py`; FTRL extension pattern is clean composition; one open question on BC constructor |
| Pitfalls | HIGH | All critical pitfalls sourced from direct codebase inspection or verified SB3 docs; DAgger OOM issue documented in CONCERNS.md |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **sb3 HF org Atari coverage for 4 games:** Run `python -m rl_zoo3.load_from_hub --algo ppo --env {Game}NoFrameskip-v4 -orga sb3` for BeamRider, Enduro, Qbert, and one more game before finalizing the game list. If missing, fall back to local RL Zoo training or reduce to the 3 confirmed games.
- **HumanCompatibleAI random dataset naming for Atari:** The `EnvironmentName` transform that maps gym IDs to dataset names is undocumented; must be validated programmatically for each chosen game before the full run.
- **ale-py version pulled by seals 0.2.1 on CC-server:** Run `pip show ale-py` after install to confirm ROM bundling. If ale-py < 0.9, an `AutoROM` step is needed.
- **BC constructor for loss_calculator injection:** Check whether `BC.__init__` already accepts `loss_calculator` or needs a two-line addition — determines exact implementation approach for `FTRLTrainer`.
- **Lavington et al. experimental config details:** DAgger round count, beta schedule type, and deterministic vs stochastic expert labels are needed to ensure faithful reproduction of Figure 4. Inspect their GitHub repo's `atari_ex.py` before finalizing Sacred configs.

## Sources

### Primary (HIGH confidence)
- `src/imitation/algorithms/dagger.py` — DAgger trainer structure, beta schedule defaults, `_all_demos` accumulation
- `src/imitation/algorithms/bc.py` — BC loss calculator injection point
- `src/imitation/regularization/regularizers.py` — existing regularizer types (not proximal)
- `src/imitation/scripts/parallel.py` — CAPTURE_MODE bug, Ray Tune overhead
- `benchmarking/sacred_output_to_markdown_summary.py` — normalization formula, rliable IQM integration
- `benchmarking/README.md` — 10-seed protocol, Sacred output schema
- `src/seals/atari.py` (seals v0.2.1) — confirmed 7-game list and wrapper types
- [SB3 Atari Wrappers docs](https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html) — frame skip warning, expected obs space
- [GNU Parallel Tutorial](https://www.gnu.org/software/parallel/parallel_tutorial.html) — `slot()`-based GPU assignment

### Secondary (MEDIUM confidence)
- [HuggingFace sb3 org](https://huggingface.co/sb3) — confirmed Breakout, Pong, SpaceInvaders; BeamRider/Enduro/Qbert unconfirmed
- Lavington et al. 2022, CoLLAs — FTRL Proposition 4.1 Eq. 6 formulation (abstract only; full PDF not accessed)
- Agarwal et al. 2021, NeurIPS — IQM and Probability of Improvement metrics
- ale-py release notes — ROM bundling since 0.9.0

### Tertiary (LOW confidence)
- [HumanCompatibleAI HuggingFace org](https://huggingface.co/HumanCompatibleAI) — Atari random-score dataset existence not confirmed; only MuJoCo confirmed present

---
*Research completed: 2026-03-19*
*Ready for roadmap: yes*
