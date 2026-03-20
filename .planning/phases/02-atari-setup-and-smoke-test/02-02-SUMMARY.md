---
phase: 02-atari-setup-and-smoke-test
plan: "02"
subsystem: experiments
tags: [atari, smoke-test, bc, dagger, ftrl, pong, breakout, sacred]

# Dependency graph
requires:
  - phase: 02-atari-setup-and-smoke-test
    plan: "01"
    provides: atari_helpers.py, random baselines pkl, expert loading
  - phase: 01-ftrl-algorithm
    provides: FTRLDAggerTrainer, BC infrastructure
provides:
  - "experiments/atari_smoke.py: standalone BC/DAgger/FTRL smoke test script"
  - "Sacred Atari named configs for 7 games (atari_pong, atari_breakout, etc.) with organization=sb3"
  - "experiments/smoke_test_output.log: confirmed all 3 methods run end-to-end on 2 Atari games"
  - "Fix: serialize.py num_shards guard preventing IndexError on short trajectories"
affects:
  - 03-benchmark-runs
  - 04-analysis-and-figures

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Standalone smoke test: directly use imitation API (no Sacred) for debuggability"
    - "CnnPolicy: ActorCriticCnnPolicy from stable_baselines3.common.policies for Atari obs"
    - "BC eval: compute mean return directly from trajectory rews (not rollout_stats) to avoid post-DAgger state issues"
    - "Script entry: add _PROJECT_ROOT to sys.path for both script and module execution modes"

key-files:
  created:
    - experiments/atari_smoke.py
    - experiments/smoke_test_output.log
  modified:
    - src/imitation/scripts/config/train_imitation.py
    - src/imitation/data/serialize.py

key-decisions:
  - "Use total_timesteps=8000 for CPU smoke test (50000 produces 80+ DAgger rounds with O(n^2) BC training growth, 6+ hours on CPU vs plan's estimated 30min for GPU)"
  - "Replace rollout_stats() with direct mean computation in evaluate_policy to avoid subtle post-DAgger buffering issues in long runs"
  - "serialize.save: clamp num_shards=max(1,len(ds)) to prevent HF datasets IndexError when single-trajectory datasets (short Breakout episodes) are sharded with num_shards>1"

patterns-established:
  - "Pattern: standalone atari_smoke.py bypasses Sacred infrastructure for simpler debugging"
  - "Pattern: each method gets fresh BC trainer + CnnPolicy for fair comparison"
  - "Pattern: separate rng instances per method call to ensure reproducibility"

requirements-completed: [INFRA-05]

# Metrics
duration: 353min
completed: 2026-03-20
---

# Phase 02 Plan 02: Atari Smoke Test Summary

Ran BC, DAgger, and FTRL on Pong and Breakout end-to-end with 3 training rounds, producing normalized scores confirming full pipeline correctness.

## What Was Built

**Task 1: Sacred Atari Named Configs + atari_smoke.py**

Added 7 Atari named configs to `src/imitation/scripts/config/train_imitation.py` (atari_pong through atari_spaceinvaders), each with `expert = dict(loader_kwargs=dict(organization="sb3"))` to route to the correct HuggingFace expert models.

Created `experiments/atari_smoke.py` (390 lines): standalone script that:
- Parses CLI args (--games, --seed, --n-rounds, --total-timesteps, --n-envs)
- Loads cached random baselines from experiments/baselines/atari_random_scores.pkl
- For each game: runs BC (expert demos + 4 epochs), DAgger (SimpleDAggerTrainer), FTRL (FTRLDAggerTrainer, alpha=1.0)
- Computes normalized scores: (agent - random) / (expert - random)
- Prints summary table

**Task 2: Smoke Test Execution**

Ran the smoke test on Pong and Breakout with seed=0, n-rounds=3, total-timesteps=8000 (CPU-appropriate budget), n-envs=8.

Results (`experiments/smoke_test_output.log`):

```
Game                BC    DAgger      FTRL
------------------------------------------
Pong            0.1253    0.1224    0.1167
Breakout        0.0172    0.0439    0.0073
```

Scores near 0 are expected (8000 timesteps on CPU is a smoke test, not a benchmark). All 6 (method, game) combinations ran to completion without errors.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] HuggingFace datasets IndexError on short trajectories**
- **Found during:** Task 2, first DAgger run on Breakout
- **Issue:** `serialize.save()` called `save_to_disk()` with default `num_shards` (derived from CPU count), which exceeds `len(dataset)=1` for single-step Breakout episodes, causing `IndexError: Index 1 out of range for dataset of size 1`
- **Fix:** Added `num_shards = max(1, len(ds))` clamp in `src/imitation/data/serialize.py` to ensure num_shards never exceeds dataset size
- **Files modified:** src/imitation/data/serialize.py
- **Commit:** 9475543

**2. [Rule 3 - Blocking] Module import path for standalone script execution**
- **Found during:** Task 2 initial run
- **Issue:** `python experiments/atari_smoke.py` failed with `ModuleNotFoundError: No module named 'experiments'` because running a script adds the script's directory to sys.path, not the project root
- **Fix:** Added `_PROJECT_ROOT = Path(__file__).resolve().parents[1]` + `sys.path.insert(0, str(_PROJECT_ROOT))` at top of atari_smoke.py
- **Files modified:** experiments/atari_smoke.py
- **Commit:** 9475543

**3. [Rule 1 - Bug] Random baselines pickle stores dict (not float) per game**
- **Found during:** Task 2 first run
- **Issue:** `random_baselines.get(game_name, 0.0)` returned `{"mean": ..., "std": ...}` not float; format string `{random_score:.2f}` crashed with TypeError
- **Fix:** Extract `baseline_entry["mean"]` with isinstance check
- **Files modified:** experiments/atari_smoke.py
- **Commit:** 9475543

**4. [Plan deviation] Reduced total_timesteps from 50000 to 8000 for CPU execution**
- **Reason:** With 50000 timesteps, DAgger runs 80+ rounds on Pong (episode length ~1600 steps / 8 envs); BC training grows O(n^2) with round number, totaling 6+ hours on CPU vs GPU
- **Impact:** Scores are lower but this is a smoke test confirming end-to-end correctness, not a benchmark
- **Decision:** Use 8000 timesteps locally; production runs on CC-server with GPUs will use 50000+

## Self-Check: PASSED

All created files exist on disk. All task commits verified in git log.
