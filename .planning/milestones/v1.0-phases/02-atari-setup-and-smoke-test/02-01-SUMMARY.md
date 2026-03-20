---
phase: 02-atari-setup-and-smoke-test
plan: "01"
subsystem: infra
tags: [atari, ale-py, shimmy, stable-baselines3, huggingface, seals, gymnasium, random-baselines]

# Dependency graph
requires:
  - phase: 01-ftrl-algorithm
    provides: FTRLTrainer, BC infrastructure, rollout utilities
provides:
  - "ATARI_GAMES dict with 7 raw ALE env IDs"
  - "make_atari_training_venv: make_atari_env + VecFrameStack(4) + VecTransposeImage -> (4,84,84)"
  - "make_atari_eval_venv: same with clip_reward=False for true episode returns"
  - "load_atari_expert: load_policy ppo-huggingface with organization='sb3'"
  - "compute_normalized_score: (agent - random) / (expert - random) with div-by-zero guard"
  - "Random baselines cached: experiments/baselines/atari_random_scores.pkl"
  - "Server setup script: setup/setup_server.sh (pip install -e '[atari]' + autorom)"
  - "24 passing tests covering ENV-01 through ENV-05"
affects:
  - 02-atari-smoke-test
  - 03-benchmark-runs
  - 04-analysis-and-figures

# Tech tracking
tech-stack:
  added: [ale-py~=0.8.1, shimmy[atari]~=0.2.1, autorom~=0.4.2, gym==0.26.2]
  patterns:
    - "Atari venv: make_atari_env + VecFrameStack(4) + VecTransposeImage for (4,84,84) channels-first obs"
    - "Expert loading: load_policy('ppo-huggingface', organization='sb3') NOT HumanCompatibleAI"
    - "Eval venv: wrapper_kwargs={'clip_reward': False} for unclipped episode returns"
    - "Random baseline collection: rollout.rollout(unwrap=False) to avoid RolloutInfoWrapper requirement"

key-files:
  created:
    - experiments/atari_helpers.py
    - experiments/collect_random_baselines.py
    - experiments/baselines/atari_random_scores.pkl
    - setup/setup_server.sh
    - tests/algorithms/test_atari_setup.py
  modified:
    - experiments/atari_helpers.py  # added VecTransposeImage (auto-fix for obs space mismatch)

key-decisions:
  - "Use VecTransposeImage after VecFrameStack to produce (4,84,84) channels-first obs matching SB3 expert; without it venv produces (84,84,4) channels-last causing obs space mismatch"
  - "Pass unwrap=False to rollout.rollout() for random baseline collection; RolloutInfoWrapper not needed for stats-only collection"
  - "Install gym==0.26.2 alongside gymnasium; HF sb3 models were pickled with old gym module and require it for deserialization"
  - "Random baselines: Pong=-20.42, Breakout=0.30, BeamRider=120.0, Enduro=0.0, Qbert=40.15, Seaquest=21.82, SpaceInvaders=63.48"

patterns-established:
  - "Pattern: Always chain make_atari_env -> VecFrameStack(4) -> VecTransposeImage for (4,84,84) obs"
  - "Pattern: Use organization='sb3' for all Atari expert loads (never 'HumanCompatibleAI')"
  - "Pattern: Separate training venv (clip_reward=True) and eval venv (clip_reward=False)"

requirements-completed: [ENV-01, ENV-02, ENV-03, ENV-04, ENV-05, INFRA-07]

# Metrics
duration: 10min
completed: 2026-03-20
---

# Phase 2 Plan 01: Atari Setup and Infrastructure Summary

**7-game Atari suite verified end-to-end: SB3 make_atari_env + VecTransposeImage for (4,84,84) obs, all 7 sb3 HF experts load and match obs space, random baselines cached (30 eps/game), setup_server.sh created**

## Performance

- **Duration:** 10 min
- **Started:** 2026-03-20T05:55:29Z
- **Completed:** 2026-03-20T06:05:48Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Created `experiments/atari_helpers.py` with 5 exports: ATARI_GAMES, make_atari_training_venv, make_atari_eval_venv, load_atari_expert, compute_normalized_score
- Collected and cached random baseline scores for all 7 Atari games to `experiments/baselines/atari_random_scores.pkl` (30 episodes each, seed=0)
- Created `setup/setup_server.sh` with `pip install -e ".[atari]"` and `autorom --accept-license` (INFRA-07)
- All 24 tests in `test_atari_setup.py` pass, covering ENV-01 through ENV-05

## Task Commits

Each task was committed atomically:

1. **Task 1: Create atari_helpers.py shared utilities and server setup script** - `49d73f3` (feat)
2. **Task 2: Collect random baselines, write tests, and verify all 7 games end-to-end** - `9f736b5` (feat)

## Files Created/Modified
- `experiments/atari_helpers.py` - ATARI_GAMES dict + 4 utility functions; VecTransposeImage added for channels-first obs space
- `experiments/collect_random_baselines.py` - Standalone script collecting random baselines using RandomPolicy + rollout.rollout(unwrap=False)
- `experiments/baselines/atari_random_scores.pkl` - Cached dict {game_name: {"mean": float, "std": float}} for all 7 games
- `setup/setup_server.sh` - Server venv setup script: create venv, pip install -e "[atari]", autorom --accept-license, verify ALE
- `tests/algorithms/test_atari_setup.py` - 24 tests across TestAtariEnvSetup, TestAtariObsSpace, TestAtariExpertLoading, TestRandomBaselines, test_normalized_score

## Decisions Made
- **VecTransposeImage required:** SB3's make_atari_env + VecFrameStack(4) produces (84,84,4) channels-last. SB3 experts internally wrap venv with VecTransposeImage and expose (4,84,84) observation_space. Adding VecTransposeImage to the learner venv is mandatory for obs space equality assertion (ENV-04).
- **gym==0.26.2 needed alongside gymnasium:** HF sb3 expert models were serialized with the old `gym` module. cloudpickle deserialization fails with "No module named 'gym'" without it installed. Both gym and gymnasium coexist at runtime.
- **unwrap=False for rollout collection:** imitation's rollout.rollout() by default calls unwrap_traj() which requires RolloutInfoWrapper. For random baseline collection we only need stats, so passing unwrap=False avoids the wrapper requirement while still computing rollout_stats correctly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added VecTransposeImage to produce channels-first obs matching expert**
- **Found during:** Task 2 (running collect_random_baselines.py, then confirmed via expert obs space check)
- **Issue:** Plan specified obs space (4,84,84) but make_atari_env + VecFrameStack(4) actually produces (84,84,4) channels-last. SB3 experts auto-apply VecTransposeImage and expose (4,84,84). Without VecTransposeImage the ENV-04 obs space assertion would fail.
- **Fix:** Added `VecTransposeImage(venv)` as final step in both make_atari_training_venv and make_atari_eval_venv
- **Files modified:** experiments/atari_helpers.py
- **Verification:** venv.observation_space.shape == (4,84,84); all 7 expert obs space assertions pass
- **Committed in:** 9f736b5 (Task 2 commit)

**2. [Rule 3 - Blocking] Used unwrap=False in rollout.rollout() to bypass RolloutInfoWrapper requirement**
- **Found during:** Task 2 (running collect_random_baselines.py, KeyError: 'rollout')
- **Issue:** rollout.rollout() default unwrap=True calls unwrap_traj() which reads traj.infos[-1]["rollout"] — requires RolloutInfoWrapper. Plan didn't mention this wrapper.
- **Fix:** Pass unwrap=False; rollout_stats still computes correctly from raw trajectories
- **Files modified:** experiments/collect_random_baselines.py
- **Verification:** Script runs and produces stats for all 7 games
- **Committed in:** 9f736b5 (Task 2 commit)

**3. [Rule 3 - Blocking] Installed ale-py~=0.8.1, shimmy~=0.2.1, autorom, gym on local machine**
- **Found during:** Task 2 (initial script run — NameNotFound for PongNoFrameskip-v4)
- **Issue:** Local machine lacked Atari dependencies (not on CC-server yet). ale-py, shimmy, autorom, gym were missing.
- **Fix:** Installed compatible versions: ale-py~=0.8.1, shimmy~=0.2.1, autorom, gym==0.26.2; ran autorom --accept-license to download ROMs
- **Verification:** All 24 tests pass locally; setup confirms the exact dependency list for setup_server.sh
- **Committed in:** Not committed (environment changes)

---

**Total deviations:** 3 auto-fixed (1 bug/correctness, 2 blocking)
**Impact on plan:** All auto-fixes necessary for correctness. The VecTransposeImage fix is the most important — without it ENV-04 would always fail. No scope creep.

## Issues Encountered
- ale-py 0.10.1 + gymnasium 1.1.1 incompatible with SB3 2.2.1 (requires gymnasium<0.30): resolved by pinning ale-py~=0.8.1 and shimmy~=0.2.1 as specified in seals[atari] requirements
- HF sb3 models require old `gym` module for cloudpickle deserialization: resolved by installing gym==0.26.2 alongside gymnasium

## User Setup Required
None - no external service configuration required beyond running setup_server.sh on the CC-server.

## Next Phase Readiness
- Atari infrastructure complete: all 7 games create correct (4,84,84) venvs, experts load from HF sb3 org, random baselines cached
- Ready for Phase 02-02: Atari smoke test (BC, DAgger, FTRL on 2 games, 3 rounds, 1 seed)
- The `atari_helpers.py` functions are the canonical way to create Atari venvs throughout the study

## Self-Check: PASSED

All files confirmed on disk. Both task commits verified in git log.

---
*Phase: 02-atari-setup-and-smoke-test*
*Completed: 2026-03-20*
