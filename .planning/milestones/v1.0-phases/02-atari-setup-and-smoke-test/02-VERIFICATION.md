---
phase: 02-atari-setup-and-smoke-test
verified: 2026-03-20T00:00:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
---

# Phase 2: Atari Setup and Smoke Test Verification Report

**Phase Goal:** The 7-game Atari suite is verified end-to-end — experts load, observation spaces match, random baselines exist, and a smoke test on a real Atari game passes all three methods
**Verified:** 2026-03-20
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|---------|
| 1  | All 7 Atari games create environments without error | VERIFIED | `make_atari_training_venv` and `make_atari_eval_venv` wrap `make_atari_env + VecFrameStack(4) + VecTransposeImage`; `test_atari_setup.py::TestAtariEnvSetup` parametrized over all 7 games |
| 2  | Learner venv observation space is Box(0,255,(4,84,84),uint8) for all 7 games | VERIFIED | `atari_helpers.py` applies `VecTransposeImage` after `VecFrameStack(n_stack=4)`; `TestAtariObsSpace` asserts `shape == (4, 84, 84)` and `dtype == np.uint8` |
| 3  | Expert policies load from HuggingFace sb3 org for all 7 games | VERIFIED | `load_atari_expert` calls `load_policy("ppo-huggingface", ..., organization="sb3")`; `TestAtariExpertLoading` loads each of 7 experts and confirms success |
| 4  | venv.observation_space == expert.observation_space for all 7 games | VERIFIED | `TestAtariExpertLoading::test_expert_loads_and_obs_match` asserts equality for all 7 games; smoke test also contains runtime assertion |
| 5  | Random baseline scores cached to disk for all 7 games | VERIFIED | `experiments/baselines/atari_random_scores.pkl` exists and contains 7 entries: Pong, Breakout, BeamRider, Enduro, Qbert, Seaquest, SpaceInvaders, each with `mean` and `std` float values |
| 6  | Server setup script installs all dependencies and downloads ROMs | VERIFIED | `setup/setup_server.sh` is executable (`-x` bit set), contains `pip install -e ".[atari]"` and `autorom --accept-license` |
| 7  | Smoke test runs BC on 2 Atari games (Pong, Breakout) to completion | VERIFIED | `smoke_test_output.log` shows BC completing on both games with normalized scores 0.1253 (Pong) and 0.0172 (Breakout); no `[BC] ERROR` lines |
| 8  | Smoke test runs DAgger and FTRL on 2 games to completion | VERIFIED | Log shows DAgger: Pong 0.1224, Breakout 0.0439; FTRL: Pong 0.1167, Breakout 0.0073; no `[DAgger] ERROR` or `[FTRL] ERROR` lines |
| 9  | Results are printed as a summary table with normalized scores | VERIFIED | Final lines of log show `=== Atari Smoke Test Results ===` table with all 3 methods and 2 games |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `experiments/atari_helpers.py` | ATARI_GAMES dict, make_atari_training_venv, make_atari_eval_venv, load_atari_expert, compute_normalized_score | VERIFIED | 121 lines; all 5 exports present and substantive; VecTransposeImage applied for channels-first obs |
| `experiments/collect_random_baselines.py` | Standalone script to collect and cache random baselines | VERIFIED | 99 lines; imports from atari_helpers, uses rollout.rollout(unwrap=False), saves to pkl |
| `experiments/baselines/atari_random_scores.pkl` | Cached random baseline scores dict | VERIFIED | Binary pickle with 7 game entries, each `{"mean": float, "std": float}`; Pong=-20.42, Breakout=0.30, etc. |
| `setup/setup_server.sh` | Server venv setup with Atari extras and ROM download | VERIFIED | 29 lines; executable; contains `pip install -e ".[atari]"` and `autorom --accept-license` |
| `tests/algorithms/test_atari_setup.py` | Tests for ENV-01 through ENV-05 | VERIFIED | 73 lines; 4 test classes (TestAtariEnvSetup, TestAtariObsSpace, TestAtariExpertLoading, TestRandomBaselines) plus test_normalized_score |
| `experiments/atari_smoke.py` | Standalone smoke test script running BC, DAgger, FTRL on 2 Atari games | VERIFIED | 398 lines (exceeds 80-line min); imports all required symbols; full CLI arg parsing |
| `src/imitation/scripts/config/train_imitation.py` | Atari named configs for Sacred experiment | VERIFIED | 7 named configs (atari_pong through atari_spaceinvaders), each with `expert = dict(loader_kwargs=dict(organization="sb3"))` |
| `experiments/smoke_test_output.log` | Confirmed smoke test run output | VERIFIED | 1207 lines; contains results table with all 3 methods and 2 games |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `experiments/atari_helpers.py` | `stable_baselines3.common.env_util.make_atari_env` | import and wrapping with VecFrameStack(4) | WIRED | Line 7: `from stable_baselines3.common.env_util import make_atari_env`; called in both venv constructors with VecFrameStack(4) |
| `experiments/atari_helpers.py` | `imitation.policies.serialize.load_policy` | load_policy with organization="sb3" | WIRED | Line 10: `from imitation.policies.serialize import load_policy`; called with `organization="sb3"` at line 91-96 |
| `experiments/collect_random_baselines.py` | `experiments/atari_helpers.py` | imports make_atari_eval_venv, ATARI_GAMES | WIRED | Line 22: `from experiments.atari_helpers import ATARI_GAMES, make_atari_eval_venv`; both used in `collect_random_baselines()` |
| `experiments/atari_smoke.py` | `experiments/atari_helpers.py` | imports make_atari_training_venv, make_atari_eval_venv, load_atari_expert, ATARI_GAMES, compute_normalized_score | WIRED | Lines 29-35: full import block; all 5 symbols used in run_bc() and run_dagger() |
| `experiments/atari_smoke.py` | `imitation.algorithms.ftrl.FTRLDAggerTrainer` | import for FTRL training | WIRED | Line 38: `from imitation.algorithms.ftrl import FTRLDAggerTrainer`; used in run_dagger() at line 229 |
| `experiments/atari_smoke.py` | `experiments/baselines/atari_random_scores.pkl` | loads cached random baselines for normalization | WIRED | `_BASELINES_PATH` defined at line 43 pointing to the pickle file; loaded in `load_random_baselines()` at line 52 |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| ENV-01 | 02-01-PLAN | 7 Atari games configured: Pong, Breakout, BeamRider, Enduro, Qbert, Seaquest, SpaceInvaders | SATISFIED | `ATARI_GAMES` dict in atari_helpers.py has all 7; TestAtariEnvSetup parametrized over all 7 |
| ENV-02 | 02-01-PLAN | Consistent Atari preprocessing: frame stacking (4), grayscale, 84x84 resize | SATISFIED | `make_atari_env` applies AtariWrapper (grayscale, 84x84, noop_max=30, frame_skip=4); `VecFrameStack(n_stack=4)` applied; result shape (4,84,84) asserted in tests |
| ENV-03 | 02-01-PLAN | Expert policies loaded from HuggingFace sb3 org for all 7 games | SATISFIED | `load_atari_expert` uses `organization="sb3"`; TestAtariExpertLoading verifies load for all 7 |
| ENV-04 | 02-01-PLAN | Expert observation space matches learner observation space (verified by assertion) | SATISFIED | TestAtariExpertLoading::test_expert_loads_and_obs_match asserts equality; smoke test contains runtime assertion |
| ENV-05 | 02-01-PLAN | Random baseline scores computed and cached for all 7 games | SATISFIED | `atari_random_scores.pkl` confirmed with 7 entries; TestRandomBaselines::test_baselines_has_all_games verifies all keys |
| INFRA-05 | 02-02-PLAN | Quick smoke-test config: 1-2 games, 1 seed, 3-5 DAgger rounds | SATISFIED | `atari_smoke.py` ran 2 games, 1 seed, 3 rounds; `smoke_test_output.log` confirms all 3 methods completed |
| INFRA-07 | 02-01-PLAN | Server setup script: create isolated Python env, install dependencies | SATISFIED | `setup/setup_server.sh` creates venv, installs `.[atari]`, runs `autorom --accept-license`, is executable |

**Orphaned requirements:** None. All 7 IDs claimed by plans are in REQUIREMENTS.md and mapped to Phase 2.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `experiments/smoke_test_output.log` | 16,19,22,191,194,197 | `Exception: an integer is required (got type bytes)` inside `warnings.warn()` | Info | These are warning message strings printed inside `UserWarning` via SB3's save_util.py — not raised exceptions. All 6 method runs completed successfully. No functional impact. |

No blockers or warnings found. The "Exception" strings are payload text within `UserWarning` messages from SB3's model deserialization compatibility shim, not unhandled Python exceptions.

### Human Verification Required

No items require human verification. All observable behaviors — environment creation, obs space shapes, expert loading, baseline values, and smoke test output — are verifiable from the codebase and log file.

### Gaps Summary

No gaps. All 9 truths are verified, all 8 artifacts are substantive and wired, all 6 key links are connected, and all 7 requirement IDs are satisfied with code evidence.

The only notable implementation deviation from the plan was reducing smoke test `total_timesteps` from 50000 to 8000 for CPU execution. This is a runtime parameter choice, not a structural gap — the script still accepts `--total-timesteps 50000` and will run with full budget on the GPU server (Phase 3).

---

_Verified: 2026-03-20_
_Verifier: Claude (gsd-verifier)_
