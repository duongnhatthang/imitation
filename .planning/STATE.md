---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
stopped_at: Completed 04-full-run-and-analysis/04-01-PLAN.md
last_updated: "2026-03-20T13:44:10.537Z"
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 8
  completed_plans: 8
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-19)

**Core value:** Fair, reproducible comparison of FTL vs FTRL vs BC across 7 Atari games with normalized scores and publication-quality figures
**Current focus:** Phase 04 — full-run-and-analysis

## Current Position

Phase: 04 (full-run-and-analysis) — COMPLETE
Plan: 2 of 2

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
| Phase 01-ftrl-algorithm P01 | 4 | 2 tasks | 2 files |
| Phase 01-ftrl-algorithm P02 | 35 | 2 tasks | 2 files |
| Phase 02-atari-setup-and-smoke-test P01 | 10 | 2 tasks | 5 files |
| Phase 02 P02 | 353 | 2 tasks | 4 files |
| Phase 03 P01 | 6 | 2 tasks | 2 files |
| Phase 03-experiment-infrastructure P02 | 2 | 2 tasks | 1 files |
| Phase 04-full-run-and-analysis P01 | 4 | 2 tasks | 3 files |
| Phase 04-full-run-and-analysis P02 | 6 | 2 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Research]: Use `FTRLLossCalculator` injected into `BC` via constructor — do NOT use `LpRegularizer` or `WeightDecayRegularizer`; proximal term is `||theta - anchor||^2` not `||theta||^2`
- [Research]: Game list is the 7 seals games (Pong, Breakout, BeamRider, Enduro, Qbert, Seaquest, SpaceInvaders) — BeamRider/Enduro/Qbert HF expert coverage must be verified in Phase 2
- [Research]: GNU parallel with `slot()-1` for GPU assignment; no Ray cluster needed
- [Phase 01-ftrl-algorithm]: eta_t = alpha / cumulative_sigma so large alpha = FTL degeneracy (ALGO-05); sigma_i = 1.0 constant (eta_t = alpha/t)
- [Phase 01-ftrl-algorithm]: BC loss_calculator injection via optional constructor param; FTRLLossCalculator uses dataclasses.replace to return BCTrainingMetrics
- [Phase 01-ftrl-algorithm]: sigma_grad must be normalized by n_batches (per-batch average) to prevent linear correction term from dominating BC loss in later rounds
- [Phase 01-ftrl-algorithm]: FTL degeneracy test validates code property (proximal_coeff < 1e-6) not reward outcome; CartPole smoke uses alpha=100.0 with 35000 timesteps for n_envs stability
- [Phase 02-atari-setup-and-smoke-test]: VecTransposeImage required after VecFrameStack(4) to produce (4,84,84) channels-first obs matching SB3 expert; without it ENV-04 obs space assertion fails
- [Phase 02-atari-setup-and-smoke-test]: gym==0.26.2 must be installed alongside gymnasium; HF sb3 models were pickled with old gym module and require it for cloudpickle deserialization
- [Phase 02-atari-setup-and-smoke-test]: rollout.rollout(unwrap=False) used for random baseline collection to bypass RolloutInfoWrapper requirement; rollout_stats works correctly without unwrapping
- [Phase 02-atari-setup-and-smoke-test]: Random baselines cached: Pong=-20.42, Breakout=0.30, BeamRider=120.0, Enduro=0.0, Qbert=40.15, Seaquest=21.82, SpaceInvaders=63.48 (30 episodes, seed=0)
- [Phase 02]: Use total_timesteps=8000 for CPU smoke test (50000 produces 80+ DAgger rounds, O(n^2) BC training growth, 6+ hours on CPU vs 30min GPU estimate)
- [Phase 02]: serialize.py: clamp num_shards=max(1,len(ds)) to prevent HuggingFace IndexError when dataset has fewer rows than default shard count (short Breakout episodes)
- [Phase 03-01]: Log ftrl/round before round_num increment so round number reflects current round (0-indexed, consistent with dagger/round_num)
- [Phase 03-01]: Pre-parser defaults must match Sacred config defaults to prevent silent observer path mismatch when only Sacred with syntax is used
- [Phase 03-01]: BC logs normalized_score with step=0 as round-0 entry so Phase 4 analysis finds a metric entry for every run regardless of algorithm
- [Phase 03-02]: --jobs N_GPUS limits concurrency to 1 job/GPU; {%} maps 1-indexed slot to 0-indexed GPU via (({%}-1)); tmux auto-relaunch via --_in-tmux flag pattern; --halt soon,fail=1 lets in-flight jobs complete before stopping
- [Phase 04-full-run-and-analysis]: Use typing.Optional for Python 3.8 compat in analyze_results.py
- [Phase 04-02]: rliable plot_interval_estimates colors param must be dict[algo_name->color] not list (rliable 1.2.0 API)
- [Phase 04-02]: BC plotted as axhline (single step=0 entry); NaN filled with 0.0 before rliable call with missing count annotated in figure title

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 2]: HumanCompatibleAI random-score datasets for Atari games are unconfirmed — may need to collect local random rollouts
- [Phase 2]: sb3 HF org expert coverage for BeamRider, Enduro, Qbert unconfirmed — game list may shrink
- [Phase 1]: Confirm whether `BC.__init__` already accepts `loss_calculator` before writing FTRLTrainer injection code

## Session Continuity

Last session: 2026-03-20T13:15:47.172Z
Stopped at: Completed 04-full-run-and-analysis/04-01-PLAN.md
Resume file: None
