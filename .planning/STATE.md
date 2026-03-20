---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
stopped_at: Completed 02-atari-setup-and-smoke-test/02-01-PLAN.md
last_updated: "2026-03-20T06:07:36.020Z"
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 4
  completed_plans: 3
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-19)

**Core value:** Fair, reproducible comparison of FTL vs FTRL vs BC across 7 Atari games with normalized scores and publication-quality figures
**Current focus:** Phase 02 — atari-setup-and-smoke-test

## Current Position

Phase: 02 (atari-setup-and-smoke-test) — EXECUTING
Plan: 1 of 2

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

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 2]: HumanCompatibleAI random-score datasets for Atari games are unconfirmed — may need to collect local random rollouts
- [Phase 2]: sb3 HF org expert coverage for BeamRider, Enduro, Qbert unconfirmed — game list may shrink
- [Phase 1]: Confirm whether `BC.__init__` already accepts `loss_calculator` before writing FTRLTrainer injection code

## Session Continuity

Last session: 2026-03-20T06:07:36.018Z
Stopped at: Completed 02-atari-setup-and-smoke-test/02-01-PLAN.md
Resume file: None
