# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-19)

**Core value:** Fair, reproducible comparison of FTL vs FTRL vs BC across 7 Atari games with normalized scores and publication-quality figures
**Current focus:** Phase 1 — FTRL Algorithm

## Current Position

Phase: 1 of 4 (FTRL Algorithm)
Plan: 0 of 2 in current phase
Status: Ready to plan
Last activity: 2026-03-19 — Roadmap created; research complete

Progress: [░░░░░░░░░░] 0%

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

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Research]: Use `FTRLLossCalculator` injected into `BC` via constructor — do NOT use `LpRegularizer` or `WeightDecayRegularizer`; proximal term is `||theta - anchor||^2` not `||theta||^2`
- [Research]: Game list is the 7 seals games (Pong, Breakout, BeamRider, Enduro, Qbert, Seaquest, SpaceInvaders) — BeamRider/Enduro/Qbert HF expert coverage must be verified in Phase 2
- [Research]: GNU parallel with `slot()-1` for GPU assignment; no Ray cluster needed

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 2]: HumanCompatibleAI random-score datasets for Atari games are unconfirmed — may need to collect local random rollouts
- [Phase 2]: sb3 HF org expert coverage for BeamRider, Enduro, Qbert unconfirmed — game list may shrink
- [Phase 1]: Confirm whether `BC.__init__` already accepts `loss_calculator` before writing FTRLTrainer injection code

## Session Continuity

Last session: 2026-03-19
Stopped at: Roadmap created; no plans written yet
Resume file: None
