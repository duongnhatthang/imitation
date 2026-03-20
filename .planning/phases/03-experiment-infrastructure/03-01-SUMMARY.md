---
phase: 03-experiment-infrastructure
plan: "01"
subsystem: infra
tags: [sacred, ftrl, atari, experiment-runner, file-storage-observer, imitation-logger]

# Dependency graph
requires:
  - phase: 02-atari-setup-and-smoke-test
    provides: run_bc, run_dagger, load_random_baselines helpers in atari_smoke.py
  - phase: 01-ftrl-algorithm
    provides: FTRLDAggerTrainer with eta_t/sigma_grad computed in extend_and_update

provides:
  - Sacred single-run entry point (run_atari_experiment.py) for (algo, game, seed) combinations
  - Per-round FTRL metric logging (ftrl/eta_t, ftrl/norm_g, ftrl/round) via imitation HierarchicalLogger
  - Isolated FileStorageObserver directories per (algo, game, seed) to prevent run-ID collisions
  - BC round-0 Sacred log_scalar entry for consistent output format across all algorithms

affects: [03-02-run-atari-benchmark, 04-analysis-and-figures]

# Tech tracking
tech-stack:
  added: [sacred, sacred.observers.FileStorageObserver]
  patterns:
    - Pre-parse argv with parse_known_args before Sacred consumes sys.argv to build observer path
    - Sacred FileStorageObserver per (algo/game/seed) leaf dir for no-collision parallel runs
    - FTRLDAggerTrainer._logger.record for per-round metrics (imitation HierarchicalLogger)
    - BC treated as round-0 algorithm; log_scalar with step=0

key-files:
  created:
    - experiments/run_atari_experiment.py
  modified:
    - src/imitation/algorithms/ftrl.py

key-decisions:
  - "Log ftrl/round before round_num increment so round number reflects current round (0-indexed)"
  - "Pre-parser defaults must match Sacred config defaults to prevent silent observer path mismatch when only with syntax is used"
  - "BC logs normalized_score with step=0 as round-0 entry (no per-round data for BC — single-round algorithm)"
  - "norm_g computed as sqrt(sum of squared L2 norms across all sigma_grad tensors) = global gradient norm"

patterns-established:
  - "Pattern: Sacred entry point pre-parses CLI flags with parse_known_args before run_commandline for observer setup"
  - "Pattern: FileStorageObserver path = {output_dir}/{algo}/{game}/{seed} for per-combination isolation"
  - "Pattern: FTRL per-round metrics logged via self._logger.record inside extend_and_update"

requirements-completed: [INFRA-01, INFRA-03, INFRA-04]

# Metrics
duration: 6min
completed: 2026-03-20
---

# Phase 03 Plan 01: Sacred Entry Point and FTRL Metric Logging Summary

**Sacred single-run entry point (run_atari_experiment.py) with isolated per-(algo,game,seed) FileStorageObserver dirs and per-round eta_t/norm_g logging added to FTRLDAggerTrainer**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-20T12:33:28Z
- **Completed:** 2026-03-20T12:39:05Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added ftrl/eta_t, ftrl/norm_g, and ftrl/round metric logging to FTRLDAggerTrainer.extend_and_update (INFRA-03)
- Created experiments/run_atari_experiment.py: Sacred experiment with config-driven algo/game/seed, isolated FileStorageObserver paths (INFRA-01, INFRA-04)
- BC normalized_score logged as step=0 Sacred scalar for consistent output format across all three algorithms (addresses RESEARCH.md Pitfall 6)
- All 14 existing FTRL tests pass after changes (264s test run, 0 failures)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add per-round metric logging to FTRLDAggerTrainer** - `6e145f6` (feat)
2. **Task 2: Create Sacred single-run entry point** - `d8eda89` (feat)

**Plan metadata:** committed with final docs commit

## Files Created/Modified
- `src/imitation/algorithms/ftrl.py` - Added norm_g computation and logger.record calls for ftrl/eta_t, ftrl/norm_g, ftrl/round after BC training in extend_and_update
- `experiments/run_atari_experiment.py` - Sacred entry point: @ex.config with algo/game/seed defaults, pre-parser for observer path, @ex.main calling run_bc/run_dagger, isolated FileStorageObserver

## Decisions Made
- Logged ftrl/round as self.round_num (before increment) so round number reflects the round just completed (0-indexed, consistent with DAgger's dagger/round_num)
- Pre-parser defaults in __main__ block explicitly documented to match Sacred config defaults — if they diverge, observer path will be wrong when Sacred `with key=val` syntax is used without --key flag
- BC uses log_scalar with step=0 (round-0) to ensure Phase 4 analysis code finds a normalized_score metric entry for every run regardless of algorithm

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- run_atari_experiment.py is ready to be invoked by the GNU parallel benchmark script (Plan 03-02)
- FTRL per-round metrics will be captured in Sacred metrics.json for Phase 4 analysis
- Sacred output structure: output/sacred/{algo}/{game}/{seed}/1/metrics.json

---
*Phase: 03-experiment-infrastructure*
*Completed: 2026-03-20*
