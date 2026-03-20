---
phase: 03-experiment-infrastructure
plan: "02"
subsystem: infra
tags: [gnu-parallel, tmux, bash, sacred, cuda, atari, multi-gpu]

# Dependency graph
requires:
  - phase: 03-01
    provides: run_atari_experiment.py Sacred entry point that benchmark script invokes
  - phase: 02-atari-setup-and-smoke-test
    provides: atari_smoke.py helpers (run_bc, run_dagger) and ATARI_GAMES dict
provides:
  - "experiments/run_atari_benchmark.sh: single-command launcher for all 63 Atari benchmark runs"
  - "Multi-GPU orchestration: 4 GPUs via GNU parallel slot-based CUDA_VISIBLE_DEVICES assignment"
  - "tmux session wrapper for SSH-resilient long-running benchmark"
affects:
  - 04-analysis-and-figures

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "GNU parallel {%} slot → 0-indexed GPU: CUDA_VISIBLE_DEVICES=$(({%}-1))"
    - "tmux auto-relaunch: script detects TMUX absence and re-spawns itself with --_in-tmux flag"
    - "cartesian product via ::: separator for ALGOS x GAMES x SEEDS"
    - "--halt soon,fail=1 to stop scheduling on first failure without killing running jobs"

key-files:
  created:
    - experiments/run_atari_benchmark.sh
  modified: []

key-decisions:
  - "--jobs N_GPUS (4) limits concurrency to one job per GPU; {%} maps 1-indexed slot to 0-indexed GPU via (({%}-1))"
  - "tmux auto-relaunch pattern: check TMUX env var, re-spawn with --_in-tmux flag to avoid infinite recursion"
  - "Sacred CLI receives both --algo/--game/--seed (pre-parser) and with algo=X (Sacred config) in same invocation to ensure observer path and Sacred config stay aligned"
  - "--halt soon,fail=1 chosen over --halt now,fail=1 to let in-flight jobs complete before stopping"

patterns-established:
  - "Pattern: shell arrays + GNU parallel ::: for cartesian product benchmarks"
  - "Pattern: source experiments/common.sh first for OMP_NUM_THREADS and TIMESTAMP"

requirements-completed: [INFRA-02, INFRA-06]

# Metrics
duration: 2min
completed: 2026-03-20
---

# Phase 03 Plan 02: Experiment Infrastructure — Benchmark Orchestrator Summary

**GNU parallel benchmark script distributing 63 Atari runs (3 algos x 7 games x 3 seeds) across 4 GPUs inside a named tmux session with slot-based CUDA_VISIBLE_DEVICES assignment**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-20T05:21:41Z
- **Completed:** 2026-03-20T05:23:00Z
- **Tasks:** 2 (1 auto + 1 checkpoint auto-approved)
- **Files modified:** 1

## Accomplishments

- Created `experiments/run_atari_benchmark.sh` — single command launches all 63 benchmark jobs
- Distributed GPU assignment via `CUDA_VISIBLE_DEVICES=$(({%}-1))` maps GNU parallel's 1-indexed slot to GPU 0–3
- tmux auto-relaunch: script detects absence of `$TMUX`, spawns `atari_bench_<TIMESTAMP>` session, then exits; inner invocation skips the check via `--_in-tmux` flag
- Sacred CLI invocation passes both pre-parser flags (`--algo`, `--game`, `--seed`) and Sacred `with` syntax in the same command, ensuring observer path and Sacred config values stay aligned
- `--halt soon,fail=1` stops scheduling new jobs on first failure without killing running ones; `--joblog` records per-job timing and exit codes for debugging

## Task Commits

1. **Task 1: Create GNU parallel benchmark orchestration script** - `f4ce3d1` (feat)
2. **Task 2: Verify experiment infrastructure** - Auto-approved checkpoint (no commit)

## Files Created/Modified

- `experiments/run_atari_benchmark.sh` — Multi-GPU GNU parallel orchestrator with tmux session management (103 lines, chmod +x)

## Decisions Made

- Used `--halt soon,fail=1` (not `now,fail=1`) so already-running jobs on other GPUs can complete when one fails — avoids partial results for in-flight games
- tmux `--_in-tmux` flag pattern chosen over a separate `--inner` boolean to keep the script self-contained without argument parsing complexity
- Job log written to `${OUTPUT_DIR}/joblog.txt` (same directory as Sacred output) for easy correlation with experiment results

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Full experiment infrastructure complete: `run_atari_experiment.py` (Plan 01) + `run_atari_benchmark.sh` (Plan 02) are ready to launch on the CC-server
- Run `bash experiments/run_atari_benchmark.sh` from the project root on the CC-server to begin the benchmark
- Phase 04 analysis can read Sacred output from `output/sacred/{algo}/{game}/{seed}/1/metrics.json`
- No blockers for Phase 04

---
*Phase: 03-experiment-infrastructure*
*Completed: 2026-03-20*
