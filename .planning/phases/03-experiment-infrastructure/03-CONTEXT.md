# Phase 3: Experiment Infrastructure - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the multi-GPU experiment orchestration layer: a single `run_atari_benchmark.sh` command that launches all 84+ experiment combinations (3 algorithms x 7 games x 4+ seeds) across 4 GPUs with GNU parallel, isolated Sacred logging per experiment, per-round metric logging, and tmux session management for SSH resilience.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase. Requirements are fully specified by REQUIREMENTS.md (INFRA-01 through INFRA-06).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `experiments/atari_helpers.py` — ATARI_GAMES dict, make_atari_training_venv, load_atari_expert, compute_normalized_score
- `experiments/atari_smoke.py` — smoke test runner with BC/DAgger/FTRL training logic (can be refactored into single-run entry point)
- Sacred experiment infrastructure in `experiments/` and `src/imitation/scripts/`
- `experiments/baselines/atari_random_scores.pkl` — cached random baselines

### Established Patterns
- Sacred FileStorageObserver for experiment logging
- Named configs per game in `src/imitation/scripts/config/train_imitation.py`
- `atari_smoke.py` already has the training loop structure for all 3 methods

### Integration Points
- GNU parallel for multi-GPU job distribution with CUDA_VISIBLE_DEVICES
- tmux for session persistence across SSH disconnects
- Sacred observers need separate directories per (algorithm, game, seed) to avoid ID collisions

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure follows standard GPU parallelism and Sacred patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
