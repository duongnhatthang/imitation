# Phase 1: FTRL Algorithm - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement FTRLDAggerTrainer as a subclass of SimpleDAggerTrainer following Lavington et al. (2022) Eq. 6. All algorithmic decisions are dictated by the paper's formulation. This phase delivers the core FTRL algorithm with tests, touching no existing files except a minimal BC constructor change.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase. The algorithm formulation is fully specified by REQUIREMENTS.md (ALGO-01 through ALGO-06) and the paper's Eq. 6.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `SimpleDAggerTrainer` in `src/imitation/algorithms/dagger.py:552` — direct superclass for FTRL
- `DAggerTrainer` in `src/imitation/algorithms/dagger.py:294` — base DAgger with round logic, trajectory collection, `extend_and_update()`
- `BC` in `src/imitation/algorithms/bc.py:268` — handles supervised training, optimizer, loss computation
- Existing rollout infrastructure in `src/imitation/data/rollout.py`

### Established Patterns
- Algorithms subclass `BaseImitationAlgorithm` or domain-specific parents
- BC handles the inner training loop with `train()` method taking `n_epochs`/`n_batches`
- DAgger rounds: collect trajectories → extend dataset → BC update via `extend_and_update()`
- Sacred experiment management exists in `experiments/`

### Integration Points
- `FTRLDAggerTrainer` subclasses `SimpleDAggerTrainer` — override `train()` to add FTRL loss terms
- BC constructor may need a hook for custom loss functions (the "two-line change" per success criterion 1)
- Tests go in `tests/algorithms/` following existing patterns

</code_context>

<specifics>
## Specific Ideas

No specific requirements — implementation follows paper's Eq. 6 formulation exactly as specified in REQUIREMENTS.md.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
