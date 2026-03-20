# Phase 4: Full Run and Analysis - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate publication-quality figures from the completed benchmark: per-game learning curves (normalized return vs environment interactions), aggregate metrics (mean and IQM with 95% CI across all 7 games), and a completion dashboard. Figures must work incrementally from partial Sacred output (runnable anytime mid-experiment for boss updates). All figures use consistent colors/styles and match Lavington et al. Figure 4 style.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — requirements are fully specified by REQUIREMENTS.md (EVAL-01 through EVAL-07). Key constraint: figures MUST support incremental generation from partial results (user requirement for boss updates every 2-3 days).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `experiments/atari_helpers.py` — ATARI_GAMES dict, compute_normalized_score
- `experiments/baselines/atari_random_scores.pkl` — cached random baselines for normalization
- Sacred FileStorageObserver output in `output/sacred/{algo}/{game}/{seed}/`
- `benchmarking/` directory with existing score normalization utilities

### Established Patterns
- Sacred `metrics.json` contains per-step logged scalars (reward, loss, eta_t, norm_g, round)
- Normalized score: `(agent - random) / (expert - random)`
- matplotlib for figure generation

### Integration Points
- Read Sacred output directories to collect per-round metrics
- Use `atari_random_scores.pkl` for random baselines
- Expert scores from HuggingFace policy rollouts (cached or computed)
- Output: PDF/PNG figures in `figures/` directory

</code_context>

<specifics>
## Specific Ideas

- Incremental figure generation: must work from partial Sacred output (some runs complete, others in progress)
- Match Lavington et al. Figure 4 style for aggregate comparison
- Completion dashboard showing which (algorithm, game, seed) runs are done/running/pending

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
