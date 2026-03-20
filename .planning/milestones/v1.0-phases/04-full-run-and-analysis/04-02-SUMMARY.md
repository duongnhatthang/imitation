---
phase: 04-full-run-and-analysis
plan: "02"
subsystem: analysis
tags: [matplotlib, rliable, numpy, learning-curves, aggregate-metrics, IQM, publication-figures]

# Dependency graph
requires:
  - phase: 04-01
    provides: "load_sacred_run, collect_results, print_dashboard, plot_config.py with COLOR_MAP/LINESTYLE_MAP/RCPARAMS"
provides:
  - "build_score_matrix: (n_seeds, n_games) array with NaN for missing runs"
  - "plot_learning_curves: 2x4 subplot grid per-game curves with BC horizontal line and DAgger/FTRL mean+std bands"
  - "plot_aggregate: rliable IQM + mean with stratified bootstrap 95% CI, colors from plot_config"
  - "CLI curves and aggregate subcommands with --output-dir and --figures-dir flags"
  - "26 passing tests covering EVAL-01 through EVAL-07"
affects:
  - "figures/learning_curves.pdf, figures/aggregate.pdf — final paper deliverables"

# Tech tracking
tech-stack:
  added: ["rliable==1.2.0 (IQM + stratified bootstrap CI)"]
  patterns:
    - "matplotlib.use('Agg') at module top for headless server compatibility"
    - "NaN-fill then nan_to_num for rliable score matrices (rliable cannot handle NaN)"
    - "rliable colors param must be dict[algo_name -> color], not list"
    - "BC plotted as axhline (single step=0); DAgger/FTRL use interpolated mean+std bands"

key-files:
  created: []
  modified:
    - "experiments/analyze_results.py"
    - "tests/test_analysis.py"

key-decisions:
  - "rliable plot_interval_estimates colors param must be a dict mapping algo name to color (not a list) — verified from rliable 1.2.0 source"
  - "BC plotted as axhline at final normalized score (not per-round) since only step=0 is logged"
  - "NaN filled with 0.0 before rliable call; missing run count annotated in figure suptitle"
  - "Round-to-timestep x-axis conversion: timestep = round * (total_timesteps / n_rounds) from run config"

patterns-established:
  - "Score matrix assembly: build_score_matrix(results, algo, games, seeds) -> (n_seeds, n_games) with NaN"
  - "Headless figure generation: matplotlib.use('Agg') before any pyplot import"

requirements-completed: [EVAL-02, EVAL-03, EVAL-06]

# Metrics
duration: 6min
completed: 2026-03-20
---

# Phase 4 Plan 02: Full Run and Analysis (Figures) Summary

**Per-game learning curves (2x4 subplot grid, BC as horizontal line, mean+std seed bands) and rliable IQM+mean aggregate figure with 95% CI, both saved as PDF and PNG, with graceful partial-results handling**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-20T13:17:28Z
- **Completed:** 2026-03-20T13:23:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Implemented `build_score_matrix` producing (n_seeds, n_games) arrays with NaN for missing runs
- Implemented `plot_learning_curves` with 2x4 subplot grid, BC as axhline, DAgger/FTRL with mean+std fill_between, Expert reference at y=1.0, per-game individual PDFs, and completion count annotation
- Implemented `plot_aggregate` using rliable `get_interval_estimates` (50000 bootstrap reps) for IQM and mean with 95% CI; NaN filled with 0.0 and missing count annotated
- Wired `curves` and `aggregate` CLI subcommands with `--output-dir` and `--figures-dir` flags
- Added 16 new tests (26 total) covering all EVAL requirements for figures

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement curves and aggregate subcommands in analyze_results.py** - `e817514` (feat)
2. **Task 2: Add tests for learning curves and aggregate figure generation** - `903b4c1` (test)

## Files Created/Modified
- `experiments/analyze_results.py` - Added build_score_matrix, plot_learning_curves, plot_aggregate, _save_single_game_curve; wired CLI subcommands; added matplotlib.use("Agg")
- `tests/test_analysis.py` - Extended with 16 new tests for EVAL-02, EVAL-03, EVAL-06; extended _create_mock_run helper with n_rounds/total_timesteps params

## Decisions Made
- **rliable colors must be dict:** `plot_interval_estimates` in rliable 1.2.0 expects `colors` as `dict[str, color]` not a list — discovered during GREEN phase, auto-fixed inline
- **BC as axhline:** BC only logs step=0 so it is rendered as a horizontal line at its final score value, not an interpolated multi-round curve
- **NaN policy:** Missing runs fill with 0.0 before rliable (documented with per-algo missing count printed); NaN matrix returned by build_score_matrix before that conversion for tests to verify

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] rliable colors parameter must be dict, not list**
- **Found during:** Task 2 (test_aggregate_figure_creation RED phase)
- **Issue:** `plot_interval_estimates(colors=[...])` raised `TypeError: list indices must be integers or slices, not str` — rliable 1.2.0 expects colors as `dict[algo_name -> color]` not a list
- **Fix:** Changed `colors=[COLOR_MAP[k] for k in scores_dict.keys()]` to `colors_dict = {k: COLOR_MAP[k] for k in scores_dict.keys()}`
- **Files modified:** `experiments/analyze_results.py`
- **Verification:** `test_aggregate_figure_creation` passes; 26/26 tests green
- **Committed in:** `903b4c1` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Necessary for correct rliable 1.2.0 API usage. No scope creep.

## Issues Encountered
- rliable was not installed in the project environment — auto-installed with `pip install rliable==1.2.0` (Rule 3 - Blocking). rliable does not expose `__version__` attribute; verified by importing `from rliable import library, metrics, plot_utils`.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All figure generation functions ready: `python experiments/analyze_results.py curves --output-dir <path>` and `python experiments/analyze_results.py aggregate --output-dir <path>`
- Figures work on partial Sacred output at any point mid-benchmark (EVAL-04 satisfied)
- Phase 04 is complete: dashboard (Plan 01) + learning curves + aggregate figure (Plan 02)

---
*Phase: 04-full-run-and-analysis*
*Completed: 2026-03-20*
