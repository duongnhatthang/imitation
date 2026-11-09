#!/usr/bin/env bash
# Shared paths for one experiment run. Export EXP_RUN_ID to resume or to group
# separate classical and Atari invocations. Otherwise each invocation starts a
# timestamped run. Raw artifacts never share the directory mirrored as plots.
export EXP_RUN_ID="${EXP_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export EXP_RUNS_DIR="${EXP_RUNS_DIR:-experiments/runs}"
export EXP_RUN_ROOT="${EXP_RUN_ROOT:-$EXP_RUNS_DIR/$EXP_RUN_ID}"
export EXP_LOG_DIR="${EXP_LOG_DIR:-$EXP_RUN_ROOT/logs}"
export EXP_LC_CLASSICAL="${EXP_LC_CLASSICAL:-$EXP_RUN_ROOT/data/classical}"
export EXP_LC_ATARI="${EXP_LC_ATARI:-$EXP_RUN_ROOT/data/atari}"
export EXP_PLOTS_CLASSICAL="${EXP_PLOTS_CLASSICAL:-$EXP_RUN_ROOT/plots/classical}"
export EXP_PLOTS_ATARI="${EXP_PLOTS_ATARI:-$EXP_RUN_ROOT/plots/atari}"
export EXP_LC_COVERAGE="${EXP_LC_COVERAGE:-$EXP_RUN_ROOT/data/coverage}"
export EXP_PLOTS_COVERAGE="${EXP_PLOTS_COVERAGE:-$EXP_RUN_ROOT/plots/coverage}"
export EXP_LR_OBS_CLASSICAL="${EXP_LR_OBS_CLASSICAL:-$EXP_RUN_ROOT/data/lr_obs_heatmap/classical}"
export EXP_LR_OBS_ATARI="${EXP_LR_OBS_ATARI:-$EXP_RUN_ROOT/data/lr_obs_heatmap/atari}"
export EXP_SMOKE_DIR="${EXP_SMOKE_DIR:-$EXP_RUN_ROOT/data/smoke}"
export EXP_SMOKE_ATARI="${EXP_SMOKE_ATARI:-$EXP_SMOKE_DIR/atari}"
export EXP_CALIBRATION="${EXP_CALIBRATION:-experiments/calibration/lr_calibration.json}"
export EXP_EXPERT_CACHE="${EXP_EXPERT_CACHE:-experiments/expert_cache}"
# The algorithm set a campaign runs. Override with ALGOS="ftl ftrl" for a
# targeted resume; the curve scripts read ALGOS first and fall back to this.
export EXP_ALGOS="${EXP_ALGOS:-ftl ftrl bc bc_iid}"
