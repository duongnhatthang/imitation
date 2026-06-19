#!/usr/bin/env bash
# Canonical output paths for FTRL experiments.
# Source this file from launcher scripts; override any variable via env
# var to redirect a single run (e.g. EXP_LC_CLASSICAL=experiments/smoke/foo).
#
# Layout convention:
#   <root>/                 ← --output-dir passed to run_experiment.py
#     <env>/                ← per-env JSON results (written by runner)
#     tb/                   ← tensorboard (written by runner)
#     scratch/              ← per-seed training byproducts (written by runner)
#     plots/                ← PNGs (written by plot_results.py)
#     run.log               ← launcher tee log

EXP_LC_CLASSICAL="${EXP_LC_CLASSICAL:-experiments/learning_curves/classical}"
EXP_LC_ATARI="${EXP_LC_ATARI:-experiments/learning_curves/atari}"
EXP_LR_OBS_CLASSICAL="${EXP_LR_OBS_CLASSICAL:-experiments/lr_obs_heatmap/classical}"
EXP_LR_OBS_ATARI="${EXP_LR_OBS_ATARI:-experiments/lr_obs_heatmap/atari}"
EXP_SMOKE_DIR="${EXP_SMOKE_DIR:-experiments/smoke}"
EXP_CALIBRATION="${EXP_CALIBRATION:-experiments/calibration/lr_calibration.json}"
EXP_EXPERT_CACHE="${EXP_EXPERT_CACHE:-experiments/expert_cache}"
