#!/usr/bin/env bash
# Learning-curves sweep over the 8 classical MDPs.
#
# Scope: classical env-group × 4 algos × 5 seeds = 160 runs, CPU-only
# (these MDPs ignore GPU per run_experiment.py device selection).
#
# Settings: samples_per_round=1, n_rounds=1000, bc_n_epochs=20,
# inner-ES on, outer-ES OFF (every run goes to the full 1000 rounds so
# the learning curves are directly comparable).
#
# Output paths come from experiments/paths.sh (override EXP_LC_CLASSICAL to
# redirect a single run, e.g. to experiments/smoke/<name> for a smoke test).
#
# Extra args ($@) forward to run_experiment, so you can pass --force-rerun,
# --seeds 3, --envs CartPole-v1, etc. without editing the script.
#
# Stop-and-review gate: after this completes, review the PNGs in
# $EXP_LC_CLASSICAL/plots/ before any further env scaling.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=./paths.sh
source experiments/paths.sh

RESULTS_DIR="$EXP_LC_CLASSICAL"
PLOTS_DIR="$EXP_LC_CLASSICAL/plots"
LOG_FILE="$EXP_LC_CLASSICAL/run.log"
mkdir -p "$RESULTS_DIR" "$PLOTS_DIR"

# CPU worker count: total - 2, floor 1.
CPU_TOTAL="$(getconf _NPROCESSORS_ONLN)"
WORKERS=$(( CPU_TOTAL - 2 ))
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
echo "[learning_curves] launching with $WORKERS parallel workers on $CPU_TOTAL-core host" | tee -a "$LOG_FILE"
echo "[learning_curves] start time: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "[learning_curves] output dir: $RESULTS_DIR" | tee -a "$LOG_FILE"

python -m imitation.experiments.ftrl.run_experiment \
    --env-group classical \
    --algos ftl ftrl bc bc_dagger \
    --seeds 5 \
    --samples-per-round 1 \
    --n-rounds 1000 \
    --bc-n-epochs 20 \
    --eval-interval 10 \
    --output-dir "$RESULTS_DIR" \
    --inner-early-stop \
    --no-outer-early-stop \
    --n-workers "$WORKERS" \
    --n-gpus 0 \
    "$@" \
    2>&1 | tee -a "$LOG_FILE"

echo "[learning_curves] sweep done at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "[learning_curves] generating plots ..." | tee -a "$LOG_FILE"

python -m imitation.experiments.ftrl.plot_results \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$PLOTS_DIR" \
    2>&1 | tee -a "$LOG_FILE"

echo "[learning_curves] complete. JSONs:"
find "$RESULTS_DIR" -name "*.json" -not -path "*/scratch/*" -not -path "*/tb/*" | wc -l | tee -a "$LOG_FILE"
echo "[learning_curves] PNGs in $PLOTS_DIR/:"
ls "$PLOTS_DIR/" 2>&1 | tee -a "$LOG_FILE"
