#!/usr/bin/env bash
# Local wave-1 sweep for the inner-early-stop + epoch-fix branch.
#
# Scope: 3 envs × 4 algos × 5 seeds = 60 runs, CPU-only (these MDPs
# ignore GPU per run_experiment.py device selection).
#
# Settings from the wave-1 plan (samples_per_round=1, n_rounds=1000,
# bc_n_epochs=20, inner-ES on, outer-ES on with disagreement_rate signal).
#
# Results -> experiments/results_wave1/
# Plots   -> experiments/plots_wave1/
#
# Stop-and-review gate: after this completes, the user reviews the 3 PNGs
# before any further env scaling.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RESULTS_DIR="experiments/results_wave1"
PLOTS_DIR="experiments/plots_wave1"
LOG_FILE="experiments/results_wave1/wave1_overall.log"
mkdir -p "$RESULTS_DIR" "$PLOTS_DIR"

# CPU worker count: total - 2, floor 1.
CPU_TOTAL="$(getconf _NPROCESSORS_ONLN)"
WORKERS=$(( CPU_TOTAL - 2 ))
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
echo "[wave1] launching with $WORKERS parallel workers on $CPU_TOTAL-core host" | tee -a "$LOG_FILE"
echo "[wave1] start time: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"

python -m imitation.experiments.ftrl.run_experiment \
    --envs Blackjack-v1 FrozenLake-v1 CartPole-v1 \
    --algos ftl ftrl bc bc_dagger \
    --seeds 5 \
    --samples-per-round 1 \
    --n-rounds 1000 \
    --bc-n-epochs 20 \
    --eval-interval 10 \
    --output-dir "$RESULTS_DIR" \
    --inner-early-stop \
    --outer-early-stop \
    --n-workers "$WORKERS" \
    --n-gpus 0 \
    2>&1 | tee -a "$LOG_FILE"

echo "[wave1] sweep done at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "[wave1] generating plots ..." | tee -a "$LOG_FILE"

python -m imitation.experiments.ftrl.plot_results \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$PLOTS_DIR" \
    2>&1 | tee -a "$LOG_FILE"

echo "[wave1] complete. JSONs:"
find "$RESULTS_DIR" -name "*.json" | wc -l | tee -a "$LOG_FILE"
echo "[wave1] PNGs in $PLOTS_DIR/:"
ls "$PLOTS_DIR/" 2>&1 | tee -a "$LOG_FILE"
