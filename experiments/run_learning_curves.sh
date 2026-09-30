#!/usr/bin/env bash
# Learning-curves sweep over the 8 classical MDPs.
#
# Scope: classical env-group × $EXP_ALGOS × 5 seeds, CPU-only
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
# Learning curves are saved under $EXP_PLOTS_CLASSICAL/learning_curves/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=./paths.sh
source experiments/paths.sh

RESULTS_DIR="$EXP_LC_CLASSICAL"
PLOTS_DIR="$EXP_PLOTS_CLASSICAL/learning_curves"
LOG_FILE="$EXP_LOG_DIR/classical.log"
mkdir -p "$RESULTS_DIR" "$PLOTS_DIR" "$EXP_LOG_DIR"

# CPU worker count: total - 2, floor 1.
# Classical cells are CPU-bound: linear policies on toy MDPs, one process each.
# Throughput scales with worker count, so the default takes the node's cores
# minus two for the driver -- but capped, because on a big shared node "all the
# cores" is both antisocial and a memory risk (each worker is its own torch
# process). N_WORKERS overrides outright; WORKER_CAP only raises the ceiling.
CPU_TOTAL="$(getconf _NPROCESSORS_ONLN)"
WORKER_CAP="${WORKER_CAP:-32}"
if [ -n "${N_WORKERS:-}" ]; then
    WORKERS="$N_WORKERS"
else
    WORKERS=$(( CPU_TOTAL - 2 ))
    if [ "$WORKERS" -gt "$WORKER_CAP" ]; then WORKERS="$WORKER_CAP"; fi
fi
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
echo "[learning_curves] launching with $WORKERS parallel workers on $CPU_TOTAL-core host" | tee -a "$LOG_FILE"
echo "[learning_curves] start time: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "[learning_curves] output dir: $RESULTS_DIR" | tee -a "$LOG_FILE"

# Algo set is overridable (ALGOS="ftl ftrl") so a targeted resume can re-run
# a subset without touching the completed cells of the other algorithms.
# Word splitting is intended here.
# shellcheck disable=SC2206
ALGO_SEL=(${ALGOS:-$EXP_ALGOS})

# ENVS="CartPole-v1 Taxi-v3" narrows the sweep to those environments. Without it
# the whole classical group runs. --env-group takes precedence over --envs in
# run_experiment, so the two are selected here rather than both being passed.
# shellcheck disable=SC2206
if [ -n "${ENVS:-}" ]; then ENV_SEL=(--envs ${ENVS}); else ENV_SEL=(--env-group classical); fi

python -m imitation.experiments.ftrl.run_experiment \
    "${ENV_SEL[@]}" \
    --algos "${ALGO_SEL[@]}" \
    --seeds 5 \
    --samples-per-round 1 \
    --n-rounds 1000 \
    --bc-n-epochs 20 \
    --eval-interval 10 \
    --output-dir "$RESULTS_DIR" \
    --expert-cache-dir "$EXP_EXPERT_CACHE" \
    --inner-early-stop \
    --no-outer-early-stop \
    --no-warm-start --beta-rampdown 0 \
    --n-workers "$WORKERS" \
    --n-gpus 0 \
    "$@" \
    2>&1 | tee -a "$LOG_FILE"

echo "[learning_curves] sweep done at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "[learning_curves] generating plots ..." | tee -a "$LOG_FILE"

python -m imitation.experiments.ftrl.plot_results \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$PLOTS_DIR" --flat-output \
    2>&1 | tee -a "$LOG_FILE"

echo "[learning_curves] complete. JSONs:"
find "$RESULTS_DIR" -mindepth 2 -maxdepth 2 -name "*.json" | wc -l | tee -a "$LOG_FILE"
echo "[learning_curves] PNGs in $PLOTS_DIR/:"
ls "$PLOTS_DIR/" 2>&1 | tee -a "$LOG_FILE"
