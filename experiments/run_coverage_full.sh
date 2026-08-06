#!/usr/bin/env bash
# Full coverage run: an FTRL sweep (which now persists per-round demos for ALL
# four algos -- ftl, ftrl, bc, bc_dagger) followed by the three post-hoc
# visualizations (t-SNE state coverage, recoverability mu(s), runtime) plus the
# standard learning-curve plots. One command produces the whole deliverable.
#
# Phase-1 validated on CartPole-v1 (the only env whose DQN reference is
# hyperparameter-tuned).  t-SNE coverage + runtime work for any env; the
# recoverability mu(s) plot needs a near-optimal DQN, so for envs OTHER than
# CartPole-v1 the DQN trains with SB3 defaults and may be under-trained ->
# plot_recoverability prints a WARNING and that mu(s) figure is less reliable
# (per-env DQN tuning is Phase 2).
#
# Overrides (env vars):
#   ENVS           space-separated env ids               (default "CartPole-v1")
#   N_ROUNDS       DAgger rounds ~= states/algo for t-SNE density  (default 1000)
#   SEEDS          number of seeds                        (default 5)
#   N_GPUS         GPUs for the runner (classical is CPU) (default 0)
#   COVERAGE_SEED  which seed's states feed the coverage/mu plots (default 0)
#   EXP_LC_COVERAGE  output dir (default experiments/learning_curves/coverage)
#
# Density note: the t-SNE panels get ~= N_ROUNDS points per algorithm (bc gets
# one round-0 dump). N_ROUNDS=1000 gives dense, legible maps; the smoke driver
# (run_coverage_analysis.sh) uses 30 and is only a wiring check.
#
# Extra args ($@) forward to run_experiment (e.g. --force-rerun, --seeds).
#
# Resume: the runner is n_rounds-aware -- re-running with the same N_ROUNDS and
# dir skips completed (algo,env,seed) JSONs (safe after an interrupt/reboot);
# a higher N_ROUNDS re-runs those cells to the new depth.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=./paths.sh
source experiments/paths.sh

ENVS="${ENVS:-CartPole-v1}"
N_ROUNDS="${N_ROUNDS:-1000}"
SEEDS="${SEEDS:-5}"
N_GPUS="${N_GPUS:-0}"
COVERAGE_SEED="${COVERAGE_SEED:-0}"

RESULTS_DIR="$EXP_LC_COVERAGE"
PLOTS_DIR="$RESULTS_DIR/plots"
COV_DIR="$RESULTS_DIR/plots/coverage"
CACHE_DIR="$RESULTS_DIR/dqn_cache"
LOG_FILE="$RESULTS_DIR/run.log"
mkdir -p "$RESULTS_DIR" "$PLOTS_DIR" "$COV_DIR" "$CACHE_DIR"

# CPU worker count: total - 2, floor 1.
CPU_TOTAL="$(getconf _NPROCESSORS_ONLN)"
WORKERS=$(( CPU_TOTAL - 2 ))
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi

# shellcheck disable=SC2206  # word-splitting is intentional for the env list
ENV_ARR=($ENVS)

echo "[coverage_full] envs=[$ENVS] n_rounds=$N_ROUNDS seeds=$SEEDS workers=$WORKERS gpus=$N_GPUS" | tee -a "$LOG_FILE"
echo "[coverage_full] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "[coverage_full] output dir: $RESULTS_DIR" | tee -a "$LOG_FILE"

python -m imitation.experiments.ftrl.run_experiment \
    --envs "${ENV_ARR[@]}" \
    --algos ftl ftrl bc bc_dagger \
    --seeds "$SEEDS" \
    --samples-per-round 1 \
    --n-rounds "$N_ROUNDS" \
    --eval-interval 10 \
    --output-dir "$RESULTS_DIR" \
    --inner-early-stop \
    --no-outer-early-stop \
    --n-workers "$WORKERS" \
    --n-gpus "$N_GPUS" \
    "$@" \
    2>&1 | tee -a "$LOG_FILE"

echo "[coverage_full] sweep done: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"

echo "[coverage_full] learning-curve plots ..." | tee -a "$LOG_FILE"
python -m imitation.experiments.ftrl.plot_results \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$PLOTS_DIR" \
    2>&1 | tee -a "$LOG_FILE"

echo "[coverage_full] t-SNE coverage + recoverability per env ..." | tee -a "$LOG_FILE"
for env in "${ENV_ARR[@]}"; do
    echo "[coverage_full]   $env" | tee -a "$LOG_FILE"
    python -m imitation.experiments.ftrl.plot_tsne_coverage \
        --results-dir "$RESULTS_DIR" --env "$env" --seed "$COVERAGE_SEED" \
        --output-dir "$COV_DIR" \
        2>&1 | tee -a "$LOG_FILE"
    python -m imitation.experiments.ftrl.plot_recoverability \
        --results-dir "$RESULTS_DIR" --env "$env" --seed "$COVERAGE_SEED" \
        --cache-dir "$CACHE_DIR" --output-dir "$COV_DIR" \
        2>&1 | tee -a "$LOG_FILE"
done

echo "[coverage_full] runtime aggregation ..." | tee -a "$LOG_FILE"
python -m imitation.experiments.ftrl.aggregate_runtime \
    --results-dir "$RESULTS_DIR" --output-dir "$COV_DIR" \
    2>&1 | tee -a "$LOG_FILE"

echo "[coverage_full] done: $(date -u +%Y-%m-%dT%H:%M:%SZ). Coverage outputs:" | tee -a "$LOG_FILE"
ls "$COV_DIR/" 2>&1 | tee -a "$LOG_FILE"
