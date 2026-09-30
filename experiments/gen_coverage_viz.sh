#!/usr/bin/env bash
# Generate coverage / recoverability / runtime viz for an already-run sweep.
# Usage: ./experiments/gen_coverage_viz.sh <results_dir> <env> [env ...]
# CPU-only: safe to run after a GPU Atari sweep. Recoverability dispatches by
# env family (exact env.P for toy-text, hub DQN for Atari, trained DQN for
# continuous classical; Blackjack skipped).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
RESULTS_DIR="${1:?usage: gen_coverage_viz.sh <results_dir> <env...>}"; shift
RESULTS_DIR="${RESULTS_DIR%/}"
if [ -z "${PLOT_ROOT:-}" ]; then
  case "$RESULTS_DIR" in
    */data/*) PLOT_ROOT="${RESULTS_DIR%/data/*}/plots/${RESULTS_DIR##*/data/}" ;;
    *) PLOT_ROOT="$RESULTS_DIR/plots" ;;
  esac
fi
COV_DIR="$PLOT_ROOT/coverage"
RUNTIME_DIR="$PLOT_ROOT/runtime"
CACHE_DIR="$RESULTS_DIR/dqn_cache"
DATASET_SEEDS="${COVERAGE_SEEDS:-$(seq 0 $((${SEEDS:-5} - 1)))}"
TSNE_SEEDS="${TSNE_SEEDS:-0 1 2}"
# Where cached PPO experts live (Atari CNN features + toy-text exact-mu expert).
EXPERT_CACHE="${EXPERT_CACHE:-${EXP_EXPERT_CACHE:-experiments/expert_cache}}"
mkdir -p "$COV_DIR" "$CACHE_DIR" "$RUNTIME_DIR"
for env in "$@"; do
  for seed in $DATASET_SEEDS; do
    echo "[gen_coverage_viz] $env dataset seed $seed"
    python -m imitation.experiments.ftrl.plot_tsne_coverage \
        --results-dir "$RESULTS_DIR" --env "$env" --seed "$seed" --output-dir "$COV_DIR" \
        --tsne-seed $TSNE_SEEDS --expert-cache "$EXPERT_CACHE"
    if [ "${RUN_RECOVERABILITY:-0}" = "1" ]; then
      python -m imitation.experiments.ftrl.plot_recoverability \
          --results-dir "$RESULTS_DIR" --env "$env" --seed "$seed" \
          --cache-dir "$CACHE_DIR" --output-dir "$COV_DIR/recoverability_seed$seed" \
          --expert-cache "$EXPERT_CACHE"
    fi
  done
done
python -m imitation.experiments.ftrl.aggregate_runtime \
    --results-dir "$RESULTS_DIR" --output-dir "$RUNTIME_DIR"
echo "[gen_coverage_viz] done -> $COV_DIR"; ls "$COV_DIR"
