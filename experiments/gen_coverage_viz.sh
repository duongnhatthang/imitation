#!/usr/bin/env bash
# Generate coverage / recoverability / runtime viz for an already-run sweep.
# Usage: ./experiments/gen_coverage_viz.sh <results_dir> <env> [env ...]
# CPU-only: safe to run after a GPU Atari sweep. Recoverability dispatches by
# env family (exact env.P for toy-text, hub DQN for Atari, trained DQN for
# continuous classical; Blackjack skipped).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
RESULTS_DIR="${1:?usage: gen_coverage_viz.sh <results_dir> <env...>}"; shift
COV_DIR="$RESULTS_DIR/plots/coverage"
CACHE_DIR="$RESULTS_DIR/dqn_cache"
SEED="${COVERAGE_SEED:-0}"
# Where cached PPO experts live (Atari CNN features + toy-text exact-mu expert).
EXPERT_CACHE="${EXPERT_CACHE:-experiments/expert_cache}"
mkdir -p "$COV_DIR" "$CACHE_DIR"
for env in "$@"; do
  echo "[gen_coverage_viz] $env"
  python -m imitation.experiments.ftrl.plot_tsne_coverage \
      --results-dir "$RESULTS_DIR" --env "$env" --seed "$SEED" --output-dir "$COV_DIR" \
      --expert-cache "$EXPERT_CACHE"
  python -m imitation.experiments.ftrl.plot_recoverability \
      --results-dir "$RESULTS_DIR" --env "$env" --seed "$SEED" \
      --cache-dir "$CACHE_DIR" --output-dir "$COV_DIR" --expert-cache "$EXPERT_CACHE"
done
python -m imitation.experiments.ftrl.aggregate_runtime \
    --results-dir "$RESULTS_DIR" --output-dir "$COV_DIR"
echo "[gen_coverage_viz] done -> $COV_DIR"; ls "$COV_DIR"
