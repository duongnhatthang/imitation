#!/usr/bin/env bash
# Phase 1 (CartPole) coverage / recoverability / runtime analysis.
#
# 1. Ensure a CartPole sweep with retained scratch demos exists (re-run a short
#    sweep only if scratch is missing).
# 2. Generate the three visualizations into $OUT/plots/coverage/.
#
# Extra args ($@) forward to run_experiment (e.g. --force-rerun).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV="CartPole-v1"
RESULTS_DIR="${COVERAGE_RESULTS_DIR:-experiments/coverage/classical}"
PLOTS_DIR="$RESULTS_DIR/plots/coverage"
CACHE_DIR="${COVERAGE_CACHE_DIR:-experiments/coverage/dqn_cache}"
mkdir -p "$PLOTS_DIR" "$CACHE_DIR"

# Require every algo's demos (not just ftrl): bc/bc_dagger persist demos too, and a
# stale ftl/ftrl-only dir would otherwise skip the sweep and drop the offline panels.
NEED_SWEEP=0
for algo in ftl ftrl bc bc_dagger; do
  if [ ! -d "$RESULTS_DIR/scratch/${algo}_${ENV}_seed0/demos" ]; then
    NEED_SWEEP=1
  fi
done
if [ "$NEED_SWEEP" -eq 1 ]; then
  echo "[coverage] missing scratch demos for one or more algos; running short CartPole sweep ..."
  python -m imitation.experiments.ftrl.run_experiment \
      --envs "$ENV" \
      --algos ftl ftrl bc bc_dagger \
      --seeds 1 \
      --samples-per-round 1 \
      --n-rounds 30 \
      --bc-n-epochs 20 \
      --eval-interval 5 \
      --output-dir "$RESULTS_DIR" \
      --inner-early-stop \
      --no-outer-early-stop \
      --n-workers 4 \
      --n-gpus 0 \
      "$@"
else
  echo "[coverage] reusing existing scratch demos in $RESULTS_DIR/scratch/"
fi

echo "[coverage] t-SNE coverage ..."
python -m imitation.experiments.ftrl.plot_tsne_coverage \
    --results-dir "$RESULTS_DIR" --env "$ENV" --seed 0 --output-dir "$PLOTS_DIR"

echo "[coverage] recoverability ..."
python -m imitation.experiments.ftrl.plot_recoverability \
    --results-dir "$RESULTS_DIR" --env "$ENV" --seed 0 \
    --cache-dir "$CACHE_DIR" --output-dir "$PLOTS_DIR"

echo "[coverage] runtime ..."
python -m imitation.experiments.ftrl.aggregate_runtime \
    --results-dir "$RESULTS_DIR" --output-dir "$PLOTS_DIR"

echo "[coverage] done. Outputs in $PLOTS_DIR/"
ls "$PLOTS_DIR/"
