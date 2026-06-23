#!/usr/bin/env bash
# E1: L2-lambda sweep over the 8 classical MDPs (FTRL algo).
#
# Runs FTRL across a grid of constant L2 weights into per-lambda output dirs.
# lambda=0 reproduces FTL; lambda=0.01 reproduces the FTRL learning-curves run.
# Each lambda gets its own --output-dir so tb/scratch/JSONs never collide.
#
# Extra args ($@) forward to run_experiment (e.g. --envs CartPole-v1 Acrobot-v1
# --seeds 1 for a smoke test, or --force-rerun).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${EXP_L2_SWEEP:-experiments/l2_sweep/classical}"
SEEDS="${L2_SWEEP_SEEDS:-3}"
LAMBDAS=(0 1e-4 1e-3 1e-2 1e-1 1)

CPU_TOTAL="$(getconf _NPROCESSORS_ONLN)"
WORKERS=$(( CPU_TOTAL - 2 ))
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi

mkdir -p "$BASE_DIR"
LOG_FILE="$BASE_DIR/run.log"
echo "[l2_sweep] start $(date -u +%Y-%m-%dT%H:%M:%SZ) workers=$WORKERS seeds=$SEEDS" | tee -a "$LOG_FILE"

# Detect if caller is overriding --envs; if so, skip --env-group (they are mutually exclusive).
ENV_GROUP_ARG=(--env-group classical)
for arg in "$@"; do
    if [ "$arg" = "--envs" ]; then
        ENV_GROUP_ARG=()
        break
    fi
done

for lam in "${LAMBDAS[@]}"; do
    tag="$(printf '%s' "$lam" | tr '.' 'p')"
    out_dir="$BASE_DIR/lam_${tag}"
    mkdir -p "$out_dir"
    echo "[l2_sweep] lambda=$lam -> $out_dir" | tee -a "$LOG_FILE"
    python -m imitation.experiments.ftrl.run_experiment \
        ${ENV_GROUP_ARG[@]+"${ENV_GROUP_ARG[@]}"} \
        --algos ftrl \
        --l2-lambda "$lam" \
        --seeds "$SEEDS" \
        --samples-per-round 1 \
        --n-rounds 1000 \
        --bc-n-epochs 20 \
        --eval-interval 10 \
        --output-dir "$out_dir" \
        --inner-early-stop \
        --no-outer-early-stop \
        --n-workers "$WORKERS" \
        --n-gpus 0 \
        "$@" \
        2>&1 | tee -a "$LOG_FILE"
done

echo "[l2_sweep] done $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
