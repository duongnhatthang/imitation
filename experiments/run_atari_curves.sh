#!/usr/bin/env bash
# Learning-curves sweep over the 7 atari-zoo games (HuggingFace experts, no
# self-training). Linear mode only: frozen expert CNN features + a trainable
# action_net (the same architecture as the classical learning-curves sweep,
# via run_experiment's --policy-mode linear).
#
# Scope: atari-zoo x 4 algos x 5 seeds = 140 runs, GPU. run_experiment shares
# N_GPUS across the CPU workers via its gpu_queue (worker w -> gpu w % N_GPUS).
#
# Settings: policy_mode=linear, samples_per_round=1, inner-ES on, outer-ES off
# (every run goes to the full n_rounds so the curves are directly comparable to
# the classical sweep).
#
# n_rounds is intentionally tunable (env N_ROUNDS, default small) -- set it from
# the smoke run's measured per-round wall-clock so 140 runs finish in <= ~1 day.
#
# Output dir comes from experiments/paths.sh (EXP_LC_ATARI). Override per run,
# e.g. for a smoke test on a single game (ENVS replaces --env-group; the two are
# mutually exclusive in run_experiment):
#   EXP_LC_ATARI="$EXP_SMOKE_ATARI" ENVS=PongNoFrameskip-v4 N_ROUNDS=20 N_GPUS=1 \
#     ./experiments/run_atari_curves.sh --seeds 1 --force-rerun
#
# Env selection: by default the whole atari-zoo group; set ENVS="A B C" (space-
# separated game IDs) to run a subset instead. Do NOT pass --envs/--env-group in
# $@ — use ENVS so the script keeps the two mutually-exclusive flags consistent.
#
# Extra args ($@) forward to run_experiment, so you can pass --seeds,
# --force-rerun, etc. without editing the script.
#
# Stop-and-review gate: after this completes, review the PNGs in
# $EXP_LC_ATARI/plots/ before any further env scaling.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=./paths.sh
source experiments/paths.sh

N_ROUNDS="${N_ROUNDS:-50}"
N_GPUS="${N_GPUS:-4}"
ENVS="${ENVS:-}"
# Env selection is mutually exclusive in run_experiment: a named group OR an
# explicit list. ENVS (space-separated IDs) overrides the default atari-zoo group.
if [ -n "$ENVS" ]; then
    # shellcheck disable=SC2206  # word-splitting is intentional for the list
    ENV_SEL=(--envs $ENVS)
else
    ENV_SEL=(--env-group atari-zoo)
fi
RESULTS_DIR="$EXP_LC_ATARI"
PLOTS_DIR="$EXP_LC_ATARI/plots"
LOG_FILE="$EXP_LC_ATARI/run.log"
mkdir -p "$RESULTS_DIR" "$PLOTS_DIR"

# CPU worker count: total - 2, floor 1.
CPU_TOTAL="$(getconf _NPROCESSORS_ONLN)"
WORKERS=$(( CPU_TOTAL - 2 ))
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
echo "[atari_curves] $WORKERS workers, $N_GPUS GPUs, n_rounds=$N_ROUNDS" | tee -a "$LOG_FILE"
echo "[atari_curves] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "[atari_curves] output dir: $RESULTS_DIR" | tee -a "$LOG_FILE"

python -m imitation.experiments.ftrl.run_experiment \
    "${ENV_SEL[@]}" \
    --policy-mode linear \
    --algos ftl ftrl bc bc_dagger \
    --seeds 5 \
    --samples-per-round 1 \
    --n-rounds "$N_ROUNDS" \
    --eval-interval 5 \
    --output-dir "$RESULTS_DIR" \
    --inner-early-stop \
    --no-outer-early-stop \
    --n-workers "$WORKERS" \
    --n-gpus "$N_GPUS" \
    "$@" \
    2>&1 | tee -a "$LOG_FILE"

echo "[atari_curves] sweep done: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "[atari_curves] plotting ..." | tee -a "$LOG_FILE"

python -m imitation.experiments.ftrl.plot_results \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$PLOTS_DIR" \
    2>&1 | tee -a "$LOG_FILE"

echo "[atari_curves] JSONs:"
find "$RESULTS_DIR" -name "*.json" -not -path "*/scratch/*" -not -path "*/tb/*" | wc -l | tee -a "$LOG_FILE"
echo "[atari_curves] PNGs in $PLOTS_DIR/:"
ls "$PLOTS_DIR/" 2>&1 | tee -a "$LOG_FILE"
