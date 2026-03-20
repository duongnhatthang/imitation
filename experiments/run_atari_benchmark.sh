#!/usr/bin/env bash
# run_atari_benchmark.sh — Multi-GPU GNU parallel orchestrator for Atari benchmark.
#
# Launches all 63 (algo x game x seed) combinations across 4 GPUs using GNU
# parallel, inside a tmux session that survives SSH disconnect.
#
# Usage (from project root):
#   bash experiments/run_atari_benchmark.sh
#
# The script will auto-relaunch itself inside a named tmux session if not
# already running inside one. Attach with the printed command.
set -euo pipefail

source experiments/common.sh

# ---------------------------------------------------------------------------
# Configuration (INFRA-06: 7 games x 3 algos x 3 seeds = 63 combinations)
# ---------------------------------------------------------------------------
GAMES=(Pong Breakout BeamRider Enduro Qbert Seaquest SpaceInvaders)
ALGOS=(bc dagger ftrl)
SEEDS=(0 1 2)
N_ROUNDS=20
TOTAL_TIMESTEPS=500000
N_ENVS=4
ALPHA=1.0
N_GPUS=4
OUTPUT_DIR="output/sacred/${TIMESTAMP}"

# ---------------------------------------------------------------------------
# tmux auto-relaunch (INFRA-02)
# ---------------------------------------------------------------------------
# If not already inside a tmux session, re-launch this script inside one so
# the benchmark survives SSH disconnection. The inner invocation receives
# --_in-tmux to skip this check.
if [[ "${1:-}" != "--_in-tmux" ]] && [ -z "${TMUX:-}" ]; then
  SESSION="atari_bench_$(date +%Y%m%d_%H%M%S)"
  tmux new-session -d -s "$SESSION" \
    "bash $0 --_in-tmux; echo 'Benchmark complete. Press enter to close.'; read"
  echo "Launched in tmux session '$SESSION'"
  echo "Attach with: tmux attach -t $SESSION"
  echo "Detach with: Ctrl-b d"
  exit 0
fi
# Shift off the --_in-tmux flag if present
[[ "${1:-}" == "--_in-tmux" ]] && shift

# ---------------------------------------------------------------------------
# Job summary
# ---------------------------------------------------------------------------
N_COMBOS=$(( ${#ALGOS[@]} * ${#GAMES[@]} * ${#SEEDS[@]} ))
echo "========================================"
echo "Atari Benchmark — GNU Parallel Launch"
echo "========================================"
echo "  Algorithms : ${ALGOS[*]}"
echo "  Games      : ${GAMES[*]}"
echo "  Seeds      : ${SEEDS[*]}"
echo "  Total jobs : ${N_COMBOS}"
echo "  GPUs       : ${N_GPUS}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  n_rounds   : ${N_ROUNDS}"
echo "  total_ts   : ${TOTAL_TIMESTEPS}"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"

BENCH_START=$(date +%s)

# ---------------------------------------------------------------------------
# GNU parallel invocation (INFRA-02)
# ---------------------------------------------------------------------------
# --jobs ${N_GPUS}          : at most 4 concurrent jobs (one per GPU)
# --eta                     : show estimated time of arrival
# --halt soon,fail=1        : stop scheduling new jobs on first failure,
#                             but let already-running jobs finish
# --joblog                  : per-job log for debugging (timing, exit codes)
# CUDA_VISIBLE_DEVICES      : {%} is 1-indexed slot → subtract 1 for 0-indexed GPU
# ::: ALGOS ::: GAMES ::: SEEDS : cartesian product of all three arrays
# ---------------------------------------------------------------------------
# Resolve the active Python so GNU parallel sub-shells use the right interpreter
# (conda activate does not propagate into parallel's shells).
# Try the current interpreter first, fall back to which python.
PYTHON_BIN="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"
echo "  Python bin : ${PYTHON_BIN}"

parallel --jobs "${N_GPUS}" --eta \
  --joblog "${OUTPUT_DIR}/joblog.txt" \
  "CUDA_VISIBLE_DEVICES=\$(( {%} - 1 )) ${PYTHON_BIN} experiments/run_atari_experiment.py \
    --algo {1} --game {2} --seed {3} --output-dir ${OUTPUT_DIR} \
    with algo={1} game={2} seed={3} \
    n_rounds=${N_ROUNDS} total_timesteps=${TOTAL_TIMESTEPS} \
    n_envs=${N_ENVS} alpha=${ALPHA}" \
  ::: "${ALGOS[@]}" \
  ::: "${GAMES[@]}" \
  ::: "${SEEDS[@]}"

# ---------------------------------------------------------------------------
# Completion summary
# ---------------------------------------------------------------------------
BENCH_END=$(date +%s)
ELAPSED=$(( BENCH_END - BENCH_START ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo "========================================"
echo "Benchmark complete!"
echo "  Elapsed    : ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Job log    : ${OUTPUT_DIR}/joblog.txt"
echo "========================================"
