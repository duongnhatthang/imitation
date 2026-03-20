#!/usr/bin/env bash
# run_atari_benchmark.sh — Multi-GPU GNU parallel orchestrator for Atari benchmark.
#
# Runs in two phases to manage memory:
#   Phase A: BC jobs (lightweight, 4 parallel)
#   Phase B: DAgger + FTRL jobs (memory-heavy, 2 parallel)
#
# Usage (from project root):
#   bash experiments/run_atari_benchmark.sh
set -euo pipefail

source experiments/common.sh

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GAMES=(Pong Breakout BeamRider Enduro Qbert Seaquest SpaceInvaders)
SEEDS=(0 1 2)
N_ROUNDS=20
TOTAL_TIMESTEPS=200000
N_ENVS=4
ALPHA=1.0
N_GPUS=4
OUTPUT_DIR="output/sacred/${TIMESTAMP}"

# ---------------------------------------------------------------------------
# tmux auto-relaunch
# ---------------------------------------------------------------------------
if [[ "${1:-}" != "--_in-tmux" ]] && [ -z "${TMUX:-}" ]; then
  SESSION="atari_bench_$(date +%Y%m%d_%H%M%S)"
  tmux new-session -d -s "$SESSION" \
    "bash $0 --_in-tmux; echo 'Benchmark complete. Press enter to close.'; read"
  echo "Launched in tmux session '$SESSION'"
  echo "Attach with: tmux attach -t $SESSION"
  echo "Detach with: Ctrl-b d"
  exit 0
fi
[[ "${1:-}" == "--_in-tmux" ]] && shift

# ---------------------------------------------------------------------------
# Resolve Python
# ---------------------------------------------------------------------------
PYTHON_BIN="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"

N_BC=$(( ${#GAMES[@]} * ${#SEEDS[@]} ))
N_DAGGER_FTRL=$(( 2 * ${#GAMES[@]} * ${#SEEDS[@]} ))

echo "========================================"
echo "Atari Benchmark — GNU Parallel Launch"
echo "========================================"
echo "  Games      : ${GAMES[*]}"
echo "  Seeds      : ${SEEDS[*]}"
echo "  Phase A    : BC (${N_BC} jobs, ${N_GPUS} parallel)"
echo "  Phase B    : DAgger+FTRL (${N_DAGGER_FTRL} jobs, 2 parallel)"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  n_rounds   : ${N_ROUNDS}"
echo "  total_ts   : ${TOTAL_TIMESTEPS}"
echo "  Python bin : ${PYTHON_BIN}"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"
BENCH_START=$(date +%s)

# ---------------------------------------------------------------------------
# Phase A: BC (lightweight — 4 parallel, one per GPU)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase A: BC (${N_BC} jobs, ${N_GPUS} parallel) ==="
echo ""

parallel --jobs "${N_GPUS}" --eta \
  --joblog "${OUTPUT_DIR}/joblog_bc.txt" \
  "CUDA_VISIBLE_DEVICES=\$(( {%} - 1 )) ${PYTHON_BIN} experiments/run_atari_experiment.py \
    --algo bc --game {1} --seed {2} --output-dir ${OUTPUT_DIR} \
    with algo=bc game={1} seed={2} \
    n_rounds=${N_ROUNDS} total_timesteps=${TOTAL_TIMESTEPS} \
    n_envs=${N_ENVS} alpha=${ALPHA}" \
  ::: "${GAMES[@]}" \
  ::: "${SEEDS[@]}" \
  || echo "Phase A: some BC jobs failed (see joblog_bc.txt)"

BC_DONE=$(date +%s)
BC_ELAPSED=$(( BC_DONE - BENCH_START ))
echo "Phase A complete in $(( BC_ELAPSED / 60 ))m $(( BC_ELAPSED % 60 ))s"

# ---------------------------------------------------------------------------
# Phase B: DAgger + FTRL (memory-heavy — 2 parallel to avoid OOM)
# DAgger accumulates all trajectories across rounds, using ~15-25GB RAM each.
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase B: DAgger + FTRL (${N_DAGGER_FTRL} jobs, 2 parallel) ==="
echo ""

# Run sequentially (1 job) — DAgger/FTRL accumulate all trajectories across
# rounds in memory. A single 20-round Pong DAgger job peaks at ~30-45GB RAM.
# Running 2+ concurrently on 93GB server causes OOM kills (signal 137).
parallel --jobs 1 --eta \
  --joblog "${OUTPUT_DIR}/joblog_dagger_ftrl.txt" \
  "CUDA_VISIBLE_DEVICES=\$(( {%} - 1 )) ${PYTHON_BIN} experiments/run_atari_experiment.py \
    --algo {1} --game {2} --seed {3} --output-dir ${OUTPUT_DIR} \
    with algo={1} game={2} seed={3} \
    n_rounds=${N_ROUNDS} total_timesteps=${TOTAL_TIMESTEPS} \
    n_envs=${N_ENVS} alpha=${ALPHA}" \
  ::: dagger ftrl \
  ::: "${GAMES[@]}" \
  ::: "${SEEDS[@]}" \
  || echo "Phase B: some DAgger/FTRL jobs failed (see joblog_dagger_ftrl.txt)"

# ---------------------------------------------------------------------------
# Completion summary
# ---------------------------------------------------------------------------
BENCH_END=$(date +%s)
ELAPSED=$(( BENCH_END - BENCH_START ))

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "  Elapsed    : $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "  BC phase   : $(( BC_ELAPSED / 60 ))m"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Job logs   : ${OUTPUT_DIR}/joblog_bc.txt"
echo "               ${OUTPUT_DIR}/joblog_dagger_ftrl.txt"
echo "========================================"

# Show failure summary
BC_FAIL=$(awk -F'\t' 'NR>1 && $7!=0' "${OUTPUT_DIR}/joblog_bc.txt" 2>/dev/null | wc -l)
DF_FAIL=$(awk -F'\t' 'NR>1 && $7!=0' "${OUTPUT_DIR}/joblog_dagger_ftrl.txt" 2>/dev/null | wc -l)
echo "  BC failures       : ${BC_FAIL}"
echo "  DAgger/FTRL fails : ${DF_FAIL}"
echo "========================================"
