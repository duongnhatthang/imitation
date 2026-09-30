#!/usr/bin/env bash
# Validate and run one named experiment campaign, classical by default; Atari is opt-in.
# Activate the desired Python environment and run this inside tmux.
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/paths.sh
export CLASSICAL_ROUNDS="${CLASSICAL_ROUNDS:-1000}"
export ATARI_ROUNDS="${ATARI_ROUNDS:-200}"
export SEEDS="${SEEDS:-5}"
export SAMPLES_PER_ROUND="${SAMPLES_PER_ROUND:-1}"
export TRAJECTORIES_PER_ROUND="${TRAJECTORIES_PER_ROUND:-1}"
export ALGOS="${ALGOS:-$EXP_ALGOS}"
export RUN_ATARI="${RUN_ATARI:-0}"
case "$RUN_ATARI" in 0|1) ;; *) echo "RUN_ATARI must be 0 or 1" >&2; exit 2 ;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$EXP_LOG_DIR"
STAGE=starting
status() {
    STAGE="$1"
    printf '%s\n' "$STAGE" > "$EXP_RUN_ROOT/status.tmp"
    mv "$EXP_RUN_ROOT/status.tmp" "$EXP_RUN_ROOT/status.txt"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $STAGE"
}
trap 'rc=$?; if [ "$rc" -ne 0 ]; then status "failed: $STAGE (exit $rc)"; fi' EXIT
python - <<'PY'
import datetime
import json
import os
from pathlib import Path
root = Path(os.environ['EXP_RUN_ROOT'])
manifest = {
    'run_id': os.environ['EXP_RUN_ID'],
    'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_revision': os.environ.get('SOURCE_REVISION', 'unknown'),
    'algorithms': os.environ['ALGOS'].split(),
    'classical_rounds': int(os.environ['CLASSICAL_ROUNDS']),
    'atari_rounds': int(os.environ['ATARI_ROUNDS']),
    'seeds': int(os.environ['SEEDS']),
    'samples_per_round': int(os.environ['SAMPLES_PER_ROUND']),
    'minimum_trajectories_per_round': int(os.environ['TRAJECTORIES_PER_ROUND']),
    'policy_mode': 'linear',
    'warm_start': False,
    'dagger_beta': 0,
    'bc_iid_beta': 1,
    'run_atari': os.environ['RUN_ATARI'] == '1',
    'classical_envs': os.environ.get('ENVS', '').split(),
    'atari_envs': os.environ.get('ATARI_ENVS', '').split(),
    'bc_sampling': 'prefix',
    'bc_iid_sampling': 'uniform',
    'bc_prefix_sampling': 'prefix',
    'bc_pool_sampling': 'uniform',
    'outer_early_stop': False,
}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
PY
status validation
python -m pytest -q tests/experiments/test_bc_iid.py \
    tests/experiments/test_expert_dataset.py \
    tests/experiments/test_run_experiment.py \
    tests/experiments/test_bc_baselines.py \
    -k 'bc_iid_uses_only or fixed_bc_defaults or bc_prefix_preserves or baseline_data_and_evaluation_budget or shared_data_is_collected or per_trajectory or bc_prefix or bc_pool' \
    > "$EXP_LOG_DIR/validation.log" 2>&1
status smoke
python -m imitation.experiments.ftrl.run_experiment \
    --envs CartPole-v1 --algos $ALGOS --seeds 1 \
    --samples-per-round 1 --traj-per-round 1 --n-rounds 3 \
    --bc-n-epochs 1 --eval-interval 1 --policy-mode linear \
    --no-warm-start --beta-rampdown 0 \
    --no-inner-early-stop --no-outer-early-stop --n-workers 2 --n-gpus 0 \
    --expert-cache-dir "$EXP_EXPERT_CACHE" --output-dir "$EXP_SMOKE_DIR" \
    > "$EXP_LOG_DIR/smoke.log" 2>&1
status classical_training
# No hardcoded 8: let run_learning_curves.sh size itself to the node (cores - 2,
# capped) unless CLASSICAL_WORKERS or an exported N_WORKERS says otherwise.
env ${CLASSICAL_WORKERS:+N_WORKERS="$CLASSICAL_WORKERS"} bash experiments/run_learning_curves.sh \
    --seeds "$SEEDS" --n-rounds "$CLASSICAL_ROUNDS" \
    --samples-per-round "$SAMPLES_PER_ROUND" --traj-per-round "$TRAJECTORIES_PER_ROUND"
status classical_plots
CLASSICAL_PLOT_ENVS="${ENVS:-$(python -c 'from imitation.experiments.ftrl.env_utils import ENV_GROUPS; print(" ".join(ENV_GROUPS["classical"]))')}"
PLOT_ROOT="$EXP_PLOTS_CLASSICAL" bash experiments/gen_coverage_viz.sh "$EXP_LC_CLASSICAL" $CLASSICAL_PLOT_ENVS \
    > "$EXP_LOG_DIR/classical-coverage.log" 2>&1
if [ "$RUN_ATARI" = 1 ]; then
status atari_training
ENVS="${ATARI_ENVS:-}" N_WORKERS="${ATARI_WORKERS:-${N_WORKERS:-4}}" N_ROUNDS="$ATARI_ROUNDS" bash experiments/run_atari_curves.sh \
    --seeds "$SEEDS" --samples-per-round "$SAMPLES_PER_ROUND" --traj-per-round "$TRAJECTORIES_PER_ROUND"
status atari_plots
ATARI_PLOT_ENVS="${ATARI_ENVS:-$(python -c 'from imitation.experiments.ftrl.env_utils import ENV_GROUPS; print(" ".join(ENV_GROUPS["atari-zoo"]))')}"
PLOT_ROOT="$EXP_PLOTS_ATARI" bash experiments/gen_coverage_viz.sh "$EXP_LC_ATARI" $ATARI_PLOT_ENVS \
    > "$EXP_LOG_DIR/atari-coverage.log" 2>&1
fi
status complete
