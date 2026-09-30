#!/usr/bin/env bash
# Compatibility entry point. New runs use the named-run layout in paths.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
source experiments/paths.sh
case "${1:-all}" in
  all) exec bash experiments/run_bc_iid_experiments.sh ;;
  classical)
    bash experiments/run_learning_curves.sh --traj-per-round "${TRAJECTORIES_PER_ROUND:-1}"
    envs=$(python -c 'from imitation.experiments.ftrl.env_utils import ENV_GROUPS; print(" ".join(ENV_GROUPS["classical"]))')
    PLOT_ROOT="$EXP_PLOTS_CLASSICAL" bash experiments/gen_coverage_viz.sh "$EXP_LC_CLASSICAL" $envs
    ;;
  atari)
    N_ROUNDS="${N_ROUNDS:-200}" bash experiments/run_atari_curves.sh --traj-per-round "${TRAJECTORIES_PER_ROUND:-1}"
    envs="${ENVS:-$(python -c 'from imitation.experiments.ftrl.env_utils import ENV_GROUPS; print(" ".join(ENV_GROUPS["atari-zoo"]))')}"
    PLOT_ROOT="$EXP_PLOTS_ATARI" bash experiments/gen_coverage_viz.sh "$EXP_LC_ATARI" $envs
    ;;
  *) echo 'Usage: rerun_sampling_v2.sh classical|atari|all' >&2; exit 2 ;;
esac
