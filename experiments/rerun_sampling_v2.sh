#!/usr/bin/env bash
# Run corrected learning curves, then coverage from those same training datasets.
# Usage: bash experiments/rerun_sampling_v2.sh classical|atari|all
# Run long experiments inside an existing tmux session.
set -euo pipefail
cd "$(dirname "$0")/.."
export EXP_LC_CLASSICAL="${EXP_LC_CLASSICAL:-experiments/learning_curves/sampling_v2/classical}"
export EXP_LC_ATARI="${EXP_LC_ATARI:-experiments/learning_curves/sampling_v2/atari}"
export COVERAGE_SEEDS="0 1 2 3 4"
export TSNE_SEEDS="${TSNE_SEEDS:-0 1 2}"

run_group() {
  local group="$1" results envs
  if [ "$group" = classical ]; then
    bash experiments/run_learning_curves.sh --trajectories-per-round 1
    results="$EXP_LC_CLASSICAL"
    envs=$(python -c 'from imitation.experiments.ftrl.env_utils import ENV_GROUPS; print(" ".join(ENV_GROUPS["classical"]))')
  else
    bash experiments/run_atari_curves.sh --trajectories-per-round 1
    results="$EXP_LC_ATARI"
    envs="${ENVS:-$(python -c 'from imitation.experiments.ftrl.env_utils import ENV_GROUPS; print(" ".join(ENV_GROUPS["atari-zoo"]))')}"
  fi
  # Environment identifiers deliberately expand to separate arguments.
  bash experiments/gen_coverage_viz.sh "$results" $envs
}

case "${1:-classical}" in
  classical) run_group classical ;;
  atari) run_group atari ;;
  all) run_group classical; run_group atari ;;
  *) echo "Usage: $0 classical|atari|all" >&2; exit 2 ;;
esac
