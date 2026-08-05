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
#
# Resume / extend (Way 1 — re-run fresh): the runner is n_rounds-aware. Re-running
# this script:
#   * same N_ROUNDS, same dir  -> resumes (skips completed (algo,env,seed) JSONs;
#     safe after an interruption/reboot).
#   * higher N_ROUNDS          -> re-runs those configs to the new depth (the
#     shorter run's JSON is NOT reused). Rounds 1..k are statistically identical
#     whether you target k or 2k, so this yields a valid deeper curve.
# To extend only some games to more rounds (e.g. 100 -> 200), set both:
#   N_ROUNDS=200 ENVS="SeaquestNoFrameskip-v4 MsPacmanNoFrameskip-v4" \
#     EXP_LC_ATARI=experiments/learning_curves/atari_r200 ./experiments/run_atari_curves.sh
# (Use a fresh EXP_LC_ATARI dir to keep the shallower run for comparison, or the
# same dir to overwrite those envs in place.)
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

# --- Disk safety (shared server) -------------------------------------------
# HuggingFace `datasets` stages load_from_disk / save_to_disk copies through
# $TMPDIR (default /tmp, on the shared ROOT filesystem). DAgger (ftl/ftrl)
# reloads the *accumulated* demo set every round, so over a deep run the many
# concurrent workers pile temp copies onto / until it fills node-wide -- this
# is exactly what sank the first atari_r200 sweep (ENOSPC on /). Three
# independent guards, cheapest first:
#
# 1. Redirect all temp + HF cache to a RUN-LOCAL dir on the same (big) volume
#    as the results, and remove it on exit -- keeps churn off shared / and
#    leaves nothing behind.
# 2. Keep small demo datasets in RAM (IN_MEMORY_MAX_SIZE) so per-round reloads
#    don't leave memory-mapped temp copies at all.
# 3. A background watchdog that aborts the sweep if free space on the results
#    volume drops below a floor -- a fail-safe so we can never fill a disk
#    other people share, regardless of what the code above misses.
export TMPDIR="$RESULTS_DIR/_tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export HF_DATASETS_CACHE="$RESULTS_DIR/_hfcache"
# Keep demo datasets <= 2 GiB in RAM instead of mmap-ing a temp copy.
export HF_DATASETS_IN_MEMORY_MAX_SIZE="${HF_DATASETS_IN_MEMORY_MAX_SIZE:-2147483648}"
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"
cleanup_tmp() { rm -rf "$TMPDIR" "$HF_DATASETS_CACHE"; }
trap cleanup_tmp EXIT

DISK_FLOOR_GB="${DISK_FLOOR_GB:-30}"
disk_watchdog() {
    local target_pid="$1" avail_gb
    while kill -0 "$target_pid" 2>/dev/null; do
        avail_gb=$(df -B1G --output=avail "$RESULTS_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')
        if [ -n "$avail_gb" ] && [ "$avail_gb" -lt "$DISK_FLOOR_GB" ]; then
            echo "[atari_curves] DISK GUARD: only ${avail_gb}G free on the results volume" \
                 "(< ${DISK_FLOOR_GB}G floor) -- aborting sweep to protect the shared disk." \
                 | tee -a "$LOG_FILE"
            pkill -TERM -f run_experiment 2>/dev/null || true
            sleep 5
            pkill -KILL -f run_experiment 2>/dev/null || true
            return 1
        fi
        sleep 30
    done
}

# CPU worker count: total - 2, floor 1.
CPU_TOTAL="$(getconf _NPROCESSORS_ONLN)"
WORKERS=$(( CPU_TOTAL - 2 ))
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
echo "[atari_curves] $WORKERS workers, $N_GPUS GPUs, n_rounds=$N_ROUNDS" | tee -a "$LOG_FILE"
echo "[atari_curves] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "[atari_curves] output dir: $RESULTS_DIR" | tee -a "$LOG_FILE"

# Algo set is overridable (ALGOS="ftl ftrl") so a targeted resume can re-run
# only the interactive methods without re-touching completed bc/bc_dagger cells.
# shellcheck disable=SC2206  # word-splitting is intentional for the list
ALGO_SEL=(${ALGOS:-ftl ftrl bc bc_dagger})

# Run under the disk watchdog: launch the sweep in the background, capture its
# PID (process substitution keeps python as the job, so $! is python's PID),
# and let the watchdog abort it if the shared disk gets dangerously low.
python -m imitation.experiments.ftrl.run_experiment \
    "${ENV_SEL[@]}" \
    --policy-mode linear \
    --algos "${ALGO_SEL[@]}" \
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
    > >(tee -a "$LOG_FILE") 2>&1 &
RUN_PID=$!
disk_watchdog "$RUN_PID" &
WATCH_PID=$!
wait "$RUN_PID" && RUN_RC=0 || RUN_RC=$?
kill "$WATCH_PID" 2>/dev/null || true
wait "$WATCH_PID" 2>/dev/null || true
if [ "$RUN_RC" -ne 0 ]; then
    echo "[atari_curves] sweep exited non-zero (rc=$RUN_RC) -- see disk guard / errors above." \
        | tee -a "$LOG_FILE"
    exit "$RUN_RC"
fi

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
