#!/usr/bin/env bash
# Launch and monitor a BC-iid campaign on a configured remote host.
#
#   ./run.sh status          one status check, prints a summary, saves a snapshot
#   ./run.sh watch [MIN]     status every MIN minutes (default 10) until Ctrl-C
#   ./run.sh start           start or resume the campaign (safe to re-run)
#   ./run.sh stop            stop the campaign (progress is kept)
#   ./run.sh push            push tracked code to the server (experiments/sync_results.sh)
#   ./run.sh pull            pull plots back (experiments/sync_results.sh)
#   ./run.sh runs            list every run directory on the server, with counts
#   ./run.sh cells           rounds completed per active cell, slowest last
#
# Every check writes runstate/latest.txt and appends runstate/history.tsv, so
# progress can be read without an SSH session.
#
# Overrides:  RUN_ID=... HOST=... ATARI_ROUNDS=... GPUS=0 ./run.sh status
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Private connection settings belong in this ignored file or the environment.
[ ! -f "$HERE/experiments/.local/runner.env" ] || source "$HERE/experiments/.local/runner.env"
HOST="${HOST:-}"
RUN_ATARI="${RUN_ATARI:-0}"
RUN_ID="${RUN_ID:-classical-cold-start-v1}"
REMOTE_REPO="${REMOTE_REPO:-imitation}"          # relative to remote $HOME
ATARI_ROUNDS="${ATARI_ROUNDS:-200}"
CLASSICAL_ROUNDS="${CLASSICAL_ROUNDS:-1000}"
CONDA_ENV="${CONDA_ENV:-imitation}"
# Empty selects every available device; set GPUS to restrict visible indices.
GPUS="${GPUS:-}"

# What the runner actually produces: one JSON per (env, algo, seed), written as
# <data dir>/<env>/<algo>_<policy-mode>_seed<N>.json.
#   classical  = 8 envs x 4 algos x 5 seeds = 160
#   atari-zoo  = 7 envs x 4 algos x 5 seeds = 140
CLASSICAL_TOTAL=160
ATARI_TOTAL=140

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$HERE/runstate"
HISTORY="$STATE_DIR/history.tsv"
LATEST="$STATE_DIR/latest.txt"
mkdir -p "$STATE_DIR"
# run_id is appended LAST so rows written before it existed still parse; they
# simply have no $15 and are therefore excluded from every per-run rate below.
[ -s "$HISTORY" ] || printf 'epoch\tiso\trun_dir\tstage\tclassical\tatari\tatari_round_dirs\tatari_cells_active\tworkers\tlast_result_age_min\tdisk_avail_gb\tclassical_round_dirs\tclassical_cells_active\tleaders\trun_id\n' > "$HISTORY"

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR "$HOST")
PROBE_KEYS="manifest_run_atari manifest_classical_envs_n manifest_atari_envs_n manifest_algos manifest_algo_names manifest_seeds manifest_loop_algos run_dir stage classical atari atari_round_dirs atari_cells_active classical_round_dirs classical_cells_active workers leaders python_procs driver last_result_age_min runs_on_disk expert_cache_files disk_avail_gb disk_used_pct cpu_total loadavg gpu tmux log_tail_b64"

die() { echo "error: $*" >&2; exit 1; }

# SSH joins command arguments before the remote shell parses them. Quote each
# value so empty settings and lists retain their positional argument boundaries.
ssh_bash() {
  local remote_command
  printf -v remote_command '%q ' bash -s -- "$@"
  "${SSH[@]}" "$remote_command"
}

# ---------------------------------------------------------------- probe -----
# Runs on the server, prints key=value lines, changes nothing.
probe() {
  ssh_bash "$RUN_ID" "$REMOTE_REPO" "$CONDA_ENV" <<'PROBE'
set -u
RID="$1"; REPO="$2"; ENVNAME="$3"
EXP="$HOME/$REPO/experiments"
ROOT="$EXP/runs/$RID"
cells() { find "$1" -mindepth 2 -maxdepth 2 -name '*.json' 2>/dev/null | wc -l | tr -d ' '; }

echo "run_id=$RID"
# The campaign shape comes from the run's own manifest, so a six-algo run, a
# narrowed ENVS sweep or a different seed count all report correct totals
# instead of the four-algo numbers this script used to hardcode.
(
for h in "$HOME/miniconda3/etc/profile.d/conda.sh" \
         "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "/opt/conda/etc/profile.d/conda.sh"; do
  [ -f "$h" ] && { . "$h" >/dev/null; break; }
done
command -v conda >/dev/null || exit 1
conda activate "$ENVNAME" >/dev/null || exit 1
python - "$ROOT/manifest.json" <<'MANIFEST'
import json, sys
try:
    m = json.load(open(sys.argv[1]))
except Exception:
    print("manifest_algos=")
    sys.exit(0)
print("manifest_run_atari=%d" % bool(m.get("run_atari", True)))
print("manifest_classical_envs_n=%d" % len(m.get("classical_envs") or range(8)))
print("manifest_atari_envs_n=%d" % len(m.get("atari_envs") or range(7)))
algos = m.get("algorithms") or []
print("manifest_algos=%d" % len(algos))
print("manifest_algo_names=%s" % ",".join(algos))
print("manifest_seeds=%s" % (m.get("seeds") or ""))
# Only the algorithms that run the round loop write a round-* series; plain BC
# writes a single round-000.
loop = [a for a in algos if a in ("ftl", "ftrl", "bc_iid", "bc_prefix", "bc_pool")]
print("manifest_loop_algos=%d" % len(loop))
MANIFEST
) 2>/dev/null || echo "manifest_algos="
[ -d "$ROOT" ] && echo "run_dir=present" || echo "run_dir=MISSING"
echo "stage=$(tr -d '\n' < "$ROOT/status.txt" 2>/dev/null)"
echo "classical=$(cells "$ROOT/data/classical")"
echo "atari=$(cells "$ROOT/data/atari")"
echo "atari_round_dirs=$(find "$ROOT/data/atari/scratch" -maxdepth 3 -type d -name 'round-*' 2>/dev/null | wc -l | tr -d ' ')"
echo "atari_cells_active=$(find "$ROOT/data/atari/scratch" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
echo "classical_round_dirs=$(find "$ROOT/data/classical/scratch" -maxdepth 3 -type d -name 'round-*' 2>/dev/null | wc -l | tr -d ' ')"
echo "classical_cells_active=$(find "$ROOT/data/classical/scratch" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
# run_experiment parallelises with multiprocessing's "spawn" context, so each
# pool child runs as `python -c "from multiprocessing.spawn import ..."` and its
# command line does NOT mention run_experiment. Counting by pattern therefore
# finds only the parent and always reports 1. Count the parent's children.
LEADER=$(pgrep -f 'imitation\.experiments\.ftrl\.run_experiment' 2>/dev/null | head -1)
if [ -n "${LEADER:-}" ]; then
  # Not every child is a worker: multiprocessing also starts a resource_tracker,
  # which is why this used to read one higher than --n-workers and invited the
  # question "why 9 when I asked for 8".
  W=0
  for c in $(pgrep -P "$LEADER" 2>/dev/null); do
    case "$(tr '\0' ' ' < "/proc/$c/cmdline" 2>/dev/null)" in
      *resource_tracker*) ;;
      *) W=$(( W + 1 )) ;;
    esac
  done
  echo "workers=$W"
else
  echo "workers=0"
fi
echo "leaders=$(pgrep -fc 'imitation\.experiments\.ftrl\.run_experiment' 2>/dev/null || echo 0)"
echo "python_procs=$(pgrep -c python 2>/dev/null || echo 0)"
echo "driver=$(pgrep -fc 'run_bc_iid_experiments' 2>/dev/null || echo 0)"

# Only the two production data trees. $ROOT/data also holds data/smoke, whose
# JSONs would otherwise masquerade as a fresh result for the rest of the run.
NEW=$(find "$ROOT/data/classical" "$ROOT/data/atari" -mindepth 2 -maxdepth 2 -name '*.json' -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
if [ -n "${NEW:-}" ]; then
  echo "last_result_age_min=$(( ( $(date +%s) - ${NEW%.*} ) / 60 ))"
else
  echo "last_result_age_min="
fi

echo "runs_on_disk=$(ls -1 "$EXP/runs" 2>/dev/null | paste -sd, -)"
echo "expert_cache_files=$(find "$EXP/expert_cache" -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "disk_avail_gb=$(df -B1G --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9')"
echo "disk_used_pct=$(df -h "$HOME" 2>/dev/null | tail -1 | awk '{print $5}')"
echo "cpu_total=$(nproc 2>/dev/null || echo)"
echo "loadavg=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null | tr ' ' '/')"
echo "gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd';' -)"
echo "tmux=$(tmux list-sessions 2>/dev/null | cut -d: -f1 | paste -sd, -)"
echo "log_tail_b64=$(tail -c 3000 "$ROOT/logs/atari.log" 2>/dev/null | base64 | tr -d '\n')"
PROBE
}

# --------------------------------------------------------------- status -----
cmd_status() {
  local k v out rc epoch iso
  local ALGOS_N ALGOS_LOOP_N
  epoch=$(date -u +%s); iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  for k in $PROBE_KEYS; do eval "$k="; done
  run_dir='?' ; classical=0; atari=0; atari_round_dirs=0
  atari_cells_active=0; classical_round_dirs=0; classical_cells_active=0; workers=0; leaders=0; python_procs=0; driver=0

  out=$(probe 2>&1); rc=$?
  if [ $rc -ne 0 ]; then
    run_dir="ssh-failed"
    stage="ssh error: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-160)"
  else
    while IFS='=' read -r k v; do
      case " $PROBE_KEYS " in *" $k "*) eval "$k=\$v" ;; esac
    done <<< "$out"
  fi

  # Totals follow the manifest when it is readable, and fall back to the
  # documented defaults otherwise. CLASSICAL_ENVS_N / ATARI_ENVS_N override the
  # env-group sizes for a sweep narrowed with ENVS.
  # Precedence: the run's manifest, then an explicit ALGOS/SEEDS override for a
  # run whose manifest is unreadable, then the four-algo historical defaults.
  local n_algos n_seeds n_loop
  # shellcheck disable=SC2086
  if [ -n "${ALGOS:-}" ]; then
    ALGOS_N="$(printf '%s\n' $ALGOS | wc -w | tr -d ' ')"
    ALGOS_LOOP_N="$(printf '%s\n' $ALGOS | grep -cE '^(ftl|ftrl|bc_iid|bc_prefix|bc_pool)$' || true)"
  fi
  n_algos="${manifest_algos:-${ALGOS_N:-4}}"
  n_seeds="${manifest_seeds:-${SEEDS:-5}}"
  n_loop="${manifest_loop_algos:-${ALGOS_LOOP_N:-3}}"
  [ -n "$n_algos" ] || n_algos=4
  [ -n "$n_seeds" ] || n_seeds=5
  [ -n "$n_loop" ] || n_loop=3
  local classical_envs_n atari_envs_n
  classical_envs_n="${manifest_classical_envs_n:-${CLASSICAL_ENVS_N:-8}}"
  atari_envs_n="${manifest_atari_envs_n:-${ATARI_ENVS_N:-7}}"
  if [ "${manifest_run_atari:-$RUN_ATARI}" = 0 ]; then atari_envs_n=0; fi
  CLASSICAL_TOTAL=$(( n_algos * classical_envs_n * n_seeds ))
  ATARI_TOTAL=$(( n_algos * atari_envs_n * n_seeds ))
  CLASSICAL_UNITS=$(( n_loop * classical_envs_n * n_seeds * CLASSICAL_ROUNDS \
                      + ( n_algos - n_loop ) * classical_envs_n * n_seeds ))
  ATARI_UNITS=$(( n_loop * atari_envs_n * n_seeds * ATARI_ROUNDS \
                  + ( n_algos - n_loop ) * atari_envs_n * n_seeds ))

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$epoch" "$iso" "$run_dir" "${stage:-unknown}" "$classical" "$atari" \
    "$atari_round_dirs" "$atari_cells_active" "$workers" \
    "${last_result_age_min:-}" "${disk_avail_gb:-}" \
    "$classical_round_dirs" "$classical_cells_active" "$leaders" \
    "$RUN_ID" >> "$HISTORY"

  {
    echo "checked_at        $iso"
    echo "host / run        $HOST : $RUN_ID"
    echo "campaign          ${manifest_algos:-?} algos x ${manifest_seeds:-?} seeds  (${manifest_algo_names:-unknown})"
    echo "run directory     $run_dir"
    echo "stage             ${stage:-unknown}"
    echo "classical cells   $classical / $CLASSICAL_TOTAL"
    echo "atari cells       $atari / $ATARI_TOTAL"
    echo "classical rounds  $classical_round_dirs  (across $classical_cells_active active cells)"
    echo "atari rounds      $atari_round_dirs  (across $atari_cells_active active cells)"
    echo "pool workers      $workers   sweep: $leaders   driver: $driver   python procs: $python_procs"
    echo "newest result     ${last_result_age_min:-n/a} min old"
    echo "eta (cells)       $(eta_cells)"
    echo "eta (rounds)      $(eta_rounds)"
    echo "disk free         ${disk_avail_gb:-?} GB (${disk_used_pct:-?} used)"
    # During the smoke stage this is the smoke's own 2 workers, not the
    # campaign's -- the classical count only appears at stage classical_training.
    echo "cpu               $workers workers / ${cpu_total:-?} cores   load ${loadavg:-?}"
    # Classical is CPU-only by design -- linear policies on toy MDPs do not repay
    # a host-to-device round trip -- so 0% gpu during classical_training is
    # correct, not a fault. The GPUs matter only for the atari stage.
    echo "gpu               ${gpu:-n/a}"
    echo "tmux sessions     ${tmux:-none}"
    echo "runs on server    ${runs_on_disk:-none}"
    echo "expert cache      ${expert_cache_files:-?} files"
    echo "verdict           $(verdict)"
    echo "--- last lines of atari.log ---"
    if [ -n "${log_tail_b64:-}" ]; then printf '%s' "$log_tail_b64" | base64 -d 2>/dev/null; else echo "(no log)"; fi
  } > "$LATEST"

  cat "$LATEST"
}

# One line saying whether a human is needed. Uses the vars set by cmd_status.
verdict() {
  if [ "$run_dir" = "ssh-failed" ]; then
    echo "cannot reach $HOST"
  elif [ "$run_dir" = "MISSING" ]; then
    echo "run directory absent on server; nothing is running"
  elif [ "${stage:-}" = complete ]; then
    echo "campaign complete -- run ./run.sh pull"
  elif [ "$ATARI_TOTAL" -gt 0 ] && [ "$atari" -ge "$ATARI_TOTAL" ]; then
    echo "atari sweep complete ($atari/$ATARI_TOTAL) -- run ./run.sh pull"
  elif [ "$leaders" -gt 0 ]; then
    if [ "$workers" -gt 1 ]; then
      echo "running normally, $workers pool workers"
    else
      echo "sweep alive but only $workers pool worker(s) -- check parallelism"
    fi
  elif [ "$driver" -gt 0 ]; then
    echo "driver alive at stage '${stage}' with no sweep running -- plotting, or between stages"
  elif [ -n "${last_result_age_min:-}" ] && [ "${last_result_age_min:-999}" -lt 90 ] 2>/dev/null; then
    echo "no workers seen but a result landed ${last_result_age_min}m ago; re-check shortly"
  else
    case "${stage:-}" in
      complete) echo "campaign finished" ;;
      failed*)  echo "stage failed -- ${stage}; read the log tail" ;;
      *)        echo "nothing running; ./run.sh start will resume" ;;
    esac
  fi
}

# Cell-completion rate over the recorded history, for whichever stage is running.
eta_cells() {
  local col total
  case "${stage:-}" in
    classical_training) col=5; total="$CLASSICAL_TOTAL" ;;
    atari_training)     col=6; total="$ATARI_TOTAL" ;;
    *) echo "n/a (only meaningful during a training stage)"; return ;;
  esac
  COL="$col" TOTAL="$total" RID="$RUN_ID" awk -F'\t' '
    NR > 1 && $3 == "present" && $15 == ENVIRON["RID"] \
      && $(ENVIRON["COL"]) ~ /^[0-9]+$/ {
      if (ft == "") { ft = $1; fc = $(ENVIRON["COL"]) + 0 }
      lt = $1; lc = $(ENVIRON["COL"]) + 0
    }
    END {
      total = ENVIRON["TOTAL"] + 0
      left = total - lc
      if (ft == "") { print "n/a (no history for this run yet)"; exit }
      if (left <= 0) { print "stage complete"; exit }
      span = (lt - ft) / 3600.0
      if (span < 0.5) { print "n/a (needs ~30 min of history)"; exit }
      if (lc - fc <= 0) { printf "no cell finished in %.1f h of watching\n", span; exit }
      rate = (lc - fc) / span
      printf "~%.1f h (%.1f cells/h, %d of %d cells left)\n", left / rate, rate, left, total
    }' "$HISTORY"
}


# Cell JSONs appear only when a whole cell finishes, so early in a stage the
# cell count sits at 0 for hours. The scratch round-* directories advance
# continuously, which gives a usable rate much sooner. Totals per stage:
# Totals count only the cells that produce a round series: ftl, ftrl and bc_iid
# each write one round-* dir per round, while plain BC trains once on a fixed
# dataset and writes a single round-000. So it is 3/4 of the cells x n_rounds
# plus 1 for each BC cell -- classical 120x1000+40 = 120,040, atari 105x200+35
# = 21,035. Using cells x rounds (160,000 / 28,000) overstates the denominator
# by a third and makes every round-based ETA pessimistic.
eta_rounds() {
  local col total
  case "${stage:-}" in
    classical_training) col=12; total="$CLASSICAL_UNITS" ;;
    atari_training)     col=7;  total="$ATARI_UNITS" ;;
    *) echo "n/a (only meaningful during a training stage)"; return ;;
  esac
  COL="$col" TOTAL="$total" RID="$RUN_ID" awk -F'\t' '
    NR > 1 && $3 == "present" && $15 == ENVIRON["RID"] \
      && $(ENVIRON["COL"]) ~ /^[0-9]+$/ {
      v = $(ENVIRON["COL"]) + 0
      if (v > 0 && ft == "") { ft = $1; fv = v }
      if (v > 0) { lt = $1; lv = v }
    }
    END {
      total = ENVIRON["TOTAL"] + 0
      if (ft == "" || lv == 0) { print "n/a (no rounds for this run yet)"; exit }
      left = total - lv
      if (left <= 0) { print "stage looks complete"; exit }
      span = (lt - ft) / 3600.0
      if (span < 0.4) { printf "n/a (%.0f of %d rounds; needs ~25 min of history)\n", lv, total; exit }
      if (lv - fv <= 0) { printf "STALLED: no new rounds in %.1f h\n", span; exit }
      rate = (lv - fv) / span
      printf "~%.1f h for this stage (%.0f of %d rounds, %.0f rounds/h)\n", left / rate, lv, total, rate
    }' "$HISTORY"
}

# ---------------------------------------------------------------- watch -----
cmd_watch() {
  local mins="${1:-10}"
  case "$mins" in ''|*[!0-9]*) die "watch interval must be a whole number of minutes" ;; esac
  [ "$mins" -ge 1 ] || die "watch interval must be at least 1 minute"
  echo "watching $HOST:$RUN_ID every ${mins} min -- Ctrl-C to stop"
  echo "snapshot: $LATEST"
  echo "history:  $HISTORY"
  while :; do
    echo; echo "================ $(date -u +%Y-%m-%dT%H:%M:%SZ) ================"
    cmd_status
    sleep $(( mins * 60 ))
  done
}

# ---------------------------------------------------------------- start -----
cmd_start() {
  local n_gpus=0
  echo "starting/resuming $RUN_ID on $HOST (conda env: $CONDA_ENV, GPUs $GPUS)"
  [ -n "${ALGOS:-}" ] && echo "  algos: $ALGOS"
  [ -n "${SEEDS:-}" ] && echo "  seeds: $SEEDS"
  [ -n "${ENVS:-}" ] && echo "  envs:  $ENVS   (set CLASSICAL_ENVS_N for correct status totals)"
  [ -n "${CLASSICAL_WORKERS:-}" ] && echo "  classical workers: $CLASSICAL_WORKERS"
  ssh_bash "$RUN_ID" "$REMOTE_REPO" "$ATARI_ROUNDS" "$CLASSICAL_ROUNDS" "$CONDA_ENV" "$GPUS" "$n_gpus" "${ALGOS:-}" "${SEEDS:-}" "${ENVS:-}" "${CLASSICAL_WORKERS:-}" "$RUN_ATARI" "${ATARI_ENVS:-}" <<'REMOTE'
set -u
RID="$1"; REPO="$2"; AR="$3"; CR="$4"; ENVNAME="$5"; GPUSET="$6"; NGPU="$7"
ALGOS_IN="${8:-}"; SEEDS_IN="${9:-}"; ENVS_IN="${10:-}"; CW_IN="${11:-}"; RUN_ATARI_IN="${12:-0}"; ATARI_ENVS_IN="${13:-}"
cd "$HOME/$REPO/experiments" || { echo "no such directory: $HOME/$REPO/experiments"; exit 1; }

if pgrep -f 'run_bc_iid_experiments|imitation.experiments.ftrl.run_experiment' > /dev/null; then
  echo "#############################################################"
  echo "## REFUSED: a sweep is already running. NOTHING WAS STARTED. "
  echo "## Your new code is NOT in use -- these are the old processes."
  echo "## Run ./run.sh stop, confirm it reports 'stopped', then start."
  echo "#############################################################"
  pgrep -af 'run_bc_iid_experiments|imitation.experiments.ftrl.run_experiment' | head -3
  exit 3
fi

# Activate the conda env. A non-login shell (which is what ssh gives us) has no
# conda on PATH, so `python` falls back to the system one and the driver dies in
# its first python block with "failed: starting (exit 1)" -- which is exactly
# what the three orphan run directories from 2026-09-14 are.
for h in "$HOME/miniconda3/etc/profile.d/conda.sh" \
         "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "/opt/conda/etc/profile.d/conda.sh"; do
  [ -f "$h" ] && { . "$h"; break; }
done
command -v conda > /dev/null || { echo "PREFLIGHT FAIL: conda not found on the server"; exit 1; }
conda activate "$ENVNAME" || { echo "PREFLIGHT FAIL: cannot activate conda env '$ENVNAME'"; exit 1; }

# Select the requested devices, or expose all devices by default.
if [ -n "$GPUSET" ]; then export CUDA_VISIBLE_DEVICES="$GPUSET"; else unset CUDA_VISIBLE_DEVICES; fi
NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"
export N_GPUS="$NGPU"

# Fail here, loudly and in one second, rather than in a detached background job.
python - <<'CHECK' || { echo "PREFLIGHT FAIL: env '$ENVNAME' cannot run the experiment code"; exit 1; }
import sys
import torch, gymnasium, stable_baselines3          # noqa: F401
from imitation.experiments.ftrl import run_experiment  # noqa: F401
import os
print("PREFLIGHT OK: python %s, torch %s, cuda %s, visible gpus %d (CUDA_VISIBLE_DEVICES=%s)"
      % (sys.version.split()[0], torch.__version__,
         torch.cuda.is_available(), torch.cuda.device_count(),
         os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")))
CHECK

# EXP_RUN_ID is what makes this a resume. Without it, paths.sh invents a new
# timestamped run directory on every launch and no earlier work is reused.
export EXP_RUN_ID="$RID"
export ATARI_ROUNDS="$AR" CLASSICAL_ROUNDS="$CR" RUN_ATARI="$RUN_ATARI_IN"
export ATARI_ENVS="$ATARI_ENVS_IN"
# Only export what was actually asked for: an empty string would override the
# driver's own defaults with nothing. The manifest records whatever wins, which
# is what run.sh reads back for the cell totals.
[ -n "$ALGOS_IN" ] && export ALGOS="$ALGOS_IN"
[ -n "$SEEDS_IN" ] && export SEEDS="$SEEDS_IN"
[ -n "$ENVS_IN" ] && export ENVS="$ENVS_IN"
[ -n "$CW_IN" ] && export CLASSICAL_WORKERS="$CW_IN"
mkdir -p "runs/$RID/logs"
LOG="runs/$RID/logs/driver.log"
echo "=== launch $(date -u +%Y-%m-%dT%H:%M:%SZ) env=$ENVNAME run=$RID gpus=$GPUSET (n=$NGPU) ===" >> "$LOG"

# Keep long jobs in a named tmux session, including all campaign settings.
command -v tmux > /dev/null || { echo "PREFLIGHT FAIL: tmux not found"; exit 1; }
SESSION="imitation-$RID"
[[ "$RID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || { echo "invalid run ID"; exit 1; }
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "REFUSED: tmux session $SESSION already exists"; exit 3
fi
LAUNCH=(env -u CUDA_VISIBLE_DEVICES -u ALGOS -u SEEDS -u ENVS -u CLASSICAL_WORKERS)
for key in EXP_RUN_ID ATARI_ROUNDS CLASSICAL_ROUNDS RUN_ATARI ATARI_ENVS N_GPUS CUDA_VISIBLE_DEVICES ALGOS SEEDS ENVS CLASSICAL_WORKERS PATH CONDA_PREFIX SOURCE_REVISION; do
  if [ "${!key+x}" = x ]; then LAUNCH+=("$key=${!key}"); fi
done
LAUNCH+=(bash "$PWD/run_bc_iid_experiments.sh")
printf -v COMMAND '%q ' "${LAUNCH[@]}"
printf -v REDIRECT ' >> %q 2>&1' "$PWD/$LOG"
tmux new-session -d -s "$SESSION" -c "$PWD" "$COMMAND$REDIRECT"
PID="$(tmux display-message -p -t "=$SESSION" '#{pane_pid}')"
echo "$PID" > "runs/$RID/driver.pid"
sleep 2
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "started in tmux session $SESSION, pid $PID; log runs/$RID/logs/driver.log"
  echo "stage: $(cat "runs/$RID/status.txt" 2>/dev/null)"
else
  echo "driver exited during startup; last log lines:"; tail -30 "$LOG"; exit 1
fi
REMOTE
  local launch_rc=$?
  [ "$launch_rc" -eq 0 ] || return "$launch_rc"
  echo
  echo "then: bash run.sh status   (and bash run.sh watch to keep an eye on it)"
}

# ----------------------------------------------------------------- stop -----
cmd_stop() {
  read -r -p "stop the run on $HOST? completed cells are kept. (yes/no) " a
  [ "$a" = yes ] || { echo "cancelled"; exit 0; }
  ssh_bash <<'STOP'
set -u
# The pool workers are spawned as `python -c "from multiprocessing.spawn import
# ..."`, so their command lines do NOT mention run_experiment -- the same fact
# the probe documents when counting them. `pkill -f ...run_experiment` therefore
# reaches the leader only, and a leader that ignores the signal, or children that
# outlive it, leave the sweep running while `start` refuses to launch a
# replacement. Kill the children explicitly, verify, then escalate.
sweep_pids() { pgrep -f 'imitation\.experiments\.ftrl\.run_experiment' 2>/dev/null; }
all_pids() {
  local l
  for l in $(sweep_pids); do echo "$l"; pgrep -P "$l" 2>/dev/null; done
}

pkill -f run_bc_iid_experiments 2>/dev/null

for sig in TERM TERM KILL; do
  pids=$(all_pids | sort -u)
  [ -z "$pids" ] && break
  echo "sending SIG$sig to: $(echo $pids | tr '\n' ' ')"
  # Children first, so the leader cannot hand out more work on its way down.
  for p in $(echo "$pids" | tac); do kill "-$sig" "$p" 2>/dev/null; done
  sleep 5
done

left=$(all_pids | sort -u)
if [ -n "$left" ]; then
  echo "STILL RUNNING after SIGKILL:"
  ps -o pid,etime,cmd -p $(echo $left | tr ' ' ',') 2>/dev/null | head -12
  exit 1
fi
echo "stopped -- no sweep or driver processes left"
STOP
}

# ------------------------------------------------------------ push / pull ---
# CLAUDE.md: use experiments/sync_results.sh for all transfers, never raw rsync.
# It pushes only git-tracked files (via `git ls-files`) and has no --delete, so
# it cannot repeat the 2026-09-14 wipe of experiments/runs/.
cmd_push() { bash "$HERE/experiments/sync_results.sh" push; }
cmd_pull() {
  bash "$HERE/experiments/sync_results.sh" pull "$RUN_ID"
  echo "plots (if any) are under experiments/plots/$RUN_ID/"
}

# --------------------------------------------------------------- cells -----
# Rounds completed per active cell, slowest last. The aggregate round counter
# cannot tell "everything is progressing evenly" from "four cells are done and
# four are stuck", which is the question whenever throughput looks wrong.
cmd_cells() {
  echo "$RUN_ID on $HOST -- rounds per active cell (scratch demo dirs)"
  ssh_bash "$RUN_ID" "$REMOTE_REPO" <<'CELLS'
set -u
RID="$1"; REPO="$2"
for stage in classical atari; do
  SCRATCH="$HOME/$REPO/experiments/runs/$RID/data/$stage/scratch"
  [ -d "$SCRATCH" ] || continue
  echo "--- $stage ---"
  for d in "$SCRATCH"/*/; do
    [ -d "$d" ] || continue
    n=$(find "$d/demos" -maxdepth 1 -type d -name 'round-*' 2>/dev/null | wc -l | tr -d ' ')
    age=$(( ( $(date +%s) - $(stat -c %Y "$d" 2>/dev/null || echo 0) ) / 60 ))
    printf '%6s rounds  %4s min since last write  %s\n' "$n" "$age" "$(basename "$d")"
  done | sort -rn
done
CELLS
}

case "${1:-status}" in
  -h|--help|help) ;;
  *) [ -n "$HOST" ] || die "Set HOST or add it to experiments/.local/runner.env" ;;
esac

case "${1:-status}" in
  status) cmd_status ;;
  cells)  cmd_cells ;;
  watch)  shift; cmd_watch "$@" ;;
  start)  cmd_start ;;
  stop)   cmd_stop ;;
  push)   cmd_push ;;
  runs)   cmd_runs ;;
  pull)   cmd_pull ;;
  -h|--help|help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}" ;;
  *) die "unknown command: $1   (try ./run.sh --help)" ;;
esac
