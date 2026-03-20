# Phase 3: Experiment Infrastructure - Research

**Researched:** 2026-03-20
**Domain:** GNU parallel multi-GPU orchestration, Sacred FileStorageObserver, per-round metric logging, tmux session management
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
None — all implementation choices are at Claude's discretion.

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase. Requirements are fully specified by REQUIREMENTS.md (INFRA-01 through INFRA-06).

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INFRA-01 | Sacred experiment entry point for running single (algorithm, game, seed) combination | `atari_smoke.py` is the refactoring base; entry point must accept algorithm/game/seed CLI args and attach a `FileStorageObserver` to a unique path |
| INFRA-02 | GPU orchestrator assigns experiments to 4 GPUs and launches via tmux | GNU parallel with `{%}` slot-number modulo 4 controls `CUDA_VISIBLE_DEVICES`; `tmux new-session` wraps the `parallel` invocation |
| INFRA-03 | Each experiment logs per-round metrics: reward, loss, eta_t, norm(g_t), round number | DAgger trainer emits `dagger/round_num` and `dagger/mean_episode_reward`; must add `eta_t`, `norm_g`, `loss` by subclassing or callback in `FTRLDAggerTrainer.extend_and_update` and in `run_dagger_round` callback |
| INFRA-04 | Separate Sacred FileStorageObserver directories per experiment (no run ID collisions) | Use `--file_storage output/sacred/{algo}/{game}/{seed}` CLI flag on each sub-process; Sacred run IDs are per-observer-dir, so unique dirs = no collisions |
| INFRA-06 | Full benchmark config: 7 games, 3+ seeds, 20+ DAgger rounds | Drive from shell arrays (7 games × 3 seeds × 3 algos = 63–84+ combos); `total_timesteps` must be high enough to yield 20+ rounds given `rollout_round_min_timesteps=500` and `n_envs=8` |
</phase_requirements>

---

## Summary

Phase 3 builds the shell scaffolding that takes the already-working single-game `atari_smoke.py` and scales it to 84+ parallel experiments across 4 GPUs. The work has three independent parts: (1) a Python entry point `experiments/run_atari_experiment.py` that wraps `atari_smoke.py` logic into a single Sacred-instrumented run for one `(algorithm, game, seed)` triple, (2) a `run_atari_benchmark.sh` bash script that uses GNU parallel with slot-based `CUDA_VISIBLE_DEVICES` assignment and launches everything inside a named tmux session, and (3) per-round metric emission plumbed through `FTRLDAggerTrainer` and its DAgger base class.

The Sacred collision problem (INFRA-04) is entirely avoided by giving each sub-process its own `--file_storage` directory. Sacred's `FileStorageObserver` assigns run IDs relative to its own directory root, so if every `(algo, game, seed)` combination writes to a distinct path there is no shared counter. This pattern is already used in the project's existing `imit_benchmark.sh` (per-environment `log_dir` on the `parallel` command line).

The per-round logging (INFRA-03) requires the smallest code change: `FTRLDAggerTrainer.extend_and_update` already has `eta_t` in scope. Adding `_logger.record` calls at the end of `extend_and_update` (before `return`) and a matching callback in the BC round-end path will emit all required scalars. `norm(g_t)` is the L2 norm of the list of `sigma_grad` tensors computed at round start.

**Primary recommendation:** Build `run_atari_experiment.py` as a thin Sacred wrapper around the existing `run_bc` / `run_dagger` helpers from `atari_smoke.py`, add three `_logger.record` calls to `FTRLDAggerTrainer.extend_and_update`, then write `run_atari_benchmark.sh` using `parallel --jobs 4 ... "CUDA_VISIBLE_DEVICES=$(({%}-1))"`.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| GNU parallel | system (≥20200522) | Job distribution, slot-based GPU assignment | Already used in bc_benchmark.sh / imit_benchmark.sh |
| Sacred | 0.8.x (project dep) | Experiment tracking, config injection, FileStorageObserver | Already wired into train_imitation.py |
| tmux | system | SSH-resilient session persistence | Standard on the CC-server |
| Python (argparse / Sacred CLI) | 3.8+ | Entry point argument parsing | Already used across experiments/ |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch (th.norm) | installed | Compute norm(g_t) for logging | Inside extend_and_update sigma_grad |
| imitation HierarchicalLogger | project | Per-round metric emission | `self._logger.record(...)` pattern already in DAgger |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Sacred FileStorageObserver | Plain CSV / JSON writer | Sacred gives structured per-run metadata for free; CSV is simpler but has no run hierarchy |
| GNU parallel | Ray, SLURM | Ray is already in the project (parallel.py) but requires a cluster daemon; GNU parallel has no setup cost on a single 4-GPU server |
| tmux | screen, nohup | tmux is already on the server per project_server_setup.md |

**Installation:** No new packages — all dependencies already satisfied.

---

## Architecture Patterns

### Recommended Project Structure
```
experiments/
├── run_atari_experiment.py   # INFRA-01: Sacred entry point, single (algo, game, seed)
├── run_atari_benchmark.sh    # INFRA-02 + INFRA-06: GNU parallel orchestrator
├── atari_smoke.py            # Unchanged — reuse run_bc / run_dagger helpers
├── atari_helpers.py          # Unchanged
└── baselines/
    └── atari_random_scores.pkl

output/
└── sacred/
    └── {algo}/               # e.g., bc, dagger, ftrl
        └── {game}/           # e.g., Pong, Breakout
            └── {seed}/       # e.g., 0, 1, 2
                └── 1/        # Sacred run ID — always 1 per unique dir
                    ├── run.json
                    ├── metrics.json
                    └── per_round_log.jsonl  # INFRA-03
```

### Pattern 1: Sacred Entry Point with FileStorageObserver

**What:** A Sacred experiment attached inside `main()` so the observer path is constructed from `--algo`, `--game`, `--seed` arguments before `ex.run_commandline()` is called.

**When to use:** Every sub-process invocation from `run_atari_benchmark.sh`.

**Example:**
```python
# experiments/run_atari_experiment.py
import sacred
from sacred.observers import FileStorageObserver

ex = sacred.Experiment("atari_run")

def main():
    import sys
    # Parse algo/game/seed from argv BEFORE Sacred consumes them.
    # Use a pre-pass with argparse (ignore_unknown=True), then build observer path.
    import argparse
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--algo", default="dagger")
    pre.add_argument("--game", default="Pong")
    pre.add_argument("--seed", type=int, default=0)
    pre.add_argument("--output-dir", default="output/sacred")
    known, _ = pre.parse_known_args()

    obs_path = f"{known.output_dir}/{known.algo}/{known.game}/{known.seed}"
    ex.observers.append(FileStorageObserver(obs_path))
    ex.run_commandline()
```

### Pattern 2: GNU Parallel with Slot-Based GPU Assignment

**What:** `parallel` distributes jobs using `{%}` (1-indexed slot number); `$(( {%} - 1 ))` maps to GPU index 0–3. The `--jobs 4` flag ensures at most 4 concurrent jobs (one per GPU).

**When to use:** `run_atari_benchmark.sh` outer loop.

**Example:**
```bash
# run_atari_benchmark.sh
#!/usr/bin/env bash
set -euo pipefail
source experiments/common.sh

GAMES=(Pong Breakout BeamRider Enduro Qbert Seaquest SpaceInvaders)
ALGOS=(bc dagger ftrl)
SEEDS=(0 1 2)
OUTPUT_DIR="output/sacred/${TIMESTAMP}"
N_ROUNDS=20
TOTAL_TIMESTEPS=500000   # ~20-25 rounds with n_envs=8, min_timesteps=500

export OMP_NUM_THREADS=2

parallel --jobs 4 --eta \
  "CUDA_VISIBLE_DEVICES=$(( {%} - 1 )) \
   python experiments/run_atari_experiment.py \
     --algo {1} --game {2} --seed {3} \
     --n-rounds ${N_ROUNDS} \
     --total-timesteps ${TOTAL_TIMESTEPS} \
     --output-dir ${OUTPUT_DIR}" \
  ::: "${ALGOS[@]}" \
  ::: "${GAMES[@]}" \
  ::: "${SEEDS[@]}"
```

### Pattern 3: tmux Session Wrapper

**What:** Wrap the `parallel` launch in a named tmux session so it survives SSH disconnection. The session name encodes the timestamp for easy identification.

**When to use:** Final `run_atari_benchmark.sh` invocation pattern.

**Example:**
```bash
# In run_atari_benchmark.sh, caller wraps via:
SESSION="atari_bench_${TIMESTAMP}"
tmux new-session -d -s "${SESSION}" \
  "bash run_atari_benchmark.sh --inner; tmux wait-for -S done-${SESSION}"
echo "Launched tmux session: ${SESSION}"
echo "Attach with: tmux attach -t ${SESSION}"
```

Alternatively, the script itself detects if it's running in tmux and re-launches if not:
```bash
if [ -z "${TMUX:-}" ]; then
  SESSION="atari_bench_${TIMESTAMP}"
  tmux new-session -d -s "$SESSION" "bash $0 --_in-tmux $*"
  echo "Launched in tmux session '$SESSION'. Attach with: tmux attach -t $SESSION"
  exit 0
fi
```

### Pattern 4: Per-Round Metric Logging (INFRA-03)

**What:** Emit `round`, `reward`, `loss`, `eta_t`, `norm_g_t` at the end of each DAgger/FTRL round. Use both the imitation `HierarchicalLogger` (for TensorBoard/stdout) and a JSONL sidecar file for easy downstream parsing.

**When to use:** `FTRLDAggerTrainer.extend_and_update` and `SimpleDAggerTrainer.extend_and_update` override / callback.

**Example (FTRL — add to end of extend_and_update, after `self.round_num += 1`):**
```python
# norm(g_t) = sqrt(sum of squared L2 norms across all param tensors)
import torch as th
norm_g = float(
    th.sqrt(sum(th.sum(g ** 2) for g in sigma_grad)).item()
)
logging.info(
    f"round={self.round_num - 1} eta_t={eta_t:.6f} norm_g={norm_g:.6f}"
)
# Emit via imitation logger (picked up by TensorBoard / stdout format strs)
self._logger.record("ftrl/eta_t", eta_t)
self._logger.record("ftrl/norm_g", norm_g)
self._logger.record("ftrl/round", self.round_num - 1)
# Per-round JSONL sidecar written from the entry point via Sacred _run.log_scalar
```

**For BC and DAgger (no eta_t / norm_g):** `SimpleDAggerTrainer.train()` already records `dagger/round_num` and `dagger/mean_episode_reward` at line 688. Loss is recorded inside `BC._log_batch` at every batch. The per-round log file just needs to aggregate these at round boundaries.

### Pattern 5: Sacred FileStorageObserver — No-Collision Directory Layout

**What:** Each Sacred observer is initialized with a fully unique leaf directory `output/sacred/{algo}/{game}/{seed}`. Sacred assigns run IDs sequentially within that leaf. Since no two processes share a leaf, IDs never conflict.

**When to use:** Always — the alternative (shared root + Sacred's locking) fails under high concurrency.

**Example:**
```python
# Unique per (algo, game, seed):
obs = FileStorageObserver("output/sacred/dagger/Pong/0")
# Sacred will create:  output/sacred/dagger/Pong/0/1/run.json
#                      output/sacred/dagger/Pong/0/1/metrics.json
```

This is the same pattern used in `imit_benchmark.sh`:
```bash
logging.log_dir="${LOG_ROOT}/{env_config_name}_{seed}"
```

### Anti-Patterns to Avoid

- **Shared FileStorageObserver root across parallel processes:** Sacred uses a simple integer counter file; concurrent processes will read/write the same counter and produce duplicate or skipped IDs. Fix: one leaf per run.
- **`CUDA_VISIBLE_DEVICES` set from a shared counter file:** Produces race conditions. Fix: derive GPU index purely from `{%}` (GNU parallel slot, process-local).
- **Launching experiments with `&` and `wait` instead of GNU parallel:** Loses load balancing, no retry, no progress display. Fix: use `parallel --jobs 4`.
- **Writing per-round logs inside the training loop without flushing:** Logs lost on crash. Fix: flush after every `_logger.dump()` call, or open the JSONL sidecar in line-buffered mode.
- **Using `total_timesteps` that is too small for 20+ rounds:** With `rollout_round_min_timesteps=500`, `n_envs=8`, each round consumes ≥500 timesteps. For 20 rounds: `total_timesteps >= 20 * 500 = 10000`. But in practice Atari episodes can be long; use `total_timesteps=500000` for robust 20+ rounds.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Job distribution across GPUs | Custom process pool with subprocess.Popen | GNU parallel with `{%}` | Handles retries, progress, load balancing; already in project |
| Experiment tracking / config management | Custom JSON config system | Sacred FileStorageObserver (already wired) | Run metadata, config snapshot, metrics JSON auto-written |
| SSH persistence | Custom daemon / systemd service | tmux | Zero configuration on CC-server; already documented in project memory |
| Per-round metric emission | Custom log file writer | `self._logger.record` (imitation HierarchicalLogger) | Already emits to TensorBoard and stdout; downstream Phase 4 reads it |
| Sacred run-ID uniqueness | Locking / distributed counter | Unique directory per run | Sacred's own mechanism, zero added code |

---

## Common Pitfalls

### Pitfall 1: Sacred Run-ID Collision Under Parallelism
**What goes wrong:** Two processes share a `FileStorageObserver` root. Sacred reads the current max run ID from the directory listing, both see the same number, and both write `run.json` to the same path (e.g., `3/run.json`), corrupting both runs.
**Why it happens:** Sacred uses directory enumeration (no file lock) to pick the next run ID.
**How to avoid:** Give every `(algo, game, seed)` its own observer root directory. Pass `--file_storage output/sacred/{algo}/{game}/{seed}` from the `parallel` command.
**Warning signs:** Duplicate `run.json` files; `metrics.json` mixing metrics from two different runs.

### Pitfall 2: Wrong GPU Assignment When {%} is 0-indexed vs 1-indexed
**What goes wrong:** GNU parallel's `{%}` is 1-indexed (1–N). Using `CUDA_VISIBLE_DEVICES={%}` directly maps jobs to GPU 1–4, skipping GPU 0.
**Why it happens:** Off-by-one; `{%}` starts at 1.
**How to avoid:** Use `$(( {%} - 1 ))` to get 0-indexed GPU IDs. Verified in GNU parallel documentation (slot function semantics).
**Warning signs:** `CUDA error: invalid device ordinal` for the first GPU if 4 GPUs are indexed 0–3.

### Pitfall 3: total_timesteps Too Small for 20+ Rounds
**What goes wrong:** DAgger's `train()` loop exits before reaching 20 rounds. With `rollout_round_min_timesteps=500` and `n_envs=8`, each `generate_trajectories` call collects at least 500 steps. But Atari episodes can last thousands of frames; one episode alone may exceed the `total_timesteps` budget.
**Why it happens:** `total_timesteps` is a lower bound on environment steps, not a round count. If one rollout episode is long, it can exhaust the budget before round 20.
**How to avoid:** Set `total_timesteps=500000` (≈20–25 rounds for typical Atari episodes). Validate with a 2-game smoke test that counts actual rounds.
**Warning signs:** `run.json` shows fewer than 20 rounds in `metrics.json`.

### Pitfall 4: OMP_NUM_THREADS Not Set — CPU Contention
**What goes wrong:** 4 concurrent PyTorch processes each spawn 16+ OpenMP threads. The server has limited CPU cores; threads contend and training slows to a crawl.
**Why it happens:** PyTorch defaults to all available CPU cores per process.
**How to avoid:** `export OMP_NUM_THREADS=2` (already in `experiments/common.sh`). Source `common.sh` in `run_atari_benchmark.sh`.
**Warning signs:** `htop` shows 100% CPU on all cores; GPU utilization low.

### Pitfall 5: tmux Session Exits Before Jobs Finish
**What goes wrong:** `tmux new-session -d "bash run_atari_benchmark.sh"` exits when the shell command finishes, which may happen before `parallel` detaches all child processes.
**Why it happens:** The shell exits and tmux destroys the session. `parallel` child processes may get SIGHUP.
**How to avoid:** Keep the shell alive: `tmux new-session -d -s "bench" "bash run_atari_benchmark.sh; read -p 'Done. Press enter to exit.'"`.
**Warning signs:** tmux session disappears while jobs are still running.

### Pitfall 6: Per-Round Log Metrics Missing for BC (No DAgger Rounds)
**What goes wrong:** BC does not have rounds in the same sense — it trains in epochs over a fixed dataset. Logging `round_num`, `eta_t`, `norm_g_t` for BC requires special handling (those fields don't exist for BC).
**Why it happens:** The requirements ask for per-round logs for all algorithms.
**How to avoid:** For BC: log epoch as "round 0"; set `eta_t=null` and `norm_g=null`. Or treat BC as a single-round algorithm (round 0 only). Document clearly in JSONL schema.
**Warning signs:** Phase 4 analysis code crashes on missing fields for BC runs.

---

## Code Examples

### Single-Run Entry Point (INFRA-01) — Minimal Sacred Wrapper

```python
# experiments/run_atari_experiment.py  (skeleton)
# Source: derived from atari_smoke.py + imit_benchmark.sh observer pattern

import argparse
import sys
from pathlib import Path
import sacred
from sacred.observers import FileStorageObserver

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

ex = sacred.Experiment("atari_run")

@ex.config
def cfg():
    algo = "dagger"         # bc | dagger | ftrl
    game = "Pong"
    seed = 0
    n_rounds = 20
    total_timesteps = 500000
    n_envs = 8
    alpha = 1.0             # FTRL only
    output_dir = "output/sacred"

@ex.main
def run(algo, game, seed, n_rounds, total_timesteps, n_envs, alpha, output_dir, _run):
    from experiments.atari_smoke import run_bc, run_dagger, load_random_baselines
    from experiments.atari_helpers import ATARI_GAMES
    import numpy as np

    game_id = ATARI_GAMES[game]
    baselines = load_random_baselines()
    random_score = baselines[game]["mean"]
    rng = np.random.default_rng(seed)

    if algo == "bc":
        norm_score = run_bc(game, game_id, seed, n_envs, random_score, rng)
        _run.log_scalar("normalized_score", norm_score)
    elif algo in ("dagger", "ftrl"):
        use_ftrl = (algo == "ftrl")
        norm_score = run_dagger(
            game, game_id, seed, n_envs, n_rounds, total_timesteps,
            random_score, rng, use_ftrl=use_ftrl, alpha=alpha,
        )
        _run.log_scalar("normalized_score", norm_score)
    return {"normalized_score": norm_score}


if __name__ == "__main__":
    # Pre-parse to build observer path before Sacred consumes argv
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--algo", default="dagger")
    pre.add_argument("--game", default="Pong")
    pre.add_argument("--seed", type=int, default=0)
    pre.add_argument("--output-dir", default="output/sacred")
    known, _ = pre.parse_known_args()
    obs_path = f"{known.output_dir}/{known.algo}/{known.game}/{known.seed}"
    ex.observers.append(FileStorageObserver(obs_path))
    ex.run_commandline()
```

### Per-Round Logging in FTRLDAggerTrainer (INFRA-03 — delta to ftrl.py)

```python
# Add at end of FTRLDAggerTrainer.extend_and_update, after self.round_num += 1:
import torch as th
norm_g = float(th.sqrt(sum(th.sum(g ** 2) for g in sigma_grad)).item())
self._logger.record("ftrl/round", self.round_num - 1)
self._logger.record("ftrl/eta_t", eta_t)
self._logger.record("ftrl/norm_g", norm_g)
# loss is already logged inside BC.train via _log_batch -> bc/neglogp etc.
```

### Per-Round Logging for DAgger (INFRA-03 — existing hooks)

```python
# Already in SimpleDAggerTrainer.train() at lines 687-690:
self._logger.record("dagger/total_timesteps", total_timestep_count)
self._logger.record("dagger/round_num", round_num)
self._logger.record("dagger/round_episode_count", round_episode_count)
self._logger.record("dagger/round_timestep_count", round_timestep_count)
# dagger/mean_episode_reward is recorded via record_mean inside the trajectory loop
# Add: self._logger.record("dagger/mean_bc_loss", ...) if needed
# Loss is accessible from BCTrainingMetrics inside BC._log_batch
```

### GNU Parallel GPU Assignment (INFRA-02)

```bash
# Source: GNU parallel documentation, slot() function
# {%} is 1-indexed slot number (1..N_JOBS)
CUDA_VISIBLE_DEVICES=$(( {%} - 1 )) python experiments/run_atari_experiment.py ...
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Ray Tune for parallelism (parallel.py) | GNU parallel for single-server 4-GPU runs | Project decision (STATE.md) | No daemon setup; simpler on single server |
| Shared Sacred log root | Per-(algo,game,seed) FileStorageObserver dir | This phase | Eliminates run-ID collision |

**Deprecated/outdated:**
- `experiments/parallel.py` (Ray-based): Not used for this benchmark; too heavy for single-server setup.

---

## Open Questions

1. **Round count determination for total_timesteps**
   - What we know: `rollout_round_min_timesteps=500`, `n_envs=8`, Atari episodes average ~400–2000 frames depending on game.
   - What's unclear: Whether `total_timesteps=500000` reliably produces 20+ rounds for all 7 games without being excessive.
   - Recommendation: Use `total_timesteps=500000` as default; add an assertion in `run_atari_experiment.py` that logs final round count and warns if < 20.

2. **Per-round reward logging for BC**
   - What we know: BC trains in epochs, not DAgger rounds. INFRA-03 asks for per-round logs including "reward".
   - What's unclear: Whether to evaluate BC after each epoch or just report a final score.
   - Recommendation: BC logs a single "round 0" entry with final normalized score. Skip per-round reward for BC (document in schema). This is consistent with BC not being an online algorithm.

3. **JSONL sidecar vs. Sacred metrics.json**
   - What we know: Sacred `_run.log_scalar(key, value, step)` writes to `metrics.json` in the run directory.
   - What's unclear: Whether Phase 4 analysis reads Sacred `metrics.json` directly or expects a different format.
   - Recommendation: Use `_run.log_scalar` exclusively for per-round metrics (Sacred handles storage). No custom JSONL sidecar needed. Phase 4 can read `metrics.json` using `json.load`.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (setup.cfg `[tool:pytest]`) |
| Config file | `setup.cfg` |
| Quick run command | `pytest tests/algorithms/ -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INFRA-01 | Entry point runs single (algo, game, seed) and produces Sacred output | smoke | `pytest tests/algorithms/test_experiment_entry_point.py -x` | Wave 0 |
| INFRA-02 | GNU parallel assigns correct CUDA_VISIBLE_DEVICES per slot | unit (env var inspection) | `pytest tests/algorithms/test_benchmark_script.py::test_gpu_assignment -x` | Wave 0 |
| INFRA-03 | Per-round metrics logged: round, reward, loss, eta_t, norm_g | unit | `pytest tests/algorithms/test_ftrl.py::test_per_round_logging -x` | ❌ Wave 0 |
| INFRA-04 | No Sacred run-ID collision under concurrent runs | unit | `pytest tests/algorithms/test_experiment_entry_point.py::test_no_id_collision -x` | Wave 0 |
| INFRA-06 | Full benchmark config launches 7 games × 3 seeds × 3 algos | smoke (dry-run echo mode) | `pytest tests/algorithms/test_benchmark_script.py::test_full_config_combinations -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/algorithms/test_ftrl.py -x -q`
- **Per wave merge:** `pytest tests/algorithms/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/algorithms/test_experiment_entry_point.py` — covers INFRA-01, INFRA-04
- [ ] `tests/algorithms/test_benchmark_script.py` — covers INFRA-02, INFRA-06 (dry-run / echo-mode)
- [ ] Add `test_per_round_logging` to existing `tests/algorithms/test_ftrl.py` — covers INFRA-03

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection — `experiments/atari_smoke.py`, `experiments/bc_benchmark.sh`, `experiments/imit_benchmark.sh`, `experiments/common.sh`
- `src/imitation/algorithms/ftrl.py` — `FTRLDAggerTrainer.extend_and_update`, `eta_t` computation, `sigma_grad`
- `src/imitation/algorithms/dagger.py` — `SimpleDAggerTrainer.train()`, existing `_logger.record` calls at lines 687–690
- `src/imitation/algorithms/bc.py` — `BCTrainingMetrics`, `_log_batch`, `_logger.record` patterns
- `src/imitation/scripts/train_imitation.py` — Sacred experiment + FileStorageObserver attachment pattern
- `src/imitation/scripts/ingredients/logging.py` — `setup_logging`, `make_log_dir`, Sacred symlink pattern
- `src/imitation/scripts/config/train_imitation.py` — named configs for all 7 Atari games already defined
- `.planning/STATE.md` — decision: "GNU parallel with slot()-1 for GPU assignment; no Ray cluster needed"

### Secondary (MEDIUM confidence)
- `experiments/dagger_benchmark.sh` — per-env `log_dir` pattern for avoiding Sacred collisions under parallel runs
- GNU parallel manual (slot `{%}` semantics, 1-indexed) — standard behavior documented in parallel(1) man page

### Tertiary (LOW confidence)
- None — all critical claims verified from codebase or project decisions.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already installed and in use
- Architecture: HIGH — patterns directly observed in existing benchmark scripts and source
- Pitfalls: HIGH — Sacred collision pattern directly traced in codebase; GPU indexing verified against STATE.md decisions

**Research date:** 2026-03-20
**Valid until:** 2026-04-19 (stable domain — GNU parallel, Sacred, tmux are not fast-moving)
