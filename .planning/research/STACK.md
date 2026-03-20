# Technology Stack: FTRL Atari Benchmark

**Project:** DAgger vs FTRL Empirical Study — Atari Milestone
**Researched:** 2026-03-19
**Scope:** Additive dependencies only. Existing stack (PyTorch, SB3, Sacred, gymnasium ~0.29) is documented in `.planning/codebase/STACK.md`.

---

## Recommended Stack — New Dependencies

### Atari Environment Layer

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `seals[atari]` | `~0.2.1` (already pinned) | Atari game wrappers with masked scores and fixed-length episodes | Already the project's Atari dependency. Registers 7 games as `seals/{Game}-v5` with `MaskScoreWrapper` and `AutoResetWrapper` for fixed episode length — required for fair IL comparison. |
| `ale-py` | pulled in by `seals[atari]` | ALE backend for Atari emulation | seals pins ale-py transitively. As of ale-py 0.9+, ROMs are bundled in the PyPI wheel — no `AutoROM` or `ale-import-roms` step needed on CC-server. |

**Confidence:** HIGH (verified via seals source code and ale-py release notes)

**Critical note:** seals 0.2.1 targets gymnasium ~0.26–0.29 (the existing constraint). ale-py 0.11.x requires Python 3.9+ and targets gymnasium 1.x. The existing `gymnasium ~0.29` pin prevents upgrading ale-py to latest. This is intentional — stay on seals 0.2.1 + gymnasium 0.29 + ale-py compatible with that range. Do NOT upgrade to gymnasium 1.x for this milestone.

**Confidence on gymnasium lock:** HIGH (verified from seals PyPI metadata and ale-py release notes)

### The 7 seals Atari Games

These are the only games seals 0.2.1 wraps. They cover the standard 7-game RL Zoo Atari benchmark:

| `gym_id` | Underlying ALE |
|----------|----------------|
| `seals/BeamRider-v5` | BeamRider |
| `seals/Breakout-v5` | Breakout |
| `seals/Enduro-v5` | Enduro |
| `seals/Pong-v5` | Pong |
| `seals/Qbert-v5` | Q*bert |
| `seals/Seaquest-v5` | Seaquest |
| `seals/SpaceInvaders-v5` | SpaceInvaders |

Masked variants (`seals/{Game}-Unmasked-v5`) expose the score. Use the masked default for training; unmasked is for inspection only.

**Confidence:** HIGH (read directly from seals v0.2.1 source `src/seals/atari.py`)

### Expert Policy Acquisition

| Component | Details | Why |
|-----------|---------|-----|
| `huggingface_sb3` | `~3.0` (already in project) | Downloads SB3-format `.zip` policy files from HF Hub |
| HF organization | `sb3` (not the default `HumanCompatibleAI`) | `HumanCompatibleAI` org has PPO models for seals MuJoCo environments but only one Atari model (`ppo-AsteroidsNoFrameskip-v4`). The `sb3` org hosts `ppo-BreakoutNoFrameskip-v4`, `ppo-PongNoFrameskip-v4`, `ppo-SpaceInvadersNoFrameskip-v4`, and more. |
| env name format | `BreakoutNoFrameskip-v4` (NoFrameskip-v4 suffix) | The `sb3` HF repos follow the RL Zoo naming convention: `ppo-{Game}NoFrameskip-v4`. The seals `seals/Breakout-v5` env is internally backed by the same ALE ROM but seals wraps it. |

**Configuration pattern** — in Sacred config override:
```python
expert = dict(
    policy_type="ppo-huggingface",
    loader_kwargs=dict(
        organization="sb3",
        env_name="BreakoutNoFrameskip-v4",  # HF repo name, not seals gym_id
    ),
)
```

**Confidence:** MEDIUM — confirmed `sb3` org has `ppo-BreakoutNoFrameskip-v4`, `ppo-PongNoFrameskip-v4`, `ppo-SpaceInvadersNoFrameskip-v4` on HuggingFace. Coverage for all 7 seals games (BeamRider, Enduro, Qbert) is not fully confirmed via search and needs manual verification before full benchmark run. Risk is manageable: run `python -m rl_zoo3.load_from_hub --algo ppo --env {Game}NoFrameskip-v4 -orga sb3` to check coverage.

**Fallback:** If `sb3` org is missing a game, use `rl_zoo3.load_from_hub` directly to download from the RL Zoo (`-orga sb3`) and pass a local path via `policy_type="ppo"` with `loader_kwargs={"path": "/path/to/model"}`.

### Multi-GPU Experiment Parallelism

**Recommended tool:** GNU `parallel` (not Ray, not Slurm)

| Tool | Version | Purpose | Why |
|------|---------|---------|-----|
| GNU `parallel` | system package | Distribute experiment jobs across 4 GPUs | Perfectly suited: no SLURM, no daemon, works over SSH, tmux-native. The existing `run_all_benchmarks.sh` runs sequentially; replacing it with GNU parallel gives immediate 4x throughput. |

**Why not Ray?** Ray is already in the project (`ray~2.0.0` in `parallel` extra) and `imitation-parallel` exists. However Ray requires a cluster daemon, adds overhead, and the existing benchmark pattern (Sacred CLI calls) maps more naturally to process-level parallelism. For 4 GPUs running independent Sacred experiments, GNU parallel is lower friction.

**Why not Slurm?** CC-server is a 4-GPU workstation, not a cluster. The existing `run_all_benchmarks_on_slurm.sh` targets a cluster — this server uses direct process execution.

**Confidence:** HIGH

**GPU assignment pattern:**
```bash
# Run up to 4 jobs in parallel, assigning each to one GPU by slot index
parallel -j4 \
    'CUDA_VISIBLE_DEVICES={=1 $_=slot()-1 =} python -m imitation.scripts.train_imitation {1} with {2} seed={3} 2>&1 | tee logs/{1}_{2}_seed{3}.log' \
    ::: bc dagger ftrl \
    ::: atari_breakout atari_pong atari_seaquest \
    ::: 1 2 3
```

The `slot()-1` expression maps parallel's job slot (1-indexed) to GPU IDs (0-indexed). With `-j4`, jobs queue and the next available GPU picks up the next job automatically.

**Confidence:** HIGH (GNU parallel `slot()` GPU assignment is a documented pattern)

### Logging and Monitoring for Remote tmux Runs

**Strategy:** `2>&1 | tee` at the process level, tmux `pipe-pane` at the session level.

| Component | Purpose | Why |
|-----------|---------|-----|
| `tee logs/{run}.log` | Per-run log file | Each Sacred run's stdout+stderr goes to a named log file. Survives tmux detach. Monitorable with `tail -f`. |
| `tmux pipe-pane` | Full session capture | Optionally pipe entire tmux window output to a session log for forensics after disconnects. |
| Sacred file observer | Structured metrics | Sacred already writes metrics to `sacred_output/`. The `sacred_output_to_csv.py` script in `benchmarking/` converts this to CSV for analysis. |
| TensorBoard (existing) | Live training curves | SB3 integration already in the codebase. Run `tensorboard --logdir logs/` on CC-server and SSH tunnel port 6006 for remote monitoring. |

**Log file naming convention:**
```
logs/{algorithm}_{game}_seed{N}.log
logs/{algorithm}_{game}_seed{N}_sacred/  # Sacred output dir
```

**Remote monitoring command:**
```bash
# On local machine — tail all active run logs
ssh CC-server "tail -f /path/to/experiment/logs/*.log"
```

**Confidence:** HIGH (tmux + tee is standard practice; Sacred's file observer is already in the codebase)

---

## Installation on CC-Server (Python venv)

The server requires a Python virtual environment, not conda.

```bash
# Create venv (Python 3.9+ recommended for SB3 2.x compatibility)
python3 -m venv .venv
source .venv/bin/activate

# Install imitation with Atari support
pip install -e ".[atari]"

# Verify ALE + seals Atari envs register correctly
python -c "import ale_py; import gymnasium as gym; import seals; env = gym.make('seals/Pong-v5'); print(env)"

# Install GNU parallel (if not present)
sudo apt-get install -y parallel  # or use local install without sudo
```

**ale-py ROM verification:** With ale-py >= 0.9, ROMs are bundled. No `AutoROM` step needed. If you get `rom not found` errors, the ale-py version is too old — check `pip show ale-py`.

**Confidence:** HIGH for venv setup; MEDIUM for ROM bundling (depends on which ale-py version seals 0.2.1 pulls in)

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Multi-GPU dispatch | GNU parallel | Ray (already in project) | Ray adds daemon overhead; Sacred CLI jobs are independent processes — no distributed state needed |
| Multi-GPU dispatch | GNU parallel | xargs | xargs lacks slot-to-GPU mapping, no tmux integration |
| Expert source | `sb3` HF org | Train from scratch with RL Zoo | Training adds compute cost and reproducibility risk; pre-trained `sb3` experts are the standard for this benchmark |
| Expert source | `sb3` HF org | `HumanCompatibleAI` org | HumanCompatibleAI only has seals MuJoCo + CartPole + Asteroids; missing Breakout, Pong, Seaquest, etc. |
| Environment format | `seals/{Game}-v5` | `{Game}NoFrameskip-v4` bare ALE | seals adds `MaskScoreWrapper` (hides on-screen score) which is essential for fair IL: without it, the agent can exploit the displayed score as a reward signal |
| Gymnasium version | `~0.29` (keep existing) | Gymnasium 1.x | Upgrading breaks seals 0.2.1 compatibility; seals is not yet updated to gymnasium 1.x |

---

## Version Compatibility Matrix

| Package | Current Pinned | Latest (2026-03) | Compatible? |
|---------|---------------|-----------------|-------------|
| seals | 0.2.1 | 0.2.1 | YES — already latest |
| gymnasium | ~0.29 | 1.2.3 | LOCKED — do not upgrade |
| ale-py | pulled by seals | 0.11.2 | Must match seals' constraint (~0.8.x likely) |
| stable-baselines3 | ~2.2.1 | 2.7.1 | LOCKED — upgrading may break imitation internals |
| huggingface_sb3 | ~3.0 | 3.x | OK as-is |
| GNU parallel | system | 20260222 | No pip install; apt or brew |

**Confidence:** HIGH for gymnasium/SB3 lock; MEDIUM for exact ale-py version pulled by seals 0.2.1 (needs `pip install seals[atari]~=0.2.1` and `pip show ale-py` on CC-server to confirm)

---

## Open Questions (Needs Validation on CC-Server)

1. **ale-py version compatibility:** Run `pip show ale-py` after `pip install -e ".[atari]"` to confirm which ale-py version seals 0.2.1 resolves to and whether ROMs are bundled.

2. **sb3 HF org Atari coverage:** Manually verify `sb3/ppo-BeamRiderNoFrameskip-v4`, `sb3/ppo-EnduroNoFrameskip-v4`, `sb3/ppo-QbertNoFrameskip-v4` exist before committing to all 7 games. Confirmed available: Breakout, Pong, SpaceInvaders. The `HumanCompatibleAI/ppo-AsteroidsNoFrameskip-v4` is confirmed (but Asteroids is not a seals game).

3. **seals Atari env registration:** After installing, run `python -c "import seals; import gymnasium; print([e for e in gymnasium.envs.registry if 'seals' in e and any(g in e for g in ['Pong','Breakout','Seaquest'])])"` to confirm all 7 seals Atari envs register under the current gymnasium version.

4. **GPU availability format:** Confirm `nvidia-smi` shows 4 GPUs numbered 0-3 on CC-server before using `slot()-1` GPU assignment.

---

## Sources

- seals v0.2.1 source: `src/seals/atari.py` — game list and wrappers (HIGH confidence)
- ale-py PyPI: [ale-py 0.11.2](https://pypi.org/project/ale-py/) — latest version, ROM bundling confirmed (HIGH)
- ale-py release notes: [ALE Release Notes](https://ale.farama.org/release_notes/index.html) — ROM bundling since 0.9.0, gymnasium 1.x migration (HIGH)
- gymnasium PyPI: [gymnasium 1.2.3](https://pypi.org/project/gymnasium/) — current stable (HIGH)
- stable-baselines3 PyPI: [SB3 2.7.1](https://pypi.org/project/stable-baselines3/) — current stable (HIGH)
- HuggingFace sb3 org: [sb3 models](https://huggingface.co/sb3) — confirmed ppo-BreakoutNoFrameskip-v4, ppo-PongNoFrameskip-v4, ppo-SpaceInvadersNoFrameskip-v4 (MEDIUM — not all 7 games confirmed)
- HuggingFace HumanCompatibleAI: [HumanCompatibleAI models](https://huggingface.co/HumanCompatibleAI) — only MuJoCo seals + CartPole + Asteroids (HIGH — searched exhaustively)
- RL Zoo HF docs: [Using RL-Baselines3-Zoo at Hugging Face](https://huggingface.co/docs/hub/rl-baselines3-zoo) — load_from_hub CLI (HIGH)
- GNU parallel docs: [GNU Parallel Tutorial](https://www.gnu.org/software/parallel/parallel_tutorial.html) — slot()-based GPU assignment (HIGH)
- imitation serialize.py: `src/imitation/policies/serialize.py` — organization parameter in HF loader (HIGH — read source)
- shimmy: Atari support removed from shimmy 2.0 since ale-py now has native gymnasium bindings (MEDIUM via release notes search)
