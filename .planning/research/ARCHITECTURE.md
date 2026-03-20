# Architecture Patterns

**Domain:** Online imitation learning — FTRL extension of DAgger + multi-GPU Atari experiment runner
**Researched:** 2026-03-19

---

## Recommended Architecture

Three cleanly separated components with explicit ownership boundaries:

```
┌────────────────────────────────────────────────────────────────────┐
│  Component 1: FTRL Algorithm Layer                                 │
│  src/imitation/algorithms/ftrl.py                                  │
│                                                                    │
│  FTRLTrainer (extends SimpleDAggerTrainer)                         │
│    └── FTRLLossCalculator (replaces BehaviorCloningLossCalculator) │
│          └── BC.loss_calculator (injection point)                  │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │ trains a policy
┌─────────────────────────────────▼──────────────────────────────────┐
│  Component 2: Atari Experiment Infrastructure                      │
│  experiments/                                                      │
│                                                                    │
│  run_atari_benchmark.sh   — assigns game→GPU, launches tmux panes  │
│  configs/atari_games.py   — named_config per game for Sacred       │
│  scripts/run_single.py    — single (algorithm, game, seed) entry   │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │ produces sacred output dirs
┌─────────────────────────────────▼──────────────────────────────────┐
│  Component 3: Evaluation & Figure Pipeline                         │
│  experiments/                                                      │
│                                                                    │
│  collect_baselines.py     — random / expert score collection       │
│  normalize_scores.py      — (agent - random) / (expert - random)  │
│  plot_curves.py           — normalized performance curves          │
└────────────────────────────────────────────────────────────────────┘
```

---

## Component Boundaries

| Component | Responsibility | Communicates With | Does NOT Own |
|-----------|---------------|-------------------|--------------|
| `FTRLTrainer` | Algorithm correctness — FTRL loss, anchor weight management, training loop | `BC.train()`, `SimpleDAggerTrainer.train()` | GPU assignment, Sacred config, evaluation |
| `FTRLLossCalculator` | Compute FTRL loss (Eq. 6: neglogp + L2 regularization anchored to previous iterate) | `BC.optimizer`, `BC.policy` | Dataset loading, rollout collection |
| Atari experiment infra | GPU assignment, tmux orchestration, Sacred named configs per game | `train_imitation_ex` Sacred command | Algorithm implementation, score normalization |
| Evaluation pipeline | Load Sacred output, compute normalized scores, produce figures | Sacred filesystem output | Training logic |

---

## How FTRL Modifies the BC Training Loop

### The Formulation (Lavington et al. 2022, Proposition 4.1, Eq. 6)

FTRL differs from DAgger (FTL) only in the loss function passed to each BC update step. Standard DAgger minimizes the unregularized cumulative loss. FTRL adds a proximal (L2) term anchored to the **previous round's policy parameters**:

```
L_FTRL(θ; round t) = NLL(θ) + (λ/2) * ||θ - θ_{t-1}||²
```

where `θ_{t-1}` is a snapshot of policy parameters from the end of round `t-1`.

This is memory-equivalent to FTL because `θ_{t-1}` is a single parameter snapshot (not the full history), and the quadratic term introduces a linear correction equivalent to the sum of all past gradients in the standard FTRL derivation.

### What Changes in Code

The change is **surgical** — one class replaces one collaborator inside `BC`:

**Before (DAgger/FTL):**
```
BC.loss_calculator = BehaviorCloningLossCalculator(ent_weight, l2_weight)
# loss = neglogp + ent_loss + l2_loss_on_current_weights
```

**After (FTRL):**
```
BC.loss_calculator = FTRLLossCalculator(
    ent_weight,
    l2_weight,          # existing L2 on ||θ||² (unchanged)
    ftrl_lambda,        # new: regularization strength toward anchor
    anchor_params,      # new: snapshot of θ_{t-1} as List[Tensor]
)
# loss = neglogp + ent_loss + l2_loss + ftrl_lambda/2 * ||θ - anchor||²
```

`FTRLLossCalculator.__call__` adds the proximal term:

```python
# Pseudocode for FTRLLossCalculator
anchor_penalty = sum(
    th.sum(th.square(p - a))
    for p, a in zip(policy.parameters(), self.anchor_params)
) / 2
loss = neglogp + ent_loss + l2_loss + self.ftrl_lambda * anchor_penalty
```

### Anchor Update Protocol

`FTRLTrainer` inherits the `SimpleDAggerTrainer.train()` round loop and adds one step at round boundaries:

```
Round t:
  1. Collect rollouts (inherited from SimpleDAggerTrainer, unchanged)
  2. Snapshot θ_t = copy.deepcopy(list(bc_trainer.policy.parameters()))
  3. Call extend_and_update() — this calls BC.train() with the updated anchor
  4. Update FTRLLossCalculator.anchor_params = θ_t before training begins
```

The anchor must be updated **before** `BC.train()` is called for round `t`, so it reflects `θ_{t-1}` (the parameters that produced the rollouts just collected). This is implemented by overriding `extend_and_update()` in `FTRLTrainer`.

### What is Not Changed

- `DAggerTrainer.__init__` and all demonstration collection logic — completely unchanged
- `BetaSchedule` — unchanged, FTRL uses the same interpolation between expert and robot actions
- `BC.train()` — no modifications; the loss calculator is a constructor-injected collaborator
- `InteractiveTrajectoryCollector` — unchanged
- `SimpleDAggerTrainer.train()` round loop — only `extend_and_update` is overridden

---

## Class Hierarchy

```
BaseImitationAlgorithm
└── DAggerTrainer
    └── SimpleDAggerTrainer
        └── FTRLTrainer   ← new class, ~60 lines of new code

BehaviorCloningLossCalculator   (frozen dataclass)
└── FTRLLossCalculator          ← new class, ~30 lines of new code
    extra fields: ftrl_lambda: float, anchor_params: Optional[List[Tensor]]
```

`FTRLLossCalculator` cannot literally subclass `BehaviorCloningLossCalculator` because it is a frozen dataclass, but it can replicate the interface and be injected identically via `BC(loss_calculator=...)`. Confirm whether BC exposes `loss_calculator` as a constructor argument; currently it is constructed internally from `ent_weight` and `l2_weight`. **BC will need a one-line change to accept an optional external `loss_calculator` argument**, or `FTRLTrainer` replaces `bc_trainer.loss_calculator` after construction.

The safest approach: add `loss_calculator: Optional[BehaviorCloningLossCalculator] = None` to `BC.__init__`, defaulting to the existing behavior. This is a backward-compatible, two-line change.

---

## Data Flow for Multi-GPU Parallelism

The Atari benchmark is **embarrassingly parallel at the (algorithm, game, seed) level**. There is no data sharing between runs. The correct parallelism model is:

```
CC-server (4 GPUs)
│
├── GPU 0  →  games [0..2]   — DAgger seeds 1-3, FTRL seeds 1-3, BC seeds 1-3
├── GPU 1  →  games [3..5]   — same
├── GPU 2  →  games [6..8]   — same
└── GPU 3  →  games [9..11]  — same
```

Each GPU process is a **completely independent** `train_imitation` Sacred run. No inter-GPU communication, no shared memory, no distributed training library needed.

### Orchestration Approach: tmux + CUDA_VISIBLE_DEVICES

Do not use Ray Tune for this (the existing `parallel.py` uses Ray Tune for hyperparameter search, which is overkill for a fixed benchmark sweep). Use a simple bash script:

```bash
# run_atari_benchmark.sh
GAMES=(Pong Breakout SpaceInvaders MsPacman Qbert BeamRider Enduro Seaquest DemonAttack CrazyClimber)
ALGOS=(bc dagger ftrl)
SEEDS=(1 2 3)
GPU_ID=0

for GAME in "${GAMES[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU_ID \
    python -m imitation.scripts.train_imitation $ALGO \
      with atari_${GAME,,} seed=$SEED \
    >> logs/${ALGO}_${GAME}_seed${SEED}.log 2>&1 &
  GPU_ID=$(( (GPU_ID + 1) % 4 ))
done
```

A tmux session wraps this so disconnects do not kill it. Each run writes to its own Sacred output directory and its own log file. The constraint "4 GPUs, all utilized" is satisfied by round-robin game-to-GPU assignment before launching all processes in parallel.

### Data Flow Direction

```
HuggingFace RL Zoo
    │ download expert policy (once per game)
    ▼
Expert Policy (cached to disk)
    │ generate demonstrations (rollout.generate_trajectories)
    ▼
Demo .npz files (scratch_dir/demos/round-000/)
    │ loaded by DAgger/FTRL _try_load_demos()
    ▼
PyTorch DataLoader (in-memory, per-run)
    │ batches to BC.train()
    ▼
Trained Policy (.pt file in scratch_dir)
    │ eval_policy runs N episodes
    ▼
Sacred output (return_mean, episode_len, etc.)
    │ sacred_output_to_csv.py (existing benchmarking tool)
    ▼
CSV with (algorithm, game, seed, mean_return)
    │ normalize_scores.py
    ▼
Normalized scores CSV  →  plot_curves.py  →  Figure (PDF/PNG)
```

---

## Sacred Integration for FTRL

Following the existing pattern, FTRL is added as a new command to `train_imitation_ex`:

```python
# In src/imitation/scripts/train_imitation.py
@train_imitation_ex.command
def ftrl(
    bc: Dict[str, Any],
    dagger: Mapping[str, Any],
    ftrl: Mapping[str, Any],   # new ingredient
    _run,
    _rnd: np.random.Generator,
) -> Mapping[str, Mapping[str, float]]:
    ...
```

And a new `ftrl` config block in `train_imitation_ex.config`:

```python
# In src/imitation/scripts/config/train_imitation.py
@train_imitation_ex.config
def config():
    dagger = dict(...)       # unchanged
    ftrl = dict(
        ftrl_lambda=1.0,     # regularization toward previous iterate
        total_timesteps=1e5, # same interface as dagger
        beta_schedule=None,
    )
```

Named configs for Atari games follow the same pattern as the existing `bc_seals_ant`, `dagger_seals_ant` configs — one file per `(algorithm, game)` pair loaded from JSON.

---

## Suggested Build Order

### Phase 1: FTRL Algorithm (prerequisite for everything else)

Build `FTRLLossCalculator` and `FTRLTrainer` first. Validate correctness on a tiny environment (CartPole or seals/CartPole) before any Atari work. The FTRL loss can be unit-tested independently of training by constructing a mock policy and verifying the anchor penalty term is zero at round 0, and positive after one round.

**Deliverables:**
- `src/imitation/algorithms/ftrl.py`
- Unit tests in `tests/test_ftrl.py`
- Smoke test: `python -m imitation.scripts.train_imitation ftrl with fast`

### Phase 2: Atari Game Configs and Expert Acquisition

Add Sacred named configs for each Atari game. Verify HuggingFace RL Zoo expert policies load correctly for each game. Collect random baseline scores (trivially: run random policy for N episodes per game). Verify score normalization formula produces sensible outputs.

**Deliverables:**
- `src/imitation/scripts/config/atari_games.py` (named configs)
- `experiments/collect_baselines.py`
- Verified expert download for all 10+ games

### Phase 3: Multi-GPU Experiment Runner

Write the bash orchestration script. Test on 2 games and 2 GPUs before committing to the full benchmark. The smoke test config (1-2 games, 3 rounds, reduced timesteps) is critical here for validating the orchestration before paying full GPU-hours.

**Deliverables:**
- `experiments/run_atari_benchmark.sh`
- `experiments/run_smoke_test.sh`
- Verified Sacred output structure for all three algorithms

### Phase 4: Evaluation and Figure Generation

Collect Sacred outputs into CSV, apply normalization, generate curves. This phase has no algorithmic risk but moderate pipeline engineering complexity (handling missing runs, aligning seeds, etc.).

**Deliverables:**
- `experiments/normalize_scores.py`
- `experiments/plot_curves.py`
- Figure matching style of Lavington et al. 2022 Figure 4

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Modifying BC.train() Directly

**What goes wrong:** Adding FTRL-specific logic inside `BC.train()` (e.g., `if self.is_ftrl: ...`) contaminates BC with algorithm-specific branching and makes future extension harder.

**Why bad:** BC is used standalone and by DAgger. FTRL is a client of BC, not a modification of it. The existing codebase correctly uses composition (loss calculator injection) for exactly this kind of variation.

**Instead:** Keep all FTRL logic in `FTRLLossCalculator` and `FTRLTrainer`. BC should not know FTRL exists.

### Anti-Pattern 2: Storing Full Parameter History for the Anchor

**What goes wrong:** Accumulating all past policy snapshots to compute the "true" FTRL update. This grows memory linearly with rounds.

**Why bad:** The whole point of Proposition 4.1 in Lavington et al. is that FTRL can be reformulated using only the previous iterate, not all past iterates. Storing full history contradicts the paper's formulation and destroys the memory efficiency claim.

**Instead:** Store exactly one anchor snapshot (`θ_{t-1}`). Overwrite it at each round boundary.

### Anti-Pattern 3: Using Ray Tune for Benchmark Parallelism

**What goes wrong:** Using the existing `parallel.py` (Ray Tune-based) to run the Atari benchmark sweep. Ray Tune adds significant overhead, complexity, and scheduler interference for a fixed benchmark with no hyperparameter search.

**Why bad:** The benchmark uses fixed hyperparameters. Ray Tune is built for search. Its worker scheduling may not honor `CUDA_VISIBLE_DEVICES` assignments reliably across the 4-GPU setup.

**Instead:** Use `CUDA_VISIBLE_DEVICES` + bash `&` for background processes + tmux for session persistence. This is what `run_all_benchmarks.sh` in benchmarking/ already does for the existing benchmark.

### Anti-Pattern 4: Evaluating During FTRL Training Rounds

**What goes wrong:** Running the log_rollouts_venv evaluation pass (which calls `RolloutStatsComputer`) every batch during FTRL training. For Atari, each evaluation episode is long.

**Why bad:** Mid-training evaluations can consume 30-50% of total training time for Atari environments. The benchmark only needs final policy scores, not per-batch curves.

**Instead:** Set `log_rollouts_n_episodes=0` during training. Run evaluation once at the end via `policy_evaluation.eval_policy`.

### Anti-Pattern 5: Tight Coupling Between Experiment Runner and Algorithm

**What goes wrong:** Encoding algorithm-specific logic (FTRL lambda schedule, anchor reset logic) inside the bash runner or Sacred config rather than inside `FTRLTrainer`.

**Why bad:** Makes it impossible to test the algorithm in isolation and forces algorithm bugs to manifest only during full benchmark runs.

**Instead:** The bash runner only sets `CUDA_VISIBLE_DEVICES` and launches Sacred commands. All algorithmic configuration stays in Sacred named configs and `FTRLTrainer` defaults.

---

## Scalability Considerations

| Concern | At 10 games x 3 seeds x 3 algos = 90 runs | At 20+ games |
|---------|-------------------------------------------|--------------|
| GPU utilization | 90 runs / 4 GPUs = ~23 sequential per GPU. Full utilization throughout. | Same pattern, just more sequential runs per GPU. |
| Storage | Each Sacred run writes ~100MB. Total ~9GB. | Linear with games. |
| Expert caching | Each game needs one expert download. Cache to shared disk. | Download once, reuse across algorithms and seeds. |
| Normalization | Random scores collected once per game (not per algorithm). | Same. |

---

## Sources

- Direct code analysis: `src/imitation/algorithms/dagger.py`, `src/imitation/algorithms/bc.py` (HIGH confidence — source of truth)
- Direct code analysis: `src/imitation/regularization/regularizers.py` (HIGH confidence)
- Direct code analysis: `src/imitation/scripts/parallel.py`, `benchmarking/run_all_benchmarks.sh` (HIGH confidence)
- Lavington et al. (2022) "Improved Policy Optimization for Online Imitation Learning" — Proposition 4.1 and Eq. 6 for the FTRL loss formulation. Referenced from PROJECT.md (MEDIUM confidence — paper content not directly retrieved, relies on project description)
