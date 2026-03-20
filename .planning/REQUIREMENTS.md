# Requirements: DAgger vs FTRL Empirical Study

**Defined:** 2026-03-19
**Core Value:** Fair, reproducible comparison of FTL vs FTRL vs BC across Atari games with normalized scores and publication-quality figures

## v1 Requirements

Requirements for initial benchmark. Each maps to roadmap phases.

### Algorithm

- [ ] **ALGO-01**: FTRL algorithm implemented as `FTRLDAggerTrainer` subclassing `SimpleDAggerTrainer`
- [ ] **ALGO-02**: FTRL loss includes round-dependent L2 regularization `(1/(2*eta_t))||w||^2` with `eta_t = alpha/sqrt(t)`
- [ ] **ALGO-03**: FTRL loss includes linear anchor term `-w^T * g_t` where `g_t` accumulates from previous iterates
- [ ] **ALGO-04**: g_t stored as `dict[str, Tensor]`, detached, constant within round, updated between rounds
- [ ] **ALGO-05**: FTRL degenerates to FTL when alpha → infinity (verified by test)
- [ ] **ALGO-06**: FTRL passes smoke test on CartPole matching or exceeding BC performance

### Environment

- [ ] **ENV-01**: 7 seals Atari games configured: Pong, Breakout, BeamRider, Enduro, Qbert, Seaquest, SpaceInvaders
- [ ] **ENV-02**: Consistent Atari preprocessing: frame stacking (4), grayscale, 84x84 resize
- [ ] **ENV-03**: Expert policies loaded from HuggingFace `sb3` org for all 7 games
- [ ] **ENV-04**: Expert observation space matches learner observation space (verified by assertion)
- [ ] **ENV-05**: Random baseline scores computed and cached for all 7 games

### Experiment Infrastructure

- [ ] **INFRA-01**: Sacred experiment entry point for running single (algorithm, game, seed) combination
- [ ] **INFRA-02**: GPU orchestrator assigns experiments to 4 GPUs and launches via tmux
- [ ] **INFRA-03**: Each experiment logs per-round metrics: reward, loss, eta_t, norm(g_t), round number
- [ ] **INFRA-04**: Separate Sacred FileStorageObserver directories per experiment (no run ID collisions)
- [ ] **INFRA-05**: Quick smoke-test config: 1-2 games, 1 seed, 3-5 DAgger rounds
- [ ] **INFRA-06**: Full benchmark config: 7 games, 3+ seeds, 20+ DAgger rounds
- [ ] **INFRA-07**: Server setup script: create isolated Python env, install dependencies, clone repo

### Evaluation & Visualization

- [ ] **EVAL-01**: Normalized scores computed as `(agent - random) / (expert - random)` for each game
- [ ] **EVAL-02**: Learning curves plotted: normalized return vs environment interactions for each game
- [ ] **EVAL-03**: Aggregate metrics across games: mean and IQM with 95% confidence intervals
- [ ] **EVAL-04**: Figure generation works on partial results (incremental — runnable anytime mid-experiment)
- [ ] **EVAL-05**: Completion dashboard showing which (algorithm, game, seed) runs are done/running/pending
- [ ] **EVAL-06**: Final publication-quality figures comparing FTL, FTRL, BC, and Expert baselines
- [ ] **EVAL-07**: All figures use consistent colors/styles across methods

## v2 Requirements

Deferred to future. Tracked but not in current roadmap.

### Extended Games

- **EXT-01**: Extend to full ALE game suite (20+ games)
- **EXT-02**: Support raw ALE environments beyond seals wrappers

### Additional Algorithms

- **ALG-01**: AdaFTRL (adaptive FTRL variant)
- **ALG-02**: Alt-FTRL (alternative reformulation)

### Advanced Analysis

- **ADV-01**: Probability of Improvement statistical tests between methods
- **ADV-02**: Per-game scatter plots and performance profiles
- **ADV-03**: Alpha sensitivity analysis across games

## Out of Scope

| Feature | Reason |
|---------|--------|
| Linear policy (pretrained features) setting | End-to-end only per user decision |
| Continuous control (MuJoCo) experiments | Atari discrete actions only |
| GAIL/AIRL/adversarial methods | Not relevant to FTL vs FTRL comparison |
| Hyperparameter search (Optuna) | Use paper's settings or reasonable defaults |
| Training expert policies from scratch | Use HuggingFace pretrained experts |
| Within-game multi-GPU parallelism | Networks too small to benefit; per-game parallelism instead |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ALGO-01 | TBD | Pending |
| ALGO-02 | TBD | Pending |
| ALGO-03 | TBD | Pending |
| ALGO-04 | TBD | Pending |
| ALGO-05 | TBD | Pending |
| ALGO-06 | TBD | Pending |
| ENV-01 | TBD | Pending |
| ENV-02 | TBD | Pending |
| ENV-03 | TBD | Pending |
| ENV-04 | TBD | Pending |
| ENV-05 | TBD | Pending |
| INFRA-01 | TBD | Pending |
| INFRA-02 | TBD | Pending |
| INFRA-03 | TBD | Pending |
| INFRA-04 | TBD | Pending |
| INFRA-05 | TBD | Pending |
| INFRA-06 | TBD | Pending |
| INFRA-07 | TBD | Pending |
| EVAL-01 | TBD | Pending |
| EVAL-02 | TBD | Pending |
| EVAL-03 | TBD | Pending |
| EVAL-04 | TBD | Pending |
| EVAL-05 | TBD | Pending |
| EVAL-06 | TBD | Pending |
| EVAL-07 | TBD | Pending |

**Coverage:**
- v1 requirements: 25 total
- Mapped to phases: 0
- Unmapped: 25 ⚠️

---
*Requirements defined: 2026-03-19*
*Last updated: 2026-03-19 after initial definition*
