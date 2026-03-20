# DAgger vs FTRL Empirical Study

## What This Is

An empirical study verifying whether DAgger (FTL) is better than, or at least no worse than, its regularized variant FTRL for online imitation learning across a suite of 7 Atari games with discrete action spaces. Built on top of an existing popular imitation learning library that already implements BC and DAgger. The goal is to produce normalized performance curves (between random and expert) testing this hypothesis broadly, similar to Figure 4 in Lavington et al. (2022).

## Core Value

Verify empirically whether FTL (DAgger) is better than or at least no worse than FTRL across discrete-action Atari tasks, with normalized scores and publication-quality figures.

## Requirements

### Validated

- ✓ BC (Behavioral Cloning) algorithm — existing
- ✓ DAgger (FTL) algorithm — existing
- ✓ Trajectory collection and rollout infrastructure — existing
- ✓ Sacred experiment management — existing
- ✓ HuggingFace integration for expert policies — existing
- ✓ Score normalization utilities in benchmarking/ — existing

### Active

- [ ] FTRL algorithm implementation (regularized DAgger per Lavington et al. Eq. 6)
- [ ] Atari game suite setup (7 seals games with consistent preprocessing; ALE extension as v2)
- [ ] Expert policy acquisition from HuggingFace RL Zoo for all 7 games
- [ ] Random policy baseline scores for all 7 games
- [ ] Normalized evaluation pipeline (score = (agent - random) / (expert - random))
- [ ] Multi-GPU parallel experiment runner (4 GPUs on CC-server)
- [ ] Experiment scripts with logging for remote tmux execution
- [ ] Quick smoke-test configuration (1-2 games, few rounds)
- [ ] Full benchmark configuration (7 games, full training)
- [ ] Figure generation script producing normalized performance curves

### Out of Scope

- Alt-FTRL and AdaFTRL variants — not needed for this study
- Linear policy (pretrained features) setting — end-to-end neural net only
- Continuous control (MuJoCo) experiments — Atari discrete actions only
- GAIL, AIRL, or other adversarial methods — not relevant to this comparison
- Hyperparameter search — use paper's recommended settings or reasonable defaults

## Context

- **Paper reference**: Lavington et al. (2022) "Improved Policy Optimization for Online Imitation Learning" (CoLLAs 2022)
- **FTRL formulation**: Memory-efficient reformulation from Proposition 4.1, Eq. 6:
  `w_{t+1} = argmin_w [ Σ l_i(w) - ⟨w, Σ_{i=1}^{t-1} ∇l_i(w_t)⟩ + (1/(2η_t)) ||w - w_t||^2 ]`
  Three terms: (1) cumulative loss (same as FTL), (2) linear correction using gradients of past losses at current weights, (3) proximal term centered on current weights w_t. η_t = 1/(Σ σ_i).
- **Key insight from paper**: Figure 4 suggests FTL (DAgger) often has good empirical performance. Our goal is to verify whether this holds broadly across many discrete-action tasks — i.e., whether FTRL's regularization helps or is unnecessary.
- **Codebase**: Fork of the `imitation` library (HumanCompatibleAI) — well-structured with BC, DAgger, Sacred scripts, and benchmarking tools
- **Server**: CC-server accessible via `ssh CC-server`, 4 GPUs, needs venv and repo setup
- **Workflow**: Code edits locally → push to GitHub → sync to server → run in tmux with logging

## Constraints

- **Server setup**: Must use an isolated Python environment (venv or conda) on CC-server
- **GPU parallelism**: 4 GPUs available — design experiments to utilize all 4 concurrently
- **Remote execution**: All training/eval runs via tmux sessions for resilience to disconnects
- **Logging**: All processes must produce readable logs for monitoring progress
- **Expert fairness**: Same expert policy (from HuggingFace RL Zoo) used across all methods for each game
- **Normalization**: Scores normalized between random (0.0) and expert (1.0) for cross-game comparison

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| FTRL only (no Alt-FTRL/AdaFTRL) | Simplest comparison, paper shows FTRL is representative | — Pending |
| End-to-end neural net only | Practical setting users care about | — Pending |
| HuggingFace RL Zoo experts | Standard, reproducible, no extra compute for expert training | — Pending |
| Quick smoke test before full run | Catch bugs early, save GPU hours | — Pending |
| Isolated Python env on server | User requirement, cleaner than system Python | — Pending |

---
*Last updated: 2026-03-19 after initialization*
