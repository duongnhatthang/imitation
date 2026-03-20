# DAgger vs FTRL Empirical Study

## What This Is

An empirical study comparing DAgger (FTL) against its regularized variant FTRL for online imitation learning across a standard suite of 10+ Atari games. Built on top of an existing popular imitation learning library that already implements BC and DAgger. The goal is to produce normalized performance curves (between random and expert) showing whether FTRL improves upon or matches DAgger, similar to Figure 4 in Lavington et al. (2022).

## Core Value

Produce a fair, reproducible comparison of FTL vs FTRL vs BC (plus expert baseline) across a broad Atari benchmark with normalized scores, generating publication-quality figures.

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
- [ ] Atari game suite setup (10+ games with consistent preprocessing)
- [ ] Expert policy acquisition from HuggingFace RL Zoo for all games
- [ ] Random policy baseline scores for all games
- [ ] Normalized evaluation pipeline (score = (agent - random) / (expert - random))
- [ ] Multi-GPU parallel experiment runner (4 GPUs on CC-server)
- [ ] Experiment scripts with logging for remote tmux execution
- [ ] Quick smoke-test configuration (1-2 games, few rounds)
- [ ] Full benchmark configuration (10+ games, full training)
- [ ] Figure generation script producing normalized performance curves

### Out of Scope

- Alt-FTRL and AdaFTRL variants — not needed for this study
- Linear policy (pretrained features) setting — end-to-end neural net only
- Continuous control (MuJoCo) experiments — Atari discrete actions only
- GAIL, AIRL, or other adversarial methods — not relevant to this comparison
- Hyperparameter search — use paper's recommended settings or reasonable defaults

## Context

- **Paper reference**: Lavington et al. (2022) "Improved Policy Optimization for Online Imitation Learning" (CoLLAs 2022)
- **FTRL formulation**: Memory-efficient reformulation from Proposition 4.1, Eq. 6 — adds L2 regularization with a linear correction term anchored to previous iterates, matching FTL's memory footprint
- **Key insight from paper**: FTRL often matches or outperforms FTL (DAgger) in average cumulative loss and sometimes in return; both dominate on-policy methods (OGD, AdaGrad)
- **Codebase**: Fork of the `imitation` library (HumanCompatibleAI) — well-structured with BC, DAgger, Sacred scripts, and benchmarking tools
- **Server**: CC-server accessible via `ssh CC-server`, 4 GPUs, needs venv and repo setup
- **Workflow**: Code edits locally → push to GitHub → sync to server → run in tmux with logging

## Constraints

- **Server setup**: Must use Python virtual environment (not conda) on CC-server
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
| Virtual environment on server | User requirement, cleaner than system Python | — Pending |

---
*Last updated: 2026-03-19 after initialization*
