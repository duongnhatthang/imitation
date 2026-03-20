# DAgger vs FTRL Empirical Study

## What This Is

An empirical study verifying whether DAgger (FTL) is better than, or at least no worse than, its regularized variant FTRL for online imitation learning across a suite of 7 Atari games with discrete action spaces. Built on top of the `imitation` library (HumanCompatibleAI). Produces normalized performance curves (between random and expert) testing this hypothesis broadly, similar to Figure 4 in Lavington et al. (2022).

## Core Value

Verify empirically whether FTL (DAgger) is better than or at least no worse than FTRL across discrete-action Atari tasks, with normalized scores and publication-quality figures.

## Current State

**v1.0 shipped** — All code infrastructure complete. Ready to run full benchmark on CC-server.

- FTRL algorithm implemented and tested (14 test variants passing)
- 7 Atari games verified end-to-end with expert policies from HuggingFace sb3
- Multi-GPU benchmark script with GNU parallel + tmux
- Analysis pipeline: dashboard, learning curves, aggregate IQM+CI figures
- ~3,975 LOC of new Python/shell across 4 phases

**Known tech debt:**
- Per-round Sacred logging gap — DAgger/FTRL log only final score, not per-round time series (learning curves degenerate)
- Expert scores not cached to pkl for offline analysis
- Server operational verification pending (tmux, GPU isolation)

## Requirements

### Validated

- ✓ BC (Behavioral Cloning) algorithm — existing
- ✓ DAgger (FTL) algorithm — existing
- ✓ Trajectory collection and rollout infrastructure — existing
- ✓ Sacred experiment management — existing
- ✓ HuggingFace integration for expert policies — existing
- ✓ Score normalization utilities in benchmarking/ — existing
- ✓ FTRL algorithm implementation — v1.0
- ✓ Atari game suite setup (7 games) — v1.0
- ✓ Expert policy acquisition from HuggingFace RL Zoo — v1.0
- ✓ Random policy baseline scores — v1.0
- ✓ Normalized evaluation pipeline — v1.0
- ✓ Multi-GPU parallel experiment runner — v1.0
- ✓ Experiment scripts with logging — v1.0
- ✓ Quick smoke-test configuration — v1.0
- ✓ Full benchmark configuration — v1.0
- ✓ Figure generation script — v1.0

### Active

(None — next milestone requirements TBD)

### Out of Scope

- Alt-FTRL and AdaFTRL variants — not needed for this study
- Linear policy (pretrained features) setting — end-to-end neural net only
- Continuous control (MuJoCo) experiments — Atari discrete actions only
- GAIL, AIRL, or other adversarial methods — not relevant to this comparison
- Hyperparameter search — use paper's recommended settings or reasonable defaults

## Context

- **Paper reference**: Lavington et al. (2022) "Improved Policy Optimization for Online Imitation Learning" (CoLLAs 2022)
- **FTRL formulation**: Memory-efficient reformulation from Proposition 4.1, Eq. 6
- **Codebase**: Fork of the `imitation` library (HumanCompatibleAI) with FTRL extension
- **Server**: CC-server accessible via `ssh CC-server`, 4 GPUs, isolated Python venv
- **Workflow**: Code edits locally → push to GitHub → sync to server → run in tmux with logging

## Constraints

- **Server setup**: Isolated Python environment (venv) on CC-server
- **GPU parallelism**: 4 GPUs — experiments distributed via GNU parallel
- **Remote execution**: All runs via tmux for SSH resilience
- **Logging**: Per-round metrics + Sacred FileStorageObserver per experiment
- **Expert fairness**: Same HuggingFace RL Zoo expert per game across all methods
- **Normalization**: Scores normalized between random (0.0) and expert (1.0)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| FTRL only (no Alt-FTRL/AdaFTRL) | Simplest comparison, paper shows FTRL is representative | ✓ Good |
| End-to-end neural net only | Practical setting users care about | ✓ Good |
| HuggingFace RL Zoo experts | Standard, reproducible, no extra compute | ✓ Good |
| Quick smoke test before full run | Catch bugs early, save GPU hours | ✓ Good — caught serialize.py bug |
| Isolated Python env on server | User requirement, cleaner than system Python | ✓ Good |
| eta_t = alpha/cumulative_sigma | Large alpha = weak proximal = FTL degeneracy | ✓ Good — verified by test |
| VecTransposeImage for obs spaces | SB3 experts expect channels-first (4,84,84) | ✓ Good — discovered during Phase 2 |
| rliable for aggregate metrics | NeurIPS 2021 standard for RL benchmarks | ✓ Good |

---
*Last updated: 2026-03-20 after v1.0 milestone*
