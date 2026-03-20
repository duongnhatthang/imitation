# Roadmap: DAgger vs FTRL Empirical Study

## Overview

This study compares FTL (DAgger), FTRL, and BC on 7 Atari games to produce normalized performance curves following Lavington et al. 2022. The existing imitation library provides the scaffold; the work is four sequenced phases: implement the ~90-line FTRL algorithm, verify Atari game setup and expert availability, wire up the multi-GPU Sacred experiment runner, then execute the full benchmark and produce publication-quality figures.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: FTRL Algorithm** - Implement and verify FTRLTrainer + FTRLLossCalculator against Lavington et al. Eq. 6 (completed 2026-03-20)
- [x] **Phase 2: Atari Setup and Smoke Test** - Verify 7-game expert coverage, random baselines, and run a smoke test on real Atari before paying full compute costs (completed 2026-03-20)
- [x] **Phase 3: Experiment Infrastructure** - Sacred named configs, multi-GPU GNU parallel runner, per-experiment logging and isolation (completed 2026-03-20)
- [ ] **Phase 4: Full Run and Analysis** - Execute 300+ runs, collect results, produce normalized learning curves and aggregate figures

## Phase Details

### Phase 1: FTRL Algorithm
**Goal**: A correct, tested FTRL implementation is available that can be dropped into any experiment script
**Depends on**: Nothing (first phase)
**Requirements**: ALGO-01, ALGO-02, ALGO-03, ALGO-04, ALGO-05, ALGO-06
**Success Criteria** (what must be TRUE):
  1. `FTRLDAggerTrainer` can be imported and instantiated without modifying any existing file other than the two-line BC constructor change
  2. FTRL loss per Eq. 6: proximal term `(1/(2η_t))||w - w_t||^2` centered on current weights, plus linear correction `-⟨w, Σ ∇l_i(w_t)⟩` using gradients of past losses at current weights
  3. Setting alpha to a very large value produces training curves statistically indistinguishable from plain DAgger (FTL degeneracy test passes)
  4. Running FTRL on CartPole for 5 rounds completes without error and achieves reward matching or exceeding BC
**Plans**: 2 plans

Plans:
- [ ] 01-01-PLAN.md — BC loss_calculator injection + FTRLLossCalculator and FTRLDAggerTrainer implementation
- [ ] 01-02-PLAN.md — Unit tests (proximal centering, linear correction, anchor freezing) + FTL degeneracy + CartPole smoke test

### Phase 2: Atari Setup and Smoke Test
**Goal**: The 7-game Atari suite is verified end-to-end — experts load, observation spaces match, random baselines exist, and a smoke test on a real Atari game passes all three methods
**Depends on**: Phase 1
**Requirements**: ENV-01, ENV-02, ENV-03, ENV-04, ENV-05, INFRA-05, INFRA-07
**Success Criteria** (what must be TRUE):
  1. All 7 seals Atari games (Pong, Breakout, BeamRider, Enduro, Qbert, Seaquest, SpaceInvaders) have confirmed HuggingFace `sb3` org expert policies that load without error
  2. `assert venv.observation_space == expert_policy.observation_space` passes for all 7 games with `make_atari_env`-based construction
  3. Random baseline scores for all 7 games are available (from HumanCompatibleAI datasets or locally collected) and cached to disk
  4. Smoke test on 2 Atari games with 1 seed and 3 DAgger rounds runs all three methods (BC, DAgger, FTRL) to completion and produces Sacred output with normalized scores
  5. CC-server Python venv is set up with all dependencies installed and the repo synced
**Plans**: 2 plans

Plans:
- [ ] 02-01-PLAN.md — Atari helpers (env creation, expert loading, random baselines), tests for ENV-01 through ENV-05, server setup script
- [ ] 02-02-PLAN.md — Sacred Atari named configs and smoke test running BC, DAgger, FTRL on Pong and Breakout

### Phase 3: Experiment Infrastructure
**Goal**: A single command launches all 84+ experiment combinations (3 algorithms x 7 games x 4+ seeds) across 4 GPUs with isolated logging and no Sacred run-ID collisions
**Depends on**: Phase 2
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-06
**Success Criteria** (what must be TRUE):
  1. `run_atari_benchmark.sh` distributes jobs across 4 GPUs via GNU parallel with explicit `CUDA_VISIBLE_DEVICES` assignment and no process claims the wrong GPU
  2. Each (algorithm, game, seed) combination writes to its own Sacred observer directory; concurrent runs produce no ID collisions
  3. Each experiment produces a per-round log file showing reward, loss, eta_t, norm(g_t), and round number
  4. Full benchmark config (7 games, 3+ seeds, 20+ rounds) launches in a tmux session and survives SSH disconnect
**Plans**: 2 plans

Plans:
- [ ] 03-01-PLAN.md — Sacred single-run entry point + FTRL per-round metric logging
- [ ] 03-02-PLAN.md — GNU parallel multi-GPU orchestration script with tmux and full benchmark config

### Phase 4: Full Run and Analysis
**Goal**: Publication-quality figures comparing FTL, FTRL, BC, and Expert are generated from the completed benchmark
**Depends on**: Phase 3
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07
**Success Criteria** (what must be TRUE):
  1. Normalized scores `(agent - random) / (expert - random)` are computed for every completed (algorithm, game, seed) run
  2. Per-game learning curves (normalized return vs environment interactions) can be generated at any point mid-run from partial Sacred output
  3. A completion dashboard shows which (algorithm, game, seed) combinations are done, running, or pending
  4. Aggregate figure shows mean and IQM with 95% CI across all 7 games for each method, matching Lavington et al. Figure 4 style
  5. All figures use consistent colors and line styles across methods and are saved as publication-quality PDF/PNG
**Plans**: 2 plans

Plans:
- [ ] 04-01-PLAN.md — Plot config, Sacred output reader, completion dashboard, and test scaffolding
- [ ] 04-02-PLAN.md — Per-game learning curves, aggregate metrics (mean + IQM with 95% CI), and publication figures

## Progress

**Execution Order:**
Phases execute in order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. FTRL Algorithm | 2/2 | Complete   | 2026-03-20 |
| 2. Atari Setup and Smoke Test | 2/2 | Complete   | 2026-03-20 |
| 3. Experiment Infrastructure | 2/2 | Complete   | 2026-03-20 |
| 4. Full Run and Analysis | 1/2 | In Progress|  |
