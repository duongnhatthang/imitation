---
phase: 01-ftrl-algorithm
plan: "01"
subsystem: algorithm
tags: [ftrl, dagger, imitation-learning, pytorch, bc, loss-calculator]

# Dependency graph
requires: []
provides:
  - FTRLLossCalculator: frozen dataclass computing proximal + linear correction terms per Eq. 6
  - FTRLDAggerTrainer: SimpleDAggerTrainer subclass with per-round FTRL loss injection
  - BC.loss_calculator: optional injection point allowing custom loss calculators
affects:
  - 01-ftrl-algorithm (plan 02: tests)
  - 02-evaluation (needs FTRLDAggerTrainer importable and functional)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Loss calculator injection: BC accepts optional loss_calculator param; FTRLLossCalculator matches BehaviorCloningLossCalculator callable protocol"
    - "Frozen dataclass for stateless callables: FTRLLossCalculator uses @dataclasses.dataclass(frozen=True) with detached tensor fields"
    - "Pre-round FTRL setup: extend_and_update override calls _try_load_demos first, then snapshots anchor and computes sigma_grad before bc_trainer.train()"

key-files:
  created:
    - src/imitation/algorithms/ftrl.py
  modified:
    - src/imitation/algorithms/bc.py

key-decisions:
  - "eta_t = alpha / cumulative_sigma (not 1/(alpha*cumulative_sigma)): large alpha = large eta_t = weak proximal = FTL degeneracy, matching ALGO-05 semantics"
  - "sigma_i = 1.0 constant per round (eta_t = alpha/t): simplest implementation matching practical FTRL; documented as Phase 3+ hyperparameter"
  - "FTRLLossCalculator returns BCTrainingMetrics via dataclasses.replace rather than subclassing: avoids touching BCLogger code"
  - "sigma_grad computed by iterating bc_trainer._demo_data_loader with backward() only (no optimizer.step()): pure gradient accumulation at w_t"
  - "policy.zero_grad() called after sigma_grad extraction: prevents stale gradients from contaminating first batch of BC.train()"

patterns-established:
  - "Pattern: BC loss calculator injection via self.bc_trainer.loss_calculator = FTRLLossCalculator(...) at round start"
  - "Pattern: anchor_params and sigma_grad must be detach().clone() tensors (no graph connection to inner training loop)"

requirements-completed: [ALGO-01, ALGO-02, ALGO-03, ALGO-04]

# Metrics
duration: 4min
completed: 2026-03-20
---

# Phase 01 Plan 01: FTRL Algorithm Implementation Summary

**FTRLLossCalculator (proximal + linear correction per Eq. 6) injected into BC via new loss_calculator param; FTRLDAggerTrainer overrides extend_and_update to snapshot anchor and accumulate sigma_grad before each round**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-20T04:43:48Z
- **Completed:** 2026-03-20T04:47:51Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- BC constructor now accepts optional `loss_calculator` parameter with full backward compatibility
- `FTRLLossCalculator` implements Eq. 6 loss with correctly-centered proximal term `(1/(2*eta_t))||w - w_t||^2` and linear correction `-<w, sigma_grad>` (negative sign)
- `FTRLDAggerTrainer` subclasses `SimpleDAggerTrainer`, overrides `extend_and_update` to do pre-round setup: snapshot anchor, accumulate sigma_grad over full dataset, inject new `FTRLLossCalculator`, then train

## Task Commits

Each task was committed atomically:

1. **Task 1: Add loss_calculator injection to BC constructor** - `591c71c` (feat)
2. **Task 2: Implement FTRLLossCalculator and FTRLDAggerTrainer** - `603574e` (feat)

## Files Created/Modified

- `src/imitation/algorithms/ftrl.py` - New file: FTRLLossCalculator (frozen dataclass) and FTRLDAggerTrainer (SimpleDAggerTrainer subclass)
- `src/imitation/algorithms/bc.py` - Added optional `loss_calculator` param to `BC.__init__`; uses `loss_calculator or BehaviorCloningLossCalculator(...)`

## Decisions Made

- **eta_t parameterization:** Used `eta_t = alpha / cumulative_sigma` so large alpha = weak proximal = FTL degeneracy. This matches ALGO-05: "FTRL degenerates to FTL when alpha -> infinity." The research note (Pitfall 2) confirmed this direction.
- **sigma_i = 1.0 constant:** Simplest approach per research recommendation. eta_t = alpha/t. Documented as Phase 3+ hyperparameter.
- **No super() call in extend_and_update:** Called `_try_load_demos()` directly then `bc_trainer.train()` to avoid double demo-loading (parent would call `_try_load_demos()` again). Also required to insert sigma_grad computation between demo loading and training.
- **FTRLLossCalculator returns BCTrainingMetrics via dataclasses.replace:** Avoids subclassing BCTrainingMetrics and touching BCLogger. Extra terms (proximal, linear correction) absorbed into the combined `loss` field.

## Deviations from Plan

None - plan executed exactly as written. Package not installed in base environment was handled by running `pip install -e .` (Rule 3 - blocking issue during verification; fixed immediately, no code deviation).

## Issues Encountered

- `imitation` package not installed in base conda environment. Fixed with `pip install -e /Users/thangduong/Desktop/imitation`. All verifications then passed cleanly.

## Next Phase Readiness

- `FTRLDAggerTrainer` and `FTRLLossCalculator` are importable and structurally correct.
- Plan 01-02 (tests) can now build on these classes with pytest fixtures from `tests/algorithms/conftest.py`.
- Blocker resolved: `BC.__init__` now accepts `loss_calculator` (confirmed in plan 01-01).

---
*Phase: 01-ftrl-algorithm*
*Completed: 2026-03-20*
