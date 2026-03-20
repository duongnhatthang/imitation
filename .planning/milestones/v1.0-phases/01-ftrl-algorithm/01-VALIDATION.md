---
phase: 1
slug: ftrl-algorithm
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-19
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | setup.cfg (pytest section) |
| **Quick run command** | `pytest tests/algorithms/test_ftrl.py -x -q` |
| **Full suite command** | `pytest tests/algorithms/test_ftrl.py -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/algorithms/test_ftrl.py -x -q`
- **After every plan wave:** Run `pytest tests/algorithms/test_ftrl.py -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | ALGO-01 | unit | `python -c "from imitation.algorithms.bc import BC; import inspect; sig = inspect.signature(BC.__init__); assert 'loss_calculator' in sig.parameters"` | n/a (code check) | ⬜ pending |
| 1-01-02 | 01 | 1 | ALGO-01..04 | unit | `python -c "from imitation.algorithms.ftrl import FTRLLossCalculator, FTRLDAggerTrainer"` | n/a (code check) | ⬜ pending |
| 1-02-01 | 02 | 2 | ALGO-01 | unit | `pytest tests/algorithms/test_ftrl.py::test_ftrl_instantiation -x` | ❌ W0 | ⬜ pending |
| 1-02-02 | 02 | 2 | ALGO-02 | unit | `pytest tests/algorithms/test_ftrl.py::test_proximal_term_centering -x` | ❌ W0 | ⬜ pending |
| 1-02-03 | 02 | 2 | ALGO-03 | unit | `pytest tests/algorithms/test_ftrl.py::test_linear_correction_sign -x` | ❌ W0 | ⬜ pending |
| 1-02-04 | 02 | 2 | ALGO-03 | unit | `pytest tests/algorithms/test_ftrl.py::test_sigma_grad_at_current_weights -x` | ❌ W0 | ⬜ pending |
| 1-02-05 | 02 | 2 | ALGO-04 | unit | `pytest tests/algorithms/test_ftrl.py::test_anchor_frozen_during_round -x` | ❌ W0 | ⬜ pending |
| 1-02-06 | 02 | 2 | ALGO-05 | integration | `pytest tests/algorithms/test_ftrl.py::test_ftl_degeneracy -x` | ❌ W0 | ⬜ pending |
| 1-02-07 | 02 | 2 | ALGO-06 | integration | `pytest tests/algorithms/test_ftrl.py::test_cartpole_smoke -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/algorithms/test_ftrl.py` — stubs for ALGO-01 through ALGO-06
- [ ] Existing `tests/algorithms/conftest.py` — shared fixtures already available

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
