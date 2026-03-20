---
phase: 4
slug: full-run-and-analysis
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-20
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x + inline verification |
| **Config file** | setup.cfg (pytest section) |
| **Quick run command** | `pytest tests/algorithms/test_analysis.py -x -q` |
| **Full suite command** | `pytest tests/algorithms/test_analysis.py -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick verification
- **After every plan wave:** Run full test suite
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 4-01-01 | 01 | 1 | EVAL-01, EVAL-04 | inline | `python -c "import ast; ast.parse(open('experiments/analyze_results.py').read()); print('PASS')"` | pending |
| 4-01-02 | 01 | 1 | EVAL-05 | inline | `python -c "import ast; ast.parse(open('experiments/completion_dashboard.py').read()); print('PASS')"` | pending |
| 4-02-01 | 02 | 2 | EVAL-02, EVAL-06, EVAL-07 | unit | `pytest tests/algorithms/test_analysis.py -x -q` | pending |
| 4-02-02 | 02 | 2 | EVAL-03 | inline | `grep -q "IQM\|interquartile_mean" experiments/analyze_results.py && echo "PASS"` | pending |

*Status: pending · green · red · flaky*

---

## Wave 0 Requirements

No Wave 0 test scaffolding needed — all verification commands are self-contained.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Figure visual quality matches Lavington et al. | EVAL-06 | Visual comparison | Open generated PDF, compare layout/style with paper Figure 4 |
| Incremental generation from partial runs | EVAL-04 | Needs partial Sacred output | Run analysis with incomplete benchmark data, verify figures generate |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify — self-contained inline commands
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] No Wave 0 test file dependencies
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** ready
