---
phase: 3
slug: experiment-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-20
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x + shell script verification |
| **Config file** | setup.cfg (pytest section) |
| **Quick run command** | `pytest tests/algorithms/test_experiment_infra.py -x -q` |
| **Full suite command** | `pytest tests/algorithms/test_experiment_infra.py -v` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick verification
- **After every plan wave:** Run full test suite
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 1 | INFRA-01 | unit | `python -c "from experiments.run_atari_experiment import main"` | ❌ W0 | ⬜ pending |
| 3-01-02 | 01 | 1 | INFRA-03 | unit | `grep -c "log_scalar" src/imitation/algorithms/ftrl.py` | n/a | ⬜ pending |
| 3-02-01 | 02 | 2 | INFRA-02, INFRA-04 | script | `bash experiments/run_atari_benchmark.sh --dry-run` | ❌ W0 | ⬜ pending |
| 3-02-02 | 02 | 2 | INFRA-06 | script | `grep -c "total_timesteps" experiments/run_atari_benchmark.sh` | n/a | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `experiments/run_atari_experiment.py` — single-run Sacred entry point
- [ ] `experiments/run_atari_benchmark.sh` — GNU parallel orchestrator

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| tmux session survives SSH disconnect | INFRA-06 | Requires SSH access | SSH to CC-server, launch benchmark, disconnect, reconnect, verify tmux session |
| GPU isolation across 4 GPUs | INFRA-02 | Requires multi-GPU server | Check nvidia-smi during benchmark run |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
