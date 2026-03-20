---
phase: 3
slug: experiment-infrastructure
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-20
---

# Phase 3 -- Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x + inline Python verification + shell syntax check |
| **Config file** | setup.cfg (pytest section) |
| **Quick run command** | `pytest tests/algorithms/test_ftrl.py -x -q` |
| **Full suite command** | `pytest tests/algorithms/test_ftrl.py -v && bash -n experiments/run_atari_benchmark.sh` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick verification
- **After every plan wave:** Run full test suite
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 3-01-01 | 01 | 1 | INFRA-03 | inline | `python -c "import inspect; from imitation.algorithms.ftrl import FTRLDAggerTrainer; src = inspect.getsource(FTRLDAggerTrainer.extend_and_update); assert 'ftrl/eta_t' in src and 'ftrl/norm_g' in src and 'ftrl/round' in src; print('PASS')"` | pending |
| 3-01-02 | 01 | 1 | INFRA-01, INFRA-04 | inline | `python -c "import ast; ast.parse(open('experiments/run_atari_experiment.py').read()); print('PASS: valid Python')" && grep -q "FileStorageObserver" experiments/run_atari_experiment.py && grep -q "parse_known_args" experiments/run_atari_experiment.py && grep -q "step=0" experiments/run_atari_experiment.py && echo "PASS: all checks"` | pending |
| 3-02-01 | 02 | 2 | INFRA-02, INFRA-04 | script | `bash -n experiments/run_atari_benchmark.sh && grep -q "CUDA_VISIBLE_DEVICES.*{%}" experiments/run_atari_benchmark.sh && grep -q "tmux" experiments/run_atari_benchmark.sh && echo "PASS"` | pending |
| 3-02-02 | 02 | 2 | INFRA-06 | script | `grep -c ":::" experiments/run_atari_benchmark.sh | xargs test 3 -eq && echo "PASS: 3 cartesian product axes"` | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

No Wave 0 test scaffolding needed -- all verification commands are self-contained inline checks (AST parse, grep, bash -n) that do not depend on separate test files.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| tmux session survives SSH disconnect | INFRA-02 | Requires SSH access | SSH to CC-server, launch benchmark, disconnect, reconnect, verify tmux session |
| GPU isolation across 4 GPUs | INFRA-02 | Requires multi-GPU server | Check nvidia-smi during benchmark run |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify -- self-contained inline commands
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] No Wave 0 test file dependencies
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** ready
