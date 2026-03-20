---
phase: 2
slug: atari-setup-and-smoke-test
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-20
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x + verification scripts |
| **Config file** | setup.cfg (pytest section) |
| **Quick run command** | `pytest tests/atari/ -x -q` |
| **Full suite command** | `pytest tests/atari/ -v` |
| **Estimated runtime** | ~120 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick verification scripts
- **After every plan wave:** Run full test suite
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 1 | ENV-01, ENV-03 | integration | `python -c "from huggingface_sb3 import load_from_hub; load_from_hub('sb3/ppo-PongNoFrameskip-v4', 'ppo-PongNoFrameskip-v4.zip')"` | n/a | ⬜ pending |
| 2-01-02 | 01 | 1 | ENV-02, ENV-04 | integration | `python scripts/verify_atari_obs_spaces.py` | ❌ W0 | ⬜ pending |
| 2-01-03 | 01 | 1 | ENV-05 | integration | `python scripts/collect_random_baselines.py --verify` | ❌ W0 | ⬜ pending |
| 2-02-01 | 02 | 2 | INFRA-07 | script | `bash scripts/setup_server.sh --dry-run` | ❌ W0 | ⬜ pending |
| 2-02-02 | 02 | 2 | INFRA-05 | integration | `python experiments/atari_smoke_test.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `scripts/verify_atari_obs_spaces.py` — obs space verification for all 7 games
- [ ] `scripts/collect_random_baselines.py` — random baseline collection/caching
- [ ] `scripts/setup_server.sh` — server environment setup
- [ ] `experiments/atari_smoke_test.py` — smoke test entry point

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CC-server env works | INFRA-07 | Requires SSH access | SSH to CC-server, activate venv, run `python -c "import imitation"` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
