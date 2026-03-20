# Retrospective

## Milestone: v1.0 — DAgger vs FTRL Benchmark

**Shipped:** 2026-03-20
**Phases:** 4 | **Plans:** 8

### What Was Built
- FTRL algorithm (Lavington et al. Eq. 6) with proximal + linear correction terms
- 7-game Atari verification suite with HuggingFace sb3 experts
- Multi-GPU benchmark orchestration (GNU parallel + tmux + Sacred)
- Analysis pipeline with dashboard, learning curves, aggregate IQM+CI figures

### What Worked
- Infrastructure-first approach: smoke test on CartPole/Pong caught bugs early
- Wave-based execution: clean dependency chains between plans
- Auto-fixing deviations (serialize.py bug, VecTransposeImage, rliable API)

### What Was Inefficient
- Per-round Sacred logging not wired end-to-end — learning curves will show flat lines for DAgger/FTRL until callbacks are added
- Phase 2 smoke test ran CPU-only (~6 hours) — would be ~30min on GPU

### Patterns Established
- `atari_helpers.py` as shared utility module across all phases
- Sacred FileStorageObserver with `{algo}/{game}/{seed}` directory isolation
- `plot_config.py` for consistent colors/styles across all figures

### Key Lessons
- Always verify the full data pipeline end-to-end, not just individual components
- Per-round evaluation callbacks must be designed upfront, not bolted on after training loops are finalized

### Cost Observations
- Model mix: ~30% opus (planning), ~60% sonnet (execution/research/verification), ~10% orchestration
- 4 phases completed in a single autonomous session
- Notable: researcher agents were efficient at finding existing codebase patterns

---

## Cross-Milestone Trends

| Metric | v1.0 |
|--------|------|
| Phases | 4 |
| Plans | 8 |
| Requirements | 25 |
| Satisfied | 24 (96%) |
| Tech Debt Items | 6 |
