# Feature Landscape

**Domain:** Imitation learning empirical study — DAgger (FTL) vs FTRL vs BC on Atari benchmark
**Researched:** 2026-03-19
**Confidence:** HIGH (based on direct codebase inspection + verified literature)

---

## Table Stakes

Features that are required for the results to be credible. Missing any of these means
reviewers will reject or discount the study.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| 10+ Atari game suite | Single-game results are not generalizable; all IL papers on Atari use multi-game benchmarks | Med | Must cover diverse game types (score-based, sparse reward, fast/slow dynamics) |
| Consistent Atari preprocessing | ALE standard: grayscale, 84x84 resize, frame-stack 4, frame-skip 4 | Low | sb3 + AtariWrapper handles this; seals Atari envs enforce constant-length episodes |
| Same expert policy for all methods per game | Expert fairness: one source of truth | Low | HuggingFace RL Zoo (sb3/ppo-*NoFrameskip-v4) is the standard; existing infra uses `ppo-huggingface` loader |
| Normalization to [0, 1] relative to random/expert | Cross-game comparison requires same scale | Low | Formula is `(score - random_score) / (expert_score - random_score)`. Already implemented in `benchmarking/sacred_output_to_markdown_summary.py` |
| Random policy baseline scores per game | Needed for normalization denominator | Low | Existing infra fetches from `HumanCompatibleAI/random-{env}` datasets on HuggingFace — must verify these exist for Atari |
| Multiple seeds (minimum 5, ideally 10) | Single-seed results are not statistically valid; existing benchmark uses 10 seeds | Med | 10 seeds × 3 algorithms × 10+ games = 300+ runs; design for parallel execution across 4 GPUs |
| BC baseline | Required comparison point (offline IL lower bound) | Low | Already implemented; needs Atari-specific config/hyperparams |
| DAgger baseline | The "FTL" comparison target; primary vs-algorithm | Low | Already implemented; needs Atari-specific named configs |
| FTRL implementation | Core contribution; must match Lavington et al. Eq. 6 exactly | High | Not yet implemented; this is the study's novel contribution |
| Expert performance reference line | Shows what normalized score of 1.0 looks like on the figures | Low | Captured in `run.json` as `expert_stats.monitor_return_mean` |
| Smoke-test configuration | Catch bugs before burning 4-GPU hours | Low | 1-2 games, 3 DAgger rounds, 1 seed; fast enough to run in < 10 min |
| Sacred experiment tracking | Reproducibility, config logging, result storage | Low | Already in use; must write Atari named configs for Sacred |
| tmux-safe remote execution | Server sessions must survive disconnects | Low | Already a workflow constraint; wrap scripts in tmux with tee logging |

---

## Differentiators

Features that strengthen the study beyond the minimum credible bar. These set the work
apart from a simple "we ran DAgger on Atari" note.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| IQM aggregate metric with 95% CI | Modern best practice (Agarwal et al., NeurIPS 2021 Outstanding Paper); more robust than mean/median | Low | `rliable` library already integrated in `benchmarking/sacred_output_to_markdown_summary.py`; extend to Atari |
| Probability of Improvement metric | Directly answers "does FTRL beat DAgger?" statistically; already supported via `compute_probability_of_improvement.py` | Low | Plug Atari runs into existing script |
| Per-round learning curves | Shows convergence behavior across DAgger rounds, not just final performance; matches Figure 4 of Lavington et al. | Med | Requires logging normalized score at each round, not just at the end |
| Per-game score table | Lets readers see which game types FTRL helps/hurts | Low | CSV output already supported via `sacred_output_to_csv.py` |
| Stratified bootstrap CIs on curves | Uncertainty bands on performance curves; standard in recent IL papers | Med | `rliable` supports this; requires storing per-round scores |
| Game diversity across difficulty tiers | Credibility: include easy (Pong), medium (Breakout, BeamRider), hard (Montezuma's Revenge) | Low | Selection criterion, not a code feature |
| Full hyperparameter config in Sacred | Any reader can reproduce the exact run | Low | Sacred already logs this; ensure Atari configs are complete and committed |
| GPU assignment strategy | Uses all 4 GPUs efficiently; CUDA_VISIBLE_DEVICES per game group | Low | Shell script concern; one GPU per game, 4 games parallel |

---

## Anti-Features

Features to explicitly NOT build. Building these wastes time and dilutes the study's focus.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Hyperparameter search for FTRL | Out of scope per PROJECT.md; introduces confound (is FTRL better or just better-tuned?) | Use paper's recommended regularization weight or a single reasonable default; document the choice |
| Alt-FTRL and AdaFTRL variants | Out of scope per PROJECT.md; adds runs without changing the core FTL vs FTRL conclusion | Reference Lavington et al. for these variants; note they are future work |
| GAIL/AIRL comparison | Different problem class (reward learning vs IL); not relevant to FTL vs FTRL question | Exclude from all result tables |
| MuJoCo/continuous control environments | Out of scope per PROJECT.md | The existing seals-MuJoCo benchmark already exists for these |
| Custom/novel Atari preprocessing | Invites reviewer complaints about unfair comparison | Use standard ALE preprocessing via AtariWrapper + seals constant-episode wrapper exactly as done in prior work |
| Interactive/human-in-the-loop evaluation | Requires human raters; massively expensive | Use automated policy evaluation against the same environment |
| Learned reward / RLHF components | Orthogonal to the IL comparison question | Out of scope |
| Web UI or visualization dashboard | No research value for an empirical study | matplotlib figures saved to disk are sufficient |
| Hyperparameter sensitivity analysis | Multiplies compute by 5-10x | If space permits in a paper, add as appendix with 2-3 regularization values; not in MVP |

---

## Feature Dependencies

```
Random baseline scores (HuggingFace datasets) → Normalization pipeline
Expert policies (HuggingFace RL Zoo)          → DAgger/FTRL training + Normalization pipeline
Normalization pipeline                         → Per-game score table, learning curves, IQM
Sacred Atari named configs                     → BC / DAgger / FTRL runs
FTRL implementation                            → FTRL Sacred named configs → FTRL runs
Per-round score logging                        → Learning curves → Performance profiles
All 300+ runs complete                         → IQM aggregate metrics, Probability of Improvement
```

---

## Atari Game Selection

**Recommendation:** Use a 10-game suite that covers diverse difficulty tiers. All games
must have a pretrained PPO expert available via `sb3/ppo-*NoFrameskip-v4` on HuggingFace
(confirmed available: Pong, Breakout; the sb3 org lists 7 Atari games from the OpenAI
benchmark set). Cross-check against HumanCompatibleAI random-score datasets before
finalizing the list.

**Confirmed available on HuggingFace (sb3 org):**
- PongNoFrameskip-v4
- BreakoutNoFrameskip-v4
- SpaceInvadersNoFrameskip-v4
- (4 more from the OpenAI 7-game set — verify names before relying on them)

**Standard Atari benchmark games used in IL papers** (MEDIUM confidence — from multiple
papers including the DAgger-adjacent literature):
- Pong (easy, near-perfect experts achievable)
- Breakout (medium, visual complexity)
- BeamRider (medium, dense rewards)
- Seaquest (medium, requires resource management)
- Qbert (hard, IL methods often struggle here — good stress test)
- SpaceInvaders (medium, multiple enemies)
- Enduro (medium, driving)
- MsPacman (medium, navigation)
- Alien (medium, partial observability)
- Montezuma's Revenge (hard, sparse reward — may show BC collapse clearly)

**Practical constraint:** Only use games for which both (a) a PPO expert exists on
HuggingFace and (b) a HumanCompatibleAI random score dataset exists. Verify this before
finalizing the suite. If random-score datasets don't exist for Atari, computing random
baselines by running a random policy for N episodes is a necessary prerequisite step.

---

## Evaluation Protocol

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Seeds per configuration | 10 | Matches existing imitation library benchmark standard |
| Evaluation episodes per checkpoint | 100 | Standard in Atari RL literature |
| Normalization formula | `(score - random_score) / (expert_score - random_score)` | Already in codebase; gives [0,1] range where 0 = random, 1 = expert |
| Score used | `monitor_return_mean` over evaluation episodes | Matches existing Sacred result schema |
| Aggregate metric | IQM with 95% stratified bootstrap CI | rliable best practice; already integrated |
| Statistical comparison | Probability of Improvement (FTRL vs DAgger) | Already supported via `compute_probability_of_improvement.py` |
| DAgger rounds | Follow Lavington et al. 2022 (paper uses online rounds with dataset aggregation) | Ensures fair comparison with the reference paper's Figure 4 |

---

## MVP Recommendation

Prioritize these features for the initial working end-to-end pass:

1. **Atari named configs for Sacred** — BC and DAgger first; proves the pipeline works before FTRL is implemented
2. **Random baseline scores** — Verify HumanCompatibleAI datasets exist for chosen games; if not, add a random-rollout script immediately (blocks normalization)
3. **FTRL implementation** — Core contribution; implement, unit-test, add Sacred config
4. **Smoke test** — 2 games, 1 seed, 3 rounds; confirm Sacred output, normalization, and figure generation end-to-end
5. **Full benchmark run** — 10+ games, 10 seeds; dispatch across 4 GPUs via tmux
6. **Figure generation** — Normalized performance curves (per-round); final tables via CSV script

Defer:
- Probability of Improvement computation — run after full benchmark completes, adds 10 minutes not weeks
- Performance profiles (tau-curves) — nice to have, add if the paper submission target requires it
- Sensitivity analysis on FTRL regularization weight — post-hoc experiment if reviewers ask

---

## Sources

- Codebase inspection: `/Users/thangduong/Desktop/imitation/benchmarking/sacred_output_to_markdown_summary.py` — normalization formula, rliable IQM integration confirmed (HIGH confidence)
- Codebase inspection: `/Users/thangduong/Desktop/imitation/benchmarking/README.md` — 10-seed protocol, Sacred output schema confirmed (HIGH confidence)
- Codebase inspection: `/Users/thangduong/Desktop/imitation/src/imitation/policies/serialize.py` — HuggingFace expert loading confirmed (HIGH confidence)
- Agarwal et al. (2021) "Deep Reinforcement Learning at the Edge of the Statistical Precipice" (NeurIPS Outstanding Paper) — IQM, Performance Profiles, Probability of Improvement (HIGH confidence)
- rliable library: https://github.com/google-research/rliable — already a dependency in this repo (HIGH confidence)
- Lavington et al. (2022) "Improved Policy Optimization for Online Imitation Learning" (CoLLAs 2022): https://proceedings.mlr.press/v199/lavington22a.html — FTRL formulation, Atari experimental setup (MEDIUM confidence — full PDF not accessed; abstract only)
- HuggingFace sb3 org (confirmed): https://huggingface.co/sb3/ppo-PongNoFrameskip-v4, https://huggingface.co/sb3/ppo-BreakoutNoFrameskip-v4 — at least 7 Atari PPO models confirmed available (MEDIUM confidence)
- HumanCompatibleAI HuggingFace org: https://huggingface.co/HumanCompatibleAI — MuJoCo experts confirmed; Atari experts NOT confirmed present (LOW confidence for Atari coverage)
- WebSearch synthesis: standard Atari IL benchmark game selection, normalized score formula (MEDIUM confidence)
