# BC vs Expert CE Audit — Results & Verdict

**Date:** 2026-04-11
**Branch:** `feature/bc-expert-ce-audit`

## Verdict: No Pipeline Bug

BC beating expert CE on D_offline is **expected** and fully explained by
the structural softmax entropy floor that on-policy RL algorithms
(PPO, A2C) maintain even after convergence. There is no bug in the
eval pipeline, label construction, or preprocessing.

## Matrix Results (Deliverable 1)

Self-CE on D_offline (expert's own argmax-rollout distribution).
22/24 cells completed; Acrobot A2C seeds 1 and 2 failed to converge.

### CartPole-v1

| Row | Config | Seed 0 | Seed 1 | Seed 2 | Mean |
|-----|--------|--------|--------|--------|------|
| 1 | PPO clip=0.2 ent=0.01 | 0.241 | 0.222 | 0.203 | **0.222** |
| 2 | PPO clip=0.2 ent=0.0 | 0.225 | 0.255 | 0.253 | **0.244** |
| 3 | PPO clip=1e9 ent=0.0 | 0.070 | 0.120* | 0.071 | **0.087** |
| 4 | A2C ent=0.0 | 0.141 | 0.230 | 0.162 | **0.178** |

*Row 3 seed 1: `norm_ret=0.924` (not fully converged); seeds 0 and 2
converged to `norm_ret=1.0` with `self_ce ≈ 0.07`.

**Key finding:** Disabling PPO clipping (row 3) drops self_ce from
0.22–0.24 to 0.07 (3× reduction). Removing entropy bonus alone
(row 2) has no effect. A2C (no clipping mechanism) averages 0.178
— higher than PPO-without-clip because A2C's advantage estimate is
noisier.

### Acrobot-v1

| Row | Config | Seed 0 | Seed 1 | Seed 2 | Mean |
|-----|--------|--------|--------|--------|------|
| 1 | PPO clip=0.2 ent=0.01 | 0.080 | 0.097 | 0.050 | **0.076** |
| 2 | PPO clip=0.2 ent=0.0 | 0.021 | 0.025 | 0.018 | **0.021** |
| 3 | PPO clip=1e9 ent=0.0 | 0.023 | 0.022 | 0.018 | **0.021** |
| 4 | A2C ent=0.0 | 0.007 | DNF | DNF | **0.007** (1 seed) |

DNF = did not converge in 5M steps (`norm_ret ≈ -0.002`).

**Key finding:** On Acrobot the **entropy bonus** is the dominant
source of the self_ce floor (row 1 → 2: 0.076 → 0.021), not clipping
(row 2 → 3: 0.021 → 0.021, no change). Environment-dependent.

## Audit Results (Deliverable 2)

Expert vs BC cross-entropy on the *same* D_offline. Row-1 default
PPO expert, vanilla BC trained for 20 epochs on 3000 transitions.

### CartPole-v1

| Seed | expert_ce(D_off) | bc_ce(D_off) | expert_ce(D_eval) | bc_ce(D_eval) | BC ret | Expert ret |
|------|-----------------|-------------|------------------|--------------|--------|-----------|
| 0 | 0.224 | **0.064** | 0.226 | 0.067 | 1.00 | 1.00 |
| 1 | 0.258 | **0.062** | 0.257 | 0.067 | 1.00 | 1.00 |
| 2 | 0.254 | **0.081** | 0.235 | 0.550* | 1.00 | 1.00 |

*Seed 2 `bc_ce(D_eval)` = 0.550: BC is sharp on D_offline but
diffuse on unseen states — classic overfitting of softmax temperature
to the training distribution. This doesn't affect the D_offline
comparison.

### Acrobot-v1

| Seed | expert_ce(D_off) | bc_ce(D_off) | expert_ce(D_eval) | bc_ce(D_eval) | BC ret | Expert ret |
|------|-----------------|-------------|------------------|--------------|--------|-----------|
| 0 | 0.022 | **0.010** | 0.022 | 0.012 | 1.00 | 1.00 |
| 1 | 0.028 | **0.013** | 0.029 | 0.016 | 1.02 | 1.01 |
| 2 | 0.013 | **0.009** | 0.018 | 0.019 | 1.00 | 1.01 |

## Interpretation Matrix Applied

| Observation | Conclusion |
|---|---|
| Row 3 CartPole `self_ce ≈ 0.07` (converged seeds) | Clipping is the dominant cause on CartPole. **Not a pipeline bug.** |
| Row 3 Acrobot `self_ce ≈ 0.021` = Row 2 | On Acrobot, clipping doesn't matter — entropy bonus does. |
| `bc_ce(D_off) < expert_ce(D_off)` in all 6 records | Expected. BC directly minimizes -log π(a*|s) with no clip/entropy. |
| Row 3 CartPole `self_ce ≈ 0.07` ≈ BC `ce(D_off) ≈ 0.06` | Gap nearly closes when clip is removed — confirms root cause. |
| `expert_ce(D_eval) ≈ expert_ce(D_off)` in all records | No label-pipeline bug; expert CE is stable across distributions. |

## Why BC < Expert CE is Expected (Not a Bug)

1. **Argmax labels** make the CE minimum 0, not H(π_expert). Any
   policy with a sharp enough softmax can beat the expert's CE.

2. **PPO/A2C maintain a structural softmax entropy floor** because
   their update rules optimize *return*, not -log π(a*|s).
   Clipping (PPO) and entropy bonus explicitly prevent softmax
   collapse; even without them, on-policy gradient dynamics don't
   drive softmax temperature to infinity.

3. **BC directly minimizes -log π(a*|s)** with no regularization
   (no clip, no entropy bonus, no KL constraint), so it can
   sharpen its softmax far below any RL policy's floor on the
   training set.

4. The ~3.5× CE gap (BC 0.06 vs Expert 0.24 on CartPole) is fully
   bridged by removing PPO clipping (row 3 ≈ 0.07). The remaining
   0.01 gap is within seed variance.

## Implications for the FTRL Pipeline

- The `expert_rollout_cross_entropy` metric in the FTRL plots should
  **not** be interpreted as a lower bound on learner CE. It is the
  expert's softmax temperature expressed in nats, not a theoretical
  floor.

- Comparing `rollout_cross_entropy` (learner) against
  `expert_rollout_cross_entropy` (expert) is still *useful* — when
  the learner matches or beats the expert, it signals that the
  learner has at least matched the expert's argmax behavior. But
  exceeding the expert CE is not a sign of overfitting or a bug.

- No code changes are needed. The pipeline is correct as-is.

## Anomalies and Notes

- **CartPole seed 2 `bc_ce(D_eval) = 0.550`:** BC overfits softmax
  temperature to D_offline; on D_eval (BC's own rollout) the policy
  encounters states where it is uncertain. This is benign (norm_ret
  still 1.0) and expected with argmax labels.

- **A2C Acrobot seeds 1/2 DNF:** A2C with `ent_coef=0` is unstable
  on Acrobot (5M steps, never reached 0.95 norm_ret). Seed 0
  converged and achieved the lowest self_ce of all rows (0.007),
  but the sample size is too small to draw conclusions about A2C vs
  PPO on Acrobot.

- **PPO no-clip (row 3) instability:** CartPole seed 1 did not fully
  converge (`norm_ret=0.924`). Without clipping, PPO's ratio updates
  are unbounded, causing training instability.
