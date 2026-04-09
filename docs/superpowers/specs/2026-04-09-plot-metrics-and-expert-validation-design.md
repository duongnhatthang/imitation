# Plot Metrics Enhancement & Expert Quality Validation

**Date**: 2026-04-09
**Status**: Approved
**Goal**: Add normalized expected return, on-policy disagreement rate, IQM, and expert quality validation to the experiment pipeline.

## Background

Current plots show per-round cross-entropy, cumulative loss, and cumulative regret. The prof wants:
- Normalized expected return (0=random, 1=expert) to show actual policy performance
- On-policy disagreement rate with expert (from DAgger paper, Ross et al. 2011)
- IQM (Agarwal et al. 2021) instead of mean +/- std for robustness to outliers
- Expert quality validation to catch degenerate experts early (MountainCar incident)

## New Metrics

### Normalized Expected Return

```
normalized_return = (learner_return - random_return) / (expert_return - random_return)
```

- 0 = random policy, 1 = expert policy
- Computed by rolling out learner policy for N episodes, averaging undiscounted returns
- Evaluated at intervals (not every round) via `--eval-interval` (default 5)
- Always evaluated at round 1 and final round

### On-Policy Disagreement Rate

From Ross et al. (2011):

```
disagreement_rate = (1/M) * sum_{t=1}^{M} I(pi(s_t) != pi*(s_t))
```

- States visited by learner's own policy (on-policy)
- At each state, query both learner and expert for deterministic action
- Count fraction of disagreements
- Evaluated at same intervals as normalized return (piggybacks on same rollouts)

### IQM (Interquartile Mean)

From Agarwal et al. (2021) "Deep RL at the Edge of the Statistical Precipice":
- Drop bottom 25% and top 25% of seeds, average middle 50%
- 95% stratified bootstrap confidence intervals
- Use `rliable` library
- Applied to ALL plotted metrics across all seeds

## Plot Layout (4 subplots per env)

1. **Per-round imitation loss** (log scale y-axis) — renamed from "cross-entropy"
2. **Normalized expected return** (0-1 scale)
3. **On-policy disagreement rate** (0-1 scale)
4. **Cumulative regret** (vs expert)

All show IQM + 95% bootstrap CI. Expert baseline shown where applicable.

## Expert Quality Validation

### Reference Baselines (hardcoded)

Known optimal/expert and random scores per environment:

**Classical MDPs:**

| Environment | random_score | expert_score | Source |
|---|---|---|---|
| CartPole-v1 | 22.0 | 500.0 | Known max |
| FrozenLake-v1 | 0.015 | 1.0 | Known optimal |
| CliffWalking-v0 | -56957.0 | -13.0 | Known optimal |
| Acrobot-v1 | -499.0 | -85.0 | PPO typical |
| MountainCar-v0 | -200.0 | -110.0 | Gymnasium solved |
| Taxi-v3 | -763.0 | 7.9 | Q-learning optimal |
| Blackjack-v1 | -0.40 | -0.06 | Basic strategy |
| LunarLander-v2 | -176.0 | 250.0 | PPO typical |

**Atari (NoFrameskip-v4):**

| Environment | random_score | expert_score (PPO) | Source |
|---|---|---|---|
| Pong | -20.7 | 20.5 | DQN Zoo / CleanRL |
| Breakout | 1.7 | 405.7 | DQN Zoo / CleanRL |
| SpaceInvaders | 148.0 | 1019.8 | DQN Zoo / CleanRL |
| BeamRider | 363.9 | 2835.7 | DQN Zoo / CleanRL |
| Qbert | 163.9 | 15228.3 | DQN Zoo / CleanRL |
| MsPacman | 307.3 | 2152.8 | DQN Zoo / CleanRL |
| Enduro | 0.0 | 986.7 | DQN Zoo / CleanRL |
| Seaquest | 68.4 | 1518.3 | DQN Zoo / CleanRL |
| Freeway | 0.0 | 33.0 | DQN Zoo / CleanRL |
| Atlantis | 12850.0 | 2036749.0 | DQN Zoo / CleanRL |
| DemonAttack | 152.1 | 13788.4 | DQN Zoo / CleanRL |
| CrazyClimber | 10780.5 | 119344.7 | DQN Zoo / CleanRL |
| Asterix | 210.0 | 3738.5 | DQN Zoo / CleanRL |
| Frostbite | 65.2 | 933.6 | DQN Zoo / CleanRL |
| Kangaroo | 52.0 | 5325.3 | DQN Zoo / CleanRL |
| BankHeist | 14.2 | 1213.5 | DQN Zoo / CleanRL |

### Validation Rules

- **Test suite (fail hard)**: After training/loading expert, evaluate and compare to reference. Fail if expert achieves < 80% of `(reference_expert - reference_random)` range above random.
- **Experiment runner (loud warning)**: Same check, but logs WARNING and continues.

### Baseline Computation at Experiment Time

When caching an expert, also compute and cache:
- `expert_return`: mean return over 20 episodes with expert policy
- `random_return`: mean return over 100 episodes with uniform random policy

Stored in `{cache_dir}/{env_name}/baselines.json`.

## Evaluation at Intervals

New CLI flag: `--eval-interval N` (default 5).

At rounds `{1, N, 2N, ..., final}`:
1. Roll out learner policy for 10 episodes
2. At each step, also query expert for action → compute disagreement rate
3. Compute mean undiscounted return → normalize with cached baselines
4. Store `normalized_return` and `disagreement_rate` in per_round metrics

Rounds without evaluation have `null` for these fields in JSON.

## JSON Results Extension

```json
{
  "baselines": {
    "expert_return": 500.0,
    "random_return": 22.1
  },
  "per_round": [
    {
      "round": 1,
      "cross_entropy": 0.34,
      "normalized_return": 0.15,
      "disagreement_rate": 0.42,
      ...
    },
    {
      "round": 2,
      "cross_entropy": 0.28,
      "normalized_return": null,
      "disagreement_rate": null,
      ...
    }
  ]
}
```

## Dependencies

- `rliable` — for IQM + stratified bootstrap CI
- All existing dependencies unchanged

## Files

### New: `src/imitation/experiments/ftrl/env_baselines.py`
- `REFERENCE_BASELINES` dict: hardcoded reference scores per env
- `validate_expert_quality(env_name, measured_return)` — returns bool + message
- `compute_baselines(expert_policy, venv, rng)` — evaluates expert + random, returns dict
- `load_or_compute_baselines(env_name, venv, expert_policy, cache_dir, rng)` — cache-aware

### Modified: `src/imitation/experiments/ftrl/run_experiment.py`
- Add `--eval-interval` CLI flag
- At eval rounds: roll out learner, compute disagreement + normalized return
- Store baselines in result JSON
- Add expert quality warning check

### Modified: `src/imitation/experiments/ftrl/plot_results.py`
- 4-subplot layout
- IQM + bootstrap CI via `rliable`
- Rename "cross-entropy" to "imitation loss"
- Log scale for imitation loss subplot
- Handle `null` eval metrics (interpolate or plot only at eval points)

### Modified: `src/imitation/experiments/ftrl/experts.py`
- After training/loading expert, compute and cache baselines

### New tests:
- Expert quality validation tests (fail hard if expert is degenerate)
- Smoke tests for new classical MDPs (LunarLander, Taxi) — from original plan Task 7
- Atari integration test — from original plan Task 9

## Compatibility

All changes are backward-compatible:
- Old JSON results without `baselines`/`normalized_return`/`disagreement_rate` still plot fine (those subplots just show no data)
- Classical MDPs and Atari both supported via the same baselines infrastructure
