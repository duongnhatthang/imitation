# Why BC-iid beats the old BC-DAgger (meeting item 2)

Runs compared: `20260910-bc-iid-v1` (new) vs `legacy-sampling-v2` (old, Sep 9).
Code delta: `74ef4e2` → `2e5bac4`, of which the load-bearing commit is
`74c65b4 feat(ftrl): replace BC-DAgger with expert-rollout BC-iid`.

## Short answer

The premise "we only changed the m>1 sampling, so nothing should move" is right
about FTL/FTRL and wrong about the BC baseline. `bc_dagger` was not re-sampled ,
it was **deleted and replaced by a different algorithm**. The two learners draw
their labels from a different number of expert episodes:

| | old `bc_dagger` | new `bc_iid` |
|---|---|---|
| data source | one **fixed pre-collected pool**, shared with fixed BC | a **fresh expert episode collected every round** |
| pool size | `⌈N/H⌉` episodes (N = total label budget) | `T` episodes (one per round) |
| round-t dataset | first `t` transitions of that pool | 1 uniform state from each of `t` distinct episodes |
| effective sample size at t=1000 | ≈ #episodes in pool (**2** for CartPole) | ≈ **1000** |
| training | fresh policy + fresh `bc.BC` each round, batch ≤32 | warm-started FTRL/DAgger trainer, β≡1, batch 1 |

The pool was collected with `make_sample_until(min_timesteps=budget, min_episodes=1)`
(AND semantics), so it holds the *smallest* number of complete expert episodes whose
total length reaches the label budget. With `--n-rounds 1000 --samples-per-round 1`
the budget is 1000 transitions, so for CartPole (H=500 expert) the pool is **2
episodes**. Because the pool size equals the budget, the `subsample_strategy="uniform"`
draw (`rng.choice(len(pool), size=budget, replace=False)`) was a **no-op permutation** ,
it selected essentially the whole pool. Old BC-DAgger and fixed BC therefore both
trained on "the first ⌈N/H⌉ expert episodes, entire".

So the change is not about m. It is about how many *independent* expert episodes the
labels come from: **2 vs 1000**. That is exactly the temporal-redundancy /
effective-sample-size effect, and it is already visible in the existing runs.

## Evidence from the runs we already have

Disagreement rate at n=1000 observations (IQM over 5 seeds, read off the learning
curves), with the number of episodes the legacy pool could hold:

| env | expert H (approx) | pool episodes | legacy BC-DAgger | fixed BC | new BC-iid | FTL |
|---|---|---|---|---|---|---|
| CartPole-v1 | 500 | 2 | 0.086 | 0.086 | 0.003 | 0.003 |
| LunarLander-v2 | ~250 | ~4 | ~0.31 | 0.315 | ~0.025 | ~0.025 |
| MountainCar-v0 | ~120 | ~8 | ~0.079 | 0.079 | ~0.004 | ~0.0035 |
| Taxi-v3 | ~13 | ~77 | ~0.43 | 0.434 | ~0.03 | ~0.01 |
| Acrobot-v1 | ~80 | ~12 | ~0.006 | 0.010 | ~0.005 | ~0.005 |
| CliffWalking-v0 | ~13 | ~77 | →0 by n≈250 | 0.000 | →0 by n≈30 | →0 by n≈20 |
| FrozenLake-v1 | ~10 | ~100 | →0 by n≈120 | 0.000 | →0 by n≈5 | →0 by n≈5 |
| Blackjack-v1 | ~1.5 | ~600 | noisy ~3e-3 | 0.000 | →0 by n≈200 | →0 by n≈100 |

Two things to notice.

1. **Legacy BC-DAgger's asymptote is the fixed-BC line, in every environment**, and
   it reaches it by n≈50–100 and then stays flat to n=1000. Extra labels from the same
   2–3 episodes buy nothing. That is an information ceiling, not an optimization
   ceiling.
2. **The size of the gap tracks the episode count of the pool**, not anything
   env-independent. Acrobot (~12 episodes in the pool) shows almost no gap;
   CartPole (2 episodes) shows a 30x gap. An optimizer-side explanation would predict
   a roughly uniform gap. Taxi is the one apparent exception, and the likely reason is
   state-space size rather than episode count: 77 episodes cover only a fraction of
   Taxi's ~300 valid starts, so coverage still binds.

## The other things that changed with it (confounds to rule out)

`bc_iid` routes through `_run_dagger_variant`, so it inherited FTL's whole training
stack. Relative to old `bc_dagger` it also gained:

* **Warm start across rounds** (`warm_start=True`) vs a fresh policy every round.
  Over 1000 rounds this is ~1000x more accumulated optimization.
* **Optimizer batch size 1** (`initial_batch_size = max(1, min(bc_batch_size,
  samples_per_round)) = 1`) vs `min(32, k)`. ~32x more gradient steps per epoch.
* `ConstantL2Schedule(0.0)` through the FTRL trainer instead of plain `bc.BC`
  (should be equivalent, worth confirming).
* `ExponentialBetaSchedule(1.0)` ⇒ β≡1, i.e. the expert is always in control, so the
  rollouts really are `d^{π^E}`.
* `protocol_version` 2→3, expert-dataset `version` 3→4, and fixed BC's default
  `subsample_strategy` changed `uniform`→`prefix` with `shuffle=False` on collection.
  Fixed BC's numbers barely moved (CartPole 0.086→0.092), which is expected: when the
  pool is only as large as the budget, prefix and uniform select the same set.

The `_inner_train` minibatch guard added in the same commit
(`if len(train_subset) < minibatch_size: ...`) is inert for these configs
(DAgger trainers already run at batch 1; fixed BC has 900 training rows vs batch 32),
so it is not a candidate explanation.

## Proposed ablation (settles data vs optimization)

One environment where the gap is largest (CartPole-v1), 5 seeds, 1000 rounds,
linear policy, 4 cells:

| cell | data | training |
|---|---|---|
| A | fixed pool of `⌈N/H⌉` episodes (legacy) | cold restart, batch ≤32 (legacy) |
| B | fixed pool of `⌈N/H⌉` episodes (legacy) | warm-started FTRL trainer, batch 1 (new) |
| C | fresh episode per round, 1 uniform state (new) | cold restart, batch ≤32 (legacy) |
| D | fresh episode per round, 1 uniform state (new) | warm-started FTRL trainer, batch 1 (new) = current `bc_iid` |

Prediction if the diagnosis above is right: **C ≈ D ≫ A ≈ B**. If instead B ≈ D, the
gap is optimization and the "iid sampling" reading of the result does not hold.

Also worth logging directly: `trajectories_collected` for the shared pool per env
(already recorded in `_expert_dataset_source`), to replace the "expert H (approx)"
column above with measured numbers.

## Fairness caveat to state in the paper

BC-iid spends **one full expert episode per label** (≈500k expert env steps for 1000
CartPole labels); legacy BC-DAgger and fixed BC spend ≈1000 expert env steps total.
On the "expert queries / retained observations" x-axis they are matched, but on
expert *demonstration effort* BC-iid is ~H times more expensive. FTL is in the same
position (one learner rollout per round), so FTL vs BC-iid is a fair pairing , but
BC-iid vs fixed BC is not, and the plot caption currently only says "collection and
evaluation queries are additional".

This also means **BC-prefix (meeting item 4) is close to the old `bc_dagger`**: same
fixed pool, chronological instead of permuted. Designing it to differ from BC-iid on
the sampling axis alone (same trainer, same warm start, same batch size) is what makes
it answer Chicheng's question.

---

## Update (same day): the m > 1 sampling change is provably not the cause

The suspicion that started this , "we only sample one state per trajectory and
one trajectory per round, so the sampling change should not matter" , is now a
checked fact rather than a belief.

`_uniform_round_demos` was rewritten to sample per trajectory (meeting item 1),
and the m = 1 path was deliberately built to take **no** draw from the RNG when
the round holds exactly one trajectory. Rerunning `ftl`, `ftrl`, `bc_iid` and
`bc` on CartPole against the pre-change tree, same seed and same expert cache,
gives byte-identical per-round metrics. So at the campaign setting
(`--samples-per-round 1 --traj-per-round 1`) the round-sampling code is a
no-op, stream included, and cannot explain anything about
`legacy-sampling-v2` versus `20260910-bc-iid-v1`.

## Update: the ablation is now three baselines, not a side experiment

Rather than a separate 2x2, the decomposition is carried by three BC variants
that share the trainer and differ only in data (see
`BC_BASELINES_HANDOFF_2026-09-15.md`):

* `bc_prefix` , first t transitions of the fixed offline dataset, in order
* `bc_pool` , t uniform draws from that same fixed pool (the old `bc_dagger`)
* `bc_iid` , t draws, one from each of t fresh expert episodes

`bc_prefix → bc_pool` measures ordering and time-truncation inside a fixed pool;
`bc_pool → bc_iid` measures the number of independent expert episodes. The
prediction from this document is that the second step carries nearly all of the
gap, and that its size tracks `⌈N/H⌉` across environments , large on CartPole
(2 episodes), negligible on Acrobot (~12).

The optimization confounds listed above are removed by construction, because all
three now use the same warm-started trainer at the same batch size. What was
previously an argument from the shape of the curves becomes a direct
measurement.
