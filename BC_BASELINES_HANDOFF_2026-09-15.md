# BC baseline implementation handoff

This note records the earlier baseline work carried into the current branch.
Current launch settings are in `RUN.md`; the proposed agnostic study is in
`experiments/AGNOSTIC_EXPERIMENT_PLAN.md`. Historical result files retain their
original settings and are not evidence for the new cold-start protocol.

## Sampling semantics

| Algorithm | Training data |
|---|---|
| `bc` | A fixed budget of chronological expert transitions by default |
| `bc_iid` | One uniformly chosen state from each fresh expert trajectory |
| `ftl`, `ftrl` | One uniformly chosen state from each learner-controlled trajectory, labeled by the expert |
| `bc_prefix` | Increasing prefixes of the canonical fixed BC dataset |
| `bc_pool` | Increasing uniformly permuted subsets of that same fixed BC dataset |

The last two algorithms remain available for sampling diagnostics but are off
by default. The canonical pool is truncated before permutation, so the optional
pool comparison does not introduce transitions outside fixed BC's dataset.

For more than one sample per round, sampling uses distinct complete trajectories
and one uniform state within each selected trajectory. Collection increases the
trajectory count when needed. Retained labels, oracle calls, and environment
steps are different budgets and are recorded separately.

## Reproducibility and visualization

Python, NumPy, and Torch use seeded per-cell initialization. Shared expert
training uses a separate fixed seed, so whichever cell first fills the cache
does not choose a different expert. Result metadata binds settings and source
fingerprints; changed protocols require new output directories.

The default coverage difference compares FTL with BC-iid. Coverage plots and
runtime figures use saved artifacts and cannot by themselves establish a causal
improvement from interactive data collection.

The old diagnostic interpretation is in
`BC_IID_VS_BC_DAGGER_DIAGNOSIS_2026-09-15.md`. In particular, the removed
`bc_dagger` baseline used a fixed expert pool, whereas BC-iid collects fresh
expert episodes. These are different sampling procedures.
