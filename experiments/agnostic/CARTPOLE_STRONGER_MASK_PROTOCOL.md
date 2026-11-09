# CartPole stronger-mask pilot: protocol `/2`

STATUS: implemented and covered by local unit tests on synthetic fixtures.
No job of this protocol has been run. Scientific execution stays blocked
until independent Codex checks and a fresh, separate Opus review pass. This
file pins the approved contract; it does not authorize a launch.

Protocol `/1` (`CARTPOLE_PILOT_PROTOCOL.md`, source `2896ffe`) is historical:
x-only mask, fixed BC fitted on growing prefixes, online minibatch 1. Its
records are not rerun or reinterpreted by this protocol. One short
supersession note was added at the top of that file; nothing else in it
changed.

## Approved six-run contract

- Env `CartPole-v1`, 500 step cap (checked at runtime), seed 300. Seed 301
  is reserved for the audit.
- Restrictions: `identity` (full `[x, x_dot, theta, theta_dot]`) and
  `cart_position_angular_velocity_zero` (learner sees `[0, x_dot, theta, 0]`).
  These are the only run choices. `cart_position_zero` stays registered
  only so historical checkpoints load.
- Methods: fixed offline BC, BC-iid, FTL. 3 x 2 = six runs, all fresh,
  including the full-observation controls.
- Learner: the existing linear policy (expert clone, frozen features,
  reinitialized `action_net`), mask applied before the frozen features on
  every path. The expert, the stored expert data and the normalization
  references (expert 500, random 22.95) are the existing ones.
- 1000 labels, 1 per round, lr 1e-3, at most 20 epochs per fit, held-out
  NLL early stopping with its original fallback for small data, outer early
  stopping off, `warm_start` false, beta 0 for FTL, mixture off.
- Evaluation: 100 deterministic episodes, normalized return unchanged.
  Online: round 0, 1, every 10, and 1000. Fixed BC: once.
- Not part of this contract: BC-prefix or BC-pool, any second mask, extra
  seeds or envs, tuning, retries.

## Methods

| | Fixed BC | BC-iid | FTL |
|---|---|---|---|
| Labels | stored pool: complete expert episodes, first 1000 transitions in order | stored stream: one uniform state from each of 1000 independent expert episodes, replayed one per round | collected in run: one learner-controlled episode per round, expert labels, one uniform state kept |
| Data held | all 1000 | exactly t after round t | exactly t after round t |
| Fits | one cold fit (`_fit_fixed_bc`, torch reseeded) | cold refit every round on all t | same as BC-iid |
| Record | one row, one checkpoint, `reference: "flat"` | one row per round | one row per round |

Fixed BC refuses any `n_rounds` other than the pool size: it never fits
prefixes. Its row records the trained pairs hash (equal to the full pool
hash), the effective batch, logical acquisition cost and separate fit and
evaluation timers.

For analysis of these runs: the x-axis is the number of labels trained on.
BC-iid and FTL are curves. Fixed BC is a frozen policy drawn as a flat line,
labelled as using all 1000 labels, not as a value at each x. FTL's expert
queries are a separate cost, not the x-axis.

## Batching (actual behavior)

`ExperimentConfig.grow_batch_with_data` (default false, which keeps the
original `min(32, samples_per_round)` for every round) is true in this
pilot. Before each round's data are loaded, batch and minibatch are set to
`min(32, accumulated labels)`. Fixed BC uses `min(32, dataset size)`.

Held-out stopping runs only when `floor(0.1 n) >= 32`, that is `n >= 320`;
below that each fit runs 20 epochs on all `n`. When the split runs and
leaves fewer training examples than the batch, the batch shrinks to the
split (it never does at these sizes, since the split needs `n >= 320`).

Loaders shuffle and drop the incomplete final batch every epoch. So with
`n` training examples and batch `b`, each epoch uses `floor(n / b) * b`
examples (a different subset each epoch). Examples: round 63 trains one
batch of 32 per epoch; fixed BC and the online fit at round 1000 both train
on 900 examples (100 held out) in 28 batches of 32, 4 dropped per epoch.

Each fit record gives the loader actually trained on: `train_batch_size`,
`train_examples`, `train_batches_per_epoch`, `train_drop_last`.

## Data reuse

The data job is unchanged and keeps schema `agnostic-cartpole-restriction-pilot/1/data`.
A run verifies the data record from the same bytes it hashes:

- schema, protocol `/1`, completion, env, cap, seed;
- config digest, seed, budget, and the pool, stream and action rules;
- the producer's source file map against its combined digest, and the same
  file set as the current source;
- package versions equal to the executing environment's;
- expert digest and the full preparation identity;
- finite baselines;
- both NPZ files: file digest, pairs digest and size.

Data from the same source version pass after these checks. Data from
another source version are refused unless `--expected-data-sha256` equals
the SHA256 of the data `result.json` bytes. A pin that does not match
refuses even same-source data. There is no other bypass. The run record keeps
`data.acceptance` (rule, pin, producer protocol, producer source and package
versions) separately from its own execution `source`.

The existing data record is pinned at
`2588f231c5d98b9c22f321231dc6f61d36326c1843330aa5a552fa502c4abc8c`. A
read-only local check confirmed its config, source map and NPZ digests pass.
It was produced by a different source version (`restriction.py`,
`restriction_pilot.py` and `run_experiment.py` changed), so the pin is
required. Package versions (Python 3.10.21, torch 2.4.1+cu121) and the
preparation identity can only be checked in the execution environment.

## Audit

One audit job (`audit` command, protocol `/2`): the existing seed 301 x-grid
audit (cart positions -1.8, -0.6, 0.6, 1.8, base steps 0 to 475 by 25) with
`restriction_id = cart_position_angular_velocity_zero`. States, physics
checks and labels do not depend on the mask. Every pair differs only in `x`
and shares `x_dot`, `theta` and `theta_dot`, so the x-only witnesses also
collide under the stronger mask. The audit establishes only that expert
labels conflict under the mask. It checks no pair that differs in angular
velocity, so it says nothing about extra conflicts from hiding angular
velocity, and nothing about an error floor on natural or on-policy state
distributions. No angular-velocity grid is generated. With the same expert,
the result should reproduce the x-only audit's 80 conflicting of 120 checked
pairs; any other count should be treated as a determinism failure.

## Limits

- Attribution is to the joint mask `{x, theta_dot}`. There is no corrected
  x-only run, so no incremental angular-velocity effect can be claimed, and
  protocol `/1` results are not comparable (different batching and BC).
- Resource bounds for any later launch: at most 4 remote workers, 2 h per
  job including cleanup, 6 h for the stage starting at the first audit,
  overall deadline 2026-10-07T01:01 UTC. The in-process
  `--job-limit-seconds` must leave room for cleanup inside 2 h.
- Run time under the new batch rule has not been measured. Protocol `/1`
  timers suggest about 3,300 s of evaluation per run when episodes reach the
  cap; this is an estimate only.

## Commands (templates; paths are placeholders)

```bash
python -m imitation.experiments.agnostic.restriction_pilot audit \
  --preparation-dir PREP --output-dir OUT/audit \
  --deadline 2026-10-07T01:01:00Z --job-limit-seconds LIMIT

python -m imitation.experiments.agnostic.restriction_pilot run \
  --preparation-dir PREP --data-dir DATA \
  --expected-data-sha256 2588f231c5d98b9c22f321231dc6f61d36326c1843330aa5a552fa502c4abc8c \
  --seed 300 --method {bc,bc_iid,ftl} \
  --restriction {identity,cart_position_angular_velocity_zero} \
  --output-dir OUT/run-METHOD-RESTRICTION \
  --deadline 2026-10-07T01:01:00Z --job-limit-seconds LIMIT
```
