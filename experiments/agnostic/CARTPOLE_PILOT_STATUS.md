# CartPole restriction pilot: status

Status snapshot: 2026-09-30 22:22 UTC, after the audit and data gates passed.

This note is the current status of the paired CartPole restriction pilot
described in [`CARTPOLE_PILOT_PROTOCOL.md`](CARTPOLE_PILOT_PROTOCOL.md).
[`STUDY_RESULTS.md`](STUDY_RESULTS.md) and
[`EXPLAINER_BINS_AUDIT_TOY.md`](EXPLAINER_BINS_AUDIT_TOY.md) are earlier dated
snapshots and are left unchanged.

## What is done: the position mask audit

The restriction `cart_position_zero` shows the learner every CartPole
observation with the cart position x replaced by 0. The expert is not
restricted: it always labels the full, unmasked state.

The audit asks a narrow question: under this mask, are there two valid states
that the learner cannot tell apart but that the expert labels differently?

- **Setup.** 20 base states taken from one expert episode (reset seed 301).
  For each base, x was set to each value on a fixed grid
  {-1.8, -0.6, +0.6, +1.8} with the other three coordinates unchanged, and the
  expert labelled each resulting full state. Every pair of grid values within
  a base gives 20 x 6 = 120 pairs.
- **Result.** 120 pairs checked, 80 conflicting, 0 invalid. A conflicting pair
  has identical masked observations and different expert actions.
- **Verification.** An independent operator gate confirmed status complete,
  scheduler exit 0, artifact SHA-256
  `f803629e71dc7e87b0bff4c81f30ea05f38c0c3946fa4240da327f7c2cd58b91`, the
  source fingerprint (source commit `2896ffe`), and every conflicting pair.

![First conflicting pair](results/cartpole_position_mask_witness.png)

The figure shows the first conflicting pair in stored order (base state 0,
x = -1.8 and x = +0.6). The expert pushes left in one state and right in the
other, yet the learner receives exactly the same input for both. Equality was
checked on the stored full precision arrays; the figure rounds numbers only
for reading.

### How to read the counts

The 80 of 120 figure describes the chosen diagnostic grid. It is **not** a
probability that such states occur naturally, and it is **not** an error floor.
The audit is an existence proof: no deterministic policy of the masked
observation can match the expert on both states of a conflicting pair. It
bounds neither return nor on-policy imitation error, and no labels from the
audit are used for training.

## Shared data verified; learning runs dispatched

The shared-data job completed with exit 0. Result, source, expert and dataset
hashes were independently checked. Both datasets retain 1,000 labels, but their
acquisition is different:

| Dataset | Complete expert episodes | Environment steps | Retained labels |
| --- | ---: | ---: | ---: |
| Fixed BC chronological pool | 2 | 1,000 | 1,000 |
| BC-iid independent-episode stream | 1,000 | 500,000 | 1,000 |

Their content hashes differ. Full and restricted runs of each method consume
the same corresponding dataset. The original normalization measurement used
500 expert and 500 random episodes, yielding returns 500.0 and 22.95. The data
job took 312.47 seconds of controller-measured worker time, including both
acquisition and normalization.

- **Six seed 300 runs:** dispatched under the frozen manifest after both gates
  passed. Four run concurrently, with two waiting for a worker.
- **No completed learning comparison is available yet.** The audit and shared
  datasets are not evidence of an FTL performance advantage.
- **Limits:** original stage deadline 2026-10-01 04:09 UTC, at most 2 hours
  per job, at most 4 workers.
- **No wider experiment is approved.**

## Files

- Figure: `results/cartpole_position_mask_witness.png` (new; earlier figures
  are unchanged).
- The figure is drawn only from the verified audit record. It does not rerun
  the audit or roll out any environment.
