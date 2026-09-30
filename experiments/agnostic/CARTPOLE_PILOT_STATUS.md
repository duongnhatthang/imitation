# CartPole restriction pilot: status

Status snapshot: 2026-09-30, after the position mask audit was verified.

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

## What is not done yet

- **No learning curves exist yet.** Nothing in this note measures BC, DAgger,
  or any learner under the mask.
- **Shared data job:** running now on private compute.
- **Six seed 300 runs:** will start only after the shared data job passes its
  gate.
- **Limits:** original stage deadline 2026-10-01 04:09 UTC, at most 2 hours
  per job, at most 4 workers.
- **No wider experiment is approved.**

## Files

- Figure: `results/cartpole_position_mask_witness.png` (new; earlier figures
  are unchanged).
- The figure is drawn only from the verified audit record. It does not rerun
  the audit or roll out any environment.
