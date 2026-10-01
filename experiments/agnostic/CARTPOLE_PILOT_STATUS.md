# CartPole restriction pilot: status

Status snapshot: September 30, 2026, 5:45 pm Phoenix time
(October 1, 00:45 UTC), after the audit and data gates passed.

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

| Method | Full observations | Cart position hidden |
| --- | --- | --- |
| Fixed BC | Complete, 1,000 labels | Complete, 1,000 labels |
| FTL | Timed out; last saved round 646, last evaluation 640 | Timed out; last saved round 717, last evaluation 710 |
| BC-iid | Still running | Still running |

Both FTL jobs were stopped by the external two-hour limit. Their saved JSON
files still say `running`, with `snapshot.final=false`; the campaign inventory
is authoritative for their terminal `timed_out` status. Saved counts cover
only the snapshot, not unrecorded work before termination. All recorded
checkpoints, results and attempt logs from this snapshot have been preserved.
No job has been restarted and no limit has been extended.

The four terminal jobs consumed 21,044.06 worker-seconds in total: 6,643.72
for the two completed BC jobs and 14,400.35 for the two timed-out FTL jobs.
This excludes the two ongoing BC-iid jobs and shared audit/data preparation.

**The pilot is incomplete.** A six-way comparison at 1,000 labels is not
available. Both fixed-BC records report return 500 from their first one-label
prefix through their last evaluation, in both observation conditions. The independent Codex and verified Opus 5.5 reviews found no evidence of
an expert-head or evaluation-policy mix-up. Saved heads changed during
training and differ from the expert; frozen feature weights equal the
expert's, as the protocol requires. Reloading a saved masked checkpoint
preserves identical logits for inputs that differ only in cart position.
A record-level check confirms that every recorded trained evaluation across
all six runs reached 500 for all 100 episodes. This is a ceiling on the
observed return metric, not evidence of an FTL advantage. The role of the
inherited expert features is a plausible explanation, not a measured causal
conclusion.

### Runtime and interpretation findings

The previous pipeline gives FTL and BC-iid minibatches of one throughout
these runs, while fixed BC uses up to 32. It also fits the round-loop methods
after every label, while fixed BC fits only the 101 plotted prefixes. Thus
fixed BC and BC-iid differ in optimizer work as well as data acquisition.
This was preserved pipeline behavior, but it is material and must not be
silently changed as a runtime-only adjustment.

For example, at the 300-label full-observation fit, the recorded 20 epochs
imply 6,000 optimizer steps for FTL/BC-iid versus 180 for fixed BC. These
counts are derived from source and recorded epochs, not separate optimizer
instrumentation. Per-phase timers cover different work and are not clean
optimizer benchmarks. A batch-size correction, different budget, changed
restriction or episode limit requires consultation before a new run.

Two additional limits apply to later analysis. Evaluation episodes are not
paired across runs, and later round-loop head initializations can diverge
because collection and fitting consume random numbers differently. Shared
data and the initial head do match across observation conditions; this does
not establish common randomness at all later rounds. On-policy cross-entropy
and disagreement describe each learner's own visited states.

For a timed-out job, the stored in-flight operation text can lag by a round.
It must not be used to infer the exact operation at termination. Final wall
time comes from the controller; work after the saved snapshot is unknown.

The next report will retain all partial curves and distinguish completed
runs from timeouts. Existing BC-iid jobs continue with their original limits.
The stage deadline remains October 1, 04:09 UTC; no wider experiment or
scientific setting change is approved.

## Files

- Figure: `results/cartpole_position_mask_witness.png` (new; earlier figures
  are unchanged).
- The figure is drawn only from the verified audit record. It does not rerun
  the audit or roll out any environment.
