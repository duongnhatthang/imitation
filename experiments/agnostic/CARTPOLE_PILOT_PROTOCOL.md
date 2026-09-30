# CartPole restriction pilot: protocol and CLI

STATUS: the first CartPole step of `PIPELINE_RESTRICTION_PROPOSAL.md` is
approved. Implementation status is as of 2026-09-30 22:05 UTC, before any
scientific launch: code and tests exist and no pilot job has been run. Later
execution does not change this protocol snapshot. Every other mask, env,
seed set, budget, cap change and the mixture toggle remain proposals and are
not implemented.

## Fixed design

- Env `CartPole-v1`, original 500 step limit (checked at runtime), pilot seed
  300. Seed 301 is reserved for the mask audit and refused for data and runs.
- Expert: the existing qualified preparation, verified by record and
  checkpoint digests, loaded on CPU, never retrained. It always reads the full
  observation.
- Learner: the previous linear policy (expert clone, frozen features,
  reinitialized `action_net`), built by `restriction.make_linear_policy`.
  Restriction `identity` or `cart_position_zero` (x := 0; x_dot, theta,
  theta_dot kept), applied before the frozen features on every path. Building
  the mask draws no random numbers, so the head initialization is identical
  across conditions.
- Methods: FTL (beta 0), fixed BC, BC-iid. Six runs: 3 methods x 2
  restrictions.
- 1000 rounds or labels, 1 sample per round, lr 1e-3, up to 20 epochs per fit
  with the original held-out NLL early stopping, outer early stopping off,
  warm start off, FTL L2 0.
- Evaluation after round 1, every 10 rounds and at the end: 100 deterministic
  episodes, normalized return, learner and expert rollout cross-entropy,
  disagreement, raw episode returns, and a checkpoint that stores its
  restriction. No mixture is computed.

## Data acquisition

The `data` job acquires, once, what full and restricted runs share:

- Fixed BC pool: complete expert episodes in order until at least 1000
  transitions; the first 1000 are kept and the overshoot is recorded. At every
  plotted budget B, fixed BC cold-fits the first B transitions with the
  original fixed-BC routine (torch reseeded before each fit). This is the
  fixed-BC learning curve, not an extra method.
- BC-iid stream: 1000 independent complete expert episodes, one uniformly
  selected pre-action state from each. BC-iid replays it in order, one state
  per round, through the same trainer as FTL.
- Baselines: 500 deterministic expert episodes and 500 random episodes, as in
  `env_baselines.compute_baselines`, on a dedicated environment.

Pool, stream and baselines use dedicated seed streams, independent of each
other and of evaluation. Stored data keep full-state observations and expert
labels. Runs verify file and content digests before starting and record the
prefix digest at every evaluated budget; BC-iid also verifies that the data it
trained on equal the stream. FTL data differ by design (learner visitation).
Physical acquisition is charged to the data job; each run reports its logical
consumption separately.

## Mask audit

On the diagnostic reset seed 301, the qualified expert runs one deterministic
episode; its states at steps 0, 25, ..., 475 are the bases. For every base and
every cart position in {-1.8, -0.6, 0.6, 1.8}, the simulator state is set
directly and checked: nonterminal under the env's own thresholds, observation
map verified against `step`, and one physics step under each action moving
paired states identically except for the position offset. The expert labels
full observations. Every checked state and pair is kept. Status
`conflict_found` proves only that conflicting expert labels exist under the
mask; it does not prove a positive on-policy error, and no classifier-based
Bayes error is claimed. With no conflicts the status is
`restriction_uncertified`; no other mask is selected automatically. Audit
labels never reach training.

## Commands

Placeholders: `PY` is the environment's Python, `SRC` the source snapshot's
`src` directory, `PREP` the qualified CartPole preparation directory, `OUT`
a stage root, `DEADLINE` the stage deadline (at most 6 hours after launch and
never after `2026-10-07T01:01:00Z`). Every output directory must be new.

```bash
export PYTHONPATH=SRC
M=imitation.experiments.agnostic.restriction_pilot
COMMON="--preparation-dir PREP --deadline DEADLINE --job-limit-seconds 7200"

# Stage A (independent jobs)
PY -m $M audit $COMMON --output-dir OUT/audit
PY -m $M data  $COMMON --output-dir OUT/data --seed 300

# Stage B, after review of Stage A
for method in ftl bc bc_iid; do
  for r in identity cart_position_zero; do
    PY -m $M run $COMMON --data-dir OUT/data --seed 300 \
      --method $method --restriction $r --output-dir OUT/run-$method-$r
  done
done
```

Use `imitation.experiments.agnostic.campaign` with at most 4 workers and
`timeout_seconds` 7200 per job. A job's `expected_result` is
`<output dir>/result.json`; useful `expected_fields` are `status`
(`complete`), `schema`, `seed`, and for runs `method` and `restriction_id`.
A retry needs a new output directory, because outputs are never overwritten.

## Records

Every `result.json` carries `protocol`, `schema`, `status` (`running`,
`complete`, `partial` at the deadline, `failed`), the expert and preparation
digests, source file digests, package versions, the requested, per-job and
effective deadlines, and `episode_cap`. Paths in records are relative.
Unknown values are null, never zero.

Snapshots: `result.json` is replaced atomically while a job runs. Every record
carries `snapshot` with `written_at_utc` and `final`. `final` is true once the
job itself wrote `complete`, `partial` or `failed`; it is false while the
status is `running`. A scheduler kill (for example the campaign timeout) can
leave a `running` snapshot. Such a record is only the last snapshot, not a
final cost: work after it is unknown, and the job's final status and wall cost
come from the external campaign inventory. Every non-complete record also
names the operation that was in flight (`interruption.in_flight_operation`).

- Audit: `audit_status`, the full `audit` record, and `expert_predict_calls`,
  which is null unless the audit finished.
- Data: `datasets` (file digest, pairs digest, collection statistics including
  episode lengths, reset seeds and overshoot), `baselines`, `baselines_costs`,
  `live_counters` (pool and stream queries, resets and steps, counted when an
  operation returns), and `baselines_observed` (observed baseline env steps
  and resets; its expert predict calls are null unless the measurement
  finished).
- Run: `config` (with `mixture: false`), `data` digests, `records` (every
  round for FTL and BC-iid, every evaluated budget for fixed BC, with wall
  time), `trained_pairs_sha256`, and `accounting`:
  - `observed`: env steps, resets and finished episodes counted by the run's
    environment wrapper when a step or reset returns. These are exact physical
    counts as of the snapshot.
  - `completed_records`: subtotals over finished rounds, fits and evaluations
    only (collection and evaluation steps, expert and learner predict calls,
    retained labels, fits and epochs).
  - `in_flight`: the operation in progress, its env steps (observed minus
    completed), and its expert and learner predict calls, which are null
    because unfinished queries are not observed.
  - `totals`: `exact` is true only if the job completed with no unattributed
    steps; then the query totals equal the completed subtotals. Otherwise
    `env_steps` is the observed count and the query totals are null.
  - `reconciled` (observed steps equal completed collection plus evaluation),
    the shared acquisition (not charged to the run), and wall time by phase.

## Deliberate differences from the previous pipeline

- Fixed BC and BC-iid data come from the shared data job, not from each
  run's environment, so both conditions train on identical data.
- Baselines are measured once per data job with a seeded random-action
  sampler, not read from the expert cache.
- The expert comes from the verified preparation, not the expert cache.
- Deadlines are checked between rounds and fits; a started round or
  evaluation completes first, and the external controller enforces hard
  limits.
