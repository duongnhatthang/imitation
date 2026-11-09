# Fixed-BC correction amendment (draft proposal)

> **STATUS: PROPOSAL PENDING USER CHOICES. NOT APPROVED, NOT FROZEN, NOT
> AUTHORIZED FOR EXECUTION.**
>
> - No protocol choice in this file is approved. Nothing here may be
>   implemented as scientific code, run, or frozen until the user explicitly
>   approves it.
> - Settled by the user, and likely to supersede the exact-bin design in
>   Section 3: reuse the previous learned-policy pipeline
>   (`experiments/run_learning_curves.sh`) wherever possible, comparing runs
>   with and without a learner-only restriction (for example an observation
>   mask) while changing as few variables as possible, and target all 8
>   original classical environments if feasible. That pipeline's default
>   policy mode is linear (frozen expert features plus a new trainable
>   head), so any restriction must act on observations before those
>   features. The fixed-BC definition in Section 2 is unaffected.
> - Also settled: restricted and unrestricted runs go in the same graphs, one
>   color per method, solid restricted and dashed unrestricted, and training
>   runs long enough for the restricted learning curves to plateau. The
>   proposed comparison is sketched in
>   [`STUDY_RESULTS.md`](STUDY_RESULTS.md#proposed-next-comparison-not-approved).
> - The user has asked that FTL mixture evaluation be off by default in future
>   work. That is an intended future default only; it is not implemented.
> - Episode length: the user asked to consider both longer training and
>   longer environment episode time limits. Any time-limit comparison would
>   apply the same cap to all methods and both restriction conditions, keep
>   the cap factor separate from the mask factor, re-evaluate the expert and
>   random normalization references at the new cap, and distinguish timeouts
>   from natural terminal failures. No automatic cap change; none is
>   implemented. Still requiring explicit consultation: the exact masks, the
>   exact time limits, the numeric training budget, the plateau criterion,
>   and other numeric settings. Hiding a feature alone does not guarantee
>   agnosticity; each mask needs verified opposite labels at masked-together
>   states and distribution-specific error evidence.
> - No corrected fixed-BC outcome exists, and no new experiment has been run.
> - No implementation plan exists yet; none should be written or executed
>   before the choices above are made.

This file records the fixed-BC error in the agnostic study and one possible
way to add the intended fixed-BC baseline as an append-only addendum.
Current diagnostic results are in [`STUDY_RESULTS.md`](STUDY_RESULTS.md);
their fixed-BC parts were an alias and have been removed, and their
FTL-DAgger versus BC-iid parts stand as diagnostic results.

## 1. What was wrong

The agnostic study was meant to compare FTL-DAgger with two baselines, BC-iid
and fixed BC, at equal retained-label budgets B. The executed "fixed BC" refit
the first B labels of BC-iid's own stream (one uniformly selected state from
each of B independent expert episodes) and reused BC-iid's evaluation. It
therefore equaled BC-iid by construction, and the producer raised an error
if it did not. It is not fixed BC. In all new reports and analysis it is
named **`bc_iid_refit_alias`**. It supports no scientific statement about
fixed BC, including equality with BC-iid.

The divergence started in the pre-execution plan
([`../AGNOSTIC_EXPERIMENT_PLAN.md`](../AGNOSTIC_EXPERIMENT_PLAN.md), fixed-BC
paragraphs in "Data budgets and current annotation-cost caveat"), which
treated a shared iid pool as a useful control and a chronological prefix as
a confound. That was an error in our proposal, carried into code, analyzers,
and tests. It is not the intended definition.

## 2. Corrected definition (from the original fixed-BC runner)

This is the definition already implemented in the original FTRL runner
(`src/imitation/experiments/ftrl/run_experiment.py`, `_shared_expert_data`,
`_run_bc`, and `bc_prefix`) and pinned by its tests
(`tests/experiments/test_bc_iid.py`, `tests/experiments/test_bc_baselines.py`).

- **Fixed BC.** The fixed deterministic expert is rolled out in **one
  collection pass**: complete episodes, one after another, until the total
  number of transitions is at least Bmax = 4096. Every pre-action state of
  every collected episode is kept, labeled with the executed expert action,
  in chronological order (episode order, then time step). The tail of the
  last episode beyond Bmax is collected, paid for, recorded, and discarded.
  At each checkpoint B the same exact ERM is fit on the first B transitions
  of that pool. There is no one-per-episode subsampling.
- **BC-iid** (unchanged, already correct). The same expert is rolled out once
  per retained label, and one uniformly selected state from each independent
  episode is kept.
- **Prefix, not minibatch order.** "Prefix" selects which transitions are in
  the size-B dataset. The learners are exact count-based ERM, so the order in
  which a fit visits its data has no effect.

Because the pool is nested, the fit at B is exactly what a standalone fixed-BC
run with budget B would produce, so the original `bc` (one fit at the budget)
and `bc_prefix` (fits along the prefix) coincide here.

## 3. Quantities this proposal would reuse (not approved)

If this proposal were approved as is, it would reuse the existing exact-bin
design below. The user has since chosen to reuse the previous learned-policy
pipeline on all 8 classical environments where feasible, so this exact-bin
reuse is likely superseded; its details remain pending consultation.

Classical: CartPole-v1 and Acrobot-v1, mild and severe quantizers, 20 seeds
1000..1019 (80 cells), the frozen experts, B = 4096, checkpoints 128, 256,
512, 1024, 2048, 4096, primary budget 4096, 100 evaluation episodes per
checkpoint on the frozen evaluation reset seeds. Toy: the full primary grid,
100 seeds 1000..1099 (6400 cells), B = 4096, batch 16, same checkpoints,
exact dynamic-programming evaluation. Campaign hard cap
2026-10-07T01:01:00Z, at most 4 workers, bounded jobs. Nothing would be
reselected.

The toy secondary batch-sensitivity check (320 cells) gets no fixed-BC
addendum. It remains a secondary, descriptive FTL-DAgger versus BC-iid
result; its old fixed-BC fields are the alias and are invalid.

## 4. Common random numbers and pairing

*Sections 4 to 9 describe what this proposal would do only if approved;
none of it is authorized or implemented.*

No new randomness and no choices:

- Classical fixed-BC episode j is reset with the existing frozen training
  reset seed j of that cell, which is also the reset seed of FTL-DAgger and
  BC-iid episode j. Because the expert and the dynamics are deterministic,
  fixed-BC episode j reproduces BC-iid episode j exactly; the addendum checks
  this.
- Toy fixed-BC episode e consumes the full row of the existing expert episode
  tape of BC-iid episode e: round `e // 16`, row `e % 16`, all H pre-action
  states.
- Consequence, disclosed: the fixed-BC data at B include the BC-iid samples
  of its first few episodes, so fixed BC and BC-iid are positively coupled.
  Their difference is secondary and descriptive only.
- Each classical cell collects its own pool, including separate collection
  for mild and severe cells of the same seed, matching the per-cell cost
  accounting of the original cells.

## 5. Append-only sidecars and identity gates

Original FTL-DAgger and BC-iid raw results, manifests, analyses, and figures
are immutable. Each corrected fixed-BC result is a new sidecar record bound
to the sha256 of the original cell (classical) or seed (toy) result it
extends. Legacy producer and analyzer source files stay byte-identical so the
original records keep verifiable source identities.

Classical sidecars run these gates, in order, before any fixed-BC fit is
evaluated. Any failure is recorded as status `rejected` with its details and
stops that cell. There is no silent fallback and no automatic rerun of the
other methods; the affected condition is held pending diagnosis.

1. **Identity.** Base result sha256 as frozen; base record complete; legacy
   manifest, expert, preparation, quantizer, stage 1 and stage 2 source
   hashes, and package pins all match.
2. **Streams.** Training reset seeds, selection uniforms, and evaluation
   reset seeds recomputed from the frozen streams equal the base record.
3. **Replay.** Every pool episode j reproduces the base BC-iid episode j:
   length, and the bin and label at its recorded selection index.
4. **Re-evaluation.** The stored FTL-DAgger final and BC-iid final tables at
   all six checkpoints are re-evaluated on the same 100 evaluation reset
   seeds and must reproduce every stored per-episode return, length, and
   disagreement exactly. This validates reuse of both whole curves. Its cost
   is validation, not training or primary evaluation.

Toy sidecars check the base result identity and source, recompute the exact
class costs, and check that every fixed-BC tape row, at the time selected by
the original tape, reproduces the original BC-iid retained sample, and that
the original BC-iid counts at every checkpoint are reproduced.

Fixed BC then gets its **own** evaluation at every checkpoint: 100 episodes
on the frozen evaluation reset seeds (classical) or exact dynamic
programming (toy). Its table or policy may legitimately equal BC-iid's; that
is recorded as an observation, never required and never an error.

## 6. Cost accounting

- Fixed BC, physical, once per cell: the pool episodes, their env steps and
  expert action entries (executed action and label counted once), overshoot
  collected and not retained, one fit per checkpoint, and its own evaluation.
- Per checkpoint, the logical standalone cost of a fixed-BC run at that B
  (episodes until at least B transitions, including that episode's overshoot)
  is reported separately and never summed.
- Fixed-BC cost is never attributed to BC-iid, and BC-iid's acquisition is
  never reported as fixed BC's.
- The total study spend sums every distinct piece of work: the original
  FTL-DAgger, BC-iid, and mixture work, the wasted `bc_iid_refit_alias` fits,
  the new fixed-BC work, the new validation re-evaluation, and all earlier
  stages. Failed and partial attempts are included.
- The previously reported "logical standalone fixed BC" cost described the
  alias and is withdrawn. The corrected fixed BC needs on the order of 4,500
  expert entries per CartPole cell and about 4,200 per Acrobot cell, versus
  about 2,048,000 and 349,000 for BC-iid.

## 7. Statistics

Everything reuses the original bootstrap design: 95% level, 10,000
resamples, bootstrap seed 20260930, whole training seeds as the resampling
unit, and one seed-index matrix shared by every contrast.

- **Classical primary contrasts** at B = 4096, paired by seed, native return:
  (1) FTL final minus BC-iid final; (2) FTL final minus fixed BC.
- **Original family** (contrast 1 over the 4 conditions): reported
  unchanged at 98.75%, labeled as the original prespecified family.
- **Amended family** (both contrasts over the 4 conditions, 8 tests):
  Bonferroni 99.375%, reported for all 8, including contrast 1 again at
  this level. Any claim that involves fixed BC uses the amended family.
- **Toy**: original 6-test family (BC-iid cost minus FTL final cost,
  kappa 1, alpha > 0, each H and alpha, balanced q) at 99.1667%; amended
  12-test family adding fixed-BC cost minus FTL final cost, at 99.5833%.
- Significant and null results are both reported. Neither family is chosen
  by its effect. BC-iid versus fixed BC, learning curves, areas, and
  disagreements are secondary, unadjusted, and descriptive.

## 8. Interpretation limits stated in advance

- Under expert control the toy never reaches its recovery state, and its
  nominal state is redrawn independently at every step. Chronological fixed
  BC and BC-iid therefore have the same data law in the toy, and their
  learned policies are expected to coincide. The toy still tests the
  interactive-recovery hypothesis (FTL-DAgger against each expert-data
  baseline). It cannot show a temporal-correlation penalty of fixed BC
  relative to BC-iid.
- In the classical tasks, fixed BC differs from BC-iid both in trajectory
  correlation and in its state weighting (all states of a few trajectories
  versus one state from each of many). The addendum cannot separate these two
  causes, and no additional baseline is proposed to do so.
- The Acrobot severe condition was a floor for FTL-DAgger and BC-iid; a floor
  for fixed BC would be reported as degenerate, not as a null.

## 9. Disclosure and proposed freeze procedure

This is a post-result correction. The FTL-DAgger and BC-iid results were
inspected before this proposal. The definition comes from the original
fixed-BC code and the user's specification, not from any data; no corrected
fixed-BC outcome has been computed or inspected. Nothing has been frozen. If
the user approves a design, a public amendment manifest
(`fixed_bc_amendment_manifest.json`) would be frozen and hashed before any
sidecar runs. It would bind the original manifests, every base result hash, the
original analysis summaries, the new producer and analyzer source hashes,
the gates, contrasts, families, and levels above. Behavioral unit tests on
synthetic fixtures could run before that freeze; no scientific fixed-BC
result could. Any implementation would be reviewed independently before it
runs, and again by a fresh reviewer before any scientific execution, and
only after the user approves the protocol.
