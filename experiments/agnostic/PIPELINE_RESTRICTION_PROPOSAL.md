# Proposal: learner-only observation restriction on the original pipeline

Status snapshot: 2026-09-30 22:05 UTC, before the approved pilot launch.

STATUS (updated 2026-09-30): the user approved ONLY the initial six-run
CartPole pilot in section 7 ("First step"), together with the implementation
and the mask, identity, checkpoint, and data-contract gates it needs. That
pilot is being implemented. It has NOT run, and no result from it exists.
Everything else below remains a PROPOSAL and is NOT approved: the wider
eight-environment pilot (S1), the 240-run main campaign, every mask other
than the pilot's CartPole `x := 0`, any time-limit (cap) change, learning
rate tuning, the budget and plateau rules, and the main seeds. No claim that
all 8 envs fit the deadline or that any mask is certified.

Plain-language background on the earlier bin-based diagnostic, its audit
bound, and the toy is in [`EXPLAINER_BINS_AUDIT_TOY.md`](EXPLAINER_BINS_AUDIT_TOY.md).

## 0. Context

- The bin-based diagnostic (coarse-observation table learner) is diagnostic
  only; its fixed-BC fields were a BC-iid refit alias and are invalid. It has
  no unrestricted data; both conditions run fresh. Its bins are not reused:
  the masks here keep observations continuous (for CartPole only `x` is
  replaced by 0).
- Deadline: the original one, 2026-10-07 01:01 UTC. At most 4 workers.
- Proposed target main design (not approved; never started automatically):
  8 envs x 2 conditions (full, restricted) x 3 methods (FTL, fixed BC,
  BC-iid) x 5 seeds (original `--seeds 5` default) = 240 runs.

## 1. Original pipeline facts relied on (read from code)

- `create_linear_policy` deep-copies the expert, freezes all but `action_net`,
  re-inits `action_net`; the full class contains the expert. Torch is seeded
  after expert load. `_run_dagger_variant` (FTL beta 0; BC-iid beta 1, one
  uniform state per expert episode) and `_run_bc` both build through it.
- `_run_bc` does ONE final fit on the chronological pool and returns one point.
- Eval: 100 deterministic learner episodes, expert labels the same obs; CE via
  `evaluate_actions`; normalized return from `baselines.json`; disagreement.
- Defaults: 1000 rounds, 1 sample/round, 20 epochs, eval every 10, inner
  held-out NLL early stop, lr 1e-3. `outer_early_stop` defaults True; set off.
- Caps (gymnasium 0.29.1 specs): CartPole 500, FrozenLake 100, Acrobot 500,
  MountainCar 200, Taxi 200, LunarLander 1000; CliffWalking and Blackjack none
  (`env_utils` fallback 200 and 20).
- `policy.save` stores state_dict plus constructor data; `load_policy_checkpoint`
  rebuilds a plain `ActorCriticPolicy` and loads strictly.

## 2. Mask placement and integration

- Wrap the learner's frozen extractor: `x -> phi_E(M(x))`, `M` fixed,
  parameter-free, dimension-preserving. SB3 routes `predict`,
  `evaluate_actions`, and `forward` through the extractor, so loss, collection,
  eval, CE, and disagreement all see `M`. The expert is a separate object and
  sees unmasked obs. Demos keep raw full-state obs and expert labels.
  Optimizer, loss, head, lr unchanged.
- Paired expert data is a DESIGN REQUIREMENT, not a consequence of seeds or
  unmasked demos. The original pipeline does not guarantee it: one venv serves
  collection and `_compute_round_eval`, so learners with different eval episode
  lengths advance stochastic env RNG differently, and methods share `rng`.
  Fixed BC: one canonical pool shared by both conditions. BC-iid: share/replay
  one independently collected one-state-per-expert-episode stream, or use
  dedicated reset/selection streams; require data-hash equality across
  conditions. FTL collection intentionally differs (learner visitation). Keep
  physical sharing (one stored artifact, compute saved) separate from logical
  standalone runs (each run reads it as its own). The exact minimal collector
  or cache change must be reviewed before implementation.
- Not an env wrapper: the expert reads the same venv obs.
- Integration (our `experiments/ftrl` extension in this worktree; no upstream
  `imitation.algorithms` edits, original checkout untouched): add an explicit,
  picklable config field `restriction_id` (default `"identity"`) resolved by a
  registry in a policy-factory function that both `_run_dagger_variant` and
  `_run_bc` call. Identity goes through the same factory path. Also add a
  `max_episode_steps` config field passed to `make_env` and to every cache key.
  No process-global monkeypatching.
- New files: `src/imitation/experiments/agnostic/restriction.py` (registry,
  masked extractor, audits), a new runner, and
  `tests/experiments/test_agnostic_restriction.py`.

## 3. Pitfalls and leakage paths

1. SB3 shares one extractor across `features_extractor`, `pi_...`, `vf_...`;
   replace all, assert identity, assert only `action_net` is trainable.
2. Checkpoints: the plain loader cannot guarantee reconstruction. A wrapped
   extractor changes state-dict keys (and buffers), so strict loading may fail;
   if the map lives outside the state dict it is omitted. Needed: restriction id
   and cap in the result manifest, a custom loader, and a round-trip test
   (masked actions and CE equal before and after reload).
3. Building `M` must not draw torch RNG, or head init differs by condition.
4. Caches: `baselines.json`, expert `model.zip`, and the expert-data cache must
   be namespaced by cap and expert provenance digest. The data cache must not
   include the mask (sharing across conditions is intended).
5. Representatives: pooled one-hot classes map to the smallest encoded state id
   in the class; continuous maps stay inside the declared obs range. This only
   avoids malformed inputs. It does NOT keep frozen features in-distribution;
   the feature distribution changes intentionally under restriction.
6. Identity regression: compare substantive identity (per-round metrics, policy
   state-dict hashes, dataset hashes, RNG-dependent outputs) with deterministic
   settings on one env and seed, excluding paths, timestamps, and config
   metadata fields. Behavioral regression: fixed-BC and BC-iid data hashes
   must match between full and restricted runs of the same seed.

## 4. Fixed BC curve (settled by user; extension required)

Collect one chronological expert pool at the max budget once; at every plotted
budget B, cold refit on the first B labels with the unchanged fit routine. This
is the fixed-BC curve, not an extra method; it needs an extension since
`_run_bc` fits once. Record pool hash, per-B prefix hashes, nested-prefix
identity, and overshoot (whole episodes past max budget). Pilot measures cost.

## 5. Candidate mild masks (structural; expert verification required)

Only the CartPole row is approved, and only for the initial pilot. All other
rows are unapproved candidates.

| Env | Proposed M (dim kept) | Possible label conflict (unverified) |
|---|---|---|
| CartPole | x := 0 | Expert may use cart position; equal (xd, th, thd) at different x may get different pushes |
| Acrobot | s2 := abs(s2) (fold theta2 sign) | Mirrored elbow angles may get different torques |
| MountainCar | vel binned: edges -0.07 + 0.01k (k=0..14), vel clipped to [-0.07, 0.07], 0.07 in last bin, value := bin center | Expert switch points may fall inside a bin |
| LunarLander | both leg contacts := 0 | Near-ground contact and airborne states may get different actions |
| FrozenLake 4x4 | pool (r,c) with (r, c xor 1) in rows 0-1 | Merged nonterminal cells may have different expert moves |
| CliffWalking | pool column pairs in row 2 | Merged cells (e.g. near column 11) may differ |
| Taxi | passenger in taxi: pool destination Y with B | Navigation and drop-off may differ |
| Blackjack | usable-ace block := [1,0] | Soft vs hard totals may get different hit/stick |

Masks are fixed before any FTL result and never changed because a method loses.

## 6. Certification

Masking alone does not guarantee a nonrealizable setting.
- Discrete: enumerate pooled classes over reachable NONTERMINAL states only
  (terminal holes, cliff, goal excluded). A class with differing expert actions
  proves global nonrealizability for deterministic policies of `M(s)`. Positive
  error under d^{pi_E} additionally needs expert visitation mass on two
  conflicting members; deterministic-path experts (FrozenLake, CliffWalking) may
  give zero.
- Continuous: paired audit only with simulator-valid partners (set full
  simulator state, recompute obs, check consistency). Editing the observation is
  not enough. LunarLander contacts come from physics, so writing contact flags
  does not establish a reachable pair; expect "uncertified" unless validated.
  A paired audit proves global nonrealizability only, never positive on-policy
  error. A finite learned classifier's error is not a Bayes-error lower bound.
  This conflict existence test has a different scope from the earlier bin
  audit: that audit gave a formal lower confidence bound on disagreement
  under one fixed reference distribution, whereas this test promises no
  distributional lower bound at all.
- Failures are labeled "restriction uncertified" and reported, not dropped.

## 7. Staged process within the deadline

- S0 build and tests: factory identity regression, expert unmasked,
  checkpoint round trip, BC prefix hashes, cap-namespaced caches.
- First step (APPROVED by the user on 2026-09-30; being implemented; not run;
  no result exists), after S0 and before any wider pilot: CartPole-v1 only, one excluded seed 300 (chosen now, not
  previously inspected), 3 methods x {full, restricted} = 6 runs. Restricted
  hides only cart position (x := 0); x_dot, theta, theta_dot unchanged.
  B = 1000 (1000 rounds, 1 sample per round), up to 20 epochs, lr 1e-3, inner
  early stop on, outer early stop off, eval every 10 rounds with 100
  episodes, original 500-step limit, existing qualified expert unchanged.
  Fixed BC: one chronological pool collected once and shared by full and
  restricted, cold prefix fits for the curve. FTL beta 0, warm start false;
  all other settings as in the previous pipeline. Bounds: 2 h per job, at
  most 4 workers, 6 h for the whole stage, and the original hard deadline
  2026-10-07 01:01 UTC. New identity, checkpoint round-trip, and mask-audit
  gates run first. If the audit shows no expert conflicts for x := 0, the
  mask is recorded "restriction uncertified" and we consult before claiming
  agnosticity or replacing it. Goal: measure cost and validate the pipeline
  and the observation effect, not select a design where FTL wins. Cap
  changes may be considered (user allowed), but no exact cap is approved; this
  step keeps the cap unchanged to isolate the mask.
- Candidate main seeds: 3100..3104, disjoint from pilot seed 300 and from the
  earlier classical diagnostic seeds 1000..1019; no effects seen. Nothing
  needs to run to fix these ids.
- S1 wider pilot (conditional future step, only after the first step is
  reviewed and separately approved): all 8 envs, both conditions, 3 methods,
  at the maximum candidate budget where feasible. It measures per-cell
  wall-clock (including BC refits), timeout categories, and optimization
  diagnostics. Pilot results are not reported as evidence and not used for
  any effect-based choice. Neither this 48-run pilot nor the 240-run main
  campaign starts automatically; each needs concrete further approval.
  Numeric rules beyond the first step remain proposals. All 8 envs remain
  the target.
- Hyperparameters: lr 1e-3. Tuning only if optimization diagnostics justify it
  (for example training CE on the aggregated data not decreasing, or divergence),
  not because returns are low. If triggered: grid {3e-4, 1e-3, 3e-3} chosen by a
  neutral excluded supervised-fit calibration (held-out NLL on the common fixed
  expert pool, pilot seeds), one frozen choice shared by all methods and both
  conditions per env. Never selected on FTL return or FTL advantage.
- No gating or dropping: low full-condition return for any method is diagnosed
  (fit: training CE; coverage/generalization: on-policy CE vs expert-pool CE)
  and reported. All 8 envs stay in.
- Budgets: 1000, 2000, 4000 rounds, one shared budget per env for all methods
  and conditions, chosen by the predeclared plateau rule (seed-mean, last 20% vs
  previous 20% of eval points, proposed |change| <= 0.02 normalized return and
  <= 0.01 disagreement, all cells). Plateau is empirical, not class optimality.
  If the pilot shows a budget is infeasible, report "not plateaued at feasible
  budget"; eval cadence is not changed under budget pressure.
- Caps: diagnose timeouts in both conditions, separating successful survival
  at the cap (CartPole), truncated unfinished tasks, and natural termination.
  A cap change is a separate factor, identical for all methods and conditions,
  with expert and random references re-evaluated. If the expert is retrained
  for a cap, features and class change, so all cells at that cap use that
  expert and the factor is disclosed. Longer caps do not fix natural failures.
  No specific cap is approved.

## 8. Analysis

- Primary: original curves (normalized return, rollout CE, disagreement).
- Additive predeclared comparison: paired-by-seed FTL - BC and FTL - BC-iid per
  condition; interaction = restricted minus full, seed-bootstrap CI per env.
- Additive efficiency: rounds to a common predeclared per-env target, the same
  for all methods (not reached is recorded), and AUC over a common budget range.
- Plots: one color per method, dashed full, solid restricted, expert line.

## 9. Decisions needing approval

Already approved: the concrete first CartPole step in section 7, including
its `x := 0` mask (six runs, not yet run). Still needing approval:

1. Candidate masks in section 5 other than the pilot's CartPole mask (choice
   and strength per env).
2. After the first CartPole step is reviewed, separately: wider-pilot
   coverage and seeds, candidate main seeds 3100..3104, and the go/no-go
   runtime rule for the 240-run campaign.
3. Numeric rules: plateau thresholds, lr-tuning trigger, per-env efficiency
   target, and cap candidates the timeout diagnosis may consider.
