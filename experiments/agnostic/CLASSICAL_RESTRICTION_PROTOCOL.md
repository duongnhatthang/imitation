# Classical restriction tasks: protocol `agnostic-classical-restriction/1`

The CartPole pilot protocol `/2` (`CARTPOLE_STRONGER_MASK_PROTOCOL.md`) is
unchanged: `restriction_pilot` keeps it as the default `RunnerSpec`
(`CARTPOLE_V2`), with the same records, checks, restrictions and CLI.

## What is the same as `/2`

The generic runner (`task_pilot`) calls the same `restriction_pilot`
functions with an explicit `RunnerSpec`; nothing is copied or monkeypatched.
Data job, fixed BC (one cold fit on the whole pool, one evaluation, a flat
reference), BC-iid (one uniform pre-action state per independent expert
episode, replayed one per round), FTL (learner rollouts, expert labels,
beta 0), cold refits, linear learner with only `action_net` trained and the
mask before the frozen features, lr 1e-3, at most 20 epochs, inner early
stopping on, outer off, `min(32, accumulated)` batches, 100 deterministic
evaluation episodes at 0, 1, every 10 and the last round, the original
normalized return, deadlines, job limits (at most 7200 s) and accounting.
Seed 301 is reserved for audits. Default budget 1000.

## Tasks (`tasks.TASKS`)

| Env | Cap | Observation | Ladder (screened in order) |
|---|---|---|---|
| CartPole-v1 | 500 | 4 floats | `cart_position_angular_velocity_zero`, `cart_position_velocity_angular_velocity_zero` |
| FrozenLake-v1 (4x4, not slippery) | 100 | one-hot 16 | `frozenlake_column_pairs`, `frozenlake_column_hidden`, `frozenlake_column_hidden_row_pairs` |
| CliffWalking-v0 | 200 (fallback) | one-hot 48 | `cliffwalking_column_pairs`, `cliffwalking_column_hidden`, `cliffwalking_column_hidden_row_pairs` |
| Acrobot-v1 | 500 | 6 floats | `acrobot_theta2_dot_zero`, `acrobot_theta_dots_zero`, `acrobot_theta2_and_dots_zero` |
| MountainCar-v0 | 200 | 2 floats | `mountaincar_velocity_zero`, `mountaincar_position_zero` (alternatives, no order of severity) |
| Taxi-v3 | 200 | one-hot 500 | `taxi_destination_hidden`, `taxi_destination_passenger_hidden` |
| Blackjack-v1 (sab, no natural) | 20 (fallback) | one-hot 32+11+2 | `blackjack_usable_ace_hidden`, `blackjack_usable_ace_dealer_hidden` |
| LunarLander-v2 | 1000 | 8 floats | `lunarlander_angular_velocity_zero`, `lunarlander_velocities_zero`, `lunarlander_velocities_angle_zero` |

The cap is read from the original factory `env_utils.make_env` (exactly one
TimeLimit); the registered cap is None for CliffWalking and Blackjack.
`tasks.describe` refuses if the factory disagrees with the declaration and
records the registry kwargs, configured kwargs, raw and effective caps,
wrapper chain, unwrapped flags and, for tabular tasks, a digest of `env.P`.
Every data and run record carries this description and its digest, and data
checks compare them.

## Restrictions

Zeroing rungs set the listed learner features to 0 (for Acrobot both
`cos th2` and `sin th2`); the simulator and the expert are not changed.

Categorical rungs map a state index to the smallest-index reachable
nonterminal state of its class (the visible factors), then re-encode it. A
class with no reachable nonterminal state keeps identity. Reachability is a
breadth-first search over `env.P` from the reset support (FrozenLake,
CliffWalking, Taxi) or an exact enumeration of the Blackjack draw process; it
means reachable under some action sequence. Tables are rebuilt on use and
must match `tasks.FROZEN_TABLE_SHA256`. Inputs that are not one-hot in every
block are refused. Alias checkpoints save the table digest and refuse a
mismatch on load. A non-identity rung that leaves a single distinct
observation on the data job's pool is refused.

## Experts

An expert directory holds exactly one of:

- `preparation.json` from `run_classical prepare-expert`, which now accepts
  all eight tasks (fallback caps included) with the original trainer,
  thresholds and qualification;
- `candidate_qualification.json` from `task_pilot qualify-candidate`: an
  existing checkpoint, copied and pinned by digest, re-measured with the
  original qualification (fresh deterministic episodes, same phase RNG, step
  cap and frozen per-environment threshold). Its training provenance is
  recorded as unknown. A failing candidate is kept; nothing is retried.

Loading fails closed for both. The record is parsed from the bytes that are
hashed. Every requested qualification episode must be present with a finite
return and a length within the cap, the steps must fit the budget, the
evaluator counters must report a completed evaluation that matches the
episodes, and the reported mean, threshold and reset seed must equal the
recomputed mean and the config. Quality (the per-environment threshold and the
global 0.80 check) is computed from the recomputed mean and the current
frozen threshold, never read from the record.

- A candidate's config must be rebuilt exactly by the current source (task
  digest, cap, cap rule, frozen thresholds), and its package versions must
  equal the current runtime's, so qualification runs where the jobs run.
- A preparation must have no convergence override, the frozen threshold and
  the current cap, a config rebuilt exactly by the current
  `build_preparation_config`, a consistent source map whose environment,
  rollout, evaluation and training files (`rollouts.py`, `env_baselines.py`,
  `env_utils.py`, `eval_utils.py`, `expert_training.py`, `util.py`) equal the
  current bytes, and the current gymnasium version. Other package versions
  and the preparation driver's own source may differ, so the historical
  CartPole and Acrobot preparations still load.

Data and run jobs refuse an expert that passes only the per-environment
threshold (`relaxed_only`) unless `--allow-relaxed-expert` is given, which is
recorded.

## References

By default a run normalizes with its own data job's baselines (as in `/2`).
`--reference-data-dir` with `--expected-reference-sha256` uses the baselines
of another data job. The pin selects exact bytes; the record must also be
complete, carry exactly the generic protocol and data schema, the current
task description and digest and the current package versions, have a
consistent config (digest, rules, a valid non-audit seed equal to the
record's) and source map (combined digest, the protocol's file set), come from
the same expert checkpoint and policy state, and hold complete baseline
evidence (500 expert and 500 random episodes in config and costs, consistent
step counts). Its seed may differ from the run's. For CartPole only, the
pilot data pair (protocol `agnostic-cartpole-restriction-pilot/1`, schema
`agnostic-cartpole-restriction-pilot/1/data`, the data job that `/2` reuses
unchanged) is also accepted, bound by that pair, the 500 step cap and the
pilot's file set, so its reference can be kept. The reference provenance is
recorded in the run config.

## Audits

`task_pilot audit` records the evidence type and its limits:

- `reachable_enumeration` (FrozenLake, CliffWalking, Taxi, Blackjack): every
  reachable nonterminal state is labelled and grouped by masked encoding.
- `constructed_unwrapped_state` (Acrobot, MountainCar): base states from one
  expert trajectory; one simulator coordinate feeding only hidden features is
  replaced by predeclared grid values; states must be inside bounds,
  nonterminal, and match the simulator's own observation map. Valid, not
  proven reachable.
- `cartpole_x_grid` (CartPole): the existing x-grid audit.
- `unverified` (LunarLander): no witness is attempted; the agnostic claim is
  held.

A conflict shows only that expert labels conflict under the mask; it is not
an error floor on any on-policy distribution. Low BC return is neither a
certificate nor a return bound.

## Planned analysis: coverage maps (postprocessing only)

Coverage maps for masked offline BC, BC-iid and FTL will use the retained
1000-label training datasets, shown in both the original state view and the
masked-input view, with one shared projection and shared axes across methods.
Offline BC is one fixed dataset, not a growing prefix. Generic offline BC
writes no scratch demos, so an adapter is still required that reads its
pinned pool. Saved samples do not represent entire visitation trajectories or
final-policy occupancy. This postprocessing plan authorizes no new training
or evaluation, and coverage or projection overlap does not establish a return
ceiling or an error floor.

## Not implemented here

Screening decisions (0.10 drop and two standard errors), freeze manifests,
plots and the eight-row report are not part of this code.
