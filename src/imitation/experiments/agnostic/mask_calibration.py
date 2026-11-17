"""Outcome-aware calibration of observation masks (pure functions).

A mask is worth running when it is scientifically eligible (the restricted
policy class cannot reach the full optimum, yet can do better than the lower
reference) and the four imitation methods show it in practice: at least one
method is measurably degraded, the masked methods are not all collapsed to
the floor or to the uniform-random reference, and they are not all at the
expert reference. This module decides that from declared evidence and never
runs anything: no training, no retries, no exact solvers, and no automatic
promotion of another mask.

Frozen protocol (`PROTOCOL`, version /6)
----------------------------------------
/6 was frozen before any calibration run under it; no calibration was
launched under /2, /3, /4, /5 or /6 before. Reports carry the protocol that
produced them, and an earlier receipt is never reinterpreted under /6. The
numerical constants and the uniform-random normalization are unchanged since
/3 (the constants since /1), the random comparison since /5, and the seeded
random binning and legacy certificate rules since /4.

/6 differs from /5 only in binding metadata (see ``Random-offset grid
binning`` below):

* Evidence whose ``mask`` is a random-offset grid (`tasks.RANDOM_GRID_SPECS`)
  must carry a ``binning`` of kind `SEEDED_RANDOM_GRID` with exactly the
  keys `GRID_BINNING_FIELDS`. It is checked, with its types, against the
  static grid registry and the frozen descriptor digest, and its
  ``restriction_sha256`` against the digest of the restriction record rebuilt
  from the frozen tables; no environment is built. Under /5 such evidence
  needed no binding, which is why the version changes.
* A grid mask carries no certificate: its eligibility stays
  `ELIGIBILITY_UNKNOWN`, never a known ceiling. No point optimum is required
  or implied. A certificate on a grid mask is invalid evidence.

/5 differs from /4 only in the random comparison (review F1: a Blackjack map
whose masked methods all collapsed to the random return, but not to the
lowest payout, was accepted under /4):

* ``random_comparison`` is computed on every arm of every environment, not
  only on return objectives without a supported floor.
* Every environment outside `GOAL_SUCCESS_ENVS` rejects as
  `NO_BETTER_THAN_RANDOM` when every masked method is at or below the
  uniform-random reference, whether or not its floor is known.
  `GOAL_SUCCESS_ENVS` keep the goal-success criterion alone, since the far
  random reference compresses their normalized means.
* On environments with a known floor basis, a method indistinct from random
  is a flag only; the `INCONCLUSIVE_NOISE` rule for no method above random is
  kept only where the floor is unknown (CartPole, LunarLander).

/4 differs from /3 only in scientific-eligibility metadata (see ``Seeded
random binning`` below):

* A certificate may declare ``binning_kind`` `SEEDED_RANDOM_BINNING`: the
  mask is a predeclared seeded random state binning (`tasks.RANDOM_ALIAS_SPECS`),
  a different generator from the ordinary aligned grid, so its
  ``ordinary_binning`` is absent or False and never makes it ineligible. Its
  binding to the static registry is checked, never trusted from the kind.
* Certificates without ``binning_kind`` are legacy ordinary-grid certificates
  and keep exactly their /3 meaning: ``ordinary_binning`` False is
  ineligible, and CliffWalking needs it explicit. Any other kind is invalid.
* A certificate's ``lower_reference`` may be labelled by
  ``lower_reference_kind`` (required on seeded random certificates); reports
  name it ``lower_benchmark``, apart from the normalization reference.

/3 differs from /2 in four places:

* Each row must record ``evaluation_deterministic: True``, the flag the
  production runner (`run_experiment._compute_round_eval`) persists. The
  evidence-level declaration cannot stand in for it.
* Only uniform-random normalization references are accepted, on every MDP
  including CliffWalking; /2 also accepted a stalled-policy reference there.
* On `GOAL_SUCCESS_ENVS` the near-reference rule reads masked goal success,
  not the normalized mean, which the far uniform-random reference compresses
  toward 1 (see ``near_reference``).
* Return objectives without a supported floor (CartPole, LunarLander) reject
  as `NO_BETTER_THAN_RANDOM` when every masked method is at or below the
  uniform-random reference.

/2 replaced /1, which required every method to pass the effect gate and
rejected on any single compressed method; /1 could reject a mask only because
one method (say FTL) stayed robust, which selects benchmarks against an
informative result.

* Evidence: every one of `REQUIRED_METHODS` must have a final ``full`` and
  ``masked`` row for every fixed selection seed, each with at least
  `MIN_EPISODES` episodes of raw outcome evidence. The evidence must declare
  ``deterministic_evaluation: True``; an absent declaration is missing
  evidence and is never imputed, and any other value is invalid. Every row
  must also carry ``evaluation_deterministic``: absent (or None) is missing,
  anything but True is invalid. A legacy row alias ``deterministic_evaluation``
  may remain only when it is also True; it never replaces the row flag.
* Effect per method: the full-minus-masked gap, averaged over seeds. Seeds
  are fixed, so its standard error is evaluation-episode noise:
  ``SE = sqrt(sum_s se_s**2) / S`` with ``se_s`` from paired per-episode
  differences when pairing is declared (identical ``evaluation_episode_seeds``
  required), else Welch's unpaired variance. Pairing is never inferred.
* Per-method gate (`effect_gate`): ``effect >= MIN_NORM_GAP`` (inclusive,
  with a `GAP_TOLERANCE` float guard) else `INSUFFICIENT_EFFECT`; then
  ``raw_gap > SIGNIFICANCE_SE * SE`` (strict) else `INCONCLUSIVE_NOISE`.
* Detection (aggregate): the mask shows an effect when at least one method
  passes its gate (``passing_methods``). A method that fails is the flag
  ``insufficient_effect:<method>`` or ``inconclusive_noise:<method>``, never
  a reason by itself, so a robust method cannot disqualify an informative
  mask. When no method passes, the reason is `INSUFFICIENT_EFFECT` if every
  method's gate is insufficient effect (rejected), else `INCONCLUSIVE_NOISE`.
* Effect scale: reference-normalized return gap,
  ``raw_gap / (expert_return - random_return)``. CliffWalking-v0
  (`GOAL_SUCCESS_ENVS`) instead uses the goal-success rate reduction (success
  is ``goal_terminated``), so its gate does not depend on the lower
  reference. Reference normalization is an affine rescaling, not an
  optimal-policy ceiling, and reference uncertainty is reported separately,
  never folded into the SE.
* Lower reference: ``reference.reference_type`` must be `REFERENCE_RANDOM`
  (the default when absent): ``random_return`` is the measured return of the
  uniform-random policy. Any other type, including the retired
  `REFERENCE_STALLED`, is invalid evidence on every environment; a stalled
  return is never read as a random return. Values are reported as given,
  never rewritten.
* All near the reference (``near_reference``, criterion
  ``near_reference_criterion``): `MASKED_NEAR_REFERENCE` rejects when every
  masked method is at least `NEAR_OPTIMAL` on the environment's criterion
  scale: goal-success rate on `GOAL_SUCCESS_ENVS`, normalized mean
  elsewhere. On `GOAL_SUCCESS_ENVS` the normalized means are still reported
  (``normalized_all_near``); when they are all at least `NEAR_OPTIMAL` while
  goal success is not, the flag `REFERENCE_COMPRESSION` records that the
  uniform-random reference compresses partial goal success toward 1.
* Floor (per masked method, the flag ``masked_floor:<method>``) rests on a
  declared basis (``floor_basis``). Goal-only objectives
  (`GOAL_OBJECTIVE_RULES`): constant returns or all-timeout episodes, with
  no successes. Declared lowest payouts (`LOWEST_PAYOUT_RETURNS`, Blackjack
  -1): every episode at that payout; a constant 0 or +1 is not a floor.
  Return objectives (CartPole, LunarLander) have no supported floor:
  ``floor`` is None and the flag `FLOOR_UNKNOWN` is raised. Constant
  returns alone never make a floor (a policy that always balances CartPole
  for 195 steps is partial competence, not failure). `FLOOR_COLLAPSE`
  rejects only when every masked method is at a supported floor. On
  `GOAL_SUCCESS_ENVS` it also rejects when every masked method has zero goal
  success, even with varied returns. No masked method with partial goal
  success (above 0 and below its control) is the flag
  ``no_partial_goal_success``, not a reason.
* Random comparison (``random_comparison`` of every arm). This compares
  against the measured uniform-random return; it claims no floor.
  `AT_OR_BELOW_RANDOM` when the normalized mean is at most `GAP_TOLERANCE`;
  `ABOVE_RANDOM` when it exceeds `SIGNIFICANCE_SE` times its episode-noise
  SE (``normalized_se``, combined over seeds like the gate SE); else
  `INDISTINCT_FROM_RANDOM`. A masked method not above random is the flag
  ``<comparison>:<method>``. Outside `GOAL_SUCCESS_ENVS`,
  `NO_BETTER_THAN_RANDOM` rejects when every masked method is at or below
  random. Only without a supported floor (CartPole, LunarLander), when none
  is above random but some is indistinct, the reason is
  `INCONCLUSIVE_NOISE`; with a known floor basis that is only the flag.
* Normalization compression: on goal-only objectives
  (`GOAL_OBJECTIVE_RULES`: CliffWalking, Taxi, FrozenLake, MountainCar,
  Acrobot), a masked method with zero successes whose normalized mean is
  still at least `MIN_NORM_GAP` is the flag
  ``normalization_compressed:<method>`` plus
  ``success_rate_presentation_required``: its results must be shown as
  success rates. It never rejects alone. Return objectives (CartPole,
  LunarLander) are exempt.
* Degenerate SE: a method whose gate SE is exactly 0 (every evaluation
  episode within each seed identical, as with one start state and
  deterministic dynamics and evaluation) is the flag
  ``degenerate_se:<method>``; its real sample size is the number of seeds.
  ``single_start`` is True when every row's ``initial_state_digests`` are one
  digest, None when any row lacks them. The between-seed SE and the per-seed
  success summary (``seeds_with_success`` of ``seeds``) are reported. The
  gate itself is unchanged.
* Full control: ``full_control.verified is True``, or a shared
  ``checkpoint_sha256`` with ``return_quality_passed is True`` (every row
  carrying ``expert_sha256`` must match it). Each full arm must also show
  outcome evidence of competence: no floor, and nonzero success on goal-only
  objectives. A return-objective full arm without success is flagged
  ``full_arm_zero_success:<method>``, not disqualified. No NLL thresholds are
  applied; a present NLL must only be finite.

Scientific eligibility (``eligibility``, outcome independent)
-------------------------------------------------------------
An optional certificate from an exact analysis outside this module (for
example exhaustive search over deterministic memoryless policies, or exact
DP). The module only checks it and never derives a ceiling from observed
returns. Fields: ``verified``, ``sha256`` (certificate digest),
``mask_sha256`` (equals the mask's), ``metric`` (`ELIGIBILITY_METRICS`),
``policy_class``, ``method``, ``full_optimum``, ``best_in_class``,
``lower_reference``, optional ``goal_feasible``, ``ordinary_binning`` and
``lower_reference_kind`` (`LOWER_REFERENCE_KINDS`), and for seeded random
binnings ``binning_kind`` and ``binning``. Status
(``scientific_eligibility.status``):

* `ELIGIBILITY_UNKNOWN` when absent, `ELIGIBILITY_UNVERIFIED` when not
  ``verified is True``. Neither blocks a mask, except where required below;
  the positive witness (``witness_verified``) is still required.
* Rejected as `SCIENTIFICALLY_INELIGIBLE`: a legacy certificate with
  ``ordinary_binning`` False (bins must come from the declared aligned
  ladder), ``goal_feasible`` False on a goal-only objective,
  ``best_in_class <= lower_reference`` (at the floor), or
  ``best_in_class >= full_optimum`` (realizable).
* Otherwise `ELIGIBILITY_ELIGIBLE`, with the normalized class gap
  ``(full_optimum - best_in_class) / (full_optimum - lower_reference)``;
  a class gap below `MIN_NORM_GAP` is reported as ``small_class_gap``.
* `ELIGIBILITY_REQUIRED_ENVS` (CliffWalking-v0) must carry a verified
  certificate with ``policy_class`` `DETERMINISTIC_MEMORYLESS`, an explicit
  ``goal_feasible``, and either ``ordinary_binning`` True (legacy) or a
  validated seeded random binding; otherwise the reason is
  `ELIGIBILITY_UNVERIFIED` (inconclusive). Row pairs (best in class -15
  against the full optimum -13) qualify with a small class gap; column pairs
  (goal infeasible) are rejected.

Lower benchmark versus normalization reference. ``lower_reference`` is the
certificate's structural benchmark for the class gap and the at-floor rule
only (reported as ``scientific_eligibility.lower_benchmark`` with its
``kind``). It never normalizes returns: normalized scores always use the
evidence ``reference`` (uniform random 0, expert 1). On CliffWalking the
certificate uses the stall benchmark -200 (-1 per step to the cap,
`LOWER_STALL_BENCHMARK`), while the uniform-random normalization reference
is about -2113.

Seeded random binning (``binning_kind`` `SEEDED_RANDOM_BINNING`)
-----------------------------------------------------------------
The certificate and the evidence each carry the same ``binning`` mapping with
exactly the keys `BINNING_FIELDS`; `seeded_random_binning` builds it::

    {"kind": "seeded_random", "generator": "random_state_binning/1",
     "restriction_id": "<evidence mask, a tasks.RANDOM_ALIAS_SPECS key>",
     "env_name": "<evidence env_name, the spec's env>",
     "partition_seed": int, "n_bins": int,       # the spec's seed and count
     "partition_sha256": "<tasks.random_partition_sha256(id)>",
     "frozen_table_sha256": "<tasks.FROZEN_TABLE_SHA256[id]>",
     "restriction_sha256": "<evidence mask_sha256>"}

Every value is compared, with its type, against the static registry (no
environment, reachable set or expert is built); the certificate's
``ordinary_binning``, if present, must be False, and its
``lower_reference_kind`` is required. Evidence whose ``mask`` is a random
binning must carry ``binning``; evidence or a certificate with a binding on
any other mask is invalid, as is a legacy certificate on a random mask.
Every failure is `INVALID_EVIDENCE` (status `ELIGIBILITY_INVALID` for a
certificate). ``restriction_sha256`` binds the actual restriction record
digest the runs used; it equals ``mask_sha256`` and every masked row's, but
this module cannot recompute it without rebuilding the restriction.

Random-offset grid binning (``binning`` kind `SEEDED_RANDOM_GRID`)
-------------------------------------------------------------------
The evidence of a grid mask carries ``binning`` with exactly the keys
`GRID_BINNING_FIELDS`; `seeded_random_grid_binning` builds it::

    {"kind": "seeded_random_grid", "generator": "random_offset_grid/1",
     "restriction_id": "<evidence mask, a tasks.RANDOM_GRID_SPECS key>",
     "env_name": "<evidence env_name, the spec's env>",
     "seed": int, "n_bins": int,                # the spec's seed and K
     "grid_descriptor_sha256": "<random_grid.FROZEN_GRID_SHA256[id]>",
     "restriction_sha256": "<evidence mask_sha256>"}

``n_bins`` is the per-axis count ``K`` (``K + 1`` cells on linear and
unbounded axes, ``K`` arcs on angle pairs, binary contacts unchanged). Every
value is compared, with its type, against the static registry. The grid is
rebuilt from the frozen tables by `random_grid.get_grid`, which refuses a
descriptor that differs from its frozen digest, and ``restriction_sha256``
must equal `restriction_record_sha256` of that rebuilt record (the digest
the runs and every masked row carry). No environment, Box or expert is
built; the actual Box bits are checked by the runs, not here. Evidence with a
grid binding on any other mask, and a grid mask without one, are invalid.

A grid mask has no certificate kind under /6, so its eligibility is
`ELIGIBILITY_UNKNOWN`: a label-conflict witness on constructed states is not
a ceiling. Extension point: a future proof that only bounds the class value
(an interval, for example an initial-step alias loss) needs its own
certificate kind with interval fields, kept apart from ``best_in_class``,
which always means an exact point optimum of the class.

Scientific eligibility and practical detection are separate. A mask that is
eligible but rejected only because every masked method is near the reference
keeps its full outcome report and is marked ``benign_agnostic``: agnostic
with small regret, not a reason to search for another mask.

Decision: `ACCEPTED` only with a qualified control, a verified witness,
complete valid evidence, no required eligibility missing, at least one method
passing its gate, and no reject reason. `REJECTED` when any of
`REJECT_REASONS` holds. Otherwise `INCONCLUSIVE` (missing or invalid
evidence, noise, unqualified control, unverified witness or eligibility).

Evidence schema (`calibrate_mask`)
----------------------------------
::

    {
      "env_name": "CliffWalking-v0",
      "mask": "<restriction id>",
      "mask_sha256": "<64 hex>",
      "witness_verified": True,
      "deterministic_evaluation": True,     # required, never imputed
      "selection_seeds": [0, 1, 2],
      "pairing": "paired" | "unpaired",     # absent means unpaired
      "full_control": {"verified": True}
                    | {"checkpoint_sha256": "<64 hex>",
                       "return_quality_passed": True},
      "reference": {"expert_return": float, "random_return": float,
                    "reference_type": "random_policy",  # absent means this
                    "uncertainty": None | {name: finite >= 0}},  # key required
      "eligibility": {                      # optional except on Cliff
          "verified": True, "sha256": "<64 hex>", "mask_sha256": "<64 hex>",
          "metric": "return" | "goal_success",
          "policy_class": "deterministic_memoryless",
          "method": "exhaustive_search",
          "full_optimum": float, "best_in_class": float,
          "lower_reference": float,
          "lower_reference_kind": "stall_benchmark"
                                | "uniform_random_exact_value",
          "goal_feasible": bool, "ordinary_binning": bool,
          "binning_kind": "seeded_random",  # absent on legacy certificates
          "binning": {...},                 # seeded random only, as above
      },                                    # never on a grid mask
      "binning": {...},                     # required on a random or grid mask
      "rows": [{
          "method": "ftl" | "bc" | "bc_iid" | "bc_prefix",
          "arm": "full" | "masked",
          "seed": int,
          "final": True,                    # placeholders are missing
          "mask_sha256": "<64 hex>",        # masked rows, equals the mask's
          "expert_sha256": "<64 hex>",      # optional, shared checkpoint
          "evaluation_deterministic": True,  # required, the runner's flag
          "deterministic_evaluation": True,  # legacy alias, must be True
          "episode_returns": [float],       # raw, unclipped
          "episode_lengths": [int],
          "episode_terminated": [bool],
          "episode_truncated": [bool],
          "episode_successes": [bool],      # when the env's success is used
          "evaluation_episode_seeds": [int],  # required when paired
          "initial_state_digests": ["<64 hex>"],  # optional
          "nll": float,                     # optional, must be finite
      }, ...],
    }

`eval_utils.episode_evidence` produces the per-row episode fields.
"""

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from imitation.experiments.agnostic import pilot, random_grid, restriction, tasks
from imitation.experiments.ftrl import eval_utils

PROTOCOL = "agnostic-mask-calibration/6"

MIN_NORM_GAP = 0.10
SIGNIFICANCE_SE = 2.0
NEAR_OPTIMAL = 0.95
MIN_EPISODES = 100
GAP_TOLERANCE = 1e-9
REQUIRED_METHODS = ("ftl", "bc", "bc_iid", "bc_prefix")
ARMS = ("full", "masked")

ACCEPTED = "accepted"
REJECTED = "rejected"
INCONCLUSIVE = "inconclusive"

CONTROL_UNQUALIFIED = "control_unqualified"
WITNESS_UNVERIFIED = "witness_unverified"
ELIGIBILITY_UNVERIFIED = "eligibility_unverified"
MISSING_OUTCOMES = "missing_outcomes"
INVALID_EVIDENCE = "invalid_evidence"
INSUFFICIENT_EFFECT = "insufficient_effect"
INCONCLUSIVE_NOISE = "inconclusive_noise"
FLOOR_COLLAPSE = "floor_collapse"
MASKED_NEAR_REFERENCE = "masked_near_reference"
NO_BETTER_THAN_RANDOM = "no_better_than_random"
SCIENTIFICALLY_INELIGIBLE = "scientifically_ineligible"
REJECT_REASONS = frozenset(
    {
        INSUFFICIENT_EFFECT,
        FLOOR_COLLAPSE,
        MASKED_NEAR_REFERENCE,
        NO_BETTER_THAN_RANDOM,
        SCIENTIFICALLY_INELIGIBLE,
    },
)
# Flag prefix (``normalization_compressed:<method>``); never a reason.
NORMALIZATION_COMPRESSED = "normalization_compressed"
SUCCESS_RATE_PRESENTATION_REQUIRED = "success_rate_presentation_required"
DEGENERATE_SE = "degenerate_se"
# Flag: normalized means all near 1 while goal success is not.
REFERENCE_COMPRESSION = "reference_compression"

# ``random_comparison`` of every arm (also the flag prefix
# ``<comparison>:<method>`` when a masked arm is not above random).
ABOVE_RANDOM = "above_random"
AT_OR_BELOW_RANDOM = "at_or_below_random"
INDISTINCT_FROM_RANDOM = "indistinct_from_random"

NORMALIZED_RETURN = "normalized_return"
GOAL_SUCCESS = "goal_success"
# Environments whose effect is the raw goal-success reduction.
GOAL_SUCCESS_ENVS = frozenset({"CliffWalking-v0"})
# Success rules of goal-only objectives: every non-goal episode is a failure,
# so a high normalized score without any success is compression, not partial
# competence. Return-objective envs (CartPole, LunarLander) are excluded.
GOAL_OBJECTIVE_RULES = frozenset({"goal_terminated", "terminal_reward_positive"})

# What an arm's ``floor`` rests on (``floor_basis``). Goal-only objectives:
# constant returns or all timeouts, with zero goal success. Declared lowest
# payouts: every episode returns exactly the lowest possible return. Any other
# objective has no supported floor: ``floor`` is None (unknown, flag
# `FLOOR_UNKNOWN`), never inferred from constant returns.
FLOOR_GOAL_FAILURE = "goal_failure"
FLOOR_LOWEST_PAYOUT = "lowest_payout"
FLOOR_UNKNOWN = "floor_unknown"
LOWEST_PAYOUT_RETURNS = {"Blackjack-v1": -1.0}  # a loss; natural=False

# ``reference.reference_type``: what ``random_return`` holds. Absent means
# `REFERENCE_RANDOM`, the only accepted type. `REFERENCE_STALLED` was accepted
# on CliffWalking under /2 and is named only to refuse it explicitly.
REFERENCE_RANDOM = "random_policy"
REFERENCE_STALLED = "stalled_policy"
REFERENCE_TYPES = (REFERENCE_RANDOM,)

# Scientific eligibility certificates.
ELIGIBILITY_METRICS = ("return", "goal_success")
DETERMINISTIC_MEMORYLESS = "deterministic_memoryless"
ELIGIBILITY_REQUIRED_ENVS = frozenset({"CliffWalking-v0"})
ELIGIBILITY_ELIGIBLE = "eligible"
ELIGIBILITY_UNKNOWN = "unknown"
ELIGIBILITY_UNVERIFIED = "unverified"
ELIGIBILITY_INVALID = "invalid"
ELIGIBILITY_NOT_ORDINARY = "not_ordinary_binning"
ELIGIBILITY_GOAL_INFEASIBLE = "goal_infeasible"
ELIGIBILITY_AT_FLOOR = "at_floor"
ELIGIBILITY_REALIZABLE = "realizable"
INELIGIBLE_STATUSES = frozenset(
    {
        ELIGIBILITY_NOT_ORDINARY,
        ELIGIBILITY_GOAL_INFEASIBLE,
        ELIGIBILITY_AT_FLOOR,
        ELIGIBILITY_REALIZABLE,
    },
)

# Certificate ``binning_kind``: the only declared kind. A certificate without
# one is a legacy ordinary-grid certificate (reported as
# `LEGACY_ORDINARY_BINNING`); any other value is invalid.
SEEDED_RANDOM_BINNING = "seeded_random"
BINNING_KINDS = (SEEDED_RANDOM_BINNING,)
LEGACY_ORDINARY_BINNING = "legacy_ordinary"
# Exact keys of a seeded random ``binning`` (see `seeded_random_binning`).
BINNING_FIELDS = (
    "kind",
    "generator",
    "restriction_id",
    "env_name",
    "partition_seed",
    "n_bins",
    "partition_sha256",
    "frozen_table_sha256",
    "restriction_sha256",
)
_BINNING_INT_FIELDS = ("partition_seed", "n_bins")
# Evidence ``binning`` kind of a random-offset grid mask (see
# `seeded_random_grid_binning`). It is not a certificate kind: grid masks
# carry no certificate under /6.
SEEDED_RANDOM_GRID = "seeded_random_grid"
GRID_BINNING_FIELDS = (
    "kind",
    "generator",
    "restriction_id",
    "env_name",
    "seed",
    "n_bins",
    "grid_descriptor_sha256",
    "restriction_sha256",
)
_GRID_INT_FIELDS = ("seed", "n_bins")

# Certificate ``lower_reference_kind``: what its ``lower_reference`` is. It is
# the eligibility benchmark only, never the normalization reference.
LOWER_STALL_BENCHMARK = "stall_benchmark"
LOWER_UNIFORM_RANDOM_VALUE = "uniform_random_exact_value"
LOWER_REFERENCE_KINDS = (LOWER_STALL_BENCHMARK, LOWER_UNIFORM_RANDOM_VALUE)
LOWER_BENCHMARK_ROLE = (
    "certificate lower benchmark for the class gap and the at-floor rule; "
    "not the normalization reference (normalized scores use the uniform-random "
    "reference as 0 and the expert as 1)"
)

EPISODE_FIELDS = (
    "episode_returns",
    "episode_lengths",
    "episode_terminated",
    "episode_truncated",
    "episode_successes",
    "evaluation_episode_seeds",
    "initial_state_digests",
)
NLL_KEYS = ("nll", "val_nll", "final_nll")
NORMALIZATION_NOTE = (
    "Normalized scores are an affine map through expert and random reference "
    "returns, not an optimal-policy ceiling; values near or above 1 do not "
    "mean optimal."
)
BENIGN_NOTE = (
    "Scientifically eligible (the class cannot reach the full optimum) but "
    "every masked method is near the expert reference: report as agnostic "
    "with small regret, do not search for another mask."
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _base_env(env_name: str) -> str:
    return env_name.rsplit("/", 1)[-1]


def success_used(env_name: str) -> Optional[bool]:
    """Whether calibration uses success for ``env_name`` (None if unknown)."""
    rule = eval_utils.success_rule_for(env_name)
    return None if rule is None else rule.used_for_calibration


def goal_objective(env_name: str) -> bool:
    """Whether ``env_name`` scores only goal success (see `GOAL_OBJECTIVE_RULES`)."""
    rule = eval_utils.success_rule_for(env_name)
    return rule is not None and rule.name in GOAL_OBJECTIVE_RULES


def floor_basis(env_name: str) -> Optional[str]:
    """What an outcome floor rests on for ``env_name`` (None when unsupported)."""
    if goal_objective(env_name):
        return FLOOR_GOAL_FAILURE
    if _base_env(env_name) in LOWEST_PAYOUT_RETURNS:
        return FLOOR_LOWEST_PAYOUT
    return None


def criterion_for(env_name: str) -> str:
    """Effect scale of ``env_name``: `GOAL_SUCCESS` or `NORMALIZED_RETURN`."""
    return (
        GOAL_SUCCESS if _base_env(env_name) in GOAL_SUCCESS_ENVS else NORMALIZED_RETURN
    )


def missing_outcome_fields(
    row: Mapping[str, Any],
    env_name: str,
    paired: bool = False,
) -> List[str]:
    """Outcome fields ``row`` lacks for calibration on ``env_name``.

    Unknown environments report ``episode_successes`` as missing, since their
    outcome cannot be judged.
    """
    required = [
        "episode_returns",
        "episode_lengths",
        "episode_terminated",
        "episode_truncated",
    ]
    if success_used(env_name) is not False:
        required.append("episode_successes")
    if paired:
        required.append("evaluation_episode_seeds")
    return [key for key in required if row.get(key) is None]


def seeded_random_binning(
    restriction_id: str,
    restriction_sha256: str,
) -> Dict[str, Any]:
    """The ``binning`` of a seeded random restriction, from static data only.

    Reads `tasks.RANDOM_ALIAS_SPECS`, `tasks.random_partition_sha256` and
    `tasks.FROZEN_TABLE_SHA256`; builds no environment and no restriction.

    Args:
        restriction_id: A key of `tasks.RANDOM_ALIAS_SPECS`.
        restriction_sha256: Digest of the actual restriction record the runs
            used (the evidence ``mask_sha256``).

    Returns:
        The mapping with exactly the keys `BINNING_FIELDS`.

    Raises:
        ValueError: If ``restriction_id`` is not a registered random binning
            or ``restriction_sha256`` is not 64 lowercase hex.
    """
    spec = None
    if isinstance(restriction_id, str):
        spec = tasks.RANDOM_ALIAS_SPECS.get(restriction_id)
    if spec is None:
        raise ValueError(f"{restriction_id!r} is not a registered random binning")
    if not _is_sha(restriction_sha256):
        raise ValueError("restriction_sha256 must be 64 lowercase hex")
    return {
        "kind": SEEDED_RANDOM_BINNING,
        "generator": spec.generator,
        "restriction_id": spec.restriction_id,
        "env_name": spec.env_name,
        "partition_seed": spec.seed,
        "n_bins": spec.n_bins,
        "partition_sha256": tasks.random_partition_sha256(spec.restriction_id),
        "frozen_table_sha256": tasks.FROZEN_TABLE_SHA256[spec.restriction_id],
        "restriction_sha256": restriction_sha256,
    }


def restriction_record_sha256(restriction_id: str) -> str:
    """Digest the runs record for a restriction, rebuilt without an environment.

    It is ``pilot.config_digest(restriction.describe_restriction(id))``; for a
    random grid the record is rebuilt from the frozen tables and refused when
    its descriptor differs from `random_grid.FROZEN_GRID_SHA256`.

    Raises:
        ValueError: If ``restriction_id`` is not registered.
        RuntimeError: If a frozen descriptor or table digest differs.
    """
    return pilot.config_digest(restriction.describe_restriction(restriction_id))


def seeded_random_grid_binning(
    restriction_id: str,
    restriction_sha256: str,
) -> Dict[str, Any]:
    """The ``binning`` of a random-offset grid, from static data only.

    Reads `tasks.RANDOM_GRID_SPECS` and `random_grid.FROZEN_GRID_SHA256` and
    rebuilds the grid from its frozen tables (`random_grid.get_grid`); builds
    no environment.

    Args:
        restriction_id: A key of `tasks.RANDOM_GRID_SPECS`.
        restriction_sha256: Digest of the actual restriction record the runs
            used (the evidence ``mask_sha256``); it must equal
            `restriction_record_sha256` of the rebuilt grid.

    Returns:
        The mapping with exactly the keys `GRID_BINNING_FIELDS`.

    Raises:
        ValueError: If ``restriction_id`` is not a registered grid, its
            rebuilt descriptor differs from the frozen digest, or
            ``restriction_sha256`` is malformed or not the rebuilt digest.
    """
    if not random_grid.is_grid_id(restriction_id):
        raise ValueError(f"{restriction_id!r} is not a registered random grid")
    if not _is_sha(restriction_sha256):
        raise ValueError("restriction_sha256 must be 64 lowercase hex")
    spec = tasks.RANDOM_GRID_SPECS[restriction_id]
    try:
        grid = random_grid.get_grid(restriction_id)
        actual = restriction_record_sha256(restriction_id)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"{restriction_id}: frozen grid does not rebuild: {exc}")
    frozen = random_grid.FROZEN_GRID_SHA256.get(restriction_id)
    if grid.descriptor_sha256 != frozen:
        raise ValueError(f"{restriction_id}: descriptor differs from the frozen one")
    declared = (spec.restriction_id, spec.env_name, spec.seed, spec.n_bins)
    if declared != (
        grid.restriction_id,
        grid.env_name,
        grid.seed,
        grid.n_bins,
    ) or (spec.generator, spec.seed) != (
        tasks.RANDOM_GRID_GENERATOR,
        tasks.RANDOM_GRID_SEED,
    ):
        raise ValueError(f"{restriction_id}: spec differs from the frozen grid")
    if restriction_sha256 != actual:
        raise ValueError(
            "restriction_sha256 is not the digest of the rebuilt restriction record",
        )
    return {
        "kind": SEEDED_RANDOM_GRID,
        "generator": spec.generator,
        "restriction_id": spec.restriction_id,
        "env_name": spec.env_name,
        "seed": spec.seed,
        "n_bins": spec.n_bins,
        "grid_descriptor_sha256": frozen,
        "restriction_sha256": restriction_sha256,
    }


def _is_random_mask(evidence: Mapping[str, Any]) -> bool:
    mask = evidence.get("mask")
    return isinstance(mask, str) and mask in tasks.RANDOM_ALIAS_SPECS


def _is_grid_mask(evidence: Mapping[str, Any]) -> bool:
    return random_grid.is_grid_id(evidence.get("mask"))


def _grid_binning_errors(binning: Mapping, evidence: Mapping[str, Any]) -> List[str]:
    """Why a grid ``binning`` is not the static binding of the evidence mask."""
    if set(binning) != set(GRID_BINNING_FIELDS):
        return [f"grid binning fields must be exactly {list(GRID_BINNING_FIELDS)}"]
    errors = [
        f"grid binning {key} must be an int"
        for key in _GRID_INT_FIELDS
        if not _is_int(binning[key])
    ]
    errors += [
        f"grid binning {key} must be a string"
        for key in GRID_BINNING_FIELDS
        if key not in _GRID_INT_FIELDS and not isinstance(binning[key], str)
    ]
    if errors:
        return errors
    if binning["restriction_id"] != evidence.get("mask"):
        errors.append("grid binning restriction_id is not the evidence mask")
    if binning["env_name"] != evidence.get("env_name"):
        errors.append("grid binning env_name is not the evidence env_name")
    try:
        expected = seeded_random_grid_binning(
            binning["restriction_id"],
            evidence.get("mask_sha256"),
        )
    except ValueError as exc:
        return errors + [f"grid binning: {exc}"]
    errors += [
        f"grid binning {key} {binning[key]!r} is not the static {expected[key]!r}"
        for key in GRID_BINNING_FIELDS
        if binning[key] != expected[key]
    ]
    return errors


def _binning_errors(binning: Any, evidence: Mapping[str, Any]) -> List[str]:
    """Why ``binning`` is not the static binding of the evidence mask."""
    if not isinstance(binning, Mapping):
        return ["binning must be a mapping"]
    if isinstance(binning.get("kind"), str) and binning["kind"] == SEEDED_RANDOM_GRID:
        return _grid_binning_errors(binning, evidence)
    if set(binning) != set(BINNING_FIELDS):
        return [f"binning fields must be exactly {list(BINNING_FIELDS)}"]
    errors = [
        f"binning {key} must be an int"
        for key in _BINNING_INT_FIELDS
        if not _is_int(binning[key])
    ]
    errors += [
        f"binning {key} must be a string"
        for key in BINNING_FIELDS
        if key not in _BINNING_INT_FIELDS and not isinstance(binning[key], str)
    ]
    if errors:
        return errors
    if binning["restriction_id"] != evidence.get("mask"):
        errors.append("binning restriction_id is not the evidence mask")
    if binning["env_name"] != evidence.get("env_name"):
        errors.append("binning env_name is not the evidence env_name")
    try:
        expected = seeded_random_binning(
            binning["restriction_id"],
            evidence.get("mask_sha256"),
        )
    except ValueError as exc:
        return errors + [f"binning: {exc}"]
    errors += [
        f"binning {key} {binning[key]!r} is not the static {expected[key]!r}"
        for key in BINNING_FIELDS
        if binning[key] != expected[key]
    ]
    return errors


def _validate_binning(evidence: Mapping[str, Any], problems: "_Problems") -> None:
    """A random or grid mask must carry its static binding; no other mask may."""
    if "binning" not in evidence:
        if _is_random_mask(evidence) or _is_grid_mask(evidence):
            problems.add(
                INVALID_EVIDENCE,
                f"binning metadata is missing for the seeded random mask "
                f"{evidence.get('mask')!r}",
            )
        return
    for error in _binning_errors(evidence["binning"], evidence):
        problems.add(INVALID_EVIDENCE, error)


def effect_gate(effect: float, raw_gap: float, raw_se: float) -> Optional[str]:
    """Gate one method's effect; None passes.

    Args:
        effect: Effect on the criterion scale (normalized gap or success
            reduction).
        raw_gap: The same gap on the scale its SE was computed on.
        raw_se: Standard error of ``raw_gap``.

    Returns:
        `INSUFFICIENT_EFFECT` if ``effect < MIN_NORM_GAP`` (equality passes),
        else `INCONCLUSIVE_NOISE` unless ``raw_gap > SIGNIFICANCE_SE * raw_se``,
        else None.
    """
    if effect < MIN_NORM_GAP - GAP_TOLERANCE:
        return INSUFFICIENT_EFFECT
    if not raw_gap > SIGNIFICANCE_SE * raw_se:
        return INCONCLUSIVE_NOISE
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class _Problems:
    def __init__(self):
        self.items: List[Tuple[str, str]] = []

    def add(self, reason: str, message: str) -> None:
        self.items.append((reason, message))

    def reasons(self) -> List[str]:
        return sorted({reason for reason, _ in self.items})


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value,
        (bool, np.bool_),
    )


def _is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256.match(value))


def _validate_reference(
    evidence: Mapping[str, Any],
    env_name: str,
    problems: _Problems,
) -> None:
    ref = evidence.get("reference")
    if not isinstance(ref, Mapping):
        problems.add(INVALID_EVIDENCE, "reference must be a mapping")
        return
    ref_type = ref.get("reference_type", REFERENCE_RANDOM)
    if ref_type not in REFERENCE_TYPES:
        problems.add(
            INVALID_EVIDENCE,
            f"reference_type {ref_type!r} on {env_name}: only the uniform-random "
            f"reference ({REFERENCE_RANDOM!r}) is accepted",
        )
    expert, random = ref.get("expert_return"), ref.get("random_return")
    if not (_is_finite_number(expert) and _is_finite_number(random)):
        problems.add(INVALID_EVIDENCE, "reference returns must be finite numbers")
    elif not float(expert) > float(random):
        problems.add(INVALID_EVIDENCE, "reference expert_return must exceed random")
    if "uncertainty" not in ref:
        problems.add(
            INVALID_EVIDENCE,
            "reference.uncertainty must be explicit (None when unavailable)",
        )
    else:
        unc = ref["uncertainty"]
        if unc is not None and not (
            isinstance(unc, Mapping)
            and all(_is_finite_number(v) and float(v) >= 0 for v in unc.values())
        ):
            problems.add(
                INVALID_EVIDENCE,
                "reference.uncertainty must be None or finite non-negative values",
            )


def _validate_eligibility(
    evidence: Mapping[str, Any],
    problems: _Problems,
) -> bool:
    """Check the optional eligibility certificate; True when well formed."""
    cert = evidence.get("eligibility")
    if cert is None:
        return True
    if not isinstance(cert, Mapping):
        problems.add(INVALID_EVIDENCE, "eligibility must be a mapping")
        return False
    errors = []
    if _is_grid_mask(evidence):
        errors.append(
            "a random grid mask carries no certificate under /6; its "
            "eligibility stays unknown",
        )
    if not _is_bool(cert.get("verified")):
        errors.append("verified must be a bool")
    if not _is_sha(cert.get("sha256")):
        errors.append("sha256 must be 64 lowercase hex")
    if cert.get("mask_sha256") != evidence.get("mask_sha256"):
        errors.append("mask_sha256 differs from the mask")
    if cert.get("metric") not in ELIGIBILITY_METRICS:
        errors.append(f"metric must be one of {list(ELIGIBILITY_METRICS)}")
    for key in ("policy_class", "method"):
        if not (isinstance(cert.get(key), str) and cert[key]):
            errors.append(f"{key} must be a non-empty string")
    for key in ("goal_feasible", "ordinary_binning"):
        if cert.get(key) is not None and not _is_bool(cert[key]):
            errors.append(f"{key} must be a bool when present")
    random_kind = cert.get("binning_kind") == SEEDED_RANDOM_BINNING
    if "binning_kind" in cert and (
        not isinstance(cert["binning_kind"], str)
        or cert["binning_kind"] not in BINNING_KINDS
    ):
        errors.append("binning_kind must be a known declared kind")
    if random_kind:
        errors += _binning_errors(cert.get("binning"), evidence)
        errors += _binning_errors(evidence.get("binning"), evidence)
        if cert.get("binning") != evidence.get("binning"):
            errors.append("certificate binning differs from evidence binning")
        if "ordinary_binning" in cert and cert["ordinary_binning"] is not False:
            errors.append("seeded random bins cannot claim ordinary_binning")
        if "lower_reference_kind" not in cert:
            errors.append("seeded random certificate needs lower_reference_kind")
    elif "binning" in cert or _is_random_mask(evidence):
        errors.append("legacy certificate cannot bind a seeded random mask")
    if "lower_reference_kind" in cert and (
        not isinstance(cert["lower_reference_kind"], str)
        or cert["lower_reference_kind"] not in LOWER_REFERENCE_KINDS
    ):
        errors.append("lower_reference_kind must be a known lower benchmark")
    values = [cert.get(k) for k in ("full_optimum", "best_in_class", "lower_reference")]
    if not all(_is_finite_number(v) for v in values):
        errors.append("full_optimum, best_in_class, lower_reference must be finite")
    else:
        full, best, lower = (float(v) for v in values)
        if best > full:
            errors.append("best_in_class exceeds full_optimum")
        if not full > lower:
            errors.append("full_optimum must exceed lower_reference")
        if cert.get("metric") == "goal_success" and not all(
            0.0 <= v <= 1.0 for v in (full, best, lower)
        ):
            errors.append("goal_success values must lie in [0, 1]")
    for error in errors:
        problems.add(INVALID_EVIDENCE, f"eligibility: {error}")
    return not errors


def _eligibility(
    evidence: Mapping[str, Any],
    env_name: str,
    well_formed: bool,
    problems: _Problems,
) -> Dict[str, Any]:
    """Scientific eligibility status, adding its reasons to ``problems``."""
    required = _base_env(env_name) in ELIGIBILITY_REQUIRED_ENVS
    cert = evidence.get("eligibility")
    result: Dict[str, Any] = {
        "required": required,
        "status": ELIGIBILITY_UNKNOWN,
        "certificate": None,
        "class_gap": None,
        "small_class_gap": None,
        "binning_kind": None,
        "lower_benchmark": None,
    }
    if cert is None:
        if required:
            problems.add(
                ELIGIBILITY_UNVERIFIED,
                f"{env_name} needs a verified eligibility certificate",
            )
        return result
    if not well_formed:
        result["status"] = ELIGIBILITY_INVALID
        return result
    result["certificate"] = dict(cert)
    random_kind = cert.get("binning_kind") == SEEDED_RANDOM_BINNING
    result["binning_kind"] = (
        SEEDED_RANDOM_BINNING if random_kind else LEGACY_ORDINARY_BINNING
    )
    result["lower_benchmark"] = {
        "value": float(cert["lower_reference"]),
        "kind": cert.get("lower_reference_kind"),
        "role": LOWER_BENCHMARK_ROLE,
    }
    if cert["verified"] is not True:
        result["status"] = ELIGIBILITY_UNVERIFIED
        if required:
            problems.add(ELIGIBILITY_UNVERIFIED, "eligibility is not verified")
        return result
    if required:
        missing = []
        if cert["policy_class"] != DETERMINISTIC_MEMORYLESS:
            missing.append(f"policy_class {DETERMINISTIC_MEMORYLESS}")
        if not random_kind and cert.get("ordinary_binning") is None:
            missing.append("ordinary_binning")
        if cert.get("goal_feasible") is None:
            missing.append("goal_feasible")
        if missing:
            result["status"] = ELIGIBILITY_UNVERIFIED
            problems.add(
                ELIGIBILITY_UNVERIFIED,
                f"{env_name} eligibility needs {', '.join(missing)}",
            )
            return result
    full = float(cert["full_optimum"])
    best = float(cert["best_in_class"])
    lower = float(cert["lower_reference"])
    if not random_kind and cert.get("ordinary_binning") is False:
        status = ELIGIBILITY_NOT_ORDINARY
    elif cert.get("goal_feasible") is False and goal_objective(env_name):
        status = ELIGIBILITY_GOAL_INFEASIBLE
    elif best <= lower:
        status = ELIGIBILITY_AT_FLOOR
    elif best >= full:
        status = ELIGIBILITY_REALIZABLE
    else:
        status = ELIGIBILITY_ELIGIBLE
        gap = (full - best) / (full - lower)
        result.update(class_gap=gap, small_class_gap=gap < MIN_NORM_GAP)
    result["status"] = status
    if status in INELIGIBLE_STATUSES:
        problems.add(
            SCIENTIFICALLY_INELIGIBLE,
            f"eligibility {status} (best in class {best:.6g}, full optimum "
            f"{full:.6g}, lower reference {lower:.6g}, {cert['policy_class']})",
        )
    return result


def _validate_row(
    row: Mapping[str, Any],
    env_name: str,
    paired: bool,
    evidence: Mapping[str, Any],
    problems: _Problems,
) -> None:
    key = f"{row.get('method')}/{row.get('arm')}/seed {row.get('seed')}"
    if row.get("final") is not True or row.get("placeholder") is True:
        problems.add(MISSING_OUTCOMES, f"{key}: not a final row")
    actual = row.get("evaluation_deterministic")
    if actual is None:
        problems.add(
            MISSING_OUTCOMES,
            f"{key}: evaluation_deterministic is not recorded (never imputed)",
        )
    elif actual is not True:
        problems.add(
            INVALID_EVIDENCE,
            f"{key}: evaluation_deterministic is {actual!r}, not True",
        )
    if (
        "deterministic_evaluation" in row
        and row["deterministic_evaluation"] is not True
    ):
        problems.add(
            INVALID_EVIDENCE,
            f"{key}: legacy deterministic_evaluation "
            f"{row['deterministic_evaluation']!r} is not a consistent True",
        )
    missing = missing_outcome_fields(row, env_name, paired)
    if missing:
        problems.add(MISSING_OUTCOMES, f"{key}: missing {', '.join(missing)}")
    returns = row.get("episode_returns")
    if returns is None:
        return
    n = len(returns)
    if n < MIN_EPISODES:
        problems.add(MISSING_OUTCOMES, f"{key}: {n} < {MIN_EPISODES} episodes")
    if not all(_is_finite_number(r) for r in returns):
        problems.add(INVALID_EVIDENCE, f"{key}: non-finite episode return")
    checks = {
        "episode_lengths": lambda v: _is_int(v) and v > 0,
        "episode_terminated": _is_bool,
        "episode_truncated": _is_bool,
        "episode_successes": _is_bool,
        "evaluation_episode_seeds": lambda v: _is_int(v) and v >= 0,
        "initial_state_digests": _is_sha,
    }
    for field, ok in checks.items():
        values = row.get(field)
        if values is None:
            continue
        if len(values) != n:
            problems.add(INVALID_EVIDENCE, f"{key}: {field} length != returns")
        elif not all(ok(v) for v in values):
            problems.add(INVALID_EVIDENCE, f"{key}: malformed {field}")
    terminated, truncated = row.get("episode_terminated"), row.get("episode_truncated")
    if (
        terminated is not None
        and truncated is not None
        and len(terminated)
        == len(
            truncated,
        )
    ):
        if not all(bool(a) or bool(b) for a, b in zip(terminated, truncated)):
            problems.add(
                INVALID_EVIDENCE, f"{key}: episode neither ended nor timed out"
            )
    seeds = row.get("evaluation_episode_seeds")
    if seeds is not None and len(set(seeds)) != len(seeds):
        problems.add(INVALID_EVIDENCE, f"{key}: duplicate evaluation seeds")
    for nll_key in NLL_KEYS:
        value = row.get(nll_key)
        if value is not None and not _is_finite_number(value):
            problems.add(INVALID_EVIDENCE, f"{key}: non-finite {nll_key}")
    if row.get("arm") == "masked" and row.get("mask_sha256") != evidence.get(
        "mask_sha256",
    ):
        problems.add(INVALID_EVIDENCE, f"{key}: mask_sha256 differs from the mask")
    control = evidence.get("full_control") or {}
    checkpoint = (
        control.get("checkpoint_sha256") if isinstance(control, Mapping) else None
    )
    if "expert_sha256" in row:
        if not _is_sha(row["expert_sha256"]):
            problems.add(INVALID_EVIDENCE, f"{key}: malformed expert_sha256")
        elif checkpoint is not None and row["expert_sha256"] != checkpoint:
            problems.add(
                INVALID_EVIDENCE, f"{key}: expert_sha256 not the shared checkpoint"
            )


def _index_rows(
    evidence: Mapping[str, Any],
    problems: _Problems,
) -> Dict[Tuple[str, str, int], Mapping[str, Any]]:
    seeds = evidence.get("selection_seeds")
    if (
        not isinstance(seeds, Sequence)
        or not seeds
        or not all(_is_int(s) for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        problems.add(INVALID_EVIDENCE, "selection_seeds must be distinct ints")
        seeds = []
    rows = evidence.get("rows")
    if not isinstance(rows, Sequence):
        problems.add(MISSING_OUTCOMES, "no rows")
        rows = []
    index: Dict[Tuple[str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (row.get("method"), row.get("arm"), row.get("seed"))
        if key[0] not in REQUIRED_METHODS or key[1] not in ARMS:
            problems.add(INVALID_EVIDENCE, f"unexpected row {key}")
        elif key[2] not in seeds:
            problems.add(INVALID_EVIDENCE, f"row {key} outside the selection seeds")
        elif key in index:
            problems.add(INVALID_EVIDENCE, f"duplicate row {key}")
        else:
            index[key] = row
    for method in REQUIRED_METHODS:
        for arm in ARMS:
            for seed in seeds:
                if (method, arm, seed) not in index:
                    problems.add(
                        MISSING_OUTCOMES, f"missing row {method}/{arm}/seed {seed}"
                    )
    return index


def _validate(evidence: Mapping[str, Any]):
    problems = _Problems()
    env_name = evidence.get("env_name")
    if not isinstance(env_name, str):
        problems.add(INVALID_EVIDENCE, "env_name must be a string")
        env_name = ""
    elif success_used(env_name) is None:
        problems.add(MISSING_OUTCOMES, f"no outcome rule for {env_name}")
    deterministic = evidence.get("deterministic_evaluation")
    if deterministic is None:
        problems.add(
            MISSING_OUTCOMES,
            "deterministic_evaluation is not declared (never imputed)",
        )
    elif deterministic is not True:
        problems.add(
            INVALID_EVIDENCE,
            f"protocol requires deterministic evaluation, got {deterministic!r}",
        )
    pairing = evidence.get("pairing", "unpaired")
    if pairing not in ("paired", "unpaired"):
        problems.add(
            INVALID_EVIDENCE, f"pairing must be paired or unpaired: {pairing!r}"
        )
    paired = pairing == "paired"
    if not _is_sha(evidence.get("mask_sha256")):
        problems.add(INVALID_EVIDENCE, "mask_sha256 must be 64 lowercase hex")
    _validate_reference(evidence, env_name, problems)
    control = evidence.get("full_control")
    if not isinstance(control, Mapping):
        problems.add(INVALID_EVIDENCE, "full_control must be a mapping")
    elif control.get("checkpoint_sha256") is not None and not _is_sha(
        control["checkpoint_sha256"],
    ):
        problems.add(INVALID_EVIDENCE, "full_control.checkpoint_sha256 malformed")
    index = _index_rows(evidence, problems)
    for row in index.values():
        _validate_row(row, env_name, paired, evidence, problems)
    if paired:
        for (method, arm, seed), row in index.items():
            other = index.get((method, "masked", seed))
            if arm != "full" or other is None:
                continue
            a, b = row.get("evaluation_episode_seeds"), other.get(
                "evaluation_episode_seeds"
            )
            if a is not None and b is not None and list(a) != list(b):
                problems.add(
                    INVALID_EVIDENCE,
                    f"{method}/seed {seed}: paired evaluation seeds differ",
                )
    return problems, index, env_name, pairing


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _seed_gap(
    full: np.ndarray, masked: np.ndarray, paired: bool
) -> Tuple[float, float]:
    """Mean full-minus-masked gap of one seed and its standard error."""
    if paired:
        diff = full - masked
        return float(diff.mean()), float(diff.std(ddof=1) / math.sqrt(diff.size))
    var = full.var(ddof=1) / full.size + masked.var(ddof=1) / masked.size
    return float(full.mean() - masked.mean()), float(math.sqrt(var))


def _gap(
    rows_full: Sequence[Mapping[str, Any]],
    rows_masked: Sequence[Mapping[str, Any]],
    field: str,
    paired: bool,
) -> Dict[str, Any]:
    """Seed-averaged gap, its episode-noise SE, and between-seed diagnostics."""
    gaps, ses = [], []
    for full, masked in zip(rows_full, rows_masked):
        g, se = _seed_gap(
            np.asarray(full[field], dtype=float),
            np.asarray(masked[field], dtype=float),
            paired,
        )
        gaps.append(g)
        ses.append(se)
    between = (
        float(np.std(gaps, ddof=1) / math.sqrt(len(gaps))) if len(gaps) > 1 else None
    )
    return {
        "gap": float(np.mean(gaps)),
        "se": float(math.sqrt(sum(s * s for s in ses)) / len(ses)),
        "between_seed_se": between,
        "zero_se_seeds": sum(1 for s in ses if s == 0.0),
    }


def _arm_summary(
    rows: Sequence[Mapping[str, Any]],
    expert: float,
    random: float,
    uses_success: bool,
    env_name: str,
) -> Dict[str, Any]:
    goal_only = goal_objective(env_name)
    basis = floor_basis(env_name)
    returns = np.concatenate(
        [np.asarray(r["episode_returns"], dtype=float) for r in rows]
    )
    terminated = np.concatenate(
        [np.asarray(r["episode_terminated"], dtype=bool) for r in rows]
    )
    truncated = np.concatenate(
        [np.asarray(r["episode_truncated"], dtype=bool) for r in rows]
    )
    success_rate = None
    seed_rates = None
    if uses_success:
        seed_rates = [
            float(np.asarray(r["episode_successes"], dtype=float).mean()) for r in rows
        ]
        success_rate = float(
            np.concatenate(
                [np.asarray(r["episode_successes"], dtype=float) for r in rows],
            ).mean(),
        )
    no_success = success_rate is None or success_rate == 0.0
    constant = bool(np.ptp(returns) == 0.0)
    all_timeouts = bool(np.all(truncated & ~terminated))
    raw_mean = float(returns.mean())
    with_success = None if seed_rates is None else sum(1 for r in seed_rates if r > 0)
    floor: Optional[bool] = None
    if basis == FLOOR_GOAL_FAILURE:
        floor = bool((constant or all_timeouts) and no_success)
    elif basis == FLOOR_LOWEST_PAYOUT:
        lowest = LOWEST_PAYOUT_RETURNS[_base_env(env_name)]
        floor = bool(np.all(returns == lowest))
    normalized = (raw_mean - random) / (expert - random)
    # Episode-noise SE of the seed-averaged mean, combined like the gate SE.
    seed_vars = [
        float(np.var(r["episode_returns"], ddof=1)) / len(r["episode_returns"])
        for r in rows
    ]
    normalized_se = math.sqrt(sum(seed_vars)) / len(rows) / (expert - random)
    if normalized <= GAP_TOLERANCE:
        comparison = AT_OR_BELOW_RANDOM
    elif normalized > SIGNIFICANCE_SE * normalized_se:
        comparison = ABOVE_RANDOM
    else:
        comparison = INDISTINCT_FROM_RANDOM
    return {
        "episodes": int(returns.size),
        "raw_mean": raw_mean,
        "normalized_mean": normalized,
        "normalized_se": normalized_se,
        "random_comparison": comparison,
        "success_rate": success_rate,
        "seed_success_rates": seed_rates,
        "seeds_with_success": with_success,
        "seed_success_summary": (
            None
            if seed_rates is None
            else f"{with_success}/{len(seed_rates)} seeds with success"
        ),
        "constant_returns": constant,
        "timeout_fraction": float(np.mean(truncated & ~terminated)),
        "floor": floor,
        "floor_basis": basis,
        "zero_success": bool(uses_success and success_rate == 0.0),
        # Goal-only objectives: no success means every episode failed.
        "outcome_failure": bool(goal_only and success_rate == 0.0),
    }


def _method_stats(
    index: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    method: str,
    seeds: Sequence[int],
    expert: float,
    random: float,
    paired: bool,
    criterion: str,
    uses_success: bool,
    env_name: str,
) -> Dict[str, Any]:
    full_rows = [index[(method, "full", s)] for s in seeds]
    masked_rows = [index[(method, "masked", s)] for s in seeds]
    ret = _gap(full_rows, masked_rows, "episode_returns", paired)
    raw_gap, raw_se = ret["gap"], ret["se"]
    norm_gap = raw_gap / (expert - random)
    stats: Dict[str, Any] = {
        "full": _arm_summary(full_rows, expert, random, uses_success, env_name),
        "masked": _arm_summary(masked_rows, expert, random, uses_success, env_name),
        "raw_gap": raw_gap,
        "raw_se": raw_se,
        "normalized_gap": norm_gap,
        "normalized_se": raw_se / (expert - random),
    }
    if criterion == GOAL_SUCCESS:
        suc = _gap(full_rows, masked_rows, "episode_successes", paired)
        stats.update(success_gap=suc["gap"], success_se=suc["se"])
        stats["effect"], gate = suc["gap"], suc
    else:
        stats["effect"], gate = norm_gap, ret
    stats["gate"] = effect_gate(stats["effect"], gate["gap"], gate["se"])
    stats["gate_between_seed_se"] = gate["between_seed_se"]
    stats["gate_zero_se_seeds"] = gate["zero_se_seeds"]
    stats["degenerate_se"] = gate["se"] == 0.0
    stats["effective_sample_size"] = len(seeds) if stats["degenerate_se"] else None
    return stats


def _single_start(index: Mapping[Tuple[str, str, int], Mapping[str, Any]]):
    """True when every row starts from one digest, None without digests."""
    digests = set()
    for row in index.values():
        values = row.get("initial_state_digests")
        if values is None:
            return None
        digests.update(values)
    return len(digests) == 1 if digests else None


def _control(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    control = evidence.get("full_control")
    control = control if isinstance(control, Mapping) else {}
    verified = control.get("verified") is True
    checkpoint = control.get("checkpoint_sha256")
    shared_quality = (
        _is_sha(checkpoint) and control.get("return_quality_passed") is True
    )
    return {
        "verified_flag": verified,
        "checkpoint_sha256": checkpoint if _is_sha(checkpoint) else None,
        "return_quality_passed": control.get("return_quality_passed") is True,
        "upstream_qualified": verified or shared_quality,
    }


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def _decide(reasons: Iterable[str]) -> str:
    reasons = set(reasons)
    if reasons & REJECT_REASONS:
        return REJECTED
    return INCONCLUSIVE if reasons else ACCEPTED


def calibrate_mask(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """Decide whether one mask on one environment passes calibration.

    Data problems never raise; they become reasons. The input is not
    mutated. See the module docstring for the schema and protocol.

    Args:
        evidence: Calibration evidence for one (environment, mask).

    Returns:
        JSON-ready report with ``decision``, sorted ``reasons``, ``messages``,
        ``flags``, ``per_method`` statistics (empty when evidence is missing
        or invalid), ``criterion`` and ``near_reference_criterion``,
        ``near_reference`` (None when evidence is missing or invalid),
        ``passing_methods``, ``most_robust_method`` (smallest
        effect; reported, not decisive), ``scientific_eligibility``,
        ``benign_agnostic``, ``single_start``, ``control``, ``reference``,
        and the frozen ``constants``.

    Raises:
        TypeError: If ``evidence`` is not a mapping.
    """
    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping")
    problems, index, env_name, pairing = _validate(evidence)
    _validate_binning(evidence, problems)
    well_formed = _validate_eligibility(evidence, problems)
    eligibility = _eligibility(evidence, env_name, well_formed, problems)
    control = _control(evidence)
    if not control["upstream_qualified"]:
        problems.add(CONTROL_UNQUALIFIED, "full control is not verified upstream")
    if evidence.get("witness_verified") is not True:
        problems.add(WITNESS_UNVERIFIED, "mask witness is not verified")
    ref = (
        evidence.get("reference")
        if isinstance(evidence.get("reference"), Mapping)
        else {}
    )
    uncertainty = ref.get("uncertainty")
    flags: List[str] = []
    if uncertainty is None:
        flags.append("reference_uncertainty_unavailable")
    if eligibility["small_class_gap"]:
        flags.append("small_class_gap")

    criterion = criterion_for(env_name) if env_name else NORMALIZED_RETURN
    per_method: Dict[str, Any] = {}
    passing: List[str] = []
    most_robust = None
    single_start = None
    near_reference = None
    complete = not ({MISSING_OUTCOMES, INVALID_EVIDENCE} & set(problems.reasons()))
    if complete:
        expert, random = float(ref["expert_return"]), float(ref["random_return"])
        seeds = list(evidence["selection_seeds"])
        uses_success = bool(success_used(env_name))
        for method in REQUIRED_METHODS:
            per_method[method] = _method_stats(
                index,
                method,
                seeds,
                expert,
                random,
                pairing == "paired",
                criterion,
                uses_success,
                env_name,
            )
        single_start = _single_start(index)
        if single_start:
            flags.append("deterministic_single_start")
        passing, near_reference = _judge(per_method, criterion, problems, flags)
        most_robust = min(REQUIRED_METHODS, key=lambda m: per_method[m]["effect"])
    control["qualified"] = control["upstream_qualified"] and (
        CONTROL_UNQUALIFIED not in problems.reasons()
    )
    reasons = problems.reasons()
    benign = (
        MASKED_NEAR_REFERENCE in reasons
        and eligibility["status"] == ELIGIBILITY_ELIGIBLE
    )
    return {
        "protocol": PROTOCOL,
        "env_name": env_name,
        "mask": evidence.get("mask"),
        "mask_sha256": evidence.get("mask_sha256"),
        "binning": (
            dict(evidence["binning"])
            if isinstance(evidence.get("binning"), Mapping)
            else None
        ),
        "decision": _decide(reasons),
        "reasons": reasons,
        "messages": [f"{reason}: {message}" for reason, message in problems.items],
        "flags": flags,
        "criterion": criterion,
        # The near-reference rule reads the same scale as the effect gate.
        "near_reference_criterion": criterion,
        "near_reference": near_reference,
        "pairing": pairing,
        "deterministic_evaluation": evidence.get("deterministic_evaluation"),
        "single_start": single_start,
        "passing_methods": passing,
        "most_robust_method": most_robust,
        "per_method": per_method,
        "scientific_eligibility": eligibility,
        "benign_agnostic": benign,
        "benign_note": BENIGN_NOTE if benign else None,
        "control": control,
        "reference": {
            "expert_return": ref.get("expert_return"),
            "random_return": ref.get("random_return"),
            "reference_type": ref.get("reference_type", REFERENCE_RANDOM),
            "uncertainty": "unavailable" if uncertainty is None else dict(uncertainty),
        },
        "normalization_note": NORMALIZATION_NOTE,
        "constants": {
            "min_norm_gap": MIN_NORM_GAP,
            "significance_se": SIGNIFICANCE_SE,
            "near_optimal": NEAR_OPTIMAL,
            "min_episodes": MIN_EPISODES,
            "required_methods": list(REQUIRED_METHODS),
        },
    }


def _judge(
    per_method: Mapping[str, Dict[str, Any]],
    criterion: str,
    problems: _Problems,
    flags: List[str],
) -> Tuple[List[str], Dict[str, Any]]:
    """Add outcome, floor, random, and effect reasons for complete evidence.

    Returns:
        Methods passing their effect gate (empty when the control is
        unqualified, since effects against it are not judged), and the
        near-reference details on the ``criterion`` scale.
    """
    compressed = False
    for method, stats in per_method.items():
        full, masked = stats["full"], stats["masked"]
        if full["floor"] or full["outcome_failure"]:
            problems.add(
                CONTROL_UNQUALIFIED,
                f"{method}: full arm shows outcome failure "
                f"(normalized {full['normalized_mean']:.3f} is not competence)",
            )
        elif full["zero_success"]:
            # Return objective: competence rests on upstream qualification.
            flags.append(f"full_arm_zero_success:{method}")
        if masked["floor"]:
            flags.append(f"masked_floor:{method}")
        if masked["outcome_failure"] and masked["normalized_mean"] >= MIN_NORM_GAP:
            compressed = True
            flags.append(f"{NORMALIZATION_COMPRESSED}:{method}")
        if stats["degenerate_se"]:
            flags.append(f"{DEGENERATE_SE}:{method}")
    if compressed:
        flags.append(SUCCESS_RATE_PRESENTATION_REQUIRED)
    if any(s["masked"]["floor"] is None for s in per_method.values()):
        flags.append(FLOOR_UNKNOWN)
    if all(s["masked"]["floor"] is True for s in per_method.values()):
        problems.add(FLOOR_COLLAPSE, "every masked method is at the outcome floor")
    if criterion == GOAL_SUCCESS:
        rates = {m: s["masked"]["success_rate"] for m, s in per_method.items()}
        if all(rate == 0.0 for rate in rates.values()):
            problems.add(FLOOR_COLLAPSE, "every masked method has zero goal success")
        elif not any(
            0.0 < rates[m] < per_method[m]["full"]["success_rate"] for m in rates
        ):
            flags.append("no_partial_goal_success")
    comparisons = {m: s["masked"]["random_comparison"] for m, s in per_method.items()}
    for method, comparison in comparisons.items():
        if comparison != ABOVE_RANDOM:
            flags.append(f"{comparison}:{method}")
    if criterion != GOAL_SUCCESS and all(
        c == AT_OR_BELOW_RANDOM for c in comparisons.values()
    ):
        problems.add(
            NO_BETTER_THAN_RANDOM,
            "every masked normalized mean is at or below the uniform-random "
            "reference (0); uninformative, whether or not at a known floor",
        )
    elif (
        any(s["masked"]["floor_basis"] is None for s in per_method.values())
        and ABOVE_RANDOM not in comparisons.values()
    ):
        # Unknown floor only; on known floors indistinct is a flag.
        problems.add(
            INCONCLUSIVE_NOISE,
            "no masked method is above the uniform-random reference by more "
            f"than {SIGNIFICANCE_SE} SE",
        )
    normalized = {m: s["masked"]["normalized_mean"] for m, s in per_method.items()}
    normalized_all_near = all(v >= NEAR_OPTIMAL for v in normalized.values())
    if criterion == GOAL_SUCCESS:
        values = {m: s["masked"]["success_rate"] for m, s in per_method.items()}
    else:
        values = normalized
    all_near = all(v >= NEAR_OPTIMAL for v in values.values())
    if all_near:
        problems.add(
            MASKED_NEAR_REFERENCE,
            f"every masked {criterion} value is at least {NEAR_OPTIMAL}",
        )
    elif normalized_all_near:
        flags.append(REFERENCE_COMPRESSION)
    near_reference = {
        "criterion": criterion,
        "threshold": NEAR_OPTIMAL,
        "masked": values,
        "all_near": all_near,
        "normalized_all_near": normalized_all_near,
    }
    if CONTROL_UNQUALIFIED in problems.reasons():
        # Effects against an unqualified control are not judged.
        return [], near_reference
    passing = [m for m, s in per_method.items() if s["gate"] is None]
    for method, stats in per_method.items():
        if stats["gate"] is not None:
            flags.append(f"{stats['gate']}:{method}")
    if not passing:
        gates = {s["gate"] for s in per_method.values()}
        reason = (
            INSUFFICIENT_EFFECT
            if gates == {INSUFFICIENT_EFFECT}
            else (INCONCLUSIVE_NOISE)
        )
        problems.add(
            reason,
            "no method passes the effect gate: "
            + "; ".join(
                f"{m} effect {s['effect']:.6f} (raw gap {s['raw_gap']:.6g}, "
                f"SE {s['raw_se']:.6g}, {s['gate']})"
                for m, s in per_method.items()
            ),
        )
    return passing, near_reference


# ---------------------------------------------------------------------------
# Surveillance
# ---------------------------------------------------------------------------


def survey_calibrations(evidences: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Calibrate every candidate and summarize per environment.

    Legacy results without outcome evidence are listed under
    ``missing_outcomes``; unknown is never a pass. A mask is selected only
    when it is the single accepted candidate of its environment; several
    accepted candidates need human review, and nothing is promoted.

    Args:
        evidences: Calibration evidence, one per (environment, mask).

    Returns:
        JSON-ready summary with ``environments`` (candidates in input order,
        ``accepted``, ``selected_mask``, ``status``), ``missing_outcomes``,
        and ``reports``.
    """
    environments: Dict[str, Dict[str, Any]] = {}
    missing: List[Dict[str, Any]] = []
    reports = []
    for evidence in evidences:
        report = calibrate_mask(evidence)
        reports.append(report)
        env = environments.setdefault(report["env_name"], {"candidates": []})
        env["candidates"].append(
            {
                "mask": report["mask"],
                "decision": report["decision"],
                "reasons": report["reasons"],
                "passing_methods": report["passing_methods"],
                "eligibility": report["scientific_eligibility"]["status"],
                "benign_agnostic": report["benign_agnostic"],
            },
        )
        if MISSING_OUTCOMES in report["reasons"]:
            missing.append({"env_name": report["env_name"], "mask": report["mask"]})
    for env in environments.values():
        accepted = [c["mask"] for c in env["candidates"] if c["decision"] == ACCEPTED]
        env["accepted"] = accepted
        env["selected_mask"] = accepted[0] if len(accepted) == 1 else None
        env["status"] = {0: "none_accepted", 1: "selected"}.get(
            len(accepted),
            "multiple_accepted_needs_review",
        )
    return {
        "protocol": PROTOCOL,
        "environments": environments,
        "missing_outcomes": missing,
        "reports": reports,
    }
