"""Exact agnostic DAgger toy: layered finite MDP with a four-policy learner class.

Each layer has nominal states N0, N1 (drawn as N_z, z ~ Bernoulli(alpha)) and a
recovery state R. The learner sees only the mode (nominal or recovery), so its
class is the four deterministic policies (a_N, a_R). FTL-DAgger (beta = 0) is
compared with BC-iid (beta = 1) and a matching fixed-BC control using exact
count-based ERM, real vectorized rollouts, and exact matrix-DP evaluation.

Usage::

    python -m imitation.experiments.agnostic.toy --output DIR [options]

The defaults are a small pilot. The result file is ``DIR/result.json``; exit
codes are 0 (status ``complete``), 2 (invalid invocation or refused output),
3 (``partial_deadline``) and 4 (``partial_error``).
"""

import argparse
import dataclasses
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROTOCOL_VERSION = "agnostic-toy-v1"
EXPERT = "expert"
CANONICAL_TIE_ORDER = (0, 1, 2, 3)
ANNOTATION_MODES = ("deferred", "full_trajectory")
CAMPAIGN_CAP_UTC = dt.datetime(2026, 10, 7, 1, 1, tzinfo=dt.timezone.utc)
EXIT_OK, EXIT_INVALID, EXIT_PARTIAL, EXIT_ERROR = 0, 2, 3, 4
RESULTS_NAME = "result.json"
CLAIM_NAME = ".claim"
PROGRESS_INTERVAL_SECONDS = 10.0  # minimum spacing of round-boundary publishes
_TAPE_TAG = 0x7A6779  # separates toy tapes from any other use of the same seed
_SELECT, _NOMINAL, _SLIP = 0, 1, 2

SEMANTICS = {
    "cost": "J(pi): expected expert-action disagreements over exactly H actions; "
    "lower is better. Used only by the exact evaluator, never shown to learners.",
    "expert_relative_cost": "J(pi) - J(expert), with J(expert) = 0.",
    "class_excess": "J(pi) - min over the four class policies of J.",
    "policy_ids": "Canonical ID p is (a_N, a_R) = (p // 2, p % 2); the physical "
    "action is the canonical action XOR orientation.",
    "erm": "Cold exact ERM on retained (mode, physical label) counts, minimizing "
    "total 0-1 mistakes; ties go to the smallest canonical ID. The data-free "
    "initial policy is ID 0 for every arm.",
    "rounds": "Round n freezes behavior pi_n, rolls out K independent full "
    "H-action episodes, retains one (mode, label) per episode at a pre-action "
    "time T uniform on 0..H-1 drawn independently of transitions, then refits "
    "to give pi_(n+1). The last round may be partial so the retained total "
    "equals the budget exactly.",
    "ftl": "FTL-DAgger, beta = 0: the current deterministic learner acts.",
    "bc_iid": "beta = 1: the deterministic expert acts; refit every round.",
    "bc_fixed": "Fit once at checkpoint B on the first B retained labels of the "
    "shared expert stream (acquisition order, no later labels). A matching "
    "offline control that must equal BC-iid under exact ERM. It differs from a "
    "fixed-B0 replay curve, which is not implemented.",
    "shared_expert_acquisition": "Physical expert-driven acquisition, done once "
    "and shared by BC-iid and BC-fixed. Their logical counters describe the "
    "data each arm consumes and are not additive.",
    "annotation": "Counted at the expert oracle as action entries actually "
    "requested, not vectorized calls. deferred: the FTL rollout requests "
    "labels only at selected states. full_trajectory: it requests labels at "
    "every visited state and discards unselected ones. The expert driver "
    "requests an action for every state it acts in and the retained label "
    "reuses it. No cross-state label cache. The environment's transition "
    "predicate is not an oracle request and is never shown to the trainer.",
    "status": "complete: every cell finished and the run ended before the "
    "deadline. partial_deadline: the deadline was observed; no rollout starts "
    "after an observed expiration, but a round already in progress finishes, "
    "so the run can end after the deadline by at most one round. "
    "partial_error: an exception stopped the run.",
    "cell_status": "complete; overran_deadline (all rounds acquired but the "
    "deadline was observed after the last round, not claimed to meet the "
    "cap); partial_deadline; partial_error; running.",
    "acquisition": "Physical data acquired per arm, updated after each "
    "successful rollout, including a rollout whose round was never "
    "committed. Committed training counters are in latest_round and "
    "checkpoints.",
    "in_flight": "A rollout that raised before returning: its env steps are "
    "not observable (null, bounded by env_steps_upper_bound); expert entries "
    "are those observed at the oracle before the failure.",
    "latest_round": "Snapshot after the last committed round regardless of the "
    "checkpoint schedule; bc_fixed is fit only at checkpoints (null otherwise).",
    "tapes": "Per (seed, horizon, round), selection times, nominal uniforms and "
    "slip uniforms come from separate streams, with a fixed value per "
    "episode/time whether or not it is consumed. FTL and the expert stream "
    "share tapes (common random numbers). Variants of alpha, kappa, q and "
    "orientation with equal seed and horizon also share tapes: they are "
    "paired, not independent replicates.",
    "behavior_mixture": "Uniform mixture over FTL behavior policies "
    "pi_1..pi_N (each round weighted equally, including a partial last round; "
    "not label-weighted), drawn once per episode. Its cost equals "
    "H * (approximation_term + regret / N).",
    "post_update": "pi_(N+1), the ERM fit after the last completed round.",
    "F": "F_b(p) = (1/H) sum_t E_{s ~ d_t^b} 1[p(s) != expert(s)] from exact "
    "occupancy; equals (1 - rbar_b) e_p + rbar_b w_p.",
    "approximation_term": "A_N = min_p (1/N) sum_n F_{pi_n}(p) over behavior "
    "policies pi_1..pi_N.",
    "regret": "Reg_N = sum_n F_{pi_n}(pi_n) - min_p sum_n F_{pi_n}(p); not "
    "clamped, may be negative.",
    "onpolicy_disagreement": "F_pi(pi) = J(pi) / H.",
    "expert_occupancy_floor": "min_p F_expert(p) = alpha.",
    "evaluation": "Exact matrix dynamic programming with no randomness and no "
    "training queries; analytic evaluations are counted separately.",
    "alpha_zero": "N1 is removed from the admissible state space.",
}


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _is_real(x: Any) -> bool:
    ok = isinstance(x, (int, float, np.integer, np.floating))
    return ok and not isinstance(x, bool) and math.isfinite(x)


@dataclasses.dataclass(frozen=True)
class ToyMDP:
    """One toy condition. ``q`` is the canonical recovery action."""

    horizon: int
    alpha: float
    kappa: float
    q: int
    orientation: int

    def __post_init__(self):
        if not _is_int(self.horizon) or self.horizon < 1:
            raise ValueError("horizon must be an integer >= 1")
        if not _is_real(self.alpha) or not 0.0 <= self.alpha < 0.5:
            raise ValueError("alpha must be finite in [0, 0.5)")
        if not _is_real(self.kappa) or not 0.0 <= self.kappa <= 1.0:
            raise ValueError("kappa must be finite in [0, 1]")
        for name in ("q", "orientation"):
            if not _is_int(getattr(self, name)) or getattr(self, name) not in (0, 1):
                raise ValueError("{} must be 0 or 1".format(name))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "kappa", float(self.kappa))

    @property
    def states(self) -> Tuple[str, ...]:
        return ("N0", "R") if self.alpha == 0.0 else ("N0", "N1", "R")

    def expert_action(self, state: str) -> int:
        if state not in self.states:
            raise ValueError(
                "{} is not admissible (states {})".format(state, self.states)
            )
        canonical = self.q if state == "R" else int(state == "N1")
        return canonical ^ self.orientation


def physical_actions(policy_id: int, orientation: int) -> Tuple[int, int]:
    """Physical (nominal, recovery) actions of a canonical class policy."""
    if not _is_int(policy_id) or not 0 <= policy_id <= 3:
        raise ValueError("policy_id must be in 0..3")
    return (policy_id >> 1) ^ orientation, (policy_id & 1) ^ orientation


def class_policy_actions(mdp: ToyMDP, policy_id: int) -> Dict[str, int]:
    nominal, recovery = physical_actions(policy_id, mdp.orientation)
    return {s: recovery if s == "R" else nominal for s in mdp.states}


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """Exact cost and occupancy; ``occupancy[t]`` is the law before action t."""

    cost: float
    occupancy: np.ndarray
    states: Tuple[str, ...]

    @property
    def recovery_mean(self) -> float:
        return float(self.occupancy[:, self.states.index("R")].mean())


def _mistakes(mdp: ToyMDP, actions: Mapping[str, int]) -> np.ndarray:
    if set(actions) != set(mdp.states):
        raise ValueError("actions must cover exactly the states {}".format(mdp.states))
    if any(a not in (0, 1) for a in actions.values()):
        raise ValueError("actions must be 0 or 1")
    return np.array([float(actions[s] != mdp.expert_action(s)) for s in mdp.states])


def evaluate_actions(mdp: ToyMDP, actions: Mapping[str, int]) -> Evaluation:
    """Matrix DP for a stationary physical action map over admissible states."""
    states = mdp.states
    wrong = _mistakes(mdp, actions)
    n = len(states)
    fresh = np.zeros(n)
    fresh[0] = 1.0 - mdp.alpha
    if mdp.alpha > 0:
        fresh[1] = mdp.alpha
    stay = np.zeros(n)
    stay[states.index("R")] = 1.0
    trans = np.empty((n, n))
    for i, s in enumerate(states):
        if not wrong[i]:
            trans[i] = fresh
        elif s == "R":
            trans[i] = stay
        else:
            trans[i] = mdp.kappa * stay + (1.0 - mdp.kappa) * fresh
    occupancy = np.empty((mdp.horizon, n))
    d = fresh
    for t in range(mdp.horizon):
        occupancy[t] = d
        d = d @ trans
    return Evaluation(float((occupancy @ wrong).sum()), occupancy, states)


def evaluate_policy(mdp: ToyMDP, policy_id: int) -> Evaluation:
    return evaluate_actions(mdp, class_policy_actions(mdp, policy_id))


def evaluate_expert(mdp: ToyMDP) -> Evaluation:
    return evaluate_actions(mdp, {s: mdp.expert_action(s) for s in mdp.states})


def class_costs(mdp: ToyMDP) -> List[float]:
    return [evaluate_policy(mdp, p).cost for p in range(4)]


def expert_occupancy_floor(mdp: ToyMDP) -> float:
    occupancy = evaluate_expert(mdp).occupancy
    return min(
        float((occupancy @ _mistakes(mdp, class_policy_actions(mdp, p))).mean())
        for p in range(4)
    )


def fit_erm(
    counts: np.ndarray,
    orientation: int,
    tie_order: Sequence[int] = CANONICAL_TIE_ORDER,
) -> int:
    """Exact 0-1 ERM from counts[mode, physical label]; first in tie_order wins."""
    counts = np.asarray(counts)
    if counts.shape != (2, 2) or np.any(counts < 0):
        raise ValueError("counts must be a nonnegative 2x2 array")
    if sorted(tie_order) != [0, 1, 2, 3]:
        raise ValueError("tie_order must be a permutation of 0..3")
    best, best_mistakes = None, None
    for p in tie_order:
        nominal, recovery = physical_actions(p, orientation)
        mistakes = counts[0, 1 - nominal] + counts[1, 1 - recovery]
        if best_mistakes is None or mistakes < best_mistakes:
            best, best_mistakes = p, mistakes
    return int(best)


@dataclasses.dataclass(frozen=True)
class Tapes:
    """Exogenous randomness for K episodes of one round."""

    select: np.ndarray  # (K,) retained pre-action time in 0..H-1
    nominal: np.ndarray  # (K, H) uniform for the fresh nominal draw before action t
    slip: np.ndarray  # (K, H) uniform for the slip after a wrong nominal action t


def episode_tapes(seed: int, round_index: int, n_episodes: int, horizon: int) -> Tapes:
    def stream(kind: int) -> np.random.Generator:
        seq = np.random.SeedSequence(
            entropy=[_TAPE_TAG, int(seed)],
            spawn_key=(int(horizon), int(round_index), kind),
        )
        return np.random.default_rng(seq)

    return Tapes(
        select=stream(_SELECT).integers(0, horizon, size=n_episodes),
        nominal=stream(_NOMINAL).random((n_episodes, horizon)),
        slip=stream(_SLIP).random((n_episodes, horizon)),
    )


def _expert_physical(mdp: ToyMDP, recovery: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.where(
        recovery, mdp.q ^ mdp.orientation, z.astype(np.int64) ^ mdp.orientation
    )


class ExpertOracle:
    """Simulated deterministic expert, the only source of labels for training.

    ``entries`` counts action entries actually requested (one per full state),
    not vectorized calls. The environment's own transition predicate does not
    go through the oracle and is never shown to the trainer.
    """

    def __init__(self, mdp: ToyMDP):
        self._mdp = mdp
        self.entries = 0

    def request(self, recovery: np.ndarray, z: np.ndarray) -> np.ndarray:
        recovery = np.asarray(recovery, dtype=bool)
        self.entries += int(recovery.size)
        return _expert_physical(self._mdp, recovery, np.asarray(z))


@dataclasses.dataclass(frozen=True)
class Rollout:
    modes: np.ndarray  # retained learner-visible mode: 0 nominal, 1 recovery
    labels: np.ndarray  # retained physical expert label
    episodes: int
    env_steps: int


def rollout(
    mdp: ToyMDP,
    behavior: Any,
    tapes: Tapes,
    oracle: ExpertOracle,
    annotation: str = "deferred",
) -> Rollout:
    """Execute K full H-action episodes, retaining the state at ``tapes.select``.

    The expert behavior requests an action for every batch state at every step
    and the retained label reuses that action. A learner behavior requests
    labels only at the selected states (``deferred``) or at every visited state
    (``full_trajectory``, unselected labels are discarded).
    """
    k, horizon = tapes.nominal.shape
    if horizon != mdp.horizon:
        raise ValueError("tape horizon does not match the MDP")
    if annotation not in ANNOTATION_MODES:
        raise ValueError("annotation must be one of {}".format(ANNOTATION_MODES))
    by_expert = isinstance(behavior, str) and behavior == EXPERT
    if not by_expert:
        act_nominal, act_recovery = physical_actions(behavior, mdp.orientation)
    recovery = np.zeros(k, dtype=bool)
    z = tapes.nominal[:, 0] < mdp.alpha
    modes = np.full(k, -1, dtype=np.int64)
    labels = np.full(k, -1, dtype=np.int64)
    steps = 0
    for t in range(horizon):
        chosen = tapes.select == t
        if by_expert:
            action = oracle.request(recovery, z)
            labels[chosen] = action[chosen]
        else:
            action = np.where(recovery, act_recovery, act_nominal)
            if annotation == "full_trajectory":
                labels[chosen] = oracle.request(recovery, z)[chosen]
            elif chosen.any():
                labels[chosen] = oracle.request(recovery[chosen], z[chosen])
        modes[chosen] = recovery[chosen]
        # Environment dynamics predicate: not an oracle request.
        correct = action == _expert_physical(mdp, recovery, z)
        steps += k
        if t + 1 < horizon:
            slipped = ~recovery & ~correct & (tapes.slip[:, t] < mdp.kappa)
            recovery = (recovery & ~correct) | slipped
            z = tapes.nominal[:, t + 1] < mdp.alpha
    if np.any(modes < 0):
        raise ValueError("selection times must lie in 0..H-1")
    return Rollout(modes, labels, k, steps)


def _counts(modes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.bincount(2 * modes + labels, minlength=4).reshape(2, 2)


def _validate_schedule(
    budget: int, batch: int, checkpoints: Sequence[int]
) -> Tuple[int, ...]:
    if not _is_int(budget) or budget < 1:
        raise ValueError("budget must be an integer >= 1")
    if not _is_int(batch) or batch < 1:
        raise ValueError("batch must be an integer >= 1")
    checkpoints = tuple(checkpoints)
    if not checkpoints or not all(_is_int(c) for c in checkpoints):
        raise ValueError("checkpoints must be a nonempty list of integers")
    if any(b <= a for a, b in zip(checkpoints, checkpoints[1:])) or checkpoints[0] < 1:
        raise ValueError("checkpoints must be positive and strictly increasing")
    if checkpoints[-1] != budget:
        raise ValueError("the last checkpoint must equal the budget")
    if any(c % batch for c in checkpoints[:-1]):
        raise ValueError("checkpoints before the budget must be multiples of batch")
    return tuple(int(c) for c in checkpoints)


def run_cell(
    mdp: ToyMDP,
    seed: int,
    budget: int,
    batch: int,
    checkpoints: Sequence[int],
    annotation_mode: str = "deferred",
    tie_order: Sequence[int] = CANONICAL_TIE_ORDER,
    should_stop: Optional[Callable[[], bool]] = None,
    record: Optional[Dict[str, Any]] = None,
    on_round: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Run FTL-DAgger, BC-iid and fixed BC for one condition and seed.

    ``tie_order`` is a diagnostic hook for symmetry checks; the CLI always uses
    the canonical order. ``record`` is filled in place so that a caller keeps
    the acquisition ledger and latest committed round if interrupted.
    ``should_stop`` is checked before every round and once after the last
    round; ``on_round`` is called after every committed round.
    """
    checkpoints = _validate_schedule(budget, batch, checkpoints)
    if annotation_mode not in ANNOTATION_MODES:
        raise ValueError("annotation_mode must be one of {}".format(ANNOTATION_MODES))
    u = mdp.orientation
    empty = np.zeros((2, 2), dtype=np.int64)
    initial = fit_erm(empty, u, tie_order)

    evals = [evaluate_policy(mdp, p) for p in range(4)]
    expert_eval = evaluate_expert(mdp)
    mistakes = [_mistakes(mdp, class_policy_actions(mdp, p)) for p in range(4)]
    f = np.array(
        [[(evals[b].occupancy @ m).mean() for m in mistakes] for b in range(4)]
    )
    costs = np.array([e.cost for e in evals])
    optimum, expert_cost = float(costs.min()), expert_eval.cost
    record = {} if record is None else record
    record.update(
        status="running",
        rounds_completed=0,
        exact={
            "class_costs": costs.tolist(),
            "class_optimum_cost": optimum,
            "class_optimum_policy_id": int(np.argmin(costs)),
            "expert_cost": expert_cost,
            "expert_occupancy_floor": min(
                float((expert_eval.occupancy @ m).mean()) for m in mistakes
            ),
            "recovery_occupancy_mean": [e.recovery_mean for e in evals],
            "analytic_policy_evaluations": 5,
        },
        checkpoints=[],
        latest_round=None,
        acquisition={
            arm: dict(rollouts=0, episodes=0, env_steps=0, expert_action_entries=0)
            for arm in ("ftl", "shared_expert")
        },
        in_flight=None,
    )
    acquisition = record["acquisition"]
    oracles = {"ftl": ExpertOracle(mdp), "shared_expert": ExpertOracle(mdp)}

    def acquire(arm: str, behavior: Any, tapes: Tapes, round_index: int) -> Rollout:
        k = tapes.select.shape[0]
        before = oracles[arm].entries
        in_flight = {
            "arm": arm,
            "round_index": round_index,
            "episodes_launched": k,
            "env_steps": None,
            "env_steps_upper_bound": k * mdp.horizon,
            "expert_action_entries_observed": None,
        }
        record["in_flight"] = in_flight
        try:
            out = rollout(mdp, behavior, tapes, oracles[arm], annotation_mode)
        except BaseException:
            in_flight["expert_action_entries_observed"] = oracles[arm].entries - before
            raise
        ledger = acquisition[arm]
        ledger["rollouts"] += 1
        ledger["episodes"] += out.episodes
        ledger["env_steps"] += out.env_steps
        ledger["expert_action_entries"] += oracles[arm].entries - before
        record["in_flight"] = None
        return out

    def costs_of(p: int, prefix: str = "") -> Dict[str, float]:
        return {
            prefix + "cost": float(costs[p]),
            prefix + "expert_relative_cost": float(costs[p] - expert_cost),
            prefix + "class_excess": float(costs[p] - optimum),
        }

    ftl_policy = bc_policy = initial
    ftl_counts, bc_counts = empty.copy(), empty.copy()
    pool_modes: List[np.ndarray] = []
    pool_labels: List[np.ndarray] = []
    behavior_counts = np.zeros(4, dtype=np.int64)
    fixed_fits = 0
    collected = 0
    for round_index in itertools.count():
        if collected == budget:
            break
        if should_stop is not None and should_stop():
            record["status"] = "partial_deadline"
            return record
        k = min(batch, budget - collected)
        tapes = episode_tapes(seed, round_index, k, mdp.horizon)
        learner = acquire("ftl", ftl_policy, tapes, round_index)
        expert = acquire("shared_expert", EXPERT, tapes, round_index)

        # Commit the round: aggregate retained labels and refit every arm.
        behavior_counts[ftl_policy] += 1
        ftl_counts += _counts(learner.modes, learner.labels)
        ftl_policy = fit_erm(ftl_counts, u, tie_order)
        pool_modes.append(expert.modes)
        pool_labels.append(expert.labels)
        bc_counts += _counts(expert.modes, expert.labels)
        bc_policy = fit_erm(bc_counts, u, tie_order)
        collected += k
        n = round_index + 1
        record["rounds_completed"] = n

        is_checkpoint = collected in checkpoints
        ftl_ledger, shared_ledger = acquisition["ftl"], acquisition["shared_expert"]
        ftl = dict(
            rounds=n,
            episodes=ftl_ledger["episodes"],
            env_steps=ftl_ledger["env_steps"],
            retained_labels=collected,
            expert_action_entries=ftl_ledger["expert_action_entries"],
            fits=n,
        )
        shared = dict(
            episodes=shared_ledger["episodes"],
            env_steps=shared_ledger["env_steps"],
            expert_action_entries=shared_ledger["expert_action_entries"],
            retained_labels=collected,
        )
        totals = behavior_counts @ f
        best_total = float(totals.min())
        mixture = float(behavior_counts @ costs) / n
        logical = {
            "retained_labels": collected,
            "logical_episodes": shared["episodes"],
            "logical_env_steps": shared["env_steps"],
            "logical_expert_action_entries": shared["expert_action_entries"],
        }
        ftl_record = {"post_update_policy_id": ftl_policy}
        ftl_record.update(costs_of(ftl_policy, "post_update_"))
        ftl_record.update(
            post_update_onpolicy_disagreement=float(f[ftl_policy, ftl_policy]),
            behavior_policy_id_counts=behavior_counts.tolist(),
            behavior_mixture_cost=mixture,
            behavior_mixture_expert_relative_cost=mixture - expert_cost,
            behavior_mixture_class_excess=mixture - optimum,
            approximation_term=best_total / n,
            regret=float(behavior_counts @ np.diag(f)) - best_total,
            retained_counts=ftl_counts.tolist(),
            counters=dict(ftl),
        )
        bc_record = {"policy_id": bc_policy}
        bc_record.update(costs_of(bc_policy))
        bc_record.update(
            expert_occupancy_disagreement=float(
                (expert_eval.occupancy @ mistakes[bc_policy]).mean()
            ),
            retained_counts=bc_counts.tolist(),
            counters=dict(logical, rounds=n, fits=n),
        )
        fixed_record = None
        if is_checkpoint:
            prefix_modes = np.concatenate(pool_modes)[:collected]
            prefix_labels = np.concatenate(pool_labels)[:collected]
            fixed_counts = _counts(prefix_modes, prefix_labels)
            fixed_policy = fit_erm(fixed_counts, u, tie_order)
            fixed_fits += 1
            fixed_record = {"policy_id": fixed_policy}
            fixed_record.update(costs_of(fixed_policy))
            fixed_record.update(
                retained_counts=fixed_counts.tolist(),
                counters=dict(logical, fits=fixed_fits),
            )
        snapshot = {
            "retained_labels": collected,
            "rounds": n,
            "is_checkpoint": is_checkpoint,
            "ftl": ftl_record,
            "bc_iid": bc_record,
            "bc_fixed": fixed_record,
            "shared_expert_acquisition": shared,
        }
        record["latest_round"] = snapshot
        if is_checkpoint:
            record["checkpoints"].append(snapshot)
        if on_round is not None:
            on_round()
    if should_stop is not None and should_stop():
        # Every round was acquired, but the deadline passed during the last one.
        record["status"] = "overran_deadline"
        return record
    record["status"] = "complete"
    return record


def _require_utc(value: Any, name: str) -> dt.datetime:
    if not isinstance(value, dt.datetime) or value.tzinfo is None:
        raise ValueError("{} must be a timezone-aware UTC datetime".format(name))
    if value.utcoffset() != dt.timedelta(0):
        raise ValueError("{} must have a zero UTC offset".format(name))
    return value.astimezone(dt.timezone.utc)


def _iso(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat()


def _grid(values: Sequence[Any], name: str, check: Callable[[Any], bool]) -> Tuple:
    values = tuple(values)
    if not values:
        raise ValueError("{} must be nonempty".format(name))
    if not all(check(v) for v in values):
        raise ValueError("{} has an invalid value".format(name))
    if len(set(values)) != len(values):
        raise ValueError("{} has duplicates".format(name))
    return values


@dataclasses.dataclass(frozen=True)
class ToyConfig:
    """Validated experiment grid; construction fails before any acquisition."""

    seeds: Tuple[int, ...] = (0,)
    horizons: Tuple[int, ...] = (16,)
    alphas: Tuple[float, ...] = (0.0, 0.1)
    kappas: Tuple[float, ...] = (0.0, 1.0)
    qs: Tuple[int, ...] = (0, 1)
    orientations: Tuple[int, ...] = (0, 1)
    budget: int = 64
    batch: int = 16
    checkpoints: Optional[Tuple[int, ...]] = None  # None: batch doubling to budget
    annotation_mode: str = "deferred"
    deadline: dt.datetime = CAMPAIGN_CAP_UTC

    def __post_init__(self):
        def put(name, value):
            object.__setattr__(self, name, value)

        put("seeds", _grid(self.seeds, "seeds", lambda s: _is_int(s) and s >= 0))
        put(
            "horizons",
            _grid(self.horizons, "horizons", lambda h: _is_int(h) and h >= 1),
        )
        put(
            "alphas",
            _grid(self.alphas, "alphas", lambda a: _is_real(a) and 0 <= a < 0.5),
        )
        put(
            "kappas",
            _grid(self.kappas, "kappas", lambda k: _is_real(k) and 0 <= k <= 1),
        )
        put("qs", _grid(self.qs, "qs", lambda v: _is_int(v) and v in (0, 1)))
        put(
            "orientations",
            _grid(
                self.orientations, "orientations", lambda v: _is_int(v) and v in (0, 1)
            ),
        )
        if not _is_int(self.batch) or self.batch < 1:
            raise ValueError("batch must be an integer >= 1")
        if not _is_int(self.budget) or self.budget < 1:
            raise ValueError("budget must be an integer >= 1")
        checkpoints = self.checkpoints
        if checkpoints is None:
            checkpoints, c = [], self.batch
            while c < self.budget:
                checkpoints.append(c)
                c *= 2
            checkpoints.append(self.budget)
        put("checkpoints", _validate_schedule(self.budget, self.batch, checkpoints))
        if self.annotation_mode not in ANNOTATION_MODES:
            raise ValueError(
                "annotation_mode must be one of {}".format(ANNOTATION_MODES)
            )
        deadline = _require_utc(self.deadline, "deadline")
        if deadline > CAMPAIGN_CAP_UTC:
            raise ValueError(
                "deadline is past the campaign cap {}".format(_iso(CAMPAIGN_CAP_UTC))
            )
        put("deadline", deadline)

    def cells(self) -> List[Tuple]:
        return list(
            itertools.product(
                self.seeds,
                self.horizons,
                self.alphas,
                self.kappas,
                self.qs,
                self.orientations,
            )
        )

    def to_json(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out = {k: list(v) if isinstance(v, tuple) else v for k, v in out.items()}
        out["deadline"] = _iso(self.deadline)
        out["tie_order"] = list(CANONICAL_TIE_ORDER)
        return out


def source_identity() -> Dict[str, Any]:
    here = pathlib.Path(__file__).resolve()
    revision = None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(here.parent),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            revision = proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "toy_module": __name__,
        "sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        "git_revision": revision,
        "git_revision_note": "HEAD of the enclosing checkout; uncommitted edits "
        "are identified only by source_sha256.",
    }


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def run_experiment(
    config: ToyConfig,
    clock: Callable[[], dt.datetime] = _utc_now,
    publish: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_interval_seconds: float = PROGRESS_INTERVAL_SECONDS,
) -> Dict[str, Any]:
    """Run the grid, checking the deadline before every cell and round.

    The deadline is also checked after each cell's last round and at the end;
    finishing after it is reported as ``partial_deadline``. Never raises for
    failures during acquisition: the result carries a ``partial_*`` status and
    the cell inventory. ``publish`` receives the result at start, at round
    boundaries (at most every ``progress_interval_seconds`` by the clock),
    and after each cell.
    """
    if not isinstance(config, ToyConfig):
        raise TypeError("config must be a ToyConfig")
    cells = config.cells()
    started = _require_utc(clock(), "clock()")
    source = source_identity()
    result: Dict[str, Any] = {
        "protocol": PROTOCOL_VERSION,
        "source_sha256": source["sha256"],
        "seed": config.seeds[0] if len(config.seeds) == 1 else None,
        "status": "running",
        "config": config.to_json(),
        "source": source,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "semantics": SEMANTICS,
        "started_utc": _iso(started),
        "ended_utc": None,
        "deadline_utc": _iso(config.deadline),
        "campaign_cap_utc": _iso(CAMPAIGN_CAP_UTC),
        "deadline_overshoot": None,
        "inventory": {
            "cells_planned": len(cells),
            "cells_complete": [],
            "cells_overran_deadline": [],
            "partial_cell": None,
        },
        "cells": [],
        "error": None,
    }
    times = {"seen": started, "published": started}

    def expired() -> bool:
        times["seen"] = _require_utc(clock(), "clock()")
        return times["seen"] >= config.deadline

    def progress() -> None:
        elapsed = (times["seen"] - times["published"]).total_seconds()
        if publish is not None and elapsed >= progress_interval_seconds:
            publish(result)
            times["published"] = times["seen"]

    def mark_partial(status: str) -> None:
        result["status"] = status
        last = result["cells"][-1] if result["cells"] else None
        if last is not None and last["status"] not in ("complete", "overran_deadline"):
            last["status"] = status
            result["inventory"]["partial_cell"] = {
                "index": last["index"],
                "rounds_completed": last.get("rounds_completed", 0),
            }

    try:
        if publish is not None:
            publish(result)
        for index, (seed, horizon, alpha, kappa, q, u) in enumerate(cells):
            if expired():
                mark_partial("partial_deadline")
                break
            record: Dict[str, Any] = {
                "index": index,
                "status": "running",
                "cell": {
                    "seed": seed,
                    "horizon": horizon,
                    "alpha": alpha,
                    "kappa": kappa,
                    "q": q,
                    "orientation": u,
                },
            }
            result["cells"].append(record)
            run_cell(
                ToyMDP(horizon, alpha, kappa, q, u),
                seed,
                config.budget,
                config.batch,
                config.checkpoints,
                config.annotation_mode,
                should_stop=expired,
                record=record,
                on_round=progress,
            )
            if record["status"] == "overran_deadline":
                result["inventory"]["cells_overran_deadline"].append(index)
                mark_partial("partial_deadline")
                break
            if record["status"] != "complete":
                mark_partial("partial_deadline")
                break
            result["inventory"]["cells_complete"].append(index)
            if publish is not None:
                publish(result)
                times["published"] = times["seen"]
        else:
            result["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:  # preserve a safe partial
        mark_partial("partial_error")
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
    try:
        ended = _require_utc(clock(), "clock()")
        result["ended_utc"] = _iso(ended)
    except Exception as exc:  # the clock itself failed
        mark_partial("partial_error")
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
        return result
    if ended >= config.deadline:
        result["deadline_overshoot"] = {
            "ended_utc": _iso(ended),
            "seconds_past_deadline": (ended - config.deadline).total_seconds(),
            "note": "no rollout started after an observed expiration; at most the "
            "round in progress when the deadline passed ran past it",
        }
        if result["status"] == "complete":
            result["status"] = "partial_deadline"
    return result


class OutputRefused(Exception):
    pass


def claim_output_dir(
    path: str, clock: Callable[[], dt.datetime] = _utc_now
) -> pathlib.Path:
    """Claim a fresh or empty directory with an exclusive lock file."""
    out = pathlib.Path(path)
    if out.is_symlink():
        raise OutputRefused("output path is a symlink")
    if not out.parent.is_dir():
        raise OutputRefused("parent directory {} does not exist".format(out.parent))
    try:
        os.mkdir(str(out))
    except FileExistsError:
        if not out.is_dir() or any(out.iterdir()):
            raise OutputRefused(
                "output {} exists and is not empty; resume is not "
                "supported".format(out)
            )
    try:
        fd = os.open(str(out / CLAIM_NAME), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise OutputRefused("output {} was claimed concurrently".format(out))
    with os.fdopen(fd, "w") as handle:
        handle.write("claimed by {} at {}\n".format(PROTOCOL_VERSION, _iso(clock())))
        handle.flush()
        os.fsync(handle.fileno())
    return out


def publish_json(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    """Atomically replace ``path`` via a same-directory temp file and fsync."""
    text = json.dumps(obj, allow_nan=False, indent=1) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".result.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def parse_deadline(text: str) -> dt.datetime:
    value = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("deadline must be ISO 8601, e.g. 2026-10-07T01:01:00Z")
    return _require_utc(parsed, "deadline")


def _parser() -> argparse.ArgumentParser:
    defaults = ToyConfig()
    p = argparse.ArgumentParser(
        prog="python -m imitation.experiments.agnostic.toy",
        description="Exact agnostic DAgger toy (small pilot by default).",
    )
    p.add_argument("--output", help="fresh or empty output directory")
    p.add_argument("--seeds", type=int, nargs="+", default=list(defaults.seeds))
    p.add_argument("--horizons", type=int, nargs="+", default=list(defaults.horizons))
    p.add_argument("--alphas", type=float, nargs="+", default=list(defaults.alphas))
    p.add_argument("--kappas", type=float, nargs="+", default=list(defaults.kappas))
    p.add_argument("--qs", type=int, nargs="+", default=list(defaults.qs))
    p.add_argument(
        "--orientations", type=int, nargs="+", default=list(defaults.orientations)
    )
    p.add_argument(
        "--budget",
        type=int,
        default=defaults.budget,
        help="retained labels per arm (exact cap)",
    )
    p.add_argument(
        "--batch", type=int, default=defaults.batch, help="episodes per round"
    )
    p.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=None,
        help="retained-label checkpoints; default doubles from batch",
    )
    p.add_argument(
        "--annotation-mode", choices=ANNOTATION_MODES, default=defaults.annotation_mode
    )
    p.add_argument(
        "--deadline",
        default=_iso(CAMPAIGN_CAP_UTC),
        help="timezone-aware UTC deadline, not past the campaign cap",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the plan without acquisition",
    )
    return p


def main(
    argv: Optional[Sequence[str]] = None,
    clock: Callable[[], dt.datetime] = _utc_now,
) -> int:
    """CLI entry point. ``clock`` is injectable for tests; the real CLI uses UTC
    wall time, and every deadline is capped at ``CAMPAIGN_CAP_UTC``."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if not exc.code else EXIT_INVALID
    try:
        config = ToyConfig(
            seeds=tuple(args.seeds),
            horizons=tuple(args.horizons),
            alphas=tuple(args.alphas),
            kappas=tuple(args.kappas),
            qs=tuple(args.qs),
            orientations=tuple(args.orientations),
            budget=args.budget,
            batch=args.batch,
            checkpoints=None if args.checkpoints is None else tuple(args.checkpoints),
            annotation_mode=args.annotation_mode,
            deadline=parse_deadline(args.deadline),
        )
        if not args.dry_run and not args.output:
            raise ValueError("--output is required unless --dry-run is given")
    except ValueError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_INVALID
    if args.dry_run:
        plan = {
            "config": config.to_json(),
            "cells_planned": len(config.cells()),
            "rounds_per_cell": -(-config.budget // config.batch),
        }
        print(json.dumps(plan, indent=1))
        return EXIT_OK
    try:
        out = claim_output_dir(args.output, clock)
    except (OutputRefused, OSError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_INVALID
    path = out / RESULTS_NAME
    result = run_experiment(config, clock, publish=lambda r: publish_json(path, r))
    publish_json(path, result)
    print(
        "{}: {} of {} cells complete -> {}".format(
            result["status"],
            len(result["inventory"]["cells_complete"]),
            result["inventory"]["cells_planned"],
            path,
        )
    )
    if result["status"] == "complete":
        return EXIT_OK
    return EXIT_PARTIAL if result["status"] == "partial_deadline" else EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
