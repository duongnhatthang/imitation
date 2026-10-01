"""Single-environment rollout evaluation with factual cost accounting.

The evaluator drives one SB3 ``VecEnv`` (``n_envs=1``, autoreset) with a
behavior policy and counts what actually happened: real environment
transitions, completed episodes, predict invocations, and action entries
returned by those invocations. When the behavior policy is the expert, its
executed action is also the expert label, so the expert is queried once per
step, never twice.

Every rollout is bounded by an absolute timezone-aware UTC deadline and a
finite environment-step budget. Stopping early is not an error: the result is
marked incomplete and all counts, including the interrupted episode's
transitions, are preserved. If a policy query or an environment step raises,
`EvaluationError` carries the same partial result so callers can still record
what was spent.
"""

import dataclasses
import datetime
import time
from typing import Callable, List, Optional

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

UTC = datetime.timezone.utc

# Named phases with their own counters and RNG streams. Keep this order fixed:
# the index is the SeedSequence spawn key, so reordering changes every stream.
PHASES = (
    "expert_training",
    "expert_qualification",
    "train",
    "reference",
    "eval",
)

UNAVAILABLE = "unavailable"


class DeadlineExceeded(RuntimeError):
    """Raised when an absolute UTC deadline has passed before a step starts."""


class StepBudgetExhausted(RuntimeError):
    """Labels a rollout stopped by its environment-step budget."""


def utc_now() -> datetime.datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.datetime.now(UTC)


def parse_utc_deadline(text: str) -> datetime.datetime:
    """Parse an ISO 8601 timestamp with explicit offset into aware UTC.

    Args:
        text: Timestamp such as ``2026-10-07T01:01:00Z``. A trailing ``Z`` is
            accepted; a missing offset is rejected.

    Returns:
        The deadline converted to UTC.

    Raises:
        ValueError: If the timestamp is malformed or has no offset.
    """
    value = text.strip()
    if value.endswith(("Z", "z")):
        value = value[:-1] + "+00:00"
    parsed = datetime.datetime.fromisoformat(value)
    return ensure_aware_utc(parsed)


def ensure_aware_utc(value: datetime.datetime) -> datetime.datetime:
    """Return ``value`` in UTC, rejecting naive datetimes.

    Args:
        value: Datetime to check.

    Returns:
        The same instant with UTC tzinfo.

    Raises:
        ValueError: If ``value`` is naive.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"Deadline must be timezone-aware, got {value!r}")
    return value.astimezone(UTC)


def check_deadline(
    deadline: datetime.datetime,
    what: str,
    clock: Callable[[], datetime.datetime] = utc_now,
) -> None:
    """Raise `DeadlineExceeded` if ``deadline`` is not in the future.

    Args:
        deadline: Aware UTC deadline.
        what: Description of the step about to start, for the error message.
        clock: Returns the current aware time.

    Raises:
        DeadlineExceeded: If the deadline has passed.
    """
    now = clock()
    if now >= deadline:
        raise DeadlineExceeded(
            f"Deadline {deadline.isoformat()} passed before {what} "
            f"(now {now.isoformat()})",
        )


def make_phase_rng(seed: int, phase: str) -> np.random.Generator:
    """Return a reproducible RNG stream for ``phase`` derived from ``seed``.

    Different phases get statistically independent streams even for the same
    seed, so sampling, evaluation, and training never share draws.

    Args:
        seed: Non-negative base seed.
        phase: One of `PHASES`.

    Returns:
        A fresh generator.

    Raises:
        ValueError: If ``phase`` is unknown.
    """
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; expected one of {PHASES}")
    seq = np.random.SeedSequence(int(seed), spawn_key=(PHASES.index(phase),))
    return np.random.default_rng(seq)


@dataclasses.dataclass
class PhaseCounters:
    """Resources consumed by one phase.

    ``*_predict_calls`` counts invocations of ``predict`` that returned;
    ``*_action_entries`` counts the actions those invocations returned. A
    query or step that raises is not counted as a returned query or as a
    transition. They differ once batched
    queries are used. When the behavior policy is the expert, expert counters
    equal behavior counters because a single query serves as both the executed
    action and the label.
    """

    phase: str
    env_transitions: int = 0
    completed_episodes: int = 0
    behavior_predict_calls: int = 0
    behavior_action_entries: int = 0
    expert_predict_calls: int = 0
    expert_action_entries: int = 0
    partial_episode_transitions: int = 0
    partial_episode_return: float = 0.0
    elapsed_seconds: float = 0.0
    stop_reason: str = "not_started"

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict."""
        return dataclasses.asdict(self)


def unavailable_counters(phase: str, reason: str) -> dict:
    """Counter record for a phase whose costs an inherited routine hides.

    Args:
        phase: Phase name.
        reason: Why the counts are unavailable.

    Returns:
        A dict with the `PhaseCounters` fields set to ``"unavailable"``.
    """
    record = {
        field.name: UNAVAILABLE
        for field in dataclasses.fields(PhaseCounters)
        if field.name != "phase"
    }
    record["phase"] = phase
    record["unavailable_reason"] = reason
    return record


@dataclasses.dataclass
class EvaluationResult:
    """Outcome of `evaluate_episodes`.

    ``episode_returns`` are undiscounted native environment returns of
    completed episodes only. ``disagreement`` is the fraction of executed steps
    where the deterministic expert action differs from the behavior action, or
    None if no expert was supplied.
    """

    episode_returns: List[float]
    episode_lengths: List[int]
    disagreement: Optional[float]
    counters: PhaseCounters
    completed: bool


class EvaluationError(RuntimeError):
    """A policy query or env step raised during `evaluate_episodes`.

    The original exception is chained as ``__cause__``; ``result`` holds the
    factual counts and completed episodes up to the failure, with
    ``counters.stop_reason == "error"``.
    """

    def __init__(self, result: "EvaluationResult", cause: BaseException):
        super().__init__(f"Evaluation failed: {type(cause).__name__}: {cause}")
        self.result = result


def _predict(policy, obs: np.ndarray, deterministic: bool) -> np.ndarray:
    actions, _ = policy.predict(obs, deterministic=deterministic)
    return np.asarray(actions)


def evaluate_episodes(
    venv: VecEnv,
    policy,
    n_episodes: int,
    *,
    phase: str,
    deadline: datetime.datetime,
    max_env_steps: int,
    behavior_is_expert: bool = False,
    expert=None,
    deterministic: bool = True,
    env_seed: Optional[int] = None,
    clock: Callable[[], datetime.datetime] = utc_now,
) -> EvaluationResult:
    """Roll out ``policy`` for ``n_episodes`` complete episodes on one env.

    Args:
        venv: Single-environment SB3 VecEnv with autoreset, for example from
            ``env_utils.make_env(env, n_envs=1, rng=...)``. It is reset here.
        policy: Behavior policy exposing SB3-style ``predict(obs, deterministic)``.
        n_episodes: Number of complete episodes requested.
        phase: One of `PHASES`, recorded in the counters.
        deadline: Aware absolute deadline, checked before every step.
        max_env_steps: Positive cap on environment transitions.
        behavior_is_expert: True if ``policy`` is the expert. Its actions are
            then counted as expert entries and it is not queried a second time.
        expert: Optional deterministic expert queried for disagreement when the
            behavior is not the expert. Ignored for querying if
            ``behavior_is_expert``.
        deterministic: Passed to the behavior policy's ``predict``.
        env_seed: If given, the env is reset with this seed; the following
            episodes continue the env's own RNG stream.
        clock: Returns the current aware time.

    Returns:
        Episode returns, lengths, optional disagreement, and counters.

    Raises:
        ValueError: On a non-positive budget, naive deadline, unknown phase,
            or a VecEnv with more than one environment.
        EvaluationError: If seeding, reset, a policy query, or an env step
            raises. It carries the partial result.
    """
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}")
    if int(max_env_steps) <= 0:
        raise ValueError(f"max_env_steps must be positive, got {max_env_steps}")
    if int(n_episodes) <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    deadline = ensure_aware_utc(deadline)
    if venv.num_envs != 1:
        raise ValueError(f"Expected a single-env VecEnv, got {venv.num_envs}")

    counters = PhaseCounters(phase=phase, stop_reason="running")
    returns: List[float] = []
    lengths: List[int] = []
    mismatches = 0
    labelled_steps = 0
    ep_return = 0.0
    ep_len = 0
    started = time.monotonic()

    def result() -> EvaluationResult:
        disagreement: Optional[float] = None
        if behavior_is_expert or expert is not None:
            disagreement = mismatches / labelled_steps if labelled_steps else None
        return EvaluationResult(
            episode_returns=returns,
            episode_lengths=lengths,
            disagreement=disagreement,
            counters=counters,
            completed=counters.stop_reason == "completed",
        )

    try:
        if env_seed is not None:
            venv.seed(int(env_seed))
        obs = venv.reset()
        while len(returns) < n_episodes:
            if counters.env_transitions >= max_env_steps:
                counters.stop_reason = "step_budget"
                break
            if clock() >= deadline:
                counters.stop_reason = "deadline"
                break

            actions = _predict(policy, obs, deterministic)
            counters.behavior_predict_calls += 1
            counters.behavior_action_entries += int(actions.shape[0])
            if behavior_is_expert:
                counters.expert_predict_calls += 1
                counters.expert_action_entries += int(actions.shape[0])
                labelled_steps += int(actions.shape[0])
            elif expert is not None:
                labels = _predict(expert, obs, True)
                counters.expert_predict_calls += 1
                counters.expert_action_entries += int(labels.shape[0])
                mismatches += int(np.sum(labels.reshape(-1) != actions.reshape(-1)))
                labelled_steps += int(labels.shape[0])

            obs, rewards, dones, _ = venv.step(actions)
            counters.env_transitions += 1
            ep_return += float(rewards[0])
            ep_len += 1
            if dones[0]:
                returns.append(ep_return)
                lengths.append(ep_len)
                counters.completed_episodes += 1
                ep_return = 0.0
                ep_len = 0
        else:
            counters.stop_reason = "completed"
    except Exception as exc:
        counters.stop_reason = "error"
        _finalize(counters, ep_len, ep_return, started)
        raise EvaluationError(result(), exc) from exc
    finally:
        _finalize(counters, ep_len, ep_return, started)
    return result()


def _finalize(
    counters: PhaseCounters,
    ep_len: int,
    ep_return: float,
    started: float,
) -> None:
    counters.partial_episode_transitions = ep_len
    counters.partial_episode_return = ep_return
    counters.elapsed_seconds = time.monotonic() - started
    if counters.stop_reason == "running":
        counters.stop_reason = "error"
