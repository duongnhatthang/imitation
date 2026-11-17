"""Unified policy evaluation for FTRL experiments.

Single entry point for rolling out a policy, measuring its return, and
(optionally) collecting expert argmax actions at every visited state. The
caller is responsible for aggregating rollout batches across eval points
into a running D_eval buffer — see the spec
docs/superpowers/specs/2026-04-10-ftrl-eval-distribution-fix.md §3.4.

Episode outcome evidence
------------------------
Every rollout also records, per completed episode, its length, end flags,
success, and a digest of its initial observation. These are evidence for
outcome-aware mask calibration (`imitation.experiments.agnostic.
mask_calibration`), not training signals.

End flags. SB3 VecEnvs return one ``done`` and set
``info["TimeLimit.truncated"] = truncated and not terminated``. A raw Gym step
that is both terminated and truncated (for example a goal reached on the
TimeLimit step) is therefore reported as terminated only. Without raw flags
the recorded pair is an inferred, mutually exclusive classification
(`FLAGS_SB3_EXCLUSIVE`) and does not claim to know whether the raw step was
also truncated. `RawTerminationInfoWrapper`, placed outside the TimeLimit,
copies the raw Gym flags into the info dict; when both raw keys are present
they are recorded as given (`FLAGS_RAW`).

Success. A terminal ``info["is_success"]`` wins when present. Otherwise the
per-environment rule from `SUCCESS_RULES` applies (see each rule's
description). Unknown environments record None, never a guessed value. The
rules read the terminal reward, end flags, and episode return; none needs the
terminal observation, so ``info["terminal_observation"]`` is not consulted.

Paired episodes. With ``episode_seeds`` every episode starts from an explicit
``venv.seed(s); venv.reset()``. The VecEnv's own auto-reset after each done
consumes its hidden RNG and is discarded (counted in ``reset_accounting``);
the explicit seeded reset reproduces the same initial state for any policy.
Pairing is a property of these explicit seeds, never of a shared starting
RNG.
"""

import dataclasses
import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

from imitation.data import types
from imitation.util import util

__all__ = [
    "RolloutBatch",
    "EvalResult",
    "eval_policy_rollout",
    "compute_sampled_action_ce",
    "SuccessRule",
    "SUCCESS_RULES",
    "IS_SUCCESS_RULE",
    "FLAGS_RAW",
    "FLAGS_SB3_EXCLUSIVE",
    "RAW_TERMINATED_KEY",
    "RAW_TRUNCATED_KEY",
    "RawTerminationInfoWrapper",
    "success_rule_for",
    "episode_success",
    "episode_end_flags",
    "observation_digest",
    "episode_evidence",
]

RAW_TERMINATED_KEY = "raw_terminated"
RAW_TRUNCATED_KEY = "raw_truncated"
FLAGS_RAW = "raw_gym_flags"
FLAGS_SB3_EXCLUSIVE = "sb3_time_limit_inferred_exclusive"


@dataclasses.dataclass(frozen=True)
class SuccessRule:
    """How one environment's episode success is read from its outcome.

    Attributes:
        name: Stable rule identifier recorded with the evidence.
        description: The exact criterion, in words.
        used_for_calibration: False when calibration must ignore the success
            indicator (it is recorded for reference only).
    """

    name: str
    description: str
    used_for_calibration: bool = True


IS_SUCCESS_RULE = SuccessRule(
    "info_is_success",
    "terminal info['is_success'] supplied by the environment or a wrapper",
)
_TERMINAL_REWARD_POSITIVE = SuccessRule(
    "terminal_reward_positive",
    "terminated with a positive terminal reward (the goal pays 1, holes pay 0)",
)
_GOAL_TERMINATED = SuccessRule(
    "goal_terminated",
    "terminated; the environment terminates only at its goal, and a goal "
    "reached on the TimeLimit step still counts (SB3 reports it terminated)",
)
_SURVIVED = SuccessRule(
    "survived_to_time_limit",
    "TimeLimit truncated without termination (the pole never fell)",
)
_LANDING_BONUS = SuccessRule(
    "landing_bonus",
    "terminated with terminal reward +100, the Gym landing bonus that "
    "replaces the step reward when the lander comes to rest",
)
_POSITIVE_RETURN = SuccessRule(
    "positive_return",
    "episode return > 0 (a win); recorded only, calibration ignores it",
    used_for_calibration=False,
)

SUCCESS_RULES: Dict[str, SuccessRule] = {
    "FrozenLake-v1": _TERMINAL_REWARD_POSITIVE,
    "FrozenLakeVariedStart-v0": _TERMINAL_REWARD_POSITIVE,
    "FrozenLakeSlippery-v0": _TERMINAL_REWARD_POSITIVE,
    "CliffWalking-v0": _GOAL_TERMINATED,
    "Taxi-v3": _GOAL_TERMINATED,
    "Acrobot-v1": _GOAL_TERMINATED,
    "MountainCar-v0": _GOAL_TERMINATED,
    "CartPole-v1": _SURVIVED,
    "LunarLander-v2": _LANDING_BONUS,
    "LunarLander-v3": _LANDING_BONUS,
    "Blackjack-v1": _POSITIVE_RETURN,
}

_LANDING_BONUS_REWARD = 100.0


def _base_env_id(env_id: Optional[str]) -> Optional[str]:
    """Strip a Gymnasium namespace (``ns/Name-v0`` becomes ``Name-v0``)."""
    if env_id is None:
        return None
    return env_id.rsplit("/", 1)[-1]


def success_rule_for(env_id: Optional[str]) -> Optional[SuccessRule]:
    """Known success rule of ``env_id``, or None for an unknown environment."""
    return SUCCESS_RULES.get(_base_env_id(env_id) or "")


def episode_success(
    env_id: Optional[str],
    *,
    terminated: bool,
    truncated: bool,
    terminal_reward: float,
    episode_return: float,
    terminal_info: Optional[Mapping[str, Any]] = None,
) -> Optional[bool]:
    """Success of one finished episode, or None when no rule is known.

    Args:
        env_id: Gymnasium id, namespace optional.
        terminated: Episode termination flag (see the module note on flags).
        truncated: Episode truncation flag.
        terminal_reward: Reward of the final step.
        episode_return: Undiscounted episode return.
        terminal_info: Final step info; ``is_success`` there wins.

    Returns:
        True or False under `IS_SUCCESS_RULE` or the environment's rule, else
        None.
    """
    if terminal_info is not None and "is_success" in terminal_info:
        return bool(terminal_info["is_success"])
    rule = success_rule_for(env_id)
    if rule is None:
        return None
    if rule is _TERMINAL_REWARD_POSITIVE:
        return bool(terminated and terminal_reward > 0)
    if rule is _GOAL_TERMINATED:
        return bool(terminated)
    if rule is _SURVIVED:
        return bool(truncated and not terminated)
    if rule is _LANDING_BONUS:
        return bool(terminated and math.isclose(terminal_reward, _LANDING_BONUS_REWARD))
    if rule is _POSITIVE_RETURN:
        return bool(episode_return > 0)
    raise AssertionError(f"unhandled success rule {rule.name}")


def episode_end_flags(info: Mapping[str, Any]) -> Tuple[bool, bool, str]:
    """``(terminated, truncated, source)`` of an episode that just ended.

    Uses the raw keys written by `RawTerminationInfoWrapper` when both are
    present, else infers exclusive flags from SB3's ``TimeLimit.truncated``.
    """
    if RAW_TERMINATED_KEY in info and RAW_TRUNCATED_KEY in info:
        return bool(info[RAW_TERMINATED_KEY]), bool(info[RAW_TRUNCATED_KEY]), FLAGS_RAW
    truncated = bool(info.get("TimeLimit.truncated", False))
    return not truncated, truncated, FLAGS_SB3_EXCLUSIVE


class RawTerminationInfoWrapper(gym.Wrapper):
    """Copy the raw Gym ``terminated``/``truncated`` flags into ``info``.

    Place it outside every TimeLimit so it sees the final flags. SB3 drops
    the distinction when both are True; these keys preserve it.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info[RAW_TERMINATED_KEY] = bool(terminated)
        info[RAW_TRUNCATED_KEY] = bool(truncated)
        return obs, reward, terminated, truncated, info


def observation_digest(obs: Any) -> str:
    """SHA-256 of one observation's dtype, shape, and bytes."""
    arr = np.ascontiguousarray(np.asarray(obs))
    h = hashlib.sha256()
    h.update(f"{arr.dtype.str}|{arr.shape}|".encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _validate_episode_seeds(
    episode_seeds: Optional[Sequence[int]],
    n_episodes: int,
) -> Optional[List[int]]:
    if episode_seeds is None:
        return None
    seeds = list(episode_seeds)
    if len(seeds) != n_episodes:
        raise ValueError(
            f"episode_seeds must list exactly n_episodes={n_episodes} seeds, "
            f"got {len(seeds)}",
        )
    for seed in seeds:
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError(
                f"episode_seeds must be non-negative ints, got {seed!r}",
            )
    if len(set(seeds)) != len(seeds):
        raise ValueError("episode_seeds must be distinct")
    return [int(seed) for seed in seeds]


def _venv_env_id(venv: VecEnv) -> Optional[str]:
    try:
        spec = venv.get_attr("spec", indices=[0])[0]
    except Exception:  # Not every VecEnv or wrapper exposes a spec.
        return None
    return getattr(spec, "id", None)


@dataclasses.dataclass
class RolloutBatch:
    """Raw transitions collected during one eval call.

    Owned by the caller; aggregated across eval points into D_eval^t.

    Note:
        `obs` is assumed to be an ndarray (Box or one-hot-encoded Discrete
        observation spaces used by the 8 classical MDPs in this PR's scope).
        Dict observation spaces are not supported here — extend this class
        if/when that becomes necessary.
    """

    obs: np.ndarray  # shape (n_steps, *obs_shape)
    expert_actions: np.ndarray  # shape (n_steps,) — a*(s) = argmax π*(·|s)
    learner_actions: np.ndarray  # shape (n_steps,) — deterministic learner action


@dataclasses.dataclass
class EvalResult:
    """Return value from `eval_policy_rollout`.

    Fields after ``current_round_ce`` are optional episode outcome evidence
    (see the module docstring). They default to None so results built by
    older code stay valid. Per-episode lists align with ``episode_returns``.
    ``episode_successes`` is None when no success rule is known.
    ``reset_accounting`` counts ``explicit_resets``, VecEnv ``auto_resets``,
    and ``discarded_auto_resets`` (auto-reset states never played);
    ``n_steps`` counts environment steps only, never resets.
    """

    mean_return: float
    episode_returns: List[float]
    n_steps: int
    rollout_batch: Optional[RolloutBatch]
    current_round_disagreement: Optional[float]
    current_round_ce: Optional[float]
    episode_lengths: Optional[List[int]] = None
    episode_terminated: Optional[List[bool]] = None
    episode_truncated: Optional[List[bool]] = None
    episode_successes: Optional[List[bool]] = None
    evaluation_episode_seeds: Optional[List[int]] = None
    initial_state_digests: Optional[List[str]] = None
    success_rule: Optional[str] = None
    termination_flag_source: Optional[str] = None
    reset_accounting: Optional[Dict[str, int]] = None


def episode_evidence(result: EvalResult) -> Dict[str, Any]:
    """JSON-ready outcome evidence of ``result`` for persisting with a row."""
    return {
        "episode_returns": [float(r) for r in result.episode_returns],
        "episode_lengths": _maybe_list(result.episode_lengths, int),
        "episode_terminated": _maybe_list(result.episode_terminated, bool),
        "episode_truncated": _maybe_list(result.episode_truncated, bool),
        "episode_successes": _maybe_list(result.episode_successes, bool),
        "evaluation_episode_seeds": _maybe_list(result.evaluation_episode_seeds, int),
        "initial_state_digests": _maybe_list(result.initial_state_digests, str),
        "success_rule": result.success_rule,
        "termination_flag_source": result.termination_flag_source,
        "reset_accounting": (
            None
            if result.reset_accounting is None
            else {k: int(v) for k, v in result.reset_accounting.items()}
        ),
    }


def _maybe_list(values: Optional[Sequence[Any]], cast) -> Optional[List[Any]]:
    return None if values is None else [cast(v) for v in values]


def eval_policy_rollout(
    policy: BasePolicy,
    venv: VecEnv,
    n_episodes: int,
    deterministic: bool = True,
    expert_policy: Optional[BasePolicy] = None,
    safety_step_cap: int = 1_000_000,
    episode_seeds: Optional[Sequence[int]] = None,
    env_id: Optional[str] = None,
) -> EvalResult:
    """Roll out `policy` for exactly `n_episodes` complete episodes.

    Args:
        policy: The policy to evaluate.
        venv: Single-env VecEnv (num_envs must be 1).
        n_episodes: Exact number of complete episodes to collect.
        deterministic: Pass to policy.predict.
        expert_policy: If provided, query at every step and store
            the expert's deterministic action in the returned RolloutBatch.
        safety_step_cap: Hard ceiling on environment steps. Exceeding
            raises RuntimeError.
        episode_seeds: Optional reset seed per episode, exactly
            ``n_episodes`` distinct non-negative ints. Each episode then
            starts from ``venv.seed(seed); venv.reset()``, so policies
            evaluated with the same list are paired episode by episode.
            None keeps the single initial reset and VecEnv auto-resets.
        env_id: Gymnasium id used to pick the success rule. None reads it
            from the env spec when available.

    Returns:
        EvalResult with mean_return computed from Monitor episode_returns
        (matching SB3's evaluate_policy) when available, plus episode
        outcome evidence.

    Raises:
        ValueError: If venv.num_envs != 1 or ``episode_seeds`` is malformed.
        RuntimeError: If safety_step_cap is hit before n_episodes complete.
    """
    if venv.num_envs != 1:
        raise ValueError(
            f"eval_policy_rollout requires num_envs=1, got {venv.num_envs}"
        )
    seeds = _validate_episode_seeds(episode_seeds, n_episodes)
    if env_id is None:
        env_id = _venv_env_id(venv)

    obs_buf: List[np.ndarray] = []
    expert_act_buf: List[int] = []
    learner_act_buf: List[int] = []
    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    episode_terminated: List[bool] = []
    episode_truncated: List[bool] = []
    episode_successes: List[Optional[bool]] = []
    success_sources: List[str] = []
    flag_sources: List[str] = []
    initial_digests: List[str] = []
    resets = {"explicit_resets": 0, "auto_resets": 0, "discarded_auto_resets": 0}

    def explicit_reset(episode: int):
        if seeds is not None:
            venv.seed(seeds[episode])
        resets["explicit_resets"] += 1
        return venv.reset()

    total_steps = 0
    current_return = 0.0
    current_length = 0

    obs = explicit_reset(0)
    initial_digests.append(observation_digest(obs[0]))
    while len(episode_returns) < n_episodes:
        if total_steps >= safety_step_cap:
            raise RuntimeError(
                f"eval_policy_rollout hit safety_step_cap={safety_step_cap} "
                f"after collecting {len(episode_returns)}/{n_episodes} "
                f"episodes. Env may be non-terminating."
            )

        learner_action, _ = policy.predict(obs, deterministic=deterministic)
        if expert_policy is not None:
            expert_action, _ = expert_policy.predict(obs, deterministic=True)
            obs_buf.append(np.asarray(obs[0]))
            expert_act_buf.append(int(expert_action[0]))
            learner_act_buf.append(int(learner_action[0]))

        obs, rewards, dones, infos = venv.step(learner_action)
        total_steps += 1
        current_length += 1
        current_return += float(rewards[0])

        if dones[0]:
            info = infos[0]
            if "episode" in info and "r" in info["episode"]:
                episode_return = float(info["episode"]["r"])
            else:
                episode_return = current_return
            episode_returns.append(episode_return)
            terminated, truncated, flag_source = episode_end_flags(info)
            episode_lengths.append(current_length)
            episode_terminated.append(terminated)
            episode_truncated.append(truncated)
            flag_sources.append(flag_source)
            episode_successes.append(
                episode_success(
                    env_id,
                    terminated=terminated,
                    truncated=truncated,
                    terminal_reward=float(rewards[0]),
                    episode_return=episode_return,
                    terminal_info=info,
                ),
            )
            success_sources.append(
                (
                    IS_SUCCESS_RULE.name
                    if "is_success" in info
                    else getattr(success_rule_for(env_id), "name", "")
                ),
            )
            current_return = 0.0
            current_length = 0
            # The VecEnv already auto-reset; keep that state only when it
            # starts the next unseeded episode.
            resets["auto_resets"] += 1
            if len(episode_returns) == n_episodes:
                resets["discarded_auto_resets"] += 1
            elif seeds is not None:
                resets["discarded_auto_resets"] += 1
                obs = explicit_reset(len(episode_returns))
                initial_digests.append(observation_digest(obs[0]))
            else:
                initial_digests.append(observation_digest(obs[0]))

    rollout_batch: Optional[RolloutBatch] = None
    current_round_disagreement: Optional[float] = None
    current_round_ce: Optional[float] = None
    if expert_policy is not None:
        obs_arr = np.stack(obs_buf, axis=0)
        expert_arr = np.asarray(expert_act_buf, dtype=np.int64)
        learner_arr = np.asarray(learner_act_buf, dtype=np.int64)
        rollout_batch = RolloutBatch(
            obs=obs_arr,
            expert_actions=expert_arr,
            learner_actions=learner_arr,
        )
        current_round_disagreement = float(np.mean(expert_arr != learner_arr))
        current_round_ce = compute_sampled_action_ce(policy, obs_arr, expert_arr)

    known = all(s is not None for s in episode_successes)
    success_rule = "|".join(sorted(set(success_sources))) if known else None
    return EvalResult(
        mean_return=float(np.mean(episode_returns)),
        episode_returns=episode_returns,
        n_steps=total_steps,
        rollout_batch=rollout_batch,
        current_round_disagreement=current_round_disagreement,
        current_round_ce=current_round_ce,
        episode_lengths=episode_lengths,
        episode_terminated=episode_terminated,
        episode_truncated=episode_truncated,
        episode_successes=([bool(s) for s in episode_successes] if known else None),
        evaluation_episode_seeds=seeds,
        initial_state_digests=initial_digests,
        success_rule=success_rule,
        termination_flag_source="|".join(sorted(set(flag_sources))),
        reset_accounting=resets,
    )


def compute_sampled_action_ce(
    policy: BasePolicy,
    obs: np.ndarray,
    expert_actions: np.ndarray,
    batch_size: int = 2048,
) -> float:
    """Compute -1/|D| * sum log π(a*|s) on an arbitrary aggregated buffer.

    Single batched forward pass, no autograd. Uses policy.evaluate_actions
    to exactly match the loss form minimized by
    `imitation.algorithms.bc.BehaviorCloningLossCalculator`.

    Args:
        policy: Policy with `.evaluate_actions(obs, acts)`.
        obs: Aggregated state buffer, shape (N, *obs_shape).
        expert_actions: Aggregated expert argmax actions, shape (N,).
        batch_size: Mini-batch size for the forward pass.

    Returns:
        Mean sampled-action cross-entropy (scalar float).

    Note:
        Assumes the policy has a uniform train/eval mode across all
        submodules (true for the MLP ActorCritic policies used by the
        classical FTRL pipeline). Callers with policies that have
        independently-toggled submodules (e.g. frozen BatchNorm) should
        snapshot mode per-submodule themselves.
    """
    was_training = policy.training
    policy.eval()
    device = policy.device
    total = 0.0
    n = int(obs.shape[0])
    try:
        with th.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_obs = obs[start:end]
                batch_acts = expert_actions[start:end]
                tensor_obs = types.map_maybe_dict(
                    lambda x: util.safe_to_tensor(x).to(device),
                    types.maybe_unwrap_dictobs(batch_obs),
                )
                tensor_acts = util.safe_to_tensor(batch_acts).to(device)
                _, log_prob, _ = policy.evaluate_actions(tensor_obs, tensor_acts)
                total += float(-log_prob.sum().item())
    finally:
        if was_training:
            policy.train()
    return total / max(n, 1)
