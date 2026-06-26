"""Unified policy evaluation for FTRL experiments.

Single entry point for rolling out a policy, measuring its return, and
(optionally) collecting expert argmax actions at every visited state. The
caller is responsible for aggregating rollout batches across eval points
into a running D_eval buffer — see the spec
docs/superpowers/specs/2026-04-10-ftrl-eval-distribution-fix.md §3.4.
"""

import dataclasses
from typing import List, Optional

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
    "CEBreakdown",
    "compute_ce_breakdown",
]


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
    """Return value from `eval_policy_rollout`."""

    mean_return: float
    episode_returns: List[float]
    n_steps: int
    rollout_batch: Optional[RolloutBatch]
    current_round_disagreement: Optional[float]
    current_round_ce: Optional[float]


def eval_policy_rollout(
    policy: BasePolicy,
    venv: VecEnv,
    n_episodes: int,
    deterministic: bool = True,
    expert_policy: Optional[BasePolicy] = None,
    safety_step_cap: int = 1_000_000,
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

    Returns:
        EvalResult with mean_return computed from Monitor episode_returns
        (matching SB3's evaluate_policy) when available.

    Raises:
        ValueError: If venv.num_envs != 1.
        RuntimeError: If safety_step_cap is hit before n_episodes complete.
    """
    if venv.num_envs != 1:
        raise ValueError(
            f"eval_policy_rollout requires num_envs=1, got {venv.num_envs}"
        )

    obs_buf: List[np.ndarray] = []
    expert_act_buf: List[int] = []
    learner_act_buf: List[int] = []
    episode_returns: List[float] = []

    total_steps = 0
    current_return = 0.0

    obs = venv.reset()
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
        current_return += float(rewards[0])

        if dones[0]:
            info = infos[0]
            if "episode" in info and "r" in info["episode"]:
                episode_returns.append(float(info["episode"]["r"]))
            else:
                episode_returns.append(current_return)
            current_return = 0.0

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

    return EvalResult(
        mean_return=float(np.mean(episode_returns)),
        episode_returns=episode_returns,
        n_steps=total_steps,
        rollout_batch=rollout_batch,
        current_round_disagreement=current_round_disagreement,
        current_round_ce=current_round_ce,
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


@dataclasses.dataclass
class CEBreakdown:
    """Sampled-action CE split by whether the learner's argmax matches expert.

    `ce_correct`/`conf_correct` cover states where the learner's deterministic
    action equals the expert's; `ce_wrong`/`conf_wrong` cover the rest. CE is the
    mean of -log pi(a*|s) within the bucket; confidence is the mean max softmax
    probability. Empty buckets report NaN (count 0).
    """

    ce_correct: float
    ce_wrong: float
    n_correct: int
    n_wrong: int
    conf_correct: float
    conf_wrong: float


def compute_ce_breakdown(
    policy: BasePolicy,
    obs: np.ndarray,
    expert_actions: np.ndarray,
    learner_actions: np.ndarray,
    batch_size: int = 2048,
) -> CEBreakdown:
    """Split sampled-action CE into correct- vs wrong-argmax buckets.

    Args:
        policy: Policy with `.evaluate_actions` and `.get_distribution`.
        obs: Rollout state buffer, shape (N, *obs_shape).
        expert_actions: Expert argmax actions, shape (N,).
        learner_actions: Learner deterministic actions, shape (N,).
        batch_size: Mini-batch size for the forward pass.

    Returns:
        CEBreakdown with per-bucket mean CE, counts, and mean confidence.
    """
    correct_mask = np.asarray(learner_actions) == np.asarray(expert_actions)
    wrong_mask = ~correct_mask

    was_training = policy.training
    policy.eval()
    device = policy.device

    ce_sum = {"correct": 0.0, "wrong": 0.0}
    conf_sum = {"correct": 0.0, "wrong": 0.0}
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
                neglogp = (-log_prob).cpu().numpy()
                probs = policy.get_distribution(tensor_obs).distribution.probs
                max_prob = probs.max(dim=1).values.cpu().numpy()

                cm = correct_mask[start:end]
                ce_sum["correct"] += float(neglogp[cm].sum())
                ce_sum["wrong"] += float(neglogp[~cm].sum())
                conf_sum["correct"] += float(max_prob[cm].sum())
                conf_sum["wrong"] += float(max_prob[~cm].sum())
    finally:
        if was_training:
            policy.train()

    n_correct = int(correct_mask.sum())
    n_wrong = int(wrong_mask.sum())

    def _mean(total: float, count: int) -> float:
        return total / count if count > 0 else float("nan")

    return CEBreakdown(
        ce_correct=_mean(ce_sum["correct"], n_correct),
        ce_wrong=_mean(ce_sum["wrong"], n_wrong),
        n_correct=n_correct,
        n_wrong=n_wrong,
        conf_correct=_mean(conf_sum["correct"], n_correct),
        conf_wrong=_mean(conf_sum["wrong"], n_wrong),
    )
