"""Tests for unified policy evaluation in eval_utils."""

import numpy as np
import pytest
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.experiments.ftrl import env_utils, experts
from imitation.experiments.ftrl.eval_utils import (
    EvalResult,
    RolloutBatch,
    compute_sampled_action_ce,
    eval_policy_rollout,
)


def test_eval_matches_sb3_evaluate_policy_cartpole(tmp_path):
    """eval_policy_rollout must match SB3's evaluate_policy on CartPole."""
    rng = np.random.default_rng(0)
    venv = env_utils.make_env("CartPole-v1", n_envs=1, rng=rng)
    expert = experts.get_or_train_expert(
        "CartPole-v1", venv, cache_dir=tmp_path, rng=rng, seed=0
    )

    venv.reset()
    ours = eval_policy_rollout(expert, venv, n_episodes=20, deterministic=True)
    venv.reset()
    sb3_mean, _ = evaluate_policy(
        expert, venv, n_eval_episodes=20, deterministic=True
    )

    assert isinstance(ours, EvalResult)
    assert len(ours.episode_returns) == 20
    assert abs(ours.mean_return - sb3_mean) < 1e-4


def test_eval_raises_on_safety_cap():
    """Hitting safety_step_cap must raise RuntimeError, not silently truncate."""
    rng = np.random.default_rng(0)
    venv = env_utils.make_env("CartPole-v1", n_envs=1, rng=rng)

    class NullPolicy:
        def predict(self, obs, deterministic=True):
            # Action 0 keeps CartPole running for a while but hits cap
            return np.zeros(len(obs), dtype=np.int64), None

    with pytest.raises(RuntimeError, match="safety_step_cap"):
        eval_policy_rollout(
            NullPolicy(), venv, n_episodes=1000, safety_step_cap=50
        )


def test_rollout_batch_expert_actions_are_argmax(tmp_path):
    """When expert_policy is provided, rollout_batch.expert_actions must be
    the expert's deterministic argmax at every visited state."""
    rng = np.random.default_rng(0)
    venv = env_utils.make_env("CartPole-v1", n_envs=1, rng=rng)
    expert = experts.get_or_train_expert(
        "CartPole-v1", venv, cache_dir=tmp_path, rng=rng, seed=0
    )

    res = eval_policy_rollout(
        expert, venv, n_episodes=5, deterministic=True, expert_policy=expert
    )
    assert res.rollout_batch is not None
    # When learner == expert, disagreement must be zero.
    assert res.current_round_disagreement == 0.0
    # Shapes line up
    assert res.rollout_batch.obs.shape[0] == res.rollout_batch.expert_actions.shape[0]
    assert res.rollout_batch.obs.shape[0] == res.n_steps


def test_compute_sampled_action_ce_matches_bc_loss_calculator(tmp_path):
    """compute_sampled_action_ce must match bc.BehaviorCloningLossCalculator's
    neglogp on the same (obs, acts)."""
    import torch as th
    from imitation.algorithms.bc import BehaviorCloningLossCalculator

    rng = np.random.default_rng(0)
    venv = env_utils.make_env("CartPole-v1", n_envs=1, rng=rng)
    expert = experts.get_or_train_expert(
        "CartPole-v1", venv, cache_dir=tmp_path, rng=rng, seed=0
    )

    res = eval_policy_rollout(
        expert, venv, n_episodes=3, deterministic=True, expert_policy=expert
    )
    obs = res.rollout_batch.obs
    acts = res.rollout_batch.expert_actions

    ours = compute_sampled_action_ce(expert, obs, acts)

    calc = BehaviorCloningLossCalculator(ent_weight=0.0, l2_weight=0.0)
    with th.no_grad():
        metrics = calc(expert, obs, acts)
    theirs = metrics.neglogp.item()

    assert abs(ours - theirs) < 1e-4
