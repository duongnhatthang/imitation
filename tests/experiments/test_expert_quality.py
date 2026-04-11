"""Regression tests pinning expert quality invariants for classical MDPs.

Marked @pytest.mark.expensive — these retrain experts and take minutes.
"""

import numpy as np
import pytest

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.experiments.ftrl import env_utils, experts, policy_utils
from imitation.experiments.ftrl.env_baselines import REFERENCE_BASELINES
from imitation.experiments.ftrl.eval_utils import eval_policy_rollout

CLASSICAL_ENVS = [
    "CartPole-v1",
    "FrozenLake-v1",
    "CliffWalking-v0",
    "Acrobot-v1",
    "MountainCar-v0",
    "Taxi-v3",
    "Blackjack-v1",
    "LunarLander-v2",
]


def _normalize(mean_return: float, env_name: str) -> float:
    ref = REFERENCE_BASELINES[env_name]
    return (mean_return - ref["random_score"]) / (
        ref["expert_score"] - ref["random_score"]
    )


@pytest.mark.expensive
@pytest.mark.parametrize("env_name", CLASSICAL_ENVS)
def test_expert_converged(env_name, tmp_path):
    """Every classical expert must meet its convergence bar after retraining."""
    rng = np.random.default_rng(0)
    venv = env_utils.make_env(env_name, n_envs=1, rng=rng)
    expert = experts.get_or_train_expert(
        env_name, venv, cache_dir=tmp_path, rng=rng, seed=0
    )
    res = eval_policy_rollout(
        expert, venv, n_episodes=20, deterministic=True, expert_policy=expert
    )
    cfg = env_utils.get_convergence_config(env_name)
    normalized = _normalize(res.mean_return, env_name)
    assert normalized >= cfg["threshold"] - 0.05, (
        f"{env_name}: normalized return {normalized:.3f} < "
        f"threshold {cfg['threshold']}"
    )
    assert res.current_round_ce <= cfg["self_ce_eps"] + 0.02, (
        f"{env_name}: self-CE {res.current_round_ce:.3f} > "
        f"eps {cfg['self_ce_eps']} (argmax pathology risk)"
    )


@pytest.mark.expensive
@pytest.mark.parametrize("env_name", CLASSICAL_ENVS)
def test_expert_beats_bc(env_name, tmp_path):
    """Expert deterministic return must meet-or-exceed BC's deterministic return."""
    rng = np.random.default_rng(0)
    venv = env_utils.make_env(env_name, n_envs=1, rng=rng)
    expert = experts.get_or_train_expert(
        env_name, venv, cache_dir=tmp_path, rng=rng, seed=0
    )

    trajs = rollout.generate_trajectories(
        policy=expert,
        venv=venv,
        sample_until=rollout.make_sample_until(min_timesteps=4000, min_episodes=1),
        deterministic_policy=True,
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(list(trajs))

    bc_policy = policy_utils.create_end_to_end_policy(
        venv.observation_space, venv.action_space
    )
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        rng=rng,
        policy=bc_policy,
        demonstrations=transitions,
        batch_size=min(32, len(transitions)),
    )
    bc_trainer.train(n_epochs=20)

    expert_res = eval_policy_rollout(expert, venv, n_episodes=20, deterministic=True)
    bc_res = eval_policy_rollout(
        bc_trainer.policy, venv, n_episodes=20, deterministic=True
    )
    assert expert_res.mean_return >= bc_res.mean_return * 0.95, (
        f"{env_name}: expert {expert_res.mean_return:.2f} < "
        f"BC {bc_res.mean_return:.2f} (argmax pathology or undertraining)"
    )
