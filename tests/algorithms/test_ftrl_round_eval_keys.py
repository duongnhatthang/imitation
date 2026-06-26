"""_compute_round_eval emits additive CE-breakdown keys (E2 wiring)."""

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.experiments.ftrl import policy_utils, run_experiment


def _tiny_expert():
    from stable_baselines3 import PPO

    venv = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    model = PPO("MlpPolicy", venv, n_steps=64, batch_size=64, n_epochs=1, seed=0)
    return model, venv


def test_round_eval_has_breakdown_keys():
    model, venv = _tiny_expert()
    expert_policy = model.policy
    learner = policy_utils.create_linear_policy(expert_policy)
    baselines = {"expert_return": 200.0, "random_return": 10.0}

    out = run_experiment._compute_round_eval(learner, expert_policy, venv, baselines)

    # Existing keys preserved.
    for k in [
        "rollout_cross_entropy",
        "expert_rollout_cross_entropy",
        "normalized_return",
        "disagreement_rate",
        "d_eval_size",
    ]:
        assert k in out
    # New additive keys present.
    for k in [
        "rollout_ce_correct",
        "rollout_ce_wrong",
        "n_correct",
        "n_wrong",
        "conf_correct",
        "conf_wrong",
    ]:
        assert k in out
    assert out["n_correct"] + out["n_wrong"] == out["d_eval_size"]


def test_nan_to_none_maps_nan_and_rounds():
    from imitation.experiments.ftrl.run_experiment import _nan_to_none

    assert _nan_to_none(float("nan")) is None
    assert _nan_to_none(0.1234567) == 0.123457
