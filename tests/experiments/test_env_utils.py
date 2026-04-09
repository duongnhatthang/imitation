"""Tests for env_utils: new MDPs, wrappers, and env groups."""

import gymnasium as gym
import numpy as np
import pytest

from imitation.experiments.ftrl.env_utils import (
    ENV_CONFIGS,
    FlattenTupleObsWrapper,
    OneHotObsWrapper,
    make_env,
)


class TestNewClassicalMDPs:
    """Test that the 3 new classical MDPs are configured and create properly."""

    def test_taxi_in_configs(self):
        assert "Taxi-v3" in ENV_CONFIGS
        assert ENV_CONFIGS["Taxi-v3"]["obs_type"] == "discrete"
        assert ENV_CONFIGS["Taxi-v3"]["obs_size"] == 500

    def test_blackjack_in_configs(self):
        assert "Blackjack-v1" in ENV_CONFIGS
        assert ENV_CONFIGS["Blackjack-v1"]["obs_type"] == "tuple"

    def test_lunarlander_in_configs(self):
        assert "LunarLander-v2" in ENV_CONFIGS
        assert ENV_CONFIGS["LunarLander-v2"]["obs_type"] == "continuous"

    def test_make_env_taxi(self):
        rng = np.random.default_rng(0)
        venv = make_env("Taxi-v3", n_envs=1, rng=rng)
        obs = venv.reset()
        assert obs.shape == (1, 500)  # one-hot encoded
        venv.close()

    def test_make_env_blackjack(self):
        rng = np.random.default_rng(0)
        venv = make_env("Blackjack-v1", n_envs=1, rng=rng)
        obs = venv.reset()
        assert obs.shape == (1, 45)  # flattened one-hot: 32 + 11 + 2
        venv.close()

    def test_make_env_lunarlander(self):
        rng = np.random.default_rng(0)
        venv = make_env("LunarLander-v2", n_envs=1, rng=rng)
        obs = venv.reset()
        assert obs.shape == (1, 8)
        venv.close()


class TestFlattenTupleObsWrapper:
    """Test the FlattenTupleObsWrapper for Blackjack-v1."""

    def test_wraps_tuple_obs(self):
        env = gym.make("Blackjack-v1")
        wrapped = FlattenTupleObsWrapper(env)
        assert wrapped.observation_space.shape == (45,)
        obs, _ = wrapped.reset()
        assert obs.shape == (45,)
        assert obs.dtype == np.float32
        # Exactly one 1.0 in each one-hot segment
        assert np.sum(obs[:32]) == 1.0
        assert np.sum(obs[32:43]) == 1.0
        assert np.sum(obs[43:45]) == 1.0
        wrapped.close()

    def test_rejects_non_tuple(self):
        env = gym.make("CartPole-v1")
        with pytest.raises(TypeError, match="Tuple"):
            FlattenTupleObsWrapper(env)
        env.close()
