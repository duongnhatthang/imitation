"""Tests for env_utils: new MDPs, wrappers, and env groups."""

import gymnasium as gym
import numpy as np
import pytest

from imitation.experiments.ftrl.env_utils import (
    ENV_CONFIGS,
    ENV_GROUPS,
    FlattenTupleObsWrapper,
    OneHotObsWrapper,
    is_atari,
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


class TestEnvGroups:
    """Test environment group definitions."""

    def test_classical_group_has_8_envs(self):
        assert len(ENV_GROUPS["classical"]) == 8

    def test_atari_zoo_group(self):
        assert "PongNoFrameskip-v4" in ENV_GROUPS["atari-zoo"]
        assert len(ENV_GROUPS["atari-zoo"]) == 8

    def test_atari_all_is_union(self):
        expected = (
            set(ENV_GROUPS["atari-zoo"])
            | set(ENV_GROUPS["atari-fast"])
            | set(ENV_GROUPS["atari-medium"])
        )
        assert set(ENV_GROUPS["atari-all"]) == expected

    def test_all_is_classical_plus_atari(self):
        expected = set(ENV_GROUPS["classical"]) | set(ENV_GROUPS["atari-all"])
        assert set(ENV_GROUPS["all"]) == expected


class TestIsAtari:
    """Test Atari environment detection."""

    def test_atari_envs(self):
        assert is_atari("PongNoFrameskip-v4") is True
        assert is_atari("BreakoutNoFrameskip-v4") is True

    def test_classical_envs(self):
        assert is_atari("CartPole-v1") is False
        assert is_atari("Taxi-v3") is False
