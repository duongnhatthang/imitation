"""Tests for policy_utils: clone-and-reset linear policy creation."""

import copy

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from imitation.experiments.ftrl.policy_utils import (
    create_linear_policy,
    freeze_feature_layers,
    reinitialize_action_net,
)


class TestCreateLinearPolicy:
    """Test that create_linear_policy uses clone-and-reset."""

    def test_mlp_policy_action_net_is_fresh(self):
        """Linear policy should have reinitialized action_net."""
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64]))
        expert_policy = model.policy

        expert_action_weights = {
            name: param.clone()
            for name, param in expert_policy.named_parameters()
            if name.startswith("action_net")
        }

        linear_policy = create_linear_policy(expert_policy)

        for name, param in linear_policy.named_parameters():
            if name.startswith("action_net") and "weight" in name:
                assert not th.equal(param, expert_action_weights[name]), (
                    f"action_net param {name} was not reinitialized"
                )
        env.close()

    def test_mlp_policy_features_frozen(self):
        """All non-action_net params should be frozen."""
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64]))
        linear_policy = create_linear_policy(model.policy)

        for name, param in linear_policy.named_parameters():
            if name.startswith("action_net"):
                assert param.requires_grad is True, f"{name} should be trainable"
            else:
                assert param.requires_grad is False, f"{name} should be frozen"
        env.close()

    def test_mlp_policy_features_match_expert(self):
        """Frozen features should be identical to expert's."""
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64]))
        expert_policy = model.policy

        expert_state = {
            name: param.clone()
            for name, param in expert_policy.named_parameters()
            if not name.startswith("action_net")
        }

        linear_policy = create_linear_policy(expert_policy)

        for name, param in linear_policy.named_parameters():
            if not name.startswith("action_net"):
                assert th.equal(param, expert_state[name]), (
                    f"Feature param {name} doesn't match expert"
                )
        env.close()

    def test_does_not_modify_expert(self):
        """Creating linear policy should not modify the original expert."""
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64]))
        expert_policy = model.policy

        original_state = copy.deepcopy(expert_policy.state_dict())
        create_linear_policy(expert_policy)

        for name, param in expert_policy.named_parameters():
            assert th.equal(param, original_state[name]), (
                f"Expert param {name} was modified"
            )
        env.close()
