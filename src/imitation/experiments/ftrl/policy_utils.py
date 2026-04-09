"""Policy creation utilities for FTRL experiments.

Provides end-to-end and linear policy modes for comparing FTL vs FTRL.
Linear mode freezes the feature extractor + MLP hidden layers from an
expert and only trains the final action_net layer.
"""

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy


def create_end_to_end_policy(
    obs_space: spaces.Space,
    act_space: spaces.Space,
) -> ActorCriticPolicy:
    """Create a fresh [64,64] MLP policy for end-to-end training.

    Args:
        obs_space: Observation space.
        act_space: Action space.

    Returns:
        An untrained ActorCriticPolicy with [64,64] architecture.
    """
    return ActorCriticPolicy(
        obs_space,
        act_space,
        lr_schedule=lambda _: 1e-3,
        net_arch=[64, 64],
    )


def freeze_feature_layers(policy: ActorCriticPolicy) -> None:
    """Freeze all parameters except action_net.

    Sets requires_grad=False on all parameters whose name does not
    start with "action_net".

    Args:
        policy: The policy to partially freeze.
    """
    for name, param in policy.named_parameters():
        if not name.startswith("action_net"):
            param.requires_grad = False


def reinitialize_action_net(policy: ActorCriticPolicy) -> None:
    """Reinitialize action_net parameters with Xavier uniform / zeros.

    Args:
        policy: The policy whose action_net to reinitialize.
    """
    for name, param in policy.named_parameters():
        if name.startswith("action_net"):
            if param.dim() >= 2:
                th.nn.init.xavier_uniform_(param)
            elif param.dim() == 1:
                th.nn.init.zeros_(param)


def create_linear_policy(
    expert_policy: ActorCriticPolicy,
) -> ActorCriticPolicy:
    """Create a linear-mode policy by cloning expert and resetting action_net.

    Deep-copies the expert's entire policy, freezes all parameters except
    action_net, and reinitializes action_net. Works with any architecture
    (MLP, CNN, etc.) without needing to know internals.

    Args:
        expert_policy: Trained expert policy to clone features from.

    Returns:
        A policy with frozen hidden layers and fresh action_net.
    """
    import copy

    policy = copy.deepcopy(expert_policy)
    freeze_feature_layers(policy)
    reinitialize_action_net(policy)
    return policy
