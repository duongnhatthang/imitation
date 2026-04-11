"""Tests for FTRL experiment utilities: env_utils, experts, policy_utils."""

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from imitation.experiments.ftrl import env_utils, experts, policy_utils


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


# --- OneHotObsWrapper Tests ---


def test_one_hot_wrapper_frozenlake(rng):
    """OneHotObsWrapper converts FrozenLake discrete obs to one-hot Box."""
    import gymnasium as gym

    env = gym.make("FrozenLake-v1", is_slippery=False)
    wrapped = env_utils.OneHotObsWrapper(env)

    assert isinstance(wrapped.observation_space, spaces.Box)
    assert wrapped.observation_space.shape == (16,)
    assert wrapped.observation_space.dtype == np.float32

    obs, _ = wrapped.reset()
    assert obs.shape == (16,)
    assert obs.sum() == 1.0  # one-hot
    assert obs.dtype == np.float32

    env.close()


def test_one_hot_wrapper_cliffwalking(rng):
    """OneHotObsWrapper works for CliffWalking (48 states)."""
    import gymnasium as gym

    env = gym.make("CliffWalking-v0")
    wrapped = env_utils.OneHotObsWrapper(env)

    assert wrapped.observation_space.shape == (48,)
    obs, _ = wrapped.reset()
    assert obs.shape == (48,)
    assert obs.sum() == 1.0

    env.close()


def test_one_hot_wrapper_rejects_continuous():
    """OneHotObsWrapper raises TypeError for non-Discrete spaces."""
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    with pytest.raises(TypeError, match="Discrete"):
        env_utils.OneHotObsWrapper(env)
    env.close()


# --- make_env Tests ---


def test_make_env_cartpole(rng):
    """make_env for CartPole returns continuous obs without one-hot."""
    venv = env_utils.make_env("CartPole-v1", n_envs=1, rng=rng)
    assert isinstance(venv.observation_space, spaces.Box)
    assert venv.observation_space.shape == (4,)

    obs = venv.reset()
    assert obs.shape == (1, 4)
    venv.close()


def test_make_env_frozenlake_auto_onehot(rng):
    """make_env for FrozenLake auto-applies one-hot wrapper."""
    venv = env_utils.make_env("FrozenLake-v1", n_envs=1, rng=rng)
    assert isinstance(venv.observation_space, spaces.Box)
    assert venv.observation_space.shape == (16,)

    obs = venv.reset()
    assert obs.shape == (1, 16)
    assert obs[0].sum() == 1.0
    venv.close()


def test_make_env_cliffwalking_auto_onehot(rng):
    """make_env for CliffWalking auto-applies one-hot wrapper."""
    venv = env_utils.make_env("CliffWalking-v0", n_envs=1, rng=rng)
    assert venv.observation_space.shape == (48,)
    obs = venv.reset()
    assert obs.shape == (1, 48)
    venv.close()


# --- Expert Pipeline Tests ---


def test_train_and_cache_expert(tmp_path, rng):
    """get_or_train_expert trains PPO and caches the model for discrete env."""
    venv = env_utils.make_env("FrozenLake-v1", n_envs=1, rng=rng)
    policy = experts.get_or_train_expert(
        "FrozenLake-v1",
        venv,
        cache_dir=tmp_path,
        rng=rng,
        seed=0,
    )

    assert isinstance(policy, th.nn.Module)
    # Verify cache exists
    cache_file = tmp_path / "FrozenLake-v1" / "model.zip"
    assert cache_file.exists()

    # Loading again should use cache
    policy2 = experts.get_or_train_expert(
        "FrozenLake-v1",
        venv,
        cache_dir=tmp_path,
        rng=rng,
    )
    assert isinstance(policy2, th.nn.Module)
    venv.close()


def test_make_expert_trajectories(tmp_path, rng):
    """make_expert_trajectories collects the requested number of episodes."""
    venv = env_utils.make_env("CartPole-v1", n_envs=1, rng=rng)

    # Train a quick expert
    model = PPO("MlpPolicy", venv, seed=0, verbose=0)
    model.learn(total_timesteps=5000)

    trajs = experts.make_expert_trajectories(
        expert=model.policy,
        venv=venv,
        n_trajectories=3,
        rng=rng,
    )

    assert len(trajs) >= 3
    for t in trajs:
        assert len(t.obs) == len(t.acts) + 1
        assert len(t.rews) == len(t.acts)
    venv.close()


# --- Policy Utils Tests ---


def test_create_end_to_end_policy():
    """End-to-end policy has correct architecture and all params trainable."""
    obs_space = spaces.Box(-1, 1, (4,))
    act_space = spaces.Discrete(2)
    policy = policy_utils.create_end_to_end_policy(obs_space, act_space)

    assert isinstance(policy, ActorCriticPolicy)
    # All params should be trainable
    for name, p in policy.named_parameters():
        assert p.requires_grad, f"{name} should be trainable"


def test_freeze_feature_layers():
    """freeze_feature_layers freezes everything except action_net."""
    obs_space = spaces.Box(-1, 1, (4,))
    act_space = spaces.Discrete(2)
    policy = policy_utils.create_end_to_end_policy(obs_space, act_space)

    policy_utils.freeze_feature_layers(policy)

    for name, p in policy.named_parameters():
        if name.startswith("action_net"):
            assert p.requires_grad, f"{name} should remain trainable"
        else:
            assert not p.requires_grad, f"{name} should be frozen"


def test_reinitialize_action_net():
    """reinitialize_action_net resets action_net weights."""
    obs_space = spaces.Box(-1, 1, (4,))
    act_space = spaces.Discrete(2)
    policy = policy_utils.create_end_to_end_policy(obs_space, act_space)

    # Record original weights
    original_weight = policy.action_net.weight.data.clone()

    # Modify weights
    policy.action_net.weight.data.fill_(999.0)
    assert (policy.action_net.weight.data == 999.0).all()

    # Reinitialize
    policy_utils.reinitialize_action_net(policy)
    assert not (policy.action_net.weight.data == 999.0).any()


def test_create_linear_policy():
    """Linear policy has frozen features from expert and fresh action_net."""
    obs_space = spaces.Box(-1, 1, (4,))
    act_space = spaces.Discrete(2)

    # Create a "trained" expert
    expert = policy_utils.create_end_to_end_policy(obs_space, act_space)
    # Give it distinctive weights
    with th.no_grad():
        for p in expert.mlp_extractor.parameters():
            p.fill_(0.42)

    linear_policy = policy_utils.create_linear_policy(expert)

    # mlp_extractor weights should match expert
    for (name_e, pe), (name_l, pl) in zip(
        expert.mlp_extractor.named_parameters(),
        linear_policy.mlp_extractor.named_parameters(),
    ):
        assert th.allclose(pe, pl), f"mlp_extractor.{name_e} should match expert"

    # mlp_extractor should be frozen
    for name, p in linear_policy.named_parameters():
        if name.startswith("action_net"):
            assert p.requires_grad
        elif name.startswith("mlp_extractor"):
            assert not p.requires_grad, f"{name} should be frozen"

    # action_net should be reinitialized (not matching expert)
    # Very unlikely to match after Xavier init
    assert not th.allclose(
        expert.action_net.weight.data,
        linear_policy.action_net.weight.data,
    )


def test_train_classical_expert_converges_cartpole(tmp_path):
    """CartPole expert trainer must return a converged policy within the cap."""
    import numpy as np
    from imitation.experiments.ftrl import env_utils
    from imitation.experiments.ftrl.expert_training import (
        train_classical_expert_until_converged,
    )
    from imitation.experiments.ftrl.eval_utils import eval_policy_rollout

    rng = np.random.default_rng(0)
    venv = env_utils.make_env("CartPole-v1", n_envs=1, rng=rng)
    policy = train_classical_expert_until_converged(
        env_name="CartPole-v1",
        cache_dir=tmp_path,
        rng=rng,
        seed=0,
    )
    res = eval_policy_rollout(
        policy, venv, n_episodes=20, deterministic=True, expert_policy=policy
    )
    cfg = env_utils.get_convergence_config("CartPole-v1")
    normalized = (res.mean_return - 22.0) / (500.0 - 22.0)
    assert normalized >= cfg["threshold"] - 0.05
    assert res.current_round_ce <= cfg["self_ce_eps"] + 0.02


def test_train_classical_expert_raises_on_non_convergence(tmp_path):
    """Unreachable threshold must raise RuntimeError."""
    import numpy as np
    import pytest
    from imitation.experiments.ftrl.expert_training import (
        train_classical_expert_until_converged,
    )

    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="failed to converge"):
        train_classical_expert_until_converged(
            env_name="CartPole-v1",
            cache_dir=tmp_path,
            rng=rng,
            seed=0,
            convergence_override={
                "chunk_timesteps": 500,
                "min_timesteps": 500,
                "max_timesteps": 1000,
                "threshold": 0.99,
                "self_ce_eps": 0.001,
                "patience": 1,
            },
        )


def test_get_or_train_expert_uses_new_trainer(tmp_path):
    """get_or_train_expert routes classical envs through the convergence trainer."""
    import numpy as np
    from imitation.experiments.ftrl import env_utils, experts
    from imitation.experiments.ftrl.eval_utils import eval_policy_rollout

    rng = np.random.default_rng(0)
    venv = env_utils.make_env("CartPole-v1", n_envs=1, rng=rng)
    policy = experts.get_or_train_expert(
        "CartPole-v1", venv, cache_dir=tmp_path, rng=rng, seed=0
    )
    # Expert must be converged (argmax policy at ceiling).
    res = eval_policy_rollout(policy, venv, n_episodes=20, deterministic=True)
    normalized = (res.mean_return - 22.0) / (500.0 - 22.0)
    assert normalized >= 0.90
    # Cache file exists so a second call short-circuits.
    assert (tmp_path / "CartPole-v1" / "model.zip").exists()
