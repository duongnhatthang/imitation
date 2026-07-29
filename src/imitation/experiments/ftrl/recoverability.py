"""Recoverability constant mu(s) = max_a Q(s,a) - min_a Q(s,a).

Q comes from a separately-trained DQN *reference* expert (SB3 DQN exposes
per-action Q via ``q_net``; its greedy policy makes V(s)=max_a Q(s,a)). The PPO
expert used for imitation is unchanged; mu characterizes the environment's
recoverability, and figures must state this provenance.
"""

import pathlib
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

DQN_DEFAULT_TIMESTEPS: Dict[str, int] = {
    "CartPole-v1": 50_000,
    "Acrobot-v1": 100_000,
    "MountainCar-v0": 120_000,
    "LunarLander-v2": 100_000,
}

# Per-environment DQN constructor kwargs overrides.  The SB3 DQN default
# learning_starts=50_000 means a 50k-step run never trains, and the default
# hyperparameters leave CartPole far from optimal (return ~280/500), which
# compresses the Q-scale and makes mu(s) meaningless.  These are the SB3
# rl-baselines3-zoo CartPole DQN hyperparameters, which reliably reach the
# 500 return cap -> a near-optimal reference and a well-scaled Q-function.
_DQN_ENV_KWARGS: Dict[str, Dict[str, Any]] = {
    "CartPole-v1": {
        "learning_rate": 2.3e-3,
        "batch_size": 64,
        "buffer_size": 100_000,
        "learning_starts": 1_000,
        "gamma": 0.99,
        "target_update_interval": 10,
        "train_freq": 256,
        "gradient_steps": 128,
        "exploration_fraction": 0.16,
        "exploration_final_eps": 0.04,
        "policy_kwargs": {"net_arch": [256, 256]},
    },
    "Acrobot-v1": {
        "learning_rate": 6.3e-4,
        "batch_size": 128,
        "buffer_size": 50_000,
        "learning_starts": 0,
        "gamma": 0.99,
        "target_update_interval": 250,
        "train_freq": 4,
        "gradient_steps": -1,
        "exploration_fraction": 0.12,
        "exploration_final_eps": 0.1,
        "policy_kwargs": {"net_arch": [256, 256]},
    },
    "MountainCar-v0": {
        "learning_rate": 4e-3,
        "batch_size": 128,
        "buffer_size": 10_000,
        "learning_starts": 1_000,
        "gamma": 0.98,
        "target_update_interval": 600,
        "train_freq": 16,
        "gradient_steps": 8,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.07,
        "policy_kwargs": {"net_arch": [256, 256]},
    },
    "LunarLander-v2": {
        "learning_rate": 6.3e-4,
        "batch_size": 128,
        "buffer_size": 50_000,
        "learning_starts": 0,
        "gamma": 0.99,
        "target_update_interval": 250,
        "train_freq": 4,
        "gradient_steps": -1,
        "exploration_fraction": 0.12,
        "exploration_final_eps": 0.1,
        "policy_kwargs": {"net_arch": [256, 256]},
    },
}


def mu_from_q(q_values: np.ndarray) -> np.ndarray:
    """Recoverability per state from a [N, A] array of action-values."""
    return q_values.max(axis=1) - q_values.min(axis=1)


def dqn_q_values(dqn: DQN, obs: np.ndarray) -> np.ndarray:
    """Per-action Q values [N, A] for a batch of observations."""
    obs_t = th.as_tensor(np.asarray(obs), dtype=th.float32, device=dqn.device)
    with th.no_grad():
        q = dqn.q_net(obs_t)
    return q.cpu().numpy()


def recoverability(dqn: DQN, obs: np.ndarray) -> np.ndarray:
    """mu(s) for each observation in a [N, ...] batch."""
    return mu_from_q(dqn_q_values(dqn, obs))


def hub_dqn_q_values(dqn: DQN, obs: np.ndarray) -> np.ndarray:
    """Per-action Q for Atari obs via SB3 preprocessing (CHW transpose + /255)."""
    from imitation.experiments.ftrl import coverage_features

    chw = coverage_features.to_chw(np.asarray(obs))
    obs_t, _ = dqn.policy.obs_to_tensor(chw)
    with th.no_grad():
        q = dqn.q_net(obs_t)
    return q.cpu().numpy()


def get_or_train_dqn_reference(
    env_name: str, cache_dir, total_timesteps: Optional[int] = None, seed: int = 0
) -> DQN:
    """Load a cached DQN reference expert or train and cache one."""
    cache_dir = pathlib.Path(cache_dir)
    model_file = cache_dir / env_name.replace("/", "_") / "dqn_reference.zip"
    if model_file.exists():
        return DQN.load(model_file, device="auto")
    if total_timesteps is None:
        total_timesteps = DQN_DEFAULT_TIMESTEPS.get(env_name, 100_000)
    env_kwargs = _DQN_ENV_KWARGS.get(env_name, {})
    model = DQN("MlpPolicy", env_name, seed=seed, verbose=0, **env_kwargs)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    # DQN (notably on CartPole) can degrade late in training, so the final policy
    # is often far from optimal even when the Q-scale looks right.  Evaluate
    # periodically and keep the best-scoring checkpoint as the reference.
    eval_env = Monitor(gym.make(env_name))
    best_dir = model_file.parent / "_best"
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        eval_freq=2_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=eval_cb)
    eval_env.close()
    best_file = best_dir / "best_model.zip"
    if best_file.exists():
        model = DQN.load(best_file, device="auto")
    model.save(model_file)
    return model


def reference_returns(
    dqn: DQN, env_name: str, n_episodes: int = 20, seed: int = 0
) -> np.ndarray:
    """Per-episode greedy returns over deterministically-seeded episodes.

    Episode ``i`` resets with ``seed + i`` so the result is reproducible across
    calls (the previous unseeded evaluation fluctuated run to run).

    Args:
        dqn: The DQN reference expert.
        env_name: Gymnasium environment id.
        n_episodes: Number of evaluation episodes.
        seed: Base seed; episode i uses ``seed + i``.

    Returns:
        Array of shape ``[n_episodes]`` of episodic returns.
    """
    env = gym.make(env_name)
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        total = 0.0
        while not done:
            action, _ = dqn.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    env.close()
    return np.asarray(returns, dtype=float)


def reference_return(
    dqn: DQN, env_name: str, n_episodes: int = 20, seed: int = 0
) -> float:
    """Mean greedy episodic return (deterministic given ``seed``)."""
    return float(reference_returns(dqn, env_name, n_episodes, seed).mean())


def load_hub_dqn(env_name: str) -> DQN:
    """Load a pretrained sb3/dqn-<Game> Atari DQN from the HuggingFace hub.

    Args:
        env_name: Gymnasium environment id (e.g. ``"PongNoFrameskip-v4"``).

    Returns:
        A DQN model loaded from the HuggingFace hub.
    """
    from huggingface_sb3 import load_from_hub

    path = load_from_hub(repo_id=f"sb3/dqn-{env_name}", filename=f"dqn-{env_name}.zip")
    return DQN.load(path, device="auto")


def normalized_return(dqn_return, random_return, expert_return) -> float:
    """Return normalized to [random=0, expert=1]; robust to negative scales.

    Args:
        dqn_return: The DQN reference agent's mean episodic return.
        random_return: The random policy's mean episodic return.
        expert_return: The expert policy's mean episodic return.

    Returns:
        Normalized return where 0 corresponds to random and 1 to expert.
        Robust to environments with negative returns (e.g. Acrobot).
    """
    denom = expert_return - random_return
    if abs(denom) < 1e-8:
        return 0.0
    return float((dqn_return - random_return) / denom)


def recoverability_mu(
    env_name: str, obs, cache_dir, expert_policy=None
) -> Optional[np.ndarray]:
    """Dispatch mu(s) over visited obs to the right backend for this env family.

    Dispatch logic:
    - ``Blackjack-v1``: returns ``None`` (stochastic optimum; mu not reliable).
    - Atari (``NoFrameskip`` envs): loads a pretrained hub DQN and computes mu.
    - Toy-text (``obs_type == "discrete"``): uses exact tabular policy evaluation
      w.r.t. the PPO expert (requires ``expert_policy``).
    - Continuous classical: trains or loads a cached DQN reference and computes mu.

    Args:
        env_name: Gymnasium environment id.
        obs: Batch of observations, shape ``[N, ...]``.
        cache_dir: Directory for caching trained DQN models.
        expert_policy: Expert policy (required for toy-text / discrete obs envs).

    Returns:
        Array of shape ``[N]`` with mu(s) per observation, or ``None`` for
        Blackjack.
    """
    from imitation.experiments.ftrl import env_utils, recoverability_tabular

    obs = np.asarray(obs)
    if env_name == "Blackjack-v1":
        return None
    if env_utils.is_atari(env_name):
        dqn = load_hub_dqn(env_name)
        return mu_from_q(hub_dqn_q_values(dqn, obs))
    obs_type = env_utils.ENV_CONFIGS.get(env_name, {}).get("obs_type")
    if obs_type == "discrete":  # toy-text: exact env.P mu w.r.t. the PPO expert
        mu_by_state = recoverability_tabular.exact_mu_per_state(env_name, expert_policy)
        ids = recoverability_tabular.state_ids_from_onehot(obs)
        return mu_by_state[ids]
    dqn = get_or_train_dqn_reference(env_name, cache_dir)  # continuous classical
    return recoverability(dqn, obs)
