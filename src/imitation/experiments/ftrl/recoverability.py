"""Recoverability constant mu(s) = max_a Q(s,a) - min_a Q(s,a).

Q comes from a separately-trained DQN *reference* expert (SB3 DQN exposes
per-action Q via ``q_net``; its greedy policy makes V(s)=max_a Q(s,a)). The PPO
expert used for imitation is unchanged; mu characterizes the environment's
recoverability, and figures must state this provenance.
"""

import pathlib
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

DQN_DEFAULT_TIMESTEPS: Dict[str, int] = {"CartPole-v1": 50_000}

# Per-environment DQN constructor kwargs overrides.  The SB3 DQN default
# learning_starts=50_000 means a 50k-step run never trains; we lower it for
# small environments so that the 50k-step budget actually includes gradient
# updates.  learning_rate=1e-3 is tuned for CartPole on CPU.
_DQN_ENV_KWARGS: Dict[str, Dict[str, Any]] = {
    "CartPole-v1": {"learning_starts": 500, "learning_rate": 1e-3},
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


def get_or_train_dqn_reference(
    env_name: str, cache_dir, total_timesteps: int = None, seed: int = 0
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
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_file)
    return model


def reference_return(dqn: DQN, env_name: str, n_episodes: int = 20) -> float:
    """Mean episodic return of the (greedy) DQN reference."""
    env = gym.make(env_name)
    mean, _ = evaluate_policy(dqn, env, n_eval_episodes=n_episodes, deterministic=True)
    env.close()
    return float(mean)
