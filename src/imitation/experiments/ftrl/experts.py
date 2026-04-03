"""Expert training and trajectory collection for FTRL experiments.

Provides get_or_train_expert (tries HuggingFace first, trains PPO as fallback)
and make_expert_trajectories for collecting demonstration data.
"""

import logging
import pathlib
from typing import List, Optional, Sequence

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

from imitation.data import rollout, types
from imitation.policies import serialize

from . import env_utils

logger = logging.getLogger(__name__)


# Map from our env names to HuggingFace model IDs (sb3 org).
# Only environments with known good HF models are listed.
_HF_ENV_MAP = {
    "CartPole-v1": "CartPole-v1",
    "Acrobot-v1": "Acrobot-v1",
    "MountainCar-v0": "MountainCar-v0",
}


def get_or_train_expert(
    env_name: str,
    venv: VecEnv,
    cache_dir: pathlib.Path,
    rng: np.random.Generator,
    seed: int = 0,
) -> BasePolicy:
    """Load an expert policy from HuggingFace or train one with PPO.

    Tries HuggingFace sb3 models first. If that fails (no model available or
    env has discrete obs with one-hot wrapper), falls back to training PPO
    locally and caching the result.

    Args:
        env_name: Gymnasium environment ID.
        venv: Vectorized environment (may have OneHotObsWrapper applied).
        cache_dir: Directory for caching trained expert models.
        rng: Random state.
        seed: Random seed for PPO training.

    Returns:
        A trained expert policy compatible with venv.
    """
    cache_path = pathlib.Path(cache_dir) / env_name.replace("/", "_")
    model_file = cache_path / "model.zip"

    # Try loading cached model first
    if model_file.exists():
        logger.info(f"Loading cached expert from {model_file}")
        model = PPO.load(model_file, env=venv)
        return model.policy

    # Try HuggingFace for envs that don't need one-hot wrapper
    config = env_utils.ENV_CONFIGS.get(env_name, {})
    is_discrete_obs = config.get("obs_type") == "discrete"

    if not is_discrete_obs and env_name in _HF_ENV_MAP:
        try:
            hf_env_name = _HF_ENV_MAP[env_name]
            logger.info(f"Trying HuggingFace model for {hf_env_name}")
            policy = serialize.load_policy(
                "ppo-huggingface",
                venv,
                env_name=hf_env_name,
                organization="sb3",
            )
            return policy
        except Exception as e:
            logger.warning(f"HuggingFace load failed for {env_name}: {e}")

    # Fallback: train PPO
    logger.info(
        f"Training PPO expert for {env_name} "
        f"({config.get('ppo_timesteps', 100_000)} timesteps)",
    )
    ppo_timesteps = config.get("ppo_timesteps", 100_000)
    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=ppo_timesteps)

    # Cache the trained model
    cache_path.mkdir(parents=True, exist_ok=True)
    model.save(model_file)
    logger.info(f"Saved expert to {model_file}")

    return model.policy


def make_expert_trajectories(
    expert: BasePolicy,
    venv: VecEnv,
    n_trajectories: int,
    rng: np.random.Generator,
) -> List[types.TrajectoryWithRew]:
    """Collect expert demonstration trajectories.

    Args:
        expert: The expert policy to roll out.
        venv: Vectorized environment.
        n_trajectories: Number of complete episodes to collect.
        rng: Random state.

    Returns:
        List of expert trajectories with rewards.
    """
    sample_until = rollout.make_sample_until(
        min_episodes=n_trajectories,
        min_timesteps=None,
    )
    trajectories = rollout.generate_trajectories(
        policy=expert,
        venv=venv,
        sample_until=sample_until,
        deterministic_policy=True,
        rng=rng,
    )
    return list(trajectories)
