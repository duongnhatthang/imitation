"""Expert training and trajectory collection for FTRL experiments.

Supports three expert sourcing strategies:
- Classical MDPs: Train PPO with MlpPolicy [64,64] and cache.
- Atari Tier 1: Download pre-trained PPO from HuggingFace model zoo.
- Atari Tier 2/3: Train PPO with CnnPolicy and cache.
"""

import logging
import pathlib
from typing import List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

from imitation.data import rollout, types

from . import env_utils

logger = logging.getLogger(__name__)


def get_or_train_expert(
    env_name: str,
    venv: VecEnv,
    cache_dir: pathlib.Path,
    rng: np.random.Generator,
    seed: int = 0,
) -> BasePolicy:
    """Load a cached expert, download from HuggingFace, or train with PPO.

    Routing logic:
    - If cached model exists on disk: load it.
    - If Atari tier 1 (hub_repo_id available): download from HuggingFace.
    - If Atari tier 2/3: train PPO with CnnPolicy.
    - If classical MDP: train PPO with MlpPolicy [64,64].

    Args:
        env_name: Gymnasium environment ID.
        venv: Vectorized environment.
        cache_dir: Directory for caching trained/downloaded expert models.
        rng: Random state.
        seed: Random seed for PPO training.

    Returns:
        A trained expert policy compatible with venv.
    """
    cache_path = pathlib.Path(cache_dir) / env_name.replace("/", "_")
    model_file = cache_path / "model.zip"

    # Load cached model if available
    if model_file.exists():
        logger.info(f"Loading cached expert from {model_file}")
        model = PPO.load(model_file, env=venv)
        return model.policy

    if env_utils.is_atari(env_name):
        return _get_atari_expert(env_name, venv, cache_dir, seed)
    else:
        return _train_classical_expert(env_name, venv, cache_dir, rng, seed)


def _get_atari_expert(
    env_name: str,
    venv: VecEnv,
    cache_dir: pathlib.Path,
    seed: int,
) -> BasePolicy:
    """Get an Atari expert: download from hub or train."""
    from . import atari_utils

    config = atari_utils.ATARI_CONFIGS.get(env_name, {})

    if "hub_repo_id" in config:
        # Tier 1: download from HuggingFace
        model_path = atari_utils.download_hub_expert(env_name, cache_dir)
        model = PPO.load(model_path, env=venv)
    else:
        # Tier 2/3: train PPO with CnnPolicy
        ppo_timesteps = config.get("ppo_timesteps", 10_000_000)
        logger.info(
            f"Training PPO CnnPolicy expert for {env_name} "
            f"({ppo_timesteps} timesteps)"
        )
        train_venv = atari_utils.make_atari_venv(env_name, n_envs=8, seed=seed)
        model = PPO(
            "CnnPolicy",
            train_venv,
            n_steps=128,
            n_epochs=4,
            batch_size=256,
            learning_rate=2.5e-4,
            clip_range=0.1,
            vf_coef=0.5,
            ent_coef=0.01,
            seed=seed,
            verbose=0,
        )
        model.learn(total_timesteps=ppo_timesteps)
        train_venv.close()

        # Cache
        cache_path = cache_dir / env_name.replace("/", "_")
        cache_path.mkdir(parents=True, exist_ok=True)
        model.save(cache_path / "model.zip")
        logger.info(f"Saved Atari expert to {cache_path / 'model.zip'}")

    # Evaluate
    from stable_baselines3.common.evaluation import evaluate_policy

    mean_reward, std_reward = evaluate_policy(
        model.policy, venv, n_eval_episodes=10, deterministic=True,
    )
    logger.info(
        f"Expert quality for {env_name}: reward={mean_reward:.1f}±{std_reward:.1f}"
    )

    from .env_baselines import load_or_compute_baselines, validate_expert_quality

    rng = np.random.default_rng(seed)

    # Compute and cache baselines
    load_or_compute_baselines(env_name, venv, model.policy, cache_dir, rng)

    # Warn if expert quality is below reference
    is_ok, msg = validate_expert_quality(env_name, mean_reward)
    if not is_ok:
        logger.warning(f"WARNING: {msg}")

    return model.policy


def _train_classical_expert(
    env_name: str,
    venv: VecEnv,
    cache_dir: pathlib.Path,
    rng: np.random.Generator,
    seed: int,
) -> BasePolicy:
    """Train a classical MDP expert with PPO MlpPolicy [64,64]."""
    config = env_utils.ENV_CONFIGS.get(env_name, {})
    ppo_timesteps = config.get("ppo_timesteps", 100_000)
    ppo_kwargs = config.get("ppo_kwargs", {})
    ppo_n_envs = config.get("ppo_n_envs", None)
    logger.info(
        f"Training PPO expert for {env_name} ({ppo_timesteps} timesteps)",
    )

    if ppo_n_envs and ppo_n_envs > 1:
        train_venv = env_utils.make_env(env_name, n_envs=ppo_n_envs, rng=rng)
    else:
        train_venv = venv

    model = PPO(
        "MlpPolicy",
        train_venv,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=0,
        **ppo_kwargs,
    )
    model.learn(total_timesteps=ppo_timesteps)

    if ppo_n_envs and ppo_n_envs > 1:
        train_venv.close()

    from stable_baselines3.common.evaluation import evaluate_policy

    mean_reward, std_reward = evaluate_policy(
        model.policy, venv, n_eval_episodes=20, deterministic=True,
    )
    logger.info(
        f"Expert quality for {env_name}: "
        f"reward={mean_reward:.1f}±{std_reward:.1f} "
        f"(trained {ppo_timesteps} steps)",
    )

    cache_path = cache_dir / env_name.replace("/", "_")
    cache_path.mkdir(parents=True, exist_ok=True)
    model.save(cache_path / "model.zip")
    logger.info(f"Saved expert to {cache_path / 'model.zip'}")

    from .env_baselines import load_or_compute_baselines, validate_expert_quality

    # Compute and cache baselines
    load_or_compute_baselines(env_name, venv, model.policy, cache_dir, rng)

    # Warn if expert quality is below reference
    is_ok, msg = validate_expert_quality(env_name, mean_reward)
    if not is_ok:
        logger.warning(f"WARNING: {msg}")

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
