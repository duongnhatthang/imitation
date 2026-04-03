"""Expert training and trajectory collection for FTRL experiments.

Always trains PPO experts with a consistent [64,64] architecture and caches
them to disk. This ensures the expert's network architecture matches what
create_linear_policy expects (same mlp_extractor structure).
"""

import logging
import pathlib
from typing import List, Optional, Sequence

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
    """Load a cached expert or train one with PPO.

    Always uses PPO with net_arch=[64,64] to ensure the expert architecture
    matches what create_linear_policy expects. Trained models are cached to
    disk for reuse across seeds and runs.

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

    # Load cached model if available
    if model_file.exists():
        logger.info(f"Loading cached expert from {model_file}")
        model = PPO.load(model_file, env=venv)
        return model.policy

    # Train PPO expert
    config = env_utils.ENV_CONFIGS.get(env_name, {})
    ppo_timesteps = config.get("ppo_timesteps", 100_000)
    logger.info(
        f"Training PPO expert for {env_name} ({ppo_timesteps} timesteps)",
    )
    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=ppo_timesteps)

    # Evaluate expert quality
    from stable_baselines3.common.evaluation import evaluate_policy

    mean_reward, std_reward = evaluate_policy(
        model, venv, n_eval_episodes=20, deterministic=True,
    )
    logger.info(
        f"Expert quality for {env_name}: "
        f"reward={mean_reward:.1f}±{std_reward:.1f} "
        f"(trained {ppo_timesteps} steps)",
    )

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
