"""Atari environment utilities for FTRL experiments.

Provides Atari game configs (3 tiers), vectorized env creation with standard
Atari wrappers, and HuggingFace model zoo downloads.
"""

import logging
import pathlib
import shutil
from typing import Dict

from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

logger = logging.getLogger(__name__)

# Atari game configurations, organized by tier.
# Tier 1: HuggingFace model zoo (no training needed).
# Tier 2: Fast self-trained (~1-2M steps, ~30 min).
# Tier 3: Medium self-trained (~5-10M steps, ~2-5 hours).
ATARI_CONFIGS: Dict[str, dict] = {
    # Tier 1: HuggingFace model zoo
    "PongNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 1_000_000,
        "hub_repo_id": "sb3/ppo-PongNoFrameskip-v4",
        "hub_filename": "ppo-PongNoFrameskip-v4.zip",
    },
    "BreakoutNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 10_000_000,
        "hub_repo_id": "sb3/ppo-BreakoutNoFrameskip-v4",
        "hub_filename": "ppo-BreakoutNoFrameskip-v4.zip",
    },
    "SpaceInvadersNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 10_000_000,
        "hub_repo_id": "sb3/ppo-SpaceInvadersNoFrameskip-v4",
        "hub_filename": "ppo-SpaceInvadersNoFrameskip-v4.zip",
    },
    "BeamRiderNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 10_000_000,
        "hub_repo_id": "sb3/ppo-BeamRiderNoFrameskip-v4",
        "hub_filename": "ppo-BeamRiderNoFrameskip-v4.zip",
    },
    "QbertNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 10_000_000,
        "hub_repo_id": "sb3/ppo-QbertNoFrameskip-v4",
        "hub_filename": "ppo-QbertNoFrameskip-v4.zip",
    },
    "MsPacmanNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 10_000_000,
        "hub_repo_id": "sb3/ppo-MsPacmanNoFrameskip-v4",
        "hub_filename": "ppo-MsPacmanNoFrameskip-v4.zip",
    },
    "EnduroNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 10_000_000,
        "hub_repo_id": "sb3/ppo-EnduroNoFrameskip-v4",
        "hub_filename": "ppo-EnduroNoFrameskip-v4.zip",
    },
    "SeaquestNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 10_000_000,
        "hub_repo_id": "sb3/ppo-SeaquestNoFrameskip-v4",
        "hub_filename": "ppo-SeaquestNoFrameskip-v4.zip",
    },
    # Tier 2: Fast self-trained
    "FreewayNoFrameskip-v4": {
        "tier": 2,
        "ppo_timesteps": 1_000_000,
    },
    "AtlantisNoFrameskip-v4": {
        "tier": 2,
        "ppo_timesteps": 2_000_000,
    },
    "DemonAttackNoFrameskip-v4": {
        "tier": 2,
        "ppo_timesteps": 2_000_000,
    },
    "CrazyClimberNoFrameskip-v4": {
        "tier": 2,
        "ppo_timesteps": 2_000_000,
    },
    # Tier 3: Medium self-trained
    "AsterixNoFrameskip-v4": {
        "tier": 3,
        "ppo_timesteps": 10_000_000,
    },
    "FrostbiteNoFrameskip-v4": {
        "tier": 3,
        "ppo_timesteps": 10_000_000,
    },
    "KangarooNoFrameskip-v4": {
        "tier": 3,
        "ppo_timesteps": 10_000_000,
    },
    "BankHeistNoFrameskip-v4": {
        "tier": 3,
        "ppo_timesteps": 10_000_000,
    },
}

# Short name -> full env ID mapping
_SHORT_NAMES: Dict[str, str] = {}
for _env_id in ATARI_CONFIGS:
    _short = _env_id.replace("NoFrameskip-v4", "")
    _SHORT_NAMES[_short] = _env_id


def get_atari_env_id(name: str) -> str:
    """Convert a short game name to a full Atari env ID.

    Args:
        name: Short name (e.g. "Pong") or full ID ("PongNoFrameskip-v4").

    Returns:
        Full environment ID string.

    Raises:
        ValueError: If the name is not recognized.
    """
    if name in ATARI_CONFIGS:
        return name
    if name in _SHORT_NAMES:
        return _SHORT_NAMES[name]
    raise ValueError(
        f"Unknown Atari game: {name}. "
        f"Known short names: {sorted(_SHORT_NAMES.keys())}"
    )


def make_atari_venv(
    env_name: str,
    n_envs: int,
    seed: int = 0,
) -> VecEnv:
    """Create a vectorized Atari environment with standard wrappers.

    Applies: AtariWrapper (frame skip, grayscale, resize 84x84) ->
    VecFrameStack(4) -> RolloutInfoWrapper.

    Args:
        env_name: Full Atari env ID (e.g. "PongNoFrameskip-v4").
        n_envs: Number of parallel environments.
        seed: Random seed.

    Returns:
        A VecEnv ready for PPO training or FTRL experiments.
    """
    from stable_baselines3.common.env_util import make_atari_env

    venv = make_atari_env(env_name, n_envs=n_envs, seed=seed)
    venv = VecFrameStack(venv, n_stack=4)
    return venv


def download_hub_expert(
    env_name: str,
    cache_dir: pathlib.Path,
) -> pathlib.Path:
    """Download a pre-trained expert from HuggingFace model hub.

    Args:
        env_name: Full Atari env ID (e.g. "PongNoFrameskip-v4").
        cache_dir: Directory to cache downloaded models.

    Returns:
        Path to the downloaded .zip model file.

    Raises:
        ValueError: If the env has no hub_repo_id configured.
    """
    config = ATARI_CONFIGS.get(env_name)
    if config is None or "hub_repo_id" not in config:
        raise ValueError(
            f"No HuggingFace model available for {env_name}. "
            f"Tier 1 games only: "
            f"{[k for k, v in ATARI_CONFIGS.items() if v.get('hub_repo_id')]}"
        )

    cache_path = cache_dir / env_name.replace("/", "_")
    model_file = cache_path / "model.zip"

    if model_file.exists():
        logger.info(f"Using cached hub expert: {model_file}")
        return model_file

    from huggingface_sb3 import load_from_hub

    logger.info(f"Downloading expert from HuggingFace: {config['hub_repo_id']}")
    downloaded_path = load_from_hub(
        repo_id=config["hub_repo_id"],
        filename=config["hub_filename"],
    )

    # Copy to our cache structure
    cache_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded_path, model_file)
    logger.info(f"Cached hub expert to {model_file}")

    return model_file
