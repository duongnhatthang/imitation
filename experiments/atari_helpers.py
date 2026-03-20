"""Shared Atari helper utilities for the empirical study.

Provides environment construction, expert loading, and score normalization
for the 7-game Atari benchmark suite.
"""

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack, VecTransposeImage

from imitation.policies.serialize import load_policy

# The 7 Atari games used in the empirical study.
# Keys: display names. Values: raw ALE env IDs (used by SB3 make_atari_env
# and as env_name for HuggingFace expert loading).
ATARI_GAMES = {
    "Pong": "PongNoFrameskip-v4",
    "Breakout": "BreakoutNoFrameskip-v4",
    "BeamRider": "BeamRiderNoFrameskip-v4",
    "Enduro": "EnduroNoFrameskip-v4",
    "Qbert": "QbertNoFrameskip-v4",
    "Seaquest": "SeaquestNoFrameskip-v4",
    "SpaceInvaders": "SpaceInvadersNoFrameskip-v4",
}


def make_atari_training_venv(game_id: str, n_envs: int = 8, seed: int = 0) -> VecEnv:
    """Create a vectorised Atari training environment.

    Applies SB3's AtariWrapper (grayscale, 84x84 resize, reward clipping,
    noop_max=30, frame_skip=4) then stacks 4 frames via VecFrameStack and
    transposes to channels-first via VecTransposeImage to match the SB3
    expert's observation space.
    Resulting observation space: Box(0, 255, (4, 84, 84), uint8).

    Args:
        game_id: Raw ALE env ID, e.g. "PongNoFrameskip-v4".
        n_envs: Number of parallel environments.
        seed: Random seed for environment initialisation.

    Returns:
        A VecEnv with obs space Box(0, 255, (4, 84, 84), uint8).
    """
    venv = make_atari_env(game_id, n_envs=n_envs, seed=seed)
    venv = VecFrameStack(venv, n_stack=4)
    return VecTransposeImage(venv)


def make_atari_eval_venv(game_id: str, n_envs: int = 8, seed: int = 0) -> VecEnv:
    """Create a vectorised Atari evaluation environment (unclipped rewards).

    Same as make_atari_training_venv but passes clip_reward=False so that
    episode returns reflect true game scores rather than clipped {-1, 0, +1}.
    Includes VecTransposeImage to produce channels-first obs matching the
    SB3 expert. Use this venv for score normalisation.

    Args:
        game_id: Raw ALE env ID, e.g. "PongNoFrameskip-v4".
        n_envs: Number of parallel environments.
        seed: Random seed for environment initialisation.

    Returns:
        A VecEnv with obs space Box(0, 255, (4, 84, 84), uint8) and
        unclipped rewards.
    """
    venv = make_atari_env(
        game_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_kwargs={"clip_reward": False},
    )
    venv = VecFrameStack(venv, n_stack=4)
    return VecTransposeImage(venv)


def load_atari_expert(venv: VecEnv, game_id: str):
    """Load a pre-trained PPO expert from the HuggingFace sb3 organisation.

    Uses imitation's load_policy with organization="sb3" (NOT "HumanCompatibleAI",
    which is the default in expert.py but incorrect for Atari). The HuggingFace
    repo ID formed is sb3/ppo-{game_id}.

    Args:
        venv: The environment the expert will run in. Must have obs space
            Box(0, 255, (4, 84, 84), uint8) to match the SB3 expert.
        game_id: Raw ALE env ID, e.g. "PongNoFrameskip-v4". Must match the
            sb3 HuggingFace repo naming convention.

    Returns:
        A stable_baselines3 BasePolicy for the given game.
    """
    return load_policy(
        "ppo-huggingface",
        venv=venv,
        env_name=game_id,
        organization="sb3",
    )


def compute_normalized_score(
    agent_score: float,
    random_score: float,
    expert_score: float,
) -> float:
    """Compute a normalised performance score.

    Normalises the agent's episode return relative to random and expert
    baselines: 0.0 = random-level, 1.0 = expert-level.

    Args:
        agent_score: Mean episode return of the agent being evaluated.
        random_score: Mean episode return of a random policy (cached baseline).
        expert_score: Mean episode return of the expert policy.

    Returns:
        Normalised score in [0, 1] (can exceed range for super-human agents).
        Returns 0.0 if expert_score == random_score (division-by-zero guard).
    """
    if expert_score == random_score:
        return 0.0
    return (agent_score - random_score) / (expert_score - random_score)
