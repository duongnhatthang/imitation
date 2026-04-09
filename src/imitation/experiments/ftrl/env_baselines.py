"""Reference baselines and expert quality validation for FTRL experiments.

Reference baselines serve two purposes:

1. **Expert quality validation**: detect degenerate experts early by checking
   that measured expert return exceeds a threshold between random and known-good
   expert performance.

2. **Normalized return computation**: map raw returns to a 0-1 scale where
   0 = random policy and 1 = expert policy, enabling cross-environment
   comparison.

Adding a new MDP requires adding its reference baseline here -- the tests in
``tests/experiments/test_env_baselines.py`` enforce that every environment in
``ENV_CONFIGS`` and ``ATARI_CONFIGS`` has a corresponding entry.

Random scores come from uniform-random rollouts (100 episodes).
Expert scores come from well-trained PPO agents or published benchmarks.
"""

import json
import logging
import pathlib
from typing import Dict, Tuple

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference baselines: {env_name: {"random_score": float, "expert_score": float}}
# ---------------------------------------------------------------------------
REFERENCE_BASELINES: Dict[str, Dict[str, float]] = {
    # ---- Classical MDPs ----
    "CartPole-v1": {"random_score": 22.0, "expert_score": 500.0},
    "FrozenLake-v1": {"random_score": 0.015, "expert_score": 1.0},
    "CliffWalking-v0": {"random_score": -56957.0, "expert_score": -13.0},
    "Acrobot-v1": {"random_score": -499.0, "expert_score": -85.0},
    "MountainCar-v0": {"random_score": -200.0, "expert_score": -110.0},
    "Taxi-v3": {"random_score": -763.0, "expert_score": 7.9},
    "Blackjack-v1": {"random_score": -0.40, "expert_score": -0.06},
    "LunarLander-v2": {"random_score": -176.0, "expert_score": 250.0},
    # ---- Atari Tier 1 ----
    "PongNoFrameskip-v4": {"random_score": -20.7, "expert_score": 20.5},
    "BreakoutNoFrameskip-v4": {"random_score": 1.7, "expert_score": 405.7},
    "SpaceInvadersNoFrameskip-v4": {
        "random_score": 148.0,
        "expert_score": 1019.8,
    },
    "BeamRiderNoFrameskip-v4": {"random_score": 363.9, "expert_score": 2835.7},
    "QbertNoFrameskip-v4": {"random_score": 163.9, "expert_score": 15228.3},
    "MsPacmanNoFrameskip-v4": {"random_score": 307.3, "expert_score": 2152.8},
    "EnduroNoFrameskip-v4": {"random_score": 0.0, "expert_score": 986.7},
    "SeaquestNoFrameskip-v4": {"random_score": 68.4, "expert_score": 1518.3},
    # ---- Atari Tier 2 ----
    "FreewayNoFrameskip-v4": {"random_score": 0.0, "expert_score": 33.0},
    "AtlantisNoFrameskip-v4": {
        "random_score": 12850.0,
        "expert_score": 2036749.0,
    },
    "DemonAttackNoFrameskip-v4": {
        "random_score": 152.1,
        "expert_score": 13788.4,
    },
    "CrazyClimberNoFrameskip-v4": {
        "random_score": 10780.5,
        "expert_score": 119344.7,
    },
    # ---- Atari Tier 3 ----
    "AsterixNoFrameskip-v4": {"random_score": 210.0, "expert_score": 3738.5},
    "FrostbiteNoFrameskip-v4": {"random_score": 65.2, "expert_score": 933.6},
    "KangarooNoFrameskip-v4": {"random_score": 52.0, "expert_score": 5325.3},
    "BankHeistNoFrameskip-v4": {"random_score": 14.2, "expert_score": 1213.5},
}

# Minimum fraction of (expert - random) that a measured expert must achieve
# to be considered acceptable.
EXPERT_QUALITY_THRESHOLD: float = 0.80


def validate_expert_quality(
    env_name: str,
    measured_return: float,
) -> Tuple[bool, str]:
    """Check whether a measured expert return meets quality expectations.

    The expert is considered acceptable if::

        measured_return >= random_score + threshold * (expert_score - random_score)

    For unknown environments (not in ``REFERENCE_BASELINES``), validation
    is skipped and the expert is assumed OK.

    Args:
        env_name: Environment ID.
        measured_return: Mean return measured from the expert policy.

    Returns:
        A ``(is_ok, message)`` tuple. ``is_ok`` is True if the expert passes
        or the environment is unknown; ``message`` explains the outcome.
    """
    if env_name not in REFERENCE_BASELINES:
        return True, f"No reference baseline for {env_name}; skipping validation."

    bl = REFERENCE_BASELINES[env_name]
    random_score = bl["random_score"]
    expert_score = bl["expert_score"]
    threshold_value = random_score + EXPERT_QUALITY_THRESHOLD * (
        expert_score - random_score
    )

    if measured_return >= threshold_value:
        return True, (
            f"{env_name}: expert return {measured_return:.1f} >= "
            f"threshold {threshold_value:.1f} "
            f"({EXPERT_QUALITY_THRESHOLD:.0%} of [{random_score}, {expert_score}])."
        )
    else:
        return False, (
            f"{env_name}: expert return {measured_return:.1f} is below "
            f"threshold {threshold_value:.1f} "
            f"({EXPERT_QUALITY_THRESHOLD:.0%} of [{random_score}, {expert_score}]). "
            f"The expert may be degenerate."
        )


def compute_random_return(venv: VecEnv, n_episodes: int = 100) -> float:
    """Estimate the mean return of a uniform-random policy.

    Args:
        venv: Vectorized environment (reset before calling).
        n_episodes: Number of episodes to average over.

    Returns:
        Mean episodic return across ``n_episodes`` episodes.
    """
    obs = venv.reset()
    episode_returns: list = []
    running_returns = np.zeros(venv.num_envs)

    while len(episode_returns) < n_episodes:
        actions = np.array([venv.action_space.sample() for _ in range(venv.num_envs)])
        obs, rewards, dones, infos = venv.step(actions)
        running_returns += rewards
        for i, done in enumerate(dones):
            if done:
                episode_returns.append(running_returns[i])
                running_returns[i] = 0.0
                if len(episode_returns) >= n_episodes:
                    break

    return float(np.mean(episode_returns[:n_episodes]))


def compute_baselines(
    expert_policy,
    venv: VecEnv,
    rng: np.random.Generator,
    n_expert_episodes: int = 20,
    n_random_episodes: int = 100,
) -> Dict[str, float]:
    """Compute expert and random return baselines for an environment.

    Args:
        expert_policy: A policy with a ``predict(obs, deterministic=True)``
            method (e.g. an SB3 BaseAlgorithm).
        venv: Vectorized environment.
        rng: Random number generator (unused currently, reserved for future).
        n_expert_episodes: Number of expert rollout episodes.
        n_random_episodes: Number of random rollout episodes.

    Returns:
        Dict with keys ``"expert_return"`` and ``"random_return"``.
    """
    from imitation.data import rollout

    # Expert return
    expert_trajs = rollout.rollout(
        expert_policy,
        venv,
        rollout.make_sample_until(min_episodes=n_expert_episodes),
        rng=rng,
    )
    expert_returns = [float(np.sum(t.rews)) for t in expert_trajs]
    expert_return = float(np.mean(expert_returns))

    # Random return
    random_return = compute_random_return(venv, n_episodes=n_random_episodes)

    return {
        "expert_return": expert_return,
        "random_return": random_return,
    }


def load_or_compute_baselines(
    env_name: str,
    venv: VecEnv,
    expert_policy,
    cache_dir: pathlib.Path,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Load cached baselines or compute and cache them.

    Baselines are stored at ``{cache_dir}/{env_name}/baselines.json``.

    Args:
        env_name: Environment ID (used for cache path).
        venv: Vectorized environment.
        expert_policy: Policy with ``predict(obs, deterministic=True)``.
        cache_dir: Root cache directory.
        rng: Random number generator.

    Returns:
        Dict with keys ``"expert_return"`` and ``"random_return"``.
    """
    cache_path = cache_dir / env_name / "baselines.json"

    if cache_path.exists():
        logger.info(f"Loading cached baselines from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    logger.info(f"Computing baselines for {env_name}...")
    baselines = compute_baselines(expert_policy, venv, rng)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(baselines, f, indent=2)
    logger.info(f"Cached baselines to {cache_path}")

    return baselines
