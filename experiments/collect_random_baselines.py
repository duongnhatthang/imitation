"""Collect and cache random baseline scores for all 7 Atari games.

Runs a RandomPolicy for 30 episodes on each game (using the eval venv with
unclipped rewards) and saves results to experiments/baselines/atari_random_scores.pkl.

Usage:
    python experiments/collect_random_baselines.py

The resulting pickle file maps game names to dicts with "mean" and "std" keys,
e.g. {"Pong": {"mean": -20.8, "std": 1.2}, ...}.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.atari_helpers import ATARI_GAMES, make_atari_eval_venv  # noqa: E402
from imitation.data import rollout  # noqa: E402
from imitation.policies.base import RandomPolicy  # noqa: E402

CACHE_PATH = Path(__file__).resolve().parent / "baselines" / "atari_random_scores.pkl"
N_EPISODES = 30
SEED = 0


def collect_random_baselines(
    n_episodes: int = N_EPISODES,
    seed: int = SEED,
    n_envs: int = 4,
) -> dict:
    """Collect random agent episode returns for all 7 Atari games.

    Args:
        n_episodes: Minimum number of episodes to collect per game.
        seed: Random seed for rollout collection.
        n_envs: Number of parallel environments per game.

    Returns:
        Dict mapping game name to {"mean": float, "std": float}.
    """
    baselines = {}
    rng = np.random.default_rng(seed)

    for game_name, game_id in ATARI_GAMES.items():
        print(f"Collecting random baseline for {game_name} ({game_id})...")
        venv = make_atari_eval_venv(game_id, n_envs=n_envs, seed=seed)
        random_policy = RandomPolicy(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
        )
        trajs = rollout.rollout(
            random_policy,
            venv,
            rollout.make_sample_until(min_episodes=n_episodes),
            rng=rng,
            unwrap=False,
        )
        venv.close()

        stats = rollout.rollout_stats(trajs)
        mean_score = float(stats["return_mean"])
        std_score = float(stats["return_std"])
        baselines[game_name] = {"mean": mean_score, "std": std_score}
        print(f"  {game_name}: mean={mean_score:.2f}, std={std_score:.2f}")

    return baselines


def main():
    """Collect baselines and save to disk."""
    print(f"Collecting random baselines for {len(ATARI_GAMES)} Atari games...")
    print(f"Episodes per game: {N_EPISODES}, seed: {SEED}")
    print()

    baselines = collect_random_baselines(n_episodes=N_EPISODES, seed=SEED)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(baselines, f)

    print()
    print("=" * 50)
    print(f"Saved to: {CACHE_PATH}")
    print()
    print("Summary:")
    for game_name, scores in baselines.items():
        print(f"  {game_name:15s}: mean={scores['mean']:8.2f}, std={scores['std']:7.2f}")

    return baselines


if __name__ == "__main__":
    main()
