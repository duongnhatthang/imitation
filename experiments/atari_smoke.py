"""Atari smoke test: runs BC, DAgger, and FTRL on Pong and Breakout.

Standalone script (no Sacred) that verifies the full imitation pipeline works
end-to-end on real Atari environments before paying full compute costs.

Usage:
    python experiments/atari_smoke.py [--games Pong,Breakout] [--seed 0]
        [--n-rounds 3] [--total-timesteps 50000] [--n-envs 8]
"""

import argparse
import pickle
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional

# Ensure the project root is on sys.path so that both
# `python experiments/atari_smoke.py` and `python -m experiments.atari_smoke`
# can resolve the `experiments` package.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from experiments.atari_helpers import (
    ATARI_GAMES,
    compute_normalized_score,
    load_atari_expert,
    make_atari_eval_venv,
    make_atari_training_venv,
)
from imitation.algorithms import bc as bc_module
from imitation.algorithms import dagger as dagger_module
from imitation.algorithms.ftrl import FTRLDAggerTrainer
from imitation.data import rollout


# Path to cached random baselines relative to this file
_BASELINES_PATH = Path(__file__).parent / "baselines" / "atari_random_scores.pkl"


def load_random_baselines() -> Dict[str, float]:
    """Load cached random policy scores from disk.

    Returns:
        Dict mapping game display name to random policy mean return.
    """
    with open(_BASELINES_PATH, "rb") as f:
        return pickle.load(f)


def evaluate_policy(policy, eval_venv, rng: np.random.Generator, n_episodes: int = 10) -> float:
    """Evaluate a policy on eval_venv and return mean episode return.

    Args:
        policy: SB3 policy or BC trainer policy.
        eval_venv: Evaluation VecEnv with unclipped rewards.
        rng: Random number generator.
        n_episodes: Number of episodes to evaluate over.

    Returns:
        Mean episode return across n_episodes.
    """
    trajs = rollout.rollout(
        policy,
        eval_venv,
        rollout.make_min_episodes(n_episodes),
        rng=rng,
        unwrap=False,
    )
    # Compute mean return directly from trajectory rewards to avoid any
    # rollout_stats edge cases after long DAgger training runs.
    returns = [float(sum(t.rews)) for t in trajs]
    return float(np.mean(returns))


def make_bc_trainer(
    venv,
    rng: np.random.Generator,
    custom_logger=None,
) -> bc_module.BC:
    """Create a fresh BC trainer with CnnPolicy for Atari.

    Args:
        venv: VecEnv whose observation/action spaces define BC trainer.
        rng: Random number generator.
        custom_logger: Optional imitation logger.

    Returns:
        A fresh BC instance with ActorCriticCnnPolicy.
    """
    policy = ActorCriticCnnPolicy(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        lr_schedule=lambda _: 1e-4,
    )
    return bc_module.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=policy,
        rng=rng,
        custom_logger=custom_logger,
    )


def run_bc(
    game_name: str,
    game_id: str,
    seed: int,
    n_envs: int,
    random_score: float,
    rng: np.random.Generator,
) -> Optional[float]:
    """Run BC on a single Atari game and return normalized score.

    Args:
        game_name: Display name, e.g. "Pong".
        game_id: Raw ALE env ID, e.g. "PongNoFrameskip-v4".
        seed: Random seed.
        n_envs: Number of parallel environments.
        random_score: Cached random policy score for normalization.
        rng: Random number generator.

    Returns:
        Normalized score in [0, 1] or None if an error occurred.
    """
    venv = make_atari_training_venv(game_id, n_envs=n_envs, seed=seed)
    eval_venv = make_atari_eval_venv(game_id, n_envs=n_envs, seed=seed)
    try:
        expert = load_atari_expert(venv, game_id)
        assert venv.observation_space == expert.observation_space, (
            f"Obs space mismatch: venv={venv.observation_space} "
            f"vs expert={expert.observation_space}"
        )

        # Collect expert score for normalization
        expert_score = evaluate_policy(expert, eval_venv, rng=rng, n_episodes=10)
        print(f"  [{game_name}] Expert score: {expert_score:.2f}")
        sys.stdout.flush()

        # Collect expert demonstrations for BC
        expert_trajs = rollout.rollout(
            expert,
            venv,
            rollout.make_min_episodes(20),
            rng=rng,
            unwrap=False,
        )
        transitions = rollout.flatten_trajectories(expert_trajs)

        # Create and train BC
        bc_trainer = make_bc_trainer(venv, rng)
        bc_trainer.set_demonstrations(transitions)
        bc_trainer.train(n_epochs=4, log_rollouts_venv=None, progress_bar=False)

        # Evaluate
        agent_score = evaluate_policy(bc_trainer.policy, eval_venv, rng=rng, n_episodes=10)
        norm_score = compute_normalized_score(agent_score, random_score, expert_score)
        print(f"  [{game_name}] BC agent score: {agent_score:.2f}, normalized: {norm_score:.4f}")
        sys.stdout.flush()
        return norm_score
    finally:
        venv.close()
        eval_venv.close()


def run_dagger(
    game_name: str,
    game_id: str,
    seed: int,
    n_envs: int,
    n_rounds: int,
    total_timesteps: int,
    random_score: float,
    rng: np.random.Generator,
    use_ftrl: bool = False,
    alpha: float = 1.0,
) -> Optional[float]:
    """Run DAgger (or FTRL) on a single Atari game and return normalized score.

    Args:
        game_name: Display name, e.g. "Pong".
        game_id: Raw ALE env ID.
        seed: Random seed.
        n_envs: Number of parallel environments.
        n_rounds: Number of DAgger rounds (used for rollout_round_min_episodes).
        total_timesteps: Total environment timesteps for training.
        random_score: Cached random policy score for normalization.
        rng: Random number generator.
        use_ftrl: If True, use FTRLDAggerTrainer; otherwise use SimpleDAggerTrainer.
        alpha: FTRL alpha parameter (only used if use_ftrl=True).

    Returns:
        Normalized score or None if an error occurred.
    """
    method = "FTRL" if use_ftrl else "DAgger"
    venv = make_atari_training_venv(game_id, n_envs=n_envs, seed=seed)
    eval_venv = make_atari_eval_venv(game_id, n_envs=n_envs, seed=seed)
    try:
        expert = load_atari_expert(venv, game_id)
        assert venv.observation_space == expert.observation_space, (
            f"Obs space mismatch: venv={venv.observation_space} "
            f"vs expert={expert.observation_space}"
        )

        # Expert score for normalization
        expert_score = evaluate_policy(expert, eval_venv, rng=rng, n_episodes=10)
        print(f"  [{game_name}] Expert score: {expert_score:.2f}")
        sys.stdout.flush()

        # Create BC trainer with fresh CnnPolicy
        bc_trainer = make_bc_trainer(venv, rng)

        scratch_dir = tempfile.mkdtemp(prefix=f"atari_{method.lower()}_{game_name}_")

        trainer_kwargs = dict(
            venv=venv,
            scratch_dir=scratch_dir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
        )

        if use_ftrl:
            trainer = FTRLDAggerTrainer(alpha=alpha, **trainer_kwargs)
        else:
            trainer = dagger_module.SimpleDAggerTrainer(**trainer_kwargs)

        trainer.train(
            total_timesteps=total_timesteps,
            bc_train_kwargs=dict(n_epochs=4, log_rollouts_venv=None, progress_bar=False),
        )

        # Evaluate
        agent_score = evaluate_policy(trainer.bc_trainer.policy, eval_venv, rng=rng, n_episodes=10)
        norm_score = compute_normalized_score(agent_score, random_score, expert_score)
        print(f"  [{game_name}] {method} agent score: {agent_score:.2f}, normalized: {norm_score:.4f}")
        sys.stdout.flush()
        return norm_score
    finally:
        venv.close()
        eval_venv.close()


def main():
    parser = argparse.ArgumentParser(
        description="Atari smoke test: BC, DAgger, FTRL on 2 Atari games."
    )
    parser.add_argument(
        "--games",
        type=str,
        default="Pong,Breakout",
        help="Comma-separated list of game display names (default: Pong,Breakout).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=3,
        help="Number of DAgger rounds (used as rollout_round_min_episodes, default: 3).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50000,
        help="Total environment timesteps per method per game (default: 50000).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8).",
    )
    args = parser.parse_args()

    game_names = [g.strip() for g in args.games.split(",")]
    seed = args.seed
    n_rounds = args.n_rounds
    total_timesteps = args.total_timesteps
    n_envs = args.n_envs

    # Validate game names
    for g in game_names:
        if g not in ATARI_GAMES:
            print(f"ERROR: Unknown game '{g}'. Available: {list(ATARI_GAMES.keys())}")
            sys.exit(1)

    # Load cached random baselines
    print(f"Loading random baselines from {_BASELINES_PATH}")
    random_baselines = load_random_baselines()
    print(f"Loaded baselines for: {list(random_baselines.keys())}")

    rng = np.random.default_rng(seed)

    # Results storage: results[method][game] = normalized_score
    results: Dict[str, Dict[str, Optional[float]]] = {
        "BC": {},
        "DAgger": {},
        "FTRL": {},
    }

    for game_name in game_names:
        game_id = ATARI_GAMES[game_name]
        baseline_entry = random_baselines.get(game_name, {"mean": 0.0})
        random_score = baseline_entry["mean"] if isinstance(baseline_entry, dict) else float(baseline_entry)
        print(f"\n{'='*60}")
        print(f"Game: {game_name} ({game_id}), random_score={random_score:.2f}")
        print(f"{'='*60}")

        # --- BC ---
        print(f"\n[BC] Running on {game_name}...")
        try:
            results["BC"][game_name] = run_bc(
                game_name=game_name,
                game_id=game_id,
                seed=seed,
                n_envs=n_envs,
                random_score=random_score,
                rng=np.random.default_rng(seed),
            )
        except Exception:
            print(f"[BC] ERROR on {game_name}:")
            traceback.print_exc()
            results["BC"][game_name] = None

        # --- DAgger ---
        print(f"\n[DAgger] Running on {game_name}...")
        try:
            results["DAgger"][game_name] = run_dagger(
                game_name=game_name,
                game_id=game_id,
                seed=seed,
                n_envs=n_envs,
                n_rounds=n_rounds,
                total_timesteps=total_timesteps,
                random_score=random_score,
                rng=np.random.default_rng(seed),
                use_ftrl=False,
            )
        except Exception:
            print(f"[DAgger] ERROR on {game_name}:")
            traceback.print_exc()
            results["DAgger"][game_name] = None

        # --- FTRL ---
        print(f"\n[FTRL] Running on {game_name}...")
        try:
            results["FTRL"][game_name] = run_dagger(
                game_name=game_name,
                game_id=game_id,
                seed=seed,
                n_envs=n_envs,
                n_rounds=n_rounds,
                total_timesteps=total_timesteps,
                random_score=random_score,
                rng=np.random.default_rng(seed),
                use_ftrl=True,
                alpha=1.0,
            )
        except Exception:
            print(f"[FTRL] ERROR on {game_name}:")
            traceback.print_exc()
            results["FTRL"][game_name] = None

    # Print summary table
    print(f"\n{'='*70}")
    print("=== Atari Smoke Test Results ===")
    print(f"{'='*70}")
    col_w = 10
    header = f"{'Game':<12}" + "".join(f"{m:>{col_w}}" for m in ["BC", "DAgger", "FTRL"])
    print(header)
    print("-" * len(header))
    for game_name in game_names:
        row = f"{game_name:<12}"
        for method in ["BC", "DAgger", "FTRL"]:
            score = results[method].get(game_name)
            if score is None:
                row += f"{'ERROR':>{col_w}}"
            else:
                row += f"{score:>{col_w}.4f}"
        print(row)
    print(f"{'='*70}")
    print("Note: scores normalized as (agent - random) / (expert - random).")
    print("      0.0 = random-level, 1.0 = expert-level.")


if __name__ == "__main__":
    main()
