"""Experiment runner for FTL vs FTRL vs BC comparison.

Runs all (env × algo × seed) combinations with multiprocessing parallelism.
Results are saved as JSON files for downstream plotting.

Usage:
    python -m imitation.experiments.ftrl.run_experiment --envs CartPole-v1 --seeds 3
    python -m imitation.experiments.ftrl.run_experiment --n-workers 8  # full run
"""

import argparse
import dataclasses
import json
import logging
import multiprocessing
import os
import pathlib
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch as th
from stable_baselines3.common import policies as sb3_policies

from imitation.algorithms import bc, ftrl
from imitation.data import rollout, types
from imitation.experiments.ftrl import env_utils, experts, policy_utils
from imitation.util import logger as imit_logger

logger = logging.getLogger(__name__)

ALL_ENVS = list(env_utils.ENV_CONFIGS.keys())
ALL_ALGOS = ["ftl", "ftrl", "bc"]


@dataclasses.dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    algo: str
    env_name: str
    seed: int
    policy_mode: str  # "end_to_end" or "linear"
    n_rounds: int
    samples_per_round: int
    l2_lambda: float
    l2_decay: bool
    warm_start: bool
    beta_rampdown: int
    bc_n_epochs: int
    output_dir: pathlib.Path
    expert_cache_dir: pathlib.Path


def _evaluate_policy_cross_entropy(
    policy: sb3_policies.ActorCriticPolicy,
    transitions: types.Transitions,
) -> float:
    """Compute mean negative log-prob of expert actions under the policy."""
    from imitation.util import util

    policy.eval()
    with th.no_grad():
        tensor_obs = types.map_maybe_dict(
            util.safe_to_tensor,
            types.maybe_unwrap_dictobs(transitions.obs),
        )
        acts = util.safe_to_tensor(transitions.acts)
        _, log_prob, _ = policy.evaluate_actions(tensor_obs, acts)
        cross_entropy = -log_prob.mean().item()
    policy.train()
    return cross_entropy


def run_single(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a single (algo, env, seed) experiment.

    Args:
        config: Full experiment configuration.

    Returns:
        Results dict with per-round metrics.
    """
    start_time = time.time()
    rng = np.random.default_rng(config.seed)

    # Force CPU for classical MDPs
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Create env
    venv = env_utils.make_env(config.env_name, n_envs=1, rng=rng)

    # Get expert
    expert_policy = experts.get_or_train_expert(
        config.env_name,
        venv,
        cache_dir=config.expert_cache_dir,
        rng=rng,
        seed=config.seed,
    )

    # Create output dir
    env_dir = config.output_dir / config.env_name.replace("/", "_")
    env_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "algo": config.algo,
        "env": config.env_name,
        "seed": config.seed,
        "policy_mode": config.policy_mode,
        "config": {
            "l2_lambda": config.l2_lambda,
            "l2_decay": config.l2_decay,
            "n_rounds": config.n_rounds,
            "samples_per_round": config.samples_per_round,
            "warm_start": config.warm_start,
            "beta_rampdown": config.beta_rampdown,
            "bc_n_epochs": config.bc_n_epochs,
        },
        "per_round": [],
    }

    if config.algo in ("ftl", "ftrl"):
        result["per_round"] = _run_dagger_variant(config, venv, expert_policy, rng)
    elif config.algo == "bc":
        result["per_round"] = _run_bc(config, venv, expert_policy, rng)
    else:
        raise ValueError(f"Unknown algo: {config.algo}")

    elapsed = time.time() - start_time
    result["elapsed_seconds"] = round(elapsed, 1)

    # Save result
    out_file = env_dir / f"{config.algo}_{config.policy_mode}_seed{config.seed}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved {out_file} ({elapsed:.1f}s)")

    venv.close()
    return result


def _run_dagger_variant(
    config: ExperimentConfig,
    venv,
    expert_policy,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Run FTL or FTRL (DAgger variants)."""
    from imitation.algorithms.dagger import LinearBetaSchedule

    # Create policy
    if config.policy_mode == "linear":
        policy = policy_utils.create_linear_policy(expert_policy)
        use_trainable_params_loss = True
    else:
        policy = policy_utils.create_end_to_end_policy(
            venv.observation_space,
            venv.action_space,
        )
        use_trainable_params_loss = False

    # L2 schedule
    if config.algo == "ftl":
        l2_schedule = ftrl.ConstantL2Schedule(0.0)
    elif config.l2_decay:
        l2_schedule = ftrl.DecayingL2Schedule(config.l2_lambda)
    else:
        l2_schedule = ftrl.ConstantL2Schedule(config.l2_lambda)

    # Create custom logger (suppress output)
    custom_logger = imit_logger.configure(
        str(config.output_dir / "tb" / f"{config.algo}_{config.env_name}_{config.seed}"),
        format_strs=[],
    )

    # Create BC trainer
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        rng=rng,
        policy=policy,
        custom_logger=custom_logger,
    )

    # Create scratch dir for this run
    scratch_dir = (
        config.output_dir
        / "scratch"
        / f"{config.algo}_{config.env_name}_{config.seed}"
    )

    # Create FTRL trainer
    trainer = ftrl.FTRLTrainer(
        venv=venv,
        scratch_dir=scratch_dir,
        bc_trainer=bc_trainer,
        expert_policy=expert_policy,
        rng=rng,
        l2_schedule=l2_schedule,
        warm_start=config.warm_start,
        track_per_round_loss=True,
        use_trainable_params_loss=use_trainable_params_loss,
        beta_schedule=LinearBetaSchedule(config.beta_rampdown),
        custom_logger=custom_logger,
    )

    # Train
    total_timesteps = config.n_rounds * config.samples_per_round
    trainer.train(
        total_timesteps=total_timesteps,
        rollout_round_min_episodes=1,
        rollout_round_min_timesteps=config.samples_per_round,
    )

    # Extract metrics and compute expert baseline CE per round
    from imitation.data import serialize

    per_round = []
    cum_obs = 0
    for m in trainer.get_metrics():
        # m.round_num is post-increment (1-indexed); demo dirs are 0-indexed
        demo_round = m.round_num - 1
        round_dir = trainer._demo_dir_path_for_round(demo_round)
        demo_paths = trainer._get_demo_paths(round_dir)
        round_demos = []
        for p in demo_paths:
            round_demos.extend(serialize.load(p))
        round_transitions = rollout.flatten_trajectories(round_demos)
        expert_ce = _evaluate_policy_cross_entropy(expert_policy, round_transitions)
        cum_obs += len(round_transitions)

        per_round.append({
            "round": m.round_num,
            "n_observations": cum_obs,
            "cross_entropy": round(m.cross_entropy, 6),
            "l2_norm": round(m.l2_norm, 6),
            "total_loss": round(m.total_loss, 6),
            "expert_cross_entropy": round(expert_ce, 6),
        })

    return per_round


def _run_bc(
    config: ExperimentConfig,
    venv,
    expert_policy,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Run BC baseline.

    Collects total data upfront, trains BC, then evaluates on round-sized chunks.
    """
    total_timesteps = config.n_rounds * config.samples_per_round

    # Collect expert trajectories with min_timesteps to guarantee enough data
    sample_until = rollout.make_sample_until(
        min_timesteps=total_timesteps,
        min_episodes=1,
    )
    trajs = rollout.generate_trajectories(
        policy=expert_policy,
        venv=venv,
        sample_until=sample_until,
        deterministic_policy=True,
        rng=rng,
    )
    all_transitions = rollout.flatten_trajectories(list(trajs))

    # Trim to exact total_timesteps if we got more
    if len(all_transitions) > total_timesteps:
        all_transitions = all_transitions[:total_timesteps]

    # Create policy
    if config.policy_mode == "linear":
        policy = policy_utils.create_linear_policy(expert_policy)
    else:
        policy = policy_utils.create_end_to_end_policy(
            venv.observation_space,
            venv.action_space,
        )

    custom_logger = imit_logger.configure(
        str(config.output_dir / "tb" / f"bc_{config.env_name}_{config.seed}"),
        format_strs=[],
    )

    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        rng=rng,
        policy=policy,
        demonstrations=all_transitions,
        batch_size=min(32, len(all_transitions)),
        custom_logger=custom_logger,
    )

    # Train on all data
    bc_trainer.train(n_epochs=config.bc_n_epochs)

    # Evaluate on round-sized chunks
    per_round = []
    chunk_size = config.samples_per_round
    cum_obs = 0
    for round_num in range(config.n_rounds):
        start_idx = round_num * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_transitions))
        if start_idx >= len(all_transitions):
            break
        chunk = all_transitions[start_idx:end_idx]
        cum_obs += len(chunk)
        ce = _evaluate_policy_cross_entropy(bc_trainer.policy, chunk)
        expert_ce = _evaluate_policy_cross_entropy(expert_policy, chunk)

        # Compute L2 norm for consistency
        l2_norms = [th.sum(th.square(w)).item() for w in bc_trainer.policy.parameters()]
        l2_norm = sum(l2_norms) / 2

        per_round.append({
            "round": round_num + 1,
            "n_observations": cum_obs,
            "cross_entropy": round(ce, 6),
            "l2_norm": round(l2_norm, 6),
            "total_loss": round(ce, 6),  # BC has no L2 penalty in loss
            "expert_cross_entropy": round(expert_ce, 6),
        })

    return per_round


def _run_single_wrapper(args):
    """Wrapper for multiprocessing.Pool.map (unpacks config)."""
    config = args
    try:
        return run_single(config)
    except Exception as e:
        logger.error(f"Failed: {config.algo}/{config.env_name}/seed{config.seed}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "algo": config.algo, "env": config.env_name,
                "seed": config.seed}


def build_configs(args: argparse.Namespace) -> List[ExperimentConfig]:
    """Build list of experiment configs from CLI args."""
    configs = []
    for env_name in args.envs:
        for algo in args.algos:
            for seed in range(args.seeds):
                configs.append(ExperimentConfig(
                    algo=algo,
                    env_name=env_name,
                    seed=seed,
                    policy_mode=args.policy_mode,
                    n_rounds=args.n_rounds,
                    samples_per_round=args.samples_per_round,
                    l2_lambda=args.l2_lambda,
                    l2_decay=args.l2_decay,
                    warm_start=args.warm_start,
                    beta_rampdown=args.beta_rampdown,
                    bc_n_epochs=args.bc_n_epochs,
                    output_dir=pathlib.Path(args.output_dir),
                    expert_cache_dir=pathlib.Path(args.expert_cache_dir),
                ))
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Run FTL vs FTRL vs BC experiments on classical MDPs",
    )
    parser.add_argument("--envs", nargs="+", default=ALL_ENVS,
                        choices=ALL_ENVS, help="Environments to test")
    parser.add_argument("--algos", nargs="+", default=ALL_ALGOS,
                        choices=ALL_ALGOS, help="Algorithms to run")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds")
    parser.add_argument("--n-rounds", type=int, default=20,
                        help="Number of DAgger rounds")
    parser.add_argument("--samples-per-round", type=int, default=500,
                        help="Min timesteps per DAgger round")
    parser.add_argument("--policy-mode", choices=["end_to_end", "linear"],
                        default="linear", help="Policy training mode")
    parser.add_argument("--l2-lambda", type=float, default=0.01,
                        help="L2 regularization weight for FTRL")
    parser.add_argument("--l2-decay", action="store_true",
                        help="Use decaying L2 schedule (lambda/n)")
    parser.add_argument("--warm-start", action="store_true", default=True,
                        help="Keep policy weights between rounds (default)")
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false",
                        help="Reinitialize trainable params each round")
    parser.add_argument("--beta-rampdown", type=int, default=15,
                        help="Rounds for beta schedule linear rampdown")
    parser.add_argument("--bc-n-epochs", type=int, default=20,
                        help="Number of BC training epochs")
    parser.add_argument("--output-dir", type=str, default="experiments/results",
                        help="Directory for results")
    parser.add_argument("--expert-cache-dir", type=str,
                        default="experiments/expert_cache",
                        help="Directory for caching trained experts")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Number of parallel workers (1=sequential)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    configs = build_configs(args)
    total = len(configs)
    logger.info(
        f"Running {total} experiments: "
        f"{len(args.envs)} envs × {len(args.algos)} algos × {args.seeds} seeds"
    )
    logger.info(f"Policy mode: {args.policy_mode}, workers: {args.n_workers}")

    # Pre-train and cache experts sequentially before parallel dispatch.
    # Without this, parallel workers all see "no cache" simultaneously and
    # redundantly train the same expert (e.g. 15 workers each training
    # MountainCar for 1M steps instead of one training + 14 cache hits).
    expert_cache_dir = pathlib.Path(args.expert_cache_dir)
    for env_name in args.envs:
        rng = np.random.default_rng(0)
        venv = env_utils.make_env(env_name, n_envs=1, rng=rng)
        experts.get_or_train_expert(
            env_name, venv, cache_dir=expert_cache_dir, rng=rng, seed=0,
        )
        venv.close()

    start_time = time.time()

    if args.n_workers <= 1:
        results = []
        for i, config in enumerate(configs):
            logger.info(f"[{i+1}/{total}] {config.algo}/{config.env_name}/seed{config.seed}")
            results.append(run_single(config))
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(args.n_workers) as pool:
            results = pool.map(_run_single_wrapper, configs)

    elapsed = time.time() - start_time

    # Summary
    errors = [r for r in results if "error" in r]
    successes = [r for r in results if "error" not in r]
    logger.info(
        f"Done: {len(successes)}/{total} succeeded, "
        f"{len(errors)} failed, {elapsed:.0f}s total"
    )
    if errors:
        for e in errors:
            logger.error(f"  FAILED: {e['algo']}/{e['env']}/seed{e['seed']}: {e['error']}")


if __name__ == "__main__":
    main()
