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

ALL_ALGOS = ["ftl", "ftrl", "bc"]


def resolve_envs(
    env_group: Optional[str] = None,
    envs: Optional[List[str]] = None,
) -> List[str]:
    """Resolve environment list from --env-group or --envs.

    Args:
        env_group: Name of a predefined environment group (e.g. "classical",
            "atari-zoo"). Mutually exclusive with ``envs``.
        envs: Explicit list of environment names. Mutually exclusive with
            ``env_group``.

    Returns:
        List of environment names to run.

    Raises:
        ValueError: If both ``env_group`` and ``envs`` are specified, or if
            ``env_group`` is not a recognised group name.
    """
    if env_group and envs:
        raise ValueError("Specify --env-group or --envs, not both")
    if env_group:
        if env_group not in env_utils.ENV_GROUPS:
            raise ValueError(
                f"Unknown env group: {env_group}. "
                f"Available: {list(env_utils.ENV_GROUPS.keys())}"
            )
        return env_utils.ENV_GROUPS[env_group]
    if envs:
        return envs
    return list(env_utils.ENV_CONFIGS.keys())  # default: classical only


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
    eval_interval: int
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


def _evaluate_learner_metrics(
    learner_policy,
    expert_policy,
    venv,
    baselines: Dict[str, float],
    n_episodes: int = 10,
    max_steps: int = 10000,
) -> Dict[str, Optional[float]]:
    """Evaluate normalized return and on-policy disagreement rate.

    Rolls out the learner policy for n_episodes (or until max_steps), at each
    step also querying the expert to compute disagreement. Returns normalized
    return (0=random, 1=expert) and disagreement rate (fraction of steps
    where actions differ).
    """
    learner_policy.eval()

    episode_returns: List[float] = []
    total_steps = 0
    total_disagreements = 0
    current_return = 0.0
    current_episode_steps = 0

    obs = venv.reset()
    while len(episode_returns) < n_episodes and total_steps < max_steps:
        learner_action = learner_policy.predict(obs, deterministic=True)[0]
        expert_action = expert_policy.predict(obs, deterministic=True)[0]

        total_steps += 1
        current_episode_steps += 1
        if learner_action[0] != expert_action[0]:
            total_disagreements += 1

        obs, rewards, dones, infos = venv.step(learner_action)
        current_return += rewards[0]

        if dones[0]:
            episode_returns.append(current_return)
            current_return = 0.0
            current_episode_steps = 0

    # If we hit max_steps mid-episode, count the partial return as an episode
    if not episode_returns:
        episode_returns.append(current_return)

    learner_policy.train()

    mean_return = float(np.mean(episode_returns))
    expert_ret = baselines["expert_return"]
    random_ret = baselines["random_return"]
    score_range = expert_ret - random_ret

    if abs(score_range) < 1e-8:
        normalized_return = 0.0
    else:
        normalized_return = (mean_return - random_ret) / score_range

    disagreement_rate = total_disagreements / max(total_steps, 1)

    return {
        "normalized_return": round(normalized_return, 6),
        "disagreement_rate": round(disagreement_rate, 6),
    }


def run_single(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a single (algo, env, seed) experiment.

    Args:
        config: Full experiment configuration.

    Returns:
        Results dict with per-round metrics.
    """
    start_time = time.time()
    rng = np.random.default_rng(config.seed)

    # Device selection: classical MDPs use CPU (tiny networks, GPU adds overhead).
    # Atari uses the worker's assigned GPU if available, else CPU.
    use_gpu = env_utils.is_atari(config.env_name) and _WORKER_GPU_ID is not None
    if not use_gpu and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Create env
    if env_utils.is_atari(config.env_name):
        from imitation.experiments.ftrl.atari_utils import make_atari_venv

        venv = make_atari_venv(config.env_name, n_envs=1, seed=config.seed)
        # Atari CNN policies expect CHW obs (transposed from HWC).
        # VecTransposeImage handles this so BC/DAgger see the same obs space
        # as the policy.
        from stable_baselines3.common.vec_env import is_vecenv_wrapped, VecTransposeImage

        if not is_vecenv_wrapped(venv, VecTransposeImage):
            venv = VecTransposeImage(venv)
    else:
        venv = env_utils.make_env(config.env_name, n_envs=1, rng=rng)

    # Get expert
    expert_policy = experts.get_or_train_expert(
        config.env_name,
        venv,
        cache_dir=config.expert_cache_dir,
        rng=rng,
        seed=config.seed,
    )

    # Load or compute baselines for normalized return
    from imitation.experiments.ftrl.env_baselines import (
        load_or_compute_baselines,
        validate_expert_quality,
    )

    baselines = load_or_compute_baselines(
        config.env_name,
        venv,
        expert_policy,
        config.expert_cache_dir,
        rng,
    )

    # Warn if expert quality is below reference
    is_ok, msg = validate_expert_quality(
        config.env_name,
        baselines["expert_return"],
    )
    if not is_ok:
        logger.warning(f"WARNING: {msg}")

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
        "baselines": baselines,
        "per_round": [],
    }

    if config.algo in ("ftl", "ftrl"):
        result["per_round"] = _run_dagger_variant(
            config,
            venv,
            expert_policy,
            rng,
            baselines,
        )
    elif config.algo == "bc":
        result["per_round"] = _run_bc(config, venv, expert_policy, rng, baselines)
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
    baselines: Dict[str, float],
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
        str(
            config.output_dir / "tb" / f"{config.algo}_{config.env_name}_{config.seed}"
        ),
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
        config.output_dir / "scratch" / f"{config.algo}_{config.env_name}_{config.seed}"
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

    # Extract metrics and compute rollout CE on an aggregated D_eval buffer
    from imitation.data import serialize
    from imitation.experiments.ftrl.eval_utils import (
        compute_sampled_action_ce,
        eval_policy_rollout,
    )

    metrics = list(trainer.get_metrics())
    total_rounds = len(metrics)

    d_eval_obs: List[np.ndarray] = []
    d_eval_expert_acts: List[np.ndarray] = []

    per_round: List[Dict[str, Any]] = []
    cum_obs = 0
    for m in metrics:
        # m.round_num is post-increment (1-indexed); demo dirs are 0-indexed
        demo_round = m.round_num - 1
        round_dir = trainer._demo_dir_path_for_round(demo_round)
        demo_paths = trainer._get_demo_paths(round_dir)
        round_demos = []
        for p in demo_paths:
            round_demos.extend(serialize.load(p))
        round_transitions = rollout.flatten_trajectories(round_demos)
        cum_obs += len(round_transitions)

        round_data: Dict[str, Any] = {
            "round": m.round_num,
            "n_observations": cum_obs,
            "train_cross_entropy": round(m.cross_entropy, 6),
            "l2_norm": round(m.l2_norm, 6),
            "total_loss": round(m.total_loss, 6),
            "rollout_cross_entropy": None,
            "normalized_return": None,
            "disagreement_rate": None,
            "d_eval_size": sum(a.shape[0] for a in d_eval_obs),
        }

        # Evaluate at intervals: round 1, every eval_interval, and final round.
        # NOTE: For DAgger, bc_trainer.policy is the *final* trained policy at
        # this point (trainer.train() runs all rounds). Per-round evaluation
        # during training would be more informative but requires a bigger
        # refactor. This is acceptable for the first version.
        is_first = m.round_num == 1
        is_interval = m.round_num % config.eval_interval == 0
        is_final = m.round_num == total_rounds
        if is_first or is_interval or is_final:
            eval_res = eval_policy_rollout(
                bc_trainer.policy,
                venv,
                n_episodes=20,
                deterministic=True,
                expert_policy=expert_policy,
            )
            d_eval_obs.append(eval_res.rollout_batch.obs)
            d_eval_expert_acts.append(eval_res.rollout_batch.expert_actions)
            agg_obs = np.concatenate(d_eval_obs, axis=0)
            agg_acts = np.concatenate(d_eval_expert_acts, axis=0)
            round_data["rollout_cross_entropy"] = round(
                compute_sampled_action_ce(bc_trainer.policy, agg_obs, agg_acts),
                6,
            )
            expert_ret = baselines["expert_return"]
            random_ret = baselines["random_return"]
            score_range = expert_ret - random_ret
            if abs(score_range) < 1e-8:
                norm_ret = 0.0
            else:
                norm_ret = (eval_res.mean_return - random_ret) / score_range
            round_data["normalized_return"] = round(norm_ret, 6)
            round_data["disagreement_rate"] = round(
                eval_res.current_round_disagreement, 6
            )
            round_data["d_eval_size"] = int(agg_obs.shape[0])

        per_round.append(round_data)

    return per_round


def _run_bc(
    config: ExperimentConfig,
    venv,
    expert_policy,
    rng: np.random.Generator,
    baselines: Dict[str, float],
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

        round_data = {
            "round": round_num + 1,
            "n_observations": cum_obs,
            "cross_entropy": round(ce, 6),
            "l2_norm": round(l2_norm, 6),
            "total_loss": round(ce, 6),  # BC has no L2 penalty in loss
            "expert_cross_entropy": round(expert_ce, 6),
            "normalized_return": None,
            "disagreement_rate": None,
        }

        is_first = round_num == 0
        is_interval = (round_num + 1) % config.eval_interval == 0
        is_final = round_num == config.n_rounds - 1
        if is_first or is_interval or is_final:
            eval_metrics = _evaluate_learner_metrics(
                bc_trainer.policy,
                expert_policy,
                venv,
                baselines,
            )
            round_data.update(eval_metrics)

        per_round.append(round_data)

    return per_round


def _result_path(config: ExperimentConfig) -> pathlib.Path:
    """Return the output JSON path for a given experiment config."""
    env_dir = config.output_dir / config.env_name.replace("/", "_")
    return env_dir / f"{config.algo}_{config.policy_mode}_seed{config.seed}.json"


def _is_already_done(config: ExperimentConfig) -> bool:
    """Check if this experiment has already been run (result JSON exists)."""
    return _result_path(config).exists()


_WORKER_GPU_ID: Optional[int] = None


def _worker_init(gpu_queue):
    """Pool initializer: assign each worker to a GPU from the queue."""
    global _WORKER_GPU_ID
    try:
        gpu_id = gpu_queue.get_nowait()
    except Exception:
        gpu_id = None
    if gpu_id is not None:
        _WORKER_GPU_ID = gpu_id
        # Best-effort CUDA device assignment
        try:
            import torch as _th

            if _th.cuda.is_available():
                _th.cuda.set_device(gpu_id)
        except Exception:
            pass


def _run_single_wrapper(args):
    """Wrapper for multiprocessing.Pool.map (unpacks config)."""
    config = args
    # Resume support: skip already-completed experiments
    if _is_already_done(config):
        out_file = _result_path(config)
        try:
            with open(out_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass  # corrupted, re-run

    try:
        return run_single(config)
    except Exception as e:
        logger.error(f"Failed: {config.algo}/{config.env_name}/seed{config.seed}: {e}")
        import traceback

        traceback.print_exc()
        return {
            "error": str(e),
            "algo": config.algo,
            "env": config.env_name,
            "seed": config.seed,
        }


def build_configs(args: argparse.Namespace) -> List[ExperimentConfig]:
    """Build list of experiment configs from CLI args."""
    configs = []
    for env_name in args.envs:
        for algo in args.algos:
            for seed in range(args.seeds):
                configs.append(
                    ExperimentConfig(
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
                        eval_interval=args.eval_interval,
                        output_dir=pathlib.Path(args.output_dir),
                        expert_cache_dir=pathlib.Path(args.expert_cache_dir),
                    )
                )
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Run FTL vs FTRL vs BC experiments on classical MDPs",
    )
    parser.add_argument("--envs", nargs="+", default=None, help="Environments to test")
    parser.add_argument(
        "--env-group",
        type=str,
        default=None,
        choices=list(env_utils.ENV_GROUPS.keys()),
        help="Predefined environment group to run",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=ALL_ALGOS,
        choices=ALL_ALGOS,
        help="Algorithms to run",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument(
        "--n-rounds", type=int, default=20, help="Number of DAgger rounds"
    )
    parser.add_argument(
        "--samples-per-round",
        type=int,
        default=500,
        help="Min timesteps per DAgger round",
    )
    parser.add_argument(
        "--policy-mode",
        choices=["end_to_end", "linear"],
        default="linear",
        help="Policy training mode",
    )
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=0.01,
        help="L2 regularization weight for FTRL",
    )
    parser.add_argument(
        "--l2-decay", action="store_true", help="Use decaying L2 schedule (lambda/n)"
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        default=True,
        help="Keep policy weights between rounds (default)",
    )
    parser.add_argument(
        "--no-warm-start",
        dest="warm_start",
        action="store_false",
        help="Reinitialize trainable params each round",
    )
    parser.add_argument(
        "--beta-rampdown",
        type=int,
        default=15,
        help="Rounds for beta schedule linear rampdown",
    )
    parser.add_argument(
        "--bc-n-epochs", type=int, default=20, help="Number of BC training epochs"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="Evaluate learner every N rounds (also first and last)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Directory for results",
    )
    parser.add_argument(
        "--expert-cache-dir",
        type=str,
        default="experiments/expert_cache",
        help="Directory for caching trained experts",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers (1=sequential)",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=0,
        help="Number of GPUs to distribute workers across (0=CPU only)",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run experiments even if result JSON already exists",
    )
    args = parser.parse_args()
    args.envs = resolve_envs(env_group=args.env_group, envs=args.envs)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Pass GPU count to workers via env var
    os.environ["FTRL_N_GPUS"] = str(args.n_gpus)

    all_configs = build_configs(args)
    total_requested = len(all_configs)

    # Resume support: skip configs whose result JSON already exists
    if args.force_rerun:
        configs = all_configs
        skipped = 0
    else:
        configs = [c for c in all_configs if not _is_already_done(c)]
        skipped = total_requested - len(configs)

    total = len(configs)
    logger.info(
        f"Running {total} new experiments ({skipped} already cached, "
        f"{total_requested} total requested): "
        f"{len(args.envs)} envs × {len(args.algos)} algos × {args.seeds} seeds"
    )
    logger.info(
        f"Policy mode: {args.policy_mode}, workers: {args.n_workers}, "
        f"GPUs: {args.n_gpus}"
    )

    if total == 0:
        logger.info("All experiments already cached. Nothing to run.")
        return

    # Pre-train and cache experts sequentially before parallel dispatch.
    # Without this, parallel workers all see "no cache" simultaneously and
    # redundantly train the same expert (e.g. 15 workers each training
    # MountainCar for 1M steps instead of one training + 14 cache hits).
    expert_cache_dir = pathlib.Path(args.expert_cache_dir)
    for env_name in args.envs:
        rng = np.random.default_rng(0)
        if env_utils.is_atari(env_name):
            from imitation.experiments.ftrl.atari_utils import make_atari_venv
            from stable_baselines3.common.vec_env import VecTransposeImage

            venv = make_atari_venv(env_name, n_envs=1, seed=0)
            venv = VecTransposeImage(venv)
        else:
            venv = env_utils.make_env(env_name, n_envs=1, rng=rng)
        experts.get_or_train_expert(
            env_name,
            venv,
            cache_dir=expert_cache_dir,
            rng=rng,
            seed=0,
        )
        venv.close()

    start_time = time.time()

    def _fmt_eta(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}m"
        return f"{seconds / 3600:.1f}h"

    if args.n_workers <= 1:
        results = []
        for i, config in enumerate(configs):
            t0 = time.time()
            logger.info(
                f"[{i+1}/{total}] {config.algo}/{config.env_name}/seed{config.seed}"
            )
            results.append(run_single(config))
            elapsed_so_far = time.time() - start_time
            done = i + 1
            avg_per_exp = elapsed_so_far / done
            remaining = (total - done) * avg_per_exp
            logger.info(
                f"Progress: {done}/{total} done | "
                f"elapsed {_fmt_eta(elapsed_so_far)} | "
                f"avg {_fmt_eta(avg_per_exp)}/exp | "
                f"ETA {_fmt_eta(remaining)}"
            )
    else:
        ctx = multiprocessing.get_context("spawn")
        # Build a queue of GPU IDs to hand out to workers (cycling).
        gpu_queue = ctx.Queue()
        if args.n_gpus > 0:
            for w in range(args.n_workers):
                gpu_queue.put(w % args.n_gpus)
        else:
            for _ in range(args.n_workers):
                gpu_queue.put(None)

        with ctx.Pool(
            args.n_workers,
            initializer=_worker_init,
            initargs=(gpu_queue,),
        ) as pool:
            results = []
            for i, result in enumerate(
                pool.imap_unordered(_run_single_wrapper, configs)
            ):
                results.append(result)
                elapsed_so_far = time.time() - start_time
                done = i + 1
                avg_per_exp = elapsed_so_far / done
                # With n_workers parallel, effective time per exp is
                # avg_per_exp (wall-clock). ETA = remaining_exps * avg_per_exp
                # but divided by parallelism: remaining / n_workers * wall_per_batch
                remaining_exps = total - done
                # Conservative ETA: assumes same throughput continues
                eta = remaining_exps * (elapsed_so_far / done)
                if done % max(1, total // 20) == 0 or done == total:
                    logger.info(
                        f"Progress: {done}/{total} done | "
                        f"elapsed {_fmt_eta(elapsed_so_far)} | "
                        f"throughput {done/elapsed_so_far*60:.1f} exp/min | "
                        f"ETA {_fmt_eta(eta)}"
                    )

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
            logger.error(
                f"  FAILED: {e['algo']}/{e['env']}/seed{e['seed']}: {e['error']}"
            )


if __name__ == "__main__":
    main()
