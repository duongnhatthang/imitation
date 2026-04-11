"""Experiment runner for FTL vs FTRL vs BC comparison.

Runs all (env × algo × seed) combinations with multiprocessing parallelism.
Results are saved as JSON files for downstream plotting.

Usage:
    python -m imitation.experiments.ftrl.run_experiment --envs CartPole-v1 --seeds 3
    python -m imitation.experiments.ftrl.run_experiment --n-workers 8  # full run
"""

import argparse
import dataclasses
import gc
import json
import logging
import multiprocessing
import os
import pathlib
import shutil
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


def _free_memory() -> None:
    """Run GC and drop PyTorch's CUDA cache to keep RSS bounded across rounds."""
    gc.collect()
    if th.cuda.is_available():
        th.cuda.empty_cache()

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
    batch_size: int = 256,
) -> float:
    """Compute mean negative log-prob of expert actions under the policy.

    Processes transitions in ``batch_size`` chunks so Atari rounds with
    thousands of frame-stacked observations do not OOM the GPU when passed
    through a CNN feature extractor in a single forward pass.
    """
    from imitation.util import util

    device = policy.device
    policy.eval()
    obs_all = types.maybe_unwrap_dictobs(transitions.obs)
    acts_all = transitions.acts
    total_n = len(acts_all)
    total_log_prob_sum = 0.0
    total_count = 0
    with th.no_grad():
        for start in range(0, total_n, batch_size):
            end = min(start + batch_size, total_n)
            obs_chunk = types.map_maybe_dict(lambda x: x[start:end], obs_all)
            tensor_obs = types.map_maybe_dict(
                lambda x: util.safe_to_tensor(x).to(device),
                obs_chunk,
            )
            acts_chunk = util.safe_to_tensor(acts_all[start:end]).to(device)
            _, log_prob, _ = policy.evaluate_actions(tensor_obs, acts_chunk)
            total_log_prob_sum += float(log_prob.sum().item())
            total_count += int(log_prob.numel())
            del tensor_obs, acts_chunk, log_prob
    policy.train()
    if total_count == 0:
        return 0.0
    return -total_log_prob_sum / total_count


def _evaluate_learner_metrics(
    learner_policy,
    expert_policy,
    venv,
    baselines: Dict[str, float],
    n_episodes: int = 5,
    max_steps: int = 50000,
) -> Dict[str, Optional[float]]:
    """Evaluate normalized return and on-policy disagreement rate.

    Rolls out the learner policy for n_episodes (or until max_steps), at each
    step also querying the expert to compute disagreement. Returns normalized
    return (0=random, 1=expert) and disagreement rate (fraction of steps
    where actions differ).

    Episode returns are read from Monitor's ``info['episode']['r']`` field.
    This matches the methodology used by ``evaluate_policy`` to measure the
    baseline ``expert_return``. It is essential for Atari: SB3's make_vec_env
    wraps the raw gym env with Monitor first, then ``AtariWrapper``, which
    means Monitor observes real game-over transitions (full-game returns),
    whereas ``dones[0]`` at the outer venv fires on every life loss via
    ``EpisodicLifeEnv``. Reading ``dones[0]`` would give per-life returns and
    underestimate normalized return by a factor of n_lives.
    """
    learner_policy.eval()

    episode_returns: List[float] = []
    total_steps = 0
    total_disagreements = 0

    obs = venv.reset()
    while len(episode_returns) < n_episodes and total_steps < max_steps:
        learner_action = learner_policy.predict(obs, deterministic=True)[0]
        expert_action = expert_policy.predict(obs, deterministic=True)[0]

        total_steps += 1
        if learner_action[0] != expert_action[0]:
            total_disagreements += 1

        obs, rewards, dones, infos = venv.step(learner_action)

        # Monitor populates info['episode'] only on real episode end (real
        # env done). For Atari this filters out EpisodicLifeEnv's life-loss
        # terminations; for classical MDPs it coincides with dones[0].
        ep_info = infos[0].get("episode") if infos and len(infos) > 0 else None
        if ep_info is not None:
            episode_returns.append(float(ep_info["r"]))

    learner_policy.train()

    disagreement_rate = total_disagreements / max(total_steps, 1)

    if not episode_returns:
        # No complete episode within the budget. Return disagreement only.
        return {
            "normalized_return": None,
            "disagreement_rate": round(disagreement_rate, 6),
        }

    mean_return = float(np.mean(episode_returns))
    expert_ret = baselines["expert_return"]
    random_ret = baselines["random_return"]
    score_range = expert_ret - random_ret

    if abs(score_range) < 1e-8:
        normalized_return = 0.0
    else:
        normalized_return = (mean_return - random_ret) / score_range

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

    # Seed torch's global RNG AFTER loading the expert. ``PPO.load`` resets
    # torch's RNG state (via ``torch.load``), which would clobber any manual
    # seeding done earlier and make linear-policy ``action_net`` init identical
    # across seeds. Seeding here ensures that ``create_linear_policy``'s
    # ``xavier_uniform_`` init and the BC dataloader shuffle differ per seed.
    th.manual_seed(config.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(config.seed)

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

    # Create scratch dir for this run. Clear any stale contents from a
    # previous partial run so DAgger doesn't refuse to overwrite its demos.
    scratch_dir = (
        config.output_dir / "scratch" / f"{config.algo}_{config.env_name}_{config.seed}"
    )
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)

    # Create FTRL trainer. track_per_round_loss=False because FTRLTrainer's
    # internal _compute_round_loss does an unbatched forward pass on the
    # entire round's transitions, which OOMs on Atari. We compute round CE
    # ourselves below via the batched _evaluate_policy_cross_entropy.
    trainer = ftrl.FTRLTrainer(
        venv=venv,
        scratch_dir=scratch_dir,
        bc_trainer=bc_trainer,
        expert_policy=expert_policy,
        rng=rng,
        l2_schedule=l2_schedule,
        warm_start=config.warm_start,
        track_per_round_loss=False,
        use_trainable_params_loss=use_trainable_params_loss,
        beta_schedule=LinearBetaSchedule(config.beta_rampdown),
        custom_logger=custom_logger,
    )

    # Per-round training loop. Replaces ``trainer.train(total_timesteps=...)``
    # with an explicit loop of (collect one round of demos -> extend_and_update
    # -> evaluate learner on-policy) so that:
    #   1. Per-round learner eval reflects the learner *at that round*, not the
    #      final post-training policy.
    #   2. Reported ``n_observations`` is nominal (round_num * samples_per_round)
    #      so BC and DAgger share the same x-axis even when DAgger's
    #      per-round rollout overshoots ``samples_per_round`` because
    #      ``rollout_round_min_episodes=1`` forces at least one full episode.
    from imitation.data import serialize

    per_round = []
    for round_idx in range(config.n_rounds):
        round_num = round_idx + 1  # 1-indexed

        # Collect one round of demos via the trainer's beta-mixture collector.
        collector = trainer.create_trajectory_collector()
        sample_until = rollout.make_sample_until(
            min_timesteps=max(config.samples_per_round, trainer.batch_size),
            min_episodes=1,
        )
        rollout.generate_trajectories(
            policy=expert_policy,
            venv=collector,
            sample_until=sample_until,
            deterministic_policy=True,
            rng=collector.rng,
        )

        # Train BC on all demos collected so far (dataset aggregation).
        # FTRLTrainer.extend_and_update updates the L2 weight and runs BC.
        trainer.extend_and_update(bc_train_kwargs=None)

        # Compute learner CE and expert CE on this round's freshly collected
        # demos. Both use the batched helper to avoid OOM on Atari.
        round_dir = trainer._demo_dir_path_for_round(round_idx)
        demo_paths = trainer._get_demo_paths(round_dir)
        round_demos = []
        for p in demo_paths:
            round_demos.extend(serialize.load(p))
        round_transitions = rollout.flatten_trajectories(round_demos)
        learner_ce = _evaluate_policy_cross_entropy(
            bc_trainer.policy, round_transitions
        )
        expert_ce = _evaluate_policy_cross_entropy(expert_policy, round_transitions)

        # L2 norm of trainable params (matches FTRLTrainer.use_trainable_params_loss).
        l2_norms = [
            th.sum(th.square(w)).item()
            for w in bc_trainer.policy.parameters()
            if w.requires_grad
        ]
        l2_norm = sum(l2_norms) / 2
        total_loss = learner_ce + config.l2_lambda * l2_norm

        round_data = {
            "round": round_num,
            "n_observations": round_num * config.samples_per_round,
            "cross_entropy": round(learner_ce, 6),
            "l2_norm": round(l2_norm, 6),
            "total_loss": round(total_loss, 6),
            "expert_cross_entropy": round(expert_ce, 6),
            "normalized_return": None,
            "disagreement_rate": None,
        }

        # Evaluate the learner at round 1, every eval_interval, and final round.
        is_first = round_num == 1
        is_interval = round_num % config.eval_interval == 0
        is_final = round_num == config.n_rounds
        if is_first or is_interval or is_final:
            eval_metrics = _evaluate_learner_metrics(
                bc_trainer.policy,
                expert_policy,
                venv,
                baselines,
            )
            round_data.update(eval_metrics)

        per_round.append(round_data)

        # Drop references and flush caches so RSS does not accumulate across
        # rounds. Cached torch allocations and stale ``round_demos`` /
        # ``round_transitions`` are freed each round; the DAgger dataset
        # inside the trainer grows cumulatively by design.
        del round_demos, round_transitions
        _free_memory()

    return per_round


def _run_bc(
    config: ExperimentConfig,
    venv,
    expert_policy,
    rng: np.random.Generator,
    baselines: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Run BC baseline.

    Collects ``n_rounds * samples_per_round`` transitions upfront, then trains
    BC incrementally on growing prefixes: at round k the trainer is fit on the
    first ``k * samples_per_round`` transitions. This makes the BC per-round
    curve meaningful (learner at round k reflects training on k chunks of data)
    and comparable to DAgger's per-round curve on an identical x-axis.
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

    # We'll call set_demonstrations() per round with growing prefixes; the
    # initial dataset just satisfies BC's required constructor argument.
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        rng=rng,
        policy=policy,
        demonstrations=all_transitions[: config.samples_per_round],
        batch_size=min(32, config.samples_per_round),
        custom_logger=custom_logger,
    )

    chunk_size = config.samples_per_round
    per_round = []
    for round_idx in range(config.n_rounds):
        round_num = round_idx + 1
        prefix_end = min(round_num * chunk_size, len(all_transitions))
        prefix = all_transitions[:prefix_end]

        # Incremental BC: fit on growing prefix. Warm-start from the previous
        # round's weights (matches FTL/FTRL warm-start semantics).
        bc_trainer.set_demonstrations(prefix)
        bc_trainer.train(n_epochs=config.bc_n_epochs)

        # Evaluate CE on the most recent chunk (last samples_per_round).
        chunk_start = max(0, prefix_end - chunk_size)
        chunk = all_transitions[chunk_start:prefix_end]
        ce = _evaluate_policy_cross_entropy(bc_trainer.policy, chunk)
        expert_ce = _evaluate_policy_cross_entropy(expert_policy, chunk)

        l2_norms = [th.sum(th.square(w)).item() for w in bc_trainer.policy.parameters()]
        l2_norm = sum(l2_norms) / 2

        round_data = {
            "round": round_num,
            "n_observations": round_num * chunk_size,
            "cross_entropy": round(ce, 6),
            "l2_norm": round(l2_norm, 6),
            "total_loss": round(ce, 6),  # BC has no L2 penalty in loss
            "expert_cross_entropy": round(expert_ce, 6),
            "normalized_return": None,
            "disagreement_rate": None,
        }

        is_first = round_num == 1
        is_interval = round_num % config.eval_interval == 0
        is_final = round_num == config.n_rounds
        if is_first or is_interval or is_final:
            eval_metrics = _evaluate_learner_metrics(
                bc_trainer.policy,
                expert_policy,
                venv,
                baselines,
            )
            round_data.update(eval_metrics)

        per_round.append(round_data)

        # Keep RSS bounded across rounds.
        _free_memory()

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
    parser.add_argument(
        "--shard-idx",
        type=int,
        default=0,
        help="Shard index (0-based) for splitting work across processes",
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=1,
        help="Total number of shards. Each process runs configs[shard_idx::n_shards]",
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

    # Shard support: split work across independent processes (e.g., one per GPU).
    # Each shard sees its slice and skips the rest entirely.
    if args.n_shards > 1:
        all_configs = all_configs[args.shard_idx :: args.n_shards]
        logger.info(
            f"Shard {args.shard_idx}/{args.n_shards}: "
            f"processing {len(all_configs)} of the total configs",
        )

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
