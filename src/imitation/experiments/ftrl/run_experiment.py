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
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th

from imitation.algorithms import bc, ftrl
from imitation.algorithms.dagger import _save_dagger_demo
from imitation.data import rollout, serialize, types
from imitation.experiments.ftrl import env_utils, experts, policy_utils
from imitation.util import logger as imit_logger

logger = logging.getLogger(__name__)

ALL_ALGOS = ["ftl", "ftrl", "bc", "bc_dagger"]


def _free_memory() -> None:
    """Run GC, drop PyTorch's CUDA cache, and trim glibc's heap.

    glibc's malloc doesn't automatically return freed pages to the OS,
    so RSS grows monotonically even when Python objects are collected.
    ``malloc_trim(0)`` forces the allocator to release pages, preventing
    RSS from ballooning across experiments and triggering the OOM killer
    when multiple shards run in parallel.
    """
    gc.collect()
    if th.cuda.is_available():
        th.cuda.empty_cache()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # Not Linux or libc not found — skip silently


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
    learning_rate: float = 1e-3
    result_name_override: Optional[str] = None
    early_stop: bool = False
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.005
    subsample_strategy: str = "uniform"  # "uniform" or "prefix"


def _compute_round_eval(
    policy,
    expert_policy,
    venv,
    baselines: Dict[str, float],
) -> Dict[str, Any]:
    """Roll out the current policy, evaluate all metrics on that rollout only.

    Returns a dict with ``rollout_cross_entropy``,
    ``expert_rollout_cross_entropy``, ``normalized_return``,
    ``disagreement_rate``, ``d_eval_size``.
    """
    from imitation.experiments.ftrl.eval_utils import (
        compute_sampled_action_ce,
        eval_policy_rollout,
    )

    eval_res = eval_policy_rollout(
        policy,
        venv,
        n_episodes=10,
        deterministic=True,
        expert_policy=expert_policy,
    )
    obs = eval_res.rollout_batch.obs
    expert_acts = eval_res.rollout_batch.expert_actions

    rollout_ce = compute_sampled_action_ce(policy, obs, expert_acts)
    expert_rollout_ce = compute_sampled_action_ce(expert_policy, obs, expert_acts)

    expert_ret = baselines["expert_return"]
    random_ret = baselines["random_return"]
    score_range = expert_ret - random_ret
    if abs(score_range) < 1e-8:
        norm_ret = 0.0
    else:
        norm_ret = (eval_res.mean_return - random_ret) / score_range

    return {
        "rollout_cross_entropy": round(float(rollout_ce), 6),
        "expert_rollout_cross_entropy": round(float(expert_rollout_ce), 6),
        "normalized_return": round(float(norm_ret), 6),
        "disagreement_rate": round(
            float(eval_res.current_round_disagreement), 6
        ),
        "d_eval_size": int(obs.shape[0]),
    }


def _should_early_stop(
    rce_history: List[float],
    patience: int,
    min_delta: float,
    expert_ce_floor: Optional[float] = None,
) -> bool:
    """Return True if rollout_ce has plateaued AND is near expert-level.

    Two-criterion stop, both must hold:

    1. **Rolling-mean plateau.** Compare the mean of the last ``patience``
       eval points against the mean of the ``patience`` eval points
       immediately before that window. If the improvement is less than
       ``min_delta``, the signal has plateaued. Requires at least
       ``2 * patience`` eval points before the first check.
       Using means rather than mins makes the check noise-robust — a
       single lucky-low early eval no longer pins a false "best ever".

    2. **Absolute sanity gate.** If ``expert_ce_floor`` is provided, also
       require the current rolling mean to be within ``2 * expert_ce_floor``
       of zero. Prevents early-stopping while rollout_ce is still
       clearly far from expert-level (as happened on noisy LunarLander
       BC (growing dataset) runs with the old min-based criterion).
    """
    if patience < 1 or len(rce_history) < 2 * patience:
        return False
    window = rce_history[-patience:]
    prior = rce_history[-2 * patience : -patience]
    current_mean = float(np.mean(window))
    prior_mean = float(np.mean(prior))
    plateau = (prior_mean - current_mean) < min_delta
    if not plateau:
        return False
    if expert_ce_floor is not None and current_mean > 2.0 * expert_ce_floor:
        return False
    return True


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
            "eval_interval": config.eval_interval,
            "warm_start": config.warm_start,
            "beta_rampdown": config.beta_rampdown,
            "bc_n_epochs": config.bc_n_epochs,
            "learning_rate": config.learning_rate,
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
    elif config.algo == "bc_dagger":
        result["per_round"] = _run_bc_dagger(
            config, venv, expert_policy, rng, baselines
        )
    else:
        raise ValueError(f"Unknown algo: {config.algo}")

    elapsed = time.time() - start_time
    result["elapsed_seconds"] = round(elapsed, 1)

    # Save result
    out_file = _result_path(config)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved {out_file} ({elapsed:.1f}s)")

    venv.close()
    _free_memory()
    return result


def _truncate_trajectory(traj: types.Trajectory, n: int) -> types.Trajectory:
    """Return the first ``n`` transitions of ``traj`` as a new Trajectory.

    Mid-episode cut => terminal=False. Used to make each DAgger round contribute
    exactly samples_per_round expert labels, matching BC / BC (growing dataset) bookkeeping.
    """
    assert 0 < n < len(traj), (n, len(traj))
    new_infos = None if traj.infos is None else traj.infos[:n]
    if isinstance(traj, types.TrajectoryWithRew):
        return dataclasses.replace(
            traj,
            obs=traj.obs[: n + 1],
            acts=traj.acts[:n],
            infos=new_infos,
            rews=traj.rews[:n],
            terminal=False,
        )
    return dataclasses.replace(
        traj,
        obs=traj.obs[: n + 1],
        acts=traj.acts[:n],
        infos=new_infos,
        terminal=False,
    )


def _truncate_round_demos(
    round_dir: pathlib.Path, n_target: int, rng: np.random.Generator
) -> None:
    """Rewrite ``round_dir`` so the flattened transitions total exactly ``n_target``.

    Keeps whole saved trajectories while their cumulative length <= n_target,
    then cuts the next trajectory mid-episode to fill the remainder. Matches
    BC / BC (growing dataset) which slice upfront-collected transitions to an exact N.
    """
    # Demos are saved as HuggingFace dataset directories (suffix is still .npz).
    demo_paths = sorted(p for p in round_dir.iterdir() if p.name.endswith(".npz"))
    trajs: List[types.Trajectory] = []
    for p in demo_paths:
        trajs.extend(serialize.load(p))

    kept: List[types.Trajectory] = []
    cum = 0
    for traj in trajs:
        if cum + len(traj) <= n_target:
            kept.append(traj)
            cum += len(traj)
            if cum == n_target:
                break
        else:
            remaining = n_target - cum
            if remaining > 0:
                kept.append(_truncate_trajectory(traj, remaining))
                cum = n_target
            break

    if cum != n_target:
        raise RuntimeError(
            f"Round at {round_dir} collected {sum(len(t) for t in trajs)} "
            f"transitions; only {cum} usable for target {n_target}."
        )

    for p in demo_paths:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    for idx, traj in enumerate(kept):
        _save_dagger_demo(traj, idx, round_dir, rng, prefix="truncated")


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
        optimizer_kwargs={"lr": config.learning_rate},
        custom_logger=custom_logger,
    )

    # Create scratch dir for this run. Clear any stale contents from a
    # previous partial run so DAgger doesn't refuse to overwrite its demos.
    scratch_dir = (
        config.output_dir / "scratch" / f"{config.algo}_{config.env_name}_{config.seed}"
    )
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)

    # Create FTRL trainer.
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

    rce_history: List[float] = []
    per_round: List[Dict[str, Any]] = []

    # Round 0: evaluate the fresh (untrained) policy.
    round0_eval = _compute_round_eval(
        bc_trainer.policy, expert_policy, venv, baselines,
    )
    rce_history.append(round0_eval["rollout_cross_entropy"])
    per_round.append(
        {
            "round": 0,
            "n_observations": 0,
            "train_cross_entropy": None,
            "l2_norm": None,
            "total_loss": None,
            **round0_eval,
        }
    )

    cum_obs = 0
    stopped_early = False
    for round_num in range(1, config.n_rounds + 1):
        # --- 1. Eval current policy BEFORE collecting new training data ---
        is_first = round_num == 1
        is_interval = round_num % config.eval_interval == 0
        is_final = round_num == config.n_rounds
        should_eval = is_first or is_interval or is_final

        eval_data: Optional[Dict[str, Any]] = None
        if should_eval:
            eval_data = _compute_round_eval(
                bc_trainer.policy, expert_policy, venv, baselines,
            )
            rce_history.append(eval_data["rollout_cross_entropy"])

            if config.early_stop and _should_early_stop(
                rce_history,
                config.early_stop_patience,
                config.early_stop_min_delta,
                expert_ce_floor=baselines.get("expert_self_ce"),
            ):
                stopped_early = True
                logger.info(
                    f"{config.algo}/{config.env_name}/seed{config.seed}: "
                    f"early stop at round {round_num} "
                    f"(rollout_ce plateau over "
                    f"{config.early_stop_patience} eval points)"
                )

        # --- 2. Collect expert demos for this round ---
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

        round_dir = trainer._demo_dir_path_for_round()
        _truncate_round_demos(round_dir, config.samples_per_round, rng)

        # --- 3. Train on all accumulated demos ---
        trainer.extend_and_update({})
        cum_obs += config.samples_per_round

        metrics = list(trainer.get_metrics())
        m = metrics[-1]
        round_data: Dict[str, Any] = {
            "round": round_num,
            "n_observations": cum_obs,
            "train_cross_entropy": round(m.cross_entropy, 6),
            "l2_norm": round(m.l2_norm, 6),
            "total_loss": round(m.total_loss, 6),
            "rollout_cross_entropy": None,
            "expert_rollout_cross_entropy": None,
            "normalized_return": None,
            "disagreement_rate": None,
            "d_eval_size": None,
        }

        if eval_data is not None:
            round_data.update(eval_data)

        per_round.append(round_data)
        if stopped_early:
            break

        _free_memory()

    return per_round


def _collect_and_subsample_transitions(
    all_transitions: "Union[types.TransitionsMinimal, list]",
    n_target: int,
    strategy: str,
    rng: np.random.Generator,
) -> "Union[types.TransitionsMinimal, list]":
    """Select n_target transitions from ``all_transitions``.

    "prefix"  → return ``all_transitions[:n_target]`` (original behavior).
    "uniform" → return ``n_target`` transitions picked uniformly without
                replacement across the full pool.
    """
    if strategy == "prefix":
        return all_transitions[:n_target]
    if strategy == "uniform":
        if len(all_transitions) < n_target:
            raise ValueError(
                f"uniform subsample needs {n_target} transitions, "
                f"pool has {len(all_transitions)}"
            )
        idx = rng.choice(len(all_transitions), size=n_target, replace=False)
        idx.sort()  # stable order for reproducibility across backends
        if isinstance(all_transitions, types.TransitionsMinimal):
            # Build a new Transitions(-like) dataclass with each numpy field
            # gathered by the index array. ``__getitem__`` only supports
            # int/slice, so we replace fields directly.
            field_updates = {
                f.name: getattr(all_transitions, f.name)[idx]
                for f in dataclasses.fields(all_transitions)
            }
            return dataclasses.replace(all_transitions, **field_updates)
        return [all_transitions[i] for i in idx]
    raise ValueError(f"Unknown subsample strategy: {strategy!r}")


def _run_bc(
    config: ExperimentConfig,
    venv,
    expert_policy,
    rng: np.random.Generator,
    baselines: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Fixed BC baseline: train once on the full expert dataset.

    Returns a single-round result. The plotter draws BC as horizontal
    reference lines on all subplots.
    """
    total_timesteps = config.n_rounds * config.samples_per_round

    if config.policy_mode == "linear":
        policy = policy_utils.create_linear_policy(expert_policy)
    else:
        policy = policy_utils.create_end_to_end_policy(
            venv.observation_space, venv.action_space
        )

    sample_until = rollout.make_sample_until(
        min_timesteps=total_timesteps, min_episodes=1
    )
    trajs = rollout.generate_trajectories(
        policy=expert_policy,
        venv=venv,
        sample_until=sample_until,
        deterministic_policy=True,
        rng=rng,
    )
    all_transitions = rollout.flatten_trajectories(list(trajs))
    if len(all_transitions) > total_timesteps:
        all_transitions = _collect_and_subsample_transitions(
            all_transitions,
            n_target=total_timesteps,
            strategy=config.subsample_strategy,
            rng=rng,
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
        optimizer_kwargs={"lr": config.learning_rate},
        custom_logger=custom_logger,
    )
    bc_trainer.train(n_epochs=config.bc_n_epochs)

    eval_data = _compute_round_eval(
        bc_trainer.policy, expert_policy, venv, baselines,
    )

    l2_norms = [
        th.sum(th.square(w)).item() for w in bc_trainer.policy.parameters()
    ]
    l2_norm = sum(l2_norms) / 2

    return [
        {
            "round": 0,
            "n_observations": total_timesteps,
            "train_cross_entropy": None,
            "l2_norm": round(l2_norm, 6),
            "total_loss": None,
            **eval_data,
        }
    ]


def _run_bc_dagger(
    config: ExperimentConfig,
    venv,
    expert_policy,
    rng: np.random.Generator,
    baselines: Dict[str, float],
) -> List[Dict[str, Any]]:
    """BC (growing dataset) baseline.

    Per-round ERM on a growing PREFIX of the expert dataset, sized to
    match DAgger's aggregated observation budget. Eval uses the same
    aggregated D_eval^t buffer construction as FTL/FTRL+DAgger (spec §3.4).

    A round-0 eval (fresh policy, before any training) is also emitted,
    and the outer round loop early-stops when rollout_ce plateaus.
    """
    total_timesteps = config.n_rounds * config.samples_per_round

    sample_until = rollout.make_sample_until(
        min_timesteps=total_timesteps, min_episodes=1
    )
    trajs = rollout.generate_trajectories(
        policy=expert_policy,
        venv=venv,
        sample_until=sample_until,
        deterministic_policy=True,
        rng=rng,
    )
    all_transitions = rollout.flatten_trajectories(list(trajs))
    if len(all_transitions) < total_timesteps:
        raise RuntimeError(
            f"BC (growing dataset): collected {len(all_transitions)} "
            f"transitions, need {total_timesteps}"
        )
    all_transitions = _collect_and_subsample_transitions(
        all_transitions,
        n_target=total_timesteps,
        strategy=config.subsample_strategy,
        rng=rng,
    )

    warm_start = env_utils.is_atari(config.env_name)

    # Build the initial fresh policy for round 0.
    if config.policy_mode == "linear":
        policy = policy_utils.create_linear_policy(expert_policy)
    else:
        policy = policy_utils.create_end_to_end_policy(
            venv.observation_space, venv.action_space
        )

    rce_history: List[float] = []
    per_round: List[Dict[str, Any]] = []

    # Round 0: evaluate fresh policy.
    round0_eval = _compute_round_eval(
        policy, expert_policy, venv, baselines,
    )
    rce_history.append(round0_eval["rollout_cross_entropy"])
    per_round.append(
        {
            "round": 0,
            "n_observations": 0,
            "train_cross_entropy": None,
            "l2_norm": None,
            "total_loss": None,
            **round0_eval,
        }
    )

    stopped_early = False
    for round_num in range(1, config.n_rounds + 1):
        # --- 1. Eval current policy ---
        is_first = round_num == 1
        is_interval = round_num % config.eval_interval == 0
        is_final = round_num == config.n_rounds
        should_eval = is_first or is_interval or is_final

        eval_data: Optional[Dict[str, Any]] = None
        if should_eval:
            eval_data = _compute_round_eval(
                policy, expert_policy, venv, baselines,
            )
            rce_history.append(eval_data["rollout_cross_entropy"])

            if config.early_stop and _should_early_stop(
                rce_history,
                config.early_stop_patience,
                config.early_stop_min_delta,
                expert_ce_floor=baselines.get("expert_self_ce"),
            ):
                stopped_early = True
                logger.info(
                    f"bc_dagger/{config.env_name}/seed{config.seed}: "
                    f"early stop at round {round_num} "
                    f"(rollout_ce plateau over "
                    f"{config.early_stop_patience} eval points)"
                )

        # --- 2. Train on growing prefix of expert data ---
        k = round_num * config.samples_per_round
        prefix = all_transitions[:k]

        if not warm_start:
            if config.policy_mode == "linear":
                policy = policy_utils.create_linear_policy(expert_policy)
            else:
                policy = policy_utils.create_end_to_end_policy(
                    venv.observation_space, venv.action_space
                )

        custom_logger = imit_logger.configure(
            str(
                config.output_dir
                / "tb"
                / f"bc_dagger_{config.env_name}_{config.seed}_r{round_num}"
            ),
            format_strs=[],
        )
        bc_trainer = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            rng=rng,
            policy=policy,
            demonstrations=prefix,
            batch_size=min(32, len(prefix)),
            optimizer_kwargs={"lr": config.learning_rate},
            custom_logger=custom_logger,
        )
        bc_trainer.train(n_epochs=config.bc_n_epochs)
        policy = bc_trainer.policy

        l2_norms = [th.sum(th.square(w)).item() for w in policy.parameters()]
        l2_norm = sum(l2_norms) / 2

        round_data: Dict[str, Any] = {
            "round": round_num,
            "n_observations": k,
            "train_cross_entropy": None,
            "l2_norm": round(l2_norm, 6),
            "total_loss": None,
            "rollout_cross_entropy": None,
            "expert_rollout_cross_entropy": None,
            "normalized_return": None,
            "disagreement_rate": None,
            "d_eval_size": None,
        }

        if eval_data is not None:
            round_data.update(eval_data)

        per_round.append(round_data)
        if stopped_early:
            break

        _free_memory()

    return per_round


def _result_path(config: ExperimentConfig) -> pathlib.Path:
    """Return the output JSON path for a given experiment config."""
    env_dir = config.output_dir / config.env_name.replace("/", "_")
    name = config.result_name_override or config.algo
    return env_dir / f"{name}_{config.policy_mode}_seed{config.seed}.json"


def _is_already_done(config: ExperimentConfig) -> bool:
    """Check if this experiment has already been run with matching config.

    Checks that the result JSON exists AND that its stored config matches the
    current ``samples_per_round``, ``n_rounds``, and ``eval_interval``.  This
    prevents stale results from a previous run with different parameters from
    being silently reused.
    """
    out_file = _result_path(config)
    if not out_file.exists():
        return False
    try:
        with open(out_file) as f:
            cached_cfg = json.load(f).get("config", {})
        return (
            cached_cfg.get("samples_per_round") == config.samples_per_round
            and cached_cfg.get("n_rounds") == config.n_rounds
            and cached_cfg.get("eval_interval") == config.eval_interval
        )
    except (json.JSONDecodeError, OSError):
        return False


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
    # Resume support: skip already-completed experiments whose config matches.
    if _is_already_done(config):
        out_file = _result_path(config)
        try:
            with open(out_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
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
                        learning_rate=args.learning_rate,
                        early_stop=args.early_stop,
                        early_stop_patience=args.early_stop_patience,
                        early_stop_min_delta=args.early_stop_min_delta,
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
        "--n-rounds",
        type=int,
        default=60,
        help="Max number of DAgger rounds (subject to early-stop)",
    )
    parser.add_argument(
        "--samples-per-round",
        type=int,
        default=50,
        help="Min timesteps per DAgger round",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        default=False,
        help="Enable early stopping on rollout_ce plateau",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help=(
            "Stop training when rollout_ce has not improved by "
            "--early-stop-min-delta over this many consecutive eval points."
        ),
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.005,
        help="Minimum improvement in rollout_ce to count as progress",
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
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for BC optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=2,
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

    # Clean stale results from seeds outside the current run.
    # Prevents old 5-seed files from poisoning a new 3-seed run.
    expected_seeds = set(range(args.seeds))
    output_dir = pathlib.Path(args.output_dir)
    for env_name in args.envs:
        env_dir = output_dir / env_name.replace("/", "_")
        if not env_dir.exists():
            continue
        for f in env_dir.glob("*.json"):
            # Extract seed from filename like "ftl_linear_seed4.json"
            parts = f.stem.rsplit("seed", 1)
            if len(parts) == 2 and parts[1].isdigit():
                file_seed = int(parts[1])
                if file_seed not in expected_seeds:
                    logger.info(f"Removing stale result: {f} (seed {file_seed} not in current run)")
                    f.unlink()

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
