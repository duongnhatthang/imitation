"""Experiment runner for FTL vs FTRL vs BC comparison.

Runs all (env × algo × seed) combinations with multiprocessing parallelism.
Results are saved as JSON files for downstream plotting.

Usage:
    python -m imitation.experiments.ftrl.run_experiment --envs CartPole-v1 --seeds 3
    python -m imitation.experiments.ftrl.run_experiment --n-workers 8  # full run
"""

import argparse
import copy
import dataclasses
import gc
import hashlib
import json
import logging
import math
import multiprocessing
import os
import pathlib
import random
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th

from imitation.algorithms import bc, ftrl
from imitation.algorithms.dagger import _save_dagger_demo
from imitation.data import rollout, serialize, types
from imitation.experiments.ftrl import env_utils, experts, policy_utils
from imitation.util import logger as imit_logger

logger = logging.getLogger(__name__)

ALL_ALGOS = ["ftl", "ftrl", "bc", "bc_iid", "bc_prefix", "bc_pool"]
DEFAULT_ALGOS = ["ftl", "ftrl", "bc", "bc_iid"]

# Algorithms that collect fresh expert-labeled rollouts each round.
INTERACTIVE_ALGOS = ("ftl", "ftrl", "bc_iid")
# BC baselines that replay a fixed offline expert dataset one round at a time
# through the same trainer as bc_iid, so only the data differs:
#   bc_prefix - round t trains on the first t transitions of the chronological
#               dataset (identical to fixed BC's dataset, consumed in order)
#   bc_pool   - round t trains on t uniform draws from that same fixed pool
#               (this is the behavior of the removed ``bc_dagger``)
OFFLINE_BC_ALGOS = ("bc_prefix", "bc_pool")
# Everything that runs the shared round loop in ``_run_dagger_variant``.
ROUND_LOOP_ALGOS = INTERACTIVE_ALGOS + OFFLINE_BC_ALGOS

# The expert is cached per environment and reused by every cell, so training it
# must not depend on which cell happened to miss the cache first. Pinning this
# keeps a fresh expert_cache reproducible; an existing cache is unaffected.
EXPERT_TRAINING_SEED = 0


def _seed_everything(seed: int) -> None:
    """Seed every global RNG a cell can reach.

    ``run_single`` seeds its own ``np.random.Generator`` and torch; this also
    pins Python's ``random`` and numpy's legacy global RNG, which SB3 and the
    gymnasium wrappers still use, so rerunning the same (env, algo, seed) with
    the same code reproduces the same result.
    """
    random.seed(seed)
    np.random.seed(int(seed) % (2**32))
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def _config_metadata(config):
    """Record every experiment setting and the actual uncommitted source version."""
    settings = dataclasses.asdict(config)
    for key in ("output_dir", "expert_cache_dir"):
        settings[key] = str(settings[key])
    settings["protocol_version"] = 5
    digest = hashlib.sha256()
    for name in (
        "run_experiment.py",
        "expert_dataset.py",
        "eval_utils.py",
        "policy_utils.py",
    ):
        digest.update(pathlib.Path(__file__).with_name(name).read_bytes())
    package_root = pathlib.Path(__file__).parents[2]
    for rel in (
        "algorithms/ftrl.py",
        "algorithms/dagger.py",
        "algorithms/bc.py",
        "data/rollout.py",
    ):
        digest.update((package_root / rel).read_bytes())
    settings["implementation_sha256"] = digest.hexdigest()
    return settings


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
    # --- Outer-loop early stop: cross-round disagreement_rate plateau ---
    outer_early_stop: bool = True
    outer_early_stop_patience: int = 5
    # Plateau threshold in disagreement_rate units (0.005 ≈ 0.5 pp absolute).
    outer_early_stop_min_delta: float = 0.005
    # Stop only when the rolling-mean disagreement_rate is <= this ceiling
    # (i.e. the policy already agrees with the expert at least ~95% of the time).
    outer_early_stop_disagreement_ceiling: float = 0.05
    # --- Inner-loop early stop: per-round val-NLL plateau on held-out split ---
    inner_early_stop: bool = True
    inner_early_stop_patience: int = 5
    inner_early_stop_min_delta: float = 1e-4
    inner_early_stop_val_frac: float = 0.1
    # Minimum held-out val-set size below which we fall back to a fixed
    # bc_n_epochs budget (matches the `min_val_size` parameter of
    # `_split_transitions_for_val`).
    inner_early_stop_min_val_size: int = 32
    inner_early_stop_min_epochs: int = 3
    # None selects the algorithm default: temporal BC, uniform interactive data.
    subsample_strategy: Optional[str] = None
    trajectories_per_round: int = (
        1  # minimum complete rollouts, independent of training batches
    )
    bc_batch_size: int = 32  # cap; effective per-call is min(this, dataset_size)

    def __post_init__(self) -> None:
        if self.beta_rampdown < 0:
            raise ValueError("beta_rampdown must be non-negative")
        if self.subsample_strategy is None:
            self.subsample_strategy = (
                "prefix" if self.algo in ("bc", "bc_prefix") else "uniform"
            )
        if self.subsample_strategy not in ("uniform", "prefix"):
            raise ValueError("subsample_strategy must be 'uniform' or 'prefix'")
        if self.algo in ("bc_iid", "bc_pool") and self.subsample_strategy != "uniform":
            raise ValueError(f"{self.algo} requires uniform sampling")
        if self.algo == "bc_prefix" and self.subsample_strategy != "prefix":
            raise ValueError("bc_prefix requires prefix sampling")
        if self.trajectories_per_round < 1 or self.samples_per_round < 1:
            raise ValueError(
                "trajectories_per_round and samples_per_round must be positive"
            )
        # Within-round independence: the m samples retained in a round must come
        # from m *distinct* trajectories, so a round never collects fewer
        # complete episodes than the samples it keeps. At m = 1 this is a no-op.
        if self.algo in INTERACTIVE_ALGOS:
            self.trajectories_per_round = max(
                self.trajectories_per_round, self.samples_per_round
            )


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
        n_episodes=100,
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
        "disagreement_rate": round(float(eval_res.current_round_disagreement), 6),
        "d_eval_size": int(obs.shape[0]),
        "episode_returns": [float(value) for value in eval_res.episode_returns],
    }


def _compute_val_nll(
    policy,
    val_obs: np.ndarray,
    val_acts: np.ndarray,
    batch_size: int,
) -> float:
    """Mean negative log-likelihood of expert actions under the current policy.

    Evaluated on a held-out validation slice with no gradient. Returns
    ``float('inf')`` if the validation slice is empty so the caller's
    early-stop check treats it as "no improvement".

    Thin wrapper over ``eval_utils.compute_sampled_action_ce`` that adds
    the empty-slice ``inf`` convention. ``batch_size`` is accepted for
    backwards-compatible call sites but ignored — the underlying helper
    chooses its own batching.
    """
    del batch_size  # underlying helper handles batching
    if int(val_obs.shape[0]) == 0:
        return float("inf")
    from imitation.experiments.ftrl.eval_utils import compute_sampled_action_ce

    return float(compute_sampled_action_ce(policy, val_obs, val_acts))


def _split_transitions_for_val(
    n_transitions: int,
    seed: int,
    round_num: int,
    val_frac: float,
    min_val_size: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (train_idx, val_idx) numpy arrays or (None, None) on fallback.

    ``n_val = floor(val_frac * n_transitions)``. Fallback ``(None, None)``
    triggers when ``n_val < min_val_size`` (strict). When ``n_val ==
    min_val_size`` the split proceeds.

    Deterministic for a given ``(seed, round_num)`` pair via the documented
    ``SeedSequence`` list-seed form, which avoids the collision pattern of
    arithmetic seed combination (e.g. seed=2/round=0 and seed=1/round=1).
    """
    n_val = int(val_frac * n_transitions)
    if n_val < min_val_size:
        return None, None
    rng = np.random.default_rng([int(seed), int(round_num)])
    perm = rng.permutation(n_transitions)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def _inner_train(
    trainer_or_bc,
    config: "ExperimentConfig",
    round_num: int,
    is_dagger: bool,
) -> Dict[str, Any]:
    """Run one round of inner BC training, with optional val-split early stopping.

    For ``inner_early_stop=False`` (this task) runs the underlying BC training
    once with ``n_epochs=config.bc_n_epochs``. For DAgger callers this routes
    through ``extend_and_update`` so the round counter advances and demos
    aggregate; fixed BC calls ``bc_trainer.train`` directly.

    The val-split early-stop branch lands in Task 7.

    Args:
        trainer_or_bc: A ``SimpleDAggerTrainer`` (or subclass) if ``is_dagger``;
            otherwise a ``BC`` instance.
        config: Full experiment config; reads ``bc_n_epochs`` and the
            ``inner_early_stop_*`` family.
        round_num: Current DAgger round (used for val-split seeding when
            ``inner_early_stop=True``).
        is_dagger: ``True`` routes to ``extend_and_update``; ``False`` routes to
            ``bc_trainer.train``.

    Returns:
        Dict with keys ``inner_es_stop_epoch`` (int, equals ``bc_n_epochs`` when
        early-stop didn't fire), ``inner_es_val_nll_best`` (float or None),
        ``inner_es_val_nll_trajectory`` (list[float]), ``inner_es_fallback``
        (str | None).
    """
    max_epochs = int(config.bc_n_epochs)

    if not config.inner_early_stop:
        if is_dagger:
            trainer_or_bc.extend_and_update({"n_epochs": max_epochs})
        else:
            trainer_or_bc.train(n_epochs=max_epochs, progress_bar=False)
        return {
            "inner_es_stop_epoch": max_epochs,
            "inner_es_val_nll_best": None,
            "inner_es_val_nll_trajectory": [],
            "inner_es_fallback": None,
        }

    # --- inner_early_stop=True: held-out-val early stopping ---

    # For DAgger callers we use FTRLTrainer.extend_only (loads this round's
    # demos, updates L2 weight, advances round_num, no training). We track
    # current_round so the post-train per-round metrics get appended to the
    # right slot. BC callers already hold the full dataset on their trainer.
    if is_dagger:
        dagger_current_round = trainer_or_bc.extend_only()
        bc_trainer = trainer_or_bc.bc_trainer
        # DAgger's _try_load_demos rebuilds the loader as a th_data.DataLoader,
        # then set_demonstrations wraps it in _WrappedDataLoader (no .dataset).
        # Use _all_demos directly.
        full_transitions = rollout.flatten_trajectories(trainer_or_bc._all_demos)
    else:
        dagger_current_round = None
        bc_trainer = trainer_or_bc
        full_transitions = bc_trainer._demo_data_loader.dataset

    # 3. Decide val split.
    train_idx, val_idx = _split_transitions_for_val(
        n_transitions=len(full_transitions),
        seed=int(config.seed),
        round_num=int(round_num),
        val_frac=float(config.inner_early_stop_val_frac),
        min_val_size=int(config.inner_early_stop_min_val_size),
    )

    if train_idx is None:
        # Dataset too small for a meaningful val split → fixed budget on full set.
        bc_trainer.train(n_epochs=max_epochs, progress_bar=False)
        if is_dagger and dagger_current_round is not None:
            trainer_or_bc.track_round_loss(dagger_current_round)
        return {
            "inner_es_stop_epoch": max_epochs,
            "inner_es_val_nll_best": None,
            "inner_es_val_nll_trajectory": [],
            "inner_es_fallback": "dataset_too_small",
        }

    # 4. Carve the train and val slices.
    if isinstance(full_transitions, types.TransitionsMinimal):
        # TransitionsMinimal.__getitem__ only accepts int/slice; build new
        # dataclass instances by field-wise gather (same pattern as
        # _collect_and_subsample_transitions).
        train_subset = dataclasses.replace(
            full_transitions,
            **{
                f.name: getattr(full_transitions, f.name)[train_idx]
                for f in dataclasses.fields(full_transitions)
            },
        )
        val_subset = dataclasses.replace(
            full_transitions,
            **{
                f.name: getattr(full_transitions, f.name)[val_idx]
                for f in dataclasses.fields(full_transitions)
            },
        )
        val_obs = np.asarray(val_subset.obs)
        val_acts = np.asarray(val_subset.acts)
    else:
        # List-of-dicts fallback.
        train_subset = [full_transitions[int(i)] for i in train_idx]
        val_subset = [full_transitions[int(i)] for i in val_idx]
        val_obs = np.stack([t["obs"] for t in val_subset]).astype(np.float32)
        val_acts = np.stack([t["acts"] for t in val_subset])

    # 5. Re-bind BC trainer to train-only.
    original_batch_size = bc_trainer.batch_size
    original_minibatch_size = bc_trainer.minibatch_size
    if len(train_subset) < original_minibatch_size:
        bc_trainer.batch_size = bc_trainer.minibatch_size = len(train_subset)
    bc_trainer.set_demonstrations(train_subset)

    # 6. Loop epochs with val-NLL early stop.
    patience = int(config.inner_early_stop_patience)
    min_delta = float(config.inner_early_stop_min_delta)
    min_epochs = int(config.inner_early_stop_min_epochs)
    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    val_nll_trajectory: List[float] = []
    stop_epoch = max_epochs
    for epoch in range(max_epochs):
        bc_trainer.train(n_epochs=1, progress_bar=False, reset_tensorboard=False)
        val_nll = _compute_val_nll(
            bc_trainer.policy,
            val_obs,
            val_acts,
            batch_size=int(config.bc_batch_size),
        )
        val_nll_trajectory.append(float(val_nll))
        if val_nll + min_delta < best_val:
            best_val = float(val_nll)
            best_state = copy.deepcopy(bc_trainer.policy.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
        if epoch + 1 >= min_epochs and patience_ctr >= patience:
            stop_epoch = epoch + 1
            break

    # 7. Restore best weights (if we ever saw an improvement). Note: only the
    #    policy state_dict is restored. The BC optimizer (Adam) keeps its
    #    accumulated momentum state from epochs after the best one — this is a
    #    known compromise of restore-best patterns. Acceptable for warm-start
    #    DAgger; revisit if we observe instability in wave-1 traces.
    if best_state is not None:
        bc_trainer.policy.load_state_dict(best_state)

    # 8. Restore the full data loader for subsequent training and diagnostics.
    bc_trainer.batch_size = original_batch_size
    bc_trainer.minibatch_size = original_minibatch_size
    bc_trainer.set_demonstrations(full_transitions)

    # 9. For DAgger callers, append per-round metrics now that training is
    #    done. extend_only deferred this step so we could insert the val
    #    split + ES loop between data load and metric tracking.
    if is_dagger and dagger_current_round is not None:
        trainer_or_bc.track_round_loss(dagger_current_round)

    # Compute the actually-best logged val NLL across the trajectory. This
    # differs from `best_val` when the very first val_nll was finite but
    # subsequent ones never beat it by `min_delta` — in that case `best_state`
    # stays None but the trajectory still has finite values worth reporting.
    finite_traj = [v for v in val_nll_trajectory if math.isfinite(v)]
    val_nll_best = min(finite_traj) if finite_traj else None
    return {
        "inner_es_stop_epoch": stop_epoch,
        "inner_es_val_nll_best": val_nll_best,
        "inner_es_val_nll_trajectory": val_nll_trajectory,
        "inner_es_fallback": None,
    }


def _should_outer_early_stop(
    disagreement_history: List[float],
    patience: int,
    min_delta: float,
    disagreement_ceiling: float,
) -> bool:
    """Return True if disagreement_rate has plateaued AND is below the ceiling.

    Two-criterion stop, both must hold:

    1. **Rolling-mean plateau.** Compare the mean of the last ``patience``
       eval points against the mean of the ``patience`` eval points
       immediately before that window. If the improvement is less than
       ``min_delta``, the signal has plateaued. Requires at least
       ``2 * patience`` eval points before the first check.

    2. **Absolute disagreement ceiling.** Only allow stopping if the
       current rolling mean is at or below ``disagreement_ceiling``
       (default 0.05 = "agrees with expert at least 95% of the time").
       Prevents stopping on a high-disagreement plateau.
    """
    if patience < 1 or len(disagreement_history) < 2 * patience:
        return False
    window = disagreement_history[-patience:]
    prior = disagreement_history[-2 * patience : -patience]
    current_mean = float(np.mean(window))
    prior_mean = float(np.mean(prior))
    plateau = (prior_mean - current_mean) < min_delta
    if not plateau:
        return False
    if current_mean > disagreement_ceiling:
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
    _seed_everything(config.seed)
    rng = np.random.default_rng(config.seed)

    # Device selection: classical MDPs use CPU (tiny networks, GPU adds overhead).
    # Atari uses the worker's assigned GPU if available, else CPU.
    use_gpu = env_utils.is_atari(config.env_name) and _WORKER_GPU_ID is not None
    # Recorded in the result JSON: "which device did this cell actually run on"
    # should be answerable from the artifacts, not from process archaeology. It
    # is also passed explicitly to every trainer below, because "auto" is not
    # safe here: SB3's get_device("auto") returns cuda whenever CUDA is merely
    # *available*, and a CPU-only cell would then put its 4x2 linear policy on a
    # GPU and train it at batch size 1 -- several times slower than the CPU, and
    # with every worker piling onto the same card.
    # CUDA_VISIBLE_DEVICES is deliberately left alone: CUDA may already be
    # initialized in this process, and changing visibility afterwards makes
    # torch report devices it then refuses to deserialize onto.
    device_used = f"cuda:{_WORKER_GPU_ID}" if use_gpu else "cpu"
    logger.info(
        "%s/%s/seed%s on %s",
        config.algo,
        config.env_name,
        config.seed,
        device_used,
    )

    # Create env
    if env_utils.is_atari(config.env_name):
        from imitation.experiments.ftrl.atari_utils import make_atari_venv

        venv = make_atari_venv(config.env_name, n_envs=1, seed=config.seed)
        # Atari CNN policies expect CHW obs (transposed from HWC).
        # VecTransposeImage handles this so BC/DAgger see the same obs space
        # as the policy.
        from stable_baselines3.common.vec_env import (
            VecTransposeImage,
            is_vecenv_wrapped,
        )

        if not is_vecenv_wrapped(venv, VecTransposeImage):
            venv = VecTransposeImage(venv)
    else:
        venv = env_utils.make_env(config.env_name, n_envs=1, rng=rng)

    # Get expert
    expert_policy = experts.get_or_train_expert(
        config.env_name,
        venv,
        cache_dir=config.expert_cache_dir,
        # A pinned seed and a dedicated stream: the expert is shared across all
        # cells, so it must not depend on which cell missed the cache first,
        # and a cache miss must not perturb this cell's own RNG stream.
        rng=np.random.default_rng(EXPERT_TRAINING_SEED),
        seed=EXPERT_TRAINING_SEED,
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
        "device": device_used,
        "policy_mode": config.policy_mode,
        "config": _config_metadata(config),
        "baselines": baselines,
        "per_round": [],
    }

    if config.algo in ROUND_LOOP_ALGOS:
        result["per_round"] = _run_dagger_variant(
            config,
            venv,
            expert_policy,
            rng,
            baselines,
            device=device_used,
        )
    elif config.algo == "bc":
        result["per_round"] = _run_bc(
            config, venv, expert_policy, rng, baselines, device=device_used
        )
    else:
        raise ValueError(f"Unknown algo: {config.algo}")

    elapsed = time.time() - start_time
    if hasattr(config, "_expert_dataset_source"):
        result["expert_dataset"] = config._expert_dataset_source

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
    exactly samples_per_round expert labels, matching the other baselines'
    observation budgets.
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
    fixed BC, which slices upfront-collected transitions to an exact N.
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


def _single_step_traj(flat, k: int) -> types.Trajectory:
    """Wrap transition ``k`` of a flattened batch as a 1-step pseudo-trajectory.

    ``obs = [obs_k, next_obs_k]``, ``acts = [act_k]``, ``terminal = False`` -- the
    shape ``_save_dagger_demo`` writes and the FTRL trainer's loader expects.
    """
    obs_arr = flat.obs
    next_obs_arr = flat.next_obs
    acts_arr = flat.acts
    infos_arr = getattr(flat, "infos", None)
    # Some Transition implementations expose rews when available.
    rews_arr = getattr(flat, "rews", None)

    pair_obs = np.stack([obs_arr[k], next_obs_arr[k]], axis=0)
    info = None if infos_arr is None else np.array([infos_arr[k]])
    if rews_arr is not None:
        return types.TrajectoryWithRew(
            obs=pair_obs,
            acts=np.array([acts_arr[k]]),
            infos=info,
            rews=np.array([rews_arr[k]], dtype=np.float32),
            terminal=False,
        )
    return types.Trajectory(
        obs=pair_obs,
        acts=np.array([acts_arr[k]]),
        infos=info,
        terminal=False,
    )


def _uniform_round_demos(
    round_dir: pathlib.Path,
    n_target: int,
    rng: np.random.Generator,
    per_trajectory: bool = True,
) -> None:
    """Rewrite ``round_dir`` to hold exactly ``n_target`` retained transitions.

    With ``per_trajectory=True`` (the default) the *trajectory* is the sampling
    unit: ``n_target`` distinct trajectories are drawn uniformly without
    replacement from the round's rollouts, and exactly one state is then drawn
    uniformly at random from each. Each retained sample is therefore drawn by
    picking an episode and then a uniform time step within that episode: an
    independent draw (within the round's fixed-policy batch) from the
    episode-normalized state distribution, rather than one of ``m`` draws from
    a pool in which a single long trajectory can dominate. With fixed episode
    lengths this equals uniform fixed-horizon occupancy sampling; when lengths
    differ it need not equal transition-weighted visitation, since each episode
    gets equal mass regardless of its length.
    ``ExperimentConfig.__post_init__`` raises ``trajectories_per_round`` to
    ``samples_per_round`` so enough trajectories are always available.

    With ``per_trajectory=False`` the previous behavior is used: every
    transition of the round is pooled and ``n_target`` are drawn uniformly from
    that pool, so several retained samples may share a trajectory.

    When the round collected exactly ``n_target`` trajectories there is nothing
    to choose between them, so no draw is taken from ``rng``. At ``m = 1`` with
    one trajectory per round, given identical inputs, this helper's output and
    RNG consumption are bit-for-bit identical to the pooled version. That claim
    covers this helper only; it does not imply that whole earlier campaigns
    reproduce under other protocol changes.

    Each selected transition is written as a 1-step pseudo-trajectory with
    ``terminal=False``, preserving the HuggingFace-dataset format the FTRL
    trainer reads (see ``_save_dagger_demo``).
    """
    demo_paths = sorted(p for p in round_dir.iterdir() if p.name.endswith(".npz"))
    trajs: List[types.Trajectory] = []
    for p in demo_paths:
        trajs.extend(serialize.load(p))

    selected: List[types.Trajectory] = []
    if per_trajectory:
        if len(trajs) < n_target:
            raise RuntimeError(
                f"Round at {round_dir}: collected {len(trajs)} trajectories, "
                f"need {n_target} for one-sample-per-trajectory sampling"
            )
        if len(trajs) == n_target:
            # Nothing to choose: leave the RNG stream untouched so the m = 1
            # path stays bit-compatible with the previous pooled sampling.
            traj_idx = list(range(len(trajs)))
        else:
            traj_idx = sorted(
                int(t) for t in rng.choice(len(trajs), size=n_target, replace=False)
            )
        for t in traj_idx:
            flat = rollout.flatten_trajectories([trajs[int(t)]])
            k = rng.choice(len(flat), size=1, replace=False)
            selected.append(_single_step_traj(flat, int(k[0])))
    else:
        all_flat = rollout.flatten_trajectories(trajs)
        if len(all_flat) < n_target:
            raise RuntimeError(
                f"Round at {round_dir}: collected {len(all_flat)} "
                f"transitions, need {n_target}"
            )
        idx = rng.choice(len(all_flat), size=n_target, replace=False)
        idx.sort()
        selected = [_single_step_traj(all_flat, int(k)) for k in idx]

    # Wipe the round dir and rewrite with the sampled pseudo-trajectories.
    for p in demo_paths:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    for j, traj in enumerate(selected):
        _save_dagger_demo(traj, j, round_dir, rng, prefix="uniform")


def _save_offline_round_demos(
    transitions, round_dir: pathlib.Path, rng: np.random.Generator
) -> None:
    """Write a slice of a fixed offline dataset as this round's demos.

    Used by ``bc_prefix`` and ``bc_pool``, which collect nothing. Writing them in
    exactly the format ``_uniform_round_demos`` produces means the trainer, the
    aggregation and the coverage visualisation treat offline and interactive
    rounds identically, so the only difference between those baselines and
    ``bc_iid`` is which transition arrives at round t.
    """
    for j in range(len(transitions)):
        _save_dagger_demo(
            _single_step_traj(transitions, j), j, round_dir, rng, prefix="offline"
        )


def _save_transitions_as_demos(transitions, scratch_dir, round_num, rng):
    """Persist a batch of transitions as a DAgger-format demo for coverage viz.

    Wraps the transitions' observations into a single synthetic trajectory whose
    ``obs[:-1]`` are exactly the batch's states, saved under
    ``{scratch_dir}/demos/round-{round_num:03d}/`` so ``coverage_data`` can load
    the states and their arrival round uniformly across all algorithms.

    Args:
        transitions: An imitation ``Transitions`` batch (may be empty).
        scratch_dir: Per-run scratch directory ``{output_dir}/scratch/{cell}``.
        round_num: Arrival round to encode in the demo directory name.
        rng: RNG used for the demo filename (matches ``_save_dagger_demo``).
    """
    if len(transitions) == 0:
        return
    obs = np.asarray(transitions.obs)
    next_obs = np.asarray(transitions.next_obs)
    full_obs = np.concatenate([obs, next_obs[-1:]], axis=0)  # T+1 rows
    traj = types.Trajectory(
        obs=full_obs,
        acts=np.asarray(transitions.acts),
        infos=None,
        terminal=True,
    )
    demo_dir = pathlib.Path(scratch_dir) / "demos" / f"round-{round_num:03d}"
    demo_dir.mkdir(parents=True, exist_ok=True)
    _save_dagger_demo(traj, 0, demo_dir, rng)


def _learner_only_beta(round_num: int) -> float:
    """Keep the learner in control from the first collection round."""
    return 0.0


def _run_dagger_variant(
    config: ExperimentConfig,
    venv,
    expert_policy,
    rng: np.random.Generator,
    baselines: Dict[str, float],
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Run FTL, FTRL, BC-iid, BC-prefix or BC-pool on shared mechanics.

    Every algorithm routed here uses the same configurable FTRL trainer, the
    same aggregation of retained round samples and the same on-policy
    evaluation. They differ only in where round t's transitions come from:

    * ``ftl`` / ``ftrl`` use learner control by default. An explicit positive
      beta rampdown enables initial expert mixing.
    * ``bc_iid`` holds beta = 1, so every round is a fresh expert episode, and
      retains one uniformly random state from each.
    * ``bc_prefix`` / ``bc_pool`` collect nothing at all: they replay a fixed
      offline expert dataset, in chronological order or in uniform draws.
    """
    from imitation.algorithms.dagger import ExponentialBetaSchedule, LinearBetaSchedule

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
    if config.algo in ("ftl", "bc_iid") + OFFLINE_BC_ALGOS:
        l2_schedule = ftrl.ConstantL2Schedule(0.0)
    elif config.l2_decay:
        l2_schedule = ftrl.DecayingL2Schedule(config.l2_lambda)
    else:
        l2_schedule = ftrl.ConstantL2Schedule(config.l2_lambda)

    # Per-cell unique tag so parallel sweep workers don't collide on
    # tb/scratch dirs when multiple (lr, sp) cells share env+seed.
    cell_name = config.result_name_override or config.algo
    cell_tag = f"{cell_name}_{config.env_name}_seed{config.seed}"

    # Create custom logger (suppress output)
    custom_logger = imit_logger.configure(
        str(config.output_dir / "tb" / cell_tag),
        format_strs=[],
    )

    # Create BC trainer. Cap batch_size at samples_per_round so round 0 (which
    # has exactly samples_per_round transitions) can form at least one batch.
    # _try_load_demos in dagger.py raises ValueError if len(transitions) <
    # batch_size, which would otherwise prevent any DAgger run with
    # samples_per_round < bc_batch_size (the new --samples-per-round=1 default
    # would always trip this).
    initial_batch_size = max(1, min(config.bc_batch_size, config.samples_per_round))
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        rng=rng,
        policy=policy,
        optimizer_kwargs={"lr": config.learning_rate},
        custom_logger=custom_logger,
        batch_size=initial_batch_size,
        device=device,
    )

    # Create scratch dir for this run. Clear any stale contents from a
    # previous partial run so DAgger doesn't refuse to overwrite its demos.
    scratch_dir = config.output_dir / "scratch" / cell_tag
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
        # beta = 1 keeps the expert in control of collection, which is what
        # makes bc_iid's states draws from d^{pi^E}. The offline baselines never
        # roll out, so their schedule is inert and only set for consistency.
        beta_schedule=(
            (
                LinearBetaSchedule(config.beta_rampdown)
                if config.beta_rampdown > 0
                else _learner_only_beta
            )
            if config.algo in ("ftl", "ftrl")
            else ExponentialBetaSchedule(1.0)
        ),
        custom_logger=custom_logger,
    )

    disagreement_history: List[float] = []
    per_round: List[Dict[str, Any]] = []

    # Round 0: evaluate the fresh (untrained) policy.
    round0_eval = _compute_round_eval(
        bc_trainer.policy,
        expert_policy,
        venv,
        baselines,
    )
    round0_eval["checkpoint"] = _save_policy(config, policy, 0)
    disagreement_history.append(round0_eval["disagreement_rate"])
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

    # The offline BC baselines replay a fixed dataset instead of collecting.
    # With ``subsample_strategy='prefix'`` this is byte-for-byte the artifact
    # fixed BC uses, so bc_prefix at round t trains on exactly fixed BC's first
    # t transitions.
    offline_data = None
    offline_collection_steps = 0
    if config.algo in OFFLINE_BC_ALGOS:
        offline_data = _shared_expert_data(config, venv, expert_policy)
        budget = config.n_rounds * config.samples_per_round
        if len(offline_data) < budget:
            raise RuntimeError(
                f"{config.algo}: offline dataset has {len(offline_data)} "
                f"transitions but the run needs {budget}"
            )
        source = getattr(config, "_expert_dataset_source", None) or {}
        offline_collection_steps = int(source.get("collection_steps", 0) or 0)

    cum_obs = 0
    collection_steps = 0
    stopped_early = False
    for round_num in range(1, config.n_rounds + 1):
        if offline_data is not None:
            # --- Replay this round's slice of the fixed offline dataset ---
            lo = (round_num - 1) * config.samples_per_round
            round_dir = trainer._demo_dir_path_for_round()
            round_dir.mkdir(parents=True, exist_ok=True)
            _save_offline_round_demos(
                offline_data[lo : lo + config.samples_per_round], round_dir, rng
            )
            collected = []
            # The whole dataset was collected up front, so the expert-interaction
            # cost is a constant, not something that grows with the round.
            collection_steps = offline_collection_steps
        else:
            # --- Collect expert-labeled rollouts for this round ---
            collector = trainer.create_trajectory_collector()
            sample_until = rollout.make_sample_until(
                min_timesteps=config.samples_per_round,
                min_episodes=config.trajectories_per_round,
            )
            collected = rollout.generate_trajectories(
                policy=expert_policy,
                venv=collector,
                sample_until=sample_until,
                deterministic_policy=True,
                rng=collector.rng,
            )
            collected_steps = sum(len(traj) for traj in collected)
            collection_steps += collected_steps

            round_dir = trainer._demo_dir_path_for_round()
            if config.subsample_strategy == "uniform":
                _uniform_round_demos(round_dir, config.samples_per_round, rng)
            else:
                _truncate_round_demos(round_dir, config.samples_per_round, rng)

        # --- Train on all accumulated demos ---
        inner_log = _inner_train(trainer, config, round_num=round_num, is_dagger=True)
        cum_obs += config.samples_per_round

        # --- Evaluate the policy at the reported training budget ---
        is_first = round_num == 1
        is_interval = round_num % config.eval_interval == 0
        is_final = round_num == config.n_rounds
        should_eval = is_first or is_interval or is_final

        eval_data: Optional[Dict[str, Any]] = None
        if should_eval:
            eval_data = _compute_round_eval(
                bc_trainer.policy,
                expert_policy,
                venv,
                baselines,
            )
            disagreement_history.append(eval_data["disagreement_rate"])

            if config.outer_early_stop and _should_outer_early_stop(
                disagreement_history,
                config.outer_early_stop_patience,
                config.outer_early_stop_min_delta,
                disagreement_ceiling=config.outer_early_stop_disagreement_ceiling,
            ):
                stopped_early = True
                logger.info(
                    f"{config.algo}/{config.env_name}/seed{config.seed}: "
                    f"early stop at round {round_num} "
                    f"(disagreement plateau over "
                    f"{config.outer_early_stop_patience} eval points)"
                )

        metrics = list(trainer.get_metrics())
        m = metrics[-1]
        round_data: Dict[str, Any] = {
            "round": round_num,
            "n_observations": cum_obs,
            "collection_steps": collection_steps,
            "collection_expert_queries": collection_steps,
            "trajectories_collected_this_round": len(collected),
            "train_cross_entropy": round(m.cross_entropy, 6),
            "l2_norm": round(m.l2_norm, 6),
            "total_loss": round(m.total_loss, 6),
            "rollout_cross_entropy": None,
            "expert_rollout_cross_entropy": None,
            "normalized_return": None,
            "disagreement_rate": None,
            "d_eval_size": None,
            **inner_log,
        }

        if eval_data is not None:
            round_data.update(eval_data)
            round_data["checkpoint"] = _save_policy(
                config, bc_trainer.policy, round_num
            )

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


def _save_policy(config, policy, round_num):
    """Retain evaluated policies so recovery tests do not require retraining."""
    name = config.result_name_override or config.algo
    path = (
        config.output_dir
        / "checkpoints"
        / f"{name}_{config.env_name.replace('/', '_')}_seed{config.seed}"
        / f"round-{round_num:05d}.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    policy.save(path)
    return str(path)


def _shared_expert_data(config, venv, expert_policy):
    """Cache fixed BC data with its sampling strategy and expert provenance."""
    from imitation.experiments.ftrl import expert_dataset

    budget = config.n_rounds * config.samples_per_round
    metadata = dict(
        version=5,
        env=config.env_name,
        seed=config.seed,
        budget=budget,
        strategy=config.subsample_strategy,
        expert=expert_dataset.policy_digest(expert_policy),
    )

    canonical_pool = None
    pool_source = None
    if config.algo == "bc_pool":
        pool_config = dataclasses.replace(
            config, algo="bc", subsample_strategy="prefix"
        )
        canonical_pool = _shared_expert_data(pool_config, venv, expert_policy)
        pool_source = pool_config._expert_dataset_source
        metadata["pool_sha256"] = pool_source["sha256"]

    def collect():
        data_rng = np.random.default_rng(config.seed)
        if canonical_pool is not None:
            selected = _collect_and_subsample_transitions(
                canonical_pool, budget, "uniform", data_rng
            )
            return selected, {
                "collection_steps": pool_source["collection_steps"],
                "trajectories_collected": pool_source["trajectories_collected"],
            }
        # A fresh environment makes collection independent of baseline evaluation
        # and Atari life-reset wrapper state in the worker's training environment.
        if env_utils.is_atari(config.env_name):
            from stable_baselines3.common.vec_env import VecTransposeImage
            from imitation.experiments.ftrl.atari_utils import make_atari_venv

            data_env = VecTransposeImage(
                make_atari_venv(config.env_name, n_envs=1, seed=config.seed)
            )
        else:
            data_env = env_utils.make_env(config.env_name, n_envs=1, rng=data_rng)
        try:
            trajectories = rollout.generate_trajectories(
                policy=expert_policy,
                venv=data_env,
                sample_until=rollout.make_sample_until(
                    min_timesteps=budget, min_episodes=1
                ),
                deterministic_policy=True,
                rng=data_rng,
                shuffle=config.subsample_strategy != "prefix",
            )
        finally:
            data_env.close()
        selected = _collect_and_subsample_transitions(
            rollout.flatten_trajectories(list(trajectories)),
            budget,
            config.subsample_strategy,
            data_rng,
        )
        return selected, {
            "collection_steps": sum(len(traj) for traj in trajectories),
            "trajectories_collected": len(trajectories),
        }

    data, source = expert_dataset.load_or_collect(
        config.output_dir / "expert_datasets",
        metadata,
        collect,
    )
    config._expert_dataset_source = source
    # Collection cache hits and misses must leave evaluation in the same RNG state.
    venv.seed(config.seed + 100000)
    return data


def _run_bc(
    config: ExperimentConfig,
    venv,
    expert_policy,
    rng: np.random.Generator,
    baselines: Dict[str, float],
    device: str = "cpu",
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

    all_transitions = _shared_expert_data(config, venv, expert_policy)

    bc_scratch = (
        config.output_dir / "scratch" / f"bc_{config.env_name}_seed{config.seed}"
    )
    if bc_scratch.exists():
        shutil.rmtree(bc_scratch)
    _save_transitions_as_demos(all_transitions, bc_scratch, 0, rng)

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
        batch_size=min(config.bc_batch_size, len(all_transitions)),
        optimizer_kwargs={"lr": config.learning_rate},
        custom_logger=custom_logger,
        device=device,
    )
    inner_log = _inner_train(bc_trainer, config, round_num=0, is_dagger=False)

    eval_data = _compute_round_eval(
        bc_trainer.policy,
        expert_policy,
        venv,
        baselines,
    )

    eval_data["checkpoint"] = _save_policy(config, bc_trainer.policy, config.n_rounds)
    l2_norms = [th.sum(th.square(w)).item() for w in bc_trainer.policy.parameters()]
    l2_norm = sum(l2_norms) / 2

    return [
        {
            "round": 0,
            "n_observations": total_timesteps,
            "train_cross_entropy": None,
            "l2_norm": round(l2_norm, 6),
            "total_loss": None,
            **inner_log,
            **eval_data,
        }
    ]


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
        return cached_cfg == _config_metadata(config)
    except (json.JSONDecodeError, OSError):
        return False


_WORKER_GPU_ID: Optional[int] = None
# How long a pool worker waits for its GPU assignment before falling back to CPU.
_GPU_HANDOUT_TIMEOUT_S = 30


def _worker_init(gpu_queue):
    """Pool initializer: assign each worker to a GPU from the queue.

    The parent fills the queue before creating the pool, but
    ``multiprocessing.Queue.put`` only hands the item to a feeder thread, so a
    worker that starts quickly can find it still empty. ``get_nowait`` then
    silently left that worker on CPU for the whole sweep -- on Atari that is the
    difference between using every visible card and using only some of them, and
    it is exactly the "silently misplaced worker" failure the GPU mask in run.sh
    is trying to avoid. Block briefly instead, and say what each worker got.
    """
    global _WORKER_GPU_ID
    try:
        gpu_id = gpu_queue.get(timeout=_GPU_HANDOUT_TIMEOUT_S)
    except Exception:
        gpu_id = None
        logger.warning(
            "worker %s got no GPU assignment within %ss; running on CPU",
            os.getpid(),
            _GPU_HANDOUT_TIMEOUT_S,
        )
    if gpu_id is not None:
        _WORKER_GPU_ID = gpu_id
        # Best-effort CUDA device assignment. The index counts over the visible
        # devices, so it follows CUDA_VISIBLE_DEVICES rather than the physical
        # card number.
        try:
            import torch as _th

            if _th.cuda.is_available():
                _th.cuda.set_device(gpu_id)
                logger.info("worker %s -> cuda:%s", os.getpid(), gpu_id)
            else:
                logger.warning(
                    "worker %s was assigned GPU %s but CUDA is unavailable",
                    os.getpid(),
                    gpu_id,
                )
        except Exception:
            logger.warning(
                "worker %s could not select GPU %s", os.getpid(), gpu_id, exc_info=True
            )


def _run_single_wrapper(args):
    """Wrapper for multiprocessing.Pool.map (unpacks config)."""
    config = args
    # Resume / --force-rerun filtering is handled in main() before configs are
    # dispatched to the pool, so the worker just runs unconditionally. An
    # earlier per-worker _is_already_done short-circuit here defeated
    # --force-rerun by returning the cached JSON inside the worker.
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
                        trajectories_per_round=getattr(
                            args, "trajectories_per_round", 1
                        ),
                        l2_lambda=args.l2_lambda,
                        l2_decay=args.l2_decay,
                        warm_start=args.warm_start,
                        beta_rampdown=args.beta_rampdown,
                        bc_n_epochs=args.bc_n_epochs,
                        eval_interval=args.eval_interval,
                        output_dir=pathlib.Path(args.output_dir),
                        expert_cache_dir=pathlib.Path(args.expert_cache_dir),
                        learning_rate=args.learning_rate,
                        subsample_strategy=getattr(args, "subsample_strategy", None),
                        outer_early_stop=args.outer_early_stop,
                        outer_early_stop_patience=args.outer_early_stop_patience,
                        outer_early_stop_min_delta=args.outer_early_stop_min_delta,
                        outer_early_stop_disagreement_ceiling=args.outer_early_stop_disagreement_ceiling,
                        inner_early_stop=args.inner_early_stop,
                        inner_early_stop_patience=args.inner_early_stop_patience,
                        inner_early_stop_min_delta=args.inner_early_stop_min_delta,
                        inner_early_stop_val_frac=args.inner_early_stop_val_frac,
                        inner_early_stop_min_val_size=args.inner_early_stop_min_val_size,
                        inner_early_stop_min_epochs=args.inner_early_stop_min_epochs,
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
        default=DEFAULT_ALGOS,
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
        default=1,
        help=(
            "Retained samples per round (default: 1). Interactive algorithms "
            "keep one state per distinct trajectory; offline BC baselines use "
            "their selected fixed-data sampling rule."
        ),
    )
    # Python 3.8 doesn't have argparse.BooleanOptionalAction, so we pair
    # store_true / store_false on a shared dest.
    parser.add_argument(
        "--outer-early-stop",
        dest="outer_early_stop",
        action="store_true",
        default=True,
        help=(
            "Enable outer-loop early stopping on disagreement_rate plateau "
            "(default: True)."
        ),
    )
    parser.add_argument(
        "--no-outer-early-stop",
        dest="outer_early_stop",
        action="store_false",
        help="Disable outer-loop early stopping.",
    )
    parser.add_argument(
        "--outer-early-stop-patience",
        type=int,
        default=5,
        help=(
            "Stop training when the tracked signal has not improved by "
            "--outer-early-stop-min-delta over this many consecutive eval points."
        ),
    )
    parser.add_argument(
        "--outer-early-stop-min-delta",
        type=float,
        default=0.005,
        help="Min improvement to count as progress for outer ES (default 0.005).",
    )
    parser.add_argument(
        "--outer-early-stop-disagreement-ceiling",
        type=float,
        default=0.05,
        help=(
            "Outer ES only fires when the rolling-mean disagreement_rate is "
            "<= this ceiling (default 0.05 = '<=5%% disagreement')."
        ),
    )
    parser.add_argument(
        "--inner-early-stop",
        dest="inner_early_stop",
        action="store_true",
        default=True,
        help="Enable val-split early stopping inside BC train (default: True).",
    )
    parser.add_argument(
        "--no-inner-early-stop",
        dest="inner_early_stop",
        action="store_false",
        help="Disable inner-loop early stopping (use full bc_n_epochs).",
    )
    parser.add_argument(
        "--inner-early-stop-patience",
        type=int,
        default=5,
        help="Epochs without val-NLL improvement before stopping (default 5).",
    )
    parser.add_argument(
        "--inner-early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Min absolute val-NLL improvement to count as progress (default 1e-4).",
    )
    parser.add_argument(
        "--inner-early-stop-val-frac",
        type=float,
        default=0.1,
        help="Fraction of D^t held out for val (default 0.1).",
    )
    parser.add_argument(
        "--inner-early-stop-min-val-size",
        type=int,
        default=32,
        help="Below this val-set size, fall back to fixed bc_n_epochs (default 32).",
    )
    parser.add_argument(
        "--inner-early-stop-min-epochs",
        type=int,
        default=3,
        help="Don't trigger inner ES before this epoch (default 3).",
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
        default=False,
        help="Keep policy weights and optimizer state between rounds",
    )
    parser.add_argument(
        "--no-warm-start",
        dest="warm_start",
        action="store_false",
        help="Reinitialize trainable params and optimizer state each round (default)",
    )
    parser.add_argument(
        "--beta-rampdown",
        type=int,
        default=0,
        help=(
            "Expert-mixing rampdown rounds; 0 keeps learner control throughout "
            "(default)"
        ),
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
    parser.add_argument(
        "--trajectories-per-round",
        "--traj-per-round",
        type=int,
        default=1,
        help=(
            "Minimum complete trajectories in each interactive round (default 1). "
            "Raised to --samples-per-round automatically so each retained sample "
            "comes from its own trajectory."
        ),
    )
    parser.add_argument(
        "--subsample-strategy",
        choices=["uniform", "prefix"],
        default=None,
        help=(
            "Sampling override. Defaults per algorithm: bc and bc_prefix use "
            "prefix; ftl, ftrl, bc_iid and bc_pool use uniform. bc_iid/bc_pool "
            "require uniform and bc_prefix requires prefix."
        ),
    )
    args = parser.parse_args()
    if args.trajectories_per_round < 1:
        parser.error("--trajectories-per-round must be positive")
    if args.samples_per_round < 1:
        parser.error("--samples-per-round must be positive")
    if args.subsample_strategy == "prefix" and set(args.algos) & {
        "bc_iid",
        "bc_pool",
    }:
        parser.error("bc_iid and bc_pool require uniform sampling")
    if args.subsample_strategy == "uniform" and "bc_prefix" in args.algos:
        parser.error("bc_prefix requires prefix sampling")
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
                    logger.info(
                        f"Removing stale result: {f} (seed {file_seed} not in current run)"
                    )
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
            from stable_baselines3.common.vec_env import VecTransposeImage

            from imitation.experiments.ftrl.atari_utils import make_atari_venv

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
        raise SystemExit(1)


if __name__ == "__main__":
    main()
