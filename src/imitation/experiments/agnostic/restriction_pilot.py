"""Runner for the approved CartPole learner-only restriction pilot.

Protocol ``/2`` (the stronger-mask pilot) for runs and the audit; the data
job and its schema are unchanged from protocol ``/1``. Protocol ``/1``, the
x-only pilot with prefix-fitted BC and online minibatch 1, is historical
(source ``2896ffe``) and is not reproduced by this module.

Three commands, each writing one fresh output directory:

``audit``
    Paired-state x-grid audit of `PILOT_RESTRICTION` with the qualified
    expert on the diagnostic reset seed `AUDIT_SEED` (see
    `restriction.audit_cartpole_position`). It reuses the fixed x-grid
    witnesses, so it shows only that expert labels conflict under the mask.
    Its labels never reach training.
``data``
    Acquires, once, the expert data that full and restricted runs share: the
    fixed-BC chronological pool (complete expert episodes until at least
    ``budget`` transitions, the first ``budget`` kept in order, the overshoot
    recorded) and the BC-iid stream (``budget`` independent complete expert
    episodes, one uniformly selected pre-action state from each). Both are
    different acquisition procedures with dedicated reset and selection
    streams, independent of each other and of evaluation. It also measures
    the normalization baselines exactly as ``env_baselines.compute_baselines``
    does (500 deterministic expert episodes, 500 uniform random episodes), on
    a dedicated environment. Physical acquisition is charged to this job.
``run``
    One (method, restriction) run on the pilot seed, restriction one of
    `RUN_RESTRICTIONS`. The learner is the previous pipeline's linear policy
    (expert clone, frozen features, reinitialized ``action_net``) built by
    `restriction.make_linear_policy`, so the restriction acts before the
    frozen features on every path while the expert keeps reading full
    observations. FTL (beta 0) and BC-iid use
    ``run_experiment._run_dagger_variant`` unchanged except for its explicit
    hooks and ``grow_batch_with_data``: FTL collects with learner control as
    before; BC-iid replays the shared stream in order, one state per round,
    through the same trainer. Fixed BC is the offline baseline of ``_run_bc``:
    one cold fit on the whole pool with ``run_experiment._fit_fixed_bc`` and
    one evaluation, a frozen policy that is a flat reference, not a curve.
    BC-prefix (`PREFIX_METHOD`, only under a spec that opts in) is the
    original ``bc_prefix`` baseline: the BC-iid trainer replaying the pool in
    chronological order, a cold refit on the first t pool labels at round t.
    Evaluation is ``_compute_round_eval`` (100 deterministic episodes,
    normalized return, learner and expert rollout cross-entropy,
    disagreement, raw episode returns); evaluated policies are checkpointed
    with their restriction. No mixture is computed. Data from another source
    version are accepted only when ``expected_data_sha256`` pins the exact
    bytes of the data record (`BATCH_RULE`, `FIXED_BC_RULE`).

Differences from the previous pipeline, all deliberate: expert data for fixed
BC and BC-iid come from the shared data job instead of the per-run
environment, so full and restricted runs see identical data (verified by
hash); baselines are measured once per data job on a dedicated environment
whose random-action sampler is seeded; the expert comes from a verified
qualified preparation rather than the expert cache; online methods train with
the batch rule fixed BC uses.

Environment and protocol: every job takes a keyword ``spec`` (`RunnerSpec`).
The default `CARTPOLE_V2` is the CartPole pilot above, unchanged. An
environment-generic spec (built by `task_pilot.runner_spec`) names another
protocol, binds a verified `tasks.Task` description into every record and
data check, loads experts with `task_experts.load_qualified_expert` and may
normalize with a pinned reference measured by another data job. A spec with
``paired_cold_fits`` runs every method under
``run_experiment.PAIRED_COLD_FITS_PROTOCOL`` (`PAIRED_COLD_FITS_RULE`); its
records add ``paired_cold_fits`` to the run config and ``fit_*``,
``eval_seed`` and ``evaluation_episode_seeds`` fields to the rows. Specs
without it keep their configs; their rows gain only the optional episode
evidence the evaluation reports.

Normalization reference: every environment, CliffWalking included,
normalizes with the uniform random return its data job measured
(`measure_baselines`, overrides only from
``env_baselines.normalization_reference``). Generic run configs describe it
as ``normalization_reference`` (`UNIFORM_RANDOM_REFERENCE_RULE`). Data and
reference records are read as recorded, never rewritten: baselines without
reference metadata hold the uniform random return, and baselines recording
another lower reference are refused (`_uniform_random_baselines`).

Capacity family: a spec whose
``learner`` is a `capacity_policy.CapacityRunner` (built by
`task_pilot.capacity_runner_spec`) replaces the linear head by a fresh
one-hidden student on the teacher's frozen encoder, on full observations only;
its teacher must be an explicit-architecture one-hidden preparation, and the
spec, teacher, encoder and actual parameter counts are bound into every data
record, run record and verified checkpoint.

Safety: every command verifies its inputs before claiming a fresh, empty
output directory, and refuses anything else without reading it. The
effective deadline is the earliest of the requested deadline, start plus the
per-job limit, and `classical.HARD_CAP`. It is checked before every data
acquisition step, before every fit or round, and before marking a result
complete; a round or an evaluation already started finishes first.
``result.json`` is replaced atomically; ``partial`` and ``failed`` records
keep everything completed so far. Unknown quantities are null.
"""

import argparse
import dataclasses
import datetime
import functools
import hashlib
import json
import logging
import math
import os
import pathlib
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.vec_env import VecEnvWrapper

from imitation.data import rollout, serialize, types
from imitation.experiments.agnostic import (
    capacity_policy,
    classical,
    pilot,
    restriction,
    rollouts,
    task_experts,
    tasks,
)
from imitation.experiments.ftrl import (
    env_baselines,
    env_utils,
    eval_utils,
    run_experiment,
)

logger = logging.getLogger(__name__)

PROTOCOL = "agnostic-cartpole-restriction-pilot/2"
AUDIT_SCHEMA = PROTOCOL + "/audit"
RUN_SCHEMA = PROTOCOL + "/run"
# The data job's producer semantics are unchanged, so it keeps protocol /1.
DATA_PROTOCOL = "agnostic-cartpole-restriction-pilot/1"
DATA_SCHEMA = DATA_PROTOCOL + "/data"

ENV_NAME = "CartPole-v1"
EPISODE_CAP = 500
PILOT_SEED = 300
AUDIT_SEED = 301
METHODS = ("ftl", "bc", "bc_iid")
# Opt-in only: a spec must list it in ``methods`` (see `RunnerSpec`).
PREFIX_METHOD = "bc_prefix"
PILOT_RESTRICTION = "cart_position_angular_velocity_zero"
# cart_position_zero stays registered for historical checkpoints only.
RUN_RESTRICTIONS = ("identity", PILOT_RESTRICTION)
DEFAULT_BUDGET = 1000
EVAL_INTERVAL = 10
BC_EPOCHS = 20
LEARNING_RATE = 1e-3
BASELINE_EPISODES = 500
MAX_JOB_LIMIT_SECONDS = 7200.0

RESULT_FILE = classical.RESULT_FILE
POOL_FILE = "fixed_bc_pool.npz"
STREAM_FILE = "bc_iid_stream.npz"
DEVICE = "cpu"

# SeedSequence spawn keys of the data job's dedicated streams.
_POOL_RESETS, _STREAM_RESETS, _STREAM_SELECTION, _BASELINE_ENV, _BASELINE_ACTIONS = (
    1,
    2,
    3,
    4,
    5,
)
_SEED_HIGH = 2**31 - 1

_SOURCE_FILES = (
    "experiments/agnostic/restriction.py",
    "experiments/agnostic/restriction_pilot.py",
    "experiments/agnostic/classical.py",
    "experiments/agnostic/pilot.py",
    "experiments/agnostic/rollouts.py",
    "experiments/ftrl/run_experiment.py",
    "experiments/ftrl/policy_utils.py",
    "experiments/ftrl/eval_utils.py",
    "experiments/ftrl/env_utils.py",
    "experiments/ftrl/env_baselines.py",
    "algorithms/ftrl.py",
    "algorithms/dagger.py",
    "algorithms/bc.py",
    "data/rollout.py",
    "util/util.py",
)
_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True)
class RunnerSpec:
    """Environment and protocol identity of one restriction runner.

    ``task`` is None only for the CartPole pilot protocol, whose records and
    checks stay exactly as they were. Otherwise it is a registered
    `tasks.Task`, and records carry its verified description. ``methods``
    are the methods its runs accept; `PREFIX_METHOD` is never a default.
    ``paired_cold_fits`` opts every run in to
    ``run_experiment.PAIRED_COLD_FITS_PROTOCOL`` (`PAIRED_COLD_FITS_RULE`).
    ``calibration_candidates`` are extra restrictions, also listed in
    ``run_restrictions``, that are not part of the task's ladder; their
    descriptions are bound into every run record of the spec.
    ``learner`` is
    None for the linear-head learner; a `capacity_policy.CapacityRunner`
    selects the one-hidden capacity student (generic tasks only).
    """

    env_name: str
    episode_cap: int
    protocol: str
    data_protocol: str
    run_restrictions: Tuple[str, ...]
    source_files: Tuple[str, ...]
    audit_seed: int
    max_job_limit_seconds: float
    task: Optional[Any] = None
    methods: Tuple[str, ...] = METHODS
    paired_cold_fits: bool = False
    calibration_candidates: Tuple[str, ...] = ()
    learner: Optional[Any] = None

    @property
    def audit_schema(self) -> str:
        return self.protocol + "/audit"

    @property
    def run_schema(self) -> str:
        return self.protocol + "/run"

    @property
    def data_schema(self) -> str:
        return self.data_protocol + "/data"


CARTPOLE_V2 = RunnerSpec(
    env_name=ENV_NAME,
    episode_cap=EPISODE_CAP,
    protocol=PROTOCOL,
    data_protocol=DATA_PROTOCOL,
    run_restrictions=RUN_RESTRICTIONS,
    source_files=_SOURCE_FILES,
    audit_seed=AUDIT_SEED,
    max_job_limit_seconds=MAX_JOB_LIMIT_SECONDS,
)

COLLECTION_RULE = (
    "FTL: one expert predict call and one learner predict call per collected "
    "step (beta 0, one environment); fixed BC and BC-iid collect nothing in "
    "the run."
)
SNAPSHOT_NOTE = (
    "result.json is replaced atomically while the job runs. A record whose "
    "snapshot is not final (status running) is the last snapshot of a job "
    "that may have been killed; work after it is unknown, not zero."
)
IN_FLIGHT_NOTE = (
    "Observed env steps and resets are counted when a step or reset returns. "
    "Completed-record subtotals cover only rounds, fits and evaluations that "
    "finished. Env steps of unfinished work are observed minus completed. "
    "Predict calls of unfinished work are not observed, so they and the query "
    "totals are null unless the job completed with nothing unattributed."
)
EVALUATION_RULE = (
    "Each evaluation step makes one learner and one expert predict call; "
    "the expert's rollout cross-entropy is a forward pass over the same states."
)
POOL_RULE = (
    "complete expert episodes until >= budget transitions; first budget kept in "
    "order"
)
STREAM_RULE = (
    "budget independent complete expert episodes; one uniformly selected "
    "pre-action state per episode"
)
BATCH_RULE = (
    "Online methods: before each round's data are loaded, batch and minibatch "
    "are min(bc_batch_size, accumulated labels). Fixed BC: min(bc_batch_size, "
    "dataset size). When the held-out split runs and leaves fewer training "
    "examples than that, the batch shrinks to the training split. Loaders "
    "shuffle and drop the incomplete final batch every epoch (drop_last). "
    "Records give the loader actually trained on (train_batch_size, "
    "train_examples, train_batches_per_epoch, train_drop_last)."
)
FIXED_BC_RULE = (
    "Offline fixed BC: one cold fit on all n_rounds pool labels with "
    "run_experiment._fit_fixed_bc (torch reseeded first) and one evaluation. "
    "The frozen policy is a flat reference across the online label axis, not "
    "a function of it; it never fits prefixes."
)
BC_PREFIX_RULE = (
    "BC-prefix: the original run_experiment bc_prefix baseline. Round t "
    "replays pool transition t through the BC-iid trainer, so the round-t fit "
    "is a cold fit (warm_start false) on the first t chronological pool "
    "labels; evaluated like BC-iid. It collects nothing and makes no expert "
    "label queries; fixed BC is unchanged."
)
PAIRED_COLD_FITS_RULE = (
    "Paired cold fits (run_experiment.PAIRED_COLD_FITS_PROTOCOL): every fit is "
    "a cold fit on its N retained labels; head initialization, Adam reset, "
    "loader shuffle, NumPy trainer stream and held-out split come from "
    "dedicated streams keyed by (seed, N) inside a scoped torch RNG, and the "
    "batch is min(bc_batch_size, N). Fixed BC on N pool labels and BC-prefix "
    "at round N therefore fit identically. Every evaluation at budget N resets "
    "each of its 100 episodes explicitly, episode i with seed (s + i) mod 2**32 "
    "where s is keyed by (seed, N), shared by all methods "
    "(evaluation_episode_seeds); auto-reset states are discarded. Episode i "
    "therefore starts from the same initial state for every method "
    "(initial_state_digests); trajectories still differ once actions differ."
)
EVALUATION_RESET_RULE = (
    "Evaluation resets come from the rollout's own reset_accounting: explicit "
    "resets pass through the run's environment wrapper and are counted there "
    "too; VecEnv auto-resets happen inside the wrapped environment and are "
    "not. Null when an evaluation did not report them."
)
UNIFORM_RANDOM_REFERENCE = "uniform_random"
UNIFORM_RANDOM_REFERENCE_RULE = (
    "Every environment: the lower normalization reference is the uniform "
    "random return measured by the data job's baseline episodes; normalized "
    "return (R - R_random) / (R_expert - R_random), unclipped."
)
_SHA256_HEX = "0123456789abcdef"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def pairs_sha256(obs: np.ndarray, acts: np.ndarray) -> str:
    """Digest of supervised (observation, expert action) pairs, in order.

    Observations are hashed as float32 and actions as int64 with their
    shapes, so the digest identifies training data independently of storage.
    Same digest as the ``fit_*_pairs_sha256`` of paired fits.
    """
    return run_experiment.pairs_sha256(obs, acts)


_ARRAYS = ("obs", "acts", "next_obs", "dones", "episode_index", "step_index")


@dataclasses.dataclass
class ExpertData:
    """Ordered full-state expert transitions with their episode positions."""

    obs: np.ndarray
    acts: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    episode_index: np.ndarray
    step_index: np.ndarray

    def __len__(self) -> int:
        return int(len(self.acts))

    def pairs_sha256(self, n: Optional[int] = None) -> str:
        """`pairs_sha256` of the first ``n`` transitions (all by default)."""
        n = len(self) if n is None else int(n)
        if not 0 < n <= len(self):
            raise ValueError(f"Prefix {n} outside 1..{len(self)}")
        return pairs_sha256(self.obs[:n], self.acts[:n])

    def transitions(self, n: Optional[int] = None) -> types.Transitions:
        """The first ``n`` transitions as imitation ``Transitions``."""
        n = len(self) if n is None else int(n)
        if not 0 < n <= len(self):
            raise ValueError(f"Prefix {n} outside 1..{len(self)}")
        return types.Transitions(
            obs=self.obs[:n].copy(),
            acts=self.acts[:n].copy(),
            next_obs=self.next_obs[:n].copy(),
            dones=self.dones[:n].copy(),
            infos=np.array([{} for _ in range(n)]),
        )

    @classmethod
    def from_rows(cls, rows: Sequence[Tuple]) -> "ExpertData":
        """Build from ``(episode, step, obs, act, next_obs, done)`` rows."""
        return cls(
            obs=np.array([r[2] for r in rows], dtype=np.float32),
            acts=np.array([r[3] for r in rows], dtype=np.int64),
            next_obs=np.array([r[4] for r in rows], dtype=np.float32),
            dones=np.array([r[5] for r in rows], dtype=bool),
            episode_index=np.array([r[0] for r in rows], dtype=np.int64),
            step_index=np.array([r[1] for r in rows], dtype=np.int64),
        )

    def save(self, path: pathlib.Path) -> None:
        """Write numeric arrays only; never replaces an existing file."""
        path = pathlib.Path(path)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(f, **{k: getattr(self, k) for k in _ARRAYS})
                f.flush()
                os.fsync(f.fileno())
            os.link(tmp, path)
        finally:
            os.unlink(tmp)

    @classmethod
    def load(cls, path: pathlib.Path) -> "ExpertData":
        with np.load(path, allow_pickle=False) as saved:
            return cls(**{k: saved[k] for k in _ARRAYS})


def _seed_stream(seed: int, key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(key,)))


def _no_check(what: str) -> None:
    del what


def _new_counters() -> Dict[str, Any]:
    return {
        "episodes": 0,
        "env_steps": 0,
        "env_resets": 0,
        "expert_predict_calls": 0,
        "expert_action_entries": 0,
    }


def _expert_episode(venv, expert, reset_seed, counters, check) -> List[Tuple]:
    """One complete deterministic expert episode on a one-env VecEnv."""
    check("expert data reset")
    venv.seed(int(reset_seed))
    obs = venv.reset()
    counters["env_resets"] += 1
    rows = []
    while True:
        check("expert data query")
        action, _ = expert.predict(obs, deterministic=True)
        counters["expert_predict_calls"] += 1
        counters["expert_action_entries"] += int(np.size(action))
        check("expert data step")
        next_obs, _, dones, infos = venv.step(action)
        counters["env_steps"] += 1
        done = bool(dones[0])
        final = infos[0]["terminal_observation"] if done else next_obs[0]
        rows.append((obs[0].copy(), int(action[0]), np.array(final), done))
        if done:
            counters["episodes"] += 1
            return rows
        obs = next_obs


def collect_chronological_pool(
    expert,
    budget: int,
    seed: int,
    *,
    check: Callable[[str], None] = _no_check,
    counters: Optional[Dict[str, Any]] = None,
    env_name: str = ENV_NAME,
) -> Tuple[ExpertData, Dict[str, Any]]:
    """Fixed-BC pool: complete expert episodes until at least ``budget`` steps.

    Episodes run in order, each reset with the next seed of the dedicated pool
    stream; the first ``budget`` transitions are kept in chronological order.

    Returns:
        The retained data and the physical collection statistics.
    """
    budget = int(budget)
    if budget <= 0:
        raise ValueError("budget must be positive")
    counters = _new_counters() if counters is None else counters
    start = time.monotonic()
    seeds = _seed_stream(seed, _POOL_RESETS)
    venv = env_utils.make_env(env_name, 1, _seed_stream(seed, _POOL_RESETS))
    rows: List[Tuple] = []
    lengths: List[int] = []
    reset_seeds: List[int] = []
    try:
        while len(rows) < budget:
            reset_seeds.append(int(seeds.integers(0, _SEED_HIGH)))
            episode = _expert_episode(venv, expert, reset_seeds[-1], counters, check)
            rows += [(len(lengths), t, *row) for t, row in enumerate(episode)]
            lengths.append(len(episode))
    finally:
        venv.close()
    stats = {
        **counters,
        "retained_labels": budget,
        "overshoot_transitions": len(rows) - budget,
        "final_episode_length": lengths[-1],
        "episode_lengths": lengths,
        "episode_reset_seeds": reset_seeds,
        "reset_seed_stream": f"SeedSequence(seed, spawn_key=({_POOL_RESETS},))",
        "elapsed_seconds": time.monotonic() - start,
    }
    return ExpertData.from_rows(rows[:budget]), stats


def collect_iid_stream(
    expert,
    n_episodes: int,
    seed: int,
    *,
    check: Callable[[str], None] = _no_check,
    counters: Optional[Dict[str, Any]] = None,
    env_name: str = ENV_NAME,
) -> Tuple[ExpertData, Dict[str, Any]]:
    """BC-iid stream: one uniformly selected state per independent episode.

    Episode ``i`` is reset with the ``i``-th seed of the dedicated stream and
    keeps time step ``floor(u_i * length)`` with ``u_i`` from a separate
    selection stream.

    Returns:
        The retained data (``step_index`` is the selected step) and the
        physical collection statistics.
    """
    n_episodes = int(n_episodes)
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    counters = _new_counters() if counters is None else counters
    start = time.monotonic()
    seeds = _seed_stream(seed, _STREAM_RESETS)
    selection = _seed_stream(seed, _STREAM_SELECTION)
    venv = env_utils.make_env(env_name, 1, _seed_stream(seed, _STREAM_RESETS))
    rows: List[Tuple] = []
    lengths: List[int] = []
    reset_seeds: List[int] = []
    try:
        for i in range(n_episodes):
            reset_seeds.append(int(seeds.integers(0, _SEED_HIGH)))
            episode = _expert_episode(venv, expert, reset_seeds[-1], counters, check)
            k = classical.select_index(float(selection.random()), len(episode))
            rows.append((i, k, *episode[k]))
            lengths.append(len(episode))
    finally:
        venv.close()
    stats = {
        **counters,
        "retained_labels": n_episodes,
        "episode_lengths": lengths,
        "episode_reset_seeds": reset_seeds,
        "reset_seed_stream": f"SeedSequence(seed, spawn_key=({_STREAM_RESETS},))",
        "selection_stream": f"SeedSequence(seed, spawn_key=({_STREAM_SELECTION},))",
        "selection_rule": "floor(u * episode_length), u uniform in [0, 1)",
        "elapsed_seconds": time.monotonic() - start,
    }
    return ExpertData.from_rows(rows), stats


class CountingVecEnv(VecEnvWrapper):
    """Count environment transitions, explicit resets and finished episodes."""

    def __init__(self, venv):
        super().__init__(venv)
        self.steps = 0
        self.resets = 0
        self.completed_episodes = 0

    def reset(self):
        self.resets += 1
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.steps += len(dones)
        self.completed_episodes += int(np.sum(dones))
        return obs, rewards, dones, infos


def _uniform_random_baselines(baselines) -> bool:
    """Whether recorded baselines hold the measured uniform random reference.

    Baselines without reference metadata hold it. Recorded metadata must
    agree: type ``uniform_random``, no ``stalled_return`` and any
    ``measured_uniform_random_return`` equal to ``random_return``. Records
    are only read, never rewritten.
    """
    if not isinstance(baselines, dict):
        return False
    kind = baselines.get("normalization_reference_type", UNIFORM_RANDOM_REFERENCE)
    measured = baselines.get(
        "measured_uniform_random_return",
        baselines.get("random_return"),
    )
    return (
        kind == UNIFORM_RANDOM_REFERENCE
        and baselines.get("stalled_return") is None
        and measured == baselines.get("random_return")
    )


def _reference_descriptor(baselines: Dict[str, Any]) -> Dict[str, Any]:
    """The lower normalization reference a run uses, from verified baselines.

    ``reference_type_recorded`` says whether the baselines carried the type
    or predate the metadata; either way they hold the uniform random return.
    """
    if not _uniform_random_baselines(baselines):
        raise classical.Refused(
            "baselines do not hold the measured uniform random reference",
        )
    return {
        "type": UNIFORM_RANDOM_REFERENCE,
        "lower_return": baselines["random_return"],
        "expert_return": baselines["expert_return"],
        "measured_uniform_random_return": baselines["random_return"],
        "reference_type_recorded": "normalization_reference_type" in baselines,
        "rule": UNIFORM_RANDOM_REFERENCE_RULE,
    }


def _evaluation_resets(evals: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summed ``reset_accounting`` of completed evaluations, never imputed.

    Each count is null unless every evaluation reported it.
    """
    reports = [r.get("reset_accounting") for r in evals]
    totals: Dict[str, Any] = {}
    for key in ("explicit_resets", "auto_resets", "discarded_auto_resets"):
        values = [
            report.get(key) if isinstance(report, dict) else None for report in reports
        ]
        totals[key] = (
            sum(int(v) for v in values) if all(v is not None for v in values) else None
        )
    totals["rule"] = EVALUATION_RESET_RULE
    return totals


def measure_baselines(
    expert,
    seed: int,
    n_episodes: int,
    check: Callable[[str], None] = _no_check,
    observer: Optional[Dict[str, Any]] = None,
    env_name: str = ENV_NAME,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Normalization baselines as in ``env_baselines.compute_baselines``.

    Runs the same two routines (deterministic expert rollout with self
    cross-entropy, then uniform random actions) on a dedicated environment,
    with the random-action sampler seeded from a dedicated stream. If given,
    ``observer["venv"]`` receives the counting environment, so an interrupted
    measurement still reports its observed steps.

    ``random_return`` is the measured uniform random return in every
    environment, CliffWalking included; ``env_baselines.normalization_reference``
    supplies any overrides (none), so the original three keys are returned.

    Returns:
        The baselines and their counted costs.
    """
    start = time.monotonic()
    venv = CountingVecEnv(
        env_utils.make_env(env_name, 1, _seed_stream(seed, _BASELINE_ENV))
    )
    venv.action_space.seed(
        int(_seed_stream(seed, _BASELINE_ACTIONS).integers(0, _SEED_HIGH))
    )
    if observer is not None:
        observer["venv"] = venv
    try:
        check("baseline expert evaluation")
        expert_result = eval_utils.eval_policy_rollout(
            expert,
            venv,
            n_episodes=n_episodes,
            deterministic=True,
            expert_policy=expert,
        )
        expert_steps = venv.steps
        check("baseline random evaluation")
        random_return = env_baselines.compute_random_return(venv, n_episodes=n_episodes)
    finally:
        venv.close()
    baselines = {
        "expert_return": float(expert_result.mean_return),
        "random_return": float(random_return),
        "expert_self_ce": float(expert_result.current_round_ce),
    }
    baselines.update(
        env_baselines.normalization_reference(env_name, float(random_return)),
    )
    costs = {
        "expert_episodes": int(n_episodes),
        "random_episodes": int(n_episodes),
        "expert_env_steps": int(expert_result.n_steps),
        "random_env_steps": int(venv.steps - expert_steps),
        "env_steps_counted": int(venv.steps),
        "expert_predict_calls": 2 * int(expert_result.n_steps),
        "expert_predict_rule": (
            "eval_policy_rollout queries the expert once as the behavior policy "
            "and once as the labeler per step"
        ),
        "elapsed_seconds": time.monotonic() - start,
    }
    return baselines, costs


def pool_logical_cost(episode_lengths: Sequence[int], budget: int) -> Dict[str, Any]:
    """Acquisition of a standalone chronological collection reaching ``budget``."""
    total = 0
    for i, length in enumerate(episode_lengths):
        total += int(length)
        if total >= budget:
            return {
                "labels": int(budget),
                "episodes": i + 1,
                "env_steps": total,
                "overshoot_transitions": total - int(budget),
            }
    raise ValueError(f"Pool episodes cover fewer than {budget} transitions")


def eval_budgets(n_rounds: int, eval_interval: int) -> List[int]:
    """Historical pilot evaluations after zero: first, periodic and final."""
    points = {1, int(n_rounds)}
    points.update(range(eval_interval, n_rounds + 1, eval_interval))
    return sorted(points)


def trained_pairs(demos_dir: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    """Pairs a round-loop trainer aggregated, read back from its demos."""
    rounds = sorted(
        (int(p.name.split("-", 1)[1]), p)
        for p in pathlib.Path(demos_dir).iterdir()
        if p.is_dir() and p.name.startswith("round-")
    )
    trajectories: List[types.Trajectory] = []
    for _, round_dir in rounds:
        for path in sorted(round_dir.iterdir()):
            if path.name.endswith(".npz"):
                trajectories.extend(serialize.load(path))
    flat = rollout.flatten_trajectories(trajectories)
    return np.asarray(flat.obs), np.asarray(flat.acts)


# ---------------------------------------------------------------------------
# Shared job plumbing
# ---------------------------------------------------------------------------


def source_identity(source_files: Sequence[str] = _SOURCE_FILES) -> Dict[str, Any]:
    """Digests of the source files that determine pilot behavior."""
    files = {}
    for rel in source_files:
        path = _PACKAGE_ROOT / rel
        files[rel] = pilot.sha256_file(path) if path.is_file() else "missing"
    status = pilot._git("status", "--porcelain")
    return {
        "files": files,
        "combined_sha256": pilot.config_digest(files),
        "git_revision": pilot._git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def _spec_source(spec: RunnerSpec) -> Dict[str, Any]:
    if spec.source_files == _SOURCE_FILES:
        return source_identity()
    return source_identity(spec.source_files)


def _validate_limits(
    seed: Optional[int],
    job_limit_seconds: float,
    spec: RunnerSpec = CARTPOLE_V2,
) -> None:
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise classical.Refused(f"seed must be a non-negative integer: {seed!r}")
        if seed == spec.audit_seed:
            raise classical.Refused(
                f"seed {spec.audit_seed} is reserved for the mask audit",
            )
    limit = job_limit_seconds
    cap = spec.max_job_limit_seconds
    if (
        isinstance(limit, bool)
        or not isinstance(limit, (int, float))
        or not math.isfinite(limit)
        or not 0 < limit <= cap
    ):
        raise classical.Refused(
            f"job_limit_seconds must be in (0, {cap}]: {limit!r}",
        )


def _deadlines(
    deadline, job_limit_seconds, clock
) -> Tuple[classical.Guard, Dict[str, Any]]:
    requested = rollouts.ensure_aware_utc(deadline)
    start = clock()
    effective = min(
        requested,
        start + datetime.timedelta(seconds=float(job_limit_seconds)),
        classical.HARD_CAP,
    )
    guard = classical.Guard(effective, clock)
    return guard, {
        "requested_utc": requested.isoformat(),
        "job_limit_seconds": job_limit_seconds,
        "job_started_utc": start.isoformat(),
        "hard_cap_utc": classical.HARD_CAP.isoformat(),
        "effective_utc": guard.deadline.isoformat(),
        "enforcement": (
            "checked before every data acquisition step, fit, and round, and "
            "before marking a result complete; a started round or evaluation "
            "finishes first; an external controller enforces the hard limits"
        ),
    }


def _load_expert(
    preparation_dir,
    spec: RunnerSpec = CARTPOLE_V2,
) -> Tuple[Any, Dict[str, Any]]:
    if spec.task is None:
        model, identity = classical.load_verified_expert(
            pathlib.Path(preparation_dir),
            spec.env_name,
        )
    else:
        model, identity = task_experts.load_qualified_expert(
            pathlib.Path(preparation_dir),
            spec.env_name,
        )
    identity = dict(identity)
    # Keep machine paths out of records: the digests identify the expert.
    identity["preparation_dir_name"] = pathlib.Path(
        identity.pop("preparation_dir"),
    ).name
    return model.policy, identity


def _check_cap(venv=None, spec: RunnerSpec = CARTPOLE_V2) -> int:
    if spec.task is not None:
        # The verified description already holds the factory's effective cap.
        try:
            cap = tasks.describe(spec.task)["episode_cap"]
            if venv is not None and tasks.venv_episode_cap(venv) != cap:
                raise RuntimeError(f"Run environment cap differs from {cap}")
        except tasks.TaskMismatch as exc:
            raise classical.Refused(str(exc)) from None
        if cap != spec.episode_cap:
            raise classical.Refused(
                f"{spec.env_name} cap {cap} is not {spec.episode_cap}",
            )
        return int(cap)
    cap = gym.spec(spec.env_name).max_episode_steps
    if venv is not None:
        cap_env = venv.get_attr("spec")[0].max_episode_steps
        if cap_env != cap:
            raise RuntimeError(f"Environment cap {cap_env} differs from spec {cap}")
    if cap != spec.episode_cap:
        raise classical.Refused(
            f"{spec.env_name} time limit {cap} is not {spec.episode_cap}",
        )
    return int(cap)


def _base_record(schema, identity, limits, clock, spec=CARTPOLE_V2) -> Dict[str, Any]:
    record = {
        "protocol": spec.data_protocol if schema == spec.data_schema else spec.protocol,
        "schema": schema,
        "env_name": spec.env_name,
        "episode_cap": spec.episode_cap,
        "expert_sha256": identity["expert_sha256"],
        "preparation": identity,
        "deadline": limits,
        "source": _spec_source(spec),
        "package_versions": pilot.package_versions(),
        "started_at_utc": clock().isoformat(),
    }
    if spec.task is not None:
        record["task"] = tasks.describe(spec.task)
        record["task_sha256"] = tasks.task_sha256(spec.task)
    return record


def _start(
    output_dir,
    schema,
    identity,
    limits,
    guard,
    render_extra,
    spec: RunnerSpec = CARTPOLE_V2,
):
    """Claim the output and build lifecycle state and writer."""
    output_dir = pathlib.Path(output_dir)
    claim = classical.claim_fresh_output_dir(output_dir, guard.clock)
    base = _base_record(schema, identity, limits, guard.clock, spec)
    state: Dict[str, Any] = {
        "status": "running",
        "in_flight": None,
        "error": None,
        "finished_at_utc": None,
    }

    def render():
        return {
            **base,
            "claim": claim,
            "status": state["status"],
            "error": state["error"],
            "interruption": classical._interruption(state),
            "finished_at_utc": state["finished_at_utc"],
            "snapshot": {
                "written_at_utc": rollouts.utc_now().isoformat(),
                "final": state["status"] != "running",
                "note": SNAPSHOT_NOTE,
            },
            **render_extra(),
        }

    writer = classical.ResultWriter(output_dir / RESULT_FILE, render)
    return state, writer


def _check_expert_quality(identity, spec: RunnerSpec, allow_relaxed: bool) -> None:
    """Refuse an expert that passes only a relaxed per-environment threshold."""
    if spec.task is None:
        return
    quality = identity.get("quality") or {}
    if not quality.get("per_env_passed"):
        raise classical.Refused("The expert did not pass its qualification threshold")
    if not quality.get("global_passed") and not allow_relaxed:
        raise classical.Refused(
            "The expert fails the global 0.80 quality check (relaxed-only); "
            "using it needs an explicit allow_relaxed_expert",
        )


CAPACITY_POLICY_RULE = (
    "capacity-one-hidden: the qualified teacher's frozen features extractor "
    "(full observation, identity restriction), then a fresh one-hidden actor "
    "of the student width and a categorical head; only the actor is trained"
)


def _capacity_teacher(expert, identity, spec: RunnerSpec) -> Dict[str, Any]:
    """Refuse a capacity teacher that is not a verified one-hidden preparation.

    The preparation must claim the declared architecture explicitly (a legacy
    config claims nothing and is refused even if its modules happen to fit),
    and the loaded policy must have exactly that actor and activation.

    Returns:
        `capacity_policy.teacher_architecture` of the expert.
    """
    learner = spec.learner
    if spec.task is None:
        raise classical.Refused("the capacity family needs a generic task spec")
    if identity.get("expert_source") != "preparation":
        raise classical.Refused(
            "a capacity teacher must come from an expert preparation",
        )
    claimed = (identity.get("architecture") or {}).get("claimed") or {}
    if claimed.get("pi") != [learner.teacher_width]:
        raise classical.Refused(
            f"the teacher preparation does not claim the one-hidden "
            f"[{learner.teacher_width}] architecture: {claimed.get('pi')!r}",
        )
    try:
        description = capacity_policy.teacher_architecture(
            expert,
            learner.teacher_width,
        )
    except ValueError as exc:
        raise classical.Refused(f"incompatible capacity teacher: {exc}") from None
    if description["activation_fn"] != claimed.get("activation_fn"):
        raise classical.Refused("the teacher activation differs from its claim")
    if description["policy_state_sha256"] != identity.get("policy_state_sha256"):
        raise classical.Refused("the teacher state differs from its preparation")
    return description


def _verified_checkpoint(path, capacity, binding, expected_path) -> Dict[str, Any]:
    """Verify a capacity checkpoint against its writer's binding and the run.

    ``binding`` is what ``save_bound`` returned when it wrote ``path``
    (`run_experiment._checkpoint_fields`); ``expected_path`` is where this
    run keeps the checkpoint of the record's budget. The trusted loader then
    hashes and parses a single read of the file against the binding's file,
    state and spec digests.

    Returns:
        The row fields: the verified spec digest and a copy of the binding.

    Raises:
        ValueError: For a missing or foreign binding, another path or spec,
            or any `capacity_policy.load_checkpoint` refusal.
    """
    if (
        not isinstance(binding, dict)
        or binding.get("schema") != capacity_policy.CHECKPOINT_BINDING_SCHEMA
    ):
        raise ValueError(f"{path} has no capacity checkpoint writer binding")
    if not binding.get("path") == str(path) == str(expected_path):
        raise ValueError(
            f"the writer bound {binding.get('path')}, not {path} at the run's "
            f"{expected_path}",
        )
    if binding.get("spec_sha256") != capacity["spec_sha256"]:
        raise ValueError(f"the writer bound {path} to another capacity spec")
    policy = capacity_policy.load_checkpoint(
        path,
        expected_spec_sha256=capacity["spec_sha256"],
        expected_file_sha256=binding.get("file_sha256"),
        expected_state_sha256=binding.get("state_sha256"),
    )
    return {
        "checkpoint_capacity_spec_sha256": policy.capacity_spec_sha256,
        "checkpoint_binding": dict(binding),
    }


def _refused(exc: classical.Refused) -> int:
    logger.error("Refused: %s", exc)
    print(f"Refused: {exc}", file=sys.stderr)
    return classical.EXIT_REFUSED


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def run_audit(
    output_dir: pathlib.Path,
    preparation_dir: pathlib.Path,
    *,
    deadline: datetime.datetime,
    job_limit_seconds: float,
    clock: Callable[[], datetime.datetime] = rollouts.utc_now,
) -> int:
    """Run the predeclared paired-state mask audit into a fresh directory."""
    try:
        _validate_limits(None, job_limit_seconds)
        _check_cap()
        expert, identity = _load_expert(preparation_dir)
        guard, limits = _deadlines(deadline, job_limit_seconds, clock)
        live: Dict[str, Any] = {"audit": None}
        state, writer = _start(
            output_dir,
            AUDIT_SCHEMA,
            identity,
            limits,
            guard,
            lambda: {
                "config": {
                    "restriction_id": PILOT_RESTRICTION,
                    "reset_seed": AUDIT_SEED,
                    "grid": list(restriction.AUDIT_X_GRID),
                    "base_steps": list(restriction.AUDIT_BASE_STEPS),
                    "witness_scope": restriction.AUDIT_WITNESS_SCOPE,
                },
                "audit_status": (live["audit"] or {}).get("status"),
                # The audit reports its costs only when it finishes.
                "expert_predict_calls": (live["audit"] or {}).get(
                    "expert_predict_calls"
                ),
                "audit": live["audit"],
            },
        )
    except classical.Refused as exc:
        return _refused(exc)

    def body():
        state["in_flight"] = "paired-state audit"
        live["audit"] = restriction.audit_cartpole_position(
            expert,
            AUDIT_SEED,
            restriction_id=PILOT_RESTRICTION,
            check=guard.check,
        )

    return classical._run_lifecycle(state, writer, body, guard)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def prepare_data(
    output_dir: pathlib.Path,
    preparation_dir: pathlib.Path,
    *,
    seed: int,
    deadline: datetime.datetime,
    job_limit_seconds: float,
    budget: int = DEFAULT_BUDGET,
    baseline_episodes: int = BASELINE_EPISODES,
    clock: Callable[[], datetime.datetime] = rollouts.utc_now,
    spec: RunnerSpec = CARTPOLE_V2,
    allow_relaxed_expert: bool = False,
) -> int:
    """Acquire the shared fixed-BC pool, BC-iid stream and baselines once."""
    output_dir = pathlib.Path(output_dir)
    try:
        _validate_limits(seed, job_limit_seconds, spec)
        if int(budget) <= 0 or int(baseline_episodes) <= 0:
            raise classical.Refused("budget and baseline_episodes must be positive")
        if spec.learner is not None and allow_relaxed_expert:
            raise classical.Refused(
                "the capacity family never accepts a relaxed-only expert",
            )
        _check_cap(spec=spec)
        expert, identity = _load_expert(preparation_dir, spec)
        _check_expert_quality(identity, spec, allow_relaxed_expert)
        teacher = None
        if spec.learner is not None:
            teacher = _capacity_teacher(expert, identity, spec)
        guard, limits = _deadlines(deadline, job_limit_seconds, clock)
        live: Dict[str, Any] = {
            "counters": {"pool": _new_counters(), "stream": _new_counters()},
            "datasets": {},
            "baselines": None,
            "baselines_costs": None,
            "baseline_observer": {},
        }
        config = {
            "seed": seed,
            "budget": int(budget),
            "baseline_episodes": int(baseline_episodes),
            "pool": POOL_RULE,
            "stream": STREAM_RULE,
            "expert_action_rule": classical.ACTION_RULE,
        }
        if spec.task is not None:
            config["allow_relaxed_expert"] = bool(allow_relaxed_expert)
        if teacher is not None:
            config["learner_family"] = spec.learner.family
            config["capacity_teacher"] = teacher
        state, writer = _start(
            output_dir,
            spec.data_schema,
            identity,
            limits,
            guard,
            lambda: {
                "seed": seed,
                "config": config,
                "config_sha256": pilot.config_digest(config),
                "live_counters": live["counters"],
                "live_counters_rule": "counted when a query, reset or step "
                "returns; an interrupted operation is not counted",
                "datasets": live["datasets"],
                "baselines": live["baselines"],
                "baselines_costs": live["baselines_costs"],
                "baselines_observed": _observed_baseline(live),
            },
            spec,
        )
    except classical.Refused as exc:
        return _refused(exc)

    def store(name, file_name, data, stats):
        path = output_dir / file_name
        data.save(path)
        live["datasets"][name] = {
            "file": file_name,
            "file_sha256": pilot.sha256_file(path),
            "pairs_sha256": data.pairs_sha256(),
            "n": len(data),
            "stats": stats,
        }
        writer.write()

    def body():
        state["in_flight"] = "fixed BC pool collection"
        pool, pool_stats = collect_chronological_pool(
            expert,
            budget,
            seed,
            check=guard.check,
            counters=live["counters"]["pool"],
            env_name=spec.env_name,
        )
        store("pool", POOL_FILE, pool, pool_stats)
        state["in_flight"] = "BC-iid stream collection"
        stream, stream_stats = collect_iid_stream(
            expert,
            budget,
            seed,
            check=guard.check,
            counters=live["counters"]["stream"],
            env_name=spec.env_name,
        )
        store("stream", STREAM_FILE, stream, stream_stats)
        state["in_flight"] = "baseline measurement"
        live["baselines"], live["baselines_costs"] = measure_baselines(
            expert,
            seed,
            int(baseline_episodes),
            check=guard.check,
            observer=live["baseline_observer"],
            env_name=spec.env_name,
        )

    return classical._run_lifecycle(state, writer, body, guard)


def _observed_baseline(live) -> Optional[Dict[str, Any]]:
    """Observed baseline steps; predict calls are known only on completion."""
    venv = live["baseline_observer"].get("venv")
    if venv is None:
        return None
    costs = live["baselines_costs"]
    return {
        "env_steps": int(venv.steps),
        "env_resets": int(venv.resets),
        "expert_predict_calls": (
            None if costs is None else costs["expert_predict_calls"]
        ),
    }


def _validate_pin(expected_data_sha256: Optional[str]) -> Optional[str]:
    if expected_data_sha256 is None:
        return None
    pin = expected_data_sha256
    if (
        not isinstance(pin, str)
        or len(pin) != 64
        or any(c not in _SHA256_HEX for c in pin)
    ):
        raise classical.Refused(
            f"expected data SHA256 must be 64 lowercase hex digits: {pin!r}",
        )
    return pin


def _finite_baselines(baselines) -> bool:
    keys = ("expert_return", "random_return", "expert_self_ce")
    return isinstance(baselines, dict) and all(
        isinstance(baselines.get(k), (int, float))
        and not isinstance(baselines.get(k), bool)
        and math.isfinite(baselines[k])
        for k in keys
    )


def _verify_data(
    data_dir,
    seed,
    identity,
    n_rounds,
    expected_data_sha256=None,
    spec: RunnerSpec = CARTPOLE_V2,
):
    """Check the shared data job and return its record, data, digest, acceptance.

    The record is hashed and parsed from the same bytes. Every check below
    applies whatever the source version; a record from another source version
    is accepted only if ``expected_data_sha256`` equals its SHA256, and a pin
    that does not match refuses even same-source data. Under a generic spec
    the record must also carry the current verified task description.
    """
    data_dir = pathlib.Path(data_dir)
    path = data_dir / RESULT_FILE
    try:
        raw = path.read_bytes()
        record = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise classical.Refused(f"Unreadable data record {path}: {exc}")
    record_sha = hashlib.sha256(raw).hexdigest()
    if expected_data_sha256 is not None and record_sha != expected_data_sha256:
        raise classical.Refused(
            f"Data record {path} has SHA256 {record_sha}, not the pinned "
            f"{expected_data_sha256}",
        )
    if not isinstance(record, dict):
        raise classical.Refused(f"{path} is not a data record")
    config = record.get("config")
    config = config if isinstance(config, dict) else {}
    source = record.get("source")
    source = source if isinstance(source, dict) else {}
    files = source.get("files")
    current = _spec_source(spec)
    checks = {
        "schema": record.get("schema") == spec.data_schema,
        "protocol": record.get("protocol") == spec.data_protocol,
        "status_complete": record.get("status") == "complete",
        "env_name": record.get("env_name") == spec.env_name,
        "episode_cap": record.get("episode_cap") == spec.episode_cap,
        "seed": record.get("seed") == seed,
        "config_sha256": record.get("config_sha256") == pilot.config_digest(config),
        "config_seed": config.get("seed") == seed,
        "config_budget": isinstance(config.get("budget"), int)
        and config["budget"] >= n_rounds,
        "config_rules": config.get("pool") == POOL_RULE
        and config.get("stream") == STREAM_RULE
        and config.get("expert_action_rule") == classical.ACTION_RULE,
        "source_combined_sha256": isinstance(files, dict)
        and source.get("combined_sha256") == pilot.config_digest(files),
        "source_file_set": isinstance(files, dict)
        and sorted(files) == sorted(current["files"]),
        "package_versions": record.get("package_versions") == pilot.package_versions(),
        "expert_sha256": record.get("expert_sha256") == identity["expert_sha256"],
        "preparation_identity": record.get("preparation") == identity,
        "baselines": _finite_baselines(record.get("baselines")),
        "baseline_reference": _uniform_random_baselines(record.get("baselines")),
    }
    if spec.task is not None:
        checks["task"] = record.get("task") == tasks.describe(spec.task)
        checks["task_sha256"] = record.get("task_sha256") == tasks.task_sha256(
            spec.task,
        )
    failed = sorted(k for k, ok in checks.items() if not ok)
    if failed:
        raise classical.Refused(f"Data record {path} failed checks {failed}")
    same_source = files == current["files"]
    if not same_source and expected_data_sha256 is None:
        raise classical.Refused(
            f"Data record {path} was produced by source "
            f"{source['combined_sha256']}, not the current "
            f"{current['combined_sha256']}; reusing it requires pinning its "
            "exact SHA256 (expected_data_sha256)",
        )
    datasets = {}
    for name, file_name in (("pool", POOL_FILE), ("stream", STREAM_FILE)):
        entry = (record.get("datasets") or {}).get(name) or {}
        data_path = data_dir / file_name
        if entry.get("file") != file_name or not data_path.is_file():
            raise classical.Refused(f"Data record lacks the {name} dataset")
        if pilot.sha256_file(data_path) != entry.get("file_sha256"):
            raise classical.Refused(f"{data_path} does not match its recorded digest")
        data = ExpertData.load(data_path)
        if data.pairs_sha256() != entry.get("pairs_sha256"):
            raise classical.Refused(f"{data_path} content does not match its record")
        if entry.get("n") != len(data):
            raise classical.Refused(f"{data_path} size does not match its record")
        if len(data) < n_rounds:
            raise classical.Refused(f"{name} holds {len(data)} < {n_rounds} labels")
        datasets[name] = data
    acceptance = {
        "rule": "same_source" if same_source else "pinned_different_source",
        "expected_data_sha256": expected_data_sha256,
        "data_result_sha256": record_sha,
        "checks": sorted(checks),
        "producer_protocol": record["protocol"],
        "producer_schema": record["schema"],
        "producer_source": source,
        "producer_package_versions": record["package_versions"],
        "current_source_combined_sha256": current["combined_sha256"],
    }
    return record, datasets, record_sha, acceptance


def _verify_reference(reference_dir, expected_sha256, identity, spec: RunnerSpec):
    """Check a pinned data record whose baselines normalize a generic run.

    The pin proves which bytes were chosen; the checks prove they describe the
    current task. The reference must be a complete data job with exactly the
    generic protocol and data schema, the current task description and digest,
    the current package versions, an internally consistent config (digest,
    pool, stream and action rules, a valid non-audit seed equal to the
    record's) and source map (combined digest, the protocol's file set), the
    same expert checkpoint and policy state, and complete original baseline
    evidence: `BASELINE_EPISODES` expert and random episodes in config and
    costs, with consistent step counts. Its seed may differ from the run's and
    its source bytes may differ from the current source. For CartPole only,
    the pilot data pair is also accepted; such a record has no task field, so
    it is bound by that pair, the 500 step cap and the pilot's file set.

    Returns:
        The reference provenance block, baselines included.
    """
    pin = _validate_pin(expected_sha256)
    if pin is None:
        raise classical.Refused("A reference data record must be pinned by SHA256")
    path = pathlib.Path(reference_dir) / RESULT_FILE
    try:
        raw = path.read_bytes()
        record = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise classical.Refused(f"Unreadable reference record {path}: {exc}")
    record_sha = hashlib.sha256(raw).hexdigest()
    if record_sha != pin:
        raise classical.Refused(
            f"Reference record {path} has SHA256 {record_sha}, not the pinned {pin}",
        )
    if not isinstance(record, dict):
        raise classical.Refused(f"{path} is not a data record")
    pair = (record.get("protocol"), record.get("schema"))
    legacy = spec.env_name == CARTPOLE_V2.env_name and pair == (
        CARTPOLE_V2.data_protocol,
        CARTPOLE_V2.data_schema,
    )
    expected_files = CARTPOLE_V2.source_files if legacy else spec.source_files

    def as_dict(value):
        return value if isinstance(value, dict) else {}

    def count(value, low, high):
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and low <= value <= high
        )

    config = as_dict(record.get("config"))
    source = as_dict(record.get("source"))
    files = source.get("files")
    costs = as_dict(record.get("baselines_costs"))
    seed = record.get("seed")
    cap = spec.episode_cap
    most = BASELINE_EPISODES * cap
    checks = {
        "protocol_pair": pair == (spec.data_protocol, spec.data_schema) or legacy,
        "status_complete": record.get("status") == "complete",
        "env_name": record.get("env_name") == spec.env_name,
        "episode_cap": record.get("episode_cap") == cap,
        "task": legacy or record.get("task") == tasks.describe(spec.task),
        "task_sha256": legacy
        or record.get("task_sha256") == tasks.task_sha256(spec.task),
        "package_versions": record.get("package_versions") == pilot.package_versions(),
        "config_sha256": record.get("config_sha256") == pilot.config_digest(config),
        "config_rules": config.get("pool") == POOL_RULE
        and config.get("stream") == STREAM_RULE
        and config.get("expert_action_rule") == classical.ACTION_RULE,
        "seed": count(seed, 0, _SEED_HIGH)
        and seed != spec.audit_seed
        and config.get("seed") == seed,
        "source_integrity": isinstance(files, dict)
        and source.get("combined_sha256") == pilot.config_digest(files),
        "source_file_set": isinstance(files, dict)
        and sorted(files) == sorted(expected_files),
        "expert": record.get("expert_sha256") == identity["expert_sha256"]
        and as_dict(record.get("preparation")).get("policy_state_sha256")
        == identity.get("policy_state_sha256"),
        "baselines": _finite_baselines(record.get("baselines")),
        "baseline_reference": _uniform_random_baselines(record.get("baselines")),
        "baseline_episodes": config.get("baseline_episodes")
        == costs.get("expert_episodes")
        == costs.get("random_episodes")
        == BASELINE_EPISODES,
        "baseline_steps": count(costs.get("expert_env_steps"), BASELINE_EPISODES, most)
        and count(costs.get("random_env_steps"), BASELINE_EPISODES, most)
        and costs.get("env_steps_counted")
        == costs["expert_env_steps"] + costs["random_env_steps"],
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    if failed:
        raise classical.Refused(f"Reference record {path} failed checks {failed}")
    return {
        "source": "pinned reference data job",
        "data_result_sha256": record_sha,
        "expected_sha256": pin,
        "producer_protocol": record.get("protocol"),
        "producer_schema": record["schema"],
        "producer_seed": record.get("seed"),
        "producer_source_combined_sha256": (record.get("source") or {}).get(
            "combined_sha256",
        ),
        "baselines": record["baselines"],
        "baselines_costs": costs,
        "checks": sorted(checks),
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _experiment_config(
    method,
    seed,
    n_rounds,
    eval_interval,
    output_dir,
    env_name=ENV_NAME,
    paired_cold_fits=False,
    bc_n_epochs=None,
    eval_episodes=None,
    final_eval_episodes=None,
    capacity=False,
):
    return run_experiment.ExperimentConfig(
        algo=method,
        env_name=env_name,
        seed=seed,
        policy_mode="linear",
        n_rounds=n_rounds,
        samples_per_round=1,
        l2_lambda=0.0,
        l2_decay=False,
        warm_start=False,
        beta_rampdown=0,
        bc_n_epochs=BC_EPOCHS if bc_n_epochs is None else bc_n_epochs,
        eval_interval=eval_interval,
        eval_episodes=(run_experiment.EVAL_EPISODES if capacity else 100)
        if eval_episodes is None else eval_episodes,
        final_eval_episodes=final_eval_episodes,
        evaluate_first_round=not capacity,
        fit_rule=(
            run_experiment.FIT_RULE_PLATEAU if capacity
            else run_experiment.FIT_RULE_EPOCHS
        ),
        skip_unreported_fits=capacity,
        output_dir=pathlib.Path(output_dir),
        expert_cache_dir=pathlib.Path(output_dir) / "unused-expert-cache",
        learning_rate=run_experiment.BC_LEARNING_RATE if capacity else LEARNING_RATE,
        outer_early_stop=False,
        grow_batch_with_data=True,
        paired_cold_fits=paired_cold_fits,
    )


def run_job(
    output_dir: pathlib.Path,
    preparation_dir: pathlib.Path,
    data_dir: pathlib.Path,
    *,
    method: str,
    restriction_id: str,
    seed: int,
    deadline: datetime.datetime,
    job_limit_seconds: float,
    n_rounds: int = DEFAULT_BUDGET,
    eval_interval: int = EVAL_INTERVAL,
    expected_data_sha256: Optional[str] = None,
    clock: Callable[[], datetime.datetime] = rollouts.utc_now,
    spec: RunnerSpec = CARTPOLE_V2,
    reference_data_dir: Optional[pathlib.Path] = None,
    expected_reference_sha256: Optional[str] = None,
    allow_relaxed_expert: bool = False,
    bc_n_epochs: Optional[int] = None,
    eval_episodes: Optional[int] = None,
    final_eval_episodes: Optional[int] = None,
) -> int:
    """Run one (method, restriction) pilot job into a fresh directory.

    Under a generic ``spec`` the baselines that normalize returns come from
    the run's own data job unless ``reference_data_dir`` names a pinned
    reference data job (`_verify_reference`); a non-identity restriction that
    leaves a single distinct observation on the data job's pool is refused.
    ``bc_n_epochs`` (generic specs only) replaces the `BC_EPOCHS` cap of every
    fit; the default None keeps `BC_EPOCHS` and the config unchanged.
    """
    output_dir = pathlib.Path(output_dir).absolute()
    generic = spec.task is not None
    learner = spec.learner
    capacity: Optional[Dict[str, Any]] = None
    try:
        _validate_limits(seed, job_limit_seconds, spec)
        pin = _validate_pin(expected_data_sha256)
        if method not in spec.methods:
            raise classical.Refused(
                f"method must be one of {spec.methods}: {method!r}",
            )
        if restriction_id not in spec.run_restrictions:
            raise classical.Refused(
                f"restriction must be one of {spec.run_restrictions}: "
                f"{restriction_id!r}",
            )
        if int(n_rounds) <= 0 or int(eval_interval) <= 0:
            raise classical.Refused("n_rounds and eval_interval must be positive")
        for name, value in (("eval_episodes", eval_episodes),
                            ("final_eval_episodes", final_eval_episodes)):
            if value is not None and int(value) <= 0:
                raise classical.Refused(f"{name} must be positive")
        if not generic and (
            reference_data_dir is not None
            or expected_reference_sha256 is not None
            or allow_relaxed_expert
        ):
            raise classical.Refused(
                "the CartPole pilot protocol takes no external reference or "
                "relaxed-expert permission",
            )
        if bc_n_epochs is not None and (
            not generic
            or isinstance(bc_n_epochs, bool)
            or not isinstance(bc_n_epochs, int)
            or bc_n_epochs <= 0
        ):
            raise classical.Refused(
                "bc_n_epochs must be a positive integer on a generic spec: "
                f"{bc_n_epochs!r}",
            )
        if learner is not None:
            if bc_n_epochs is not None:
                raise classical.Refused("bc_n_epochs applies only to fixed-epoch fits")
            if learner.student_width is None:
                raise classical.Refused("a capacity run needs a student_width")
            if restriction_id != "identity":
                raise classical.Refused(
                    "the capacity family runs on full observations: restriction "
                    f"must be identity, not {restriction_id!r}",
                )
            if allow_relaxed_expert:
                raise classical.Refused(
                    "the capacity family never accepts a relaxed-only expert",
                )
            if reference_data_dir is not None or expected_reference_sha256:
                raise classical.Refused(
                    "the capacity family normalizes with its own data job",
                )
        _check_cap(spec=spec)
        expert, identity = _load_expert(preparation_dir, spec)
        _check_expert_quality(identity, spec, allow_relaxed_expert)
        data_record, datasets, data_sha, acceptance = _verify_data(
            data_dir,
            seed,
            identity,
            n_rounds,
            pin,
            spec,
        )
        if learner is not None:
            teacher = _capacity_teacher(expert, identity, spec)
            if data_record["config"].get("capacity_teacher") != teacher:
                raise classical.Refused(
                    "the data job's capacity teacher differs from the run's",
                )
            try:
                probe = capacity_policy.make_capacity_policy(
                    expert,
                    hidden_width=learner.student_width,
                    initialization_seed=seed,
                    teacher_hidden_width=learner.teacher_width,
                )
            except ValueError as exc:
                raise classical.Refused(str(exc)) from None
            counts = probe.capacity_spec["parameters"]
            capacity = {
                "family": learner.family,
                "student_width": learner.student_width,
                "teacher_width": learner.teacher_width,
                "student_widths_declared": list(capacity_policy.STUDENT_WIDTHS),
                "spec": probe.capacity_spec,
                "spec_sha256": probe.capacity_spec_sha256,
                "teacher_architecture": teacher,
                "teacher_preparation_architecture": identity["architecture"],
                "encoder_sha256": probe.capacity_spec["encoder"]["state_sha256"],
                "trainable_parameters": sum(
                    p.numel() for p in probe.parameters() if p.requires_grad
                ),
                "optimizer_parameters": sum(
                    p.numel() for p in probe.bc_optimizer_parameters()
                ),
                "total_parameters_excluding_critic": counts["total_excluding_critic"],
                "implementation_parameters": sum(p.numel() for p in probe.parameters()),
                "initialization_seed_source": "the run seed",
                "scope": (
                    "benchmark" if n_rounds == DEFAULT_BUDGET else ("reduced-budget")
                ),
                "eval_interval": int(eval_interval),
                "policy_mode_note": "run_experiment policy_mode 'linear' only "
                "routes the factory-built partially frozen learner; the factory "
                "is capacity_policy.make_capacity_policy",
            }
            del probe
        if method == "bc" and len(datasets["pool"]) != n_rounds:
            raise classical.Refused(
                f"fixed BC fits the whole pool once: n_rounds {n_rounds} must "
                f"equal the pool size {len(datasets['pool'])}",
            )
        reference = None
        if generic and reference_data_dir is not None:
            reference = _verify_reference(
                reference_data_dir,
                expected_reference_sha256,
                identity,
                spec,
            )
        elif expected_reference_sha256 is not None:
            raise classical.Refused("expected_reference_sha256 needs a reference")
        if generic:
            pool_obs = datasets["pool"].obs
            mask = restriction.get_restriction(restriction_id)
            distinct = {
                "pool": int(len(np.unique(pool_obs, axis=0))),
                "masked_pool": int(len(np.unique(mask.apply(pool_obs), axis=0))),
            }
            if restriction_id != "identity" and distinct["masked_pool"] == 1:
                raise classical.Refused(
                    f"{restriction_id} leaves one distinct observation on the pool",
                )
        guard, limits = _deadlines(deadline, job_limit_seconds, clock)
    except classical.Refused as exc:
        return _refused(exc)

    config = _experiment_config(
        method,
        seed,
        n_rounds,
        eval_interval,
        output_dir,
        spec.env_name,
        spec.paired_cold_fits,
        bc_n_epochs,
        eval_episodes=eval_episodes,
        final_eval_episodes=final_eval_episodes,
        capacity=capacity is not None,
    )
    experiment = {
        k: v
        for k, v in dataclasses.asdict(config).items()
        if k not in ("output_dir", "expert_cache_dir", "result_name_override")
    }
    if not spec.paired_cold_fits:
        # Absent unless opted in, so legacy configs and digests are unchanged.
        del experiment["paired_cold_fits"]
    prefix = method == PREFIX_METHOD
    shared_name = {"bc": "pool", "bc_iid": "stream", PREFIX_METHOD: "pool"}.get(
        method,
    )
    shared = datasets.get(shared_name)
    shared_entry = data_record["datasets"][shared_name] if shared_name else None
    run_config = {
        "seed": seed,
        "method": method,
        "restriction_id": restriction_id,
        "restriction": dataclasses.asdict(restriction.get_restriction(restriction_id)),
        "policy": (
            "linear: expert clone, frozen features, reinitialized action_net; "
            "restriction applied before the frozen features"
            if capacity is None
            else CAPACITY_POLICY_RULE
        ),
        "beta": {"ftl": 0.0, "bc_iid": 1.0, "bc": None, PREFIX_METHOD: None}[method],
        "eval_episodes": int(config.eval_episodes),
        "eval_deterministic": True,
        "eval_budgets": (
            [n_rounds]
            if method == "bc"
            else [0] + eval_budgets(n_rounds, eval_interval)
            if config.evaluate_first_round
            else run_experiment.eval_rounds(n_rounds, eval_interval)
        ),
        "fixed_bc": FIXED_BC_RULE if method == "bc" else None,
        "batch_rule": BATCH_RULE,
        "mixture": False,
        "experiment": experiment,
    }
    if capacity is not None:
        run_config["eval_protocol"] = run_experiment.EVAL_PROTOCOL
    if config.final_eval_episodes is not None:
        run_config["final_eval_episodes"] = int(config.final_eval_episodes)
    if prefix:
        # Added only here, so the configs of the other methods are unchanged.
        run_config["bc_prefix"] = BC_PREFIX_RULE
    if spec.paired_cold_fits:
        run_config["paired_cold_fits"] = PAIRED_COLD_FITS_RULE
    baselines = (
        data_record["baselines"] if reference is None else reference["baselines"]
    )
    if spec.calibration_candidates:
        run_config["calibration_candidates"] = {
            rid: pilot.config_digest(restriction.describe_restriction(rid))
            for rid in spec.calibration_candidates
        }
    if generic:
        run_config.update(
            restriction=restriction.describe_restriction(restriction_id),
            restriction_sha256=pilot.config_digest(
                restriction.describe_restriction(restriction_id),
            ),
            task_sha256=tasks.task_sha256(spec.task),
            allow_relaxed_expert=bool(allow_relaxed_expert),
            distinct_observations=distinct,
            normalization_reference=_reference_descriptor(baselines),
            reference=(
                reference
                or {
                    "source": "own data job",
                    "data_result_sha256": data_sha,
                    "baselines": data_record["baselines"],
                }
            ),
        )
    capacity_fields: Dict[str, Any] = {}
    if capacity is not None:
        run_config.update(learner_family=capacity["family"], capacity=capacity)
        capacity_fields = {
            "learner_family": capacity["family"],
            "student_width": capacity["student_width"],
            "capacity_spec_sha256": capacity["spec_sha256"],
        }
    demos_rel = f"scratch/{method}_{spec.env_name}_seed{seed}/demos"
    live: Dict[str, Any] = {
        "records": [],
        "venv": None,
        "trained_pairs_sha256": None,
        "setup_seconds": None,
        "total_seconds": None,
        "state": None,
    }

    def accounting():
        records = live["records"]
        venv = live["venv"]
        evals = [r for r in records if r.get("d_eval_size") is not None]
        eval_steps = sum(int(r["d_eval_size"]) for r in evals)
        fits = [r for r in records if r.get("inner_es_stop_epoch") is not None]
        if method == "ftl":
            steps = int(records[-1].get("collection_steps", 0)) if records else 0
            episodes = sum(
                int(r.get("trajectories_collected_this_round", 0)) for r in records
            )
        else:
            steps = episodes = 0
        counted = None if venv is None else int(venv.steps)
        unattributed = None if counted is None else counted - (steps + eval_steps)
        complete = (live["state"] or {}).get("status") == "complete"
        exact = complete and unattributed == 0
        queries = steps + eval_steps if exact else None
        wall = {
            "setup": live["setup_seconds"],
            "total": live["total_seconds"],
            "collection_and_training": sum(
                r["wall_seconds"].get("collect_train_seconds", 0.0) for r in records
            ),
            "fits": sum(r["wall_seconds"].get("fit", 0.0) for r in records),
            "evaluation": sum(
                r["wall_seconds"].get(
                    "eval_seconds", r["wall_seconds"].get("eval", 0.0)
                )
                for r in records
            ),
        }
        return {
            "observed": {
                "env_steps": counted,
                "env_resets": None if venv is None else int(venv.resets),
                "finished_episodes": (
                    None if venv is None else int(venv.completed_episodes)
                ),
                "rule": "counted by the run's environment wrapper when a step "
                "or reset returns",
            },
            "completed_records": {
                "collection": {
                    "env_steps": steps,
                    "episodes": episodes,
                    "expert_predict_calls": steps,
                    "expert_action_entries": steps,
                    "learner_predict_calls": steps,
                    "rule": COLLECTION_RULE,
                },
                "evaluation": {
                    "evaluations": len(evals),
                    "episodes": sum(len(r["episode_returns"]) for r in evals),
                    "env_steps": eval_steps,
                    "expert_predict_calls": eval_steps,
                    "expert_action_entries": eval_steps,
                    "learner_predict_calls": eval_steps,
                    "rule": EVALUATION_RULE,
                    "resets": _evaluation_resets(evals),
                },
                "retained_labels": int(records[-1]["n_observations"]) if records else 0,
                "training": {
                    "fits": len(fits),
                    "epochs": sum(int(r["inner_es_stop_epoch"]) for r in fits),
                    "early_stopped_fits": sum(
                        int(r["inner_es_stop_epoch"]) < config.bc_n_epochs for r in fits
                    ),
                    "fixed_budget_fallback_fits": sum(
                        r.get("inner_es_fallback") is not None for r in fits
                    ),
                    "monitored_fits": sum(
                        r.get("fit_monitor") is not None for r in fits
                    ),
                    "cap_without_plateau_fits": sum(
                        bool(r.get("fit_cap_without_plateau")) for r in fits
                    ),
                },
            },
            "in_flight": {
                "operation": (live["state"] or {}).get("in_flight"),
                "env_steps": unattributed,
                "expert_predict_calls": 0 if exact else None,
                "learner_predict_calls": 0 if exact else None,
                "note": IN_FLIGHT_NOTE,
            },
            "totals": {
                "exact": exact,
                "env_steps": counted,
                "expert_predict_calls": queries,
                "expert_action_entries": queries,
                "learner_predict_calls": queries,
            },
            "reconciled": None if counted is None else unattributed == 0,
            "shared_acquisition": (
                None
                if shared_entry is None
                else {
                    "source": "data job",
                    "dataset": shared_name,
                    "file": shared_entry["file"],
                    "pairs_sha256": shared_entry["pairs_sha256"],
                    "physical_env_steps": shared_entry["stats"]["env_steps"],
                    "physical_episodes": shared_entry["stats"]["episodes"],
                    "physical_expert_predict_calls": shared_entry["stats"][
                        "expert_predict_calls"
                    ],
                    "charged_to_this_run": False,
                }
            ),
            "wall_seconds": wall,
        }

    try:
        state, writer = _start(
            output_dir,
            spec.run_schema,
            identity,
            limits,
            guard,
            lambda: {
                "seed": seed,
                "method": method,
                "restriction_id": restriction_id,
                "config": run_config,
                "config_sha256": pilot.config_digest(run_config),
                "data": {
                    "data_result_sha256": data_sha,
                    "pool_pairs_sha256": data_record["datasets"]["pool"][
                        "pairs_sha256"
                    ],
                    "stream_pairs_sha256": data_record["datasets"]["stream"][
                        "pairs_sha256"
                    ],
                    "baselines": data_record["baselines"],
                    "acceptance": acceptance,
                },
                "scratch_demos": demos_rel if method != "bc" else None,
                "trained_pairs_sha256": live["trained_pairs_sha256"],
                "records": live["records"],
                "accounting": accounting(),
                **capacity_fields,
            },
            spec,
        )
    except classical.Refused as exc:
        return _refused(exc)
    live["state"] = state

    if capacity is None:
        factory = functools.partial(
            restriction.make_linear_policy,
            restriction_id=restriction_id,
        )
    else:
        # Initialization comes from the run seed's named streams only; no
        # global RNG advances, whatever the width.
        factory = functools.partial(
            capacity_policy.make_capacity_policy,
            hidden_width=capacity["student_width"],
            initialization_seed=seed,
            teacher_hidden_width=capacity["teacher_width"],
        )

    def relative(path):
        return os.path.relpath(path, output_dir)

    def bound_checkpoint(path, binding, round_num):
        fields = _verified_checkpoint(
            path,
            capacity,
            binding,
            run_experiment._policy_path(config, round_num),
        )
        fields["checkpoint_binding"]["path"] = relative(path)
        return fields

    def append(row):
        live["records"].append(row)
        writer.write(force=row.get("d_eval_size") is not None)

    def on_round(record, timing):
        row = dict(record)
        if "checkpoint" in row:
            if capacity is not None:
                row.update(
                    bound_checkpoint(
                        row["checkpoint"],
                        row.get("checkpoint_binding"),
                        row["round"],
                    ),
                )
            row["checkpoint"] = relative(row["checkpoint"])
        row["wall_seconds"] = dict(timing)
        t = int(row["n_observations"])
        if method == "bc_iid" and t:
            lengths = shared_entry["stats"]["episode_lengths"]
            row["logical"] = {"labels": t, "episodes": t, "env_steps": sum(lengths[:t])}
            if row.get("d_eval_size") is not None:
                row["prefix_pairs_sha256"] = shared.pairs_sha256(t)
        elif prefix and t:
            lengths = shared_entry["stats"]["episode_lengths"]
            row["logical"] = pool_logical_cost(lengths, t)
            if row.get("d_eval_size") is not None:
                row["prefix_pairs_sha256"] = shared.pairs_sha256(t)
        append(row)
        if row["round"] < n_rounds:
            state["in_flight"] = (
                f"round {row['round'] + 1}: collection, training and evaluation"
            )
            guard.check("starting the next round")
        else:
            state["in_flight"] = "reading back the trained demonstrations"

    def run_fixed_bc(venv):
        # One offline fit on the whole pool, as in run_experiment._run_bc.
        lengths = shared_entry["stats"]["episode_lengths"]
        transitions = shared.transitions(n_rounds)
        trained = pairs_sha256(transitions.obs, transitions.acts)
        if trained != shared_entry["pairs_sha256"]:
            raise RuntimeError("Fixed BC would train on data other than the pool")
        state["in_flight"] = "fixed BC fit on the full pool"
        guard.check("fixed BC fit on the full pool")
        th.manual_seed(seed)
        fit_start = time.monotonic()
        trainer, inner_log = run_experiment._fit_fixed_bc(
            config,
            venv,
            factory(expert),
            transitions,
            np.random.default_rng(seed),
            device=DEVICE,
            tb_tag="bc_full_pool",
        )
        state["in_flight"] = "fixed BC evaluation"
        eval_start = time.monotonic()
        evaluation = run_experiment._compute_round_eval(
            trainer.policy,
            expert,
            venv,
            baselines,
            n_episodes=run_experiment._eval_episodes(config, final=True),
            **run_experiment._paired_eval_kwargs(config, n_rounds),
        )
        eval_end = time.monotonic()
        saved = run_experiment._checkpoint_fields(config, trainer.policy, n_rounds)
        checkpoint = saved["checkpoint"]
        live["trained_pairs_sha256"] = trained
        verified = (
            {}
            if capacity is None
            else bound_checkpoint(
                checkpoint,
                saved.get("checkpoint_binding"),
                n_rounds,
            )
        )
        append(
            {
                "round": 0,
                "n_observations": n_rounds,
                "reference": "flat",
                "trained_pairs_sha256": trained,
                "logical": pool_logical_cost(lengths, n_rounds),
                **inner_log,
                **evaluation,
                **verified,
                "checkpoint": relative(checkpoint),
                "wall_seconds": {
                    "fit": eval_start - fit_start,
                    "eval": eval_end - eval_start,
                },
            },
        )

    def body():
        start = time.monotonic()
        state["in_flight"] = "setup"
        run_experiment._seed_everything(seed)
        rng = np.random.default_rng(seed)
        venv = CountingVecEnv(env_utils.make_env(spec.env_name, 1, rng))
        live["venv"] = venv
        try:
            _check_cap(venv, spec)
            th.manual_seed(seed)
            live["setup_seconds"] = time.monotonic() - start
            if method == "bc":
                run_fixed_bc(venv)
            else:
                state["in_flight"] = "round 0 evaluation"
                run_experiment._run_dagger_variant(
                    config,
                    venv,
                    expert,
                    rng,
                    baselines,
                    device=DEVICE,
                    policy_factory=factory,
                    offline_data=(
                        shared.transitions(n_rounds) if shared_name else None
                    ),
                    offline_source=(
                        {"collection_steps": shared_entry["stats"]["env_steps"]}
                        if shared_name
                        else None
                    ),
                    round_callback=on_round,
                )
                obs, acts = trained_pairs(output_dir / demos_rel)
                live["trained_pairs_sha256"] = pairs_sha256(obs, acts)
                if shared_name and live["trained_pairs_sha256"] != (
                    shared.pairs_sha256(n_rounds)
                ):
                    name = "BC-prefix" if prefix else "BC-iid"
                    raise RuntimeError(
                        f"{name} trained on data other than the {shared_name}",
                    )
        finally:
            venv.close()
            live["total_seconds"] = time.monotonic() - start

    return classical._run_lifecycle(state, writer, body, guard)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m imitation.experiments.agnostic.restriction_pilot",
        description="CartPole learner-only restriction pilot (approved scope).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--preparation-dir", type=pathlib.Path, required=True)
        p.add_argument("--output-dir", type=pathlib.Path, required=True)
        p.add_argument("--deadline", type=rollouts.parse_utc_deadline, required=True)
        p.add_argument("--job-limit-seconds", type=float, required=True)

    common(sub.add_parser("audit", help="paired-state mask audit (seed 301)"))
    data = sub.add_parser("data", help="shared expert data and baselines")
    common(data)
    data.add_argument("--seed", type=int, required=True)
    run = sub.add_parser("run", help="one (method, restriction) run")
    common(run)
    run.add_argument("--data-dir", type=pathlib.Path, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--method", choices=METHODS, required=True)
    run.add_argument("--restriction", choices=RUN_RESTRICTIONS, required=True)
    run.add_argument(
        "--expected-data-sha256",
        default=None,
        help="SHA256 of the data job's result.json bytes; required when that "
        "record was produced by a different source version",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "seed", None) == AUDIT_SEED:
        parser.error(f"seed {AUDIT_SEED} is reserved for the mask audit")
    common = dict(deadline=args.deadline, job_limit_seconds=args.job_limit_seconds)
    if args.command == "audit":
        code = run_audit(args.output_dir, args.preparation_dir, **common)
    elif args.command == "data":
        code = prepare_data(
            args.output_dir,
            args.preparation_dir,
            seed=args.seed,
            **common,
        )
    else:
        code = run_job(
            args.output_dir,
            args.preparation_dir,
            args.data_dir,
            method=args.method,
            restriction_id=args.restriction,
            seed=args.seed,
            expected_data_sha256=args.expected_data_sha256,
            **common,
        )
    status = "refused"
    if code != classical.EXIT_REFUSED:
        status = pilot.read_json(args.output_dir / RESULT_FILE).get("status")
    print(json.dumps({"status": status, "exit_code": code}))
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
