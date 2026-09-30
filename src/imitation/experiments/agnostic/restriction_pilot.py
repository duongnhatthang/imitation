"""Runner for the approved CartPole learner-only restriction pilot.

Three commands, each writing one fresh output directory:

``audit``
    Paired-state mask audit of ``cart_position_zero`` with the qualified
    expert on the diagnostic reset seed `AUDIT_SEED` (see
    `restriction.audit_cartpole_position`). Its labels never reach training.
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
    One (method, restriction) run on the pilot seed. The learner is the
    previous pipeline's linear policy (expert clone, frozen features,
    reinitialized ``action_net``) built by `restriction.make_linear_policy`, so
    the restriction acts before the frozen features on every path while the
    expert keeps reading full observations. FTL (beta 0) and BC-iid use
    ``run_experiment._run_dagger_variant`` unchanged except for its explicit
    hooks: FTL collects with learner control as before; BC-iid replays the
    shared stream in order, one state per round, through the same trainer.
    Fixed BC cold-fits, at every plotted budget B, the first B pool
    transitions with ``run_experiment._fit_fixed_bc`` (the routine of
    ``_run_bc``), reseeding torch before each fit so every point is the
    standalone fit at B. Evaluation is ``_compute_round_eval`` (100
    deterministic episodes, normalized return, learner and expert rollout
    cross-entropy, disagreement, raw episode returns); evaluated policies are
    checkpointed with their restriction. No mixture is computed.

Differences from the previous pipeline, all deliberate: expert data for fixed
BC and BC-iid come from the shared data job instead of the per-run
environment, so full and restricted runs see identical data (verified by
hash); baselines are measured once per data job on a dedicated environment
whose random-action sampler is seeded; the expert comes from a verified
qualified preparation rather than the expert cache.

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
from imitation.experiments.agnostic import classical, pilot, restriction, rollouts
from imitation.experiments.ftrl import (
    env_baselines,
    env_utils,
    eval_utils,
    run_experiment,
)

logger = logging.getLogger(__name__)

PROTOCOL = "agnostic-cartpole-restriction-pilot/1"
AUDIT_SCHEMA = PROTOCOL + "/audit"
DATA_SCHEMA = PROTOCOL + "/data"
RUN_SCHEMA = PROTOCOL + "/run"

ENV_NAME = "CartPole-v1"
EPISODE_CAP = 500
PILOT_SEED = 300
AUDIT_SEED = 301
METHODS = ("ftl", "bc", "bc_iid")
RESTRICTION_IDS = tuple(restriction.RESTRICTIONS)
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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def pairs_sha256(obs: np.ndarray, acts: np.ndarray) -> str:
    """Digest of supervised (observation, expert action) pairs, in order.

    Observations are hashed as float32 and actions as int64 with their
    shapes, so the digest identifies training data independently of storage.
    """
    obs = np.ascontiguousarray(obs, dtype=np.float32)
    acts = np.ascontiguousarray(acts, dtype=np.int64)
    if len(obs) != len(acts):
        raise ValueError("obs and acts differ in length")
    digest = hashlib.sha256()
    for name, array in (("obs", obs), ("acts", acts)):
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


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
    venv = env_utils.make_env(ENV_NAME, 1, _seed_stream(seed, _POOL_RESETS))
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
    venv = env_utils.make_env(ENV_NAME, 1, _seed_stream(seed, _STREAM_RESETS))
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


def measure_baselines(
    expert,
    seed: int,
    n_episodes: int,
    check: Callable[[str], None] = _no_check,
    observer: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Normalization baselines as in ``env_baselines.compute_baselines``.

    Runs the same two routines (deterministic expert rollout with self
    cross-entropy, then uniform random actions) on a dedicated environment,
    with the random-action sampler seeded from a dedicated stream. If given,
    ``observer["venv"]`` receives the counting environment, so an interrupted
    measurement still reports its observed steps.

    Returns:
        The baselines and their counted costs.
    """
    start = time.monotonic()
    venv = CountingVecEnv(
        env_utils.make_env(ENV_NAME, 1, _seed_stream(seed, _BASELINE_ENV))
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
    """Budgets the round loop evaluates: round 1, every interval, the last."""
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


def source_identity() -> Dict[str, Any]:
    """Digests of the source files that determine pilot behavior."""
    files = {}
    for rel in _SOURCE_FILES:
        path = _PACKAGE_ROOT / rel
        files[rel] = pilot.sha256_file(path) if path.is_file() else "missing"
    status = pilot._git("status", "--porcelain")
    return {
        "files": files,
        "combined_sha256": pilot.config_digest(files),
        "git_revision": pilot._git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def _validate_limits(seed: Optional[int], job_limit_seconds: float) -> None:
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise classical.Refused(f"seed must be a non-negative integer: {seed!r}")
        if seed == AUDIT_SEED:
            raise classical.Refused(f"seed {AUDIT_SEED} is reserved for the mask audit")
    limit = job_limit_seconds
    if (
        isinstance(limit, bool)
        or not isinstance(limit, (int, float))
        or not math.isfinite(limit)
        or not 0 < limit <= MAX_JOB_LIMIT_SECONDS
    ):
        raise classical.Refused(
            f"job_limit_seconds must be in (0, {MAX_JOB_LIMIT_SECONDS}]: {limit!r}",
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


def _load_expert(preparation_dir) -> Tuple[Any, Dict[str, Any]]:
    model, identity = classical.load_verified_expert(
        pathlib.Path(preparation_dir),
        ENV_NAME,
    )
    identity = dict(identity)
    # Keep machine paths out of records: the digests identify the expert.
    identity["preparation_dir_name"] = pathlib.Path(
        identity.pop("preparation_dir"),
    ).name
    return model.policy, identity


def _check_cap(venv=None) -> int:
    cap = gym.spec(ENV_NAME).max_episode_steps
    if venv is not None:
        cap_env = venv.get_attr("spec")[0].max_episode_steps
        if cap_env != cap:
            raise RuntimeError(f"Environment cap {cap_env} differs from spec {cap}")
    if cap != EPISODE_CAP:
        raise classical.Refused(f"{ENV_NAME} time limit {cap} is not {EPISODE_CAP}")
    return int(cap)


def _base_record(schema, identity, limits, clock) -> Dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "schema": schema,
        "env_name": ENV_NAME,
        "episode_cap": EPISODE_CAP,
        "expert_sha256": identity["expert_sha256"],
        "preparation": identity,
        "deadline": limits,
        "source": source_identity(),
        "package_versions": pilot.package_versions(),
        "started_at_utc": clock().isoformat(),
    }


def _start(output_dir, schema, identity, limits, guard, render_extra):
    """Claim the output and build lifecycle state and writer."""
    output_dir = pathlib.Path(output_dir)
    claim = classical.claim_fresh_output_dir(output_dir, guard.clock)
    base = _base_record(schema, identity, limits, guard.clock)
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
                    "restriction_id": restriction.AUDIT_RESTRICTION,
                    "reset_seed": AUDIT_SEED,
                    "grid": list(restriction.AUDIT_X_GRID),
                    "base_steps": list(restriction.AUDIT_BASE_STEPS),
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
) -> int:
    """Acquire the shared fixed-BC pool, BC-iid stream and baselines once."""
    output_dir = pathlib.Path(output_dir)
    try:
        _validate_limits(seed, job_limit_seconds)
        if int(budget) <= 0 or int(baseline_episodes) <= 0:
            raise classical.Refused("budget and baseline_episodes must be positive")
        _check_cap()
        expert, identity = _load_expert(preparation_dir)
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
            "pool": "complete expert episodes until >= budget transitions; "
            "first budget kept in order",
            "stream": "budget independent complete expert episodes; one "
            "uniformly selected pre-action state per episode",
            "expert_action_rule": classical.ACTION_RULE,
        }
        state, writer = _start(
            output_dir,
            DATA_SCHEMA,
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
        )
        store("pool", POOL_FILE, pool, pool_stats)
        state["in_flight"] = "BC-iid stream collection"
        stream, stream_stats = collect_iid_stream(
            expert,
            budget,
            seed,
            check=guard.check,
            counters=live["counters"]["stream"],
        )
        store("stream", STREAM_FILE, stream, stream_stats)
        state["in_flight"] = "baseline measurement"
        live["baselines"], live["baselines_costs"] = measure_baselines(
            expert,
            seed,
            int(baseline_episodes),
            check=guard.check,
            observer=live["baseline_observer"],
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


def _verify_data(data_dir, seed, identity, n_rounds):
    """Check the shared data job and return its record, data and file digest."""
    data_dir = pathlib.Path(data_dir)
    path = data_dir / RESULT_FILE
    try:
        record = pilot.read_json(path)
        record_sha = pilot.sha256_file(path)
    except (OSError, ValueError) as exc:
        raise classical.Refused(f"Unreadable data record {path}: {exc}")
    checks = {
        "schema": record.get("schema") == DATA_SCHEMA,
        "status_complete": record.get("status") == "complete",
        "env_name": record.get("env_name") == ENV_NAME,
        "episode_cap": record.get("episode_cap") == EPISODE_CAP,
        "seed": record.get("seed") == seed,
        "expert_sha256": record.get("expert_sha256") == identity["expert_sha256"],
        "policy_state_sha256": (record.get("preparation") or {}).get(
            "policy_state_sha256",
        )
        == identity["policy_state_sha256"],
        "baselines": isinstance(record.get("baselines"), dict),
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    if failed:
        raise classical.Refused(f"Data record {path} failed checks {failed}")
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
        if len(data) < n_rounds:
            raise classical.Refused(f"{name} holds {len(data)} < {n_rounds} labels")
        datasets[name] = data
    return record, datasets, record_sha


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _experiment_config(method, seed, n_rounds, eval_interval, output_dir):
    return run_experiment.ExperimentConfig(
        algo=method,
        env_name=ENV_NAME,
        seed=seed,
        policy_mode="linear",
        n_rounds=n_rounds,
        samples_per_round=1,
        l2_lambda=0.0,
        l2_decay=False,
        warm_start=False,
        beta_rampdown=0,
        bc_n_epochs=BC_EPOCHS,
        eval_interval=eval_interval,
        output_dir=pathlib.Path(output_dir),
        expert_cache_dir=pathlib.Path(output_dir) / "unused-expert-cache",
        learning_rate=LEARNING_RATE,
        outer_early_stop=False,
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
    clock: Callable[[], datetime.datetime] = rollouts.utc_now,
) -> int:
    """Run one (method, restriction) pilot job into a fresh directory."""
    output_dir = pathlib.Path(output_dir).absolute()
    try:
        _validate_limits(seed, job_limit_seconds)
        if method not in METHODS:
            raise classical.Refused(f"method must be one of {METHODS}: {method!r}")
        try:
            restriction.get_restriction(restriction_id)
        except ValueError as exc:
            raise classical.Refused(str(exc))
        if int(n_rounds) <= 0 or int(eval_interval) <= 0:
            raise classical.Refused("n_rounds and eval_interval must be positive")
        _check_cap()
        expert, identity = _load_expert(preparation_dir)
        data_record, datasets, data_sha = _verify_data(
            data_dir,
            seed,
            identity,
            n_rounds,
        )
        guard, limits = _deadlines(deadline, job_limit_seconds, clock)
    except classical.Refused as exc:
        return _refused(exc)

    config = _experiment_config(method, seed, n_rounds, eval_interval, output_dir)
    experiment = {
        k: v
        for k, v in dataclasses.asdict(config).items()
        if k not in ("output_dir", "expert_cache_dir", "result_name_override")
    }
    shared_name = {"bc": "pool", "bc_iid": "stream"}.get(method)
    shared = datasets.get(shared_name)
    shared_entry = data_record["datasets"][shared_name] if shared_name else None
    run_config = {
        "seed": seed,
        "method": method,
        "restriction_id": restriction_id,
        "restriction": dataclasses.asdict(restriction.get_restriction(restriction_id)),
        "policy": "linear: expert clone, frozen features, reinitialized action_net; "
        "restriction applied before the frozen features",
        "beta": {"ftl": 0.0, "bc_iid": 1.0, "bc": None}[method],
        "eval_episodes": 100,
        "eval_deterministic": True,
        "eval_budgets": eval_budgets(n_rounds, eval_interval),
        "mixture": False,
        "experiment": experiment,
    }
    demos_rel = f"scratch/{method}_{ENV_NAME}_seed{seed}/demos"
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
                },
                "retained_labels": int(records[-1]["n_observations"]) if records else 0,
                "training": {
                    "fits": len(fits),
                    "epochs": sum(int(r["inner_es_stop_epoch"]) for r in fits),
                    "early_stopped_fits": sum(
                        int(r["inner_es_stop_epoch"]) < BC_EPOCHS for r in fits
                    ),
                    "fixed_budget_fallback_fits": sum(
                        r.get("inner_es_fallback") is not None for r in fits
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
            RUN_SCHEMA,
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
                },
                "scratch_demos": demos_rel if method != "bc" else None,
                "trained_pairs_sha256": live["trained_pairs_sha256"],
                "records": live["records"],
                "accounting": accounting(),
            },
        )
    except classical.Refused as exc:
        return _refused(exc)
    live["state"] = state

    baselines = data_record["baselines"]
    factory = functools.partial(
        restriction.make_linear_policy,
        restriction_id=restriction_id,
    )

    def relative(path):
        return os.path.relpath(path, output_dir)

    def append(row):
        live["records"].append(row)
        writer.write(force=row.get("d_eval_size") is not None)

    def on_round(record, timing):
        row = dict(record)
        if "checkpoint" in row:
            row["checkpoint"] = relative(row["checkpoint"])
        row["wall_seconds"] = dict(timing)
        t = int(row["n_observations"])
        if method == "bc_iid" and t:
            lengths = shared_entry["stats"]["episode_lengths"]
            row["logical"] = {"labels": t, "episodes": t, "env_steps": sum(lengths[:t])}
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
        lengths = shared_entry["stats"]["episode_lengths"]
        for b in eval_budgets(n_rounds, eval_interval):
            state["in_flight"] = f"fixed BC fit at budget {b}"
            guard.check(f"fixed BC fit at budget {b}")
            th.manual_seed(seed)
            fit_start = time.monotonic()
            trainer, inner_log = run_experiment._fit_fixed_bc(
                config,
                venv,
                factory(expert),
                shared.transitions(b),
                np.random.default_rng(seed),
                device=DEVICE,
                tb_tag=f"bc_budget{b:05d}",
            )
            state["in_flight"] = f"fixed BC evaluation at budget {b}"
            eval_start = time.monotonic()
            evaluation = run_experiment._compute_round_eval(
                trainer.policy,
                expert,
                venv,
                baselines,
            )
            eval_end = time.monotonic()
            checkpoint = run_experiment._save_policy(config, trainer.policy, b)
            append(
                {
                    "round": b,
                    "n_observations": b,
                    "prefix_pairs_sha256": shared.pairs_sha256(b),
                    "logical": pool_logical_cost(lengths, b),
                    **inner_log,
                    **evaluation,
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
        venv = CountingVecEnv(env_utils.make_env(ENV_NAME, 1, rng))
        live["venv"] = venv
        try:
            _check_cap(venv)
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
                        shared.transitions(n_rounds) if method == "bc_iid" else None
                    ),
                    offline_source=(
                        {"collection_steps": shared_entry["stats"]["env_steps"]}
                        if method == "bc_iid"
                        else None
                    ),
                    round_callback=on_round,
                )
                obs, acts = trained_pairs(output_dir / demos_rel)
                live["trained_pairs_sha256"] = pairs_sha256(obs, acts)
                if method == "bc_iid" and live["trained_pairs_sha256"] != (
                    shared.pairs_sha256(n_rounds)
                ):
                    raise RuntimeError("BC-iid trained on data other than the stream")
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
    run.add_argument("--restriction", choices=RESTRICTION_IDS, required=True)
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
            **common,
        )
    status = "refused"
    if code != classical.EXIT_REFUSED:
        status = pilot.read_json(args.output_dir / RESULT_FILE).get("status")
    print(json.dumps({"status": status, "exit_code": code}))
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
