"""Stage 2 classical agnostic experiment: alias audit and learning cells.

``audit`` estimates, on an independent reference distribution, a lower bound
on the disagreement of every table over each predeclared quantizer. Each
reference episode is driven, with probability 1/2 each, by the frozen
deterministic expert or by uniform random actions, and contributes one
uniformly selected pre-action observation labelled by the expert. Its labels
are diagnostic only and never reach training.

``run-cell`` runs one (env, representation, seed) cell that jointly produces
FTL-DAgger (learner behavior, beta 0, deferred expert labels on selected
states only), BC-iid (expert behavior, beta 1, cached labels), and fixed BC
(the same exact fit on the shared BC-iid prefix, an equality control). All
arms use the same exact majority-table learner (`quantized.fit_table`). The
cell reports representation-restricted results; any misspecification
certificate comes separately from ``audit``.

Fit accounting: FTL and BC-iid each fit once per round. Fixed BC really fits
once more at every checkpoint; those fits and their time are physical training
costs (``train_fixed_bc``), which share no env or oracle cost with BC-iid. A
standalone offline fixed BC run at budget B would acquire what BC-iid acquired
up to B and fit once, which ``logical_standalone_costs`` reports.

Both records carry flat identity fields (``protocol``, an alias of ``schema``,
``env_name``, ``seed``, ``expert_sha256``, and for cells ``representation``)
that duplicate the nested config and preparation identity.

Estimand: every retained sample is one uniformly selected pre-action state of
one independent complete episode, so datasets target the episode-normalized
state distribution, not fixed-horizon or transition-weighted occupancy.

Both commands require a stage 1 preparation that is complete, qualified, and
approved, with a matching env, protocol, config digest, and checkpoint
digest. They load the frozen PPO on CPU and act with ``predict(deterministic=
True)``; nothing is retrained and the preparation is never modified.

Status and deadlines: output goes to a fresh directory claimed exclusively;
a non-empty directory is refused without reading its contents. The deadline
is clamped to `HARD_CAP` and checked before every reset, policy query, env
step, and expert query, and once more before marking a result complete.
``result.json`` is replaced atomically on progress. ``status == "complete"``
only for a fully finished run; ``partial`` (deadline) and ``failed`` keep
every completed acquisition and oracle spend. An external kill (for example
GNU ``timeout``) is the final cap and can lose progress since the last write;
such a record keeps ``status == "running"``, which is never complete.
"""

import collections
import dataclasses
import datetime
import hashlib
import io
import logging
import math
import os
import pathlib
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from imitation.experiments.agnostic import pilot, quantized, rollouts

logger = logging.getLogger(__name__)

UTC = datetime.timezone.utc
HARD_CAP = datetime.datetime(2026, 10, 7, 1, 1, tzinfo=UTC)

AUDIT_SCHEMA = "agnostic-classical-audit/1"
CELL_SCHEMA = "agnostic-classical-cell/1"
RESULT_FILE = "result.json"
CLAIM_FILE = "result.claim"
AUDIT_DATA_FILE = "audit_samples_diagnostic_only.npz"

EXIT_COMPLETE = 0
EXIT_INCOMPLETE = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3

DEFAULT_AUDIT_EPISODES = 4096
DEFAULT_DELTA = 0.05 / 6
DEFAULT_BUDGET = 4096
DEFAULT_BATCH = 16
DEFAULT_CHECKPOINTS = (128, 256, 512, 1024, 2048, 4096)
DEFAULT_EVAL_EPISODES = 100
PROGRESS_INTERVAL_SECONDS = 30.0

# Env reset seeds of different phases come from disjoint ranges.
_SEED_SPAN = 2**30
_SEED_OFFSETS = {"train": 0, "eval": _SEED_SPAN, "reference": 2 * _SEED_SPAN}
# Sub-stream keys within a phase.
_ENV_SEEDS, _SELECTION, _BEHAVIOR_CHOICE, _RANDOM_ACTIONS, _MIXTURE = range(5)

ACTION_RULE = "PPO.predict(obs, deterministic=True) on CPU"
EXPERT_INTERRUPTION_NOTE = (
    "Costs are counted when a query, reset, or step returns. An operation "
    "stopped by the deadline did not start. An external kill leaves at most "
    "the progress of the last atomic write."
)

_OWN_SOURCES = ("quantized.py", "classical.py")


class Refused(Exception):
    """Inputs or output location fail verification; nothing was run."""


class EqualityControlFailed(RuntimeError):
    """Fixed BC and BC-iid differ despite fitting the same data."""


# ---------------------------------------------------------------------------
# Deadlines, streams, and costs
# ---------------------------------------------------------------------------


def effective_deadline(requested: datetime.datetime) -> datetime.datetime:
    """Return ``requested`` as aware UTC, clamped to `HARD_CAP`."""
    return min(rollouts.ensure_aware_utc(requested), HARD_CAP)


class Guard:
    """Deadline check applied before every operation that spends resources."""

    def __init__(
        self,
        deadline: datetime.datetime,
        clock: Callable[[], datetime.datetime] = rollouts.utc_now,
    ):
        self.deadline = effective_deadline(deadline)
        self.clock = clock

    def check(self, what: str) -> None:
        rollouts.check_deadline(self.deadline, what, self.clock)


def stream(seed: int, phase: str, *keys: int) -> np.random.Generator:
    """Independent generator for a named phase and sub-stream keys."""
    spawn_key = (rollouts.PHASES.index(phase),) + tuple(int(k) for k in keys)
    return np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=spawn_key))


def reset_seeds(seed: int, phase: str, n: int) -> List[int]:
    """``n`` env reset seeds for ``phase``, one per episode."""
    draws = stream(seed, phase, _ENV_SEEDS).integers(0, _SEED_SPAN, size=n)
    return [int(x) + _SEED_OFFSETS[phase] for x in draws]


def selection_uniforms(seed: int, phase: str, n: int) -> np.ndarray:
    """One uniform per episode for choosing its retained time step."""
    return stream(seed, phase, _SELECTION).random(n)


def select_index(u: float, length: int) -> int:
    """Uniform time index ``floor(u * length)`` in ``[0, length)``."""
    return min(int(math.floor(u * length)), length - 1)


@dataclasses.dataclass
class Costs:
    """Resources actually spent. Counted only when an operation returns."""

    env_steps: int = 0
    env_resets: int = 0
    completed_episodes: int = 0
    interrupted_episodes: int = 0
    interrupted_episode_steps: int = 0
    behavior_predict_calls: int = 0
    behavior_action_entries: int = 0
    expert_predict_calls: int = 0
    expert_action_entries: int = 0
    retained_labels: int = 0
    fits: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def sum_costs(items: Sequence[Costs]) -> Dict[str, Any]:
    total = Costs()
    for item in items:
        for field in dataclasses.fields(Costs):
            setattr(
                total,
                field.name,
                getattr(total, field.name) + getattr(item, field.name),
            )
    return total.to_dict()


# ---------------------------------------------------------------------------
# Expert and episodes
# ---------------------------------------------------------------------------


class Expert:
    """Deterministic expert whose every returned query is counted."""

    def __init__(self, model, n_actions: int):
        self.model = model
        self.n_actions = int(n_actions)

    def label(self, observations: np.ndarray, costs: Costs) -> np.ndarray:
        """Return deterministic expert actions for a batch of observations."""
        batch = np.asarray(observations, dtype=np.float32)
        actions, _ = self.model.predict(batch, deterministic=True)
        actions = np.asarray(actions).reshape(-1)
        costs.expert_predict_calls += 1
        costs.expert_action_entries += int(actions.shape[0])
        if actions.shape[0] != batch.shape[0]:
            raise RuntimeError(f"Expert returned {actions.shape[0]} actions")
        if np.any(actions < 0) or np.any(actions >= self.n_actions):
            raise RuntimeError(f"Expert returned invalid actions {actions}")
        return actions.astype(np.int64)


Behavior = Callable[[np.ndarray], int]


def expert_behavior(expert: Expert, costs: Costs) -> Behavior:
    def act(obs):
        action = int(expert.label(obs[None], costs)[0])
        costs.behavior_predict_calls += 1
        costs.behavior_action_entries += 1
        return action

    return act


def table_behavior(
    table: np.ndarray,
    quantizer: quantized.Quantizer,
    costs: Costs,
) -> Behavior:
    """Learner behavior: its action depends on the observation only via its bin."""
    frozen = np.array(table, dtype=np.int64, copy=True)

    def act(obs):
        action = int(frozen[quantizer(obs)])
        costs.behavior_predict_calls += 1
        costs.behavior_action_entries += 1
        return action

    return act


def random_behavior(
    rng: np.random.Generator,
    n_actions: int,
    costs: Costs,
) -> Behavior:
    def act(obs):
        action = int(rng.integers(n_actions))
        costs.behavior_predict_calls += 1
        costs.behavior_action_entries += 1
        return action

    return act


@dataclasses.dataclass
class Episode:
    observations: List[np.ndarray]
    actions: List[int]
    episode_return: float

    @property
    def length(self) -> int:
        return len(self.actions)


def run_episode(
    env: gym.Env,
    reset_seed: int,
    act: Behavior,
    guard: Guard,
    costs: Costs,
    max_steps: int,
    what: str,
) -> Episode:
    """Run one complete episode, recording pre-action observations.

    Costs accrue as operations return; an interrupted episode adds its steps
    to the interrupted counters and yields no sample.

    Raises:
        rollouts.DeadlineExceeded: Before an operation that would start late.
        RuntimeError: If the episode exceeds ``max_steps``.
    """
    steps = 0
    try:
        guard.check(f"{what} reset")
        obs, _ = env.reset(seed=int(reset_seed))
        costs.env_resets += 1
        episode = Episode([], [], 0.0)
        while True:
            if steps >= max_steps:
                raise RuntimeError(f"{what} episode exceeded {max_steps} steps")
            guard.check(f"{what} action query")
            action = act(obs)
            guard.check(f"{what} env step")
            episode.observations.append(np.array(obs, dtype=np.float64, copy=True))
            episode.actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            costs.env_steps += 1
            episode.episode_return += float(reward)
            if terminated or truncated:
                break
    except BaseException:
        costs.interrupted_episodes += 1
        costs.interrupted_episode_steps += steps
        raise
    costs.completed_episodes += 1
    return episode


def make_env(env_name: str) -> gym.Env:
    """Single env with the registered TimeLimit."""
    return gym.make(env_name)


def _episode_cap(env: gym.Env) -> int:
    cap = env.spec.max_episode_steps if env.spec is not None else None
    if not cap:
        raise ValueError("Environment must have a finite TimeLimit")
    return int(cap)


# ---------------------------------------------------------------------------
# Preparation verification, output claim, and result writing
# ---------------------------------------------------------------------------


def load_verified_expert(
    preparation_dir: pathlib.Path,
    env_name: str,
) -> Tuple[Any, Dict[str, Any]]:
    """Load the frozen expert after verifying its stage 1 preparation.

    Returns:
        The PPO model and an identity record.

    Raises:
        Refused: If any check fails. Nothing is modified.
    """
    prep_dir = pathlib.Path(preparation_dir)
    record_path = prep_dir / pilot.PREPARATION_FILE
    try:
        record_bytes = record_path.read_bytes()
        record = pilot.read_json(record_path)
    except (OSError, ValueError) as exc:
        raise Refused(f"Unreadable preparation record {record_path}: {exc}")
    if not isinstance(record, dict) or not isinstance(record.get("config"), dict):
        raise Refused(f"{record_path} is not a preparation record")
    config = record["config"]
    expert_cfg = config.get("expert") or {}
    checks = {
        "status_complete": record.get("status") == pilot.COMPLETE_STATUS,
        "qualified": record.get("qualified") is True,
        "approved": record.get("approved") is True,
        "protocol": record.get("protocol") == pilot.PREPARATION_SCHEMA
        and config.get("schema") == pilot.PREPARATION_SCHEMA,
        "env_name": record.get("env_name") == env_name == config.get("env_name"),
        "seed": record.get("seed") == config.get("seed"),
        "cpu_expert": expert_cfg.get("device") == "cpu",
        "record_verified": pilot.verify_qualified_preparation(prep_dir, config),
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    if failed:
        raise Refused(f"Preparation {prep_dir} failed checks {failed}")
    checkpoint = record["checkpoint"]
    data = (prep_dir / checkpoint["path"]).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != checkpoint["sha256"]:
        raise Refused("Checkpoint bytes changed during verification")
    model = PPO.load(io.BytesIO(data), device="cpu")
    state_digest = pilot.policy_state_sha256(model.policy)
    if state_digest != checkpoint.get("policy_state_sha256"):
        raise Refused("Loaded policy parameters do not match the record")
    identity = {
        "preparation_dir": str(prep_dir.resolve()),
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "protocol": record["protocol"],
        "config_sha256": record["config_sha256"],
        "expert_sha256": digest,
        "policy_state_sha256": state_digest,
        "qualification_mean_return": (record.get("qualification") or {}).get(
            "mean_return",
        ),
        "action_rule": ACTION_RULE,
    }
    return model, identity


def claim_fresh_output_dir(
    output_dir: pathlib.Path,
    clock: Callable[[], datetime.datetime],
) -> Dict[str, Any]:
    """Exclusively claim a fresh output directory.

    A path that exists and is not an empty directory is refused without
    reading anything inside it.

    Raises:
        Refused: If the directory is not fresh or another invocation owns it.
    """
    output_dir = pathlib.Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise Refused(f"{output_dir} is not a fresh empty directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    claim = {"pid": os.getpid(), "claimed_at_utc": clock().isoformat()}
    try:
        fd = os.open(
            str(output_dir / CLAIM_FILE),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
    except FileExistsError:
        raise Refused(f"{output_dir} is already claimed") from None
    with os.fdopen(fd, "w") as f:
        f.write(pilot.canonical_json(claim) + "\n")
        f.flush()
        os.fsync(f.fileno())
    others = sorted(p.name for p in output_dir.iterdir() if p.name != CLAIM_FILE)
    if others:
        raise Refused(f"{output_dir} gained entries {others} before the claim")
    return dict(claim, path=CLAIM_FILE)


def source_identity() -> Dict[str, Any]:
    here = pathlib.Path(__file__).resolve().parent
    return {
        "stage1": pilot.source_fingerprint(),
        "stage2_files": {name: pilot.sha256_file(here / name) for name in _OWN_SOURCES},
    }


class ResultWriter:
    """Atomically rewrites ``result.json`` from a render function."""

    def __init__(self, path: pathlib.Path, render: Callable[[], Dict[str, Any]]):
        self.path = path
        self.render = render
        self.last = -math.inf

    def write(self, force: bool = True) -> None:
        now = time.monotonic()
        if force or now - self.last >= PROGRESS_INTERVAL_SECONDS:
            pilot.atomic_write_json(self.path, self.render())
            self.last = now


def _error(exc: BaseException) -> Dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__),
        ),
    }


def _run_lifecycle(
    state: Dict[str, Any],
    writer: ResultWriter,
    body: Callable[[], None],
    guard: Guard,
) -> int:
    """Run ``body`` and record a truthful final status.

    ``state`` must hold ``status``, ``in_flight``, and ``error`` keys used by
    the writer's render function.
    """
    state["status"] = "running"
    writer.write()
    try:
        body()
        state["in_flight"] = "final deadline check"
        guard.check("marking the result complete")
    except rollouts.DeadlineExceeded as exc:
        state["status"] = "partial"
        state["error"] = _error(exc)
    except Exception as exc:  # recorded, never silently discarded
        state["status"] = "failed"
        state["error"] = _error(exc)
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = _error(exc)
        state["finished_at_utc"] = guard.clock().isoformat()
        writer.write()
        raise
    else:
        state["status"] = "complete"
        state["in_flight"] = None
    state["finished_at_utc"] = guard.clock().isoformat()
    writer.write()
    return EXIT_COMPLETE if state["status"] == "complete" else EXIT_INCOMPLETE


def _interruption(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if state["status"] == "complete":
        return None
    return {
        "in_flight_operation": state.get("in_flight"),
        "note": EXPERT_INTERRUPTION_NOTE,
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def validate_audit_args(episodes: int, delta: float, seed: int) -> None:
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")


class AuditRun:
    """Collects the reference dataset shared by every quantizer of an env."""

    def __init__(
        self,
        env_name: str,
        *,
        seed: int,
        episodes: int,
        delta: float,
        expert: Expert,
        guard: Guard,
        env_factory: Callable[[], gym.Env],
    ):
        self.env_name = env_name
        self.seed = int(seed)
        self.episodes = int(episodes)
        self.delta = float(delta)
        self.expert = expert
        self.guard = guard
        self.env_factory = env_factory
        self.quantizers = [
            quantized.get_quantizer(env_name, rep) for rep in quantized.REPRESENTATIONS
        ]
        self.costs = {"expert_episodes": Costs(), "random_episodes": Costs()}
        self.samples: List[Dict[str, Any]] = []
        self.state: Dict[str, Any] = {"in_flight": None, "error": None}

    def collect(self, progress: Callable[[bool], None]) -> None:
        env = self.env_factory()
        try:
            cap = _episode_cap(env)
            seeds = reset_seeds(self.seed, "reference", self.episodes)
            uniforms = selection_uniforms(self.seed, "reference", self.episodes)
            use_expert = (
                stream(self.seed, "reference", _BEHAVIOR_CHOICE).random(self.episodes)
                < 0.5
            )
            for e in range(self.episodes):
                kind = "expert_episodes" if use_expert[e] else "random_episodes"
                costs = self.costs[kind]
                t0 = time.monotonic()
                self.state["in_flight"] = f"audit episode {e} ({kind})"
                try:
                    if use_expert[e]:
                        act = expert_behavior(self.expert, costs)
                    else:
                        rng = stream(self.seed, "reference", _RANDOM_ACTIONS, e)
                        act = random_behavior(rng, self.expert.n_actions, costs)
                    ep = run_episode(
                        env, seeds[e], act, self.guard, costs, cap, "audit"
                    )
                    i = select_index(uniforms[e], ep.length)
                    if use_expert[e]:
                        label = ep.actions[i]
                    else:
                        self.guard.check("audit expert label")
                        label = int(
                            self.expert.label(ep.observations[i][None], costs)[0]
                        )
                finally:
                    costs.elapsed_seconds += time.monotonic() - t0
                self.samples.append(
                    {
                        "obs": ep.observations[i],
                        "label": label,
                        "expert_behavior": bool(use_expert[e]),
                        "length": ep.length,
                        "index": i,
                        "reset_seed": seeds[e],
                    },
                )
                progress(False)
        finally:
            env.close()

    def summaries(self, complete: bool) -> Dict[str, Any]:
        out = {}
        for q in self.quantizers:
            bins = [q(s["obs"]) for s in self.samples]
            labels = [s["label"] for s in self.samples]
            counts = quantized.label_counts(
                bins, labels, q.n_bins, self.expert.n_actions
            )
            bound = quantized.alias_lower_bound(counts, self.delta)
            bound["positive_certificate"] = bool(
                complete and bound["positive_certificate"],
            )
            out[q.representation] = {
                "quantizer": q.describe(),
                "counts": counts.tolist(),
                "bound": bound,
                "claim": (
                    "positive misspecification lower bound on the reference "
                    "distribution"
                    if bound["positive_certificate"]
                    else "no certificate: misspecification unverified"
                ),
            }
        return out

    def save_samples(self, path: pathlib.Path) -> Optional[str]:
        if not self.samples:
            return None
        tmp = path.with_name(f".{path.name}.tmp")
        with open(tmp, "wb") as f:
            np.savez(
                f,
                obs=np.stack([s["obs"] for s in self.samples]),
                label=np.array([s["label"] for s in self.samples], dtype=np.int64),
                expert_behavior=np.array(
                    [s["expert_behavior"] for s in self.samples],
                ),
                length=np.array([s["length"] for s in self.samples], dtype=np.int64),
                index=np.array([s["index"] for s in self.samples], dtype=np.int64),
                reset_seed=np.array(
                    [s["reset_seed"] for s in self.samples],
                    dtype=np.int64,
                ),
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return pilot.sha256_file(path)


def audit(
    env_name: str,
    preparation_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    seed: int,
    deadline: datetime.datetime,
    episodes: int = DEFAULT_AUDIT_EPISODES,
    delta: float = DEFAULT_DELTA,
    clock: Callable[[], datetime.datetime] = rollouts.utc_now,
    env_factory: Optional[Callable[[], gym.Env]] = None,
) -> int:
    """Run the alias audit for both quantizers of ``env_name``.

    Returns:
        Exit code: complete, incomplete (partial or failed), or refused.

    Raises:
        ValueError: On invalid arguments, before anything is written.
    """
    validate_audit_args(episodes, delta, seed)
    quantized.get_quantizer(env_name, "severe")
    guard = Guard(deadline, clock)
    try:
        model, identity = load_verified_expert(preparation_dir, env_name)
        claim = claim_fresh_output_dir(output_dir, clock)
    except Refused as exc:
        return _refuse(exc)
    output_dir = pathlib.Path(output_dir)
    factory = env_factory or (lambda: make_env(env_name))
    n_actions = int(model.action_space.n)
    identities = {
        "package_versions": pilot.package_versions(),
        "source": source_identity(),
    }
    run = AuditRun(
        env_name,
        seed=seed,
        episodes=episodes,
        delta=delta,
        expert=Expert(model, n_actions),
        guard=guard,
        env_factory=factory,
    )
    config = {
        "schema": AUDIT_SCHEMA,
        "env_name": env_name,
        "seed": int(seed),
        "episodes": int(episodes),
        "delta": float(delta),
        "reference_mixture": {
            "expert_episode_probability": 0.5,
            "random_episode_probability": 0.5,
            "choice": "independent Bernoulli(0.5) per episode, whole episode",
            "sampling": "one uniform pre-action observation per complete episode",
            "labels": "deterministic expert; cached executed action on expert "
            "episodes, one query at the selected state on random episodes",
        },
        "quantizers": [q.describe() for q in run.quantizers],
        "preparation_config_sha256": identity["config_sha256"],
        "expert_sha256": identity["expert_sha256"],
    }
    state = run.state
    started = clock().isoformat()

    def render() -> Dict[str, Any]:
        complete = state.get("status") == "complete"
        return {
            "schema": AUDIT_SCHEMA,
            "protocol": AUDIT_SCHEMA,
            "env_name": env_name,
            "seed": int(seed),
            "expert_sha256": identity["expert_sha256"],
            "status": state.get("status"),
            "claim": claim,
            "config": config,
            "config_sha256": pilot.config_digest(config),
            "preparation": identity,
            "requested_deadline_utc": rollouts.ensure_aware_utc(deadline).isoformat(),
            "effective_deadline_utc": guard.deadline.isoformat(),
            "started_at_utc": started,
            "finished_at_utc": state.get("finished_at_utc"),
            "completed_samples": len(run.samples),
            "diagnostic_only": "Audit labels are never supplied to training.",
            "gate": "scientific: a representation is certified misspecified only "
            "for a positive lower bound from a complete audit",
            "representations": run.summaries(complete),
            "costs": {k: v.to_dict() for k, v in run.costs.items()},
            "total_costs": sum_costs(list(run.costs.values())),
            "samples_file": state.get("samples_file"),
            **identities,
            "interruption": _interruption(state),
            "error": state.get("error"),
        }

    writer = ResultWriter(output_dir / RESULT_FILE, render)

    def body() -> None:
        try:
            run.collect(lambda force: writer.write(force))
        finally:
            digest = run.save_samples(output_dir / AUDIT_DATA_FILE)
            state["samples_file"] = (
                {"path": AUDIT_DATA_FILE, "sha256": digest} if digest else None
            )

    return _run_lifecycle(state, writer, body, guard)


def _refuse(exc: Refused) -> int:
    logger.error("Refused: %s", exc)
    return EXIT_REFUSED


# ---------------------------------------------------------------------------
# run-cell
# ---------------------------------------------------------------------------


def validate_cell_args(
    representation: str,
    seed: int,
    budget: int,
    batch: int,
    checkpoints: Sequence[int],
    eval_episodes: int,
) -> List[int]:
    """Return the sorted checkpoints after validating the cell schedule."""
    if representation not in quantized.REPRESENTATIONS:
        raise ValueError(f"representation must be one of {quantized.REPRESENTATIONS}")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    if int(budget) <= 0 or int(batch) <= 0 or int(budget) % int(batch):
        raise ValueError("budget and batch must be positive, budget divisible by batch")
    points = sorted(int(c) for c in checkpoints)
    if not points or len(set(points)) != len(points):
        raise ValueError("checkpoints must be non-empty and unique")
    if any(c <= 0 or c % int(batch) or c > int(budget) for c in points):
        raise ValueError(
            "each checkpoint must be a positive multiple of batch <= budget"
        )
    if points[-1] != int(budget):
        raise ValueError("the last checkpoint must equal the budget")
    if int(eval_episodes) <= 0:
        raise ValueError("eval_episodes must be positive")
    return points


class CellRun:
    """State of one learning cell; `render` reports it at any moment."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        expert: Expert,
        quantizer: quantized.Quantizer,
        guard: Guard,
        env_factory: Callable[[], gym.Env],
    ):
        self.cfg = cfg
        self.expert = expert
        self.q = quantizer
        self.guard = guard
        self.env_factory = env_factory
        k, a = quantizer.n_bins, expert.n_actions
        # train_fixed_bc holds only fixed BC's own fits; its data is BC-iid's.
        self.costs = {
            "train_ftl": Costs(),
            "train_bc_iid": Costs(),
            "train_fixed_bc": Costs(),
            "eval": Costs(),
        }
        self.phase_seconds: Dict[str, float] = collections.defaultdict(float)
        self.counts = {
            "ftl": np.zeros((k, a), np.int64),
            "bc_iid": np.zeros((k, a), np.int64),
        }
        # Data-free initial table: every bin takes action 0.
        self.tables = {"ftl": np.zeros(k, np.int64), "bc_iid": np.zeros(k, np.int64)}
        self.data: Dict[str, Dict[str, List[int]]] = {
            arm: {"bins": [], "labels": [], "reset_seeds": [], "lengths": []}
            for arm in ("ftl", "bc_iid")
        }
        self.ftl_behavior_tables: List[List[int]] = []
        self.checkpoints: List[Dict[str, Any]] = []
        self.rounds_completed = 0
        self.state: Dict[str, Any] = {"in_flight": None, "error": None}

        seed, budget, n_eval = cfg["seed"], cfg["budget"], cfg["eval_episodes"]
        self.train_seeds = reset_seeds(seed, "train", budget)
        self.train_uniforms = selection_uniforms(seed, "train", budget)
        self.eval_seeds = reset_seeds(seed, "eval", n_eval)

    # -- training -----------------------------------------------------------

    def _collect(self, arm: str, env: gym.Env, cap: int, j: int) -> None:
        costs = self.costs[f"train_{arm}"]
        self.state["in_flight"] = f"train {arm} episode {j}"
        t0 = time.monotonic()
        try:
            if arm == "ftl":
                act = table_behavior(self.tables["ftl"], self.q, costs)
            else:
                act = expert_behavior(self.expert, costs)
            ep = run_episode(env, self.train_seeds[j], act, self.guard, costs, cap, arm)
            i = select_index(self.train_uniforms[j], ep.length)
            if arm == "ftl":
                # Deferred annotation: only the selected state is labelled.
                self.guard.check("deferred FTL expert annotation")
                label = int(self.expert.label(ep.observations[i][None], costs)[0])
            else:
                label = ep.actions[i]
        finally:
            elapsed = time.monotonic() - t0
            costs.elapsed_seconds += elapsed
            self.phase_seconds[f"{arm}_collect"] += elapsed
        bin_id = self.q(ep.observations[i])
        self.counts[arm][bin_id, label] += 1
        record = self.data[arm]
        record["bins"].append(bin_id)
        record["labels"].append(label)
        record["reset_seeds"].append(self.train_seeds[j])
        record["lengths"].append(ep.length)
        costs.retained_labels += 1

    def _fit(self, arm: str) -> None:
        self.state["in_flight"] = f"{arm} fit"
        self.guard.check(f"{arm} fit")
        t0 = time.monotonic()
        self.tables[arm] = quantized.fit_table(self.counts[arm].copy())
        elapsed = time.monotonic() - t0
        costs = self.costs[f"train_{arm}"]
        costs.fits += 1
        costs.elapsed_seconds += elapsed
        self.phase_seconds[f"{arm}_fit"] += elapsed

    def _fit_fixed_bc(self, n: int) -> Tuple[np.ndarray, float]:
        """Fit fixed BC on the first ``n`` BC-iid samples; return table, seconds."""
        self.state["in_flight"] = f"fixed_bc fit B={n}"
        self.guard.check("fixed_bc fit")
        prefix = self.data["bc_iid"]
        t0 = time.monotonic()
        table = quantized.fit_table(
            quantized.label_counts(
                prefix["bins"][:n],
                prefix["labels"][:n],
                self.q.n_bins,
                self.expert.n_actions,
            ),
        )
        elapsed = time.monotonic() - t0
        costs = self.costs["train_fixed_bc"]
        costs.fits += 1
        costs.elapsed_seconds += elapsed
        self.phase_seconds["fixed_bc_fit"] += elapsed
        return table, elapsed

    def run(self, progress: Callable[[bool], None]) -> None:
        batch = self.cfg["batch"]
        env = self.env_factory()
        try:
            cap = _episode_cap(env)
            for r in range(self.cfg["budget"] // batch):
                # Behavior policy of round r, before its update (pi_0 included).
                self.ftl_behavior_tables.append(self.tables["ftl"].tolist())
                for arm in ("ftl", "bc_iid"):
                    for j in range(r * batch, (r + 1) * batch):
                        self._collect(arm, env, cap, j)
                    self._fit(arm)
                self.rounds_completed = r + 1
                n = self.rounds_completed * batch
                if n in self.cfg["checkpoints"]:
                    self._checkpoint(env, cap, n)
                    progress(True)
                else:
                    progress(False)
        finally:
            env.close()

    # -- evaluation ---------------------------------------------------------

    def _evaluate(
        self,
        env: gym.Env,
        cap: int,
        tables: Sequence[np.ndarray],
        name: str,
    ) -> Dict[str, Any]:
        costs = self.costs["eval"]
        job = Costs()
        per_episode: Dict[str, List[Any]] = {
            "returns": [],
            "lengths": [],
            "disagreement": [],
        }
        t0 = time.monotonic()
        try:
            for e, reset_seed in enumerate(self.eval_seeds):
                self.state["in_flight"] = f"eval {name} episode {e}"
                act = table_behavior(tables[e], self.q, job)
                ep = run_episode(env, reset_seed, act, self.guard, job, cap, "eval")
                self.guard.check("eval expert labels")
                labels = self.expert.label(np.stack(ep.observations), job)
                per_episode["returns"].append(ep.episode_return)
                per_episode["lengths"].append(ep.length)
                per_episode["disagreement"].append(
                    float(np.mean(labels != np.asarray(ep.actions))),
                )
        finally:
            job.elapsed_seconds = time.monotonic() - t0
            for field in dataclasses.fields(Costs):
                setattr(
                    costs,
                    field.name,
                    getattr(costs, field.name) + getattr(job, field.name),
                )
            self.phase_seconds["eval"] += job.elapsed_seconds
        returns = per_episode["returns"]
        return dict(
            per_episode,
            mean_return=float(np.mean(returns)),
            mean_disagreement=float(np.mean(per_episode["disagreement"])),
            reset_seeds=list(self.eval_seeds),
            costs=job.to_dict(),
        )

    def _checkpoint(self, env: gym.Env, cap: int, n: int) -> None:
        index = self.cfg["checkpoints"].index(n)
        rounds = self.rounds_completed
        n_eval = self.cfg["eval_episodes"]
        fixed, fixed_seconds = self._fit_fixed_bc(n)
        if not np.array_equal(fixed, self.tables["bc_iid"]):
            raise EqualityControlFailed(f"Fixed BC differs from BC-iid at B={n}")
        uniforms = stream(self.cfg["seed"], "eval", _MIXTURE, index).random(n_eval)
        mixture = [select_index(u, rounds) for u in uniforms]
        # ``complete`` stays False if the deadline stops this checkpoint's evals.
        entry: Dict[str, Any] = {"budget": n, "rounds": rounds, "complete": False}
        self.checkpoints.append(entry)
        entry["train_costs_at_checkpoint"] = {
            "ftl": self.costs["train_ftl"].to_dict(),
            "bc_iid": self.costs["train_bc_iid"].to_dict(),
            "fixed_bc": self.costs["train_fixed_bc"].to_dict(),
        }
        # A standalone offline run at B: BC-iid's acquisition up to B, one fit.
        standalone = self.costs["train_bc_iid"].to_dict()
        standalone.update(
            fits=1,
            elapsed_seconds=self.phase_seconds["bc_iid_collect"] + fixed_seconds,
        )
        for arm in ("ftl", "bc_iid"):
            table = self.tables[arm]
            entry[f"{arm}_final"] = {
                "label": "final post-update policy",
                "table": table.tolist(),
                "table_sha256": quantized.table_sha256(table),
                "data_sha256": quantized.data_sha256(
                    self.data[arm]["bins"][:n],
                    self.data[arm]["labels"][:n],
                ),
                "eval": self._evaluate(env, cap, [table] * n_eval, f"{arm}_final"),
            }
        entry["fixed_bc"] = {
            "table_sha256": quantized.table_sha256(fixed),
            "data_sha256": entry["bc_iid_final"]["data_sha256"],
            "equals_bc_iid": True,
            "eval_alias": "bc_iid_final",
            "physical_acquisition": "shared with bc_iid; only its own fits are "
            "added to physical totals (train_fixed_bc)",
            "logical_standalone_costs": standalone,
            "logical_standalone_note": "acquisition of bc_iid up to B plus one fit "
            "on B labels; elapsed is bc_iid collection time plus that fit",
        }
        history = [np.asarray(t, np.int64) for t in self.ftl_behavior_tables]
        entry["ftl_mixture"] = {
            "label": "episode-uniform mixture of FTL behavior policies pi_0..pi_{R-1}",
            "n_policies": rounds,
            "mixture_indices": mixture,
            "eval": self._evaluate(
                env,
                cap,
                [history[m] for m in mixture],
                "ftl_mixture",
            ),
        }
        entry["complete"] = True

    # -- report -------------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        train = [self.costs[f"train_{arm}"] for arm in ("ftl", "bc_iid", "fixed_bc")]
        return {
            "rounds_completed": self.rounds_completed,
            "retained": {
                arm: dict(
                    self.data[arm],
                    n=len(self.data[arm]["labels"]),
                    counts=self.counts[arm].tolist(),
                    table=self.tables[arm].tolist(),
                    table_sha256=quantized.table_sha256(self.tables[arm]),
                )
                for arm in ("ftl", "bc_iid")
            },
            "ftl_behavior_tables": self.ftl_behavior_tables,
            "ftl_behavior_table_sha256": [
                quantized.table_sha256(np.asarray(t)) for t in self.ftl_behavior_tables
            ],
            "checkpoints": self.checkpoints,
            "costs": {k: v.to_dict() for k, v in self.costs.items()},
            "physical_costs": {
                "training": sum_costs(train),
                "all": sum_costs(list(self.costs.values())),
                "note": "fixed BC reuses BC-iid acquisition and evaluation; its "
                "own checkpoint fits are counted in training",
            },
            "phase_seconds": dict(self.phase_seconds),
            "seeds": {
                "train_reset_seeds": self.train_seeds,
                "train_selection_uniforms": self.train_uniforms.tolist(),
                "eval_reset_seeds": self.eval_seeds,
            },
        }


def run_cell(
    env_name: str,
    preparation_dir: pathlib.Path,
    representation: str,
    output_dir: pathlib.Path,
    *,
    seed: int,
    deadline: datetime.datetime,
    budget: int = DEFAULT_BUDGET,
    batch: int = DEFAULT_BATCH,
    checkpoints: Sequence[int] = DEFAULT_CHECKPOINTS,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    clock: Callable[[], datetime.datetime] = rollouts.utc_now,
    env_factory: Optional[Callable[[], gym.Env]] = None,
) -> int:
    """Run one learning cell; see the module docstring.

    Returns:
        Exit code: complete, incomplete (partial or failed), or refused.

    Raises:
        ValueError: On invalid arguments, before anything is written.
    """
    points = validate_cell_args(
        representation,
        seed,
        budget,
        batch,
        checkpoints,
        eval_episodes,
    )
    quantizer = quantized.get_quantizer(env_name, representation)
    guard = Guard(deadline, clock)
    try:
        model, identity = load_verified_expert(preparation_dir, env_name)
        claim = claim_fresh_output_dir(output_dir, clock)
    except Refused as exc:
        return _refuse(exc)
    output_dir = pathlib.Path(output_dir)
    factory = env_factory or (lambda: make_env(env_name))
    n_actions = int(model.action_space.n)
    identities = {
        "package_versions": pilot.package_versions(),
        "source": source_identity(),
    }
    cfg = {
        "schema": CELL_SCHEMA,
        "env_name": env_name,
        "representation": representation,
        "seed": int(seed),
        "budget": int(budget),
        "batch": int(batch),
        "checkpoints": points,
        "eval_episodes": int(eval_episodes),
        "quantizer": quantizer.describe(),
        "learner": "exact majority table (0-1 ERM); ties lowest action; unseen "
        "bins action 0; cold refit from cumulative counts each round",
        "initial_policy": "data-free table with action 0 in every bin",
        "arms": {
            "ftl": "learner behavior (deterministic table), beta 0, expert "
            "label deferred to the selected state only",
            "bc_iid": "expert behavior, beta 1, cached executed action as label",
            "fixed_bc": "same exact fit on the shared BC-iid prefix at each "
            "checkpoint; equality control",
        },
        "estimand": "episode-normalized: one uniform pre-action state per "
        "independent complete episode; not fixed-horizon occupancy",
        "pairing": "FTL and BC-iid share train reset seeds and selection "
        "uniforms; all evaluated policies share eval reset seeds",
        "preparation_config_sha256": identity["config_sha256"],
        "expert_sha256": identity["expert_sha256"],
    }
    run = CellRun(
        cfg,
        expert=Expert(model, n_actions),
        quantizer=quantizer,
        guard=guard,
        env_factory=factory,
    )
    state = run.state
    started = clock().isoformat()

    def render() -> Dict[str, Any]:
        return dict(
            run.report(),
            schema=CELL_SCHEMA,
            protocol=CELL_SCHEMA,
            env_name=env_name,
            representation=representation,
            seed=int(seed),
            expert_sha256=identity["expert_sha256"],
            status=state.get("status"),
            representation_status="representation-restricted; misspecification "
            "certificate, if any, comes from a separate audit",
            claim=claim,
            config=cfg,
            config_sha256=pilot.config_digest(cfg),
            preparation=identity,
            requested_deadline_utc=rollouts.ensure_aware_utc(deadline).isoformat(),
            effective_deadline_utc=guard.deadline.isoformat(),
            started_at_utc=started,
            finished_at_utc=state.get("finished_at_utc"),
            **identities,
            interruption=_interruption(state),
            error=state.get("error"),
        )

    writer = ResultWriter(output_dir / RESULT_FILE, render)
    return _run_lifecycle(
        state,
        writer,
        lambda: run.run(lambda force: writer.write(force)),
        guard,
    )
