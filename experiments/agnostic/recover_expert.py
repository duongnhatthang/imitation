"""Expanded expert recovery for Taxi-v3 and LunarLander-v2.

A standalone runner kept outside the immutable ``src`` snapshot. It produces
one new expert candidate per invocation and a mandatory sidecar
``acceptance.json`` that records the true training method and every quality
check. The unchanged candidate qualification (`task_experts.qualify_candidate`)
is run on the produced checkpoint and its generated fields are kept as they
are (it calls the container PPO and the training provenance unknown); the
sidecar is the record of how the expert was really made. Qualification alone
never releases an expert: ``release_allowed`` is true only when every required
check in the sidecar passes, and `verify_acceptance` re-checks the bindings
from the files.

Commands (each invocation claims a fresh output directory and runs once;
nothing is retried, and nothing written by an earlier attempt is touched)::

    python experiments/agnostic/recover_expert.py taxi \
        --config expert3-taxi-A.json --output-dir NEW_DIR \
        --candidate-ledger LEDGER_DIR --deadline 2026-10-03T00:00:00Z
    python experiments/agnostic/recover_expert.py lunar \
        --config expert3-lunar-A.json --output-dir NEW_DIR \
        --candidate-ledger LEDGER_DIR --deadline ...
    python experiments/agnostic/recover_expert.py verify-acceptance \
        --config expert3-lunar-A.json --attempt-dir DIR \
        [--candidate-ledger LEDGER_DIR]

Configs are accepted only if they equal the predeclared config that
`build_config` rebuilds from the current source for the same candidate id, so
no gate can be relaxed by editing a JSON file. Candidate B of either
environment, and LunarLander candidate C, are separate invocations by the
root; the runner never falls back on its own (see `CANDIDATE_B_RULE`).

Candidate ledger: the science commands require an existing ledger directory
shared by every output directory (the launcher pins one common ledger for all
expert3 candidates). Before any science an exclusive create-once execution
claim ``<candidate_id>.execution.json`` is made, so a candidate id trains at
most once whatever output directory is named; immediately before the
unchanged qualification a second create-once token
``<candidate_id>.heldout.json`` is made, so held-out games run at most once
per candidate. Tokens are never removed or reset, also not on failure, and
nothing is retried. The resolved ledger path and the exact token contents and
digests are recorded in the attempt. The ledger guards against duplicate
invocations through the same ledger; it does not protect against a caller
that deliberately names another ledger directory.

Resume is not supported. ``lunar-resume``, `run_lunar` with a prior attempt
and `verify_resume_source` all refuse before anything is read, written or
trained, because a resume could roll back training consumed after the latest
checkpoint and repeat held-out use. Checkpoints are still saved at every
check and kept.

Taxi (candidate A [64, 64], B [256, 256]): the deterministic transition table
of the original factory is digested and compared with the task description
digest of a freshly built factory env; the digest is recomputed and recorded,
not predeclared. Reachable nonterminal states are
enumerated from the reset support under all actions without passing through
terminal edges. A shortest legal delivery policy (unit edge lengths) is fixed
with a lowest-action-index tie break before any learner exists, and an exact
undiscounted finite-horizon DP (horizon 200) must agree with the value of that
stationary policy on every reachable state and every remaining horizon. Its
labels are distilled, uniformly over all reachable nonterminal states, into
the actor of a standard SB3 ``PPO`` ``MlpPolicy`` (full-batch Adam, actor
parameters only, critic untouched) until canonical label agreement is 100%
and self cross-entropy is at most the inherited 0.7. The model and the
pre-reload fit are saved whatever the outcome (also when distillation raises),
the fit file is written once and never rewritten, and reload evidence goes to
a separate file. The saved zip is
reloaded and rechecked, then rolled out from every reset-support state of the
original wrapped environment (cap 200): every episode must deliver with no
illegal action and with the DP optimal length and return. Then the unchanged
candidate qualification (100 games) and both the per-environment and the
common quality gates.

LunarLander (candidate A [64, 64], B [256, 256], each 4M requested; C is A
with 8M requested and its own held-out seed, a fresh single-use candidate
that never resumes A or B): PPO from scratch with the
RL Zoo style settings in the config, 16 ``DummyVecEnv`` instances from the
original factory in one process. The training budget counts actual
transitions (``model.num_timesteps``), is rounded down to whole 16384-step
rollouts, and is checked every 4 rollouts. Each check saves an immutable
checkpoint (SB3 zip including the optimizer, RNG states, progress) and runs
the frozen selection rule on 50 deterministic episodes from a fixed seed
schedule. The first passing checkpoint is copied to ``selected/`` and
qualified exactly once on held-out seeds with the unchanged qualification plus
a raw mean of at least 200. Held-out results never feed back into training.

Known limitations (also written into every acceptance record, see
`LIMITATIONS`): a selected Lunar checkpoint stranded by the deadline before
qualification stays partial, and there is no qualify-only command;
`verify_acceptance` re-checks file bindings by digest and recomputes the
qualification mean, but trusts the environment-specific booleans, so a digest
match is not semantic proof. The true training method is the
``training_method`` of the acceptance record; downstream manifests and the
final figure must carry it, since the qualification record keeps its generic
PPO container fields.
"""

import argparse
import dataclasses
import datetime
import hashlib
import json
import logging
import math
import os
import pathlib
import pickle
import random
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch as th

from imitation.experiments.agnostic import (
    classical,
    pilot,
    rollouts,
    task_experts,
    tasks,
)
from imitation.experiments.ftrl import env_baselines, env_utils, eval_utils

logger = logging.getLogger("recover_expert")

CONFIG_SCHEMA = "agnostic-expert-recovery-config/1"
ATTEMPT_SCHEMA = "agnostic-expert-recovery-attempt/1"
ACCEPTANCE_SCHEMA = "agnostic-expert-recovery-acceptance/1"
ATTEMPT_FILE = "attempt.json"
ACCEPTANCE_FILE = "acceptance.json"
CONFIG_COPY = "config.json"
QUALIFICATION_DIR = "qualification"
QUALIFICATION_RECORD = f"{QUALIFICATION_DIR}/{task_experts.CANDIDATE_FILE}"

EXIT_RELEASED = 0
EXIT_NOT_RELEASED = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3

TAXI = "Taxi-v3"
LUNAR = "LunarLander-v2"
RUNNER_PATH = pathlib.Path(__file__).resolve()
RUNNER_LABEL = "experiments/agnostic/recover_expert.py"
# Sources outside pilot.source_fingerprint() that this runner also relies on.
_EXTRA_SOURCES = (
    "experiments/agnostic/classical.py",
    "experiments/agnostic/restriction.py",
    "experiments/ftrl/policy_utils.py",
)

TAXI_METHOD = (
    "supervised distillation of an exact shortest-delivery planner into the "
    "actor of a stable_baselines3 PPO MlpPolicy container: full-batch Adam on "
    "cross-entropy of planner labels over all reachable nonterminal states; "
    "PPO.learn is never called, no environment reward is used in training, "
    "and the critic is left at its initialization"
)
LUNAR_METHOD = (
    "PPO (stable_baselines3) trained from scratch on the original "
    "LunarLander-v2 factory with the configured hyperparameters; candidate "
    "chosen by the frozen selection rule on selection seeds, then qualified "
    "once on held-out seeds"
)
QUALIFICATION_FIELDS_NOTE = (
    "The candidate qualification record is generated by the unchanged "
    "task_experts.qualify_candidate and keeps its own fields (expert algorithm "
    "PPO, training provenance unknown). The true training method is the "
    "training_method of this acceptance record, bound to the qualification "
    "record by digest."
)
RESUME_UNSUPPORTED = (
    "Resume is not supported: lunar-resume, run_lunar with a prior attempt and "
    "verify_resume_source refuse before anything is read, written or trained. "
    "Checkpoints are still saved at every check and kept. The resume clauses "
    "in the frozen config text (training_env_rng, seed_schedule) are inert."
)
CANDIDATE_B_RULE = (
    "Candidate B of an environment may be invoked only after candidate A ended "
    "with its predefined scientific failure: for Taxi a fitting failure (the "
    "distillation stop rule is not met within the update cap), for LunarLander "
    "a training or quality failure (the budget is exhausted without a passing "
    "selection check, or the single held-out qualification fails a quality "
    "gate). A deadline, partial attempt, infrastructure error or refused "
    "ledger claim does not make B eligible. The runner never invokes B; the "
    "frozen launcher enforces this rule before invoking B."
)
LEDGER_RULE = (
    "one execution claim and one held-out token per candidate id, each created "
    "once with O_EXCL in the shared ledger; tokens are never removed or reset, "
    "also not on failure, and nothing is retried"
)
LIMITATIONS = (
    "A selected LunarLander checkpoint stranded by the deadline before "
    "qualification stays partial (fail closed); there is no qualify-only "
    "command and no automatic promotion of candidate B.",
    "verify_acceptance re-checks file bindings by digest and recomputes the "
    "qualification mean from its episodes; the Taxi DP, distillation and "
    "verification booleans and the LunarLander selection booleans are trusted "
    "as written, so digest matches are not semantic proof.",
    "The ledger refuses duplicate invocations through the same ledger "
    "directory; it does not protect against deliberately naming another "
    "ledger directory. The launcher pins one common ledger.",
    "The Taxi dynamics digest is recomputed from a freshly built factory env "
    "and recorded; it is not a predeclared constant (the frozen config phrase "
    "'pinned table' refers to this recomputed digest).",
    RESUME_UNSUPPORTED,
)

CANDIDATES = ("taxi-A", "taxi-B", "lunar-A", "lunar-B", "lunar-C")
LEDGER_SCHEMA = "agnostic-expert-recovery-ledger-token/1"
LEDGER_KINDS = ("execution", "heldout")


class Refused(Exception):
    """Inputs fail verification; nothing was run."""


class RecoveryFailure(RuntimeError):
    """A required step failed; the attempt is recorded as failed."""


class BudgetViolation(RecoveryFailure):
    """Training consumed a different number of transitions than planned."""


# ---------------------------------------------------------------------------
# Predeclared configs
# ---------------------------------------------------------------------------


def _gates(env_name: str) -> Dict[str, Any]:
    return {
        "per_env_threshold": pilot.frozen_expert_threshold(env_name),
        "global_fraction": float(env_baselines.EXPERT_QUALITY_THRESHOLD),
        "reference_baselines": dict(env_baselines.REFERENCE_BASELINES[env_name]),
        "rule": "release requires the unchanged candidate qualification to be "
        "complete and its recomputed mean to pass both the per-environment "
        "threshold and the common threshold random + 0.80 * (expert - random)",
    }


def _taxi_config(candidate: str) -> Dict[str, Any]:
    net_arch = {"A": [64, 64], "B": [256, 256]}[candidate]
    convergence = env_utils.get_convergence_config(TAXI)
    return {
        "schema": CONFIG_SCHEMA,
        "candidate_id": f"taxi-{candidate}",
        "env_name": TAXI,
        "command": "taxi",
        "training_method": TAXI_METHOD,
        "fallback_rule": "candidate B is run only after candidate A fails, as a "
        "separate invocation by the root; the runner never falls back itself",
        "device": "cpu",
        "torch_threads": 1,
        "teacher": {
            "type": "shortest legal delivery planner on the pinned table",
            "edge_length": 1,
            "tie_break": "lowest action index among actions on a shortest path",
            "dp_horizon": 200,
            "rewards": {"delivery": 20, "step": -1, "illegal": -10},
            "frozen_before": "any restriction, mask or learner result",
        },
        "distillation": {
            "container": "stable_baselines3.PPO",
            "policy": "MlpPolicy",
            "net_arch": net_arch,
            "observation": "original factory one-hot of size 500",
            "state_weighting": "uniform over all reachable nonterminal states",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch": "full",
            "max_updates": 20000,
            "seed": 1,
            "trained_parameters": "mlp_extractor.policy_net and action_net only",
            "critic": "left at initialization and unused",
            "stop_rule": {
                "canonical_label_agreement": 1.0,
                "self_ce_max": float(convergence["self_ce_eps"]),
            },
        },
        "verification": {
            "reset_seed_start": 0,
            "max_resets": 20000,
            "episode_cap": 200,
            "rule": "reset the original wrapped env with seeds start, start+1, "
            "... until every reset-support state was a start state; every "
            "episode must deliver without an illegal action, with length and "
            "return equal to the DP optimum",
        },
        "qualification": {"episodes": 100, "seed": 900111},
        "gates": _gates(TAXI),
    }


def _lunar_config(candidate: str) -> Dict[str, Any]:
    # C repeats A with a larger budget and its own held-out seed.
    net_arch = {"A": [64, 64], "B": [256, 256], "C": [64, 64]}[candidate]
    qual_seed = {"A": 900112, "B": 900113, "C": 900114}[candidate]
    n_envs, n_steps, every = 16, 1024, 4
    rollout = n_envs * n_steps
    requested = {"A": 4_000_000, "B": 4_000_000, "C": 8_000_000}[candidate]
    usable, n_rollouts = usable_budget(requested, rollout)
    convergence = env_utils.get_convergence_config(LUNAR)
    return {
        "schema": CONFIG_SCHEMA,
        "candidate_id": f"lunar-{candidate}",
        "env_name": LUNAR,
        "command": "lunar",
        "training_method": LUNAR_METHOD,
        "fallback_rule": "candidate B is run only after candidate A fails, as a "
        "separate invocation by the root; the runner never falls back itself",
        "device": "cpu",
        "torch_threads": 1,
        "ppo": {
            "policy": "MlpPolicy",
            "net_arch": net_arch,
            "n_envs": n_envs,
            "vec_env": "env_utils.make_env (util.make_vec_env, parallel False)",
            "n_steps": n_steps,
            "batch_size": 64,
            "gae_lambda": 0.98,
            "gamma": 0.999,
            "n_epochs": 4,
            "ent_coef": 0.01,
            "learning_rate": 0.0003,
            "seed": 1,
            "training_env_rng": "env_utils.make_env rng SeedSequence(seed, "
            "spawn_key=(expert_training,)), or (expert_training, T) for a resume "
            "from T transitions; SB3 PPO(seed) and PPO.load then call "
            "env.seed(seed), so the first training reset of a fresh or resumed "
            "run uses reset seeds seed .. seed + n_envs - 1; global RNG states "
            "are restored from the checkpoint after load",
        },
        "budget": {
            "requested_max_env_transitions": requested,
            "rollout_env_transitions": rollout,
            "usable_max_env_transitions": usable,
            "rollouts": n_rollouts,
            "check_every_rollouts": every,
            "check_env_transitions": every * rollout,
            "rule": "count actual transitions (model.num_timesteps); round the "
            "request down to whole rollouts; check every 4 rollouts and once more "
            "at the usable bound if it is not aligned",
        },
        "selection": {
            "episodes": 50,
            "deterministic": True,
            "seed": 910001,
            "seed_schedule": "check k (1-based, counted across resumes) evaluates "
            "on one fresh env_utils.make_env env with "
            "rng=default_rng(SeedSequence(seed, spawn_key=(k,)))",
            "normalized_threshold": float(env_baselines.EXPERT_QUALITY_THRESHOLD),
            "self_ce_max": float(convergence["self_ce_eps"]),
            "min_env_transitions": int(convergence["min_timesteps"]),
            "patience": int(convergence["patience"]),
            "plateau_tolerance": 0.02,
            "rule": "inherited expert_training rule: the last `patience` "
            "normalized returns all >= threshold, current self-CE <= self_ce_max, "
            "transitions >= min, and no improvement of best normalized return or "
            "best self-CE by more than the tolerance in the last `patience` "
            "checks; the first passing check is selected",
        },
        "qualification": {"episodes": 100, "seed": qual_seed, "raw_mean_min": 200.0},
        "gates": _gates(LUNAR),
    }


def build_config(candidate_id: str) -> Dict[str, Any]:
    """The predeclared config of one candidate, rebuilt from current source.

    Raises:
        ValueError: For an unknown candidate id.
    """
    if candidate_id not in CANDIDATES:
        raise ValueError(f"Unknown candidate {candidate_id!r}; expected {CANDIDATES}")
    env, letter = candidate_id.split("-")
    return _taxi_config(letter) if env == "taxi" else _lunar_config(letter)


def load_config(path: pathlib.Path, command: str) -> Tuple[Dict[str, Any], bytes]:
    """Read a config file and require it to equal the predeclared config.

    Raises:
        Refused: If unreadable, unknown, for another command, or different.
    """
    try:
        raw = pathlib.Path(path).read_bytes()
        config = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise Refused(f"Unreadable config {path}: {exc}") from None
    if not isinstance(config, dict) or config.get("candidate_id") not in CANDIDATES:
        raise Refused(f"{path} does not name a predeclared candidate")
    expected = build_config(config["candidate_id"])
    if config != expected:
        changed = sorted(
            k for k in set(config) | set(expected) if config.get(k) != expected.get(k)
        )
        raise Refused(f"{path} differs from the predeclared config in {changed}")
    if config["command"] != command:
        raise Refused(f"{config['candidate_id']} is run by {config['command']!r}")
    return config, raw


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _error_text(what: str, exc: BaseException) -> str:
    return f"{what}: {type(exc).__name__}: {exc}"


def source_identity() -> Dict[str, Any]:
    root = pilot._PACKAGE_ROOT
    extra = {}
    for rel in _EXTRA_SOURCES:
        path = root / rel
        extra[rel] = pilot.sha256_file(path) if path.is_file() else "missing"
    identity = {
        "runner": {"path": RUNNER_LABEL, "sha256": pilot.sha256_file(RUNNER_PATH)},
        "package_sources": pilot.source_fingerprint(),
        "extra_package_sources": extra,
    }
    identity["combined_sha256"] = pilot.config_digest(
        {
            "runner": identity["runner"]["sha256"],
            "package_sources": identity["package_sources"]["files"],
            "extra": extra,
        },
    )
    return identity


def env_configs_sha256() -> str:
    return pilot.config_digest(env_utils.ENV_CONFIGS)


def _write_new_json(path: pathlib.Path, payload: Any) -> str:
    """Atomically write a JSON file that must not exist yet; return its digest."""
    path = pathlib.Path(path)
    if path.exists():
        raise RecoveryFailure(f"Refusing to overwrite {path}")
    pilot.atomic_write_json(path, payload)
    return pilot.sha256_file(path)


def _copy_exclusive(src: pathlib.Path, dst: pathlib.Path) -> str:
    data = pathlib.Path(src).read_bytes()
    dst = pathlib.Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    digest = pilot.sha256_file(dst)
    if digest != _sha(data):
        raise RecoveryFailure(f"Copied bytes of {dst} differ from {src}")
    return digest


def token_bytes(contents: Mapping[str, Any]) -> bytes:
    """The exact bytes of a ledger token with these contents."""
    return (pilot.canonical_json(dict(contents)) + "\n").encode("utf-8")


class CandidateLedger:
    """Create-once execution and held-out tokens shared across output dirs."""

    def __init__(self, path: Optional[pathlib.Path]) -> None:
        """Pin an existing ledger directory by its resolved path.

        Raises:
            Refused: If no ledger is given or it is not an existing directory.
        """
        if path is None or str(path) == "":
            raise Refused("A --candidate-ledger directory is required")
        path = pathlib.Path(path)
        if not path.is_dir():
            raise Refused(f"Candidate ledger {path} is not an existing directory")
        self.path = path.resolve(strict=True)

    def token_path(self, candidate_id: str, kind: str) -> pathlib.Path:
        if candidate_id not in CANDIDATES or kind not in LEDGER_KINDS:
            raise ValueError(f"Unknown ledger token {candidate_id!r} {kind!r}")
        return self.path / f"{candidate_id}.{kind}.json"

    def require_unclaimed(self, candidate_id: str, kind: str) -> None:
        """Early refusal; the exclusive create in `claim` is what decides.

        Raises:
            Refused: If the token already exists.
        """
        path = self.token_path(candidate_id, kind)
        if os.path.lexists(path):
            raise Refused(f"{candidate_id} already has a {kind} token at {path}")

    def claim(self, candidate_id: str, kind: str, contents: Mapping[str, Any]):
        """Exclusively create one token; it is never removed or rewritten.

        Raises:
            Refused: If the token already exists. A failure after the create
                leaves the (possibly incomplete) token in place, fail closed.
        """
        path = self.token_path(candidate_id, kind)
        contents = dict(
            contents,
            schema=LEDGER_SCHEMA,
            kind=kind,
            candidate_id=candidate_id,
            ledger=str(self.path),
            rule=LEDGER_RULE,
        )
        data = token_bytes(contents)
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
        except FileExistsError:
            raise Refused(
                f"{candidate_id} already has a {kind} token at {path}",
            ) from None
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        dir_fd = os.open(str(self.path), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        if pilot.sha256_file(path) != _sha(data):
            raise RecoveryFailure(f"Ledger token {path} bytes differ after writing")
        logger.info("Candidate ledger %s: created %s", self.path, path.name)
        return {"file": path.name, "sha256": _sha(data), "contents": contents}


def ledger_problems(
    attempt: Mapping[str, Any],
    config: Mapping[str, Any],
    checkpoint_sha256: Optional[str],
    ledger: Optional[CandidateLedger] = None,
) -> List[str]:
    """Problems with the ledger tokens an attempt records (none if consistent).

    The recorded contents must name this candidate, config and checkpoint and
    hash to the recorded digests. With ``ledger`` the attempt must name that
    ledger and its token files must hold exactly the recorded bytes.
    """
    rec = attempt.get("candidate_ledger")
    if not isinstance(rec, dict):
        return ["no candidate ledger record"]
    problems = []
    tokens = {}
    for kind, key in (("execution", "execution_claim"), ("heldout", "heldout_token")):
        token = rec.get(key)
        contents = token.get("contents") if isinstance(token, dict) else None
        if not isinstance(contents, dict):
            problems.append(f"no {kind} token")
            continue
        tokens[kind] = token
        if _sha(token_bytes(contents)) != token.get("sha256"):
            problems.append(f"{kind} token digest")
        if (
            contents.get("schema") != LEDGER_SCHEMA
            or contents.get("kind") != kind
            or contents.get("candidate_id") != config["candidate_id"]
            or contents.get("ledger") != rec.get("path")
            or token.get("file") != f"{config['candidate_id']}.{kind}.json"
        ):
            problems.append(f"{kind} token identity")
    if "execution" in tokens:
        claim = tokens["execution"]["contents"]
        if claim.get("config_sha256") != pilot.config_digest(config):
            problems.append("execution token config")
    if "execution" in tokens and "heldout" in tokens:
        held = tokens["heldout"]["contents"]
        if held.get("execution_claim_sha256") != tokens["execution"]["sha256"]:
            problems.append("held-out token is not bound to the execution claim")
        if checkpoint_sha256 is None or held.get("checkpoint_sha256") != (
            checkpoint_sha256
        ):
            problems.append("held-out token checkpoint")
    if ledger is not None:
        if rec.get("path") != str(ledger.path):
            problems.append("attempt names another ledger")
        for kind, token in tokens.items():
            path = ledger.token_path(config["candidate_id"], kind)
            actual = pilot.sha256_file(path) if path.is_file() else None
            if actual != token.get("sha256"):
                problems.append(f"{kind} token file in ledger")
    return problems


class Attempt:
    """One claimed attempt directory and its atomically rewritten record."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        config: Dict[str, Any],
        config_bytes: bytes,
        deadline: datetime.datetime,
        clock: Callable[[], datetime.datetime] = rollouts.utc_now,
        source: Optional[Dict[str, Any]] = None,
        *,
        ledger: CandidateLedger,
    ) -> None:
        """Claim the output directory, then the candidate's execution token.

        Raises:
            Refused: If the output directory is not fresh, no ledger is given,
                or the candidate already has an execution token. In the last
                case the attempt record says ``refused`` and nothing ran.
        """
        if not isinstance(ledger, CandidateLedger):
            raise Refused("A candidate ledger is required")
        self.dir = pathlib.Path(output_dir)
        self.clock = clock
        self.config = config
        self.ledger = ledger
        try:
            claim = classical.claim_fresh_output_dir(self.dir, clock)
        except classical.Refused as exc:
            raise Refused(str(exc)) from None
        with open(self.dir / CONFIG_COPY, "xb") as f:
            f.write(config_bytes)
            f.flush()
            os.fsync(f.fileno())
        self.started = time.monotonic()
        source = source if source is not None else source_identity()
        self.record: Dict[str, Any] = {
            "schema": ATTEMPT_SCHEMA,
            "status": "running",
            "candidate_id": config["candidate_id"],
            "env_name": config["env_name"],
            "training_method": config["training_method"],
            "claim": claim,
            "config_file": {"path": CONFIG_COPY, "sha256": _sha(config_bytes)},
            "config_sha256": pilot.config_digest(config),
            "deadline_utc": deadline.isoformat(),
            "started_at_utc": clock().isoformat(),
            "source": source,
            "package_versions": pilot.package_versions(),
            "env_configs_sha256_start": env_configs_sha256(),
            "runtime": {"device": "cpu", "torch_threads": th.get_num_threads()},
            "costs": {},
            "elapsed_seconds": {},
            "failure_reasons": [],
            "no_retry": "this attempt runs once; a new attempt needs a new directory",
            "candidate_b_rule": CANDIDATE_B_RULE,
            "candidate_ledger": {"path": str(ledger.path), "rule": LEDGER_RULE},
        }
        try:
            token = ledger.claim(
                config["candidate_id"],
                "execution",
                {
                    "config_sha256": self.record["config_sha256"],
                    "config_file_sha256": _sha(config_bytes),
                    "runner_sha256": source["runner"]["sha256"],
                    "source_combined_sha256": source["combined_sha256"],
                    "output_dir": str(self.dir.resolve()),
                    "output_claim": claim,
                    "claimed_at_utc": clock().isoformat(),
                    "pid": os.getpid(),
                },
            )
        except Exception as exc:  # nothing ran; a created token stays
            refused = isinstance(exc, Refused)
            self.record["status"] = "refused" if refused else "failed"
            self.record["failure_reasons"].append(_error_text("execution claim", exc))
            self.record["finished_at_utc"] = clock().isoformat()
            self.write()
            raise
        self.record["candidate_ledger"]["execution_claim"] = token
        self.write()

    def claim_heldout(self, checkpoint_sha256: str) -> Dict[str, Any]:
        """Create the one held-out token of this candidate and record it.

        Raises:
            Refused: If the candidate already has a held-out token.
        """
        qual = self.config["qualification"]
        execution = self.record["candidate_ledger"]["execution_claim"]
        token = self.ledger.claim(
            self.config["candidate_id"],
            "heldout",
            {
                "execution_claim_sha256": execution["sha256"],
                "output_dir": str(self.dir.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "qualification_seed": int(qual["seed"]),
                "qualification_episodes": int(qual["episodes"]),
                "claimed_at_utc": self.clock().isoformat(),
                "pid": os.getpid(),
            },
        )
        self.record["candidate_ledger"]["heldout_token"] = token
        self.write()
        return token

    def write(self) -> None:
        pilot.atomic_write_json(self.dir / ATTEMPT_FILE, self.record)

    def fail_reason(self, reason: str) -> None:
        self.record["failure_reasons"].append(reason)
        self.write()

    def finish(self, status: str, exc: Optional[BaseException] = None) -> str:
        self.record["status"] = status
        if exc is not None:
            self.record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            self.record["failure_reasons"].append(f"{type(exc).__name__}: {exc}")
        self.record["env_configs_sha256_end"] = env_configs_sha256()
        self.record["finished_at_utc"] = self.clock().isoformat()
        self.record["elapsed_seconds"]["total"] = time.monotonic() - self.started
        self.write()
        return pilot.sha256_file(self.dir / ATTEMPT_FILE)


# ---------------------------------------------------------------------------
# Taxi: transition table, planner and exact DP
# ---------------------------------------------------------------------------

Table = Dict[int, Dict[int, Tuple[int, int, bool]]]


def dynamics_sha256(raw_table: Mapping) -> str:
    """Digest of ``env.unwrapped.P`` in the same encoding as ``tasks``."""
    rows = {
        str(s): {
            str(a): [[float(p), int(n), float(r), bool(d)] for p, n, r, d in outcomes]
            for a, outcomes in sorted(actions.items())
        }
        for s, actions in sorted(raw_table.items())
    }
    return pilot.config_digest(rows)


def deterministic_table(raw_table: Mapping) -> Table:
    """Each (state, action) as (next state, integer reward, done).

    Raises:
        RecoveryFailure: If any transition is stochastic or a reward is not an
            integer.
    """
    table: Table = {}
    for s, actions in raw_table.items():
        row = {}
        for a, outcomes in actions.items():
            if len(outcomes) != 1 or float(outcomes[0][0]) != 1.0:
                raise RecoveryFailure(f"Transition ({s}, {a}) is not deterministic")
            _, nxt, reward, done = outcomes[0]
            if float(reward) != int(reward):
                raise RecoveryFailure(f"Reward of ({s}, {a}) is not an integer")
            row[int(a)] = (int(nxt), int(reward), bool(done))
        table[int(s)] = row
    return table


def reachable_states(table: Table, starts: Sequence[int]) -> List[int]:
    """Nonterminal states reachable from ``starts`` under any actions.

    A terminal edge ends the episode, so its successor is never entered.

    Raises:
        RecoveryFailure: If a reset state is also the target of a terminal edge.
    """
    seen = set(int(s) for s in starts)
    queue = sorted(seen)
    terminal_targets = set()
    while queue:
        s = queue.pop()
        for a in sorted(table[s]):
            nxt, _, done = table[s][a]
            if done:
                terminal_targets.add(nxt)
            elif nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    if terminal_targets & set(int(s) for s in starts):
        raise RecoveryFailure("A reset state is also the target of a terminal edge")
    return sorted(seen)


def check_reward_semantics(table: Table, states: Sequence[int], rewards) -> List[str]:
    """Problems with terminal +20, ordinary -1 and illegal -10 (self loop)."""
    problems = []
    for s in states:
        for a, (nxt, r, done) in sorted(table[s].items()):
            if done and r != rewards["delivery"]:
                problems.append(f"terminal edge ({s}, {a}) has reward {r}")
            elif not done and r not in (rewards["step"], rewards["illegal"]):
                problems.append(f"edge ({s}, {a}) has reward {r}")
            elif not done and r == rewards["illegal"] and nxt != s:
                problems.append(f"illegal edge ({s}, {a}) moves to {nxt}")
    return problems


def shortest_delivery(table: Table, states: Sequence[int]) -> Dict[int, Optional[int]]:
    """Fewest steps to a terminal edge from each state (None if impossible).

    Every edge has length 1; a terminal edge ends the path. Bellman-Ford over
    the reachable set, which is closed under nonterminal edges.
    """
    dist: Dict[int, Optional[int]] = {s: None for s in states}
    for _ in range(len(states) + 1):
        changed = False
        for s in states:
            best = dist[s]
            for nxt, _, done in table[s].values():
                tail = 0 if done else dist[nxt]
                if tail is not None and (best is None or tail + 1 < best):
                    best = tail + 1
            if best != dist[s]:
                dist[s], changed = best, True
        if not changed:
            return dist
    raise RecoveryFailure("Shortest-path iteration did not converge")


def planner_policy(table: Table, dist: Mapping[int, Optional[int]]) -> Dict[int, int]:
    """Lowest-index action on a shortest delivery path from every state.

    Raises:
        RecoveryFailure: If some reachable state cannot deliver.
    """
    missing = sorted(s for s, d in dist.items() if d is None)
    if missing:
        raise RecoveryFailure(f"{len(missing)} reachable states cannot deliver")
    policy = {}
    for s, d in dist.items():
        for a in sorted(table[s]):
            nxt, _, done = table[s][a]
            if 1 + (0 if done else dist[nxt]) == d:
                policy[s] = a
                break
    return policy


def finite_horizon_check(
    table: Table,
    states: Sequence[int],
    policy: Mapping[int, int],
    horizon: int,
) -> Dict[str, Any]:
    """Exact undiscounted DP versus the stationary policy for every horizon.

    ``V_0 = 0``; ``Q_h(s, a) = r + (0 if done else V_{h-1}(s'))``. A terminal
    edge never bootstraps. Integers throughout, so the comparison is exact.

    Returns:
        Evidence with the per-horizon mismatches, the horizon-``horizon``
        optimal values and optimal action sets, and digests of all tables.
    """
    index = {s: i for i, s in enumerate(states)}
    n_actions = len(table[states[0]])
    nxt = np.zeros((len(states), n_actions), dtype=np.int64)
    rew = np.zeros((len(states), n_actions), dtype=np.int64)
    alive = np.zeros((len(states), n_actions), dtype=np.int64)
    for s in states:
        for a in range(n_actions):
            t, r, done = table[s][a]
            rew[index[s], a] = r
            alive[index[s], a] = 0 if done else 1
            nxt[index[s], a] = 0 if done else index[t]
    pi = np.array([policy[s] for s in states], dtype=np.int64)
    rows = np.arange(len(states))
    v_opt = np.zeros(len(states), dtype=np.int64)
    v_pi = np.zeros(len(states), dtype=np.int64)
    opt_digest, pi_digest = hashlib.sha256(), hashlib.sha256()
    mismatched_horizons = []
    q = None
    for h in range(1, horizon + 1):
        q = rew + alive * v_opt[nxt]
        q_pi = rew[rows, pi] + alive[rows, pi] * v_pi[nxt[rows, pi]]
        v_opt, v_pi = q.max(axis=1), q_pi
        opt_digest.update(v_opt.tobytes())
        pi_digest.update(v_pi.tobytes())
        bad = int(np.sum(v_opt != v_pi))
        if bad:
            mismatched_horizons.append({"horizon": h, "states": bad})
    optimal_sets = {
        str(s): [int(a) for a in np.flatnonzero(q[index[s]] == v_opt[index[s]])]
        for s in states
    }
    return {
        "horizon": horizon,
        "rule": "V_0 = 0; Q_h = r + (0 if terminal else V_{h-1}(next)); "
        "policy value uses the same recursion with the planner action",
        "mismatched_horizons": mismatched_horizons,
        "equal_all_horizons": not mismatched_horizons,
        "optimal_value": {str(s): int(v_opt[index[s]]) for s in states},
        "policy_value": {str(s): int(v_pi[index[s]]) for s in states},
        "optimal_action_sets": optimal_sets,
        "policy_in_optimal_set": all(policy[s] in optimal_sets[str(s)] for s in states),
        "optimal_value_tables_sha256": opt_digest.hexdigest(),
        "policy_value_tables_sha256": pi_digest.hexdigest(),
    }


def build_teacher(
    raw_table: Mapping,
    starts: Sequence[int],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Planner labels, reachable set and DP verification, or a failure."""
    teacher_cfg = config["teacher"]
    table = deterministic_table(raw_table)
    states = reachable_states(table, starts)
    problems = check_reward_semantics(table, states, teacher_cfg["rewards"])
    if problems:
        raise RecoveryFailure(f"Reward semantics differ: {problems[:5]}")
    dist = shortest_delivery(table, states)
    policy = planner_policy(table, dist)
    dp = finite_horizon_check(table, states, policy, int(teacher_cfg["dp_horizon"]))
    if not dp["equal_all_horizons"] or not dp["policy_in_optimal_set"]:
        raise RecoveryFailure(
            f"DP disagrees with the planner policy: {dp['mismatched_horizons'][:5]}",
        )
    return {
        "dynamics_sha256": dynamics_sha256(raw_table),
        "reset_support": [int(s) for s in starts],
        "reset_support_sha256": pilot.config_digest([int(s) for s in starts]),
        "reachable_states": states,
        "reachable_states_sha256": pilot.config_digest(states),
        "distance": {str(s): int(dist[s]) for s in states},
        "max_distance": max(int(d) for d in dist.values()),
        "labels": {str(s): int(policy[s]) for s in states},
        "labels_sha256": pilot.config_digest({str(s): policy[s] for s in states}),
        "tie_break": teacher_cfg["tie_break"],
        "dp": dp,
    }


# ---------------------------------------------------------------------------
# Taxi: distillation and verification
# ---------------------------------------------------------------------------


def _is_actor(name: str) -> bool:
    return name.startswith("mlp_extractor.policy_net.") or name.startswith(
        "action_net."
    )


def _state_digest(policy, keep: Callable[[str], bool]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        if keep(name):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fit_metrics(policy, obs: np.ndarray, labels: np.ndarray, optimal_sets) -> Dict:
    preds, _ = policy.predict(obs, deterministic=True)
    preds = np.asarray(preds, dtype=np.int64)
    agree = int(np.sum(preds == labels))
    in_set = int(sum(int(p) in opt for p, opt in zip(preds, optimal_sets)))
    self_ce = eval_utils.compute_sampled_action_ce(policy, obs, preds)
    return {
        "states": int(len(labels)),
        "canonical_agreement_count": agree,
        "canonical_agreement": agree / len(labels),
        "optimal_set_agreement_count": in_set,
        "optimal_set_agreement": in_set / len(labels),
        "self_ce": float(self_ce),
        "predictions_sha256": pilot.config_digest([int(p) for p in preds]),
    }


def distill_actor(
    policy,
    obs: np.ndarray,
    labels: np.ndarray,
    optimal_sets,
    cfg: Mapping[str, Any],
    deadline: datetime.datetime,
    clock: Callable[[], datetime.datetime],
) -> Dict[str, Any]:
    """Full-batch Adam on the actor until the stop rule or the update cap.

    An exception does not propagate: the fit so far is returned with stop
    reason ``error`` so the caller can still save the model and this partial
    evidence. ``final`` is then the last completed metric pass, or None.
    """
    stop = cfg["stop_rule"]
    updates, history, metrics, reason, error = 0, [], None, None, None
    passes = losses = 0
    try:
        actor = [p for n, p in policy.named_parameters() if _is_actor(n)]
        optimizer = th.optim.Adam(actor, lr=float(cfg["learning_rate"]))
        obs_t = th.as_tensor(obs, dtype=th.float32)
        labels_t = th.as_tensor(labels, dtype=th.long)
        while True:
            metrics = dict(
                fit_metrics(policy, obs, labels, optimal_sets),
                updates=updates,
            )
            passes += 1
            if updates % 500 == 0:
                history.append(dict(metrics))
            if metrics["canonical_agreement_count"] == len(labels) and metrics[
                "self_ce"
            ] <= float(stop["self_ce_max"]):
                reason = "stop_rule_met"
                break
            if updates >= int(cfg["max_updates"]):
                reason = "max_updates"
                break
            if updates % 100 == 0 and clock() >= deadline:
                reason = "deadline"
                break
            policy.set_training_mode(True)
            loss = -policy.get_distribution(obs_t).log_prob(labels_t).mean()
            losses += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1
    except Exception as exc:  # the caller saves the model and this partial fit
        reason, error = "error", {"type": type(exc).__name__, "message": str(exc)}
    try:
        policy.set_training_mode(False)
    except Exception as exc:
        if error is None:
            reason = "error"
            error = {"type": type(exc).__name__, "message": str(exc)}
    fit = {
        "stop_reason": reason,
        "updates": updates,
        "final": metrics,
        "history_every_500_updates": history,
        # Completed forwards: two per metric pass and one per loss, which is
        # (3 * updates + 2) per state when the loop ends normally.
        "forward_state_evaluations": (2 * passes + losses) * len(labels),
    }
    if error is not None:
        fit["error"] = error
    return fit


def verify_reset_support(
    env,
    predict: Callable[[np.ndarray], int],
    expected: Mapping[int, Tuple[int, int]],
    rewards: Mapping[str, int],
    *,
    seed_start: int,
    max_resets: int,
    deadline: datetime.datetime,
    clock: Callable[[], datetime.datetime],
) -> Dict[str, Any]:
    """Roll out from reset seeds until every reset-support state was a start.

    ``expected`` maps each reset state to its DP optimal (length, return).
    Every episode must deliver (terminate with the delivery reward), take no
    illegal action, and match the optimum exactly.
    """
    covered: Dict[int, int] = {}
    episodes = []
    resets = transitions = queries = 0
    stop = "covered"
    while len(covered) < len(expected):
        if resets >= max_resets:
            stop = "max_resets"
            break
        if resets % 100 == 0 and clock() >= deadline:
            stop = "deadline"
            break
        seed = seed_start + resets
        obs, _ = env.reset(seed=seed)
        resets += 1
        start = int(np.argmax(obs))
        ret, length, illegal = 0, 0, 0
        terminated = truncated = False
        last_reward = None
        while not (terminated or truncated):
            action = int(predict(obs))
            queries += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            transitions += 1
            length += 1
            ret += int(reward)
            last_reward = int(reward)
            illegal += int(reward == rewards["illegal"])
        want = expected.get(start)
        delivered = bool(terminated and last_reward == rewards["delivery"])
        episodes.append(
            {
                "seed": seed,
                "start_state": start,
                "length": length,
                "return": ret,
                "delivered": delivered,
                "truncated": bool(truncated),
                "illegal_actions": illegal,
                "optimal": want is not None and (length, ret) == tuple(want),
            },
        )
        covered.setdefault(start, seed)
    unexpected = sorted({e["start_state"] for e in episodes} - set(expected))
    checks = {
        "reset_support_covered": stop == "covered" and len(covered) == len(expected),
        "no_unexpected_start": not unexpected,
        "all_delivered": bool(episodes) and all(e["delivered"] for e in episodes),
        "no_illegal_actions": all(e["illegal_actions"] == 0 for e in episodes),
        "optimal_length_and_return": bool(episodes)
        and all(e["optimal"] for e in episodes),
    }
    return {
        "stop_reason": stop,
        "checks": checks,
        "resets": resets,
        "env_transitions": transitions,
        "policy_queries": queries,
        "covered_states": len(covered),
        "reset_support_size": len(expected),
        "unexpected_starts": unexpected,
        "episodes": episodes,
    }


# ---------------------------------------------------------------------------
# Qualification (unchanged) and acceptance
# ---------------------------------------------------------------------------

TAXI_CHECKS = (
    "config_predeclared",
    "cpu_single_thread",
    "task_cap",
    "dynamics_deterministic_and_task_digest",
    "reachable_matches_task",
    "reward_semantics",
    "planner_complete",
    "dp_equal_all_horizons",
    "mask_representatives_covered",
    "distill_canonical_agreement",
    "distill_optimal_set_agreement",
    "distill_self_ce",
    "critic_untouched",
    "reload_policy_state_equal",
    "reload_canonical_agreement",
    "reload_self_ce",
    "reset_support_covered",
    "rollouts_delivered",
    "rollouts_no_illegal",
    "rollouts_optimal",
    "qualification_complete",
    "qualification_settings",
    "qualification_candidate_bytes",
    "qualification_loader",
    "per_env_gate",
    "global_gate",
    "env_configs_unchanged",
)
LUNAR_CHECKS = (
    "config_predeclared",
    "cpu_single_thread",
    "task_cap",
    "budget_exact",
    "budget_within_bound",
    "selection_passed",
    "selected_checkpoint_bytes",
    "qualification_complete",
    "qualification_settings",
    "qualification_candidate_bytes",
    "qualification_loader",
    "per_env_gate",
    "global_gate",
    "raw_mean_min",
    "env_configs_unchanged",
)
REQUIRED_CHECKS = {TAXI: TAXI_CHECKS, LUNAR: LUNAR_CHECKS}
REQUIRED_BINDINGS = {
    TAXI: (
        "config",
        "attempt",
        "checkpoint",
        "teacher",
        "distillation",
        "reload",
        "verification",
        "qualification_record",
    ),
    LUNAR: (
        "config",
        "attempt",
        "checkpoint",
        "selection",
        "latest_checkpoint",
        "qualification_record",
    ),
}


def qualification_summary(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    checkpoint_sha256: Optional[str],
) -> Dict[str, Any]:
    """Recompute qualification quality from the episodes, not record fields."""
    env_name = config["env_name"]
    qual_cfg = config["qualification"]
    qual = record.get("qualification") if isinstance(record, dict) else None
    qual = qual if isinstance(qual, dict) else {}
    returns = qual.get("episode_returns")
    lengths = qual.get("episode_lengths")
    rec_cfg = record.get("config") if isinstance(record, dict) else None
    rec_cfg = rec_cfg if isinstance(rec_cfg, dict) else {}
    count_ok = (
        isinstance(returns, list)
        and isinstance(lengths, list)
        and len(returns) == len(lengths) == int(qual_cfg["episodes"])
        and all(
            isinstance(r, (int, float)) and not isinstance(r, bool) and math.isfinite(r)
            for r in returns
        )
    )
    mean = float(sum(returns) / len(returns)) if count_ok else None
    raw = float(config["gates"]["per_env_threshold"]["raw_threshold"])
    quality = task_experts.expert_quality(env_name, mean, raw)
    checks = {
        "qualification_complete": record.get("status") == pilot.COMPLETE_STATUS
        and record.get("qualified") is True
        and count_ok,
        "qualification_settings": rec_cfg.get("qualification_seed")
        == int(qual_cfg["seed"])
        and rec_cfg.get("eval_episodes") == int(qual_cfg["episodes"])
        and record.get("env_name") == env_name
        and rec_cfg.get("threshold") == config["gates"]["per_env_threshold"],
        "qualification_candidate_bytes": checkpoint_sha256 is not None
        and record.get("expert_sha256") == checkpoint_sha256,
        "per_env_gate": quality["per_env_passed"],
        "global_gate": quality["global_passed"],
    }
    if "raw_mean_min" in qual_cfg:
        checks["raw_mean_min"] = mean is not None and mean >= float(
            qual_cfg["raw_mean_min"],
        )
    return {
        "checks": checks,
        "recomputed_mean_return": mean,
        "quality": quality,
        "episode_returns": returns,
        "episode_lengths": lengths,
    }


def run_qualification(
    config: Mapping[str, Any],
    checkpoint: pathlib.Path,
    attempt: Attempt,
    checkpoint_sha256: str,
    deadline: datetime.datetime,
) -> Dict[str, Any]:
    """Run the unchanged candidate qualification once and summarize it.

    The candidate's held-out token is created first; if it exists already the
    qualification is not run.

    Raises:
        Refused: If the candidate already used its held-out games.
    """
    attempt.claim_heldout(checkpoint_sha256)
    qdir = attempt.dir / QUALIFICATION_DIR
    t0 = time.monotonic()
    code = task_experts.qualify_candidate(
        config["env_name"],
        checkpoint,
        qdir,
        qualification_seed=int(config["qualification"]["seed"]),
        eval_episodes=int(config["qualification"]["episodes"]),
        deadline=deadline,
    )
    elapsed = time.monotonic() - t0
    record_path = qdir / task_experts.CANDIDATE_FILE
    record = pilot.read_json(record_path) if record_path.is_file() else {}
    summary = qualification_summary(record, config, checkpoint_sha256)
    try:
        _, identity = task_experts.load_qualified_expert(qdir, config["env_name"])
        # The identity also holds a machine path; keep only digests and quality.
        loader = {
            "loaded": True,
            "identity": {
                k: identity[k]
                for k in (
                    "expert_source",
                    "record_sha256",
                    "config_sha256",
                    "expert_sha256",
                    "policy_state_sha256",
                    "qualification_mean_return",
                    "quality",
                )
            },
        }
    except classical.Refused as exc:
        loader = {"loaded": False, "error": str(exc)}
    summary["checks"]["qualification_loader"] = loader["loaded"]
    summary.update(
        exit_code=code,
        loader=loader,
        record_sha256=pilot.sha256_file(record_path) if record_path.is_file() else None,
        counters=(record.get("counters") or {}).get("expert_qualification"),
        elapsed_seconds=elapsed,
        fields_note=QUALIFICATION_FIELDS_NOTE,
    )
    return summary


def write_acceptance(
    attempt_dir: pathlib.Path,
    config: Mapping[str, Any],
    status: str,
    checks: Mapping[str, bool],
    bindings: Mapping[str, str],
    details: Mapping[str, Any],
) -> Dict[str, Any]:
    """Write ``acceptance.json`` binding every evidence file by digest.

    ``release_allowed`` is true only for a complete attempt in which every
    required check is present and true and every required binding exists.
    """
    attempt_dir = pathlib.Path(attempt_dir)
    env_name = config["env_name"]
    required = REQUIRED_CHECKS[env_name]
    full = {name: bool(checks.get(name, False)) for name in required}
    files = {}
    for key, rel in bindings.items():
        path = attempt_dir / rel
        files[key] = {
            "path": rel,
            "sha256": pilot.sha256_file(path) if path.is_file() else None,
        }
    bound = all(
        key in files and files[key]["sha256"] is not None
        for key in REQUIRED_BINDINGS[env_name]
    )
    release = status == pilot.COMPLETE_STATUS and all(full.values()) and bound
    payload = {
        "schema": ACCEPTANCE_SCHEMA,
        "candidate_id": config["candidate_id"],
        "env_name": env_name,
        "status": status,
        "release_allowed": release,
        "training_method": config["training_method"],
        "config_sha256": pilot.config_digest(config),
        "checks": full,
        "failed_checks": sorted(k for k, ok in full.items() if not ok),
        "files": files,
        "release_rule": "status complete, every required check true and every "
        "required file bound; qualification alone never releases an expert",
        "next_steps_before_learners": "fresh data, references and mask audits "
        "on this expert before any learner run",
        "candidate_b_rule": CANDIDATE_B_RULE,
        "limitations": list(LIMITATIONS),
        **details,
    }
    _write_new_json(attempt_dir / ACCEPTANCE_FILE, payload)
    return payload


def verify_acceptance(
    attempt_dir: pathlib.Path,
    config_path: pathlib.Path,
    *,
    require_current_source: bool = True,
    ledger: Optional[CandidateLedger] = None,
) -> Dict[str, Any]:
    """Re-check an acceptance sidecar from the files; no model is loaded.

    Bindings are checked by digest and the qualification mean is recomputed;
    the environment-specific booleans are trusted as written (`LIMITATIONS`).
    The attempt must record consistent ledger tokens; with ``ledger`` they
    must also be the token files of that ledger.

    Raises:
        Refused: If anything is missing, failed, partial or forged.
    """
    attempt_dir = pathlib.Path(attempt_dir)
    try:
        config_bytes = pathlib.Path(config_path).read_bytes()
        config = json.loads(config_bytes)
    except (OSError, ValueError) as exc:
        raise Refused(f"Unreadable config: {exc}") from None
    if not isinstance(config, dict) or config.get("candidate_id") not in CANDIDATES:
        raise Refused("Config does not name a predeclared candidate")
    if config != build_config(config["candidate_id"]):
        raise Refused("Config differs from the predeclared config")
    try:
        acc_bytes = (attempt_dir / ACCEPTANCE_FILE).read_bytes()
        acc = json.loads(acc_bytes)
        attempt = json.loads((attempt_dir / ATTEMPT_FILE).read_bytes())
    except (OSError, ValueError) as exc:
        raise Refused(f"Unreadable acceptance or attempt: {exc}") from None
    env_name = config["env_name"]
    problems = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            problems.append(what)

    need(acc.get("schema") == ACCEPTANCE_SCHEMA, "schema")
    need(acc.get("candidate_id") == config["candidate_id"], "candidate_id")
    need(acc.get("env_name") == env_name, "env_name")
    need(acc.get("training_method") == config["training_method"], "training_method")
    need(acc.get("status") == pilot.COMPLETE_STATUS, "status")
    need(acc.get("release_allowed") is True, "release_allowed")
    need(acc.get("config_sha256") == pilot.config_digest(config), "config_sha256")
    checks = acc.get("checks") if isinstance(acc.get("checks"), dict) else {}
    need(set(checks) == set(REQUIRED_CHECKS[env_name]), "required check set")
    need(all(v is True for v in checks.values()), "all checks true")
    files = acc.get("files") if isinstance(acc.get("files"), dict) else {}
    for key in REQUIRED_BINDINGS[env_name]:
        entry = files.get(key) if isinstance(files.get(key), dict) else {}
        rel = entry.get("path")
        if not isinstance(rel, str) or rel.startswith("/") or ".." in rel.split("/"):
            problems.append(f"binding {key} path")
            continue
        path = attempt_dir / rel
        actual = pilot.sha256_file(path) if path.is_file() else None
        need(actual is not None and actual == entry.get("sha256"), f"binding {key}")
    need(
        (files.get("config") or {}).get("sha256") == _sha(config_bytes),
        "config file bytes",
    )
    need(attempt.get("status") == pilot.COMPLETE_STATUS, "attempt status")
    need(attempt.get("config_sha256") == pilot.config_digest(config), "attempt config")
    checkpoint_sha = (files.get("checkpoint") or {}).get("sha256")
    need(attempt.get("checkpoint_sha256") == checkpoint_sha, "attempt checkpoint")
    need(
        attempt.get("env_configs_sha256_start")
        == attempt.get("env_configs_sha256_end"),
        "ENV_CONFIGS unchanged during attempt",
    )
    if require_current_source:
        runner = ((attempt.get("source") or {}).get("runner") or {}).get("sha256")
        need(runner == pilot.sha256_file(RUNNER_PATH), "runner source")
    problems.extend(
        f"ledger: {p}" for p in ledger_problems(attempt, config, checkpoint_sha, ledger)
    )
    try:
        record = json.loads((attempt_dir / QUALIFICATION_RECORD).read_bytes())
    except (OSError, ValueError):
        record = {}
    summary = qualification_summary(record, config, checkpoint_sha)
    for name, ok in summary["checks"].items():
        need(ok, f"recomputed {name}")
    need(
        (files.get("qualification_record") or {}).get("path") == QUALIFICATION_RECORD,
        "qualification record path",
    )
    if problems:
        raise Refused(f"Acceptance refused: {problems}")
    return {
        "candidate_id": config["candidate_id"],
        "acceptance_sha256": _sha(acc_bytes),
        "checkpoint_sha256": checkpoint_sha,
        "recomputed_mean_return": summary["recomputed_mean_return"],
        "candidate_ledger": attempt["candidate_ledger"]["path"],
        "ledger_files_checked": ledger is not None,
    }


def _ready_ledger(ledger, candidate_id: str) -> CandidateLedger:
    """The pinned ledger, refusing early if the candidate already ran.

    Raises:
        Refused: If no ledger is given or the execution token exists.
    """
    if not isinstance(ledger, CandidateLedger):
        ledger = CandidateLedger(ledger)
    ledger.require_unclaimed(candidate_id, "execution")
    ledger.require_unclaimed(candidate_id, "heldout")
    return ledger


# ---------------------------------------------------------------------------
# Taxi runner
# ---------------------------------------------------------------------------


def _mask_representatives(env_name: str, reachable: Sequence[int]) -> Dict[str, Any]:
    task = tasks.get_task(env_name)
    reach = set(reachable)
    out = {}
    for rid in task.ladder:
        table = tasks.alias_table(rid)["representative"]
        reps = sorted({int(table[s]) for s in reachable})
        out[rid] = {"representatives": reps, "all_reachable": set(reps) <= reach}
    return out


def run_taxi(
    config: Dict[str, Any],
    config_bytes: bytes,
    output_dir: pathlib.Path,
    deadline: datetime.datetime,
    *,
    ledger: Optional[CandidateLedger] = None,
) -> int:
    """Run one Taxi candidate once under the shared candidate ledger.

    Raises:
        Refused: Before anything runs, if no ledger is given, the candidate
            already has a token, or the output directory is not fresh.
    """
    ledger = _ready_ledger(ledger, config["candidate_id"])
    from stable_baselines3 import PPO

    th.set_num_threads(1)
    attempt = Attempt(output_dir, config, config_bytes, deadline, ledger=ledger)
    rec = attempt.record
    checks: Dict[str, bool] = {"config_predeclared": True}
    bindings = {"config": CONFIG_COPY}
    details: Dict[str, Any] = {"qualification_fields_note": QUALIFICATION_FIELDS_NOTE}
    out = attempt.dir
    status = "failed"
    venv = None
    try:
        checks["cpu_single_thread"] = th.get_num_threads() == 1
        task = tasks.get_task(TAXI)
        description = tasks.describe(task)
        rec["task"] = {
            "task_sha256": tasks.task_sha256(task),
            "episode_cap": description["episode_cap"],
        }
        checks["task_cap"] = (
            description["episode_cap"]
            == config["verification"]["episode_cap"]
            == config["teacher"]["dp_horizon"]
        )
        rollouts.check_deadline(deadline, "Taxi teacher", attempt.clock)
        seed = int(config["distillation"]["seed"])
        venv = env_utils.make_env(
            TAXI, 1, rollouts.make_phase_rng(seed, "expert_training")
        )
        unwrapped = venv.envs[0].unwrapped
        starts = [int(s) for s in np.flatnonzero(unwrapped.initial_state_distrib > 0)]
        t0 = time.monotonic()
        teacher = build_teacher(unwrapped.P, starts, config)
        rec["elapsed_seconds"]["teacher"] = time.monotonic() - t0
        checks["dynamics_deterministic_and_task_digest"] = (
            teacher["dynamics_sha256"] == description["tabular_dynamics_sha256"]
        )
        task_states, task_evidence = tasks.reachable_nonterminal(TAXI)
        checks["reachable_matches_task"] = (
            task_states == teacher["reachable_states"]
            and task_evidence["reset_support"] == teacher["reset_support"]
        )
        checks["reward_semantics"] = checks["planner_complete"] = True
        checks["dp_equal_all_horizons"] = teacher["dp"]["equal_all_horizons"]
        masks = _mask_representatives(TAXI, teacher["reachable_states"])
        checks["mask_representatives_covered"] = all(
            m["all_reachable"] for m in masks.values()
        )
        teacher["mask_representatives"] = masks
        bindings["teacher"] = "teacher/teacher.json"
        _write_new_json(out / bindings["teacher"], teacher)
        rec["teacher"] = {
            k: teacher[k]
            for k in (
                "dynamics_sha256",
                "reset_support_sha256",
                "reachable_states_sha256",
                "labels_sha256",
                "max_distance",
            )
        }
        rec["teacher"]["file_sha256"] = pilot.sha256_file(out / bindings["teacher"])
        attempt.write()
        if not all(checks.values()):
            failed = sorted(k for k, v in checks.items() if not v)
            raise RecoveryFailure(f"Teacher checks failed: {failed}")

        states = teacher["reachable_states"]
        obs = np.zeros((len(states), 500), dtype=np.float32)
        obs[np.arange(len(states)), states] = 1.0
        labels = np.array([teacher["labels"][str(s)] for s in states], dtype=np.int64)
        opt_sets = [set(teacher["dp"]["optimal_action_sets"][str(s)]) for s in states]
        dcfg = config["distillation"]
        model = PPO(
            "MlpPolicy",
            venv,
            policy_kwargs={"net_arch": list(dcfg["net_arch"])},
            seed=seed,
            device="cpu",
            verbose=0,
        )
        critic_before = _state_digest(model.policy, lambda n: not _is_actor(n))
        rollouts.check_deadline(deadline, "Taxi distillation", attempt.clock)
        t0 = time.monotonic()
        fit = distill_actor(
            model.policy, obs, labels, opt_sets, dcfg, deadline, attempt.clock
        )
        rec["elapsed_seconds"]["distillation"] = time.monotonic() - t0
        rec["costs"]["distillation_updates"] = fit["updates"]
        rec["costs"]["distillation_forward_state_evaluations"] = fit[
            "forward_state_evaluations"
        ]
        rec["distillation"] = {
            "stop_reason": fit["stop_reason"],
            "updates": fit["updates"],
            "error": fit.get("error"),
        }
        attempt.write()
        ce_max = float(dcfg["stop_rule"]["self_ce_max"])
        final = fit["final"] or {}
        checks["distill_canonical_agreement"] = final.get(
            "canonical_agreement_count"
        ) == len(states)
        checks["distill_optimal_set_agreement"] = final.get(
            "optimal_set_agreement_count"
        ) == len(states)
        checks["distill_self_ce"] = final.get("self_ce", math.inf) <= ce_max
        # Keep the model and the pre-reload fit whatever happened above; a
        # failure to keep them is recorded and makes the attempt fail.
        lost: List[str] = []
        try:
            checks["critic_untouched"] = (
                _state_digest(model.policy, lambda n: not _is_actor(n)) == critic_before
            )
            fit["critic_state_sha256"] = critic_before
            fit["policy_state_sha256"] = pilot.policy_state_sha256(model.policy)
        except Exception as exc:
            lost.append(_error_text("policy digests", exc))
        ckpt, ckpt_sha = out / "distill" / "model.zip", None
        try:
            ckpt.parent.mkdir()
            model.save(ckpt)  # saved whatever the outcome, never discarded
            ckpt_sha = pilot.sha256_file(ckpt)
            rec["checkpoint_sha256"] = ckpt_sha
            rec["selected_vs_latest"] = {
                "selected": {"path": "distill/model.zip", "sha256": ckpt_sha},
                "latest": {"path": "distill/model.zip", "sha256": ckpt_sha},
                "note": "one distillation run; the saved model is both selected "
                "and latest",
            }
            bindings["checkpoint"] = "distill/model.zip"
            fit["checkpoint"] = {"path": "distill/model.zip", "sha256": ckpt_sha}
        except Exception as exc:
            lost.append(_error_text("model save", exc))
        fit["preservation_errors"] = list(lost)
        try:  # the pre-reload fit, written once and never rewritten
            _write_new_json(out / "distill" / "distillation.json", fit)
            bindings["distillation"] = "distill/distillation.json"
            rec["distillation"]["file_sha256"] = pilot.sha256_file(
                out / bindings["distillation"]
            )
        except Exception as exc:
            lost.append(_error_text("fit evidence", exc))
        for reason in lost:
            attempt.fail_reason(f"preservation failed: {reason}")
        reloaded, reload_error = None, None
        if fit.get("error") is None and ckpt_sha is not None:
            reload_ev: Dict[str, Any] = {
                "checkpoint": {"path": "distill/model.zip", "sha256": ckpt_sha},
                "fit_policy_state_sha256": fit.get("policy_state_sha256"),
            }
            try:
                reloaded = PPO.load(ckpt, device="cpu")
                again = fit_metrics(reloaded.policy, obs, labels, opt_sets)
                reload_ev["metrics"] = again
                reload_ev["policy_state_sha256"] = pilot.policy_state_sha256(
                    reloaded.policy
                )
                checks["reload_policy_state_equal"] = (
                    fit.get("policy_state_sha256") is not None
                    and reload_ev["policy_state_sha256"] == fit["policy_state_sha256"]
                )
                checks["reload_canonical_agreement"] = again[
                    "canonical_agreement_count"
                ] == len(states)
                checks["reload_self_ce"] = again["self_ce"] <= ce_max
            except Exception as exc:
                reload_error = _error_text("reload", exc)
                reload_ev["error"] = reload_error
                for name in (
                    "reload_policy_state_equal",
                    "reload_canonical_agreement",
                    "reload_self_ce",
                ):
                    checks[name] = False
            try:  # separate from the fit, so a reload failure loses nothing
                _write_new_json(out / "distill" / "reload.json", reload_ev)
                bindings["reload"] = "distill/reload.json"
            except Exception as exc:
                lost.append(_error_text("reload evidence", exc))
                attempt.fail_reason(f"preservation failed: {lost[-1]}")
        attempt.write()
        if fit.get("error") is not None:
            raise RecoveryFailure(
                f"Distillation raised {fit['error']}; model and partial fit kept",
            )
        if lost:
            raise RecoveryFailure(f"Distillation evidence not preserved: {lost}")
        if reload_error is not None:
            raise RecoveryFailure(
                f"Reload of the distilled model failed: {reload_error}"
            )
        if fit["stop_reason"] == "deadline":
            raise rollouts.DeadlineExceeded("Deadline during distillation")
        if not all(checks.values()):
            raise RecoveryFailure("Distillation checks failed")

        rollouts.check_deadline(
            deadline, "Taxi reset-support verification", attempt.clock
        )
        vcfg = config["verification"]
        expected = {
            s: (teacher["distance"][str(s)], teacher["dp"]["optimal_value"][str(s)])
            for s in teacher["reset_support"]
        }
        venv.close()
        venv = env_utils.make_env(TAXI, 1, rollouts.make_phase_rng(seed, "eval"))
        checks["task_cap"] = checks["task_cap"] and (
            tasks.venv_episode_cap(venv) == vcfg["episode_cap"]
        )

        def predict(o):
            return int(reloaded.predict(o, deterministic=True)[0])

        t0 = time.monotonic()
        ver = verify_reset_support(
            venv.envs[0],
            predict,
            expected,
            config["teacher"]["rewards"],
            seed_start=int(vcfg["reset_seed_start"]),
            max_resets=int(vcfg["max_resets"]),
            deadline=deadline,
            clock=attempt.clock,
        )
        rec["elapsed_seconds"]["verification"] = time.monotonic() - t0
        rec["costs"]["verification"] = {
            k: ver[k] for k in ("resets", "env_transitions", "policy_queries")
        }
        checks["reset_support_covered"] = (
            ver["checks"]["reset_support_covered"]
            and ver["checks"]["no_unexpected_start"]
        )
        checks["rollouts_delivered"] = ver["checks"]["all_delivered"]
        checks["rollouts_no_illegal"] = ver["checks"]["no_illegal_actions"]
        checks["rollouts_optimal"] = ver["checks"]["optimal_length_and_return"]
        bindings["verification"] = "verification/reset_support.json"
        _write_new_json(out / bindings["verification"], ver)
        attempt.write()
        if ver["stop_reason"] == "deadline":
            raise rollouts.DeadlineExceeded(
                "Deadline during reset-support verification"
            )
        if not all(checks.values()):
            raise RecoveryFailure("Reset-support verification failed")

        rollouts.check_deadline(deadline, "Taxi qualification", attempt.clock)
        qual = run_qualification(config, ckpt, attempt, ckpt_sha, deadline)
        checks.update(qual["checks"])
        rec["costs"]["qualification"] = qual["counters"]
        rec["elapsed_seconds"]["qualification"] = qual["elapsed_seconds"]
        details["qualification"] = {
            k: qual[k]
            for k in (
                "recomputed_mean_return",
                "quality",
                "exit_code",
                "record_sha256",
                "loader",
            )
        }
        bindings["qualification_record"] = QUALIFICATION_RECORD
        checks["env_configs_unchanged"] = (
            env_configs_sha256() == rec["env_configs_sha256_start"]
        )
        status = pilot.COMPLETE_STATUS if all(checks.values()) else "failed"
        if status != pilot.COMPLETE_STATUS:
            attempt.fail_reason(
                f"failed checks {sorted(k for k, v in checks.items() if not v)}"
            )
        bindings["attempt"] = ATTEMPT_FILE
        attempt.finish(status)
    except rollouts.DeadlineExceeded as exc:
        status = "partial"
        bindings["attempt"] = ATTEMPT_FILE
        attempt.finish(status, exc)
    except Exception as exc:  # recorded, never retried
        logger.exception("Taxi attempt failed")
        status = "failed"
        bindings["attempt"] = ATTEMPT_FILE
        attempt.finish(status, exc)
    finally:
        if venv is not None:
            venv.close()
    details["costs"] = rec["costs"]
    details["source_combined_sha256"] = rec["source"]["combined_sha256"]
    details["task"] = rec.get("task")
    details["teacher"] = rec.get("teacher")
    details["distillation"] = rec.get("distillation")
    details["candidate_ledger"] = rec.get("candidate_ledger")
    payload = write_acceptance(out, config, status, checks, bindings, details)
    return EXIT_RELEASED if payload["release_allowed"] else EXIT_NOT_RELEASED


# ---------------------------------------------------------------------------
# LunarLander: actual-step budget, selection rule, checkpoints
# ---------------------------------------------------------------------------


def usable_budget(requested: int, rollout: int) -> Tuple[int, int]:
    """Largest whole number of rollouts within ``requested`` transitions.

    Raises:
        ValueError: On non-positive inputs or a request below one rollout.
    """
    requested, rollout = int(requested), int(rollout)
    if requested <= 0 or rollout <= 0:
        raise ValueError("Budget and rollout size must be positive")
    n = requested // rollout
    if n == 0:
        raise ValueError(f"Budget {requested} is below one rollout of {rollout}")
    return n * rollout, n


def check_targets(usable: int, rollout: int, every: int) -> List[int]:
    """Cumulative transition counts at which a check runs."""
    step = int(every) * int(rollout)
    if usable % rollout or step <= 0:
        raise ValueError("Usable budget must be whole rollouts")
    targets = list(range(step, usable + 1, step))
    if not targets or targets[-1] != usable:
        targets.append(usable)
    return targets


@dataclasses.dataclass
class SelectionState:
    """The inherited expert_training convergence rule, as serializable state."""

    best_norm: Optional[float] = None
    best_self_ce: Optional[float] = None
    since_best: int = 0
    window: List[float] = dataclasses.field(default_factory=list)

    def update(
        self,
        norm: float,
        self_ce: float,
        num_timesteps: int,
        rule: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if not (math.isfinite(norm) and math.isfinite(self_ce)):
            raise RecoveryFailure(f"Non-finite selection metrics {norm}, {self_ce}")
        tol, patience = float(rule["plateau_tolerance"]), int(rule["patience"])
        best_norm = -math.inf if self.best_norm is None else self.best_norm
        best_ce = math.inf if self.best_self_ce is None else self.best_self_ce
        improved = norm > best_norm + tol or self_ce < best_ce - tol
        if improved:
            self.best_norm = max(best_norm, norm)
            self.best_self_ce = min(best_ce, self_ce)
            self.since_best = 0
        else:
            self.since_best += 1
        self.window = (self.window + [float(norm)])[-patience:]
        gates = {
            "min_transitions": num_timesteps >= int(rule["min_env_transitions"]),
            "window_full": len(self.window) >= patience,
            "window_min": bool(self.window)
            and min(self.window) >= float(rule["normalized_threshold"]),
            "self_ce": self_ce <= float(rule["self_ce_max"]),
            "plateau": self.since_best >= patience,
        }
        return {"improved": improved, "gates": gates, "passed": all(gates.values())}

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def normalize(mean_return: float, env_name: str) -> float:
    ref = env_baselines.REFERENCE_BASELINES[env_name]
    lo, hi = float(ref["random_score"]), float(ref["expert_score"])
    return (mean_return - lo) / (hi - lo)


def capture_rng_state() -> bytes:
    return pickle.dumps(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": th.get_rng_state().numpy().tobytes(),
        },
    )


class CheckpointStore:
    """Immutable per-check checkpoints plus an atomically replaced pointer."""

    def __init__(
        self, root: pathlib.Path, rng_state: Callable[[], bytes] = capture_rng_state
    ):
        self.root = pathlib.Path(root)
        self.rng_state = rng_state

    def save(self, model, name: str, progress: Dict[str, Any]) -> Dict[str, Any]:
        directory = self.root / "checkpoints" / name
        directory.mkdir(parents=True, exist_ok=False)
        model.save(directory / "model.zip")
        rng = self.rng_state()
        with open(directory / "rng_state.pkl", "xb") as f:
            f.write(rng)
            f.flush()
            os.fsync(f.fileno())
        identity = {
            "name": name,
            "dir": f"checkpoints/{name}",
            "model_sha256": pilot.sha256_file(directory / "model.zip"),
            "rng_state_sha256": _sha(rng),
            "num_timesteps": int(model.num_timesteps),
        }
        progress = dict(progress, checkpoint=identity)
        identity["progress_sha256"] = _write_new_json(
            directory / "progress.json", progress
        )
        return identity

    def point_latest(self, identity: Mapping[str, Any]) -> None:
        pilot.atomic_write_json(self.root / "latest.json", dict(identity))


def train_with_selection(
    model,
    *,
    targets: Sequence[int],
    usable: int,
    rollout: int,
    rule: Mapping[str, Any],
    state: SelectionState,
    first_check: int,
    evaluate: Callable[[Any, int], Dict[str, Any]],
    store: CheckpointStore,
    deadline: datetime.datetime,
    clock: Callable[[], datetime.datetime],
    on_check: Callable[[Dict[str, Any]], None] = lambda _: None,
    best: Optional[Dict[str, Any]] = None,
    env_name: str = LUNAR,
) -> Dict[str, Any]:
    """Learn in budget-aligned segments, check after each, stop at selection.

    Every check saves an immutable checkpoint before the selection decision
    is acted on. On any exception the in-memory model is saved as a failure
    checkpoint. The loop never asks ``learn`` for more than the remaining
    usable budget and fails if the model's actual count differs.
    """
    result: Dict[str, Any] = {
        "stop": None,
        "checks": [],
        "selected": None,
        "latest": None,
        "best": best,
        "error": None,
    }
    check = first_check - 1
    try:
        for target in targets:
            if target <= model.num_timesteps:
                continue
            segment = int(target) - int(model.num_timesteps)
            if segment % rollout or target > usable:
                raise BudgetViolation(f"Segment {segment} to {target} is not aligned")
            if clock() >= deadline:
                result["stop"] = "deadline"
                return result
            check += 1
            before = int(model.num_timesteps)
            t0 = time.monotonic()
            model.learn(total_timesteps=segment, reset_num_timesteps=False)
            learn_seconds = time.monotonic() - t0
            actual = int(model.num_timesteps)
            if actual != target or actual > usable:
                raise BudgetViolation(
                    f"Check {check}: model.num_timesteps {actual} != planned {target}",
                )
            ev = evaluate(model, check)
            norm = normalize(ev["mean_return"], env_name)
            decision = state.update(norm, ev["self_ce"], actual, rule)
            if best is None or norm > best["normalized_return"]:
                best = {"check": check, "normalized_return": norm}
            entry = {
                "check": check,
                "num_timesteps_before": before,
                "num_timesteps": actual,
                "segment_transitions": actual - before,
                "learn_seconds": learn_seconds,
                "evaluation": ev,
                "normalized_return": norm,
                "decision": decision,
                "selection_state": state.to_dict(),
                "best_selection_check": dict(best),
            }
            identity = store.save(model, f"check-{check:04d}", entry)
            if best["check"] == check:
                best = dict(best, checkpoint=identity)
            entry["checkpoint"] = identity
            store.point_latest(identity)
            result["latest"], result["best"] = identity, best
            result["checks"].append(entry)
            on_check(entry)
            if decision["passed"]:
                result["selected"] = dict(identity, check=check)
                result["stop"] = "selected"
                return result
        result["stop"] = "budget_exhausted"
        return result
    except Exception as exc:
        result["stop"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            result["failure_checkpoint"] = store.save(
                model,
                f"failure-{check:04d}",
                {"error": result["error"], "selection_state": state.to_dict()},
            )
        except Exception as save_exc:  # keep the original failure visible
            result["failure_checkpoint_error"] = (
                f"{type(save_exc).__name__}: {save_exc}"
            )
        return result


class EpisodeRecorder:
    """VecEnv pass-through that records episode lengths and end flags."""

    def __init__(self, venv) -> None:
        self.venv = venv
        self.num_envs = venv.num_envs
        self.episodes: List[Dict[str, Any]] = []
        self._length = 0

    def reset(self):
        self._length = 0
        return self.venv.reset()

    def step(self, actions):
        obs, rewards, dones, infos = self.venv.step(actions)
        self._length += 1
        if dones[0]:
            truncated = bool(infos[0].get("TimeLimit.truncated", False))
            self.episodes.append(
                {
                    "length": self._length,
                    "end": "time_limit" if truncated else "terminated",
                    "final_step_reward": float(rewards[0]),
                    "landing_or_crash": "unknown: not recorded by the environment "
                    "info; not inferred from rewards",
                },
            )
            self._length = 0
        return obs, rewards, dones, infos


def lunar_selection_eval(config: Mapping[str, Any]) -> Callable[[Any, int], Dict]:
    sel = config["selection"]

    def evaluate(model, check: int) -> Dict[str, Any]:
        rng = np.random.default_rng(
            np.random.SeedSequence(int(sel["seed"]), spawn_key=(check,))
        )
        venv = env_utils.make_env(LUNAR, 1, rng)
        try:
            recorder = EpisodeRecorder(venv)
            t0 = time.monotonic()
            res = eval_utils.eval_policy_rollout(
                model.policy,
                recorder,
                int(sel["episodes"]),
                deterministic=True,
                expert_policy=model.policy,
                safety_step_cap=int(sel["episodes"]) * 1000,
            )
        finally:
            venv.close()
        return {
            "selection_seed": int(sel["seed"]),
            "spawn_key": [check],
            "mean_return": float(res.mean_return),
            "episode_returns": [float(r) for r in res.episode_returns],
            "episodes": recorder.episodes,
            "self_ce": float(res.current_round_ce),
            "env_transitions": int(res.n_steps),
            "policy_queries": 2 * int(res.n_steps),
            "seconds": time.monotonic() - t0,
        }

    return evaluate


def _make_lunar_model(config, venv):
    from stable_baselines3 import PPO

    p = config["ppo"]
    return PPO(
        "MlpPolicy",
        venv,
        policy_kwargs={"net_arch": list(p["net_arch"])},
        n_steps=p["n_steps"],
        batch_size=p["batch_size"],
        gae_lambda=p["gae_lambda"],
        gamma=p["gamma"],
        n_epochs=p["n_epochs"],
        ent_coef=p["ent_coef"],
        learning_rate=p["learning_rate"],
        seed=p["seed"],
        device="cpu",
        verbose=0,
    )


def verify_resume_source(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Resume is not supported; nothing is read.

    Raises:
        Refused: Always.
    """
    raise Refused(RESUME_UNSUPPORTED)


def run_lunar(
    config: Dict[str, Any],
    config_bytes: bytes,
    output_dir: pathlib.Path,
    deadline: datetime.datetime,
    prior_attempt: Optional[pathlib.Path] = None,
    *,
    ledger: Optional[CandidateLedger] = None,
) -> int:
    """Run one LunarLander candidate once, from scratch, under the ledger.

    Raises:
        Refused: Before anything is read, written or trained, if a prior
            attempt is given (resume is not supported), no ledger is given,
            the candidate already has a token, or the output is not fresh.
    """
    if prior_attempt is not None:
        raise Refused(RESUME_UNSUPPORTED)
    ledger = _ready_ledger(ledger, config["candidate_id"])
    th.set_num_threads(1)
    source = source_identity()
    task = tasks.get_task(LUNAR)
    task_sha = tasks.task_sha256(task)
    attempt = Attempt(
        output_dir, config, config_bytes, deadline, source=source, ledger=ledger
    )
    rec = attempt.record
    out = attempt.dir
    rec["task"] = {"task_sha256": task_sha}
    rec["resume"] = RESUME_UNSUPPORTED
    checks: Dict[str, bool] = {
        "config_predeclared": True,
        "cpu_single_thread": th.get_num_threads() == 1,
    }
    bindings = {"config": CONFIG_COPY}
    details: Dict[str, Any] = {"qualification_fields_note": QUALIFICATION_FIELDS_NOTE}
    budget = config["budget"]
    usable, rollout = int(budget["usable_max_env_transitions"]), int(
        budget["rollout_env_transitions"]
    )
    targets = check_targets(usable, rollout, int(budget["check_every_rollouts"]))
    status, venv, result = "failed", None, None
    store = CheckpointStore(out)
    try:
        description = tasks.describe(task)
        rec["task"]["episode_cap"] = description["episode_cap"]
        checks["task_cap"] = description["episode_cap"] == task.episode_cap == 1000
        p = config["ppo"]
        venv = env_utils.make_env(
            LUNAR,
            int(p["n_envs"]),
            rollouts.make_phase_rng(int(p["seed"]), "expert_training"),
        )
        model = _make_lunar_model(config, venv)
        state, first_check, best = SelectionState(), 1, None
        if model.n_envs != int(p["n_envs"]) or model.n_envs * model.n_steps != rollout:
            raise RecoveryFailure("PPO rollout size differs from the config")
        rec["budget"] = dict(budget, check_targets=targets, start_num_timesteps=0)
        attempt.write()

        def on_check(entry):
            rec.setdefault("checks", []).append(
                {
                    k: entry[k]
                    for k in (
                        "check",
                        "num_timesteps",
                        "normalized_return",
                        "decision",
                        "checkpoint",
                    )
                }
            )
            rec["latest_checkpoint"] = entry["checkpoint"]
            rec["costs"]["training_env_transitions"] = entry["num_timesteps"]
            rec["costs"]["model_num_timesteps"] = entry["num_timesteps"]
            ev = entry["evaluation"]
            for key in ("env_transitions", "policy_queries"):
                rec["costs"][f"selection_{key}"] = (
                    rec["costs"].get(f"selection_{key}", 0) + ev[key]
                )
            rec["costs"]["selection_episodes"] = rec["costs"].get(
                "selection_episodes", 0
            ) + len(ev["episode_returns"])
            attempt.write()

        t0 = time.monotonic()
        result = train_with_selection(
            model,
            targets=targets,
            usable=usable,
            rollout=rollout,
            rule=config["selection"],
            state=state,
            first_check=first_check,
            evaluate=lunar_selection_eval(config),
            store=store,
            deadline=deadline,
            clock=attempt.clock,
            on_check=on_check,
            best=best,
        )
        rec["elapsed_seconds"]["training_and_selection"] = time.monotonic() - t0
        rec["costs"]["model_num_timesteps"] = int(model.num_timesteps)
        rec["costs"]["training_env_transitions"] = int(model.num_timesteps)
        checks["budget_within_bound"] = int(model.num_timesteps) <= usable
        checks["budget_exact"] = (result.get("error") or {}).get(
            "type"
        ) != "BudgetViolation" and all(
            e["num_timesteps"] in targets for e in result["checks"]
        )
        selection = {
            "stop": result["stop"],
            "selected": result["selected"],
            "latest": result["latest"],
            "best_selection_check": result["best"],
            "failure_checkpoint": result.get("failure_checkpoint"),
            "error": result.get("error"),
            "rule": config["selection"]["rule"],
        }
        rec["selection"] = selection
        bindings["selection"] = "selection.json"
        _write_new_json(out / "selection.json", selection)
        if result["latest"] is not None:
            bindings["latest_checkpoint"] = f"{result['latest']['dir']}/model.zip"
        attempt.write()
        checks["selection_passed"] = result["stop"] == "selected"
        if result["stop"] == "deadline":
            raise rollouts.DeadlineExceeded("Deadline before the next training segment")
        if result["stop"] == "failed":
            raise RecoveryFailure(f"Training failed: {result['error']}")
        if result["stop"] != "selected":
            raise RecoveryFailure("Budget exhausted without a passing selection check")

        sel = result["selected"]
        src = out / sel["dir"] / "model.zip"
        sel_sha = _copy_exclusive(src, out / "selected" / "model.zip")
        checks["selected_checkpoint_bytes"] = sel_sha == sel["model_sha256"]
        rec["checkpoint_sha256"] = sel_sha
        rec["selected_vs_latest"] = {
            "selected": sel,
            "latest": result["latest"],
            "same": sel["model_sha256"] == result["latest"]["model_sha256"],
        }
        bindings["checkpoint"] = "selected/model.zip"
        attempt.write()
        rollouts.check_deadline(deadline, "LunarLander qualification", attempt.clock)
        qual = run_qualification(
            config, out / "selected" / "model.zip", attempt, sel_sha, deadline
        )
        checks.update(qual["checks"])
        bindings["qualification_record"] = QUALIFICATION_RECORD
        rec["costs"]["qualification"] = qual["counters"]
        rec["elapsed_seconds"]["qualification"] = qual["elapsed_seconds"]
        lengths = qual["episode_lengths"] or []
        details["qualification"] = {
            k: qual[k]
            for k in (
                "recomputed_mean_return",
                "quality",
                "exit_code",
                "record_sha256",
                "loader",
                "episode_returns",
                "episode_lengths",
            )
        }
        details["qualification"]["episode_outcomes"] = [
            (
                "terminated before the cap (landing or crash unknown)"
                if n < 1000
                else "reached the cap (end flag not recorded by the evaluator)"
            )
            for n in lengths
        ]
        checks["env_configs_unchanged"] = (
            env_configs_sha256() == rec["env_configs_sha256_start"]
        )
        status = pilot.COMPLETE_STATUS if all(checks.values()) else "failed"
        if status != pilot.COMPLETE_STATUS:
            attempt.fail_reason(
                f"failed checks {sorted(k for k, v in checks.items() if not v)}; "
                "held-out results never resume training"
            )
        bindings["attempt"] = ATTEMPT_FILE
        attempt.finish(status)
    except rollouts.DeadlineExceeded as exc:
        status = "partial"
        bindings["attempt"] = ATTEMPT_FILE
        attempt.finish(status, exc)
    except Exception as exc:
        logger.exception("LunarLander attempt failed")
        status = "failed"
        bindings["attempt"] = ATTEMPT_FILE
        attempt.finish(status, exc)
    finally:
        if venv is not None:
            venv.close()
    details["costs"] = rec["costs"]
    details["source_combined_sha256"] = rec["source"]["combined_sha256"]
    details["task"] = rec.get("task")
    details["selected_vs_latest"] = rec.get("selected_vs_latest")
    details["resume"] = RESUME_UNSUPPORTED
    details["candidate_ledger"] = rec.get("candidate_ledger")
    payload = write_acceptance(out, config, status, checks, bindings, details)
    return EXIT_RELEASED if payload["release_allowed"] else EXIT_NOT_RELEASED


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _parse_deadline(text: str) -> datetime.datetime:
    deadline = rollouts.parse_utc_deadline(text)
    if deadline <= rollouts.utc_now():
        raise Refused(f"Deadline {deadline.isoformat()} has already passed")
    return deadline


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if argv and argv[0] == "lunar-resume":  # refused before parsing anything
        logger.error("Refused: %s", RESUME_UNSUPPORTED)
        return EXIT_REFUSED
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("taxi", "lunar"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=pathlib.Path, required=True)
        cmd.add_argument("--output-dir", type=pathlib.Path, required=True)
        cmd.add_argument("--candidate-ledger", type=pathlib.Path, required=True)
        cmd.add_argument("--deadline", required=True, help="ISO 8601 UTC, e.g. ...Z")
    ver = sub.add_parser("verify-acceptance")
    ver.add_argument("--config", type=pathlib.Path, required=True)
    ver.add_argument("--attempt-dir", type=pathlib.Path, required=True)
    ver.add_argument("--candidate-ledger", type=pathlib.Path, default=None)
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return EXIT_USAGE
    try:
        if args.command == "verify-acceptance":
            ledger = None
            if args.candidate_ledger is not None:
                ledger = CandidateLedger(args.candidate_ledger)
            out = verify_acceptance(args.attempt_dir, args.config, ledger=ledger)
            print(json.dumps(out, indent=2))
            return EXIT_RELEASED
        deadline = _parse_deadline(args.deadline)
        config, raw = load_config(args.config, args.command)
        ledger = CandidateLedger(args.candidate_ledger)
        run = run_taxi if args.command == "taxi" else run_lunar
        return run(config, raw, args.output_dir, deadline, ledger=ledger)
    except (Refused, ValueError) as exc:
        logger.error("Refused: %s", exc)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
