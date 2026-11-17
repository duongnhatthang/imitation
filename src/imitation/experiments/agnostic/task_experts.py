"""Qualified experts for the classical restriction tasks.

Two kinds of evidence qualify an expert, and records keep them apart:

``preparation``
    A stage 1 ``preparation.json`` written by ``run_classical
    prepare-expert`` (training and qualification in one protocol), verified
    by `classical.load_verified_expert`.
``candidate_qualification``
    An existing checkpoint trained outside this protocol (for example an older
    expert cache), re-measured here with the original qualification: fresh
    deterministic episodes on the original factory, the same phase RNG, step
    cap and frozen per-environment threshold as ``prepare-expert``. Its
    training provenance is unknown and recorded as such; it is never written
    as, or read as, a preparation record. The checkpoint bytes are copied into
    the qualification directory and pinned by digest. A candidate that fails
    is kept as evidence; nothing is retried.

Loading fails closed for both kinds (`load_qualified_expert`): records are
parsed from the hashed bytes, the qualification episodes are checked and
their mean recomputed, and quality comes from that mean and the current
frozen threshold.

`expert_quality` reports both checks for either kind: the per-environment
threshold the qualification used (``pilot.frozen_expert_threshold``) and the
global ``env_baselines.EXPERT_QUALITY_THRESHOLD`` (0.80) check on the same
qualification mean return. An expert that passes only the first is
``relaxed_only``; the restriction runner refuses it unless explicitly allowed,
and records that permission.
"""

import datetime
import hashlib
import io
import json
import logging
import math
import os
import pathlib
import time
from typing import Any, Callable, Dict, Optional, Tuple

from stable_baselines3 import PPO

from imitation.experiments.agnostic import (
    classical,
    pilot,
    rollouts,
    run_classical,
    tasks,
)
from imitation.experiments.ftrl import env_baselines, env_utils, expert_training

logger = logging.getLogger(__name__)

CANDIDATE_SCHEMA = "agnostic-expert-candidate-qualification/1"
CANDIDATE_FILE = "candidate_qualification.json"
CANDIDATE_CHECKPOINT = "candidate/model.zip"

TRAINING_PROVENANCE = (
    "unknown: trained outside this protocol; no preparation record exists, so "
    "the training seeds, budget and convergence evidence are not verified and "
    "disjointness of the qualification seed from training seeds is unverifiable"
)


def expert_quality(env_name: str, mean_return: Optional[float], raw_threshold):
    """Per-environment and global quality checks on one qualification mean."""
    ref = env_baselines.REFERENCE_BASELINES[env_name]
    lo, hi = float(ref["random_score"]), float(ref["expert_score"])
    fraction = float(env_baselines.EXPERT_QUALITY_THRESHOLD)
    global_raw = lo + fraction * (hi - lo)
    known = isinstance(mean_return, (int, float)) and math.isfinite(mean_return)
    per_env = bool(known and mean_return >= float(raw_threshold))
    global_ok = bool(known and mean_return >= global_raw)
    return {
        "qualification_mean_return": mean_return if known else None,
        "per_env_raw_threshold": float(raw_threshold),
        "per_env_passed": per_env,
        "global_fraction": fraction,
        "global_raw_threshold": global_raw,
        "global_passed": global_ok,
        "relaxed_only": per_env and not global_ok,
        "reference_baselines": {"random_score": lo, "expert_score": hi},
        "rule": "passed means mean_return >= raw threshold; global raw = "
        "random_score + 0.80 * (expert_score - random_score)",
    }


# ---------------------------------------------------------------------------
# Candidate qualification
# ---------------------------------------------------------------------------


def build_candidate_config(
    env_name: str,
    candidate_sha256: str,
    candidate_label: str,
    *,
    qualification_seed: int,
    eval_episodes: int,
    max_eval_env_steps: Optional[int] = None,
    cached_baselines: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Config that identifies one candidate qualification.

    Raises:
        ValueError: On an unknown task or non-positive counts.
    """
    task = tasks.get_task(env_name)
    if int(eval_episodes) <= 0:
        raise ValueError("eval_episodes must be positive")
    if int(qualification_seed) < 0:
        raise ValueError("qualification_seed must be non-negative")
    description = tasks.describe(task)
    cap = int(description["episode_cap"])
    if max_eval_env_steps is None:
        max_eval_env_steps = int(eval_episodes) * cap
    if int(max_eval_env_steps) <= 0:
        raise ValueError("max_eval_env_steps must be positive")
    threshold = pilot.frozen_expert_threshold(env_name)
    return {
        "schema": CANDIDATE_SCHEMA,
        "env_name": env_name,
        "env_episode_cap": cap,
        "env_episode_cap_rule": description["episode_cap_rule"],
        "task_sha256": tasks.task_sha256(task),
        "qualification_seed": int(qualification_seed),
        "eval_episodes": int(eval_episodes),
        "threshold": threshold,
        "global_quality": expert_quality(env_name, None, threshold["raw_threshold"]),
        "candidate": {
            "label": candidate_label,
            "sha256": candidate_sha256,
            "training_provenance": TRAINING_PROVENANCE,
            "cached_baselines_unverified": cached_baselines,
        },
        "expert": {"algorithm": "PPO", "device": "cpu"},
        "qualification": {
            "action_rule": "deterministic predict (argmax)",
            "n_envs": 1,
            "env_reset_seed": int(qualification_seed),
            "construction_rng": (
                "SeedSequence(qualification_seed, spawn_key=(expert_qualification,))"
            ),
            "max_env_steps": int(max_eval_env_steps),
        },
    }


def _cached_baselines(candidate: pathlib.Path) -> Optional[Dict[str, Any]]:
    path = candidate.parent / "baselines.json"
    if not path.is_file():
        return None
    raw = path.read_bytes()
    try:
        content = pilot.read_json(path)
    except ValueError:
        content = None
    return {
        "file": "baselines.json",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "content": content,
        "note": "cached alongside the candidate; provenance unverified, not used",
    }


def qualify_candidate(
    env_name: str,
    candidate_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    qualification_seed: int,
    eval_episodes: int,
    deadline: datetime.datetime,
    max_eval_env_steps: Optional[int] = None,
    clock: Callable[[], datetime.datetime] = rollouts.utc_now,
) -> int:
    """Re-measure an existing checkpoint with the original qualification.

    Returns:
        0 for a complete qualified candidate, 1 for failed, partial or not
        qualified, 3 if the candidate is unreadable or the output is not fresh.

    Raises:
        ValueError: On invalid arguments, before anything is written.
    """
    deadline = rollouts.ensure_aware_utc(deadline)
    candidate_path = pathlib.Path(candidate_path)
    try:
        data = candidate_path.read_bytes()
    except OSError as exc:
        logger.error("Unreadable candidate %s: %s", candidate_path, exc)
        return classical.EXIT_REFUSED
    digest = hashlib.sha256(data).hexdigest()
    # Keep machine paths out of records: the digest identifies the candidate.
    label = "/".join(candidate_path.parts[-2:])
    config = build_candidate_config(
        env_name,
        digest,
        label,
        qualification_seed=qualification_seed,
        eval_episodes=eval_episodes,
        max_eval_env_steps=max_eval_env_steps,
        cached_baselines=_cached_baselines(candidate_path),
    )
    output_dir = pathlib.Path(output_dir)
    try:
        claim = classical.claim_fresh_output_dir(output_dir, clock)
    except classical.Refused as exc:
        logger.error("Refused: %s", exc)
        return classical.EXIT_REFUSED

    record_path = output_dir / CANDIDATE_FILE
    started = time.monotonic()
    record: Dict[str, Any] = {
        "status": "running",
        "qualified": False,
        "protocol": CANDIDATE_SCHEMA,
        "env_name": env_name,
        "expert_sha256": digest,
        "claim": claim,
        "config": config,
        "config_sha256": pilot.config_digest(config),
        "deadline_utc": deadline.isoformat(),
        "started_at_utc": clock().isoformat(),
        "package_versions": pilot.package_versions(),
        "source_fingerprint": pilot.source_fingerprint(),
        "counters": {},
        "elapsed_seconds": {},
    }

    def finish(status: str, exc: Optional[BaseException] = None) -> int:
        record["status"] = status
        record["qualified"] = status == pilot.COMPLETE_STATUS
        if exc is not None:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        mean = (record.get("qualification") or {}).get("mean_return")
        record["quality"] = expert_quality(
            env_name,
            mean,
            config["threshold"]["raw_threshold"],
        )
        record["finished_at_utc"] = clock().isoformat()
        record["elapsed_seconds"]["total"] = time.monotonic() - started
        pilot.atomic_write_json(record_path, record)
        logger.info("qualify-candidate %s: status=%s", env_name, status)
        return 0 if record["qualified"] else classical.EXIT_INCOMPLETE

    pilot.atomic_write_json(record_path, record)
    try:
        checkpoint = output_dir / CANDIDATE_CHECKPOINT
        checkpoint.parent.mkdir()
        with open(checkpoint, "xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if pilot.sha256_file(checkpoint) != digest:
            raise RuntimeError("Copied candidate bytes differ from the source")
        model = PPO.load(io.BytesIO(data), device="cpu")
        record["checkpoint"] = {
            "path": CANDIDATE_CHECKPOINT,
            "sha256": digest,
            "policy_state_sha256": pilot.policy_state_sha256(model.policy),
        }
        probe = env_utils.make_env(env_name, 1, rollouts.make_phase_rng(0, "eval"))
        try:
            spaces_ok = (
                model.observation_space == probe.observation_space
                and model.action_space == probe.action_space
            )
        finally:
            probe.close()
        if not spaces_ok:
            raise ValueError("Candidate spaces differ from the original factory")
        pilot.atomic_write_json(record_path, record)
    except Exception as exc:
        return finish("failed", exc)

    try:
        rollouts.check_deadline(deadline, "candidate qualification", clock)
    except rollouts.DeadlineExceeded as exc:
        return finish("partial", exc)
    t0 = time.monotonic()
    record["counters"]["expert_qualification"] = rollouts.PhaseCounters(
        phase="expert_qualification",
    ).to_dict()
    venv = None
    result = None
    try:
        venv = env_utils.make_env(
            env_name,
            n_envs=1,
            rng=rollouts.make_phase_rng(qualification_seed, "expert_qualification"),
        )
        result = rollouts.evaluate_episodes(
            venv,
            model,
            config["eval_episodes"],
            phase="expert_qualification",
            deadline=deadline,
            max_env_steps=config["qualification"]["max_env_steps"],
            behavior_is_expert=True,
            deterministic=True,
            env_seed=config["qualification_seed"],
            clock=clock,
        )
    except rollouts.EvaluationError as exc:
        result = exc.result
        error: Optional[BaseException] = exc.__cause__ or exc
    except Exception as exc:
        error = exc
    else:
        error = None
    finally:
        if venv is not None:
            venv.close()
        record["elapsed_seconds"]["expert_qualification"] = time.monotonic() - t0
    if result is not None:
        returns = result.episode_returns
        counters = result.counters.to_dict()
        record["counters"]["expert_qualification"] = counters
        record["qualification"] = {
            "episode_returns": returns,
            "episode_lengths": result.episode_lengths,
            "mean_return": float(sum(returns) / len(returns)) if returns else None,
            "raw_threshold": config["threshold"]["raw_threshold"],
            "env_reset_seed": config["qualification_seed"],
            "counters": counters,
        }
    if error is not None:
        return finish("failed", error)
    if not result.completed:
        reason = result.counters.stop_reason
        label_error = (
            rollouts.DeadlineExceeded
            if reason == "deadline"
            else rollouts.StepBudgetExhausted
        )
        return finish("partial", label_error(f"Qualification stopped early: {reason}"))
    passed = record["qualification"]["mean_return"] >= (
        config["threshold"]["raw_threshold"]
    )
    return finish(pilot.COMPLETE_STATUS if passed else "not_qualified")


# ---------------------------------------------------------------------------
# Loading a qualified expert of either kind
# ---------------------------------------------------------------------------

# Sources that determine the environment, rollouts, evaluation and training of
# a preparation. A preparation is compatible only if these bytes equal the
# current ones; the preparation driver itself (``pilot.py``,
# ``run_classical.py`` and files added to its fingerprint later) may differ,
# so legitimate historical preparations keep loading.
PREPARATION_ENVIRONMENT_SOURCES = (
    "experiments/agnostic/rollouts.py",
    "experiments/ftrl/env_baselines.py",
    "experiments/ftrl/env_utils.py",
    "experiments/ftrl/eval_utils.py",
    "experiments/ftrl/expert_training.py",
    "util/util.py",
)
PREPARATION_COMPATIBILITY_RULE = (
    "config rebuilt by the current build_preparation_config with no "
    "convergence override, the frozen per-environment threshold and the "
    "current task cap; environment, rollout, evaluation and training source "
    "bytes and the gymnasium version unchanged; other package versions and "
    "preparation-driver source may differ"
)
# Environment-source lineage accepted for the four vector width teachers
# Their records were written before env_utils.py and env_baselines.py gained
# the FrozenLakeSlippery-v0 variant task (registration, factory entry,
# reference scores). That change only adds entries: these four tasks'
# factory configs, wrappers and reference scores are unchanged. Exactly these
# recorded and current digests are accepted, for exactly these tasks; every
# other source, task or digest must still match, and the loaded identity
# records the lineage.
APPROVED_ENVIRONMENT_SOURCE_LINEAGE = {
    "tasks": ("Acrobot-v1", "CartPole-v1", "LunarLander-v2", "MountainCar-v0"),
    "files": {
        "experiments/ftrl/env_utils.py": {
            "recorded": "f9af716a2cbcf4b6914f2d4e9e39d4f7bd3ab8145c23da58bb64fd489d8ef0c5",
            "current": "6c7d6d13a8dc4c2b15e52eb5703703fc275d493c3b8184f53750865c66ef009f",
        },
        "experiments/ftrl/env_baselines.py": {
            "recorded": "3480ed8c9d53ba023afb51eeb6e74e7c7c7ff648feeca4f2a93881a0a5898639",
            "current": "fbeb1043653d45b01a4997a4d6b522f71952a33d1fcaeb82704171d3d7b66b41",
        },
    },
    "reason": "the later source adds FrozenLake variants and tabular reference helpers; "
    "the four vector tasks' configs, wrappers and reference scores are "
    "unchanged",
}


def _environment_sources_match(env_name, files, current_sources):
    """(passed, accepted lineage entries) of the environment-source check."""
    if not isinstance(files, dict):
        return False, {}
    lineage = {}
    approved = APPROVED_ENVIRONMENT_SOURCE_LINEAGE
    for rel, sha in current_sources.items():
        recorded = files.get(rel)
        if recorded == sha:
            continue
        entry = approved["files"].get(rel)
        if (
            env_name in approved["tasks"]
            and entry is not None
            and recorded == entry["recorded"]
            and sha == entry["current"]
        ):
            lineage[rel] = dict(entry)
            continue
        return False, {}
    return True, lineage


ARCHITECTURE_RULE = (
    "an explicit-architecture preparation is rebuilt with its claimed actor "
    "widths, and the loaded policy's actual modules must match the claim and "
    "the architecture recorded at preparation"
)


def _read_record(path: pathlib.Path) -> Tuple[bytes, Any]:
    """Read a record once and parse exactly the bytes that are hashed."""
    try:
        raw = path.read_bytes()
        return raw, json.loads(raw)
    except (OSError, ValueError) as exc:
        raise classical.Refused(f"Unreadable record {path}: {exc}") from None


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_dict(value) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _evidence_checks(
    record: Dict[str, Any],
    config: Dict[str, Any],
    cap: int,
) -> Tuple[Dict[str, bool], Optional[float]]:
    """Check the recorded qualification episodes and recompute their mean.

    Every requested episode must be present with a finite return and a length
    in ``[1, cap]``; the steps must fit the configured budget; the evaluator's
    counters (stored twice by the producers) must agree with each other and
    with the episodes and report a completed evaluation; the reported mean,
    threshold and reset seed must equal the recomputed mean and the config.

    Returns:
        The named checks and the recomputed mean (None if not computable).
    """
    qual = _as_dict(record.get("qualification"))
    settings = _as_dict(config.get("qualification"))
    returns, lengths = qual.get("episode_returns"), qual.get("episode_lengths")
    requested = config.get("eval_episodes")
    lists = isinstance(returns, list) and isinstance(lengths, list)
    count = bool(
        lists
        and _is_int(requested)
        and requested > 0
        and len(returns) == len(lengths) == requested,
    )
    finite = bool(lists and all(_is_real(r) and math.isfinite(r) for r in returns))
    valid = bool(lists and all(_is_int(n) and 1 <= n <= cap for n in lengths))
    total = sum(lengths) if valid else None
    mean = float(sum(returns) / len(returns)) if count and finite else None
    reported = qual.get("mean_return")
    counters = qual.get("counters")
    stored = _as_dict(record.get("counters")).get("expert_qualification")
    max_steps = settings.get("max_env_steps")
    threshold = _as_dict(config.get("threshold")).get("raw_threshold")
    checks = {
        "episode_count": count,
        "returns_finite": finite,
        "lengths_valid": valid,
        "steps_within_budget": total is not None
        and _is_int(max_steps)
        and total <= max_steps,
        "mean_recomputed": mean is not None and _is_real(reported) and reported == mean,
        "counters": isinstance(counters, dict)
        and counters == stored
        and counters.get("phase") == "expert_qualification"
        and counters.get("stop_reason") == "completed"
        and counters.get("completed_episodes") == requested
        and total is not None
        and counters.get("env_transitions") == total
        and counters.get("partial_episode_transitions") == 0,
        "evidence_settings": _is_real(threshold)
        and qual.get("raw_threshold") == threshold
        and qual.get("env_reset_seed")
        == config.get("qualification_seed")
        == settings.get("env_reset_seed"),
    }
    return checks, mean


def _quality(env_name: str, mean: Optional[float]) -> Dict[str, Any]:
    """Quality from the recomputed mean and the current frozen threshold."""
    frozen = pilot.frozen_expert_threshold(env_name)["raw_threshold"]
    return expert_quality(env_name, mean, frozen)


def _refuse_failed(what: str, checks: Dict[str, bool]) -> None:
    failed = sorted(k for k, ok in checks.items() if not ok)
    if failed:
        raise classical.Refused(f"{what} failed checks {failed}")


def _load_candidate(expert_dir: pathlib.Path, env_name: str):
    record_path = expert_dir / CANDIDATE_FILE
    record_bytes, record = _read_record(record_path)
    if not isinstance(record, dict) or not isinstance(record.get("config"), dict):
        raise classical.Refused(f"{record_path} is not a candidate record")
    config = record["config"]
    candidate = _as_dict(config.get("candidate"))
    task = tasks.get_task(env_name)
    try:
        rebuilt = build_candidate_config(
            env_name,
            candidate.get("sha256"),
            candidate.get("label"),
            qualification_seed=config.get("qualification_seed"),
            eval_episodes=config.get("eval_episodes"),
            max_eval_env_steps=_as_dict(config.get("qualification")).get(
                "max_env_steps",
            ),
            cached_baselines=candidate.get("cached_baselines_unverified"),
        )
    except (TypeError, ValueError):
        rebuilt = None
    checkpoint = _as_dict(record.get("checkpoint"))
    data = None
    if checkpoint.get("path") == CANDIDATE_CHECKPOINT:
        try:
            data = (expert_dir / CANDIDATE_CHECKPOINT).read_bytes()
        except OSError:
            data = None
    digest = hashlib.sha256(data).hexdigest() if data is not None else None
    evidence, mean = _evidence_checks(record, config, task.episode_cap)
    quality = _quality(env_name, mean)
    checks = {
        "status_complete": record.get("status") == pilot.COMPLETE_STATUS,
        "qualified": record.get("qualified") is True,
        "protocol": record.get("protocol") == CANDIDATE_SCHEMA
        and config.get("schema") == CANDIDATE_SCHEMA,
        "env_name": record.get("env_name") == env_name == config.get("env_name"),
        "config_sha256": record.get("config_sha256") == pilot.config_digest(config),
        # Binds the task digest, cap, cap rule and frozen thresholds to the
        # current source and environment.
        "config_rebuilt": rebuilt is not None and config == rebuilt,
        "package_versions": record.get("package_versions") == pilot.package_versions(),
        "checkpoint_bytes": digest is not None
        and digest
        == checkpoint.get("sha256")
        == record.get("expert_sha256")
        == candidate.get("sha256"),
        **evidence,
        "threshold_passed": quality["per_env_passed"],
    }
    _refuse_failed(f"Candidate {expert_dir}", checks)
    model = PPO.load(io.BytesIO(data), device="cpu")
    state_digest = pilot.policy_state_sha256(model.policy)
    if state_digest != checkpoint.get("policy_state_sha256"):
        raise classical.Refused("Loaded policy parameters do not match the record")
    identity = {
        "expert_source": "candidate_qualification",
        "preparation_dir": str(expert_dir.resolve()),
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "protocol": record["protocol"],
        "config_sha256": record["config_sha256"],
        "expert_sha256": digest,
        "policy_state_sha256": state_digest,
        "qualification_mean_return": mean,
        "training_provenance": TRAINING_PROVENANCE,
        "action_rule": classical.ACTION_RULE,
        "quality": quality,
    }
    return model, identity


def _load_preparation(expert_dir: pathlib.Path, env_name: str):
    """A stage 1 preparation, checked for compatibility with the current task."""
    model, identity = classical.load_verified_expert(expert_dir, env_name)
    record_bytes, record = _read_record(expert_dir / pilot.PREPARATION_FILE)
    if hashlib.sha256(record_bytes).hexdigest() != identity["record_sha256"]:
        raise classical.Refused("Preparation record changed during verification")
    config = record["config"]
    task = tasks.get_task(env_name)
    cap = tasks.describe(task)["episode_cap"]
    # Legacy configs have no architecture claim and rebuild exactly as before;
    # an explicit claim is rebuilt from its own widths, never as [64, 64].
    expert_cfg = _as_dict(config.get("expert"))
    explicit = "architecture" in expert_cfg
    claim = _as_dict(expert_cfg.get("architecture"))
    try:
        rebuilt = run_classical.build_preparation_config(
            env_name,
            seed=config.get("seed"),
            qualification_seed=config.get("qualification_seed"),
            eval_episodes=config.get("eval_episodes"),
            convergence_override=None,
            max_eval_env_steps=_as_dict(config.get("qualification")).get(
                "max_env_steps",
            ),
            actor_net_arch=claim.get("pi", "missing") if explicit else None,
        )
    except (TypeError, ValueError):
        rebuilt = None
    source = _as_dict(record.get("source_fingerprint"))
    files = source.get("files")
    current_sources = {
        rel: pilot.sha256_file(pilot._PACKAGE_ROOT / rel)
        for rel in PREPARATION_ENVIRONMENT_SOURCES
    }
    gymnasium = pilot.package_versions()["gymnasium"]
    evidence, mean = _evidence_checks(record, config, cap)
    quality = _quality(env_name, mean)
    sources_ok, source_lineage = _environment_sources_match(env_name, files, current_sources)
    checks = {
        "convergence_override": "convergence_override" in config
        and config["convergence_override"] is None,
        "frozen_threshold": config.get("threshold")
        == record.get("threshold")
        == pilot.frozen_expert_threshold(env_name),
        "episode_cap": config.get("env_max_episode_steps") == cap == task.episode_cap,
        "config_rebuilt": rebuilt is not None and config == rebuilt,
        "source_fingerprint": isinstance(files, dict)
        and source.get("combined_sha256") == pilot.config_digest(files),
        "environment_sources": sources_ok,
        "gymnasium_version": _as_dict(record.get("package_versions")).get(
            "gymnasium",
        )
        == gymnasium,
        **evidence,
        "threshold_passed": quality["per_env_passed"],
    }
    if explicit:
        actual = expert_training.policy_architecture(model.policy)
        checks["architecture"] = (
            not run_classical.architecture_mismatches(claim, actual)
            and _as_dict(record.get("checkpoint")).get("architecture") == actual
        )
    _refuse_failed(f"Preparation {expert_dir}", checks)
    identity = dict(identity)
    if explicit:
        identity["architecture"] = {
            "claimed": claim,
            "actual": actual,
            "rule": ARCHITECTURE_RULE,
        }
    compatibility = {
        "rule": PREPARATION_COMPATIBILITY_RULE,
        "environment_sources": current_sources,
        "gymnasium": gymnasium,
    }
    if source_lineage:
        compatibility["environment_source_lineage"] = {
            "recorded_and_current": source_lineage,
            "reason": APPROVED_ENVIRONMENT_SOURCE_LINEAGE["reason"],
        }
    identity.update(
        expert_source="preparation",
        qualification_mean_return=mean,
        quality=quality,
        compatibility=compatibility,
    )
    return model, identity


def load_qualified_expert(
    expert_dir: pathlib.Path,
    env_name: str,
) -> Tuple[Any, Dict[str, Any]]:
    """Load a qualified expert from a preparation or a candidate qualification.

    Both kinds fail closed: the record is parsed from the bytes that are
    hashed, its qualification episodes are checked and their mean recomputed,
    and ``quality`` is computed from that mean and the current frozen
    threshold, never from fields of the record. A candidate's config must be
    rebuilt exactly by the current source and its packages must equal the
    current ones. A preparation must satisfy `PREPARATION_COMPATIBILITY_RULE`.

    Returns:
        The PPO model and an identity record with ``expert_source`` and
        ``quality`` (`expert_quality`).

    Raises:
        Refused: If the directory holds neither or both kinds, or any check
            fails. Nothing is modified.
    """
    expert_dir = pathlib.Path(expert_dir)
    has_prep = (expert_dir / pilot.PREPARATION_FILE).exists()
    has_candidate = (expert_dir / CANDIDATE_FILE).exists()
    if has_prep == has_candidate:
        raise classical.Refused(
            f"{expert_dir} must hold exactly one of {pilot.PREPARATION_FILE} "
            f"and {CANDIDATE_FILE}",
        )
    if has_candidate:
        return _load_candidate(expert_dir, env_name)
    return _load_preparation(expert_dir, env_name)
