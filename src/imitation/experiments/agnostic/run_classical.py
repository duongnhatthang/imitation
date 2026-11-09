"""Classical agnostic imitation runner.

Stage 1 ``prepare-expert``: train a PPO expert with the inherited
convergence trainer, then qualify it on fresh deterministic episodes against a
threshold frozen before training. Example::

    python -m imitation.experiments.agnostic.run_classical prepare-expert \\
        --env CartPole-v1 --output-dir PATH --seed 0 \\
        --qualification-seed 1000 --eval-episodes 100 \\
        --deadline 2026-10-07T01:01:00Z

Stage 2 ``audit`` and ``run-cell`` are implemented in `classical`; they use
exit code 0 for complete, 1 for partial or failed, 2 for invalid arguments,
and 3 for a refused preparation or non-fresh output directory. Example::

    python -m imitation.experiments.agnostic.run_classical run-cell \\
        --env CartPole-v1 --preparation-dir PREP --representation severe \\
        --output-dir PATH --seed 0 --deadline 2026-10-07T01:01:00Z

Their ``result.json`` has top-level ``status``, ``protocol`` (alias of
``schema``), ``env_name``, ``seed``, ``expert_sha256``, and for ``run-cell``
``representation``; the nested ``config`` and ``preparation`` keep the same
values. Only ``status == "complete"`` is complete.

Stage 1 exit codes: 0 complete and qualified (fresh or verified reuse), 1 failed,
partial, or not qualified, 2 invalid arguments, 3 refused because the output
directory holds different, incomplete, or unverifiable evidence, or another
invocation owns it.

A successful ``preparation.json`` has top-level ``status == "complete"``,
``qualified`` and ``approved`` both True, ``protocol``, ``env_name``, ``seed``,
``expert_sha256`` (alias of ``checkpoint.sha256``), ``config_sha256``, and
``source_fingerprint``. Any other status (``running``, ``failed``,
``partial``, ``not_qualified``) is never complete.

Only one invocation may own an output directory: it claims the directory by
exclusively creating ``preparation.claim`` before writing anything else, and
the claim is kept as provenance. A second invocation fails closed before
training.

This deliberately does not call ``experts.get_or_train_expert`` or
``env_baselines.load_or_compute_baselines``: both run hidden 1000-episode
evaluations and may silently reuse cached experts.
"""

import argparse
import copy
import datetime
import json
import logging
import os
import pathlib
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import gymnasium as gym
from stable_baselines3 import PPO

from imitation.experiments.agnostic import classical, pilot, quantized, rollouts
from imitation.experiments.ftrl import env_utils, expert_training
from imitation.util import util

logger = logging.getLogger(__name__)

SUPPORTED_ENVS = ("CartPole-v1", "Acrobot-v1", "MountainCar-v0")

EXIT_QUALIFIED = 0
EXIT_NOT_QUALIFIED = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3

TRAINER_NAME = (
    "imitation.experiments.ftrl.expert_training."
    "train_classical_expert_until_converged"
)
CHECKPOINT_DIR = "trainer_cache"
CLAIM_FILE = "preparation.claim"

TRAINING_INTERRUPTIBILITY = (
    "The inherited convergence trainer is not interruptible by the deadline: "
    "the deadline is checked only before training starts. During training the "
    "external supervisor's wall-clock cap is the only bound, and a run killed "
    "there leaves this record with status 'running'."
)
TRAINING_COUNTERS_REASON = (
    "The inherited trainer does not expose PPO environment steps, its "
    "50-episode convergence evaluations, or expert query counts. Configured "
    "step caps are recorded in config.convergence_config."
)
DEADLINE_ENFORCEMENT = {
    "expert_training": "checked before start only",
    "expert_qualification": "checked before start and before every env step",
}


def _training_env_reset_seeds(env_name: str, seed: int) -> List[int]:
    """Env reset seeds the inherited trainer will use for ``seed``.

    The trainer builds a training VecEnv and an evaluation VecEnv from the
    same RNG (each env is reset once at construction with a drawn seed), and
    PPO then reseeds the training VecEnv with ``seed + i``.

    Args:
        env_name: Env ID.
        seed: Training seed.

    Returns:
        Sorted unique seeds.
    """
    n_train = env_utils.ENV_CONFIGS[env_name].get("ppo_n_envs") or 1
    rng = rollouts.make_phase_rng(seed, "expert_training")
    drawn = util.make_seeds(copy.deepcopy(rng), n_train + 1)
    return sorted(set(drawn) | {seed + i for i in range(n_train)})


def build_preparation_config(
    env_name: str,
    *,
    seed: int,
    qualification_seed: int,
    eval_episodes: int,
    convergence_override: Optional[Mapping[str, Any]],
    max_eval_env_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the full config that identifies an expert preparation.

    Args:
        env_name: One of `SUPPORTED_ENVS`.
        seed: PPO training seed.
        qualification_seed: Env reset seed for qualification episodes.
        eval_episodes: Number of fresh qualification episodes.
        convergence_override: Optional partial convergence config.
        max_eval_env_steps: Cap on qualification env steps. Defaults to
            ``eval_episodes * max_episode_steps``.

    Returns:
        JSON-compatible config dict.

    Raises:
        ValueError: On unsupported env, bad counts, or overlapping seeds.
    """
    if env_name not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported env {env_name!r}; use {SUPPORTED_ENVS}")
    if int(eval_episodes) <= 0:
        raise ValueError("eval_episodes must be positive")
    if int(seed) < 0 or int(qualification_seed) < 0:
        raise ValueError("Seeds must be non-negative")
    training_seeds = _training_env_reset_seeds(env_name, int(seed))
    if int(qualification_seed) == int(seed) or qualification_seed in training_seeds:
        raise ValueError(
            f"qualification_seed {qualification_seed} overlaps training seeds "
            f"{training_seeds} for seed {seed}",
        )
    horizon = gym.spec(env_name).max_episode_steps
    if max_eval_env_steps is None:
        max_eval_env_steps = int(eval_episodes) * int(horizon)
    if int(max_eval_env_steps) <= 0:
        raise ValueError("max_eval_env_steps must be positive")

    convergence = pilot.merged_convergence_config(env_name, convergence_override)
    env_cfg = env_utils.ENV_CONFIGS[env_name]
    return {
        "schema": pilot.PREPARATION_SCHEMA,
        "env_name": env_name,
        "env_max_episode_steps": int(horizon),
        "seed": int(seed),
        "qualification_seed": int(qualification_seed),
        "eval_episodes": int(eval_episodes),
        "convergence_override": (
            dict(convergence_override) if convergence_override else None
        ),
        "convergence_config": convergence,
        "threshold": pilot.frozen_expert_threshold(env_name, convergence),
        "expert": {
            "trainer": TRAINER_NAME,
            "algorithm": "PPO",
            "policy": "MlpPolicy",
            "net_arch": [64, 64],
            "ppo_kwargs": dict(env_cfg.get("ppo_kwargs", {})),
            "ppo_n_envs": env_cfg.get("ppo_n_envs") or 1,
            "device": "cpu",
            "training_rng": "SeedSequence(seed, spawn_key=(expert_training,))",
        },
        "qualification": {
            "action_rule": "deterministic predict (argmax)",
            "n_envs": 1,
            "env_reset_seed": int(qualification_seed),
            "construction_rng": (
                "SeedSequence(qualification_seed, " "spawn_key=(expert_qualification,))"
            ),
            "max_env_steps": int(max_eval_env_steps),
        },
    }


def _error(exc: BaseException) -> Dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _inspect_existing(output_dir: pathlib.Path, config: Mapping[str, Any]) -> int:
    """Decide what to do with a non-empty output directory.

    Args:
        output_dir: Preparation directory, known to exist and be non-empty.
        config: Requested config.

    Returns:
        `EXIT_QUALIFIED` for a verified matching reuse, else `EXIT_REFUSED`.
    """
    record_path = output_dir / pilot.PREPARATION_FILE
    if not record_path.is_file():
        logger.error(
            "%s contains files but no %s; refusing to reuse or overwrite them",
            output_dir,
            pilot.PREPARATION_FILE,
        )
        return EXIT_REFUSED
    try:
        record = pilot.read_json(record_path)
    except (OSError, ValueError) as exc:
        logger.error("Unreadable %s: %s", record_path, exc)
        return EXIT_REFUSED
    if not isinstance(record, dict):
        logger.error("%s is not a JSON object", record_path)
        return EXIT_REFUSED
    if record.get("config") != dict(config):
        logger.error("%s was prepared with a different config", output_dir)
        return EXIT_REFUSED
    status = record.get("status")
    if status != pilot.COMPLETE_STATUS:
        logger.error(
            "%s has status %r; it stays as evidence of an incomplete or "
            "unqualified attempt. Use a new output directory.",
            output_dir,
            status,
        )
        return EXIT_REFUSED
    if not pilot.verify_qualified_preparation(output_dir, config):
        logger.error("%s checkpoint or record failed verification", output_dir)
        return EXIT_REFUSED
    logger.info("Reusing verified qualified expert in %s", output_dir)
    return EXIT_QUALIFIED


def _has_existing_entries(output_dir: pathlib.Path) -> bool:
    return output_dir.exists() and any(output_dir.iterdir())


def _claim_output_dir(
    output_dir: pathlib.Path,
    claimed_at: datetime.datetime,
) -> Optional[Dict[str, Any]]:
    """Exclusively claim ``output_dir`` for this invocation.

    The claim file is created with ``O_EXCL`` and never removed, so exactly
    one invocation can ever own a directory; a crash after claiming leaves the
    claim as evidence and the directory is refused thereafter.

    Args:
        output_dir: Preparation directory; created if missing.
        claimed_at: Current aware time.

    Returns:
        The claim record, or None if another invocation already owns the
        directory or it gained other entries before the claim.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    claim_path = output_dir / CLAIM_FILE
    claim = {"pid": os.getpid(), "claimed_at_utc": claimed_at.isoformat()}
    try:
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        logger.error("%s is already claimed by another invocation", output_dir)
        return None
    with os.fdopen(fd, "w") as f:
        json.dump(claim, f, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    others = sorted(p.name for p in output_dir.iterdir() if p.name != CLAIM_FILE)
    if others:
        logger.error("%s gained entries %s before it was claimed", output_dir, others)
        return None
    return dict(claim, path=CLAIM_FILE)


def prepare_expert(
    env_name: str,
    output_dir: pathlib.Path,
    *,
    seed: int,
    qualification_seed: int,
    eval_episodes: int,
    deadline: datetime.datetime,
    convergence_override: Optional[Mapping[str, Any]] = None,
    max_eval_env_steps: Optional[int] = None,
    trainer: Optional[Callable[..., Any]] = None,
    clock: Callable[[], datetime.datetime] = rollouts.utc_now,
) -> int:
    """Train and qualify an expert, recording evidence in ``output_dir``.

    Args:
        env_name: One of `SUPPORTED_ENVS`.
        output_dir: Directory for ``preparation.json`` and the checkpoint.
        seed: PPO training seed.
        qualification_seed: Env reset seed for qualification, disjoint from
            every training env seed.
        eval_episodes: Fresh deterministic qualification episodes.
        deadline: Aware absolute deadline.
        convergence_override: Optional partial convergence config, merged over
            the env's config before being passed to the trainer.
        max_eval_env_steps: Optional cap on qualification env steps.
        trainer: Replacement for the inherited trainer (tests only).
        clock: Returns the current aware time.

    Returns:
        Process exit code; `EXIT_QUALIFIED` only for a complete, qualified
        expert.

    Raises:
        ValueError: On invalid arguments, before anything is written.
    """
    deadline = rollouts.ensure_aware_utc(deadline)
    config = build_preparation_config(
        env_name,
        seed=seed,
        qualification_seed=qualification_seed,
        eval_episodes=eval_episodes,
        convergence_override=convergence_override,
        max_eval_env_steps=max_eval_env_steps,
    )
    output_dir = pathlib.Path(output_dir)
    if _has_existing_entries(output_dir):
        return _inspect_existing(output_dir, config)
    claim = _claim_output_dir(output_dir, clock())
    if claim is None:
        return EXIT_REFUSED
    if trainer is None:
        trainer = expert_training.train_classical_expert_until_converged

    record_path = output_dir / pilot.PREPARATION_FILE
    started = time.monotonic()
    elapsed: Dict[str, float] = {}
    record: Dict[str, Any] = {
        "status": "running",
        "phase": "expert_training",
        "qualified": False,
        "approved": False,
        "protocol": pilot.PREPARATION_SCHEMA,
        "env_name": env_name,
        "seed": config["seed"],
        "expert_sha256": None,
        "claim": claim,
        "config": config,
        "config_sha256": pilot.config_digest(config),
        "threshold": config["threshold"],
        "deadline_utc": deadline.isoformat(),
        "deadline_enforcement": DEADLINE_ENFORCEMENT,
        "training_interruptibility": TRAINING_INTERRUPTIBILITY,
        "started_at_utc": clock().isoformat(),
        "seeds": {
            "training_seed": config["seed"],
            "training_env_reset_seeds": _training_env_reset_seeds(
                env_name,
                config["seed"],
            ),
            "qualification_env_reset_seed": config["qualification_seed"],
        },
        "package_versions": pilot.package_versions(),
        "source_fingerprint": pilot.source_fingerprint(),
        "counters": {
            "expert_training": rollouts.unavailable_counters(
                "expert_training",
                TRAINING_COUNTERS_REASON,
            ),
        },
        "elapsed_seconds": elapsed,
    }

    def finish(status: str, exc: Optional[BaseException] = None) -> int:
        record["status"] = status
        if exc is not None:
            record["error"] = _error(exc)
        complete = status == pilot.COMPLETE_STATUS
        record["qualified"] = complete
        record["approved"] = complete
        record["finished_at_utc"] = clock().isoformat()
        elapsed["total"] = time.monotonic() - started
        pilot.atomic_write_json(record_path, record)
        logger.info("prepare-expert %s: status=%s", env_name, status)
        return EXIT_QUALIFIED if complete else EXIT_NOT_QUALIFIED

    def record_qualification(result: rollouts.EvaluationResult) -> None:
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

    try:
        rollouts.check_deadline(deadline, "expert training", clock)
    except rollouts.DeadlineExceeded as exc:
        return finish("failed", exc)

    pilot.atomic_write_json(record_path, record)
    cache_dir = output_dir / CHECKPOINT_DIR
    t0 = time.monotonic()
    try:
        trainer(
            env_name,
            cache_dir,
            rollouts.make_phase_rng(config["seed"], "expert_training"),
            config["seed"],
            convergence_override=(
                config["convergence_config"] if convergence_override else None
            ),
        )
    except Exception as exc:
        elapsed["expert_training"] = time.monotonic() - t0
        return finish("failed", exc)
    elapsed["expert_training"] = time.monotonic() - t0

    record["phase"] = "expert_qualification"
    try:
        checkpoint = cache_dir / env_name.replace("/", "_") / "model.zip"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Trainer did not write {checkpoint}")
        digest = pilot.sha256_file(checkpoint)
        model = PPO.load(checkpoint, device="cpu")
        record["checkpoint"] = {
            "path": str(checkpoint.relative_to(output_dir)),
            "sha256": digest,
            "policy_state_sha256": pilot.policy_state_sha256(model.policy),
        }
        record["expert_sha256"] = digest
        pilot.atomic_write_json(record_path, record)
    except Exception as exc:
        return finish("failed", exc)

    try:
        rollouts.check_deadline(deadline, "expert qualification", clock)
    except rollouts.DeadlineExceeded as exc:
        return finish("partial", exc)

    t0 = time.monotonic()
    venv = None
    # Factual zero counts until the evaluator reports what it spent.
    record["counters"]["expert_qualification"] = rollouts.PhaseCounters(
        phase="expert_qualification",
    ).to_dict()
    try:
        venv = env_utils.make_env(
            env_name,
            n_envs=1,
            rng=rollouts.make_phase_rng(
                config["qualification_seed"],
                "expert_qualification",
            ),
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
        elapsed["expert_qualification"] = time.monotonic() - t0
        record_qualification(exc.result)
        return finish("failed", exc.__cause__ or exc)
    except Exception as exc:
        elapsed["expert_qualification"] = time.monotonic() - t0
        return finish("failed", exc)
    finally:
        if venv is not None:
            venv.close()
    elapsed["expert_qualification"] = time.monotonic() - t0

    record_qualification(result)
    if not result.completed:
        reason = result.counters.stop_reason
        label = (
            rollouts.DeadlineExceeded
            if reason == "deadline"
            else rollouts.StepBudgetExhausted
        )
        return finish("partial", label(f"Qualification stopped early: {reason}"))
    passed = record["qualification"]["mean_return"] >= (
        config["threshold"]["raw_threshold"]
    )
    return finish(pilot.COMPLETE_STATUS if passed else "not_qualified")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-expert", help="Train and qualify an expert.")
    prep.add_argument("--env", required=True, choices=SUPPORTED_ENVS)
    prep.add_argument("--output-dir", required=True, type=pathlib.Path)
    prep.add_argument("--seed", required=True, type=int)
    prep.add_argument("--qualification-seed", required=True, type=int)
    prep.add_argument("--eval-episodes", type=int, default=100)
    prep.add_argument(
        "--deadline",
        required=True,
        type=rollouts.parse_utc_deadline,
        help="Absolute timezone-aware deadline, e.g. 2026-10-07T01:01:00Z.",
    )
    prep.add_argument(
        "--convergence-override",
        type=json.loads,
        default=None,
        help="JSON object merged over the env's convergence config.",
    )
    prep.add_argument("--max-eval-env-steps", type=int, default=None)

    aud = sub.add_parser("audit", help="Stage 2 alias audit (diagnostic labels).")
    cell = sub.add_parser("run-cell", help="Stage 2 FTL / BC-iid / fixed BC cell.")
    for stage2 in (aud, cell):
        stage2.add_argument("--env", required=True, choices=SUPPORTED_ENVS)
        stage2.add_argument("--preparation-dir", required=True, type=pathlib.Path)
        stage2.add_argument("--output-dir", required=True, type=pathlib.Path)
        stage2.add_argument("--seed", required=True, type=int)
        stage2.add_argument(
            "--deadline",
            required=True,
            type=rollouts.parse_utc_deadline,
            help="Aware deadline; clamped to classical.HARD_CAP.",
        )
    aud.add_argument("--episodes", type=int, default=classical.DEFAULT_AUDIT_EPISODES)
    aud.add_argument("--delta", type=float, default=classical.DEFAULT_DELTA)
    cell.add_argument(
        "--representation",
        required=True,
        choices=quantized.REPRESENTATIONS,
    )
    cell.add_argument("--budget", type=int, default=classical.DEFAULT_BUDGET)
    cell.add_argument("--batch", type=int, default=classical.DEFAULT_BATCH)
    cell.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=list(classical.DEFAULT_CHECKPOINTS),
    )
    cell.add_argument(
        "--eval-episodes",
        type=int,
        default=classical.DEFAULT_EVAL_EPISODES,
    )
    return parser.parse_args(argv)


def _stage2_main(args: argparse.Namespace) -> int:
    """Run ``audit`` or ``run-cell`` and print a summary of the fresh record."""
    try:
        if args.command == "audit":
            code = classical.audit(
                args.env,
                args.preparation_dir,
                args.output_dir,
                seed=args.seed,
                deadline=args.deadline,
                episodes=args.episodes,
                delta=args.delta,
            )
        else:
            code = classical.run_cell(
                args.env,
                args.preparation_dir,
                args.representation,
                args.output_dir,
                seed=args.seed,
                deadline=args.deadline,
                budget=args.budget,
                batch=args.batch,
                checkpoints=args.checkpoints,
                eval_episodes=args.eval_episodes,
            )
    except ValueError as exc:
        logger.error("Invalid arguments: %s", exc)
        return EXIT_USAGE
    # A refused directory belongs to someone else; never report its status.
    if code != classical.EXIT_REFUSED:
        record_path = args.output_dir / classical.RESULT_FILE
        record = pilot.read_json(record_path)
        print(
            json.dumps(
                {
                    "status": record.get("status"),
                    "record": str(record_path),
                    "exit_code": code,
                },
            ),
        )
    return code


def main(
    argv: Optional[Sequence[str]] = None,
    trainer: Optional[Callable[..., Any]] = None,
) -> int:
    """CLI entry point.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv``.
        trainer: Replacement for the inherited trainer (tests only).

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    if args.command != "prepare-expert":
        return _stage2_main(args)
    try:
        code = prepare_expert(
            args.env,
            args.output_dir,
            seed=args.seed,
            qualification_seed=args.qualification_seed,
            eval_episodes=args.eval_episodes,
            deadline=args.deadline,
            convergence_override=args.convergence_override,
            max_eval_env_steps=args.max_eval_env_steps,
            trainer=trainer,
        )
    except ValueError as exc:
        logger.error("Invalid arguments: %s", exc)
        return EXIT_USAGE
    record_path = args.output_dir / pilot.PREPARATION_FILE
    try:
        record = pilot.read_json(record_path)
    except (OSError, ValueError):
        return code
    if isinstance(record, dict):
        print(
            json.dumps(
                {
                    "status": record.get("status"),
                    "qualified": record.get("qualified"),
                    "expert_sha256": record.get("expert_sha256"),
                    "record": str(record_path),
                    "exit_code": code,
                },
            ),
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
