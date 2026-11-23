"""Environment-generic restriction runner for the eight classical tasks.

Protocol `PROTOCOL` runs the CartPole pilot procedure of `restriction_pilot`
on any `tasks.Task`: the same shared data job (fixed-BC chronological pool,
BC-iid stream, 500 expert and 500 random baseline episodes), the same fixed
BC, BC-iid and FTL runs through ``run_experiment`` and the same accounting and
deadlines. Nothing is copied or monkeypatched: every job passes an explicit
`restriction_pilot.RunnerSpec` (`runner_spec`), so the CartPole pilot
protocol ``/2`` and its records are untouched. Differences, all explicit in
records: every data and run record binds the verified task description and
its digest; the expert may come from a preparation or a candidate
qualification (`task_experts`), and a relaxed-only expert is refused unless
explicitly allowed; a run may normalize with a pinned reference data job; a
non-identity rung leaving one distinct pool observation is refused.

Commands (each claims one fresh output directory, except ``describe``):

``describe``
    Print the verified task description, its digest and every ladder
    restriction with its digest. Builds the environment factory only.
``qualify-candidate``
    `task_experts.qualify_candidate` on an existing checkpoint.
``audit``
    `task_audit.audit` of one ladder restriction (reset seed 301 by default).
``data``
    The shared data job for one (task, seed).
``run``
    One (method, restriction) run on one (task, seed). Method ``bc_prefix``
    opts in to `PREFIX_PROTOCOL`: the original BC-prefix baseline, cold
    refits on growing chronological pool prefixes (`rp.BC_PREFIX_RULE`).
    ``--bc-epochs N`` replaces the `rp.BC_EPOCHS` cap of every fit (recorded
    as the run config's ``experiment.bc_n_epochs``); omitted, the cap and
    the config are unchanged. No other fit or evaluation budget moves.

Opt-in flags of ``audit`` and ``run``. ``--paired-cold-fits`` selects
`PAIRED_PROTOCOL`, one spec for all four methods (fixed BC, BC-iid, FTL and
BC-prefix) whose fits and evaluations are paired by (seed, label count)
(`rp.PAIRED_COLD_FITS_RULE`). ``--calibration-candidates`` adds the task's
`tasks.CALIBRATION_CANDIDATES` to the restrictions a job accepts, binding
their descriptions into its records; only tasks that declare candidates
(currently MountainCar, Blackjack, CliffWalking, Taxi and FrozenLakeSlippery)
accept it. The random state binnings (`tasks.RANDOM_ALIAS_CANDIDATES`) are
among them, so a random binning id runs and audits through this same flag,
alone or with ``--paired-cold-fits`` or method ``bc_prefix``. Variant tasks
(`tasks.VARIANT_TASKS`) need no flag for their ladder, which is already the
new mask set. Data jobs are shared: both specs keep
the data protocol `PROTOCOL`, its source file set and its checks, so a
same-source data job (or a pinned one) serves legacy and paired runs alike.

Seed 301 is reserved for audits, as in the pilot; no other seed is special.

Learner families: the default ``linear-head`` is the protocol above, with
every default unchanged. ``--learner-family capacity-one-hidden`` selects
protocol `CAPACITY_PROTOCOL` (`capacity_runner_spec`): an explicit-architecture
one-hidden width-64 teacher preparation, full observations (``--restriction
identity`` only), one shared data job per (task, seed) for every width, and
runs whose learner is a fresh one-hidden student of ``--student-width`` (one
of `capacity_policy.STUDENT_WIDTHS`, strictly below the teacher) on the
teacher's frozen encoder. Fixed BC, BC-iid and FTL keep the runner's data,
cold-refit, 20-epoch and beta-0 mechanics. ``--n-rounds`` (default 1000, the
benchmark budget) and ``--eval-interval`` (default `CAPACITY_EVAL_INTERVAL`)
exist for this family only; a reduced budget is recorded as such.
"""

import argparse
import dataclasses
import datetime
import json
import logging
import pathlib
import sys
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from imitation.experiments.agnostic import (
    capacity_policy,
    classical,
    pilot,
    restriction,
)
from imitation.experiments.agnostic import restriction_pilot as rp
from imitation.experiments.agnostic import rollouts, task_audit, task_experts, tasks

logger = logging.getLogger(__name__)

PROTOCOL = "agnostic-classical-restriction/1"
# Opt-in marker of BC-prefix runs: their records carry it as protocol and
# schema, so readers of the three-method protocol never ingest them. Their
# data and reference jobs keep `PROTOCOL` and its unchanged checks.
PREFIX_PROTOCOL = PROTOCOL + "+bc-prefix/1"
# Opt-in repaired protocol: paired cold fits and paired evaluation seeds for
# all four methods through one spec. Data and reference jobs keep `PROTOCOL`.
PAIRED_PROTOCOL = "agnostic-classical-restriction/2"
RUN_METHODS = rp.METHODS + (rp.PREFIX_METHOD,)
AUDIT_SEED = rp.AUDIT_SEED
SOURCE_FILES = rp._SOURCE_FILES + (
    "experiments/agnostic/tasks.py",
    "experiments/agnostic/task_experts.py",
    "experiments/agnostic/task_audit.py",
    "experiments/agnostic/task_pilot.py",
)

LEGACY_FAMILY = "linear-head"
LEARNER_FAMILIES = (LEGACY_FAMILY, capacity_policy.LEARNER_FAMILY)
CAPACITY_PROTOCOL = "agnostic-capacity-one-hidden/1"
# Everything that shapes a capacity teacher, student, run or its dispatch.
CAPACITY_SOURCE_FILES = SOURCE_FILES + (
    "experiments/agnostic/capacity_policy.py",
    "experiments/agnostic/run_classical.py",
    "experiments/ftrl/expert_training.py",
    "experiments/agnostic/campaign.py",
    "experiments/agnostic/resource_budget.py",
)
# Capacity evaluations are dense initially, then periodic and at the cap.
CAPACITY_EVAL_INTERVAL = 20
FOLLOWUP_EVAL_INTERVAL = 20
FOLLOWUP_METHODS = ("ftl", "bc", rp.PREFIX_METHOD)


def _followup_spec(spec: rp.RunnerSpec) -> rp.RunnerSpec:
    """Use the same acquisition and fit protocol in both study modes."""
    return dataclasses.replace(
        spec,
        protocol=spec.protocol + "+statewise/1",
        data_protocol=spec.data_protocol + "+statewise/1",
        methods=FOLLOWUP_METHODS,
        paired_cold_fits=True,
        statewise_followup=True,
    )


def calibration_candidates_of(env_name: str) -> Tuple[str, ...]:
    """Registered `tasks.CALIBRATION_CANDIDATES` of a task outside its ladder.

    A task with none (for example FrozenLakeVariedStart, whose ladder is
    already new) refuses the opt-in rather than silently running its ladder
    only. FrozenLakeSlippery declares its random state binnings.

    Raises:
        ValueError: The task declares none, or one is not a registered
            restriction.
    """
    task = tasks.get_task(env_name)
    declared = getattr(tasks, "CALIBRATION_CANDIDATES", {}).get(task.env_name, ())
    ids: list = []
    for entry in declared:
        rid = entry if isinstance(entry, str) else entry.restriction_id
        if rid not in task.restriction_ids and rid not in ids:
            ids.append(rid)
    if not ids:
        raise ValueError(f"{env_name} declares no calibration candidates")
    for rid in ids:
        restriction.describe_restriction(rid)  # raises for an unknown id
    return tuple(ids)


def runner_spec(
    env_name: str,
    *,
    bc_prefix: bool = False,
    paired_cold_fits: bool = False,
    calibration_candidates: bool = False,
) -> rp.RunnerSpec:
    """The generic runner spec of a registered task, verified against its factory.

    ``bc_prefix`` selects the opt-in spec that runs only `rp.PREFIX_METHOD`
    under `PREFIX_PROTOCOL`; ``paired_cold_fits`` the spec of
    `PAIRED_PROTOCOL`, which runs all four methods. ``calibration_candidates``
    adds the task's candidates (`calibration_candidates_of`) to the accepted
    restrictions. Everything else, data checks included, is the spec of
    `PROTOCOL`.

    Raises:
        ValueError: For an unknown task, both ``bc_prefix`` and
            ``paired_cold_fits``, candidates requested but not declared, or
            `tasks.TaskMismatch` if the factory disagrees with the declaration.
    """
    if bc_prefix and paired_cold_fits:
        raise ValueError("the paired protocol already runs bc_prefix")
    task = tasks.get_task(env_name)
    tasks.describe(task)
    candidates = calibration_candidates_of(env_name) if calibration_candidates else ()
    if paired_cold_fits:
        protocol, methods = PAIRED_PROTOCOL, rp.METHODS + (rp.PREFIX_METHOD,)
    elif bc_prefix:
        protocol, methods = PREFIX_PROTOCOL, (rp.PREFIX_METHOD,)
    else:
        protocol, methods = PROTOCOL, rp.METHODS
    return rp.RunnerSpec(
        env_name=task.env_name,
        episode_cap=task.episode_cap,
        protocol=protocol,
        data_protocol=PROTOCOL,
        run_restrictions=task.restriction_ids + candidates,
        source_files=SOURCE_FILES,
        audit_seed=AUDIT_SEED,
        max_job_limit_seconds=rp.MAX_JOB_LIMIT_SECONDS,
        task=task,
        methods=methods,
        paired_cold_fits=paired_cold_fits,
        calibration_candidates=candidates,
    )


def capacity_runner_spec(
    env_name: str,
    student_width: Optional[int] = None,
    teacher_width: int = capacity_policy.TEACHER_WIDTH,
) -> rp.RunnerSpec:
    """Capacity-family spec of a registered task; full observations only.

    ``student_width`` is None for the shared data job.

    Raises:
        ValueError: For an unknown task or undeclared widths, or
            `tasks.TaskMismatch` if the factory disagrees with the declaration.
    """
    learner = capacity_policy.CapacityRunner(
        teacher_width=teacher_width,
        student_width=student_width,
    )
    base = runner_spec(env_name)
    return rp.RunnerSpec(
        env_name=base.env_name,
        episode_cap=base.episode_cap,
        protocol=CAPACITY_PROTOCOL,
        data_protocol=CAPACITY_PROTOCOL,
        run_restrictions=("identity",),
        source_files=CAPACITY_SOURCE_FILES,
        audit_seed=AUDIT_SEED,
        max_job_limit_seconds=rp.MAX_JOB_LIMIT_SECONDS,
        task=base.task,
        learner=learner,
    )


def _family_spec(
    env_name: str,
    learner_family: str,
    student_width: Optional[int],
    teacher_width: Optional[int],
) -> rp.RunnerSpec:
    if learner_family == LEGACY_FAMILY:
        if student_width is not None or teacher_width is not None:
            raise classical.Refused(
                "student_width and teacher_width need the capacity family",
            )
        return runner_spec(env_name)
    if learner_family != capacity_policy.LEARNER_FAMILY:
        raise classical.Refused(f"learner_family must be one of {LEARNER_FAMILIES}")
    if teacher_width is None:
        teacher_width = capacity_policy.TEACHER_WIDTH
    try:
        return capacity_runner_spec(env_name, student_width, teacher_width)
    except ValueError as exc:
        raise classical.Refused(str(exc)) from None


def describe(env_name: str) -> Dict[str, Any]:
    """Task description and every ladder restriction, with digests."""
    task = tasks.get_task(env_name)
    ladder = []
    for rid in task.ladder:
        record = restriction.describe_restriction(rid)
        ladder.append({"restriction": record, "sha256": pilot.config_digest(record)})
    return {
        "protocol": PROTOCOL,
        "task": tasks.describe(task),
        "task_sha256": tasks.task_sha256(task),
        "ladder_rule": task.ladder_rule,
        "ladder": ladder,
        "audit_evidence_type": task_audit.evidence_type(env_name),
    }


def _constructed_grids(restriction_id: str) -> Tuple[Any, ...]:
    """`task_audit.constructed_grids`, empty for a restriction with none.

    Ladder and calibration-candidate grids both resolve; a restriction without
    constructed grids records ``[]`` as before.
    """
    try:
        return task_audit.constructed_grids(restriction_id)
    except KeyError:
        return ()


def run_audit(
    output_dir: pathlib.Path,
    expert_dir: pathlib.Path,
    *,
    env_name: str,
    restriction_id: str,
    deadline: datetime.datetime,
    job_limit_seconds: float,
    reset_seed: int = AUDIT_SEED,
    clock: Callable[[], datetime.datetime] = rollouts.utc_now,
    paired_cold_fits: bool = False,
    calibration_candidates: bool = False,
) -> int:
    """Run the witness audit of one restriction into a fresh directory.

    The restriction must be one of the spec's run restrictions other than
    ``identity``: a ladder rung, or a calibration candidate when opted in.
    """
    try:
        spec = runner_spec(
            env_name,
            paired_cold_fits=paired_cold_fits,
            calibration_candidates=calibration_candidates,
        )
        rp._validate_limits(None, job_limit_seconds, spec)
        audited = tuple(r for r in spec.run_restrictions if r != "identity")
        if restriction_id not in audited:
            raise classical.Refused(
                f"restriction must be one of {audited}: {restriction_id!r}",
            )
        rp._check_cap(spec=spec)
        expert, identity = rp._load_expert(expert_dir, spec)
        guard, limits = rp._deadlines(deadline, job_limit_seconds, clock)
        live: Dict[str, Any] = {"audit": None}
        config = {
            "restriction_id": restriction_id,
            "restriction": restriction.describe_restriction(restriction_id),
            "reset_seed": int(reset_seed),
            "evidence_type": task_audit.evidence_type(env_name, restriction_id),
            "constructed_grids": [list(g) for g in _constructed_grids(restriction_id)],
        }
        if spec.calibration_candidates:
            config["calibration_candidates"] = {
                rid: pilot.config_digest(restriction.describe_restriction(rid))
                for rid in spec.calibration_candidates
            }
        state, writer = rp._start(
            output_dir,
            spec.audit_schema,
            identity,
            limits,
            guard,
            lambda: {
                "config": config,
                "config_sha256": pilot.config_digest(config),
                "audit_status": (live["audit"] or {}).get("status"),
                # The audit reports its costs only when it finishes.
                "expert_predict_calls": (live["audit"] or {}).get(
                    "expert_predict_calls",
                ),
                "audit": live["audit"],
            },
            spec,
        )
    except (classical.Refused, ValueError) as exc:
        return rp._refused(classical.Refused(str(exc)))

    def body():
        state["in_flight"] = "witness audit"
        live["audit"] = task_audit.audit(
            expert,
            env_name,
            restriction_id,
            reset_seed,
            check=guard.check,
        )

    return classical._run_lifecycle(state, writer, body, guard)


def prepare_data(
    output_dir,
    expert_dir,
    *,
    env_name: str,
    learner_family: str = LEGACY_FAMILY,
    teacher_width: Optional[int] = None,
    statewise_followup: bool = False,
    **kwargs,
) -> int:
    """`restriction_pilot.prepare_data` under the spec of ``env_name``.

    The capacity family's data job is shared by every student width.
    """
    try:
        spec = _family_spec(env_name, learner_family, None, teacher_width)
    except tasks.TaskMismatch as exc:
        return rp._refused(classical.Refused(str(exc)))
    except classical.Refused as exc:
        return rp._refused(exc)
    if statewise_followup:
        spec = _followup_spec(spec)
    return rp.prepare_data(output_dir, expert_dir, spec=spec, **kwargs)


def run_job(
    output_dir,
    expert_dir,
    data_dir,
    *,
    env_name: str,
    paired_cold_fits: bool = False,
    calibration_candidates: bool = False,
    learner_family: str = LEGACY_FAMILY,
    student_width: Optional[int] = None,
    teacher_width: Optional[int] = None,
    statewise_followup: bool = False,
    **kwargs,
) -> int:
    """`restriction_pilot.run_job` under the generic spec of ``env_name``.

    Without ``paired_cold_fits``, method `rp.PREFIX_METHOD` opts in to the
    BC-prefix spec and the three original methods keep the spec and records
    of `PROTOCOL`. With it, every method runs under `PAIRED_PROTOCOL`.
    ``kwargs`` may hold ``bc_n_epochs`` (`rp.run_job`).
    """
    try:
        if learner_family == LEGACY_FAMILY:
            if student_width is not None or teacher_width is not None:
                raise classical.Refused("student_width and teacher_width need the capacity family")
            spec = runner_spec(
                env_name,
                bc_prefix=(
                    not paired_cold_fits
                    and not statewise_followup
                    and kwargs.get("method") == rp.PREFIX_METHOD
                ),
                paired_cold_fits=paired_cold_fits,
                calibration_candidates=calibration_candidates,
            )
        else:
            if paired_cold_fits or calibration_candidates:
                raise ValueError("capacity runs use their own learner protocol")
            spec = _family_spec(env_name, learner_family, student_width, teacher_width)
    except ValueError as exc:
        return rp._refused(classical.Refused(str(exc)))
    except classical.Refused as exc:
        return rp._refused(exc)
    if statewise_followup:
        if paired_cold_fits or kwargs.get("bc_n_epochs") is not None:
            return rp._refused(
                classical.Refused("statewise follow-up fixes the cold-fit protocol"),
            )
        spec = _followup_spec(spec)
        kwargs.setdefault("eval_interval", FOLLOWUP_EVAL_INTERVAL)
    if spec.learner is not None:
        if student_width is None:
            return rp._refused(
                classical.Refused("a capacity run needs a student_width"),
            )
        kwargs.setdefault("eval_interval", CAPACITY_EVAL_INTERVAL)
    return rp.run_job(output_dir, expert_dir, data_dir, spec=spec, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer: {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m imitation.experiments.agnostic.task_pilot",
        description="Environment-generic classical restriction runner.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    # Variant tasks run like the originals, on their own (already new) ladder.
    envs = sorted(tasks.TASKS) + sorted(getattr(tasks, "VARIANT_TASKS", {}))

    desc = sub.add_parser("describe", help="print the verified task and ladder")
    desc.add_argument("--env", choices=envs, required=True)

    qual = sub.add_parser("qualify-candidate", help="qualify an existing checkpoint")
    qual.add_argument("--env", choices=envs, required=True)
    qual.add_argument("--candidate", type=pathlib.Path, required=True)
    qual.add_argument("--output-dir", type=pathlib.Path, required=True)
    qual.add_argument("--qualification-seed", type=int, required=True)
    qual.add_argument("--eval-episodes", type=int, default=100)
    qual.add_argument("--max-eval-env-steps", type=int, default=None)
    qual.add_argument("--deadline", type=rollouts.parse_utc_deadline, required=True)

    def common(p):
        p.add_argument("--env", choices=envs, required=True)
        p.add_argument("--expert-dir", type=pathlib.Path, required=True)
        p.add_argument("--output-dir", type=pathlib.Path, required=True)
        p.add_argument("--deadline", type=rollouts.parse_utc_deadline, required=True)
        p.add_argument("--job-limit-seconds", type=float, required=True)

    def opt_in(p):
        p.add_argument(
            "--paired-cold-fits",
            action="store_true",
            help=f"use the paired protocol {PAIRED_PROTOCOL}",
        )
        p.add_argument(
            "--calibration-candidates",
            action="store_true",
            help="also accept the task's declared calibration candidate masks",
        )

    aud = sub.add_parser("audit", help="witness audit of one restriction")
    common(aud)
    aud.add_argument("--restriction", required=True)
    aud.add_argument("--reset-seed", type=int, default=AUDIT_SEED)
    opt_in(aud)

    def family(p):
        p.add_argument(
            "--learner-family",
            choices=LEARNER_FAMILIES,
            default=LEGACY_FAMILY,
        )
        p.add_argument(
            "--teacher-width",
            type=int,
            choices=capacity_policy.TEACHER_WIDTHS,
            default=None,
            help="capacity family only: the qualified teacher's hidden width",
        )

    data = sub.add_parser("data", help="shared expert data and baselines")
    common(data)
    data.add_argument("--seed", type=int, required=True)
    data.add_argument("--allow-relaxed-expert", action="store_true")
    data.add_argument("--budget", type=_positive_int, default=rp.DEFAULT_BUDGET)
    data.add_argument("--statewise-followup", action="store_true")
    family(data)
    run = sub.add_parser("run", help="one (method, restriction) run")
    common(run)
    run.add_argument("--data-dir", type=pathlib.Path, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--method", choices=RUN_METHODS, required=True)
    run.add_argument("--restriction", required=True)
    run.add_argument("--expected-data-sha256", default=None)
    run.add_argument("--reference-data-dir", type=pathlib.Path, default=None)
    run.add_argument("--expected-reference-sha256", default=None)
    run.add_argument("--allow-relaxed-expert", action="store_true")
    run.add_argument(
        "--bc-epochs",
        type=_positive_int,
        default=None,
        help=f"epoch cap of a fixed-epoch BC fit (default {rp.BC_EPOCHS})",
    )
    opt_in(run)
    run.add_argument("--statewise-followup", action="store_true")
    family(run)
    run.add_argument(
        "--student-width",
        type=int,
        choices=capacity_policy.STUDENT_WIDTHS,
        default=None,
        help="capacity family only: student hidden width, below the teacher's",
    )
    run.add_argument(
        "--n-rounds",
        type=_positive_int,
        default=None,
        help=f"label budget (default {rp.DEFAULT_BUDGET})",
    )
    run.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="deterministic episodes per evaluation (capacity and statewise: 10, legacy: 100)",
    )
    run.add_argument(
        "--final-eval-episodes",
        type=int,
        default=None,
        help="episodes of the final-budget evaluation only (default --eval-episodes)",
    )
    run.add_argument(
        "--eval-interval",
        type=_positive_int,
        default=None,
        help="rounds between evaluations",
    )
    run.add_argument(
        "--eval-initial-rounds", type=int, default=None,
        help="evaluate every round through this cutoff (capacity and statewise default 60)",
    )
    run.add_argument("--eval-only-checkpoints", dest="save_every_round", action="store_false")
    run.add_argument("--full-checkpoints", dest="compact_checkpoints", action="store_false")
    return parser


def _family_arguments(parser, args) -> Dict[str, Any]:
    """Keyword arguments of the selected learner family; legacy adds none."""
    names = ("teacher_width", "student_width")
    if args.learner_family == LEGACY_FAMILY and not getattr(
        args, "statewise_followup", False,
    ):
        names += ("n_rounds", "eval_interval")
    given = {n: getattr(args, n, None) for n in names}
    given = {n: v for n, v in given.items() if v is not None}
    if args.learner_family == LEGACY_FAMILY:
        if given:
            parser.error(
                f"{sorted(given)} need --learner-family "
                f"{capacity_policy.LEARNER_FAMILY}",
            )
        return {}
    if args.command == "run" and "student_width" not in given:
        parser.error("the capacity family needs --student-width")
    return {"learner_family": args.learner_family, **given}


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "describe":
        print(json.dumps(describe(args.env), indent=2, sort_keys=True))
        return 0
    if args.command == "qualify-candidate":
        try:
            code = task_experts.qualify_candidate(
                args.env,
                args.candidate,
                args.output_dir,
                qualification_seed=args.qualification_seed,
                eval_episodes=args.eval_episodes,
                max_eval_env_steps=args.max_eval_env_steps,
                deadline=args.deadline,
            )
        except ValueError as exc:
            parser.error(str(exc))
        record_file = task_experts.CANDIDATE_FILE
    else:
        if getattr(args, "seed", None) == AUDIT_SEED:
            parser.error(f"seed {AUDIT_SEED} is reserved for the mask audit")
        common = dict(deadline=args.deadline, job_limit_seconds=args.job_limit_seconds)
        if args.command in ("data", "run"):
            common.update(_family_arguments(parser, args))
        if args.command == "audit":
            code = run_audit(
                args.output_dir,
                args.expert_dir,
                env_name=args.env,
                restriction_id=args.restriction,
                reset_seed=args.reset_seed,
                paired_cold_fits=args.paired_cold_fits,
                calibration_candidates=args.calibration_candidates,
                **common,
            )
        elif args.command == "data":
            code = prepare_data(
                args.output_dir,
                args.expert_dir,
                env_name=args.env,
                seed=args.seed,
                budget=args.budget,
                statewise_followup=args.statewise_followup,
                allow_relaxed_expert=args.allow_relaxed_expert,
                **common,
            )
        else:
            code = run_job(
                args.output_dir,
                args.expert_dir,
                args.data_dir,
                env_name=args.env,
                method=args.method,
                restriction_id=args.restriction,
                seed=args.seed,
                expected_data_sha256=args.expected_data_sha256,
                reference_data_dir=args.reference_data_dir,
                expected_reference_sha256=args.expected_reference_sha256,
                allow_relaxed_expert=args.allow_relaxed_expert,
                paired_cold_fits=args.paired_cold_fits,
                calibration_candidates=args.calibration_candidates,
                bc_n_epochs=args.bc_epochs,
                eval_episodes=args.eval_episodes,
                final_eval_episodes=args.final_eval_episodes,
                eval_initial_rounds=args.eval_initial_rounds,
                save_every_round=args.save_every_round,
                compact_checkpoints=args.compact_checkpoints,
                statewise_followup=args.statewise_followup,
                **{
                    name: getattr(args, name)
                    for name in ("n_rounds", "eval_interval")
                    if getattr(args, name) is not None
                },
                **common,
            )
        record_file = rp.RESULT_FILE
    status = "refused"
    if code != classical.EXIT_REFUSED:
        status = pilot.read_json(args.output_dir / record_file).get("status")
    print(json.dumps({"status": status, "exit_code": code}))
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
