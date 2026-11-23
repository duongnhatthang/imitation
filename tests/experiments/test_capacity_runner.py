"""Capacity family through task_pilot, RunnerSpec and the restriction runner.

A one-hidden width-64 CartPole teacher is prepared with an exact hand-built
policy (no training). One shared data job serves every width; fixed BC,
BC-iid and FTL then run for students 32 and 4 at a reduced budget with an
explicit clock before the legacy hard cutoff.
"""

import json
import pathlib

import numpy as np
import pytest
import torch as th

from imitation.experiments.agnostic import capacity_policy as cp
from imitation.experiments.agnostic import classical, pilot, restriction
from imitation.experiments.agnostic import restriction_pilot as rp
from imitation.experiments.agnostic import task_experts, task_pilot

from .test_capacity_teacher import FAR, Clock, injected_trainer, legacy_trainer, prepare

LIMIT = 3600
BUDGET = 6
SEED = 4100
WIDTHS = (32, 4)
METHODS = ("bc", "bc_iid", "ftl")
LEGACY_SOURCE_FILES = rp._SOURCE_FILES + (
    "experiments/agnostic/tasks.py",
    "experiments/agnostic/task_experts.py",
    "experiments/agnostic/task_audit.py",
    "experiments/agnostic/task_pilot.py",
)


def _read(out):
    return json.loads((pathlib.Path(out) / rp.RESULT_FILE).read_text())


@pytest.fixture(scope="module")
def teachers(tmp_path_factory):
    root = tmp_path_factory.mktemp("teachers")
    preps = {
        "one_hidden": (injected_trainer((64,)), (64,)),
        "two_hidden_claimed": (legacy_trainer((64, 64)), None),
        "one_hidden_legacy_claim": (legacy_trainer((64,)), None),
    }
    for name, (trainer, arch) in preps.items():
        assert prepare(root / name, trainer, arch) == 0
    return {name: root / name for name in preps}


def _data(out, teacher, **kwargs):
    params = dict(
        env_name="CartPole-v1",
        seed=SEED,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        budget=BUDGET,
        baseline_episodes=2,
        clock=Clock(),
    )
    params.update(kwargs)
    return task_pilot.prepare_data(out, teacher, **params)


@pytest.fixture(scope="module")
def capacity_data(teachers, tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "capacity"
    code = _data(out, teachers["one_hidden"], learner_family=cp.LEARNER_FAMILY)
    assert code == classical.EXIT_COMPLETE
    return out


def _run(out, teacher, data, method, width, **kwargs):
    params = dict(
        env_name="CartPole-v1",
        method=method,
        restriction_id="identity",
        seed=SEED,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        n_rounds=BUDGET,
        eval_interval=3,
        clock=Clock(),
        learner_family=cp.LEARNER_FAMILY,
        student_width=width,
    )
    params.update(kwargs)
    return task_pilot.run_job(out, teacher, data, **params)


@pytest.fixture(scope="module")
def capacity_runs(teachers, capacity_data, tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    runs = {}
    for width in WIDTHS:
        for method in METHODS:
            out = root / f"{method}-w{width}"
            code = _run(out, teachers["one_hidden"], capacity_data, method, width)
            assert code == classical.EXIT_COMPLETE, _read(out).get("error")
            runs[method, width] = out
    return runs


# ---------------------------------------------------------------------------
# Data job
# ---------------------------------------------------------------------------


def test_capacity_data_job_has_its_own_protocol_and_verified_teacher(
    teachers,
    capacity_data,
):
    record = _read(capacity_data)
    assert record["status"] == "complete"
    assert record["protocol"] == task_pilot.CAPACITY_PROTOCOL
    assert record["schema"] == task_pilot.CAPACITY_PROTOCOL + "/data"
    assert sorted(record["source"]["files"]) == sorted(task_pilot.CAPACITY_SOURCE_FILES)
    for rel in (
        "experiments/agnostic/capacity_policy.py",
        "experiments/agnostic/run_classical.py",
        "experiments/ftrl/expert_training.py",
        "experiments/agnostic/campaign.py",
        "experiments/agnostic/resource_budget.py",
        "algorithms/bc.py",
        "algorithms/ftrl.py",
        "experiments/ftrl/policy_utils.py",
    ):
        assert record["source"]["files"][rel] != "missing", rel
    teacher = record["config"]["capacity_teacher"]
    assert teacher["actor_hidden_widths"] == [64]
    assert (
        teacher["policy_state_sha256"] == record["preparation"]["policy_state_sha256"]
    )
    assert record["preparation"]["architecture"]["claimed"]["pi"] == [64]
    assert record["config"]["allow_relaxed_expert"] is False


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def test_every_width_and_method_runs_on_shared_data(capacity_data, capacity_runs):
    data = _read(capacity_data)
    data_sha = pilot.sha256_file(capacity_data / rp.RESULT_FILE)
    trained = {}
    for (method, width), out in capacity_runs.items():
        record = _read(out)
        config = record["config"]
        capacity = config["capacity"]
        assert record["protocol"] == task_pilot.CAPACITY_PROTOCOL
        assert record["schema"] == task_pilot.CAPACITY_PROTOCOL + "/run"
        assert record["learner_family"] == cp.LEARNER_FAMILY
        assert record["student_width"] == width
        assert record["capacity_spec_sha256"] == capacity["spec_sha256"]
        assert config["learner_family"] == cp.LEARNER_FAMILY
        assert config["restriction_id"] == "identity"
        assert capacity["spec"]["hidden_width"] == width
        assert capacity["spec_sha256"] == cp.spec_sha256(capacity["spec"])
        assert capacity["spec"]["teacher"]["policy_state_sha256"] == (
            record["preparation"]["policy_state_sha256"]
        )
        assert capacity["spec"]["initialization"]["seed"] == SEED
        assert capacity["trainable_parameters"] == 4 * width + 2 * width + width + 2
        assert capacity["optimizer_parameters"] == capacity["trainable_parameters"]
        assert capacity["scope"] == "reduced-budget"
        assert capacity["student_widths_declared"] == list(cp.STUDENT_WIDTHS)
        experiment = config["experiment"]
        assert experiment["bc_n_epochs"] == 20 and experiment["l2_lambda"] == 0.0
        assert experiment["warm_start"] is False
        assert config["beta"] == {"bc": None, "bc_iid": 1.0, "ftl": 0.0}[method]
        assert record["data"]["data_result_sha256"] == data_sha
        assert record["data"]["pool_pairs_sha256"] == (
            data["datasets"]["pool"]["pairs_sha256"]
        )
        trained[method, width] = record["trained_pairs_sha256"]
        fits = record["accounting"]["completed_records"]["training"]["fits"]
        evals = [r for r in record["records"] if r.get("d_eval_size") is not None]
        if method == "bc":
            assert fits == 1 and [r["reference"] for r in evals] == ["flat"]
            assert config["eval_budgets"] == [BUDGET]
        else:
            # Per-round checkpoint retention requires a fresh fit each round.
            assert fits == BUDGET
            assert config["eval_budgets"] == list(range(BUDGET + 1))
        for row in evals:
            policy = cp.load_checkpoint(
                out / row["checkpoint"],
                expected_spec_sha256=capacity["spec_sha256"],
                expected_file_sha256=row["checkpoint_binding"]["file_sha256"],
                expected_state_sha256=row["checkpoint_binding"]["state_sha256"],
            )
            assert policy.capacity_spec["hidden_width"] == width
            assert row["checkpoint_capacity_spec_sha256"] == capacity["spec_sha256"]
    for method in ("bc", "bc_iid"):
        assert trained[method, 32] == trained[method, 4]
    shared = {"bc": "pool", "bc_iid": "stream"}
    for method, name in shared.items():
        assert trained[method, 4] == data["datasets"][name]["pairs_sha256"]


def test_every_checkpoint_row_carries_its_writer_binding(capacity_runs):
    """Round 0, every online budget and the fixed BC endpoint are all bound."""
    for (method, width), out in capacity_runs.items():
        record = _read(out)
        capacity = record["config"]["capacity"]
        rows = [r for r in record["records"] if "checkpoint" in r]
        evaluated = [r for r in record["records"] if r.get("d_eval_size") is not None]
        assert rows == evaluated
        if method == "bc":
            assert [r["round"] for r in rows] == [0]
            names = [f"round-{BUDGET:05d}.pt"]
        else:
            assert [r["round"] for r in rows] == list(range(BUDGET + 1))
            names = [f"round-{r:05d}.pt" for r in range(BUDGET + 1)]
        assert [pathlib.Path(r["checkpoint"]).name for r in rows] == names
        for row in rows:
            binding = row["checkpoint_binding"]
            path = out / row["checkpoint"]
            assert binding["schema"] == cp.CHECKPOINT_BINDING_SCHEMA
            assert binding["path"] == row["checkpoint"]
            assert binding["file_sha256"] == pilot.sha256_file(path)
            assert binding["file_bytes"] == path.stat().st_size
            assert binding["spec_sha256"] == capacity["spec_sha256"]
            assert row["checkpoint_capacity_spec_sha256"] == capacity["spec_sha256"]
            restored = cp.load_checkpoint(
                path,
                expected_spec_sha256=capacity["spec_sha256"],
                expected_file_sha256=binding["file_sha256"],
                expected_state_sha256=binding["state_sha256"],
            )
            assert cp.state_sha256(restored) == binding["state_sha256"]
        digests = [r["checkpoint_binding"]["file_sha256"] for r in rows]
        assert len(set(digests)) == len(digests)


def _swap_actor_after_writer(path):
    saved = th.load(path, weights_only=False)
    saved["state_dict"]["action_net.bias"] = th.tensor([100.0, -100.0])
    th.save(saved, path)


@pytest.mark.parametrize("method", ["bc", "ftl"])
def test_actor_swapped_after_writer_emission_fails_the_run(
    teachers,
    capacity_data,
    tmp_path,
    monkeypatch,
    method,
):
    """A valid finite actor written over the checkpoint before verification."""
    original = rp._verified_checkpoint

    def swapped(path, *args, **kwargs):
        _swap_actor_after_writer(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(rp, "_verified_checkpoint", swapped)
    out = tmp_path / method
    code = _run(out, teachers["one_hidden"], capacity_data, method, 4)
    record = _read(out)
    assert code == classical.EXIT_INCOMPLETE and record["status"] == "failed"
    assert "writer" in record["error"]["message"]
    assert not [r for r in record["records"] if "checkpoint" in r]


def test_paired_widths_get_distinct_specs_and_identical_teacher(capacity_runs):
    specs = {
        width: _read(capacity_runs["bc", width])["config"]["capacity"]["spec"]
        for width in WIDTHS
    }
    assert specs[32]["teacher"] == specs[4]["teacher"]
    assert specs[32]["encoder"] == specs[4]["encoder"]
    assert cp.spec_sha256(specs[32]) != cp.spec_sha256(specs[4])


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"student_width": 64}, "student"),
        ({"student_width": 3}, "student"),
        ({"student_width": None}, "student"),
        ({"restriction_id": "cart_position_angular_velocity_zero"}, "restriction"),
        ({"allow_relaxed_expert": True}, "relaxed"),
        ({"teacher_width": 32}, "teacher"),
    ],
)
def test_capacity_runs_refuse_before_claiming_output(
    teachers,
    capacity_data,
    tmp_path,
    capsys,
    kwargs,
    needle,
):
    out = tmp_path / "out"
    code = _run(out, teachers["one_hidden"], capacity_data, "ftl", 4, **kwargs)
    assert code == classical.EXIT_REFUSED
    assert not out.exists()
    assert needle in capsys.readouterr().err


def test_capacity_runs_refuse_external_reference(teachers, capacity_data, tmp_path):
    code = _run(
        tmp_path / "out",
        teachers["one_hidden"],
        capacity_data,
        "bc_iid",
        4,
        reference_data_dir=capacity_data,
        expected_reference_sha256=pilot.sha256_file(capacity_data / rp.RESULT_FILE),
    )
    assert code == classical.EXIT_REFUSED
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["two_hidden_claimed", "one_hidden_legacy_claim"])
def test_incompatible_or_unclaimed_teachers_refuse(teachers, tmp_path, name):
    code = _data(tmp_path / "data", teachers[name], learner_family=cp.LEARNER_FAMILY)
    assert code == classical.EXIT_REFUSED
    assert not (tmp_path / "data").exists()


def test_capacity_runs_refuse_legacy_data_and_legacy_refuses_capacity_args(
    teachers,
    tmp_path,
):
    legacy_data = tmp_path / "legacy-data"
    assert _data(legacy_data, teachers["one_hidden"]) == classical.EXIT_COMPLETE
    code = _run(tmp_path / "a", teachers["one_hidden"], legacy_data, "bc", 4)
    assert code == classical.EXIT_REFUSED
    for extra in ({"student_width": 4}, {"teacher_width": 64}):
        code = _run(
            tmp_path / "b",
            teachers["one_hidden"],
            legacy_data,
            "bc",
            None,
            learner_family=task_pilot.LEGACY_FAMILY,
            **extra,
        )
        assert code == classical.EXIT_REFUSED
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


# ---------------------------------------------------------------------------
# Legacy defaults
# ---------------------------------------------------------------------------


def test_legacy_specs_and_defaults_are_unchanged():
    spec = task_pilot.runner_spec("CartPole-v1")
    assert spec.protocol == task_pilot.PROTOCOL == "agnostic-classical-restriction/1"
    assert spec.source_files == task_pilot.SOURCE_FILES == LEGACY_SOURCE_FILES
    assert spec.learner is None and rp.CARTPOLE_V2.learner is None
    assert rp.CARTPOLE_V2.protocol == "agnostic-cartpole-restriction-pilot/2"
    assert rp.EVAL_INTERVAL == 10 and rp.DEFAULT_BUDGET == 1000
    assert task_pilot.CAPACITY_EVAL_INTERVAL == 20
    config = rp._experiment_config("ftl", 300, 1000, 10, "/tmp/x", "CartPole-v1")
    assert config.policy_mode == "linear" and config.bc_n_epochs == 20
    args = task_pilot.build_parser().parse_args(
        [
            "run",
            "--env",
            "CartPole-v1",
            "--expert-dir",
            "e",
            "--output-dir",
            "o",
            "--deadline",
            "2026-10-07T00:00:00Z",
            "--job-limit-seconds",
            "60",
            "--data-dir",
            "d",
            "--seed",
            "300",
            "--method",
            "bc",
            "--restriction",
            "identity",
        ],
    )
    assert args.learner_family == task_pilot.LEGACY_FAMILY
    assert args.student_width is None and args.teacher_width is None
    assert args.n_rounds is None and args.eval_interval is None


def test_legacy_generic_run_still_uses_the_linear_head(teachers, tmp_path):
    data = tmp_path / "data"
    assert _data(data, teachers["two_hidden_claimed"]) == classical.EXIT_COMPLETE
    out = tmp_path / "bc"
    code = task_pilot.run_job(
        out,
        teachers["two_hidden_claimed"],
        data,
        env_name="CartPole-v1",
        method="bc",
        restriction_id="cart_position_angular_velocity_zero",
        seed=SEED,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        n_rounds=BUDGET,
        eval_interval=BUDGET,
        clock=Clock(),
    )
    assert code == classical.EXIT_COMPLETE
    record = _read(out)
    assert record["protocol"] == task_pilot.PROTOCOL
    assert "learner_family" not in record and "capacity" not in record["config"]
    assert record["config"]["policy"].startswith("linear: expert clone")
    [row] = [r for r in record["records"] if r.get("d_eval_size") is not None]
    policy = restriction.load_policy_checkpoint(out / row["checkpoint"])
    assert policy.restriction_id == "cart_position_angular_velocity_zero"
    assert not {"checkpoint_binding", "checkpoint_capacity_spec_sha256"} & set(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(command, *extra):
    base = [
        command,
        "--env",
        "CartPole-v1",
        "--expert-dir",
        "e",
        "--output-dir",
        "o",
        "--deadline",
        "2026-10-07T00:00:00Z",
        "--job-limit-seconds",
        "60",
        "--seed",
        "4100",
    ]
    if command == "run":
        base += ["--data-dir", "d", "--method", "ftl", "--restriction", "identity"]
    return base + list(extra)


def test_cli_forwards_capacity_arguments(monkeypatch):
    calls = {}

    def capture(name):
        def fake(*args, **kwargs):
            calls[name] = kwargs
            return classical.EXIT_REFUSED

        return fake

    monkeypatch.setattr(task_pilot, "run_job", capture("run"))
    monkeypatch.setattr(task_pilot, "prepare_data", capture("data"))
    family = ["--learner-family", cp.LEARNER_FAMILY]
    code = task_pilot.main(
        _cli("run", *family, "--student-width", "8", "--n-rounds", "40")
        + ["--eval-interval", "20", "--teacher-width", "64"],
    )
    assert code == classical.EXIT_REFUSED
    assert calls["run"]["learner_family"] == cp.LEARNER_FAMILY
    assert calls["run"]["student_width"] == 8 and calls["run"]["teacher_width"] == 64
    assert calls["run"]["n_rounds"] == 40 and calls["run"]["eval_interval"] == 20
    task_pilot.main(_cli("data", *family))
    assert calls["data"]["learner_family"] == cp.LEARNER_FAMILY
    # Legacy calls forward nothing new, so every old default applies.
    task_pilot.main(_cli("run"))
    assert not {"learner_family", "student_width", "n_rounds"} & set(calls["run"])


@pytest.mark.parametrize(
    "argv",
    [
        _cli("run", "--student-width", "8"),
        _cli("run", "--n-rounds", "5"),
        _cli("run", "--learner-family", cp.LEARNER_FAMILY),
        _cli("run", "--learner-family", cp.LEARNER_FAMILY, "--student-width", "64"),
        _cli("data", "--teacher-width", "64"),
    ],
)
def test_cli_rejects_inconsistent_family_arguments(argv):
    with pytest.raises(SystemExit) as excinfo:
        task_pilot.main(argv)
    assert excinfo.value.code == 2


def test_runner_state_isolation_against_global_rng(teachers):
    expert, _ = task_experts.load_qualified_expert(
        teachers["one_hidden"],
        "CartPole-v1",
    )
    th.manual_seed(SEED)
    np.random.seed(SEED)
    before = th.get_rng_state(), np.random.get_state()[1].copy()
    for width in cp.STUDENT_WIDTHS:
        cp.make_capacity_policy(
            expert.policy, hidden_width=width, initialization_seed=SEED
        )
    assert th.equal(before[0], th.get_rng_state())
    assert np.array_equal(before[1], np.random.get_state()[1])
