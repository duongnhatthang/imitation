"""Environment-generic restriction runner: experts, data, runs and refusals.

Experts are exact tabular or linear policies, so qualification outcomes are
known. The CartPole regression runs the pilot protocol and the generic
protocol on the same expert and seed and compares everything scientific.
"""

import datetime
from types import SimpleNamespace
import json
import pathlib
import shutil

import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO

from imitation.data import serialize
from imitation.experiments.agnostic import classical, restriction
from imitation.experiments.agnostic import restriction_pilot as rp
from imitation.experiments.agnostic import (
    rollouts,
    run_classical,
    task_experts,
    task_pilot,
    tasks,
)
from imitation.experiments.ftrl import env_utils

UTC = datetime.timezone.utc
FAR = datetime.datetime(2026, 10, 7, 1, 0, tzinfo=UTC)
LIMIT = 3600
# FrozenLake 4x4 deterministic: LEFT 0, DOWN 1, RIGHT 2, UP 3.
LAKE_PATH = {0: 1, 4: 1, 8: 2, 9: 1, 13: 2, 14: 2}
# CliffWalking: UP 0, RIGHT 1, DOWN 2, LEFT 3; up, along row 2, then down.
CLIFF_PATH = {36: 0, **{s: 1 for s in range(24, 35)}, 35: 2}
LAKE_BUDGET = 9  # two 6-step expert episodes; 3 transitions of overshoot
CART_BUDGET = 6


@pytest.fixture(scope="module", autouse=True)
def pilot_clock():
    """Keep CLI and default-clock fixtures inside the frozen pilot window."""
    class PilotDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            point = cls(2026, 10, 1, tzinfo=UTC)
            return point.astimezone(tz) if tz is not None else point.replace(tzinfo=None)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rollouts, "datetime", SimpleNamespace(datetime=PilotDatetime))
        yield


class Clock:
    """Deterministic clock that advances a fixed step per reading."""

    def __init__(self, step_seconds=0.001):
        self.now = datetime.datetime(2026, 10, 1, tzinfo=UTC)
        self.step = datetime.timedelta(seconds=step_seconds)

    def __call__(self):
        self.now += self.step
        return self.now


def _tabular_expert(env_name, actions, path, default=0):
    """Linear policy on the one-hot observation taking ``actions[state]``."""
    venv = env_utils.make_env(env_name, 1, np.random.default_rng(0))
    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(net_arch=dict(pi=[], vf=[])),
        device="cpu",
        seed=0,
    )
    with th.no_grad():
        weight = model.policy.action_net.weight
        weight.zero_()
        model.policy.action_net.bias.zero_()
        for s in range(weight.shape[1]):
            weight[actions.get(s, default), s] = 10.0
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    venv.close()


def _cart_expert(path):
    """Linear expert that balances CartPole (as in the pilot tests)."""
    venv = env_utils.make_env("CartPole-v1", 1, np.random.default_rng(0))
    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(net_arch=dict(pi=[], vf=[])),
        device="cpu",
        seed=0,
    )
    with th.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.action_net.weight[1] = th.tensor([1.0, 1.0, 10.0, 10.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    venv.close()


def _prepare(env_name, out, save):
    def trainer(env, cache_dir, rng, seed, convergence_override=None):
        save(pathlib.Path(cache_dir) / env / "model.zip")

    code = run_classical.prepare_expert(
        env_name,
        out,
        seed=0,
        qualification_seed=1234,
        eval_episodes=2,
        deadline=FAR,
        trainer=trainer,
    )
    assert code == 0
    return out


def _read(out, name=rp.RESULT_FILE):
    return json.loads((pathlib.Path(out) / name).read_text())


def _sha(path):
    return rp.pilot.sha256_file(path)


@pytest.fixture(scope="module")
def lake_candidate(tmp_path_factory):
    path = tmp_path_factory.mktemp("cache") / "FrozenLake-v1" / "model.zip"
    _tabular_expert("FrozenLake-v1", LAKE_PATH, path)
    (path.parent / "baselines.json").write_text('{"expert_return": 1.0}')
    return path


@pytest.fixture(scope="module")
def lake_expert(lake_candidate, tmp_path_factory):
    out = tmp_path_factory.mktemp("lake") / "expert"
    code = task_experts.qualify_candidate(
        "FrozenLake-v1",
        lake_candidate,
        out,
        qualification_seed=1234,
        eval_episodes=3,
        deadline=FAR,
        clock=Clock(),
    )
    assert code == 0
    return out


@pytest.fixture(scope="module")
def cliff_expert(tmp_path_factory):
    return _prepare(
        "CliffWalking-v0",
        tmp_path_factory.mktemp("cliff") / "prep",
        lambda p: _tabular_expert("CliffWalking-v0", CLIFF_PATH, p),
    )


@pytest.fixture(scope="module")
def cart_expert(tmp_path_factory):
    return _prepare(
        "CartPole-v1", tmp_path_factory.mktemp("cart") / "prep", _cart_expert
    )


def _data(env_name, expert, out, seed, budget, **kwargs):
    code = task_pilot.prepare_data(
        out,
        expert,
        env_name=env_name,
        seed=seed,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        budget=budget,
        baseline_episodes=2,
        clock=Clock(),
        **kwargs,
    )
    assert code == classical.EXIT_COMPLETE
    return out


@pytest.fixture(scope="module")
def lake_data(lake_expert, tmp_path_factory):
    out = tmp_path_factory.mktemp("lake-data") / "data"
    return _data("FrozenLake-v1", lake_expert, out, 300, LAKE_BUDGET)


@pytest.fixture(scope="module")
def cliff_data(cliff_expert, tmp_path_factory):
    out = tmp_path_factory.mktemp("cliff-data") / "data"
    return _data("CliffWalking-v0", cliff_expert, out, 300, 6)


@pytest.fixture(scope="module")
def cart_data(cart_expert, tmp_path_factory):
    """Pilot-protocol and generic-protocol data jobs, same expert and seed."""
    root = tmp_path_factory.mktemp("cart-data")
    code = rp.prepare_data(
        root / "pilot",
        cart_expert,
        seed=300,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        budget=CART_BUDGET,
        baseline_episodes=2,
        clock=Clock(),
    )
    assert code == classical.EXIT_COMPLETE
    _data("CartPole-v1", cart_expert, root / "generic", 300, CART_BUDGET)
    _data("CartPole-v1", cart_expert, root / "generic302", 302, CART_BUDGET)
    return root


def _run(env_name, expert, data, out, method, restriction_id, **kwargs):
    params = dict(
        env_name=env_name,
        method=method,
        restriction_id=restriction_id,
        seed=300,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        n_rounds=kwargs.pop("n_rounds", LAKE_BUDGET),
        eval_interval=kwargs.pop("eval_interval", LAKE_BUDGET),
        clock=Clock(),
    )
    params.update(kwargs)
    return task_pilot.run_job(out, expert, data, **params)


# ---------------------------------------------------------------------------
# Experts
# ---------------------------------------------------------------------------


def test_candidate_qualification_is_its_own_kind_of_evidence(
    lake_candidate,
    lake_expert,
    tmp_path,
):
    record = _read(lake_expert, task_experts.CANDIDATE_FILE)
    assert record["status"] == "complete" and record["qualified"] is True
    assert record["protocol"] == task_experts.CANDIDATE_SCHEMA
    assert not (lake_expert / "preparation.json").exists()
    assert (lake_expert / task_experts.CANDIDATE_CHECKPOINT).read_bytes() == (
        lake_candidate.read_bytes()
    )
    candidate = record["config"]["candidate"]
    assert candidate["label"] == "FrozenLake-v1/model.zip"
    assert candidate["training_provenance"].startswith("unknown")
    assert candidate["cached_baselines_unverified"]["content"] == {
        "expert_return": 1.0,
    }
    assert record["qualification"]["episode_returns"] == [1.0, 1.0, 1.0]
    assert record["quality"]["per_env_passed"] and record["quality"]["global_passed"]

    _, identity = task_experts.load_qualified_expert(lake_expert, "FrozenLake-v1")
    assert identity["expert_source"] == "candidate_qualification"
    assert identity["quality"]["relaxed_only"] is False

    # A failing candidate is kept as evidence and never loads.
    bad = tmp_path / "cache" / "FrozenLake-v1" / "model.zip"
    _tabular_expert("FrozenLake-v1", {}, bad)
    out = tmp_path / "failed"
    code = task_experts.qualify_candidate(
        "FrozenLake-v1",
        bad,
        out,
        qualification_seed=1234,
        eval_episodes=3,
        deadline=FAR,
        clock=Clock(),
    )
    failed = _read(out, task_experts.CANDIDATE_FILE)
    assert code == classical.EXIT_INCOMPLETE
    assert failed["status"] == "not_qualified" and failed["qualified"] is False
    with pytest.raises(classical.Refused, match="threshold_passed"):
        task_experts.load_qualified_expert(out, "FrozenLake-v1")
    with pytest.raises(classical.Refused, match="env_name"):
        task_experts.load_qualified_expert(lake_expert, "Taxi-v3")
    # The output must be fresh; the failed record is not overwritten.
    before = _sha(out / task_experts.CANDIDATE_FILE)
    code = task_experts.qualify_candidate(
        "FrozenLake-v1",
        lake_candidate,
        out,
        qualification_seed=1234,
        eval_episodes=3,
        deadline=FAR,
    )
    assert code == classical.EXIT_REFUSED
    assert _sha(out / task_experts.CANDIDATE_FILE) == before

    both = tmp_path / "both"
    shutil.copytree(lake_expert, both)
    (both / "preparation.json").write_text("{}")
    with pytest.raises(classical.Refused, match="exactly one"):
        task_experts.load_qualified_expert(both, "FrozenLake-v1")


def test_prepare_expert_supports_fallback_cap_tasks(cliff_expert):
    record = _read(cliff_expert, "preparation.json")
    assert record["status"] == "complete" and record["qualified"] is True
    assert record["config"]["env_max_episode_steps"] == 200
    assert record["config"]["env_registered_max_episode_steps"] is None
    assert record["qualification"]["episode_returns"] == [-13.0, -13.0]
    _, identity = task_experts.load_qualified_expert(cliff_expert, "CliffWalking-v0")
    assert identity["expert_source"] == "preparation"
    assert (
        identity["quality"]["per_env_passed"] and identity["quality"]["global_passed"]
    )


def test_relaxed_only_experts_need_explicit_permission():
    spec = task_pilot.runner_spec("Taxi-v3")
    # Taxi global 0.80 raw threshold: -763 + 0.8 * 770.9 = -146.28.
    relaxed = {"quality": task_experts.expert_quality("Taxi-v3", -200.0, -260.0)}
    assert relaxed["quality"]["relaxed_only"] is True
    with pytest.raises(classical.Refused, match="relaxed-only"):
        rp._check_expert_quality(relaxed, spec, False)
    rp._check_expert_quality(relaxed, spec, True)
    failing = {"quality": task_experts.expert_quality("Taxi-v3", -300.0, -260.0)}
    with pytest.raises(classical.Refused, match="qualification threshold"):
        rp._check_expert_quality(failing, spec, True)


# ---------------------------------------------------------------------------
# Data and runs
# ---------------------------------------------------------------------------


def test_generic_data_job_binds_the_task_and_keeps_pool_and_stream_apart(
    lake_data,
    lake_expert,
):
    record = _read(lake_data)
    assert record["status"] == "complete"
    assert record["protocol"] == task_pilot.PROTOCOL
    assert record["schema"] == task_pilot.PROTOCOL + "/data"
    assert record["env_name"] == "FrozenLake-v1" and record["episode_cap"] == 100
    task = tasks.get_task("FrozenLake-v1")
    assert record["task"] == tasks.describe(task)
    assert record["task_sha256"] == tasks.task_sha256(task)
    assert record["preparation"]["expert_source"] == "candidate_qualification"
    assert record["config"]["allow_relaxed_expert"] is False
    assert sorted(record["source"]["files"]) == sorted(task_pilot.SOURCE_FILES)

    pool = rp.ExpertData.load(lake_data / rp.POOL_FILE)
    stream = rp.ExpertData.load(lake_data / rp.STREAM_FILE)
    # Pool: complete episodes in chronological order, the first budget kept.
    np.testing.assert_array_equal(pool.episode_index, [0] * 6 + [1] * 3)
    np.testing.assert_array_equal(pool.step_index, list(range(6)) + [0, 1, 2])
    assert record["datasets"]["pool"]["stats"]["overshoot_transitions"] == 3
    # Stream: one uniformly selected state from each independent episode.
    np.testing.assert_array_equal(stream.episode_index, np.arange(LAKE_BUDGET))
    assert record["datasets"]["stream"]["stats"]["episodes"] == LAKE_BUDGET
    assert len(set(stream.step_index.tolist())) > 1
    path = [0, 4, 8, 9, 13, 14]
    for obs, act, k in zip(stream.obs, stream.acts, stream.step_index):
        assert int(obs.argmax()) == path[k] and act == LAKE_PATH[path[k]]


def test_full_and_masked_generic_runs_share_data_and_bc_fits_once(
    lake_expert,
    lake_data,
    tmp_path,
):
    data = _read(lake_data)
    masked = "frozenlake_column_pairs"
    records = {}
    for method in ("bc", "bc_iid"):
        for rid in ("identity", masked):
            out = tmp_path / f"{method}-{rid}"
            code = _run("FrozenLake-v1", lake_expert, lake_data, out, method, rid)
            assert code == classical.EXIT_COMPLETE
            records[method, rid] = (out, _read(out))
    for (method, rid), (out, record) in records.items():
        name = "pool" if method == "bc" else "stream"
        assert record["protocol"] == task_pilot.PROTOCOL
        assert record["data"]["data_result_sha256"] == _sha(lake_data / rp.RESULT_FILE)
        assert record["trained_pairs_sha256"] == data["datasets"][name]["pairs_sha256"]
        config = record["config"]
        assert config["restriction"] == json.loads(
            json.dumps(restriction.describe_restriction(rid)),
        )
        assert config["reference"]["source"] == "own data job"
        assert config["task_sha256"] == data["task_sha256"]
        fits = record["accounting"]["completed_records"]["training"]["fits"]
        evals = [r for r in record["records"] if r.get("d_eval_size") is not None]
        if method == "bc":
            # One fit on the whole pool and one evaluation: a flat reference.
            assert fits == 1 and [r["reference"] for r in evals] == ["flat"]
            assert config["eval_budgets"] == [LAKE_BUDGET]
        else:
            assert fits == LAKE_BUDGET
            assert config["eval_budgets"] == [0, 1, LAKE_BUDGET]
        for r in evals:
            policy = restriction.load_policy_checkpoint(out / r["checkpoint"])
            assert policy.restriction_id == rid
            assert len(r["episode_returns"]) == 100
    pairs = {
        key: [r.get("trained_pairs_sha256") for r in rec["records"]]
        for key, (_, rec) in records.items()
    }
    assert pairs["bc", "identity"] == pairs["bc", masked]
    assert pairs["bc_iid", "identity"] == pairs["bc_iid", masked]


def test_masked_ftl_labels_learner_states_with_the_unmasked_expert(
    lake_expert,
    lake_data,
    tmp_path,
):
    out = tmp_path / "ftl"
    rid = "frozenlake_column_hidden"
    assert _run("FrozenLake-v1", lake_expert, lake_data, out, "ftl", rid) == 0
    record = _read(out)
    expert, _ = task_experts.load_qualified_expert(lake_expert, "FrozenLake-v1")
    demos = []
    for path in sorted((out / record["scratch_demos"]).rglob("*.npz")):
        demos.extend(serialize.load(path))
    obs = np.concatenate([d.obs[:-1] for d in demos])
    acts = np.concatenate([d.acts for d in demos])
    assert len(acts) == LAKE_BUDGET
    # Stored states are full one-hot encodings; labels are the expert's on them.
    assert set(np.unique(obs)) <= {0.0, 1.0} and np.all(obs.sum(1) == 1)
    np.testing.assert_array_equal(
        acts, expert.policy.predict(obs, deterministic=True)[0]
    )


def test_cartpole_generic_protocol_reproduces_the_pilot(
    cart_expert,
    cart_data,
    tmp_path,
):
    pilot_data, generic_data = _read(cart_data / "pilot"), _read(cart_data / "generic")
    assert pilot_data["protocol"] == rp.DATA_PROTOCOL
    assert generic_data["protocol"] == task_pilot.PROTOCOL
    for name in ("pool", "stream"):
        assert (
            pilot_data["datasets"][name]["pairs_sha256"]
            == generic_data["datasets"][name]["pairs_sha256"]
        )
    assert pilot_data["baselines"] == generic_data["baselines"]

    obs = np.random.default_rng(0).normal(size=(256, 4)).astype(np.float32)
    for method in ("bc", "bc_iid"):
        common = dict(
            method=method,
            restriction_id=rp.PILOT_RESTRICTION,
            seed=300,
            deadline=FAR,
            job_limit_seconds=LIMIT,
            n_rounds=CART_BUDGET,
            eval_interval=3,
        )
        a, b = tmp_path / f"{method}-pilot", tmp_path / f"{method}-generic"
        assert (
            rp.run_job(a, cart_expert, cart_data / "pilot", clock=Clock(), **common)
            == 0
        )
        code = task_pilot.run_job(
            b,
            cart_expert,
            cart_data / "generic",
            env_name="CartPole-v1",
            clock=Clock(),
            **common,
        )
        assert code == 0
        old, new = _read(a), _read(b)
        assert old["protocol"] == rp.PROTOCOL and new["protocol"] == task_pilot.PROTOCOL
        assert old["config"]["eval_budgets"] == new["config"]["eval_budgets"]
        assert old["config"]["experiment"] == new["config"]["experiment"]
        assert old["config"]["restriction"] == new["config"]["restriction"]
        assert old["trained_pairs_sha256"] == new["trained_pairs_sha256"]
        assert len(old["records"]) == len(new["records"])
        for r, s in zip(old["records"], new["records"]):
            for key in (
                "n_observations",
                "normalized_return",
                "episode_returns",
                "disagreement_rate",
                "rollout_cross_entropy",
                "train_batch_size",
                "inner_es_stop_epoch",
                "prefix_pairs_sha256",
                "trained_pairs_sha256",
            ):
                assert r.get(key) == s.get(key), key
            if r.get("checkpoint"):
                np.testing.assert_array_equal(
                    restriction.load_policy_checkpoint(a / r["checkpoint"]).predict(
                        obs,
                        deterministic=True,
                    )[0],
                    restriction.load_policy_checkpoint(b / s["checkpoint"]).predict(
                        obs,
                        deterministic=True,
                    )[0],
                )


def test_data_checks_refuse_mismatched_seed_pin_task_and_environment(
    lake_expert,
    lake_data,
    cliff_data,
    cart_expert,
    cart_data,
    tmp_path,
    capsys,
):
    def refused(out, reason, *args, **kwargs):
        assert _run(*args, out, **kwargs) == classical.EXIT_REFUSED
        assert reason in capsys.readouterr().err
        assert not out.exists()

    lake = ("FrozenLake-v1", lake_expert, lake_data)
    refused(
        tmp_path / "seed",
        "'seed'",
        *lake,
        method="bc",
        restriction_id="identity",
        seed=299,
    )
    refused(
        tmp_path / "pin",
        "not the pinned",
        *lake,
        method="bc",
        restriction_id="identity",
        expected_data_sha256="0" * 64,
    )
    refused(
        tmp_path / "ladder",
        "restriction must be one of",
        *lake,
        method="bc",
        restriction_id="taxi_destination_hidden",
    )
    refused(
        tmp_path / "env",
        "'env_name'",
        "FrozenLake-v1",
        lake_expert,
        cliff_data,
        method="bc_iid",
        restriction_id="identity",
        n_rounds=6,
    )
    # The pilot data protocol is not generic data, even for CartPole.
    refused(
        tmp_path / "protocol",
        "'protocol'",
        "CartPole-v1",
        cart_expert,
        cart_data / "pilot",
        method="bc_iid",
        restriction_id="identity",
        n_rounds=CART_BUDGET,
        expected_data_sha256=_sha(cart_data / "pilot" / rp.RESULT_FILE),
    )
    tampered = tmp_path / "tampered"
    shutil.copytree(lake_data, tampered)
    record = _read(tampered)
    record["task"]["episode_cap"] = 101
    (tampered / rp.RESULT_FILE).write_text(json.dumps(record))
    refused(
        tmp_path / "task",
        "'task'",
        "FrozenLake-v1",
        lake_expert,
        tampered,
        method="bc",
        restriction_id="identity",
        expected_data_sha256=_sha(tampered / rp.RESULT_FILE),
    )
    # The pilot protocol takes none of the generic options.
    out = tmp_path / "pilot-reference"
    code = rp.run_job(
        out,
        cart_expert,
        cart_data / "pilot",
        method="bc",
        restriction_id="identity",
        seed=300,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        n_rounds=CART_BUDGET,
        reference_data_dir=cart_data / "pilot",
    )
    assert code == classical.EXIT_REFUSED and not out.exists()


def test_a_rung_leaving_one_distinct_pool_observation_is_refused(
    cliff_expert,
    cliff_data,
    tmp_path,
    capsys,
):
    # The expert path 36, 24, 25, ... collapses to one state once rows 2 and 3
    # are merged and the column is hidden.
    cliff = ("CliffWalking-v0", cliff_expert, cliff_data)
    out = tmp_path / "collapsed"
    code = _run(
        *cliff,
        out,
        method="bc",
        restriction_id="cliffwalking_column_hidden_row_pairs",
        n_rounds=6,
    )
    assert code == classical.EXIT_REFUSED and not out.exists()
    assert "one distinct observation" in capsys.readouterr().err
    out = tmp_path / "two"
    code = _run(
        *cliff,
        out,
        method="bc",
        restriction_id="cliffwalking_column_hidden",
        n_rounds=6,
    )
    assert code == classical.EXIT_COMPLETE
    record = _read(out)
    assert record["config"]["distinct_observations"] == {"pool": 6, "masked_pool": 2}
    assert record["episode_cap"] == 200


def test_a_pinned_reference_normalizes_generic_runs(
    cart_expert,
    cart_data,
    tmp_path,
    capsys,
    monkeypatch,
):
    reference = cart_data / "pilot"
    pin = _sha(reference / rp.RESULT_FILE)
    common = dict(
        method="bc",
        restriction_id="identity",
        seed=302,
        n_rounds=CART_BUDGET,
        reference_data_dir=reference,
    )
    args = ("CartPole-v1", cart_expert, cart_data / "generic302")
    assert _run(*args, tmp_path / "unpinned", **common) == classical.EXIT_REFUSED
    assert "must be pinned" in capsys.readouterr().err
    out = tmp_path / "short"
    assert _run(*args, out, expected_reference_sha256=pin, **common) == 3
    assert "'baseline_episodes'" in capsys.readouterr().err
    # The test references measure 2 episodes, not 500; accept them here only.
    monkeypatch.setattr(rp, "BASELINE_EPISODES", 2)
    out = tmp_path / "pinned"
    assert _run(*args, out, expected_reference_sha256=pin, **common) == 0
    record = _read(out)
    ref = _read(reference)["baselines"]
    own = _read(cart_data / "generic302")["baselines"]
    assert ref != own
    used = record["config"]["reference"]
    assert used["producer_schema"] == rp.DATA_SCHEMA and used["baselines"] == ref
    [row] = [r for r in record["records"] if r.get("d_eval_size") is not None]
    mean = float(np.mean(row["episode_returns"]))
    expected = (mean - ref["random_return"]) / (
        ref["expert_return"] - ref["random_return"]
    )
    assert row["normalized_return"] == round(expected, 6)


def test_cli_describes_tasks_and_reserves_the_audit_seed(lake_expert, tmp_path, capsys):
    assert task_pilot.main(["describe", "--env", "Blackjack-v1"]) == 0
    described = json.loads(capsys.readouterr().out)
    assert described["task"]["episode_cap"] == 20
    assert [e["restriction"]["restriction_id"] for e in described["ladder"]] == list(
        tasks.get_task("Blackjack-v1").ladder,
    )
    assert described["ladder"][0]["restriction"]["table_sha256"] == (
        tasks.FROZEN_TABLE_SHA256["blackjack_usable_ace_hidden"]
    )
    with pytest.raises(SystemExit) as exit_info:
        task_pilot.main(
            [
                "data",
                "--env",
                "FrozenLake-v1",
                "--expert-dir",
                str(lake_expert),
                "--output-dir",
                str(tmp_path / "data"),
                "--deadline",
                FAR.isoformat(),
                "--job-limit-seconds",
                "60",
                "--seed",
                "301",
            ],
        )
    assert exit_info.value.code == 2 and not (tmp_path / "data").exists()
    out = tmp_path / "audit"
    code = task_pilot.main(
        [
            "audit",
            "--env",
            "FrozenLake-v1",
            "--expert-dir",
            str(lake_expert),
            "--output-dir",
            str(out),
            "--deadline",
            FAR.isoformat(),
            "--job-limit-seconds",
            "60",
            "--restriction",
            "frozenlake_column_hidden",
        ],
    )
    record = _read(out)
    assert code == 0 and record["status"] == "complete"
    assert record["schema"] == task_pilot.PROTOCOL + "/audit"
    assert record["audit"]["evidence_type"] == "reachable_enumeration"
    assert record["audit"]["labels_used_for_training"] is False


# ---------------------------------------------------------------------------
# Fail-closed expert and reference evidence
# ---------------------------------------------------------------------------


def _tampered(source, tmp_path, name, mutate, *, rehash=True):
    """Copy an expert or data directory and edit its JSON record in place."""
    out = tmp_path / "tampered"
    shutil.copytree(source, out)
    path = out / name
    record = json.loads(path.read_text())
    mutate(record)
    if rehash and isinstance(record.get("config"), dict):
        record["config_sha256"] = rp.pilot.config_digest(record["config"])
    path.write_text(json.dumps(record))
    return out


def _qual(record):
    return record["qualification"]


def _drop_episode(record):
    _qual(record)["episode_returns"].pop()
    _qual(record)["episode_lengths"].pop()


def _all_counters(record, **values):
    _qual(record)["counters"].update(values)
    record["counters"]["expert_qualification"].update(values)


# (id, edit, refusal reason). Config edits are re-hashed, so only the
# evidence checks can refuse them.
_EVIDENCE_MUTATIONS = [
    (
        "no-episodes",
        lambda r: _qual(r).update(episode_returns=[], episode_lengths=[]),
        "episode_count",
    ),
    ("partial-episodes", _drop_episode, "episode_count"),
    (
        "self-reported-mean",
        lambda r: _qual(r)["episode_returns"].__setitem__(0, -1000.0),
        "mean_recomputed",
    ),
    (
        "non-finite-return",
        lambda r: _qual(r)["episode_returns"].__setitem__(0, float("nan")),
        "returns_finite",
    ),
    (
        "length-beyond-cap",
        lambda r: _qual(r)["episode_lengths"].__setitem__(0, 10_000),
        "lengths_valid",
    ),
    (
        "stop-reason",
        lambda r: _all_counters(r, stop_reason="deadline"),
        "counters",
    ),
    (
        "completed-count",
        lambda r: _all_counters(r, completed_episodes=1),
        "counters",
    ),
    (
        "reset-seed",
        lambda r: _qual(r).update(env_reset_seed=7),
        "evidence_settings",
    ),
]
_CANDIDATE_MUTATIONS = _EVIDENCE_MUTATIONS + [
    (
        "task-digest",
        lambda r: r["config"].update(task_sha256="0" * 64),
        "config_rebuilt",
    ),
    ("cap", lambda r: r["config"].update(env_episode_cap=999), "config_rebuilt"),
    (
        "threshold",
        lambda r: r["config"]["threshold"].update(raw_threshold=-1e9),
        "config_rebuilt",
    ),
    ("requested-count", lambda r: r["config"].update(eval_episodes=100), "count"),
    (
        "packages",
        lambda r: r["package_versions"].update(gymnasium="0.0.0"),
        "package_versions",
    ),
]


def _set_source(record, rel, value):
    files = record["source_fingerprint"]["files"]
    if value is None:
        files.pop(rel)
    else:
        files[rel] = value
    record["source_fingerprint"]["combined_sha256"] = rp.pilot.config_digest(files)


_PREPARATION_MUTATIONS = _EVIDENCE_MUTATIONS + [
    (
        "override",
        lambda r: r["config"].update(convergence_override={"patience": 6}),
        "convergence_override",
    ),
    (
        "threshold",
        lambda r: r["config"]["threshold"].update(raw_threshold=-1e9),
        "frozen_threshold",
    ),
    ("cap", lambda r: r["config"].update(env_max_episode_steps=999), "episode_cap"),
    ("requested-count", lambda r: r["config"].update(eval_episodes=5), "count"),
    (
        "environment-source",
        lambda r: _set_source(r, "experiments/ftrl/env_utils.py", "0" * 64),
        "environment_sources",
    ),
    (
        "missing-source",
        lambda r: _set_source(r, "util/util.py", None),
        "environment_sources",
    ),
    (
        "source-map",
        lambda r: r["source_fingerprint"].update(combined_sha256="0" * 64),
        "source_fingerprint",
    ),
    (
        "gymnasium",
        lambda r: r["package_versions"].update(gymnasium="0.0.0"),
        "gymnasium_version",
    ),
]


def _ids(cases):
    return [case[0] for case in cases]


@pytest.mark.parametrize(
    "name, mutate, reason", _CANDIDATE_MUTATIONS, ids=_ids(_CANDIDATE_MUTATIONS)
)
def test_candidate_loader_refuses_bogus_or_missing_evidence(
    lake_expert,
    tmp_path,
    name,
    mutate,
    reason,
):
    bad = _tampered(lake_expert, tmp_path, task_experts.CANDIDATE_FILE, mutate)
    with pytest.raises(classical.Refused, match=reason):
        task_experts.load_qualified_expert(bad, "FrozenLake-v1")


def test_candidate_quality_comes_from_recomputed_evidence(lake_expert, tmp_path):
    def forge(record):
        record["quality"] = {"per_env_passed": False, "per_env_raw_threshold": 9.0}

    edited = _tampered(lake_expert, tmp_path, task_experts.CANDIDATE_FILE, forge)
    _, identity = task_experts.load_qualified_expert(edited, "FrozenLake-v1")
    frozen = rp.pilot.frozen_expert_threshold("FrozenLake-v1")["raw_threshold"]
    assert identity["qualification_mean_return"] == 1.0
    assert identity["quality"] == task_experts.expert_quality(
        "FrozenLake-v1",
        1.0,
        frozen,
    )


@pytest.mark.parametrize(
    "name, mutate, reason", _PREPARATION_MUTATIONS, ids=_ids(_PREPARATION_MUTATIONS)
)
def test_preparation_loader_refuses_incompatible_or_missing_evidence(
    cliff_expert,
    tmp_path,
    name,
    mutate,
    reason,
):
    bad = _tampered(cliff_expert, tmp_path, "preparation.json", mutate)
    with pytest.raises(classical.Refused, match=reason):
        task_experts.load_qualified_expert(bad, "CliffWalking-v0")


def test_historical_shaped_preparation_keeps_loading(cart_expert, tmp_path):
    """Other host packages and preparation-driver bytes are not required."""

    def historical(record):
        record["package_versions"].update(python="3.10.21", torch="2.4.1+cu121")
        _set_source(record, "experiments/agnostic/pilot.py", "1" * 64)
        _set_source(record, "experiments/agnostic/run_classical.py", "2" * 64)
        _set_source(record, "experiments/agnostic/tasks.py", None)
        _set_source(record, "experiments/agnostic/task_experts.py", None)

    edited = _tampered(cart_expert, tmp_path, "preparation.json", historical)
    _, identity = task_experts.load_qualified_expert(edited, "CartPole-v1")
    frozen = rp.pilot.frozen_expert_threshold("CartPole-v1")["raw_threshold"]
    assert identity["expert_source"] == "preparation"
    assert identity["qualification_mean_return"] == 500.0
    assert identity["quality"] == task_experts.expert_quality(
        "CartPole-v1",
        500.0,
        frozen,
    )


def test_codex_reference_reproduction_is_refused(tmp_path):
    spec = task_pilot.runner_spec("FrozenLake-v1")
    identity = {"expert_sha256": "a" * 64, "policy_state_sha256": "b" * 64}
    record = {
        "schema": spec.data_schema,
        "protocol": "wrong-protocol",
        "status": "complete",
        "env_name": spec.env_name,
        "episode_cap": 100,
        "expert_sha256": identity["expert_sha256"],
        "baselines": {
            "expert_return": 1.0,
            "random_return": 0.0,
            "expert_self_ce": 0.2,
        },
        "baselines_costs": {"expert_episodes": 500, "random_episodes": 500},
        "task": {"configured_kwargs": {"is_slippery": True}},
        "task_sha256": "0" * 64,
        "package_versions": {"gymnasium": "wrong"},
        "config": {"seed": 301, "baseline_episodes": 1},
        "config_sha256": "0" * 64,
        "source": {"files": {}, "combined_sha256": "bad"},
        "seed": 301,
    }
    (tmp_path / rp.RESULT_FILE).write_text(json.dumps(record))
    pin = _sha(tmp_path / rp.RESULT_FILE)
    with pytest.raises(classical.Refused) as refused:
        rp._verify_reference(tmp_path, pin, identity, spec)
    for reason in ("protocol_pair", "task", "package_versions", "seed", "source"):
        assert reason in str(refused.value)


def _reference(directory, spec, identity):
    pin = _sha(pathlib.Path(directory) / rp.RESULT_FILE)
    return rp._verify_reference(directory, pin, identity, spec)


@pytest.fixture
def cart_reference(cart_expert, monkeypatch):
    # The test data jobs measure 2 baseline episodes, not 500.
    monkeypatch.setattr(rp, "BASELINE_EPISODES", 2)
    spec = task_pilot.runner_spec("CartPole-v1")
    return spec, rp._load_expert(cart_expert, spec)[1]


def test_legitimate_references_from_other_seeds_and_the_pilot_are_accepted(
    cart_data,
    cart_reference,
):
    spec, identity = cart_reference
    generic = _reference(cart_data / "generic302", spec, identity)
    assert generic["producer_seed"] == 302
    assert generic["producer_schema"] == spec.data_schema
    legacy = _reference(cart_data / "pilot", spec, identity)
    assert legacy["producer_seed"] == 300
    assert legacy["producer_schema"] == rp.DATA_SCHEMA


def _seed(record, value):
    record["seed"] = value
    record["config"]["seed"] = value


_REFERENCE_MUTATIONS = [
    ("protocol", lambda r: r.update(protocol=rp.DATA_PROTOCOL), True, "protocol_pair"),
    (
        "task",
        lambda r: r["task"].update(configured_env_kwargs={"x": 1}),
        True,
        "'task'",
    ),
    (
        "packages",
        lambda r: r["package_versions"].update(torch="0.0.0"),
        True,
        "package_versions",
    ),
    ("config-digest", lambda r: r.update(config_sha256="0" * 64), False, "config_sha"),
    ("rule", lambda r: r["config"].update(pool="prefix"), True, "config_rules"),
    (
        "source-map",
        lambda r: r["source"].update(combined_sha256="0" * 64),
        True,
        "source_integrity",
    ),
    ("audit-seed", lambda r: _seed(r, 301), True, "seed"),
    ("seed-mismatch", lambda r: r["config"].update(seed=303), True, "seed"),
    (
        "baseline-costs",
        lambda r: r["baselines_costs"].update(random_episodes=3),
        True,
        "baseline_episodes",
    ),
    ("incomplete", lambda r: r.update(status="partial"), True, "status_complete"),
    ("other-expert", lambda r: r.update(expert_sha256="b" * 64), True, "expert"),
]


def _drop_source_file(record):
    files = record["source"]["files"]
    files.pop("algorithms/bc.py")
    record["source"]["combined_sha256"] = rp.pilot.config_digest(files)


@pytest.mark.parametrize(
    "name, mutate, rehash, reason",
    _REFERENCE_MUTATIONS
    + [("source-file-set", _drop_source_file, True, "source_file_set")],
    ids=_ids(_REFERENCE_MUTATIONS) + ["source-file-set"],
)
def test_reference_gate_refuses_incompatible_records(
    cart_data,
    cart_reference,
    tmp_path,
    name,
    mutate,
    rehash,
    reason,
):
    spec, identity = cart_reference
    bad = _tampered(
        cart_data / "generic302",
        tmp_path,
        rp.RESULT_FILE,
        mutate,
        rehash=rehash,
    )
    with pytest.raises(classical.Refused, match=reason):
        _reference(bad, spec, identity)


def test_the_legacy_reference_pair_is_only_for_cartpole(
    cart_data,
    cart_reference,
    tmp_path,
):
    spec, identity = cart_reference

    # A generic file set under the legacy pair is not the pilot's record.
    def generic_files(record):
        files = dict(
            record["source"]["files"], **{"experiments/agnostic/tasks.py": "0" * 64}
        )
        record["source"]["files"] = files
        record["source"]["combined_sha256"] = rp.pilot.config_digest(files)

    bad = _tampered(cart_data / "pilot", tmp_path, rp.RESULT_FILE, generic_files)
    with pytest.raises(classical.Refused, match="source_file_set"):
        _reference(bad, spec, identity)
    lake = task_pilot.runner_spec("FrozenLake-v1")
    with pytest.raises(classical.Refused, match="protocol_pair"):
        _reference(cart_data / "pilot", lake, identity)
