"""BC-prefix: the original ``bc_prefix`` baseline as an opt-in generic method.

FrozenLake (deterministic, exact tabular expert) keeps the runs tiny: the
nine-label pool is one complete six-step expert episode and the first three
steps of the next one, while the BC-iid stream keeps one state from each of
nine independent episodes, so chronological prefixes and iid prefixes differ.
"""

import datetime
import functools
import json
import pathlib

import numpy as np
import pytest
import torch as th

from imitation.algorithms import ftrl
from imitation.data import serialize
from imitation.experiments.agnostic import classical, restriction
from imitation.experiments.agnostic import restriction_pilot as rp
from imitation.experiments.agnostic import task_experts, task_pilot
from imitation.experiments.ftrl import env_utils, run_experiment

UTC = datetime.timezone.utc
FAR = datetime.datetime(2026, 10, 7, 1, 0, tzinfo=UTC)
LIMIT = 3600
ENV = "FrozenLake-v1"
MASK = "frozenlake_column_pairs"
# FrozenLake 4x4 deterministic: LEFT 0, DOWN 1, RIGHT 2, UP 3.
LAKE_PATH = {0: 1, 4: 1, 8: 2, 9: 1, 13: 2, 14: 2}
BUDGET = 9  # one 6-step expert episode and 3 steps of the next
EVAL_INTERVAL = 3
SEED = 300
# Keys of a generic three-method run config before BC-prefix existed.
ORIGINAL_CONFIG_KEYS = {
    "allow_relaxed_expert",
    "batch_rule",
    "beta",
    "distinct_observations",
    "eval_budgets",
    "eval_deterministic",
    "eval_episodes",
    "experiment",
    "fixed_bc",
    "method",
    "mixture",
    "normalization_reference",
    "policy",
    "reference",
    "restriction",
    "restriction_id",
    "restriction_sha256",
    "seed",
    "task_sha256",
}


class Clock:
    """Deterministic clock that advances a fixed step per reading."""

    def __init__(self, step_seconds=0.001):
        self.now = datetime.datetime(2026, 10, 1, tzinfo=UTC)
        self.step = datetime.timedelta(seconds=step_seconds)

    def __call__(self):
        self.now += self.step
        return self.now


def _tabular_expert(path):
    """Linear policy on the one-hot observation following ``LAKE_PATH``."""
    from stable_baselines3 import PPO

    venv = env_utils.make_env(ENV, 1, np.random.default_rng(0))
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
            weight[LAKE_PATH.get(s, 0), s] = 10.0
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    venv.close()


def _read(out):
    return json.loads((pathlib.Path(out) / rp.RESULT_FILE).read_text())


@pytest.fixture(scope="module")
def lake_expert(tmp_path_factory):
    candidate = tmp_path_factory.mktemp("cache") / ENV / "model.zip"
    _tabular_expert(candidate)
    out = tmp_path_factory.mktemp("lake") / "expert"
    code = task_experts.qualify_candidate(
        ENV,
        candidate,
        out,
        qualification_seed=1234,
        eval_episodes=3,
        deadline=FAR,
        clock=Clock(),
    )
    assert code == 0
    return out


@pytest.fixture(scope="module")
def lake_data(lake_expert, tmp_path_factory):
    out = tmp_path_factory.mktemp("lake-data") / "data"
    code = task_pilot.prepare_data(
        out,
        lake_expert,
        env_name=ENV,
        seed=SEED,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        budget=BUDGET,
        baseline_episodes=2,
        clock=Clock(),
    )
    assert code == classical.EXIT_COMPLETE
    return out


def _run(expert, data, out, method, restriction_id, **kwargs):
    params = dict(
        env_name=ENV,
        method=method,
        restriction_id=restriction_id,
        seed=SEED,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        n_rounds=BUDGET,
        eval_interval=EVAL_INTERVAL,
        clock=Clock(),
    )
    params.update(kwargs)
    return task_pilot.run_job(out, expert, data, **params)


@pytest.fixture(scope="module")
def runs(lake_expert, lake_data, tmp_path_factory):
    """Every method under identity and the mask, from one shared data job."""
    root = tmp_path_factory.mktemp("runs")
    out = {}
    for method in task_pilot.RUN_METHODS:
        for rid in ("identity", MASK):
            path = root / f"{method}-{rid}"
            code = _run(lake_expert, lake_data, path, method, rid)
            assert code == classical.EXIT_COMPLETE, (method, rid)
            out[method, rid] = (path, _read(path))
    return out


def _evals(record):
    return [r for r in record["records"] if r.get("d_eval_size") is not None]


def _round_demos(run_dir, record):
    """Labeled (obs, act) pairs of every scratch round, in round order."""
    demos = pathlib.Path(run_dir) / record["scratch_demos"]
    rounds = sorted(p for p in demos.iterdir() if p.name.startswith("round-"))
    pairs = []
    for round_dir in rounds:
        trajs = []
        for path in sorted(round_dir.glob("*.npz")):
            trajs.extend(serialize.load(path))
        pairs.append(
            (
                np.concatenate([t.obs[:-1] for t in trajs]),
                np.concatenate([t.acts for t in trajs]),
            ),
        )
    return pairs


# ---------------------------------------------------------------------------
# Data: chronological pool prefixes, nested, never the iid stream
# ---------------------------------------------------------------------------


def test_prefix_runs_refit_nested_chronological_pool_prefixes(runs, lake_data):
    data = _read(lake_data)
    pool = rp.ExpertData.load(lake_data / rp.POOL_FILE)
    stream = rp.ExpertData.load(lake_data / rp.STREAM_FILE)
    lengths = data["datasets"]["pool"]["stats"]["episode_lengths"]
    # The fixture separates the two acquisitions at every positive budget.
    for b in range(1, BUDGET + 1):
        assert pool.pairs_sha256(b) != stream.pairs_sha256(b), b

    for rid in ("identity", MASK):
        out, record = runs["bc_prefix", rid]
        assert record["status"] == "complete"
        assert record["protocol"] == task_pilot.PREFIX_PROTOCOL
        assert record["schema"] == task_pilot.PREFIX_PROTOCOL + "/run"
        assert record["method"] == "bc_prefix" and record["restriction_id"] == rid
        assert record["data"]["acceptance"]["producer_protocol"] == task_pilot.PROTOCOL
        assert record["config"]["bc_prefix"] == rp.BC_PREFIX_RULE
        assert record["config"]["eval_budgets"] == [0, 1, 3, 6, 9]

        evals = _evals(record)
        assert [r["n_observations"] for r in evals] == [0, 1, 3, 6, 9]
        iid = {r["n_observations"]: r for r in _evals(runs["bc_iid", rid][1])}
        for r in evals:
            b = r["n_observations"]
            if b == 0:
                continue
            assert r["prefix_pairs_sha256"] == pool.pairs_sha256(b)
            assert r["prefix_pairs_sha256"] != iid[b]["prefix_pairs_sha256"]
            assert r["logical"] == rp.pool_logical_cost(lengths, b)
            policy = restriction.load_policy_checkpoint(out / r["checkpoint"])
            assert policy.restriction_id == rid

        # Round t holds exactly pool transition t: the replay is chronological.
        arrivals = _round_demos(out, record)
        assert len(arrivals) == BUDGET
        for t, (obs, acts) in enumerate(arrivals):
            np.testing.assert_array_equal(obs, pool.obs[t : t + 1])
            np.testing.assert_array_equal(acts, pool.acts[t : t + 1])

        # Nesting: the full prefix is exactly fixed BC's one full-pool dataset.
        bc = runs["bc", rid][1]
        assert record["trained_pairs_sha256"] == pool.pairs_sha256()
        assert record["trained_pairs_sha256"] == bc["trained_pairs_sha256"]
        shared = record["accounting"]["shared_acquisition"]
        assert shared["dataset"] == "pool" and shared["charged_to_this_run"] is False
        assert shared["pairs_sha256"] == data["datasets"]["pool"]["pairs_sha256"]

    hashes = {
        rid: [
            r.get("prefix_pairs_sha256") for r in runs["bc_prefix", rid][1]["records"]
        ]
        for rid in ("identity", MASK)
    }
    assert hashes["identity"] == hashes[MASK]


def test_prefix_config_differs_from_bc_iid_only_by_method_and_sampling(runs):
    for rid in ("identity", MASK):
        prefix = runs["bc_prefix", rid][1]["config"]
        iid = runs["bc_iid", rid][1]["config"]
        assert set(prefix) == ORIGINAL_CONFIG_KEYS | {"bc_prefix"}
        changed = sorted(
            k for k in set(prefix) | set(iid) if prefix.get(k) != iid.get(k)
        )
        assert changed == ["bc_prefix", "beta", "experiment", "method"]
        assert prefix["beta"] is None and prefix["fixed_bc"] is None
        exp_changed = sorted(
            k
            for k in prefix["experiment"]
            if prefix["experiment"][k] != iid["experiment"][k]
        )
        assert exp_changed == ["algo", "subsample_strategy"]
        assert prefix["experiment"]["algo"] == "bc_prefix"
        assert prefix["experiment"]["subsample_strategy"] == "prefix"
        for key in ("warm_start", "bc_n_epochs", "learning_rate", "inner_early_stop"):
            assert prefix["experiment"][key] == iid["experiment"][key]
        assert prefix["experiment"]["warm_start"] is False
        assert prefix["experiment"]["bc_n_epochs"] == rp.BC_EPOCHS


# ---------------------------------------------------------------------------
# Training: cold refits of the original routine
# ---------------------------------------------------------------------------


def test_prefix_fits_are_cold_refits_of_the_original_routine(
    lake_expert,
    lake_data,
    tmp_path,
    monkeypatch,
):
    calls = []
    original = ftrl.FTRLTrainer._reinitialize_trainable_params

    def spy(self):
        calls.append(self.round_num)
        return original(self)

    monkeypatch.setattr(ftrl.FTRLTrainer, "_reinitialize_trainable_params", spy)
    out = tmp_path / "prefix"
    assert _run(lake_expert, lake_data, out, "bc_prefix", MASK) == 0
    record = _read(out)
    # Every round after the first restarts from a fresh head (round 1 is
    # already a fresh head): one cold fit per round, as for BC-iid.
    assert calls == list(range(1, BUDGET))
    done = record["accounting"]["completed_records"]
    assert done["training"]["fits"] == BUDGET
    monkeypatch.setattr(ftrl.FTRLTrainer, "_reinitialize_trainable_params", original)

    # The same arguments through the unchanged original routine reproduce the
    # job exactly, round by round and weight by weight.
    expert, _ = task_experts.load_qualified_expert(lake_expert, ENV)
    pool = rp.ExpertData.load(lake_data / rp.POOL_FILE)
    baselines = _read(lake_data)["baselines"]
    config = rp._experiment_config(
        "bc_prefix",
        SEED,
        BUDGET,
        EVAL_INTERVAL,
        tmp_path / "standalone",
        ENV,
    )
    assert config.algo in run_experiment.OFFLINE_BC_ALGOS
    run_experiment._seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    venv = rp.CountingVecEnv(env_utils.make_env(ENV, 1, rng))
    th.manual_seed(SEED)
    try:
        rows = run_experiment._run_dagger_variant(
            config,
            venv,
            expert.policy,
            rng,
            baselines,
            device="cpu",
            policy_factory=functools.partial(
                restriction.make_linear_policy,
                restriction_id=MASK,
            ),
            offline_data=pool.transitions(BUDGET),
        )
    finally:
        venv.close()
    assert len(rows) == len(record["records"]) == BUDGET + 1
    for mine, theirs in zip(record["records"], rows):
        for key in (
            "n_observations",
            "train_cross_entropy",
            "normalized_return",
            "disagreement_rate",
            "episode_returns",
            "inner_es_stop_epoch",
        ):
            assert mine.get(key) == theirs.get(key), key
        if theirs.get("checkpoint"):
            a = restriction.load_policy_checkpoint(out / mine["checkpoint"])
            b = restriction.load_policy_checkpoint(theirs["checkpoint"])
            sa, sb = a.state_dict(), b.state_dict()
            assert sa.keys() == sb.keys()
            for name in sa:
                assert th.equal(sa[name], sb[name]), name


def test_prefix_makes_no_expert_label_queries(
    lake_expert,
    lake_data,
    tmp_path,
    monkeypatch,
):
    data_sha = rp.pilot.sha256_file(lake_data / rp.RESULT_FILE)
    counts = {}
    load = rp._load_expert

    def counting_load(preparation_dir, spec=rp.CARTPOLE_V2):
        policy, identity = load(preparation_dir, spec)
        predict = policy.predict

        def counted(*args, **kwargs):
            counts["predict"] = counts.get("predict", 0) + 1
            return predict(*args, **kwargs)

        monkeypatch.setattr(policy, "predict", counted)
        return policy, identity

    monkeypatch.setattr(rp, "_load_expert", counting_load)
    totals = {}
    for method in ("bc_prefix", "bc_iid"):
        counts.clear()
        out = tmp_path / method
        assert _run(lake_expert, lake_data, out, method, MASK) == 0
        acct = _read(out)["accounting"]
        done = acct["completed_records"]
        # Only evaluation queries the expert: one predict per evaluation step.
        assert done["collection"]["env_steps"] == 0
        assert done["collection"]["expert_predict_calls"] == 0
        assert acct["totals"]["exact"] is True
        assert acct["observed"]["env_steps"] == done["evaluation"]["env_steps"]
        assert acct["totals"]["expert_predict_calls"] == (
            done["evaluation"]["expert_predict_calls"]
        )
        assert counts["predict"] == done["evaluation"]["env_steps"]
        totals[method] = counts["predict"]
    # The data job is read, never extended or rewritten.
    assert rp.pilot.sha256_file(lake_data / rp.RESULT_FILE) == data_sha
    assert totals["bc_prefix"] > 0


# ---------------------------------------------------------------------------
# Opt-in only: the three original methods are unchanged
# ---------------------------------------------------------------------------


def test_original_methods_keep_their_protocol_spec_and_configs(runs):
    spec = task_pilot.runner_spec(ENV)
    assert spec == task_pilot.runner_spec(ENV, bc_prefix=False)
    assert spec.protocol == task_pilot.PROTOCOL and spec.methods == rp.METHODS
    assert rp.CARTPOLE_V2.methods == rp.METHODS
    assert rp.PREFIX_METHOD not in rp.METHODS
    prefix = task_pilot.runner_spec(ENV, bc_prefix=True)
    assert prefix.methods == (rp.PREFIX_METHOD,)
    assert prefix.data_protocol == spec.data_protocol == task_pilot.PROTOCOL
    assert prefix.source_files == spec.source_files
    assert prefix.run_restrictions == spec.run_restrictions
    for method in rp.METHODS:
        for rid in ("identity", MASK):
            record = runs[method, rid][1]
            assert record["protocol"] == task_pilot.PROTOCOL
            assert record["schema"] == task_pilot.PROTOCOL + "/run"
            assert set(record["config"]) == ORIGINAL_CONFIG_KEYS
            assert record["config"]["experiment"]["algo"] == method


def test_bc_prefix_needs_the_opt_in_spec(lake_expert, lake_data, tmp_path):
    common = dict(
        method="bc_prefix",
        restriction_id="identity",
        seed=SEED,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        n_rounds=BUDGET,
        eval_interval=EVAL_INTERVAL,
        clock=Clock(),
    )
    plain = tmp_path / "plain"
    code = rp.run_job(
        plain,
        lake_expert,
        lake_data,
        spec=task_pilot.runner_spec(ENV),
        **common,
    )
    assert code == classical.EXIT_REFUSED and not plain.exists()
    pilot = tmp_path / "pilot"
    assert rp.run_job(pilot, lake_expert, lake_data, **common) == classical.EXIT_REFUSED
    assert not pilot.exists()
    # The opt-in spec runs BC-prefix only.
    other = tmp_path / "other"
    code = rp.run_job(
        other,
        lake_expert,
        lake_data,
        spec=task_pilot.runner_spec(ENV, bc_prefix=True),
        **dict(common, method="bc_iid"),
    )
    assert code == classical.EXIT_REFUSED and not other.exists()

    with pytest.raises(SystemExit):
        rp.build_parser().parse_args(
            [
                "run",
                "--preparation-dir",
                "p",
                "--output-dir",
                "o",
                "--deadline",
                "2026-10-07T00:00:00Z",
                "--job-limit-seconds",
                "1",
                "--data-dir",
                "d",
                "--seed",
                "300",
                "--method",
                "bc_prefix",
                "--restriction",
                "identity",
            ],
        )


@pytest.mark.parametrize("method", task_pilot.RUN_METHODS)
def test_cli_dispatches_each_method_to_its_spec(tmp_path, monkeypatch, capsys, method):
    seen = {}

    def fake_run_job(output_dir, expert_dir, data_dir, *, spec, **kwargs):
        seen.update(spec=spec, kwargs=kwargs)
        pathlib.Path(output_dir).mkdir()
        (pathlib.Path(output_dir) / rp.RESULT_FILE).write_text('{"status": "complete"}')
        return 0

    monkeypatch.setattr(rp, "run_job", fake_run_job)
    out = tmp_path / "cli"
    argv = [
        "run",
        "--env",
        ENV,
        "--expert-dir",
        "e",
        "--output-dir",
        str(out),
        "--deadline",
        "2026-10-07T00:00:00Z",
        "--job-limit-seconds",
        "60",
        "--data-dir",
        "d",
        "--seed",
        str(SEED),
        "--method",
        method,
        "--restriction",
        "identity",
    ]
    assert task_pilot.main(argv) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == {
        "status": "complete",
        "exit_code": 0,
    }
    prefix = method == rp.PREFIX_METHOD
    assert seen["kwargs"]["method"] == method
    assert seen["spec"] == task_pilot.runner_spec(ENV, bc_prefix=prefix)
    assert seen["spec"].protocol == (
        task_pilot.PREFIX_PROTOCOL if prefix else task_pilot.PROTOCOL
    )
