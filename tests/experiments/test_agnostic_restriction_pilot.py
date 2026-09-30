"""CartPole restriction pilot: data acquisition, sharing, accounting, safety.

Data contracts are checked against independent replays on the real CartPole
simulator. Runs use a verified synthetic preparation and tiny budgets; the
approved pilot settings are exercised only through their recorded config.
"""

import datetime
import inspect
import json
import pathlib
import shutil
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO

from imitation.data import serialize
from imitation.experiments.agnostic import classical, restriction
from imitation.experiments.agnostic import restriction_pilot as rp
from imitation.experiments.agnostic import run_classical
from imitation.experiments.ftrl import env_utils

UTC = datetime.timezone.utc
FAR = datetime.datetime(2026, 10, 7, 1, 0, tzinfo=UTC)
REPO_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


class Clock:
    """Deterministic clock that advances a fixed step per reading."""

    def __init__(self, step_seconds=0.001):
        self.now = datetime.datetime(2026, 10, 1, tzinfo=UTC)
        self.step = datetime.timedelta(seconds=step_seconds)

    def __call__(self):
        self.now += self.step
        return self.now


@pytest.fixture(scope="module")
def weak_expert():
    """Random [64, 64] CartPole policy: short episodes of varying length."""
    venv = env_utils.make_env("CartPole-v1", 1, np.random.default_rng(0))
    policy = PPO("MlpPolicy", venv, device="cpu", seed=6).policy
    venv.close()
    return policy


def _replay(expert, reset_seed):
    """Independently replay one deterministic expert episode on raw gym."""
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=int(reset_seed))
    rows = []
    while True:
        act = int(expert.predict(obs[None], deterministic=True)[0][0])
        nxt, _, terminated, truncated, _ = env.step(act)
        rows.append((obs, act, nxt, terminated or truncated))
        if terminated or truncated:
            return rows
        obs = nxt


# ---------------------------------------------------------------------------
# Data acquisition contracts
# ---------------------------------------------------------------------------


def test_pool_is_chronological_complete_episodes_with_recorded_overshoot(
    weak_expert,
):
    budget = 60
    pool, stats = rp.collect_chronological_pool(weak_expert, budget, seed=300)

    lengths = stats["episode_lengths"]
    assert stats["episodes"] == len(lengths) >= 3
    assert sum(lengths[:-1]) < budget <= sum(lengths)
    assert stats["env_steps"] == sum(lengths)
    assert stats["overshoot_transitions"] == sum(lengths) - budget > 0
    assert stats["final_episode_length"] == lengths[-1]
    assert stats["expert_predict_calls"] == stats["expert_action_entries"]
    assert stats["expert_predict_calls"] == stats["env_steps"]
    assert stats["retained_labels"] == len(pool) == budget

    replayed = []
    for episode, reset_seed in enumerate(stats["episode_reset_seeds"]):
        rows = _replay(weak_expert, reset_seed)
        assert len(rows) == lengths[episode]
        replayed += [(episode, step, *row) for step, row in enumerate(rows)]
    for i, (episode, step, obs, act, nxt, done) in enumerate(replayed[:budget]):
        assert pool.episode_index[i] == episode and pool.step_index[i] == step
        np.testing.assert_array_equal(pool.obs[i], obs)
        np.testing.assert_array_equal(pool.next_obs[i], nxt)
        assert pool.acts[i] == act and bool(pool.dones[i]) == done


def test_iid_stream_keeps_one_uniform_state_per_independent_episode(weak_expert):
    n = 40
    stream, stats = rp.collect_iid_stream(weak_expert, n, seed=300)
    _, pool_stats = rp.collect_chronological_pool(weak_expert, 60, seed=300)

    assert len(stream) == stats["episodes"] == stats["retained_labels"] == n
    np.testing.assert_array_equal(stream.episode_index, np.arange(n))
    seeds = stats["episode_reset_seeds"]
    assert len(set(seeds)) == n
    assert not set(seeds) & set(pool_stats["episode_reset_seeds"])
    assert stats["env_steps"] == sum(stats["episode_lengths"])
    selected = stream.step_index
    assert len(set(selected.tolist())) > 1
    for i, reset_seed in enumerate(seeds):
        rows = _replay(weak_expert, reset_seed)
        assert len(rows) == stats["episode_lengths"][i]
        assert 0 <= selected[i] < len(rows)
        obs, act, nxt, done = rows[selected[i]]
        np.testing.assert_array_equal(stream.obs[i], obs)
        np.testing.assert_array_equal(stream.next_obs[i], nxt)
        assert stream.acts[i] == act and bool(stream.dones[i]) == done


def test_datasets_are_reproducible_and_prefix_hashes_are_exact(
    weak_expert,
    tmp_path,
):
    pool, _ = rp.collect_chronological_pool(weak_expert, 60, seed=300)
    again, _ = rp.collect_chronological_pool(weak_expert, 60, seed=300)
    other, _ = rp.collect_chronological_pool(weak_expert, 60, seed=299)

    assert pool.pairs_sha256() == again.pairs_sha256()
    assert pool.pairs_sha256() != other.pairs_sha256()
    for b in (1, 10, 37, 60):
        assert pool.pairs_sha256(b) == rp.pairs_sha256(pool.obs[:b], pool.acts[:b])
        prefix = pool.transitions(b)
        np.testing.assert_array_equal(prefix.obs, pool.obs[:b])
        np.testing.assert_array_equal(prefix.acts, pool.acts[:b])
    assert pool.pairs_sha256(10) != pool.pairs_sha256(11)
    path = tmp_path / "pool.npz"
    pool.save(path)
    loaded = rp.ExpertData.load(path)
    assert loaded.pairs_sha256() == pool.pairs_sha256()
    np.testing.assert_array_equal(loaded.episode_index, pool.episode_index)


# ---------------------------------------------------------------------------
# Jobs with a verified synthetic preparation
# ---------------------------------------------------------------------------


def _save_expert(path):
    """Linear expert that balances CartPole and uses the cart position."""
    env = env_utils.make_env("CartPole-v1", n_envs=1, rng=np.random.default_rng(0))
    model = PPO(
        "MlpPolicy",
        env,
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
    env.close()


@pytest.fixture(scope="module")
def prep(tmp_path_factory):
    out = tmp_path_factory.mktemp("prep") / "prep"

    def trainer(env_name, cache_dir, rng, seed, convergence_override=None):
        _save_expert(pathlib.Path(cache_dir) / env_name / "model.zip")

    code = run_classical.prepare_expert(
        "CartPole-v1",
        out,
        seed=0,
        qualification_seed=1234,
        eval_episodes=2,
        deadline=FAR,
        trainer=trainer,
    )
    assert code == 0
    return out


BUDGET = 6


@pytest.fixture(scope="module")
def data_dir(prep, tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "data"
    code = rp.prepare_data(
        out,
        prep,
        seed=300,
        deadline=FAR,
        job_limit_seconds=3600,
        budget=BUDGET,
        baseline_episodes=2,
        clock=Clock(),
    )
    assert code == classical.EXIT_COMPLETE
    return out


def _read(out):
    return json.loads((out / rp.RESULT_FILE).read_text())


def _run(prep, data_dir, out, method, restriction_id, **kwargs):
    params = dict(
        seed=300,
        deadline=FAR,
        job_limit_seconds=3600,
        n_rounds=BUDGET,
        eval_interval=3,
        clock=Clock(),
    )
    params.update(kwargs)
    return rp.run_job(
        out,
        prep,
        data_dir,
        method=method,
        restriction_id=restriction_id,
        **params,
    )


def test_data_job_records_shared_datasets_baselines_and_physical_costs(
    prep,
    data_dir,
):
    record = _read(data_dir)
    prep_record = json.loads((prep / "preparation.json").read_text())

    assert record["status"] == "complete"
    assert record["protocol"] == rp.PROTOCOL
    assert record["env_name"] == "CartPole-v1" and record["seed"] == 300
    assert record["episode_cap"] == 500
    assert record["expert_sha256"] == prep_record["expert_sha256"]
    for name, file_name in (("pool", rp.POOL_FILE), ("stream", rp.STREAM_FILE)):
        entry = record["datasets"][name]
        data = rp.ExpertData.load(data_dir / file_name)
        assert entry["file"] == file_name
        assert entry["pairs_sha256"] == data.pairs_sha256()
        assert entry["file_sha256"] == rp.pilot.sha256_file(data_dir / file_name)
        assert entry["stats"]["retained_labels"] == len(data) == BUDGET
        # Stored demos keep the full state, cart position included.
        assert np.any(data.obs[:, 0] != 0)
    pool_stats = record["datasets"]["pool"]["stats"]
    assert pool_stats["overshoot_transitions"] == pool_stats["env_steps"] - BUDGET
    baselines = record["baselines"]
    assert set(baselines) >= {"expert_return", "random_return", "expert_self_ce"}
    costs = record["baselines_costs"]
    assert costs["expert_episodes"] == costs["random_episodes"] == 2
    assert costs["env_steps_counted"] == (
        costs["expert_env_steps"] + costs["random_env_steps"]
    )


@pytest.mark.parametrize("method", ["bc", "bc_iid"])
def test_full_and_restricted_runs_train_on_the_same_exact_data(
    prep,
    data_dir,
    tmp_path,
    method,
):
    data = _read(data_dir)
    name = "pool" if method == "bc" else "stream"
    shared = rp.ExpertData.load(data_dir / data["datasets"][name]["file"])
    records = {}
    for rid in ("identity", "cart_position_zero"):
        out = tmp_path / rid
        assert _run(prep, data_dir, out, method, rid) == classical.EXIT_COMPLETE
        records[rid] = _read(out)
        record = records[rid]
        assert record["status"] == "complete"
        assert record["method"] == method and record["restriction_id"] == rid
        assert record["data"]["data_result_sha256"] == rp.pilot.sha256_file(
            data_dir / rp.RESULT_FILE,
        )
        evals = [r for r in record["records"] if r.get("normalized_return") is not None]
        budgets = [r["n_observations"] for r in evals]
        if method == "bc":
            assert budgets == [1, 3, 6]
        else:
            assert budgets == [0, 1, 3, 6]
        for r in evals:
            b = r["n_observations"]
            if b:
                assert r["prefix_pairs_sha256"] == shared.pairs_sha256(b)
            assert len(r["episode_returns"]) == 100
            policy = restriction.load_policy_checkpoint(out / r["checkpoint"])
            assert policy.restriction_id == rid
        acct = record["accounting"]
        done = acct["completed_records"]
        assert done["retained_labels"] == BUDGET
        assert done["collection"]["env_steps"] == 0
        assert acct["observed"]["env_steps"] == done["evaluation"]["env_steps"]
        assert acct["in_flight"]["env_steps"] == 0
        assert acct["totals"]["exact"] is True
        assert acct["totals"]["expert_predict_calls"] == (
            done["evaluation"]["expert_predict_calls"]
        )
        assert record["snapshot"]["final"] is True
        assert acct["shared_acquisition"]["pairs_sha256"] == shared.pairs_sha256()
        assert acct["shared_acquisition"]["charged_to_this_run"] is False
    if method == "bc":
        # Fixed BC cold-fits every plotted prefix with the unchanged routine.
        for r in records["identity"]["records"]:
            assert r["logical"]["labels"] == r["n_observations"]
            assert r["logical"]["env_steps"] >= r["n_observations"]
        done = records["identity"]["accounting"]["completed_records"]
        assert done["training"]["fits"] == 3
    else:
        for rid, record in records.items():
            assert record["trained_pairs_sha256"] == shared.pairs_sha256(BUDGET)
            done = record["accounting"]["completed_records"]
            assert done["training"]["fits"] == BUDGET
    hashes = {
        rid: [r.get("prefix_pairs_sha256") for r in rec["records"]]
        for rid, rec in records.items()
    }
    assert hashes["identity"] == hashes["cart_position_zero"]


def test_fixed_bc_points_are_cold_fits_independent_of_earlier_budgets(
    prep,
    data_dir,
    tmp_path,
):
    # Budgets [1, 3] versus [1, 2, 3]: a warm start or a wrong prefix would
    # make the policy at B = 3 depend on the earlier fits.
    weights = []
    for interval in (3, 1):
        out = tmp_path / f"interval{interval}"
        code = _run(
            prep,
            data_dir,
            out,
            "bc",
            "cart_position_zero",
            n_rounds=3,
            eval_interval=interval,
        )
        assert code == classical.EXIT_COMPLETE
        [last] = [r for r in _read(out)["records"] if r["n_observations"] == 3]
        weights.append(restriction.load_policy_checkpoint(out / last["checkpoint"]))
    first, second = (w.state_dict() for w in weights)
    assert first.keys() == second.keys()
    for name in first:
        assert th.equal(first[name], second[name]), name


def test_restricted_ftl_collects_learner_states_with_full_state_expert_labels(
    prep,
    data_dir,
    tmp_path,
):
    out = tmp_path / "ftl"
    code = _run(prep, data_dir, out, "ftl", "cart_position_zero", n_rounds=4)
    assert code == classical.EXIT_COMPLETE
    record = _read(out)
    expert, _ = classical.load_verified_expert(prep, "CartPole-v1")

    demos = []
    for round_dir in sorted((out / record["scratch_demos"]).iterdir()):
        for path in sorted(round_dir.iterdir()):
            demos.extend(serialize.load(path))
    obs = np.concatenate([d.obs[:-1] for d in demos])
    acts = np.concatenate([d.acts for d in demos])
    assert len(acts) == 4
    assert np.all(obs[:, 0] != 0)
    labels, _ = expert.policy.predict(obs, deterministic=True)
    np.testing.assert_array_equal(acts, labels)
    assert record["trained_pairs_sha256"] == rp.pairs_sha256(obs, acts)

    acct = record["accounting"]
    done = acct["completed_records"]
    collection = done["collection"]
    last = record["records"][-1]
    assert collection["env_steps"] == last["collection_steps"] > 0
    assert collection["expert_predict_calls"] == collection["env_steps"]
    assert collection["learner_predict_calls"] == collection["env_steps"]
    assert acct["observed"]["env_steps"] == (
        collection["env_steps"] + done["evaluation"]["env_steps"]
    )
    assert acct["reconciled"] is True
    assert done["training"]["fits"] == 4
    assert acct["totals"] == {
        "exact": True,
        "env_steps": acct["observed"]["env_steps"],
        "expert_predict_calls": acct["observed"]["env_steps"],
        "expert_action_entries": acct["observed"]["env_steps"],
        "learner_predict_calls": acct["observed"]["env_steps"],
    }
    assert acct["shared_acquisition"] is None
    assert record["config"]["experiment"]["warm_start"] is False
    assert record["config"]["experiment"]["outer_early_stop"] is False
    assert record["config"]["experiment"]["inner_early_stop"] is True
    assert record["config"]["mixture"] is False


def test_refusals_leave_existing_output_and_data_untouched(prep, data_dir, tmp_path):
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "keep.txt").write_text("old")
    code = _run(prep, data_dir, busy, "bc", "identity")
    assert code == classical.EXIT_REFUSED
    assert sorted(p.name for p in busy.iterdir()) == ["keep.txt"]

    assert _run(prep, data_dir, tmp_path / "s", "bc", "identity", seed=299) == (
        classical.EXIT_REFUSED
    )
    assert not (tmp_path / "s").exists()
    assert _run(prep, data_dir, tmp_path / "r", "bc", "bogus") == classical.EXIT_REFUSED

    tampered = tmp_path / "tampered"
    shutil.copytree(data_dir, tampered)
    pool = rp.ExpertData.load(tampered / rp.POOL_FILE)
    pool.acts[0] = 1 - pool.acts[0]
    (tampered / rp.POOL_FILE).unlink()
    pool.save(tampered / rp.POOL_FILE)
    assert _run(prep, tampered, tmp_path / "t", "bc", "identity") == (
        classical.EXIT_REFUSED
    )
    assert not (tmp_path / "t").exists()


def test_deadline_mid_run_keeps_a_partial_record(prep, data_dir, tmp_path):
    clock = Clock(step_seconds=1.0)
    start = clock.now
    out = tmp_path / "late"
    code = _run(
        prep,
        data_dir,
        out,
        "bc_iid",
        "cart_position_zero",
        clock=clock,
        deadline=start + datetime.timedelta(seconds=6),
    )
    record = _read(out)

    assert code == classical.EXIT_INCOMPLETE
    assert record["status"] == "partial"
    assert record["error"]["type"] == "DeadlineExceeded"
    assert 0 < len(record["records"]) < BUDGET + 1
    acct = record["accounting"]
    assert acct["completed_records"]["evaluation"]["env_steps"] > 0
    # Stopped at a round boundary: nothing is in flight, yet the job did not
    # complete, so query totals stay unknown rather than claimed exact.
    assert acct["in_flight"]["env_steps"] == 0
    assert acct["totals"]["exact"] is False


def _fail_on_step(monkeypatch, should_fail):
    """Make the counting wrapper raise on the step selected by ``should_fail``.

    The failing step never returns, so it is not counted.
    """
    original = rp.CountingVecEnv.step_wait

    def step_wait(self):
        if should_fail(self):
            raise RuntimeError("injected failure")
        return original(self)

    monkeypatch.setattr(rp.CountingVecEnv, "step_wait", step_wait)


def _assert_unknown_in_flight(record, env_steps, operation):
    acct = record["accounting"]
    assert record["status"] == "failed"
    assert record["snapshot"]["final"] is True
    assert record["interruption"]["in_flight_operation"] == operation
    assert acct["observed"]["env_steps"] == acct["totals"]["env_steps"]
    assert acct["in_flight"]["operation"] == operation
    assert acct["in_flight"]["env_steps"] == env_steps
    for key in ("expert_predict_calls", "learner_predict_calls"):
        assert acct["in_flight"][key] is None
        assert acct["totals"][key] is None
    assert acct["totals"]["expert_action_entries"] is None
    assert acct["totals"]["exact"] is False
    assert acct["reconciled"] is (env_steps == 0)


def test_failure_inside_evaluation_keeps_observed_steps_and_unknown_queries(
    prep,
    data_dir,
    tmp_path,
    monkeypatch,
):
    # Round 0 evaluates before anything else, so step 11 is inside it.
    _fail_on_step(monkeypatch, lambda venv: venv.steps == 10)
    out = tmp_path / "eval-failure"

    assert _run(prep, data_dir, out, "ftl", "identity") == classical.EXIT_INCOMPLETE
    record = _read(out)

    assert record["records"] == []
    done = record["accounting"]["completed_records"]
    assert done["evaluation"]["env_steps"] == 0
    assert record["accounting"]["observed"]["env_steps"] == 10
    _assert_unknown_in_flight(record, 10, "round 0 evaluation")


def test_failure_inside_collection_keeps_observed_steps_and_unknown_queries(
    prep,
    data_dir,
    tmp_path,
    monkeypatch,
):
    collected = []

    def inside_collection(venv):
        frames = {f.function for f in inspect.stack()}
        if "generate_trajectories" in frames:
            collected.append(1)
            return len(collected) == 3
        return False

    _fail_on_step(monkeypatch, inside_collection)
    out = tmp_path / "collection-failure"

    assert _run(prep, data_dir, out, "ftl", "identity") == classical.EXIT_INCOMPLETE
    record = _read(out)

    [round0] = record["records"]
    done = record["accounting"]["completed_records"]
    assert done["evaluation"]["env_steps"] == round0["d_eval_size"]
    assert record["accounting"]["observed"]["env_steps"] == round0["d_eval_size"] + 2
    _assert_unknown_in_flight(
        record,
        2,
        "round 1: collection, training and evaluation",
    )


def test_failure_inside_baselines_keeps_datasets_and_observed_steps(
    prep,
    tmp_path,
    monkeypatch,
):
    # Only the baseline measurement uses the counting wrapper in the data job.
    _fail_on_step(monkeypatch, lambda venv: venv.steps == 5)
    out = tmp_path / "data-failure"

    code = rp.prepare_data(
        out,
        prep,
        seed=300,
        deadline=FAR,
        job_limit_seconds=3600,
        budget=BUDGET,
        baseline_episodes=2,
        clock=Clock(),
    )
    record = _read(out)

    assert code == classical.EXIT_INCOMPLETE
    assert record["status"] == "failed"
    assert record["interruption"]["in_flight_operation"] == "baseline measurement"
    assert set(record["datasets"]) == {"pool", "stream"}
    assert record["baselines"] is None and record["baselines_costs"] is None
    assert record["baselines_observed"] == {
        "env_steps": 5,
        "env_resets": record["baselines_observed"]["env_resets"],
        "expert_predict_calls": None,
    }
    assert record["baselines_observed"]["env_resets"] >= 1


def test_job_limit_bounds_the_effective_deadline(prep, data_dir, tmp_path):
    clock = Clock(step_seconds=1.0)
    out = tmp_path / "limited"
    code = _run(prep, data_dir, out, "bc", "identity", clock=clock, job_limit_seconds=5)
    record = _read(out)

    assert code == classical.EXIT_INCOMPLETE
    assert record["status"] == "partial"
    limits = record["deadline"]
    assert limits["job_limit_seconds"] == 5
    assert limits["effective_utc"] < limits["requested_utc"]
    assert limits["hard_cap_utc"] == classical.HARD_CAP.isoformat()


def test_audit_job_keeps_every_pair_and_uses_no_training_data(prep, tmp_path):
    out = tmp_path / "audit"
    code = rp.run_audit(
        out,
        prep,
        deadline=FAR,
        job_limit_seconds=600,
        clock=Clock(),
    )
    record = _read(out)

    assert code == classical.EXIT_COMPLETE
    assert record["status"] == "complete"
    audit = record["audit"]
    assert audit["reset_seed"] == rp.AUDIT_SEED == 301
    assert audit["status"] in ("conflict_found", "restriction_uncertified")
    assert audit["labels_used_for_training"] is False
    assert len(audit["pairs"]) == audit["n_checked_pairs"] > 0
    assert record["expert_sha256"] == record["preparation"]["expert_sha256"]


_CLI_BOOTSTRAP = """
import sys
from imitation.experiments.agnostic import restriction_pilot
sys.exit(restriction_pilot.main(sys.argv[1:]))
"""


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-c", _CLI_BOOTSTRAP, *args],
        env={"PYTHONPATH": str(REPO_SRC), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_cli_audit_and_refusal_of_the_excluded_audit_seed(prep, data_dir, tmp_path):
    common = [
        "--preparation-dir",
        str(prep),
        "--deadline",
        FAR.isoformat(),
        "--job-limit-seconds",
        "600",
    ]
    audit = _cli("audit", *common, "--output-dir", str(tmp_path / "audit"))
    assert audit.returncode == 0, audit.stderr
    assert json.loads(audit.stdout.strip().splitlines()[-1])["status"] == "complete"

    refused = _cli(
        "run",
        *common,
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(tmp_path / "run"),
        "--method",
        "ftl",
        "--restriction",
        "cart_position_zero",
        "--seed",
        "301",
    )
    assert refused.returncode == classical.EXIT_USAGE, refused.stderr
    assert not (tmp_path / "run").exists()
