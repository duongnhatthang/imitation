"""Stage 2 tests for the classical alias audit and learning cells.

The expert fixture is synthetic: a hand-set linear PPO checkpoint passed
through the real stage 1 preparation (only the PPO trainer is replaced). It
supports no scientific claim. A small fake env with CartPole-shaped
observations makes costs and seeds easy to check; the smoke test at the end
runs the real CLI on real CartPole.
"""

import datetime
import json
import pathlib
import shutil
import subprocess
import sys
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO

from imitation.experiments.agnostic import classical, pilot, quantized, run_classical
from imitation.experiments.ftrl import env_utils

UTC = datetime.timezone.utc
START = datetime.datetime(2026, 1, 1, tzinfo=UTC)
DEADLINE = datetime.datetime(2026, 6, 1, tzinfo=UTC)
REPO_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
FAKE_ID = "AgnosticFakeCart-v0"


def _expert_rule(obs):
    # Depends on a velocity that the severe CartPole quantizer ignores.
    return int(obs[3] > 0)


class FakeCartEnv(gym.Env):
    """CartPole-shaped observations; a non-expert action may end the episode."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def _draw(self):
        scale = np.array([0.6, 1.0, 0.08, 1.0])
        self.obs = (self.np_random.uniform(-1, 1, 4) * scale).astype(np.float32)
        return self.obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._draw(), {}

    def step(self, action):
        wrong = int(action) != _expert_rule(self.obs)
        terminated = wrong and self.np_random.random() < 0.3
        return self._draw(), 1.0, terminated, False, {}


if FAKE_ID not in gym.registry:
    gym.register(id=FAKE_ID, entry_point=FakeCartEnv, max_episode_steps=10)


def fake_env():
    return gym.make(FAKE_ID)


class CountingExpert:
    def __init__(self, fn=_expert_rule):
        self.fn = fn
        self.calls = 0
        self.entries = 0

    def predict(self, obs, deterministic=True):
        assert deterministic
        self.calls += 1
        self.entries += len(obs)
        return np.array([self.fn(o) for o in obs]), None


class Clock:
    """Advances one second per call; ``late_from`` jumps past the deadline."""

    def __init__(self, late_from=None):
        self.calls = 0
        self.late_from = late_from

    def __call__(self):
        self.calls += 1
        if self.late_from is not None and self.calls >= self.late_from:
            return DEADLINE
        return START + datetime.timedelta(seconds=self.calls)


def _cell(expert, **cfg):
    params = dict(seed=0, budget=64, batch=16, checkpoints=[32, 64], eval_episodes=5)
    params.update(cfg)
    return classical.CellRun(
        params,
        expert=classical.Expert(expert, 2),
        quantizer=quantized.get_quantizer("CartPole-v1", "severe"),
        guard=classical.Guard(DEADLINE, Clock()),
        env_factory=fake_env,
    )


# ---------------------------------------------------------------------------
# Cell semantics on the fake env
# ---------------------------------------------------------------------------


def test_oracle_calls_are_deferred_for_ftl_and_cached_for_bc():
    expert = CountingExpert()
    run = _cell(expert)
    run.run(lambda force: None)
    ftl, bc, ev = (run.costs[k] for k in ("train_ftl", "train_bc_iid", "eval"))
    # FTL asks the expert only once per episode, at the selected state.
    assert ftl.expert_predict_calls == ftl.expert_action_entries == 64
    assert ftl.retained_labels == ftl.completed_episodes == 64
    assert ftl.behavior_action_entries == ftl.env_steps
    # BC-iid's driving action is also its label: one entry per step.
    assert bc.expert_action_entries == bc.expert_predict_calls == bc.env_steps
    assert bc.retained_labels == 64 and bc.fits == ftl.fits == 4
    # Evaluation labels each episode in one batched call.
    assert ev.expert_predict_calls == ev.completed_episodes == 2 * 3 * 5
    assert ev.expert_action_entries == ev.env_steps
    assert expert.entries == sum(c.expert_action_entries for c in (ftl, bc, ev))
    assert expert.calls == sum(c.expert_predict_calls for c in (ftl, bc, ev))
    # Retained labels are expert labels at the retained states.
    np.testing.assert_array_equal(
        quantized.label_counts(
            run.data["ftl"]["bins"], run.data["ftl"]["labels"], 9, 2
        ),
        run.counts["ftl"],
    )


def test_learner_sees_only_bins_not_full_state():
    q = quantized.get_quantizer("CartPole-v1", "severe")
    costs = classical.Costs()
    act = classical.table_behavior(np.arange(9) % 2, q, costs)
    for bin_obs in ([0.0, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.06, 0.0]):
        actions = {
            act(np.array(bin_obs) + [0, v, 0, w]) for v in (-3, 3) for w in (-3, 3)
        }
        assert len(actions) == 1
    # The rollout driver gives the table no expert access and no history.
    expert = CountingExpert()
    run = _cell(expert, budget=16, checkpoints=[16], eval_episodes=1)
    run.run(lambda force: None)
    assert run.costs["train_ftl"].expert_action_entries == 16
    assert run.ftl_behavior_tables == [[0] * 9]


def test_fixed_bc_equals_bc_iid_and_shares_evaluation_by_alias():
    run = _cell(CountingExpert())
    run.run(lambda force: None)
    for entry in run.checkpoints:
        fixed = entry["fixed_bc"]
        assert fixed["equals_bc_iid"] is True
        assert fixed["table_sha256"] == entry["bc_iid_final"]["table_sha256"]
        assert fixed["eval_alias"] == "bc_iid_final" and "eval" not in fixed
        assert entry["complete"] is True
    physical = run.report()["physical_costs"]["training"]
    assert physical["retained_labels"] == 128


class ConstantCartEnv(gym.Env):
    """Three steps from one constant CartPole-shaped observation."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(4, np.float32), {}

    def step(self, action):
        self.t += 1
        return np.zeros(4, np.float32), 1.0, self.t >= 3, False, {}


CONSTANT_ID = "AgnosticConstantCart-v0"
if CONSTANT_ID not in gym.registry:
    gym.register(id=CONSTANT_ID, entry_point=ConstantCartEnv, max_episode_steps=10)


def test_every_executed_fit_is_counted_once():
    run = classical.CellRun(
        dict(seed=7, budget=64, batch=16, checkpoints=[64], eval_episodes=2),
        expert=classical.Expert(CountingExpert(lambda o: 1), 2),
        quantizer=quantized.get_quantizer("CartPole-v1", "severe"),
        guard=classical.Guard(DEADLINE, Clock()),
        env_factory=lambda: gym.make(CONSTANT_ID),
    )
    with mock.patch.object(
        quantized,
        "fit_table",
        wraps=quantized.fit_table,
    ) as fit:
        run.run(lambda force: None)
    report = run.report()
    # 4 FTL rounds, 4 BC-iid rounds, and 1 fixed BC fit at the one checkpoint.
    assert fit.call_count == 9
    assert report["physical_costs"]["training"]["fits"] == fit.call_count
    assert report["physical_costs"]["all"]["fits"] == fit.call_count
    own = report["costs"]["train_fixed_bc"]
    assert own["fits"] == 1 and own["elapsed_seconds"] > 0
    assert own["env_steps"] == own["expert_action_entries"] == 0
    assert report["phase_seconds"]["fixed_bc_fit"] == own["elapsed_seconds"]
    (entry,) = report["checkpoints"]
    standalone = entry["fixed_bc"]["logical_standalone_costs"]
    assert standalone["fits"] == 1
    bc = entry["train_costs_at_checkpoint"]["bc_iid"]
    for key in ("env_steps", "expert_action_entries", "retained_labels"):
        assert standalone[key] == bc[key]
    assert standalone["retained_labels"] == 64
    assert standalone["elapsed_seconds"] == pytest.approx(
        report["phase_seconds"]["bc_iid_collect"] + own["elapsed_seconds"],
    )


def test_mixture_covers_every_round_behavior_policy():
    run = _cell(CountingExpert(), budget=64, checkpoints=[16, 64], eval_episodes=40)
    run.run(lambda force: None)
    history = run.ftl_behavior_tables
    assert len(history) == 4 and history[0] == [0] * 9
    ftl = run.data["ftl"]
    for r in range(1, 4):
        refit = quantized.fit_table(
            quantized.label_counts(
                ftl["bins"][: 16 * r], ftl["labels"][: 16 * r], 9, 2
            ),
        )
        assert history[r] == refit.tolist()
    first, last = run.checkpoints
    assert first["ftl_mixture"]["n_policies"] == 1
    assert set(first["ftl_mixture"]["mixture_indices"]) == {0}
    assert last["ftl_mixture"]["n_policies"] == 4
    assert set(last["ftl_mixture"]["mixture_indices"]) == {0, 1, 2, 3}
    # The final post-update policy is reported separately from the mixture.
    assert last["ftl_final"]["table"] == run.tables["ftl"].tolist()


def test_training_seeds_are_paired_and_independent_of_trajectories():
    a = _cell(CountingExpert())
    a.run(lambda force: None)
    b = _cell(CountingExpert(lambda o: 1 - _expert_rule(o)))
    b.run(lambda force: None)
    assert a.data["ftl"]["reset_seeds"] == a.data["bc_iid"]["reset_seeds"]
    assert a.data["ftl"]["reset_seeds"] == b.data["ftl"]["reset_seeds"]
    assert a.data["bc_iid"]["lengths"] != b.data["bc_iid"]["lengths"]
    np.testing.assert_array_equal(a.train_uniforms, b.train_uniforms)
    for entry in a.checkpoints:
        for name in ("ftl_final", "bc_iid_final"):
            assert entry[name]["eval"]["reset_seeds"] == a.eval_seeds
        assert entry["ftl_mixture"]["eval"]["reset_seeds"] == a.eval_seeds
    audit = set(classical.reset_seeds(0, "reference", 64))
    assert not (set(a.train_seeds) & set(a.eval_seeds))
    assert not (audit & (set(a.train_seeds) | set(a.eval_seeds)))
    other_seed = _cell(CountingExpert(), seed=1)
    assert other_seed.train_seeds != a.train_seeds


# ---------------------------------------------------------------------------
# Public paths with a verified synthetic preparation
# ---------------------------------------------------------------------------


def _save_linear_expert(path):
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
def prep_template(tmp_path_factory):
    out = tmp_path_factory.mktemp("prep") / "prep"

    def trainer(env_name, cache_dir, rng, seed, convergence_override=None):
        _save_linear_expert(pathlib.Path(cache_dir) / env_name / "model.zip")

    code = run_classical.prepare_expert(
        "CartPole-v1",
        out,
        seed=0,
        qualification_seed=1234,
        eval_episodes=3,
        deadline=datetime.datetime(2100, 1, 1, tzinfo=UTC),
        trainer=trainer,
    )
    assert code == 0
    return out


@pytest.fixture
def prep(prep_template, tmp_path):
    copy = tmp_path / "prep"
    shutil.copytree(prep_template, copy)
    return copy


def _run_cell(prep, out, clock=None, **kwargs):
    params = dict(
        seed=0,
        deadline=DEADLINE,
        budget=32,
        batch=16,
        checkpoints=[16, 32],
        eval_episodes=3,
        clock=clock or Clock(),
        env_factory=fake_env,
    )
    params.update(kwargs)
    return classical.run_cell("CartPole-v1", prep, "mild", out, **params)


def _result(out):
    return json.loads((out / classical.RESULT_FILE).read_text())


def test_run_cell_complete_record(prep, tmp_path):
    out = tmp_path / "cell"
    assert _run_cell(prep, out) == classical.EXIT_COMPLETE
    record = _result(out)
    assert record["status"] == "complete" and record["interruption"] is None
    assert record["rounds_completed"] == 2 and len(record["checkpoints"]) == 2
    assert record["preparation"]["expert_sha256"] == _prep_record(prep)["expert_sha256"]
    assert record["config_sha256"] == pilot.config_digest(record["config"])
    assert "representation-restricted" in record["representation_status"]
    config = record["config"]
    assert record["protocol"] == record["schema"] == config["schema"]
    for key in ("env_name", "seed", "representation", "expert_sha256"):
        assert record[key] == config[key]
    assert (record["env_name"], record["seed"]) == ("CartPole-v1", 0)
    assert record["representation"] == "mild"
    assert record["expert_sha256"] == record["preparation"]["expert_sha256"]


def test_audit_shares_one_reference_dataset(prep, tmp_path):
    out = tmp_path / "audit"
    code = classical.audit(
        "CartPole-v1",
        prep,
        out,
        seed=0,
        deadline=DEADLINE,
        episodes=40,
        clock=Clock(),
        env_factory=fake_env,
    )
    assert code == classical.EXIT_COMPLETE
    record = _result(out)
    config = record["config"]
    assert record["protocol"] == record["schema"] == config["schema"]
    for key in ("env_name", "seed", "expert_sha256"):
        assert record[key] == config[key]
    assert record["expert_sha256"] == _prep_record(prep)["expert_sha256"]
    reps = record["representations"]
    for rep, k in (("severe", 9), ("mild", 36)):
        bound = reps[rep]["bound"]
        assert bound["n"] == 40 and bound["K"] == k and bound["A"] == 2
        assert np.sum(reps[rep]["counts"]) == 40
    costs = record["costs"]
    assert (
        costs["expert_episodes"]["completed_episodes"]
        + costs["random_episodes"]["completed_episodes"]
        == 40
    )
    # Random episodes query the expert only at their one selected state.
    rand = costs["random_episodes"]
    assert rand["expert_action_entries"] == rand["completed_episodes"] > 0
    exp = costs["expert_episodes"]
    assert exp["expert_action_entries"] == exp["env_steps"]
    data = np.load(out / classical.AUDIT_DATA_FILE)
    assert len(data["label"]) == 40 and len(set(data["reset_seed"].tolist())) == 40
    assert record["samples_file"]["sha256"] == pilot.sha256_file(
        out / classical.AUDIT_DATA_FILE,
    )


def _prep_record(prep):
    return json.loads((prep / pilot.PREPARATION_FILE).read_text())


def _edit_record(prep, **changes):
    record = _prep_record(prep)
    record.update(changes)
    (prep / pilot.PREPARATION_FILE).write_text(json.dumps(record))


@pytest.mark.parametrize("problem", ["status", "digest", "env", "nonfresh"])
def test_refusals_write_nothing_and_do_not_read_old_output(prep, tmp_path, problem):
    out = tmp_path / "cell"
    env_name = "CartPole-v1"
    if problem == "status":
        _edit_record(prep, status="partial")
    elif problem == "digest":
        ckpt = prep / _prep_record(prep)["checkpoint"]["path"]
        ckpt.write_bytes(ckpt.read_bytes() + b"x")
    elif problem == "env":
        env_name = "Acrobot-v1"
    else:
        out.mkdir()
        (out / classical.RESULT_FILE).write_text('{"status": "complete"}')
    before = sorted(p.name for p in out.iterdir()) if out.exists() else None
    code = classical.run_cell(
        env_name,
        prep,
        "severe",
        out,
        seed=0,
        deadline=DEADLINE,
        clock=Clock(),
        env_factory=fake_env,
    )
    assert code == classical.EXIT_REFUSED
    after = sorted(p.name for p in out.iterdir()) if out.exists() else None
    assert before == after


def test_deadline_mid_run_is_partial_with_spent_costs(prep, tmp_path):
    out = tmp_path / "cell"
    assert _run_cell(prep, out, clock=Clock(late_from=400)) == classical.EXIT_INCOMPLETE
    record = _result(out)
    assert record["status"] == "partial"
    assert record["error"]["type"] == "DeadlineExceeded"
    assert record["interruption"]["in_flight_operation"].startswith("train bc_iid")
    ftl, bc = record["costs"]["train_ftl"], record["costs"]["train_bc_iid"]
    assert ftl["retained_labels"] == record["retained"]["ftl"]["n"] == 16
    assert ftl["interrupted_episodes"] == 0 and ftl["fits"] == 1
    # The interrupted BC-iid episode's steps count but it yields no sample.
    assert bc["interrupted_episodes"] == 1 and bc["interrupted_episode_steps"] > 0
    assert (
        bc["retained_labels"]
        == bc["completed_episodes"]
        == record["retained"]["bc_iid"]["n"]
    )
    assert bc["env_steps"] > sum(record["retained"]["bc_iid"]["lengths"])


def test_failed_second_arm_keeps_first_arm_costs(prep, tmp_path, monkeypatch):
    def broken(expert, costs):
        def act(obs):
            raise RuntimeError("expert driver broke")

        return act

    monkeypatch.setattr(classical, "expert_behavior", broken)
    out = tmp_path / "cell"
    assert _run_cell(prep, out) == classical.EXIT_INCOMPLETE
    record = _result(out)
    assert (
        record["status"] == "failed"
        and "expert driver broke" in record["error"]["message"]
    )
    assert "Traceback" in record["error"]["traceback"]
    assert record["costs"]["train_ftl"]["retained_labels"] == 16
    assert record["costs"]["train_ftl"]["expert_action_entries"] == 16
    assert record["costs"]["train_bc_iid"]["interrupted_episodes"] == 1


def test_late_final_operation_cannot_complete(prep, tmp_path):
    counting = Clock()
    assert _run_cell(prep, tmp_path / "a", clock=counting) == 0
    # The final deadline check is the second to last clock call.
    late = Clock(late_from=counting.calls - 1)
    assert _run_cell(prep, tmp_path / "b", clock=late) == classical.EXIT_INCOMPLETE
    record = _result(tmp_path / "b")
    assert record["status"] == "partial" and record["rounds_completed"] == 2
    assert "marking the result complete" in record["error"]["message"]


def test_deadline_is_clamped_to_hard_cap():
    far = datetime.datetime(2030, 1, 1, tzinfo=UTC)
    assert classical.Guard(far).deadline == classical.HARD_CAP
    with pytest.raises(ValueError):
        classical.Guard(datetime.datetime(2026, 1, 1))


# ---------------------------------------------------------------------------
# Real CartPole CLI smoke test (synthetic expert, no scientific claim)
# ---------------------------------------------------------------------------

# Runs the real ``main`` in a fresh process. Only ``rollouts.utc_now`` is
# replaced, before `classical` binds it as its default clock, by a clock that
# starts before the hard cap and advances in real time, so this test does not
# depend on today's date. Scientific jobs use the unpatched real clock.
_CLI_BOOTSTRAP = """
import datetime, sys, time
from imitation.experiments.agnostic import rollouts
_start = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
_t0 = time.monotonic()
rollouts.utc_now = lambda: _start + datetime.timedelta(
    seconds=time.monotonic() - _t0,
)
from imitation.experiments.agnostic import run_classical
sys.exit(run_classical.main(sys.argv[1:]))
"""


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-c", _CLI_BOOTSTRAP, *args],
        env={"PYTHONPATH": str(REPO_SRC), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_cli_audit_and_run_cell_on_real_cartpole(prep, tmp_path):
    deadline = classical.HARD_CAP.isoformat()
    common = ["--env", "CartPole-v1", "--preparation-dir", str(prep), "--seed", "0"]
    audit = _cli(
        "audit",
        *common,
        "--output-dir",
        str(tmp_path / "audit"),
        "--episodes",
        "8",
        "--deadline",
        deadline,
    )
    assert audit.returncode == 0, audit.stderr
    cell = _cli(
        "run-cell",
        *common,
        "--representation",
        "severe",
        "--output-dir",
        str(tmp_path / "cell"),
        "--budget",
        "32",
        "--checkpoints",
        "16",
        "32",
        "--eval-episodes",
        "2",
        "--deadline",
        deadline,
    )
    assert cell.returncode == 0, cell.stderr
    assert json.loads(cell.stdout.strip().splitlines()[-1])["status"] == "complete"
    record = _result(tmp_path / "cell")
    assert record["config"]["env_name"] == record["env_name"] == "CartPole-v1"
    assert record["representation"] == "severe"
    assert record["started_at_utc"].startswith("2026-01-01")
    returns = record["checkpoints"][-1]["bc_iid_final"]["eval"]["returns"]
    assert returns == [
        float(n) for n in record["checkpoints"][-1]["bc_iid_final"]["eval"]["lengths"]
    ]
    again = _cli(
        "run-cell",
        *common,
        "--representation",
        "severe",
        "--output-dir",
        str(tmp_path / "cell"),
        "--deadline",
        deadline,
    )
    assert again.returncode == classical.EXIT_REFUSED and again.stdout == ""
