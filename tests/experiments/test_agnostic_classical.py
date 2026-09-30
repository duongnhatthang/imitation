"""Stage 1 tests for the agnostic classical collector and expert preparation.

Only the expensive PPO trainer boundary is replaced. Environments, checkpoint
loading, hashing, threshold freezing, qualification rollouts, and status files
all run for real on tiny budgets.
"""

import datetime
import json
import pathlib
import subprocess
import sys
from typing import List

import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper

from imitation.experiments.agnostic import pilot, rollouts, run_classical
from imitation.experiments.ftrl import env_utils

UTC = datetime.timezone.utc
FAR_DEADLINE = datetime.datetime(2100, 1, 1, tzinfo=UTC)
REPO_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


class CountingPolicy:
    """Deterministic policy with a predict() method that counts its queries."""

    def __init__(self, action_fn):
        self.action_fn = action_fn
        self.calls = 0
        self.entries = 0

    def predict(self, obs, deterministic=True):
        obs = np.asarray(obs)
        self.calls += 1
        self.entries += obs.shape[0]
        return np.array([self.action_fn(o) for o in obs]), None


def _balance(obs):
    return int(obs[0] + obs[1] + 10 * obs[2] + 10 * obs[3] > 0)


def _cartpole_venv(seed=0):
    return env_utils.make_env("CartPole-v1", n_envs=1, rng=np.random.default_rng(seed))


class StepClock:
    """Clock that advances one second per call, starting at ``start``."""

    def __init__(self, start):
        self.now = start
        self.calls = 0

    def __call__(self):
        self.calls += 1
        value = self.now
        self.now = self.now + datetime.timedelta(seconds=1)
        return value


# ---------------------------------------------------------------------------
# Deadline and RNG streams
# ---------------------------------------------------------------------------


def test_parse_utc_deadline_accepts_z_and_rejects_naive():
    parsed = rollouts.parse_utc_deadline("2026-10-07T01:01:00Z")
    assert parsed == datetime.datetime(2026, 10, 7, 1, 1, tzinfo=UTC)
    offset = rollouts.parse_utc_deadline("2026-10-06T18:01:00-07:00")
    assert offset == parsed and offset.tzinfo == UTC
    with pytest.raises(ValueError):
        rollouts.parse_utc_deadline("2026-10-07T01:01:00")


def test_phase_rngs_are_distinct_streams_and_reproducible():
    draws = {
        phase: rollouts.make_phase_rng(0, phase).integers(0, 2**31, 8).tolist()
        for phase in rollouts.PHASES
    }
    assert len({tuple(v) for v in draws.values()}) == len(rollouts.PHASES)
    again = rollouts.make_phase_rng(0, "eval").integers(0, 2**31, 8).tolist()
    assert again == draws["eval"]
    with pytest.raises(ValueError):
        rollouts.make_phase_rng(0, "not-a-phase")


# ---------------------------------------------------------------------------
# Single-env evaluator
# ---------------------------------------------------------------------------


def test_expert_behavior_counts_transitions_and_never_double_queries():
    expert = CountingPolicy(lambda o: 0)
    venv = _cartpole_venv()
    result = rollouts.evaluate_episodes(
        venv,
        expert,
        n_episodes=3,
        phase="expert_qualification",
        behavior_is_expert=True,
        expert=expert,
        deadline=FAR_DEADLINE,
        max_env_steps=10_000,
        env_seed=11,
    )
    c = result.counters
    assert result.completed and c.stop_reason == "completed"
    assert c.completed_episodes == 3 == len(result.episode_returns)
    # CartPole pays +1 per step, so native returns equal episode lengths.
    assert result.episode_returns == [float(n) for n in result.episode_lengths]
    assert c.env_transitions == sum(result.episode_lengths)
    assert c.partial_episode_transitions == 0
    # The expert drives the rollout; its one action per step is the label.
    assert expert.calls == c.env_transitions == c.expert_predict_calls
    assert c.expert_action_entries == expert.entries == c.env_transitions
    assert c.behavior_action_entries == c.env_transitions
    assert result.disagreement == 0.0


def test_learner_behavior_queries_expert_separately_for_disagreement():
    learner = CountingPolicy(lambda o: 0)
    expert = CountingPolicy(_balance)
    result = rollouts.evaluate_episodes(
        _cartpole_venv(),
        learner,
        n_episodes=2,
        phase="eval",
        behavior_is_expert=False,
        expert=expert,
        deadline=FAR_DEADLINE,
        max_env_steps=10_000,
        env_seed=3,
    )
    c = result.counters
    assert learner.calls == c.behavior_predict_calls == c.env_transitions
    assert expert.calls == c.expert_predict_calls == c.env_transitions
    assert c.expert_action_entries == c.env_transitions
    assert 0.0 < result.disagreement <= 1.0


def test_evaluation_without_expert_reports_no_disagreement_or_expert_cost():
    learner = CountingPolicy(lambda o: 1)
    result = rollouts.evaluate_episodes(
        _cartpole_venv(),
        learner,
        n_episodes=1,
        phase="eval",
        deadline=FAR_DEADLINE,
        max_env_steps=10_000,
    )
    assert result.disagreement is None
    assert result.counters.expert_action_entries == 0


def test_step_budget_interrupts_mid_episode_and_preserves_counts():
    policy = CountingPolicy(lambda o: 0)
    result = rollouts.evaluate_episodes(
        _cartpole_venv(),
        policy,
        n_episodes=50,
        phase="eval",
        deadline=FAR_DEADLINE,
        max_env_steps=25,
        env_seed=0,
    )
    c = result.counters
    assert not result.completed and c.stop_reason == "step_budget"
    assert c.env_transitions == 25 == policy.calls
    assert c.completed_episodes == len(result.episode_returns) >= 1
    assert sum(result.episode_lengths) + c.partial_episode_transitions == 25
    assert c.partial_episode_transitions > 0
    assert c.partial_episode_return == float(c.partial_episode_transitions)


def test_deadline_interrupts_rollout_and_keeps_partial_counts():
    start = datetime.datetime(2030, 1, 1, tzinfo=UTC)
    clock = StepClock(start)
    policy = CountingPolicy(lambda o: 0)
    result = rollouts.evaluate_episodes(
        _cartpole_venv(),
        policy,
        n_episodes=50,
        phase="eval",
        deadline=start + datetime.timedelta(seconds=7),
        max_env_steps=10_000,
        clock=clock,
    )
    c = result.counters
    assert not result.completed and c.stop_reason == "deadline"
    assert 0 < c.env_transitions < 10
    assert policy.calls == c.env_transitions
    assert sum(result.episode_lengths) + c.partial_episode_transitions == (
        c.env_transitions
    )


def test_evaluator_requires_finite_budget_and_aware_deadline():
    policy = CountingPolicy(lambda o: 0)
    with pytest.raises(ValueError):
        rollouts.evaluate_episodes(
            _cartpole_venv(),
            policy,
            n_episodes=1,
            phase="eval",
            deadline=FAR_DEADLINE,
            max_env_steps=0,
        )
    with pytest.raises(ValueError):
        rollouts.evaluate_episodes(
            _cartpole_venv(),
            policy,
            n_episodes=1,
            phase="eval",
            deadline=datetime.datetime(2100, 1, 1),
            max_env_steps=10,
        )
    assert policy.calls == 0


def test_env_seed_makes_evaluation_reproducible():
    def run(seed):
        return rollouts.evaluate_episodes(
            _cartpole_venv(seed=99),
            CountingPolicy(lambda o: 0),
            n_episodes=4,
            phase="eval",
            deadline=FAR_DEADLINE,
            max_env_steps=10_000,
            env_seed=seed,
        ).episode_returns

    assert run(5) == run(5)


# ---------------------------------------------------------------------------
# Threshold and provenance helpers
# ---------------------------------------------------------------------------


def test_frozen_threshold_uses_convergence_threshold_not_generic_080():
    frozen = pilot.frozen_expert_threshold("CartPole-v1")
    assert frozen["convergence_threshold"] == 0.95
    assert frozen["raw_threshold"] == pytest.approx(22.0 + 0.95 * (500.0 - 22.0))
    merged = pilot.merged_convergence_config("MountainCar-v0", {"threshold": 0.9})
    assert merged["threshold"] == 0.9
    assert merged["max_timesteps"] == 6_000_000  # env-specific cap retained
    frozen_mc = pilot.frozen_expert_threshold("MountainCar-v0", merged)
    assert frozen_mc["raw_threshold"] == pytest.approx(-200.0 + 0.9 * 90.0)
    with pytest.raises(ValueError):
        pilot.merged_convergence_config("CartPole-v1", {"not_a_key": 1})


def test_atomic_write_json_leaves_no_temp_files(tmp_path):
    path = tmp_path / "status.json"
    pilot.atomic_write_json(path, {"a": 1})
    pilot.atomic_write_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]


# ---------------------------------------------------------------------------
# prepare-expert
# ---------------------------------------------------------------------------


def _save_cartpole_model(path: pathlib.Path, balancing: bool) -> None:
    venv = _cartpole_venv()
    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(net_arch=dict(pi=[], vf=[])),
        device="cpu",
        seed=0,
    )
    if balancing:
        with th.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
            model.policy.action_net.weight[1] = th.tensor([1.0, 1.0, 10.0, 10.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    venv.close()


class FakeTrainer:
    """Stands in for the PPO convergence trainer; writes a real checkpoint."""

    def __init__(self, output_dir, balancing=True, error=None):
        self.output_dir = pathlib.Path(output_dir)
        self.balancing = balancing
        self.error = error
        self.calls: List[dict] = []

    def __call__(self, env_name, cache_dir, rng, seed, convergence_override=None):
        status = json.loads((self.output_dir / "preparation.json").read_text())
        self.calls.append(
            dict(
                env_name=env_name,
                cache_dir=pathlib.Path(cache_dir),
                rng=rng,
                seed=seed,
                convergence_override=convergence_override,
                status_during_training=status,
            ),
        )
        if self.error is not None:
            raise self.error
        path = pathlib.Path(cache_dir) / env_name.replace("/", "_") / "model.zip"
        _save_cartpole_model(path, self.balancing)
        return None


def _prepare(out, trainer, **kwargs):
    params = dict(
        env_name="CartPole-v1",
        output_dir=out,
        seed=0,
        qualification_seed=1234,
        eval_episodes=5,
        deadline=FAR_DEADLINE,
        trainer=trainer,
    )
    params.update(kwargs)
    return run_classical.prepare_expert(**params)


def _record(out):
    return json.loads((pathlib.Path(out) / "preparation.json").read_text())


def test_prepare_expert_qualifies_real_checkpoint_and_records_evidence(tmp_path):
    out = tmp_path / "prep"
    trainer = FakeTrainer(out, balancing=True)
    code = _prepare(out, trainer)
    assert code == 0
    rec = _record(out)
    assert rec["status"] == "complete" and rec["qualified"] is True
    assert rec["approved"] is True

    # Running status with the frozen threshold existed before training began.
    during = trainer.calls[0]["status_during_training"]
    assert during["status"] == "running" and during["phase"] == "expert_training"
    assert during["threshold"]["raw_threshold"] == pytest.approx(476.1)
    assert during["qualified"] is False
    assert "not interruptible" in during["training_interruptibility"]

    # Default config: inherited trainer resolves its own identical config.
    call = trainer.calls[0]
    assert call["seed"] == 0 and call["convergence_override"] is None
    assert isinstance(call["rng"], np.random.Generator)

    ckpt = out / rec["checkpoint"]["path"]
    assert ckpt.is_file()
    assert rec["checkpoint"]["sha256"] == pilot.sha256_file(ckpt)
    assert len(rec["checkpoint"]["policy_state_sha256"]) == 64

    q = rec["qualification"]
    assert q["episode_returns"] == [500.0] * 5
    assert q["mean_return"] == 500.0
    assert q["counters"]["completed_episodes"] == 5
    assert q["counters"]["env_transitions"] == 2500
    assert q["counters"]["expert_action_entries"] == 2500
    assert q["counters"]["stop_reason"] == "completed"
    assert q["env_reset_seed"] == 1234

    training = rec["counters"]["expert_training"]
    assert training["env_transitions"] == "unavailable"
    assert training["expert_action_entries"] == "unavailable"

    assert rec["config"]["qualification_seed"] == 1234
    assert rec["config"]["convergence_config"]["threshold"] == 0.95
    assert rec["config_sha256"] == pilot.config_digest(rec["config"])
    assert 1234 not in rec["seeds"]["training_env_reset_seeds"]
    assert "stable_baselines3" in rec["package_versions"]
    assert rec["source_fingerprint"]["files"]
    assert rec["elapsed_seconds"]["total"] >= 0.0


def test_prepare_expert_unqualified_checkpoint_is_not_approved(tmp_path):
    out = tmp_path / "prep"
    code = _prepare(out, FakeTrainer(out, balancing=False))
    assert code != 0
    rec = _record(out)
    assert rec["status"] == "not_qualified"
    assert rec["qualified"] is False and rec["approved"] is False
    assert rec["qualification"]["mean_return"] < rec["threshold"]["raw_threshold"]
    assert len(rec["qualification"]["episode_returns"]) == 5


def test_trainer_failure_writes_failed_status(tmp_path):
    out = tmp_path / "prep"
    trainer = FakeTrainer(out, error=RuntimeError("failed to converge"))
    code = _prepare(out, trainer)
    assert code != 0
    rec = _record(out)
    assert rec["status"] == "failed" and rec["phase"] == "expert_training"
    assert rec["qualified"] is False and rec["approved"] is False
    assert "failed to converge" in rec["error"]["message"]
    assert rec["error"]["type"] == "RuntimeError"


def test_expired_deadline_skips_training(tmp_path):
    out = tmp_path / "prep"
    trainer = FakeTrainer(out)
    code = _prepare(out, trainer, deadline=datetime.datetime(2000, 1, 1, tzinfo=UTC))
    assert code != 0 and trainer.calls == []
    rec = _record(out)
    assert rec["status"] == "failed" and rec["error"]["type"] == "DeadlineExceeded"
    assert rec["qualified"] is False


def test_deadline_after_training_leaves_partial_unqualified(tmp_path):
    out = tmp_path / "prep"
    trainer = FakeTrainer(out)
    start = datetime.datetime(2030, 1, 1, tzinfo=UTC)

    def clock():
        # Time jumps past the deadline once training has happened.
        return start + datetime.timedelta(days=2 if trainer.calls else 0)

    code = _prepare(
        out,
        trainer,
        deadline=start + datetime.timedelta(days=1),
        clock=clock,
    )
    assert code != 0 and len(trainer.calls) == 1
    rec = _record(out)
    assert rec["status"] == "partial" and rec["qualified"] is False
    assert rec["phase"] == "expert_qualification"
    assert rec["checkpoint"]["sha256"] == pilot.sha256_file(
        out / rec["checkpoint"]["path"],
    )


def test_qualification_seed_must_be_disjoint_from_training(tmp_path):
    out = tmp_path / "prep"
    trainer = FakeTrainer(out)
    with pytest.raises(ValueError):
        _prepare(out, trainer, seed=7, qualification_seed=7)
    # MountainCar trains with 4 PPO envs reset at seed, seed+1, ..., seed+3.
    with pytest.raises(ValueError):
        _prepare(
            out,
            trainer,
            env_name="MountainCar-v0",
            seed=10,
            qualification_seed=12,
        )
    assert trainer.calls == [] and not out.exists()


def test_matching_qualified_preparation_is_reused_without_training(tmp_path):
    out = tmp_path / "prep"
    assert _prepare(out, FakeTrainer(out)) == 0
    before = (out / "preparation.json").read_bytes()
    again = FakeTrainer(out)
    assert _prepare(out, again) == 0
    assert again.calls == []
    assert (out / "preparation.json").read_bytes() == before


def test_mismatched_config_is_refused_without_overwrite(tmp_path):
    out = tmp_path / "prep"
    assert _prepare(out, FakeTrainer(out)) == 0
    before = (out / "preparation.json").read_bytes()
    other = FakeTrainer(out)
    assert _prepare(out, other, qualification_seed=999) != 0
    assert _prepare(out, other, eval_episodes=6) != 0
    assert _prepare(out, other, convergence_override={"threshold": 0.9}) != 0
    assert other.calls == []
    assert (out / "preparation.json").read_bytes() == before


def test_tampered_checkpoint_is_refused_and_not_approved(tmp_path):
    out = tmp_path / "prep"
    assert _prepare(out, FakeTrainer(out)) == 0
    rec = _record(out)
    ckpt = out / rec["checkpoint"]["path"]
    ckpt.write_bytes(ckpt.read_bytes() + b"tamper")
    again = FakeTrainer(out)
    assert _prepare(out, again) != 0
    assert again.calls == []
    assert pilot.verify_qualified_preparation(out, rec["config"]) is False


def test_preexisting_running_record_stays_visibly_incomplete(tmp_path):
    out = tmp_path / "prep"
    out.mkdir()
    config = run_classical.build_preparation_config(
        "CartPole-v1",
        seed=0,
        qualification_seed=1234,
        eval_episodes=5,
        convergence_override=None,
    )
    stale = {"status": "running", "config": config, "qualified": False}
    pilot.atomic_write_json(out / "preparation.json", stale)
    trainer = FakeTrainer(out)
    assert _prepare(out, trainer) != 0
    assert trainer.calls == []
    assert _record(out) == stale


def test_stale_checkpoint_without_record_is_refused(tmp_path):
    out = tmp_path / "prep"
    _save_cartpole_model(out / "trainer_cache" / "CartPole-v1" / "model.zip", True)
    trainer = FakeTrainer(out)
    assert _prepare(out, trainer) != 0
    assert trainer.calls == [] and not (out / "preparation.json").exists()


def test_override_is_fully_merged_before_reaching_trainer(tmp_path):
    out = tmp_path / "prep"
    trainer = FakeTrainer(out)
    _prepare(out, trainer, convergence_override={"threshold": 0.5})
    passed = trainer.calls[0]["convergence_override"]
    assert passed == pilot.merged_convergence_config("CartPole-v1", {"threshold": 0.5})
    assert set(env_utils.DEFAULT_CONVERGENCE) <= set(passed)
    assert _record(out)["threshold"]["raw_threshold"] == pytest.approx(22 + 0.5 * 478)


def test_cli_main_parses_interface_and_runs_injected_trainer(tmp_path):
    out = tmp_path / "prep"
    trainer = FakeTrainer(out)
    argv = [
        "prepare-expert",
        "--env",
        "CartPole-v1",
        "--output-dir",
        str(out),
        "--seed",
        "0",
        "--qualification-seed",
        "4321",
        "--eval-episodes",
        "3",
        "--deadline",
        "2100-01-01T00:00:00Z",
    ]
    assert run_classical.main(argv, trainer=trainer) == 0
    rec = _record(out)
    assert rec["config"]["eval_episodes"] == 3
    assert rec["deadline_utc"] == "2100-01-01T00:00:00+00:00"


def test_cli_module_rejects_expired_deadline_without_training(tmp_path):
    out = tmp_path / "prep"
    cmd = [
        sys.executable,
        "-m",
        "imitation.experiments.agnostic.run_classical",
        "prepare-expert",
        "--env",
        "Acrobot-v1",
        "--output-dir",
        str(out),
        "--seed",
        "0",
        "--qualification-seed",
        "77",
        "--eval-episodes",
        "100",
        "--deadline",
        "2000-01-01T00:00:00Z",
    ]
    env = {"PYTHONPATH": str(REPO_SRC), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0, proc.stderr
    rec = _record(out)
    assert rec["status"] == "failed" and rec["qualified"] is False
    assert rec["config"]["env_name"] == "Acrobot-v1"


# ---------------------------------------------------------------------------
# Stage 1 repairs: queue contract, config bounds, claim, failure accounting
# ---------------------------------------------------------------------------


def test_complete_record_exposes_queue_contract_fields(tmp_path):
    out = tmp_path / "prep"
    assert _prepare(out, FakeTrainer(out)) == run_classical.EXIT_QUALIFIED
    raw = (out / "preparation.json").read_text()
    # Strict JSON: the campaign queue rejects NaN and Infinity constants.
    rec = json.loads(raw, parse_constant=lambda c: pytest.fail(c))
    assert rec["status"] == "complete"
    assert rec["qualified"] is True and rec["approved"] is True
    assert rec["protocol"] == pilot.PREPARATION_SCHEMA
    assert rec["seed"] == 0 and rec["env_name"] == "CartPole-v1"
    assert rec["expert_sha256"] == rec["checkpoint"]["sha256"]
    assert rec["expert_sha256"] == pilot.sha256_file(out / rec["checkpoint"]["path"])
    assert rec["config_sha256"] == pilot.config_digest(rec["config"])
    assert len(rec["source_fingerprint"]["combined_sha256"]) == 64
    assert pilot.verify_qualified_preparation(out, rec["config"]) is True


def test_old_qualified_status_is_not_accepted_as_complete(tmp_path):
    out = tmp_path / "prep"
    assert _prepare(out, FakeTrainer(out)) == 0
    rec = _record(out)
    rec["status"] = "qualified"
    pilot.atomic_write_json(out / "preparation.json", rec)
    assert pilot.verify_qualified_preparation(out, rec["config"]) is False
    again = FakeTrainer(out)
    assert _prepare(out, again) == run_classical.EXIT_REFUSED
    assert again.calls == []


def test_verifier_rejects_expert_alias_mismatch_and_escaping_path(tmp_path):
    out = tmp_path / "prep"
    assert _prepare(out, FakeTrainer(out)) == 0
    good = _record(out)
    bad_alias = dict(good, expert_sha256="0" * 64)
    pilot.atomic_write_json(out / "preparation.json", bad_alias)
    assert pilot.verify_qualified_preparation(out, good["config"]) is False
    outside = tmp_path / "model.zip"
    outside.write_bytes((out / good["checkpoint"]["path"]).read_bytes())
    escaping = dict(good, checkpoint=dict(good["checkpoint"], path="../model.zip"))
    pilot.atomic_write_json(out / "preparation.json", escaping)
    assert pilot.verify_qualified_preparation(out, good["config"]) is False


@pytest.mark.parametrize(
    "override",
    [
        {"max_timesteps": float("inf")},
        {"max_timesteps": float("nan")},
        {"max_timesteps": 0},
        {"max_timesteps": -25_000},
        {"max_timesteps": 25_000.5},
        {"max_timesteps": True},
        {"max_timesteps": "5000000"},
        {"chunk_timesteps": 0},
        {"chunk_timesteps": float("inf")},
        {"min_timesteps": 0},
        {"min_timesteps": 60_000, "max_timesteps": 50_000},
        {"patience": 0},
        {"patience": 2.5},
        {"patience": float("inf")},
        {"threshold": float("-inf")},
        {"threshold": float("nan")},
        {"threshold": -0.01},
        {"threshold": 1.01},
        {"self_ce_eps": -0.1},
        {"self_ce_eps": float("inf")},
        {"self_ce_eps": float("nan")},
    ],
)
def test_invalid_convergence_config_is_rejected_before_any_output(tmp_path, override):
    with pytest.raises(ValueError):
        pilot.merged_convergence_config("CartPole-v1", override)
    out = tmp_path / "prep"
    trainer = FakeTrainer(out)
    with pytest.raises(ValueError):
        _prepare(out, trainer, convergence_override=override)
    assert trainer.calls == [] and not out.exists()


@pytest.mark.parametrize("override", [[("threshold", 0.9)], 5, "threshold"])
def test_non_mapping_convergence_override_is_rejected(override):
    with pytest.raises(ValueError):
        pilot.merged_convergence_config("CartPole-v1", override)


def test_convergence_config_boundaries_are_accepted_and_normalized():
    lo = pilot.merged_convergence_config(
        "CartPole-v1",
        {"threshold": 0.0, "self_ce_eps": 0.0, "patience": 1},
    )
    assert lo["threshold"] == 0.0 and lo["self_ce_eps"] == 0.0
    hi = pilot.merged_convergence_config(
        "CartPole-v1",
        {"threshold": 1, "min_timesteps": 5_000_000, "chunk_timesteps": 25_000.0},
    )
    assert hi["threshold"] == 1.0 and hi["min_timesteps"] == hi["max_timesteps"]
    assert hi["chunk_timesteps"] == 25_000 and isinstance(hi["chunk_timesteps"], int)


def test_cli_invalid_convergence_override_is_usage_error(tmp_path):
    out = tmp_path / "prep"
    trainer = FakeTrainer(out)
    argv = [
        "prepare-expert",
        "--env",
        "CartPole-v1",
        "--output-dir",
        str(out),
        "--seed",
        "0",
        "--qualification-seed",
        "4321",
        "--deadline",
        "2100-01-01T00:00:00Z",
        "--convergence-override",
        '{"patience": 0}',
    ]
    assert run_classical.main(argv, trainer=trainer) == run_classical.EXIT_USAGE
    assert trainer.calls == [] and not out.exists()


class NestedTrainer(FakeTrainer):
    """Trainer that launches a competing preparation into the same directory."""

    def __init__(self, output_dir, competitor):
        super().__init__(output_dir)
        self.competitor = competitor
        self.competitor_code = None
        self.record_after_competitor = None

    def __call__(self, env_name, cache_dir, rng, seed, convergence_override=None):
        self.competitor_code = _prepare(self.output_dir, self.competitor)
        self.record_after_competitor = _record(self.output_dir)
        return super().__call__(env_name, cache_dir, rng, seed, convergence_override)


def test_second_owner_during_training_is_refused_before_trainer(tmp_path):
    out = tmp_path / "prep"
    competitor = FakeTrainer(out)
    owner = NestedTrainer(out, competitor)
    assert _prepare(out, owner) == 0
    assert owner.competitor_code == run_classical.EXIT_REFUSED
    assert competitor.calls == []
    assert owner.record_after_competitor["status"] == "running"
    assert _record(out)["status"] == "complete"


def test_exclusive_claim_closes_check_then_write_race(tmp_path, monkeypatch):
    # Both invocations pass the emptiness check, as in a real race; the
    # exclusive claim must still stop the second one before its trainer.
    monkeypatch.setattr(run_classical, "_has_existing_entries", lambda d: False)
    out = tmp_path / "prep"
    competitor = FakeTrainer(out)
    owner = NestedTrainer(out, competitor)
    assert _prepare(out, owner) == 0
    assert owner.competitor_code == run_classical.EXIT_REFUSED
    assert competitor.calls == []
    assert owner.record_after_competitor["status"] == "running"
    rec = _record(out)
    assert rec["status"] == "complete" and len(owner.calls) == 1


def test_claim_without_record_is_refused_as_evidence(tmp_path):
    out = tmp_path / "prep"
    out.mkdir()
    (out / run_classical.CLAIM_FILE).write_text("{}")
    trainer = FakeTrainer(out)
    assert _prepare(out, trainer) == run_classical.EXIT_REFUSED
    assert trainer.calls == [] and not (out / "preparation.json").exists()


class FailAfterSteps(VecEnvWrapper):
    """VecEnv whose ``step`` raises once ``n_ok`` steps have succeeded."""

    def __init__(self, venv, n_ok):
        super().__init__(venv)
        self.n_ok = n_ok
        self.ok = 0

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        if self.ok >= self.n_ok:
            raise RuntimeError("env broke")
        self.ok += 1
        return self.venv.step_wait()


class FailingPolicy(CountingPolicy):
    def __init__(self, fail_on_call):
        super().__init__(lambda o: 0)
        self.fail_on_call = fail_on_call

    def predict(self, obs, deterministic=True):
        if self.calls + 1 == self.fail_on_call:
            self.calls += 1
            raise RuntimeError("policy broke")
        return super().predict(obs, deterministic)


def test_env_step_failure_exposes_factual_partial_counters():
    policy = CountingPolicy(lambda o: 0)
    with pytest.raises(rollouts.EvaluationError) as info:
        rollouts.evaluate_episodes(
            FailAfterSteps(_cartpole_venv(), n_ok=3),
            policy,
            n_episodes=5,
            phase="expert_qualification",
            behavior_is_expert=True,
            deadline=FAR_DEADLINE,
            max_env_steps=10_000,
            env_seed=0,
        )
    assert isinstance(info.value.__cause__, RuntimeError)
    res = info.value.result
    c = res.counters
    # Four actions were queried; the fourth step raised, so only three
    # transitions happened. No step is invented for the failed one.
    assert policy.calls == 4
    assert c.behavior_predict_calls == 4 == c.expert_predict_calls
    assert c.expert_action_entries == 4
    assert c.env_transitions == 3 and c.completed_episodes == 0
    assert c.partial_episode_transitions == 3
    assert c.partial_episode_return == 3.0
    assert c.stop_reason == "error" and res.completed is False
    assert res.episode_returns == []


def test_policy_failure_counts_only_returned_queries():
    policy = FailingPolicy(fail_on_call=3)
    with pytest.raises(rollouts.EvaluationError) as info:
        rollouts.evaluate_episodes(
            _cartpole_venv(),
            policy,
            n_episodes=5,
            phase="eval",
            deadline=FAR_DEADLINE,
            max_env_steps=10_000,
            env_seed=0,
        )
    c = info.value.result.counters
    assert c.behavior_predict_calls == 2 and c.behavior_action_entries == 2
    assert c.env_transitions == 2 and c.stop_reason == "error"
    assert str(info.value.__cause__) == "policy broke"


def test_qualification_env_failure_writes_partial_counters(tmp_path, monkeypatch):
    real_make_env = env_utils.make_env

    def make_failing_env(*args, **kwargs):
        return FailAfterSteps(real_make_env(*args, **kwargs), n_ok=3)

    out = tmp_path / "prep"
    trainer = FakeTrainer(out)
    monkeypatch.setattr(run_classical.env_utils, "make_env", make_failing_env)
    assert _prepare(out, trainer) == run_classical.EXIT_NOT_QUALIFIED
    rec = _record(out)
    assert rec["status"] == "failed" and rec["phase"] == "expert_qualification"
    assert rec["qualified"] is False and rec["approved"] is False
    assert rec["error"] == {"type": "RuntimeError", "message": "env broke"}
    counters = rec["counters"]["expert_qualification"]
    assert counters["env_transitions"] == 3
    assert counters["expert_action_entries"] == 4
    assert counters["stop_reason"] == "error"
    assert rec["qualification"]["counters"] == counters
    assert rec["qualification"]["episode_returns"] == []
    assert rec["qualification"]["mean_return"] is None


def test_qualification_step_budget_is_not_labelled_deadline(tmp_path):
    out = tmp_path / "prep"
    assert _prepare(out, FakeTrainer(out), max_eval_env_steps=10) != 0
    rec = _record(out)
    assert rec["status"] == "partial" and rec["qualified"] is False
    assert rec["error"]["type"] == "StepBudgetExhausted"
    counters = rec["counters"]["expert_qualification"]
    assert counters["stop_reason"] == "step_budget"
    assert counters["env_transitions"] == 10


def test_qualification_deadline_stop_is_labelled_deadline(tmp_path):
    out = tmp_path / "prep"
    trainer = FakeTrainer(out)
    start = datetime.datetime(2030, 1, 1, tzinfo=UTC)
    ticks = {"n": 0}

    def clock():
        # Before training: time stands still. After: one second per call.
        if not trainer.calls:
            return start
        ticks["n"] += 1
        return start + datetime.timedelta(seconds=ticks["n"])

    code = _prepare(
        out,
        trainer,
        deadline=start + datetime.timedelta(seconds=20),
        clock=clock,
    )
    assert code != 0
    rec = _record(out)
    assert rec["status"] == "partial"
    assert rec["error"]["type"] == "DeadlineExceeded"
    counters = rec["counters"]["expert_qualification"]
    assert counters["stop_reason"] == "deadline"
    assert 0 < counters["env_transitions"] < 20


@pytest.mark.parametrize("env_name", run_classical.SUPPORTED_ENVS)
def test_default_convergence_configs_pass_validation(env_name):
    merged = pilot.merged_convergence_config(env_name)
    assert merged == {
        k: (int(v) if k in pilot._COUNT_KEYS else float(v))
        for k, v in env_utils.get_convergence_config(env_name).items()
    }


def test_non_object_record_is_refused(tmp_path):
    out = tmp_path / "prep"
    out.mkdir()
    (out / "preparation.json").write_text("[1, 2]")
    trainer = FakeTrainer(out)
    assert _prepare(out, trainer) == run_classical.EXIT_REFUSED
    assert trainer.calls == []
