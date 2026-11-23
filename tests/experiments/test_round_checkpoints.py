"""Compact checkpoints preserve predictions independently of evaluation cadence."""

import pathlib
import shutil

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy

from imitation.experiments.agnostic import capacity_policy
from imitation.experiments.ftrl import policy_utils, run_experiment


def test_dense_early_evaluation_schedule():
    assert run_experiment.EVAL_EPISODES == 10
    assert run_experiment.eval_rounds(105) == list(range(61)) + [80, 100, 105]
    assert run_experiment.eval_rounds(11, 4, 3) == [0, 1, 2, 3, 4, 8, 11]
    assert run_experiment.eval_rounds(11, 4, 0) == [0, 4, 8, 11]
    assert run_experiment.eval_rounds(2) == [0, 1, 2]
    with pytest.raises(ValueError):
        run_experiment.eval_rounds(10, 4, -1)


def _policy(image):
    obs = (
        spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)
        if image
        else spaces.Box(-1, 1, (4,), dtype=np.float32)
    )
    cls = ActorCriticCnnPolicy if image else ActorCriticPolicy
    teacher = cls(
        obs, spaces.Discrete(3), lambda _: 0.001, net_arch=dict(pi=[64], vf=[64])
    )
    return policy_utils.create_linear_policy(teacher)


@pytest.mark.parametrize("image", [False, True])
def test_compact_round_trip_and_shared_frozen_state(tmp_path, image):
    policy = _policy(image)
    obs = np.zeros(
        (2,) + policy.observation_space.shape, dtype=policy.observation_space.dtype
    )
    paths = []
    for r in range(2):
        with th.no_grad():
            policy.action_net.weight.add_(0.2)
        path = tmp_path / f"round-{r}.pt"
        policy_utils.save_policy_checkpoint(path, policy, compact=True)
        paths.append(path)
        saved = th.load(path, weights_only=False)
        assert set(saved["state_dict"]) == {"action_net.weight", "action_net.bias"}
        loaded = policy_utils.load_policy_checkpoint(path)
        for key, value in policy.state_dict().items():
            assert th.equal(value, loaded.state_dict()[key]), key
        assert np.array_equal(
            policy.predict(obs, deterministic=True)[0],
            loaded.predict(obs, deterministic=True)[0],
        )
        assert {n for n, p in loaded.named_parameters() if p.requires_grad} == {
            "action_net.weight",
            "action_net.bias",
        }
    assert len(list(tmp_path.glob("frozen-*.pt"))) == 1
    moved = tmp_path / "moved"
    moved.mkdir()
    for p in list(tmp_path.glob("*.pt")):
        shutil.copy2(p, moved / p.name)
    policy_utils.load_policy_checkpoint(moved / paths[-1].name)
    base = next(moved.glob("frozen-*.pt"))
    base.write_bytes(b"wrong frozen state")
    with pytest.raises(ValueError, match="frozen|digest"):
        policy_utils.load_policy_checkpoint(moved / paths[-1].name)


def test_capacity_compact_writer_binding(tmp_path):
    teacher = _policy(False)
    student = capacity_policy.make_capacity_policy(
        teacher, hidden_width=4, initialization_seed=9
    )
    path = tmp_path / "round.pt"
    binding = student.save_bound(path, compact=True)
    saved = th.load(path, weights_only=False)
    assert set(saved["state_dict"]) == {
        n for n, p in student.named_parameters() if p.requires_grad
    }
    loaded = capacity_policy.load_checkpoint(
        path,
        expected_spec_sha256=binding["spec_sha256"],
        expected_file_sha256=binding["file_sha256"],
        expected_state_sha256=binding["state_sha256"],
    )
    assert capacity_policy.state_sha256(loaded) == capacity_policy.state_sha256(student)


def test_full_checkpoint_compatibility(tmp_path):
    policy = _policy(False)
    path = tmp_path / "old.pt"
    policy.save(path)
    loaded = policy_utils.load_policy_checkpoint(path)
    assert all(
        th.equal(v, loaded.state_dict()[k]) for k, v in policy.state_dict().items()
    )


def test_round_checkpoints_between_evaluations(tmp_path, monkeypatch):
    from .test_eval50_fit_monitor import _config
    from stable_baselines3 import PPO
    from imitation.experiments.ftrl import env_baselines

    expert = PPO("MlpPolicy", "CartPole-v1", seed=10, device="cpu")
    monkeypatch.setattr(
        run_experiment.experts, "get_or_train_expert", lambda *a, **kw: expert.policy
    )
    monkeypatch.setattr(
        env_baselines,
        "load_or_compute_baselines",
        lambda *a, **kw: {"expert_return": 100.0, "random_return": 10.0},
    )
    monkeypatch.setattr(env_baselines, "validate_expert_quality", lambda *a: (True, ""))
    config = _config(
        "bc_iid",
        tmp_path,
        n_rounds=4,
        eval_interval=3,
        eval_initial_rounds=1,
        eval_episodes=1,
        save_every_round=True,
        skip_unreported_fits=True,
        fit_max_steps=10,
        fit_max_epochs=10,
    )
    rows = run_experiment.run_single(config)["per_round"]
    assert [r["round"] for r in rows if r.get("d_eval_size")] == [0, 1, 3, 4]
    assert all(pathlib.Path(r["checkpoint"]).is_file() for r in rows)
    assert all(not r.get("fit_skipped") for r in rows[1:])
    expert.get_env().close()


def test_restricted_compact_checkpoint_keeps_mask(tmp_path):
    from imitation.experiments.agnostic import restriction

    policy = restriction.make_linear_policy(_policy(False), "cart_position_zero")
    path = tmp_path / "restricted.pt"
    policy_utils.save_policy_checkpoint(path, policy, compact=True)
    loaded = restriction.load_policy_checkpoint(path)
    assert loaded.restriction_id == policy.restriction_id
    assert all(
        th.equal(v, loaded.state_dict()[k]) for k, v in policy.state_dict().items()
    )


def test_task_cli_forwards_evaluation_parameters(monkeypatch, tmp_path):
    from imitation.experiments.agnostic import classical, task_pilot

    received = {}

    def capture(*args, **kwargs):
        received.update(kwargs)
        return classical.EXIT_REFUSED

    monkeypatch.setattr(task_pilot, "run_job", capture)
    code = task_pilot.main(
        [
            "run",
            "--env",
            "CartPole-v1",
            "--expert-dir",
            "e",
            "--output-dir",
            str(tmp_path / "run"),
            "--data-dir",
            "d",
            "--deadline",
            "2026-12-01T00:00:00Z",
            "--job-limit-seconds",
            "60",
            "--seed",
            "3100",
            "--method",
            "ftl",
            "--restriction",
            "identity",
            "--statewise-followup",
            "--eval-initial-rounds",
            "7",
            "--eval-interval",
            "13",
            "--eval-episodes",
            "3",
        ]
    )
    assert code == classical.EXIT_REFUSED
    assert received["eval_initial_rounds"] == 7
    assert received["eval_interval"] == 13
    assert received["eval_episodes"] == 3
    assert received["save_every_round"] and received["compact_checkpoints"]


def test_compact_alias_mask_ignores_nonpersistent_buffers(tmp_path):
    from imitation.experiments.agnostic import restriction

    teacher = ActorCriticPolicy(
        spaces.Box(0, 1, (16,), dtype=np.float32),
        spaces.Discrete(4),
        lambda _: 0.001,
    )
    policy = restriction.make_linear_policy(teacher, "frozenlake_column_pairs")
    assert any("representative" in n for n, _ in policy.named_buffers())
    path = tmp_path / "round.pt"
    policy_utils.save_policy_checkpoint(path, policy, compact=True)
    loaded = restriction.load_policy_checkpoint(path)
    assert all(
        th.equal(v, loaded.state_dict()[k]) for k, v in policy.state_dict().items()
    )
