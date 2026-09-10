"""BC-iid collection and training contracts on a small deterministic MDP."""

import dataclasses
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import serialize
from imitation.experiments.ftrl import coverage_data, run_experiment


class ActionHistoryEnv(gym.Env):
    """Expose elapsed time and the number of non-expert actions in the state."""

    observation_space = gym.spaces.Box(0, 7, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.elapsed = self.mistakes = 0
        return np.array([0, 0], dtype=np.float32), {}

    def step(self, action):
        self.elapsed += 1
        self.mistakes += int(action == 0)
        obs = np.array([self.elapsed, self.mistakes], dtype=np.float32)
        return obs, float(action == 1), self.elapsed == 7, False, {}


def constant_policy(obs_space, act_space, action):
    policy = ActorCriticPolicy(obs_space, act_space, lr_schedule=lambda _: 0.0)
    with th.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.fill_(-20)
        policy.action_net.bias[action] = 20
    return policy


@pytest.fixture
def tiny_experiment(tmp_path, monkeypatch):
    """Avoid expert pretraining while exercising real rollouts and BC updates."""
    from imitation.experiments.ftrl import env_baselines

    expert = constant_policy(
        ActionHistoryEnv.observation_space, ActionHistoryEnv.action_space, 1
    )
    monkeypatch.setattr(
        run_experiment.env_utils,
        "make_env",
        lambda *args, **kwargs: DummyVecEnv([ActionHistoryEnv]),
    )
    monkeypatch.setattr(
        run_experiment.experts, "get_or_train_expert", lambda *a, **kw: expert
    )
    monkeypatch.setattr(
        env_baselines,
        "load_or_compute_baselines",
        lambda *a, **kw: {"expert_return": 7.0, "random_return": 0.0},
    )
    monkeypatch.setattr(env_baselines, "validate_expert_quality", lambda *a: (True, ""))
    monkeypatch.setattr(
        run_experiment.policy_utils,
        "create_end_to_end_policy",
        lambda obs, acts: constant_policy(obs, acts, 0),
    )
    return run_experiment.ExperimentConfig(
        algo="bc_iid",
        env_name="CartPole-v1",
        seed=7,
        policy_mode="end_to_end",
        n_rounds=4,
        samples_per_round=3,
        l2_lambda=123.0,
        l2_decay=True,
        warm_start=True,
        beta_rampdown=1,
        bc_n_epochs=2,
        eval_interval=2,
        output_dir=tmp_path / "results",
        expert_cache_dir=tmp_path / "experts",
        learning_rate=0.0,
        outer_early_stop=False,
        inner_early_stop=False,
    )


@pytest.mark.parametrize(
    "trajectories,m,expected_trajectories", [(1, 1, 1), (3, 5, 3), (1, 20, 3)]
)
@pytest.mark.parametrize("early_stop", [False, True])
def test_bc_iid_uses_only_expert_rollouts_and_accumulates(
    tiny_experiment, monkeypatch, trajectories, m, expected_trajectories, early_stop
):
    config = dataclasses.replace(
        tiny_experiment,
        trajectories_per_round=trajectories,
        samples_per_round=m,
        inner_early_stop=early_stop,
        inner_early_stop_min_val_size=1,
        inner_early_stop_val_frac=0.25,
    )
    trained_sizes = []
    original_train = run_experiment._inner_train

    def train(trainer, *args, **kwargs):
        batch_size = trainer.bc_trainer.batch_size
        minibatch_size = trainer.bc_trainer.minibatch_size
        result = original_train(trainer, *args, **kwargs)
        trained_sizes.append(sum(len(traj) for traj in trainer._all_demos))
        assert trainer.bc_trainer.loss_calculator.l2_weight == 0.0
        assert trainer.bc_trainer.batch_size == batch_size
        assert trainer.bc_trainer.minibatch_size == minibatch_size
        return result

    monkeypatch.setattr(run_experiment, "_inner_train", train)
    result = run_experiment.run_single(config)
    assert trained_sizes == [m, 2 * m, 3 * m, 4 * m]
    assert [row["n_observations"] for row in result["per_round"]] == [
        0,
        m,
        2 * m,
        3 * m,
        4 * m,
    ]
    evaluated = [row["round"] for row in result["per_round"] if "checkpoint" in row]
    assert evaluated == [0, 1, 2, 4]
    for row in result["per_round"][1:]:
        assert row["trajectories_collected_this_round"] == expected_trajectories
        assert row["collection_steps"] == row["round"] * expected_trajectories * 7
    states = coverage_data.load_algo_states(
        config.output_dir, config.env_name, "bc_iid", 7
    )
    assert len(states.obs) == 4 * m
    # A learned-policy rollout would increase the second coordinate.
    np.testing.assert_array_equal(states.obs[:, 1], 0)
    assert np.any(states.obs[:, 0] > 0)
    for path in coverage_data.scratch_demo_root(
        config.output_dir, "bc_iid", config.env_name, 7
    ).glob("round-*/*.npz"):
        for traj in serialize.load(path):
            np.testing.assert_array_equal(traj.acts, 1)
    assert "expert_dataset" not in result


@pytest.mark.parametrize("strategy", [None, "prefix", "uniform"])
def test_fixed_bc_defaults_to_temporal_prefix(tiny_experiment, strategy):
    kwargs = {} if strategy is None else {"subsample_strategy": strategy}
    config = run_experiment.ExperimentConfig(
        **{
            **{
                k: v
                for k, v in dataclasses.asdict(tiny_experiment).items()
                if k != "subsample_strategy"
            },
            "algo": "bc",
            "n_rounds": 1,
            "samples_per_round": 5,
            **kwargs,
        }
    )
    result = run_experiment.run_single(config)
    states = coverage_data.load_algo_states(config.output_dir, config.env_name, "bc", 7)
    if strategy != "uniform":
        np.testing.assert_array_equal(states.obs[:, 0], [0, 1, 2, 3, 4])
        assert result["expert_dataset"]["strategy"] == "prefix"
    else:
        assert not np.array_equal(states.obs[:, 0], [0, 1, 2, 3, 4])
        assert result["expert_dataset"]["strategy"] == "uniform"


def test_bc_prefix_preserves_collection_order_across_episodes(
    tiny_experiment, monkeypatch
):
    class EpisodeOrderEnv(ActionHistoryEnv):
        observation_space = gym.spaces.Box(0, np.inf, shape=(2,), dtype=np.float32)

        def reset(self, **kwargs):
            self.episode = getattr(self, "episode", 0) + 1
            obs, info = super().reset(**kwargs)
            obs[1] = self.episode
            return obs, info

        def step(self, action):
            obs, reward, done, truncated, info = super().step(action)
            obs[1] = self.episode
            return obs, reward, done, truncated, info

    monkeypatch.setattr(
        run_experiment.env_utils,
        "make_env",
        lambda *args, **kwargs: DummyVecEnv([EpisodeOrderEnv]),
    )
    config = dataclasses.replace(
        tiny_experiment,
        algo="bc",
        seed=3,
        n_rounds=1,
        samples_per_round=10,
        subsample_strategy="prefix",
    )
    result = run_experiment.run_single(config)
    with np.load(result["expert_dataset"]["path"]) as saved:
        np.testing.assert_array_equal(
            saved["obs"][:, 0], [0, 1, 2, 3, 4, 5, 6, 0, 1, 2]
        )
        np.testing.assert_array_equal(
            saved["obs"][:, 1], [1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        )


def test_cli_defaults_and_trajectory_option(tiny_experiment, monkeypatch):
    seen = []
    monkeypatch.setattr(
        run_experiment, "run_single", lambda config: seen.append(config) or {}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment",
            "--envs",
            "CartPole-v1",
            "--seeds",
            "1",
            "--output-dir",
            str(tiny_experiment.output_dir),
            "--traj-per-round",
            "3",
        ],
    )
    run_experiment.main()
    assert {c.algo: c.subsample_strategy for c in seen} == {
        "ftl": "uniform",
        "ftrl": "uniform",
        "bc": "prefix",
        "bc_iid": "uniform",
    }
    assert all(c.trajectories_per_round == 3 for c in seen)


def test_bc_iid_rejects_prefix_sampling(tiny_experiment):
    with pytest.raises(ValueError, match="uniform"):
        dataclasses.replace(tiny_experiment, subsample_strategy="prefix")


@pytest.mark.parametrize("warm_start", [False, True])
@pytest.mark.parametrize("policy_mode", ["linear", "end_to_end"])
def test_bc_iid_trains_and_respects_warm_start(
    tiny_experiment, monkeypatch, warm_start, policy_mode
):
    from imitation.algorithms.ftrl import FTRLTrainer
    from imitation.experiments.ftrl.policy_utils import load_policy_checkpoint

    resets = []
    original_reset = FTRLTrainer._reinitialize_trainable_params

    def reset(trainer):
        original_reset(trainer)
        resets.append(trainer.round_num)

    monkeypatch.setattr(FTRLTrainer, "_reinitialize_trainable_params", reset)
    config = dataclasses.replace(
        tiny_experiment,
        warm_start=warm_start,
        policy_mode=policy_mode,
        learning_rate=0.01,
        n_rounds=3,
        eval_interval=1,
    )
    result = run_experiment.run_single(config)
    assert len(resets) == (0 if warm_start else 2)
    initial = load_policy_checkpoint(result["per_round"][0]["checkpoint"])
    trained = load_policy_checkpoint(result["per_round"][1]["checkpoint"])
    assert not th.equal(initial.action_net.weight, trained.action_net.weight)
    if policy_mode == "linear":
        for name, value in initial.state_dict().items():
            if not name.startswith("action_net"):
                assert th.equal(value, trained.state_dict()[name])


def test_bc_iid_outer_stop_preserves_final_trained_budget(tiny_experiment):
    config = dataclasses.replace(
        tiny_experiment,
        n_rounds=10,
        eval_interval=1,
        outer_early_stop=True,
        outer_early_stop_patience=2,
        outer_early_stop_disagreement_ceiling=1.0,
    )
    result = run_experiment.run_single(config)
    assert [row["round"] for row in result["per_round"]] == [0, 1, 2, 3]
    assert result["per_round"][-1]["n_observations"] == 9
    assert "checkpoint" in result["per_round"][-1]
    states = coverage_data.load_algo_states(
        config.output_dir, config.env_name, "bc_iid", 7
    )
    assert len(states.obs) == 9
