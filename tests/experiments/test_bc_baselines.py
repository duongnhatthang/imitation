"""Contracts for the offline BC baselines and per-trajectory round sampling.

The three BC variants routed through the shared round-loop trainer differ
only in which transitions arrive at round t:

* ``bc_iid``    - one uniform state from each of t freshly collected expert
                  episodes, so the labels are t independent draws from d^{pi^E}
* ``bc_pool``   - t uniform draws from ONE fixed pool of ceil(N/H) episodes
                  (the behavior of the removed ``bc_dagger``)
* ``bc_prefix`` - the first t transitions of that same fixed pool, in order
                  (byte-for-byte fixed BC's dataset, consumed sequentially)

These tests pin that "only the data differs" property, because the whole point
of the comparison is that the optimizer is held fixed.
"""

import dataclasses

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import rollout
from imitation.experiments.ftrl import coverage_data, run_experiment

EPISODE_LEN = 7


class ActionHistoryEnv(gym.Env):
    """Elapsed time and the number of non-expert actions, both in the state."""

    observation_space = gym.spaces.Box(0, np.inf, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.elapsed = self.mistakes = 0
        return np.array([0, 0], dtype=np.float32), {}

    def step(self, action):
        self.elapsed += 1
        self.mistakes += int(action == 0)
        obs = np.array([self.elapsed, self.mistakes], dtype=np.float32)
        return obs, float(action == 1), self.elapsed == EPISODE_LEN, False, {}


class EpisodeOrderEnv(ActionHistoryEnv):
    """Same, but obs[1] is the episode index so a state's source is identifiable."""

    def reset(self, **kwargs):
        self.episode = getattr(self, "episode", 0) + 1
        obs, info = super().reset(**kwargs)
        obs[1] = self.episode
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        obs[1] = self.episode
        return obs, reward, done, truncated, info


def constant_policy(obs_space, act_space, action):
    policy = ActorCriticPolicy(obs_space, act_space, lr_schedule=lambda _: 0.0)
    with th.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.fill_(-20)
        policy.action_net.bias[action] = 20
    return policy


def _install_tiny_env(monkeypatch, env_cls):
    """Point the runner at ``env_cls`` and stub out expert training/baselines."""
    from imitation.experiments.ftrl import env_baselines

    expert = constant_policy(env_cls.observation_space, env_cls.action_space, 1)
    monkeypatch.setattr(
        run_experiment.env_utils,
        "make_env",
        lambda *args, **kwargs: DummyVecEnv([env_cls]),
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


def _config(tmp_path, **overrides):
    base = dict(
        algo="bc_iid",
        env_name="CartPole-v1",
        seed=7,
        policy_mode="end_to_end",
        n_rounds=4,
        samples_per_round=1,
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
    base.update(overrides)
    return run_experiment.ExperimentConfig(**base)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    _install_tiny_env(monkeypatch, ActionHistoryEnv)
    return tmp_path


@pytest.fixture
def tiny_ordered(tmp_path, monkeypatch):
    _install_tiny_env(monkeypatch, EpisodeOrderEnv)
    return tmp_path


# --------------------------------------------------------------- item 1 -----


@pytest.mark.parametrize("m", [1, 2, 3])
def test_round_samples_come_from_distinct_trajectories(tiny_ordered, m):
    """m retained samples must be drawn from m different episodes.

    Pooling a round's transitions and drawing m from the pool lets one episode
    contribute several of them, which is exactly the dependence this sampling
    change removes. obs[1] carries the episode index, so a repeat is visible.
    """
    config = _config(tiny_ordered, algo="bc_iid", n_rounds=3, samples_per_round=m)
    assert config.trajectories_per_round == m
    run_experiment.run_single(config)
    root = coverage_data.scratch_demo_root(
        config.output_dir, "bc_iid", config.env_name, config.seed
    )
    for round_dir in sorted(root.glob("round-*")):
        episodes = []
        for path in sorted(round_dir.glob("*.npz")):
            from imitation.data import serialize

            for traj in serialize.load(path):
                assert len(traj) == 1, "a retained sample must be a single step"
                episodes.append(float(traj.obs[0][1]))
        assert len(episodes) == m
        assert len(set(episodes)) == m, f"{round_dir.name}: repeated episode {episodes}"


def test_trajectories_per_round_is_raised_to_samples_per_round(tmp_path):
    config = _config(
        tmp_path, algo="ftl", samples_per_round=5, trajectories_per_round=2
    )
    assert config.trajectories_per_round == 5
    # An explicit larger request is honored: more episodes, still one state each.
    config = _config(
        tmp_path, algo="ftl", samples_per_round=2, trajectories_per_round=9
    )
    assert config.trajectories_per_round == 9


def test_per_trajectory_is_a_no_op_at_one_sample_per_round(tmp_path):
    """The campaign setting m = 1 must be bit-identical to the old pooled draw.

    This is the claim that lets us say the sampling change cannot account for
    any difference between the existing m = 1 runs: with one trajectory in the
    round there is nothing to choose between trajectories, so the new path takes
    no extra draw and lands on the same transition from the same RNG stream.
    """
    venv = DummyVecEnv([EpisodeOrderEnv])
    trajs = rollout.generate_trajectories(
        policy=constant_policy(
            EpisodeOrderEnv.observation_space, EpisodeOrderEnv.action_space, 1
        ),
        venv=venv,
        sample_until=rollout.make_sample_until(min_episodes=3),
        deterministic_policy=True,
        rng=np.random.default_rng(0),
    )
    from imitation.algorithms.dagger import _save_dagger_demo

    def build(per_trajectory, n_target, n_trajs, tag):
        round_dir = tmp_path / f"round-{tag}"
        round_dir.mkdir(parents=True)
        for i, traj in enumerate(trajs[:n_trajs]):
            _save_dagger_demo(traj, i, round_dir, np.random.default_rng(1))
        run_experiment._uniform_round_demos(
            round_dir,
            n_target,
            np.random.default_rng(5),
            per_trajectory=per_trajectory,
        )
        from imitation.data import serialize

        out = []
        for path in sorted(round_dir.glob("*.npz")):
            out.extend(t.obs[0].tolist() for t in serialize.load(path))
        return sorted(out)

    # m = 1, one trajectory per round: identical transition, identical stream.
    assert build(True, 1, 1, "new-1") == build(False, 1, 1, "old-1")
    # m = 3 from 3 trajectories: one state each, so three distinct episodes.
    new = build(True, 3, 3, "new-3")
    assert len({row[1] for row in new}) == 3
    # The pooled path is still reachable and still returns the right count.
    assert len(build(False, 3, 3, "old-3")) == 3


# ------------------------------------------------- items 4 and the ablation --


def _round_states(config, algo):
    states = coverage_data.load_algo_states(
        config.output_dir, config.env_name, algo, config.seed
    )
    return states.obs, states.rounds


def test_bc_prefix_trains_on_fixed_bc_prefix(tiny_ordered):
    """bc_prefix at round t must hold exactly fixed BC's first t transitions."""
    budget = 10
    bc_cfg = _config(tiny_ordered, algo="bc", n_rounds=1, samples_per_round=budget)
    bc_result = run_experiment.run_single(bc_cfg)

    prefix_cfg = _config(
        tiny_ordered, algo="bc_prefix", n_rounds=budget, samples_per_round=1
    )
    prefix_result = run_experiment.run_single(prefix_cfg)

    # Same artifact, not merely the same numbers.
    assert prefix_result["expert_dataset"]["sha256"] == (
        bc_result["expert_dataset"]["sha256"]
    )
    assert prefix_result["expert_dataset"]["strategy"] == "prefix"

    with np.load(bc_result["expert_dataset"]["path"]) as saved:
        dataset_obs = saved["obs"]
    obs, rounds = _round_states(prefix_cfg, "bc_prefix")
    order = np.argsort(rounds, kind="stable")
    np.testing.assert_array_equal(obs[order], dataset_obs[:budget])
    # One transition per round, no round skipped or doubled. Demo directories
    # are numbered from the trainer's 0-based round counter, as for FTL.
    np.testing.assert_array_equal(np.sort(rounds), np.arange(budget))


def test_bc_pool_draws_from_the_same_fixed_pool(tiny_ordered):
    """bc_pool's rounds are uniform draws from one fixed pool, not fresh episodes."""
    budget = 10
    bc_result = run_experiment.run_single(
        _config(tiny_ordered, algo="bc", n_rounds=1, samples_per_round=budget)
    )
    pool_cfg = _config(
        tiny_ordered, algo="bc_pool", n_rounds=budget, samples_per_round=1
    )
    result = run_experiment.run_single(pool_cfg)
    assert result["expert_dataset"]["strategy"] == "uniform"

    with np.load(result["expert_dataset"]["path"]) as saved:
        dataset_obs = saved["obs"]
    with np.load(bc_result["expert_dataset"]["path"]) as saved:
        canonical_obs = saved["obs"]
    # The second episode overshoots the budget by four rows. Pool sampling must
    # permute the same ten rows as BC, never sample those discarded tail rows.
    assert sorted(map(tuple, dataset_obs)) == sorted(map(tuple, canonical_obs))
    assert not np.array_equal(dataset_obs, canonical_obs)
    obs, rounds = _round_states(pool_cfg, "bc_pool")
    order = np.argsort(rounds, kind="stable")
    np.testing.assert_array_equal(obs[order], dataset_obs[:budget])

    # The defining property of the ablation: every state came from the small set
    # of episodes collected once up front, so the number of distinct source
    # episodes is bounded by the pool, not by the number of rounds.
    episodes = set(obs[:, 1].tolist())
    assert len(episodes) <= int(np.ceil(budget / EPISODE_LEN)) + 1


def test_offline_baselines_never_collect_during_rounds(tiny_ordered, monkeypatch):
    """No DAgger collector is ever opened: the dataset is built once, up front.

    ``generate_trajectories`` alone is not the right probe, because on-policy
    evaluation rolls out too. Opening a trajectory collector is what a round of
    *collection* does, and an offline baseline must never do it.
    """
    from imitation.algorithms.ftrl import FTRLTrainer

    collectors = []
    original = FTRLTrainer.create_trajectory_collector

    def counted(self, *args, **kwargs):
        collectors.append(self.round_num)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FTRLTrainer, "create_trajectory_collector", counted)
    config = _config(
        tiny_ordered, algo="bc_prefix", n_rounds=5, samples_per_round=1, eval_interval=1
    )
    result = run_experiment.run_single(config)
    assert collectors == []
    rows = result["per_round"][1:]
    assert all(row["trajectories_collected_this_round"] == 0 for row in rows)
    # Expert-interaction cost is a constant, not something that grows per round.
    assert len({row["collection_steps"] for row in rows}) == 1


@pytest.mark.parametrize("algo", ["bc_iid", "bc_prefix", "bc_pool"])
def test_bc_variants_share_trainer_configuration(tiny_ordered, monkeypatch, algo):
    """Only the data may differ: L2, warm start and batch size must match."""
    seen = []
    original_train = run_experiment._inner_train

    def train(trainer, *args, **kwargs):
        seen.append(
            (
                trainer.bc_trainer.loss_calculator.l2_weight,
                trainer.bc_trainer.batch_size,
                trainer.warm_start,
                trainer.beta_schedule(trainer.round_num),
            )
        )
        return original_train(trainer, *args, **kwargs)

    monkeypatch.setattr(run_experiment, "_inner_train", train)
    config = _config(tiny_ordered, algo=algo, n_rounds=3, samples_per_round=1)
    run_experiment.run_single(config)
    assert seen, "trainer was never invoked"
    for l2_weight, batch_size, warm_start, beta in seen:
        assert l2_weight == 0.0
        assert batch_size == 1
        assert warm_start is True
        assert beta == 1.0


@pytest.mark.parametrize(
    "algo,bad", [("bc_iid", "prefix"), ("bc_pool", "prefix"), ("bc_prefix", "uniform")]
)
def test_sampling_strategy_is_enforced_per_algo(tmp_path, algo, bad):
    with pytest.raises(ValueError):
        _config(tmp_path, algo=algo, subsample_strategy=bad)


@pytest.mark.parametrize(
    "algo,expected",
    [
        ("ftl", "uniform"),
        ("ftrl", "uniform"),
        ("bc", "prefix"),
        ("bc_iid", "uniform"),
        ("bc_prefix", "prefix"),
        ("bc_pool", "uniform"),
    ],
)
def test_default_sampling_strategy_per_algo(tmp_path, algo, expected):
    assert _config(tmp_path, algo=algo).subsample_strategy == expected


# ---------------------------------------------------------- reproducibility --


@pytest.mark.parametrize("algo", ["ftl", "bc_iid", "bc_prefix", "bc_pool"])
def test_same_seed_reproduces_the_same_run(tmp_path, monkeypatch, algo):
    def run(tag):
        _install_tiny_env(monkeypatch, EpisodeOrderEnv)
        config = _config(
            tmp_path / tag,
            algo=algo,
            n_rounds=4,
            samples_per_round=2,
            eval_interval=1,
            learning_rate=0.01,
        )
        rows = run_experiment.run_single(config)["per_round"]
        return [
            (row["round"], row.get("disagreement_rate"), row.get("train_cross_entropy"))
            for row in rows
        ]

    assert run("a") == run("b")


def test_expert_training_seed_is_pinned(tmp_path, monkeypatch):
    """The shared expert must not depend on which cell missed the cache first."""
    from imitation.experiments.ftrl import env_baselines

    seeds = []
    expert = constant_policy(
        ActionHistoryEnv.observation_space, ActionHistoryEnv.action_space, 1
    )
    monkeypatch.setattr(
        run_experiment.env_utils,
        "make_env",
        lambda *args, **kwargs: DummyVecEnv([ActionHistoryEnv]),
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
    monkeypatch.setattr(
        run_experiment.experts,
        "get_or_train_expert",
        lambda *a, **kw: seeds.append(kw["seed"]) or expert,
    )
    for seed in (0, 3, 11):
        run_experiment.run_single(
            _config(tmp_path / f"s{seed}", algo="bc_iid", seed=seed, n_rounds=1)
        )
    assert seeds == [run_experiment.EXPERT_TRAINING_SEED] * 3


# ---------------------------------------------------------------- item 3 -----


def test_coverage_diff_sides_are_configurable(tmp_path):
    from imitation.experiments.ftrl import plot_tsne_coverage as P

    assert P.COVERAGE_DIFF_SIDES == {"left": ("ftl",), "right": ("bc_iid",)}
    rng = np.random.default_rng(0)
    embedding = rng.normal(size=(60, 2))
    labels = np.array(["ftl"] * 20 + ["bc_iid"] * 20 + ["bc"] * 20, dtype=object)

    default_path = tmp_path / "default.png"
    P.render_coverage_diff(embedding, labels, default_path, "Tiny-v0", n_bins=8)
    assert default_path.exists()

    custom_path = tmp_path / "custom.png"
    P.render_coverage_diff(
        embedding,
        labels,
        custom_path,
        "Tiny-v0",
        n_bins=8,
        left_algos=("ftl", "ftrl"),
        right_algos=("bc",),
    )
    assert custom_path.exists()

    # A missing side is skipped with a warning rather than crashing the sweep.
    missing_path = tmp_path / "missing.png"
    P.render_coverage_diff(
        embedding,
        labels,
        missing_path,
        "Tiny-v0",
        n_bins=8,
        right_algos=("bc_pool",),
    )
    assert not missing_path.exists()
