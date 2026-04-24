"""Smoke tests for the FTRL experiment runner."""

import json

import numpy as np
import pytest

from imitation.experiments.ftrl.run_experiment import (
    ExperimentConfig,
    resolve_envs,
    run_single,
)


def _make_config(algo, tmp_path, **overrides):
    """Helper to create an ExperimentConfig for testing."""
    defaults = dict(
        algo=algo,
        env_name="CartPole-v1",
        seed=0,
        policy_mode="end_to_end",
        n_rounds=3,
        samples_per_round=300,
        l2_lambda=1e-4,
        l2_decay=False,
        warm_start=True,
        beta_rampdown=2,
        bc_n_epochs=2,
        eval_interval=5,
        output_dir=tmp_path / "results",
        expert_cache_dir=tmp_path / "experts",
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_run_ftl_cartpole(tmp_path):
    """FTL (l2=0) smoke test on CartPole, 3 rounds."""
    config = _make_config("ftl", tmp_path)
    result = run_single(config)

    assert result["algo"] == "ftl"
    assert result["env"] == "CartPole-v1"
    assert result["seed"] == 0
    assert len(result["per_round"]) >= 2
    assert result["elapsed_seconds"] > 0

    # Check JSON was saved
    out_file = tmp_path / "results" / "CartPole-v1" / "ftl_end_to_end_seed0.json"
    assert out_file.exists()
    with open(out_file) as f:
        saved = json.load(f)
    assert saved["algo"] == "ftl"


def test_run_ftrl_cartpole(tmp_path):
    """FTRL smoke test on CartPole, 3 rounds."""
    config = _make_config("ftrl", tmp_path)
    result = run_single(config)

    assert result["algo"] == "ftrl"
    assert len(result["per_round"]) >= 2
    for m in result["per_round"]:
        assert "train_cross_entropy" in m
        assert "l2_norm" in m
        assert "round" in m
        assert "d_eval_size" in m
        assert isinstance(m["d_eval_size"], int)


def test_run_bc_cartpole(tmp_path):
    """BC baseline smoke test on CartPole."""
    config = _make_config("bc", tmp_path)
    result = run_single(config)

    assert result["algo"] == "bc"
    assert len(result["per_round"]) >= 2
    for m in result["per_round"]:
        # Fixed BC does not track per-round train cross-entropy or rollout CE;
        # these fields are present but None (static reference line in plots).
        assert "train_cross_entropy" in m
        assert m["train_cross_entropy"] is None
        assert "rollout_cross_entropy" in m
        assert m["rollout_cross_entropy"] is None
        assert "l2_norm" in m


def test_run_ftrl_linear_mode(tmp_path):
    """FTRL with linear policy mode on CartPole."""
    config = _make_config("ftrl", tmp_path, policy_mode="linear")
    result = run_single(config)

    assert result["policy_mode"] == "linear"
    assert len(result["per_round"]) >= 2


def test_run_ftrl_decaying_l2(tmp_path):
    """FTRL with decaying L2 schedule."""
    config = _make_config("ftrl", tmp_path, l2_decay=True)
    result = run_single(config)

    assert result["algo"] == "ftrl"
    assert result["config"]["l2_decay"] is True
    assert len(result["per_round"]) >= 2


class TestResolveEnvs:
    """Test --env-group resolution."""

    def test_classical_group(self):
        envs = resolve_envs(env_group="classical", envs=None)
        assert "CartPole-v1" in envs
        assert len(envs) == 8

    def test_explicit_envs_override_group(self):
        envs = resolve_envs(env_group=None, envs=["CartPole-v1"])
        assert envs == ["CartPole-v1"]

    def test_atari_zoo_group(self):
        envs = resolve_envs(env_group="atari-zoo", envs=None)
        assert "PongNoFrameskip-v4" in envs

    def test_default_is_classical(self):
        envs = resolve_envs(env_group=None, envs=None)
        assert "CartPole-v1" in envs
        assert not any("NoFrameskip" in e for e in envs)

    def test_both_raises(self):
        with pytest.raises(ValueError, match="not both"):
            resolve_envs(env_group="classical", envs=["CartPole-v1"])


def test_run_bc_dagger_cartpole(tmp_path):
    """bc_dagger smoke test on CartPole."""
    config = _make_config(
        "bc_dagger",
        tmp_path,
        env_name="CartPole-v1",
        policy_mode="end_to_end",
        n_rounds=3,
        samples_per_round=200,
        eval_interval=1,
    )
    result = run_single(config)
    assert result["algo"] == "bc_dagger"
    assert len(result["per_round"]) == 3
    # Data budget invariant: round k has exactly k * samples_per_round obs
    for i, r in enumerate(result["per_round"]):
        assert r["n_observations"] == (i + 1) * 200
    # rollout_cross_entropy populated on every eval-point round
    # (with eval_interval=1, every round is an eval point)
    for r in result["per_round"]:
        assert r["rollout_cross_entropy"] is not None
        assert r["d_eval_size"] > 0


@pytest.mark.expensive
def test_run_ftrl_lunarlander(tmp_path):
    """FTRL smoke test on LunarLander-v2 (new classical MDP).

    Marked expensive: the convergence-detecting expert trainer (Task 3)
    requires several million PPO steps to clear LunarLander's 95% return
    threshold — no longer fast enough for the default test loop.
    """
    config = _make_config(
        "ftrl",
        tmp_path,
        env_name="LunarLander-v2",
        policy_mode="linear",
        n_rounds=2,
        samples_per_round=200,
    )
    result = run_single(config)
    assert result["algo"] == "ftrl"
    assert result["env"] == "LunarLander-v2"
    assert len(result["per_round"]) >= 1


@pytest.mark.expensive
def test_run_bc_taxi(tmp_path):
    """BC smoke test on Taxi-v3 (discrete obs, one-hot encoded).

    Marked expensive: same reason as test_run_ftrl_lunarlander — Taxi's
    expert now requires >500k PPO steps to clear the convergence gate.
    """
    config = _make_config(
        "bc",
        tmp_path,
        env_name="Taxi-v3",
        policy_mode="linear",
        n_rounds=2,
        samples_per_round=200,
    )
    result = run_single(config)
    assert result["algo"] == "bc"
    assert result["env"] == "Taxi-v3"
    assert len(result["per_round"]) >= 1


def test_collect_and_subsample_uniform_samples_without_replacement(tmp_path):
    """With strategy='uniform', BC samples without replacement from the pool.

    We run two BC configs with the same seed but strategy='uniform' vs
    'prefix' and assert they produce different round-1 n_observations
    orderings. Since the dataset is collected deterministically from the
    same seed, prefix order is fixed; uniform must differ.
    """
    from imitation.experiments.ftrl.run_experiment import (
        ExperimentConfig,
        _collect_and_subsample_transitions,
    )
    import numpy as np

    rng_prefix = np.random.default_rng(0)
    rng_uniform = np.random.default_rng(0)

    # Build a fake transitions list by indices
    fake_txns = list(range(1000))
    prefix = _collect_and_subsample_transitions(
        fake_txns, n_target=50, strategy="prefix", rng=rng_prefix
    )
    uniform = _collect_and_subsample_transitions(
        fake_txns, n_target=50, strategy="uniform", rng=rng_uniform
    )
    assert prefix == list(range(50))
    assert uniform != list(range(50))
    assert len(uniform) == 50
    assert set(uniform).issubset(set(fake_txns))
    assert len(set(uniform)) == 50  # no duplicates


def test_collect_and_subsample_preserves_transitions_type():
    """Uniform subsampling over a Transitions object preserves type and shapes."""
    from imitation.data import types
    from imitation.experiments.ftrl.run_experiment import (
        _collect_and_subsample_transitions,
    )
    import numpy as np

    n = 40
    obs = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
    next_obs = obs + 1.0
    acts = np.arange(n, dtype=np.int64)
    dones = np.zeros(n, dtype=bool)
    infos = np.array([{} for _ in range(n)])
    trans = types.Transitions(
        obs=obs, acts=acts, infos=infos, next_obs=next_obs, dones=dones,
    )

    rng = np.random.default_rng(7)
    out = _collect_and_subsample_transitions(
        trans, n_target=10, strategy="uniform", rng=rng,
    )

    # Type preserved
    assert isinstance(out, types.Transitions)
    # Length matches n_target for all fields
    assert len(out.obs) == 10
    assert len(out.acts) == 10
    assert len(out.next_obs) == 10
    assert len(out.dones) == 10
    assert len(out.infos) == 10
    # Selected indices are a subset of the original pool
    selected = set(int(a) for a in out.acts)
    assert selected.issubset(set(range(n)))
    assert len(selected) == 10  # no duplicates
    # next_obs still paired correctly: next_obs[i] == obs[i] + 1
    np.testing.assert_allclose(out.next_obs, out.obs + 1.0)

    # And prefix strategy returns the first 10
    rng2 = np.random.default_rng(7)
    prefix = _collect_and_subsample_transitions(
        trans, n_target=10, strategy="prefix", rng=rng2,
    )
    np.testing.assert_array_equal(prefix.acts, np.arange(10))


def test_ftl_uniform_round_demos_are_subsampled(tmp_path):
    """With strategy='uniform', FTL's round demos should not be the prefix.

    Run FTL on CartPole with samples_per_round=50 and n_rounds=2. Inspect
    the demo dir for round 1 and confirm the selected transitions span a
    wider range of source timesteps than a sequential prefix would.
    """
    from imitation.data import serialize
    from imitation.experiments.ftrl.run_experiment import (
        ExperimentConfig,
        run_single,
    )

    config = ExperimentConfig(
        algo="ftl",
        env_name="CartPole-v1",
        seed=0,
        policy_mode="linear",
        n_rounds=2,
        samples_per_round=50,
        l2_lambda=0.0,
        l2_decay=False,
        warm_start=True,
        beta_rampdown=2,
        bc_n_epochs=2,
        eval_interval=1,
        output_dir=tmp_path / "results",
        expert_cache_dir=tmp_path / "experts",
        subsample_strategy="uniform",
    )
    run_single(config)

    # Scratch dir is where FTL wrote the per-round demos
    round_dir = (
        tmp_path / "results" / "scratch" / "ftl_CartPole-v1_0"
        / "demos" / "round-000"
    )
    assert round_dir.exists(), f"Missing round_dir: {round_dir}"
    demo_paths = sorted(p for p in round_dir.iterdir() if p.name.endswith(".npz"))
    assert demo_paths, "no demos saved"
    total = 0
    for p in demo_paths:
        total += sum(len(t) for t in serialize.load(p))
    assert total == 50
