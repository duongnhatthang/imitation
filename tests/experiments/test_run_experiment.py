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
        assert "cross_entropy" in m
        assert "l2_norm" in m
        assert "round" in m


def test_run_bc_cartpole(tmp_path):
    """BC baseline smoke test on CartPole."""
    config = _make_config("bc", tmp_path)
    result = run_single(config)

    assert result["algo"] == "bc"
    assert len(result["per_round"]) >= 2
    for m in result["per_round"]:
        assert m["cross_entropy"] >= 0


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


def test_run_ftrl_lunarlander(tmp_path):
    """FTRL smoke test on LunarLander-v2 (new classical MDP)."""
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


def test_run_bc_taxi(tmp_path):
    """BC smoke test on Taxi-v3 (discrete obs, one-hot encoded)."""
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
