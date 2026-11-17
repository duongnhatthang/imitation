"""Tests for env_baselines: reference scores and expert quality validation."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from imitation.experiments.ftrl import env_baselines
from imitation.experiments.ftrl.env_baselines import (
    REFERENCE_BASELINES,
    load_or_compute_baselines,
    validate_expert_quality,
)


class TestReferenceBaselines:
    """Test that reference baselines are complete and sane."""

    def test_all_classical_envs_have_baselines(self):
        from imitation.experiments.ftrl.env_utils import ENV_CONFIGS

        for env_name in ENV_CONFIGS:
            assert (
                env_name in REFERENCE_BASELINES
            ), f"{env_name} missing from REFERENCE_BASELINES"

    def test_all_atari_envs_have_baselines(self):
        from imitation.experiments.ftrl.atari_utils import ATARI_CONFIGS

        for env_name in ATARI_CONFIGS:
            assert (
                env_name in REFERENCE_BASELINES
            ), f"{env_name} missing from REFERENCE_BASELINES"

    def test_expert_better_than_random(self):
        for env_name, bl in REFERENCE_BASELINES.items():
            assert bl["expert_score"] > bl["random_score"], (
                f"{env_name}: expert ({bl['expert_score']}) should be > "
                f"random ({bl['random_score']})"
            )


class TestValidateExpertQuality:
    """Test expert quality validation logic."""

    def test_good_expert_passes(self):
        is_ok, msg = validate_expert_quality("CartPole-v1", 490.0)
        assert is_ok, msg

    def test_bad_expert_fails(self):
        is_ok, msg = validate_expert_quality("CartPole-v1", 50.0)
        assert not is_ok
        assert "below" in msg.lower() or "threshold" in msg.lower()

    def test_unknown_env_passes(self):
        is_ok, msg = validate_expert_quality("UnknownEnv-v0", 100.0)
        assert is_ok


CLIFF = "CliffWalking-v0"


class TestCliffBaselinesCache:
    """CliffWalking baselines keep the uniform random return, cached or fresh."""

    def _write(self, tmp_path, record):
        path = tmp_path / CLIFF / "baselines.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record, indent=2))
        return path

    def _fail_compute(self, *args, **kwargs):
        raise AssertionError("cached baselines must not be recomputed")

    def test_cached_uniform_random_baselines_are_returned_unchanged(
        self,
        tmp_path,
        monkeypatch,
    ):
        record = {
            "expert_return": -13.0,
            "random_return": -7123.25,
            "expert_self_ce": 0.01,
        }
        path = self._write(tmp_path, record)
        before = path.read_bytes()
        monkeypatch.setattr(env_baselines, "compute_baselines", self._fail_compute)
        rng = np.random.default_rng(0)
        loaded = load_or_compute_baselines(CLIFF, None, None, tmp_path, rng)
        assert loaded == record
        assert path.read_bytes() == before

    def test_historical_stalled_record_is_read_without_mutation(
        self,
        tmp_path,
        monkeypatch,
    ):
        # Diagnostic-only records from the withdrawn stalled reference.
        record = {
            "expert_return": -13.0,
            "random_return": -200.0,
            "expert_self_ce": 0.01,
            "measured_uniform_random_return": -7123.25,
            "normalization_reference_type": "stalled_policy",
            "stalled_return": -200.0,
        }
        path = self._write(tmp_path, record)
        before = path.read_bytes()
        monkeypatch.setattr(env_baselines, "compute_baselines", self._fail_compute)
        rng = np.random.default_rng(0)
        assert load_or_compute_baselines(CLIFF, None, None, tmp_path, rng) == record
        assert path.read_bytes() == before

    def test_fresh_baselines_keep_the_measured_uniform_random_return(
        self,
        tmp_path,
        monkeypatch,
    ):
        from imitation.experiments.ftrl import eval_utils

        expert = SimpleNamespace(mean_return=-13.0, current_round_ce=0.02)
        monkeypatch.setattr(
            eval_utils,
            "eval_policy_rollout",
            lambda *args, **kwargs: expert,
        )
        monkeypatch.setattr(
            env_baselines,
            "compute_random_return",
            lambda venv, n_episodes: -4321.5,
        )
        rng = np.random.default_rng(0)
        fresh = load_or_compute_baselines(CLIFF, None, None, tmp_path, rng)
        expected = {
            "expert_return": -13.0,
            "random_return": -4321.5,
            "expert_self_ce": 0.02,
        }
        assert fresh == expected
        cached = json.loads((tmp_path / CLIFF / "baselines.json").read_text())
        assert cached == expected


from stable_baselines3.common.evaluation import evaluate_policy

from imitation.experiments.ftrl.env_utils import ENV_CONFIGS, make_env
from imitation.experiments.ftrl.experts import get_or_train_expert


@pytest.mark.expensive
class TestExpertQualityHardFail:
    """Fail hard if any trained expert is degenerate."""

    @pytest.mark.parametrize("env_name", list(ENV_CONFIGS.keys()))
    def test_expert_meets_quality_threshold(self, env_name, tmp_path):
        rng = np.random.default_rng(42)
        venv = make_env(env_name, n_envs=1, rng=rng)
        expert = get_or_train_expert(
            env_name,
            venv,
            cache_dir=tmp_path,
            rng=rng,
            seed=42,
        )
        mean_return, _ = evaluate_policy(
            expert,
            venv,
            n_eval_episodes=20,
            deterministic=True,
        )
        venv.close()

        is_ok, msg = validate_expert_quality(env_name, mean_return)
        assert is_ok, msg
