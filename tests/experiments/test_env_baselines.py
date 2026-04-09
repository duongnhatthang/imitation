"""Tests for env_baselines: reference scores and expert quality validation."""

import numpy as np
import pytest

from imitation.experiments.ftrl.env_baselines import (
    REFERENCE_BASELINES,
    validate_expert_quality,
)


class TestReferenceBaselines:
    """Test that reference baselines are complete and sane."""

    def test_all_classical_envs_have_baselines(self):
        from imitation.experiments.ftrl.env_utils import ENV_CONFIGS

        for env_name in ENV_CONFIGS:
            assert env_name in REFERENCE_BASELINES, (
                f"{env_name} missing from REFERENCE_BASELINES"
            )

    def test_all_atari_envs_have_baselines(self):
        from imitation.experiments.ftrl.atari_utils import ATARI_CONFIGS

        for env_name in ATARI_CONFIGS:
            assert env_name in REFERENCE_BASELINES, (
                f"{env_name} missing from REFERENCE_BASELINES"
            )

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
