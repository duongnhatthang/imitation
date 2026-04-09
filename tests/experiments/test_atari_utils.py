"""Tests for atari_utils: config, env creation, hub downloads."""

import pytest

from imitation.experiments.ftrl.atari_utils import (
    ATARI_CONFIGS,
    get_atari_env_id,
)


class TestAtariConfigs:
    """Test Atari game configurations."""

    def test_all_tiers_present(self):
        tiers = {cfg["tier"] for cfg in ATARI_CONFIGS.values()}
        assert tiers == {1, 2, 3}

    def test_tier1_has_hub_repo(self):
        for env_id, cfg in ATARI_CONFIGS.items():
            if cfg["tier"] == 1:
                assert "hub_repo_id" in cfg, f"{env_id} missing hub_repo_id"

    def test_all_have_ppo_timesteps(self):
        for env_id, cfg in ATARI_CONFIGS.items():
            assert "ppo_timesteps" in cfg, f"{env_id} missing ppo_timesteps"


class TestGetAtariEnvId:
    """Test short name -> full env ID mapping."""

    def test_pong(self):
        assert get_atari_env_id("Pong") == "PongNoFrameskip-v4"

    def test_already_full_id(self):
        assert get_atari_env_id("PongNoFrameskip-v4") == "PongNoFrameskip-v4"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_atari_env_id("NonexistentGame")


from unittest.mock import MagicMock, patch

import numpy as np

from imitation.experiments.ftrl.experts import get_or_train_expert


class TestAtariExpertRouting:
    """Test that get_or_train_expert routes correctly for Atari games."""

    @patch("imitation.experiments.ftrl.atari_utils.download_hub_expert")
    def test_tier1_downloads_from_hub(self, mock_download, tmp_path):
        """Tier 1 Atari game should attempt HuggingFace download."""
        mock_download.side_effect = ValueError(
            "mock_hub_download: no real download in test",
        )

        # Use a mock venv to avoid needing Atari ROMs in CI
        mock_venv = MagicMock()

        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="mock_hub_download"):
            get_or_train_expert(
                "PongNoFrameskip-v4", mock_venv, cache_dir=tmp_path, rng=rng,
            )
