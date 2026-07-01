"""Tests for atari_utils: config, env creation, hub downloads."""

import pytest

from imitation.experiments.ftrl.atari_utils import ATARI_CONFIGS, get_atari_env_id


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
                "PongNoFrameskip-v4",
                mock_venv,
                cache_dir=tmp_path,
                rng=rng,
            )


class TestMakeAtariVenvEpisodeCap:
    """The per-episode TimeLimit prevents infinite-episode rollout hangs."""

    def _make_or_skip(self, **kw):
        from imitation.experiments.ftrl.atari_utils import make_atari_venv

        try:
            return make_atari_venv("PongNoFrameskip-v4", n_envs=1, seed=0, **kw)
        except Exception as e:  # pragma: no cover - depends on ROM availability
            pytest.skip(f"Atari ROMs unavailable: {type(e).__name__}: {e}")

    def test_episode_truncates_within_cap(self):
        """A random policy episode ends at ~max_episode_steps/4 agent steps.

        AtariWrapper frame-skips 4, and the TimeLimit counts base frames, so the
        agent-step bound is roughly max_episode_steps / 4. We assert the episode
        DID terminate within that bound (the property that breaks the hang).
        """
        venv = self._make_or_skip(max_episode_steps=2000)
        try:
            venv.reset()
            agent_cap = 2000 // 4
            ended = False
            for _ in range(agent_cap + 5):
                _, _, dones, _ = venv.step([venv.action_space.sample()])
                if dones[0]:
                    ended = True
                    break
            assert ended, "episode must terminate within the TimeLimit cap"
        finally:
            venv.close()

    def test_default_cap_is_finite(self):
        """The default cap is a finite positive int (no infinite episodes)."""
        from imitation.experiments.ftrl.atari_utils import (
            DEFAULT_ATARI_MAX_EPISODE_STEPS,
        )

        assert isinstance(DEFAULT_ATARI_MAX_EPISODE_STEPS, int)
        assert DEFAULT_ATARI_MAX_EPISODE_STEPS > 0
