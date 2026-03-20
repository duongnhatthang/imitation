"""Tests for Atari environment setup (ENV-01 through ENV-05)."""
import pickle
from pathlib import Path
import pytest
import numpy as np

# Import helpers
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.atari_helpers import ATARI_GAMES, make_atari_training_venv, make_atari_eval_venv, load_atari_expert, compute_normalized_score


BASELINES_PATH = Path(__file__).resolve().parents[2] / "experiments" / "baselines" / "atari_random_scores.pkl"


class TestAtariEnvSetup:
    """ENV-01: All 7 games register and create without error."""

    @pytest.mark.parametrize("game_name,game_id", list(ATARI_GAMES.items()))
    def test_game_creates(self, game_name, game_id):
        """Each game creates a venv without error."""
        venv = make_atari_training_venv(game_id, n_envs=1, seed=0)
        assert venv is not None
        venv.close()


class TestAtariObsSpace:
    """ENV-02: Obs space is Box(0,255,(4,84,84),uint8) for all games."""

    @pytest.mark.parametrize("game_name,game_id", list(ATARI_GAMES.items()))
    def test_obs_space_shape(self, game_name, game_id):
        venv = make_atari_training_venv(game_id, n_envs=1, seed=0)
        assert venv.observation_space.shape == (4, 84, 84)
        assert venv.observation_space.dtype == np.uint8
        venv.close()


class TestAtariExpertLoading:
    """ENV-03 + ENV-04: Experts load and obs spaces match."""

    @pytest.mark.parametrize("game_name,game_id", list(ATARI_GAMES.items()))
    def test_expert_loads_and_obs_match(self, game_name, game_id):
        """Expert loads from sb3 HF org and obs space matches learner venv."""
        venv = make_atari_training_venv(game_id, n_envs=1, seed=0)
        expert = load_atari_expert(venv, game_id)
        assert venv.observation_space == expert.observation_space, (
            f"Obs space mismatch for {game_name}: "
            f"venv={venv.observation_space} vs expert={expert.observation_space}"
        )
        venv.close()


class TestRandomBaselines:
    """ENV-05: Random baselines cached for all 7 games."""

    def test_baselines_file_exists(self):
        assert BASELINES_PATH.exists(), f"Baselines file not found: {BASELINES_PATH}"

    def test_baselines_has_all_games(self):
        with open(BASELINES_PATH, "rb") as f:
            baselines = pickle.load(f)
        for game_name in ATARI_GAMES:
            assert game_name in baselines, f"Missing baseline for {game_name}"
            assert "mean" in baselines[game_name]
            assert "std" in baselines[game_name]


def test_normalized_score():
    """Score normalization utility works correctly."""
    assert compute_normalized_score(50.0, 0.0, 100.0) == 0.5
    assert compute_normalized_score(100.0, 0.0, 100.0) == 1.0
    assert compute_normalized_score(0.0, 0.0, 100.0) == 0.0
    assert compute_normalized_score(10.0, 10.0, 10.0) == 0.0  # div-by-zero guard
