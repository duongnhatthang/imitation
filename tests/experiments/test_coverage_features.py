import pathlib

import numpy as np
import pytest

from imitation.experiments.ftrl import coverage_features


def test_feature_mode_selects_cnn_for_atari_else_scaler():
    assert coverage_features.feature_mode("PongNoFrameskip-v4") == "cnn"
    assert coverage_features.feature_mode("CartPole-v1") == "scaler"
    assert coverage_features.feature_mode("FrozenLake-v1") == "scaler"


def test_to_chw_transposes_hwc_and_passes_chw():
    hwc = np.zeros((3, 84, 84, 4), dtype=np.uint8)
    chw = np.zeros((3, 4, 84, 84), dtype=np.uint8)
    assert coverage_features.to_chw(hwc).shape == (3, 4, 84, 84)
    assert coverage_features.to_chw(chw).shape == (3, 4, 84, 84)


def test_scaler_mode_zero_means_and_preserves_rows():
    obs = np.arange(20, dtype=np.float32).reshape(5, 4)
    feats = coverage_features.extract_features(obs, "CartPole-v1")
    assert feats.shape == (5, 4)
    assert np.allclose(feats.mean(axis=0), 0.0, atol=1e-6)


@pytest.mark.expensive
def test_cnn_features_from_pong_expert():
    cache = pathlib.Path("experiments/expert_cache")
    if not (cache / "PongNoFrameskip-v4" / "model.zip").exists():
        pytest.skip("Pong expert not cached")
    expert = coverage_features.load_expert_policy("PongNoFrameskip-v4", cache)
    obs = np.zeros((6, 4, 84, 84), dtype=np.uint8)
    feats = coverage_features.extract_features(obs, "PongNoFrameskip-v4", expert=expert)
    assert feats.shape == (6, 512)
