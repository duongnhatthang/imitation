"""Observation-type feature extraction for t-SNE coverage maps.

Non-Atari observations are already numeric vectors upstream (toy-text one-hot,
tuple-flattened, continuous raw), so they use a plain StandardScaler. Atari
image observations are embedded through the frozen expert PPO CNN feature
extractor before t-SNE (raw-pixel t-SNE is poor and memory-heavy).
"""

import pathlib

import numpy as np
import torch as th
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO

from imitation.experiments.ftrl import env_utils


def feature_mode(env_name: str) -> str:
    """Return the feature-extraction mode for an environment."""
    return "cnn" if env_utils.is_atari(env_name) else "scaler"


def to_chw(obs: np.ndarray) -> np.ndarray:
    """Return stacked-frame obs as [N, C, H, W] (accepts NCHW or NHWC)."""
    obs = np.asarray(obs)
    if obs.ndim == 4 and obs.shape[1] != 4 and obs.shape[-1] == 4:
        return np.transpose(obs, (0, 3, 1, 2))
    return obs


def load_expert_policy(env_name: str, expert_cache):
    """Load the cached PPO expert policy (CPU) for CNN feature extraction."""
    model_file = pathlib.Path(expert_cache) / env_name.replace("/", "_") / "model.zip"
    return PPO.load(model_file, device="cpu").policy


def extract_features(obs, env_name, expert=None, chunk=512) -> np.ndarray:
    """Feature matrix [N, D] for t-SNE, dispatched by observation type."""
    obs = np.asarray(obs)
    if feature_mode(env_name) == "scaler":
        return StandardScaler().fit_transform(obs.reshape(len(obs), -1))
    if expert is None:
        raise ValueError(f"CNN features for {env_name} require an expert policy")
    chw = to_chw(obs)
    feats = []
    for start in range(0, len(chw), chunk):
        batch = chw[start : start + chunk]
        obs_t, _ = expert.obs_to_tensor(batch)
        with th.no_grad():
            feats.append(expert.extract_features(obs_t).cpu().numpy())
    return StandardScaler().fit_transform(np.concatenate(feats, axis=0))
