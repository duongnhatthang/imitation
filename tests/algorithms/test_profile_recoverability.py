"""Tests for the recoverability probe (E_rec, H5)."""
import math

import numpy as np
import gymnasium as gym

from imitation.experiments.ftrl import profile_recoverability as pr


class _ConstExpert:
    """Always predicts action 0 for every env in the batch."""

    def predict(self, obs, deterministic=True):
        n = np.asarray(obs).shape[0]
        return np.zeros(n, dtype=np.int64), None


def test_epsilon_expert_eps0_is_pure_expert():
    rng = np.random.default_rng(0)
    space = gym.spaces.Discrete(4)
    pol = pr.EpsilonExpert(_ConstExpert(), space, eps=0.0, rng=rng)
    act, _ = pol.predict(np.zeros((5, 3)))
    assert act.tolist() == [0, 0, 0, 0, 0]


def test_epsilon_expert_eps1_in_action_space():
    rng = np.random.default_rng(0)
    space = gym.spaces.Discrete(4)
    pol = pr.EpsilonExpert(_ConstExpert(), space, eps=1.0, rng=rng)
    act, _ = pol.predict(np.zeros((50, 3)))
    assert all(0 <= a < 4 for a in act)
    # With eps=1 and 4 actions, at least one should differ from the expert's 0.
    assert any(a != 0 for a in act)


def test_recoverability_scores_normalization():
    # expert=100, random=0; perturbed: 90 -> rec 0.9, 50 -> rec 0.5
    rec = pr.recoverability_scores(100.0, 0.0, {0.1: 90.0, 0.4: 50.0})
    assert math.isclose(rec[0.1], 0.9)
    assert math.isclose(rec[0.4], 0.5)


def test_recoverability_scores_zero_span_is_nan():
    rec = pr.recoverability_scores(5.0, 5.0, {0.1: 5.0})
    assert math.isnan(rec[0.1])
