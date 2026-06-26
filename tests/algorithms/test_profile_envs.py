"""Environment action-margin profiling (E_env, H4)."""

import math

import numpy as np
import torch as th

from imitation.experiments.ftrl import profile_envs


class _ToyPolicy:
    """probs row per state index; margins are top1-top2."""

    def __init__(self, probs):
        self._probs = th.tensor(probs, dtype=th.float32)
        self.device = th.device("cpu")

    def get_distribution(self, obs):
        idx = th.as_tensor(np.asarray(obs), dtype=th.long)

        class _D:
            pass

        d = _D()
        d.distribution = _D()
        d.distribution.probs = self._probs[idx]
        return d


def test_action_margins_and_fraction():
    # state 0: probs [0.5,0.5] -> margin 0.0 (tie)
    # state 1: probs [0.9,0.1] -> margin 0.8 (sharp)
    policy = _ToyPolicy([[0.5, 0.5], [0.9, 0.1]])
    obs = np.array([0, 1, 0])
    margins = profile_envs.action_margins(policy, obs)
    assert margins.shape == (3,)
    assert math.isclose(margins[0], 0.0, abs_tol=1e-6)
    assert math.isclose(margins[1], 0.8, abs_tol=1e-6)
    # threshold 0.1 -> states 0 and 2 (both ties) are "small" -> 2/3
    assert math.isclose(profile_envs.small_margin_fraction(margins, 0.1), 2.0 / 3.0)
