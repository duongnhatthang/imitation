"""Tests for the correct/wrong-argmax CE breakdown diagnostic (E2)."""

import math

import numpy as np
import torch as th

from imitation.experiments.ftrl import eval_utils


class _ToyPolicy:
    """Minimal stand-in exposing the two methods compute_ce_breakdown needs.

    Two states, two actions. Logits make action 0 highly likely in state 0
    and action 1 highly likely in state 1, so argmax = [0, 1] deterministically.
    """

    def __init__(self):
        self.training = False
        self.device = th.device("cpu")
        # row i = logits for state i
        self._logits = th.tensor([[3.0, 0.0], [0.0, 3.0]])

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def _rows(self, obs):
        # obs is a (k,) int index array selecting rows of self._logits
        idx = th.as_tensor(np.asarray(obs), dtype=th.long)
        return self._logits[idx]

    def evaluate_actions(self, obs, acts):
        logits = self._rows(obs)
        log_probs_all = th.log_softmax(logits, dim=1)
        acts = th.as_tensor(acts, dtype=th.long)
        log_prob = log_probs_all.gather(1, acts.view(-1, 1)).squeeze(1)
        return None, log_prob, None

    def get_distribution(self, obs):
        logits = self._rows(obs)
        probs = th.softmax(logits, dim=1)

        class _D:
            pass

        d = _D()
        d.distribution = _D()
        d.distribution.probs = probs
        return d


def test_ce_breakdown_splits_correct_and_wrong():
    policy = _ToyPolicy()
    # 3 states: indices 0,1,0. Expert actions = argmax = [0,1,0].
    obs = np.array([0, 1, 0])
    expert_actions = np.array([0, 1, 0])
    # Learner agrees on first two, disagrees on the third.
    learner_actions = np.array([0, 1, 1])

    out = eval_utils.compute_ce_breakdown(policy, obs, expert_actions, learner_actions)

    assert out.n_correct == 2
    assert out.n_wrong == 1
    # Correct bucket: CE = -log softmax(action=expert) on confident states -> small.
    expected_correct = -math.log(math.exp(3.0) / (math.exp(3.0) + 1.0))
    assert abs(out.ce_correct - expected_correct) < 1e-5
    # Confidence (max prob) is the same high value in both buckets here.
    assert out.conf_correct > 0.9
    assert out.conf_wrong > 0.9


def test_ce_breakdown_empty_wrong_bucket_is_nan():
    policy = _ToyPolicy()
    obs = np.array([0, 1])
    expert_actions = np.array([0, 1])
    learner_actions = np.array([0, 1])  # perfect agreement -> no wrong states

    out = eval_utils.compute_ce_breakdown(policy, obs, expert_actions, learner_actions)
    assert out.n_wrong == 0
    assert math.isnan(out.ce_wrong)
    assert math.isnan(out.conf_wrong)
    assert out.n_correct == 2
