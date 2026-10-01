"""Stage 2 tests: predeclared quantizers, exact majority ERM, alias bound."""

import itertools
import math

import numpy as np
import pytest

from imitation.experiments.agnostic import quantized


def _enumerated_minimum(counts):
    n_bins, n_actions = counts.shape
    return min(
        quantized.empirical_risk(np.array(t), counts)
        for t in itertools.product(range(n_actions), repeat=n_bins)
    )


def test_fit_is_exact_erm_with_lowest_tie_and_unseen_zero():
    rng = np.random.default_rng(0)
    for _ in range(20):
        counts = rng.integers(0, 3, size=(4, 3))
        table = quantized.fit_table(counts)
        assert quantized.empirical_risk(table, counts) == _enumerated_minimum(counts)
    counts = np.array([[0, 0, 0], [2, 5, 5], [1, 0, 1], [0, 0, 4]])
    np.testing.assert_array_equal(quantized.fit_table(counts), [0, 1, 0, 2])


def test_fit_returns_independent_table_from_counts_only():
    counts = np.array([[1, 0], [0, 2]])
    first = quantized.fit_table(counts)
    first[:] = 1
    np.testing.assert_array_equal(quantized.fit_table(counts), [0, 1])
    bins, labels = [0, 1, 1, 0], [0, 1, 1, 0]
    np.testing.assert_array_equal(
        quantized.label_counts(bins, labels, 2, 2), [[2, 0], [0, 2]]
    )


@pytest.mark.parametrize(
    "env_name,severe,mild",
    [("CartPole-v1", 9, 36), ("Acrobot-v1", 9, 36), ("MountainCar-v0", 4, 12)],
)
def test_bin_counts_include_every_possible_bin(env_name, severe, mild):
    assert quantized.get_quantizer(env_name, "severe").n_bins == severe
    assert quantized.get_quantizer(env_name, "mild").n_bins == mild


def test_cartpole_edges_are_left_closed_and_severe_ignores_velocities():
    severe = quantized.get_quantizer("CartPole-v1", "severe")
    mild = quantized.get_quantizer("CartPole-v1", "mild")
    below = np.nextafter(-0.4, -1.0)
    assert severe([below, 0, 0, 0]) == 0 * 3 + 1
    assert severe([-0.4, 0, 0, 0]) == 1 * 3 + 1
    assert severe([0.4, 0, 0.05, 0]) == 2 * 3 + 2
    assert severe([0.0, -9.0, 0.0, 9.0]) == severe([0.0, 9.0, 0.0, -9.0])
    # Zero velocity is in the non-negative sign bin.
    assert mild([0, 0.0, 0, 0.0]) == mild([0, 1.0, 0, 1.0])
    assert mild([0, -1e-9, 0, 0.0]) != mild([0, 0.0, 0, 0.0])
    ids = {mild(o) for o in itertools.product([-1, 0, 1], [-1, 1], [-1, 0, 1], [-1, 1])}
    assert ids == set(range(36))


def test_acrobot_uses_atan2_and_wraps_minus_pi_to_pi():
    q = quantized.get_quantizer("Acrobot-v1", "severe")

    def obs(t1, t2, s1=None):
        return [
            math.cos(t1),
            math.sin(t1) if s1 is None else s1,
            math.cos(t2),
            math.sin(t2),
            0,
            0,
        ]

    # sin = +0 and -0 at cos = -1 are the same angle, pi, and share a bin.
    assert q(obs(math.pi, 0, s1=0.0)) == q(obs(math.pi, 0, s1=-0.0)) == 2 * 3 + 1
    assert q(obs(-math.pi / 2, 0)) == 0 * 3 + 1
    assert q(obs(0.0, math.pi / 3 + 1e-9)) == 1 * 3 + 2
    assert q(obs(0.0, math.pi / 3 - 1e-9)) == 1 * 3 + 1
    mild = quantized.get_quantizer("Acrobot-v1", "mild")
    assert mild(obs(0, 0)[:4] + [-1, 1]) != mild(obs(0, 0)[:4] + [1, 1])


def test_mountaincar_bins():
    severe = quantized.get_quantizer("MountainCar-v0", "severe")
    mild = quantized.get_quantizer("MountainCar-v0", "mild")
    assert [severe([p, 0.05]) for p in (-1.2, -0.9, -0.5, -0.1)] == [0, 1, 2, 3]
    assert [mild([-1.2, v]) for v in (-0.02, -0.01, 0.01)] == [0, 1, 2]


def test_observations_are_validated():
    q = quantized.get_quantizer("CartPole-v1", "mild")
    with pytest.raises(ValueError):
        q([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        q([0.0, np.nan, 0.0, 0.0])
    with pytest.raises(ValueError):
        q([0.0, 0.0, np.inf, 0.0])
    with pytest.raises(ValueError):
        quantized.get_quantizer("CartPole-v1", "raw")


def test_alias_bound_uses_all_bins_and_needs_strictly_positive_bound():
    counts = np.array([[600, 400], [1000, 0]])
    summary = quantized.alias_lower_bound(counts, delta=0.01)
    slack = math.sqrt((2 * math.log(2) + math.log(100)) / (2 * 2000))
    assert summary["empirical_min_disagreement"] == pytest.approx(0.2)
    assert summary["lower_bound"] == pytest.approx(0.2 - slack)
    assert summary["positive_certificate"] is True
    # Unobserved bins still enlarge the class and the slack.
    padded = quantized.alias_lower_bound(np.vstack([counts, np.zeros((98, 2))]), 0.01)
    assert padded["K"] == 100 and padded["observed_bins"] == 2
    assert padded["lower_bound"] < summary["lower_bound"]
    few = quantized.alias_lower_bound(np.array([[3, 2]]), 0.01)
    assert few["lower_bound"] == 0.0 and few["positive_certificate"] is False
    assert quantized.alias_lower_bound(np.zeros((3, 2)), 0.01)["lower_bound"] is None
