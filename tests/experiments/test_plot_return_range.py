"""Return views follow task saturation and preserve later uncertainty."""

import numpy as np

from imitation.experiments.ftrl.plot_return_range import automatic_return_limits


def test_suboptimal_plateau_gets_its_own_range():
    lo, hi = automatic_return_limits([([0, 20, 40], [0, 0.72, 0.73], [0, 0.76, 0.75])])
    assert 0.65 < lo < 0.72
    assert 0.76 < hi < 0.85


def test_distinct_methods_and_later_uncertainty_are_included():
    lo, hi = automatic_return_limits(
        [
            ([0, 50, 100], [-20, 0.4, 0.5], [20, 0.7, 0.8]),
            ([100], [0.2], [0.3]),
        ]
    )
    assert lo < 0.2 and hi > 0.8
    assert lo > -1 and hi < 2


def test_nonfinite_or_absent_values_have_a_defined_range():
    assert automatic_return_limits([([0], [np.nan], [np.nan])]) == (-0.5, 1.5)
    assert automatic_return_limits([]) == (-0.5, 1.5)
