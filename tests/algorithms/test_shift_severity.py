"""Shift-severity analysis (E3)."""

import math

from imitation.experiments.ftrl import analyze_shift_severity as a


def test_shift_severity_ratio():
    per_round = [
        {"rollout_cross_entropy": 0.2, "expert_rollout_cross_entropy": 0.1},
        {"rollout_cross_entropy": 0.4, "expert_rollout_cross_entropy": 0.1},
    ]
    # mean rollout = 0.3, mean expert = 0.1 -> ratio 3.0
    assert math.isclose(a.shift_severity(per_round), 3.0)
