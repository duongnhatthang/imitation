"""Regression tests for the shared BC training data artifact."""

import numpy as np

from imitation.data import types


def test_shared_data_is_collected_once_and_preserves_order(tmp_path):
    from imitation.experiments.ftrl.expert_dataset import load_or_collect

    calls = []

    def collect():
        calls.append(True)
        return types.Transitions(
            obs=np.array([[3], [1], [2]]),
            acts=np.array([0, 1, 0]),
            next_obs=np.array([[4], [2], [3]]),
            dones=np.zeros(3, dtype=bool),
            infos=np.array([{}, {}, {}]),
        )

    metadata = dict(env="test", seed=7, budget=3, expert="fixed", version=1)
    first, source1 = load_or_collect(tmp_path, metadata, collect)
    second, source2 = load_or_collect(tmp_path, metadata, collect)
    assert len(calls) == 1
    # Collection order must survive the round trip, not merely match itself.
    for loaded in (first, second):
        np.testing.assert_array_equal(loaded.obs, [[3], [1], [2]])
        np.testing.assert_array_equal(loaded.acts, [0, 1, 0])
    assert source1 == source2
    assert source1["sha256"]
    load_or_collect(tmp_path, dict(metadata, seed=8), collect)
    assert len(calls) == 2
