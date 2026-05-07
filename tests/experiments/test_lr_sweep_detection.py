"""Unit tests for LR sweep saturation detection."""

import numpy as np
import pytest

from imitation.experiments.ftrl.run_lr_sweep import detect_t_sat


def test_detect_t_sat_disagreement_downward():
    """Falling-then-flat disagreement: T_sat should land near the plateau."""
    # 60 rounds; falls from 0.8 → 0.1 over first 40, flat at 0.1 after.
    rounds = np.arange(60)
    rates = list(np.where(rounds < 40, 0.8 - 0.7 * rounds / 40.0, 0.1))
    obs = list((rounds + 1) * 50)
    t_sat, sat_val = detect_t_sat(
        rates, obs, smooth_window=10, metric_direction="down"
    )
    assert t_sat is not None
    assert 35 * 50 <= t_sat <= 55 * 50  # plateau region
    assert 0.05 <= sat_val <= 0.15


def test_detect_t_sat_normalized_return_upward():
    """Rising-then-flat return (near 1): T_sat should detect the plateau."""
    rounds = np.arange(60)
    rets = list(np.where(rounds < 30, rounds / 30.0, 1.0))
    obs = list((rounds + 1) * 50)
    t_sat, sat_val = detect_t_sat(
        rets, obs, smooth_window=10, metric_direction="up"
    )
    assert t_sat is not None
    # Plateau begins at round 30; allow detection a few rounds earlier due
    # to smoothing window edge effects.
    assert 20 * 50 <= t_sat <= 50 * 50
    assert 0.9 <= sat_val <= 1.01


def test_detect_t_sat_upward_flat_at_low_level_is_saturated():
    """Flat-at-any-level is saturated for upward direction.

    Per design: no absolute level gate. A curve that stays flat at 0.05 is
    saturated at 0.05 — the return heatmap will show the low value, which is
    itself the useful signal. Fast-learning curves (CartPole with high LR)
    that start near the ceiling are similarly correctly reported.
    """
    flat = [0.05] * 60
    obs = list((np.arange(60) + 1) * 50)
    t_sat, sat_val = detect_t_sat(
        flat, obs, smooth_window=10, metric_direction="up"
    )
    assert t_sat is not None
    assert sat_val is not None
    assert 0.04 <= sat_val <= 0.06


def test_detect_t_sat_upward_oscillating_is_rejected():
    """A curve that oscillates without any flat region returns None."""
    rng = np.random.default_rng(0)
    noisy = list(rng.uniform(0.0, 1.0, size=60))
    obs = list((np.arange(60) + 1) * 50)
    t_sat, sat_val = detect_t_sat(
        noisy, obs, smooth_window=10, metric_direction="up"
    )
    assert t_sat is None
    assert sat_val is None


def test_detect_t_sat_adaptive_window_short_curve():
    """Detector should still work with fewer rounds than the default window."""
    # 23 rounds (matches samples_per_round=45 budget cell); curve rises fast.
    rounds = np.arange(23)
    rets = list(np.where(rounds < 8, rounds / 8.0, 1.0))
    obs = list((rounds + 1) * 45)
    t_sat, sat_val = detect_t_sat(
        rets, obs, smooth_window=7, metric_direction="up"
    )
    assert t_sat is not None
    assert 0.9 <= sat_val <= 1.01
