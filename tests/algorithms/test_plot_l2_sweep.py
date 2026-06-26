"""Aggregation logic for the L2-lambda sweep plots (E1)."""

import json
import math
from pathlib import Path

from imitation.experiments.ftrl import plot_l2_sweep


def test_parse_lambda_tag():
    assert plot_l2_sweep.parse_lambda_tag("lam_0") == 0.0
    assert plot_l2_sweep.parse_lambda_tag("lam_1e-2") == 0.01
    assert plot_l2_sweep.parse_lambda_tag("lam_1e-04") == 1e-4
    assert plot_l2_sweep.parse_lambda_tag("lam_1") == 1.0


def _write_run(path: Path, ce_values, dis_values):
    per_round = [
        {
            "round": i,
            "rollout_cross_entropy": ce,
            "disagreement_rate": d,
            "rollout_ce_correct": ce / 2,
            "rollout_ce_wrong": ce,
            "n_correct": 9,
            "n_wrong": 1,
        }
        for i, (ce, d) in enumerate(zip(ce_values, dis_values))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"per_round": per_round}))


def test_aggregate_lambda_auc(tmp_path):
    base = tmp_path / "classical"
    # env CartPole, two lambdas, one seed each.
    _write_run(
        base / "lam_0" / "CartPole-v1" / "ftrl_linear_seed0.json",
        [0.1, 0.2],
        [0.0, 0.1],
    )
    _write_run(
        base / "lam_1e-2" / "CartPole-v1" / "ftrl_linear_seed0.json",
        [0.3, 0.5],
        [0.2, 0.4],
    )

    agg = plot_l2_sweep.aggregate_lambda_auc(str(base))

    assert set(agg["CartPole-v1"].keys()) == {0.0, 0.01}
    assert math.isclose(agg["CartPole-v1"][0.0]["rollout_ce_auc"], 0.15)
    assert math.isclose(agg["CartPole-v1"][0.01]["disagreement_auc"], 0.3)


def test_plot_auc_vs_lambda_all_nan_guard(tmp_path):
    """Test that _plot_auc_vs_lambda handles all-NaN metrics without crashing."""
    base = tmp_path / "classical"
    # Write a run with per_round entries that have NaN values for ce_correct/ce_wrong.
    per_round = [
        {
            "round": i,
            "rollout_cross_entropy": 0.5,
            "disagreement_rate": 0.1,
            "rollout_ce_correct": None,  # Will become NaN
            "rollout_ce_wrong": None,  # Will become NaN
            "n_correct": 5,
            "n_wrong": 5,
        }
        for i in range(2)
    ]
    path = base / "lam_0" / "CartPole-v1" / "ftrl_linear_seed0.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"per_round": per_round}))

    agg = plot_l2_sweep.aggregate_lambda_auc(str(base))

    # ce_wrong_mean will be all NaN since all values are None.
    # This should not raise ValueError when plotting.
    out_path = str(tmp_path / "out.png")
    plot_l2_sweep._plot_auc_vs_lambda(agg, "ce_wrong_mean", "test ylabel", out_path)

    # Verify the PNG was created.
    assert Path(out_path).exists()
