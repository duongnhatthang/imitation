"""Tests for the FTRL plotting pipeline."""

import json
import pathlib

import numpy as np
import pytest

from imitation.experiments.ftrl.plot_results import (
    compute_cumulative_loss,
    compute_cumulative_regret,
    load_results,
    plot_all,
    plot_env,
)


def _write_fake_result(
    results_dir: pathlib.Path,
    algo: str,
    env: str,
    seed: int,
    n_rounds: int = 5,
    base_ce: float = 1.0,
):
    """Write a synthetic JSON result file."""
    env_dir = results_dir / env.replace("/", "_")
    env_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed + hash(algo) % 1000)
    samples_per_round = 500
    per_round = []
    for r in range(n_rounds):
        ce = base_ce * np.exp(-0.1 * r) + rng.normal(0, 0.05)
        ce = max(ce, 0.01)
        round_data = {
            "round": r + 1,
            "n_observations": (r + 1) * samples_per_round,
            "cross_entropy": round(ce, 6),
            "l2_norm": round(rng.uniform(0.1, 1.0), 6),
            "total_loss": round(ce + 0.001, 6),
            # normalized_return and disagreement_rate present only at even rounds
            "normalized_return": round(rng.uniform(0, 1), 6) if r % 2 == 0 else None,
            "disagreement_rate": round(rng.uniform(0, 0.5), 6) if r % 2 == 0 else None,
        }
        per_round.append(round_data)

    result = {
        "algo": algo,
        "env": env,
        "seed": seed,
        "policy_mode": "end_to_end",
        "config": {"n_rounds": n_rounds},
        "per_round": per_round,
        "baselines": {"expert_return": 500.0, "random_return": 22.0},
        "elapsed_seconds": 5.0,
    }

    out_file = env_dir / f"{algo}_end_to_end_seed{seed}.json"
    with open(out_file, "w") as f:
        json.dump(result, f)


def _populate_results(results_dir, envs=None, algos=None, seeds=3, n_rounds=5):
    """Generate a full synthetic results directory."""
    if envs is None:
        envs = ["CartPole-v1", "FrozenLake-v1"]
    if algos is None:
        algos = ["ftl", "ftrl", "bc"]

    base_ces = {"ftl": 0.8, "ftrl": 0.9, "bc": 1.2}
    for env in envs:
        for algo in algos:
            for seed in range(seeds):
                _write_fake_result(
                    results_dir,
                    algo,
                    env,
                    seed,
                    n_rounds,
                    base_ce=base_ces.get(algo, 1.0),
                )


def test_load_results(tmp_path):
    """load_results returns correct DataFrame shape."""
    results_dir = tmp_path / "results"
    _populate_results(results_dir, seeds=2, n_rounds=4)

    df = load_results(results_dir)
    # 2 envs x 3 algos x 2 seeds x 4 rounds = 48 rows
    assert len(df) == 48
    assert set(df.columns) >= {
        "algo",
        "env",
        "seed",
        "round",
        "cross_entropy",
        "l2_norm",
        "normalized_return",
        "disagreement_rate",
    }
    assert set(df["algo"].unique()) == {"ftl", "ftrl", "bc"}
    assert set(df["env"].unique()) == {"CartPole-v1", "FrozenLake-v1"}


def test_load_results_empty(tmp_path):
    """load_results returns empty DataFrame for empty dir."""
    df = load_results(tmp_path / "nonexistent")
    assert df.empty


def test_load_results_skips_errors(tmp_path):
    """load_results skips malformed JSON and error results."""
    results_dir = tmp_path / "results"
    _populate_results(results_dir, envs=["CartPole-v1"], algos=["ftl"], seeds=1)

    # Write a malformed file
    bad_file = results_dir / "CartPole-v1" / "bad.json"
    bad_file.write_text("{invalid json")

    # Write an error result
    err_file = results_dir / "CartPole-v1" / "err.json"
    err_file.write_text(
        json.dumps({"error": "boom", "algo": "x", "env": "y", "seed": 0})
    )

    df = load_results(results_dir)
    # Should only have the 1 valid file's data
    assert len(df) == 5  # 1 algo x 1 seed x 5 rounds


def test_compute_cumulative_loss(tmp_path):
    """Cumulative loss is monotonically non-decreasing."""
    results_dir = tmp_path / "results"
    _populate_results(results_dir, envs=["CartPole-v1"], algos=["ftl"], seeds=1)

    df = load_results(results_dir)
    df = compute_cumulative_loss(df)

    assert "cum_loss" in df.columns
    group = df.sort_values("round")
    cum = group["cum_loss"].values
    assert all(cum[i] <= cum[i + 1] for i in range(len(cum) - 1))


def test_compute_cumulative_regret(tmp_path):
    """Cumulative regret is non-negative (best algo has regret=0)."""
    results_dir = tmp_path / "results"
    _populate_results(results_dir, seeds=2)

    df = load_results(results_dir)
    df = compute_cumulative_loss(df)
    df = compute_cumulative_regret(df)

    assert "cum_regret" in df.columns
    # At each (env, seed, round), at least one algo should have regret ~0
    for (env, seed, rnd), group in df.groupby(["env", "seed", "round"]):
        min_regret = group["cum_regret"].min()
        assert min_regret == pytest.approx(0.0, abs=1e-10)


def test_plot_env(tmp_path):
    """plot_env generates a PNG file with 4 subplots."""
    results_dir = tmp_path / "results"
    _populate_results(results_dir, envs=["CartPole-v1"], seeds=2)

    df = load_results(results_dir)
    df = compute_cumulative_loss(df)
    df = compute_cumulative_regret(df)

    out_path = tmp_path / "plots" / "CartPole-v1.png"
    plot_env(df, "CartPole-v1", out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # non-trivial PNG


def test_plot_all(tmp_path):
    """plot_all generates one PNG per env."""
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    _populate_results(results_dir)

    paths = plot_all(results_dir, plots_dir)

    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        assert p.suffix == ".png"


def test_plot_all_with_env_filter(tmp_path):
    """plot_all respects env filter."""
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    _populate_results(results_dir)

    paths = plot_all(results_dir, plots_dir, envs=["CartPole-v1"])

    assert len(paths) == 1
    assert "CartPole-v1" in paths[0].name


def test_plot_incremental(tmp_path):
    """Plotting works with partial results (1 algo, 1 seed)."""
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    _populate_results(results_dir, algos=["ftl"], seeds=1)

    paths = plot_all(results_dir, plots_dir)

    assert len(paths) == 2
    for p in paths:
        assert p.exists()
