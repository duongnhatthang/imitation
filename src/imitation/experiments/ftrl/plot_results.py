"""Plotting for FTL vs FTRL vs BC experiment results.

Generates per-environment figures with cumulative loss and cumulative regret.
Supports incremental generation — can plot partial results as experiments complete.

Usage:
    python -m imitation.experiments.ftrl.plot_results --results-dir experiments/results/
    python -m imitation.experiments.ftrl.plot_results --envs CartPole-v1 FrozenLake-v1
"""

import argparse
import json
import logging
import pathlib
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imitation.experiments.ftrl import env_utils

logger = logging.getLogger(__name__)

# Use non-interactive backend for server/CI
matplotlib.use("Agg")

ALGO_COLORS: Dict[str, str] = {
    "ftl": "#1f77b4",   # blue
    "ftrl": "#d62728",  # red
    "bc": "#2ca02c",    # green
}

ALGO_LABELS: Dict[str, str] = {
    "ftl": "FTL (DAgger, λ=0)",
    "ftrl": "FTRL (DAgger + L2)",
    "bc": "BC (offline)",
}


def load_results(results_dir: pathlib.Path) -> pd.DataFrame:
    """Load all JSON result files into a DataFrame.

    Each row corresponds to one (algo, env, seed, round) combination.

    Args:
        results_dir: Directory containing per-env subdirs with JSON files.

    Returns:
        DataFrame with columns: algo, env, seed, policy_mode, round,
        cross_entropy, l2_norm, total_loss.
    """
    rows = []
    results_path = pathlib.Path(results_dir)

    for json_file in sorted(results_path.rglob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Skipping {json_file}: {e}")
            continue

        if "error" in data:
            logger.warning(f"Skipping failed run: {json_file}")
            continue

        for m in data.get("per_round", []):
            rows.append({
                "algo": data["algo"],
                "env": data["env"],
                "seed": data["seed"],
                "policy_mode": data.get("policy_mode", "unknown"),
                "round": m["round"],
                "cross_entropy": m["cross_entropy"],
                "l2_norm": m["l2_norm"],
                "total_loss": m["total_loss"],
            })

    if not rows:
        logger.warning(f"No results found in {results_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info(
        f"Loaded {len(df)} data points: "
        f"{df['env'].nunique()} envs, {df['algo'].nunique()} algos, "
        f"{df['seed'].nunique()} seeds"
    )
    return df


def compute_cumulative_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative cross-entropy loss per (algo, env, seed).

    Args:
        df: DataFrame from load_results.

    Returns:
        Same DataFrame with an added 'cum_loss' column.
    """
    df = df.sort_values(["algo", "env", "seed", "round"]).copy()
    df["cum_loss"] = df.groupby(["algo", "env", "seed"])["cross_entropy"].cumsum()
    return df


def compute_cumulative_regret(
    df: pd.DataFrame,
    baseline_algo: str = "ftl",
) -> pd.DataFrame:
    """Compute cumulative regret relative to best algo's cumulative loss.

    Regret for algo A at round t = cum_loss_A(t) - cum_loss_baseline(t),
    where baseline is typically the best-performing algo (FTL by default).
    Since we don't have a true expert baseline per-round, we use the minimum
    cumulative loss across algos at each round as the baseline.

    Args:
        df: DataFrame with 'cum_loss' column (from compute_cumulative_loss).
        baseline_algo: Not used — baseline is computed as min across algos.

    Returns:
        Same DataFrame with an added 'cum_regret' column.
    """
    if "cum_loss" not in df.columns:
        df = compute_cumulative_loss(df)

    # Baseline: minimum cumulative loss across algos at each (env, seed, round)
    baseline = (
        df.groupby(["env", "seed", "round"])["cum_loss"]
        .min()
        .rename("baseline_cum_loss")
    )
    df = df.merge(baseline, on=["env", "seed", "round"])
    df["cum_regret"] = df["cum_loss"] - df["baseline_cum_loss"]
    df.drop(columns=["baseline_cum_loss"], inplace=True)
    return df


def _plot_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
):
    """Plot a metric with mean ± 1 std bands across seeds.

    Args:
        ax: Matplotlib axes to plot on.
        df: DataFrame filtered to a single env.
        metric: Column name to plot.
        ylabel: Y-axis label.
    """
    algos = sorted(df["algo"].unique(), key=lambda a: list(ALGO_COLORS.keys()).index(a)
                   if a in ALGO_COLORS else 99)

    for algo in algos:
        algo_df = df[df["algo"] == algo]
        stats = algo_df.groupby("round")[metric].agg(["mean", "std"]).reset_index()
        stats["std"] = stats["std"].fillna(0)

        color = ALGO_COLORS.get(algo, "#888888")
        label = ALGO_LABELS.get(algo, algo)
        rounds = stats["round"].values
        mean = stats["mean"].values
        std = stats["std"].values

        ax.plot(rounds, mean, color=color, label=label, linewidth=2, marker="o",
                markersize=3)
        ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_env(
    df: pd.DataFrame,
    env_name: str,
    output_path: pathlib.Path,
) -> None:
    """Generate a 3-subplot figure for one environment.

    Top: per-round cross-entropy (learning curve).
    Middle: cumulative loss.
    Bottom: cumulative regret.

    Args:
        df: Full DataFrame with cum_loss and cum_regret columns.
        env_name: Environment to plot.
        output_path: Path to save the PNG.
    """
    env_df = df[df["env"] == env_name]
    if env_df.empty:
        logger.warning(f"No data for {env_name}, skipping")
        return

    # Detect policy mode for title
    policy_modes = env_df["policy_mode"].unique()
    mode_str = policy_modes[0] if len(policy_modes) == 1 else "mixed"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(f"{env_name}  ({mode_str})", fontsize=14, fontweight="bold")

    _plot_metric(ax1, env_df, "cross_entropy", "Per-Round Cross-Entropy")
    _plot_metric(ax2, env_df, "cum_loss", "Cumulative Cross-Entropy Loss")
    _plot_metric(ax3, env_df, "cum_regret", "Cumulative Regret (vs best)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot: {output_path}")


def plot_all(
    results_dir: pathlib.Path,
    output_dir: pathlib.Path,
    envs: Optional[List[str]] = None,
) -> List[pathlib.Path]:
    """Generate plots for all environments.

    Args:
        results_dir: Directory with JSON result files.
        output_dir: Directory to save PNG plots.
        envs: Optional list of envs to filter. If None, plots all found.

    Returns:
        List of saved plot file paths.
    """
    df = load_results(results_dir)
    if df.empty:
        logger.warning("No data to plot")
        return []

    df = compute_cumulative_loss(df)
    df = compute_cumulative_regret(df)

    if envs is not None:
        df = df[df["env"].isin(envs)]

    saved_paths = []
    for env_name in sorted(df["env"].unique()):
        safe_name = env_name.replace("/", "_")
        out_path = pathlib.Path(output_dir) / f"{safe_name}.png"
        plot_env(df, env_name, out_path)
        saved_paths.append(out_path)

    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Plot FTL vs FTRL vs BC experiment results",
    )
    parser.add_argument(
        "--results-dir", type=str, default="experiments/results",
        help="Directory containing JSON result files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/plots",
        help="Directory to save PNG plots",
    )
    parser.add_argument(
        "--envs", nargs="+", default=None,
        help="Filter to specific environments",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    paths = plot_all(
        pathlib.Path(args.results_dir),
        pathlib.Path(args.output_dir),
        envs=args.envs,
    )
    if paths:
        logger.info(f"Generated {len(paths)} plots in {args.output_dir}")
    else:
        logger.warning("No plots generated — check results directory")


if __name__ == "__main__":
    main()
