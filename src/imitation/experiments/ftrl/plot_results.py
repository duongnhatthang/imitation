"""Plotting for FTL vs FTRL vs BC experiment results.

Generates per-environment figures with 4 subplots:
  1. Per-round imitation loss (log scale)
  2. Normalized expected return
  3. On-policy disagreement rate
  4. Cumulative regret

Uses IQM + 95% stratified bootstrap CI (via rliable) instead of mean +/- std.
Supports incremental generation -- can plot partial results as experiments complete.

Usage:
    python -m imitation.experiments.ftrl.plot_results --results-dir experiments/results/
    python -m imitation.experiments.ftrl.plot_results --envs CartPole-v1 FrozenLake-v1
"""

import argparse
import json
import logging
import pathlib
from typing import Dict, List, Optional, Tuple

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
    "ftl": "FTL (DAgger, \u03bb=0)",
    "ftrl": "FTRL (DAgger + L2)",
    "bc": "BC (offline)",
}


def _compute_iqm_and_ci(
    values_by_seed: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute IQM and 95% stratified bootstrap CI using rliable.

    Args:
        values_by_seed: 1D array of values, one per seed.

    Returns:
        (iqm_value, ci_low, ci_high)
    """
    from rliable import library as rly
    from rliable import metrics as rly_metrics

    if len(values_by_seed) < 2:
        val = float(values_by_seed[0]) if len(values_by_seed) == 1 else 0.0
        return val, val, val

    # rliable expects dict of {algo_name: array of shape (n_runs, n_tasks)}
    data = {"a": values_by_seed.reshape(-1, 1)}
    aggregate_func = lambda x: np.array([rly_metrics.aggregate_iqm(x)])
    scores, cis = rly.get_interval_estimates(data, aggregate_func, reps=2000)
    return float(scores["a"][0]), float(cis["a"][0, 0]), float(cis["a"][1, 0])


def load_results(results_dir: pathlib.Path) -> pd.DataFrame:
    """Load all JSON result files into a DataFrame.

    Each row corresponds to one (algo, env, seed, round) combination.

    Args:
        results_dir: Directory containing per-env subdirs with JSON files.

    Returns:
        DataFrame with columns: algo, env, seed, policy_mode, round,
        cross_entropy, l2_norm, total_loss, normalized_return,
        disagreement_rate.
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
            row = {
                "algo": data["algo"],
                "env": data["env"],
                "seed": data["seed"],
                "policy_mode": data.get("policy_mode", "unknown"),
                "round": m["round"],
                "n_observations": m.get("n_observations", 0),
                "cross_entropy": m["cross_entropy"],
                "l2_norm": m["l2_norm"],
                "total_loss": m["total_loss"],
            }
            if "expert_cross_entropy" in m:
                row["expert_cross_entropy"] = m["expert_cross_entropy"]

            # New optional fields
            if "normalized_return" in m and m["normalized_return"] is not None:
                row["normalized_return"] = m["normalized_return"]
            else:
                row["normalized_return"] = np.nan
            if "disagreement_rate" in m and m["disagreement_rate"] is not None:
                row["disagreement_rate"] = m["disagreement_rate"]
            else:
                row["disagreement_rate"] = np.nan

            rows.append(row)

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

    Also computes expert cumulative loss if expert_cross_entropy is available.

    Args:
        df: DataFrame from load_results.

    Returns:
        Same DataFrame with added 'cum_loss' column (and 'expert_cum_loss'
        if expert_cross_entropy is present).
    """
    df = df.sort_values(["algo", "env", "seed", "round"]).copy()
    df["cum_loss"] = df.groupby(["algo", "env", "seed"])["cross_entropy"].cumsum()
    if "expert_cross_entropy" in df.columns:
        df["expert_cum_loss"] = (
            df.groupby(["algo", "env", "seed"])["expert_cross_entropy"].cumsum()
        )
    return df


def compute_cumulative_regret(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative regret relative to the expert's cumulative loss.

    Regret for algo A at round t = cum_loss_A(t) - expert_cum_loss(t).
    Falls back to best-algo baseline if expert data is unavailable.

    Args:
        df: DataFrame with 'cum_loss' column (from compute_cumulative_loss).

    Returns:
        Same DataFrame with an added 'cum_regret' column.
    """
    if "cum_loss" not in df.columns:
        df = compute_cumulative_loss(df)

    if "expert_cum_loss" in df.columns:
        # Use expert as baseline
        df["cum_regret"] = df["cum_loss"] - df["expert_cum_loss"]
    else:
        # Fallback: minimum cumulative loss across algos
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
    log_scale: bool = False,
    expert_baseline: Optional[pd.DataFrame] = None,
):
    """Plot a metric with IQM and 95% bootstrap CI bands across seeds.

    Args:
        ax: Matplotlib axes to plot on.
        df: DataFrame filtered to a single env.
        metric: Column name to plot.
        ylabel: Y-axis label.
        log_scale: Whether to use log scale on y-axis.
        expert_baseline: Optional DataFrame with 'mean_metric' and 'mean_obs'
            columns. Plotted as a black dashed line.
    """
    algos = sorted(
        df["algo"].unique(),
        key=lambda a: list(ALGO_COLORS.keys()).index(a)
        if a in ALGO_COLORS else 99,
    )

    for algo in algos:
        algo_df = df[df["algo"] == algo]

        # Filter to non-null values for this metric
        valid_df = algo_df.dropna(subset=[metric])
        if valid_df.empty:
            continue

        rounds = sorted(valid_df["round"].unique())
        x_vals, iqm_vals, ci_lows, ci_highs = [], [], [], []

        for rnd in rounds:
            rnd_df = valid_df[valid_df["round"] == rnd]
            values = rnd_df[metric].values

            if len(values) == 0:
                continue

            iqm, ci_lo, ci_hi = _compute_iqm_and_ci(values)
            mean_obs = rnd_df["n_observations"].mean()
            x_vals.append(mean_obs)
            iqm_vals.append(iqm)
            ci_lows.append(ci_lo)
            ci_highs.append(ci_hi)

        if not x_vals:
            continue

        x = np.array(x_vals)
        iqm = np.array(iqm_vals)
        ci_lo = np.array(ci_lows)
        ci_hi = np.array(ci_highs)

        color = ALGO_COLORS.get(algo, "#888888")
        label = ALGO_LABELS.get(algo, algo)
        ax.plot(
            x, iqm, color=color, label=label, linewidth=2,
            marker="o", markersize=3,
        )
        ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.15)

    # Expert baseline
    if expert_baseline is not None and not expert_baseline.empty:
        x = expert_baseline["mean_obs"].values
        values = expert_baseline["mean_metric"].values
        # If values are roughly constant, draw a flat line; otherwise a curve
        if np.std(values) < 0.01 * (np.mean(np.abs(values)) + 1e-8):
            ax.axhline(
                y=np.mean(values),
                color="black", linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"Expert (\u03c0*) = {np.mean(values):.3f}",
            )
        else:
            ax.plot(
                x, values,
                color="black", linestyle="--", linewidth=1.5, alpha=0.7,
                marker="s", markersize=3,
                label="Expert (\u03c0*)",
            )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Number of Observations")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_env(
    df: pd.DataFrame,
    env_name: str,
    output_path: pathlib.Path,
) -> None:
    """Generate a 4-subplot figure for one environment.

    Subplot 1: per-round imitation loss (log scale).
    Subplot 2: normalized expected return.
    Subplot 3: on-policy disagreement rate.
    Subplot 4: cumulative regret.

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

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 14), sharex=True)
    fig.suptitle(f"{env_name}  ({mode_str})", fontsize=14, fontweight="bold")

    # Get expert baselines if available (as DataFrames with mean_obs column)
    expert_ce = None
    if "expert_cross_entropy" in env_df.columns:
        expert_ce = (
            env_df.groupby("round")
            .agg(
                mean_metric=("expert_cross_entropy", "mean"),
                mean_obs=("n_observations", "mean"),
            )
            .reset_index()
        )

    # Subplot 1: Per-round imitation loss (log scale)
    _plot_metric(
        ax1, env_df, "cross_entropy", "Per-Round Imitation Loss",
        log_scale=True, expert_baseline=expert_ce,
    )

    # Subplot 2: Normalized expected return
    _plot_metric(ax2, env_df, "normalized_return", "Normalized Expected Return")
    ax2.axhline(
        y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5,
        label="Expert (1.0)",
    )
    ax2.axhline(
        y=0.0, color="gray", linestyle=":", linewidth=1, alpha=0.5,
        label="Random (0.0)",
    )
    ax2.legend(fontsize=9)

    # Subplot 3: On-policy disagreement rate
    _plot_metric(ax3, env_df, "disagreement_rate", "On-Policy Disagreement Rate")
    ax3.set_ylim(-0.05, 1.05)

    # Subplot 4: Cumulative regret
    _plot_metric(ax4, env_df, "cum_regret", "Cumulative Regret (vs Expert)")

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
        logger.warning("No plots generated -- check results directory")


if __name__ == "__main__":
    main()
