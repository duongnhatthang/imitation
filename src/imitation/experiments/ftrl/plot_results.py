"""Plotting for FTL vs FTRL vs BC+DAgger vs BC experiment results.

Generates per-environment figures with 4 subplots:
  1. Rollout cross-entropy on the aggregated D_eval^t buffer (log scale)
  2. Normalized expected return
  3. On-policy disagreement rate
  4. Cumulative regret (vs best dynamic algo)

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

logger = logging.getLogger(__name__)

# Use non-interactive backend for server/CI
matplotlib.use("Agg")

ALGO_COLORS: Dict[str, str] = {
    "ftl": "#1f77b4",  # blue
    "ftrl": "#d62728",  # red
    "bc_dagger": "#2ca02c",  # green
    "bc": "#17a663",  # dark green (dashed reference)
    "expert": "#555555",  # gray (dashed reference)
}

ALGO_LABELS: Dict[str, str] = {
    "ftl": "FTL+DAgger",
    "ftrl": "FTRL+DAgger",
    "bc_dagger": "BC+DAgger",
    "bc": "BC (fixed)",
    "expert": "Expert",
}

ALGO_LINESTYLES: Dict[str, str] = {
    "ftl": "-",
    "ftrl": "-",
    "bc_dagger": "-",
    "bc": "--",
    "expert": "--",
}

LOSS_SUBPLOT_ALGOS = {"ftl", "ftrl", "bc_dagger"}  # fixed BC excluded

LOSS_SUBTITLE_LINES = [
    (
        r"Loss: $\ell_t(\pi^t) = -\frac{1}{|D_{\mathrm{eval}}^t|}"
        r"\sum_{(s,a^*)\in D_{\mathrm{eval}}^t}\log\pi^t(a^*|s),$   "
        r"$a^*(s)=\arg\max_a \pi^*(a|s).$"
    ),
    (
        r"$D_{\mathrm{eval}}^t$: aggregated fresh rollouts of the current "
        r"learner (labeled with expert argmax)."
    ),
    (
        r"Cum. regret: $\sum_{t=1}^T [\ell_t(\pi^t)-\ell_t(\pi^*)]$ "
        r"where $\ell_t(\pi^*)$ is the expert's CE on the same $D_{\mathrm{eval}}^t$."
    ),
    (
        "BC+DAgger train set = expert-data prefix matched to DAgger's "
        "cumulative observation count."
    ),
    (
        "Note: FTL/FTRL use episode-aligned DAgger collection, so obs "
        "counts vary slightly by env/seed (<5% vs fixed-step BC)."
    ),
]


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
        DataFrame with per-round metrics plus top-level ``expert_self_ce``
        carried from the ``baselines`` block.
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

        expert_self_ce = data.get("baselines", {}).get("expert_self_ce")
        for m in data.get("per_round", []):
            row = {
                "algo": data["algo"],
                "env": data["env"],
                "seed": data["seed"],
                "policy_mode": data.get("policy_mode", "unknown"),
                "round": m["round"],
                "n_observations": m.get("n_observations", 0),
                "train_cross_entropy": m.get("train_cross_entropy"),
                "rollout_cross_entropy": m.get("rollout_cross_entropy"),
                "expert_rollout_cross_entropy": m.get(
                    "expert_rollout_cross_entropy"
                ),
                "l2_norm": m.get("l2_norm"),
                "total_loss": m.get("total_loss"),
                "normalized_return": (
                    m["normalized_return"]
                    if m.get("normalized_return") is not None
                    else np.nan
                ),
                "disagreement_rate": (
                    m["disagreement_rate"]
                    if m.get("disagreement_rate") is not None
                    else np.nan
                ),
                "expert_self_ce": expert_self_ce,
            }
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
    """Compute cumulative rollout cross-entropy per (algo, env, seed).

    Also computes cumulative expert rollout cross-entropy on the same
    aggregated D_eval^t buffer (if the field is present).

    Args:
        df: DataFrame from load_results.

    Returns:
        Same DataFrame with added ``cum_loss`` and ``cum_expert_loss``
        columns. Rows with missing ``rollout_cross_entropy`` contribute
        nothing to the cumulative sum.
    """
    df = df.sort_values(["algo", "env", "seed", "round"]).copy()
    # cumsum over eval points; NaN entries (non-eval rounds) contribute 0
    # to the running sum but keep NaN in the cum_loss column so the line
    # plot skips those rounds.
    df["_rce_filled"] = df["rollout_cross_entropy"].fillna(0.0)
    df["cum_loss"] = df.groupby(["algo", "env", "seed"])["_rce_filled"].cumsum()
    df.loc[df["rollout_cross_entropy"].isna(), "cum_loss"] = np.nan
    df.drop(columns=["_rce_filled"], inplace=True)

    if "expert_rollout_cross_entropy" in df.columns:
        df["_erce_filled"] = df["expert_rollout_cross_entropy"].fillna(0.0)
        df["cum_expert_loss"] = df.groupby(["algo", "env", "seed"])[
            "_erce_filled"
        ].cumsum()
        df.loc[
            df["expert_rollout_cross_entropy"].isna(), "cum_expert_loss"
        ] = np.nan
        df.drop(columns=["_erce_filled"], inplace=True)
    return df


def compute_cumulative_regret(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative regret vs the expert on the same D_eval^t buffer.

    Formula matches the DAgger regret bound:
    :math:`\\sum_{t=1}^T [\\ell_t(\\pi^t) - \\ell_t(\\pi^*)]`
    where :math:`\\ell_t` is the sampled-action CE on the aggregated eval
    buffer. The expert's loss on that buffer is stored per-round as
    ``expert_rollout_cross_entropy``.

    If ``expert_rollout_cross_entropy`` is missing (e.g. older JSONs),
    falls back to using the env-level ``expert_self_ce`` constant, which
    approximates expert loss when the learner is near-expert.

    Args:
        df: DataFrame with ``cum_loss`` column (from compute_cumulative_loss).

    Returns:
        Same DataFrame with an added ``cum_regret`` column.
    """
    if "cum_loss" not in df.columns:
        df = compute_cumulative_loss(df)

    if "cum_expert_loss" in df.columns and df["cum_expert_loss"].notna().any():
        df["cum_regret"] = df["cum_loss"] - df["cum_expert_loss"]
    else:
        # Fallback: treat expert loss as constant expert_self_ce per eval
        # point. Count eval points seen so far per run.
        df["_is_eval"] = df["rollout_cross_entropy"].notna().astype(int)
        df["_eval_count"] = df.groupby(["algo", "env", "seed"])[
            "_is_eval"
        ].cumsum()
        df["cum_regret"] = df["cum_loss"] - df["_eval_count"] * df["expert_self_ce"]
        df.drop(columns=["_is_eval", "_eval_count"], inplace=True)
    return df


def _plot_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    log_scale: bool = False,
    allowed_algos: Optional[set] = None,
):
    """Plot a metric with IQM and 95% bootstrap CI bands across seeds.

    Args:
        ax: Matplotlib axes to plot on.
        df: DataFrame filtered to a single env.
        metric: Column name to plot.
        ylabel: Y-axis label.
        log_scale: Whether to use log scale on y-axis.
        allowed_algos: Optional filter on algo set; if given, only those
            algos are drawn on this subplot.
    """
    algos = sorted(
        df["algo"].unique(),
        key=lambda a: list(ALGO_COLORS.keys()).index(a) if a in ALGO_COLORS else 99,
    )
    for algo in algos:
        if allowed_algos is not None and algo not in allowed_algos:
            continue
        algo_df = df[df["algo"] == algo]
        valid_df = algo_df.dropna(subset=[metric])
        if valid_df.empty:
            continue

        # Drop rounds not present in every seed for this algo — otherwise
        # single-seed rounds pull the per-round mean(n_observations) to
        # whatever that lone seed happens to have, causing the line plot
        # to jump left/right on the x-axis.
        n_seeds = valid_df["seed"].nunique()
        seed_counts = valid_df.groupby("round")["seed"].nunique()
        complete_rounds = set(seed_counts[seed_counts == n_seeds].index)
        rounds = sorted(r for r in valid_df["round"].unique() if r in complete_rounds)

        x_vals, iqm_vals, ci_lows, ci_highs = [], [], [], []
        for rnd in rounds:
            rnd_df = valid_df[valid_df["round"] == rnd]
            values = rnd_df[metric].values
            if len(values) == 0:
                continue
            iqm, ci_lo, ci_hi = _compute_iqm_and_ci(values)
            x_vals.append(rnd_df["n_observations"].mean())
            iqm_vals.append(iqm)
            ci_lows.append(ci_lo)
            ci_highs.append(ci_hi)
        if not x_vals:
            continue
        color = ALGO_COLORS.get(algo, "#888888")
        linestyle = ALGO_LINESTYLES.get(algo, "-")
        label = ALGO_LABELS.get(algo, algo)
        ax.plot(
            x_vals,
            iqm_vals,
            color=color,
            linestyle=linestyle,
            label=label,
            linewidth=2,
            marker="o" if linestyle == "-" else None,
            markersize=3,
        )
        ax.fill_between(x_vals, ci_lows, ci_highs, color=color, alpha=0.12)

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
    show_expert_on_loss: bool = True,
) -> None:
    """Generate a 4-subplot figure for one environment.

    Subplot 1: rollout cross-entropy on aggregated D_eval^t (log scale).
    Subplot 2: normalized expected return.
    Subplot 3: on-policy disagreement rate.
    Subplot 4: cumulative regret vs best dynamic algo.

    Args:
        df: Full DataFrame with cum_loss and cum_regret columns.
        env_name: Environment to plot.
        output_path: Path to save the PNG.
        show_expert_on_loss: If True, draw expert self-CE as a dashed line
            on the loss subplot.
    """
    env_df = df[df["env"] == env_name]
    if env_df.empty:
        logger.warning(f"No data for {env_name}, skipping")
        return

    policy_modes = env_df["policy_mode"].unique()
    mode_str = policy_modes[0] if len(policy_modes) == 1 else "mixed"

    # Wider figure so the subtitle has room to breathe and long formulas
    # don't overflow. Room at top is controlled via subplots_adjust;
    # bbox_inches="tight" is intentionally NOT used so the layout isn't
    # recropped out from under us.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4,
        1,
        figsize=(11, 17),
        sharex=True,
    )
    subtitle_text = "\n".join(LOSS_SUBTITLE_LINES)
    # Reserve 17% of the figure for title + subtitle. Subplot 1 top lands
    # at y=0.83, leaving y=0.83..1.00 for text. The extra headroom avoids
    # any overlap between the bottom of the multi-line subtitle and the
    # top spine of subplot 1.
    plt.subplots_adjust(
        top=0.83,
        left=0.09,
        right=0.97,
        bottom=0.05,
        hspace=0.25,
    )
    fig.suptitle(
        f"{env_name}  ({mode_str})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.955,
        subtitle_text,
        ha="center",
        va="top",
        fontsize=10,
        linespacing=1.4,
    )

    # Subplot 1: rollout_cross_entropy on the aggregated D_eval^t buffer
    _plot_metric(
        ax1,
        env_df,
        "rollout_cross_entropy",
        "Rollout CE on $D_{\\mathrm{eval}}^t$ (log)",
        log_scale=True,
        allowed_algos=LOSS_SUBPLOT_ALGOS,
    )
    if show_expert_on_loss:
        expert_self_ce_vals = env_df.get("expert_self_ce")
        if expert_self_ce_vals is not None and expert_self_ce_vals.notna().any():
            y = float(expert_self_ce_vals.dropna().iloc[0])
            ax1.axhline(
                y=y,
                color=ALGO_COLORS["expert"],
                linestyle=ALGO_LINESTYLES["expert"],
                linewidth=1.2,
                alpha=0.7,
                label=f"{ALGO_LABELS['expert']} self-CE = {y:.3f}",
            )
            ax1.legend(fontsize=9)

    # Subplot 2: Normalized expected return
    _plot_metric(
        ax2,
        env_df,
        "normalized_return",
        "Normalized Expected Return",
    )
    ax2.axhline(
        y=1.0,
        color=ALGO_COLORS["expert"],
        linestyle=ALGO_LINESTYLES["expert"],
        linewidth=1.2,
        alpha=0.7,
        label=ALGO_LABELS["expert"] + " (1.0)",
    )
    ax2.axhline(
        y=0.0, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Random (0.0)"
    )
    ax2.legend(fontsize=9)

    # Subplot 3: On-policy disagreement rate (dynamic algos + fixed BC)
    _plot_metric(
        ax3,
        env_df,
        "disagreement_rate",
        "On-Policy Disagreement Rate",
        allowed_algos={"ftl", "ftrl", "bc_dagger", "bc"},
    )
    ax3.set_ylim(-0.05, 1.05)

    # Subplot 4: Cumulative regret vs expert
    _plot_metric(
        ax4,
        env_df,
        "cum_regret",
        r"Cumulative Regret (vs Expert $\pi^*$)",
        allowed_algos=LOSS_SUBPLOT_ALGOS,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight" — we deliberately reserved the top 12% for
    # title + subtitle via subplots_adjust, and tight-cropping would
    # claw back exactly that space.
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved plot: {output_path}")


def plot_all(
    results_dir: pathlib.Path,
    output_dir: pathlib.Path,
    envs: Optional[List[str]] = None,
    show_expert_on_loss: bool = True,
) -> List[pathlib.Path]:
    """Generate plots for all environments.

    Args:
        results_dir: Directory with JSON result files.
        output_dir: Directory to save PNG plots.
        envs: Optional list of envs to filter. If None, plots all found.
        show_expert_on_loss: Forwarded to ``plot_env``.

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
        plot_env(df, env_name, out_path, show_expert_on_loss=show_expert_on_loss)
        saved_paths.append(out_path)

    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Plot FTL vs FTRL vs BC+DAgger vs BC experiment results",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="Directory containing JSON result files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/plots",
        help="Directory to save PNG plots",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Filter to specific environments",
    )
    parser.add_argument(
        "--show-expert-on-loss",
        dest="show_expert_on_loss",
        action="store_true",
        default=True,
        help="Draw expert self-CE on the loss subplot (default on)",
    )
    parser.add_argument(
        "--hide-expert-on-loss",
        dest="show_expert_on_loss",
        action="store_false",
        help="Hide expert self-CE on the loss subplot",
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
        show_expert_on_loss=args.show_expert_on_loss,
    )
    if paths:
        logger.info(f"Generated {len(paths)} plots in {args.output_dir}")
    else:
        logger.warning("No plots generated -- check results directory")


if __name__ == "__main__":
    main()
