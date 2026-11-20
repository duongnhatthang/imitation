"""Plotting for FTL vs FTRL vs BC-iid vs BC experiment results.

Generates per-environment figures with 4 subplots (linear x-axis by default):
  1. Rollout cross-entropy on the aggregated D_eval^t buffer (log y)
  2. Normalized expected return (linear y, fixed view NORMALIZED_RETURN_YLIM)
  3. On-policy disagreement rate (log y, floor 1e-3)
  4. Cumulative regret vs expert pi* (linear y)

Bands default to IQM + 95% stratified bootstrap CI (via rliable); pass
``--band sem`` for mean +/- 1 SEM instead (narrower, parametric).
Supports incremental generation -- can plot partial results as experiments complete.

Usage:
    python -m imitation.experiments.ftrl.plot_results --results-dir experiments/results/
    python -m imitation.experiments.ftrl.plot_results --envs CartPole-v1 FrozenLake-v1
"""

import argparse
import json
import logging
import math
import numbers
import pathlib
import re
import textwrap
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from imitation.experiments.ftrl.plot_return_range import (
    RANGE_RULE,
    automatic_return_limits,
)

logger = logging.getLogger(__name__)

# Use non-interactive backend for server/CI
matplotlib.use("Agg")

ALGO_COLORS: Dict[str, str] = {
    "ftl": "#1f77b4",  # blue
    "ftrl": "#d62728",  # red
    "bc_iid": "#2ca02c",  # green
    "bc_pool": "#9467bd",  # purple (fixed pool, uniform draws)
    "bc_prefix": "#8c564b",  # brown (fixed pool, chronological)
    "bc": "#17a663",  # dark green (dashed reference)
    "expert": "#555555",  # gray (dashed reference)
}

ALGO_LABELS: Dict[str, str] = {
    "ftl": "FTL+DAgger",
    "ftrl": "FTRL+DAgger",
    "bc_iid": "BC-iid",
    "bc_pool": "BC-pool",
    "bc_prefix": "BC-prefix",
    "bc": "BC (fixed)",
    "expert": "Expert",
}

ALGO_LINESTYLES: Dict[str, str] = {
    "ftl": "-",
    "ftrl": "-",
    "bc_iid": "-",
    "bc_pool": "-",
    "bc_prefix": "-",
    "bc": "--",
    "expert": "--",
}

LOSS_SUBPLOT_ALGOS = {"ftl", "ftrl", "bc_iid", "bc_pool", "bc_prefix", "bc"}

# Fixed linear display range of every normalized-return panel. It only sets the
# view: plotted values, bands and receipts are never clipped to it.
NORMALIZED_RETURN_YLIM: Tuple[float, float] = (-0.5, 1.5)


def set_normalized_return_axis(
    ax: plt.Axes, limits: Tuple[float, float] = NORMALIZED_RETURN_YLIM,
) -> None:
    """Show a normalized-return panel on the shared linear fixed view."""
    ax.set_yscale("linear")
    ax.set_ylim(*limits)


def _draw_bc_hlines(
    axes: List[plt.Axes],
    env_df: pd.DataFrame,
    metrics: List[str],
) -> Dict[str, float]:
    """Draw BC as horizontal reference lines at the stored values, unclipped.

    Returns:
        The drawn value per metric.
    """
    drawn: Dict[str, float] = {}
    bc_df = env_df[env_df["algo"] == "bc"]
    if bc_df.empty:
        return drawn
    bc_row = bc_df.iloc[0]
    color = ALGO_COLORS.get("bc", "#17a663")
    linestyle = ALGO_LINESTYLES.get("bc", "--")
    label = ALGO_LABELS.get("bc", "BC (fixed)")

    for ax, metric in zip(axes, metrics):
        val = bc_row.get(metric)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        ax.axhline(
            y=float(val),
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            alpha=0.7,
            label=f"{label} = {float(val):.3f}",
        )
        ax.legend(fontsize=9)
        drawn[metric] = float(val)
    return drawn


def _env_category(env_name: str) -> str:
    """Route env to 'atari' or 'classical' output subdir."""
    return "atari" if "NoFrameskip" in env_name else "classical"


LOSS_SUBTITLE_LINES = [
    (
        r"Loss: $\ell_t(\pi^t) = -\frac{1}{|D_{\mathrm{eval}}^t|}"
        r"\sum_{(s,a^*)\in D_{\mathrm{eval}}^t}\log\pi^t(a^*|s),$   "
        r"$a^*(s)=\arg\max_a \pi^*(a|s).$"
    ),
    (
        r"$D_{\mathrm{eval}}^t$: deterministic rollout (per run config) of the current "
        r"learner $\pi^t$ (labeled with expert argmax). Not aggregated."
    ),
    (
        r"Cumulative excess CE: $\sum_{t\ \mathrm{logged}} [\ell_t(\pi^t)-\ell_t(\pi^*)]$ "
        r"on the same $D_{\mathrm{eval}}^t$ at each logged evaluation."
    ),
    (
        "x-axis = retained training observations. Fixed BC uses the full dataset budget; "
        "collection and evaluation queries are additional."
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


def _compute_mean_and_sem(
    values_by_seed: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute the across-seed mean and a ±1 SEM band.

    SEM = sample std (ddof=1) / sqrt(n). This is a narrower, parametric
    alternative to the IQM + bootstrap-CI band; the center is the plain
    mean (not the IQM), so the two band modes are not centered identically.

    Args:
        values_by_seed: 1D array of values, one per seed.

    Returns:
        (mean, mean - sem, mean + sem). For n < 2 the band collapses to
        the point estimate.
    """
    vals = np.asarray(values_by_seed, dtype=float)
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(vals))
    if n < 2:
        return mean, mean, mean
    sem = float(np.std(vals, ddof=1) / np.sqrt(n))
    return mean, mean - sem, mean + sem


def _compute_center_and_band(
    values_by_seed: np.ndarray,
    band: str,
) -> Tuple[float, float, float]:
    """Dispatch to the chosen aggregation: ``ci`` (IQM+bootstrap) or ``sem``.

    Args:
        values_by_seed: 1D array of values, one per seed.
        band: ``"ci"`` for IQM + 95% bootstrap CI, ``"sem"`` for mean ± 1 SEM.

    Returns:
        (center, low, high).

    Raises:
        ValueError: If ``band`` is not ``"ci"`` or ``"sem"``.
    """
    if band == "ci":
        return _compute_iqm_and_ci(values_by_seed)
    if band == "sem":
        return _compute_mean_and_sem(values_by_seed)
    raise ValueError(f"Unknown band type: {band!r} (expected 'ci' or 'sem')")


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

    for json_file in sorted(
        p
        for p in results_path.rglob("*.json")
        if not any(part.startswith("_archive") for part in p.parts)
    ):
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
                "expert_rollout_cross_entropy": m.get("expert_rollout_cross_entropy"),
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

    _check_stale_data(df)
    return df


def _check_stale_data(df: pd.DataFrame) -> None:
    """Warn if seeds within the same (algo, env) have inconsistent n_observations.

    This catches stale results from a previous run (e.g. different
    samples_per_round) mixed with current results.
    """
    for (algo, env), group in df.groupby(["algo", "env"]):
        per_seed_rounds = {}
        for seed, seed_df in group.groupby("seed"):
            obs_by_round = dict(zip(seed_df["round"], seed_df["n_observations"]))
            per_seed_rounds[seed] = obs_by_round

        seeds = list(per_seed_rounds.keys())
        if len(seeds) < 2:
            continue
        ref = per_seed_rounds[seeds[0]]
        for seed in seeds[1:]:
            other = per_seed_rounds[seed]
            shared = set(ref.keys()) & set(other.keys())
            for rnd in shared:
                if ref[rnd] != other[rnd]:
                    logger.warning(
                        f"STALE DATA: {algo}/{env} round {rnd}: "
                        f"seed {seeds[0]} has n_obs={ref[rnd]}, "
                        f"seed {seed} has n_obs={other[rnd]}. "
                        f"Delete old results and re-run."
                    )
                    return


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
        df.loc[df["expert_rollout_cross_entropy"].isna(), "cum_expert_loss"] = np.nan
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
        df["_eval_count"] = df.groupby(["algo", "env", "seed"])["_is_eval"].cumsum()
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
    log_x: bool = False,
    y_clip_floor: Optional[float] = None,
    band: str = "ci",
) -> Dict[str, List[Dict[str, Any]]]:
    """Plot a metric's center line with an uncertainty band across seeds.

    Args:
        ax: Matplotlib axes to plot on.
        df: DataFrame filtered to a single env.
        metric: Column name to plot.
        ylabel: Y-axis label.
        log_scale: Whether to use log scale on y-axis.
        allowed_algos: Optional filter on algo set; if given, only those
            algos are drawn on this subplot.
        log_x: Whether to use log scale on x-axis. Rounds with
            ``n_observations == 0`` are dropped (can't render at log(0)).
        y_clip_floor: If set, values below this floor are clipped up to
            the floor before plotting. Useful for log-y plots where the
            natural metric can be zero or slightly negative.
        band: Uncertainty band type — ``"ci"`` (IQM + 95% stratified
            bootstrap CI, the default) or ``"sem"`` (mean ± 1 SEM).

    Returns:
        Drawn points per algo: center (``mean``), band ``low``/``high`` and the
        per-seed ``values`` they aggregate.
    """
    drawn: Dict[str, List[Dict[str, Any]]] = {}
    algos = sorted(
        df["algo"].unique(),
        key=lambda a: list(ALGO_COLORS.keys()).index(a) if a in ALGO_COLORS else 99,
    )
    for algo in algos:
        if allowed_algos is not None and algo not in allowed_algos:
            continue
        if algo == "bc":
            continue  # BC drawn as horizontal lines by _draw_bc_hlines
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

        x_vals, center_vals, band_lows, band_highs = [], [], [], []
        seed_vals = []
        for rnd in rounds:
            rnd_df = valid_df[valid_df["round"] == rnd]
            values = rnd_df[metric].values
            if len(values) == 0:
                continue
            if y_clip_floor is not None:
                values = np.maximum(values, y_clip_floor)
            center, band_lo, band_hi = _compute_center_and_band(values, band)
            mean_obs = rnd_df["n_observations"].mean()
            x_vals.append(mean_obs)
            center_vals.append(center)
            band_lows.append(band_lo)
            band_highs.append(band_hi)
            seed_vals.append([float(v) for v in values])
        if not x_vals:
            continue
        color = ALGO_COLORS.get(algo, "#888888")
        linestyle = ALGO_LINESTYLES.get(algo, "-")
        label = ALGO_LABELS.get(algo, algo)
        if y_clip_floor is not None:
            center_vals = [max(v, y_clip_floor) for v in center_vals]
            band_lows = [max(v, y_clip_floor) for v in band_lows]
            band_highs = [max(v, y_clip_floor) for v in band_highs]
        ax.plot(
            x_vals,
            center_vals,
            color=color,
            linestyle=linestyle,
            label=label,
            linewidth=2,
            marker="o" if linestyle == "-" else None,
            markersize=3,
        )
        ax.fill_between(x_vals, band_lows, band_highs, color=color, alpha=0.12)
        drawn[algo] = [
            {"mean": c, "low": lo, "high": hi, "values": v, "budget": x}
            for c, lo, hi, v, x in zip(center_vals, band_lows, band_highs, seed_vals, x_vals)
        ]

    if log_scale:
        ax.set_yscale("log")
    if log_x:
        ax.set_xscale("symlog", linthresh=100)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.tick_params(axis="x", which="major", labelsize=8, rotation=0)
    # Force x tick labels on every subplot (sharex otherwise hides
    # labels on all but the bottom axis).
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel("Number of Observations" + (" (log)" if log_x else ""))
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    return drawn


def plot_env(
    df: pd.DataFrame,
    env_name: str,
    output_path: pathlib.Path,
    show_expert_on_loss: bool = True,
    calibration: Optional[Dict[str, Any]] = None,
    band: str = "ci",
    return_limits: Optional[Tuple[float, float]] = NORMALIZED_RETURN_YLIM,
) -> None:
    """Generate a 4-subplot figure for one environment.

    Subplot 1: rollout cross-entropy on aggregated D_eval^t (log scale).
    Subplot 2: normalized expected return (linear, fixed view
    ``NORMALIZED_RETURN_YLIM``; values are not clipped).
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
        left=0.10,
        right=0.97,
        bottom=0.05,
        hspace=0.38,
    )
    fig.suptitle(
        f"{env_name}  ({mode_str})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    # Add calibration info line if available
    cal_line = ""
    if calibration:
        lr = calibration.get("best_lr")
        t_sat = calibration.get("best_t_sat")
        lr_str = f"lr={lr:.0e}" if lr is not None else "lr=default"
        t_sat_str = f"T_sat={t_sat} obs" if t_sat is not None else "T_sat=N/A"
        cal_line = f"\nCalibrated: {lr_str},  {t_sat_str},  max_obs=2×T_sat"

    band_desc = "IQM + 95% stratified bootstrap CI" if band == "ci" else "mean ± 1 SEM"
    band_line = f"\nBands: {band_desc} across seeds"

    fig.text(
        0.5,
        0.955,
        subtitle_text + cal_line + band_line,
        ha="center",
        va="top",
        fontsize=10,
        linespacing=1.4,
    )

    # Subplot 1: rollout_cross_entropy on the aggregated D_eval^t buffer.
    _plot_metric(
        ax1,
        env_df,
        "rollout_cross_entropy",
        r"Rollout CE on $D_{\mathrm{eval}}^t$",
        log_scale=True,
        allowed_algos=LOSS_SUBPLOT_ALGOS,
        band=band,
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
                label=rf"{ALGO_LABELS['expert']} $\pi^*$ self-CE = {y:.3f}",
            )
            ax1.legend(fontsize=9)

    # Subplot 2: Normalized expected return, unclipped on a linear axis; the
    # fixed view is applied after the BC reference lines are drawn.
    return_points = _plot_metric(
        ax2,
        env_df,
        "normalized_return",
        "Normalized Return",
        band=band,
    )
    ax2.axhline(
        y=1.0,
        color=ALGO_COLORS["expert"],
        linestyle=ALGO_LINESTYLES["expert"],
        linewidth=1.2,
        alpha=0.7,
        label=rf"{ALGO_LABELS['expert']} $\pi^*$",
    )
    ax2.legend(fontsize=9)

    # Subplot 3: On-policy disagreement rate (dynamic algos + fixed BC).
    # Log y-axis (natural for a rate in [0, 1]).
    _plot_metric(
        ax3,
        env_df,
        "disagreement_rate",
        "Disagreement Rate",
        log_scale=True,
        y_clip_floor=1e-3,
        allowed_algos={"ftl", "ftrl", "bc_iid", "bc_pool", "bc_prefix", "bc"},
        band=band,
    )

    # Subplot 4: Cumulative regret vs expert. Linear y (regret can be negative).
    _plot_metric(
        ax4,
        env_df,
        "cum_regret",
        r"Cumulative Evaluated Excess CE (vs Expert $\pi^*$)",
        allowed_algos=LOSS_SUBPLOT_ALGOS,
        band=band,
    )

    bc_drawn = _draw_bc_hlines(
        [ax1, ax2, ax3, ax4],
        env_df,
        [
            "rollout_cross_entropy",
            "normalized_return",
            "disagreement_rate",
            "cum_regret",
        ],
    )
    if return_limits is None:
        series = []
        for points in return_points.values():
            series.append((
                [p["budget"] for p in points],
                [p["low"] for p in points],
                [p["high"] for p in points],
            ))
        if "normalized_return" in bc_drawn:
            values = env_df.loc[env_df["algo"] == "bc", "normalized_return"].dropna()
            _, low, high = _compute_center_and_band(values.to_numpy(), band)
            series.append(([1], [min(low, bc_drawn["normalized_return"])],
                           [max(high, bc_drawn["normalized_return"])]))
        return_limits = automatic_return_limits(series)
        ax2.set_title("Automatic return view (" + RANGE_RULE + ")", fontsize=9)
    set_normalized_return_axis(ax2, return_limits)
    # Audit the drawn centers, band edges and the BC line (with the stored BC
    # values); the expert line at 1 and the axis ticks are not data.
    if "normalized_return" in bc_drawn:
        val = bc_drawn["normalized_return"]
        bc_values = env_df.loc[env_df["algo"] == "bc", "normalized_return"]
        return_points["bc"] = [
            {
                "mean": val,
                "low": val,
                "high": val,
                "values": [float(v) for v in bc_values.dropna()],
            },
        ]
    scale_notes = _normalized_scale_notes(return_points, return_limits)
    if scale_notes:
        fig.text(
            0.5,
            0.008,
            " ".join(scale_notes),
            ha="center",
            va="bottom",
            fontsize=9,
            gid="normalized_return_scale_note",
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
    calibration_data: Optional[Dict[str, Any]] = None,
    band: str = "ci",
    flat_output: bool = False,
    zoom_classical_return: bool = False,
) -> List[pathlib.Path]:
    """Generate plots for all environments.

    Args:
        results_dir: Directory with JSON result files.
        output_dir: Directory to save PNG plots.
        envs: Optional list of envs to filter. If None, plots all found.
        show_expert_on_loss: Forwarded to ``plot_env``.
        calibration_data: Optional dict from lr_calibration.json, keyed by env.
        band: Uncertainty band type forwarded to ``plot_env`` — ``"ci"`` or
            ``"sem"``.

        flat_output: Save directly in output_dir, without another environment-family folder.
        zoom_classical_return: Show the near-expert return range for classical environments.

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
        parent = pathlib.Path(output_dir)
        if not flat_output:
            parent /= _env_category(env_name)
        out_path = parent / f"{safe_name}.png"
        cal = (calibration_data or {}).get(env_name)
        plot_env(
            df,
            env_name,
            out_path,
            show_expert_on_loss=show_expert_on_loss,
            calibration=cal,
            band=band,
            return_limits=None
            if zoom_classical_return and _env_category(env_name) == "classical"
            else NORMALIZED_RETURN_YLIM,
        )
        saved_paths.append(out_path)

    return saved_paths


# ---------------------------------------------------------------------------
# Four-method restriction comparison (agnostic study run records)
# ---------------------------------------------------------------------------

RESTRICTION_METHODS: Tuple[str, ...] = ("bc", "bc_iid", "ftl", "bc_prefix")
RESTRICTION_SEEDS: Tuple[int, ...] = (3100, 3101, 3102, 3103, 3104)
RESTRICTION_PANELS: Tuple[Tuple[str, str], ...] = (
    ("normalized_return", "Normalized return"),
    ("disagreement_rate", "On-policy disagreement rate"),
    ("rollout_cross_entropy", "Rollout cross-entropy"),
)
RESTRICTION_LABELS: Dict[str, str] = {
    "bc": "Offline BC (one fit on the full pool)",
    "bc_iid": "BC-iid",
    "ftl": "FTL+DAgger",
    "bc_prefix": "BC-prefix (chronological pool)",
}
RESTRICTION_EVAL_EPISODES = 100

RETURN_SCALE_FOOTNOTE = "Random=0, expert=1; values outside [0,1] are valid."
OFFSCALE_FOOTNOTE = "Off-scale portions are not shown; values unchanged."
# CliffWalking-v0 is displayed on the uniform-random reference like every other
# environment, with the expert at 1. A policy that stalls for the whole episode
# (-episode_cap) is drawn as an additional benchmark line only; it is never a
# normalization reference. Historical runs stored the uniform-random reference
# and are plotted as stored. Diagnostic runs that stored the stalled reference
# (``normalization_reference_type`` ``stalled_policy``) are redisplayed from
# their raw episode means and their recorded measured uniform-random return.
STALL_REFERENCE_ENVS = frozenset({"CliffWalking-v0"})
STALL_EPISODE_CAP = 200
REFERENCE_UNIFORM = "uniform_random"
REFERENCE_STALLED = "stalled_policy"
REFERENCE_TYPES = (REFERENCE_UNIFORM, REFERENCE_STALLED)
# Historical CliffWalking column-pair mask (r, c//2); see the diagnostic note.
CLIFF_COLUMN_PAIRS_ID = "cliffwalking_column_pairs"
# An exhaustive search over the deterministic memoryless policies of this masked
# observation found none that reaches the goal; see _cliff_column_pairs_audit.
CLIFF_COLUMN_PAIRS_NOTE = (
    "Column-pair bins: no goal-reaching deterministic memoryless policy."
)
CLIFF_SUCCESS_NOT_RECORDED_NOTE = (
    "Goal success not recorded ({missing}/{total} rows); not imputed."
)
# Return of one cliff fall relative to an ordinary step (-100 instead of -1).
_CLIFF_FALL_EXTRA = 99
STALL_BENCHMARK_FOOTNOTE = (
    "Cliff stall benchmark: R={stall:.6g} (normalized {normalized}); random "
    "normalization unchanged."
)
STALLED_REDISPLAY_FOOTNOTE = (
    "Stored stall-referenced values redisplayed on measured random "
    "(R_random={random}); records unchanged."
)
REFERENCES_DIFFER_FOOTNOTE = (
    "Normalization references differ across runs; per-run values in receipt."
)
STALL_BENCHMARK_LABEL = "Stalled policy (R = {stall:.6g}), benchmark only"
EXPERT_LINE_LABEL = "Expert reference (normalized 1)"
_STALL_LINE_COLOR = "#8a5a00"
# Tolerance of the stored normalized_return, which is rounded to 6 decimals.
_STORED_NORM_ATOL = 1e-6
# Only rounding-level SEM band edges can push these axes past their bounds.
CE_AXIS_PADDING_FOOTNOTE = "CE axis below 0 shows band edges only; every CE >= 0."
DISAGREEMENT_AXIS_PADDING_FOOTNOTE = (
    "Disagreement axis outside [0, 1] shows band edges only; every rate is in [0, 1]."
)

# Fixed physical layout (inches); text block heights are measured at draw time.
_RC_WIDTH = 16.0
_RC_PANEL_HEIGHT = 4.2
_RC_PAD = 0.12
_RC_GAP = 0.12
_RC_FONTSIZE = 8.5
_RC_LEFT = 0.06
_RC_RIGHT = 0.99


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_count(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _audit_normalization_reference(run: Mapping[str, Any]) -> Dict[str, Any]:
    """The baselines a run recorded its returns with (pinned reference first).

    This is the audit reference: the stored ``normalized_return`` values were
    computed with it, whatever the figure displays. ``random_return`` is the
    recorded field verbatim, i.e. the stored lower reference. Baselines without
    ``normalization_reference_type`` predate the stalled reference and hold the
    uniform-random reference. A ``stalled_policy`` reference is accepted only on
    CliffWalking-v0, with ``random_return`` equal to ``stalled_return`` and a
    finite ``measured_uniform_random_return``.

    Raises:
        ValueError: If the baselines are missing, non-finite or contradictory.
    """
    reference = (run.get("config") or {}).get("reference") or {}
    baselines = reference.get("baselines") or (run.get("data") or {}).get(
        "baselines",
    )
    label = f"{run.get('method')} seed {run.get('seed')}"
    if not isinstance(baselines, Mapping) or not all(
        _is_finite_number(baselines.get(k)) for k in ("random_return", "expert_return")
    ):
        raise ValueError(f"{label}: no finite normalization reference baselines")
    recorded = "normalization_reference_type" in baselines
    kind = baselines.get("normalization_reference_type", REFERENCE_UNIFORM)
    if kind not in REFERENCE_TYPES:
        raise ValueError(f"{label}: unknown normalization_reference_type {kind!r}")
    lower = float(baselines["random_return"])
    measured = baselines.get("measured_uniform_random_return")
    stalled = baselines.get("stalled_return")
    for name, value in (
        ("measured_uniform_random_return", measured),
        ("stalled_return", stalled),
    ):
        if value is not None and not _is_finite_number(value):
            raise ValueError(f"{label}: {name} must be finite, got {value!r}")
    if kind == REFERENCE_STALLED:
        if run.get("env_name") not in STALL_REFERENCE_ENVS:
            raise ValueError(
                f"{label}: stalled_policy reference on {run.get('env_name')!r}; "
                f"it is allowed only on {sorted(STALL_REFERENCE_ENVS)}",
            )
        if stalled is None or float(stalled) != lower:
            raise ValueError(
                f"{label}: stalled_policy reference needs stalled_return equal to "
                f"random_return {lower}, got {stalled!r}",
            )
        if measured is None:
            raise ValueError(
                f"{label}: stalled_policy reference lacks "
                "measured_uniform_random_return",
            )
        if float(measured) >= lower:
            raise ValueError(
                f"{label}: measured_uniform_random_return {measured} is not below "
                f"the stalled return {lower}",
            )
    else:
        if stalled is not None:
            raise ValueError(
                f"{label}: uniform_random reference also records stalled_return "
                f"{stalled}",
            )
        if measured is not None and float(measured) != lower:
            raise ValueError(
                f"{label}: uniform_random reference random_return {lower} differs "
                f"from measured_uniform_random_return {measured}",
            )
        if run.get("env_name") in STALL_REFERENCE_ENVS and lower >= -STALL_EPISODE_CAP:
            # A uniform-random CliffWalking return is tens of thousands below.
            raise ValueError(
                f"{label}: uniform_random reference {lower} is not below the "
                f"stalled return {-STALL_EPISODE_CAP}; refusing to label it "
                "uniform-random",
            )
    descriptor = (run.get("config") or {}).get("normalization_reference")
    if isinstance(descriptor, Mapping) and (
        descriptor.get("type", kind) != kind
        or descriptor.get("lower_return", lower) != lower
    ):
        raise ValueError(
            f"{label}: config normalization_reference {dict(descriptor)} "
            f"disagrees with the baselines ({kind}, {lower})",
        )
    return {
        "method": run.get("method"),
        "seed": run.get("seed"),
        "normalization_reference_type": kind,
        "reference_type_recorded": recorded,
        "random_return": lower,
        "expert_return": float(baselines["expert_return"]),
        "measured_uniform_random_return": (
            lower if measured is None else float(measured)
        ),
        "stalled_return": None if stalled is None else float(stalled),
    }


def _uses_stall_reference(runs: Sequence[Mapping[str, Any]]) -> bool:
    return runs[0].get("env_name") in STALL_REFERENCE_ENVS


def _stall_return(run: Mapping[str, Any]) -> float:
    """``-episode_cap`` of a CliffWalking run: the return of stalling all episode."""
    cap = run.get("episode_cap")
    label = f"{run.get('method')} seed {run.get('seed')}"
    if not _is_count(cap) or cap <= 0:
        raise ValueError(f"{label}: episode_cap must be a positive int, got {cap!r}")
    if cap != STALL_EPISODE_CAP:
        raise ValueError(
            f"{label}: episode_cap {cap} is not the stalled-reference cap "
            f"{STALL_EPISODE_CAP}",
        )
    return -float(cap)


def _display_normalization_reference(run: Mapping[str, Any]) -> Dict[str, Any]:
    """The lower and upper references the figure normalizes this run's returns by.

    The lower reference is always the uniform-random policy. For a stored
    ``stalled_policy`` reference it is the recorded
    ``measured_uniform_random_return``, never an inferred value. On
    CliffWalking the stalled-policy benchmark (``-episode_cap``) and its value
    on this scale are added; the benchmark is not a reference.

    Raises:
        ValueError: If the CliffWalking cap, stall or expert are inconsistent.
    """
    audit = _audit_normalization_reference(run)
    stored_stalled = audit["normalization_reference_type"] == REFERENCE_STALLED
    display = {
        "method": audit["method"],
        "seed": audit["seed"],
        "lower_reference": "uniform-random policy",
        "lower_return": audit["measured_uniform_random_return"],
        "lower_return_source": (
            "recorded measured_uniform_random_return"
            if stored_stalled
            else "recorded random_return"
        ),
        "expert_return": audit["expert_return"],
    }
    if run.get("env_name") in STALL_REFERENCE_ENVS:
        label = f"{audit['method']} seed {audit['seed']}"
        stall = _stall_return(run)
        stored = audit["stalled_return"]
        if stored is not None and stored != stall:
            raise ValueError(
                f"{label}: recorded stalled_return {stored} differs from "
                f"-episode_cap {stall}",
            )
        lower, expert = display["lower_return"], display["expert_return"]
        if not lower < stall < expert:
            raise ValueError(
                f"{label}: need uniform-random {lower} < stall {stall} < expert "
                f"{expert} to draw the stalled-policy benchmark",
            )
        display["stall_return"] = stall
        display["stall_normalized"] = (stall - lower) / (expert - lower)
    return display


def _cliff_display_returns(
    group: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
) -> Optional[Dict[Tuple[str, Any], List[float]]]:
    """Verify CliffWalking stored returns; redisplay stored-stall runs only.

    Each stored ``normalized_return`` must match its raw episode mean under the
    run's recorded (audit) reference. Uniform-random runs are then plotted as
    stored (``None``). Stored ``stalled_policy`` runs are recomputed from their
    raw episode means on the recorded measured uniform-random reference; the
    run records are never modified.

    Raises:
        ValueError: If a stored value disagrees with its raw episodes.
    """
    by_cell = {(run["method"], run["seed"]): run for run in runs}
    display = {}
    recompute = False
    for cell, rows in group["evaluations"].items():
        audit = _audit_normalization_reference(by_cell[cell])
        ref = _display_normalization_reference(by_cell[cell])
        recompute = audit["normalization_reference_type"] == REFERENCE_STALLED
        values = []
        for row in rows:
            mean = float(np.mean(row["episode_returns"]))
            scale = audit["expert_return"] - audit["random_return"]
            recorded = (
                0.0 if abs(scale) < 1e-8 else (mean - audit["random_return"]) / scale
            )
            if abs(float(row["normalized_return"]) - recorded) > _STORED_NORM_ATOL:
                raise ValueError(
                    f"{cell[0]} seed {cell[1]} budget {row['n_observations']}: "
                    f"stored normalized_return {row['normalized_return']} does not "
                    f"match its episode returns under the recorded lower reference "
                    f"{audit['random_return']:.6g} ({recorded:.6g})",
                )
            values.append(
                (mean - ref["lower_return"])
                / (ref["expert_return"] - ref["lower_return"]),
            )
        display[cell] = values
    # One stored reference type per figure is enforced before this is called.
    return display if recompute else None


def _stall_benchmark(runs: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The CliffWalking stalled-policy benchmark line, else ``None``.

    Each run's stall is normalized by that run's own display references. The
    line is drawn at the mean over runs; when runs differ, the range is shaded
    and every per-run value is listed.
    """
    if not _uses_stall_reference(runs):
        return None
    refs = [_display_normalization_reference(run) for run in runs]
    raws = sorted({r["stall_return"] for r in refs})
    if len(raws) != 1:
        raise ValueError(f"Runs record different stalled returns {raws}")
    values = [r["stall_normalized"] for r in refs]
    return {
        "env_name": runs[0]["env_name"],
        "role": "additional benchmark, not a normalization reference",
        "raw_return": raws[0],
        "episode_cap": runs[0]["episode_cap"],
        "formula": "(R_stall - R_random)/(R_expert - R_random), per run",
        "per_run": [
            {"method": r["method"], "seed": r["seed"], "normalized": v}
            for r, v in zip(refs, values)
        ],
        "mean": float(np.mean(values)),
        "min": min(values),
        "max": max(values),
        "identical": len(set(values)) == 1,
    }


def _stall_value_text(benchmark: Mapping[str, Any]) -> str:
    if benchmark["identical"]:
        return f"{benchmark['mean']:.3f}"
    return f"{benchmark['min']:.3f} to {benchmark['max']:.3f} across runs, shaded"


def _cliff_column_pairs_audit(
    runs: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Checks of a CliffWalking column-pair figure, else ``None``.

    An exhaustive search found that no deterministic memoryless policy of the
    column-pair observation (one fixed start, deterministic dynamics, capped
    episodes) reaches the goal, so every capped episode returns
    ``R_stall - 99k`` for ``k`` cliff falls and returns below R_stall measure
    cliff penalties, not goal progress. Stochastic or memory-based policies are
    not covered. The figure is accepted only when every evaluation episode
    return has that form, a necessary but not sufficient check (a goal reached
    at step 101 after one fall also returns -200). Goal success is never
    inferred from returns.

    Raises:
        ValueError: If an episode return or a recorded success contradicts
            the exhaustive result.
    """
    first = runs[0]
    if (
        first.get("env_name") not in STALL_REFERENCE_ENVS
        or first.get("restriction_id") != CLIFF_COLUMN_PAIRS_ID
    ):
        return None
    stall = _stall_return(first)
    missing = total = 0
    for run in runs:
        for row in run.get("records") or []:
            if row.get("d_eval_size") is None:
                continue
            total += 1
            for value in row["episode_returns"]:
                falls = (stall - float(value)) / _CLIFF_FALL_EXTRA
                if falls < 0 or abs(falls - round(falls)) > 1e-9:
                    raise ValueError(
                        f"{run['method']} seed {run['seed']} budget "
                        f"{row['n_observations']}: episode return {value} is not "
                        f"{stall:.6g} - {_CLIFF_FALL_EXTRA}k, contradicting the "
                        "column-pair goal-infeasibility result",
                    )
            successes = row.get("episode_successes")
            if successes is None:
                missing += 1
            elif any(successes):
                raise ValueError(
                    f"{run['method']} seed {run['seed']} budget "
                    f"{row['n_observations']}: recorded goal success contradicts "
                    "the column-pair goal-infeasibility result",
                )
    return {
        "checked_rows": total,
        "success_not_recorded_rows": missing,
        "episode_returns_on_stall_lattice": True,
        "scope": "deterministic memoryless policies, exhaustive search",
    }


def _cliff_column_pairs_note(audit: Optional[Mapping[str, Any]]) -> List[str]:
    if audit is None:
        return []
    note = CLIFF_COLUMN_PAIRS_NOTE
    if audit["success_not_recorded_rows"]:
        note += " " + CLIFF_SUCCESS_NOT_RECORDED_NOTE.format(
            missing=audit["success_not_recorded_rows"],
            total=audit["checked_rows"],
        )
    return [note]


_ZEROED_NAMES = re.compile(r"features (.+?) := 0")


def _feature_names_of(run: Mapping[str, Any]) -> Optional[List[str]]:
    names = ((run.get("task") or {}).get("declared") or {}).get("feature_names")
    return list(names) if isinstance(names, (list, tuple)) else None


def _quantized_text(desc: Mapping[str, Any]) -> str:
    parts = []
    for feature in desc.get("features") or []:
        name, rule = feature.get("name"), feature.get("rule")
        if rule == "round_to_step":
            parts.append(f"{name} step {feature['step']:g}")
        elif rule == "sign_dead_zone":
            m = feature["magnitude"]
            parts.append(
                f"{name} in {{-{m:g}, 0, {m:g}}} (dead zone {feature['dead_zone']:g})",
            )
        else:
            parts.append(f"{name} {rule}")
    if not parts:
        raise ValueError("quantized restriction descriptor records no features")
    return "Quantized: " + "; ".join(parts) + "."


def _random_bins_text(
    desc: Mapping[str, Any],
    env_name: str,
    rid: str,
) -> str:
    """Random-bin K and partition seed, cross-checked against the run."""
    partition = desc["partition"]
    if (
        not isinstance(partition, Mapping)
        or partition.get("kind") != "random_state_binning"
        or partition.get("env_name") != env_name
        or partition.get("restriction_id") != rid
        or not _is_count(partition.get("n_bins"))
        or not _is_count(partition.get("seed"))
        or partition["n_bins"] != desc.get("reachable_classes")
    ):
        raise ValueError(
            f"{rid}: random-bin partition descriptor does not match the run "
            f"({env_name}, {desc.get('reachable_classes')} reachable classes)",
        )
    states = partition.get("n_reachable_nonterminal")
    over = f" over {states} reachable states" if _is_count(states) else ""
    return (
        f"Random bins: K={partition['n_bins']}{over}, partition seed "
        f"{partition['seed']}."
    )


_GRID_KINDS = ("linear", "unbounded", "circular", "binary")
_GRID_RECORD_EXTRAS = ("kind", "descriptor_sha256", "nominal_cells")


def _check_random_grid_binding(
    desc: Mapping[str, Any],
    rid: str,
    runs: Sequence[Mapping[str, Any]],
) -> None:
    """Refuse a grid record that is not exactly its frozen registered map.

    The recorded descriptor must be byte-for-byte (canonical JSON) the record
    `restriction.describe_restriction` rebuilds from the frozen tables (no
    environment is built); its ``descriptor_sha256`` must be the digest of
    the descriptor body and equal `random_grid.FROZEN_GRID_SHA256`; and any
    ``config.restriction_sha256`` a run records must be the digest of that
    full record.

    Raises:
        ValueError: If any of these differ.
    """
    from imitation.experiments.agnostic import pilot, random_grid, restriction, tasks

    if not random_grid.is_grid_id(rid):
        raise ValueError(f"{rid}: not a registered random grid")
    try:
        rebuilt = json.loads(json.dumps(restriction.describe_restriction(rid)))
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"{rid}: frozen random grid does not rebuild: {exc}")
    body = {k: v for k, v in desc.items() if k not in _GRID_RECORD_EXTRAS}
    record_sha = pilot.config_digest(rebuilt)
    configs = [run.get("config") or {} for run in runs]
    if not (
        pilot.canonical_json(desc) == pilot.canonical_json(rebuilt)
        and tasks._digest(body)
        == desc.get("descriptor_sha256")
        == random_grid.FROZEN_GRID_SHA256.get(rid)
        and all(
            c["restriction_sha256"] == record_sha
            for c in configs
            if "restriction_sha256" in c
        )
    ):
        raise ValueError(
            f"{rid}: random grid record is not the frozen registered grid",
        )


def _random_grid_text(
    desc: Mapping[str, Any],
    rid: str,
    runs: Sequence[Mapping[str, Any]],
) -> str:
    """Random-offset grid K per axis and seed, cross-checked against the run.

    The record must be the frozen registered grid
    (`_check_random_grid_binding`). Every axis must carry the cell table its
    kind implies for the recorded K (``K`` edges and ``K + 1``
    representatives on a scalar axis, ``K`` arcs on an angle pair); binary
    flags pass through and are named as unchanged.
    """
    _check_random_grid_binding(desc, rid, runs)
    k, seed, axes = desc.get("n_bins"), desc.get("seed"), desc.get("axes")
    ok = (
        _is_count(k)
        and _is_count(seed)
        and k >= 1
        and rid.endswith(f"_random_grid{k}_seed{seed}")
        and isinstance(axes, (list, tuple))
        and axes
        and all(isinstance(a, Mapping) and a.get("kind") in _GRID_KINDS for a in axes)
    )
    for axis in axes if ok else ():
        kind = axis["kind"]
        if kind in ("linear", "unbounded"):
            ok &= len(axis.get("edges") or ()) == k
            ok &= len(axis.get("centers") or ()) == k + 1
        elif kind == "circular":
            ok &= len(axis.get("edge_cos") or ()) == k
            ok &= len(axis.get("center_cos") or ()) == k
    if not ok:
        raise ValueError(f"{rid}: random grid descriptor does not match the run")
    kinds = {a["kind"] for a in axes}
    cells = []
    if kinds & {"linear", "unbounded"}:
        cells.append(f"{k + 1} cells per scalar axis")
    if "circular" in kinds:
        cells.append(f"{k} arcs per angle")
    text = f"Random-offset bins: K={k} per axis ({', '.join(cells)}); seed {seed}"
    flags = [str(a.get("name")) for a in axes if a["kind"] == "binary"]
    if flags:
        named = all("contact" in n for n in flags)
        text += (
            "; leg contacts unchanged" if named else f"; {', '.join(flags)} unchanged"
        )
    return text + "."


def _zero_mask_text(
    desc: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    rid: str,
) -> str:
    indices = desc.get("zeroed_indices") or []
    described = None
    match = _ZEROED_NAMES.search(str(desc.get("description") or ""))
    if match:
        described = [n.strip() for n in match.group(1).split(",")]
    declared = {tuple(n) for n in map(_feature_names_of, runs) if n is not None}
    if len(declared) > 1:
        raise ValueError(f"{rid}: runs declare different task feature names")
    names = None
    if declared:
        (features,) = declared
        if not all(_is_count(i) and 0 <= i < len(features) for i in indices):
            raise ValueError(f"{rid}: zeroed indices {indices} exceed feature names")
        names = [features[i] for i in indices]
        if described is not None and sorted(described) != sorted(names):
            raise ValueError(
                f"{rid}: task feature names {names} at zeroed indices {indices} "
                f"disagree with the recorded description {described}",
            )
    names = names or described
    if names:
        return f"Mask: {', '.join(names)} zeroed."
    if desc.get("description"):
        return f"Mask: {desc['description']}"
    return f"Mask: zeroed indices {list(indices)}."


def _restriction_observation(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Short human description of the learner observation, from the records only.

    Reads ``config.restriction`` (identical on every run) and, for zero masks,
    the run's declared task feature names. Nothing is inferred from the
    restriction id beyond ``identity``.

    Raises:
        ValueError: If runs disagree or a descriptor contradicts the run.
    """
    rid = runs[0]["restriction_id"]
    env_name = runs[0]["env_name"]
    descs = [(run.get("config") or {}).get("restriction") for run in runs]
    if any(d != descs[0] for d in descs):
        raise ValueError(f"{rid}: runs record different restriction descriptors")
    desc = descs[0]
    if not isinstance(desc, Mapping):
        if rid == "identity":
            return {"kind": "identity", "text": "Observation: full."}
        return {"kind": "unrecorded", "text": f"Mask: {rid} (no recorded descriptor)."}
    if desc.get("restriction_id", rid) != rid or desc.get("env_name", env_name) != (
        env_name
    ):
        raise ValueError(
            f"{rid}: restriction descriptor names {desc.get('restriction_id')} on "
            f"{desc.get('env_name')}, not the run's {rid} on {env_name}",
        )
    kind = desc.get("kind")
    if kind == "random_grid":
        return {"kind": "random_grid", "text": _random_grid_text(desc, rid, runs)}
    if kind == "quantized":
        return {"kind": "quantized", "text": _quantized_text(desc)}
    if kind == "alias":
        if desc.get("partition") is not None:
            text = _random_bins_text(desc, env_name, rid)
            return {"kind": "random_bins", "text": text}
        hidden, classes = desc.get("hidden"), desc.get("reachable_classes")
        if not isinstance(hidden, str) or not _is_count(classes):
            raise ValueError(f"{rid}: alias descriptor lacks hidden or class count")
        text = f"Aliased bins: {hidden} hidden; {classes} reachable classes."
        return {"kind": "categorical_alias", "text": text}
    if kind is not None:
        # Never label an unknown map as the full observation.
        raise ValueError(f"{rid}: unknown restriction descriptor kind {kind!r}")
    if not desc.get("zeroed_indices"):
        return {"kind": "identity", "text": "Observation: full."}
    return {"kind": "zero_mask", "text": _zero_mask_text(desc, runs, rid)}


def _normalized_view_audit(
    points: Mapping[str, Sequence[Mapping[str, Any]]],
    limits: Tuple[float, float] = NORMALIZED_RETURN_YLIM,
) -> Dict[str, Any]:
    """What lies outside the fixed normalized-return view; data are unchanged."""
    lo, hi = limits
    pts = [p for series in points.values() for p in series]
    drawn = [p[k] for p in pts for k in ("low", "high", "mean") if k in p]
    return {
        "y_limits": list(limits),
        "fixed": True,
        "data_modified": False,
        "outside": {
            "means": sum(not lo <= p["mean"] <= hi for p in pts if "mean" in p),
            "band_edges": sum(
                not lo <= p[k] <= hi for p in pts for k in ("low", "high")
            ),
            "seed_values": sum(not lo <= v <= hi for p in pts for v in p["values"]),
            "min": min(drawn) if drawn else None,
            "max": max(drawn) if drawn else None,
        },
    }


def _normalized_scale_notes(
    points: Mapping[str, Sequence[Mapping[str, Any]]],
    limits: Tuple[float, float] = NORMALIZED_RETURN_YLIM,
) -> List[str]:
    """Scale note if any drawn or source value leaves [0,1]; off-scale if hidden."""
    pts = [p for series in points.values() for p in series]
    notes = []
    drawn = [p[k] for p in pts for k in ("low", "high", "mean") if k in p]
    if any(not 0 <= v <= 1 for v in drawn + [v for p in pts for v in p["values"]]):
        notes.append(RETURN_SCALE_FOOTNOTE)
    outside = _normalized_view_audit(points, limits)["outside"]
    if outside["means"] or outside["band_edges"]:
        notes.append(OFFSCALE_FOOTNOTE)
    return notes


def _value_text(values: Sequence[float]) -> str:
    return (
        f"{values[0]:.6g}"
        if len(values) == 1
        else f"[{values[0]:.6g}, {values[-1]:.6g}]"
    )


def _paired_protocol(group: Mapping[str, Any], runs) -> Dict[str, bool]:
    """Whether every run declares paired cold fits and shares episode starts.

    Starts count as paired only when, for every training seed and evaluated
    budget, every method recorded the same episode seed list (Offline BC at
    its endpoint budget only).

    Raises:
        ValueError: If runs mix fit protocols.
    """
    declared = {
        ((run.get("config") or {}).get("experiment") or {}).get("paired_cold_fits")
        is True
        for run in runs
    }
    if len(declared) != 1:
        raise ValueError("Runs mix fit protocols: paired_cold_fits on only some runs")
    if declared != {True}:
        return {"paired_cold_fits": False, "paired_starts": False}

    def seeds_of(row):
        for field in ("evaluation_episode_seeds", "episode_seeds"):
            if isinstance(row.get(field), (list, tuple)):
                return list(row[field])
        return None

    paired = True
    for seed in group["seeds"]:
        lists: Dict[int, List[Any]] = {}
        for method in RESTRICTION_METHODS:
            for row in group["evaluations"][(method, seed)]:
                lists.setdefault(row["n_observations"], []).append(seeds_of(row))
        for found in lists.values():
            if found[0] is None or not found[0] or any(s != found[0] for s in found):
                paired = False
    return {"paired_cold_fits": True, "paired_starts": paired}


def _plain_caveats(caveats: Iterable[str]) -> List[str]:
    lines = list(caveats)
    for line in lines:
        if not isinstance(line, str) or not line.strip():
            raise ValueError(f"Caveats must be non-empty strings, got {line!r}")
    return lines


def _seed_text(seeds: Sequence[int]) -> str:
    ordered = sorted(seeds)
    if len(ordered) > 2 and ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}-{ordered[-1]}"
    return ", ".join(str(s) for s in ordered)


def _validated_restriction_group(
    runs: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    """Check one complete four-method, all-seed group; return its evaluations.

    Only records with ``d_eval_size`` are evaluations; every other record is a
    training-only round and is ignored.

    Raises:
        ValueError: On any mixed, missing, duplicate, incomplete or
            out-of-contract run or evaluation.
    """
    seeds = tuple(seeds)
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError(f"Need at least two distinct training seeds, got {seeds}")
    if not runs:
        raise ValueError("No runs to plot")
    for field in ("env_name", "restriction_id"):
        found = {run.get(field) for run in runs}
        if len(found) != 1:
            raise ValueError(f"Runs mix {field} values: {sorted(map(str, found))}")

    cells: Dict[Tuple[Any, Any], Mapping[str, Any]] = {}
    for run in runs:
        key = (run.get("method"), run.get("seed"))
        if key in cells:
            raise ValueError(f"Duplicate run for method/seed {key}")
        if run.get("status") != "complete":
            raise ValueError(f"Run {key} is not complete: {run.get('status')!r}")
        cells[key] = run
    methods = {method for method, _ in cells}
    if methods != set(RESTRICTION_METHODS):
        raise ValueError(
            f"Expected methods {sorted(RESTRICTION_METHODS)}, "
            f"got {sorted(map(str, methods))}",
        )
    for method in RESTRICTION_METHODS:
        found = {seed for m, seed in cells if m == method}
        if found != set(seeds):
            raise ValueError(
                f"{method}: expected training seeds {sorted(seeds)}, "
                f"got {sorted(found, key=str)}",
            )

    evaluations: Dict[Tuple[str, Any], List[Mapping[str, Any]]] = {}
    for (method, seed), run in cells.items():
        label = f"{method} seed {seed}"
        rows = [r for r in run.get("records") or [] if r.get("d_eval_size") is not None]
        if method == "bc" and len(rows) != 1:
            raise ValueError(
                f"{label}: Offline BC must have exactly one evaluated full fit, "
                f"got {len(rows)}",
            )
        if not rows:
            raise ValueError(f"{label}: no evaluated records")
        for row in rows:
            budget = row.get("n_observations")
            where = f"{label} budget {budget}"
            if not _is_count(budget) or budget < 0:
                raise ValueError(f"{where}: invalid evaluated budget n_observations")
            for metric, _ in RESTRICTION_PANELS:
                if not _is_finite_number(row.get(metric)):
                    raise ValueError(
                        f"{where}: {metric} must be a finite number, "
                        f"got {row.get(metric)!r}",
                    )
            if row["rollout_cross_entropy"] < 0:
                raise ValueError(
                    f"{where}: rollout cross-entropy "
                    f"{row['rollout_cross_entropy']} is negative",
                )
            if not 0 <= row["disagreement_rate"] <= 1:
                raise ValueError(
                    f"{where}: disagreement rate {row['disagreement_rate']} "
                    "is outside [0, 1]",
                )
            episodes = row.get("episode_returns")
            if (
                not isinstance(episodes, (list, tuple))
                or len(episodes) != RESTRICTION_EVAL_EPISODES
            ):
                count = len(episodes) if isinstance(episodes, (list, tuple)) else None
                raise ValueError(
                    f"{where}: expected {RESTRICTION_EVAL_EPISODES} evaluation "
                    f"episodes, got {count}",
                )
            if not all(_is_finite_number(r) for r in episodes):
                raise ValueError(f"{where}: episode returns must be finite numbers")
        rows = sorted(rows, key=lambda r: r["n_observations"])
        budgets = [r["n_observations"] for r in rows]
        if len(set(budgets)) != len(budgets):
            raise ValueError(f"{label}: repeated evaluated budget in {budgets}")
        evaluations[(method, seed)] = rows

    grids: Dict[str, List[int]] = {}
    for method in RESTRICTION_METHODS:
        reference = [r["n_observations"] for r in evaluations[(method, seeds[0])]]
        for seed in seeds[1:]:
            budgets = [r["n_observations"] for r in evaluations[(method, seed)]]
            if budgets != reference:
                raise ValueError(
                    f"{method}: evaluated budget grid of seed {seed} {budgets} "
                    f"differs from seed {seeds[0]} {reference}",
                )
        grids[method] = reference
    loop_grid = grids["bc_iid"]
    for method in ("ftl", "bc_prefix"):
        if grids[method] != loop_grid:
            raise ValueError(
                f"Evaluated budget grids differ across methods: "
                f"bc_iid {loop_grid}, {method} {grids[method]}",
            )
    if grids["bc"] != [loop_grid[-1]]:
        raise ValueError(
            f"Offline BC budget {grids['bc']} differs from the final round-loop "
            f"budget {loop_grid[-1]}",
        )
    for run in runs:
        _display_normalization_reference(run)
    kinds = {
        _audit_normalization_reference(run)["normalization_reference_type"]
        for run in runs
    }
    if len(kinds) != 1:
        raise ValueError(
            f"Runs mix normalization reference types {sorted(kinds)}; one figure "
            "needs one stored reference type",
        )

    group = {
        "env_name": runs[0]["env_name"],
        "restriction_id": runs[0]["restriction_id"],
        "seeds": list(seeds),
        "budgets": loop_grid,
        "evaluations": evaluations,
    }
    group["protocol"] = _paired_protocol(group, runs)
    if _uses_stall_reference(runs):
        display = _cliff_display_returns(group, runs)
        if display is not None:
            group["display_returns"] = display
    return group


def _restriction_points(group: Dict[str, Any]) -> Dict[str, Dict[str, List[Any]]]:
    """Per metric and method, the plotted mean +/- 1 SEM over seeds per budget.

    Normalized return uses the group's ``display_returns`` when present
    (CliffWalking runs that stored the stalled reference), else the recorded
    values.
    """
    display = group.get("display_returns")
    points: Dict[str, Dict[str, List[Any]]] = {}
    for metric, _ in RESTRICTION_PANELS:
        points[metric] = {}
        for method in RESTRICTION_METHODS:
            per_seed = [group["evaluations"][(method, s)] for s in group["seeds"]]
            series = []
            for i, row in enumerate(per_seed[0]):
                if metric == "normalized_return" and display is not None:
                    values = [display[(method, s)][i] for s in group["seeds"]]
                else:
                    values = [float(rows[i][metric]) for rows in per_seed]
                mean, low, high = _compute_mean_and_sem(np.asarray(values))
                series.append(
                    {
                        "n_observations": int(row["n_observations"]),
                        "values": values,
                        "mean": mean,
                        "sem": (high - low) / 2,
                        "low": low,
                        "high": high,
                    },
                )
            points[metric][method] = series
    return points


def restriction_comparison_footnotes(
    runs: Sequence[Mapping[str, Any]],
    plotted_points: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    caveats: Iterable[str] = (),
    y_limits: Optional[Mapping[str, Sequence[float]]] = None,
    protocol: Optional[Mapping[str, bool]] = None,
) -> Dict[str, List[str]]:
    """Short visible footnotes for a restriction comparison.

    One line each: the aggregation and Offline BC setup; the learner
    observation, mask or binning from the recorded restriction descriptor; on
    CliffWalking the stalled-policy benchmark and, on the column-pair mask, its
    goal-infeasibility note. A scale line is added only when needed: when a
    return value or band leaves [0, 1], when plotted means or bands leave the
    fixed ``NORMALIZED_RETURN_YLIM`` view, when references differ across runs,
    or when stored stall-referenced values are redisplayed. When drawn
    ``y_limits`` are given, a short note is added if the cross-entropy axis
    extends below 0 or the disagreement axis outside [0, 1]. Pairing, fit
    protocol, per-run references and audits stay in the plot receipt.

    Args:
        runs: The original run records being plotted.
        plotted_points: ``{metric: {method: [point, ...]}}`` where each point
            has the per-seed ``values``, the ``mean`` and the band ``low`` and
            ``high``.
        caveats: Caller-provided plain-text lines, kept verbatim.
        y_limits: Optional ``{metric: (lower, upper)}`` of the drawn axes.
        protocol: The validated group's ``protocol``; derived from ``runs``
            when omitted.

    Returns:
        ``{"footnotes": [...], "caveats": [...]}``.
    """
    seeds = sorted({run["seed"] for run in runs})
    if protocol is None:
        _validated_restriction_group(runs, seeds)
    epochs = sorted(
        {
            ((run.get("config") or {}).get("experiment") or {}).get("bc_n_epochs")
            for run in runs
        }
        - {None},
    )
    cap = f", epoch cap {'/'.join(map(str, epochs))}" if epochs else ""
    footnotes = [
        f"Mean +/- 1 SEM; n={len(seeds)} seeds ({_seed_text(seeds)}); "
        f"Offline BC: one full-pool fit{cap}.",
        _restriction_observation(runs)["text"],
    ]
    references = [_audit_normalization_reference(run) for run in runs]
    kinds = {r["normalization_reference_type"] for r in references}
    if len(kinds) != 1:
        raise ValueError(
            f"Runs mix normalization reference types {sorted(kinds)}; one figure "
            "needs one stored reference type",
        )
    # The display lower reference is the uniform-random return in every case.
    random_returns = sorted({r["measured_uniform_random_return"] for r in references})
    expert_returns = sorted({r["expert_return"] for r in references})
    benchmark = _stall_benchmark(runs)
    if benchmark is not None:
        footnotes.append(
            STALL_BENCHMARK_FOOTNOTE.format(
                stall=benchmark["raw_return"],
                normalized=_stall_value_text(benchmark),
            ),
        )
    footnotes += _cliff_column_pairs_note(_cliff_column_pairs_audit(runs))

    scale = []
    if kinds == {REFERENCE_STALLED}:
        scale.append(
            STALLED_REDISPLAY_FOOTNOTE.format(random=_value_text(random_returns)),
        )
    if len(random_returns) > 1 or len(expert_returns) > 1:
        scale.append(REFERENCES_DIFFER_FOOTNOTE)
    scale += _normalized_scale_notes(plotted_points.get("normalized_return", {}))
    y_limits = y_limits or {}
    if "disagreement_rate" in y_limits:
        lo, hi = y_limits["disagreement_rate"]
        if lo < 0 or hi > 1:
            scale.append(DISAGREEMENT_AXIS_PADDING_FOOTNOTE)
    if "rollout_cross_entropy" in y_limits:
        if y_limits["rollout_cross_entropy"][0] < 0:
            scale.append(CE_AXIS_PADDING_FOOTNOTE)
    if scale:
        footnotes.append(" ".join(scale))
    return {"footnotes": footnotes, "caveats": _plain_caveats(caveats)}


def _wrap_lines(lines: Sequence[str], width: int, bullet: str = "") -> str:
    indent = " " * len(bullet)
    return "\n".join(
        textwrap.fill(
            line,
            width=width,
            initial_indent=bullet,
            subsequent_indent=indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for line in lines
    )


def _layout_restriction_figure(fig, axes, legend, blocks) -> None:
    """Stack title, panels, legend and text blocks top to bottom without overlap.

    Text is rewrapped until it fits the figure width, then the figure height
    is set from the measured extents of every element.
    """
    usable = (_RC_RIGHT - _RC_LEFT) * _RC_WIDTH
    for _ in range(3):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        dpi = fig.dpi
        for text, lines, bullet, width in blocks:
            for _ in range(20):
                if text.get_window_extent(renderer).width / dpi <= usable:
                    break
                width[0] = max(20, int(width[0] * 0.92))
                text.set_text(_wrap_lines(lines, width[0], bullet))
        heights = [t.get_window_extent(renderer).height / dpi for t, *_ in blocks]
        tight = [ax.get_tightbbox(renderer) for ax in axes]
        below = max(ax.bbox.y0 - t.y0 for ax, t in zip(axes, tight)) / dpi
        above = max(t.y1 - ax.bbox.y1 for ax, t in zip(axes, tight)) / dpi
        title_h = fig._suptitle.get_window_extent(renderer).height / dpi
        legend_h = legend.get_window_extent(renderer).height / dpi

        y = _RC_PAD
        text_tops = []
        for h in reversed(heights):
            y += h
            text_tops.append(y)
            y += _RC_GAP
        legend_top = y + legend_h
        axes_bottom = legend_top + _RC_GAP + below
        axes_top = axes_bottom + _RC_PANEL_HEIGHT
        height = axes_top + above + _RC_GAP + title_h + _RC_PAD

        fig.set_size_inches(_RC_WIDTH, height)
        fig.subplots_adjust(
            left=_RC_LEFT,
            right=_RC_RIGHT,
            bottom=axes_bottom / height,
            top=axes_top / height,
            wspace=0.24,
        )
        fig._suptitle.set_y(1 - _RC_PAD / height)
        legend.set_bbox_to_anchor((0.5, legend_top / height), fig.transFigure)
        for (text, *_), top in zip(blocks, reversed(text_tops)):
            text.set_y(top / height)


def plot_restriction_comparison(
    runs: Sequence[Mapping[str, Any]],
    output_path: pathlib.Path,
    caveats: Iterable[str] = (),
    seeds: Sequence[int] = RESTRICTION_SEEDS,
) -> Dict[str, Any]:
    """Plot BC, BC-iid, FTL and BC-prefix for one environment and restriction.

    Three linear-axis panels (normalized return, disagreement rate, rollout
    cross-entropy) show the across-seed mean +/- 1 SEM at each evaluated
    budget. The normalized-return panel uses the fixed view
    ``NORMALIZED_RETURN_YLIM``; values and bands are never clipped, and the
    receipt counts what lies outside the view. The other axes start at 0 and
    keep every value. Offline BC is drawn as a flat reference at its single
    full-pool fit.

    Args:
        runs: Already provenance-verified original run records, exactly one
            complete run per method and seed, all with one ``env_name`` and
            ``restriction_id``.
        output_path: Image path to write.
        caveats: Plain-text lines drawn below the legend and footnotes.
        seeds: The expected training seeds.

    Returns:
        A receipt with the plotted points, short footnotes, verbatim caveats,
        the observation descriptor, normalization references, protocol,
        evaluation, column-pair audit and y-limits with the off-view audit.

    Raises:
        ValueError: If the group or any evaluation violates the contract.
    """
    group = _validated_restriction_group(runs, seeds)
    caveat_lines = _plain_caveats(caveats)
    points = _restriction_points(group)
    benchmark = _stall_benchmark(runs)

    fig, axes = plt.subplots(1, 3, figsize=(_RC_WIDTH, 7.0))
    y_limits = {}
    for ax, (metric, ylabel) in zip(axes, RESTRICTION_PANELS):
        extremes = []
        if metric == "normalized_return" and benchmark is not None:
            ax.axhline(
                1.0,
                color=ALGO_COLORS["expert"],
                linestyle="--",
                linewidth=1.2,
                label=EXPERT_LINE_LABEL,
                zorder=3,
            )
            # Drawn above the method curves so coincident curves cannot hide it.
            ax.axhline(
                benchmark["mean"],
                color=_STALL_LINE_COLOR,
                linestyle=":",
                linewidth=2.0,
                label=STALL_BENCHMARK_LABEL.format(stall=benchmark["raw_return"]),
                zorder=3,
            )
            if not benchmark["identical"]:
                ax.axhspan(
                    benchmark["min"],
                    benchmark["max"],
                    color=_STALL_LINE_COLOR,
                    alpha=0.10,
                    lw=0,
                )
            extremes += [1.0, benchmark["min"], benchmark["max"]]
        for method in RESTRICTION_METHODS:
            series = points[metric][method]
            color = ALGO_COLORS[method]
            label = RESTRICTION_LABELS[method]
            extremes += [p[k] for p in series for k in ("low", "high", "mean")]
            if method == "bc":
                (point,) = series
                ax.axhline(
                    point["mean"],
                    color=color,
                    linestyle=ALGO_LINESTYLES["bc"],
                    linewidth=1.8,
                    label=label,
                )
                ax.axhspan(point["low"], point["high"], color=color, alpha=0.12, lw=0)
                continue
            xs = [p["n_observations"] for p in series]
            ax.plot(
                xs,
                [p["mean"] for p in series],
                color=color,
                linestyle=ALGO_LINESTYLES[method],
                linewidth=2,
                marker="o",
                markersize=3,
                label=label,
            )
            ax.fill_between(
                xs,
                [p["low"] for p in series],
                [p["high"] for p in series],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        if metric == "normalized_return":
            set_normalized_return_axis(ax)
        else:
            # Both metrics are validated nonnegative (rates also at most 1), so
            # the axis starts at 0 unless a band edge is below it.
            lo, hi = ax.get_ylim()
            if metric == "disagreement_rate":
                hi = min(hi, 1.0)
            ax.set_ylim(min(0.0, min(extremes)), max(hi, max(extremes)))
        y_limits[metric] = [float(v) for v in ax.get_ylim()]
        ax.set_xlabel("Retained expert labels")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    notes = restriction_comparison_footnotes(
        runs,
        points,
        caveat_lines,
        y_limits,
        protocol=group["protocol"],
    )

    rid = group["restriction_id"]
    observation = "Full observation" if rid == "identity" else f"Learner mask: {rid}"
    fig.suptitle(
        f"{group['env_name']}  |  {observation}  |  BC, BC-iid, FTL, BC-prefix",
        fontsize=13,
        fontweight="bold",
        va="top",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), len(RESTRICTION_METHODS)),
        frameon=False,
        fontsize=10,
    )
    blocks = []
    sections = [(notes["footnotes"], "* ")]
    if caveat_lines:
        sections.append((["Caveats:"] + caveat_lines, ""))
    for lines, bullet in sections:
        width = [200]
        text = fig.text(
            _RC_LEFT,
            0.0,
            _wrap_lines(lines, width[0], bullet),
            ha="left",
            va="top",
            fontsize=_RC_FONTSIZE,
            linespacing=1.35,
            parse_math=False,
        )
        blocks.append((text, lines, bullet, width))
    _layout_restriction_figure(fig, axes, legend, blocks)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    logger.info(f"Saved restriction comparison: {output_path}")

    references = [_audit_normalization_reference(run) for run in runs]
    override = None
    if "display_returns" in group:
        # Diagnostic runs that stored the stalled reference: the display goes
        # back to the recorded uniform-random reference; records are unchanged.
        override = {
            "env_name": group["env_name"],
            "reason": "stored stalled_policy values redisplayed on uniform-random",
            "original_stored_lower_reference": "stalled policy",
            "original_stored_lower_returns": sorted(
                {r["random_return"] for r in references},
            ),
            "measured_uniform_random_returns": sorted(
                {r["measured_uniform_random_return"] for r in references},
            ),
            "chosen_lower_reference": "uniform-random policy",
            "formula": "(R - R_random)/(R_expert - R_random)",
            "clipped": False,
            "recomputed_from": "mean of each row's recorded episode_returns",
            "stored_values_verified_against": "recorded R_stall and R_expert",
        }
    return {
        "env_name": group["env_name"],
        "restriction_id": rid,
        "seeds": group["seeds"],
        "methods": list(RESTRICTION_METHODS),
        "panels": [metric for metric, _ in RESTRICTION_PANELS],
        "budgets": group["budgets"],
        "band": "mean +/- 1 SEM across seeds (sample std, ddof=1, / sqrt(n))",
        "points": points,
        "y_limits": y_limits,
        "normalized_return_view": _normalized_view_audit(
            points["normalized_return"],
        ),
        "observation": _restriction_observation(runs),
        "evaluation": {"episodes": RESTRICTION_EVAL_EPISODES, "deterministic": True},
        "column_pairs_audit": _cliff_column_pairs_audit(runs),
        "footnotes": notes["footnotes"],
        "caveats": notes["caveats"],
        "normalization": {
            "consistent": len(
                {(r["random_return"], r["expert_return"]) for r in references},
            )
            == 1,
            "references": references,
            "display_references": [
                _display_normalization_reference(run) for run in runs
            ],
            "display_override": override,
            "stall_benchmark": benchmark,
        },
        "protocol": group["protocol"],
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot FTL vs FTRL vs BC-iid vs BC experiment results",
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
    parser.add_argument(
        "--calibration-file",
        type=str,
        default=None,
        help="Path to lr_calibration.json for displaying LR and T_sat in titles",
    )
    parser.add_argument(
        "--band",
        choices=["ci", "sem"],
        default="ci",
        help="Uncertainty band: 'ci' (IQM + 95%% bootstrap CI, default) or "
        "'sem' (mean ± 1 SEM)",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Save directly in output-dir without a family subfolder",
    )
    parser.add_argument(
        "--zoom-classical-return",
        action="store_true",
        help="Choose each classical return range from its later curve bands",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    calibration_data = None
    if args.calibration_file:
        with open(args.calibration_file) as f:
            calibration_data = json.load(f)

    paths = plot_all(
        pathlib.Path(args.results_dir),
        pathlib.Path(args.output_dir),
        envs=args.envs,
        show_expert_on_loss=args.show_expert_on_loss,
        calibration_data=calibration_data,
        band=args.band,
        flat_output=args.flat_output,
        zoom_classical_return=args.zoom_classical_return,
    )
    if paths:
        logger.info(f"Generated {len(paths)} plots in {args.output_dir}")
    else:
        logger.warning("No plots generated -- check results directory")


if __name__ == "__main__":
    main()
