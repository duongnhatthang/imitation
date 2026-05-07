"""Plot learning curves from LR sweep results.

Two subplots per environment:
  - Top: raw disagreement rate (IQM + 95% bootstrap CI)
  - Bottom: smoothed disagreement rate with T_sat markers + ranking

Usage:
    python -m imitation.experiments.ftrl.plot_lr_sweep \
        --envs CartPole-v1 --results-dir experiments/lr_sweep
"""

import argparse
import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from imitation.experiments.ftrl.run_lr_sweep import (
    analyze_sweep_results,
    detect_t_sat,
)

logger = logging.getLogger(__name__)

_TAB10 = plt.cm.tab10


def _get_lr_color(lr: float, all_lrs: List[float]) -> Any:
    """Assign a distinct color from tab10 colormap."""
    idx = sorted(all_lrs).index(lr)
    return _TAB10(idx % 10)


def _compute_iqm_and_ci(values: np.ndarray) -> Tuple[float, float, float]:
    """Compute IQM and 95% stratified bootstrap CI using rliable."""
    from rliable import library as rly
    from rliable import metrics as rly_metrics

    if len(values) < 2:
        val = float(values[0]) if len(values) == 1 else 0.0
        return val, val, val

    data = {"a": values.reshape(-1, 1)}
    aggregate_func = lambda x: np.array([rly_metrics.aggregate_iqm(x)])
    scores, cis = rly.get_interval_estimates(data, aggregate_func, reps=2000)
    return float(scores["a"][0]), float(cis["a"][0, 0]), float(cis["a"][1, 0])


def load_sweep_results(
    results_dir: pathlib.Path, env_name: str
) -> Dict[float, List[Dict[str, Any]]]:
    """Load all ftl_lr*.json for an env, grouped by LR."""
    env_dir = results_dir / env_name.replace("/", "_")
    by_lr: Dict[float, List[Dict]] = {}
    for json_file in sorted(env_dir.glob("ftl_lr*_linear_seed*.json")):
        with open(json_file) as f:
            data = json.load(f)
        lr = data["config"]["learning_rate"]
        by_lr.setdefault(lr, []).append(data)
    return by_lr


def _build_iqm_curve(
    seed_results: List[Dict[str, Any]],
    metric_key: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Build IQM curve with bootstrap CI from seed results."""
    all_obs: Dict[int, List[float]] = {}
    all_vals: Dict[int, List[float]] = {}
    for sr in seed_results:
        for p in sr["per_round"]:
            r = p["round"]
            val = p.get(metric_key)
            if val is None:
                continue
            all_obs.setdefault(r, []).append(p["n_observations"])
            all_vals.setdefault(r, []).append(val)

    rounds = sorted(all_obs.keys())
    x_vals, iqm_vals, ci_lo, ci_hi = [], [], [], []
    for r in rounds:
        x_vals.append(float(np.mean(all_obs[r])))
        arr = np.array(all_vals[r])
        iqm, lo, hi = _compute_iqm_and_ci(arr)
        iqm_vals.append(iqm)
        ci_lo.append(lo)
        ci_hi.append(hi)

    return x_vals, iqm_vals, ci_lo, ci_hi


def _smooth(values: List[float], window: int = 10) -> List[float]:
    """Simple moving-average smoother (same length, edge-padded)."""
    arr = np.array(values)
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return list(np.convolve(padded, kernel, mode="valid"))


def plot_env_sweep(
    env_name: str,
    by_lr: Dict[float, List[Dict[str, Any]]],
    output_dir: pathlib.Path,
) -> None:
    """Plot learning curves for one environment."""
    all_lrs = sorted(by_lr.keys())
    if not all_lrs:
        logger.warning(f"No data for {env_name}")
        return

    # Get ranking from analyze_sweep_results
    all_results = []
    for lr_results in by_lr.values():
        all_results.extend(lr_results)
    calibration = analyze_sweep_results(all_results, list(by_lr.keys()))
    cal = calibration.get(env_name, {})
    ranked = cal.get("ranked", [])
    best_lr = cal.get("best_lr")
    near_best_threshold = cal.get("near_best_threshold", 0)

    smooth_window = 20

    # Build ranking data for color-coded display
    ranking_entries = []  # list of (lr, t_sat_str, is_best)
    for s in ranked:
        competitive = (
            s["mean_best_disagreement"] is not None
            and s["mean_best_disagreement"] <= near_best_threshold
        )
        if not competitive:
            continue
        t_str = str(s["t_sat"]) if s["t_sat"] else "N/A"
        ranking_entries.append((s["lr"], t_str, s["lr"] == best_lr))

    best_d_str = f"{cal.get('best_disagreement', 0):.4f}"
    method_str = (
        f"T_sat detected on smoothed (window={smooth_window}) IQM curve across seeds.  "
        f"Ranking = smallest T_sat among LRs with best_disagree "
        f"<= 2x overall best ({best_d_str})"
    )

    fig, (ax_raw, ax_smooth) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True,
    )

    # Title line 1: env name
    fig.text(0.5, 0.99, f"{env_name} — LR Sweep (FTL+DAgger)",
             ha="center", va="top", fontsize=13, fontweight="bold")
    # Title line 2: method
    fig.text(0.5, 0.965, method_str,
             ha="center", va="top", fontsize=9, style="italic",
             color="0.4")

    # Title line 3: color-coded ranking using matplotlib text offsets
    if ranking_entries:
        from matplotlib.offsetbox import HPacker, TextArea, AnchoredOffsetbox

        text_areas = []
        for i, (lr, t_str, is_best) in enumerate(ranking_entries):
            color = _get_lr_color(lr, all_lrs)
            lr_text = f"lr={lr:.0e} T={t_str}"
            if is_best:
                lr_text = f"[{lr_text}]"
            ta = TextArea(
                lr_text,
                textprops=dict(
                    color=color, fontsize=9, fontweight="bold" if is_best else "normal",
                ),
            )
            text_areas.append(ta)
            if i < len(ranking_entries) - 1:
                sep = TextArea(" > ", textprops=dict(color="0.3", fontsize=9))
                text_areas.append(sep)

        pack = HPacker(children=text_areas, align="center", pad=0, sep=0)
        ab = AnchoredOffsetbox(
            loc="upper center", child=pack,
            bbox_to_anchor=(0.5, 0.945),
            bbox_transform=fig.transFigure,
            frameon=False, pad=0,
        )
        fig.add_artist(ab)

    # T_sat lookup from ranking data
    t_sat_by_lr = {s["lr"]: s.get("t_sat") for s in cal.get("all_lr_results", [])}

    for lr in all_lrs:
        seed_results = by_lr[lr]
        color = _get_lr_color(lr, all_lrs)
        label = f"lr={lr:.0e}"

        x_vals, iqm_vals, ci_lo, ci_hi = _build_iqm_curve(
            seed_results, "disagreement_rate",
        )

        # --- Top subplot: raw IQM ---
        ax_raw.plot(x_vals, iqm_vals, color=color, label=label,
                    linewidth=1.2, alpha=0.85)
        ax_raw.fill_between(x_vals, ci_lo, ci_hi, color=color, alpha=0.10)

        # --- Bottom subplot: smoothed IQM + T_sat ---
        smoothed = _smooth(iqm_vals, window=smooth_window)
        smoothed_lo = _smooth(ci_lo, window=smooth_window)
        smoothed_hi = _smooth(ci_hi, window=smooth_window)

        ax_smooth.plot(x_vals, smoothed, color=color, label=label,
                       linewidth=1.8)
        ax_smooth.fill_between(x_vals, smoothed_lo, smoothed_hi,
                               color=color, alpha=0.12)

        # T_sat marker from the analysis (computed on mean curve)
        t_sat_obs = t_sat_by_lr.get(lr)
        if t_sat_obs is not None:
            idx = min(range(len(x_vals)),
                      key=lambda i: abs(x_vals[i] - t_sat_obs))
            ax_smooth.axvline(x=t_sat_obs, color=color, linestyle="--",
                              alpha=0.6, linewidth=1.0)
            ax_smooth.plot(x_vals[idx], smoothed[idx], "v",
                           color=color, markersize=10, zorder=5,
                           markeredgecolor="black", markeredgewidth=0.5)
            ax_smooth.annotate(
                f"{t_sat_obs}",
                xy=(x_vals[idx], smoothed[idx]),
                xytext=(5, 8), textcoords="offset points",
                fontsize=6, color=color, fontweight="bold",
            )

    # Formatting
    ax_raw.set_ylabel("Disagreement Rate (IQM)")
    ax_raw.set_title("Raw", fontsize=10, loc="left")
    ax_raw.legend(fontsize=8, loc="upper right", ncol=2)
    ax_raw.grid(True, alpha=0.3)

    ax_smooth.set_ylabel("Disagreement Rate (smoothed IQM)")
    ax_smooth.set_title(
        f"Smoothed (window={smooth_window}) + T_sat detection",
        fontsize=10, loc="left",
    )
    ax_smooth.legend(fontsize=8, loc="upper right", ncol=2)
    ax_smooth.grid(True, alpha=0.3)
    ax_smooth.set_xlabel("Number of Expert Labels (observations)")

    fig.tight_layout(rect=[0, 0, 1, 0.91])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{env_name.replace('/', '_')}_lr_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot LR sweep learning curves",
    )
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument(
        "--results-dir", type=str, default="experiments/lr_sweep",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/plots_lr_sweep",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results_dir = pathlib.Path(args.results_dir)
    output_dir = pathlib.Path(args.output_dir)

    for env_name in args.envs:
        by_lr = load_sweep_results(results_dir, env_name)
        if not by_lr:
            logger.warning(f"No results for {env_name}")
            continue
        logger.info(
            f"{env_name}: {len(by_lr)} LR values, "
            f"{sum(len(v) for v in by_lr.values())} total runs"
        )
        plot_env_sweep(env_name, by_lr, output_dir)


if __name__ == "__main__":
    main()
