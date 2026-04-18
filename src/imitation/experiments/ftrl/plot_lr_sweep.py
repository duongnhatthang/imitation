"""Plot learning curves from LR sweep results.

Shows disagreement rate and normalized return vs cumulative observations
for each LR value, with IQM + 95% bootstrap CI across seeds.
T_sat markers shown on the disagreement rate subplot.

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

from imitation.experiments.ftrl.run_lr_sweep import detect_t_sat

logger = logging.getLogger(__name__)


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


def _get_lr_color(lr: float, all_lrs: List[float]) -> Any:
    """Assign a color from a perceptually uniform colormap."""
    cmap = plt.cm.viridis
    idx = sorted(all_lrs).index(lr)
    return cmap(idx / max(len(all_lrs) - 1, 1))


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
    """Build IQM curve with bootstrap CI from seed results.

    Returns (x_vals, iqm_vals, ci_lo, ci_hi).
    """
    # Collect per-seed values aligned by round
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


def _compute_t_sat_for_lr(
    seed_results: List[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[float]]:
    """Compute median T_sat across seeds for one LR."""
    all_t_sat = []
    for sr in seed_results:
        per_round = sr["per_round"]
        n_obs = [p["n_observations"] for p in per_round]
        disagree = [p.get("disagreement_rate") for p in per_round]
        t_sat, _ = detect_t_sat(disagree, n_obs)
        if t_sat is not None:
            all_t_sat.append(t_sat)

    if len(all_t_sat) > len(seed_results) / 2:
        return int(np.median(all_t_sat)), None
    return None, None


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

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{env_name} — LR Sweep (FTL+DAgger)", fontsize=14, y=0.98)

    metrics = [
        ("disagreement_rate", "Disagreement Rate (IQM)", axes[0]),
        ("normalized_return", "Normalized Return (IQM)", axes[1]),
    ]

    for metric_key, metric_label, ax in metrics:
        for lr in all_lrs:
            seed_results = by_lr[lr]
            color = _get_lr_color(lr, all_lrs)
            x_vals, iqm_vals, ci_lo, ci_hi = _build_iqm_curve(
                seed_results, metric_key
            )

            label = f"lr={lr:.0e}"
            ax.plot(x_vals, iqm_vals, color=color, label=label, linewidth=1.5)
            ax.fill_between(x_vals, ci_lo, ci_hi, color=color, alpha=0.15)

            # Add T_sat marker on disagreement rate subplot only
            if metric_key == "disagreement_rate":
                t_sat_obs, _ = _compute_t_sat_for_lr(seed_results)
                if t_sat_obs is not None:
                    # Find the y-value at T_sat
                    idx = min(
                        range(len(x_vals)),
                        key=lambda i: abs(x_vals[i] - t_sat_obs),
                    )
                    ax.axvline(
                        x=t_sat_obs, color=color, linestyle=":",
                        alpha=0.5, linewidth=1,
                    )
                    ax.plot(
                        x_vals[idx], iqm_vals[idx], "v",
                        color=color, markersize=8, zorder=5,
                    )

        ax.set_ylabel(metric_label)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Number of Expert Labels (observations)")
    fig.tight_layout()

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
