"""Combined 2-D heatmap: LR × samples-per-round on normalized return.

Produces a single figure with 3 rows (envs) × 2 columns (metrics):
  - Column 1: IQM T_sat (observations to saturation). Non-saturating cells
    are rendered with a hatched pattern and 'N/A'.
  - Column 2: IQM normalized return evaluated at T_sat (or at the budget
    cap for non-saturating cells). Shared colorbar per column.

Usage:
    python -m imitation.experiments.ftrl.plot_lr_obs_heatmap \
        --envs CartPole-v1 Blackjack-v1 FrozenLake-v1 \
        --results-dir experiments/lr_obs_sweep
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

    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) < 2:
        v = float(values[0])
        return v, v, v
    data = {"a": values.reshape(-1, 1)}
    agg = lambda x: np.array([rly_metrics.aggregate_iqm(x)])
    scores, cis = rly.get_interval_estimates(data, agg, reps=2000)
    return float(scores["a"][0]), float(cis["a"][0, 0]), float(cis["a"][1, 0])


def _t_sat_and_return_for_cell(
    seed_results: List[Dict[str, Any]],
    metric_direction: str,
    saturation_metric: str,
) -> Tuple[Optional[float], float, float, float]:
    """Compute IQM T_sat (across seeds) and IQM normalized return at T_sat.

    Returns (t_sat, return_iqm, return_lo, return_hi). ``t_sat`` is None if
    no seed reached saturation; ``return_*`` is then the IQM of the final-window
    mean across seeds.
    """
    per_seed_t_sat: List[Optional[int]] = []
    per_seed_return_at_t_sat: List[float] = []
    per_seed_final_mean: List[float] = []

    for sr in seed_results:
        rows = sr["per_round"]
        obs_sat = [
            r["n_observations"] for r in rows if r.get(saturation_metric) is not None
        ]
        vals = [
            r[saturation_metric] for r in rows if r.get(saturation_metric) is not None
        ]
        if not vals:
            continue
        window = max(5, min(20, len(vals) // 3))
        t, _ = detect_t_sat(
            vals, obs_sat,
            smooth_window=window,
            metric_direction=metric_direction,
        )
        per_seed_t_sat.append(t)

        ret_vals = [r.get("normalized_return") for r in rows]
        ret_obs = [r["n_observations"] for r in rows]
        if t is not None:
            best_idx = None
            best_diff = None
            for i, ro in enumerate(ret_obs):
                if ret_vals[i] is None:
                    continue
                diff = abs(ro - t)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_idx = i
            if best_idx is not None:
                per_seed_return_at_t_sat.append(float(ret_vals[best_idx]))

        valid_ret = [v for v in ret_vals if v is not None]
        if valid_ret:
            per_seed_final_mean.append(float(np.mean(valid_ret[-window:])))

    sat_seeds = [t for t in per_seed_t_sat if t is not None]
    if sat_seeds:
        t_sat_iqm = float(np.median(sat_seeds))
    else:
        t_sat_iqm = None

    if per_seed_return_at_t_sat:
        r_iqm, r_lo, r_hi = _compute_iqm_and_ci(np.array(per_seed_return_at_t_sat))
    elif per_seed_final_mean:
        r_iqm, r_lo, r_hi = _compute_iqm_and_ci(np.array(per_seed_final_mean))
    else:
        r_iqm, r_lo, r_hi = float("nan"), float("nan"), float("nan")

    return t_sat_iqm, r_iqm, r_lo, r_hi


def _load_env_grid(
    results_dir: pathlib.Path, env_name: str,
) -> Dict[Tuple[float, int], List[Dict[str, Any]]]:
    """Load all ftl_lr*_sp*.json for an env, grouped by (lr, samples_per_round)."""
    env_dir = results_dir / env_name.replace("/", "_")
    grid: Dict[Tuple[float, int], List[Dict]] = {}
    if not env_dir.exists():
        return grid
    for p in sorted(env_dir.glob("ftl_lr*_sp*_linear_seed*.json")):
        with open(p) as f:
            data = json.load(f)
        lr = data["config"]["learning_rate"]
        sp = data["config"]["samples_per_round"]
        grid.setdefault((lr, sp), []).append(data)
    return grid


def _draw_heatmap(
    ax,
    matrix: np.ndarray,
    na_mask: np.ndarray,
    lr_values: List[float],
    sp_values: List[int],
    title: str,
    cmap: str,
    annot_values: List[List[str]],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Draw a single heatmap with annotations and hatched N/A cells.

    Annotation color is chosen per-cell from the cell's background luminance
    (via the colormap output) so light text never lands on light cells.
    """
    display = np.where(na_mask, np.nan, matrix)
    im = ax.imshow(
        display, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax.set_xticks(range(len(sp_values)))
    ax.set_xticklabels([str(s) for s in sp_values])
    ax.set_yticks(range(len(lr_values)))
    ax.set_yticklabels([f"{lr:.0e}" for lr in lr_values])
    ax.set_xlabel("samples_per_round")
    ax.set_ylabel("learning_rate")
    ax.set_title(title, fontsize=10)

    cmap_obj = plt.get_cmap(cmap)
    lo = vmin if vmin is not None else float(np.nanmin(display))
    hi = vmax if vmax is not None else float(np.nanmax(display))
    span = hi - lo if hi > lo else 1.0

    for i in range(len(lr_values)):
        for j in range(len(sp_values)):
            if na_mask[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, hatch="///", edgecolor="0.4", linewidth=0,
                ))
                color = "0.1"
            else:
                # Choose contrasting text color from the cell's RGB luminance.
                norm = (matrix[i, j] - lo) / span
                norm = float(np.clip(norm, 0.0, 1.0))
                r, g, b, _ = cmap_obj(norm)
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                color = "black" if lum > 0.55 else "white"
            ax.text(
                j, i, annot_values[i][j],
                ha="center", va="center",
                fontsize=8, color=color,
            )
    return im


def plot_combined_heatmap(
    envs: List[str],
    results_dir: pathlib.Path,
    output_path: pathlib.Path,
    saturation_metric: str = "normalized_return",
):
    """Render the combined 3 rows × 2 cols heatmap."""
    direction = "up" if saturation_metric == "normalized_return" else "down"

    # Extra vertical headroom for the multi-line subtitle.
    fig, axes = plt.subplots(
        len(envs), 2, figsize=(12, 4 * len(envs) + 1), squeeze=False,
    )

    all_t_sat: List[float] = []
    all_return: List[float] = []
    env_data: Dict[str, Optional[Dict[str, Any]]] = {}

    for env_name in envs:
        grid = _load_env_grid(results_dir, env_name)
        if not grid:
            logger.warning(f"No results for {env_name}")
            env_data[env_name] = None
            continue

        lrs = sorted({k[0] for k in grid})
        sps = sorted({k[1] for k in grid})

        t_sat_mat = np.full((len(lrs), len(sps)), np.nan)
        ret_mat = np.full((len(lrs), len(sps)), np.nan)
        ret_lo = np.full((len(lrs), len(sps)), np.nan)
        ret_hi = np.full((len(lrs), len(sps)), np.nan)

        for (lr, sp), seeds in grid.items():
            i = lrs.index(lr)
            j = sps.index(sp)
            t, ri, rlo, rhi = _t_sat_and_return_for_cell(
                seeds, metric_direction=direction,
                saturation_metric=saturation_metric,
            )
            if t is not None:
                t_sat_mat[i, j] = t
                all_t_sat.append(t)
            ret_mat[i, j] = ri
            ret_lo[i, j] = rlo
            ret_hi[i, j] = rhi
            if not np.isnan(ri):
                all_return.append(ri)

        env_data[env_name] = dict(
            lrs=lrs, sps=sps, t_sat=t_sat_mat,
            ret=ret_mat, ret_lo=ret_lo, ret_hi=ret_hi,
        )

    t_vmin = float(np.nanmin(all_t_sat)) if all_t_sat else 0.0
    t_vmax = float(np.nanmax(all_t_sat)) if all_t_sat else 1.0
    r_vmin = float(np.nanmin(all_return)) if all_return else 0.0
    r_vmax = float(np.nanmax(all_return)) if all_return else 1.0

    for row, env_name in enumerate(envs):
        data = env_data.get(env_name)
        ax_t, ax_r = axes[row][0], axes[row][1]
        if data is None:
            ax_t.set_title(f"{env_name} — no data", fontsize=10)
            ax_r.set_title(f"{env_name} — no data", fontsize=10)
            continue

        t_na = np.isnan(data["t_sat"])
        t_ann = [
            [f"{int(data['t_sat'][i, j])}" if not t_na[i, j] else "N/A"
             for j in range(len(data["sps"]))]
            for i in range(len(data["lrs"]))
        ]
        im_t = _draw_heatmap(
            ax_t, data["t_sat"], t_na, data["lrs"], data["sps"],
            f"{env_name} — T_sat (obs)", "viridis_r", t_ann,
            vmin=t_vmin, vmax=t_vmax,
        )

        r_na = np.isnan(data["ret"])
        r_ann = [
            [f"{data['ret'][i, j]:.2f}\n[{data['ret_lo'][i, j]:.2f},{data['ret_hi'][i, j]:.2f}]"
             if not r_na[i, j] else "N/A"
             for j in range(len(data["sps"]))]
            for i in range(len(data["lrs"]))
        ]
        im_r = _draw_heatmap(
            ax_r, data["ret"], r_na, data["lrs"], data["sps"],
            f"{env_name} — normalized return at T_sat", "viridis", r_ann,
            vmin=r_vmin, vmax=r_vmax,
        )

        fig.colorbar(im_t, ax=ax_t, fraction=0.046, pad=0.04)
        fig.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"LR × samples_per_round sensitivity "
        f"(saturation on {saturation_metric})",
        fontsize=13, fontweight="bold", y=0.995,
    )
    subtitle = (
        "T_sat (left): median observations-to-saturation across seeds. "
        "N/A = no contiguous flat region in the cell's round budget "
        "(typical when samples_per_round is large → few rounds → noisier curve).\n"
        "Return at T_sat (right): IQM normalized return across seeds, "
        "evaluated at T_sat (or at the budget cap if N/A). "
        "Cell text = IQM\\n[lo, hi] where [lo, hi] is the 95% stratified-bootstrap CI."
    )
    fig.text(
        0.5, 0.965, subtitle,
        ha="center", va="top", fontsize=10, style="italic", color="0.25",
    )
    headroom = 0.93 - 0.01 * len(envs)
    fig.tight_layout(rect=[0, 0, 1, max(0.85, headroom)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2-D LR × samples_per_round heatmap",
    )
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--results-dir", type=str, default="experiments/lr_obs_sweep")
    parser.add_argument(
        "--output-path",
        type=str,
        default="experiments/plots_lr_obs_heatmap/lr_obs_heatmap.png",
    )
    parser.add_argument(
        "--saturation-metric",
        choices=["normalized_return", "disagreement_rate"],
        default="normalized_return",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    plot_combined_heatmap(
        envs=args.envs,
        results_dir=pathlib.Path(args.results_dir),
        output_path=pathlib.Path(args.output_path),
        saturation_metric=args.saturation_metric,
    )


if __name__ == "__main__":
    main()
