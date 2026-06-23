"""Plot the L2-lambda sweep (E1): rollout-CE/disagreement AUC vs lambda, and
the correct/wrong-argmax CE breakdown for FTL (lambda=0) vs FTRL (lambda=0.01).
"""

import argparse
import glob
import json
import os
from typing import Dict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_lambda_tag(dirname: str) -> float:
    """Convert a 'lam_<tag>' directory name to its float lambda value."""
    tag = os.path.basename(dirname)
    assert tag.startswith("lam_"), f"not a lambda dir: {dirname}"
    return float(tag[len("lam_") :].replace("p", "."))


def _auc(per_round, key):
    vals = [r.get(key) for r in per_round if r.get(key) is not None]
    arr = np.array(vals, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def aggregate_lambda_auc(base_dir: str) -> Dict[str, Dict[float, Dict[str, float]]]:
    """Aggregate per-lambda, per-env AUC metrics, averaged over seeds."""
    out: Dict[str, Dict[float, Dict[str, float]]] = {}
    for lam_dir in sorted(glob.glob(os.path.join(base_dir, "lam_*"))):
        lam = parse_lambda_tag(lam_dir)
        for env_dir in sorted(glob.glob(os.path.join(lam_dir, "*"))):
            if not os.path.isdir(env_dir):
                continue
            env = os.path.basename(env_dir)
            seed_metrics = []
            for f in sorted(glob.glob(os.path.join(env_dir, "*_seed*.json"))):
                with open(f) as fh:
                    pr = json.load(fh)["per_round"]
                seed_metrics.append(
                    {
                        "rollout_ce_auc": _auc(pr, "rollout_cross_entropy"),
                        "disagreement_auc": _auc(pr, "disagreement_rate"),
                        "ce_correct_mean": _auc(pr, "rollout_ce_correct"),
                        "ce_wrong_mean": _auc(pr, "rollout_ce_wrong"),
                    }
                )
            if not seed_metrics:
                continue
            avg = {
                k: float(np.nanmean([m[k] for m in seed_metrics]))
                for k in seed_metrics[0]
            }
            out.setdefault(env, {})[lam] = avg
    return out


def _plot_auc_vs_lambda(agg, metric_key, ylabel, out_path):
    plt.figure(figsize=(9, 6))
    for env, by_lam in sorted(agg.items()):
        lams = sorted(by_lam.keys())
        ys = [by_lam[l][metric_key] for l in lams]
        # Plot lambda=0 at a small positive x for the log axis.
        xs = [l if l > 0 else 1e-5 for l in lams]
        (line,) = plt.plot(xs, ys, marker="o", label=env)
        if not np.all(np.isnan(ys)):
            best_i = int(np.nanargmin(ys))
            plt.scatter(
                [xs[best_i]],
                [ys[best_i]],
                s=160,
                facecolors="none",
                edgecolors=line.get_color(),
                linewidths=2,
                zorder=5,
            )
    plt.xscale("log")
    plt.xlabel("L2 lambda (lambda=0 shown at 1e-5)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs L2 lambda (circle = per-env argmin)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def _plot_ce_breakdown(agg, out_path, ftl_lam=0.0, ftrl_lam=0.01):
    envs = [e for e in sorted(agg) if ftl_lam in agg[e] and ftrl_lam in agg[e]]
    x = np.arange(len(envs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    for off, lam, lbl in [(-w / 2, ftl_lam, "FTL"), (w / 2, ftrl_lam, "FTRL")]:
        corr = [agg[e][lam]["ce_correct_mean"] for e in envs]
        wrong = [agg[e][lam]["ce_wrong_mean"] for e in envs]
        ax.bar(x + off, corr, w, label=f"{lbl} correct-argmax")
        ax.bar(
            x + off,
            wrong,
            w,
            bottom=np.nan_to_num(corr),
            alpha=0.5,
            label=f"{lbl} wrong-argmax",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean rollout CE (stacked: correct + wrong bucket)")
    ax.set_title("CE decomposition by argmax correctness: FTL vs FTRL")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot L2-lambda sweep (E1)")
    parser.add_argument("--base-dir", default="experiments/l2_sweep/classical")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out_dir = args.output_dir or os.path.join(args.base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    agg = aggregate_lambda_auc(args.base_dir)
    _plot_auc_vs_lambda(
        agg,
        "rollout_ce_auc",
        "rollout CE AUC",
        os.path.join(out_dir, "rollout_ce_vs_lambda.png"),
    )
    _plot_auc_vs_lambda(
        agg,
        "disagreement_auc",
        "disagreement AUC",
        os.path.join(out_dir, "disagreement_vs_lambda.png"),
    )
    _plot_ce_breakdown(agg, os.path.join(out_dir, "ce_breakdown_ftl_vs_ftrl.png"))
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
