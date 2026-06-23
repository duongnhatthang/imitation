"""E3: correlate FTL covariate-shift severity with FTRL's rollout-CE advantage."""

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _mean(per_round, key):
    vals = [r.get(key) for r in per_round if r.get(key) is not None]
    arr = np.array(vals, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def shift_severity(per_round: List[dict]) -> float:
    """Mean rollout CE divided by mean expert-state CE over evaluated rounds."""
    roll = _mean(per_round, "rollout_cross_entropy")
    exp = _mean(per_round, "expert_rollout_cross_entropy")
    return roll / exp if exp not in (0.0, float("nan")) else float("nan")


def _algo_auc_and_severity(results_dir, env, algo):
    files = sorted(
        glob.glob(os.path.join(results_dir, env, f"{algo}_linear_seed*.json"))
    )
    ce_aucs, sevs = [], []
    for f in files:
        pr = json.load(open(f))["per_round"]
        ce_aucs.append(_mean(pr, "rollout_cross_entropy"))
        sevs.append(shift_severity(pr))
    return (
        float(np.nanmean(ce_aucs)) if ce_aucs else float("nan"),
        float(np.nanmean(sevs)) if sevs else float("nan"),
    )


def analyze(results_dir: str) -> Dict[str, dict]:
    """Per-env FTL shift severity and FTRL advantage (FTL-FTRL rollout CE AUC)."""
    envs = [
        d
        for d in sorted(os.listdir(results_dir))
        if os.path.isdir(os.path.join(results_dir, d))
        and d not in ("tb", "scratch", "plots")
    ]
    out = {}
    for env in envs:
        ftl_auc, ftl_sev = _algo_auc_and_severity(results_dir, env, "ftl")
        ftrl_auc, _ = _algo_auc_and_severity(results_dir, env, "ftrl")
        if any(np.isnan([ftl_auc, ftrl_auc, ftl_sev])):
            continue
        out[env] = {
            "ftl_shift_severity": ftl_sev,
            "ftl_rollout_ce_auc": ftl_auc,
            "ftrl_rollout_ce_auc": ftrl_auc,
            "ftrl_advantage": ftl_auc - ftrl_auc,
        }
    return out


def _spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    parser = argparse.ArgumentParser(description="Shift-severity analysis (E3)")
    parser.add_argument(
        "--results-dir", default="experiments/learning_curves/classical"
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out_dir = args.output_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    res = analyze(args.results_dir)
    envs = sorted(res)
    sev = [res[e]["ftl_shift_severity"] for e in envs]
    adv = [res[e]["ftrl_advantage"] for e in envs]
    rho = _spearman(np.array(sev), np.array(adv)) if len(envs) > 1 else float("nan")

    with open(os.path.join(out_dir, "shift_severity.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "env",
                "ftl_shift_severity",
                "ftrl_advantage",
                "ftl_rollout_ce_auc",
                "ftrl_rollout_ce_auc",
            ]
        )
        for e in envs:
            r = res[e]
            w.writerow(
                [
                    e,
                    r["ftl_shift_severity"],
                    r["ftrl_advantage"],
                    r["ftl_rollout_ce_auc"],
                    r["ftrl_rollout_ce_auc"],
                ]
            )

    plt.figure(figsize=(8, 6))
    plt.scatter(sev, adv)
    for e, xs, ys in zip(envs, sev, adv):
        plt.annotate(e, (xs, ys), fontsize=8)
    plt.axhline(0, color="grey", lw=0.8)
    plt.xlabel("FTL shift severity (rollout CE / expert-state CE)")
    plt.ylabel("FTRL advantage (FTL - FTRL rollout CE AUC)")
    plt.title(f"Shift severity vs FTRL advantage (Spearman rho={rho:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shift_severity_scatter.png"), dpi=130)
    plt.close()
    print(f"Spearman rho = {rho:.3f}; wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
