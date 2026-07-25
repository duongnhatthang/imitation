"""Aggregate per-run wall-clock (elapsed_seconds) into a table + bar chart."""

import argparse
import json
import pathlib
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def collect_runtimes(results_dir) -> pd.DataFrame:
    """Scan result JSONs and return one row per run with its elapsed seconds."""
    results_dir = pathlib.Path(results_dir)
    rows: List[dict] = []
    for json_file in sorted(results_dir.rglob("*.json")):
        if "scratch" in json_file.parts or "tb" in json_file.parts:
            continue
        try:
            data = json.loads(json_file.read_text())
        except (ValueError, OSError):
            continue
        if "elapsed_seconds" not in data:
            continue
        rows.append(
            {
                "env": data.get("env", json_file.parent.name),
                "algo": data.get("algo", "unknown"),
                "seed": data.get("seed", -1),
                "elapsed_seconds": float(data["elapsed_seconds"]),
            }
        )
    return pd.DataFrame(rows, columns=["env", "algo", "seed", "elapsed_seconds"])


def write_runtime_csv(df: pd.DataFrame, path) -> None:
    """Write the per-run runtime table to CSV."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["env", "algo", "seed"]).to_csv(path, index=False)


def plot_runtime_bar(df: pd.DataFrame, path) -> None:
    """Grouped bar chart of mean elapsed seconds by (env, algo)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(["env", "algo"])["elapsed_seconds"].agg(["mean", "std"])
    grouped = grouped.reset_index()
    envs = sorted(grouped["env"].unique())
    algos = sorted(grouped["algo"].unique())
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(envs) * len(algos)), 5))
    width = 0.8 / max(len(algos), 1)
    for i, algo in enumerate(algos):
        sub = grouped[grouped["algo"] == algo].set_index("env").reindex(envs)
        xs = [j + i * width for j in range(len(envs))]
        ax.bar(
            xs,
            sub["mean"].fillna(0.0).values,
            width=width,
            yerr=sub["std"].fillna(0.0).values,
            label=algo,
            capsize=3,
        )
    ax.set_xticks([j + width * (len(algos) - 1) / 2 for j in range(len(envs))])
    ax.set_xticklabels(envs, rotation=30, ha="right")
    ax.set_ylabel("elapsed_seconds (mean +/- std)")
    ax.set_title("Per-run wall-clock by env and algorithm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Optional[list] = None) -> None:
    """CLI entry point: aggregate FTRL run wall-clock stats to CSV + bar chart.

    Args:
        argv: Argument list (uses sys.argv if None).
    """
    parser = argparse.ArgumentParser(description="Aggregate FTRL run wall-clock.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    out = pathlib.Path(args.output_dir)
    df = collect_runtimes(args.results_dir)
    write_runtime_csv(df, out / "runtime.csv")
    plot_runtime_bar(df, out / "runtime.png")
    print(f"Wrote {out / 'runtime.csv'} and {out / 'runtime.png'} ({len(df)} runs)")


if __name__ == "__main__":
    main()
