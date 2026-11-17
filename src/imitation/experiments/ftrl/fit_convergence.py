"""Summarize the per-fit training-loss monitor (``fit-monitor/1``) of runs.

Run it where the results live (the execution host); it reads result JSON
files and writes a small summary, never copies raw data anywhere.

    python -m imitation.experiments.ftrl.fit_convergence RESULTS_DIR [...] \
        [--json summary.json] [--plot fit_convergence.png] [--title TEXT]

It accepts ``run_experiment`` results (``per_round``) and restriction-runner
``result.json`` records (``records``), recursively. Every fit row that carries
``fit_monitor`` is grouped by (environment, method, arm, student width,
training labels N) and reported with:

* ``fits``: fits in the group (one per seed for a given N)
* ``cap_no_plateau``: fraction that used the whole epoch cap while the exact
  training NLL was still falling (``fit_cap_without_plateau``)
* ``val_improving_at_cap``: fraction whose best held-out epoch was one of the
  last two allowed epochs, among fits with a held-out split
* median final training NLL and argmax accuracy on the data actually trained
  on, median epochs run and gradient steps.

The markdown table goes to stdout. ``--plot`` draws, per environment, the
median final training NLL against N for each (method, arm, width), with the
fraction of under-trained fits underneath. It is a diagnostic, not a main
figure.
"""

import argparse
import collections
import json
import math
import pathlib
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROTOCOL = "fit-monitor/1"


def _iter_json(paths: Iterable[pathlib.Path]):
    for path in paths:
        path = pathlib.Path(path)
        files = [path] if path.is_file() else sorted(path.rglob("*.json"))
        for f in files:
            try:
                with open(f) as handle:
                    yield f, json.load(handle)
            except (OSError, ValueError):
                continue


def _identity(record: Dict[str, Any]) -> Dict[str, Any]:
    config = record.get("config") or {}
    experiment = config.get("experiment") or {}
    capacity = config.get("capacity") or {}
    return {
        "env": record.get("env_name")
        or record.get("env")
        or experiment.get("env_name")
        or config.get("env_name"),
        "method": config.get("method") or record.get("algo") or experiment.get("algo"),
        "arm": config.get("restriction_id") or "full",
        "width": capacity.get("student_width") or config.get("student_width"),
        "seed": config.get("seed", record.get("seed")),
    }


def collect(paths: Iterable[pathlib.Path]) -> List[Dict[str, Any]]:
    """One row per monitored fit."""
    rows = []
    for f, record in _iter_json(paths):
        if not isinstance(record, dict):
            continue
        fits = record.get("records") or record.get("per_round") or []
        if not isinstance(fits, list):
            continue
        ident = None
        for fit in fits:
            if not isinstance(fit, dict) or fit.get("fit_monitor") != PROTOCOL:
                continue
            ident = ident or _identity(record)
            rows.append(
                {
                    **ident,
                    "file": str(f),
                    "labels": fit.get("n_observations"),
                    "nll": fit.get("fit_train_nll_final"),
                    "accuracy": fit.get("fit_train_accuracy_final"),
                    "epochs": fit.get("fit_epochs_run"),
                    "cap": fit.get("fit_epoch_cap"),
                    "steps": fit.get("fit_gradient_steps"),
                    "cap_no_plateau": bool(fit.get("fit_cap_without_plateau")),
                    "val_improving_at_cap": fit.get("fit_val_improving_at_cap"),
                    "stop": fit.get("fit_stop_reason"),
                    "examples": fit.get("fit_train_examples"),
                }
            )
    return rows


def _median(values):
    values = [v for v in values if v is not None and not (
        isinstance(v, float) and math.isnan(v))]
    return statistics.median(values) if values else None


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        key = (row["env"], row["method"], row["arm"], row["width"], row["labels"])
        groups[key].append(row)
    out = []
    for key in sorted(groups, key=lambda k: tuple("" if v is None else str(v).zfill(8) for v in k)):
        group = groups[key]
        held_out = [r for r in group if r["val_improving_at_cap"] is not None]
        out.append(
            {
                "env": key[0],
                "method": key[1],
                "arm": key[2],
                "width": key[3],
                "labels": key[4],
                "fits": len(group),
                "cap_no_plateau": sum(r["cap_no_plateau"] for r in group) / len(group),
                "val_improving_at_cap": (
                    None
                    if not held_out
                    else sum(bool(r["val_improving_at_cap"]) for r in held_out)
                    / len(held_out)
                ),
                "median_train_nll": _median(r["nll"] for r in group),
                "median_train_accuracy": _median(r["accuracy"] for r in group),
                "median_epochs": _median(r["epochs"] for r in group),
                "epoch_cap": _median(r["cap"] for r in group),
                "median_gradient_steps": _median(r["steps"] for r in group),
            }
        )
    return out


def _fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(summary: List[Dict[str, Any]]) -> str:
    cols = [
        ("env", "env"),
        ("method", "method"),
        ("arm", "arm"),
        ("width", "width"),
        ("labels", "N"),
        ("fits", "fits"),
        ("cap_no_plateau", "cap, no plateau"),
        ("val_improving_at_cap", "val improving at cap"),
        ("median_train_nll", "train NLL"),
        ("median_train_accuracy", "train acc"),
        ("median_epochs", "epochs"),
        ("median_gradient_steps", "grad steps"),
    ]
    lines = [
        "| " + " | ".join(c[1] for c in cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for row in summary:
        lines.append("| " + " | ".join(_fmt(row[c[0]]) for c in cols) + " |")
    return "\n".join(lines)


def plot(summary: List[Dict[str, Any]], path: pathlib.Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    envs = sorted({r["env"] for r in summary if r["env"]})
    if not envs:
        raise SystemExit("no monitored fits to plot")
    fig, axes = plt.subplots(
        2,
        len(envs),
        figsize=(max(7.5, 4.2 * len(envs)), 6.4),
        squeeze=False,
        sharex="col",
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    for col, env in enumerate(envs):
        series = collections.defaultdict(list)
        for r in summary:
            if r["env"] == env and r["labels"] is not None:
                label = f"{r['method']} / {r['arm']}"
                if r["width"] is not None:
                    label += f" / width {r['width']}"
                series[label].append(r)
        top, bottom = axes[0][col], axes[1][col]
        for label, points in sorted(series.items()):
            points.sort(key=lambda r: r["labels"])
            xs = [p["labels"] for p in points]
            top.plot(xs, [p["median_train_nll"] for p in points], marker=".", label=label)
            bottom.plot(xs, [p["cap_no_plateau"] for p in points], marker=".")
        top.set_title(env)
        top.set_ylabel("median final train NLL")
        bottom.set_ylabel("fraction: epoch cap,\nno loss plateau")
        bottom.set_xlabel("training labels N")
        bottom.set_ylim(-0.05, 1.05)
        top.legend(fontsize=6)
    import textwrap

    fig.suptitle(
        textwrap.fill(title + " [diagnostic: fit monitor, median over seeds]", 90),
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path, default=None)
    parser.add_argument("--plot", type=pathlib.Path, default=None)
    parser.add_argument("--title", default="BC fit convergence")
    args = parser.parse_args(argv)
    rows = collect(args.paths)
    if not rows:
        print("no fit-monitor/1 rows found (runs predating the monitor?)")
        return 1
    summary = summarize(rows)
    total = len(rows)
    flagged = sum(r["cap_no_plateau"] for r in rows)
    print(f"{total} monitored fits; {flagged} ({flagged / total:.1%}) hit the epoch "
          "cap while the training loss was still falling\n")
    print(markdown(summary))
    if args.json:
        args.json.write_text(json.dumps({"protocol": PROTOCOL, "groups": summary}, indent=1))
    if args.plot:
        plot(summary, args.plot, args.title)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
