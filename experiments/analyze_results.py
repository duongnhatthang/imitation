"""Analysis CLI for Atari benchmark Sacred output.

Reads Sacred FileStorageObserver output and provides:
- dashboard: completion status for all 63 (algo, game, seed) combinations (EVAL-05)
- curves: learning curve figures per game (Plan 02 — not yet implemented)
- aggregate: aggregate metrics figure with IQM (Plan 02 — not yet implemented)

Sacred output directory structure (from run_atari_benchmark.sh):
    {output_dir}/{algo}/{game}/{seed}/{run_id}/
    Each run_id directory contains: run.json, config.json, metrics.json

Usage:
    python experiments/analyze_results.py dashboard
    python experiments/analyze_results.py dashboard --output-dir output/sacred/2026-03-20T12:00:00+00:00
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path (same pattern as atari_smoke.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.atari_helpers import ATARI_GAMES
from experiments.plot_config import ALGO_DISPLAY_NAMES, ALGO_KEYS, SEEDS

# ---------------------------------------------------------------------------
# Constants derived from benchmark configuration
# ---------------------------------------------------------------------------

GAMES = list(ATARI_GAMES.keys())
# Total expected combinations: 3 algos x 7 games x 3 seeds = 63
TOTAL_EXPECTED = len(ALGO_KEYS) * len(GAMES) * len(SEEDS)

# Status symbols for the dashboard table
STATUS_SYMBOLS = {
    "COMPLETED": "OK",
    "RUNNING": "RUN",
    "FAILED": "FAIL",
    "INTERRUPTED": "INT",
    "PENDING": "---",
    "UNKNOWN": "???",
}


# ---------------------------------------------------------------------------
# Sacred output loading
# ---------------------------------------------------------------------------


def load_sacred_run(obs_path: Path) -> Optional[Dict]:
    """Load a single Sacred run from an observer path.

    The observer path is the directory that contains Sacred run subdirectories
    (each subdirectory is named with a sequential integer, e.g. "1", "2").

    Args:
        obs_path: Path to the Sacred observer directory, e.g.:
            output/sacred/{TIMESTAMP}/{algo}/{game}/{seed}

    Returns:
        Dict with keys:
            - "status": str (COMPLETED, RUNNING, FAILED, INTERRUPTED, etc.)
            - "config": dict (from config.json)
            - "rounds": list[int] (step indices from metrics.json)
            - "normalized_scores": list[float] (values from metrics.json)
        Returns None if:
            - obs_path does not exist or contains no run.json
            - The run status is FAILED or INTERRUPTED (skip failed runs)
    """
    if not obs_path.exists():
        return None

    # Find the latest run subdirectory by integer run ID.
    # Use sorted with int() key to handle "1", "2", "10" correctly.
    run_jsons = sorted(
        obs_path.glob("*/run.json"),
        key=lambda p: int(p.parent.name),
    )
    if not run_jsons:
        return None

    run_dir = run_jsons[-1].parent

    # Load run status from run.json
    try:
        with open(run_dir / "run.json") as f:
            run_info = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    status = run_info.get("status", "UNKNOWN")

    # Only include COMPLETED or RUNNING runs (EVAL-04 incremental).
    # FAILED/INTERRUPTED runs are skipped (return None).
    if status not in ("COMPLETED", "RUNNING"):
        return None

    # Load config.json (graceful if missing)
    config = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            config = {}

    # Load metrics.json (graceful if missing or empty)
    metrics = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except (OSError, json.JSONDecodeError):
            metrics = {}

    # Extract normalized_score time series.
    # BC logs step=0 only; DAgger/FTRL log steps 0..N-1.
    ns = metrics.get("normalized_score", {})
    rounds = ns.get("steps", [])
    normalized_scores = ns.get("values", [])

    return {
        "status": status,
        "config": config,
        "rounds": rounds,
        "normalized_scores": normalized_scores,
    }


# ---------------------------------------------------------------------------
# Collect all results
# ---------------------------------------------------------------------------


def collect_results(output_dir: Path) -> Dict[Tuple, Dict]:
    """Walk the output directory and load all Sacred runs.

    Walks {output_dir}/{algo}/{game}/{seed}/ for all known algorithm keys,
    games, and seeds. Missing directories are silently skipped.

    Args:
        output_dir: The timestamped Sacred output root, e.g.:
            output/sacred/2026-03-20T12:00:00+00:00/

    Returns:
        Dict keyed by (algo_key, game, seed) -> run data dict.
        Only contains entries for runs that loaded successfully
        (COMPLETED or RUNNING status). Missing/failed runs are absent.
    """
    results = {}

    for algo in ALGO_KEYS:
        for game in GAMES:
            for seed in SEEDS:
                obs_path = output_dir / algo / game / str(seed)
                run_data = load_sacred_run(obs_path)
                if run_data is not None:
                    results[(algo, game, seed)] = run_data

    return results


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def print_dashboard(results: Dict, output_dir: Path) -> None:
    """Print and save a completion dashboard for all 63 expected runs.

    Cross-references all expected (algo, game, seed) combinations against
    the results dict. Counts COMPLETED, RUNNING, FAILED/INTERRUPTED, and
    PENDING (not found) runs. Prints a summary line and per-game table.

    Also saves the dashboard text to figures/dashboard.txt.

    Args:
        results: Dict from collect_results() — keys are (algo_key, game, seed).
        output_dir: The output directory (used for context in the header).
    """
    # We need to check the raw filesystem for FAILED/INTERRUPTED counts
    # (collect_results only returns COMPLETED/RUNNING; we need to walk again
    # for accurate FAILED counts).
    done = 0
    running = 0
    failed = 0
    pending = 0

    # Build a full status map including FAILED/INTERRUPTED
    full_status: Dict[Tuple, str] = {}
    for algo in ALGO_KEYS:
        for game in GAMES:
            for seed in SEEDS:
                key = (algo, game, seed)
                if key in results:
                    status = results[key]["status"]
                    full_status[key] = status
                    if status == "COMPLETED":
                        done += 1
                    elif status == "RUNNING":
                        running += 1
                    else:
                        pending += 1
                else:
                    # Check if the directory exists but run failed
                    obs_path = output_dir / algo / game / str(seed)
                    run_jsons = sorted(obs_path.glob("*/run.json")) if obs_path.exists() else []
                    if run_jsons:
                        try:
                            with open(run_jsons[-1]) as f:
                                status = json.load(f).get("status", "UNKNOWN")
                        except (OSError, json.JSONDecodeError):
                            status = "UNKNOWN"
                        full_status[key] = status
                        if status in ("FAILED", "INTERRUPTED"):
                            failed += 1
                        else:
                            pending += 1
                    else:
                        full_status[key] = "PENDING"
                        pending += 1

    # Summary line
    summary = (
        f"Done: {done}  Running: {running}  Failed: {failed}  "
        f"Pending: {pending} / {TOTAL_EXPECTED}"
    )

    # Per-game table: rows=games, cols=algo x seed
    header_parts = ["Game".ljust(14)]
    for algo in ALGO_KEYS:
        display = ALGO_DISPLAY_NAMES[algo]
        for seed in SEEDS:
            header_parts.append(f"{display}[{seed}]".rjust(10))
    header = " ".join(header_parts)
    separator = "-" * len(header)

    rows = []
    for game in GAMES:
        row_parts = [game.ljust(14)]
        for algo in ALGO_KEYS:
            for seed in SEEDS:
                status = full_status.get((algo, game, seed), "PENDING")
                symbol = STATUS_SYMBOLS.get(status, "???")
                row_parts.append(symbol.rjust(10))
        rows.append(" ".join(row_parts))

    lines = [
        "=" * len(header),
        "Atari Benchmark Dashboard",
        f"Output: {output_dir}",
        "=" * len(header),
        summary,
        separator,
        header,
        separator,
    ] + rows + [separator]

    dashboard_text = "\n".join(lines)
    print(dashboard_text)

    # Save to figures/dashboard.txt
    figures_dir = _PROJECT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = figures_dir / "dashboard.txt"
    try:
        dashboard_path.write_text(dashboard_text + "\n")
        print(f"\nDashboard saved to: {dashboard_path}")
    except OSError as e:
        print(f"Warning: Could not save dashboard: {e}")


# ---------------------------------------------------------------------------
# Auto-detect output directory
# ---------------------------------------------------------------------------


def _auto_detect_output_dir() -> Optional[Path]:
    """Find the most recently modified Sacred output directory.

    Looks for directories matching output/sacred/*/ and returns the one
    with the most recent modification time.

    Returns:
        Path to the most recently modified Sacred output directory, or None
        if no directories are found.
    """
    sacred_base = _PROJECT_ROOT / "output" / "sacred"
    if not sacred_base.exists():
        return None

    candidates = [p for p in sacred_base.iterdir() if p.is_dir()]
    if not candidates:
        return None

    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_dashboard(args: argparse.Namespace) -> None:
    """Run the dashboard subcommand."""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _auto_detect_output_dir()
        if output_dir is None:
            print("No Sacred output directories found under output/sacred/.")
            print("Run the benchmark first, or pass --output-dir explicitly.")
            return

    if not output_dir.exists():
        print(f"No runs found: output directory does not exist: {output_dir}")
        return

    results = collect_results(output_dir)

    if not results:
        print(f"No runs found in: {output_dir}")
        print("The directory exists but contains no COMPLETED or RUNNING runs.")
        # Still show the dashboard (all PENDING)
        print_dashboard({}, output_dir)
        return

    print_dashboard(results, output_dir)


def _cmd_curves(args: argparse.Namespace) -> None:
    """Run the curves subcommand (Plan 02 — not yet implemented)."""
    print("Not yet implemented")


def _cmd_aggregate(args: argparse.Namespace) -> None:
    """Run the aggregate subcommand (Plan 02 — not yet implemented)."""
    print("Not yet implemented")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Atari benchmark Sacred output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # dashboard subcommand
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Show completion status for all 63 (algo, game, seed) combinations.",
    )
    dashboard_parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help=(
            "Path to the Sacred output directory "
            "(e.g. output/sacred/2026-03-20T12:00:00+00:00/). "
            "Defaults to the most recently modified output/sacred/*/ directory."
        ),
    )
    dashboard_parser.set_defaults(func=_cmd_dashboard)

    # curves subcommand (Plan 02)
    curves_parser = subparsers.add_parser(
        "curves",
        help="Generate per-game learning curve figures (Plan 02).",
    )
    curves_parser.set_defaults(func=_cmd_curves)

    # aggregate subcommand (Plan 02)
    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="Generate aggregate metrics figure with IQM (Plan 02).",
    )
    aggregate_parser.set_defaults(func=_cmd_aggregate)

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)
