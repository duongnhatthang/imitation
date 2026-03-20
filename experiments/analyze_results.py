"""Analysis CLI for Atari benchmark Sacred output.

Reads Sacred FileStorageObserver output and provides:
- dashboard: completion status for all 63 (algo, game, seed) combinations (EVAL-05)
- curves: learning curve figures per game (EVAL-02, EVAL-06)
- aggregate: aggregate metrics figure with IQM (EVAL-03, EVAL-06)

Sacred output directory structure (from run_atari_benchmark.sh):
    {output_dir}/{algo}/{game}/{seed}/{run_id}/
    Each run_id directory contains: run.json, config.json, metrics.json

Usage:
    python experiments/analyze_results.py dashboard
    python experiments/analyze_results.py dashboard --output-dir output/sacred/2026-03-20T12:00:00+00:00
    python experiments/analyze_results.py curves --output-dir output/sacred/2026-03-20T12:00:00+00:00
    python experiments/analyze_results.py aggregate --output-dir output/sacred/2026-03-20T12:00:00+00:00
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend — must be set before any pyplot import
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path (same pattern as atari_smoke.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.atari_helpers import ATARI_GAMES
from experiments.plot_config import (
    ALGO_DISPLAY_NAMES,
    ALGO_KEYS,
    COLOR_MAP,
    LINESTYLE_MAP,
    LINEWIDTH_MAP,
    SEEDS,
    apply_rcparams,
)

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
# Score matrix assembly for rliable
# ---------------------------------------------------------------------------


def build_score_matrix(
    results: Dict[Tuple, Dict],
    algo: str,
    games: List[str],
    seeds: List[int],
) -> np.ndarray:
    """Build a (n_seeds, n_games) score matrix for one algorithm.

    Uses the final normalized_score from each completed run.
    Missing runs are filled with NaN.

    Args:
        results: Dict from collect_results() — keys are (algo_key, game, seed).
        algo: Algorithm key (e.g. "dagger", "ftrl", "bc").
        games: List of game display names in order.
        seeds: List of seed integers in order.

    Returns:
        numpy array of shape (len(seeds), len(games)).
        NaN for missing or incomplete runs.
    """
    mat = np.full((len(seeds), len(games)), np.nan)
    for j, game in enumerate(games):
        for i, seed in enumerate(seeds):
            run = results.get((algo, game, seed))
            if run is not None and run.get("normalized_scores"):
                mat[i, j] = run["normalized_scores"][-1]  # final round score
    return mat


# ---------------------------------------------------------------------------
# Learning curves figure
# ---------------------------------------------------------------------------


def plot_learning_curves(results: Dict[Tuple, Dict], output_dir_figures: Path) -> None:
    """Generate per-game learning curve figures.

    Creates a 2x4 subplot grid (7 games + legend panel). For each game:
    - BC is plotted as a horizontal line (single step=0 entry).
    - DAgger/FTRL are plotted as mean ± std across seeds over rounds.
    - Expert reference line at y=1.0.

    Saves:
    - {output_dir_figures}/learning_curves.pdf
    - {output_dir_figures}/learning_curves.png
    - {output_dir_figures}/learning_curves/{game}.pdf (per-game)

    Handles partial data: missing seeds/games are silently skipped.
    Annotates figure with completion count (e.g. "42/63 runs complete").

    Args:
        results: Dict from collect_results() — keys are (algo_key, game, seed).
        output_dir_figures: Directory to save figures (will be created if needed).
    """
    apply_rcparams()

    if not results:
        print("Warning: No results available for learning curves. Skipping figure generation.")
        return

    # Create output directories
    os.makedirs(output_dir_figures, exist_ok=True)
    per_game_dir = output_dir_figures / "learning_curves"
    os.makedirs(per_game_dir, exist_ok=True)

    # Count completed runs for annotation
    completed_count = sum(1 for v in results.values() if v["status"] == "COMPLETED")
    suptitle = f"Learning Curves ({completed_count}/{TOTAL_EXPECTED} runs complete)"

    # Create 2x4 subplot grid (7 games + 1 legend panel)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(suptitle, fontsize=13)

    # Determine x-axis range for BC horizontal lines
    # Use the maximum round count across all DAgger/FTRL runs
    max_round = 0
    for (algo, game, seed), run in results.items():
        if algo in ("dagger", "ftrl") and run.get("rounds"):
            max_round = max(max_round, max(run["rounds"]))
    if max_round == 0:
        max_round = 19  # fallback default

    for idx, game in enumerate(GAMES):
        row, col = divmod(idx, 4)
        ax = axes[row, col]

        legend_handles = []
        legend_labels = []

        for algo in ALGO_KEYS:
            display_name = ALGO_DISPLAY_NAMES[algo]
            color = COLOR_MAP[display_name]
            linestyle = LINESTYLE_MAP[display_name]
            linewidth = LINEWIDTH_MAP[display_name]

            # Collect seeds for this algo/game combo
            algo_rounds = []
            algo_scores = []
            algo_configs = []
            for seed in SEEDS:
                run = results.get((algo, game, seed))
                if run is not None and run.get("normalized_scores"):
                    algo_rounds.append(run["rounds"])
                    algo_scores.append(run["normalized_scores"])
                    algo_configs.append(run.get("config", {}))

            if not algo_scores:
                # No data for this algo/game — skip
                continue

            if algo == "bc":
                # BC has a single step=0 entry — plot as horizontal line
                bc_score = np.mean([s[0] for s in algo_scores if s])
                handle = ax.axhline(
                    y=bc_score,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=display_name,
                )
                legend_handles.append(handle)
                legend_labels.append(display_name)
            else:
                # DAgger/FTRL — interpolate onto common round grid, compute mean/std
                # Determine total_timesteps and n_rounds from config for x-axis conversion
                # Use first available config; fall back to round indices if missing
                cfg = algo_configs[0] if algo_configs else {}
                total_timesteps = cfg.get("total_timesteps", None)
                n_rounds = cfg.get("n_rounds", None)

                max_r = max(len(r) for r in algo_rounds) if algo_rounds else 1
                common_rounds = np.arange(max_r)

                interpolated = []
                for rounds, scores in zip(algo_rounds, algo_scores):
                    if len(rounds) >= 2:
                        interp_scores = np.interp(common_rounds, rounds, scores)
                    elif len(rounds) == 1:
                        # Only one data point — use constant interpolation
                        interp_scores = np.full(max_r, scores[0])
                    else:
                        continue
                    interpolated.append(interp_scores)

                if not interpolated:
                    continue

                arr = np.array(interpolated)  # (n_seeds, max_rounds)
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)

                # Convert rounds to environment interactions
                if total_timesteps is not None and n_rounds is not None and n_rounds > 0:
                    timesteps = common_rounds * (total_timesteps / n_rounds)
                    xlabel_val = "Environment Interactions"
                else:
                    timesteps = common_rounds
                    xlabel_val = "Round"

                line, = ax.plot(
                    timesteps,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=display_name,
                )
                ax.fill_between(
                    timesteps,
                    mean - std,
                    mean + std,
                    alpha=0.2,
                    color=color,
                )
                legend_handles.append(line)
                legend_labels.append(display_name)

        # Expert reference line at y=1.0
        expert_handle = ax.axhline(
            y=1.0,
            color=COLOR_MAP["Expert"],
            linestyle=LINESTYLE_MAP["Expert"],
            linewidth=LINEWIDTH_MAP["Expert"],
            label="Expert",
        )
        legend_handles.append(expert_handle)
        legend_labels.append("Expert")

        ax.set_title(game)
        ax.set_xlabel("Environment Interactions")
        ax.set_ylabel("Normalized Score")

        # Save individual per-game figure
        _save_single_game_curve(results, game, per_game_dir, max_round)

    # Use the last subplot (axes[1,3]) for the shared legend
    legend_ax = axes[1, 3]
    legend_ax.axis("off")
    if legend_handles:
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="center",
            frameon=True,
            fontsize=11,
        )

    fig.tight_layout()

    curves_pdf = output_dir_figures / "learning_curves.pdf"
    curves_png = output_dir_figures / "learning_curves.png"
    fig.savefig(str(curves_pdf), bbox_inches="tight", dpi=300)
    fig.savefig(str(curves_png), bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Learning curves saved to: {curves_pdf}")
    print(f"                          {curves_png}")


def _save_single_game_curve(
    results: Dict[Tuple, Dict],
    game: str,
    per_game_dir: Path,
    max_round: int,
) -> None:
    """Save a single per-game learning curve figure.

    Args:
        results: Full results dict.
        game: Game display name.
        per_game_dir: Directory to save the figure.
        max_round: Maximum round number (for BC horizontal line extent).
    """
    apply_rcparams()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for algo in ALGO_KEYS:
        display_name = ALGO_DISPLAY_NAMES[algo]
        color = COLOR_MAP[display_name]
        linestyle = LINESTYLE_MAP[display_name]
        linewidth = LINEWIDTH_MAP[display_name]

        algo_rounds = []
        algo_scores = []
        algo_configs = []
        for seed in SEEDS:
            run = results.get((algo, game, seed))
            if run is not None and run.get("normalized_scores"):
                algo_rounds.append(run["rounds"])
                algo_scores.append(run["normalized_scores"])
                algo_configs.append(run.get("config", {}))

        if not algo_scores:
            continue

        if algo == "bc":
            bc_score = np.mean([s[0] for s in algo_scores if s])
            ax.axhline(
                y=bc_score,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=display_name,
            )
        else:
            cfg = algo_configs[0] if algo_configs else {}
            total_timesteps = cfg.get("total_timesteps", None)
            n_rounds = cfg.get("n_rounds", None)

            max_r = max(len(r) for r in algo_rounds) if algo_rounds else 1
            common_rounds = np.arange(max_r)

            interpolated = []
            for rounds, scores in zip(algo_rounds, algo_scores):
                if len(rounds) >= 2:
                    interp_scores = np.interp(common_rounds, rounds, scores)
                elif len(rounds) == 1:
                    interp_scores = np.full(max_r, scores[0])
                else:
                    continue
                interpolated.append(interp_scores)

            if not interpolated:
                continue

            arr = np.array(interpolated)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)

            if total_timesteps is not None and n_rounds is not None and n_rounds > 0:
                timesteps = common_rounds * (total_timesteps / n_rounds)
            else:
                timesteps = common_rounds

            ax.plot(timesteps, mean, color=color, linestyle=linestyle, linewidth=linewidth, label=display_name)
            ax.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color=color)

    ax.axhline(
        y=1.0,
        color=COLOR_MAP["Expert"],
        linestyle=LINESTYLE_MAP["Expert"],
        linewidth=LINEWIDTH_MAP["Expert"],
        label="Expert",
    )

    ax.set_title(game)
    ax.set_xlabel("Environment Interactions")
    ax.set_ylabel("Normalized Score")
    ax.legend()
    fig.tight_layout()

    out_path = per_game_dir / f"{game}.pdf"
    fig.savefig(str(out_path), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregate figure
# ---------------------------------------------------------------------------


def plot_aggregate(results: Dict[Tuple, Dict], output_dir_figures: Path) -> None:
    """Generate aggregate metrics figure (mean and IQM with 95% CI via rliable).

    Uses rliable's stratified bootstrap to compute 95% CI for mean and IQM
    across all 7 games and 3 seeds per algorithm.

    Saves:
    - {output_dir_figures}/aggregate.pdf
    - {output_dir_figures}/aggregate.png

    Handles partial data: missing runs filled with 0.0 (documented in title).
    Skips figure generation if zero runs are available.

    Args:
        results: Dict from collect_results() — keys are (algo_key, game, seed).
        output_dir_figures: Directory to save figures (will be created if needed).
    """
    # Guard rliable import with clear error message
    try:
        from rliable import library as rly
        from rliable import metrics as rly_metrics
        from rliable import plot_utils
    except ImportError:
        print("Install rliable: pip install rliable==1.2.0")
        sys.exit(1)

    apply_rcparams()

    if not results:
        print("Warning: No results available for aggregate figure. Skipping figure generation.")
        return

    os.makedirs(output_dir_figures, exist_ok=True)

    # Count completed runs
    completed_count = sum(1 for v in results.values() if v["status"] == "COMPLETED")

    # Build score matrices: (n_seeds, n_games) per algorithm
    scores_dict_raw = {}
    missing_per_algo = {}
    for algo in ALGO_KEYS:
        display_name = ALGO_DISPLAY_NAMES[algo]
        mat = build_score_matrix(results, algo, GAMES, SEEDS)
        assert mat.shape == (len(SEEDS), len(GAMES)), (
            f"Expected shape ({len(SEEDS)}, {len(GAMES)}), got {mat.shape}"
        )
        missing_count = int(np.isnan(mat).sum())
        missing_per_algo[display_name] = missing_count
        scores_dict_raw[display_name] = mat

    # Replace NaN with 0.0 for rliable (document how many are missing)
    scores_dict = {}
    for k, mat in scores_dict_raw.items():
        scores_dict[k] = np.nan_to_num(mat, nan=0.0)

    # Compute aggregate metrics with stratified bootstrap CI
    aggregate_func = lambda x: np.array([
        rly_metrics.aggregate_mean(x),
        rly_metrics.aggregate_iqm(x),
    ])

    # Suppress numpy warnings from bootstrap resampling
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agg_scores, agg_cis = rly.get_interval_estimates(
            scores_dict, aggregate_func, reps=50000
        )

    # Plot using rliable's plot_interval_estimates
    # colors must be a dict mapping algorithm display name to color
    colors_dict = {k: COLOR_MAP[k] for k in scores_dict.keys()}
    fig, axes = plot_utils.plot_interval_estimates(
        agg_scores,
        agg_cis,
        metric_names=["Mean", "IQM"],
        algorithms=list(scores_dict.keys()),
        xlabel="Normalized Score",
        colors=colors_dict,
    )

    # Annotate with completion count
    annotation = f"Aggregate Metrics ({completed_count}/{TOTAL_EXPECTED} runs complete)"
    fig.suptitle(annotation, fontsize=12, y=1.02)
    fig.tight_layout()

    agg_pdf = output_dir_figures / "aggregate.pdf"
    agg_png = output_dir_figures / "aggregate.png"
    fig.savefig(str(agg_pdf), bbox_inches="tight", dpi=300)
    fig.savefig(str(agg_png), bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Aggregate figure saved to: {agg_pdf}")
    print(f"                           {agg_png}")

    # Print missing run counts per algorithm
    for algo_name, missing in missing_per_algo.items():
        if missing > 0:
            print(f"  Note: {algo_name} has {missing}/{len(SEEDS) * len(GAMES)} missing runs (filled with 0.0)")


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
    """Run the curves subcommand."""
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

    figures_dir = Path(args.figures_dir)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(figures_dir / "learning_curves", exist_ok=True)

    results = collect_results(output_dir)
    plot_learning_curves(results, figures_dir)


def _cmd_aggregate(args: argparse.Namespace) -> None:
    """Run the aggregate subcommand."""
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

    figures_dir = Path(args.figures_dir)
    os.makedirs(figures_dir, exist_ok=True)

    results = collect_results(output_dir)
    plot_aggregate(results, figures_dir)


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

    # curves subcommand
    curves_parser = subparsers.add_parser(
        "curves",
        help="Generate per-game learning curve figures.",
    )
    curves_parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help=(
            "Path to the Sacred output directory. "
            "Defaults to the most recently modified output/sacred/*/ directory."
        ),
    )
    curves_parser.add_argument(
        "--figures-dir",
        metavar="DIR",
        default="figures/",
        help="Directory to save figures (default: figures/).",
    )
    curves_parser.set_defaults(func=_cmd_curves)

    # aggregate subcommand
    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="Generate aggregate metrics figure with IQM.",
    )
    aggregate_parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help=(
            "Path to the Sacred output directory. "
            "Defaults to the most recently modified output/sacred/*/ directory."
        ),
    )
    aggregate_parser.add_argument(
        "--figures-dir",
        metavar="DIR",
        default="figures/",
        help="Directory to save figures (default: figures/).",
    )
    aggregate_parser.set_defaults(func=_cmd_aggregate)

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)
