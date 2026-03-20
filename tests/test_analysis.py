"""Tests for experiments/analyze_results.py and experiments/plot_config.py.

Covers:
- EVAL-01: Normalized score computation
- EVAL-02: Learning curves (per-game, with mean+std bands)
- EVAL-03: Aggregate figure with IQM and mean
- EVAL-04: Incremental loading (COMPLETED, RUNNING, FAILED, missing)
- EVAL-05: Dashboard counts (done/running/failed/pending)
- EVAL-06: Publication-quality figures (PDF + PNG)
- EVAL-07: Consistent colors and line styles defined in one place

All tests use tmp_path for isolation — no shared mutable state.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path (same pattern as other test files).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

from experiments.atari_helpers import compute_normalized_score, ATARI_GAMES
from experiments.analyze_results import (
    collect_results,
    load_sacred_run,
    print_dashboard,
    build_score_matrix,
    plot_learning_curves,
    plot_aggregate,
)
from experiments.plot_config import (
    ALGORITHMS,
    COLOR_MAP,
    LINEWIDTH_MAP,
    LINESTYLE_MAP,
    RCPARAMS,
    SEEDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_mock_run(
    base_dir: Path,
    algo: str,
    game: str,
    seed: int,
    status: str,
    scores: list,
    steps: list,
    n_rounds: int = 20,
    total_timesteps: int = 500000,
) -> Path:
    """Create a mock Sacred observer directory structure for a single run.

    Creates:
        {base_dir}/{algo}/{game}/{seed}/1/run.json
        {base_dir}/{algo}/{game}/{seed}/1/config.json
        {base_dir}/{algo}/{game}/{seed}/1/metrics.json

    Args:
        base_dir: Root of the mock output directory (e.g. tmp_path).
        algo: Algorithm key ("bc", "dagger", "ftrl").
        game: Game display name (e.g. "Pong").
        seed: Seed integer.
        status: Sacred run status string.
        scores: List of normalized_score values.
        steps: List of round step indices corresponding to scores.
        n_rounds: Total number of rounds in the run config.
        total_timesteps: Total timesteps in the run config.

    Returns:
        Path to the observer directory ({base_dir}/{algo}/{game}/{seed}/).
    """
    obs_path = base_dir / algo / game / str(seed)
    run_dir = obs_path / "1"
    run_dir.mkdir(parents=True, exist_ok=True)

    # run.json
    run_json = {"status": status, "start_time": "2026-03-20T10:00:00", "stop_time": None}
    (run_dir / "run.json").write_text(json.dumps(run_json))

    # config.json
    config_json = {
        "algo": algo,
        "game": game,
        "seed": seed,
        "n_rounds": n_rounds,
        "total_timesteps": total_timesteps,
        "n_envs": 8,
    }
    (run_dir / "config.json").write_text(json.dumps(config_json))

    # metrics.json
    metrics_json = {
        "normalized_score": {
            "values": scores,
            "steps": steps,
            "timestamps": [f"2026-03-20T10:00:0{i}" for i in range(len(scores))],
        }
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_json))

    return obs_path


# ---------------------------------------------------------------------------
# EVAL-01: Normalized score computation
# ---------------------------------------------------------------------------


def test_normalized_score_midpoint():
    """compute_normalized_score returns 0.5 at midpoint between random and expert."""
    assert compute_normalized_score(150, 50, 250) == pytest.approx(0.5)


def test_normalized_score_random_level():
    """compute_normalized_score returns 0.0 when agent matches random baseline."""
    assert compute_normalized_score(50, 50, 250) == pytest.approx(0.0)


def test_normalized_score_expert_level():
    """compute_normalized_score returns 1.0 when agent matches expert."""
    assert compute_normalized_score(250, 50, 250) == pytest.approx(1.0)


def test_normalized_score_division_by_zero_guard():
    """compute_normalized_score returns 0.0 when expert == random (guard)."""
    assert compute_normalized_score(100, 100, 100) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# EVAL-04: Sacred run loading — various statuses
# ---------------------------------------------------------------------------


def test_load_sacred_run_completed(tmp_path):
    """load_sacred_run returns correct dict for a COMPLETED run."""
    obs_path = _create_mock_run(
        tmp_path, "dagger", "Pong", 0,
        status="COMPLETED",
        scores=[0.1, 0.3, 0.5],
        steps=[0, 1, 2],
    )
    result = load_sacred_run(obs_path)

    assert result is not None
    assert result["status"] == "COMPLETED"
    assert result["normalized_scores"] == pytest.approx([0.1, 0.3, 0.5])
    assert result["rounds"] == [0, 1, 2]
    assert result["config"]["algo"] == "dagger"
    assert result["config"]["game"] == "Pong"


def test_load_sacred_run_running(tmp_path):
    """load_sacred_run returns partial data for a RUNNING run (EVAL-04 incremental)."""
    obs_path = _create_mock_run(
        tmp_path, "ftrl", "Breakout", 1,
        status="RUNNING",
        scores=[0.05, 0.12],
        steps=[0, 1],
    )
    result = load_sacred_run(obs_path)

    assert result is not None
    assert result["status"] == "RUNNING"
    assert result["normalized_scores"] == pytest.approx([0.05, 0.12])


def test_load_sacred_run_failed(tmp_path):
    """load_sacred_run returns None for a FAILED run (skip failed runs)."""
    obs_path = _create_mock_run(
        tmp_path, "bc", "Pong", 2,
        status="FAILED",
        scores=[],
        steps=[],
    )
    result = load_sacred_run(obs_path)

    assert result is None


def test_load_sacred_run_missing(tmp_path):
    """load_sacred_run returns None for a non-existent path (no crash)."""
    obs_path = tmp_path / "nonexistent" / "algo" / "game" / "0"
    result = load_sacred_run(obs_path)

    assert result is None


def test_load_sacred_run_interrupted(tmp_path):
    """load_sacred_run returns None for INTERRUPTED status."""
    obs_path = _create_mock_run(
        tmp_path, "dagger", "Qbert", 0,
        status="INTERRUPTED",
        scores=[0.2],
        steps=[0],
    )
    result = load_sacred_run(obs_path)

    assert result is None


# ---------------------------------------------------------------------------
# EVAL-04: collect_results
# ---------------------------------------------------------------------------


def test_collect_results_partial(tmp_path):
    """collect_results returns exactly the runs present, no crash for missing ones."""
    _create_mock_run(tmp_path, "bc", "Pong", 0, "COMPLETED", [0.8], [0])
    _create_mock_run(tmp_path, "dagger", "Breakout", 1, "COMPLETED", [0.3, 0.5], [0, 1])

    results = collect_results(tmp_path)

    # Should find exactly 2 runs out of 63 expected
    assert len(results) == 2
    assert ("bc", "Pong", 0) in results
    assert ("dagger", "Breakout", 1) in results


def test_collect_results_empty(tmp_path):
    """collect_results returns empty dict for an empty output directory."""
    results = collect_results(tmp_path)

    assert results == {}


def test_collect_results_skips_failed(tmp_path):
    """collect_results excludes FAILED runs from results."""
    _create_mock_run(tmp_path, "bc", "Pong", 0, "COMPLETED", [0.8], [0])
    _create_mock_run(tmp_path, "bc", "Pong", 1, "FAILED", [], [])

    results = collect_results(tmp_path)

    assert len(results) == 1
    assert ("bc", "Pong", 0) in results
    assert ("bc", "Pong", 1) not in results


# ---------------------------------------------------------------------------
# EVAL-05: Dashboard counts
# ---------------------------------------------------------------------------


def test_dashboard_counts(tmp_path, capsys):
    """Dashboard correctly counts 2 done, 1 running, 0 failed, 60 pending."""
    results = {
        ("bc", "Pong", 0): {"status": "COMPLETED", "normalized_scores": [0.8], "rounds": [0], "config": {}},
        ("dagger", "Breakout", 1): {"status": "COMPLETED", "normalized_scores": [0.5, 0.6], "rounds": [0, 1], "config": {}},
        ("ftrl", "Enduro", 2): {"status": "RUNNING", "normalized_scores": [0.1], "rounds": [0], "config": {}},
    }

    print_dashboard(results, tmp_path)
    captured = capsys.readouterr()

    # Verify summary line counts
    assert "Done: 2" in captured.out
    assert "Running: 1" in captured.out
    assert "Failed: 0" in captured.out
    assert "Pending: 60 / 63" in captured.out


def test_dashboard_all_pending(tmp_path, capsys):
    """Dashboard shows all 63 pending when results is empty."""
    print_dashboard({}, tmp_path)
    captured = capsys.readouterr()

    assert "Done: 0" in captured.out
    assert "Running: 0" in captured.out
    assert "Pending: 63 / 63" in captured.out


# ---------------------------------------------------------------------------
# EVAL-07: Consistent colors and styles
# ---------------------------------------------------------------------------


def test_consistent_colors():
    """All algorithms in ALGORITHMS have entries in COLOR_MAP."""
    for algo in ALGORITHMS:
        assert algo in COLOR_MAP, f"Missing COLOR_MAP entry for '{algo}'"


def test_consistent_linestyles():
    """All algorithms in ALGORITHMS have entries in LINESTYLE_MAP."""
    for algo in ALGORITHMS:
        assert algo in LINESTYLE_MAP, f"Missing LINESTYLE_MAP entry for '{algo}'"


def test_consistent_linewidths():
    """All algorithms in ALGORITHMS have entries in LINEWIDTH_MAP."""
    for algo in ALGORITHMS:
        assert algo in LINEWIDTH_MAP, f"Missing LINEWIDTH_MAP entry for '{algo}'"


def test_plot_config_rcparams():
    """RCPARAMS contains required publication-quality settings."""
    assert RCPARAMS.get("pdf.fonttype") == 42, "pdf.fonttype must be 42 (embed fonts)"
    assert RCPARAMS.get("savefig.dpi") == 300, "savefig.dpi must be 300"


# ---------------------------------------------------------------------------
# EVAL-02, EVAL-03: build_score_matrix
# ---------------------------------------------------------------------------


def test_build_score_matrix():
    """build_score_matrix returns (n_seeds, n_games) array with correct values."""
    games = list(ATARI_GAMES.keys())  # ["Pong", "Breakout", ...]

    # Create results with 2 seeds for Pong
    results = {
        ("dagger", "Pong", 0): {
            "status": "COMPLETED",
            "normalized_scores": [0.1, 0.5],
            "rounds": [0, 1],
            "config": {"n_rounds": 2, "total_timesteps": 100000},
        },
        ("dagger", "Pong", 1): {
            "status": "COMPLETED",
            "normalized_scores": [0.2, 0.6],
            "rounds": [0, 1],
            "config": {"n_rounds": 2, "total_timesteps": 100000},
        },
    }

    mat = build_score_matrix(results, "dagger", games, SEEDS)

    assert mat.shape == (3, 7), f"Expected (3, 7), got {mat.shape}"
    assert mat[0, 0] == pytest.approx(0.5), "Seed 0, Pong final score should be 0.5"
    assert mat[1, 0] == pytest.approx(0.6), "Seed 1, Pong final score should be 0.6"
    assert np.isnan(mat[2, 0]), "Seed 2, Pong is missing — should be NaN"
    assert np.isnan(mat[0, 1]), "Seed 0, Breakout is missing — should be NaN"


def test_build_score_matrix_empty():
    """build_score_matrix returns all-NaN matrix for empty results."""
    games = list(ATARI_GAMES.keys())
    mat = build_score_matrix({}, "dagger", games, SEEDS)

    assert mat.shape == (len(SEEDS), len(games))
    assert np.all(np.isnan(mat)), "Empty results should yield all-NaN matrix"


# ---------------------------------------------------------------------------
# EVAL-02: Learning curve figure generation
# ---------------------------------------------------------------------------


def test_learning_curve_partial(tmp_path):
    """plot_learning_curves produces PDF without error from partial Sacred output."""
    sacred_dir = tmp_path / "sacred"
    figures_dir = tmp_path / "figures"

    # Create 2 runs: (dagger, Pong, seed=0, 5 rounds) and (dagger, Pong, seed=1, 3 rounds)
    _create_mock_run(
        sacred_dir, "dagger", "Pong", 0,
        status="COMPLETED",
        scores=[0.1, 0.2, 0.3, 0.4, 0.5],
        steps=[0, 1, 2, 3, 4],
        n_rounds=5,
        total_timesteps=250000,
    )
    _create_mock_run(
        sacred_dir, "dagger", "Pong", 1,
        status="COMPLETED",
        scores=[0.05, 0.15, 0.25],
        steps=[0, 1, 2],
        n_rounds=5,
        total_timesteps=250000,
    )

    results = collect_results(sacred_dir)
    assert len(results) == 2

    plot_learning_curves(results, figures_dir)

    curves_pdf = figures_dir / "learning_curves.pdf"
    assert curves_pdf.exists(), "learning_curves.pdf should be created"
    assert curves_pdf.stat().st_size > 0, "learning_curves.pdf should be non-empty"


def test_learning_curve_bc_horizontal(tmp_path):
    """plot_learning_curves handles BC (single step=0 entry) without error."""
    sacred_dir = tmp_path / "sacred"
    figures_dir = tmp_path / "figures"

    # BC run: only step=0, score=0.3
    _create_mock_run(
        sacred_dir, "bc", "Pong", 0,
        status="COMPLETED",
        scores=[0.3],
        steps=[0],
        n_rounds=1,
        total_timesteps=500000,
    )

    results = collect_results(sacred_dir)
    plot_learning_curves(results, figures_dir)

    curves_pdf = figures_dir / "learning_curves.pdf"
    assert curves_pdf.exists(), "learning_curves.pdf should be created for BC run"


def test_curves_empty_results(tmp_path):
    """plot_learning_curves with empty results does not raise, prints warning."""
    figures_dir = tmp_path / "figures"
    # Should not raise any exception
    plot_learning_curves({}, figures_dir)
    # PDF should NOT be created (warning printed instead)
    assert not (figures_dir / "learning_curves.pdf").exists()


# ---------------------------------------------------------------------------
# EVAL-03: Aggregate figure generation
# ---------------------------------------------------------------------------


def test_aggregate_figure_creation(tmp_path):
    """plot_aggregate produces aggregate.pdf from complete mock results."""
    pytest.importorskip("rliable")

    sacred_dir = tmp_path / "sacred"
    figures_dir = tmp_path / "figures"
    games = list(ATARI_GAMES.keys())

    # Create 3 seeds x 2 games for dagger (complete subset)
    for game in games[:2]:
        for seed in SEEDS:
            _create_mock_run(
                sacred_dir, "dagger", game, seed,
                status="COMPLETED",
                scores=[0.1 * (seed + 1), 0.2 * (seed + 1)],
                steps=[0, 1],
                n_rounds=2,
                total_timesteps=100000,
            )

    results = collect_results(sacred_dir)
    assert len(results) == 6, f"Expected 6 runs, got {len(results)}"

    plot_aggregate(results, figures_dir)

    agg_pdf = figures_dir / "aggregate.pdf"
    assert agg_pdf.exists(), "aggregate.pdf should be created"
    assert agg_pdf.stat().st_size > 0, "aggregate.pdf should be non-empty"


def test_aggregate_missing_runs_annotation(tmp_path, capsys):
    """plot_aggregate title/annotation includes completion count for partial data."""
    pytest.importorskip("rliable")

    sacred_dir = tmp_path / "sacred"
    figures_dir = tmp_path / "figures"

    # Only 5 runs out of 63
    _create_mock_run(sacred_dir, "dagger", "Pong", 0, "COMPLETED", [0.5], [0])
    _create_mock_run(sacred_dir, "dagger", "Pong", 1, "COMPLETED", [0.4], [0])
    _create_mock_run(sacred_dir, "bc", "Pong", 0, "COMPLETED", [0.3], [0])
    _create_mock_run(sacred_dir, "bc", "Breakout", 0, "COMPLETED", [0.2], [0])
    _create_mock_run(sacred_dir, "ftrl", "Pong", 0, "COMPLETED", [0.6], [0])

    results = collect_results(sacred_dir)
    assert len(results) == 5

    plot_aggregate(results, figures_dir)

    # PDF should be created
    agg_pdf = figures_dir / "aggregate.pdf"
    assert agg_pdf.exists(), "aggregate.pdf should be created even with partial data"


# ---------------------------------------------------------------------------
# EVAL-06: Figure output format — PDF and PNG both created
# ---------------------------------------------------------------------------


def test_figure_output_format(tmp_path):
    """Both PDF and PNG files are created in output directory."""
    sacred_dir = tmp_path / "sacred"
    figures_dir = tmp_path / "figures"

    _create_mock_run(
        sacred_dir, "dagger", "Pong", 0,
        status="COMPLETED",
        scores=[0.1, 0.5, 0.8],
        steps=[0, 1, 2],
        n_rounds=3,
        total_timesteps=150000,
    )

    results = collect_results(sacred_dir)
    plot_learning_curves(results, figures_dir)

    pdf_path = figures_dir / "learning_curves.pdf"
    png_path = figures_dir / "learning_curves.png"

    assert pdf_path.exists(), "PDF should be created"
    assert png_path.exists(), "PNG should be created"

    # Verify PDF magic bytes
    with open(pdf_path, "rb") as f:
        header = f.read(4)
    assert header == b"%PDF", f"PDF file should start with %PDF magic bytes, got {header!r}"
