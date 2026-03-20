"""Tests for experiments/analyze_results.py and experiments/plot_config.py.

Covers:
- EVAL-01: Normalized score computation
- EVAL-04: Incremental loading (COMPLETED, RUNNING, FAILED, missing)
- EVAL-05: Dashboard counts (done/running/failed/pending)
- EVAL-07: Consistent colors and line styles defined in one place

All tests use tmp_path for isolation — no shared mutable state.
"""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path (same pattern as other test files).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

from experiments.atari_helpers import compute_normalized_score
from experiments.analyze_results import (
    collect_results,
    load_sacred_run,
    print_dashboard,
)
from experiments.plot_config import (
    ALGORITHMS,
    COLOR_MAP,
    LINEWIDTH_MAP,
    LINESTYLE_MAP,
    RCPARAMS,
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
        "n_rounds": 20,
        "total_timesteps": 500000,
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
