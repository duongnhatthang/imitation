"""Tests for the FTRL plotting pipeline."""

import copy
import itertools
import json
import pathlib

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from imitation.experiments.ftrl.plot_results import (
    ALGO_COLORS,
    _compute_center_and_band,
    _compute_mean_and_sem,
    compute_cumulative_loss,
    compute_cumulative_regret,
    load_results,
    plot_all,
    plot_env,
    plot_restriction_comparison,
    restriction_comparison_footnotes,
)


def _write_fake_result(
    results_dir: pathlib.Path,
    algo: str,
    env: str,
    seed: int,
    n_rounds: int = 5,
    base_ce: float = 1.0,
):
    """Write a synthetic JSON result file."""
    env_dir = results_dir / env.replace("/", "_")
    env_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed + hash(algo) % 1000)
    samples_per_round = 500
    per_round = []
    for r in range(n_rounds):
        ce = base_ce * np.exp(-0.1 * r) + rng.normal(0, 0.05)
        ce = max(ce, 0.01)
        round_data = {
            "round": r + 1,
            "n_observations": (r + 1) * samples_per_round,
            "train_cross_entropy": round(ce, 6),
            "rollout_cross_entropy": round(ce, 6),
            "l2_norm": round(rng.uniform(0.1, 1.0), 6),
            "total_loss": round(ce + 0.001, 6),
            # normalized_return and disagreement_rate present only at even rounds
            "normalized_return": round(rng.uniform(0, 1), 6) if r % 2 == 0 else None,
            "disagreement_rate": round(rng.uniform(0, 0.5), 6) if r % 2 == 0 else None,
        }
        per_round.append(round_data)

    result = {
        "algo": algo,
        "env": env,
        "seed": seed,
        "policy_mode": "end_to_end",
        "config": {"n_rounds": n_rounds},
        "per_round": per_round,
        "baselines": {
            "expert_return": 500.0,
            "random_return": 22.0,
            "expert_self_ce": 0.05,
        },
        "elapsed_seconds": 5.0,
    }

    out_file = env_dir / f"{algo}_end_to_end_seed{seed}.json"
    with open(out_file, "w") as f:
        json.dump(result, f)


def _populate_results(results_dir, envs=None, algos=None, seeds=3, n_rounds=5):
    """Generate a full synthetic results directory."""
    if envs is None:
        envs = ["CartPole-v1", "FrozenLake-v1"]
    if algos is None:
        algos = ["ftl", "ftrl", "bc"]

    base_ces = {"ftl": 0.8, "ftrl": 0.9, "bc": 1.2}
    for env in envs:
        for algo in algos:
            for seed in range(seeds):
                _write_fake_result(
                    results_dir,
                    algo,
                    env,
                    seed,
                    n_rounds,
                    base_ce=base_ces.get(algo, 1.0),
                )


def test_load_results(tmp_path):
    """load_results returns correct DataFrame shape."""
    results_dir = tmp_path / "results"
    _populate_results(results_dir, seeds=2, n_rounds=4)

    df = load_results(results_dir)
    # 2 envs x 3 algos x 2 seeds x 4 rounds = 48 rows
    assert len(df) == 48
    assert set(df.columns) >= {
        "algo",
        "env",
        "seed",
        "round",
        "rollout_cross_entropy",
        "l2_norm",
        "normalized_return",
        "disagreement_rate",
    }
    assert set(df["algo"].unique()) == {"ftl", "ftrl", "bc"}
    assert set(df["env"].unique()) == {"CartPole-v1", "FrozenLake-v1"}


def test_load_results_empty(tmp_path):
    """load_results returns empty DataFrame for empty dir."""
    df = load_results(tmp_path / "nonexistent")
    assert df.empty


def test_load_results_skips_errors(tmp_path):
    """load_results skips malformed JSON and error results."""
    results_dir = tmp_path / "results"
    _populate_results(results_dir, envs=["CartPole-v1"], algos=["ftl"], seeds=1)

    # Write a malformed file
    bad_file = results_dir / "CartPole-v1" / "bad.json"
    bad_file.write_text("{invalid json")

    # Write an error result
    err_file = results_dir / "CartPole-v1" / "err.json"
    err_file.write_text(
        json.dumps({"error": "boom", "algo": "x", "env": "y", "seed": 0})
    )

    df = load_results(results_dir)
    # Should only have the 1 valid file's data
    assert len(df) == 5  # 1 algo x 1 seed x 5 rounds


def test_compute_cumulative_loss(tmp_path):
    """Cumulative loss is monotonically non-decreasing."""
    results_dir = tmp_path / "results"
    _populate_results(results_dir, envs=["CartPole-v1"], algos=["ftl"], seeds=1)

    df = load_results(results_dir)
    df = compute_cumulative_loss(df)

    assert "cum_loss" in df.columns
    group = df.sort_values("round")
    cum = group["cum_loss"].values
    assert all(cum[i] <= cum[i + 1] for i in range(len(cum) - 1))


def test_compute_cumulative_regret():
    """Cumulative excess CE subtracts expert CE at logged evaluation points."""
    df = pd.DataFrame(
        {
            "algo": ["bc_iid"] * 4,
            "env": ["CartPole-v1"] * 4,
            "seed": [0] * 4,
            "round": [0, 1, 2, 3],
            "rollout_cross_entropy": [1.0, None, 0.5, 0.2],
            "expert_rollout_cross_entropy": [0.1, None, 0.2, 0.3],
        }
    )
    result = compute_cumulative_regret(compute_cumulative_loss(df))
    np.testing.assert_allclose(result["cum_regret"], [0.9, np.nan, 1.2, 1.1])


def test_plot_env(tmp_path):
    """plot_env generates a PNG file with 4 subplots."""
    results_dir = tmp_path / "results"
    _populate_results(results_dir, envs=["CartPole-v1"], seeds=2)
    _write_fake_result(results_dir, "bc_iid", "CartPole-v1", 0)

    df = load_results(results_dir)
    df = compute_cumulative_loss(df)
    df = compute_cumulative_regret(df)

    out_path = tmp_path / "plots" / "CartPole-v1.png"
    plot_env(df, "CartPole-v1", out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # non-trivial PNG


def test_plot_all_flat_output(tmp_path):
    results_dir = tmp_path / "data"
    _populate_results(results_dir, envs=["CartPole-v1"], seeds=1)
    output = tmp_path / "plots" / "classical" / "learning_curves"
    paths = plot_all(results_dir, output, flat_output=True)
    assert paths == [output / "CartPole-v1.png"]
    assert paths[0].is_file()


def test_plot_all(tmp_path):
    """plot_all generates one PNG per env."""
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    _populate_results(results_dir)

    paths = plot_all(results_dir, plots_dir)

    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        assert p.suffix == ".png"


def test_plot_all_with_env_filter(tmp_path):
    """plot_all respects env filter."""
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    _populate_results(results_dir)

    paths = plot_all(results_dir, plots_dir, envs=["CartPole-v1"])

    assert len(paths) == 1
    assert "CartPole-v1" in paths[0].name


def test_plot_incremental(tmp_path):
    """Plotting works with partial results (1 algo, 1 seed)."""
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    _populate_results(results_dir, algos=["ftl"], seeds=1)

    paths = plot_all(results_dir, plots_dir)

    assert len(paths) == 2
    for p in paths:
        assert p.exists()


def test_compute_mean_and_sem():
    """mean ± SEM: center is the mean, half-width is std(ddof=1)/sqrt(n)."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, lo, hi = _compute_mean_and_sem(values)
    expected_sem = np.std(values, ddof=1) / np.sqrt(len(values))
    assert mean == pytest.approx(3.0)
    assert lo == pytest.approx(3.0 - expected_sem)
    assert hi == pytest.approx(3.0 + expected_sem)
    # SEM band is symmetric about the mean.
    assert (hi - mean) == pytest.approx(mean - lo)


def test_compute_mean_and_sem_single_value():
    """A single seed collapses the band to the point estimate (no div-by-zero)."""
    mean, lo, hi = _compute_mean_and_sem(np.array([2.5]))
    assert mean == lo == hi == pytest.approx(2.5)


def test_compute_center_and_band_dispatch():
    """ci -> IQM+bootstrap, sem -> mean±SEM; unknown band raises."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sem_center, _, _ = _compute_center_and_band(values, "sem")
    assert sem_center == pytest.approx(3.0)  # mean center
    # ci path returns a finite center and an ordered interval.
    ci_center, ci_lo, ci_hi = _compute_center_and_band(values, "ci")
    assert ci_lo <= ci_center <= ci_hi
    with pytest.raises(ValueError, match="Unknown band"):
        _compute_center_and_band(values, "stderr")


def test_plot_all_band_sem(tmp_path):
    """plot_all with band='sem' produces PNGs (SEM path renders end-to-end)."""
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots_sem"
    _populate_results(results_dir)

    paths = plot_all(results_dir, plots_dir, band="sem")

    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 1000


# ---------------------------------------------------------------------------
# Four-method restriction comparison (linear axes, values never clipped)
# ---------------------------------------------------------------------------

SEEDS = (3100, 3101, 3102, 3103, 3104)
GRID = [0, 1, 50, 100]
PLACEHOLDER_BUDGETS = [25, 75]
METHODS = ("bc", "bc_iid", "ftl", "bc_prefix")
METRICS = ("normalized_return", "disagreement_rate", "rollout_cross_entropy")
BASELINES = {"expert_return": -80.5, "random_return": -500.0}
RETURN_NOTE = "Random=0, expert=1; values outside [0,1] are valid."


def _mentions(notes, text):
    """How many footnote lines contain ``text`` (scale notes share one line)."""
    return sum(text in note for note in notes)


def _in_range_values():
    """Per method and metric, an array (seed, evaluated budget) inside [0, 1]."""
    values = {}
    for m, method in enumerate(METHODS):
        n_points = 1 if method == "bc" else len(GRID)
        base = np.arange(len(SEEDS))[:, None] * 0.03 + np.arange(n_points) * 0.05
        values[method] = {
            "normalized_return": 0.2 + 0.1 * m + base,
            "disagreement_rate": 0.4 - 0.05 * m + base / 2,
            "rollout_cross_entropy": 0.6 + 0.1 * m + base,
        }
    return values


def _out_of_range_values():
    values = _in_range_values()
    # Seed means below 0 for FTL and BC, above 1 for BC-prefix.
    values["ftl"]["normalized_return"][:, 2] = [-0.8, -0.6, -0.4, -0.2, 0.1]
    values["bc_prefix"]["normalized_return"][:, 3] = [1.1, 1.3, 1.2, 1.05, 1.4]
    values["bc"]["normalized_return"][:, 0] = [-0.25, -0.1, 0.05, -0.3, -0.2]
    return values


def _comparison_run(method, seed_index, values, env="Acrobot-v1", rid="mask_a"):
    grid = [GRID[-1]] if method == "bc" else GRID
    records = []
    for i, budget in enumerate(grid):
        metrics = {k: float(values[method][k][seed_index, i]) for k in METRICS}
        records.append(
            {
                "round": budget,
                "n_observations": budget,
                "d_eval_size": 400 + i,
                "episode_returns": [-100.0 - i] * 100,
                "expert_rollout_cross_entropy": 0.01,
                **metrics,
            },
        )
        if method != "bc" and i < len(PLACEHOLDER_BUDGETS):
            # Non-evaluated round: carries a training loss and placeholder
            # metric values that must never reach the figure.
            records.append(
                {
                    "round": PLACEHOLDER_BUDGETS[i],
                    "n_observations": PLACEHOLDER_BUDGETS[i],
                    "d_eval_size": None,
                    "normalized_return": 0.0,
                    "rollout_cross_entropy": 7.0,
                    "disagreement_rate": None,
                    "total_loss": 0.3,
                },
            )
    records.sort(key=lambda r: r["n_observations"])
    return {
        "status": "complete",
        "env_name": env,
        "restriction_id": rid,
        "method": method,
        "seed": SEEDS[seed_index],
        "config": {
            "eval_episodes": 100,
            "reference": {"source": "pinned", "baselines": dict(BASELINES)},
        },
        "data": {"baselines": {"expert_return": 1.0, "random_return": 0.0}},
        "records": records,
    }


def _comparison_runs(values=None):
    values = _in_range_values() if values is None else values
    return [
        _comparison_run(method, s, values)
        for method in METHODS
        for s in range(len(SEEDS))
    ]


@pytest.fixture
def saved_figures(monkeypatch):
    """Capture every Figure the plotting code saves, before it is closed."""
    captured = []
    original = matplotlib.figure.Figure.savefig

    def recording_savefig(self, *args, **kwargs):
        captured.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", recording_savefig)
    return captured


def _flat(text):
    return " ".join(text.split())


def _drawn_text(fig):
    return _flat(" ".join(t.get_text() for t in fig.texts))


def _seed_mean_and_sem(column):
    column = np.asarray(column, dtype=float)
    return column.mean(), column.std(ddof=1) / np.sqrt(len(column))


def _panel_lines(ax, method):
    return [
        line
        for line in ax.get_lines()
        if line.get_color() == ALGO_COLORS[method]
        and not line.get_label().startswith("_")
    ]


def test_restriction_comparison_preserves_out_of_range_values(
    tmp_path,
    saved_figures,
):
    values = _out_of_range_values()
    out = tmp_path / "Acrobot-v1__mask_a.png"
    receipt = plot_restriction_comparison(_comparison_runs(values), out)

    assert out.is_file()
    (fig,) = saved_figures
    assert len(fig.axes) == 3
    assert receipt["panels"] == list(METRICS)
    for ax in fig.axes:
        assert ax.get_yscale() == "linear"
        assert ax.get_xscale() == "linear"

    ret_ax = fig.axes[0]
    curve_lows, curve_highs = [], []
    for method in ("bc_iid", "ftl", "bc_prefix"):
        (line,) = _panel_lines(ret_ax, method)
        assert list(line.get_xdata()) == GRID  # placeholders at 25/75 excluded
        stats = [
            _seed_mean_and_sem(values[method]["normalized_return"][:, i])
            for i in range(len(GRID))
        ]
        np.testing.assert_allclose(
            line.get_ydata(),
            [mean for mean, _ in stats],
            rtol=0,
            atol=1e-12,
        )
        curve_lows += [mean - sem for mean, sem in stats]
        curve_highs += [mean + sem for mean, sem in stats]
    assert min(_panel_lines(ret_ax, "ftl")[0].get_ydata()) < 0
    assert max(_panel_lines(ret_ax, "bc_prefix")[0].get_ydata()) > 1

    (bc_line,) = _panel_lines(ret_ax, "bc")
    bc_mean, bc_sem = _seed_mean_and_sem(values["bc"]["normalized_return"][:, 0])
    assert bc_mean < 0
    np.testing.assert_allclose(bc_line.get_ydata(), [bc_mean, bc_mean], atol=1e-12)

    # Every band is drawn in full; the fixed view does not change the data.
    band_y = np.concatenate(
        [c.get_paths()[0].vertices[:, 1] for c in ret_ax.collections],
    )
    assert band_y.min() == pytest.approx(min(curve_lows), abs=1e-12)
    assert band_y.max() == pytest.approx(max(curve_highs), abs=1e-12)
    (bc_band,) = ret_ax.patches
    bc_band_y = np.asarray(bc_band.get_xy())[:, 1]
    assert bc_band_y.min() == pytest.approx(bc_mean - bc_sem, abs=1e-12)
    assert bc_band_y.max() == pytest.approx(bc_mean + bc_sem, abs=1e-12)
    assert ret_ax.get_ylim() == (-0.5, 1.5)
    assert min(curve_lows) < -0.5  # FTL's band edge lies outside the view

    # The receipt carries the same plotted mean/SEM points and per-seed values.
    ftl_points = receipt["points"]["normalized_return"]["ftl"]
    assert [p["n_observations"] for p in ftl_points] == GRID
    mean, sem = _seed_mean_and_sem(values["ftl"]["normalized_return"][:, 2])
    assert ftl_points[2]["mean"] == pytest.approx(mean, abs=1e-12)
    assert ftl_points[2]["sem"] == pytest.approx(sem, abs=1e-12)
    assert ftl_points[2]["values"] == list(values["ftl"]["normalized_return"][:, 2])
    (bc_point,) = receipt["points"]["normalized_return"]["bc"]
    assert bc_point["n_observations"] == GRID[-1]
    assert bc_point["low"] == pytest.approx(bc_mean - bc_sem, abs=1e-12)


def test_restriction_comparison_return_footnote_only_when_out_of_range(
    tmp_path,
    saved_figures,
):
    inside = plot_restriction_comparison(_comparison_runs(), tmp_path / "in.png")
    outside = plot_restriction_comparison(
        _comparison_runs(_out_of_range_values()),
        tmp_path / "out.png",
    )
    fig_in, fig_out = saved_figures

    assert not _mentions(inside["footnotes"], RETURN_NOTE)
    assert RETURN_NOTE not in _drawn_text(fig_in)
    assert _mentions(outside["footnotes"], RETURN_NOTE) == 1
    assert RETURN_NOTE in _drawn_text(fig_out)
    # In-range CE and disagreement axes need no axis note either.
    assert not _padding_notes(inside["footnotes"])


def test_restriction_comparison_single_out_of_range_seed_triggers_footnote(
    tmp_path,
):
    values = _in_range_values()
    values["bc_iid"]["normalized_return"][4, 1] = 1.0000001
    receipt = plot_restriction_comparison(
        _comparison_runs(values),
        tmp_path / "x.png",
    )
    assert _mentions(receipt["footnotes"], RETURN_NOTE) == 1


def test_restriction_comparison_states_evaluation_protocol(tmp_path, saved_figures):
    receipt = plot_restriction_comparison(_comparison_runs(), tmp_path / "x.png")
    text = _flat(" ".join(receipt["footnotes"]))
    assert "n=5 seeds (3100-3104)" in text
    # Protocol and references are kept in the receipt, not the footer.
    assert receipt["evaluation"] == {"episodes": 100, "deterministic": True}
    assert receipt["protocol"] == {"paired_cold_fits": False, "paired_starts": False}
    refs = receipt["normalization"]["references"]
    assert {(r["random_return"], r["expert_return"]) for r in refs} == {(-500, -80.5)}
    drawn = _drawn_text(saved_figures[0])
    for note in receipt["footnotes"]:
        assert _flat(note) in drawn


def test_restriction_comparison_reports_inconsistent_normalization(tmp_path):
    runs = _comparison_runs()
    runs[7]["config"]["reference"]["baselines"]["expert_return"] = -90.0
    receipt = plot_restriction_comparison(runs, tmp_path / "x.png")
    text = _flat(" ".join(receipt["footnotes"]))
    assert "R_expert = -80.5" not in text
    assert "differ" in text
    refs = receipt["normalization"]["references"]
    assert len(refs) == 20
    assert {r["expert_return"] for r in refs} == {-80.5, -90.0}


def test_restriction_comparison_caveats_retained_and_drawn(tmp_path, saved_figures):
    caveats = (
        "Source: pinned inventories, hashes verified.",
        "Plateau qualification is separate; this is not a convergence claim.",
    )
    receipt = plot_restriction_comparison(
        _comparison_runs(),
        tmp_path / "x.png",
        caveats=caveats,
    )
    assert receipt["caveats"] == list(caveats)
    drawn = _drawn_text(saved_figures[0])
    for caveat in caveats:
        assert caveat in drawn


@pytest.mark.parametrize("cliff", [False, True])
def test_restriction_comparison_layout_has_no_overlap(tmp_path, saved_figures, cliff):
    caveats = tuple(
        f"Caveat {i}: " + "long verified provenance statement " * 12 for i in range(4)
    )
    # CliffWalking adds the expert and stall lines, so the legend wraps.
    runs = _cliff_runs() if cliff else _comparison_runs(_out_of_range_values())
    plot_restriction_comparison(runs, tmp_path / "x.png", caveats=caveats)
    (fig,) = saved_figures
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = {f"axes{i}": ax.get_tightbbox(renderer) for i, ax in enumerate(fig.axes)}
    boxes["legend"] = fig.legends[0].get_window_extent(renderer)
    boxes["title"] = fig._suptitle.get_window_extent(renderer)
    notes = [text for text in fig.texts if text is not fig._suptitle]
    for i, text in enumerate(notes):
        boxes[f"text{i}"] = text.get_window_extent(renderer)
    assert len(notes) == 2  # footnotes and caveats

    frame = fig.bbox
    for name, box in boxes.items():
        assert box.x0 >= frame.x0 and box.x1 <= frame.x1, name
        assert box.y0 >= frame.y0 and box.y1 <= frame.y1, name
    for (a, box_a), (b, box_b) in itertools.combinations(boxes.items(), 2):
        assert not box_a.overlaps(box_b), (a, b)


def _find_run(runs, method, seed):
    return next(r for r in runs if r["method"] == method and r["seed"] == seed)


def _mutate_eval(runs, method, seed, index, **changes):
    run = _find_run(runs, method, seed)
    evals = [r for r in run["records"] if r.get("d_eval_size") is not None]
    evals[index].update(changes)
    return runs


def _shift_grid(run):
    for rec in run["records"]:
        if rec.get("d_eval_size") is not None and rec["n_observations"] == 50:
            rec["n_observations"] = 60


def _shift_seed_grid(runs):
    _shift_grid(_find_run(runs, "ftl", 3102))
    return runs


def _shift_method_grid(runs):
    for run in runs:
        if run["method"] == "bc_iid":
            _shift_grid(run)
    return runs


def _second_bc_eval(runs):
    for run in runs:
        if run["method"] == "bc":
            run["records"].append(dict(run["records"][0]))
    return runs


def _replace_last(runs, **changes):
    return [*runs[:-1], {**runs[-1], **changes}]


INVALID_CASES = {
    "duplicate": (lambda runs: runs + [copy.deepcopy(runs[0])], "uplicate"),
    "missing_seed": (
        lambda runs: [r for r in runs if (r["method"], r["seed"]) != ("ftl", 3104)],
        "seed",
    ),
    "missing_method": (
        lambda runs: [r for r in runs if r["method"] != "bc_prefix"],
        "method",
    ),
    "mixed_env": (lambda runs: _replace_last(runs, env_name="X"), "env"),
    "mixed_restriction": (
        lambda runs: _replace_last(runs, restriction_id="identity"),
        "restriction",
    ),
    "incomplete": (lambda runs: _replace_last(runs, status="partial"), "complete"),
    "nan_return": (
        lambda runs: _mutate_eval(runs, "ftl", 3101, 1, normalized_return=np.nan),
        "finite",
    ),
    "inf_ce": (
        lambda runs: _mutate_eval(
            runs,
            "bc_iid",
            3100,
            0,
            rollout_cross_entropy=np.inf,
        ),
        "finite",
    ),
    "missing_metric": (
        lambda runs: _mutate_eval(runs, "bc_prefix", 3102, 2, disagreement_rate=None),
        "finite",
    ),
    "negative_ce": (
        lambda runs: _mutate_eval(
            runs,
            "ftl",
            3103,
            3,
            rollout_cross_entropy=-0.01,
        ),
        "cross-entropy",
    ),
    "disagreement_above_one": (
        lambda runs: _mutate_eval(runs, "bc", 3100, 0, disagreement_rate=1.2),
        "disagreement",
    ),
    "episodes_99": (
        lambda runs: _mutate_eval(runs, "ftl", 3100, 1, episode_returns=[0.0] * 99),
        "100",
    ),
    "seed_grid": (_shift_seed_grid, "budget"),
    "method_grid": (_shift_method_grid, "budget"),
    "bc_two_evals": (_second_bc_eval, "Offline BC"),
    "bc_budget": (
        lambda runs: _mutate_eval(runs, "bc", 3101, 0, n_observations=90),
        "budget",
    ),
}


@pytest.mark.parametrize("case", sorted(INVALID_CASES))
def test_restriction_comparison_rejects_invalid_groups(tmp_path, case):
    mutate, match = INVALID_CASES[case]
    runs = mutate(_comparison_runs())
    out = tmp_path / "x.png"
    with pytest.raises(ValueError, match=match):
        plot_restriction_comparison(runs, out)
    assert not out.exists()


def _point(values, low, high):
    return {"n_observations": 1, "values": values, "low": low, "high": high}


def _padding_notes(notes):
    return [n for n in notes if "band edges only" in n]


def test_footnotes_return_scale_and_axis_padding_conditions():
    runs = _comparison_runs()
    inside = {
        "normalized_return": {"ftl": [_point([0.2, 0.4], 0.2, 0.4)]},
        "disagreement_rate": {"ftl": [_point([0.0, 0.1], 0.0, 0.1)]},
        "rollout_cross_entropy": {"ftl": [_point([0.0, 0.3], 0.0, 0.3)]},
    }
    notes = restriction_comparison_footnotes(runs, inside)["footnotes"]
    assert not _mentions(notes, RETURN_NOTE)
    assert not _padding_notes(notes)  # no drawn limits, no padding claim

    exact = {"disagreement_rate": (0.0, 1.0), "rollout_cross_entropy": (0.0, 5.0)}
    notes = restriction_comparison_footnotes(runs, inside, y_limits=exact)
    assert not _padding_notes(notes["footnotes"])

    padded = {
        "disagreement_rate": (0.2, 1.04),
        "rollout_cross_entropy": (-0.0151, 0.315),
    }
    notes = restriction_comparison_footnotes(runs, inside, y_limits=padded)
    (axis_line,) = _padding_notes(notes["footnotes"])
    assert "CE axis below 0 shows band edges only; every CE >= 0." in axis_line
    assert "Disagreement axis outside [0, 1] shows band edges only" in axis_line
    assert not _mentions(notes["footnotes"], RETURN_NOTE)
    for note in notes["footnotes"]:
        assert len(note) <= 200 and chr(0x2014) not in note

    band_only = copy.deepcopy(inside)
    band_only["normalized_return"]["ftl"][0]["high"] = 1.02
    notes = restriction_comparison_footnotes(runs, band_only)["footnotes"]
    assert _mentions(notes, RETURN_NOTE) == 1

    caveats = ("Plain caveat line.",)
    out = restriction_comparison_footnotes(runs, inside, caveats=caveats)
    assert out["caveats"] == list(caveats)


def _near_bound_values():
    """CE with zeros (one seed nonzero) and disagreement touching 0 and 1."""
    values = _in_range_values()
    # One nonzero seed: the SEM band low rounds to about -7e-18, not below 0.
    values["ftl"]["rollout_cross_entropy"][:, 1] = [0, 0, 0, 0, 0.2884212144312105]
    values["bc_iid"]["rollout_cross_entropy"][:, 3] = 0.0
    values["ftl"]["disagreement_rate"][:, 3] = 0.0
    values["bc_prefix"]["disagreement_rate"][:, 0] = [1.0, 1.0, 1.0, 1.0, 0.97]
    return values


def _band_vertices(ax):
    return np.concatenate([c.get_paths()[0].vertices[:, 1] for c in ax.collections])


def test_restriction_comparison_axes_near_bounds_keep_values(
    tmp_path,
    saved_figures,
):
    values = _near_bound_values()
    receipt = plot_restriction_comparison(
        _comparison_runs(values),
        tmp_path / "x.png",
    )
    (fig,) = saved_figures
    _, dis_ax, ce_ax = fig.axes

    # The receipt records the limits actually drawn in the saved figure.
    dis_lo, dis_hi = dis_ax.get_ylim()
    ce_lo, ce_hi = ce_ax.get_ylim()
    assert receipt["y_limits"]["disagreement_rate"] == [dis_lo, dis_hi]
    assert receipt["y_limits"]["rollout_cross_entropy"] == [ce_lo, ce_hi]
    all_ce = [
        p[k]
        for series in receipt["points"]["rollout_cross_entropy"].values()
        for p in series
        for k in ("low", "high")
    ]
    dis_edges = [
        p[k]
        for series in receipt["points"]["disagreement_rate"].values()
        for p in series
        for k in ("low", "high")
    ]
    # Axes start at 0 (or the lowest band edge if rounding puts it below 0).
    assert ce_lo == min(0.0, min(all_ce)) and -1e-12 < ce_lo <= 0
    assert dis_lo == min(0.0, min(dis_edges)) and dis_hi == max(1.0, max(dis_edges))
    expected = int(ce_lo < 0) + int(dis_lo < 0 or dis_hi > 1)
    padding = _padding_notes(receipt["footnotes"])
    assert sum(n.count("band edges only") for n in padding) == expected
    assert not _mentions(receipt["footnotes"], RETURN_NOTE)

    # Values and bands are unchanged: no clipping at 0 or 1.
    column = values["ftl"]["rollout_cross_entropy"][:, 1]
    mean, sem = _seed_mean_and_sem(column)
    point = receipt["points"]["rollout_cross_entropy"]["ftl"][1]
    assert point["values"] == list(column)
    assert point["mean"] == pytest.approx(mean, abs=1e-15)
    assert point["low"] == pytest.approx(mean - sem, abs=1e-15)
    assert point["low"] > -1e-12  # rounding scale only, never a real crossing
    (ftl_ce,) = _panel_lines(ce_ax, "ftl")
    assert ftl_ce.get_ydata()[1] == pytest.approx(mean, abs=1e-15)
    ce_band = _band_vertices(ce_ax)
    ce_band = np.concatenate([ce_band, np.asarray(ce_ax.patches[0].get_xy())[:, 1]])
    assert ce_band.min() == pytest.approx(min(all_ce), abs=1e-15)
    assert ce_band.max() == pytest.approx(max(all_ce), abs=1e-15)
    dis_high = max(dis_edges)
    assert dis_high <= 1 + 1e-12
    assert _band_vertices(dis_ax).max() == pytest.approx(dis_high, abs=1e-15)


def test_restriction_comparison_no_padding_note_when_axes_in_bounds(
    tmp_path,
    saved_figures,
):
    receipt = plot_restriction_comparison(_comparison_runs(), tmp_path / "x.png")
    (fig,) = saved_figures
    _, dis_ax, ce_ax = fig.axes
    assert ce_ax.get_ylim()[0] >= 0
    assert 0 <= dis_ax.get_ylim()[0] and dis_ax.get_ylim()[1] <= 1
    assert not _padding_notes(receipt["footnotes"])
    assert "band edges only" not in _drawn_text(fig)


def test_restriction_comparison_layout_near_bounds(tmp_path, saved_figures):
    plot_restriction_comparison(
        _comparison_runs(_near_bound_values()),
        tmp_path / "x.png",
        caveats=("Near-zero metrics; every note must fit the layout.",),
    )
    (fig,) = saved_figures
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = {f"axes{i}": ax.get_tightbbox(renderer) for i, ax in enumerate(fig.axes)}
    boxes["legend"] = fig.legends[0].get_window_extent(renderer)
    boxes["title"] = fig._suptitle.get_window_extent(renderer)
    notes = [text for text in fig.texts if text is not fig._suptitle]
    assert len(notes) == 2
    for i, text in enumerate(notes):
        boxes[f"text{i}"] = text.get_window_extent(renderer)
    frame = fig.bbox
    for name, box in boxes.items():
        assert box.x0 >= frame.x0 and box.x1 <= frame.x1, name
        assert box.y0 >= frame.y0 and box.y1 <= frame.y1, name
    for (a, box_a), (b, box_b) in itertools.combinations(boxes.items(), 2):
        assert not box_a.overlaps(box_b), (a, b)


# ---------------------------------------------------------------------------
# CliffWalking: uniform-random display reference plus stalled-policy benchmark
# ---------------------------------------------------------------------------

CLIFF_BASELINES = {"expert_return": -13.0, "random_return": -2120.0}
# Diagnostic runs that stored the stalled reference, as env_baselines records it.
STALLED_BASELINES = {
    "expert_return": -13.0,
    "random_return": -200.0,
    "stalled_return": -200.0,
    "normalization_reference_type": "stalled_policy",
    "measured_uniform_random_return": -2120.0,
}
CLIFF_CAP = 200
CLIFF_STALL_NORM = (-200.0 + 2120.0) / (-13.0 + 2120.0)
CLIFF_NOTE = (
    "Cliff stall benchmark: R=-200 (normalized 0.911); random normalization "
    "unchanged."
)
STALL_LABEL = "Stalled policy (R = -200), benchmark only"
EXPERT_LABEL = "Expert reference (normalized 1)"


def _cliff_mean_return(m, seed_index, i):
    # Spans returns worse than the stall (-200) and between stall and expert.
    return -250.0 + 8.0 * seed_index + 50.0 * i + 5.0 * m


def _stored_norm(mean, baselines=CLIFF_BASELINES):
    rand, expert = baselines["random_return"], baselines["expert_return"]
    return round((mean - rand) / (expert - rand), 6)


def _set_cliff_row(row, episodes, baselines=CLIFF_BASELINES):
    row["episode_returns"] = [float(r) for r in episodes]
    row["normalized_return"] = _stored_norm(float(np.mean(episodes)), baselines)


def _cliff_runs(env="CliffWalking-v0", baselines=CLIFF_BASELINES, rid="mask_a"):
    runs = []
    for m, method in enumerate(METHODS):
        for s in range(len(SEEDS)):
            run = _comparison_run(method, s, _in_range_values(), env=env, rid=rid)
            run["episode_cap"] = CLIFF_CAP
            run["config"]["reference"]["baselines"] = dict(baselines)
            evals = [r for r in run["records"] if r.get("d_eval_size") is not None]
            for i, row in enumerate(evals):
                index = len(GRID) - 1 if method == "bc" else i
                mean = _cliff_mean_return(m, s, index)
                _set_cliff_row(row, [mean - 3.0] * 50 + [mean + 3.0] * 50, baselines)
            runs.append(run)
    return runs


def _uniform(mean, baselines=CLIFF_BASELINES):
    rand = baselines.get("measured_uniform_random_return", baselines["random_return"])
    return (mean - rand) / (baselines["expert_return"] - rand)


def _labelled_lines(ax, label):
    return [line for line in ax.get_lines() if line.get_label() == label]


def _stored_values(runs, method, index):
    rows = []
    for s in SEEDS:
        run = _find_run(runs, method, s)
        rows.append([r for r in run["records"] if r.get("d_eval_size") is not None])
    return [r[index]["normalized_return"] for r in rows]


def test_cliff_plots_stored_uniform_random_values_with_stall_line(
    tmp_path,
    saved_figures,
):
    runs = _cliff_runs()
    original = copy.deepcopy(runs)
    receipt = plot_restriction_comparison(runs, tmp_path / "cliff.png")
    assert runs == original  # inputs, including stored values, are not mutated

    # Historical records are plotted exactly as stored, unclipped.
    (fig,) = saved_figures
    for method in METHODS:
        series = receipt["points"]["normalized_return"][method]
        for k, point in enumerate(series):
            stored = _stored_values(runs, method, k)
            assert point["values"] == stored
            mean, sem = _seed_mean_and_sem(stored)
            assert point["mean"] == pytest.approx(mean, abs=1e-12)
            assert point["sem"] == pytest.approx(sem, abs=1e-12)
        if method != "bc":
            (line,) = _panel_lines(fig.axes[0], method)
            np.testing.assert_allclose(
                line.get_ydata(),
                [p["mean"] for p in series],
                atol=1e-12,
            )
    values = _in_range_values()
    ftl_ce = receipt["points"]["rollout_cross_entropy"]["ftl"][2]["values"]
    assert ftl_ce == list(values["ftl"]["rollout_cross_entropy"][:, 2])

    # Return panel: dotted stall benchmark and dashed expert line at 1.
    return_ax = fig.axes[0]
    (stall_line,) = _labelled_lines(return_ax, STALL_LABEL)
    assert stall_line.get_linestyle() == ":"
    np.testing.assert_allclose(stall_line.get_ydata(), [CLIFF_STALL_NORM] * 2)
    (expert_line,) = _labelled_lines(return_ax, EXPERT_LABEL)
    assert expert_line.get_linestyle() == "--"
    np.testing.assert_allclose(expert_line.get_ydata(), [1.0, 1.0])
    lo, hi = receipt["y_limits"]["normalized_return"]
    assert lo <= CLIFF_STALL_NORM <= hi and lo <= 1.0 <= hi
    for ax in fig.axes[1:3]:
        assert not _labelled_lines(ax, STALL_LABEL)
        assert not _labelled_lines(ax, EXPERT_LABEL)
    (legend,) = fig.legends
    legend_text = [t.get_text() for t in legend.get_texts()]
    assert STALL_LABEL in legend_text and EXPERT_LABEL in legend_text
    assert "stalled-policy lower reference" not in return_ax.get_ylabel()

    notes = receipt["footnotes"]
    assert notes[2] == CLIFF_NOTE
    assert notes.count(CLIFF_NOTE) == 1
    assert not _mentions(notes, RETURN_NOTE)  # every value lies in [0, 1]
    assert len(notes) == 3
    drawn = _drawn_text(fig)
    for note in notes:
        assert _flat(note) in drawn
        assert chr(0x2014) not in note

    norm = receipt["normalization"]
    assert norm["display_override"] is None
    # The pinned reference wins over the run's data baselines.
    assert {r["random_return"] for r in norm["references"]} == {-2120.0}
    assert {r["expert_return"] for r in norm["references"]} == {-13.0}
    assert {r["normalization_reference_type"] for r in norm["references"]} == {
        "uniform_random",
    }
    assert not any(r["reference_type_recorded"] for r in norm["references"])
    for ref in norm["display_references"]:
        assert ref["lower_reference"] == "uniform-random policy"
        assert ref["lower_return"] == -2120.0
        assert ref["lower_return_source"] == "recorded random_return"
        assert ref["stall_return"] == -200.0
        assert ref["stall_normalized"] == pytest.approx(CLIFF_STALL_NORM)
    bench = norm["stall_benchmark"]
    assert bench["role"] == "additional benchmark, not a normalization reference"
    assert bench["raw_return"] == -200.0 and bench["episode_cap"] == CLIFF_CAP
    assert bench["identical"] is True
    assert len(bench["per_run"]) == len(runs)
    assert bench["mean"] == bench["min"] == bench["max"]
    assert bench["mean"] == pytest.approx(CLIFF_STALL_NORM)


def test_cliff_anchor_points_and_unclipped_values(tmp_path):
    runs = _cliff_runs()
    for run in runs:
        evals = [r for r in run["records"] if r.get("d_eval_size") is not None]
        if run["method"] == "ftl":
            _set_cliff_row(evals[0], [-2120.0] * 100)  # uniform-random reference
            _set_cliff_row(evals[1], [-13.0] * 100)  # expert reference
            _set_cliff_row(evals[2], [-200.0] * 100)  # stalled policy
            _set_cliff_row(evals[3], [-3000.0] * 100)  # worse than random
        if run["method"] == "bc_prefix":
            _set_cliff_row(evals[3], [-5.0] * 100)  # better than the expert
    receipt = plot_restriction_comparison(runs, tmp_path / "x.png")
    ftl = receipt["points"]["normalized_return"]["ftl"]
    assert ftl[0]["values"] == [0.0] * len(SEEDS)
    assert ftl[1]["values"] == [1.0] * len(SEEDS)
    np.testing.assert_allclose(ftl[2]["values"], CLIFF_STALL_NORM, atol=1e-6)
    np.testing.assert_allclose(ftl[3]["values"], -880.0 / 2107.0, atol=1e-6)
    assert ftl[3]["mean"] < 0 and ftl[3]["low"] < 0
    prefix = receipt["points"]["normalized_return"]["bc_prefix"][3]
    np.testing.assert_allclose(prefix["values"], 2115.0 / 2107.0, atol=1e-6)
    assert prefix["high"] > 1
    assert receipt["y_limits"]["normalized_return"] == [-0.5, 1.5]
    assert -0.5 < ftl[3]["low"] and prefix["high"] < 1.5  # inside the view
    assert receipt["footnotes"][2] == CLIFF_NOTE
    assert _mentions(receipt["footnotes"], RETURN_NOTE) == 1


def test_stored_stalled_records_redisplayed_on_measured_uniform_random(
    tmp_path,
    saved_figures,
):
    runs = _cliff_runs(baselines=STALLED_BASELINES)
    original = copy.deepcopy(runs)
    receipt = plot_restriction_comparison(runs, tmp_path / "cliff.png")
    assert runs == original

    # Recomputed from raw episode means with the recorded measured R_random.
    for m, method in enumerate(METHODS):
        grid = [GRID[-1]] if method == "bc" else GRID
        for k, point in enumerate(receipt["points"]["normalized_return"][method]):
            i = GRID.index(grid[k])
            expected = [
                _uniform(_cliff_mean_return(m, s, i), STALLED_BASELINES)
                for s in range(len(SEEDS))
            ]
            np.testing.assert_allclose(point["values"], expected, atol=1e-12)
            assert point["values"] != _stored_values(runs, method, k)

    notes = receipt["footnotes"]
    assert notes[2] == CLIFF_NOTE
    assert notes[3] == (
        "Stored stall-referenced values redisplayed on measured random "
        "(R_random=-2120); records unchanged."
    )
    drawn = _drawn_text(saved_figures[0])
    for note in notes:
        assert _flat(note) in drawn
        assert chr(0x2014) not in note
    (stall_line,) = _labelled_lines(saved_figures[0].axes[0], STALL_LABEL)
    np.testing.assert_allclose(stall_line.get_ydata(), [CLIFF_STALL_NORM] * 2)

    norm = receipt["normalization"]
    override = norm["display_override"]
    assert override["original_stored_lower_reference"] == "stalled policy"
    assert override["original_stored_lower_returns"] == [-200.0]
    assert override["measured_uniform_random_returns"] == [-2120.0]
    assert override["chosen_lower_reference"] == "uniform-random policy"
    assert override["clipped"] is False
    assert override["stored_values_verified_against"] == (
        "recorded R_stall and R_expert"
    )
    for ref in norm["references"]:
        assert ref["normalization_reference_type"] == "stalled_policy"
        assert ref["reference_type_recorded"] is True
        assert ref["random_return"] == -200.0
        assert ref["stalled_return"] == -200.0
        assert ref["measured_uniform_random_return"] == -2120.0
    for ref in norm["display_references"]:
        assert ref["lower_return"] == -2120.0
        assert ref["lower_return_source"] == "recorded measured_uniform_random_return"
    assert norm["stall_benchmark"]["mean"] == pytest.approx(CLIFF_STALL_NORM)


def test_cliff_stall_line_with_per_seed_references_shows_range(
    tmp_path,
    saved_figures,
):
    other = {"expert_return": -15.0, "random_return": -2000.0}
    runs = _cliff_runs()
    for run in runs:
        if run["seed"] == SEEDS[-1]:
            run["config"]["reference"]["baselines"] = dict(other)
            for row in run["records"]:
                if row.get("d_eval_size") is not None:
                    _set_cliff_row(row, row["episode_returns"], other)
    receipt = plot_restriction_comparison(runs, tmp_path / "x.png")
    bench = receipt["normalization"]["stall_benchmark"]
    seeds_norm = {r["seed"]: r["normalized"] for r in bench["per_run"]}
    other_norm = 1800.0 / 1985.0
    assert seeds_norm[SEEDS[-1]] == pytest.approx(other_norm)
    assert seeds_norm[SEEDS[0]] == pytest.approx(CLIFF_STALL_NORM)
    assert bench["identical"] is False
    assert bench["min"] == pytest.approx(min(other_norm, CLIFF_STALL_NORM))
    assert bench["max"] == pytest.approx(max(other_norm, CLIFF_STALL_NORM))
    expected_mean = (16 * CLIFF_STALL_NORM + 4 * other_norm) / 20
    assert bench["mean"] == pytest.approx(expected_mean)

    ax = saved_figures[0].axes[0]
    (stall_line,) = _labelled_lines(ax, STALL_LABEL)
    np.testing.assert_allclose(stall_line.get_ydata(), [expected_mean] * 2)
    notes = receipt["footnotes"]
    assert notes[2] == (
        "Cliff stall benchmark: R=-200 (normalized 0.907 to 0.911 across runs, "
        "shaded); random normalization unchanged."
    )
    assert (
        "Normalization references differ across runs; per-run values in receipt."
        in notes[3]
    )
    refs = receipt["normalization"]["references"]
    assert {r["random_return"] for r in refs} == {-2120.0, -2000.0}


def test_non_cliff_runs_keep_recorded_normalization(tmp_path, saved_figures):
    values = _out_of_range_values()
    receipt = plot_restriction_comparison(_comparison_runs(values), tmp_path / "x")
    point = receipt["points"]["normalized_return"]["ftl"][2]
    assert point["values"] == list(values["ftl"]["normalized_return"][:, 2])
    assert receipt["normalization"]["display_override"] is None
    assert receipt["normalization"]["stall_benchmark"] is None
    assert {
        r["lower_reference"] for r in receipt["normalization"]["display_references"]
    } == {
        "uniform-random policy",
    }
    assert CLIFF_NOTE not in receipt["footnotes"]
    fig = saved_figures[0]
    assert "CliffWalking" not in _drawn_text(fig)
    assert not _labelled_lines(fig.axes[0], STALL_LABEL)
    assert not _labelled_lines(fig.axes[0], EXPERT_LABEL)
    assert _mentions(receipt["footnotes"], RETURN_NOTE) == 1


def _cliff_mutate(method, seed, mutate):
    def apply(runs):
        mutate(_find_run(runs, method, seed))
        return runs

    return apply


def _first_eval(run):
    return next(r for r in run["records"] if r.get("d_eval_size") is not None)


CLIFF_INVALID_CASES = {
    "missing_cap": (_cliff_mutate("ftl", 3101, lambda r: r.pop("episode_cap")), "cap"),
    "wrong_cap": (
        _cliff_mutate("ftl", 3101, lambda r: r.update(episode_cap=500)),
        "cap",
    ),
    "zero_cap": (lambda runs: [{**r, "episode_cap": 0} for r in runs], "cap"),
    "bool_cap": (lambda runs: [{**r, "episode_cap": True} for r in runs], "cap"),
    "missing_raw_returns": (
        _cliff_mutate(
            "bc_iid",
            3100,
            lambda r: _first_eval(r).update(episode_returns=None),
        ),
        "100",
    ),
    "missing_expert_reference": (
        _cliff_mutate(
            "bc",
            3102,
            lambda r: r["config"]["reference"]["baselines"].pop("expert_return"),
        ),
        "reference",
    ),
    "expert_not_above_stall": (
        _cliff_mutate(
            "bc",
            3102,
            lambda r: r["config"]["reference"]["baselines"].update(
                expert_return=-250.0,
            ),
        ),
        "stalled-policy benchmark",
    ),
    "stored_norm_mismatch": (
        _cliff_mutate(
            "bc_prefix",
            3103,
            lambda r: _first_eval(r).update(
                normalized_return=_first_eval(r)["normalized_return"] + 0.01,
            ),
        ),
        "stored normalized_return",
    ),
}


@pytest.mark.parametrize("case", sorted(CLIFF_INVALID_CASES))
def test_cliff_display_normalization_rejects_unverifiable_runs(tmp_path, case):
    mutate, match = CLIFF_INVALID_CASES[case]
    runs = mutate(_cliff_runs())
    out = tmp_path / "x.png"
    with pytest.raises(ValueError, match=match):
        plot_restriction_comparison(runs, out)
    assert not out.exists()


def test_explicit_uniform_reference_plots_stored_values(tmp_path):
    baselines = {
        **CLIFF_BASELINES,
        "normalization_reference_type": "uniform_random",
        "measured_uniform_random_return": CLIFF_BASELINES["random_return"],
    }
    runs = _cliff_runs(baselines=baselines)
    receipt = plot_restriction_comparison(runs, tmp_path / "x.png")
    assert receipt["footnotes"][2] == CLIFF_NOTE
    refs = receipt["normalization"]["references"]
    assert {r["random_return"] for r in refs} == {-2120.0}
    assert receipt["normalization"]["display_override"] is None
    ftl = receipt["points"]["normalized_return"]["ftl"][0]
    assert ftl["values"] == _stored_values(runs, "ftl", 0)
    assert all(
        r["reference_type_recorded"] for r in receipt["normalization"]["references"]
    )


def test_varied_start_cliff_has_no_stall_line(tmp_path, saved_figures):
    runs = _cliff_runs(env="CliffWalkingVariedStart-v0")
    receipt = plot_restriction_comparison(runs, tmp_path / "x.png")
    assert receipt["normalization"]["display_override"] is None
    assert receipt["normalization"]["stall_benchmark"] is None
    assert CLIFF_NOTE not in receipt["footnotes"]
    ftl = receipt["points"]["normalized_return"]["ftl"][0]
    assert ftl["values"] == _stored_values(runs, "ftl", 0)
    fig = saved_figures[0]
    assert "stall" not in _drawn_text(fig).lower()
    assert not _labelled_lines(fig.axes[0], STALL_LABEL)


def _stalled_cohort(**changes):
    def apply(runs):
        for run in runs:
            run["config"]["reference"]["baselines"].update(changes)
        return runs

    return apply


def _drop_stalled_key(key):
    def apply(runs):
        for run in runs:
            run["config"]["reference"]["baselines"].pop(key)
        return runs

    return apply


def _one_run_historical(runs):
    run = _find_run(runs, "ftl", 3102)
    run["config"]["reference"]["baselines"] = dict(CLIFF_BASELINES)
    for row in run["records"]:
        if row.get("d_eval_size") is not None:
            _set_cliff_row(row, row["episode_returns"])
    return runs


REFERENCE_METADATA_INVALID_CASES = {
    "unknown_type": (
        "CliffWalking-v0",
        _stalled_cohort(normalization_reference_type="stall"),
        "unknown normalization_reference_type",
    ),
    "stalled_on_non_cliff": ("Acrobot-v1", lambda runs: runs, "allowed only on"),
    "stalled_on_varied_start": (
        "CliffWalkingVariedStart-v0",
        lambda runs: runs,
        "allowed only on",
    ),
    "stalled_value_conflict": (
        "CliffWalking-v0",
        _stalled_cohort(stalled_return=-199.0),
        "stalled_return equal to random_return",
    ),
    "stalled_missing_value": (
        "CliffWalking-v0",
        _drop_stalled_key("stalled_return"),
        "stalled_return equal to random_return",
    ),
    "stalled_cap_conflict": (
        "CliffWalking-v0",
        lambda runs: [{**r, "episode_cap": 150} for r in runs],
        "cap",
    ),
    "stalled_return_not_minus_cap": (
        "CliffWalking-v0",
        _stalled_cohort(random_return=-150.0, stalled_return=-150.0),
        "differs from -episode_cap",
    ),
    "stalled_missing_measured_random": (
        "CliffWalking-v0",
        _drop_stalled_key("measured_uniform_random_return"),
        "lacks measured_uniform_random_return",
    ),
    "non_finite_measured_random": (
        "CliffWalking-v0",
        _stalled_cohort(measured_uniform_random_return=float("nan")),
        "must be finite",
    ),
    "uniform_with_stalled_return": (
        "CliffWalking-v0",
        _stalled_cohort(normalization_reference_type="uniform_random"),
        "also records stalled_return",
    ),
    "mixed_cohort": ("CliffWalking-v0", _one_run_historical, "mix normalization"),
    "measured_random_not_below_stall": (
        "CliffWalking-v0",
        _stalled_cohort(measured_uniform_random_return=-150.0),
        "is not below the stalled return",
    ),
    "unlabelled_stall_value": (
        "CliffWalking-v0",
        _drop_stalled_key("normalization_reference_type"),
        "also records stalled_return",
    ),
    "untyped_minus_200_only": (
        "CliffWalking-v0",
        lambda runs: _drop_stalled_key("measured_uniform_random_return")(
            _drop_stalled_key("stalled_return")(
                _drop_stalled_key("normalization_reference_type")(runs),
            ),
        ),
        "refusing to label it uniform-random",
    ),
    "descriptor_disagrees": (
        "CliffWalking-v0",
        lambda runs: [
            {
                **r,
                "config": {
                    **r["config"],
                    "normalization_reference": {
                        "type": "uniform_random",
                        "lower_return": -2120.0,
                    },
                },
            }
            for r in runs
        ],
        "disagrees with the baselines",
    ),
}


@pytest.mark.parametrize("case", sorted(REFERENCE_METADATA_INVALID_CASES))
def test_invalid_reference_metadata_rejected(tmp_path, case):
    env, mutate, match = REFERENCE_METADATA_INVALID_CASES[case]
    runs = mutate(_cliff_runs(env=env, baselines=STALLED_BASELINES))
    out = tmp_path / "x.png"
    with pytest.raises(ValueError, match=match):
        plot_restriction_comparison(runs, out)
    assert not out.exists()


def test_footnotes_reject_mixed_reference_types_even_with_protocol():
    runs = _one_run_historical(_cliff_runs(baselines=STALLED_BASELINES))
    with pytest.raises(ValueError, match="mix normalization"):
        restriction_comparison_footnotes(
            runs,
            {},
            protocol={"paired_cold_fits": False, "paired_starts": False},
        )


def test_stalled_reference_read_from_data_baselines(tmp_path):
    runs = _cliff_runs(baselines=STALLED_BASELINES)
    for run in runs:
        run["data"]["baselines"] = run["config"].pop("reference")["baselines"]
        run["config"]["normalization_reference"] = {
            "type": "stalled_policy",
            "lower_return": -200.0,
        }
    receipt = plot_restriction_comparison(runs, tmp_path / "x.png")
    override = receipt["normalization"]["display_override"]
    assert override["original_stored_lower_reference"] == "stalled policy"
    assert override["measured_uniform_random_returns"] == [-2120.0]
    assert override["chosen_lower_reference"] == "uniform-random policy"


def test_uniform_reference_with_conflicting_measured_random_rejected(tmp_path):
    baselines = {**CLIFF_BASELINES, "measured_uniform_random_return": -1000.0}
    out = tmp_path / "x.png"
    with pytest.raises(ValueError, match="differs from measured_uniform_random"):
        plot_restriction_comparison(_cliff_runs(baselines=baselines), out)
    assert not out.exists()


# Historical column-pair Cliff: no memoryless policy reaches the goal, so each
# capped episode returns -200 - 99k for k cliff falls.
def _column_pair_runs(falls_by_method, successes=None, baselines=CLIFF_BASELINES):
    runs = _cliff_runs(baselines=baselines, rid="cliffwalking_column_pairs")
    for run in runs:
        for row in run["records"]:
            if row.get("d_eval_size") is not None:
                falls = falls_by_method[run["method"]]
                _set_cliff_row(row, [-200.0 - 99.0 * falls] * 100, baselines)
                if successes is not None:
                    row["episode_successes"] = [successes] * 100
    return runs


COLUMN_PAIR_FALLS = {"bc": 0, "bc_iid": 0, "bc_prefix": 0, "ftl": 13}


@pytest.mark.parametrize("baselines", [CLIFF_BASELINES, STALLED_BASELINES])
def test_cliff_column_pairs_diagnostic_note(tmp_path, saved_figures, baselines):
    runs = _column_pair_runs(COLUMN_PAIR_FALLS, baselines=baselines)
    receipt = plot_restriction_comparison(runs, tmp_path / "x.png")
    notes = receipt["footnotes"]
    rows = sum(r.get("d_eval_size") is not None for run in runs for r in run["records"])
    (diagnostic,) = [n for n in notes if n.startswith("Column-pair bins")]
    assert diagnostic == (
        "Column-pair bins: no goal-reaching deterministic memoryless policy. "
        f"Goal success not recorded ({rows}/{rows} rows); not imputed."
    )
    assert notes[2] == CLIFF_NOTE
    assert receipt["column_pairs_audit"]["success_not_recorded_rows"] == rows
    drawn = _drawn_text(saved_figures[0])
    for note in notes:
        assert _flat(note) in drawn
        assert chr(0x2014) not in note
    # Stalled methods sit on the stall benchmark line; FTL's falls are below it.
    points = receipt["points"]["normalized_return"]
    np.testing.assert_allclose(
        points["bc_iid"][0]["values"],
        CLIFF_STALL_NORM,
        atol=1e-6,
    )
    assert points["ftl"][0]["mean"] == pytest.approx(
        CLIFF_STALL_NORM - 13 * 99 / 2107,
        abs=1e-6,
    )


def test_cliff_column_pairs_recorded_failures_need_no_missing_note(tmp_path):
    runs = _column_pair_runs(COLUMN_PAIR_FALLS, successes=False)
    notes = plot_restriction_comparison(runs, tmp_path / "x.png")["footnotes"]
    assert "Column-pair bins: no goal-reaching deterministic memoryless policy." in notes
    assert not any("not recorded" in n for n in notes)


def test_cliff_column_pairs_note_only_on_that_mask(tmp_path):
    notes = plot_restriction_comparison(_cliff_runs(), tmp_path / "x.png")["footnotes"]
    assert not any(n.startswith("Column-pair") for n in notes)


def _goal_return(runs):
    row = _first_eval(_find_run(runs, "ftl", 3101))
    _set_cliff_row(row, [-15.0] * 100)
    return runs


def _off_lattice_return(runs):
    row = _first_eval(_find_run(runs, "bc_iid", 3100))
    _set_cliff_row(row, [-250.0] * 100)
    return runs


@pytest.mark.parametrize(
    "mutate,successes",
    [(_goal_return, None), (_off_lattice_return, None), (lambda r: r, True)],
)
def test_cliff_column_pairs_contradiction_fails_closed(tmp_path, mutate, successes):
    runs = mutate(_column_pair_runs(COLUMN_PAIR_FALLS, successes=successes))
    out = tmp_path / "x.png"
    with pytest.raises(ValueError, match="goal-infeasibility"):
        plot_restriction_comparison(runs, out)
    assert not out.exists()


def test_production_round_eval_uses_deterministic_actions(monkeypatch):
    """The footnotes say evaluation is deterministic; pin the production call."""
    from imitation.experiments.ftrl import eval_utils, run_experiment

    calls = []

    class _Stop(Exception):
        pass

    def spy(*args, **kwargs):
        calls.append(kwargs)
        raise _Stop

    monkeypatch.setattr(eval_utils, "eval_policy_rollout", spy)
    baselines = {"expert_return": 1.0, "random_return": 0.0}
    for eval_seed in (None, 7):
        with pytest.raises(_Stop):
            run_experiment._compute_round_eval(
                object(),
                object(),
                object(),
                baselines,
                eval_seed=eval_seed,
            )
    assert [c.get("deterministic") for c in calls] == [True, True]
    assert calls[0]["n_episodes"] == calls[1]["n_episodes"] == run_experiment.EVAL_EPISODES


# ---------------------------------------------------------------------------
# Paired cold-fit protocol (receipt audit)
# ---------------------------------------------------------------------------

def _episode_seeds(seed, budget):
    return [seed * 1000 + budget * 7 + k for k in range(100)]


def _paired_runs(field="evaluation_episode_seeds"):
    runs = _comparison_runs()
    for run in runs:
        run["config"]["experiment"] = {"paired_cold_fits": True}
        for row in run["records"]:
            if row.get("d_eval_size") is not None:
                row[field] = _episode_seeds(run["seed"], row["n_observations"])
    return runs


@pytest.mark.parametrize("field", ["evaluation_episode_seeds", "episode_seeds"])
def test_paired_protocol_footnotes_when_episode_seeds_match(
    tmp_path,
    saved_figures,
    field,
):
    receipt = plot_restriction_comparison(_paired_runs(field), tmp_path / "x.png")
    # Pairing is audited in the receipt; the short footer does not restate it.
    assert receipt["protocol"] == {"paired_cold_fits": True, "paired_starts": True}
    assert "paired" not in _flat(" ".join(receipt["footnotes"]))
    drawn = _drawn_text(saved_figures[0])
    for note in receipt["footnotes"]:
        assert _flat(note) in drawn


def _drop_seeds(runs):
    del _first_eval(_find_run(runs, "ftl", 3102))["evaluation_episode_seeds"]
    return runs


def _shift_seeds(runs):
    row = _first_eval(_find_run(runs, "bc_iid", 3100))
    row["evaluation_episode_seeds"] = [s + 1 for s in row["evaluation_episode_seeds"]]
    return runs


def _shift_bc_seeds(runs):
    row = _first_eval(_find_run(runs, "bc", 3104))
    row["evaluation_episode_seeds"] = list(reversed(row["evaluation_episode_seeds"]))
    return runs


@pytest.mark.parametrize("mutate", [_drop_seeds, _shift_seeds, _shift_bc_seeds])
def test_paired_fits_with_unpaired_starts_say_unpaired(tmp_path, mutate):
    receipt = plot_restriction_comparison(mutate(_paired_runs()), tmp_path / "x.png")
    assert receipt["protocol"] == {"paired_cold_fits": True, "paired_starts": False}


def test_legacy_runs_keep_original_protocol_footnotes(tmp_path):
    runs = _comparison_runs()
    for run in runs:  # episode seeds alone do not claim the paired protocol
        for row in run["records"]:
            row["evaluation_episode_seeds"] = _episode_seeds(run["seed"], 0)
    receipt = plot_restriction_comparison(runs, tmp_path / "x.png")
    assert receipt["protocol"] == {"paired_cold_fits": False, "paired_starts": False}


@pytest.mark.parametrize("value", [None, False, "yes"])
def test_mixed_fit_protocols_rejected(tmp_path, value):
    runs = _paired_runs()
    experiment = runs[5]["config"]["experiment"]
    if value is None:
        del experiment["paired_cold_fits"]
    else:
        experiment["paired_cold_fits"] = value
    out = tmp_path / "x.png"
    with pytest.raises(ValueError, match="protocol"):
        plot_restriction_comparison(runs, out)
    assert not out.exists()


def test_paired_cliff_runs_combine_protocol_and_stall_notes(tmp_path):
    runs = _cliff_runs()
    for run in runs:
        run["config"]["experiment"] = {"paired_cold_fits": True}
        for row in run["records"]:
            if row.get("d_eval_size") is not None:
                row["episode_seeds"] = _episode_seeds(
                    run["seed"], row["n_observations"]
                )
    receipt = plot_restriction_comparison(runs, tmp_path / "x.png")
    assert receipt["footnotes"][2] == CLIFF_NOTE
    assert receipt["protocol"] == {"paired_cold_fits": True, "paired_starts": True}


# ---------------------------------------------------------------------------
# Fixed normalized-return view and concise footers
# ---------------------------------------------------------------------------

YLIM = (-0.5, 1.5)
SCALE_NOTE = "Random=0, expert=1; values outside [0,1] are valid."
OFFSCALE_NOTE = "Off-scale portions are not shown; values unchanged."
SETUP_NOTE = "Mean +/- 1 SEM; n=5 seeds (3100-3104); Offline BC: one full-pool fit."
ACROBOT_NAMES = [
    "cos_theta1",
    "sin_theta1",
    "cos_theta2",
    "sin_theta2",
    "theta1_dot",
    "theta2_dot",
]
ZERO_DESCRIPTOR = {
    "restriction_id": "acrobot_theta_dots_zero",
    "zeroed_indices": [4, 5],
    "obs_shape": [6],
    "description": (
        "Acrobot-v1 learner features theta1_dot, theta2_dot := 0; kept: "
        "cos_theta1, sin_theta1, cos_theta2, sin_theta2."
    ),
}
RANDOM_RID = "taxi_random_bins120_seed20261003"
RANDOM_DESCRIPTOR = {
    "kind": "alias",
    "restriction_id": RANDOM_RID,
    "env_name": "Taxi-v3",
    "hidden": "every factor: 120 balanced random bins (partition seed 20261003)",
    "reachable_nonterminal_states": 400,
    "reachable_classes": 120,
    "rule": "random binning rule text",
    "description": "long description that is not shown",
    "partition": {
        "kind": "random_state_binning",
        "restriction_id": RANDOM_RID,
        "env_name": "Taxi-v3",
        "seed": 20261003,
        "n_bins": 120,
        "n_reachable_nonterminal": 400,
    },
    "partition_sha256": "0" * 64,
    "groups": [[0, 1]],
}
ALIAS_DESCRIPTOR = {
    "kind": "alias",
    "restriction_id": "frozenlake_column_pairs",
    "env_name": "FrozenLake-v1",
    "hidden": "column within pairs {2k, 2k+1}",
    "reachable_nonterminal_states": 11,
    "reachable_classes": 8,
    "rule": "classes are the visible factors; long rule text",
    "table_sha256": "1" * 64,
}
QUANTIZED_DESCRIPTOR = {
    "kind": "quantized",
    "restriction_id": "mountaincar_bins_pos005_vel015",
    "env_name": "MountainCar-v0",
    "features": [
        {"name": "position", "index": 0, "rule": "round_to_step", "step": 0.05},
        {
            "name": "velocity",
            "index": 1,
            "rule": "sign_dead_zone",
            "dead_zone": 0.015,
            "magnitude": 0.035,
        },
    ],
    "description": "long description that is not shown",
}


def _with_restriction(runs, descriptor, env=None, feature_names=None):
    rid = descriptor["restriction_id"]
    for run in runs:
        run["restriction_id"] = rid
        if env is not None:
            run["env_name"] = env
        run["config"]["restriction_id"] = rid
        run["config"]["restriction"] = copy.deepcopy(descriptor)
        if feature_names is not None:
            run["task"] = {"declared": {"feature_names": list(feature_names)}}
    return runs


def _acrobot_runs(values=None):
    return _with_restriction(
        _comparison_runs(values),
        ZERO_DESCRIPTOR,
        feature_names=ACROBOT_NAMES,
    )


def _footer_lines(fig):
    """Rendered footnote lines (the first text block below the legend)."""
    notes = [t for t in fig.texts if t is not fig._suptitle]
    return notes[0].get_text().splitlines()


def _ftl_lineset(ax, label):
    return [ln for ln in ax.get_lines() if ln.get_label() == label]


def _legacy_df(normalized_by_round, bc_normalized):
    rows = []
    for seed in range(3):
        for rnd, values in enumerate(normalized_by_round, start=1):
            rows.append(
                {
                    "algo": "ftl",
                    "env": "CartPole-v1",
                    "seed": seed,
                    "policy_mode": "end_to_end",
                    "round": rnd,
                    "n_observations": 100 * rnd,
                    "rollout_cross_entropy": 0.5 / rnd,
                    "expert_rollout_cross_entropy": 0.01,
                    "normalized_return": values[seed],
                    "disagreement_rate": 0.2 / rnd,
                    "expert_self_ce": 0.01,
                },
            )
        rows.append(
            {
                "algo": "bc",
                "env": "CartPole-v1",
                "seed": seed,
                "policy_mode": "end_to_end",
                "round": 1,
                "n_observations": 300,
                "rollout_cross_entropy": 0.4,
                "expert_rollout_cross_entropy": 0.01,
                "normalized_return": bc_normalized,
                "disagreement_rate": 0.3,
                "expert_self_ce": 0.01,
            },
        )
    return compute_cumulative_regret(compute_cumulative_loss(pd.DataFrame(rows)))


def test_plot_env_normalized_return_linear_fixed_view_unclipped(
    tmp_path,
    saved_figures,
):
    by_round = [[-0.9, -0.8, -0.7], [0.3, 0.4, 0.5], [1.8, 2.0, 2.2]]
    df = _legacy_df(by_round, bc_normalized=-0.7)
    plot_env(df, "CartPole-v1", tmp_path / "x.png", band="sem")
    (fig,) = saved_figures
    _, ret_ax, dis_ax, _ = fig.axes

    assert ret_ax.get_yscale() == "linear"
    assert ret_ax.get_ylim() == YLIM
    assert dis_ax.get_yscale() == "log"  # other metrics keep their settings
    (line,) = _ftl_lineset(ret_ax, "FTL+DAgger")
    means = [np.mean(v) for v in by_round]
    np.testing.assert_allclose(line.get_ydata(), means, rtol=0, atol=1e-12)
    sems = [np.std(v, ddof=1) / np.sqrt(3) for v in by_round]
    band = _band_vertices(ret_ax)
    assert band.min() == pytest.approx(means[0] - sems[0], abs=1e-12)
    assert band.max() == pytest.approx(means[2] + sems[2], abs=1e-12)
    (bc_line,) = [
        ln for ln in ret_ax.get_lines() if ln.get_label().startswith("BC (fixed)")
    ]
    np.testing.assert_allclose(bc_line.get_ydata(), [-0.7, -0.7], atol=1e-12)
    assert bc_line.get_label() == "BC (fixed) = -0.700"
    assert _plot_env_scale_note(fig) == f"{SCALE_NOTE} {OFFSCALE_NOTE}"


def _plot_env_scale_note(fig):
    notes = [t for t in fig.texts if t.get_gid() == "normalized_return_scale_note"]
    assert len(notes) <= 1
    return notes[0].get_text() if notes else None


def _plot_env_note_for(tmp_path, saved_figures, by_round, bc_normalized):
    df = _legacy_df(by_round, bc_normalized=bc_normalized)
    plot_env(df, "CartPole-v1", tmp_path / "x.png", band="sem")
    fig = saved_figures[-1]
    assert fig.axes[1].get_ylim() == YLIM
    return _plot_env_scale_note(fig)


@pytest.mark.parametrize(
    "by_round,bc_normalized",
    [
        ([[-0.9, -0.8, -0.7], [0.3, 0.4, 0.5]], 0.5),  # center below -0.5
        ([[0.2, 0.3, 0.4], [1.7, 1.8, 1.9]], 0.5),  # center above 1.5
        ([[-1.0, 0.0, 0.2], [0.3, 0.4, 0.5]], 0.5),  # only a band edge below
        ([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]], 1.6),  # only the Offline BC line
    ],
)
def test_plot_env_offscale_note(tmp_path, saved_figures, by_round, bc_normalized):
    note = _plot_env_note_for(tmp_path, saved_figures, by_round, bc_normalized)
    assert note == f"{SCALE_NOTE} {OFFSCALE_NOTE}"


def test_plot_env_in_range_has_no_scale_note(tmp_path, saved_figures):
    # The expert line at 1 and the [-0.5, 1.5] ticks must not trigger a note.
    by_round = [[0.0, 0.1, 0.2], [0.8, 0.9, 1.0]]
    assert _plot_env_note_for(tmp_path, saved_figures, by_round, 1.0) is None


@pytest.mark.parametrize(
    "by_round,bc_normalized",
    [
        ([[-0.3, -0.2, -0.1], [1.2, 1.3, 1.4]], 0.5),  # centers out of [0,1]
        ([[-0.05, 0.5, 0.6], [0.3, 0.4, 0.5]], 0.5),  # one seed only; band inside
        ([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]], -0.2),  # Offline BC line
    ],
)
def test_plot_env_out_of_unit_in_view_has_scale_note_only(
    tmp_path,
    saved_figures,
    by_round,
    bc_normalized,
):
    note = _plot_env_note_for(tmp_path, saved_figures, by_round, bc_normalized)
    assert note == SCALE_NOTE


def test_plot_env_scale_note_fits_without_overlap(tmp_path, saved_figures):
    by_round = [[-0.9, -0.8, -0.7], [1.8, 2.0, 2.2]]
    _plot_env_note_for(tmp_path, saved_figures, by_round, -0.7)
    fig = saved_figures[-1]
    (note,) = [t for t in fig.texts if t.get_gid() == "normalized_return_scale_note"]
    renderer = fig.canvas.get_renderer()
    box = note.get_window_extent(renderer)
    figure_box = fig.bbox
    assert box.x0 >= figure_box.x0 and box.x1 <= figure_box.x1
    assert box.y0 >= figure_box.y0
    for ax in fig.axes:
        assert not box.overlaps(ax.get_tightbbox(renderer))


def _offscale_values():
    values = _out_of_range_values()
    values["bc_prefix"]["normalized_return"][:, 3] = [1.6, 1.9, 1.7, 1.8, 2.0]
    values["ftl"]["normalized_return"][:, 2] = [-1.4, -1.2, -0.9, -1.0, -0.6]
    return values


def test_restriction_normalized_axis_fixed_view_values_unchanged(
    tmp_path,
    saved_figures,
):
    values = _offscale_values()
    runs = _acrobot_runs(values)
    original = copy.deepcopy(runs)
    receipt = plot_restriction_comparison(runs, tmp_path / "x.png")
    assert runs == original
    (fig,) = saved_figures
    ret_ax = fig.axes[0]
    assert ret_ax.get_yscale() == "linear"
    assert ret_ax.get_ylim() == YLIM
    assert receipt["y_limits"]["normalized_return"] == list(YLIM)

    for method in ("ftl", "bc_prefix"):
        (line,) = _panel_lines(ret_ax, method)
        expected = [
            _seed_mean_and_sem(values[method]["normalized_return"][:, i])[0]
            for i in range(len(GRID))
        ]
        np.testing.assert_allclose(line.get_ydata(), expected, rtol=0, atol=1e-12)
    assert min(_panel_lines(ret_ax, "ftl")[0].get_ydata()) < YLIM[0]
    assert max(_panel_lines(ret_ax, "bc_prefix")[0].get_ydata()) > YLIM[1]
    highs = [
        p["high"]
        for series in receipt["points"]["normalized_return"].values()
        for p in series
    ]
    assert _band_vertices(ret_ax).max() == pytest.approx(max(highs), abs=1e-12)
    point = receipt["points"]["normalized_return"]["ftl"][2]
    assert point["values"] == list(values["ftl"]["normalized_return"][:, 2])
    mean, sem = _seed_mean_and_sem(values["ftl"]["normalized_return"][:, 2])
    assert point["mean"] == pytest.approx(mean, abs=1e-12)
    assert point["sem"] == pytest.approx(sem, abs=1e-12)

    view = receipt["normalized_return_view"]
    assert view["y_limits"] == list(YLIM) and view["fixed"] is True
    assert view["data_modified"] is False
    assert view["outside"]["means"] == 2
    assert view["outside"]["band_edges"] >= 2
    assert view["outside"]["min"] == pytest.approx(min(p["low"] for s in receipt[
        "points"]["normalized_return"].values() for p in s))
    assert view["outside"]["max"] == pytest.approx(max(highs))

    scale = [n for n in receipt["footnotes"] if n.startswith(SCALE_NOTE)]
    assert scale == [f"{SCALE_NOTE} {OFFSCALE_NOTE}"]
    assert "nothing clipped" not in _drawn_text(fig)


def test_restriction_in_range_values_use_fixed_view_without_scale_note(
    tmp_path,
    saved_figures,
):
    receipt = plot_restriction_comparison(_acrobot_runs(), tmp_path / "x.png")
    assert saved_figures[0].axes[0].get_ylim() == YLIM
    assert receipt["normalized_return_view"]["outside"]["means"] == 0
    assert not any(SCALE_NOTE in n or OFFSCALE_NOTE in n for n in receipt["footnotes"])


def test_restriction_out_of_unit_but_in_view_has_scale_note_only(tmp_path):
    values = _out_of_range_values()
    values["ftl"]["normalized_return"][:, 2] = [-0.3, -0.2, -0.1, -0.2, 0.1]
    receipt = plot_restriction_comparison(_acrobot_runs(values), tmp_path / "x.png")
    assert receipt["normalized_return_view"]["outside"]["band_edges"] == 0
    assert SCALE_NOTE in receipt["footnotes"]
    assert not any(OFFSCALE_NOTE in n for n in receipt["footnotes"])


def test_restriction_footer_is_concise(tmp_path, saved_figures):
    receipt = plot_restriction_comparison(_acrobot_runs(), tmp_path / "x.png")
    notes = receipt["footnotes"]
    assert notes == [
        SETUP_NOTE,
        "Mask: theta1_dot, theta2_dot zeroed.",
    ]
    lines = _footer_lines(saved_figures[0])
    assert len(lines) <= 4
    assert all(len(line) <= 120 for line in lines)
    drawn = _drawn_text(saved_figures[0])
    for token in ("sha", "hash", "inventory", "checkpoint", "padding", "paired"):
        assert token not in drawn.lower()


def test_restriction_footer_mask_from_description_without_task(tmp_path):
    runs = _with_restriction(_comparison_runs(), ZERO_DESCRIPTOR)
    notes = plot_restriction_comparison(runs, tmp_path / "x.png")["footnotes"]
    assert notes[1] == "Mask: theta1_dot, theta2_dot zeroed."


def test_restriction_footer_names_must_match_description(tmp_path):
    runs = _with_restriction(
        _comparison_runs(),
        ZERO_DESCRIPTOR,
        feature_names=list(reversed(ACROBOT_NAMES)),
    )
    with pytest.raises(ValueError, match="feature names"):
        plot_restriction_comparison(runs, tmp_path / "x.png")


def test_restriction_footer_random_bins_k_and_seed(tmp_path, saved_figures):
    runs = _with_restriction(_comparison_runs(), RANDOM_DESCRIPTOR, env="Taxi-v3")
    receipt = plot_restriction_comparison(runs, tmp_path / "x.png")
    assert receipt["footnotes"][1] == (
        "Random bins: K=120 over 400 reachable states, partition seed 20261003."
    )
    assert receipt["observation"]["kind"] == "random_bins"
    assert len(_footer_lines(saved_figures[0])) <= 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("env_name", "CliffWalking-v0"),
        ("restriction_id", "taxi_random_bins80_seed20261003"),
        ("n_bins", 80),
    ],
)
def test_restriction_random_bins_descriptor_cross_checked(tmp_path, field, value):
    descriptor = copy.deepcopy(RANDOM_DESCRIPTOR)
    descriptor["partition"][field] = value
    runs = _with_restriction(_comparison_runs(), descriptor, env="Taxi-v3")
    with pytest.raises(ValueError, match="partition"):
        plot_restriction_comparison(runs, tmp_path / "x.png")


def test_restriction_footer_categorical_alias(tmp_path):
    runs = _with_restriction(
        _comparison_runs(),
        ALIAS_DESCRIPTOR,
        env="FrozenLake-v1",
    )
    notes = plot_restriction_comparison(runs, tmp_path / "x.png")["footnotes"]
    assert notes[1] == (
        "Aliased bins: column within pairs {2k, 2k+1} hidden; 8 reachable classes."
    )


def test_restriction_footer_quantized_map(tmp_path):
    runs = _with_restriction(
        _comparison_runs(),
        QUANTIZED_DESCRIPTOR,
        env="MountainCar-v0",
    )
    notes = plot_restriction_comparison(runs, tmp_path / "x.png")["footnotes"]
    assert notes[1] == (
        "Quantized: position step 0.05; velocity in {-0.035, 0, 0.035} "
        "(dead zone 0.015)."
    )


def _with_grid(runs, rid):
    """Runs bound to a frozen grid: its record and the record's digest."""
    from imitation.experiments.agnostic import pilot, restriction, tasks

    descriptor = json.loads(json.dumps(restriction.describe_restriction(rid)))
    runs = _with_restriction(
        runs,
        descriptor,
        env=tasks.RANDOM_GRID_SPECS[rid].env_name,
    )
    for run in runs:
        run["config"]["restriction_sha256"] = pilot.config_digest(descriptor)
    return runs


@pytest.mark.parametrize(
    "env,k,expected",
    [
        (
            "Acrobot-v1",
            32,
            "Random-offset bins: K=32 per axis (33 cells per scalar axis, 32 "
            "arcs per angle); seed 20261003.",
        ),
        (
            "LunarLander-v2",
            8,
            "Random-offset bins: K=8 per axis (9 cells per scalar axis); seed "
            "20261003; leg contacts unchanged.",
        ),
    ],
)
def test_restriction_footer_random_grid_k_and_seed(
    tmp_path,
    saved_figures,
    env,
    k,
    expected,
):
    from imitation.experiments.agnostic import tasks

    rid = tasks.random_grid_id(env, k, tasks.RANDOM_GRID_SEED)
    values = _offscale_values()
    runs = _with_grid(_comparison_runs(values), rid)
    original = copy.deepcopy(runs)
    receipt = plot_restriction_comparison(runs, tmp_path / "x.png")
    assert runs == original
    assert receipt["footnotes"][1] == expected
    assert receipt["observation"] == {"kind": "random_grid", "text": expected}
    (fig,) = saved_figures
    assert fig.axes[0].get_ylim() == YLIM
    (line,) = _panel_lines(fig.axes[0], "ftl")
    np.testing.assert_allclose(
        line.get_ydata(),
        [
            _seed_mean_and_sem(values["ftl"]["normalized_return"][:, i])[0]
            for i in range(len(GRID))
        ],
        rtol=0,
        atol=1e-12,
    )
    assert f"{SCALE_NOTE} {OFFSCALE_NOTE}" in receipt["footnotes"]
    lines = _footer_lines(fig)
    assert len(lines) <= 4 and all(len(line) <= 120 for line in lines)
    drawn = _drawn_text(fig).lower()
    for token in ("sha", "hash", "descriptor", "offset 0x"):
        assert token not in drawn


def _grid_notes(runs):
    inside = {"normalized_return": {"ftl": [_point([0.2, 0.4], 0.2, 0.4)]}}
    return restriction_comparison_footnotes(runs, inside)["footnotes"]


def test_restriction_footer_accepts_every_frozen_grid():
    from imitation.experiments.agnostic import tasks

    assert len(tasks.RANDOM_GRID_SPECS) == 32
    for rid, spec in tasks.RANDOM_GRID_SPECS.items():
        note = _grid_notes(_with_grid(_comparison_runs(), rid))[1]
        assert note.startswith(f"Random-offset bins: K={spec.n_bins} per axis (")
        assert f"; seed {spec.seed}" in note


def _tamper_axis(field):
    def edit(desc):
        axis = next(a for a in desc["axes"] if a.get(field))
        axis[field][0] = axis[field][0] + 1e-3

    return edit


def _grid_tampers():
    from imitation.experiments.agnostic import random_grid, tasks

    other = tasks.random_grid_id("CartPole-v1", 16, tasks.RANDOM_GRID_SEED)
    return {
        "edges": _tamper_axis("edges"),
        "centers": _tamper_axis("centers"),
        "axis_kind": lambda d: d["axes"][0].update(kind="binary"),
        "axis_dropped": lambda d: d["axes"].pop(),
        "axis_key": lambda d: d["axes"][0].update(extra=1),
        "count": lambda d: d.update(n_bins=16),
        "seed": lambda d: d.update(seed=d["seed"] + 1),
        "generator": lambda d: d.update(generator="numpy.default_rng"),
        "offsets": lambda d: d["offsets"].reverse(),
        "box_bits": lambda d: d["box_low_bits"].reverse(),
        "nominal_cells": lambda d: d.update(nominal_cells=d["nominal_cells"] + 1),
        "zero_sha": lambda d: d.update(descriptor_sha256="0" * 64),
        "other_grid_sha": lambda d: d.update(
            descriptor_sha256=random_grid.FROZEN_GRID_SHA256[other],
        ),
        "extra_key": lambda d: d.update(note="x"),
        "missing_key": lambda d: d.pop("description"),
        "count_type": lambda d: d.update(n_bins=float(d["n_bins"])),
    }


@pytest.mark.parametrize("name", sorted(_grid_tampers()))
def test_restriction_footer_refuses_tampered_grid(name):
    from imitation.experiments.agnostic import tasks

    rid = tasks.random_grid_id("CartPole-v1", 8, tasks.RANDOM_GRID_SEED)
    runs = _with_grid(_comparison_runs(), rid)
    desc = copy.deepcopy(runs[0]["config"]["restriction"])
    _grid_tampers()[name](desc)
    for run in runs:
        run["config"]["restriction"] = copy.deepcopy(desc)
    with pytest.raises(ValueError, match="random grid"):
        _grid_notes(runs)


def test_restriction_footer_refuses_grid_with_copied_sha_but_changed_body():
    from imitation.experiments.agnostic import pilot, tasks

    rid = tasks.random_grid_id("Acrobot-v1", 8, tasks.RANDOM_GRID_SEED)
    runs = _with_grid(_comparison_runs(), rid)
    desc = copy.deepcopy(runs[0]["config"]["restriction"])
    _tamper_axis("edges")(desc)
    for run in runs:
        # Authentic stored descriptor SHA kept; record digest recomputed.
        run["config"]["restriction"] = copy.deepcopy(desc)
        run["config"]["restriction_sha256"] = pilot.config_digest(desc)
    with pytest.raises(ValueError, match="random grid"):
        _grid_notes(runs)


@pytest.mark.parametrize("stored", ["0" * 64, "descriptor", "one_run"])
def test_restriction_footer_refuses_wrong_grid_record_digest(stored):
    from imitation.experiments.agnostic import tasks

    rid = tasks.random_grid_id("MountainCar-v0", 8, tasks.RANDOM_GRID_SEED)
    runs = _with_grid(_comparison_runs(), rid)
    desc = runs[0]["config"]["restriction"]
    if stored == "one_run":
        runs[-1]["config"]["restriction_sha256"] = "f" * 64
    else:
        value = desc["descriptor_sha256"] if stored == "descriptor" else stored
        for run in runs:
            run["config"]["restriction_sha256"] = value
    with pytest.raises(ValueError, match="random grid"):
        _grid_notes(runs)


def test_restriction_footer_grid_without_record_digest_still_bound():
    from imitation.experiments.agnostic import tasks

    rid = tasks.random_grid_id("LunarLander-v2", 8, tasks.RANDOM_GRID_SEED)
    runs = _with_grid(_comparison_runs(), rid)
    for run in runs:
        del run["config"]["restriction_sha256"]
    assert _grid_notes(runs)[1].startswith("Random-offset bins: K=8 per axis")
    for run in runs:
        run["config"]["restriction"]["descriptor_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="random grid"):
        _grid_notes(runs)


def test_restriction_unknown_descriptor_kind_is_refused(tmp_path):
    descriptor = dict(ZERO_DESCRIPTOR, kind="random_grid_v2", zeroed_indices=[])
    runs = _with_restriction(_comparison_runs(), descriptor)
    with pytest.raises(ValueError, match="unknown restriction descriptor kind"):
        plot_restriction_comparison(runs, tmp_path / "x.png")


def test_restriction_footer_identity_and_missing_descriptor(tmp_path):
    runs = _comparison_runs()
    for run in runs:
        run["restriction_id"] = "identity"
    notes = plot_restriction_comparison(runs, tmp_path / "x.png")["footnotes"]
    assert notes[1] == "Observation: full."
    notes = plot_restriction_comparison(_comparison_runs(), tmp_path / "y.png")[
        "footnotes"
    ]
    assert notes[1] == "Mask: mask_a (no recorded descriptor)."


def test_restriction_mixed_descriptors_rejected(tmp_path):
    runs = _acrobot_runs()
    runs[3]["config"]["restriction"]["zeroed_indices"] = [5]
    with pytest.raises(ValueError, match="restriction descriptor"):
        plot_restriction_comparison(runs, tmp_path / "x.png")


def test_restriction_ce_and_disagreement_axes_start_at_zero(tmp_path, saved_figures):
    values = _in_range_values()
    for method in METHODS:
        values[method]["disagreement_rate"][:] = 0.0
        values[method]["rollout_cross_entropy"][:] = 0.0
    receipt = plot_restriction_comparison(_acrobot_runs(values), tmp_path / "x.png")
    (fig,) = saved_figures
    for ax, metric in zip(fig.axes[1:], METRICS[1:]):
        lo, hi = ax.get_ylim()
        assert lo == 0.0 and hi > 0
        assert receipt["y_limits"][metric] == [lo, hi]
        assert np.all(_band_vertices(ax) == 0.0)
    assert len(receipt["footnotes"]) == 2


def test_restriction_cliff_column_pairs_footer(tmp_path, saved_figures):
    runs = _column_pair_runs(COLUMN_PAIR_FALLS)
    receipt = plot_restriction_comparison(runs, tmp_path / "x.png")
    notes = receipt["footnotes"]
    assert (
        "Cliff stall benchmark: R=-200 (normalized 0.911); random normalization "
        "unchanged." in notes
    )
    rows = sum(r.get("d_eval_size") is not None for run in runs for r in run["records"])
    assert (
        "Column-pair bins: no goal-reaching deterministic memoryless policy. "
        f"Goal success not recorded ({rows}/{rows} rows); not imputed." in notes
    )
    assert len(notes) <= 5
    lines = _footer_lines(saved_figures[0])
    assert len(lines) <= 5
    assert "exhaustive" not in _drawn_text(saved_figures[0])
    audit = receipt["column_pairs_audit"]
    assert audit == {
        "checked_rows": rows,
        "success_not_recorded_rows": rows,
        "episode_returns_on_stall_lattice": True,
        "scope": "deterministic memoryless policies, exhaustive search",
    }
    # The stall and expert lines stay drawn above the curves.
    ax = saved_figures[0].axes[0]
    for label in (STALL_LABEL, EXPERT_LABEL):
        (line,) = _labelled_lines(ax, label)
        assert line.get_zorder() == 3


def test_restriction_receipt_keeps_full_audit(tmp_path):
    caveats = ("Publisher caveat kept verbatim.",)
    runs = _cliff_runs(baselines=STALLED_BASELINES)
    receipt = plot_restriction_comparison(runs, tmp_path / "x.png", caveats=caveats)
    assert receipt["caveats"] == list(caveats)
    assert receipt["protocol"] == {"paired_cold_fits": False, "paired_starts": False}
    assert receipt["evaluation"] == {"episodes": 100, "deterministic": True}
    norm = receipt["normalization"]
    assert len(norm["references"]) == 20
    assert norm["display_override"]["original_stored_lower_returns"] == [-200.0]
    assert norm["stall_benchmark"]["per_run"]
    assert any(
        n.startswith("Stored stall-referenced values redisplayed on measured random")
        for n in receipt["footnotes"]
    )


def test_restriction_footer_text_lengths_quantified(tmp_path):
    lengths = {}
    cases = {
        "acrobot": _acrobot_runs(_offscale_values()),
        "cliff_pairs": _column_pair_runs(COLUMN_PAIR_FALLS),
    }
    for name, runs in cases.items():
        notes = plot_restriction_comparison(runs, tmp_path / f"{name}.png")[
            "footnotes"
        ]
        lengths[name] = (len(notes), sum(len(n) for n in notes))
    assert lengths["acrobot"][0] == 3 and lengths["acrobot"][1] < 260
    assert lengths["cliff_pairs"][0] <= 5 and lengths["cliff_pairs"][1] < 480
