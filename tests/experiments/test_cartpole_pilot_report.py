"""Public behavior of the CartPole pilot report: offline BC is a flat reference.

Synthetic records only; the real pilot root is never read here.
"""

import importlib.util
import pathlib

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ANALYZER = REPO_ROOT / "experiments" / "agnostic" / "analyze_cartpole_pilot.py"

_spec = importlib.util.spec_from_file_location("cartpole_pilot_report", ANALYZER)
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)

CAP = 500
N_EPISODES = 4
N_ROUNDS = 30
POOL = "pool-digest"
BASELINES = {"expert_return": 500.0, "random_return": 20.0}


def _norm(mean):
    return round((mean - 20.0) / 480.0, 6)


def _record(labels, *, returns=None, ce=0.3, dis=0.1, ece=0.2, digest=None):
    returns = [float(CAP)] * N_EPISODES if returns is None else returns
    mean = sum(returns) / len(returns)
    return {
        "round": labels,
        "n_observations": labels,
        "episode_returns": returns,
        "normalized_return": _norm(mean),
        "rollout_cross_entropy": ce,
        "disagreement_rate": dis,
        "expert_rollout_cross_entropy": ece,
        "d_eval_size": CAP * N_EPISODES,
        "prefix_pairs_sha256": digest,
        "checkpoint": f"checkpoints/round-{labels:05d}.pt",
        "inner_es_stop_epoch": 20,
        "inner_es_fallback": None,
        "wall_seconds": {"fit": 1.5, "eval": 30.0},
    }


def _bc_run(records):
    return {"records": records, "data": {"baselines": BASELINES}}


def _prefix_run():
    """Historical layout: one cold fit per prefix, the last on the whole pool."""
    return _bc_run(
        [
            _record(1, returns=[100.0, 200.0, 300.0, 400.0], ce=0.9, digest="p1"),
            _record(10, ce=0.7, digest="p10"),
            _record(20, ce=0.5, digest="p20"),
            _record(
                N_ROUNDS,
                returns=[480.0, 500.0, 500.0, 500.0],
                ce=0.25,
                dis=0.04,
                ece=0.21,
                digest=POOL,
            ),
        ]
    )


def _reference(rec):
    return pilot.bc_reference("run-bc-identity", rec, N_EPISODES, CAP, N_ROUNDS, POOL)


def test_bc_reference_uses_only_the_full_budget_record():
    ref = _reference(_prefix_run())
    assert ref["labels"] == N_ROUNDS
    # Stored metrics pass through unchanged; no prefix value is mixed in.
    assert ref["normalized_return"] == _norm(495.0)
    assert ref["rollout_cross_entropy"] == 0.25
    assert ref["disagreement_rate"] == 0.04
    assert ref["expert_rollout_cross_entropy"] == 0.21
    assert ref["mean_return"] == 495.0
    assert ref["episodes_at_cap"] == 3
    assert ref["pool_pairs_sha256"] == POOL
    assert (ref["reference_fit_seconds"], ref["reference_eval_seconds"]) == (1.5, 30.0)


def test_bc_reference_refuses_without_a_full_budget_endpoint():
    rec = _prefix_run()
    rec["records"].pop()
    with pytest.raises(pilot.PilotInputError, match="full budget"):
        _reference(rec)


def test_bc_reference_refuses_a_duplicate_endpoint():
    rec = _prefix_run()
    rec["records"].append(_record(N_ROUNDS, digest=POOL))
    with pytest.raises(pilot.PilotInputError, match="found 2"):
        _reference(rec)


def test_bc_reference_refuses_an_endpoint_not_on_the_whole_pool():
    rec = _prefix_run()
    rec["records"][-1]["prefix_pairs_sha256"] = "another-digest"
    with pytest.raises(pilot.PilotInputError, match="whole offline pool"):
        _reference(rec)


def test_bc_reference_refuses_an_unevaluated_endpoint():
    rec = _prefix_run()
    rec["records"][-1]["episode_returns"] = None
    with pytest.raises(pilot.PilotInputError, match="no saved evaluation"):
        _reference(rec)


def test_bc_reference_keeps_the_normalized_return_check():
    rec = _prefix_run()
    rec["records"][-1]["normalized_return"] = 1.0  # raw mean 495 is not 1.0
    with pytest.raises(pilot.PilotInputError, match="normalized_return"):
        _reference(rec)


def _online_points(ends):
    return {
        j: [{"labels": b} for b in [0, 1, 10, 20, 30] if b <= end]
        for j, end in zip(pilot.ONLINE_IDS, ends)
    }


def test_matched_budget_uses_the_four_online_curves_only():
    points = _online_points([20, 30, 30, 30])
    # Offline BC labels are not part of the rule, even when present.
    points.update({j: [{"labels": N_ROUNDS}] for j in pilot.BC_IDS})
    assert pilot.online_matched_budget(points) == 20


def test_matched_budget_refuses_without_a_shared_online_budget():
    points = {j: [{"labels": b}] for j, b in zip(pilot.ONLINE_IDS, [1, 10, 20, 30])}
    with pytest.raises(pilot.PilotInputError, match="four online runs"):
        pilot.online_matched_budget(points)


def _curve_points(job_id, ends_at):
    shift = 0.1 if job_id.endswith("cart_position_zero") else 0.0
    return [
        pilot.evaluation_point(
            job_id, _record(b, ce=0.8 - 0.01 * b + shift), N_EPISODES, CAP, BASELINES
        )
        for b in [0, 1, 10, 20, 30]
        if b <= ends_at
    ]


def test_figure_draws_offline_bc_flat_and_online_curves_by_condition():
    points = {j: _curve_points(j, 20 if "ftl" in j else 30) for j in pilot.ONLINE_IDS}
    refs = {
        "run-bc-identity": _reference(_prefix_run()),
        "run-bc-cart_position_zero": {
            **_reference(_prefix_run()),
            "rollout_cross_entropy": 0.55,
            "disagreement_rate": 0.28,
        },
    }
    rows = {
        j: {"controller_state": "timed_out" if "ftl" in j else "complete"}
        for j in pilot.ONLINE_IDS
    }
    fig = pilot.curves_figure(
        points, refs, rows, BASELINES, CAP, True, N_EPISODES, N_ROUNDS
    )
    keys = [
        "normalized_return",
        "rollout_cross_entropy",
        "disagreement_rate",
        "expert_rollout_cross_entropy",
    ]
    try:
        for ax, key in zip(fig.axes, keys):
            bc = [
                ln
                for ln in ax.get_lines()
                if mcolors.same_color(ln.get_color(), pilot.COLOR["bc"])
            ]
            # Exactly one flat line per condition, at the stored reference value.
            assert len(bc) == 2
            for ln in bc:
                ys = list(ln.get_ydata())
                assert len(set(ys)) == 1
                assert ln.get_marker() in ("None", None, "")
            drawn = {ln.get_linestyle(): ln.get_ydata()[0] for ln in bc}
            assert drawn == {
                "--": refs["run-bc-identity"][key],
                "-": refs["run-bc-cart_position_zero"][key],
            }
            for m in pilot.ONLINE:
                lines = [
                    ln
                    for ln in ax.get_lines()
                    if mcolors.same_color(ln.get_color(), pilot.COLOR[m])
                    and ln.get_linestyle() != "None"
                ]
                # Full observation dashed, cart position hidden solid.
                assert sorted(ln.get_linestyle() for ln in lines) == ["-", "--"]
                for ln, r in zip(lines, pilot.RESTRICTIONS):
                    assert list(ln.get_xdata()) == [
                        p["labels"] for p in points[f"run-{m}-{r}"]
                    ]
    finally:
        plt.close(fig)
