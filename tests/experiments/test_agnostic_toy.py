"""Behavior tests for the exact agnostic DAgger toy experiment.

Reference quantities here come from the contract's scalar recurrence and closed
forms, not from the production matrix dynamic program.
"""

import datetime as dt
import hashlib
import itertools
import json
import math
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from imitation.experiments.agnostic import toy

UTC = dt.timezone.utc
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def recurrence(horizon, alpha, kappa, q, policy_id):
    """Contract recurrence: returns (total cost, list of r_t before each action)."""
    a_nominal, a_recovery = policy_id >> 1, policy_id & 1
    e = alpha if a_nominal == 0 else 1.0 - alpha
    w = 1.0 if a_recovery != q else 0.0
    r, total, rs = 0.0, 0.0, []
    for _ in range(horizon):
        rs.append(r)
        total += e * (1.0 - r) + w * r
        r = kappa * e * (1.0 - r) + w * r
    return total, rs


def closed_optimum(horizon, alpha, kappa):
    if kappa == 0:
        return horizon * alpha
    assert kappa == 1
    return alpha * horizon / (1 + alpha) + alpha**2 / (1 + alpha) ** 2 * (
        1 - (-alpha) ** horizon
    )


def strict_json_load(path):
    def reject(token):
        raise AssertionError("non-finite JSON token {}".format(token))

    return json.loads(pathlib.Path(path).read_text(), parse_constant=reject)


class FakeClock:
    """Injectable clock advancing a fixed step per call."""

    def __init__(self, start, step_seconds):
        self.now = start
        self.step = dt.timedelta(seconds=step_seconds)
        self.last = None

    def __call__(self):
        self.last = self.now
        self.now = self.now + self.step
        return self.last


START = dt.datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


def small_config(**overrides):
    kwargs = dict(
        seeds=(0,),
        horizons=(4,),
        alphas=(0.0, 0.3),
        kappas=(0.0, 1.0),
        qs=(0, 1),
        orientations=(0, 1),
        budget=12,
        batch=4,
        checkpoints=(4, 8, 12),
        deadline=dt.datetime(2026, 10, 1, tzinfo=UTC),
    )
    kwargs.update(overrides)
    return toy.ToyConfig(**kwargs)


# Exact evaluator versus independent closed forms.


@pytest.mark.parametrize("horizon", [1, 2, 5, 16])
@pytest.mark.parametrize("alpha", [0.0, 0.02, 0.1, 0.3, 0.49])
@pytest.mark.parametrize("kappa", [0.0, 0.37, 1.0])
def test_class_costs_match_contract_recurrence(horizon, alpha, kappa):
    for q, u in itertools.product((0, 1), (0, 1)):
        mdp = toy.ToyMDP(horizon, alpha, kappa, q, u)
        for p in range(4):
            expected, rs = recurrence(horizon, alpha, kappa, q, p)
            evaluation = toy.evaluate_policy(mdp, p)
            assert evaluation.cost == pytest.approx(expected, abs=1e-12)
            assert evaluation.recovery_mean == pytest.approx(np.mean(rs), abs=1e-12)
            np.testing.assert_allclose(evaluation.occupancy.sum(axis=1), 1.0)


@pytest.mark.parametrize("horizon", [1, 2, 3, 16, 64])
@pytest.mark.parametrize("alpha", [0.0, 0.02, 0.1, 0.3, 0.49])
@pytest.mark.parametrize("kappa", [0.0, 1.0])
def test_class_optimum_matches_closed_forms(horizon, alpha, kappa):
    for q, u in itertools.product((0, 1), (0, 1)):
        mdp = toy.ToyMDP(horizon, alpha, kappa, q, u)
        costs = toy.class_costs(mdp)
        assert min(costs) == pytest.approx(closed_optimum(horizon, alpha, kappa))
        if kappa == 1 and alpha > 0 and horizon > 1:
            # Unique optimum: majority nominal action and correct recovery action.
            assert int(np.argmin(costs)) == q
            assert sorted(costs)[1] > min(costs) + 1e-12
    if horizon == 1:
        assert closed_optimum(1, alpha, kappa) == pytest.approx(alpha)
    if alpha == 0:
        assert closed_optimum(horizon, alpha, kappa) == pytest.approx(0.0)


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])
def test_expert_has_zero_cost_and_never_reaches_recovery(alpha, kappa):
    for q, u in itertools.product((0, 1), (0, 1)):
        mdp = toy.ToyMDP(7, alpha, kappa, q, u)
        evaluation = toy.evaluate_expert(mdp)
        assert evaluation.cost == 0.0
        assert evaluation.recovery_mean == 0.0
        # Expert-occupancy floor for the class is exactly alpha.
        assert toy.expert_occupancy_floor(mdp) == pytest.approx(alpha)


def test_alpha_zero_removes_n1_from_domain():
    mdp = toy.ToyMDP(5, 0.0, 1.0, 1, 1)
    assert mdp.states == ("N0", "R")
    assert toy.ToyMDP(5, 0.2, 1.0, 1, 1).states == ("N0", "N1", "R")
    with pytest.raises(ValueError):
        mdp.expert_action("N1")
    with pytest.raises(ValueError):
        toy.evaluate_actions(mdp, {"N0": 1, "N1": 0, "R": 0})
    assert toy.evaluate_policy(mdp, 0).occupancy.shape == (5, 2)
    # Sampling from the realizable control never produces an N1 label.
    tapes = toy.episode_tapes(3, 0, 2000, 5)
    for behavior in (toy.EXPERT, 2, 3):
        out = toy.rollout(mdp, behavior, tapes, toy.ExpertOracle(mdp))
        nominal = out.modes == 0
        assert np.all(out.labels[nominal] == mdp.orientation)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(horizon=0, alpha=0.1, kappa=1.0, q=0, orientation=0),
        dict(horizon=3, alpha=0.5, kappa=1.0, q=0, orientation=0),
        dict(horizon=3, alpha=float("nan"), kappa=1.0, q=0, orientation=0),
        dict(horizon=3, alpha=0.1, kappa=1.5, q=0, orientation=0),
        dict(horizon=3, alpha=0.1, kappa=float("inf"), q=0, orientation=0),
        dict(horizon=3, alpha=0.1, kappa=1.0, q=2, orientation=0),
        dict(horizon=3, alpha=0.1, kappa=1.0, q=0, orientation=-1),
    ],
)
def test_invalid_mdp_rejected(kwargs):
    with pytest.raises(ValueError):
        toy.ToyMDP(**kwargs)


# Exact ERM.


def test_erm_empty_data_gives_initial_policy_zero():
    for u in (0, 1):
        assert toy.fit_erm(np.zeros((2, 2), int), u) == 0


@pytest.mark.parametrize("q", [0, 1])
@pytest.mark.parametrize("u", [0, 1])
def test_single_recovery_label_forces_correct_recovery_action(q, u):
    label = q ^ u
    for n0, n1 in itertools.product(range(4), range(4)):
        counts = np.zeros((2, 2), int)
        counts[0, 0], counts[0, 1] = n0, n1
        counts[1, label] = 1
        p = toy.fit_erm(counts, u)
        assert (p & 1) == q
        assert toy.physical_actions(p, u)[1] == label


def test_erm_tie_order_is_respected():
    # With no recovery data both recovery actions tie.
    counts = np.array([[3, 1], [0, 0]])
    assert toy.fit_erm(counts, 0) == 0
    assert toy.fit_erm(counts, 0, tie_order=(1, 0, 3, 2)) == 1
    with pytest.raises(ValueError):
        toy.fit_erm(counts, 0, tie_order=(0, 0, 1, 2))


# Sampler.


@pytest.mark.parametrize("behavior", [2, "expert"])
def test_sampler_occupancy_matches_analytic_distribution(behavior):
    horizon, alpha, kappa, q, u = 4, 0.3, 0.6, 1, 1
    mdp = toy.ToyMDP(horizon, alpha, kappa, q, u)
    n = 120000
    tapes = toy.episode_tapes(11, 0, n, horizon)
    behavior = toy.EXPERT if behavior == "expert" else behavior
    out = toy.rollout(mdp, behavior, tapes, toy.ExpertOracle(mdp))
    assert out.episodes == n and out.env_steps == n * horizon
    state = np.where(out.modes == 1, 2, np.where(out.labels == u, 0, 1))
    if behavior == toy.EXPERT:
        rs = [0.0] * horizon
    else:
        _, rs = recurrence(horizon, alpha, kappa, q, behavior)

    def check(observed, count, p):
        # Binomial standard error; 6 SE per entry is far outside fixed-seed noise.
        se = math.sqrt(max(p * (1 - p), 1e-12) / count)
        assert abs(observed / count - p) <= 6 * se + 1e-9

    marginal = np.zeros(3)
    for t in range(horizon):
        at_t = tapes.select == t
        check(at_t.sum(), n, 1.0 / horizon)
        r = rs[t]
        d_t = np.array([(1 - r) * (1 - alpha), (1 - r) * alpha, r])
        marginal += d_t / horizon
        for s in range(3):
            check(np.sum(state[at_t] == s), at_t.sum(), d_t[s])
    for s in range(3):
        check(np.sum(state == s), n, marginal[s])


# Runner pathwise controls.


def test_kappa_zero_ftl_and_bc_iid_identical_pathwise():
    mdp = toy.ToyMDP(8, 0.3, 0.0, 1, 0)
    checkpoints = tuple(range(4, 65, 4))
    cell = toy.run_cell(mdp, seed=5, budget=64, batch=4, checkpoints=checkpoints)
    assert cell["status"] == "complete"
    for c in cell["checkpoints"]:
        assert c["ftl"]["retained_counts"] == c["bc_iid"]["retained_counts"]
        assert c["ftl"]["post_update_policy_id"] == c["bc_iid"]["policy_id"]
        assert c["ftl"]["retained_counts"][1] == [0, 0]


def test_orientation_relabel_is_pathwise_identical():
    kwargs = dict(seed=2, budget=40, batch=3, checkpoints=(3, 9, 21, 39, 40))
    a = toy.run_cell(toy.ToyMDP(6, 0.3, 1.0, 1, 0), **kwargs)
    b = toy.run_cell(toy.ToyMDP(6, 0.3, 1.0, 1, 1), **kwargs)
    for ca, cb in zip(a["checkpoints"], b["checkpoints"]):
        for arm in ("ftl", "bc_iid", "bc_fixed"):
            counts_a = np.array(ca[arm]["retained_counts"])
            counts_b = np.array(cb[arm]["retained_counts"])
            np.testing.assert_array_equal(counts_b, counts_a[:, ::-1])
        assert ca["ftl"]["post_update_policy_id"] == cb["ftl"]["post_update_policy_id"]
        assert ca["ftl"]["post_update_cost"] == cb["ftl"]["post_update_cost"]
        assert ca["ftl"]["behavior_mixture_cost"] == cb["ftl"]["behavior_mixture_cost"]
        assert ca["bc_iid"]["policy_id"] == cb["bc_iid"]["policy_id"]


def test_q_symmetry_holds_with_relabeled_ties():
    kwargs = dict(seed=4, budget=48, batch=4, checkpoints=tuple(range(4, 49, 4)))
    a = toy.run_cell(toy.ToyMDP(6, 0.2, 1.0, 0, 1), **kwargs)
    b = toy.run_cell(toy.ToyMDP(6, 0.2, 1.0, 1, 1), tie_order=(1, 0, 3, 2), **kwargs)
    for ca, cb in zip(a["checkpoints"], b["checkpoints"]):
        assert cb["ftl"]["post_update_policy_id"] == (
            ca["ftl"]["post_update_policy_id"] ^ 1
        )
        assert cb["ftl"]["post_update_cost"] == pytest.approx(
            ca["ftl"]["post_update_cost"], abs=1e-12
        )
        assert cb["bc_iid"]["policy_id"] == ca["bc_iid"]["policy_id"] ^ 1
        counts_a = np.array(ca["ftl"]["retained_counts"])
        counts_b = np.array(cb["ftl"]["retained_counts"])
        np.testing.assert_array_equal(counts_b[0], counts_a[0])
        np.testing.assert_array_equal(counts_b[1], counts_a[1, ::-1])
    # Same q and tapes, canonical ties: behavior and hence retained data differ.
    c = toy.run_cell(toy.ToyMDP(6, 0.2, 1.0, 1, 1), **kwargs)
    assert any(
        cc["ftl"]["retained_counts"] != cb["ftl"]["retained_counts"]
        for cc, cb in zip(c["checkpoints"], b["checkpoints"])
    )


def test_ftl_recovery_arm_correct_once_any_recovery_label_retained():
    q = 1  # canonical tie picks the wrong recovery action for q = 1
    mdp = toy.ToyMDP(4, 0.3, 1.0, q, 0)
    cell = toy.run_cell(mdp, seed=1, budget=30, batch=1, checkpoints=range(1, 31))
    seen_recovery = False
    for c in cell["checkpoints"]:
        if sum(c["ftl"]["retained_counts"][1]) > 0:
            seen_recovery = True
            assert c["ftl"]["post_update_policy_id"] & 1 == q
        else:
            assert c["ftl"]["post_update_policy_id"] & 1 == 0
    assert seen_recovery


@pytest.mark.parametrize("horizon", [1, 3])
@pytest.mark.parametrize("mode", ["deferred", "full_trajectory"])
def test_counters_match_actual_rollouts_with_partial_batch(monkeypatch, horizon, mode):
    calls = {"expert": 0, "learner": 0, "steps": 0}
    real = toy.rollout

    def spy(mdp, behavior, tapes, *args):
        n = tapes.select.shape[0]
        calls["expert" if behavior == toy.EXPERT else "learner"] += n
        calls["steps"] += n * tapes.nominal.shape[1]
        return real(mdp, behavior, tapes, *args)

    monkeypatch.setattr(toy, "rollout", spy)
    mdp = toy.ToyMDP(horizon, 0.3, 1.0, 1, 0)
    cell = toy.run_cell(
        mdp, seed=0, budget=5, batch=2, checkpoints=(2, 4, 5), annotation_mode=mode
    )
    assert calls["expert"] == 5 and calls["learner"] == 5
    assert calls["steps"] == 2 * 5 * horizon
    assert [c["rounds"] for c in cell["checkpoints"]] == [1, 2, 3]
    for c in cell["checkpoints"]:
        b = c["retained_labels"]
        ftl = c["ftl"]["counters"]
        assert ftl["episodes"] == b and ftl["env_steps"] == b * horizon
        assert ftl["retained_labels"] == b == sum(map(sum, c["ftl"]["retained_counts"]))
        assert ftl["expert_action_entries"] == (
            b if mode == "deferred" else b * horizon
        )
        assert ftl["fits"] == c["rounds"]
        shared = c["shared_expert_acquisition"]
        assert shared == {
            "episodes": b,
            "env_steps": b * horizon,
            "expert_action_entries": b * horizon,
            "retained_labels": b,
        }
        for arm in ("bc_iid", "bc_fixed"):
            counters = c[arm]["counters"]
            assert counters["logical_episodes"] == b
            assert counters["logical_env_steps"] == b * horizon
            assert counters["logical_expert_action_entries"] == b * horizon
            assert counters["retained_labels"] == b
        assert c["bc_iid"]["counters"]["fits"] == c["rounds"]
    assert [c["bc_fixed"]["counters"]["fits"] for c in cell["checkpoints"]] == [1, 2, 3]


def test_bc_fixed_matches_bc_iid_at_every_checkpoint():
    for q, u, kappa in itertools.product((0, 1), (0, 1), (0.0, 1.0)):
        mdp = toy.ToyMDP(5, 0.3, kappa, q, u)
        cell = toy.run_cell(mdp, seed=9, budget=23, batch=3, checkpoints=(3, 6, 21, 23))
        for c in cell["checkpoints"]:
            assert c["bc_fixed"]["policy_id"] == c["bc_iid"]["policy_id"]
            assert c["bc_fixed"]["retained_counts"] == c["bc_iid"]["retained_counts"]
            assert c["bc_fixed"]["cost"] == c["bc_iid"]["cost"]


def test_mixture_regret_identity_on_policy_changing_sequence():
    horizon, alpha, kappa, q = 6, 0.3, 1.0, 1
    mdp = toy.ToyMDP(horizon, alpha, kappa, q, 0)
    cell = toy.run_cell(mdp, seed=3, budget=19, batch=2, checkpoints=(2, 8, 18, 19))
    j = [recurrence(horizon, alpha, kappa, q, p)[0] for p in range(4)]
    rbar = [np.mean(recurrence(horizon, alpha, kappa, q, p)[1]) for p in range(4)]
    e = [alpha, alpha, 1 - alpha, 1 - alpha]
    w = [float((p & 1) != q) for p in range(4)]

    def f(b, p):
        return (1 - rbar[b]) * e[p] + rbar[b] * w[p]

    changing = False
    for c in cell["checkpoints"]:
        ftl = c["ftl"]
        counts = ftl["behavior_policy_id_counts"]
        n = c["rounds"]
        assert sum(counts) == n  # uniform weight per round, including partial last
        changing = changing or sum(1 for x in counts if x) >= 2
        totals = [sum(counts[b] * f(b, p) for b in range(4)) for p in range(4)]
        a_n = min(totals) / n
        reg = sum(counts[b] * f(b, b) for b in range(4)) - min(totals)
        mix = sum(counts[b] * j[b] for b in range(4)) / n
        assert ftl["approximation_term"] == pytest.approx(a_n, abs=1e-12)
        assert ftl["regret"] == pytest.approx(reg, abs=1e-12)
        assert ftl["behavior_mixture_cost"] == pytest.approx(mix, abs=1e-12)
        assert mix == pytest.approx(horizon * (a_n + reg / n), abs=1e-12)
        post = ftl["post_update_policy_id"]
        assert ftl["post_update_cost"] == pytest.approx(j[post], abs=1e-12)
        assert ftl["post_update_class_excess"] == pytest.approx(j[post] - min(j))
        assert ftl["post_update_onpolicy_disagreement"] == pytest.approx(
            j[post] / horizon
        )
    assert changing


# Experiment driver: replay, deadline, failure handling.


def test_deterministic_replay_and_independent_evaluation():
    config = small_config()
    first = toy.run_experiment(config, clock=FakeClock(START, 0))
    second = toy.run_experiment(config, clock=FakeClock(START, 0))
    assert first["status"] == "complete"
    assert first["cells"] == second["cells"]
    assert len(first["cells"]) == 16
    for cell in first["cells"]:
        spec = cell["cell"]
        mdp = toy.ToyMDP(
            spec["horizon"],
            spec["alpha"],
            spec["kappa"],
            spec["q"],
            spec["orientation"],
        )
        last = cell["checkpoints"][-1]
        assert last["retained_labels"] == config.budget
        ftl_eval = toy.evaluate_policy(mdp, last["ftl"]["post_update_policy_id"])
        bc_eval = toy.evaluate_policy(mdp, last["bc_iid"]["policy_id"])
        assert last["ftl"]["post_update_cost"] == ftl_eval.cost
        assert last["bc_iid"]["cost"] == bc_eval.cost
        assert cell["exact"]["class_optimum_cost"] == min(toy.class_costs(mdp))


def test_deadline_stops_before_rounds_and_cells(monkeypatch):
    clock = FakeClock(START, 1)
    deadline = START + dt.timedelta(seconds=43)
    rollout_times = []
    real = toy.rollout

    def spy(mdp, behavior, tapes, *args):
        rollout_times.append(clock.last)
        return real(mdp, behavior, tapes, *args)

    monkeypatch.setattr(toy, "rollout", spy)
    result = toy.run_experiment(small_config(deadline=deadline), clock=clock)
    assert result["status"] == "partial_deadline"
    assert rollout_times and all(t < deadline for t in rollout_times)
    inventory = result["inventory"]
    assert 0 < len(inventory["cells_complete"]) < inventory["cells_planned"]
    statuses = [cell["status"] for cell in result["cells"]]
    assert statuses[-1] == "partial_deadline"
    assert all(s == "complete" for s in statuses[:-1])
    partial = result["cells"][-1]
    assert partial["rounds_completed"] < 3
    assert len(partial["checkpoints"]) == partial["rounds_completed"]
    assert inventory["partial_cell"]["rounds_completed"] == partial["rounds_completed"]

    rollout_times.clear()
    expired = toy.run_experiment(
        small_config(deadline=START), clock=FakeClock(START, 1)
    )
    assert expired["status"] == "partial_deadline"
    assert expired["cells"] == [] and rollout_times == []


def test_exception_preserves_partial_inventory(monkeypatch):
    real = toy.rollout
    count = {"n": 0}

    def flaky(mdp, behavior, tapes, *args):
        count["n"] += 1
        if count["n"] == 15:
            raise RuntimeError("injected failure")
        return real(mdp, behavior, tapes, *args)

    monkeypatch.setattr(toy, "rollout", flaky)
    result = toy.run_experiment(small_config(), clock=FakeClock(START, 0))
    assert result["status"] == "partial_error"
    assert "injected failure" in result["error"]
    # Two rollouts per round, three rounds per cell: two cells finish first.
    assert result["inventory"]["cells_complete"] == [0, 1]
    assert result["cells"][-1]["status"] == "partial_error"
    json.dumps(result, allow_nan=False)


def one_cell_config(**overrides):
    kwargs = dict(
        seeds=(0,),
        horizons=(4,),
        alphas=(0.3,),
        kappas=(1.0,),
        qs=(1,),
        orientations=(0,),
        budget=4,
        batch=4,
        checkpoints=(4,),
        deadline=dt.datetime(2026, 10, 1, tzinfo=UTC),
    )
    kwargs.update(overrides)
    return toy.ToyConfig(**kwargs)


def test_stop_between_checkpoints_preserves_latest_committed_round():
    calls = {"n": 0}

    def stop_on_fourth_call():
        calls["n"] += 1
        return calls["n"] == 4

    mdp = toy.ToyMDP(4, 0.3, 1.0, 1, 0)
    cell = toy.run_cell(
        mdp, 0, 20, 4, checkpoints=(20,), should_stop=stop_on_fourth_call
    )
    assert cell["status"] == "partial_deadline" and cell["rounds_completed"] == 3
    assert cell["checkpoints"] == [] and cell["in_flight"] is None
    latest = cell["latest_round"]
    assert (latest["rounds"], latest["retained_labels"]) == (3, 12)
    assert not latest["is_checkpoint"] and latest["bc_fixed"] is None
    assert cell["acquisition"]["ftl"]["episodes"] == 12
    assert cell["acquisition"]["shared_expert"]["env_steps"] == 12 * 4
    # Same paired tapes: identical to a run whose budget ends at 12 labels.
    reference = toy.run_cell(mdp, 0, 12, 4, checkpoints=(12,))["checkpoints"][-1]
    for key in ("ftl", "bc_iid", "shared_expert_acquisition"):
        assert latest[key] == reference[key]


def test_progress_is_published_at_round_boundaries():
    snapshots = []
    toy.run_experiment(
        one_cell_config(budget=12, checkpoints=(12,)),
        clock=FakeClock(START, 1),
        publish=lambda r: snapshots.append(json.loads(json.dumps(r))),
        progress_interval_seconds=0,
    )
    running = [
        s["cells"][0]["latest_round"]["rounds"]
        for s in snapshots
        if s["cells"] and s["cells"][0]["status"] == "running"
    ]
    assert running == [1, 2, 3]


def test_failed_expert_rollout_keeps_completed_learner_acquisition(monkeypatch):
    real = toy.rollout

    def expert_fails(mdp, behavior, tapes, *args):
        if behavior == toy.EXPERT:
            raise RuntimeError("expert acquisition failed")
        return real(mdp, behavior, tapes, *args)

    monkeypatch.setattr(toy, "rollout", expert_fails)
    result = toy.run_experiment(one_cell_config(), clock=FakeClock(START, 0))
    assert result["status"] == "partial_error"
    cell = result["cells"][0]
    assert cell["rounds_completed"] == 0 and cell["latest_round"] is None
    assert cell["acquisition"]["ftl"] == {
        "rollouts": 1,
        "episodes": 4,
        "env_steps": 16,
        "expert_action_entries": 4,
    }
    assert set(cell["acquisition"]["shared_expert"].values()) == {0}
    assert cell["in_flight"] == {
        "arm": "shared_expert",
        "round_index": 0,
        "episodes_launched": 4,
        "env_steps": None,
        "env_steps_upper_bound": 16,
        "expert_action_entries_observed": 0,
    }
    json.dumps(result, allow_nan=False)


def test_finishing_after_deadline_is_not_complete(monkeypatch):
    deadline = START + dt.timedelta(days=1)
    late = deadline + dt.timedelta(seconds=1)
    rollouts = []
    real = toy.rollout

    def spy(*args):
        rollouts.append(args[1])
        return real(*args)

    def clock_from(values):
        return lambda: values.pop(0) if values else late

    monkeypatch.setattr(toy, "rollout", spy)
    config = one_cell_config(
        horizons=(1,),
        alphas=(0.1,),
        budget=1,
        batch=1,
        checkpoints=(1,),
        deadline=deadline,
    )
    # Deadline passes during the only round: data kept, cap not claimed.
    result = toy.run_experiment(config, clock=clock_from([START, START, START]))
    assert result["status"] == "partial_deadline"
    assert result["cells"][0]["status"] == "overran_deadline"
    assert len(result["cells"][0]["checkpoints"]) == 1
    assert result["inventory"]["cells_overran_deadline"] == [0]
    assert result["inventory"]["cells_complete"] == []
    assert result["deadline_overshoot"]["seconds_past_deadline"] == 1.0
    assert len(rollouts) == 2  # no rollout after the observed expiration
    # Every cell in time, but the run itself ends late.
    result = toy.run_experiment(config, clock=clock_from([START] * 4))
    assert result["status"] == "partial_deadline"
    assert result["inventory"]["cells_complete"] == [0]
    assert result["ended_utc"] == late.isoformat()


def recording_oracle_class(instances):
    class RecordingOracle(toy.ExpertOracle):
        def __init__(self, mdp):
            super().__init__(mdp)
            self.sizes = []
            instances.append(self)

        def request(self, recovery, z):
            self.sizes.append(len(recovery))
            return super().request(recovery, z)

    return RecordingOracle


def test_rollout_requests_only_what_annotation_mode_requires():
    mdp = toy.ToyMDP(5, 0.3, 1.0, 1, 0)
    k = 7
    tapes = toy.episode_tapes(2, 0, k, 5)
    instances = []
    oracle_class = recording_oracle_class(instances)
    deferred = toy.rollout(mdp, 2, tapes, oracle_class(mdp), "deferred")
    full = toy.rollout(mdp, 2, tapes, oracle_class(mdp), "full_trajectory")
    expert = toy.rollout(mdp, toy.EXPERT, tapes, oracle_class(mdp))
    per_step = [int(np.sum(tapes.select == t)) for t in range(5)]
    assert instances[0].sizes == [n for n in per_step if n]
    assert instances[0].entries == k == len(deferred.labels)
    assert instances[1].sizes == [k] * 5 and instances[1].entries == k * 5
    assert instances[2].sizes == [k] * 5
    # Full-trajectory requests change cost accounting, not the retained data.
    np.testing.assert_array_equal(deferred.labels, full.labels)
    np.testing.assert_array_equal(deferred.modes, full.modes)
    assert deferred.env_steps == full.env_steps == expert.env_steps == k * 5


@pytest.mark.parametrize("mode", toy.ANNOTATION_MODES)
def test_reported_expert_entries_equal_oracle_requests(monkeypatch, mode):
    instances = []
    monkeypatch.setattr(toy, "ExpertOracle", recording_oracle_class(instances))
    mdp = toy.ToyMDP(3, 0.3, 1.0, 1, 0)
    cell = toy.run_cell(mdp, 0, 5, 2, checkpoints=(2, 4, 5), annotation_mode=mode)
    last = cell["checkpoints"][-1]
    ftl_entries = last["ftl"]["counters"]["expert_action_entries"]
    shared_entries = last["shared_expert_acquisition"]["expert_action_entries"]
    assert len(instances) == 2
    assert sorted(sum(o.sizes) for o in instances) == sorted(
        [ftl_entries, shared_entries]
    )
    assert ftl_entries == cell["acquisition"]["ftl"]["expert_action_entries"]
    assert ftl_entries == (5 if mode == "deferred" else 15)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(seeds=(0, 0)),
        dict(seeds=()),
        dict(alphas=(0.1, 0.1)),
        dict(kappas=(float("nan"),)),
        dict(budget=0),
        dict(batch=0),
        dict(checkpoints=(4, 8)),
        dict(checkpoints=(4, 6, 12)),
        dict(checkpoints=(8, 4, 12)),
        dict(annotation_mode="cached"),
        dict(deadline=dt.datetime(2026, 10, 1)),
        dict(deadline=dt.datetime(2026, 10, 8, tzinfo=UTC)),
    ],
)
def test_invalid_config_rejected_before_acquisition(monkeypatch, overrides):
    def forbidden(*args, **kwargs):
        raise AssertionError("acquisition attempted")

    monkeypatch.setattr(toy, "rollout", forbidden)
    with pytest.raises(ValueError):
        toy.run_experiment(small_config(**overrides), clock=FakeClock(START, 0))


# CLI.


def run_cli(*args):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    return subprocess.run(
        [sys.executable, "-m", "imitation.experiments.agnostic.toy", *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def toy_sha256():
    return hashlib.sha256(pathlib.Path(toy.__file__).read_bytes()).hexdigest()


def test_cli_main_small_pilot_completes_with_identity_fields(tmp_path):
    out = tmp_path / "pilot"
    assert toy.main(["--output", str(out)], clock=FakeClock(START, 0)) == toy.EXIT_OK
    result = strict_json_load(out / "result.json")
    assert result["status"] == "complete"
    assert result["protocol"] == toy.PROTOCOL_VERSION
    assert result["source_sha256"] == toy_sha256()
    assert result["seed"] == 0
    started = dt.datetime.fromisoformat(result["started_utc"])
    ended = dt.datetime.fromisoformat(result["ended_utc"])
    assert started.utcoffset() == dt.timedelta(0) and ended >= started
    assert result["deadline_overshoot"] is None
    assert result["config"]["checkpoints"] == [16, 32, 64]
    assert len(result["cells"]) == result["inventory"]["cells_planned"] == 16
    assert all(c["status"] == "complete" for c in result["cells"])
    assert [p.name for p in out.iterdir() if not p.name.startswith(".")] == [
        "result.json"
    ]


def test_cli_main_multi_seed_has_null_top_level_seed(tmp_path):
    out = tmp_path / "seeds"
    argv = ["--output", str(out), "--seeds", "3", "4", "--budget", "4"]
    argv += ["--batch", "4", "--alphas", "0.1", "--kappas", "1"]
    assert toy.main(argv, clock=FakeClock(START, 0)) == toy.EXIT_OK
    result = strict_json_load(out / "result.json")
    assert result["seed"] is None and result["config"]["seeds"] == [3, 4]


def test_cli_help_exits_zero():
    proc = run_cli("--help")
    assert proc.returncode == 0 and "--deadline" in proc.stdout


def test_cli_expired_deadline_writes_partial_artifact(tmp_path):
    out = tmp_path / "expired"
    proc = run_cli("--output", str(out), "--deadline", "2020-01-01T00:00:00Z")
    assert proc.returncode == toy.EXIT_PARTIAL
    result = strict_json_load(out / "result.json")
    assert result["status"] == "partial_deadline"
    assert result["protocol"] == toy.PROTOCOL_VERSION
    assert result["source_sha256"] == toy_sha256()
    assert result["cells"] == []
    assert result["inventory"]["cells_complete"] == []
    assert result["deadline_utc"] == "2020-01-01T00:00:00+00:00"


def test_cli_refuses_existing_output_and_leaves_bytes_untouched(tmp_path):
    out = tmp_path / "taken"
    out.mkdir()
    marker = out / "result.json"
    marker.write_bytes(b"previous run bytes\x00\x01")
    proc = run_cli("--output", str(out), "--budget", "4", "--batch", "4")
    assert proc.returncode == toy.EXIT_INVALID
    assert marker.read_bytes() == b"previous run bytes\x00\x01"
    assert sorted(p.name for p in out.iterdir()) == ["result.json"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--alphas", "nan"],
        ["--alphas", "0.5"],
        ["--kappas", "inf"],
        ["--seeds", "1", "1"],
        ["--horizons", "0"],
        ["--batch", "0"],
        ["--budget", "-4"],
        ["--qs", "2"],
        ["--checkpoints", "16", "32"],
        ["--deadline", "2026-10-01T00:00:00"],
        ["--deadline", "2026-10-01T00:00:00+02:00"],
        ["--deadline", "2026-10-08T00:00:00Z"],
    ],
)
def test_cli_invalid_arguments_create_no_output(tmp_path, argv):
    out = tmp_path / "never"
    assert toy.main(["--output", str(out)] + argv) == toy.EXIT_INVALID
    assert not out.exists()
