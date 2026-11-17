"""Witness audits, coverage and footers of the random-offset grids.

Only simulator states and deterministic rule experts are used; no checkpoint
is loaded, no policy is trained and nothing here is scientific evidence.
Coverage checks read static records only and forbid every environment
builder.
"""

import copy
import importlib.util
import json
import math
import pathlib

import gymnasium as gym
import numpy as np
import pytest

from imitation.experiments.agnostic import random_grid, restriction, task_audit, tasks
from imitation.experiments.ftrl import env_utils
from imitation.experiments.ftrl import plot_results as pr

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SEED = tasks.RANDOM_GRID_SEED
GRID_IDS = tuple(tasks.RANDOM_GRID_SPECS)
CART, ACRO, MCAR, LUNAR = (
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "LunarLander-v2",
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


amc = _load(
    "grid_masked_coverage",
    REPO_ROOT / "experiments" / "agnostic" / "analyze_masked_coverage.py",
)
tsne = _load(
    "grid_masked_tsne_coverage",
    REPO_ROOT / "experiments" / "agnostic" / "plot_masked_tsne_coverage.py",
)


def gid(env_name, k):
    return tasks.random_grid_id(env_name, k, SEED)


class RuleExpert:
    """Deterministic expert given by a rule on the full observation."""

    def __init__(self, rule):
        self.rule = rule
        self.calls = 0

    def predict(self, obs, deterministic=True):
        assert deterministic
        self.calls += 1
        return np.array([self.rule(o) for o in np.asarray(obs)]), None


def _bits(values):
    return np.asarray(values, dtype=np.float32).tobytes()


def _sim_obs_after_one_step(env_name, state, action):
    """Fresh simulator: set ``state``, step once; (observation, successor)."""
    sim = gym.make(env_name).unwrapped
    sim.reset(seed=0)
    sim.state = np.array(state, dtype=np.float64)
    obs, *_ = sim.step(action)
    after = np.array(sim.state, dtype=np.float64)
    sim.close()
    return obs, after


# ---------------------------------------------------------------------------
# Registration and dispatch
# ---------------------------------------------------------------------------


def test_every_grid_is_accepted_with_its_evidence_type():
    assert len(GRID_IDS) == 32
    for rid in GRID_IDS:
        env_name = tasks.RANDOM_GRID_SPECS[rid].env_name
        expected = (
            "constructed_box2d_state"
            if env_name == LUNAR
            else "constructed_grid_cell_state"
        )
        assert task_audit.evidence_type(env_name, rid) == expected
    # Environment-level evidence types (and the ladder audits) are unchanged.
    assert task_audit.evidence_type(CART) == "cartpole_x_grid"
    assert task_audit.evidence_type(CART, "cart_position_zero") == "cartpole_x_grid"
    assert task_audit.evidence_type(ACRO) == "constructed_unwrapped_state"
    with pytest.raises(ValueError, match="grids"):
        task_audit.audit(RuleExpert(lambda o: 0), CART, gid(ACRO, 8), 301)
    with pytest.raises(KeyError):
        task_audit.constructed_grids(gid(CART, 8))


# ---------------------------------------------------------------------------
# CartPole, Acrobot, MountainCar: constructed grid-cell states
# ---------------------------------------------------------------------------


def _recheck(record, env_name, rid):
    """Independent re-verification of every recorded conflict and state."""
    grid = restriction.get_restriction(rid)
    task = tasks.get_task(env_name)
    by_key = {}
    for s in record["states"]:
        obs = np.asarray(s["observation"], dtype=np.float32)
        assert _bits(grid.apply(obs)) == bytes.fromhex(s["masked_bits"])
        assert _bits(s["masked_observation"]) == bytes.fromhex(s["masked_bits"])
        if s["simulator_valid"]:
            assert s["invalid_reasons"] == [] and all(s["checks"].values())
        else:
            assert s["invalid_reasons"]
        by_key[(s["base"], s["coordinate"], s["fraction"])] = s
    conflicts = [p for p in record["pairs"] if p["conflict"]]
    for p in conflicts:
        a, b = (
            by_key[(p["base"], None if f is None else p["coordinate"], f)]
            for f in (p["fraction_a"], p["fraction_b"])
        )
        assert a["simulator_valid"] and b["simulator_valid"]
        assert a["masked_bits"] == b["masked_bits"]
        assert a["expert_action"] != b["expert_action"]
        # Both full observations match a fresh simulator's step output map.
        for member in (a, b):
            for action in range(task.n_actions):
                obs, after = _sim_obs_after_one_step(
                    env_name,
                    member["simulator_state"],
                    action,
                )
                assert _bits(obs) == _bits(task_audit._obs_map(env_name, after))
    return conflicts


@pytest.mark.parametrize("k", [2, 32])
def test_cartpole_grid_conflicts_are_same_cell_valid_states(k):
    rid = gid(CART, k)
    expert = RuleExpert(lambda o: int(o[2] + 0.5 * o[3] > 0))
    checks = []
    record = task_audit.audit(expert, CART, rid, 301, check=checks.append)
    json.dumps(record, allow_nan=False)
    assert record["evidence_type"] == "constructed_grid_cell_state"
    assert record["status"] == "conflict_found"
    assert record["restriction"]["kind"] == "random_grid"
    assert record["grid"]["descriptor_sha256"] == random_grid.FROZEN_GRID_SHA256[rid]
    assert record["evidence"]["limits"] == task_audit.GRID_CLAIM
    conflicts = _recheck(record, CART, rid)
    assert len(conflicts) == record["n_conflicting_pairs"] > 0
    # Every coordinate is searched on every base: base + two fractions each.
    n_bases = len(record["base_states"])
    assert record["n_states"] == n_bases * (1 + 2 * 4)
    assert record["n_checked_pairs"] == n_bases * 4 * 3
    # Constructed values lie in the base's cell, cut at the nonterminal bounds.
    bounds = record["construction_bounds"]
    assert bounds["x"] == [-2.4, 2.4]
    assert bounds["x_dot"] == [-10.0, 10.0] == bounds["theta_dot"]
    for s in record["states"]:
        if s["coordinate"] is not None:
            low, high = bounds[s["coordinate"]]
            assert low <= s["value"] <= high
    # One deadline check before every query.
    assert checks.count("audit expert query") == record["n_states"]
    assert checks.count("audit simulator step") == 2 * record["n_states"]
    assert record["simulator_steps"] == 2 * record["n_states"]
    assert record["labels_used_for_training"] is False


def test_constant_expert_never_conflicts_and_deadline_stops_the_audit():
    rid = gid(CART, 8)
    record = task_audit.audit(RuleExpert(lambda o: 1), CART, rid, 301)
    assert record["status"] == "restriction_uncertified"
    assert record["n_conflicting_pairs"] == 0 and record["n_checked_pairs"] > 0
    assert record["pair_reason_counts"]["same_expert_action"] > 0

    def deadline(what):
        if what == "audit expert query":
            raise TimeoutError("deadline")

    with pytest.raises(TimeoutError):
        task_audit.audit(RuleExpert(lambda o: 1), CART, rid, 301, check=deadline)


def test_tail_cells_are_cut_at_the_documented_finite_limit():
    grid = random_grid.get_grid(gid(CART, 2))
    x_dot = grid.axes[1]
    assert x_dot.kind == "unbounded"
    low_tail = task_audit._scalar_cell(x_dot, 0, (-10.0, 10.0))
    high_tail = task_audit._scalar_cell(x_dot, 2, (-10.0, 10.0))
    assert low_tail["cell_left"] is None and low_tail["low"] == -10.0
    assert high_tail["cell_right"] is None and high_tail["high"] == 10.0
    assert low_tail["high"] == x_dot.edges[0] and not low_tail["empty"]
    for cell in (low_tail, high_tail):
        for f in task_audit.GRID_FRACTIONS:
            v = task_audit._grid_value(cell, f)
            assert math.isfinite(v) and cell["low"] < v < cell["high"]
    # A cell entirely beyond the limit is empty, never extended to infinity.
    far = random_grid.get_grid(gid(CART, 256)).axes[1]
    assert far.edges[0] < -10.0
    assert task_audit._scalar_cell(far, 0, (-10.0, 10.0))["empty"]
    json.dumps(low_tail, allow_nan=False)


@pytest.mark.parametrize("name", ["theta1", "theta2"])
def test_acrobot_arcs_at_the_angle_seam_keep_unit_pairs_in_their_cell(name):
    straddles = 0
    for k in tasks.RANDOM_GRID_COUNTS:
        grid = random_grid.get_grid(gid(ACRO, k))
        axis = next(a for a in grid.axes if a.name == name)
        ic, is_ = axis.indices
        for j in range(k):
            cell = task_audit._arc_cell(axis, j)
            assert 0 < cell["width"] < 2 * math.pi
            assert cell["width"] == pytest.approx(2 * math.pi / k, abs=1e-12)
            straddles += cell["start"] + cell["width"] > math.pi
            for f in task_audit.GRID_FRACTIONS:
                angle = task_audit._grid_value(cell, f)
                assert -math.pi <= angle < math.pi
                obs = np.zeros(6, dtype=np.float32)
                obs[ic], obs[is_] = np.cos(angle), np.sin(angle)
                masked = grid.apply(obs)
                assert task_audit._grid_cell(axis, masked) == j
    # Every K has exactly one arc across the simulator's +/-pi seam.
    assert straddles == len(tasks.RANDOM_GRID_COUNTS)


def test_acrobot_grid_audit_uses_simulator_angles_and_unit_pairs():
    rid = gid(ACRO, 2)
    expert = RuleExpert(lambda o: 2 if o[1] > 0 else 0)  # sign of sin(theta1)
    record = task_audit.audit(expert, ACRO, rid, 301)
    json.dumps(record, allow_nan=False)
    assert record["status"] == "conflict_found"
    conflicts = _recheck(record, ACRO, rid)
    assert {p["coordinate"] for p in conflicts} >= {"theta1"}
    for s in record["states"]:
        obs = np.asarray(s["observation"], dtype=np.float64)
        for c, si in ((0, 1), (2, 3)):
            assert obs[c] ** 2 + obs[si] ** 2 == pytest.approx(1.0, abs=1e-6)
        th1, th2 = s["simulator_state"][:2]
        assert _bits(s["observation"]) == _bits(
            task_audit._obs_map(ACRO, np.array(s["simulator_state"])),
        )
        assert -math.pi <= th2 <= math.pi and -math.pi <= th1 <= math.pi
    # Terminal constructed states are rejected with their reason.
    for s in record["states"]:
        if "terminal_height" in s["invalid_reasons"]:
            assert not s["simulator_valid"]
            th1, th2 = s["simulator_state"][:2]
            assert -math.cos(th1) - math.cos(th2 + th1) > 1.0


def test_mountaincar_goal_states_are_terminal_and_never_witnesses():
    probe = gym.make(MCAR).unwrapped
    assert task_audit._grid_validity(MCAR, probe, np.array([0.55, 0.01])) == [
        "terminal_goal",
    ]
    assert task_audit._grid_validity(MCAR, probe, np.array([0.55, -0.01])) == []
    assert task_audit._grid_validity(MCAR, probe, np.array([-1.3, 0.0])) == [
        "outside_simulator_bounds",
    ]
    cart = gym.make(CART).unwrapped
    assert task_audit._grid_validity(CART, cart, np.array([2.4, 0, 0, 0])) == [
        "terminal_cart_position",
    ]
    probe.close()
    cart.close()


def test_invalid_members_and_cell_escapes_are_never_witnesses(monkeypatch):
    rid = gid(MCAR, 2)
    expert = RuleExpert(lambda o: 2 if o[1] > 0 else 0)
    clean = task_audit.audit(expert, MCAR, rid, 301)
    assert clean["status"] == "conflict_found"

    real = task_audit._grid_validity

    def every_constructed_state_terminal(env_name, sim, state):
        reasons = real(env_name, sim, state)
        base_states = {tuple(b["simulator_state"]) for b in clean["base_states"]}
        if tuple(float(v) for v in state) not in base_states:
            reasons = reasons + ["terminal_goal"]
        return reasons

    monkeypatch.setattr(task_audit, "_grid_validity", every_constructed_state_terminal)
    rejected = task_audit.audit(expert, MCAR, rid, 301)
    assert rejected["status"] == "restriction_uncertified"
    assert rejected["n_conflicting_pairs"] == 0
    assert rejected["n_invalid_pairs"] == rejected["n_checked_pairs"] > 0
    assert rejected["pair_reason_counts"]["terminal_goal"] > 0
    assert all(
        any("terminal_goal" in r for r in p["reasons"]) for p in rejected["pairs"]
    )
    monkeypatch.undo()

    # A value outside the declared cell is caught by the frozen map itself.
    def outside(cell, fraction):
        if "width" in cell:
            return cell["start"]
        return cell["high"] + 0.5 * (cell["high"] - cell["low"])

    monkeypatch.setattr(task_audit, "_grid_value", outside)
    escaped = task_audit.audit(expert, MCAR, rid, 301)
    flagged = [
        s
        for s in escaped["states"]
        if "frozen_map_assigns_another_cell" in s["invalid_reasons"]
    ]
    assert flagged and not any(s["checks"]["in_declared_cell"] for s in flagged)
    for p in escaped["pairs"]:
        if p["conflict"]:
            assert p["fraction_a"] is not None or p["fraction_b"] is not None
            assert p["masked_equal"] and p["both_valid"]
    assert all(
        not p["conflict"]
        for p in escaped["pairs"]
        if "masked_observations_differ" in p["reasons"]
    )


def test_observation_map_mismatch_fails_closed(monkeypatch):
    real = task_audit._obs_map

    def corrupted(env_name, state):
        obs = real(env_name, state)
        return np.nextafter(obs, np.float32(np.inf))

    monkeypatch.setattr(task_audit, "_obs_map", corrupted)
    with pytest.raises(RuntimeError, match="observation map differs"):
        task_audit.audit(RuleExpert(lambda o: 0), CART, gid(CART, 4), 301)


def test_old_cartpole_x_audit_is_unchanged():
    expert = RuleExpert(lambda o: int(o[0] > 0))
    record = task_audit.audit(
        expert,
        CART,
        "cart_position_velocity_angular_velocity_zero",
        301,
    )
    assert record["evidence_type"] == "cartpole_x_grid"
    assert record["grid"] == list(restriction.AUDIT_X_GRID)
    assert "fractions" not in record


# ---------------------------------------------------------------------------
# LunarLander: rigid velocity edits inside the base's grid cells
# ---------------------------------------------------------------------------


def test_lunar_grid_cells_clamp_to_the_declared_box():
    grid = random_grid.get_grid(gid(LUNAR, 8))
    base_obs = np.array([0.1, 0.9, 0.2, -7.0, 0.05, 0.3, 0.0, 0.0], dtype=np.float32)
    rows, cells = task_audit._lunar_grid_cells(grid, grid.restriction_id, base_obs)
    assert [r[:2] for r in rows] == [r for r in task_audit.LUNAR_GRID_AXES]
    vy = next(c for c in cells if c["index"] == 3)
    assert vy["cell"] == 0 and vy["cell_left"] is None
    assert vy["low"] == -5.0 and vy["base_outside_declared_box"] is True
    axis = grid.axes[3]
    assert vy["high"] == axis.edges[0]
    masked = grid.apply(base_obs)
    for index, _, values in rows:
        for value in values:
            assert value == float(np.float32(value))  # exact Box2D velocity
            feature = task_audit._lunar_feature(index, value)
            edited = base_obs.copy()
            edited[index] = feature
            assert -5.0 <= feature <= 5.0
            assert _bits(grid.apply(edited)) == _bits(masked)
    json.dumps(cells, allow_nan=False)


def test_lunar_grid_audit_keeps_every_existing_check():
    rid = gid(LUNAR, 8)
    sign = RuleExpert(lambda o: 1 if o[5] > 0 else 3)
    found = task_audit.audit(sign, LUNAR, rid, 301)
    json.dumps(found, allow_nan=False)
    assert found["evidence_type"] == "constructed_box2d_state"
    assert found["simulator"]["step_source_sha256"] == (
        task_audit.LUNAR_STEP_SOURCE_SHA256
    )
    assert found["grids"] is None and "zeroed_indices" not in found
    assert found["grid"]["descriptor_sha256"] == random_grid.FROZEN_GRID_SHA256[rid]
    assert found["status"] == "conflict_found" and found["n_conflicting_pairs"] > 0
    grid = random_grid.get_grid(rid)
    names = dict(task_audit.LUNAR_GRID_AXES)
    for b in found["base_states"]:
        assert b["checks"]["replay_bitwise"]
        if b["excluded_reason"] is None:
            assert [c["coordinate"] for c in b["grid_cells"]] == list(names.values())
    for s in found["states"]:
        obs = np.asarray(s["observation"], dtype=np.float32)
        assert _bits(s["masked_observation"]) == _bits(grid.apply(obs))
        if s["coordinate"] is None:
            continue
        assert s["coordinate"] in names.values()
        assert set(s["checks"]) >= {
            "positions_angles_awake_unchanged",
            "other_features_bitwise_equal",
            "edited_feature_equals_target",
            "airborne",
            "observation_copy_equals_source",
            "in_base_grid_cell",
        }
        assert s["simulator_valid"] == all(s["checks"].values())
    for p in found["pairs"]:
        if p["conflict"]:
            assert p["both_one_real_step_execute"] is True
            assert p["masked_equal"] and p["both_valid"] and p["both_labels_stable"]
    members = [s for s in found["states"] if s["one_real_step"] is not None]
    assert members and all(len(s["one_real_step"]["actions"]) == 4 for s in members)
    # Every valid base: the base state plus two values per velocity axis.
    assert len(found["states"]) == 7 * found["n_valid_bases"]

    blind = RuleExpert(lambda o: 1 if o[4] > 0 else 3)  # angle only
    none = task_audit.audit(blind, LUNAR, rid, 301)
    assert none["status"] == "restriction_uncertified"
    assert none["n_conflicting_pairs"] == 0 and none["n_checked_pairs"] > 0


def test_lunar_ladder_records_keep_their_fields():
    rid = "lunarlander_angular_velocity_zero"
    record = task_audit.audit(RuleExpert(lambda o: 0), LUNAR, rid, 301)
    assert record["grids"] == [
        [i, n, list(g)] for i, n, g in task_audit.CONSTRUCTED_GRIDS[rid]
    ]
    assert record["zeroed_indices"] == [5]
    assert "grid" not in record and "fractions" not in record
    assert all("grid_cells" not in b for b in record["base_states"])
    assert all("in_base_grid_cell" not in s["checks"] for s in record["states"])


# ---------------------------------------------------------------------------
# Coverage: recorded descriptor, frozen digest, no environment
# ---------------------------------------------------------------------------


def _record(rid):
    """The run-config record as read back from JSON."""
    return json.loads(json.dumps(restriction.describe_restriction(rid)))


def _forbidden(*args, **kwargs):
    raise AssertionError("forbidden call")


def forbid_environment(mp):
    """Make every environment builder and live restriction lookup raise."""
    mp.setattr(gym, "make", _forbidden)
    mp.setattr(env_utils, "make_env", _forbidden)
    mp.setattr(random_grid, "describe_box", _forbidden)
    for name in ("describe", "_raw_env", "reachable_nonterminal", "alias_table"):
        mp.setattr(tasks, name, _forbidden)
    for name in ("describe_restriction", "get_restriction"):
        mp.setattr(restriction, name, _forbidden)


def test_every_grid_is_bound_to_its_own_environment_only():
    for rid, spec in tasks.RANDOM_GRID_SPECS.items():
        for env_name in (CART, ACRO, MCAR, LUNAR, "FrozenLake-v1"):
            task = tasks.get_task(env_name)
            bound = env_name == spec.env_name
            assert (amc.restriction_kind(rid, task) == "random_grid") is bound
            assert (rid in amc.permitted_restrictions(task)) is bound
    assert amc.restriction_kind(gid(CART, 3), tasks.get_task(CART)) is None


def _boundary_inputs(grid, rng):
    rows = [rng.normal(0, 2, size=(64, grid.obs_shape[0]))]
    for axis in grid.axes:
        for edge in axis.edges:
            for v in (edge, np.nextafter(np.float32(edge), np.float32(-np.inf))):
                row = rng.normal(0, 0.1, size=(1, grid.obs_shape[0]))
                row[0, axis.indices[0]] = v
                rows.append(row)
    return np.concatenate(rows).astype(np.float32)


@pytest.mark.parametrize("rid", GRID_IDS)
def test_recorded_grid_rebuilds_the_frozen_map_without_environment(rid):
    task = tasks.get_task(tasks.RANDOM_GRID_SPECS[rid].env_name)
    record = _record(rid)
    registered = random_grid.get_grid(rid)
    inputs = _boundary_inputs(registered, np.random.default_rng(0))
    with pytest.MonkeyPatch.context() as mp:
        forbid_environment(mp)
        mask, info = amc.mask_from_description(record, rid, task)
    assert info["kind"] == "random_grid" and info["registry"] == "calibration_candidate"
    assert info["descriptor_sha256"] == random_grid.FROZEN_GRID_SHA256[rid]
    assert info["n_bins"] == registered.n_bins and info["seed"] == SEED
    assert _bits(mask.apply(inputs)) == _bits(registered.apply(inputs))
    assert mask.apply(inputs.astype(np.float64)).dtype == np.float64


_RECORD_EXTRA = ("kind", "descriptor_sha256", "nominal_cells")


def _tampered(rid):
    """One edit per case; ``rehashed`` cases also recompute the digest, so
    only the frozen digest or the registered grid can catch them."""
    base = _record(rid)
    linear = next(i for i, a in enumerate(base["axes"]) if a["kind"] == "linear")

    def first_edge(d):
        d["axes"][linear]["edges"][0] += 1e-9

    edits = {
        "n_bins": lambda d: d.update(n_bins=d["n_bins"] * 2),
        "n_bins_float": lambda d: d.update(n_bins=float(d["n_bins"])),
        "seed": lambda d: d.update(seed=d["seed"] + 1),
        "generator": lambda d: d.update(generator="random_offset_grid/2"),
        "env": lambda d: d.update(env_name=MCAR if d["env_name"] != MCAR else CART),
        "edge": first_edge,
        "center": lambda d: d["axes"][linear]["centers"].__setitem__(0, 0.0),
        "offset": lambda d: d["offsets"].__setitem__(0, "0x1.0p-1"),
        "box": lambda d: d["box_low_bits"].__setitem__(0, "00000000"),
        "axis_kind": lambda d: d["axes"][linear].update(kind="unbounded"),
        "rid": lambda d: d.update(restriction_id=gid(d["env_name"], 4)),
    }
    cases = {}
    for name, fn in edits.items():
        for rehashed in (False, True):
            d = copy.deepcopy(base)
            fn(d)
            if rehashed:
                body = {k: v for k, v in d.items() if k not in _RECORD_EXTRA}
                d["descriptor_sha256"] = tasks._digest(body)
            cases[f"{name}/{'rehashed' if rehashed else 'stale'}"] = d
    for name, fn in {
        "digest": lambda d: d.update(descriptor_sha256="0" * 64),
        "nominal": lambda d: d.update(nominal_cells=d["nominal_cells"] + 1),
        "missing": lambda d: d.pop("offsets"),
        "extra": lambda d: d.update(partition={}),
        "kind": lambda d: d.update(kind="quantized"),
    }.items():
        d = copy.deepcopy(base)
        fn(d)
        cases[name] = d
    return cases


@pytest.mark.parametrize("rid", [gid(CART, 8), gid(ACRO, 32), gid(LUNAR, 2)])
def test_tampered_grid_records_fail_closed(rid):
    task = tasks.get_task(tasks.RANDOM_GRID_SPECS[rid].env_name)
    for name, record in _tampered(rid).items():
        with pytest.raises(amc.Ineligible):
            amc.mask_from_description(record, rid, task)
            pytest.fail(f"tampered {name} was accepted")
    # Another environment's task refuses a valid record.
    other = tasks.get_task(MCAR if task.env_name != MCAR else CART)
    with pytest.raises(amc.Ineligible):
        amc.mask_from_description(_record(rid), rid, other)


def test_ladder_and_quantized_coverage_paths_are_unchanged():
    task = tasks.get_task(CART)
    mask, info = amc.mask_from_description(
        _record("cart_position_angular_velocity_zero"),
        "cart_position_angular_velocity_zero",
        task,
    )
    assert info == {
        "kind": "zeroing",
        "zeroed_features": ["x", "theta_dot"],
        "registry_agrees": True,
        "registry": "ladder",
    }


@pytest.mark.parametrize("env_name", [CART, ACRO, LUNAR])
def test_tsne_footer_names_k_and_seed_concisely(env_name):
    rid = gid(env_name, 32)
    task = tasks.get_task(env_name)
    _, info = amc.mask_from_description(_record(rid), rid, task)
    text = tsne.annotation(env_name, 3100, rid, info, "b" * 64)
    assert "Random-offset grid: K=32 per axis" in text
    assert f"seed {SEED}" in text
    assert info["descriptor_sha256"][:12] not in text
    assert ("leg contacts unchanged" in text) is (env_name == LUNAR)
    assert ("32 arcs per angle" in text) is (env_name == ACRO)
    assert "33 cells per scalar axis" in text
    assert chr(0x2014) not in text


# ---------------------------------------------------------------------------
# Learning-curve footer
# ---------------------------------------------------------------------------


def _runs(rid):
    record = _record(rid)
    return [
        {
            "restriction_id": rid,
            "env_name": record["env_name"],
            "config": {"restriction": copy.deepcopy(record)},
        }
        for _ in range(3)
    ]


def test_learning_curve_grid_line_is_concise_and_from_the_record():
    cases = {
        gid(CART, 32): "Random-offset bins: K=32 per axis (33 cells per scalar "
        "axis); seed 20261003.",
        gid(ACRO, 32): "Random-offset bins: K=32 per axis (33 cells per scalar "
        "axis, 32 arcs per angle); seed 20261003.",
        gid(LUNAR, 32): "Random-offset bins: K=32 per axis (33 cells per scalar "
        "axis); seed 20261003; leg contacts unchanged.",
    }
    for rid, text in cases.items():
        assert pr._restriction_observation(_runs(rid)) == {
            "kind": "random_grid",
            "text": text,
        }
        assert len(text) <= 100


@pytest.mark.parametrize("field", ["n_bins", "seed", "axes", "kind"])
def test_learning_curve_grid_line_cross_checks_the_record(field):
    rid = gid(ACRO, 8)
    runs = _runs(rid)
    for run in runs:
        desc = run["config"]["restriction"]
        if field == "n_bins":
            desc["n_bins"] = 16
        elif field == "seed":
            desc["seed"] = SEED + 1
        elif field == "axes":
            desc["axes"][2]["centers"] = desc["axes"][2]["centers"][:-1]
        else:
            desc["kind"] = "random_grid_v2"
    with pytest.raises(ValueError):
        pr._restriction_observation(runs)
