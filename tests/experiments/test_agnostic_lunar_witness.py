"""Tests for the LunarLander constructed Box2D state witness (`task_audit`).

Only Box2D and fake rule experts are used; no checkpoint is loaded.
"""

import json

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.box2d import lunar_lander as ll

from imitation.experiments.agnostic import restriction, task_audit

LADDER = (
    "lunarlander_angular_velocity_zero",
    "lunarlander_velocities_zero",
    "lunarlander_velocities_angle_zero",
)
NAMES = {2: "lander_vx", 3: "lander_vy", 5: "lander_angular_velocity"}


class RuleExpert:
    """Deterministic expert given by a rule on the full observation."""

    def __init__(self, rule):
        self.rule = rule
        self.calls = 0

    def predict(self, obs, deterministic=True):
        assert deterministic
        self.calls += 1
        obs = np.asarray(obs)
        return np.array([self.rule(o) for o in obs]), None


def _heuristic(o):
    return int(ll.heuristic(type("E", (), {"continuous": False})(), o))


def heuristic_expert():
    return RuleExpert(_heuristic)


def _no_check(what):
    del what


def _airborne_states(seed=301, limit=400):
    """(env, recorded observation) at airborne steps of a heuristic landing."""
    env = gym.make(task_audit.LUNAR_ENV)
    obs, _ = env.reset(seed=seed)
    sim = env.unwrapped
    for _ in range(limit):
        obs, _, terminated, truncated, _ = env.step(_heuristic(obs))
        if terminated or truncated:
            break
        if obs[6] == 0 and obs[7] == 0 and task_audit._touching_contacts(sim) == 0:
            yield env, np.asarray(obs, dtype=np.float32)
    env.close()


def test_zero_length_step_reproduces_the_real_step_on_airborne_states():
    checked = 0
    for env, recorded in _airborne_states():
        sim = env.unwrapped
        before = task_audit._body_bits(sim)
        rng = sim.np_random.bit_generator.state
        refresh = task_audit._zero_dt_step(sim)
        assert task_audit._refresh_problems(refresh) == []
        assert refresh["world_step_calls"] == [list(task_audit.LUNAR_WORLD_STEP)]
        assert task_audit._body_bits(sim) == before
        assert sim.np_random.bit_generator.state == rng
        if refresh["obs"][6] == 0 and refresh["obs"][7] == 0:
            assert refresh["obs"].tobytes() == recorded.tobytes()
            assert task_audit._box2d_obs_map(sim).tobytes() == refresh["obs"].tobytes()
            checked += 1
    assert checked >= 50


def test_landing_bases_are_excluded_as_contact_transitions(monkeypatch):
    monkeypatch.setattr(task_audit, "CONSTRUCTED_BASE_STRIDE", 1)
    record = task_audit.audit(heuristic_expert(), task_audit.LUNAR_ENV, LADDER[0], 301)
    excluded = record["n_excluded_bases"]
    assert excluded.get("contact_transition", 0) >= 1
    assert excluded.get("not_airborne", 0) >= 1
    valid = {b["base"] for b in record["base_states"] if b["excluded_reason"] is None}
    assert {s["base"] for s in record["states"]} == valid
    assert {p["base"] for p in record["pairs"]} <= valid
    for b in record["base_states"]:
        if b["excluded_reason"] == "contact_transition":
            assert not b["checks"]["source_map_reproduces_recorded"]


@pytest.mark.parametrize("index", [5, 2, 3])
def test_rigid_velocity_edit_and_exact_restore(index):
    name = NAMES[index]
    seen = 0
    for env, recorded in _airborne_states():
        sim = env.unwrapped
        pose, velocities = task_audit._pose_bits(sim), task_audit._velocities(sim)
        joints = task_audit._joint_rows(sim)
        for value in (-2.5, 1.0) if index == 5 else (-5.0, 7.5):
            delta = task_audit._apply_velocity_edit(sim, name, value)
            assert task_audit._pose_bits(sim) == pose
            lander = task_audit._velocities(sim)[0]
            kept = {5: (0, 1), 2: (1, 2), 3: (0, 2)}[index]
            assert all(lander[i] == velocities[0][i] for i in kept)
            residual = task_audit._joint_residual(
                joints,
                task_audit._joint_rows(sim),
                delta,
                index == 5,
            )
            assert residual <= task_audit.LUNAR_JOINT_TOLERANCE
            obs = task_audit._zero_dt_step(sim)["obs"]
            assert obs[index] == task_audit._lunar_feature(index, value)
            assert (
                np.delete(obs, index).tobytes() == np.delete(recorded, index).tobytes()
            )
            task_audit._set_velocities(sim, velocities)
            assert task_audit._velocities(sim) == velocities
            assert task_audit._pose_bits(sim) == pose
            assert task_audit._zero_dt_step(sim)["obs"].tobytes() == recorded.tobytes()
        seen += 1
        if seen == 20:
            break
    assert seen == 20


def test_grid_values_map_to_the_predeclared_observation_values():
    for rid in LADDER:
        for index, name, grid in task_audit.CONSTRUCTED_GRIDS[rid]:
            assert NAMES[index] == name
            assert [float(task_audit._lunar_feature(index, v)) for v in grid] == [
                float(np.float32(x)) for x in (-1.0, -0.4, 0.4, 1.0)
            ]
            assert index in restriction.get_restriction(rid).zeroed_indices
    assert 4 not in {i for i, _, _ in task_audit.CONSTRUCTED_GRIDS[LADDER[2]]}


def test_zero_length_step_restores_state_when_the_step_raises(monkeypatch):
    env = gym.make(task_audit.LUNAR_ENV)
    env.reset(seed=301)
    sim = env.unwrapped
    world, rng = sim.world, sim.np_random.bit_generator.state
    prev, game_over = sim.prev_shaping, sim.game_over

    def broken(self, action):
        self.np_random.uniform(-1.0, 1.0)
        self.prev_shaping, self.game_over = -1.0, True
        self.world.Step(1.0 / 50, 180, 60)
        raise ValueError("boom")

    monkeypatch.setattr(ll.LunarLander, "step", broken)
    with pytest.raises(ValueError, match="boom"):
        task_audit._zero_dt_step(sim)
    assert sim.world is world
    assert sim.np_random.bit_generator.state == rng
    assert (sim.prev_shaping, sim.game_over) == (prev, game_over)
    env.close()


def test_a_sign_of_angular_velocity_expert_conflicts_and_a_blind_one_does_not():
    sign = RuleExpert(lambda o: 1 if o[5] > 0 else 3)
    found = task_audit.audit(sign, task_audit.LUNAR_ENV, LADDER[0], 301)
    assert found["evidence_type"] == "constructed_box2d_state"
    assert found["status"] == "conflict_found" and found["n_conflicting_pairs"] > 0
    for p in found["pairs"]:
        if p["conflict"]:
            assert p["both_one_real_step_execute"] is True
            assert p["expert_action_a"] != p["expert_action_b"]
    members = [s for s in found["states"] if s["one_real_step"] is not None]
    assert members and all(
        s["one_real_step"]["one_real_step_executes"] for s in members
    )
    assert all(len(s["one_real_step"]["actions"]) == 4 for s in members)
    json.dumps(found, allow_nan=False)

    blind = RuleExpert(lambda o: _heuristic(np.where(np.arange(8) == 5, 0.0, o)))
    none = task_audit.audit(blind, task_audit.LUNAR_ENV, LADDER[0], 301)
    assert none["status"] == "restriction_uncertified"
    assert none["n_conflicting_pairs"] == 0 and none["n_checked_pairs"] > 0
    assert all(p["masked_equal"] for p in none["pairs"])
    assert none["n_one_real_step_members"] == 0


def test_rung_coordinates_and_masked_equality():
    expert = heuristic_expert()
    first = task_audit.audit(expert, task_audit.LUNAR_ENV, LADDER[0], 301)
    assert {p["coordinate"] for p in first["pairs"]} == {"lander_angular_velocity"}
    for rid in LADDER[1:]:
        record = task_audit.audit(expert, task_audit.LUNAR_ENV, rid, 301)
        assert {p["coordinate"] for p in record["pairs"]} == {
            "lander_vx",
            "lander_vy",
            "lander_angular_velocity",
        }
        assert all(p["masked_equal"] for p in record["pairs"])
        assert record["n_invalid_pairs"] == 0
        assert all(s["simulator_valid"] for s in record["states"])
        assert record["zeroed_indices"] == list(
            restriction.get_restriction(rid).zeroed_indices,
        )
    # Every valid base contributes the base state plus four values per coordinate.
    per_base = 1 + 4 * len(task_audit.CONSTRUCTED_GRIDS[LADDER[0]])
    assert len(first["states"]) == per_base * first["n_valid_bases"]
    bases = [b["step"] for b in first["base_states"]]
    assert bases == list(range(0, first["trajectory"]["steps"], 10))


def test_unstable_labels_and_failed_real_steps_never_count(monkeypatch):
    class BatchDependent:
        def predict(self, obs, deterministic=True):
            obs = np.asarray(obs)
            flip = len(obs) > 1
            return np.array([int((o[5] > 0) != flip) + 1 for o in obs]), None

    unstable = task_audit.audit(BatchDependent(), task_audit.LUNAR_ENV, LADDER[0], 301)
    assert unstable["n_conflicting_pairs"] == 0
    assert not any(s["label_stable"] for s in unstable["states"])

    def refused(*args, **kwargs):
        return {"actions": [], "one_real_step_executes": False}

    monkeypatch.setattr(task_audit, "_one_real_step", refused)
    sign = RuleExpert(lambda o: 1 if o[5] > 0 else 3)
    record = task_audit.audit(sign, task_audit.LUNAR_ENV, LADDER[0], 301)
    assert record["status"] == "restriction_uncertified"
    assert any(p["both_one_real_step_execute"] is False for p in record["pairs"])


def test_a_corrupted_recorded_observation_raises():
    env = gym.make(task_audit.LUNAR_ENV)
    obs, _ = env.reset(seed=301)
    recorded, actions = [obs], []
    for _ in range(20):
        actions.append(_heuristic(obs))
        obs, *_ = env.step(actions[-1])
        recorded.append(obs)
    env.close()
    task_audit._lunar_replay(301, actions, recorded, 20, _no_check).close()
    corrupted = list(recorded)
    corrupted[20] = np.nextafter(recorded[20], np.float32(np.inf))
    with pytest.raises(RuntimeError, match="differs from the recorded"):
        task_audit._lunar_replay(301, actions, corrupted, 20, _no_check)


def test_simulator_pins_refuse(monkeypatch):
    monkeypatch.setattr(task_audit, "LUNAR_STEP_SOURCE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="step source"):
        task_audit.audit(heuristic_expert(), task_audit.LUNAR_ENV, LADDER[0], 301)
    monkeypatch.undo()
    for kwargs, what in (
        ({"enable_wind": True}, "wind"),
        ({"continuous": True}, "continuous"),
    ):
        env = gym.make(task_audit.LUNAR_ENV, **kwargs)
        with pytest.raises(RuntimeError, match=what):
            task_audit._lunar_simulator_pins(env)
        env.close()
    env = gym.make(task_audit.LUNAR_ENV)
    pins = task_audit._lunar_simulator_pins(env)
    assert pins["step_source_sha256"] == task_audit.LUNAR_STEP_SOURCE_SHA256
    assert pins["wrappers"] == list(task_audit.LUNAR_WRAPPERS)
    env.close()


def test_other_environments_keep_their_evidence_and_grids():
    assert {
        e: task_audit.evidence_type(e)
        for e in (
            "FrozenLake-v1",
            "CliffWalking-v0",
            "Taxi-v3",
            "Blackjack-v1",
            "Acrobot-v1",
            "MountainCar-v0",
            "CartPole-v1",
            "Pendulum-v1",
        )
    } == {
        "FrozenLake-v1": "reachable_enumeration",
        "CliffWalking-v0": "reachable_enumeration",
        "Taxi-v3": "reachable_enumeration",
        "Blackjack-v1": "reachable_enumeration",
        "Acrobot-v1": "constructed_unwrapped_state",
        "MountainCar-v0": "constructed_unwrapped_state",
        "CartPole-v1": "cartpole_x_grid",
        "Pendulum-v1": "unverified",
    }
    old = {
        "acrobot_theta2_dot_zero": ((3, "theta2_dot", (-8.0, -3.0, 3.0, 8.0)),),
        "acrobot_theta_dots_zero": (
            (2, "theta1_dot", (-4.0, -1.5, 1.5, 4.0)),
            (3, "theta2_dot", (-8.0, -3.0, 3.0, 8.0)),
        ),
        "acrobot_theta2_and_dots_zero": (
            (1, "theta2", (-2.5, -0.8, 0.8, 2.5)),
            (2, "theta1_dot", (-4.0, -1.5, 1.5, 4.0)),
            (3, "theta2_dot", (-8.0, -3.0, 3.0, 8.0)),
        ),
        "mountaincar_velocity_zero": ((1, "velocity", (-0.05, -0.02, 0.02, 0.05)),),
        "mountaincar_position_zero": ((0, "position", (-1.1, -0.8, -0.5, -0.2)),),
    }
    assert {
        k: v for k, v in task_audit.CONSTRUCTED_GRIDS.items() if k not in LADDER
    } == old
    assert task_audit.CONSTRUCTED_BASE_STRIDE == 10
