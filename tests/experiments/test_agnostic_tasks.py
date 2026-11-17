"""Classical restriction tasks: caps, frozen maps, masks, checkpoints, audits.

Maps are checked on reachable nonterminal inputs against the rung's own class
rule and against independent random rollouts of the real environments; masks
are checked on real policies; audits use rule experts whose answers are known.
"""

import collections

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO

from imitation.experiments.agnostic import restriction, run_classical, task_audit, tasks
from imitation.experiments.ftrl import env_utils

CAPS = {
    "CartPole-v1": (500, 500),
    "FrozenLake-v1": (100, 100),
    "CliffWalking-v0": (None, 200),
    "Acrobot-v1": (500, 500),
    "MountainCar-v0": (200, 200),
    "Taxi-v3": (200, 200),
    "Blackjack-v1": (None, 20),
    "LunarLander-v2": (1000, 1000),
}
# Distinct masked outputs on reachable nonterminal inputs, per rung.
CLASS_COUNTS = {
    "frozenlake_column_pairs": 8,
    "frozenlake_column_hidden": 4,
    "frozenlake_column_hidden_row_pairs": 2,
    "cliffwalking_column_pairs": 19,
    "cliffwalking_column_hidden": 4,
    "cliffwalking_column_hidden_row_pairs": 2,
    "taxi_destination_hidden": 125,
    "taxi_destination_passenger_hidden": 25,
    "blackjack_usable_ace_hidden": 180,
    "blackjack_usable_ace_dealer_hidden": 18,
}
ZEROED = {
    "cart_position_velocity_angular_velocity_zero": ("CartPole-v1", (0, 1, 3)),
    "acrobot_theta2_dot_zero": ("Acrobot-v1", (5,)),
    "acrobot_theta_dots_zero": ("Acrobot-v1", (4, 5)),
    "acrobot_theta2_and_dots_zero": ("Acrobot-v1", (2, 3, 4, 5)),
    "mountaincar_velocity_zero": ("MountainCar-v0", (1,)),
    "mountaincar_position_zero": ("MountainCar-v0", (0,)),
    "lunarlander_angular_velocity_zero": ("LunarLander-v2", (5,)),
    "lunarlander_velocities_zero": ("LunarLander-v2", (2, 3, 5)),
    "lunarlander_velocities_angle_zero": ("LunarLander-v2", (2, 3, 4, 5)),
}


def _random_expert(env_name, seed=0):
    venv = env_utils.make_env(env_name, 1, np.random.default_rng(0))
    policy = PPO("MlpPolicy", venv, device="cpu", seed=seed).policy
    venv.close()
    return policy


# ---------------------------------------------------------------------------
# Caps and environment description
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_name", sorted(CAPS))
def test_effective_caps_come_from_the_original_factory(env_name):
    registered, effective = CAPS[env_name]
    task = tasks.get_task(env_name)
    record = tasks.describe(task)

    assert gym.spec(env_name).max_episode_steps == registered
    assert record["registered_max_episode_steps"] == registered
    assert task.episode_cap == record["episode_cap"] == effective
    assert tasks.effective_episode_cap(env_name) == effective
    assert record["episode_cap_rule"].startswith(
        "registered" if registered else "env_utils fallback",
    )
    assert record["observation_space"]["shape"] == [task.obs_dim]
    config = run_classical.build_preparation_config(
        env_name,
        seed=0,
        qualification_seed=1000,
        eval_episodes=100,
        convergence_override=None,
    )
    assert config["env_max_episode_steps"] == effective
    assert config["qualification"]["max_env_steps"] == 100 * effective
    # Tasks with a registered cap keep their previous config keys and digests.
    assert ("env_episode_cap_rule" in config) == (registered is None)


@pytest.mark.parametrize("env_name", ["CliffWalking-v0", "Blackjack-v1"])
def test_fallback_cap_truncates_real_episodes(env_name):
    venv = env_utils.make_env(env_name, 1, np.random.default_rng(0))
    venv.reset()
    # CliffWalking: walking into the left wall never terminates. Blackjack can
    # never last 20 steps, so its cap is only a bound.
    action = np.array([3 if env_name == "CliffWalking-v0" else 1])
    steps = 0
    while True:
        _, _, done, info = venv.step(action)
        steps += 1
        if done[0]:
            break
    venv.close()
    if env_name == "CliffWalking-v0":
        assert steps == 200 and info[0]["TimeLimit.truncated"]
    else:
        assert steps < 20


def test_recorded_flags_bind_the_original_configuration():
    lake = tasks.describe(tasks.get_task("FrozenLake-v1"))
    jack = tasks.describe(tasks.get_task("Blackjack-v1"))
    assert lake["configured_env_kwargs"] == {"is_slippery": False}
    assert lake["registered_kwargs"] == {"map_name": "4x4"}
    assert lake["tabular_dynamics_sha256"]
    assert jack["registered_kwargs"] == {"natural": False, "sab": True}
    assert jack["unwrapped_attributes"] == {"natural": False, "sab": True}
    with pytest.raises(tasks.TaskMismatch):
        tasks.describe(
            tasks.Task(**{**vars(tasks.get_task("Taxi-v3")), "n_actions": 5})
        )


# ---------------------------------------------------------------------------
# Frozen categorical maps
# ---------------------------------------------------------------------------


def _observed_states(env_name, episodes=300):
    """Pre-action joint states seen by uniform random play on the factory."""
    rng = np.random.default_rng(1)
    venv = env_utils.make_env(env_name, 1, rng)
    venv.action_space.seed(1)
    jack = env_name == "Blackjack-v1"
    seen = set()
    obs = venv.reset()
    done_episodes = 0
    while done_episodes < episodes:
        if jack:
            o = obs[0]
            seen.add(
                tasks.blackjack_index(
                    o[:32].argmax(),
                    o[32:43].argmax(),
                    o[43:].argmax(),
                ),
            )
        else:
            seen.add(int(obs[0].argmax()))
        obs, _, done, _ = venv.step(np.array([venv.action_space.sample()]))
        done_episodes += int(done[0])
    venv.close()
    return seen


@pytest.mark.parametrize(
    "env_name",
    ["FrozenLake-v1", "CliffWalking-v0", "Taxi-v3", "Blackjack-v1"],
)
def test_reachable_sets_cover_independent_random_play(env_name):
    reachable, evidence = tasks.reachable_nonterminal(env_name)
    observed = _observed_states(env_name)
    assert observed <= set(reachable)
    if env_name == "CliffWalking-v0":
        # Cliff cells reset to the start and are never observed; the goal ends.
        assert set(reachable) == set(range(36)) | {36}
    if env_name == "FrozenLake-v1":
        assert evidence["terminal_states_reached"] == [5, 7, 11, 12, 15]


@pytest.mark.parametrize("restriction_id", sorted(CLASS_COUNTS))
def test_alias_maps_on_reachable_inputs(restriction_id):
    found = restriction.get_restriction(restriction_id)
    record = tasks.alias_table(restriction_id)
    rung = tasks.ALIAS_RUNGS[restriction_id]
    reachable = set(tasks.reachable_nonterminal(found.env_name)[0])
    table = np.asarray(found.representative)

    assert record["table_sha256"] == tasks.FROZEN_TABLE_SHA256[restriction_id]
    for s in reachable:
        rep = int(table[s])
        assert rep in reachable  # valid, reachable and nonterminal
        assert table[rep] == rep  # idempotent
        assert rung.key(rep) == rung.key(s)  # visible factors are kept
        assert rep == min(r for r in reachable if rung.key(r) == rung.key(s))
    assert len({int(table[s]) for s in reachable}) == CLASS_COUNTS[restriction_id]

    states = np.array(sorted(reachable))
    obs = found.encode(states)
    masked = found.apply(obs)
    assert masked.dtype == obs.dtype and masked.shape == obs.shape
    offset = 0
    for n in found.blocks:  # Blackjack keeps three one-hot blocks
        np.testing.assert_array_equal(masked[:, offset : offset + n].sum(1), 1)
        offset += n
    np.testing.assert_array_equal(found.apply(masked), masked)
    np.testing.assert_array_equal(found.decode(masked), table[states])
    with th.no_grad():
        torch_masked = restriction.ObservationMask(restriction_id)(th.as_tensor(obs))
    np.testing.assert_array_equal(torch_masked.numpy(), masked)


def test_alias_maps_refuse_non_one_hot_inputs():
    found = restriction.get_restriction("taxi_destination_hidden")
    bad = found.encode(np.array([7, 9]))
    bad[0, 3] = 1.0
    with pytest.raises(ValueError, match="one-hot"):
        found.apply(bad)
    with pytest.raises(ValueError, match="one-hot"):
        restriction.ObservationMask("taxi_destination_hidden")(th.as_tensor(bad))


def test_hidden_taxi_and_blackjack_factors_collapse():
    taxi = restriction.get_restriction("taxi_destination_passenger_hidden")
    env = gym.make("Taxi-v3").unwrapped
    a, b = env.encode(2, 3, 0, 1), env.encode(2, 3, 4, 2)
    assert taxi.representative[a] == taxi.representative[b]
    assert taxi.representative[env.encode(2, 4, 0, 1)] != taxi.representative[a]
    jack = restriction.get_restriction("blackjack_usable_ace_hidden")
    soft, hard = tasks.blackjack_index(15, 6, 1), tasks.blackjack_index(15, 6, 0)
    assert jack.representative[soft] == jack.representative[hard] == hard
    other = tasks.blackjack_index(15, 7, 0)
    assert jack.representative[other] != jack.representative[hard]


def test_changed_alias_table_is_refused(monkeypatch):
    tasks._alias.cache_clear()
    monkeypatch.setitem(tasks.FROZEN_TABLE_SHA256, "taxi_destination_hidden", "0" * 64)
    with pytest.raises(RuntimeError, match="differs from the frozen"):
        tasks.alias_table("taxi_destination_hidden")
    monkeypatch.undo()
    tasks._alias.cache_clear()
    tasks.alias_table("taxi_destination_hidden")


# ---------------------------------------------------------------------------
# Continuous masks and the teacher/learner boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("restriction_id", sorted(ZEROED))
def test_only_listed_coordinates_are_zeroed_for_the_learner(restriction_id):
    env_name, hidden = ZEROED[restriction_id]
    task = tasks.get_task(env_name)
    found = restriction.get_restriction(restriction_id)
    assert found.zeroed_indices == hidden
    assert restriction_id in task.ladder

    rng = np.random.default_rng(0)
    obs = rng.uniform(-1, 1, size=(32, task.obs_dim)).astype(np.float32) + 2.0
    masked = found.apply(obs)
    kept = [i for i in range(task.obs_dim) if i not in hidden]
    np.testing.assert_array_equal(masked[:, list(hidden)], 0)
    np.testing.assert_array_equal(masked[:, kept], obs[:, kept])

    expert = _random_expert(env_name)
    learner = restriction.make_linear_policy(expert, restriction_id)
    with th.no_grad():
        learner.action_net.weight.normal_()
    moved = obs.copy()
    moved[:, list(hidden)] += 0.7
    tensor, moved_tensor = th.as_tensor(obs), th.as_tensor(moved)
    with th.no_grad():
        np.testing.assert_array_equal(
            learner.get_distribution(tensor).distribution.probs.numpy(),
            learner.get_distribution(moved_tensor).distribution.probs.numpy(),
        )
        # The expert is unmasked: it still reads the hidden coordinates.
        assert not np.array_equal(
            expert.get_distribution(tensor).distribution.probs.numpy(),
            expert.get_distribution(moved_tensor).distribution.probs.numpy(),
        )


@pytest.mark.parametrize(
    "env_name, restriction_id",
    [
        ("FrozenLake-v1", "frozenlake_column_pairs"),
        ("Blackjack-v1", "blackjack_usable_ace_dealer_hidden"),
        ("Acrobot-v1", "acrobot_theta2_and_dots_zero"),
    ],
)
def test_restricted_checkpoints_reload_with_identical_actions(
    env_name,
    restriction_id,
    tmp_path,
):
    expert = _random_expert(env_name)
    th.manual_seed(3)
    policy = restriction.make_linear_policy(expert, restriction_id)
    with th.no_grad():
        policy.action_net.weight.normal_()
    found = restriction.get_restriction(restriction_id)
    if isinstance(found, restriction.AliasRestriction):
        obs = found.encode(np.arange(int(np.prod(found.blocks))))
    else:
        obs = np.random.default_rng(0).normal(size=(64, 6)).astype(np.float32)
    path = tmp_path / "policy.pt"
    policy.save(path)
    loaded = restriction.load_policy_checkpoint(path)

    assert loaded.restriction_id == restriction_id
    np.testing.assert_array_equal(
        loaded.predict(obs, deterministic=True)[0],
        policy.predict(obs, deterministic=True)[0],
    )
    saved = th.load(path, weights_only=False)
    if isinstance(found, restriction.AliasRestriction):
        assert saved["data"]["restriction_sha256"] == found.table_sha256
        saved["data"]["restriction_sha256"] = "0" * 64
        th.save(saved, path)
        with pytest.raises(ValueError, match="not the saved"):
            restriction.load_policy_checkpoint(path)
    else:
        assert "restriction_sha256" not in saved["data"]


# ---------------------------------------------------------------------------
# Witness audits
# ---------------------------------------------------------------------------


class RuleExpert:
    """Deterministic expert given by a rule on the full observation."""

    def __init__(self, rule):
        self.rule = rule

    def predict(self, obs, deterministic=True):
        assert deterministic
        obs = np.asarray(obs)
        return np.array([self.rule(o) for o in obs]), None


def test_enumerated_audit_groups_every_reachable_state():
    # Move right in even columns and down in odd ones.
    expert = RuleExpert(lambda o: 2 if int(o.argmax()) % 2 == 0 else 1)
    pairs = task_audit.audit(expert, "FrozenLake-v1", "frozenlake_column_pairs", 301)
    states, _ = tasks.reachable_nonterminal("FrozenLake-v1")
    assert pairs["evidence_type"] == "reachable_enumeration"
    assert pairs["states"] == states and pairs["n_states"] == 11
    table = restriction.get_restriction("frozenlake_column_pairs").representative
    by_class = collections.defaultdict(set)
    for s, a in zip(pairs["states"], pairs["expert_actions"]):
        by_class[table[s]].add(a)
    expected = sum(len(actions) > 1 for actions in by_class.values())
    assert pairs["n_conflicting_classes"] == expected > 0
    assert pairs["status"] == "conflict_found"

    rows = task_audit.audit(
        RuleExpert(lambda o: int(o.argmax()) // 4),
        "FrozenLake-v1",
        "frozenlake_column_hidden",
        301,
    )
    assert rows["status"] == "no_conflict_on_reachable"


def test_constructed_audit_finds_velocity_witnesses_only_when_velocity_is_hidden():
    expert = RuleExpert(lambda o: 2 if o[1] > 0 else 0)  # pump with the velocity
    hidden = task_audit.audit(
        expert, "MountainCar-v0", "mountaincar_velocity_zero", 301
    )
    kept = task_audit.audit(expert, "MountainCar-v0", "mountaincar_position_zero", 301)

    assert (
        hidden["evidence_type"]
        == kept["evidence_type"]
        == ("constructed_unwrapped_state")
    )
    assert hidden["status"] == "conflict_found" and hidden["n_conflicting_pairs"] > 0
    assert kept["status"] == "restriction_uncertified"
    for record in (hidden, kept):
        assert all(s["observation_map_verified"] for s in record["states"])
        assert all(p["masked_equal"] for p in record["pairs"])
        assert record["n_checked_pairs"] == len(record["pairs"]) > 0
    for p in hidden["pairs"]:
        if p["conflict"]:
            assert (p["value_a"] > 0) != (p["value_b"] > 0)


def test_acrobot_constructed_states_are_checked_against_the_simulator():
    expert = RuleExpert(lambda o: 2 if o[5] > 0 else 0)
    record = task_audit.audit(expert, "Acrobot-v1", "acrobot_theta2_dot_zero", 301)
    assert record["status"] == "conflict_found"
    assert all(s["observation_map_verified"] for s in record["states"])
    assert {s["coordinate"] for s in record["states"]} == {"theta2_dot"}


def test_lunarlander_and_cartpole_audits_report_their_evidence():
    calls = []
    expert = RuleExpert(lambda o: calls.append(1) or 0)
    lunar = task_audit.audit(
        expert,
        "LunarLander-v2",
        "lunarlander_angular_velocity_zero",
        301,
    )
    # Constructed Box2D states since the Lunar witness; a constant expert
    # never conflicts (focused tests: test_agnostic_lunar_witness.py).
    assert lunar["evidence_type"] == "constructed_box2d_state"
    assert lunar["status"] == "restriction_uncertified"
    assert lunar["expert_predict_calls"] > 0 and calls
    cart = task_audit.audit(
        RuleExpert(lambda o: int(o[0] > 0)),
        "CartPole-v1",
        "cart_position_velocity_angular_velocity_zero",
        301,
    )
    assert cart["evidence_type"] == "cartpole_x_grid"
    assert cart["status"] == "conflict_found"
    with pytest.raises(ValueError, match="ladder"):
        task_audit.audit(expert, "CartPole-v1", "cart_position_zero", 301)
