"""Standard row/column binning candidates and the slippery FrozenLake task.

Covers the ``FrozenLakeSlippery-v0`` variant task (the standard stochastic
4x4 FrozenLake), the CliffWalking row-pair and column-triple calibration
candidates, the wider Blackjack player-sum bins and the CliffWalking
uniform-random reference with its diagnostic stall benchmark. Every check is
synthetic or exact dynamic programming on the tabular transition tables; no
learner runs.

Bounds on masked policies below hold only for deterministic memoryless
policies of the masked observation. Stochastic or history-dependent masked
policies can do better and are not bounded here.
"""

import collections
import itertools
from fractions import Fraction

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO

from imitation.experiments.agnostic import restriction, task_audit, tasks
from imitation.experiments.ftrl import env_baselines, env_utils, expert_training

ORIGINAL_ENVS = (
    "CartPole-v1",
    "FrozenLake-v1",
    "CliffWalking-v0",
    "Acrobot-v1",
    "MountainCar-v0",
    "Taxi-v3",
    "Blackjack-v1",
    "LunarLander-v2",
)
# Task digests recorded before any replacement mask; they must never move.
EXISTING_TASK_DIGESTS = {
    "FrozenLake-v1": "7e6760d99dc8898d6f52ff4a90cd25b8911efda94efbee7f52386c09378015a4",
    "CliffWalking-v0": (
        "01127136af3841e64bc22eb04373553af3b8db159c9c6ba8f20f24348fd24967"
    ),
    "Blackjack-v1": "f6877a8cfdbe70ad61bd28f4092b4d15e04bcae0dd54cd180fd3fb1ab80182fb",
}
EXISTING_FROZEN_TABLES = {
    "frozenlake_column_pairs": (
        "ceeaea50bb590eb134d3a282573a8fbc2b08e9a9dd27b5d558d5f44a802f3b10"
    ),
    "frozenlake_column_hidden": (
        "e14eafe001fea5acee0c66e01bcd5ed223461994ba130de9c2e8c02164c6f1ea"
    ),
    "frozenlake_column_hidden_row_pairs": (
        "3cb6422a91a104ef9de2eda9c7ac4bd60391459523d293f3b02db247415c9cdb"
    ),
    "cliffwalking_column_pairs": (
        "5edeb6077203af829b57f44912fe12af786fba1e774667fd5b0fa3e3f69cc639"
    ),
    "cliffwalking_column_hidden": (
        "741b118c6dd6c254a5f6a065b07d6cb8fe7bea2e309115ceafe6035767334cb4"
    ),
    "cliffwalking_column_hidden_row_pairs": (
        "56af4e8f488cf5bd0571df4e8b17c127d020c2eb304ab7a661e6c07b5c141bfe"
    ),
    "blackjack_player_sum_bins2_dealer_ace_hidden": (
        "c66418ab0fbaa47ddaa4a0ecf847284c7b20e36ad9b083ae8e28241d1bcc7d0f"
    ),
    "blackjack_player_sum_bins3_dealer_ace_hidden": (
        "5bb2b8d23c22c33522c2680a294b29da2eb45141519e4ed3999e93b3a7d81128"
    ),
    "blackjack_player_sum_bins4_dealer_ace_hidden": (
        "b4e5577555d4c25425d1d33900e6726290ae7547c88e75b81a6504f613036c11"
    ),
}

SLIPPERY = "FrozenLakeSlippery-v0"
SLIPPERY_LADDER = (
    "frozenlake_slippery_column_pairs",
    "frozenlake_slippery_column_hidden",
    "frozenlake_slippery_column_hidden_row_pairs",
)
# Exact finite-horizon values from state 0, cap 100, intended 1/3 slips.
SLIPPERY_RANDOM_SCORE = 0.013939795959171435
SLIPPERY_OPTIMAL_SCORE = 0.7441902878292673
# FrozenLake action order.
FL_LEFT, FL_DOWN, FL_RIGHT, FL_UP = 0, 1, 2, 3

CLIFF = "CliffWalking-v0"
CLIFF_CANDIDATES = ("cliffwalking_row_pairs", "cliffwalking_column_triples")
CLIFF_START, CLIFF_GOAL, CLIFF_CAP = 36, 47, 200
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

BLACKJACK = "Blackjack-v1"
# restriction id -> (high player-sum bin width, reachable classes)
WIDE_BLACKJACK_BINS = {
    "blackjack_player_sum_bins6_dealer_ace_hidden": (6, 3),
    "blackjack_player_sum_bins8_dealer_ace_hidden": (8, 3),
    "blackjack_player_sum_bins10_dealer_ace_hidden": (10, 2),
}

# Digests of the new tables, frozen before any calibration run.
NEW_FROZEN_TABLES = {
    "frozenlake_slippery_column_pairs": (
        "ceeaea50bb590eb134d3a282573a8fbc2b08e9a9dd27b5d558d5f44a802f3b10"
    ),
    "frozenlake_slippery_column_hidden": (
        "e14eafe001fea5acee0c66e01bcd5ed223461994ba130de9c2e8c02164c6f1ea"
    ),
    "frozenlake_slippery_column_hidden_row_pairs": (
        "3cb6422a91a104ef9de2eda9c7ac4bd60391459523d293f3b02db247415c9cdb"
    ),
    "cliffwalking_row_pairs": (
        "37d71edc9db4c5736f59aadd934ebe0d84945ea063bde3607b007f7a06cc3976"
    ),
    "cliffwalking_column_triples": (
        "97c48e710daf86fe5027a06c7cf982f8bbe14d64e5d39bb62319679a059a5d03"
    ),
    "blackjack_player_sum_bins6_dealer_ace_hidden": (
        "d112edab4ca43f67e204df5fded9dc6a410a87abc3f00a1641639118d309f34a"
    ),
    "blackjack_player_sum_bins8_dealer_ace_hidden": (
        "8579086117518528767d40ba45abc9ffeb382589c5bbb798ead6d49421b2f210"
    ),
    "blackjack_player_sum_bins10_dealer_ace_hidden": (
        "43ea38ffd3d5043b1d035b675565a32e45308ed99053874d49c6a5273f863322"
    ),
}

# Reference baselines of the original environments and the varied start task.
UNCHANGED_REFERENCES = {
    "CartPole-v1": {"random_score": 22.0, "expert_score": 500.0},
    "FrozenLake-v1": {"random_score": 0.015, "expert_score": 1.0},
    "CliffWalking-v0": {"random_score": -56957.0, "expert_score": -13.0},
    "Acrobot-v1": {"random_score": -499.0, "expert_score": -85.0},
    "MountainCar-v0": {"random_score": -200.0, "expert_score": -110.0},
    "Taxi-v3": {"random_score": -763.0, "expert_score": 7.9},
    "Blackjack-v1": {"random_score": -0.40, "expert_score": -0.06},
    "LunarLander-v2": {"random_score": -176.0, "expert_score": 250.0},
    "FrozenLakeVariedStart-v0": {
        "random_score": 0.09037647670287287,
        "expert_score": 1.0,
    },
}


class RuleExpert:
    """Deterministic expert given by a rule on the full observation."""

    def __init__(self, rule):
        self.rule = rule

    def predict(self, obs, deterministic=True):
        assert deterministic
        obs = np.asarray(obs)
        return np.array([self.rule(o) for o in obs]), None


def _torch_mask(restriction_id, obs):
    with th.no_grad():
        return restriction.ObservationMask(restriction_id)(th.as_tensor(obs)).numpy()


def _table(env_name, **kwargs):
    env = gym.make(env_name, **kwargs)
    try:
        return {
            s: {a: [tuple(o) for o in outs] for a, outs in acts.items()}
            for s, acts in env.unwrapped.P.items()
        }
    finally:
        env.close()


def _start(n, state):
    return [Fraction(int(s == state)) for s in range(n)]


# ---------------------------------------------------------------------------
# Identities and registries
# ---------------------------------------------------------------------------


def test_original_tasks_ladders_and_tables_are_unchanged():
    assert tuple(tasks.TASKS) == ORIGINAL_ENVS
    assert env_utils.ENV_GROUPS["classical"] == list(ORIGINAL_ENVS)
    for env_name, digest in EXISTING_TASK_DIGESTS.items():
        assert tasks.task_sha256(tasks.get_task(env_name)) == digest
    for rid, digest in EXISTING_FROZEN_TABLES.items():
        assert tasks.FROZEN_TABLE_SHA256[rid] == digest
        assert tasks.alias_table(rid)["table_sha256"] == digest
    new = set(NEW_FROZEN_TABLES)
    for task in tasks.TASKS.values():
        assert not new & set(task.restriction_ids)
    # The old row-pair rule still hides columns and merges rows.
    key = tasks._grid_key(12, "row_pairs")
    assert key(0) == key(11) == key(23) == (0,)


def test_new_tables_are_frozen_and_registered():
    for rid, digest in NEW_FROZEN_TABLES.items():
        assert tasks.FROZEN_TABLE_SHA256[rid] == digest, rid
        record = tasks.alias_table(rid)
        assert record["table_sha256"] == digest
        assert isinstance(
            restriction.get_restriction(rid), restriction.AliasRestriction
        )


# Authorized random binning candidates, appended after the strategic ones.
TAXI = "Taxi-v3"
RANDOM_CANDIDATES = {
    env_name: tuple(f"{prefix}_random_bins{k}_seed20261003" for k in counts)
    for env_name, prefix, counts in (
        (CLIFF, "cliffwalking", (32, 28, 24, 20, 16, 12, 8, 4)),
        (SLIPPERY, "frozenlake_slippery", (9, 8, 7, 6, 5, 4, 3, 2)),
        (TAXI, "taxi", (360, 320, 280, 240, 200, 160, 120, 80, 40, 20, 10)),
        (
            BLACKJACK,
            "blackjack",
            (252, 224, 196, 168, 140, 112, 84, 56, 28, 14, 7),
        ),
    )
}


def test_calibration_candidates_extend_the_registry_off_the_ladders():
    # Strategic candidates keep their exact order as a prefix.
    cliff = tasks.calibration_candidates(CLIFF)
    assert cliff[: len(CLIFF_CANDIDATES)] == CLIFF_CANDIDATES
    blackjack = tasks.calibration_candidates(BLACKJACK)
    assert blackjack[:3] == (
        "blackjack_player_sum_bins2_dealer_ace_hidden",
        "blackjack_player_sum_bins3_dealer_ace_hidden",
        "blackjack_player_sum_bins4_dealer_ace_hidden",
    )
    n_strategic_blackjack = 3 + len(WIDE_BLACKJACK_BINS)
    assert blackjack[3:n_strategic_blackjack] == tuple(WIDE_BLACKJACK_BINS)
    # Exactly the random candidates follow; Taxi and slippery have only those.
    assert dict(tasks.RANDOM_ALIAS_CANDIDATES) == RANDOM_CANDIDATES
    assert cliff[len(CLIFF_CANDIDATES) :] == RANDOM_CANDIDATES[CLIFF]
    assert blackjack[n_strategic_blackjack:] == RANDOM_CANDIDATES[BLACKJACK]
    assert tasks.calibration_candidates(TAXI) == RANDOM_CANDIDATES[TAXI]
    assert tasks.calibration_candidates(SLIPPERY) == RANDOM_CANDIDATES[SLIPPERY]
    assert set(tasks.CALIBRATION_CANDIDATES) == {
        "MountainCar-v0",
        BLACKJACK,
        CLIFF,
        TAXI,
        SLIPPERY,
        # Random-offset grids (test_agnostic_random_grid.py) append these.
        "CartPole-v1",
        "Acrobot-v1",
        "LunarLander-v2",
    }
    for env_name in (CLIFF, BLACKJACK, TAXI, SLIPPERY):
        task = tasks.get_task(env_name)
        assert not set(tasks.calibration_candidates(env_name)) & set(
            task.restriction_ids,
        )


# ---------------------------------------------------------------------------
# FrozenLakeSlippery-v0
# ---------------------------------------------------------------------------


def test_slippery_env_is_the_standard_stochastic_frozenlake():
    env = gym.make(SLIPPERY)
    standard = gym.make("FrozenLake-v1", is_slippery=True)
    deterministic = gym.make("FrozenLake-v1", is_slippery=False)
    try:
        assert env.unwrapped.P == standard.unwrapped.P  # transitions and rewards
        assert env.unwrapped.P != deterministic.unwrapped.P
        np.testing.assert_array_equal(env.unwrapped.desc, standard.unwrapped.desc)
        np.testing.assert_array_equal(
            env.unwrapped.initial_state_distrib,
            standard.unwrapped.initial_state_distrib,
        )
        assert env.unwrapped.initial_state_distrib.tolist() == [1.0] + [0.0] * 15
        assert env.spec.max_episode_steps == 100
        assert env.spec.kwargs == {"map_name": "4x4", "is_slippery": True}
        assert tasks._tabular_dynamics_sha256(
            env.unwrapped,
        ) == tasks._tabular_dynamics_sha256(standard.unwrapped)
        # Every move slips: the intended and both perpendicular directions,
        # each with probability 1/3, on every nonterminal state.
        desc = env.unwrapped.desc.flatten()
        for s, acts in env.unwrapped.P.items():
            for outs in acts.values():
                if desc[s] in (b"H", b"G"):
                    assert outs == [(1.0, s, 0, True)]
                    continue
                assert len(outs) == 3 and all(p == 1.0 / 3.0 for p, *_ in outs)
                for _, nxt, reward, done in outs:
                    assert reward == float(desc[nxt] == b"G")
                    assert done == (desc[nxt] in (b"H", b"G"))
    finally:
        env.close()
        standard.close()
        deterministic.close()


def test_slippery_factory_and_task_declaration():
    assert SLIPPERY in env_utils.VARIANT_ENVS
    assert SLIPPERY not in env_utils.ENV_GROUPS["classical"]
    assert env_utils.ENV_CONFIGS[SLIPPERY]["obs_type"] == "discrete"
    assert env_utils.ENV_CONFIGS[SLIPPERY]["obs_size"] == 16
    assert tasks.effective_episode_cap(SLIPPERY) == 100
    task = tasks.get_task(SLIPPERY)
    assert SLIPPERY not in tasks.TASKS and tasks.VARIANT_TASKS[SLIPPERY] == task
    assert task.ladder == SLIPPERY_LADDER and task.ladder_rule == "nested"
    assert (task.episode_cap, task.blocks, task.n_actions) == (100, (16,), 4)
    record = tasks.describe(task)
    assert record["registered_kwargs"] == {"map_name": "4x4", "is_slippery": True}
    assert record["configured_env_kwargs"] == {}
    assert record["episode_cap"] == 100
    assert record["wrappers"][-1] == "FrozenLakeEnv"
    assert (
        "TimeLimit" in record["wrappers"] and "OneHotObsWrapper" in record["wrappers"]
    )
    assert record["reset_distribution"] == {
        "support": [0],
        "probabilities": [1.0] + [0.0] * 15,
        "rule": "fixed start state 0 (the map's S cell), as in FrozenLake-v1",
    }
    lake = tasks.describe(tasks.get_task("FrozenLake-v1"))
    assert record["tabular_dynamics_sha256"] != lake["tabular_dynamics_sha256"]
    assert record["tabular_dynamics_sha256"] == tasks._tabular_dynamics_sha256(
        gym.make("FrozenLake-v1", is_slippery=True).unwrapped,
    )
    # The deterministic original keeps its identity.
    assert tasks.task_sha256(tasks.get_task("FrozenLake-v1")) == (
        EXISTING_TASK_DIGESTS["FrozenLake-v1"]
    )
    assert "reset_distribution" not in lake


def test_slippery_factory_encodes_and_caps_episodes():
    venv = env_utils.make_env(SLIPPERY, 1, np.random.default_rng(0))
    try:
        obs = venv.reset()
        assert obs.shape == (1, 16) and int(obs[0].argmax()) == 0
        # Pressing LEFT slips along column 0 until the hole at 12 or the cap.
        steps = 0
        while True:
            _, _, done, info = venv.step(np.array([FL_LEFT]))
            steps += 1
            if done[0]:
                break
        assert steps <= 100
        if steps == 100:
            assert info[0]["TimeLimit.truncated"]
    finally:
        venv.close()


def test_slippery_reachability_and_alias_tables():
    reachable, evidence = tasks.reachable_nonterminal(SLIPPERY)
    assert reachable == [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]
    assert evidence["reset_support"] == [0]
    assert evidence["terminal_states_reached"] == [5, 7, 11, 12, 15]
    for rid, classes in zip(SLIPPERY_LADDER, (8, 4, 2)):
        original = rid.replace("frozenlake_slippery_", "frozenlake_")
        found = restriction.get_restriction(rid)
        record = tasks.alias_table(rid)
        assert record["env_name"] == SLIPPERY == found.env_name
        assert record["reachable_classes"] == classes
        assert record["hidden"] == tasks.ALIAS_RUNGS[original].hidden
        # Same reachable set as the deterministic lake, so the same tables.
        assert found.representative == (
            restriction.get_restriction(original).representative
        )
        obs = found.encode(np.array(reachable))
        masked = found.apply(obs)
        np.testing.assert_array_equal(found.apply(masked), masked)
        np.testing.assert_array_equal(_torch_mask(rid, obs), masked)
    assert task_audit.evidence_type(SLIPPERY) == "reachable_enumeration"
    record = task_audit.audit(
        RuleExpert(lambda o: int(o.argmax()) % 4),
        SLIPPERY,
        "frozenlake_slippery_column_hidden",
        301,
    )
    assert record["status"] == "conflict_found" and record["n_states"] == 11


def test_rational_transitions_recover_the_intended_thirds():
    P = env_baselines.rational_transitions(_table(SLIPPERY))
    probs = {p for acts in P.values() for outs in acts.values() for p, *_ in outs}
    assert probs == {Fraction(1, 3), Fraction(1)}
    for acts in P.values():
        for outs in acts.values():
            assert sum(p for p, *_ in outs) == 1
    with pytest.raises(ValueError, match="denominator"):
        env_baselines.rational_transitions({0: {0: [(0.1234567, 0, 0.0, True)]}})


def test_finite_horizon_value_of_a_fixed_policy_on_a_hand_solved_chain():
    # 0 -a0-> 1 -a0-> goal (reward 1); a1 stays.
    P = {
        0: {0: [(1.0, 1, 0.0, False)], 1: [(1.0, 0, 0.0, False)]},
        1: {0: [(1.0, 2, 1.0, True)], 1: [(1.0, 1, 0.0, False)]},
        2: {0: [(1.0, 2, 0.0, True)], 1: [(1.0, 2, 0.0, True)]},
    }
    value = env_baselines.finite_horizon_value
    go = {0: 0, 1: 0, 2: 0}
    assert value(P, [1, 0, 0], horizon=2, n_actions=2, policy=go) == 1
    assert value(P, [1, 0, 0], horizon=1, n_actions=2, policy=go) == 0
    assert value(P, [1, 0, 0], horizon=9, n_actions=2, policy={0: 0, 1: 1, 2: 0}) == 0
    with pytest.raises(ValueError, match="optimal"):
        value(P, [1, 0, 0], horizon=2, n_actions=2, optimal=True, policy=go)


def test_slippery_references_are_exact_finite_horizon_values():
    P = env_baselines.rational_transitions(_table(SLIPPERY))
    mu = _start(16, 0)
    random = env_baselines.finite_horizon_value(P, mu, horizon=100, n_actions=4)
    optimal = env_baselines.finite_horizon_value(
        P,
        mu,
        horizon=100,
        n_actions=4,
        optimal=True,
    )
    assert isinstance(random, Fraction) and isinstance(optimal, Fraction)
    assert float(random) == SLIPPERY_RANDOM_SCORE
    assert float(optimal) == SLIPPERY_OPTIMAL_SCORE
    assert 0 < optimal < 1
    assert env_baselines.REFERENCE_BASELINES[SLIPPERY] == {
        "random_score": SLIPPERY_RANDOM_SCORE,
        "expert_score": SLIPPERY_OPTIMAL_SCORE,
    }
    meta = env_baselines.reference_metadata(SLIPPERY)
    assert meta["normalization_reference_type"] == "uniform_random"
    assert meta["random_score_exact"] == str(random)
    assert meta["expert_score_exact"] == str(optimal)
    assert "nonstationary" in meta["expert_score_rule"]
    # The finite-horizon optimum is an upper benchmark: a stationary policy
    # (greedy in the undiscounted infinite-horizon values) falls short.
    v = {s: 0.0 for s in P}
    for _ in range(2000):
        v = {
            s: max(
                sum(float(p) * (r + (0 if d else v[n])) for p, n, r, d in P[s][a])
                for a in range(4)
            )
            for s in P
        }
    greedy = {
        s: max(
            range(4),
            key=lambda a, s=s: sum(
                float(p) * (r + (0 if d else v[n])) for p, n, r, d in P[s][a]
            ),
        )
        for s in P
    }
    stationary = env_baselines.finite_horizon_value(
        P,
        mu,
        horizon=100,
        n_actions=4,
        policy=greedy,
    )
    assert random < stationary < optimal
    ok, _ = env_baselines.validate_expert_quality(SLIPPERY, float(stationary))
    assert ok


def test_row_only_deterministic_policies_are_bounded_below_the_benchmark():
    P = env_baselines.rational_transitions(_table(SLIPPERY))
    mu = _start(16, 0)
    found = restriction.get_restriction("frozenlake_slippery_column_hidden")
    reachable, _ = tasks.reachable_nonterminal(SLIPPERY)
    reps = sorted({found.representative[s] for s in reachable})
    assert len(reps) == 4
    assert all(found.representative[s] // 4 == s // 4 for s in reachable)
    values = {}
    for actions in itertools.product(range(4), repeat=4):
        row_action = dict(zip(reps, actions))
        policy = {s: row_action.get(found.representative[s], 0) for s in range(16)}
        values[actions] = env_baselines.finite_horizon_value(
            P,
            mu,
            horizon=100,
            n_actions=4,
            policy=policy,
        )
    assert len(values) == 256
    best = max(values.values())
    optimal = env_baselines.finite_horizon_value(
        P,
        mu,
        horizon=100,
        n_actions=4,
        optimal=True,
    )
    # Bound for deterministic memoryless row-only policies only.
    assert 0 < best < optimal
    best_rows = [a for a, v in values.items() if v == best]
    assert best_rows == [(FL_LEFT, FL_LEFT, FL_DOWN, FL_RIGHT)]
    assert abs(float(best) - 0.16666563540941204) < 1e-15


# ---------------------------------------------------------------------------
# CliffWalking regular bins
# ---------------------------------------------------------------------------


def _cliff_optimal(s):
    """The full-observation shortest-path expert (13 steps from the start)."""
    row, col = divmod(int(s), 12)
    if s == CLIFF_START:
        return UP
    if col == 11:
        return DOWN
    if row == 2:
        return RIGHT
    return DOWN


def _cliff_expert():
    return RuleExpert(lambda o: _cliff_optimal(int(o.argmax())))


def _rollout(P, action_of, start=CLIFF_START, cap=CLIFF_CAP):
    s, total = start, 0
    for step in range(1, cap + 1):
        ((p, nxt, reward, done),) = P[s][action_of(s)]
        assert p == 1.0
        total += reward
        if done:
            return total, step, True
        s = nxt
    return total, cap, False


def _class_policy(restriction_id, choice):
    """Deterministic memoryless policy of the masked observation."""
    rep = restriction.get_restriction(restriction_id).representative
    return lambda s: choice(rep[s])


def _simple_goal_paths(P, max_len):
    """Lengths of every repetition-free state path from the start to the goal."""
    lengths = collections.Counter()
    paths = {}

    def walk(path, actions):
        s = path[-1]
        for a in range(4):
            ((_, nxt, _, done),) = P[s][a]
            if done:
                lengths[len(path)] += 1
                paths.setdefault(len(path), []).append((path, actions + [a]))
            elif nxt not in path and len(path) < max_len:
                walk(path + [nxt], actions + [a])

    walk([CLIFF_START], [])
    return lengths, paths


@pytest.mark.parametrize(
    "restriction_id, hidden, classes",
    [
        ("cliffwalking_row_pairs", "row within pairs {0,1}, {2,3}", 24),
        (
            "cliffwalking_column_triples",
            "column within triples {3k, 3k+1, 3k+2}",
            13,
        ),
    ],
)
def test_cliff_bins_are_regular_axis_bins(restriction_id, hidden, classes):
    rows, cols = {"cliffwalking_row_pairs": (2, 1)}.get(restriction_id, (1, 3))
    found = restriction.get_restriction(restriction_id)
    record = tasks.alias_table(restriction_id)
    assert record["hidden"] == hidden and record["env_name"] == CLIFF
    reachable, _ = tasks.reachable_nonterminal(CLIFF)
    assert len(reachable) == 37
    assert record["reachable_classes"] == classes
    by_class = collections.defaultdict(set)
    for s in reachable:
        r, c = divmod(s, 12)
        by_class[(r // rows, c // cols)].add(s)
    assert len(by_class) == classes
    for members in by_class.values():
        assert {found.representative[s] for s in members} == {min(members)}
    obs = found.encode(np.array(reachable))
    masked = found.apply(obs)
    np.testing.assert_array_equal(found.apply(masked), masked)
    np.testing.assert_array_equal(_torch_mask(restriction_id, obs), masked)
    assert tasks._grid_bins(12, rows, cols)(CLIFF_START) == (3 // rows, 0)


def test_cliff_row_pairs_merge_the_start_with_the_row_above():
    record = task_audit.audit(_cliff_expert(), CLIFF, "cliffwalking_row_pairs", 301)
    assert record["evidence_type"] == "reachable_enumeration"
    conflicts = [c for c in record["classes"] if c["conflict"]]
    assert len(conflicts) == 1
    assert conflicts[0]["states"] == [24, 36]
    assert conflicts[0]["expert_actions"] == [RIGHT, UP]


def test_cliff_row_pairs_masked_minimum_is_fifteen_steps():
    P = _table(CLIFF)
    assert _rollout(P, _cliff_optimal) == (-13, 13, True)
    # A goal-reaching run of a deterministic memoryless policy in this
    # deterministic environment never repeats a state (a repeat loops forever,
    # and a cliff fall repeats the start), so it is a simple path. Simple
    # paths move one cell per step, flipping (row + col) parity, and the start
    # and goal differ in parity: every goal path has odd length.
    lengths, paths = _simple_goal_paths(P, max_len=14)
    assert min(lengths) == 13 and lengths[13] == 1 and lengths[14] == 0
    ((path, actions),) = paths[13]
    assert path[:2] == [CLIFF_START, 24] and actions[:2] == [UP, RIGHT]
    rep = restriction.get_restriction("cliffwalking_row_pairs").representative
    assert rep[CLIFF_START] == rep[24]  # one masked observation, two actions

    # An explicit 15-step route the masked policy can follow: up twice, right
    # along row 1, down twice in the last column.
    def choice(r):
        row, col = divmod(r, 12)
        if col == 11:
            return DOWN
        return UP if row >= 2 else RIGHT

    policy = _class_policy("cliffwalking_row_pairs", choice)
    assert _rollout(P, policy) == (-15, 15, True)
    # Two steps short of optimal: just below 1 against the uniform reference.
    ceiling = expert_training._normalize(-15.0, CLIFF)
    assert ceiling == pytest.approx(56942 / 56944, abs=1e-12) and ceiling < 1


@pytest.mark.parametrize(
    "restriction_id, merged",
    [
        ("cliffwalking_column_pairs", [34, 35]),
        ("cliffwalking_column_triples", [33, 34, 35]),
    ],
)
def test_cliff_column_bins_merge_the_last_turn(restriction_id, merged):
    if restriction_id in tasks.TASKS[CLIFF].ladder:
        assert restriction_id not in tasks.calibration_candidates(CLIFF)
    record = task_audit.audit(_cliff_expert(), CLIFF, restriction_id, 301)
    conflicts = [c for c in record["classes"] if c["conflict"]]
    assert len(conflicts) == 1 and conflicts[0]["states"] == merged
    assert conflicts[0]["distinct_actions"] == [RIGHT, DOWN]
    # The expert's label at each class representative, run under the mask:
    # it walks to the last turn and stalls against the wall. This is one
    # controller, not a bound on the masked policy class.
    P = _table(CLIFF)
    policy = _class_policy(restriction_id, _cliff_optimal)
    assert _rollout(P, policy) == (-CLIFF_CAP, CLIFF_CAP, False)
    # Stalling is the diagnostic benchmark, not the zero of the scale.
    assert expert_training._normalize(-200.0, CLIFF) == pytest.approx(56757 / 56944)


def test_cliff_row_pairs_representative_controller_is_worse_than_stalling():
    # The representative of the start's class is 24, labelled RIGHT: from the
    # start that walks into the cliff and back, every time.
    P = _table(CLIFF)
    policy = _class_policy("cliffwalking_row_pairs", _cliff_optimal)
    total, steps, reached = _rollout(P, policy)
    assert (steps, reached) == (CLIFF_CAP, False)
    assert total == -100 * CLIFF_CAP
    normalized = expert_training._normalize(float(total), CLIFF)
    stall = env_baselines.STALL_BENCHMARK_RETURNS[CLIFF]
    assert normalized < expert_training._normalize(stall, CLIFF)
    assert normalized == pytest.approx((total + 56957) / 56944)


# ---------------------------------------------------------------------------
# Blackjack wider player-sum bins
# ---------------------------------------------------------------------------


def _player_class(player_sum, width):
    return "low" if player_sum < 12 else (player_sum - 12) // width


@pytest.mark.parametrize("restriction_id", sorted(WIDE_BLACKJACK_BINS))
def test_wide_blackjack_bins_merge_sums_and_hide_dealer_and_ace(restriction_id):
    width, n_classes = WIDE_BLACKJACK_BINS[restriction_id]
    found = restriction.get_restriction(restriction_id)
    record = tasks.alias_table(restriction_id)
    assert record["env_name"] == BLACKJACK
    assert f"(sum - 12) // {width}" in record["hidden"]
    reachable, _ = tasks.reachable_nonterminal(BLACKJACK)
    table = np.asarray(found.representative)
    assert table.min() >= 0 and table.max() < 32 * 11 * 2
    by_class = collections.defaultdict(set)
    for s in reachable:
        by_class[_player_class(tasks.blackjack_factors(s)[0], width)].add(s)
    assert len(by_class) == record["reachable_classes"] == n_classes
    for members in by_class.values():
        assert {int(table[s]) for s in members} == {min(members)}
        assert table[min(members)] == min(members)
    obs = found.encode(np.array(reachable))
    masked = found.apply(obs)
    np.testing.assert_array_equal(found.apply(masked), masked)
    np.testing.assert_array_equal(_torch_mask(restriction_id, obs), masked)


@pytest.mark.parametrize("restriction_id", sorted(WIDE_BLACKJACK_BINS))
def test_wide_blackjack_bins_conflict_where_a_threshold_is_merged(restriction_id):
    width, _ = WIDE_BLACKJACK_BINS[restriction_id]

    def threshold_17(o):
        return int(o[:32].argmax() < 17)

    record = task_audit.audit(RuleExpert(threshold_17), BLACKJACK, restriction_id, 301)
    assert record["evidence_type"] == "reachable_enumeration"
    conflicts = [c for c in record["classes"] if c["conflict"]]
    assert len(conflicts) == 1
    merged = {tasks.blackjack_factors(s)[0] for s in conflicts[0]["states"]}
    assert merged == {s for s in range(12, 22) if (s - 12) // width == 5 // width}
    assert {16, 17} <= merged


# ---------------------------------------------------------------------------
# CliffWalking uniform-random reference and diagnostic stall benchmark
# ---------------------------------------------------------------------------


def test_cliff_reference_is_uniform_random_and_stall_is_a_benchmark():
    for env_name, reference in UNCHANGED_REFERENCES.items():
        assert env_baselines.REFERENCE_BASELINES[env_name] == reference
        meta = env_baselines.reference_metadata(env_name)
        assert meta["normalization_reference_type"] == "uniform_random"
    meta = env_baselines.reference_metadata(CLIFF)
    assert meta["stall_benchmark_return"] == -200.0
    assert "never a lower normalization reference" in meta["stall_benchmark_rule"]
    assert "not measured under the current 200-step cap" in meta["random_score_rule"]
    # A 200-step episode returns at least -100 per step, so the historic
    # score cannot come from the current cap.
    assert env_baselines.REFERENCE_BASELINES[CLIFF]["random_score"] < -100 * CLIFF_CAP
    assert env_baselines.STALL_BENCHMARK_RETURNS == {CLIFF: -float(CLIFF_CAP)}


def test_normalization_reference_overrides_nothing():
    for env_name in (*UNCHANGED_REFERENCES, None, "CliffWalkingVariedStart-v0"):
        assert env_baselines.normalization_reference(env_name, -61234.5) == {}


def test_cliff_expert_qualification_uses_the_uniform_random_reference():
    threshold = -56957.0 + env_baselines.EXPERT_QUALITY_THRESHOLD * 56944.0
    assert env_baselines.validate_expert_quality(CLIFF, -13.0)[0]
    assert env_baselines.validate_expert_quality(CLIFF, threshold + 1.0)[0]
    assert not env_baselines.validate_expert_quality(CLIFF, threshold - 1.0)[0]
    assert expert_training._normalize(-13.0, CLIFF) == 1.0
    assert expert_training._normalize(-56957.0, CLIFF) == 0.0


def _untrained_policy(env_name):
    venv = env_utils.make_env(env_name, 1, np.random.default_rng(0))
    policy = PPO("MlpPolicy", venv, device="cpu", seed=0).policy
    venv.close()
    return policy


@pytest.mark.parametrize("env_name", [CLIFF, "FrozenLake-v1"])
def test_compute_baselines_keeps_the_uniform_random_rollout(env_name):
    policy = _untrained_policy(env_name)
    venv = env_utils.make_env(env_name, 1, np.random.default_rng(0))
    venv.seed(0)
    venv.action_space.seed(0)
    try:
        baselines = env_baselines.compute_baselines(
            policy,
            venv,
            np.random.default_rng(0),
            n_expert_episodes=2,
            n_random_episodes=3,
        )
    finally:
        venv.close()
    assert set(baselines) == {"expert_return", "random_return", "expert_self_ce"}
