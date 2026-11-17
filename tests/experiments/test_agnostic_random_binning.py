"""Random state binning calibration candidates.

Covers the pure partition function (`tasks.random_state_partition`), the
frozen random binning registry, its static partition descriptor, the
`restriction.AliasRestriction` numpy and torch forms on every reachable
input, the runner and audit opt-in, and restricted checkpoints. Every test is
synthetic or protocol level: environments are built locally, experts are
rule-based or untrained, and nothing is trained or launched.
"""

import random
import signal

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO

from imitation.experiments.agnostic import classical, restriction
from imitation.experiments.agnostic import restriction_pilot as rp
from imitation.experiments.agnostic import rollouts, task_audit, task_pilot, tasks
from imitation.experiments.ftrl import env_utils

SEED = 20261003
CLIFF = "CliffWalking-v0"
SLIPPERY = "FrozenLakeSlippery-v0"
TAXI = "Taxi-v3"
BLACKJACK = "Blackjack-v1"
# Predeclared bin counts and the actual reachable nonterminal state counts.
COUNTS = {
    CLIFF: (32, 28, 24, 20, 16, 12, 8, 4),
    SLIPPERY: (9, 8, 7, 6, 5, 4, 3, 2),
    TAXI: (360, 320, 280, 240, 200, 160, 120, 80, 40, 20, 10),
    BLACKJACK: (252, 224, 196, 168, 140, 112, 84, 56, 28, 14, 7),
}
N_REACHABLE = {CLIFF: 37, SLIPPERY: 11, TAXI: 400, BLACKJACK: 280}
PREFIX = {
    CLIFF: "cliffwalking",
    SLIPPERY: "frozenlake_slippery",
    TAXI: "taxi",
    BLACKJACK: "blackjack",
}
RANDOM_IDS = {
    env: tuple(f"{PREFIX[env]}_random_bins{k}_seed{SEED}" for k in ks)
    for env, ks in COUNTS.items()
}
ALL_RANDOM = tuple(rid for ids in RANDOM_IDS.values() for rid in ids)
ENV_OF = {rid: env for env, ids in RANDOM_IDS.items() for rid in ids}

# Static pins, derived from the pinned generator and reachable sets before any
# calibration run. They must never move.
EXPECTED_REACHABLE = {
    CLIFF: (
        "2dc009c14ef334c834b0f1d3f5c286d209e72fe743d797b1af008acaad7c2b7c",
        "e0f6025773eaf3853673a5b9a5b14270dd324c2a69301517c016af87d9616635",
    ),
    SLIPPERY: (
        "bddd533c2f0960bc18030394aebf6b2a4da5b13325fad622a92018077cd32c9d",
        "66d4fe125bc59d2fe81ef643312c26a3bfc5f7e74e030757511652c026c443aa",
    ),
    TAXI: (
        "df52ebb3e452304deee1b2ebd2f6fc51764eeb8d2f92ed5e639fbe233fee9e0e",
        "3881a6349e32fd951e38458cea65a6eeaa1fffda53b200524ac3517a2f72dbce",
    ),
    BLACKJACK: (
        "9f0f4e37f5f6d704f03ebb8eda5ca1f45dc97882956f616861685886557a0b73",
        "bbb207e3113a284b80cdf24b26637d61d16efd622dd96c958d93e703c1bbd72c",
    ),
}
EXPECTED_TABLES = {
    "cliffwalking_random_bins32_seed20261003": (
        "72833be47d149afaf7a56ff02608a7eefa07e181f69c6403159a54f18553c184"
    ),
    "cliffwalking_random_bins28_seed20261003": (
        "4a4704347bc453a13db60f3b96487e09a13a473ec6644279cbfeeef2f0b8f027"
    ),
    "cliffwalking_random_bins24_seed20261003": (
        "fa36ac6234e9b89f00448b35f0a3231b5ba4b2acf67bbf43431408558b93eb44"
    ),
    "cliffwalking_random_bins20_seed20261003": (
        "72667ae39f08062c81704bd9719bc4134cca29633f9c89f11b2768d44fadcb71"
    ),
    "cliffwalking_random_bins16_seed20261003": (
        "06fdb2dfe71ce12f79bd7c8b9804cfa30ba0a3ffd1ccc56a8e36e14fd43d03a9"
    ),
    "cliffwalking_random_bins12_seed20261003": (
        "86dba59bcddff0d03757a79213c39b566f20c5b4dfa40ba0e67acdbf985e7700"
    ),
    "cliffwalking_random_bins8_seed20261003": (
        "164768a60098c5a763df4a59f91ad51015a25fa6f161343970224992c75fe698"
    ),
    "cliffwalking_random_bins4_seed20261003": (
        "f3884cab7ca2aab5f65754d78d187035050f580e52183ae16721028cc028a9fd"
    ),
    "frozenlake_slippery_random_bins9_seed20261003": (
        "4df0feea2c6ea08f4104e7d30686dd7fbce4ea09a6c65fdeff6e3681d617968f"
    ),
    "frozenlake_slippery_random_bins8_seed20261003": (
        "29639b9826c472d672360c24a2bdced3437ebd9af85e4fc2e94eb094f8506219"
    ),
    "frozenlake_slippery_random_bins7_seed20261003": (
        "d2e20f6e1cd7eb803f9ceea473b38b858ac1f52ff46711e0c7302ce7e1f9fffc"
    ),
    "frozenlake_slippery_random_bins6_seed20261003": (
        "f9510804cb48643709c9c2b5bb7be8830ce581f3174cbf648d9ead959ca17274"
    ),
    "frozenlake_slippery_random_bins5_seed20261003": (
        "9d5b9ead16e3d9844f03debf7bc5904147c456cf40f150fcfdc8366234dbd626"
    ),
    "frozenlake_slippery_random_bins4_seed20261003": (
        "eaf8d4a72bf6c7a50b808358a4bf4aaf6142a138fde1ab44a372735621ebb835"
    ),
    "frozenlake_slippery_random_bins3_seed20261003": (
        "357766baebf80f9479a4001febc02273a0acf1f8094ee1b83410c78e9fe5a8aa"
    ),
    "frozenlake_slippery_random_bins2_seed20261003": (
        "58cf1dfe9e2a705cc3f2f132700bc9e9a0df1514e12adf27d4e48622f0f1baa4"
    ),
    "taxi_random_bins360_seed20261003": (
        "2fbff26d6bdb806c7f32ad823df5a4b6158ac729c88aceeb8d76044115a06d20"
    ),
    "taxi_random_bins320_seed20261003": (
        "51ef6020801b1cc8f115e4c202b51e68244f9ee55260f4fa1d98e443a815873c"
    ),
    "taxi_random_bins280_seed20261003": (
        "94a2fa5d515716f748cc1f546a4f5badb63196ee6914b8d7f1ce44d2053b2838"
    ),
    "taxi_random_bins240_seed20261003": (
        "1ab225c77844ac67cc260b1a57e05fdbf26ba98a41b8a50ce4994740e0bf44d4"
    ),
    "taxi_random_bins200_seed20261003": (
        "cf1ddcdf2d083ae02889ca2ae119c68c72cb21f59edc61d4e5b663d556bc134c"
    ),
    "taxi_random_bins160_seed20261003": (
        "7bd15e21219b5d86a19c1a62b62efaa899a7b8638cee1ea57ada3a23c292e32b"
    ),
    "taxi_random_bins120_seed20261003": (
        "ca0f907d7fe87fb0b1452e32c0208c338a2b9c1d63fc87bc62e74c518f3cf9ef"
    ),
    "taxi_random_bins80_seed20261003": (
        "74715ec72e402c8b6b39d542e45991c03d2686b6e02eb8e70b4510ba2add6aa1"
    ),
    "taxi_random_bins40_seed20261003": (
        "ff212052e179721f87f9c7139141d309f53323fe2281e5c894d10380e0b60162"
    ),
    "taxi_random_bins20_seed20261003": (
        "fd0d2704748f15e29551bb0c5859c1c38a1091f918cdb63b71a22b84446d523f"
    ),
    "taxi_random_bins10_seed20261003": (
        "ea363b805aae1a91955a3880e93bd33014c69f7d5988f9f39b4417483aeb52ba"
    ),
    "blackjack_random_bins252_seed20261003": (
        "be4e7979ca3d738a0bc1220f4be1bfcaef0861b8744965da00cad1393e5fc5aa"
    ),
    "blackjack_random_bins224_seed20261003": (
        "98159592277d22bfe4a7200a80fceedf4acd0062fe12118a6f8021a8706bb4b6"
    ),
    "blackjack_random_bins196_seed20261003": (
        "8efc0c8dcf5b13e11bcfe554fe90a42f325ce932298180ce1579031dbcc34618"
    ),
    "blackjack_random_bins168_seed20261003": (
        "e16cf1601f5177b9cc9271d9b2f3b5ccadc02d6f6fa76d0114a3b77d56da304c"
    ),
    "blackjack_random_bins140_seed20261003": (
        "ae7eb5a92c3dc3d4338ed1e98700a91b0a677dc8d46881e291a0162956504c7a"
    ),
    "blackjack_random_bins112_seed20261003": (
        "71e22cfce62f63e813c39ed9dc73fe8a12c8e7fa9a546102f03792bce1a8022a"
    ),
    "blackjack_random_bins84_seed20261003": (
        "96f52f692fb26a1923cf96530f342a64417b3a237b57decec9f77348eb5d357f"
    ),
    "blackjack_random_bins56_seed20261003": (
        "833b04e6a38598bb8044fa40329c73738109f4ed10c613b223ad710b6a3581e8"
    ),
    "blackjack_random_bins28_seed20261003": (
        "cbbe132d3c8097bb1f42304d00bbdcb6817a18d6a01715a25ff611ba716a1491"
    ),
    "blackjack_random_bins14_seed20261003": (
        "74df0b7d6092bef370f3c701d50d22a64aa6ddc69d4861dcd589027b3583e126"
    ),
    "blackjack_random_bins7_seed20261003": (
        "701a89eec750bf22c7e37ecc021ef683e3cfbcd909b3e5ad6672359fd8222f62"
    ),
}

# Identities recorded before random binning; they must never move.
EXISTING_TASK_DIGESTS = {
    CLIFF: "01127136af3841e64bc22eb04373553af3b8db159c9c6ba8f20f24348fd24967",
    TAXI: "274c50730f86271b93e53ccf1c37878ee6e7b80d377304e055a6cfe8c7fca422",
    BLACKJACK: "f6877a8cfdbe70ad61bd28f4092b4d15e04bcae0dd54cd180fd3fb1ab80182fb",
    SLIPPERY: "bc28cf88ddaccd9ba699702202bf91544f4c42108297786460b86dd8307faaa5",
}
EXISTING_RESTRICTION_DIGESTS = {
    "cliffwalking_column_pairs": (
        "5ff782bb293cdb1523cb9b524681ed95d1e76ba2c64f131de4c4539e4e645d09"
    ),
    "taxi_destination_hidden": (
        "c25c8391a5bb1618938f34147625ae95834e5659cbfbc0c059234d8b74197f43"
    ),
    "blackjack_usable_ace_hidden": (
        "6261444c57104673d8e18ded1e7c35e85077edc7a9d8607e9cc92def51cbfafe"
    ),
    "frozenlake_slippery_column_pairs": (
        "5a2ee25d04f94b395f74917d5aef275d4be0ff0560b42770ddfdf051a8dabf43"
    ),
}
STRATEGIC_CANDIDATES = {
    "MountainCar-v0": (
        "mountaincar_bins_pos005_vel005",
        "mountaincar_bins_pos005_vel015",
        "mountaincar_bins_pos010_vel005",
        "mountaincar_bins_pos010_vel015",
    ),
    BLACKJACK: tuple(
        f"blackjack_player_sum_bins{w}_dealer_ace_hidden" for w in (2, 3, 4, 6, 8, 10)
    ),
    CLIFF: ("cliffwalking_row_pairs", "cliffwalking_column_triples"),
}


@pytest.fixture(autouse=True)
def _sixty_second_limit():
    """Fail any test of this module that runs longer than 60 seconds."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def expire(signum, frame):
        raise TimeoutError("test exceeded 60 seconds")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(60)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


class RuleExpert:
    """Deterministic expert given by a rule on the full observation."""

    def __init__(self, rule):
        self.rule = rule

    def predict(self, obs, deterministic=True):
        assert deterministic
        return np.array([self.rule(o) for o in np.asarray(obs)]), None


def _random_expert(env_name, seed=0):
    venv = env_utils.make_env(env_name, 1, np.random.default_rng(0))
    policy = PPO("MlpPolicy", venv, device="cpu", seed=seed).policy
    venv.close()
    return policy


def _rng_states():
    return (
        np.random.get_state(),
        th.get_rng_state().clone(),
        random.getstate(),
    )


def _assert_rng_unchanged(before):
    after = _rng_states()
    legacy_a, legacy_b = before[0], after[0]
    assert legacy_a[0] == legacy_b[0]
    np.testing.assert_array_equal(legacy_a[1], legacy_b[1])
    assert legacy_a[2:] == legacy_b[2:]
    assert th.equal(before[1], after[1])
    assert before[2] == after[2]


# ---------------------------------------------------------------------------
# Pure partition function
# ---------------------------------------------------------------------------


def test_partition_golden_values_pin_the_generator():
    # Pinned output of random_state_binning/1 (legacy RandomState stream).
    assert tasks.random_state_partition(range(10), 3, SEED) == (
        (5, 8, 6, 3),
        (0, 7, 4),
        (1, 9, 2),
    )
    expected = np.random.RandomState(SEED).permutation(np.arange(37))
    flat = sum(tasks.random_state_partition(range(37), 4, SEED), ())
    np.testing.assert_array_equal(flat, expected)
    assert tasks.RANDOM_BINNING_GENERATOR == "random_state_binning/1"


@pytest.mark.parametrize("n_states", [1, 2, 11, 37, 100])
def test_partition_is_balanced_with_exactly_k_nonempty_disjoint_bins(n_states):
    states = list(range(3, 3 + 2 * n_states, 2))  # sparse, not 0..n-1
    for k in range(1, n_states + 1):
        bins = tasks.random_state_partition(states, k, SEED)
        assert len(bins) == k
        sizes = sorted(len(b) for b in bins)
        assert sizes[0] >= 1
        assert sizes[-1] - sizes[0] <= 1
        assert sum(sizes) == n_states
        assert sorted(s for b in bins for s in b) == states


def test_partition_endpoints_one_bin_and_singletons():
    states = list(range(37))
    (only,) = tasks.random_state_partition(states, 1, SEED)
    assert sorted(only) == states
    table = tasks.random_alias_table(48, states, 1, SEED)
    assert set(table[s] for s in states) == {only[0]}
    assert table[37:] == tuple(range(37, 48))

    singletons = tasks.random_state_partition(states, 37, SEED)
    assert all(len(b) == 1 for b in singletons)
    assert tasks.random_alias_table(48, states, 37, SEED) == tuple(range(48))


def test_table_is_idempotent_first_member_and_identity_outside():
    states = [1, 4, 5, 9, 12, 13, 20, 30]
    for k in range(1, len(states) + 1):
        bins = tasks.random_state_partition(states, k, SEED)
        table = tasks.random_alias_table(32, states, k, SEED)
        assert len(table) == 32
        for s in range(32):
            assert table[table[s]] == table[s]
            if s not in states:
                assert table[s] == s
        for b in bins:
            assert all(table[s] == b[0] for s in b)
        assert len({table[s] for s in states}) == k


def test_partition_is_reproducible_order_free_and_seed_dependent():
    states = list(range(37))
    first = tasks.random_state_partition(states, 8, SEED)
    shuffled = list(states)
    random.Random(1).shuffle(shuffled)
    assert tasks.random_state_partition(shuffled, 8, SEED) == first
    assert tasks.random_state_partition(np.array(states), 8, SEED) == first
    assert tasks.random_state_partition(tuple(states), 8, np.uint32(SEED)) == first
    by_seed = {tasks.random_state_partition(states, 8, s) for s in range(6)}
    assert len(by_seed) == 6
    # One permutation per seed, reused across bin counts.
    for k in (1, 4, 8, 20, 37):
        flat = sum(tasks.random_state_partition(states, k, SEED), ())
        assert flat == sum(first, ())


def test_partition_draws_no_ambient_random_numbers():
    np.random.seed(5)
    th.manual_seed(5)
    random.seed(5)
    before = _rng_states()
    first = tasks.random_state_partition(range(400), 40, SEED)
    tasks.random_alias_table(500, range(400), 40, SEED)
    _assert_rng_unchanged(before)
    # A different ambient state gives the same partition.
    np.random.seed(6)
    th.manual_seed(6)
    assert tasks.random_state_partition(range(400), 40, SEED) == first


@pytest.mark.parametrize(
    "states, n_bins, seed, n_states, match",
    [
        ([0, 1, 2], True, SEED, None, "n_bins must be an integer"),
        ([0, 1, 2], 2.0, SEED, None, "n_bins must be an integer"),
        ([0, 1, 2], np.float64(2), SEED, None, "n_bins must be an integer"),
        ([0, 1, 2], "2", SEED, None, "n_bins must be an integer"),
        ([0, 1, 2], 0, SEED, None, "outside 1..3"),
        ([0, 1, 2], -1, SEED, None, "outside 1..3"),
        ([0, 1, 2], 4, SEED, None, "outside 1..3"),
        ([0, 1, 2], 2, -1, None, "seed -1 is outside"),
        ([0, 1, 2], 2, 2**32, None, "outside 0..4294967295"),
        ([0, 1, 2], 2, False, None, "seed must be an integer"),
        ([0, 1, 2], 2, 1.0, None, "seed must be an integer"),
        ([0, True, 2], 2, SEED, None, "state must be an integer"),
        ([0, 1.0, 2], 2, SEED, None, "state must be an integer"),
        (np.array([0.0, 1.0]), 1, SEED, None, "state must be an integer"),
        (np.array([True, False]), 1, SEED, None, "state must be an integer"),
        (np.array([[0, 1]]), 1, SEED, None, "1-D"),
        ("012", 1, SEED, None, "sequence of integers"),
        ([], 1, SEED, None, "must not be empty"),
        ([0, 1, 1], 2, SEED, None, "must not repeat"),
        ([0, -1, 2], 2, SEED, None, "negative"),
        ([0, 1, 5], 2, SEED, 5, "not below n_states 5"),
        ([0, 1, 2], 2, SEED, 3.0, "n_states must be an integer"),
    ],
)
def test_partition_refuses_invalid_inputs(states, n_bins, seed, n_states, match):
    with pytest.raises(ValueError, match=match):
        tasks.random_state_partition(states, n_bins, seed, n_states=n_states)


def test_alias_table_refuses_out_of_range_states():
    with pytest.raises(ValueError, match="not below n_states 4"):
        tasks.random_alias_table(4, [0, 4], 1, SEED)
    with pytest.raises(ValueError, match="n_states must be an integer"):
        tasks.random_alias_table(True, [0], 1, SEED)


# ---------------------------------------------------------------------------
# Registry, frozen digests and static descriptor
# ---------------------------------------------------------------------------


def test_registry_ids_counts_and_actual_state_counts():
    assert tasks.RANDOM_BINNING_SEED == SEED
    assert dict(tasks.RANDOM_BINNING_COUNTS) == COUNTS
    assert dict(tasks.RANDOM_ALIAS_CANDIDATES) == RANDOM_IDS
    assert tuple(tasks.RANDOM_ALIAS_SPECS) == ALL_RANDOM
    assert "cliffwalking_random_bins32_seed20261003" in ALL_RANDOM
    for env_name, ks in COUNTS.items():
        states, _ = tasks.reachable_nonterminal(env_name)
        assert len(states) == N_REACHABLE[env_name]
        assert tasks.FROZEN_REACHABLE_SHA256[env_name]["n_states"] == len(states)
        assert all(1 < k < len(states) for k in ks)
        assert list(ks) == sorted(set(ks), reverse=True)
        for rid, k in zip(RANDOM_IDS[env_name], ks):
            spec = tasks.RANDOM_ALIAS_SPECS[rid]
            assert (spec.env_name, spec.n_bins, spec.seed) == (env_name, k, SEED)
            assert spec.generator == tasks.RANDOM_BINNING_GENERATOR


def test_frozen_digests_are_static_literals():
    for env_name, (states_sha, evidence_sha) in EXPECTED_REACHABLE.items():
        pin = tasks.FROZEN_REACHABLE_SHA256[env_name]
        assert (pin["states_sha256"], pin["evidence_sha256"]) == (
            states_sha,
            evidence_sha,
        )
        states, evidence = tasks.reachable_nonterminal(env_name)
        assert tasks.reachable_digests(states, evidence) == pin
    assert set(EXPECTED_TABLES) == set(ALL_RANDOM)
    for rid, digest in EXPECTED_TABLES.items():
        assert tasks.FROZEN_TABLE_SHA256[rid] == digest
        record = tasks.alias_table(rid)
        assert record["table_sha256"] == record["frozen_table_sha256"] == digest


@pytest.mark.parametrize("restriction_id", ALL_RANDOM)
def test_alias_record_binds_partition_membership_and_table(restriction_id):
    env_name = ENV_OF[restriction_id]
    task = tasks.get_task(env_name)
    states, evidence = tasks.reachable_nonterminal(env_name)
    record = tasks.alias_table(restriction_id)
    spec = tasks.RANDOM_ALIAS_SPECS[restriction_id]
    groups = tasks.random_state_partition(states, spec.n_bins, SEED)
    assert record["groups"] == [list(g) for g in groups]
    assert record["blocks"] == list(task.blocks)
    assert record["reachable_nonterminal_states"] == len(states)
    assert record["reachable_classes"] == spec.n_bins
    n = int(np.prod(task.blocks))
    assert record["identity_classes_without_reachable_state"] == n - len(states)
    assert record["reachability"] == evidence
    table = record["representative"]
    assert tuple(table) == tasks.random_alias_table(n, states, spec.n_bins, SEED)
    assert tasks.table_sha256(task.blocks, tuple(table)) == record["table_sha256"]
    assert record["partition"] == tasks.random_partition_descriptor(restriction_id)
    assert record["partition_sha256"] == tasks._digest(record["partition"])
    assert record["partition_sha256"] == tasks.random_partition_sha256(
        restriction_id,
    )
    assert "first" in record["hidden"]


def test_static_descriptor_needs_no_environment(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("environment built")

    monkeypatch.setattr(gym, "make", refuse)
    monkeypatch.setattr(env_utils, "make_env", refuse)
    monkeypatch.setattr(tasks, "reachable_nonterminal", refuse)
    for rid in ALL_RANDOM:
        env_name = ENV_OF[rid]
        descriptor = tasks.random_partition_descriptor(rid)
        assert descriptor["kind"] == "random_state_binning"
        assert descriptor["generator"] == "random_state_binning/1"
        assert descriptor["restriction_id"] == rid
        assert descriptor["env_name"] == env_name
        assert descriptor["seed"] == SEED
        assert descriptor["n_bins"] == tasks.RANDOM_ALIAS_SPECS[rid].n_bins
        assert descriptor["n_reachable_nonterminal"] == N_REACHABLE[env_name]
        assert descriptor["reachable_nonterminal_sha256"] == (
            EXPECTED_REACHABLE[env_name][0]
        )
        assert descriptor["reachability_evidence_sha256"] == (
            EXPECTED_REACHABLE[env_name][1]
        )
        assert descriptor["frozen_table_sha256"] == EXPECTED_TABLES[rid]
        assert descriptor["nested_across_bin_counts"] is False
        assert descriptor["blocks"] == list(tasks.get_task(env_name).blocks)
    for bad in ("taxi_destination_hidden", "nope", None, 3):
        with pytest.raises(ValueError, match="not a random binning"):
            tasks.random_partition_descriptor(bad)


def test_random_binning_id_is_exact_and_strict():
    assert tasks.random_binning_id(CLIFF, 20, SEED) == (
        "cliffwalking_random_bins20_seed20261003"
    )
    assert tasks.random_binning_id(TAXI, np.int64(40), np.uint32(SEED)) == (
        "taxi_random_bins40_seed20261003"
    )
    # A dynamic id may name an unregistered count; lookups still refuse it.
    unregistered = tasks.random_binning_id(CLIFF, 5, SEED)
    assert unregistered not in tasks.RANDOM_ALIAS_SPECS
    with pytest.raises(ValueError):
        restriction.get_restriction(unregistered)
    for n_bins, seed in (
        (20.9, SEED),
        (20.0, SEED),
        (20, 20261003.9),
        (20, float(SEED)),
        (True, SEED),
        (20, True),
        (np.bool_(True), SEED),
        ("20", SEED),
        (20, None),
    ):
        with pytest.raises(ValueError, match="must be an integer"):
            tasks.random_binning_id(CLIFF, n_bins, seed)
    for n_bins in (0, -4):
        with pytest.raises(ValueError, match="not positive"):
            tasks.random_binning_id(CLIFF, n_bins, SEED)
    for seed in (-1, 2**32):
        with pytest.raises(ValueError, match="outside"):
            tasks.random_binning_id(CLIFF, 20, seed)
    for env_name in ("FrozenLake-v1", "cliffwalking", None, 3):
        with pytest.raises(ValueError, match="has no random binning"):
            tasks.random_binning_id(env_name, 20, SEED)


def _refines(fine, coarse):
    bin_of = {s: i for i, part in enumerate(coarse) for s in part}
    return all(len({bin_of[s] for s in part}) == 1 for part in fine)


def test_rule_says_refinement_is_not_guaranteed_rather_than_never():
    assert "refinement is not guaranteed across bin counts" in (
        tasks.RANDOM_BINNING_RULE
    )
    assert "not nested" not in tasks.RANDOM_BINNING_RULE
    assert tasks.random_partition_descriptor(RANDOM_IDS[CLIFF][0])["rule"] == (
        tasks.RANDOM_BINNING_RULE
    )
    # One shared permutation: 37 states with K=4 (boundaries 10, 19, 28)
    # refine K=2 (boundary 19), but K=8 (boundaries 5, ..., 33) does not
    # refine K=4. So some pairs nest and no pair is guaranteed to.
    states, _ = tasks.reachable_nonterminal(CLIFF)
    assert len(states) == 37
    parts = {k: tasks.random_state_partition(states, k, SEED) for k in (2, 4, 8)}
    assert [len(p) for p in parts[2]] == [19, 18]
    assert [len(p) for p in parts[4]] == [10, 9, 9, 9]
    assert _refines(parts[4], parts[2])
    assert not _refines(parts[8], parts[4])


def _rebuild(restriction_id):
    tasks._random_alias.cache_clear()
    try:
        return tasks.alias_table(restriction_id)
    finally:
        tasks._random_alias.cache_clear()


def test_changed_table_or_reachable_set_is_refused(monkeypatch):
    rid = "frozenlake_slippery_random_bins9_seed20261003"
    monkeypatch.setitem(tasks.FROZEN_TABLE_SHA256, rid, "0" * 64)
    with pytest.raises(RuntimeError, match="differs from the frozen"):
        _rebuild(rid)
    # No digest is accepted lazily: a missing literal is refused too.
    monkeypatch.delitem(tasks.FROZEN_TABLE_SHA256, rid)
    with pytest.raises(RuntimeError, match="differs from the frozen None"):
        _rebuild(rid)
    monkeypatch.undo()

    pin = dict(tasks.FROZEN_REACHABLE_SHA256[SLIPPERY], states_sha256="0" * 64)
    monkeypatch.setitem(tasks.FROZEN_REACHABLE_SHA256, SLIPPERY, pin)
    with pytest.raises(RuntimeError, match="reachable set"):
        _rebuild(rid)
    monkeypatch.undo()

    original = tasks.random_state_partition

    def other_seed(states, n_bins, seed, **kwargs):
        return original(states, n_bins, seed + 1, **kwargs)

    monkeypatch.setattr(tasks, "random_state_partition", other_seed)
    with pytest.raises(RuntimeError, match="differs from the frozen"):
        _rebuild(rid)
    monkeypatch.undo()
    assert _rebuild(rid)["table_sha256"] == EXPECTED_TABLES[rid]


def test_original_tasks_ladders_and_old_records_are_unchanged():
    for env_name, digest in EXISTING_TASK_DIGESTS.items():
        task = tasks.get_task(env_name)
        assert tasks.task_sha256(task) == digest
        assert not set(task.restriction_ids) & set(ALL_RANDOM)
        assert "random" not in " ".join(tasks.describe(task)["declared"]["ladder"])
    for rid, digest in EXISTING_RESTRICTION_DIGESTS.items():
        record = restriction.describe_restriction(rid)
        assert tasks._digest(record) == digest, rid
        assert "partition" not in record
        assert "smallest-index" in record["description"]
    assert not set(tasks.ALIAS_RUNGS) & set(ALL_RANDOM)
    assert restriction.ALIAS_RESTRICTIONS == tuple(tasks.ALIAS_RUNGS)
    assert restriction.RANDOM_ALIAS_RESTRICTIONS == ALL_RANDOM


def test_calibration_candidates_append_random_after_strategic():
    for env_name, strategic in STRATEGIC_CANDIDATES.items():
        assert tasks.calibration_candidates(env_name)[: len(strategic)] == strategic
    for env_name, ids in RANDOM_IDS.items():
        candidates = tasks.calibration_candidates(env_name)
        assert candidates[-len(ids) :] == ids
        assert candidates[: -len(ids)] == STRATEGIC_CANDIDATES.get(env_name, ())
    # Random-offset grids (test_agnostic_random_grid.py) append after these.
    assert tasks.calibration_candidates("MountainCar-v0") == (
        STRATEGIC_CANDIDATES["MountainCar-v0"]
        + tasks.RANDOM_GRID_CANDIDATES["MountainCar-v0"]
    )
    assert tasks.calibration_candidates("FrozenLake-v1") == ()
    assert tasks.calibration_candidates(env_utils.FROZENLAKE_VARIED_START) == ()


# ---------------------------------------------------------------------------
# AliasRestriction numpy and torch forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("restriction_id", ALL_RANDOM)
def test_numpy_and_torch_agree_on_every_reachable_input(restriction_id):
    env_name = ENV_OF[restriction_id]
    found = restriction.get_restriction(restriction_id)
    assert isinstance(found, restriction.AliasRestriction)
    assert found.env_name == env_name
    assert found.table_sha256 == EXPECTED_TABLES[restriction_id]
    assert restriction.restriction_digest(found) == EXPECTED_TABLES[restriction_id]
    states, _ = tasks.reachable_nonterminal(env_name)
    obs = task_audit._encode_states(env_name, states)
    assert obs.dtype == np.float32
    np.testing.assert_array_equal(found.decode(obs), states)

    masked = found.apply(obs)
    mask = restriction.ObservationMask(restriction_id)
    assert not list(mask.parameters())
    with th.no_grad():
        torch_masked = mask(th.as_tensor(obs)).numpy()
    np.testing.assert_array_equal(masked, torch_masked)
    assert masked.dtype == obs.dtype and masked.shape == obs.shape
    table = np.asarray(found.representative)
    np.testing.assert_array_equal(found.decode(masked), table[states])
    np.testing.assert_array_equal(found.apply(masked), masked)  # idempotent
    record = tasks.alias_table(restriction_id)
    for group in record["groups"]:
        rows = masked[[states.index(s) for s in group]]
        assert (rows == found.encode(group[0])).all()
    # Unbatched inputs give the same rows.
    with th.no_grad():
        for i in range(len(states)):
            single = found.apply(obs[i])
            np.testing.assert_array_equal(single, masked[i])
            np.testing.assert_array_equal(
                mask(th.as_tensor(obs[i])).numpy(),
                masked[i],
            )


def test_description_says_random_first_shuffled_representative():
    for rid in ALL_RANDOM:
        record = restriction.describe_restriction(rid)
        spec = tasks.RANDOM_ALIAS_SPECS[rid]
        assert record["kind"] == "alias"
        assert "first shuffled member" in record["description"]
        assert f"{spec.n_bins} balanced random bins" in record["description"]
        assert str(SEED) in record["description"]
        assert "smallest-index" not in record["description"]
        assert record["obs_shape"] == [
            sum(tasks.get_task(spec.env_name).blocks),
        ]
        assert record["partition"]["restriction_id"] == rid


def test_masks_refuse_inputs_that_are_not_one_hot():
    rid = RANDOM_IDS[BLACKJACK][0]
    found = restriction.get_restriction(rid)
    bad = np.zeros((2, 45), dtype=np.float32)
    with pytest.raises(ValueError, match="not one-hot"):
        found.apply(bad)
    with pytest.raises(ValueError, match="not one-hot"):
        restriction.ObservationMask(rid)(th.as_tensor(bad))


# ---------------------------------------------------------------------------
# Runner, audit and checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_name", sorted(COUNTS))
def test_runner_accepts_random_ids_only_when_opted_in(env_name):
    ids = RANDOM_IDS[env_name]
    assert task_pilot.calibration_candidates_of(env_name)[-len(ids) :] == ids
    plain = task_pilot.runner_spec(env_name)
    assert not set(plain.run_restrictions) & set(ALL_RANDOM)
    for kwargs in ({}, {"paired_cold_fits": True}, {"bc_prefix": True}):
        spec = task_pilot.runner_spec(
            env_name,
            calibration_candidates=True,
            **kwargs,
        )
        assert spec.run_restrictions[-len(ids) :] == ids
        assert spec.calibration_candidates[-len(ids) :] == ids
        others = set(ALL_RANDOM) - set(ids)
        assert not set(spec.run_restrictions) & others
    paired = task_pilot.runner_spec(
        env_name,
        calibration_candidates=True,
        paired_cold_fits=True,
    )
    assert paired.protocol == task_pilot.PAIRED_PROTOCOL
    assert rp.PREFIX_METHOD in paired.methods


def test_slippery_opt_in_now_declares_only_random_candidates():
    assert task_pilot.calibration_candidates_of(SLIPPERY) == RANDOM_IDS[SLIPPERY]
    with pytest.raises(ValueError, match="declares no calibration candidates"):
        task_pilot.calibration_candidates_of(env_utils.FROZENLAKE_VARIED_START)


def test_cli_parses_random_ids_with_existing_flags():
    parser = task_pilot.build_parser()
    common = [
        "--env",
        TAXI,
        "--expert-dir",
        "e",
        "--output-dir",
        "o",
        "--deadline",
        "2099-01-01T00:00:00Z",
        "--job-limit-seconds",
        "60",
    ]
    rid = "taxi_random_bins40_seed20261003"
    for method, extra in (
        ("ftl", ["--paired-cold-fits"]),
        (rp.PREFIX_METHOD, []),
        (rp.PREFIX_METHOD, ["--paired-cold-fits"]),
    ):
        args = parser.parse_args(
            ["run", *common, "--data-dir", "d", "--seed", "1", "--method", method]
            + ["--restriction", rid, "--calibration-candidates", *extra],
        )
        assert args.restriction == rid
        assert args.calibration_candidates
        assert args.paired_cold_fits == bool(extra)
    args = parser.parse_args(
        ["audit", *common, "--restriction", rid, "--calibration-candidates"],
    )
    assert args.restriction == rid


def test_cross_environment_random_ids_are_refused(tmp_path):
    taxi_id = "taxi_random_bins40_seed20261003"
    kwargs = dict(
        method="ftl",
        restriction_id=taxi_id,
        seed=1,
        deadline=rollouts.parse_utc_deadline("2099-01-01T00:00:00Z"),
        job_limit_seconds=60.0,
        paired_cold_fits=True,
        calibration_candidates=True,
    )
    code = task_pilot.run_job(
        tmp_path / "out",
        tmp_path / "expert",
        tmp_path / "data",
        env_name=CLIFF,
        **kwargs,
    )
    assert code == classical.EXIT_REFUSED
    assert not (tmp_path / "out").exists()
    # Without the opt-in the task's own random ids are refused as well.
    kwargs.update(restriction_id=RANDOM_IDS[CLIFF][0], calibration_candidates=False)
    code = task_pilot.run_job(
        tmp_path / "out",
        tmp_path / "expert",
        tmp_path / "data",
        env_name=CLIFF,
        **kwargs,
    )
    assert code == classical.EXIT_REFUSED
    expert = RuleExpert(lambda o: 0)
    with pytest.raises(ValueError, match="not on the CliffWalking-v0 ladder"):
        task_audit.audit(expert, CLIFF, taxi_id, 301)
    cliff_expert = _random_expert(CLIFF)
    with pytest.raises(ValueError, match="needs observation shape"):
        restriction.make_linear_policy(cliff_expert, taxi_id)


@pytest.mark.parametrize("restriction_id", RANDOM_IDS[SLIPPERY])
def test_enumerated_audit_groups_states_by_random_bin(restriction_id):
    # Down on even states, right on odd ones: a full-state rule.
    expert = RuleExpert(lambda o: 1 if int(np.argmax(o)) % 2 == 0 else 2)
    record = task_audit.audit(expert, SLIPPERY, restriction_id, 301)
    assert record["evidence_type"] == "reachable_enumeration"
    assert record["restriction"]["partition"]["restriction_id"] == restriction_id
    groups = tasks.alias_table(restriction_id)["groups"]
    assert record["n_classes"] == len(groups)
    assert sorted(sorted(c["states"]) for c in record["classes"]) == sorted(
        sorted(g) for g in groups
    )
    for c in record["classes"]:
        group = next(g for g in groups if set(g) == set(c["states"]))
        assert c["representative"] == group[0]
        parity = {s % 2 for s in c["states"]}
        assert c["conflict"] == (len(parity) > 1)


@pytest.mark.parametrize("restriction_id", RANDOM_IDS[CLIFF][::3])
def test_checkpoints_reload_and_refuse_a_changed_table(restriction_id, tmp_path):
    expert = _random_expert(CLIFF)
    before = {k: v.clone() for k, v in expert.state_dict().items()}
    th.manual_seed(3)
    policy = restriction.make_linear_policy(expert, restriction_id)
    for key, value in expert.state_dict().items():
        assert th.equal(value, before[key])  # expert never mutated
    with th.no_grad():
        policy.action_net.weight.normal_()
    states, _ = tasks.reachable_nonterminal(CLIFF)
    obs = task_audit._encode_states(CLIFF, states)
    tensor = th.as_tensor(obs)
    with th.no_grad():
        seen = policy.features_extractor.mask(tensor).numpy()
        probs = policy.get_distribution(tensor).distribution.probs.numpy()
    for group in tasks.alias_table(restriction_id)["groups"]:
        rows = [states.index(s) for s in group]
        # The learner input is bitwise one row per bin; batched matmul may
        # still round differently per row, hence the tolerance on probs.
        assert (seen[rows] == seen[rows[0]]).all()
        np.testing.assert_allclose(
            probs[rows],
            np.broadcast_to(probs[rows[0]], probs[rows].shape),
            rtol=0,
            atol=1e-6,
        )

    path = tmp_path / "policy.pt"
    policy.save(path)
    loaded = restriction.load_policy_checkpoint(path)
    assert loaded.restriction_id == restriction_id
    assert isinstance(
        loaded.features_extractor.mask.restriction,
        restriction.AliasRestriction,
    )
    np.testing.assert_array_equal(
        loaded.predict(obs, deterministic=True)[0],
        policy.predict(obs, deterministic=True)[0],
    )
    saved = th.load(path, weights_only=False)
    assert saved["data"]["restriction_id"] == restriction_id
    assert saved["data"]["restriction_sha256"] == EXPECTED_TABLES[restriction_id]

    params = dict(saved["data"], restriction_sha256="0" * 64)
    with pytest.raises(ValueError, match="not the saved"):
        restriction.RestrictedActorCriticPolicy(**params)
    # Another random binning's digest is refused on reload too.
    saved["data"]["restriction_sha256"] = EXPECTED_TABLES[RANDOM_IDS[CLIFF][-1]]
    th.save(saved, path)
    with pytest.raises(ValueError, match="not the saved"):
        restriction.load_policy_checkpoint(path)
