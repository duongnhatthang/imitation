"""Opt-in replacement masks and the FrozenLakeVariedStart-v0 task.

Covers the MountainCar quantized restrictions, the Blackjack player-sum bin
aliases, the opt-in calibration candidate registry and the new varied-start
FrozenLake task. Every test is synthetic or protocol level: environments are
built locally and experts are rule-based or untrained.
"""

import collections
import itertools
from fractions import Fraction

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from stable_baselines3 import PPO

from imitation.experiments.agnostic import restriction, task_audit, tasks
from imitation.experiments.ftrl import env_baselines, env_utils

# Digests of every pre-existing restriction description and task description,
# recorded before the replacement masks were added. They must never move.
EXISTING_RESTRICTION_DIGESTS = {
    "acrobot_theta2_and_dots_zero": (
        "f606b7d1eca89aadd4c42df684fd042ddea6e736fdb27b5dda01ecf5257a8e21"
    ),
    "acrobot_theta2_dot_zero": (
        "419084743713927a2ae0470513f234b6703896c50408129d03c733f263ad5d70"
    ),
    "acrobot_theta_dots_zero": (
        "77c73a97e10b1284bf136c405228a41c0f9dbd3993b6e0087c4ef563ab947be4"
    ),
    "cart_position_angular_velocity_zero": (
        "9cd68277c24117f92d37670ef0ba11f54eb9a7dc8773ab3926367b9e25e2b634"
    ),
    "cart_position_velocity_angular_velocity_zero": (
        "3c3ceb9be433bc29d99e2f029bd88ce6869cb7ae066a1ca484f3219ea43cf075"
    ),
    "cart_position_zero": (
        "b37ee6ee95a854e338148222860eda9b67a868724b90c45c3bc3b32c7e5fade9"
    ),
    "identity": "ce17b3017a00aebb32d4c4107e18e0ef51a721b270eb84ac4048ab4613e91e86",
    "lunarlander_angular_velocity_zero": (
        "831aca65d04b648fdc68cdfc1e57972e42849f76b2551fd4f91279399e252c7e"
    ),
    "lunarlander_velocities_angle_zero": (
        "8b986a0eeb826bb13162e116997409d84d0e7e50c5c5de7fa68365c3353a39a4"
    ),
    "lunarlander_velocities_zero": (
        "a9ea0ae5e5ac851bd2ca28def1ac29a3431973a0ded091d8fad7e343ce67d4d6"
    ),
    "mountaincar_position_zero": (
        "be2afe00f85c361358eeef9b49cdc813d8fe24a60b8cd1d95895bd2fa5c75da7"
    ),
    "mountaincar_velocity_zero": (
        "395b2e8b256a5b5bff4848766f510fe14139929afdcfd88e370bc54ddd5c4307"
    ),
    "blackjack_usable_ace_dealer_hidden": (
        "29efc4dfeebf7bdeeffee599771dc0243fa53089d83af7b2fc605ad8a02646a6"
    ),
    "blackjack_usable_ace_hidden": (
        "6261444c57104673d8e18ded1e7c35e85077edc7a9d8607e9cc92def51cbfafe"
    ),
    "cliffwalking_column_hidden": (
        "2f2a0609223db9f1b3759f5d08dac5632a50b720ed001a40ed1f348103aa81d1"
    ),
    "cliffwalking_column_hidden_row_pairs": (
        "c9f25e1c5bd0ea9f0ee308f9a732bff31224a2016a0d4135069a27b53bb32422"
    ),
    "cliffwalking_column_pairs": (
        "5ff782bb293cdb1523cb9b524681ed95d1e76ba2c64f131de4c4539e4e645d09"
    ),
    "frozenlake_column_hidden": (
        "2b5d3b80b427022dd3257d8c829c33497e5299c02dad50878d8f25f17efe8049"
    ),
    "frozenlake_column_hidden_row_pairs": (
        "e9939e9f8ab57900db922620300a8d23bb9d236a4cbc42dd0dd375e28dfb55f6"
    ),
    "frozenlake_column_pairs": (
        "a063a8f4378b8110b1754636790c9d6b896d5226cffd93277519fe36468e4d41"
    ),
    "taxi_destination_hidden": (
        "c25c8391a5bb1618938f34147625ae95834e5659cbfbc0c059234d8b74197f43"
    ),
    "taxi_destination_passenger_hidden": (
        "80e31bb21788caafea2c4ddb9e2922835801494506ee050109456aa8f8625863"
    ),
}
EXISTING_TASK_DIGESTS = {
    "CartPole-v1": "bbe157c72171cd88f0a5e679e3f32f54c0d2756b13cfd381641172318fd1efc3",
    "FrozenLake-v1": "7e6760d99dc8898d6f52ff4a90cd25b8911efda94efbee7f52386c09378015a4",
    "CliffWalking-v0": (
        "01127136af3841e64bc22eb04373553af3b8db159c9c6ba8f20f24348fd24967"
    ),
    "Acrobot-v1": "82426d16d673aa9741879f6a8116829bc7ae360c508bbb893d78ed9080bf218c",
    "MountainCar-v0": (
        "1da99f9c7f6798270bd19a0132c80149e6f50de8a28c6e9b8ba41db2b860245c"
    ),
    "Taxi-v3": "274c50730f86271b93e53ccf1c37878ee6e7b80d377304e055a6cfe8c7fca422",
    "Blackjack-v1": "f6877a8cfdbe70ad61bd28f4092b4d15e04bcae0dd54cd180fd3fb1ab80182fb",
    "LunarLander-v2": (
        "37e378d602a85c2701349721e9f4819ef11d4b6bde3b3eb86fe7ea8666e64c84"
    ),
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
    "taxi_destination_hidden": (
        "949469d8309e8871b1c427d86ec1ddc63a0e5739483227c46fba4465f1046c1a"
    ),
    "taxi_destination_passenger_hidden": (
        "a225e30160f5710a3886249405154138e788f955fd0e03ee7761fa4dec22e632"
    ),
    "blackjack_usable_ace_hidden": (
        "da0225fe2e322c7c17650ee6064ade7039266077cba075cb3d306f28a544a4bb"
    ),
    "blackjack_usable_ace_dealer_hidden": (
        "ee81f71455823860f5f7be9db0029000fc04ff68373e465f617a3688120af3d7"
    ),
}
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
ORIGINAL_LADDERS = {
    "CartPole-v1": (
        "cart_position_angular_velocity_zero",
        "cart_position_velocity_angular_velocity_zero",
    ),
    "FrozenLake-v1": (
        "frozenlake_column_pairs",
        "frozenlake_column_hidden",
        "frozenlake_column_hidden_row_pairs",
    ),
    "CliffWalking-v0": (
        "cliffwalking_column_pairs",
        "cliffwalking_column_hidden",
        "cliffwalking_column_hidden_row_pairs",
    ),
    "Acrobot-v1": (
        "acrobot_theta2_dot_zero",
        "acrobot_theta_dots_zero",
        "acrobot_theta2_and_dots_zero",
    ),
    "MountainCar-v0": ("mountaincar_velocity_zero", "mountaincar_position_zero"),
    "Taxi-v3": ("taxi_destination_hidden", "taxi_destination_passenger_hidden"),
    "Blackjack-v1": (
        "blackjack_usable_ace_hidden",
        "blackjack_usable_ace_dealer_hidden",
    ),
    "LunarLander-v2": (
        "lunarlander_angular_velocity_zero",
        "lunarlander_velocities_zero",
        "lunarlander_velocities_angle_zero",
    ),
}

MOUNTAINCAR = "MountainCar-v0"
# restriction id -> (position step, velocity dead zone)
QUANTIZED = {
    "mountaincar_bins_pos005_vel005": (0.05, 0.005),
    "mountaincar_bins_pos005_vel015": (0.05, 0.015),
    "mountaincar_bins_pos010_vel005": (0.1, 0.005),
    "mountaincar_bins_pos010_vel015": (0.1, 0.015),
}
VELOCITY_MAGNITUDE = 0.035

BLACKJACK = "Blackjack-v1"
# restriction id -> (high player-sum bin width, reachable classes)
BLACKJACK_BINS = {
    "blackjack_player_sum_bins2_dealer_ace_hidden": (2, 6),
    "blackjack_player_sum_bins3_dealer_ace_hidden": (3, 5),
    "blackjack_player_sum_bins4_dealer_ace_hidden": (4, 4),
}

VARIED = "FrozenLakeVariedStart-v0"
VARIED_SUPPORT = (0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14)
VARIED_LADDER = (
    "frozenlake_varied_start_column_pairs",
    "frozenlake_varied_start_column_hidden",
    "frozenlake_varied_start_column_hidden_row_pairs",
)
# Uniform-random success probability over the varied reset, 100-step cap.
VARIED_RANDOM_SCORE = 0.09037647670287287


class RuleExpert:
    """Deterministic expert given by a rule on the full observation."""

    def __init__(self, rule):
        self.rule = rule

    def predict(self, obs, deterministic=True):
        assert deterministic
        obs = np.asarray(obs)
        return np.array([self.rule(o) for o in obs]), None


def _random_expert(env_name, seed=0):
    venv = env_utils.make_env(env_name, 1, np.random.default_rng(0))
    policy = PPO("MlpPolicy", venv, device="cpu", seed=seed).policy
    venv.close()
    return policy


def _torch_mask(restriction_id, obs):
    with th.no_grad():
        return restriction.ObservationMask(restriction_id)(th.as_tensor(obs)).numpy()


# ---------------------------------------------------------------------------
# Existing identities are unchanged
# ---------------------------------------------------------------------------


def test_existing_restriction_descriptions_are_unchanged():
    for rid, digest in EXISTING_RESTRICTION_DIGESTS.items():
        assert tasks._digest(restriction.describe_restriction(rid)) == digest, rid


@pytest.mark.parametrize("env_name", ORIGINAL_ENVS)
def test_existing_tasks_and_ladders_are_unchanged(env_name):
    task = tasks.get_task(env_name)
    assert task.ladder == ORIGINAL_LADDERS[env_name]
    assert tasks.task_sha256(task) == EXISTING_TASK_DIGESTS[env_name]
    assert "reset_distribution" not in tasks.describe(task)


def test_existing_frozen_tables_and_registries_are_unchanged():
    for rid, digest in EXISTING_FROZEN_TABLES.items():
        assert tasks.FROZEN_TABLE_SHA256[rid] == digest
        assert tasks.alias_table(rid)["table_sha256"] == digest
    assert tuple(tasks.TASKS) == ORIGINAL_ENVS
    assert env_utils.ENV_GROUPS["classical"] == list(ORIGINAL_ENVS)
    for ladder in ORIGINAL_LADDERS.values():
        assert not set(ladder) & (set(QUANTIZED) | set(BLACKJACK_BINS))


def test_calibration_candidates_are_opt_in_and_off_the_ladders():
    # Later candidates (test_agnostic_standard_binning.py) only append.
    # Random-offset grids (test_agnostic_random_grid.py) append after these.
    assert tasks.CALIBRATION_CANDIDATES[MOUNTAINCAR] == tuple(QUANTIZED) + (
        tasks.RANDOM_GRID_CANDIDATES[MOUNTAINCAR]
    )
    assert tasks.CALIBRATION_CANDIDATES[BLACKJACK][:3] == tuple(BLACKJACK_BINS)
    for env_name, candidates in tasks.CALIBRATION_CANDIDATES.items():
        task = tasks.get_task(env_name)
        assert not set(candidates) & set(task.restriction_ids)
        assert tasks.calibration_candidates(env_name) == candidates
        for rid in candidates:
            restriction.get_restriction(rid)  # registered
    assert tasks.calibration_candidates("CartPole-v1") == (
        tasks.RANDOM_GRID_CANDIDATES["CartPole-v1"]
    )


# ---------------------------------------------------------------------------
# MountainCar quantized restrictions
# ---------------------------------------------------------------------------


def _expected_quantized(obs, step, dead_zone):
    """Independent float64 reference: round position, three-level velocity."""
    obs = np.asarray(obs, dtype=np.float64)
    position = np.round(obs[..., 0] / step) * step
    velocity = np.where(
        np.abs(obs[..., 1]) <= dead_zone,
        0.0,
        np.sign(obs[..., 1]) * VELOCITY_MAGNITUDE,
    )
    return np.stack([position, velocity], axis=-1)


def _mountaincar_space():
    env = gym.make(MOUNTAINCAR)
    space = env.observation_space
    env.close()
    return space


def _boundary_inputs(step, dead_zone):
    """Bin edges, dead-zone edges, bounds and their float32 neighbours."""
    space = _mountaincar_space()
    low, high = space.low.astype(np.float64), space.high.astype(np.float64)
    centres = np.arange(-1.2, 0.6 + 1e-9, step)
    edges = np.concatenate([centres - step / 2, centres + step / 2, centres])
    positions = [low[0], high[0], 0.0]
    for e in edges:
        f = np.float32(e)
        positions += [
            f,
            np.nextafter(f, np.float32(-2)),
            np.nextafter(f, np.float32(2)),
        ]
    velocities = [low[1], high[1], 0.0, -0.0]
    for d in (dead_zone, -dead_zone, 0.005, 0.015, -0.005, -0.015):
        f = np.float32(d)
        velocities += [
            f,
            np.nextafter(f, np.float32(-1)),
            np.nextafter(f, np.float32(1)),
        ]
    grid = np.array(
        list(itertools.product(positions, velocities)),
        dtype=np.float32,
    )
    rng = np.random.default_rng(0)
    random = rng.uniform(low, high, size=(4096, 2)).astype(np.float32)
    inside = (grid[:, 0] >= space.low[0]) & (grid[:, 0] <= space.high[0])
    return np.concatenate([grid[inside], random])


@pytest.mark.parametrize("restriction_id", sorted(QUANTIZED))
def test_quantized_mask_values_and_bounds(restriction_id):
    step, dead_zone = QUANTIZED[restriction_id]
    found = restriction.get_restriction(restriction_id)
    assert isinstance(found, restriction.QuantizedRestriction)
    assert found.obs_shape == (2,) and found.env_name == MOUNTAINCAR
    space = _mountaincar_space()

    obs = _boundary_inputs(step, dead_zone)
    masked = found.apply(obs)
    assert masked.dtype == np.float32 and masked.shape == obs.shape
    expected = _expected_quantized(obs, step, dead_zone).astype(np.float32)
    expected = np.clip(expected, space.low, space.high)
    np.testing.assert_array_equal(masked, expected)
    assert all(space.contains(row) for row in masked)
    assert set(np.unique(masked[:, 1]).tolist()) == {
        float(np.float32(-VELOCITY_MAGNITUDE)),
        0.0,
        float(np.float32(VELOCITY_MAGNITUDE)),
    }
    # Velocity is zero exactly when |v| <= dead zone.
    zero = np.abs(obs[:, 1].astype(np.float64)) <= dead_zone
    np.testing.assert_array_equal(masked[:, 1] == 0, zero)
    # Every position output is a grid point k * step (within float32).
    k = np.round(masked[:, 0].astype(np.float64) / step)
    np.testing.assert_allclose(masked[:, 0], k * step, rtol=0, atol=1e-6)
    # Idempotent, and a fresh copy: the input is never written.
    before = obs.copy()
    np.testing.assert_array_equal(found.apply(masked), masked)
    np.testing.assert_array_equal(obs, before)


def test_quantized_mask_hand_checked_values():
    fine = restriction.get_restriction("mountaincar_bins_pos005_vel005")
    coarse = restriction.get_restriction("mountaincar_bins_pos010_vel015")
    obs = np.array(
        [
            [0.123, 0.005],
            [-0.524, 0.0051],
            [-1.2, -0.02],
            [0.6, 0.07],
            [0.0, -0.005],
            [-0.476, 0.015],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        fine.apply(obs),
        [
            [0.1, 0.0],
            [-0.5, 0.035],
            [-1.2, -0.035],
            [0.6, 0.035],
            [0.0, 0.0],
            [-0.5, 0.035],
        ],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        coarse.apply(obs),
        [
            [0.1, 0.0],
            [-0.5, 0.0],
            [-1.2, -0.035],
            [0.6, 0.035],
            [0.0, 0.0],
            [-0.5, 0.0],
        ],
        rtol=0,
        atol=1e-7,
    )


@pytest.mark.parametrize("restriction_id", sorted(QUANTIZED))
def test_quantized_numpy_and_torch_agree_bitwise(restriction_id):
    step, dead_zone = QUANTIZED[restriction_id]
    found = restriction.get_restriction(restriction_id)
    obs = _boundary_inputs(step, dead_zone)
    masked = found.apply(obs)
    tensor = th.as_tensor(obs.copy())
    state = th.random.get_rng_state()
    torch_masked = _torch_mask(restriction_id, tensor)
    assert th.equal(th.random.get_rng_state(), state)  # draws nothing
    assert masked.tobytes() == torch_masked.tobytes()
    np.testing.assert_array_equal(tensor.numpy(), obs)  # input untouched
    # Batch shapes beyond one leading axis, and a single observation.
    stacked = obs[:24].reshape(2, 3, 4, 2)
    assert (
        found.apply(stacked).tobytes() == _torch_mask(restriction_id, stacked).tobytes()
    )
    np.testing.assert_array_equal(found.apply(obs[0]), masked[0])
    np.testing.assert_array_equal(_torch_mask(restriction_id, obs[0]), masked[0])
    # float64 inputs keep their dtype in both forms.
    wide = obs.astype(np.float64)
    assert found.apply(wide).dtype == np.float64
    assert found.apply(wide).tobytes() == _torch_mask(restriction_id, wide).tobytes()


@pytest.mark.parametrize("restriction_id", sorted(QUANTIZED))
def test_quantized_mask_is_parameter_free_and_validates_inputs(restriction_id):
    found = restriction.get_restriction(restriction_id)
    mask = restriction.ObservationMask(restriction_id)
    assert list(mask.parameters()) == [] and len(mask.state_dict()) == 0
    for bad in (np.zeros((5, 3), np.float32), np.zeros((), np.float32)):
        with pytest.raises(ValueError, match="shape"):
            found.apply(bad)
        with pytest.raises(ValueError, match="shape"):
            mask(th.as_tensor(bad))
    nan = np.array([[np.nan, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        found.apply(nan)
    with pytest.raises(ValueError, match="finite"):
        mask(th.as_tensor(nan))


def test_quantized_descriptor_is_serialized_in_full():
    digests = set()
    for rid, (step, dead_zone) in QUANTIZED.items():
        record = restriction.describe_restriction(rid)
        assert record["kind"] == "quantized"
        assert record["restriction_id"] == rid
        assert record["env_name"] == MOUNTAINCAR
        assert record["obs_shape"] == [2]
        position, velocity = record["features"]
        assert position == {
            "name": "position",
            "index": 0,
            "rule": "round_to_step",
            "step": step,
            "dead_zone": None,
            "magnitude": None,
            "low": float(np.float32(-1.2)),
            "high": float(np.float32(0.6)),
        }
        assert velocity == {
            "name": "velocity",
            "index": 1,
            "rule": "sign_dead_zone",
            "step": None,
            "dead_zone": dead_zone,
            "magnitude": VELOCITY_MAGNITUDE,
            "low": float(np.float32(-0.07)),
            "high": float(np.float32(0.07)),
        }
        found = restriction.get_restriction(rid)
        body = {k: v for k, v in record.items() if k != "descriptor_sha256"}
        assert record["descriptor_sha256"] == found.descriptor_sha256
        assert found.descriptor_sha256 == tasks._digest(
            {k: v for k, v in body.items() if k != "kind"},
        )
        digests.add(found.descriptor_sha256)
    assert len(digests) == len(QUANTIZED)
    space = _mountaincar_space()
    for rid in QUANTIZED:
        features = restriction.get_restriction(rid).features
        assert [f.low for f in features] == [float(v) for v in space.low]
        assert [f.high for f in features] == [float(v) for v in space.high]


def test_quantized_learner_ignores_within_bin_moves_and_leaves_expert_alone():
    rid = "mountaincar_bins_pos005_vel005"
    expert = _random_expert(MOUNTAINCAR)
    before = {k: v.clone() for k, v in expert.state_dict().items()}
    learner = restriction.make_linear_policy(expert, rid)
    for key, value in expert.state_dict().items():
        assert th.equal(value, before[key])  # expert never mutated
    assert not isinstance(expert, restriction.RestrictedActorCriticPolicy)
    with th.no_grad():
        learner.action_net.weight.normal_()
    obs = np.array([[-0.51, 0.004], [-0.49, -0.004]], dtype=np.float32)
    tensor = th.as_tensor(obs)
    with th.no_grad():
        probs = learner.get_distribution(tensor).distribution.probs.numpy()
        expert_probs = expert.get_distribution(tensor).distribution.probs.numpy()
    np.testing.assert_array_equal(probs[0], probs[1])
    assert not np.array_equal(expert_probs[0], expert_probs[1])


@pytest.mark.parametrize("restriction_id", sorted(QUANTIZED))
def test_quantized_checkpoints_reload_and_refuse_a_changed_descriptor(
    restriction_id,
    tmp_path,
):
    expert = _random_expert(MOUNTAINCAR)
    th.manual_seed(3)
    policy = restriction.make_linear_policy(expert, restriction_id)
    with th.no_grad():
        policy.action_net.weight.normal_()
    found = restriction.get_restriction(restriction_id)
    obs = _boundary_inputs(*QUANTIZED[restriction_id])[:256]
    path = tmp_path / "policy.pt"
    policy.save(path)
    loaded = restriction.load_policy_checkpoint(path)

    assert loaded.restriction_id == restriction_id
    assert isinstance(loaded.features_extractor.mask.restriction, type(found))
    np.testing.assert_array_equal(
        loaded.predict(obs, deterministic=True)[0],
        policy.predict(obs, deterministic=True)[0],
    )
    saved = th.load(path, weights_only=False)
    assert saved["data"]["restriction_sha256"] == found.descriptor_sha256
    saved["data"]["restriction_sha256"] = "0" * 64
    th.save(saved, path)
    with pytest.raises(ValueError, match="not the saved"):
        restriction.load_policy_checkpoint(path)


@pytest.mark.parametrize("restriction_id", sorted(QUANTIZED))
def test_quantized_candidates_have_constructed_witness_audits(restriction_id):
    # Pump with the velocity: inside the velocity dead zone the sign is hidden.
    expert = RuleExpert(lambda o: 2 if o[1] > 0 else 0)
    record = task_audit.audit(expert, MOUNTAINCAR, restriction_id, 301)
    assert record["evidence_type"] == "constructed_unwrapped_state"
    assert record["restriction"]["kind"] == "quantized"
    assert record["status"] == "conflict_found"
    assert all(s["observation_map_verified"] for s in record["states"])
    assert all(p["masked_equal"] for p in record["pairs"])
    for p in record["pairs"]:
        if p["conflict"]:
            assert p["coordinate"] == "velocity"
            assert (p["value_a"] > 0) != (p["value_b"] > 0)
    with pytest.raises(ValueError, match="ladder"):
        task_audit.audit(expert, "CartPole-v1", restriction_id, 301)


# ---------------------------------------------------------------------------
# Blackjack player-sum bins
# ---------------------------------------------------------------------------


def _player_class(player_sum, width):
    return "low" if player_sum < 12 else (player_sum - 12) // width


@pytest.mark.parametrize("restriction_id", sorted(BLACKJACK_BINS))
def test_blackjack_bins_merge_sums_and_hide_dealer_and_ace(restriction_id):
    width, n_classes = BLACKJACK_BINS[restriction_id]
    found = restriction.get_restriction(restriction_id)
    assert isinstance(found, restriction.AliasRestriction)
    record = tasks.alias_table(restriction_id)
    assert record["table_sha256"] == tasks.FROZEN_TABLE_SHA256[restriction_id]
    assert record["env_name"] == BLACKJACK
    reachable, _ = tasks.reachable_nonterminal(BLACKJACK)
    table = np.asarray(found.representative)

    by_class = collections.defaultdict(set)
    for s in reachable:
        player_sum, _, _ = tasks.blackjack_factors(s)
        by_class[_player_class(player_sum, width)].add(s)
    assert len(by_class) == record["reachable_classes"] == n_classes
    for members in by_class.values():
        rep = min(members)
        for s in members:
            assert table[s] == rep  # dealer card and usable ace are hidden
        assert table[rep] == rep
    # Low sums are one class; high classes are (sum - 12) // width.
    sums = sorted({tasks.blackjack_factors(s)[0] for s in reachable})
    assert sums == list(range(4, 22))
    expected_high = {(s - 12) // width for s in range(12, 22)}
    assert set(by_class) == {"low"} | expected_high

    states = np.array(reachable)
    obs = found.encode(states)
    masked = found.apply(obs)
    np.testing.assert_array_equal(found.apply(masked), masked)
    np.testing.assert_array_equal(_torch_mask(restriction_id, obs), masked)


@pytest.mark.parametrize("restriction_id", sorted(BLACKJACK_BINS))
def test_blackjack_bins_conflict_only_where_a_threshold_is_merged(restriction_id):
    width, _ = BLACKJACK_BINS[restriction_id]

    def threshold_17(o):
        return int(o[:32].argmax() < 17)  # hit below 17, ignore the dealer

    record = task_audit.audit(RuleExpert(threshold_17), BLACKJACK, restriction_id, 301)
    assert record["evidence_type"] == "reachable_enumeration"
    assert record["restriction"]["restriction_id"] == restriction_id
    conflicts = [c for c in record["classes"] if c["conflict"]]
    assert len(conflicts) == 1
    merged = {tasks.blackjack_factors(s)[0] for s in conflicts[0]["states"]}
    expected = {s for s in range(12, 22) if (s - 12) // width == (17 - 12) // width}
    assert merged == expected and 16 in merged and 17 in merged

    # An expert that only reads the visible bin never conflicts.
    def by_bin(o):
        return int(_player_class(int(o[:32].argmax()), width) in ("low", 0))

    clean = task_audit.audit(RuleExpert(by_bin), BLACKJACK, restriction_id, 301)
    assert clean["status"] == "no_conflict_on_reachable"


# ---------------------------------------------------------------------------
# FrozenLakeVariedStart-v0
# ---------------------------------------------------------------------------


def test_varied_start_env_keeps_the_original_dynamics():
    env = gym.make(VARIED)
    original = gym.make("FrozenLake-v1", is_slippery=False)
    try:
        assert isinstance(env.unwrapped, FrozenLakeEnv)
        assert type(env.unwrapped) is env_utils.FrozenLakeVariedStartEnv
        assert env.spec.max_episode_steps == 100
        assert env.unwrapped.P == original.unwrapped.P  # transitions and rewards
        assert tasks._tabular_dynamics_sha256(
            env.unwrapped
        ) == tasks._tabular_dynamics_sha256(original.unwrapped)
        np.testing.assert_array_equal(env.unwrapped.desc, original.unwrapped.desc)
        assert (env.unwrapped.nrow, env.unwrapped.ncol) == (4, 4)
        mu = env.unwrapped.initial_state_distrib
        expected = np.zeros(16)
        expected[list(VARIED_SUPPORT)] = 1.0 / len(VARIED_SUPPORT)
        np.testing.assert_array_equal(mu, expected)
        desc = env.unwrapped.desc.flatten()
        assert set(VARIED_SUPPORT) == {s for s in range(16) if desc[s] in (b"S", b"F")}
        starts = {env.reset(seed=seed)[0] for seed in range(200)}
        assert starts == set(VARIED_SUPPORT)
    finally:
        env.close()
        original.close()


def test_varied_start_factory_caps_and_encodes_like_frozenlake():
    assert env_utils.ENV_CONFIGS[VARIED]["obs_type"] == "discrete"
    assert env_utils.ENV_CONFIGS[VARIED]["obs_size"] == 16
    assert VARIED not in env_utils.ENV_GROUPS["classical"]
    assert tasks.effective_episode_cap(VARIED) == 100
    venv = env_utils.make_env(VARIED, 1, np.random.default_rng(0))
    try:
        assert venv.observation_space.shape == (16,)
        # Walking up from row 0 never terminates; the cap truncates at 100.
        for seed in range(50):
            venv.seed(seed)
            obs = venv.reset()
            if int(obs[0].argmax()) < 4:
                break
        steps = 0
        while True:
            _, _, done, info = venv.step(np.array([3]))
            steps += 1
            if done[0]:
                break
        assert steps == 100 and info[0]["TimeLimit.truncated"]
    finally:
        venv.close()


def test_varied_start_task_declares_its_own_ladder_and_reset():
    task = tasks.get_task(VARIED)
    assert VARIED not in tasks.TASKS and tasks.VARIANT_TASKS[VARIED] == task
    assert task.ladder == VARIED_LADDER and task.ladder_rule == "nested"
    assert (task.episode_cap, task.blocks, task.n_actions) == (100, (16,), 4)
    assert task.observation == "one_hot"
    record = tasks.describe(task)
    assert record["episode_cap"] == 100
    assert record["registered_kwargs"] == {}
    assert record["reset_distribution"] == {
        "support": list(VARIED_SUPPORT),
        "probabilities": [1.0 / 11 if s in VARIED_SUPPORT else 0.0 for s in range(16)],
        "rule": "uniform over the safe nonterminal states",
    }
    lake = tasks.describe(tasks.get_task("FrozenLake-v1"))
    assert record["tabular_dynamics_sha256"] == lake["tabular_dynamics_sha256"]
    assert tasks.task_sha256(task) != tasks.task_sha256(tasks.get_task("FrozenLake-v1"))


def test_varied_start_describe_refuses_a_changed_reset(monkeypatch):
    tasks._describe.cache_clear()
    monkeypatch.setitem(tasks.DECLARED_RESET_SUPPORT, VARIED, (0,))
    with pytest.raises(tasks.TaskMismatch, match="reset"):
        tasks.describe(tasks.get_task(VARIED))
    monkeypatch.undo()
    tasks._describe.cache_clear()
    tasks.describe(tasks.get_task(VARIED))


def test_varied_start_reachability_and_alias_tables():
    reachable, evidence = tasks.reachable_nonterminal(VARIED)
    assert reachable == list(VARIED_SUPPORT)
    assert evidence["reset_support"] == list(VARIED_SUPPORT)
    assert evidence["terminal_states_reached"] == [5, 7, 11, 12, 15]
    for rid, counts in zip(VARIED_LADDER, (8, 4, 2)):
        original = rid.replace("frozenlake_varied_start_", "frozenlake_")
        found = restriction.get_restriction(rid)
        record = tasks.alias_table(rid)
        assert record["env_name"] == VARIED == found.env_name
        assert record["table_sha256"] == tasks.FROZEN_TABLE_SHA256[rid]
        assert record["reachable_classes"] == counts
        assert record["hidden"] == tasks.ALIAS_RUNGS[original].hidden
        # Same reachable set as the original reset, so the same tables.
        assert (
            found.representative == restriction.get_restriction(original).representative
        )
        obs = found.encode(np.array(reachable))
        np.testing.assert_array_equal(_torch_mask(rid, obs), found.apply(obs))
    assert task_audit.evidence_type(VARIED) == "reachable_enumeration"
    rows = task_audit.audit(
        RuleExpert(lambda o: int(o.argmax()) % 4),
        VARIED,
        "frozenlake_varied_start_column_hidden",
        301,
    )
    assert rows["status"] == "conflict_found" and rows["n_states"] == 11


def _table(env):
    return {
        s: {a: [tuple(o) for o in outs] for a, outs in acts.items()}
        for s, acts in env.unwrapped.P.items()
    }


def _varied_P():
    env = gym.make(VARIED)
    try:
        return _table(env)
    finally:
        env.close()


# The declared reset as exact rationals (the env stores float64 1/11).
EXACT_RESET = [
    Fraction(1, len(VARIED_SUPPORT)) if s in VARIED_SUPPORT else Fraction(0)
    for s in range(16)
]


def test_finite_horizon_value_on_a_hand_solved_chain():
    # 0 -a0-> 1 -a0-> goal (reward 1); a1 stays. Uniform random, horizon 2:
    # success needs a0 twice, probability 1/4; horizon 3 adds two orders, 1/2.
    P = {
        0: {0: [(1.0, 1, 0.0, False)], 1: [(1.0, 0, 0.0, False)]},
        1: {0: [(1.0, 2, 1.0, True)], 1: [(1.0, 1, 0.0, False)]},
        2: {0: [(1.0, 2, 0.0, True)], 1: [(1.0, 2, 0.0, True)]},
    }
    value = env_baselines.finite_horizon_value
    assert value(P, [1, 0, 0], horizon=2, n_actions=2) == Fraction(1, 4)
    assert value(P, [1, 0, 0], horizon=3, n_actions=2) == Fraction(1, 2)
    assert value(P, [1, 0, 0], horizon=1, n_actions=2, optimal=True) == 0
    assert value(P, [1, 0, 0], horizon=2, n_actions=2, optimal=True) == 1


def test_varied_start_reference_is_the_exact_uniform_random_value():
    P = _varied_P()
    mu = EXACT_RESET
    exact = env_baselines.finite_horizon_value(P, mu, horizon=100, n_actions=4)
    assert isinstance(exact, Fraction)
    assert float(exact) == VARIED_RANDOM_SCORE
    reference = env_baselines.REFERENCE_BASELINES[VARIED]
    assert reference == {"random_score": VARIED_RANDOM_SCORE, "expert_score": 1.0}
    # Not the old single-start reference.
    assert reference != env_baselines.REFERENCE_BASELINES["FrozenLake-v1"]
    optimal = env_baselines.finite_horizon_value(
        P,
        mu,
        horizon=100,
        n_actions=4,
        optimal=True,
    )
    assert optimal == 1
    ok, _ = env_baselines.validate_expert_quality(VARIED, 1.0)
    assert ok


def _success(P, policy, start, cap=100):
    s = start
    for _ in range(cap):
        ((p, nxt, reward, done),) = P[s][policy(s)]
        assert p == 1.0
        if done:
            return Fraction(int(reward))
        s = nxt
    return Fraction(0)


def test_row_only_policies_cannot_solve_the_varied_start_task():
    P = _varied_P()
    found = restriction.get_restriction("frozenlake_varied_start_column_hidden")
    reps = sorted({found.representative[s] for s in VARIED_SUPPORT})
    assert len(reps) == 4
    assert all(found.representative[s] // 4 == s // 4 for s in VARIED_SUPPORT)
    n = len(VARIED_SUPPORT)
    values = []
    for actions in itertools.product(range(4), repeat=4):
        choice = dict(zip(reps, actions))

        def policy(s, choice=choice):
            return choice[found.representative[s]]

        values.append(sum(_success(P, policy, s) for s in VARIED_SUPPORT) / n)
    assert len(values) == 256
    best = max(values)
    assert best == Fraction(6, 11) and 0 < best < 1
    # The full observation solves every start within the cap.
    full = env_baselines.finite_horizon_value(
        P,
        [1.0 if s in VARIED_SUPPORT else 0.0 for s in range(16)],
        horizon=100,
        n_actions=4,
        optimal=True,
    )
    assert full == len(VARIED_SUPPORT)
