"""Random-offset grids of the continuous classical tasks (`random_grid`).

Every test is synthetic: arrays are built by hand, Box metadata comes from
local factories, and experts are untrained policies on the declared Box. No
training, no rollout and no network.
"""

import base64
import inspect
import math

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy

from imitation.experiments.agnostic import (
    random_grid,
    random_grid_tables,
    restriction,
    task_pilot,
    tasks,
)
from imitation.experiments.ftrl import policy_utils

SEED = 20261003
COUNTS = (2, 4, 8, 16, 32, 64, 128, 256)
PREFIX = {
    "CartPole-v1": "cartpole",
    "Acrobot-v1": "acrobot",
    "MountainCar-v0": "mountaincar",
    "LunarLander-v2": "lunarlander",
}
ENVS = tuple(PREFIX)
ALL_IDS = tuple(f"{PREFIX[e]}_random_grid{k}_seed{SEED}" for e in ENVS for k in COUNTS)
F32MAX = float(np.finfo(np.float32).max)

# Actual gymnasium 0.29.1 Box bounds (float32 values as Python floats).
ACTUAL_BOX = {
    "CartPole-v1": (
        [-4.800000190734863, -F32MAX, -0.41887903213500977, -F32MAX],
        [4.800000190734863, F32MAX, 0.41887903213500977, F32MAX],
    ),
    "Acrobot-v1": (
        [-1.0, -1.0, -1.0, -1.0, -12.566370964050293, -28.274333953857422],
        [1.0, 1.0, 1.0, 1.0, 12.566370964050293, 28.274333953857422],
    ),
    "MountainCar-v0": (
        [-1.2000000476837158, -0.07000000029802322],
        [0.6000000238418579, 0.07000000029802322],
    ),
    "LunarLander-v2": (
        [-1.5, -1.5, -5.0, -5.0, -3.1415927410125732, -5.0, -0.0, -0.0],
        [1.5, 1.5, 5.0, 5.0, 3.1415927410125732, 5.0, 1.0, 1.0],
    ),
}
KINDS = {
    "CartPole-v1": ["linear", "unbounded", "linear", "unbounded"],
    "Acrobot-v1": ["circular"] * 4 + ["linear"] * 2,
    "MountainCar-v0": ["linear", "linear"],
    "LunarLander-v2": ["linear"] * 6 + ["binary", "binary"],
}

# Pins copied from the frozen registry once; they must never move.
EXPECTED_TABLES_SHA256 = (
    "911bcf3fb79a00beaa9648c73c2a608e819d5f60aa794cf58d5dbb4648202e78"
)
EXPECTED_GRID_SHA256 = {
    "cartpole_random_grid2_seed20261003": (
        "3bf0fc3df985f5dd9a91a34dec489b72f034551fca0cdf79f890ca0440d70925"
    ),
    "cartpole_random_grid4_seed20261003": (
        "15cc2c6ff25b096ea7bb11a164d8c4a09c06bea1e1db6f590d249912dbc8a7d9"
    ),
    "cartpole_random_grid8_seed20261003": (
        "465c38ab91e18b58f52a722b4c741d0a822c79bbdbc0737202d2038f9fb1990e"
    ),
    "cartpole_random_grid16_seed20261003": (
        "116c1e16c61b225729ccbc57c1e9148eac9c5bf7881d12012dcd0b7a25bf742e"
    ),
    "cartpole_random_grid32_seed20261003": (
        "86b60f80c51d6167589ccffbce58ef3aa64b7321174b1762559ce553739de8a9"
    ),
    "cartpole_random_grid64_seed20261003": (
        "f203081efd1234eebde3e3e696771d171cd279da6fbc7069fdfb898d1a886603"
    ),
    "cartpole_random_grid128_seed20261003": (
        "b4815aa7f71705ffbfaea05370291b07ffa3b27970bc5d17fd27459d0a6fc9f9"
    ),
    "cartpole_random_grid256_seed20261003": (
        "6f43e9c3acc0088c49437ff4b6d1b31a07e768eb3fdd837fc60885da43879157"
    ),
    "acrobot_random_grid2_seed20261003": (
        "5e505be15168b01b0a044521b4c5b56a95df21ff3559ec4285b58f45899b296c"
    ),
    "acrobot_random_grid4_seed20261003": (
        "cc9671fc26041c6ad45c60bf773d88d4d1a49c7843a7b4918954dda9eae0427f"
    ),
    "acrobot_random_grid8_seed20261003": (
        "5c15d78d07e0d1cf614831b3f93dbdf9c4a0f6021151dadaf485209036d69d66"
    ),
    "acrobot_random_grid16_seed20261003": (
        "434bfbd66dd03814b81be7c74b763cc469f0595f026643d1c6ad969adc8e7cc2"
    ),
    "acrobot_random_grid32_seed20261003": (
        "b3b9877f8ecd62d67b3f42a19fe4ab124b96924542ea7553f4f94c11e82bc2b6"
    ),
    "acrobot_random_grid64_seed20261003": (
        "9355ed0918aaf928ff002a126e93eedb0569a517440491761390c77dcd38577d"
    ),
    "acrobot_random_grid128_seed20261003": (
        "118d2e024640654970e81c9e9039cfbe607acaf891585307d2216024a12b5f98"
    ),
    "acrobot_random_grid256_seed20261003": (
        "0d911dc313b4100e4dff2e14f7152c5433fe2fde7c933b12868cb53fc4760aee"
    ),
    "mountaincar_random_grid2_seed20261003": (
        "03fb9def96df685a84c5662a73e70f952920b0d32917643195a3eee9e2a01908"
    ),
    "mountaincar_random_grid4_seed20261003": (
        "a16b296fe79d74084368a4f8ff3a1d0b65ef6143ae12722a0e10430de44998a9"
    ),
    "mountaincar_random_grid8_seed20261003": (
        "31a7eb9704657da3d6b773b4e34e7745363a29b355cd33b2c07f629c63a7bccf"
    ),
    "mountaincar_random_grid16_seed20261003": (
        "0cf04b617b4eaf7be5a694102741efa9911235ea10f559ec0b9fcc491cbe27ab"
    ),
    "mountaincar_random_grid32_seed20261003": (
        "98153829114a529d441a7805fd768be5a208e29587fe8381bc6f099170f4919b"
    ),
    "mountaincar_random_grid64_seed20261003": (
        "e0e11e29c27167cbe55b30f59c1cb6b55f505ca0677e5f1c5d75117aa68ff07a"
    ),
    "mountaincar_random_grid128_seed20261003": (
        "ad7c63ddab2a904bea3874921074709c28c4a84184f0b363c20831f6b9e33258"
    ),
    "mountaincar_random_grid256_seed20261003": (
        "69a2e7361561a0174958505feba9c00f171bb427cbcfd256a1879d5045487a0b"
    ),
    "lunarlander_random_grid2_seed20261003": (
        "309f47ea219db2a6fb1c7fa2ab9a5d9161943c9379ec1e9872ddab75fdbb846c"
    ),
    "lunarlander_random_grid4_seed20261003": (
        "6d57bb1987375503caee780f1dced6f2fe8e7b71b7a0dea0b24c11065bf32725"
    ),
    "lunarlander_random_grid8_seed20261003": (
        "059a16fd7adc86789d22351dd3eb25e5e023aa84195362551e023b94da3dc1be"
    ),
    "lunarlander_random_grid16_seed20261003": (
        "2c38f116e3f2c29a369c3a297b2b4de468fef221ce47bb279937107dcc3b33a6"
    ),
    "lunarlander_random_grid32_seed20261003": (
        "91a1e42d1ae4a3edb0ab644c5efcd1c78bcc23fb2b65185ede98b8db6f1fe698"
    ),
    "lunarlander_random_grid64_seed20261003": (
        "63b925721732c6480cb500e5ea665488dec012d58e2bf9e91fa5034e88fc3314"
    ),
    "lunarlander_random_grid128_seed20261003": (
        "b58d8da388d8d72ea15a9aaa8ef8147c5dc862ef293ceb144b3eac427bcb1397"
    ),
    "lunarlander_random_grid256_seed20261003": (
        "dc4e923b5eaf09c0cf01fa9127b6374c81150d44d43be9dbb9c4e775758c4bbb"
    ),
}

# Digests of the original task descriptions (test_agnostic_replacement_masks).
LEGACY_TASK_DIGESTS = {
    "CartPole-v1": "bbe157c72171cd88f0a5e679e3f32f54c0d2756b13cfd381641172318fd1efc3",
    "Acrobot-v1": "82426d16d673aa9741879f6a8116829bc7ae360c508bbb893d78ed9080bf218c",
    "MountainCar-v0": (
        "1da99f9c7f6798270bd19a0132c80149e6f50de8a28c6e9b8ba41db2b860245c"
    ),
    "LunarLander-v2": (
        "37e378d602a85c2701349721e9f4819ef11d4b6bde3b3eb86fe7ea8666e64c84"
    ),
}
LEGACY_RESTRICTION_DIGESTS = {
    "identity": "ce17b3017a00aebb32d4c4107e18e0ef51a721b270eb84ac4048ab4613e91e86",
    "cart_position_angular_velocity_zero": (
        "9cd68277c24117f92d37670ef0ba11f54eb9a7dc8773ab3926367b9e25e2b634"
    ),
    "acrobot_theta2_dot_zero": (
        "419084743713927a2ae0470513f234b6703896c50408129d03c733f263ad5d70"
    ),
    "mountaincar_velocity_zero": (
        "395b2e8b256a5b5bff4848766f510fe14139929afdcfd88e370bc54ddd5c4307"
    ),
    "lunarlander_velocities_zero": (
        "a9ea0ae5e5ac851bd2ca28def1ac29a3431973a0ded091d8fad7e343ce67d4d6"
    ),
}
LEGACY_LADDERS = {
    "CartPole-v1": (
        "cart_position_angular_velocity_zero",
        "cart_position_velocity_angular_velocity_zero",
    ),
    "Acrobot-v1": (
        "acrobot_theta2_dot_zero",
        "acrobot_theta_dots_zero",
        "acrobot_theta2_and_dots_zero",
    ),
    "MountainCar-v0": ("mountaincar_velocity_zero", "mountaincar_position_zero"),
    "LunarLander-v2": (
        "lunarlander_angular_velocity_zero",
        "lunarlander_velocities_zero",
        "lunarlander_velocities_angle_zero",
    ),
}


def _box(env_name, low=None, high=None):
    lo, hi = ACTUAL_BOX[env_name]
    return gym.spaces.Box(
        low=np.array(lo if low is None else low, dtype=np.float32),
        high=np.array(hi if high is None else high, dtype=np.float32),
        dtype=np.float32,
    )


def _torch(rid, obs):
    with th.no_grad():
        return restriction.ObservationMask(rid)(th.as_tensor(obs)).numpy()


def _bits(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint32 if a.dtype == np.float32 else np.uint64)


def _nextafter32(values, direction):
    v = np.asarray(values, dtype=np.float32)
    return np.nextafter(v, np.float32(direction * np.inf))


def _probe_inputs(grid, n_random=2000, seed=0):
    """Random, boundary-adjacent and extreme float32 inputs of one grid."""
    d = grid.obs_shape[0]
    rng = np.random.default_rng(seed)
    lo, hi = (np.array(b, dtype=np.float64) for b in ACTUAL_BOX[grid.env_name])
    finite_lo = np.where(np.abs(lo) > 1e30, -50.0, lo)
    finite_hi = np.where(np.abs(hi) > 1e30, 50.0, hi)
    span = finite_hi - finite_lo
    rows = [rng.uniform(finite_lo - 0.2 * span, finite_hi + 0.2 * span, (n_random, d))]
    # Log-scale magnitudes over the whole float32 range, both signs.
    mags = 10.0 ** rng.uniform(-45, 38.5, (n_random, d))
    rows.append(np.clip(mags * rng.choice([-1.0, 1.0], (n_random, d)), -F32MAX, F32MAX))
    extremes = [0.0, -0.0, 1e30, -1e30, F32MAX, -F32MAX, 1.4e-45, -1.4e-45]
    for value in extremes:
        rows.append(np.full((1, d), value))
    base = np.concatenate(rows).astype(np.float32)
    probes = [base]
    for axis in grid.axes:
        if axis.kind in ("linear", "unbounded"):
            points = np.concatenate(
                [
                    np.asarray(axis.edges, dtype=np.float32),
                    _nextafter32(axis.edges, 1),
                    _nextafter32(axis.edges, -1),
                    np.asarray(axis.centers, dtype=np.float32),
                    np.float32([lo[axis.indices[0]], hi[axis.indices[0]]]),
                ]
            )
            block = np.repeat(base[:1], len(points), axis=0)
            block[:, axis.indices[0]] = points
            probes.append(block)
        elif axis.kind == "circular":
            ec = np.asarray(axis.edge_cos, dtype=np.float32)
            es = np.asarray(axis.edge_sin, dtype=np.float32)
            pairs = [
                (ec, es),
                (_nextafter32(ec, 1), es),
                (_nextafter32(ec, -1), es),
                (ec, _nextafter32(es, 1)),
                (ec, _nextafter32(es, -1)),
                (-ec, -es),  # opposite rays
                (np.float32(axis.center_cos), np.float32(axis.center_sin)),
                # theta = +/- pi, 0, +/- pi / 2 and the zero pair.
                (
                    np.float32([-1, -1, 1, 0, 0, 0, 0]),
                    np.float32([0, -0.0, 0, 1, -1, 0, -0.0]),
                ),
            ]
            for c, s in pairs:
                block = np.repeat(base[:1], len(c), axis=0)
                block[:, axis.indices[0]] = c
                block[:, axis.indices[1]] = s
                probes.append(block)
    return np.concatenate(probes).astype(np.float32)


# ---------------------------------------------------------------------------
# Registration and legacy invariance
# ---------------------------------------------------------------------------


def test_registers_exactly_32_grids_eight_counts_per_env():
    assert tasks.RANDOM_GRID_SEED == SEED
    assert tasks.RANDOM_GRID_COUNTS == COUNTS
    assert tasks.RANDOM_GRID_GENERATOR == "random_offset_grid/1"
    assert tuple(tasks.RANDOM_GRID_SPECS) == ALL_IDS
    assert restriction.RANDOM_GRID_RESTRICTIONS == ALL_IDS
    assert set(tasks.RANDOM_GRID_CANDIDATES) == set(ENVS)
    for env_name in ENVS:
        ids = tasks.RANDOM_GRID_CANDIDATES[env_name]
        assert ids == tuple(tasks.random_grid_id(env_name, k, SEED) for k in COUNTS)
        for k, rid in zip(COUNTS, ids):
            spec = tasks.RANDOM_GRID_SPECS[rid]
            assert (spec.env_name, spec.n_bins, spec.seed) == (env_name, k, SEED)
            found = restriction.get_restriction(rid)
            assert isinstance(found, random_grid.RandomGridRestriction)
            assert (found.env_name, found.n_bins, found.seed) == (env_name, k, SEED)


def test_grids_stay_off_ladders_and_task_descriptions():
    for env_name in ENVS:
        task = tasks.get_task(env_name)
        assert task.ladder == LEGACY_LADDERS[env_name]
        assert not set(ALL_IDS) & set(task.restriction_ids)
    assert not set(ALL_IDS) & set(tasks.FROZEN_TABLE_SHA256)
    for rid, digest in LEGACY_RESTRICTION_DIGESTS.items():
        assert tasks._digest(restriction.describe_restriction(rid)) == digest, rid


# Strategic candidates declared before the grids; the grids only append.
STRATEGIC = {
    "MountainCar-v0": (
        "mountaincar_bins_pos005_vel005",
        "mountaincar_bins_pos005_vel015",
        "mountaincar_bins_pos010_vel005",
        "mountaincar_bins_pos010_vel015",
    ),
}


def grid_ids(env_name):
    return tuple(f"{PREFIX[env_name]}_random_grid{k}_seed{SEED}" for k in COUNTS)


def test_grids_are_appended_calibration_candidates_after_strategic():
    seen = []
    for env_name in ENVS:
        candidates = tasks.calibration_candidates(env_name)
        assert candidates == STRATEGIC.get(env_name, ()) + grid_ids(env_name)
        assert tasks.CALIBRATION_CANDIDATES[env_name] == candidates
        assert tasks.RANDOM_GRID_CANDIDATES[env_name] == grid_ids(env_name)
        seen += [rid for rid in candidates if rid in ALL_IDS]
    assert tuple(seen) == ALL_IDS and len(set(seen)) == 32
    # No other environment picks up a grid.
    for env_name, candidates in tasks.CALIBRATION_CANDIDATES.items():
        if env_name not in ENVS:
            assert not set(candidates) & set(ALL_IDS)


@pytest.mark.parametrize("env_name", ENVS)
def test_runner_spec_selects_all_grids_only_when_opted_in(env_name):
    task = tasks.get_task(env_name)
    ids = grid_ids(env_name)
    assert task_pilot.calibration_candidates_of(env_name)[-len(ids) :] == ids
    for kwargs in ({}, {"paired_cold_fits": True}, {"bc_prefix": True}):
        plain = task_pilot.runner_spec(env_name, **kwargs)
        assert plain.run_restrictions == task.restriction_ids
        assert plain.calibration_candidates == ()
        spec = task_pilot.runner_spec(env_name, calibration_candidates=True, **kwargs)
        assert spec.calibration_candidates == STRATEGIC.get(env_name, ()) + ids
        assert spec.run_restrictions == task.restriction_ids + (
            spec.calibration_candidates
        )
        assert not set(spec.run_restrictions) & (set(ALL_IDS) - set(ids))


def test_runner_and_audit_membership_agree_on_all_32_grids():
    runner, audit = set(), set()
    for env_name in ENVS:
        task = tasks.get_task(env_name)
        spec = task_pilot.runner_spec(env_name, calibration_candidates=True)
        runner |= set(spec.run_restrictions) & set(ALL_IDS)
        # The membership expression of `task_audit.audit_restriction`.
        allowed = task.ladder + tasks.calibration_candidates(env_name)
        audit |= set(allowed) & set(ALL_IDS)
    assert runner == audit == set(ALL_IDS)


@pytest.mark.parametrize("env_name", ENVS)
def test_legacy_task_digest_is_unchanged_and_carries_no_box_bits(env_name):
    task = tasks.get_task(env_name)
    record = tasks.describe(task)
    assert tasks.task_sha256(task) == LEGACY_TASK_DIGESTS[env_name]
    assert set(record["observation_space"]) == {"type", "shape", "dtype"}


def test_random_grid_id_validates_and_get_restriction_refuses_unregistered():
    assert tasks.random_grid_id("Acrobot-v1", 8, SEED) == (
        "acrobot_random_grid8_seed20261003"
    )
    for bad in (("Taxi-v3", 8, SEED), (None, 8, SEED)):
        with pytest.raises(ValueError, match="no random grid"):
            tasks.random_grid_id(*bad)
    for bad in ((True, SEED), (8.0, SEED), (0, SEED), (8, -1), (8, 2**32), (8, 1.5)):
        with pytest.raises(ValueError):
            tasks.random_grid_id("CartPole-v1", *bad)
    for rid in (
        tasks.random_grid_id("CartPole-v1", 3, SEED),  # count off the ladder
        tasks.random_grid_id("CartPole-v1", 8, SEED + 1),  # seed not frozen
        "cartpole_random_grid8",
        None,
        8,
    ):
        with pytest.raises(ValueError):
            restriction.get_restriction(rid)
        with pytest.raises(ValueError, match="not a random grid"):
            random_grid.get_grid(rid)


# ---------------------------------------------------------------------------
# Box metadata, offsets, frozen tables and pins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_name", ENVS)
def test_declared_box_bits_match_actual_gym_metadata(env_name):
    lo, hi = ACTUAL_BOX[env_name]
    bits = random_grid.DECLARED_BOX_BITS[env_name]
    assert bits["low"] == random_grid.float32_bits(lo)
    assert bits["high"] == random_grid.float32_bits(hi)
    space = gym.make(env_name).observation_space
    assert random_grid.float32_bits(space.low) == bits["low"]
    assert random_grid.float32_bits(space.high) == bits["high"]
    record = random_grid.describe_box(env_name)
    assert record["low_bits"] == list(bits["low"])
    kinds = {i: f.kind for f in random_grid.GRID_FEATURES[env_name] for i in f.indices}
    assert [kinds[i] for i in range(len(lo))] == KINDS[env_name]


def test_cartpole_velocity_sentinels_are_unbounded_not_finite_bins():
    for k in COUNTS:
        grid = random_grid.get_grid(tasks.random_grid_id("CartPole-v1", k, SEED))
        for axis in grid.axes[1], grid.axes[3]:
            assert axis.kind == "unbounded" and axis.low is None and axis.high is None
            assert axis.scale == 1.0
            # Interior edges are moderate, not spread over +/- 3.4e38.
            assert max(abs(e) for e in axis.edges) < 1e4
            assert all(abs(c) < 1e4 for c in axis.centers)
            # Each edge is the tan image of its offset grid point.
            for m, edge in enumerate(axis.edges):
                t = 0.5 + math.atan(edge) / math.pi
                assert abs(t - (axis.offset + m) / k) < 1e-12


@pytest.mark.parametrize("env_name", ENVS)
def test_check_box_refuses_other_bounds_shapes_and_dtypes(env_name):
    grid = random_grid.get_grid(tasks.RANDOM_GRID_CANDIDATES[env_name][0])
    grid.check_box(_box(env_name))
    lo, hi = ACTUAL_BOX[env_name]
    d = len(lo)
    nudged = list(hi)
    nudged[0] = float(np.nextafter(np.float32(hi[0]), np.float32(np.inf)))
    bad = [
        _box(env_name, high=nudged),
        gym.spaces.Box(-np.inf, np.inf, (d,), np.float32),
        gym.spaces.Box(np.float64(-1), np.float64(1), (d,), np.float64),
        gym.spaces.Box(-1, 1, (d + 1,), np.float32),
        gym.spaces.Discrete(3),
    ]
    for space in bad:
        with pytest.raises(tasks.TaskMismatch):
            grid.check_box(space)


def test_offsets_regenerate_from_a_local_random_state():
    for env_name in ENVS:
        drawn = random_grid.draw_offsets(env_name)
        assert tuple(u.hex() for u in drawn) == random_grid.FROZEN_OFFSETS[env_name]
        assert len(drawn) == len(ACTUAL_BOX[env_name][0])
    # One stream per coordinate index, contact and sin offsets included.
    lunar = random_grid.FROZEN_OFFSETS["LunarLander-v2"]
    for env_name in ENVS:
        offsets = random_grid.FROZEN_OFFSETS[env_name]
        assert offsets == lunar[: len(offsets)]


def test_offsets_are_shared_by_every_count():
    for env_name in ENVS:
        offsets = [float.fromhex(h) for h in random_grid.FROZEN_OFFSETS[env_name]]
        for rid in tasks.RANDOM_GRID_CANDIDATES[env_name]:
            grid = random_grid.get_grid(rid)
            assert grid.offsets == random_grid.FROZEN_OFFSETS[env_name]
            for axis in grid.axes:
                assert axis.offset == offsets[axis.indices[0]]


def test_frozen_libm_tables_match_regeneration_and_pins():
    assert random_grid.FROZEN_TABLES_SHA256 == EXPECTED_TABLES_SHA256
    assert random_grid_tables.TABLES_SHA256 == EXPECTED_TABLES_SHA256
    frozen = random_grid.frozen_tables()
    regen = random_grid.generate_libm_tables()
    assert set(frozen) == set(regen)
    assert len(frozen) == 2 * len(COUNTS) * 2 + 2 * len(COUNTS) * 4
    for key, values in frozen.items():
        a = np.array(values, dtype=np.float64)
        b = np.array(regen[key], dtype=np.float64)
        if key.endswith(("/centers", "center_cos", "center_sin")):
            np.testing.assert_array_equal(a, a.astype(np.float32))
            np.testing.assert_array_max_ulp(
                a.astype(np.float32),
                b.astype(np.float32),
                maxulp=1,
            )
        else:
            np.testing.assert_array_max_ulp(a, b, maxulp=4)


def test_descriptor_digests_are_pinned():
    assert random_grid.FROZEN_GRID_SHA256 == EXPECTED_GRID_SHA256
    assert len(EXPECTED_GRID_SHA256) == 32
    for rid in ALL_IDS:
        grid = random_grid.get_grid(rid)
        assert grid.descriptor_sha256 == EXPECTED_GRID_SHA256[rid]
        assert restriction.restriction_digest(grid) == EXPECTED_GRID_SHA256[rid]
        assert tasks._digest(grid.descriptor()) == EXPECTED_GRID_SHA256[rid]


@pytest.fixture
def fresh_cache():
    random_grid.get_grid.cache_clear()
    random_grid._frozen_tables_json.cache_clear()
    yield
    random_grid.get_grid.cache_clear()
    random_grid._frozen_tables_json.cache_clear()


def test_descriptor_pin_mismatch_is_refused(monkeypatch, fresh_cache):
    rid = ALL_IDS[0]
    monkeypatch.setitem(random_grid.FROZEN_GRID_SHA256, rid, "0" * 64)
    with pytest.raises(RuntimeError, match="differs from the frozen"):
        random_grid.get_grid(rid)


def test_tampered_tables_or_offsets_are_refused(monkeypatch, fresh_cache):
    key = sorted(random_grid_tables.TABLES)[0]
    tables = dict(random_grid_tables.TABLES)
    raw = bytearray(base64.b64decode(tables[key]))
    raw[0] ^= 1  # one ulp of the first value
    tables[key] = base64.b64encode(bytes(raw)).decode()
    with monkeypatch.context() as m:
        m.setattr(random_grid_tables, "TABLES", tables)
        with pytest.raises(RuntimeError, match="tables"):
            random_grid.get_grid("cartpole_random_grid2_seed20261003")
    random_grid._frozen_tables_json.cache_clear()
    random_grid.get_grid.cache_clear()
    offsets = dict(random_grid.FROZEN_OFFSETS)
    offsets["MountainCar-v0"] = ("0x1.0p-1", offsets["MountainCar-v0"][1])
    monkeypatch.setattr(random_grid, "FROZEN_OFFSETS", offsets)
    with pytest.raises(RuntimeError, match="differs from the frozen"):
        random_grid.get_grid("mountaincar_random_grid2_seed20261003")


# ---------------------------------------------------------------------------
# Structure and hand-checked values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rid", ALL_IDS)
def test_grid_structure(rid):
    grid = random_grid.get_grid(rid)
    k = grid.n_bins
    assert [a.kind for a in grid.axes for _ in a.indices] == KINDS[grid.env_name]
    assert grid.box_low_bits == random_grid.DECLARED_BOX_BITS[grid.env_name]["low"]
    expected_cells = 1
    for axis in grid.axes:
        if axis.kind in ("linear", "unbounded"):
            assert len(axis.edges) == k and len(axis.centers) == k + 1
            assert all(b > a for a, b in zip(axis.edges, axis.edges[1:]))
            expected_cells *= k + 1
        elif axis.kind == "circular":
            assert len(axis.edge_cos) == len(axis.center_cos) == k
            c = np.array(axis.center_cos, dtype=np.float64)
            s = np.array(axis.center_sin, dtype=np.float64)
            assert np.all(np.abs(c * c + s * s - 1) < 2.0**-22)
            assert np.allclose(np.hypot(axis.edge_cos, axis.edge_sin), 1, atol=1e-15)
            expected_cells *= k
        else:
            assert not axis.edges and not axis.centers
            expected_cells *= 2
        if axis.kind == "linear":
            lo, hi = axis.low, axis.high
            w = (hi - lo) / k
            assert axis.edges == tuple(lo + (axis.offset + m) * w for m in range(k))
            assert lo < axis.centers[0] < axis.edges[0]
            assert axis.edges[-1] < axis.centers[-1] < hi
    assert grid.nominal_cells == expected_cells
    record = restriction.describe_restriction(rid)
    assert record["kind"] == "random_grid"
    assert record["descriptor_sha256"] == grid.descriptor_sha256
    assert record["nominal_cells"] == expected_cells
    assert record["generator"] == "random_offset_grid/1"
    assert record["seed"] == SEED and record["n_bins"] == k
    assert record["rule"] == random_grid.RANDOM_GRID_RULE


def test_mountaincar_k2_hand_checked_values():
    grid = random_grid.get_grid("mountaincar_random_grid2_seed20261003")
    u0, u1 = (float.fromhex(h) for h in random_grid.FROZEN_OFFSETS["MountainCar-v0"])
    lo, hi = -1.2000000476837158, 0.6000000238418579
    w = (hi - lo) / 2
    edges = (lo + u0 * w, lo + (u0 + 1) * w)
    assert grid.axes[0].edges == edges
    centers = [(lo + edges[0]) / 2, (edges[0] + edges[1]) / 2, (edges[1] + hi) / 2]
    assert grid.axes[0].centers == tuple(float(np.float32(c)) for c in centers)
    obs = np.array(
        [[-2.0, 0.0], [edges[0] - 0.01, 0.0], [0.0, 0.0], [5.0, 0.0]],
        dtype=np.float32,
    )
    out = grid.apply(obs)
    np.testing.assert_array_equal(
        out[:, 0], np.float32([centers[0], centers[0], centers[1], centers[2]])
    )


# ---------------------------------------------------------------------------
# Mapping: parity, finiteness, idempotence, contacts, circular cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rid", ALL_IDS)
def test_numpy_and_torch_agree_bitwise_and_outputs_are_stable(rid):
    grid = random_grid.get_grid(rid)
    obs = _probe_inputs(grid, seed=ALL_IDS.index(rid))
    out = grid.apply(obs)
    out_t = _torch(rid, obs)
    assert out.dtype == out_t.dtype == np.float32
    np.testing.assert_array_equal(_bits(out), _bits(out_t))
    assert np.all(np.isfinite(out))
    # Idempotent, also through torch.
    np.testing.assert_array_equal(_bits(grid.apply(out)), _bits(out))
    np.testing.assert_array_equal(_bits(_torch(rid, out)), _bits(out))
    # Row-wise: batching and leading shape do not change values.
    np.testing.assert_array_equal(_bits(grid.apply(obs[3])), _bits(out[3]))
    stacked = obs[:12].reshape(3, 4, -1)
    np.testing.assert_array_equal(
        _bits(grid.apply(stacked)), _bits(out[:12].reshape(3, 4, -1))
    )
    np.testing.assert_array_equal(
        _bits(_torch(rid, stacked)), _bits(out[:12].reshape(3, 4, -1))
    )
    # float64 input: same values in float64.
    out64 = grid.apply(obs.astype(np.float64))
    assert out64.dtype == np.float64
    np.testing.assert_array_equal(out64, out.astype(np.float64))
    np.testing.assert_array_equal(_torch(rid, obs.astype(np.float64)), out64)
    # Every output coordinate is a frozen representative or a passed flag.
    for axis in grid.axes:
        if axis.kind in ("linear", "unbounded"):
            assert set(out[:, axis.indices[0]].tolist()) <= set(axis.centers)
        elif axis.kind == "circular":
            pairs = set(
                zip(out[:, axis.indices[0]].tolist(), out[:, axis.indices[1]].tolist())
            )
            assert pairs <= set(zip(axis.center_cos, axis.center_sin))
        else:
            i = axis.indices[0]
            np.testing.assert_array_equal(_bits(out[:, i]), _bits(obs[:, i]))


@pytest.mark.skipif(not th.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("rid", ALL_IDS)
def test_numpy_and_cuda_grid_mask_agree_bitwise(rid):
    grid = random_grid.get_grid(rid)
    mask = random_grid.GridMask(grid).to("cuda:0")

    def cuda(obs):
        with th.no_grad():
            return mask(th.as_tensor(obs, device="cuda:0")).cpu().numpy()

    # Same probes as the CPU parity test: edges +/- 1 ulp, seams, extremes.
    obs = _probe_inputs(grid, seed=ALL_IDS.index(rid))
    out = grid.apply(obs)
    out_c = cuda(obs)
    assert out_c.dtype == np.float32
    np.testing.assert_array_equal(_bits(out_c), _bits(out))
    np.testing.assert_array_equal(_bits(cuda(out)), _bits(out))  # idempotent
    out64 = grid.apply(obs.astype(np.float64))
    np.testing.assert_array_equal(_bits(cuda(obs.astype(np.float64))), _bits(out64))


@pytest.mark.parametrize("rid", ALL_IDS)
def test_edges_are_half_open_and_out_of_box_values_clamp(rid):
    grid = random_grid.get_grid(rid)
    base = grid.apply(_probe_inputs(grid, n_random=1)[:1])[0]
    for axis in grid.axes:
        if axis.kind not in ("linear", "unbounded"):
            continue
        i = axis.indices[0]
        for m, edge in enumerate(axis.edges):
            at = np.float32(edge)
            # float32 values at or above the float64 edge are in cell m + 1.
            for value in (
                at,
                np.nextafter(at, np.float32(np.inf)),
                np.nextafter(at, np.float32(-np.inf)),
            ):
                row = base.copy()
                row[i] = value
                expected = (
                    axis.centers[m + 1] if float(value) >= edge else axis.centers[m]
                )
                assert grid.apply(row)[i] == np.float32(expected)
        for value, cell in ((-F32MAX, 0), (-1e30, 0), (1e30, -1), (F32MAX, -1)):
            row = base.copy()
            row[i] = value
            assert grid.apply(row)[i] == np.float32(axis.centers[cell])
        if axis.kind == "linear":
            for value, cell in ((axis.low - 1.0, 0), (axis.high + 1.0, -1)):
                row = base.copy()
                row[i] = value
                assert grid.apply(row)[i] == np.float32(axis.centers[cell])


@pytest.mark.parametrize("k", COUNTS)
def test_acrobot_arcs_zero_pair_and_angle_seams(k):
    rid = tasks.random_grid_id("Acrobot-v1", k, SEED)
    grid = random_grid.get_grid(rid)
    axis = grid.axes[0]
    obs = np.zeros((1, 6), dtype=np.float32)
    out = grid.apply(obs)
    # The zero pair (invalid simulator input) goes to arc 0.
    assert (out[0, 0], out[0, 1]) == (
        np.float32(axis.center_cos[0]),
        np.float32(axis.center_sin[0]),
    )
    np.testing.assert_array_equal(_torch(rid, obs), out)
    # Angles sampled on a fine grid land in the arc that contains them.
    theta = np.linspace(-math.pi, math.pi, 4001)[:-1] + 1e-4
    rows = np.zeros((len(theta), 6), dtype=np.float32)
    rows[:, 0], rows[:, 1] = np.cos(theta), np.sin(theta)
    rows[:, 2], rows[:, 3] = np.cos(theta), np.sin(theta)
    out = grid.apply(rows)
    arc = np.floor(((theta / (2 * math.pi)) * k - axis.offset) % k).astype(int)
    expected_c = np.float32(axis.center_cos)[arc]
    expected_s = np.float32(axis.center_sin)[arc]
    distance = np.abs(((theta / (2 * math.pi)) * k - axis.offset) % 1)
    clear = (distance > 1e-6) & (distance < 1 - 1e-6)
    np.testing.assert_array_equal(out[clear, 0], expected_c[clear])
    np.testing.assert_array_equal(out[clear, 1], expected_s[clear])
    # theta = +pi and -pi (sin = +0 and -0) share one arc.
    seam = np.zeros((2, 6), dtype=np.float32)
    seam[:, 0] = -1.0
    seam[1, 1] = -0.0
    out = grid.apply(seam)
    np.testing.assert_array_equal(out[0], out[1])
    # Magnitude does not matter: a scaled pair keeps its arc.
    scaled = rows[:50] * np.float32(1e-30)
    scaled[:, 4:] = 0
    np.testing.assert_array_equal(
        grid.apply(scaled)[:, :4], grid.apply(rows[:50])[:, :4]
    )


@pytest.mark.parametrize("k", COUNTS[1:])
def test_every_nonzero_pair_is_decided_by_exactly_one_arc(k):
    # The arc-0 fallback must only ever see the zero pair.
    grid = random_grid.get_grid(tasks.random_grid_id("Acrobot-v1", k, SEED))
    obs = _probe_inputs(grid, n_random=3000, seed=k).astype(np.float64)
    for axis in grid.axes[:2]:
        c, s = obs[:, axis.indices[0]], obs[:, axis.indices[1]]
        nonzero = (c != 0) | (s != 0)
        ec, es = np.array(axis.edge_cos), np.array(axis.edge_sin)
        ahead = ec * s[:, None] - es * c[:, None] >= 0
        starts = (ahead & ~np.roll(ahead, -1, axis=-1)).sum(-1)
        assert np.all(starts[nonzero] == 1)
        assert np.all(starts[~nonzero] == 0)
        assert (~nonzero).any()


def test_lunar_contacts_pass_through_unchanged():
    for rid in tasks.RANDOM_GRID_CANDIDATES["LunarLander-v2"]:
        grid = random_grid.get_grid(rid)
        obs = np.zeros((6, 8), dtype=np.float32)
        obs[:, 6] = [0.0, 1.0, 0.0, 1.0, -0.0, 0.25]
        obs[:, 7] = [0.0, 0.0, 1.0, 1.0, 1.0, 7.0]
        for out in (grid.apply(obs), _torch(rid, obs)):
            np.testing.assert_array_equal(_bits(out[:, 6:]), _bits(obs[:, 6:]))


# ---------------------------------------------------------------------------
# Fail closed and side effects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_name", ENVS)
def test_inputs_fail_closed(env_name):
    rid = tasks.RANDOM_GRID_CANDIDATES[env_name][2]
    grid = random_grid.get_grid(rid)
    mask = restriction.ObservationMask(rid)
    d = grid.obs_shape[0]
    good = np.zeros((2, d), dtype=np.float32)
    for bad in (np.zeros((2, d + 1), np.float32), np.zeros((), np.float32)):
        with pytest.raises(ValueError, match="shape"):
            grid.apply(bad)
        with pytest.raises(ValueError, match="shape"):
            mask(th.as_tensor(bad))
    for dtype in (np.int64, np.float16, np.bool_):
        with pytest.raises(ValueError, match="dtype"):
            grid.apply(good.astype(dtype))
        with pytest.raises(ValueError, match="dtype"):
            mask(th.as_tensor(good.astype(dtype)))
    with pytest.raises(ValueError, match="dtype"):
        grid.apply(np.array([["a"] * d]))
    for value in (np.nan, np.inf, -np.inf):
        bad = good.copy()
        bad[1, -1] = value
        with pytest.raises(ValueError, match="not finite"):
            grid.apply(bad)
        with pytest.raises(ValueError, match="not finite"):
            mask(th.as_tensor(bad))


def test_no_global_rng_side_effects(fresh_cache):
    np.random.seed(5)
    th.manual_seed(5)
    np_state = np.random.get_state()
    th_state = th.get_rng_state()
    for env_name in ENVS:
        random_grid.draw_offsets(env_name)
    random_grid.generate_libm_tables()
    for rid in ALL_IDS:
        grid = random_grid.get_grid(rid)
        obs = np.zeros((3, grid.obs_shape[0]), dtype=np.float32)
        grid.apply(obs)
        _torch(rid, obs)
    after = np.random.get_state()
    assert after[0] == np_state[0] and np.array_equal(after[1], np_state[1])
    assert after[2:] == np_state[2:]
    assert th.equal(th.get_rng_state(), th_state)


def test_mask_is_parameter_free_with_non_persistent_buffers():
    rid = "acrobot_random_grid16_seed20261003"
    mask = restriction.ObservationMask(rid)
    assert not list(mask.parameters())
    assert mask.state_dict() == {}
    assert len(list(mask.buffers())) > 0


def test_construction_api_takes_no_policy_or_data():
    assert list(inspect.signature(random_grid.get_grid).parameters) == [
        "restriction_id"
    ]
    assert list(inspect.signature(random_grid.draw_offsets).parameters) == [
        "env_name",
        "seed",
    ]
    assert list(inspect.signature(random_grid.generate_libm_tables).parameters) == []


# ---------------------------------------------------------------------------
# Learner integration and checkpoints
# ---------------------------------------------------------------------------


def _expert(env_name, seed=0):
    th.manual_seed(seed)
    n = tasks.get_task(env_name).n_actions
    return ActorCriticPolicy(
        _box(env_name),
        gym.spaces.Discrete(n),
        lambda _: 3e-4,
        net_arch=dict(pi=[16], vf=[16]),
    )


@pytest.mark.parametrize("env_name", ENVS)
def test_learner_maps_every_path_once_and_leaves_expert_alone(env_name):
    rid = tasks.RANDOM_GRID_CANDIDATES[env_name][3]
    grid = random_grid.get_grid(rid)
    expert = _expert(env_name)
    before = {k: v.clone() for k, v in expert.state_dict().items()}
    learner = restriction.make_linear_policy(expert, rid)
    for key, value in expert.state_dict().items():
        assert th.equal(value, before[key])
    with th.no_grad():
        learner.action_net.weight.normal_()
    obs = _probe_inputs(grid, n_random=50)[:200]
    masked = grid.apply(obs)
    with th.no_grad():
        latent = learner.mlp_extractor.forward_actor(
            learner.extract_features(th.as_tensor(obs), learner.pi_features_extractor)
        )
        reference = expert.mlp_extractor.forward_actor(
            expert.extract_features(th.as_tensor(masked), expert.pi_features_extractor)
        )
        probs = learner.get_distribution(th.as_tensor(obs)).distribution.probs
        probs_masked = learner.get_distribution(th.as_tensor(masked)).distribution.probs
        values, log_prob, _ = learner.evaluate_actions(
            th.as_tensor(obs), th.zeros(len(obs), dtype=th.long)
        )
        values_m, log_prob_m, _ = learner.evaluate_actions(
            th.as_tensor(masked), th.zeros(len(obs), dtype=th.long)
        )
    np.testing.assert_array_equal(latent.numpy(), reference.numpy())
    np.testing.assert_array_equal(probs.numpy(), probs_masked.numpy())
    np.testing.assert_array_equal(values.numpy(), values_m.numpy())
    np.testing.assert_array_equal(log_prob.numpy(), log_prob_m.numpy())
    np.testing.assert_array_equal(
        learner.predict(obs, deterministic=True)[0],
        learner.predict(masked, deterministic=True)[0],
    )


def test_restricted_policy_refuses_a_different_box():
    rid = "cartpole_random_grid8_seed20261003"
    expert = ActorCriticPolicy(
        gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
        gym.spaces.Discrete(2),
        lambda _: 3e-4,
    )
    with pytest.raises(tasks.TaskMismatch):
        restriction.make_linear_policy(expert, rid)


@pytest.mark.parametrize("rid", [ALL_IDS[0], ALL_IDS[11], ALL_IDS[21], ALL_IDS[31]])
def test_checkpoint_roundtrip_and_tamper_rejection(rid, tmp_path):
    grid = random_grid.get_grid(rid)
    policy = restriction.make_linear_policy(_expert(grid.env_name), rid)
    with th.no_grad():
        policy.action_net.weight.normal_()
    obs = _probe_inputs(grid, n_random=20)[:100]
    path = tmp_path / "policy.pt"
    policy.save(path)
    loaded = restriction.load_policy_checkpoint(path)
    assert loaded.restriction_id == rid
    assert loaded.features_extractor.mask.restriction is grid
    np.testing.assert_array_equal(
        loaded.predict(obs, deterministic=True)[0],
        policy.predict(obs, deterministic=True)[0],
    )
    with pytest.raises(TypeError):
        policy_utils.load_policy_checkpoint(path)
    saved = th.load(path, weights_only=False)
    assert saved["data"]["restriction_sha256"] == grid.descriptor_sha256
    saved["data"]["restriction_sha256"] = "0" * 64
    th.save(saved, path)
    with pytest.raises(ValueError, match="not the saved"):
        restriction.load_policy_checkpoint(path)
