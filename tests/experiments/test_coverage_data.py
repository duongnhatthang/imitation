import numpy as np

from imitation.data import serialize, types
from imitation.experiments.ftrl import coverage_data


def _make_traj(n, d, val):
    # obs has n+1 rows (imitation convention); acts has n rows.
    obs = np.full((n + 1, d), float(val), dtype=np.float32)
    acts = np.zeros((n,), dtype=np.int64)
    rews = np.zeros((n,), dtype=np.float32)
    return types.TrajectoryWithRew(
        obs=obs, acts=acts, infos=None, terminal=True, rews=rews
    )


def _write_demo(root, algo, env, seed, round_num, traj):
    d = (
        coverage_data.scratch_demo_root(root, algo, env, seed)
        / f"round-{round_num:03d}"
    )
    d.mkdir(parents=True, exist_ok=True)
    serialize.save(d / f"demo-{round_num}.npz", [traj])


def test_load_algo_states_parses_round_and_drops_final_obs(tmp_path):
    _write_demo(tmp_path, "ftrl", "CartPole-v1", 0, 0, _make_traj(3, 4, 1.0))
    _write_demo(tmp_path, "ftrl", "CartPole-v1", 0, 2, _make_traj(2, 4, 2.0))
    cs = coverage_data.load_algo_states(tmp_path, "CartPole-v1", "ftrl", 0)
    assert cs.algo == "ftrl"
    assert cs.obs.shape == (5, 4)  # 3 + 2 states, final obs dropped
    assert sorted(set(cs.rounds.tolist())) == [0, 2]
    assert (cs.rounds == 0).sum() == 3 and (cs.rounds == 2).sum() == 2


def test_load_algo_states_missing_returns_none(tmp_path):
    assert coverage_data.load_algo_states(tmp_path, "CartPole-v1", "bc", 0) is None


def test_save_transitions_as_demos_roundtrips(tmp_path):
    from imitation.experiments.ftrl import run_experiment

    obs = np.arange(12, dtype=np.float32).reshape(4, 3)
    trans = types.Transitions(
        obs=obs,
        acts=np.zeros(4, dtype=np.int64),
        infos=np.array([{}] * 4),
        next_obs=obs + 1.0,
        dones=np.zeros(4, dtype=bool),
    )
    scratch = tmp_path / "scratch" / "bc_CartPole-v1_seed0"
    run_experiment._save_transitions_as_demos(
        trans, scratch, 0, np.random.default_rng(0)
    )
    cs = coverage_data.load_algo_states(tmp_path, "CartPole-v1", "bc", 0)
    assert cs is not None
    assert cs.obs.shape == (4, 3)
    assert (cs.rounds == 0).all()
    assert np.allclose(cs.obs, obs)  # obs[:-1] of the synthetic traj == original states


def _make_image_traj(n):
    """Trajectory with Atari-shaped (84, 84, 4) uint8 obs; n+1 obs rows."""
    obs = np.zeros((n + 1, 84, 84, 4), dtype=np.uint8)
    acts = np.zeros((n,), dtype=np.int64)
    rews = np.zeros((n,), dtype=np.float32)
    return types.TrajectoryWithRew(
        obs=obs, acts=acts, infos=None, terminal=True, rews=rews
    )


def test_load_algo_states_preserves_image_obs(tmp_path):
    T = 3
    _write_demo(tmp_path, "ftrl", "PongNoFrameskip-v4", 0, 0, _make_image_traj(T))
    cs = coverage_data.load_algo_states(tmp_path, "PongNoFrameskip-v4", "ftrl", 0)
    assert cs is not None
    # Shape must be (T, H, W, C) — NOT flattened to (T, 28224)
    assert cs.obs.shape == (T, 84, 84, 4), f"expected (3,84,84,4), got {cs.obs.shape}"
    # dtype must NOT be float32 (the old bug forced dtype=float32 on every obs)
    assert (
        cs.obs.dtype != np.float32
    ), f"obs dtype was coerced to float32; got {cs.obs.dtype}"


def test_pool_and_standardize(tmp_path):
    _write_demo(tmp_path, "bc", "CartPole-v1", 0, 0, _make_traj(4, 4, 5.0))
    _write_demo(tmp_path, "ftrl", "CartPole-v1", 0, 1, _make_traj(4, 4, 9.0))
    states = coverage_data.load_env_states(
        tmp_path, "CartPole-v1", 0, algos=["bc", "ftrl"]
    )
    pooled = coverage_data.pool(states)
    assert pooled.obs.shape == (8, 4)
    assert set(pooled.algo.tolist()) == {"bc", "ftrl"}
    feats = coverage_data.standardize_features(pooled.obs)
    assert feats.shape == (8, 4)
    assert np.allclose(feats.mean(axis=0), 0.0, atol=1e-6)
