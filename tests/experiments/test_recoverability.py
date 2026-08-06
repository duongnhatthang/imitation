import numpy as np
import pytest

from imitation.experiments.ftrl import recoverability, recoverability_tabular


def test_mu_from_q_is_max_minus_min():
    q = np.array([[1.0, 5.0, 2.0], [0.0, 0.0, 0.0], [-3.0, 1.0, -1.0]])
    mu = recoverability.mu_from_q(q)
    assert np.allclose(mu, [4.0, 0.0, 4.0])


def test_mu_nonnegative():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((100, 4))
    assert (recoverability.mu_from_q(q) >= 0).all()


@pytest.mark.expensive
def test_dqn_reference_reaches_expert_and_mu_reasonable(tmp_path):
    env_name = "CartPole-v1"
    expert_optimal = 500.0  # CartPole-v1 episodic return cap
    dqn = recoverability.get_or_train_dqn_reference(
        env_name, tmp_path, total_timesteps=50_000, seed=0
    )
    ret = recoverability.reference_return(dqn, env_name, n_episodes=20)
    # The reference must be NEAR-optimal, not merely "solved": a large gap below
    # the expert optimum means mu(s) is computed from an under-trained Q-function
    # (compressed/miscalibrated values) and is unreliable. A small gap is fine.
    assert ret >= 0.95 * expert_optimal, (
        f"DQN reference return {ret:.0f} is >5% below CartPole's optimum "
        f"{expert_optimal:.0f}; reference is under-trained, mu unreliable."
    )
    import gymnasium as gym

    env = gym.make(env_name)
    obs = np.stack([env.reset(seed=i)[0] for i in range(64)]).astype(np.float32)
    q = recoverability.dqn_q_values(dqn, obs)
    mu = recoverability.mu_from_q(q)
    assert mu.shape == (64,)
    assert (mu >= 0).all()
    # A near-optimal CartPole DQN (gamma=0.99) has a well-scaled value function;
    # V=max_a Q should be on the discounted-return scale (tens), not compressed
    # to single digits as it is when the reference is under-trained.
    assert q.max(axis=1).mean() > 40.0


def test_normalized_return_handles_negative_scales():
    # Acrobot-like: random -500, expert -80, dqn -90 -> normalized ~0.976
    n = recoverability.normalized_return(-90.0, -500.0, -80.0)
    assert 0.95 <= n <= 1.0


def test_recoverability_mu_dispatch_blackjack_returns_none(tmp_path):
    obs = np.zeros((4, 45))
    assert recoverability.recoverability_mu("Blackjack-v1", obs, tmp_path) is None


def test_recoverability_mu_toytext_uses_tabular(tmp_path, monkeypatch):
    monkeypatch.setattr(
        recoverability_tabular,
        "exact_mu_per_state",
        lambda env, expert, gamma=0.99: np.array([0.0, 1.0, 2.0, 3.0]),
    )
    obs = np.eye(4)[[3, 1]]  # state ids 3, 1
    mu = recoverability.recoverability_mu(
        "FrozenLake-v1", obs, tmp_path, expert_policy=object()
    )
    assert list(mu) == [3.0, 1.0]


@pytest.mark.expensive
def test_hub_dqn_qnet_shape_pong():
    dqn = recoverability.load_hub_dqn("PongNoFrameskip-v4")
    obs = np.zeros((5, 4, 84, 84), dtype=np.uint8)
    q = recoverability.dqn_q_values(dqn, obs)
    assert q.shape[0] == 5 and q.shape[1] >= 3  # Pong has 6 actions
    assert (recoverability.mu_from_q(q) >= 0).all()


def test_load_hub_dqn_prefers_local_cache(tmp_path, monkeypatch):
    env = "PongNoFrameskip-v4"
    local_dir = tmp_path / env
    local_dir.mkdir(parents=True)
    local_zip = local_dir / f"dqn-{env}.zip"
    local_zip.write_bytes(b"stub")
    captured = {}

    def fake_load(path, device="auto", custom_objects=None):
        captured["path"] = str(path)
        return "STUB_DQN"

    monkeypatch.setattr(recoverability.DQN, "load", staticmethod(fake_load))
    out = recoverability.load_hub_dqn(env, expert_cache=tmp_path)
    assert out == "STUB_DQN"
    assert captured["path"] == str(local_zip)  # local used, hub never touched
