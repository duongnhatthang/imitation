import numpy as np
import pytest

from imitation.experiments.ftrl import recoverability


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
