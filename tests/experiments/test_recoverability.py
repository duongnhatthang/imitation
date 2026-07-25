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
    dqn = recoverability.get_or_train_dqn_reference(
        env_name, tmp_path, total_timesteps=50_000, seed=0
    )
    ret = recoverability.reference_return(dqn, env_name, n_episodes=20)
    assert ret >= 195.0  # near CartPole's 500 cap; comfortably above random
    import gymnasium as gym

    env = gym.make(env_name)
    obs = np.stack([env.reset(seed=i)[0] for i in range(64)]).astype(np.float32)
    mu = recoverability.recoverability(dqn, obs)
    assert mu.shape == (64,)
    assert (mu >= 0).all()
