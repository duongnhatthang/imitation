import gymnasium as gym
import numpy as np

from imitation.experiments.ftrl import recoverability_tabular as rt


class _GreedyDownRightExpert:
    """Stub expert: deterministic policy, one-hot obs -> fixed action per state."""

    def __init__(self, actions):
        self._actions = actions  # array [nS] of action ids

    def obs_to_tensor(self, obs):
        return np.asarray(obs), None

    def get_distribution(self, obs):
        import torch as th

        ids = np.asarray(obs).argmax(axis=1)
        probs = np.zeros((len(ids), 4), dtype=np.float32)
        probs[np.arange(len(ids)), self._actions[ids]] = 1.0

        class _D:
            distribution = type("x", (), {"probs": th.as_tensor(probs)})()

        return _D()


def test_state_ids_from_onehot():
    oh = np.eye(4)[[2, 0, 3]]
    assert list(rt.state_ids_from_onehot(oh)) == [2, 0, 3]


def test_exact_mu_matches_independent_q_on_deterministic_frozenlake():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    nS = env.observation_space.n
    # Expert always goes RIGHT(2) then DOWN(1) toward the goal; arbitrary but fixed.
    actions = np.full(nS, 2, dtype=int)
    expert = _GreedyDownRightExpert(actions)
    mu = rt.exact_mu_per_state("FrozenLake-v1", expert, gamma=0.9, is_slippery=False)
    # Independent recomputation of Q from env.P under the same policy.
    P = env.unwrapped.P
    probs = rt.expert_action_probs(expert, nS)
    V = np.zeros(nS)
    for _ in range(2000):
        Vn = np.zeros(nS)
        for s in range(nS):
            for a in range(env.action_space.n):
                q = sum(
                    p * (r + 0.9 * (0.0 if d else V[s2])) for p, s2, r, d in P[s][a]
                )
                Vn[s] += probs[s, a] * q
        V = Vn
    Q = np.zeros((nS, env.action_space.n))
    for s in range(nS):
        for a in range(env.action_space.n):
            Q[s, a] = sum(
                p * (r + 0.9 * (0.0 if d else V[s2])) for p, s2, r, d in P[s][a]
            )
    expected = Q.max(axis=1) - Q.min(axis=1)
    assert np.allclose(mu, expected, atol=1e-4)


def test_exact_mu_defaults_to_pipeline_deterministic_frozenlake():
    """exact_mu_per_state with no env_kwargs must use is_slippery=False (pipeline)."""
    env_det = gym.make("FrozenLake-v1", is_slippery=False)
    nS = env_det.observation_space.n
    actions = np.full(nS, 2, dtype=int)
    expert = _GreedyDownRightExpert(actions)

    # Call with NO env_kwargs — should auto-default to pipeline's is_slippery=False.
    mu_default = rt.exact_mu_per_state("FrozenLake-v1", expert, gamma=0.99)

    # Independent recomputation on the deterministic model with gamma=0.99.
    P = env_det.unwrapped.P
    probs = rt.expert_action_probs(expert, nS)
    V = np.zeros(nS)
    for _ in range(5000):
        Vn = np.zeros(nS)
        for s in range(nS):
            for a in range(env_det.action_space.n):
                q = sum(
                    p * (r + 0.99 * (0.0 if d else V[s2])) for p, s2, r, d in P[s][a]
                )
                Vn[s] += probs[s, a] * q
        if np.max(np.abs(Vn - V)) < 1e-9:
            V = Vn
            break
        V = Vn
    Q = np.zeros((nS, env_det.action_space.n))
    for s in range(nS):
        for a in range(env_det.action_space.n):
            Q[s, a] = sum(
                p * (r + 0.99 * (0.0 if d else V[s2])) for p, s2, r, d in P[s][a]
            )
    env_det.close()
    expected = Q.max(axis=1) - Q.min(axis=1)
    assert np.allclose(mu_default, expected, atol=1e-4)
