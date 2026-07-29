"""Exact recoverability mu(s) for tabular (env.P) environments.

mu(s) = max_a Q^{pi^E}(s,a) - min_a Q^{pi^E}(s,a), computed exactly by policy
evaluation on the environment's transition model, w.r.t. the PPO expert policy.
"""

import gymnasium as gym
import numpy as np

from imitation.experiments.ftrl import env_utils


def state_ids_from_onehot(obs: np.ndarray) -> np.ndarray:
    """Map one-hot observations back to integer state ids."""
    return np.asarray(obs).reshape(len(obs), -1).argmax(axis=1)


def expert_action_probs(expert_policy, n_states: int) -> np.ndarray:
    """Query pi^E(.|s) for every state via one-hot observations -> [nS, nA]."""
    onehot = np.eye(n_states, dtype=np.float32)
    obs_t, _ = expert_policy.obs_to_tensor(onehot)
    dist = expert_policy.get_distribution(obs_t)
    return np.asarray(dist.distribution.probs.detach().cpu().numpy())


def exact_mu_per_state(
    env_name: str, expert_policy, gamma: float = 0.99, **env_kwargs
) -> np.ndarray:
    """Per-state mu(s) via exact policy evaluation on env.P (indexed by state id).

    Args:
        env_name: Gymnasium environment id (e.g. ``"FrozenLake-v1"``).
        expert_policy: Expert policy with ``obs_to_tensor`` and ``get_distribution``.
        gamma: Discount factor for policy evaluation.
        **env_kwargs: Additional keyword arguments forwarded to ``gym.make``.
            If not provided, defaults to the pipeline's ``ENV_CONFIGS`` entry for
            ``env_name`` so that mu is computed on the same transition model used
            for demos/expert training (e.g. ``is_slippery=False`` for FrozenLake-v1).
            An explicit caller-supplied ``env_kwargs`` always takes precedence.

    Returns:
        Array of shape ``[nS]`` with mu(s) = max_a Q(s,a) - min_a Q(s,a).
    """
    if not env_kwargs:
        env_kwargs = dict(env_utils.ENV_CONFIGS.get(env_name, {}).get("env_kwargs", {}))
    env = gym.make(env_name, **env_kwargs)
    model = env.unwrapped.P
    n_s = env.observation_space.n
    n_a = env.action_space.n
    env.close()
    probs = expert_action_probs(expert_policy, n_s)

    def q_of(state, action, value):
        return sum(
            p * (r + gamma * (0.0 if d else value[s2]))
            for p, s2, r, d in model[state][action]
        )

    value = np.zeros(n_s)
    for _ in range(5000):
        nxt = np.zeros(n_s)
        for s in range(n_s):
            nxt[s] = sum(probs[s, a] * q_of(s, a, value) for a in range(n_a))
        if np.max(np.abs(nxt - value)) < 1e-9:
            value = nxt
            break
        value = nxt

    q = np.zeros((n_s, n_a))
    for s in range(n_s):
        for a in range(n_a):
            q[s, a] = q_of(s, a, value)
    return q.max(axis=1) - q.min(axis=1)
