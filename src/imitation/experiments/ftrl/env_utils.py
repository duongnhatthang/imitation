"""Environment utilities for FTRL experiments.

Provides a OneHotObsWrapper for discrete-state environments and a unified
make_env factory that auto-applies it where needed.
"""

from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import util


# Environment configurations for the 5 classical MDPs.
# obs_type: "discrete" means Discrete obs space that needs one-hot encoding.
# obs_size: size of the Discrete space (only for discrete obs_type).
# ppo_timesteps: PPO training steps to solve the env (benchmarked locally with
#   net_arch=[64,64], n_steps=256, batch_size=64, 4 parallel envs).
#   FrozenLake: 5k steps (~1s), CartPole: 20k (~7s).
#   CliffWalking/Acrobot/MountainCar: may need 200k+ (tune on server).
ENV_CONFIGS: Dict[str, dict] = {
    "CartPole-v1": {
        "obs_type": "continuous",
        "ppo_timesteps": 25_000,
        "env_kwargs": {},
    },
    "FrozenLake-v1": {
        "obs_type": "discrete",
        "obs_size": 16,
        "ppo_timesteps": 10_000,
        "env_kwargs": {"is_slippery": False},
    },
    "CliffWalking-v0": {
        "obs_type": "discrete",
        "obs_size": 48,
        "ppo_timesteps": 200_000,
        "env_kwargs": {},
    },
    "Acrobot-v1": {
        "obs_type": "continuous",
        "ppo_timesteps": 200_000,
        "env_kwargs": {},
    },
    "MountainCar-v0": {
        "obs_type": "continuous",
        "ppo_timesteps": 1_000_000,
        "env_kwargs": {},
    },
}


class OneHotObsWrapper(gym.ObservationWrapper):
    """Converts a Discrete observation space to a one-hot Box space.

    Wraps environments like FrozenLake and CliffWalking that return integer
    observations, converting them to one-hot float32 vectors suitable for
    neural network policies.
    """

    def __init__(self, env: gym.Env) -> None:
        """Builds OneHotObsWrapper.

        Args:
            env: Environment with a Discrete observation space.

        Raises:
            TypeError: If the observation space is not Discrete.
        """
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise TypeError(
                f"OneHotObsWrapper requires Discrete obs space, "
                f"got {type(env.observation_space).__name__}",
            )
        self._n = int(env.observation_space.n)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._n,),
            dtype=np.float32,
        )

    def observation(self, obs) -> np.ndarray:
        """Convert integer observation to one-hot vector.

        Args:
            obs: Integer observation from the wrapped environment.

        Returns:
            One-hot encoded float32 array of shape (n,).
        """
        one_hot = np.zeros(self._n, dtype=np.float32)
        one_hot[int(obs)] = 1.0
        return one_hot


def make_env(
    env_name: str,
    n_envs: int,
    rng: np.random.Generator,
    env_kwargs: Optional[dict] = None,
) -> VecEnv:
    """Create a vectorized environment, auto-applying one-hot for discrete obs.

    Args:
        env_name: Gymnasium environment ID.
        n_envs: Number of parallel environments.
        rng: Random state for seeding.
        env_kwargs: Override env_kwargs from ENV_CONFIGS. If None, uses
            the defaults from ENV_CONFIGS (or empty dict if env not in configs).

    Returns:
        A VecEnv with RolloutInfoWrapper applied. Discrete-obs envs also
        get OneHotObsWrapper.
    """
    config = ENV_CONFIGS.get(env_name, {})
    is_discrete = config.get("obs_type") == "discrete"

    if env_kwargs is None:
        env_kwargs = config.get("env_kwargs", {})

    def post_wrapper(env, _):
        if is_discrete:
            env = OneHotObsWrapper(env)
        return RolloutInfoWrapper(env)

    return util.make_vec_env(
        env_name,
        n_envs=n_envs,
        post_wrappers=[post_wrapper],
        rng=rng,
        env_make_kwargs=env_kwargs,
    )
