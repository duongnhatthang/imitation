# Atari + Expanded MDPs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand FTL vs FTRL vs BC experiments from 5 classical MDPs to 8 classical + 16 Atari games, using pre-trained HuggingFace models where available and self-trained PPO experts elsewhere.

**Architecture:** Modular approach — new `atari_utils.py` handles Atari env creation and HuggingFace model downloads. Existing files (`env_utils.py`, `experts.py`, `policy_utils.py`, `run_experiment.py`) are extended with Atari routing. `create_linear_policy` is generalized to clone-and-reset (deepcopy expert, freeze all except `action_net`, reinitialize `action_net`) so it works with any architecture (MLP or CNN).

**Tech Stack:** stable-baselines3, gymnasium[atari], ale-py, huggingface_sb3, torch

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/imitation/experiments/ftrl/atari_utils.py` | **Create** | Atari env configs (3 tiers), `make_atari_venv()`, `download_hub_expert()`, `get_atari_env_id()` |
| `src/imitation/experiments/ftrl/env_utils.py` | **Modify** | Add 3 new classical MDPs, `ENV_GROUPS`, `is_atari()`, `FlattenTupleObsWrapper` |
| `src/imitation/experiments/ftrl/policy_utils.py` | **Modify** | Generalize `create_linear_policy` to clone-and-reset |
| `src/imitation/experiments/ftrl/experts.py` | **Modify** | Add HuggingFace download branch, Atari self-training branch |
| `src/imitation/experiments/ftrl/run_experiment.py` | **Modify** | Add `--env-group`, Atari env routing, GPU support for Atari |
| `tests/experiments/test_env_utils.py` | **Create** | Tests for new wrappers, env groups, `is_atari()` |
| `tests/experiments/test_atari_utils.py` | **Create** | Tests for Atari env creation and config |
| `tests/experiments/test_policy_utils.py` | **Create** | Tests for generalized clone-and-reset `create_linear_policy` |

---

### Task 1: Add 3 new classical MDPs to `env_utils.py`

**Files:**
- Modify: `src/imitation/experiments/ftrl/env_utils.py`
- Create: `tests/experiments/test_env_utils.py`

- [ ] **Step 1: Write failing tests for Taxi-v3, Blackjack-v1, LunarLander-v3**

```python
# tests/experiments/test_env_utils.py
"""Tests for env_utils: new MDPs, wrappers, and env groups."""

import gymnasium as gym
import numpy as np
import pytest

from imitation.experiments.ftrl.env_utils import (
    ENV_CONFIGS,
    FlattenTupleObsWrapper,
    OneHotObsWrapper,
    make_env,
)


class TestNewClassicalMDPs:
    """Test that the 3 new classical MDPs are configured and create properly."""

    def test_taxi_in_configs(self):
        assert "Taxi-v3" in ENV_CONFIGS
        assert ENV_CONFIGS["Taxi-v3"]["obs_type"] == "discrete"
        assert ENV_CONFIGS["Taxi-v3"]["obs_size"] == 500

    def test_blackjack_in_configs(self):
        assert "Blackjack-v1" in ENV_CONFIGS
        assert ENV_CONFIGS["Blackjack-v1"]["obs_type"] == "tuple"

    def test_lunarlander_in_configs(self):
        assert "LunarLander-v3" in ENV_CONFIGS
        assert ENV_CONFIGS["LunarLander-v3"]["obs_type"] == "continuous"

    def test_make_env_taxi(self):
        rng = np.random.default_rng(0)
        venv = make_env("Taxi-v3", n_envs=1, rng=rng)
        obs = venv.reset()
        assert obs.shape == (1, 500)  # one-hot encoded
        venv.close()

    def test_make_env_blackjack(self):
        rng = np.random.default_rng(0)
        venv = make_env("Blackjack-v1", n_envs=1, rng=rng)
        obs = venv.reset()
        assert obs.shape == (1, 45)  # flattened one-hot: 32 + 11 + 2
        venv.close()

    def test_make_env_lunarlander(self):
        rng = np.random.default_rng(0)
        venv = make_env("LunarLander-v3", n_envs=1, rng=rng)
        obs = venv.reset()
        assert obs.shape == (1, 8)
        venv.close()


class TestFlattenTupleObsWrapper:
    """Test the FlattenTupleObsWrapper for Blackjack-v1."""

    def test_wraps_tuple_obs(self):
        env = gym.make("Blackjack-v1")
        wrapped = FlattenTupleObsWrapper(env)
        assert wrapped.observation_space.shape == (45,)
        obs, _ = wrapped.reset()
        assert obs.shape == (45,)
        assert obs.dtype == np.float32
        # Exactly one 1.0 in each one-hot segment
        assert np.sum(obs[:32]) == 1.0
        assert np.sum(obs[32:43]) == 1.0
        assert np.sum(obs[43:45]) == 1.0
        wrapped.close()

    def test_rejects_non_tuple(self):
        env = gym.make("CartPole-v1")
        with pytest.raises(TypeError, match="Tuple"):
            FlattenTupleObsWrapper(env)
        env.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/experiments/test_env_utils.py -v`
Expected: FAIL — `FlattenTupleObsWrapper` not found, new envs not in `ENV_CONFIGS`

- [ ] **Step 3: Implement `FlattenTupleObsWrapper` and add new MDPs to `ENV_CONFIGS`**

Add to `src/imitation/experiments/ftrl/env_utils.py`:

```python
# Add to ENV_CONFIGS dict:
    "Taxi-v3": {
        "obs_type": "discrete",
        "obs_size": 500,
        "ppo_timesteps": 100_000,
        "env_kwargs": {},
    },
    "Blackjack-v1": {
        "obs_type": "tuple",
        "obs_sizes": [32, 11, 2],  # Tuple(Discrete(32), Discrete(11), Discrete(2))
        "ppo_timesteps": 50_000,
        "env_kwargs": {},
    },
    "LunarLander-v3": {
        "obs_type": "continuous",
        "ppo_timesteps": 300_000,
        "env_kwargs": {},
    },
```

Add `FlattenTupleObsWrapper` class:

```python
class FlattenTupleObsWrapper(gym.ObservationWrapper):
    """Converts a Tuple of Discrete observation spaces to a concatenated one-hot Box.

    For Blackjack-v1: Tuple(Discrete(32), Discrete(11), Discrete(2)) → Box(45,).
    Each discrete component is one-hot encoded and concatenated.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Tuple):
            raise TypeError(
                f"FlattenTupleObsWrapper requires Tuple obs space, "
                f"got {type(env.observation_space).__name__}",
            )
        self._sizes = []
        for space in env.observation_space.spaces:
            if not isinstance(space, gym.spaces.Discrete):
                raise TypeError(
                    f"FlattenTupleObsWrapper requires all Tuple elements to be "
                    f"Discrete, got {type(space).__name__}",
                )
            self._sizes.append(int(space.n))
        total = sum(self._sizes)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(total,), dtype=np.float32,
        )

    def observation(self, obs) -> np.ndarray:
        parts = []
        for i, size in enumerate(self._sizes):
            one_hot = np.zeros(size, dtype=np.float32)
            one_hot[int(obs[i])] = 1.0
            parts.append(one_hot)
        return np.concatenate(parts)
```

Update the `post_wrapper` in `make_env` to handle the `"tuple"` obs type:

```python
def post_wrapper(env, _):
    if is_discrete:
        env = OneHotObsWrapper(env)
    elif is_tuple:
        env = FlattenTupleObsWrapper(env)
    return RolloutInfoWrapper(env)
```

Specifically, replace the `make_env` function body to detect `"tuple"` obs_type:

```python
def make_env(
    env_name: str,
    n_envs: int,
    rng: np.random.Generator,
    env_kwargs: Optional[dict] = None,
) -> VecEnv:
    config = ENV_CONFIGS.get(env_name, {})
    obs_type = config.get("obs_type", "continuous")

    if env_kwargs is None:
        env_kwargs = config.get("env_kwargs", {})

    def post_wrapper(env, _):
        if obs_type == "discrete":
            env = OneHotObsWrapper(env)
        elif obs_type == "tuple":
            env = FlattenTupleObsWrapper(env)
        return RolloutInfoWrapper(env)

    return util.make_vec_env(
        env_name,
        n_envs=n_envs,
        post_wrappers=[post_wrapper],
        rng=rng,
        env_make_kwargs=env_kwargs,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/experiments/test_env_utils.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/imitation/experiments/ftrl/env_utils.py tests/experiments/test_env_utils.py
git commit -m "feat: add Taxi, Blackjack, LunarLander to classical MDPs"
```

---

### Task 2: Add `ENV_GROUPS` and `is_atari()` to `env_utils.py`

**Files:**
- Modify: `src/imitation/experiments/ftrl/env_utils.py`
- Modify: `tests/experiments/test_env_utils.py`

- [ ] **Step 1: Write failing tests for `ENV_GROUPS` and `is_atari()`**

Append to `tests/experiments/test_env_utils.py`:

```python
from imitation.experiments.ftrl.env_utils import ENV_GROUPS, is_atari


class TestEnvGroups:
    """Test environment group definitions."""

    def test_classical_group_has_8_envs(self):
        assert len(ENV_GROUPS["classical"]) == 8

    def test_atari_zoo_group(self):
        assert "PongNoFrameskip-v4" in ENV_GROUPS["atari-zoo"]
        assert len(ENV_GROUPS["atari-zoo"]) == 8

    def test_atari_all_is_union(self):
        expected = (
            set(ENV_GROUPS["atari-zoo"])
            | set(ENV_GROUPS["atari-fast"])
            | set(ENV_GROUPS["atari-medium"])
        )
        assert set(ENV_GROUPS["atari-all"]) == expected

    def test_all_is_classical_plus_atari(self):
        expected = set(ENV_GROUPS["classical"]) | set(ENV_GROUPS["atari-all"])
        assert set(ENV_GROUPS["all"]) == expected


class TestIsAtari:
    """Test Atari environment detection."""

    def test_atari_envs(self):
        assert is_atari("PongNoFrameskip-v4") is True
        assert is_atari("BreakoutNoFrameskip-v4") is True

    def test_classical_envs(self):
        assert is_atari("CartPole-v1") is False
        assert is_atari("Taxi-v3") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/experiments/test_env_utils.py::TestEnvGroups -v`
Expected: FAIL — `ENV_GROUPS` and `is_atari` not defined

- [ ] **Step 3: Implement `ENV_GROUPS` and `is_atari()`**

Add to `src/imitation/experiments/ftrl/env_utils.py` after `ENV_CONFIGS`:

```python
def is_atari(env_name: str) -> bool:
    """Check if an environment name refers to an Atari game."""
    return "NoFrameskip" in env_name


ENV_GROUPS: Dict[str, list] = {
    "classical": list(ENV_CONFIGS.keys()),
    "atari-zoo": [
        "PongNoFrameskip-v4",
        "BreakoutNoFrameskip-v4",
        "SpaceInvadersNoFrameskip-v4",
        "BeamRiderNoFrameskip-v4",
        "QbertNoFrameskip-v4",
        "MsPacmanNoFrameskip-v4",
        "EnduroNoFrameskip-v4",
        "SeaquestNoFrameskip-v4",
    ],
    "atari-fast": [
        "FreewayNoFrameskip-v4",
        "AtlantisNoFrameskip-v4",
        "DemonAttackNoFrameskip-v4",
        "CrazyClimberNoFrameskip-v4",
    ],
    "atari-medium": [
        "AsterixNoFrameskip-v4",
        "FrostbiteNoFrameskip-v4",
        "KangarooNoFrameskip-v4",
        "BankHeistNoFrameskip-v4",
    ],
}
ENV_GROUPS["atari-all"] = (
    ENV_GROUPS["atari-zoo"] + ENV_GROUPS["atari-fast"] + ENV_GROUPS["atari-medium"]
)
ENV_GROUPS["all"] = ENV_GROUPS["classical"] + ENV_GROUPS["atari-all"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/experiments/test_env_utils.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/imitation/experiments/ftrl/env_utils.py tests/experiments/test_env_utils.py
git commit -m "feat: add ENV_GROUPS and is_atari() for environment selection"
```

---

### Task 3: Create `atari_utils.py` — Atari env creation and HuggingFace download

**Files:**
- Create: `src/imitation/experiments/ftrl/atari_utils.py`
- Create: `tests/experiments/test_atari_utils.py`

- [ ] **Step 1: Write failing tests for Atari configs and `get_atari_env_id()`**

```python
# tests/experiments/test_atari_utils.py
"""Tests for atari_utils: config, env creation, hub downloads."""

import pytest

from imitation.experiments.ftrl.atari_utils import (
    ATARI_CONFIGS,
    get_atari_env_id,
)


class TestAtariConfigs:
    """Test Atari game configurations."""

    def test_all_tiers_present(self):
        tiers = {cfg["tier"] for cfg in ATARI_CONFIGS.values()}
        assert tiers == {1, 2, 3}

    def test_tier1_has_hub_repo(self):
        for env_id, cfg in ATARI_CONFIGS.items():
            if cfg["tier"] == 1:
                assert "hub_repo_id" in cfg, f"{env_id} missing hub_repo_id"

    def test_all_have_ppo_timesteps(self):
        for env_id, cfg in ATARI_CONFIGS.items():
            assert "ppo_timesteps" in cfg, f"{env_id} missing ppo_timesteps"


class TestGetAtariEnvId:
    """Test short name → full env ID mapping."""

    def test_pong(self):
        assert get_atari_env_id("Pong") == "PongNoFrameskip-v4"

    def test_already_full_id(self):
        assert get_atari_env_id("PongNoFrameskip-v4") == "PongNoFrameskip-v4"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_atari_env_id("NonexistentGame")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/experiments/test_atari_utils.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create `atari_utils.py` with configs and `get_atari_env_id()`**

```python
# src/imitation/experiments/ftrl/atari_utils.py
"""Atari environment utilities for FTRL experiments.

Provides Atari game configs (3 tiers), vectorized env creation with standard
Atari wrappers, and HuggingFace model zoo downloads.
"""

import logging
import pathlib
from typing import Dict, Optional

import numpy as np
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util import util

logger = logging.getLogger(__name__)

# Atari game configurations, organized by tier.
# Tier 1: HuggingFace model zoo (no training needed).
# Tier 2: Fast self-trained (~1-2M steps, ~30 min).
# Tier 3: Medium self-trained (~5-10M steps, ~2-5 hours).
ATARI_CONFIGS: Dict[str, dict] = {
    # Tier 1: HuggingFace model zoo
    "PongNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 1_000_000,
        "hub_repo_id": "sb3/ppo-PongNoFrameskip-v4",
        "hub_filename": "ppo-PongNoFrameskip-v4.zip",
    },
    "BreakoutNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 10_000_000,
        "hub_repo_id": "sb3/ppo-BreakoutNoFrameskip-v4",
        "hub_filename": "ppo-BreakoutNoFrameskip-v4.zip",
    },
    "SpaceInvadersNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 10_000_000,
        "hub_repo_id": "sb3/ppo-SpaceInvadersNoFrameskip-v4",
        "hub_filename": "ppo-SpaceInvadersNoFrameskip-v4.zip",
    },
    "BeamRiderNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 10_000_000,
        "hub_repo_id": "sb3/ppo-BeamRiderNoFrameskip-v4",
        "hub_filename": "ppo-BeamRiderNoFrameskip-v4.zip",
    },
    "QbertNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 10_000_000,
        "hub_repo_id": "sb3/ppo-QbertNoFrameskip-v4",
        "hub_filename": "ppo-QbertNoFrameskip-v4.zip",
    },
    "MsPacmanNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 10_000_000,
        "hub_repo_id": "sb3/ppo-MsPacmanNoFrameskip-v4",
        "hub_filename": "ppo-MsPacmanNoFrameskip-v4.zip",
    },
    "EnduroNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 10_000_000,
        "hub_repo_id": "sb3/ppo-EnduroNoFrameskip-v4",
        "hub_filename": "ppo-EnduroNoFrameskip-v4.zip",
    },
    "SeaquestNoFrameskip-v4": {
        "tier": 1,
        "ppo_timesteps": 10_000_000,
        "hub_repo_id": "sb3/ppo-SeaquestNoFrameskip-v4",
        "hub_filename": "ppo-SeaquestNoFrameskip-v4.zip",
    },
    # Tier 2: Fast self-trained
    "FreewayNoFrameskip-v4": {
        "tier": 2,
        "ppo_timesteps": 1_000_000,
    },
    "AtlantisNoFrameskip-v4": {
        "tier": 2,
        "ppo_timesteps": 2_000_000,
    },
    "DemonAttackNoFrameskip-v4": {
        "tier": 2,
        "ppo_timesteps": 2_000_000,
    },
    "CrazyClimberNoFrameskip-v4": {
        "tier": 2,
        "ppo_timesteps": 2_000_000,
    },
    # Tier 3: Medium self-trained
    "AsterixNoFrameskip-v4": {
        "tier": 3,
        "ppo_timesteps": 10_000_000,
    },
    "FrostbiteNoFrameskip-v4": {
        "tier": 3,
        "ppo_timesteps": 10_000_000,
    },
    "KangarooNoFrameskip-v4": {
        "tier": 3,
        "ppo_timesteps": 10_000_000,
    },
    "BankHeistNoFrameskip-v4": {
        "tier": 3,
        "ppo_timesteps": 10_000_000,
    },
}

# Short name → full env ID mapping
_SHORT_NAMES: Dict[str, str] = {}
for _env_id in ATARI_CONFIGS:
    _short = _env_id.replace("NoFrameskip-v4", "")
    _SHORT_NAMES[_short] = _env_id


def get_atari_env_id(name: str) -> str:
    """Convert a short game name to a full Atari env ID.

    Args:
        name: Short name (e.g. "Pong") or full ID ("PongNoFrameskip-v4").

    Returns:
        Full environment ID string.

    Raises:
        ValueError: If the name is not recognized.
    """
    if name in ATARI_CONFIGS:
        return name
    if name in _SHORT_NAMES:
        return _SHORT_NAMES[name]
    raise ValueError(
        f"Unknown Atari game: {name}. "
        f"Known short names: {sorted(_SHORT_NAMES.keys())}"
    )


def make_atari_venv(
    env_name: str,
    n_envs: int,
    seed: int = 0,
) -> VecEnv:
    """Create a vectorized Atari environment with standard wrappers.

    Applies: AtariWrapper (frame skip, grayscale, resize 84x84) →
    VecFrameStack(4) → RolloutInfoWrapper.

    Args:
        env_name: Full Atari env ID (e.g. "PongNoFrameskip-v4").
        n_envs: Number of parallel environments.
        seed: Random seed.

    Returns:
        A VecEnv ready for PPO training or FTRL experiments.
    """
    from stable_baselines3.common.env_util import make_atari_env

    venv = make_atari_env(env_name, n_envs=n_envs, seed=seed)
    venv = VecFrameStack(venv, n_stack=4)
    return venv


def download_hub_expert(
    env_name: str,
    cache_dir: pathlib.Path,
) -> pathlib.Path:
    """Download a pre-trained expert from HuggingFace model hub.

    Args:
        env_name: Full Atari env ID (e.g. "PongNoFrameskip-v4").
        cache_dir: Directory to cache downloaded models.

    Returns:
        Path to the downloaded .zip model file.

    Raises:
        ValueError: If the env has no hub_repo_id configured.
    """
    config = ATARI_CONFIGS.get(env_name)
    if config is None or "hub_repo_id" not in config:
        raise ValueError(
            f"No HuggingFace model available for {env_name}. "
            f"Tier 1 games only: "
            f"{[k for k, v in ATARI_CONFIGS.items() if v.get('hub_repo_id')]}"
        )

    cache_path = cache_dir / env_name.replace("/", "_")
    model_file = cache_path / "model.zip"

    if model_file.exists():
        logger.info(f"Using cached hub expert: {model_file}")
        return model_file

    from huggingface_sb3 import load_from_hub

    logger.info(
        f"Downloading expert from HuggingFace: {config['hub_repo_id']}"
    )
    downloaded_path = load_from_hub(
        repo_id=config["hub_repo_id"],
        filename=config["hub_filename"],
    )

    # Copy to our cache structure
    cache_path.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(downloaded_path, model_file)
    logger.info(f"Cached hub expert to {model_file}")

    return model_file
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/experiments/test_atari_utils.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/imitation/experiments/ftrl/atari_utils.py tests/experiments/test_atari_utils.py
git commit -m "feat: add atari_utils with game configs, env creation, and hub downloads"
```

---

### Task 4: Generalize `create_linear_policy` to clone-and-reset

**Files:**
- Modify: `src/imitation/experiments/ftrl/policy_utils.py`
- Create: `tests/experiments/test_policy_utils.py`

- [ ] **Step 1: Write failing tests for the generalized clone-and-reset approach**

```python
# tests/experiments/test_policy_utils.py
"""Tests for policy_utils: clone-and-reset linear policy creation."""

import copy

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from imitation.experiments.ftrl.policy_utils import (
    create_linear_policy,
    freeze_feature_layers,
    reinitialize_action_net,
)


class TestCreateLinearPolicy:
    """Test that create_linear_policy uses clone-and-reset."""

    def test_mlp_policy_action_net_is_fresh(self):
        """Linear policy should have reinitialized action_net."""
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64]))
        expert_policy = model.policy

        # Save expert action_net weights for comparison
        expert_action_weights = {
            name: param.clone()
            for name, param in expert_policy.named_parameters()
            if name.startswith("action_net")
        }

        linear_policy = create_linear_policy(expert_policy)

        # action_net should be different (reinitialized)
        for name, param in linear_policy.named_parameters():
            if name.startswith("action_net"):
                assert not th.equal(param, expert_action_weights[name]), (
                    f"action_net param {name} was not reinitialized"
                )
        env.close()

    def test_mlp_policy_features_frozen(self):
        """All non-action_net params should be frozen."""
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64]))
        linear_policy = create_linear_policy(model.policy)

        for name, param in linear_policy.named_parameters():
            if name.startswith("action_net"):
                assert param.requires_grad is True, f"{name} should be trainable"
            else:
                assert param.requires_grad is False, f"{name} should be frozen"
        env.close()

    def test_mlp_policy_features_match_expert(self):
        """Frozen features should be identical to expert's."""
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64]))
        expert_policy = model.policy

        expert_state = {
            name: param.clone()
            for name, param in expert_policy.named_parameters()
            if not name.startswith("action_net")
        }

        linear_policy = create_linear_policy(expert_policy)

        for name, param in linear_policy.named_parameters():
            if not name.startswith("action_net"):
                assert th.equal(param, expert_state[name]), (
                    f"Feature param {name} doesn't match expert"
                )
        env.close()

    def test_does_not_modify_expert(self):
        """Creating linear policy should not modify the original expert."""
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64]))
        expert_policy = model.policy

        original_state = copy.deepcopy(expert_policy.state_dict())
        create_linear_policy(expert_policy)

        for name, param in expert_policy.named_parameters():
            assert th.equal(param, original_state[name]), (
                f"Expert param {name} was modified"
            )
        env.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/experiments/test_policy_utils.py -v`
Expected: FAIL — `create_linear_policy` signature changed (no longer takes obs_space, act_space)

- [ ] **Step 3: Implement clone-and-reset `create_linear_policy`**

Replace the `create_linear_policy` function in `src/imitation/experiments/ftrl/policy_utils.py`:

```python
def create_linear_policy(
    expert_policy: ActorCriticPolicy,
) -> ActorCriticPolicy:
    """Create a linear-mode policy by cloning expert and resetting action_net.

    Deep-copies the expert's entire policy, freezes all parameters except
    action_net, and reinitializes action_net. Works with any architecture
    (MLP, CNN, etc.) without needing to know internals.

    Args:
        expert_policy: Trained expert policy to clone features from.

    Returns:
        A policy with frozen hidden layers and fresh action_net.
    """
    import copy

    policy = copy.deepcopy(expert_policy)
    freeze_feature_layers(policy)
    reinitialize_action_net(policy)
    return policy
```

Remove the `obs_space` and `act_space` parameters. The function no longer needs them since it clones the expert directly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/experiments/test_policy_utils.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Update callers of `create_linear_policy` in `run_experiment.py`**

In `src/imitation/experiments/ftrl/run_experiment.py`, update the two call sites.

In `_run_dagger_variant` (around line 154), change:
```python
        policy = policy_utils.create_linear_policy(
            expert_policy,
            venv.observation_space,
            venv.action_space,
        )
```
to:
```python
        policy = policy_utils.create_linear_policy(expert_policy)
```

In `_run_bc` (around line 281), change:
```python
        policy = policy_utils.create_linear_policy(
            expert_policy,
            venv.observation_space,
            venv.action_space,
        )
```
to:
```python
        policy = policy_utils.create_linear_policy(expert_policy)
```

- [ ] **Step 6: Run existing tests to verify nothing broke**

Run: `pytest tests/experiments/test_run_experiment.py -v`
Expected: All existing tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/imitation/experiments/ftrl/policy_utils.py src/imitation/experiments/ftrl/run_experiment.py tests/experiments/test_policy_utils.py
git commit -m "refactor: generalize create_linear_policy to clone-and-reset"
```

---

### Task 5: Extend `experts.py` with Atari expert sourcing

**Files:**
- Modify: `src/imitation/experiments/ftrl/experts.py`

- [ ] **Step 1: Write failing test for Atari expert loading**

Append to `tests/experiments/test_atari_utils.py`:

```python
from unittest.mock import MagicMock, patch

from imitation.experiments.ftrl.experts import get_or_train_expert


class TestAtariExpertRouting:
    """Test that get_or_train_expert routes correctly for Atari games."""

    @patch("imitation.experiments.ftrl.atari_utils.download_hub_expert")
    def test_tier1_downloads_from_hub(self, mock_download, tmp_path):
        """Tier 1 Atari game should attempt HuggingFace download."""
        # Mock the download to return a fake path
        mock_download.side_effect = ValueError("mock: no real download in test")

        from imitation.experiments.ftrl.atari_utils import make_atari_venv

        # We just test that the routing logic calls download_hub_expert
        # for tier 1 games. The actual download is mocked.
        with pytest.raises(ValueError, match="mock"):
            rng = np.random.default_rng(0)
            venv = make_atari_venv("PongNoFrameskip-v4", n_envs=1, seed=0)
            get_or_train_expert(
                "PongNoFrameskip-v4", venv, cache_dir=tmp_path, rng=rng,
            )
            venv.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/experiments/test_atari_utils.py::TestAtariExpertRouting -v`
Expected: FAIL — `get_or_train_expert` doesn't know about Atari yet

- [ ] **Step 3: Extend `experts.py` with Atari routing**

Replace the `get_or_train_expert` function in `src/imitation/experiments/ftrl/experts.py`:

```python
def get_or_train_expert(
    env_name: str,
    venv: VecEnv,
    cache_dir: pathlib.Path,
    rng: np.random.Generator,
    seed: int = 0,
) -> BasePolicy:
    """Load a cached expert, download from HuggingFace, or train with PPO.

    Routing logic:
    - If cached model exists on disk → load it.
    - If Atari tier 1 (hub_repo_id available) → download from HuggingFace.
    - If Atari tier 2/3 → train PPO with CnnPolicy.
    - If classical MDP → train PPO with MlpPolicy [64,64].

    Args:
        env_name: Gymnasium environment ID.
        venv: Vectorized environment.
        cache_dir: Directory for caching trained/downloaded expert models.
        rng: Random state.
        seed: Random seed for PPO training.

    Returns:
        A trained expert policy compatible with venv.
    """
    from . import atari_utils
    from .env_utils import is_atari

    cache_path = pathlib.Path(cache_dir) / env_name.replace("/", "_")
    model_file = cache_path / "model.zip"

    # Load cached model if available
    if model_file.exists():
        logger.info(f"Loading cached expert from {model_file}")
        model = PPO.load(model_file, env=venv)
        return model.policy

    if is_atari(env_name):
        return _get_atari_expert(env_name, venv, cache_dir, seed)
    else:
        return _train_classical_expert(env_name, venv, cache_dir, rng, seed)


def _get_atari_expert(
    env_name: str,
    venv: VecEnv,
    cache_dir: pathlib.Path,
    seed: int,
) -> BasePolicy:
    """Get an Atari expert: download from hub or train."""
    from . import atari_utils

    config = atari_utils.ATARI_CONFIGS.get(env_name, {})

    if "hub_repo_id" in config:
        # Tier 1: download from HuggingFace
        model_path = atari_utils.download_hub_expert(env_name, cache_dir)
        model = PPO.load(model_path, env=venv)
    else:
        # Tier 2/3: train PPO with CnnPolicy
        ppo_timesteps = config.get("ppo_timesteps", 10_000_000)
        logger.info(
            f"Training PPO CnnPolicy expert for {env_name} "
            f"({ppo_timesteps} timesteps)"
        )
        train_venv = atari_utils.make_atari_venv(env_name, n_envs=8, seed=seed)
        model = PPO(
            "CnnPolicy",
            train_venv,
            n_steps=128,
            n_epochs=4,
            batch_size=256,
            learning_rate=2.5e-4,
            clip_range=0.1,
            vf_coef=0.5,
            ent_coef=0.01,
            seed=seed,
            verbose=0,
        )
        model.learn(total_timesteps=ppo_timesteps)
        train_venv.close()

        # Cache
        cache_path = cache_dir / env_name.replace("/", "_")
        cache_path.mkdir(parents=True, exist_ok=True)
        model.save(cache_path / "model.zip")
        logger.info(f"Saved Atari expert to {cache_path / 'model.zip'}")

    # Evaluate
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(
        model.policy, venv, n_eval_episodes=10, deterministic=True,
    )
    logger.info(
        f"Expert quality for {env_name}: reward={mean_reward:.1f}±{std_reward:.1f}"
    )

    return model.policy


def _train_classical_expert(
    env_name: str,
    venv: VecEnv,
    cache_dir: pathlib.Path,
    rng: np.random.Generator,
    seed: int,
) -> BasePolicy:
    """Train a classical MDP expert with PPO MlpPolicy [64,64]."""
    config = env_utils.ENV_CONFIGS.get(env_name, {})
    ppo_timesteps = config.get("ppo_timesteps", 100_000)
    ppo_kwargs = config.get("ppo_kwargs", {})
    ppo_n_envs = config.get("ppo_n_envs", None)
    logger.info(
        f"Training PPO expert for {env_name} ({ppo_timesteps} timesteps)",
    )

    if ppo_n_envs and ppo_n_envs > 1:
        train_venv = env_utils.make_env(env_name, n_envs=ppo_n_envs, rng=rng)
    else:
        train_venv = venv

    model = PPO(
        "MlpPolicy",
        train_venv,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=0,
        **ppo_kwargs,
    )
    model.learn(total_timesteps=ppo_timesteps)

    if ppo_n_envs and ppo_n_envs > 1:
        train_venv.close()

    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(
        model.policy, venv, n_eval_episodes=20, deterministic=True,
    )
    logger.info(
        f"Expert quality for {env_name}: "
        f"reward={mean_reward:.1f}±{std_reward:.1f} "
        f"(trained {ppo_timesteps} steps)",
    )

    cache_path = cache_dir / env_name.replace("/", "_")
    cache_path.mkdir(parents=True, exist_ok=True)
    model.save(cache_path / "model.zip")
    logger.info(f"Saved expert to {cache_path / 'model.zip'}")

    return model.policy
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/experiments/test_atari_utils.py -v`
Expected: All tests PASS (the mock test verifies routing)

Run: `pytest tests/experiments/test_run_experiment.py -v`
Expected: Existing classical tests still PASS

- [ ] **Step 5: Commit**

```bash
git add src/imitation/experiments/ftrl/experts.py tests/experiments/test_atari_utils.py
git commit -m "feat: add Atari expert sourcing (HuggingFace + self-trained)"
```

---

### Task 6: Update `run_experiment.py` with `--env-group` and Atari routing

**Files:**
- Modify: `src/imitation/experiments/ftrl/run_experiment.py`

- [ ] **Step 1: Write failing test for `--env-group` resolution**

Append to `tests/experiments/test_run_experiment.py`:

```python
from imitation.experiments.ftrl.run_experiment import resolve_envs


class TestResolveEnvs:
    """Test --env-group resolution."""

    def test_classical_group(self):
        envs = resolve_envs(env_group="classical", envs=None)
        assert "CartPole-v1" in envs
        assert len(envs) == 8

    def test_explicit_envs_override_group(self):
        envs = resolve_envs(env_group=None, envs=["CartPole-v1"])
        assert envs == ["CartPole-v1"]

    def test_atari_zoo_group(self):
        envs = resolve_envs(env_group="atari-zoo", envs=None)
        assert "PongNoFrameskip-v4" in envs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/experiments/test_run_experiment.py::TestResolveEnvs -v`
Expected: FAIL — `resolve_envs` not found

- [ ] **Step 3: Implement `resolve_envs` and update `main()`**

Add `resolve_envs` function to `run_experiment.py`:

```python
def resolve_envs(
    env_group: Optional[str] = None,
    envs: Optional[List[str]] = None,
) -> List[str]:
    """Resolve environment list from --env-group or --envs.

    Args:
        env_group: Name of a group from ENV_GROUPS.
        envs: Explicit list of env names.

    Returns:
        List of environment IDs to run.

    Raises:
        ValueError: If neither or both are specified.
    """
    if env_group and envs:
        raise ValueError("Specify --env-group or --envs, not both")
    if env_group:
        if env_group not in env_utils.ENV_GROUPS:
            raise ValueError(
                f"Unknown env group: {env_group}. "
                f"Available: {list(env_utils.ENV_GROUPS.keys())}"
            )
        return env_utils.ENV_GROUPS[env_group]
    if envs:
        return envs
    return list(env_utils.ENV_CONFIGS.keys())  # default: classical only
```

Update `run_single` to route Atari vs classical env creation. Replace the env creation line (line ~91):

```python
    # Create env
    if env_utils.is_atari(config.env_name):
        from .atari_utils import make_atari_venv
        venv = make_atari_venv(config.env_name, n_envs=1, seed=config.seed)
    else:
        venv = env_utils.make_env(config.env_name, n_envs=1, rng=rng)
```

Remove `os.environ["CUDA_VISIBLE_DEVICES"] = ""` — it should only apply to classical MDPs. Instead:

```python
    # Force CPU for classical MDPs (Atari uses GPU for CNN)
    if not env_utils.is_atari(config.env_name):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

Update `main()` parser to add `--env-group` and update `--envs`:

```python
    parser.add_argument("--envs", nargs="+", default=None,
                        help="Specific environments to test")
    parser.add_argument("--env-group", type=str, default=None,
                        choices=list(env_utils.ENV_GROUPS.keys()),
                        help="Environment group to test (alternative to --envs)")
```

Update `main()` to use `resolve_envs`:

```python
    env_list = resolve_envs(env_group=args.env_group, envs=args.envs)
    args.envs = env_list
```

Update the pre-training loop in `main()` to handle Atari envs:

```python
    for env_name in args.envs:
        rng = np.random.default_rng(0)
        if env_utils.is_atari(env_name):
            from imitation.experiments.ftrl.atari_utils import make_atari_venv
            venv = make_atari_venv(env_name, n_envs=1, seed=0)
        else:
            venv = env_utils.make_env(env_name, n_envs=1, rng=rng)
        experts.get_or_train_expert(
            env_name, venv, cache_dir=expert_cache_dir, rng=rng, seed=0,
        )
        venv.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/experiments/test_run_experiment.py -v`
Expected: All tests PASS (including new `TestResolveEnvs`)

- [ ] **Step 5: Commit**

```bash
git add src/imitation/experiments/ftrl/run_experiment.py tests/experiments/test_run_experiment.py
git commit -m "feat: add --env-group CLI arg and Atari env routing"
```

---

### Task 7: End-to-end smoke test for a new classical MDP

**Files:**
- Modify: `tests/experiments/test_run_experiment.py`

- [ ] **Step 1: Write smoke test for LunarLander-v3**

Append to `tests/experiments/test_run_experiment.py`:

```python
def test_run_ftrl_lunarlander(tmp_path):
    """FTRL smoke test on LunarLander-v3 (new classical MDP)."""
    config = _make_config(
        "ftrl", tmp_path,
        env_name="LunarLander-v3",
        policy_mode="linear",
        n_rounds=2,
        samples_per_round=200,
    )
    result = run_single(config)

    assert result["algo"] == "ftrl"
    assert result["env"] == "LunarLander-v3"
    assert len(result["per_round"]) >= 1


def test_run_bc_taxi(tmp_path):
    """BC smoke test on Taxi-v3 (discrete obs, one-hot encoded)."""
    config = _make_config(
        "bc", tmp_path,
        env_name="Taxi-v3",
        policy_mode="linear",
        n_rounds=2,
        samples_per_round=200,
    )
    result = run_single(config)

    assert result["algo"] == "bc"
    assert result["env"] == "Taxi-v3"
    assert len(result["per_round"]) >= 1
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/experiments/test_run_experiment.py::test_run_ftrl_lunarlander tests/experiments/test_run_experiment.py::test_run_bc_taxi -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/experiments/test_run_experiment.py
git commit -m "test: add smoke tests for LunarLander and Taxi"
```

---

### Task 8: Verify Atari dependency installation

**Files:** None (shell commands only)

- [ ] **Step 1: Check if Atari dependencies are installed**

Run:
```bash
python -c "import ale_py; print('ale_py:', ale_py.__version__)"
python -c "import gymnasium; env = gymnasium.make('PongNoFrameskip-v4'); print('Pong obs space:', env.observation_space); env.close()"
python -c "from huggingface_sb3 import load_from_hub; print('huggingface_sb3 OK')"
```
Expected: All succeed. If `ale_py` or ROM import fails, install:
```bash
pip install "gymnasium[atari]" ale-py autorom
AutoROM --accept-license
```

- [ ] **Step 2: Verify SB3 Atari wrapper stack**

Run:
```bash
python -c "
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
venv = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
venv = VecFrameStack(venv, n_stack=4)
obs = venv.reset()
print('Obs shape:', obs.shape)  # Should be (1, 4, 84, 84)
venv.close()
print('Atari wrapper stack OK')
"
```
Expected: `Obs shape: (1, 4, 84, 84)`

- [ ] **Step 3: Commit dependency notes (if any installs needed)**

If new dependencies were installed, add them to `setup.cfg` or `pyproject.toml` under an `[atari]` extra:
```bash
git add setup.cfg  # or pyproject.toml
git commit -m "deps: add gymnasium[atari] and autorom to atari extras"
```

---

### Task 9: End-to-end integration test — download HuggingFace Pong expert

**Files:**
- Modify: `tests/experiments/test_atari_utils.py`

This test is marked `@pytest.mark.expensive` since it downloads a model and creates an Atari env.

- [ ] **Step 1: Write integration test**

Append to `tests/experiments/test_atari_utils.py`:

```python
import numpy as np


@pytest.mark.expensive
class TestAtariIntegration:
    """Integration tests that require Atari ROMs and network access."""

    def test_download_and_load_pong_expert(self, tmp_path):
        """Download Pong expert from HuggingFace and verify it loads."""
        from stable_baselines3 import PPO

        from imitation.experiments.ftrl.atari_utils import (
            download_hub_expert,
            make_atari_venv,
        )

        model_path = download_hub_expert("PongNoFrameskip-v4", tmp_path)
        assert model_path.exists()

        venv = make_atari_venv("PongNoFrameskip-v4", n_envs=1, seed=0)
        model = PPO.load(model_path, env=venv)

        # Verify the model can predict actions
        obs = venv.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1,)

        venv.close()

    def test_pong_linear_policy_creation(self, tmp_path):
        """Create a linear policy from Pong expert using clone-and-reset."""
        from stable_baselines3 import PPO

        from imitation.experiments.ftrl.atari_utils import (
            download_hub_expert,
            make_atari_venv,
        )
        from imitation.experiments.ftrl.policy_utils import create_linear_policy

        model_path = download_hub_expert("PongNoFrameskip-v4", tmp_path)
        venv = make_atari_venv("PongNoFrameskip-v4", n_envs=1, seed=0)
        model = PPO.load(model_path, env=venv)

        linear_policy = create_linear_policy(model.policy)

        # Verify action_net is trainable, rest frozen
        trainable = [
            name for name, p in linear_policy.named_parameters() if p.requires_grad
        ]
        frozen = [
            name for name, p in linear_policy.named_parameters() if not p.requires_grad
        ]
        assert all("action_net" in n for n in trainable)
        assert len(frozen) > 0
        assert not any("action_net" in n for n in frozen)

        # Verify it can predict actions
        obs = venv.reset()
        import torch as th
        with th.no_grad():
            action, _, _ = linear_policy(
                th.tensor(obs, dtype=th.float32)
            )
        assert action.shape[0] == 1

        venv.close()
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/experiments/test_atari_utils.py::TestAtariIntegration -v -m expensive`
Expected: PASS (requires network + Atari ROMs)

- [ ] **Step 3: Commit**

```bash
git add tests/experiments/test_atari_utils.py
git commit -m "test: add Atari integration tests (download + linear policy)"
```

---

### Task 10: Final verification and push

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite (non-expensive)**

Run: `pytest tests/experiments/ -v -m "not expensive"`
Expected: All tests PASS

- [ ] **Step 2: Run linting**

Run:
```bash
black src/imitation/experiments/ftrl/ tests/experiments/
isort src/imitation/experiments/ftrl/ tests/experiments/
flake8 src/imitation/experiments/ftrl/ tests/experiments/
```
Expected: Clean

- [ ] **Step 3: Fix any lint issues and commit**

```bash
git add -u
git commit -m "style: fix lint issues"
```

- [ ] **Step 4: Push branch**

```bash
git push -u origin feature/atari-expanded-mdps
```
