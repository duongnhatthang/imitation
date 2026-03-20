# Coding Conventions

**Analysis Date:** 2026-03-19

## Naming Patterns

**Files:**
- Module files use `snake_case`: `bc.py`, `reward_nets.py`, `rollout.py`
- Package directories use `snake_case`: `algorithms/`, `rewards/`, `policies/`, `util/`
- Test files follow pattern `test_<module_name>.py`: `test_bc.py`, `test_reward_nets.py`

**Functions:**
- Use `snake_case` for all function names: `make_vec_env()`, `save_policy()`, `oric()`
- Private/internal functions prefixed with single underscore: `_on_epoch_end()`, `_make_and_iterate_loader()`
- Descriptive names indicating action/purpose: `unwrap_traj()`, `check_fixed_horizon()`

**Variables:**
- Local variables use `snake_case`: `batch_size`, `expert_data_type`, `observation_space`
- Private attributes use single underscore: `self._demo_data_loader`, `self._bc_logger`, `self._horizon`
- Constants use `UPPER_SNAKE_CASE`: `CARTPOLE_ENV_NAME`, `ISO_TIMESTAMP`, `PENDULUM_ENV_NAME`

**Types:**
- Classes use `PascalCase`: `BatchIteratorWithEpochEndCallback`, `BehaviorCloningLossCalculator`, `RewardNet`
- Type hints use full typing module imports: `Dict`, `Optional`, `Union`, `Callable`, `Sequence`
- Imported as: `from typing import Dict, Optional, Union, Callable, Iterable, Sequence, Tuple, Type`

**Aliases:**
- `torch` imported as `th`: `import torch as th`
- `numpy` imported as `np`: `import numpy as np`
- Internal modules with descriptive imports: `from imitation.algorithms import base as algo_base`
- Logger imported as: `from imitation.util import logger as imit_logger`

## Code Style

**Formatting:**
- Black code formatter (version ~23.9.1)
- Target Python version: 3.8+
- Line length: 88 characters (Black standard)
- Tool configuration in: `.pre-commit-config.yaml`

**Linting:**
- Flake8 (version 6.1.0) with plugins:
  - `flake8-blind-except`: catch bare except clauses
  - `flake8-builtins`: check for shadowing builtins
  - `flake8-commas`: enforce trailing commas
  - `flake8-debugger`: detect debugger imports
  - `flake8-docstrings`: enforce docstring presence
  - `flake8-isort`: enforce import sorting
  - `darglint`: docstring parameter checking
- Configuration in: `setup.cfg`
- Max line length: 88 characters
- Ignored rules: `E203` (whitespace before colon), `D102` (missing docstring in public method), `D103` (missing docstring in public function), `D105` (missing docstring in magic method)

**Type Checking:**
- MyPy (version ~0.990) enabled in pre-commit hooks
- PyType (version 2023.9.27, non-Windows only)
- Configuration in: `mypy.ini` (ignore missing imports)
- Strict typing encouraged but not enforced everywhere

## Import Organization

**Order:**
1. Standard library imports: `import os`, `import dataclasses`, `from typing import ...`
2. Third-party imports: `import numpy as np`, `import torch as th`, `from gymnasium import spaces`, `from stable_baselines3.common import ...`
3. First-party imports: `from imitation.data import types`, `from imitation.util import util`, `from imitation.algorithms import base`

**Example from `src/imitation/algorithms/bc.py`:**
```python
import dataclasses
import itertools
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

import gymnasium as gym
import numpy as np
import torch as th
import tqdm
from stable_baselines3.common import policies, torch_layers, utils, vec_env

from imitation.algorithms import base as algo_base
from imitation.data import rollout, types
from imitation.policies import base as policy_base
from imitation.util import logger as imit_logger
from imitation.util import util
```

**Path Aliases:**
- `known_first_party=imitation` in isort config
- Multi-line imports use parentheses with trailing commas
- One import per line or grouped logically with parentheses

## Error Handling

**Patterns:**
- Explicit exception types (never bare `except:`)
- Meaningful error messages with context
- Raise `ValueError` for invalid arguments or state
- Raise `TypeError` for type mismatches
- Raise `KeyError` for missing dictionary keys
- Raise `AssertionError` for internal consistency checks

**Examples from codebase:**
```python
# ValueError for parameter validation
if self.batch_size % self.minibatch_size != 0:
    raise ValueError("Batch size must be a multiple of minibatch size.")

# ValueError for missing required data
if traj.infos is None:
    raise ValueError("Trajectory must have infos to unwrap")

# TypeError for type errors
raise TypeError("Expected policy to be BaseAlgorithm or BasePolicy")

# Assertions for internal consistency
assert len(res.obs) == len(res.acts) + 1
assert self._idx == 0

# Try-except with specific exception handling
try:
    # operation
except ValueError as e:
    # handle
```

## Logging

**Framework:** HierarchicalLogger from `imitation.util.logger`

**Patterns:**
- Inject logger as constructor parameter: `custom_logger: Optional[imit_logger.HierarchicalLogger] = None`
- Create logger instance if not provided: `self.logger = custom_logger or imit_logger.configure(tmpdir)`
- Use hierarchical logging: `self._logger.record("bc/epoch", epoch_number)`
- Dump logs with tensorboard step: `self._logger.dump(self._tensorboard_step)`
- Record metrics by calling `.record(key, value)` multiple times, then `.dump(step)`

**Example from `src/imitation/algorithms/bc.py` BCLogger class:**
```python
def log_batch(self, batch_num, num_samples_so_far, metrics_dict):
    self._logger.record("batch_size", self.batch_size)
    self._logger.record("bc/epoch", self._current_epoch)
    self._logger.record("bc/batch", batch_num)
    self._logger.record("bc/samples_so_far", num_samples_so_far)
    for k, v in metrics_dict.items():
        self._logger.record(f"bc/{k}", float(v) if v is not None else None)
    self._logger.dump(self._tensorboard_step)
```

## Comments

**When to Comment:**
- Docstrings required for all public functions, classes, and methods
- Docstrings required for module-level functions and classes
- Docstring convention: Google style (configured in flake8)
- Inline comments for non-obvious logic or algorithmic complexity
- Comments explaining WHY, not WHAT (code should be self-documenting for WHAT)

**JSDoc/TSDoc:**
- Use Google-style docstrings (not NumPy style)
- Required sections: `Args`, `Returns`, `Raises` (if applicable)
- Type annotations in docstrings using reStructuredText format

**Example from codebase:**
```python
def oric(x: np.ndarray) -> np.ndarray:
    """Optimal rounding under integer constraints.

    Given a vector of real numbers such that the sum is an integer, returns a vector
    of rounded integers that preserves the sum and which minimizes the Lp-norm of the
    difference between the rounded and original vectors for all p >= 1. Algorithm from
    https://arxiv.org/abs/1501.00014. Runs in O(n log n) time.

    Args:
        x: A 1D vector of real numbers that sum to an integer.

    Returns:
        A 1D vector of rounded integers, preserving the sum.
    """
```

## Function Design

**Size:** Prefer smaller functions (under 50 lines); larger functions use docstrings to explain sections

**Parameters:**
- Use keyword-only arguments with `*` separator for clarity: `def __init__(self, *, observation_space, ...)`
- Type hints for all parameters
- Default values for optional parameters
- Order: required params first, then optional params

**Return Values:**
- Type hints required
- Return tuples explicitly typed: `Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]`
- Document return values in docstring under `Returns` section
- Can return `None` (explicitly typed in annotations)

**Example from `src/imitation/rewards/reward_nets.py`:**
```python
def preprocess(
    self,
    state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
    done: np.ndarray,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Preprocess a batch of input transitions and convert it to PyTorch tensors.

    Args:
        state: The observation input. Its shape is
            `(batch_size,) + observation_space.shape`.
        action: The action input. Its shape is
            `(batch_size,) + action_space.shape`.
        next_state: The observation input. Its shape is
            `(batch_size,) + observation_space.shape`.
        done: Whether the episode has terminated. Its shape is `(batch_size,)`.

    Returns:
        Preprocessed transitions: a Tuple of tensors containing
        observations, actions, next observations and dones.
    """
```

## Module Design

**Exports:**
- Use explicit exports via `__all__` when needed
- Public classes and functions at module level
- Private functions prefixed with underscore
- Re-exports from submodules in package `__init__.py`

**Barrel Files:**
- Package `__init__.py` files import key classes/functions for convenience
- Example: `src/imitation/algorithms/__init__.py` may export `BC`, `AIRL`, etc.
- Allows `from imitation.algorithms import BC` instead of `from imitation.algorithms.bc import BC`

**Dataclasses:**
- Heavily used for type-safe data containers
- Example: `BCTrainingMetrics`, `BatchIteratorWithEpochEndCallback`
- Use frozen dataclasses where immutability is desired: `@dataclasses.dataclass(frozen=True)`

---

*Convention analysis: 2026-03-19*
