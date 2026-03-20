# Testing Patterns

**Analysis Date:** 2026-03-19

## Test Framework

**Runner:**
- pytest (version ~7.1.2)
- Config: `setup.cfg` under `[tool:pytest]` section
- Additional plugin: pytest-cov (~3.0.0) for coverage tracking
- Additional plugin: pytest-xdist (~2.5.0) for parallel test execution
- Additional plugin: pytest-timeout (~2.1.0) for test timeouts
- Additional plugin: pytest-notebook (==0.8.0) for notebook testing

**Assertion Library:**
- Built-in pytest assertions (no separate library)
- Uses `assert` statements with optional pytest comparisons

**Run Commands:**
```bash
pytest                              # Run all tests
pytest -x                           # Stop on first failure
pytest -v                           # Verbose output
pytest tests/algorithms/test_bc.py  # Run specific test file
pytest -n auto                      # Parallel execution with pytest-xdist
pytest --cov=imitation              # Run with coverage
pytest -m "not expensive"           # Skip expensive tests
```

## Test File Organization

**Location:**
- Tests are co-located alongside source code in a separate `tests/` directory
- Mirror directory structure: `tests/algorithms/`, `tests/data/`, `tests/util/`, `tests/rewards/`, etc.
- Mirrors `src/imitation/` structure

**Naming:**
- Test files: `test_<module_name>.py`
- Test functions: `test_<functionality_description>()`
- Test classes: Can use class grouping, but flat function structure is preferred
- Examples: `test_bc.py`, `test_reward_nets.py`, `test_base.py`

**Structure:**
```
tests/
├── conftest.py              # Global fixtures
├── algorithms/
│   ├── conftest.py          # Algorithm-specific fixtures
│   ├── test_bc.py
│   ├── test_dagger.py
│   └── test_adversarial.py
├── data/
│   ├── test_types.py
│   ├── test_buffer.py
│   └── test_rollout.py
├── util/
│   ├── test_util.py
│   └── test_networks.py
└── rewards/
    ├── test_reward_nets.py
    └── test_reward_fn.py
```

## Test Structure

**Suite Organization:**

Test files organize multiple test functions by feature. Complex test suites may use comments to section tests:

```python
########################
# HYPOTHESIS STRATEGIES
########################

# Define property-based test strategies

##############
# SMOKE TESTS
##############

@hypothesis.given(...)
def test_smoke_bc_creation(...):
    # Basic creation test

##############
# FEATURE TESTS
##############

def test_specific_feature():
    # Specific feature test
```

**Patterns:**

Given-When-Then structure with comments:
```python
def test_trainer_train_arguments(trainer, pendulum_expert_policy, rng):
    # GIVEN
    trainer = bc.BC(...)
    # WHEN
    trainer.train(...)
    # THEN
    assert trainer.policy is not None
```

Fixture-based test setup:
```python
def test_something(custom_logger, cartpole_venv, rng):
    # Fixtures injected as parameters
    # custom_logger from conftest.py
    # cartpole_venv from conftest.py
    # rng from conftest.py
```

## Fixtures and Setup

**Global Fixtures** in `tests/conftest.py`:
- `custom_logger`: Creates HierarchicalLogger with tmpdir
- `rng`: Creates numpy.random.Generator with seed=0
- `torch_single_threaded`: Session-level fixture to set PyTorch to single-threaded (for parallel test execution)
- `cartpole_venv`: Parameterized fixture with vecenv(1) and vecenv(4) variants

**Algorithm-Specific Fixtures** in `tests/algorithms/conftest.py`:
- `cartpole_expert_policy`: Loads pre-trained CartPole policy
- `cartpole_expert_trajectories`: Generates expert trajectories (60)
- `cartpole_bc_trainer`: Creates BC trainer with CartPole expert data
- `pendulum_expert_policy`: Loads pre-trained Pendulum policy
- `pendulum_expert_trajectories`: Generates expert trajectories (60)
- `pendulum_venv`: Creates 8-env vectorized environment
- `pendulum_single_venv`: Creates 1-env vectorized environment
- `multi_obs_venv`: Creates multi-observation environment for testing

**Fixture Pattern:**
```python
@pytest.fixture
def custom_logger(tmpdir: str) -> logger.HierarchicalLogger:
    return logger.configure(tmpdir)

@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=0)

@pytest.fixture(params=[1, 4], ids=lambda n: f"vecenv({n})")
def cartpole_venv(request, rng) -> VecEnv:
    num_envs = request.param
    return util.make_vec_env(...)
```

**Test Data Fixtures:**
```python
@pytest.fixture
def trajectory(obs_space, act_space, length) -> types.Trajectory:
    """Fixture to generate trajectory of length `length` iid sampled from spaces."""
    raw_obs = [obs_space.sample() for _ in range(length + 1)]
    acts = np.array([act_space.sample() for _ in range(length)])
    return types.Trajectory(obs=obs, acts=acts, infos=infos, terminal=True)

@pytest.fixture
def transitions(transitions_min: types.TransitionsMinimal, obs_space) -> types.Transitions:
    next_obs = np.array([obs_space.sample() for _ in range(length)])
    dones = np.zeros(length, dtype=bool)
    return types.Transitions(...)
```

## Mocking

**Framework:** Not extensively used; Hypothesis for property-based testing instead

**Patterns:**
- Minimal use of mocks; prefer real objects and fixtures
- When mocking needed: use `unittest.mock` (from Python standard library)
- Prefer monkeypatch for pytest fixture injection over manual mocking

**What to Mock:**
- External services (not present in codebase)
- Slow operations when not needed for test logic
- Third-party library behavior if affecting test

**What NOT to Mock:**
- The code under test
- Core imitation library components
- Environment behavior (use real Gym environments)
- Data structures (use fixtures instead)

## Property-Based Testing

**Framework:** Hypothesis (version ~6.54.1)

**Strategy Definition Pattern** from `tests/algorithms/test_bc.py`:
```python
########################
# HYPOTHESIS STRATEGIES
########################

env_names = st.shared(
    st.sampled_from(["Pendulum-v1", "seals/CartPole-v0"]),
    key="env_name",
)
rngs = st.shared(st.builds(np.random.default_rng), key="rng")
env_numbers = st.integers(min_value=1, max_value=10)
envs = st.builds(
    lambda name, num, rng: util.make_vec_env(name, n_envs=num, rng=rng),
    name=env_names,
    num=env_numbers,
    rng=rngs,
)
batch_sizes = st.integers(min_value=1, max_value=50)

# Use st.shared() to ensure same value is used across test
# Use st.builds() to construct complex objects
# Use st.sampled_from() for discrete choices
```

**Property-Based Test Pattern:**
```python
@hypothesis.given(
    env_name=env_names,
    bc_args=bc_args,
    expert_data_type=expert_data_types,
    rng=rngs,
)
@hypothesis.settings(
    deadline=None,  # No time limit (expert trajectories cached)
    max_examples=15,  # Limit number of examples generated
    suppress_health_check=[hypothesis.HealthCheck.data_too_large],
)
def test_smoke_bc_creation(env_name, bc_args, expert_data_type, rng, pytestconfig):
    cache = pytestconfig.cache
    bc.BC(
        **bc_args,
        demonstrations=make_expert_transition_loader(...)
    )
```

## Test Markers

**Configuration in `setup.cfg`:**
```ini
[tool:pytest]
markers =
    expensive: mark a test as expensive (deselect with '-m "not expensive"')
```

**Usage Pattern:**
```python
@pytest.mark.expensive
def test_expensive_operation():
    # Long-running test
    pass
```

**Parametrized Tests:**
```python
@pytest.mark.parametrize("obs_space", OBS_SPACES)
@pytest.mark.parametrize("act_space", ACT_SPACES)
@pytest.mark.parametrize("length", LENGTHS)
def test_trajectory(obs_space, act_space, length, trajectory):
    # Test runs for all combinations of parameter values
    pass
```

## Coverage

**Requirements:** No explicit target enforced in CI, but coverage tracking enabled

**Configuration in `setup.cfg`:**
```ini
[coverage:run]
source = imitation
include =
    src/*
    tests/*
omit =
    src/imitation/scripts/config/*

[coverage:report]
exclude_lines =
    if self.debug:
    pragma: no cover
    raise NotImplementedError
    if __name__ == .__main__.:
```

**View Coverage:**
```bash
pytest --cov=imitation --cov-report=html  # Generate HTML coverage report
pytest --cov=imitation --cov-report=term  # Terminal report
```

**Coverage Tool:** coverage (version ~6.4.2)

**Excluded Lines:**
- `if self.debug:` debug-only code
- Lines marked `# pragma: no cover`
- `raise NotImplementedError` for abstract methods
- `if __name__ == "__main__":` script entry points

## Test Types

**Unit Tests:**
- Test individual functions and classes in isolation
- Use fixtures for dependencies
- Focus on correctness of specific logic
- Example: `test_make_data_loader()` in `tests/algorithms/test_base.py`

**Integration Tests:**
- Test interactions between multiple components
- Use real environments (Gym/Gymnasium)
- Test algorithm training pipelines
- Example: `test_smoke_bc_training()` in `tests/algorithms/test_bc.py`

**Smoke Tests:**
- Basic creation and operation tests
- Verify no crashes or exceptions
- Use property-based generation for variety
- Example: All tests prefixed with `test_smoke_` (BC creation, BC training)

**Regression Tests:**
- Test specific bug fixes
- Include expected behavior demonstration
- Example: `test_check_fixed_horizon()` validates episode length consistency

## Common Patterns

**Async/Callback Testing:**
```python
# Test callback execution
def on_epoch_end(epoch_num):
    # Callback logic
    pass

trainer.train(
    on_epoch_end=on_epoch_end,
    n_epochs=3,
)
# Verify callback was called
```

**Error Testing:**
```python
# Test expected exceptions
with pytest.raises(ValueError, match="Batch size must be a multiple.*"):
    base.make_data_loader([], batch_size=5)

# Test exception message patterns
with pytest.raises(ValueError, match="Expected batch size.*"):
    _make_and_iterate_loader(batch_iterable, batch_size=wrong_batch_size)
```

**Determinism Testing:**
```python
def test_trainer_reproducible(init_trainer_fn, pendulum_venv):
    """Test that results are reproducible with same RNG seed."""
    # Run training with same RNG
    trainer1 = init_trainer_fn(pendulum_venv)
    trainer1.train(...)

    trainer2 = init_trainer_fn(pendulum_venv)
    trainer2.train(...)

    # Verify identical results
    assert_equal_policies(trainer1.policy, trainer2.policy)
```

**Data Validation Testing:**
```python
# Test shape and dtype validation
for wrong_batch_size in [4, 6, 42]:
    with pytest.raises(ValueError, match="Expected batch size.*"):
        _make_and_iterate_loader(batch_iterable, batch_size=wrong_batch_size)
```

## Test Configuration

**Timeout:**
- Global timeout: 590 seconds (just before CircleCI's 10-minute timeout)
- Set in `setup.cfg` under `[tool:pytest]`
- Prevents test hangs from going unnoticed

**Warning Filters:**
- Ignore specific deprecation warnings from dependencies
- Configuration in `setup.cfg`:
  ```ini
  filterwarnings =
      ignore:Using or importing the ABCs from 'collections':DeprecationWarning:(google|pkg_resources)
      ignore:Parameters to load are deprecated:Warning:gym
      ignore:The binary mode of fromstring is deprecated:DeprecationWarning:gym
  ```

**PyTest Cache:**
- Used for expert trajectory caching across test runs
- Access via `pytestconfig.cache.mkdir()` in fixtures
- Speeds up Hypothesis tests by avoiding regeneration of expert data

---

*Testing analysis: 2026-03-19*
