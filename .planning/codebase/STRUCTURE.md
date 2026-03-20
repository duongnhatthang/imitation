# Codebase Structure

**Analysis Date:** 2026-03-19

## Directory Layout

```
imitation/
├── src/imitation/                          # Main package source code
│   ├── algorithms/                         # Learning algorithms
│   │   ├── adversarial/                    # GAIL, AIRL implementations
│   │   ├── base.py                         # Base classes for all algorithms
│   │   ├── bc.py                           # Behavioral Cloning
│   │   ├── dagger.py                       # DAgger (interactive learning)
│   │   ├── density.py                      # Density-based reward modeling
│   │   ├── mce_irl.py                      # Maximum Causal Entropy IRL
│   │   ├── preference_comparisons.py       # Preference comparison learning
│   │   └── sqil.py                         # Soft Q Imitation Learning
│   ├── data/                               # Data types and collection
│   │   ├── buffer.py                       # FIFO ring buffers for transitions
│   │   ├── rollout.py                      # Trajectory collection and manipulation
│   │   ├── serialize.py                    # Save/load trajectories to .npz files
│   │   ├── types.py                        # Data type definitions (Transitions, Trajectory, etc.)
│   │   ├── wrappers.py                     # Environment wrappers (RolloutInfoWrapper, etc.)
│   │   └── huggingface_utils.py            # Hugging Face dataset integration
│   ├── policies/                           # Policy management and utilities
│   │   ├── base.py                         # Base policy classes
│   │   ├── serialize.py                    # Policy save/load utilities
│   │   ├── exploration_wrapper.py          # Exploration strategy wrapper
│   │   ├── interactive.py                  # Interactive policy utilities
│   │   └── replay_buffer_wrapper.py        # Replay buffer environment wrapper
│   ├── regularization/                     # Training regularization
│   │   ├── regularizers.py                 # Regularizer implementations
│   │   └── updaters.py                     # Update strategies for regularizers
│   ├── rewards/                            # Reward function implementations
│   │   ├── reward_function.py              # Abstract reward function interface
│   │   ├── reward_nets.py                  # Neural network reward models
│   │   ├── reward_wrapper.py               # Environment wrapper for learned rewards
│   │   └── serialize.py                    # Reward model save/load
│   ├── scripts/                            # Command-line interfaces (Sacred)
│   │   ├── config/                         # Sacred configuration definitions
│   │   │   ├── tuned_hps/                  # Pre-tuned hyperparameter JSON configs
│   │   │   └── train_rl.py                 # Sacred config for RL training
│   │   ├── ingredients/                    # Sacred ingredients (sub-configs)
│   │   │   ├── demonstrations.py           # Demo loading ingredient
│   │   │   ├── environment.py              # Environment setup ingredient
│   │   │   ├── logging.py                  # Logging ingredient
│   │   │   ├── policy_evaluation.py        # Policy eval ingredient
│   │   │   ├── rl.py                       # RL algorithm ingredient
│   │   │   └── reward.py                   # Reward function ingredient
│   │   ├── analyze.py                      # Analysis utilities
│   │   ├── convert_trajs.py                # Trajectory format conversion
│   │   ├── eval_policy.py                  # Policy evaluation script
│   │   ├── parallel.py                     # Parallel experiment runner
│   │   ├── train_adversarial.py            # GAIL/AIRL training CLI
│   │   ├── train_imitation.py              # BC/DAgger training CLI
│   │   ├── train_preference_comparisons.py # Preference learning CLI
│   │   ├── train_rl.py                     # RL agent training CLI
│   │   └── tuning.py                       # Hyperparameter tuning utilities
│   ├── testing/                            # Testing utilities for users
│   │   └── test_*.py                       # Test helper modules
│   ├── util/                               # Utility functions and helpers
│   │   ├── logger.py                       # Hierarchical logging system
│   │   ├── networks.py                     # Neural network building utilities
│   │   ├── registry.py                     # Algorithm/component registry
│   │   ├── sacred.py                       # Sacred configuration utilities
│   │   ├── sacred_file_parsing.py          # Sacred file parsing helpers
│   │   ├── util.py                         # General utilities (tensor conversion, env creation)
│   │   └── video_wrapper.py                # Video recording wrapper
│   ├── __init__.py                         # Package initialization
│   └── py.typed                            # PEP 561 marker for type hints
├── tests/                                  # Test suite
│   ├── algorithms/                         # Algorithm tests (mirrors src structure)
│   │   ├── test_bc.py                      # BC algorithm tests
│   │   ├── test_adversarial.py             # GAIL/AIRL tests
│   │   ├── test_dagger.py                  # DAgger tests
│   │   ├── test_mce_irl.py                 # MCE-IRL tests
│   │   ├── test_preference_comparisons.py  # Preference learning tests
│   │   ├── test_density_baselines.py       # Density baseline tests
│   │   ├── test_base.py                    # Base algorithm tests
│   │   └── conftest.py                     # Algorithm test fixtures
│   ├── data/                               # Data layer tests
│   │   ├── test_types.py                   # Type definition tests
│   │   ├── test_rollout.py                 # Rollout collection tests
│   │   ├── test_buffer.py                  # Buffer tests
│   │   └── test_serialize.py               # Serialization tests
│   ├── policies/                           # Policy tests
│   │   ├── test_base.py                    # Policy base class tests
│   │   └── test_serialize.py               # Policy serialization tests
│   ├── rewards/                            # Reward tests
│   │   ├── test_reward_nets.py             # Reward network tests
│   │   ├── test_reward_wrapper.py          # Reward wrapper tests
│   │   └── test_reward_fn.py               # Reward function tests
│   ├── scripts/                            # Script/CLI tests
│   │   └── test_*.py                       # Script-specific tests
│   ├── util/                               # Utility tests
│   │   ├── test_logger.py                  # Logger tests
│   │   ├── test_networks.py                # Network builder tests
│   │   ├── test_util.py                    # General utility tests
│   │   └── test_registry.py                # Registry tests
│   ├── testdata/                           # Fixed test data (policies, rewards, demos)
│   ├── conftest.py                         # Root test configuration
│   ├── test_benchmarking.py                # Benchmark suite tests
│   ├── test_examples.py                    # Example script execution tests
│   ├── test_experiments.py                 # Experiment pipeline tests
│   └── test_regularization.py              # Regularization tests
├── examples/                               # Usage examples
│   ├── quickstart.py                       # Simple BC training example
│   ├── quickstart.sh                       # CLI example commands
│   └── train_dagger_atari_interactive_policy.py  # DAgger with Atari
├── docs/                                   # Sphinx documentation
├── experiments/                            # Experimental scripts and results
├── runners/                                # Experiment runner utilities
├── ci/                                     # CI/CD scripts
├── .github/                                # GitHub Actions workflows
├── .circleci/                              # CircleCI config
├── setup.py                                # Package installation config
├── setup.cfg                               # Additional setup config
├── pyproject.toml                          # Project metadata
├── mypy.ini                                # MyPy type checking config
├── .pre-commit-config.yaml                 # Pre-commit hooks
└── README.md                               # Project overview
```

## Directory Purposes

**src/imitation/algorithms/:**
- Purpose: Algorithm implementations (BC, DAgger, GAIL, AIRL, MCE-IRL, Density, SQIL, Preference Comparisons)
- Contains: Individual algorithm modules, adversarial learning shared code
- Key files: `base.py` (base classes), `bc.py` (behavioral cloning), `adversarial/common.py` (GAIL/AIRL shared code)

**src/imitation/data/:**
- Purpose: Data representation, collection, storage, and serialization
- Contains: Type definitions, trajectory collection utilities, buffers, environment wrappers
- Key files: `types.py` (Transitions, Trajectory definitions), `rollout.py` (trajectory collection), `buffer.py` (FIFO buffers)

**src/imitation/rewards/:**
- Purpose: Reward function abstractions and neural network implementations
- Contains: Reward network architectures, reward wrappers, serialization
- Key files: `reward_nets.py` (network implementations), `reward_wrapper.py` (environment integration)

**src/imitation/policies/:**
- Purpose: Policy management, serialization, and utilities
- Contains: Policy save/load, exploration wrappers, interactive utilities
- Key files: `serialize.py` (policy persistence), `base.py` (policy abstractions)

**src/imitation/util/:**
- Purpose: Cross-cutting utilities used throughout codebase
- Contains: Logging, network builders, configuration, type conversions, registries
- Key files: `logger.py` (HierarchicalLogger), `networks.py` (network construction), `util.py` (general helpers)

**src/imitation/scripts/:**
- Purpose: Command-line interfaces using Sacred for reproducibility and configuration
- Contains: Training scripts, evaluation, analysis tools, configuration definitions
- Key files: `train_rl.py` (RL training), `train_adversarial.py` (GAIL/AIRL), `train_imitation.py` (BC/DAgger)

**src/imitation/regularization/:**
- Purpose: Regularization techniques for reward/policy training
- Contains: Regularizer implementations and update strategies
- Key files: `regularizers.py`, `updaters.py`

**tests/:**
- Purpose: Comprehensive test suite mirroring source structure
- Contains: Unit and integration tests for all modules
- Key files: Algorithm-specific tests (`test_bc.py`, `test_adversarial.py`), fixture configuration (`conftest.py`)

## Key File Locations

**Entry Points:**
- `src/imitation/scripts/train_rl.py`: Train RL agents (expert data collection)
- `src/imitation/scripts/train_adversarial.py`: GAIL/AIRL training entry point
- `src/imitation/scripts/train_imitation.py`: BC/DAgger training entry point
- `src/imitation/scripts/train_preference_comparisons.py`: Preference learning entry point
- `examples/quickstart.py`: Simple BC training example

**Configuration:**
- `setup.py`: Package metadata, dependencies, entry points
- `pyproject.toml`: Project configuration metadata
- `src/imitation/scripts/config/`: Sacred experiment configurations
- `src/imitation/scripts/ingredients/`: Reusable configuration modules

**Core Logic:**
- `src/imitation/algorithms/base.py`: Base algorithm classes (BaseImitationAlgorithm, DemonstrationAlgorithm)
- `src/imitation/algorithms/bc.py`: Behavioral Cloning implementation
- `src/imitation/algorithms/dagger.py`: DAgger interactive learning
- `src/imitation/algorithms/adversarial/common.py`: GAIL/AIRL shared code
- `src/imitation/data/types.py`: Core data type definitions
- `src/imitation/data/rollout.py`: Trajectory collection and manipulation
- `src/imitation/rewards/reward_nets.py`: Reward neural network models

**Testing:**
- `tests/conftest.py`: Pytest configuration and shared fixtures
- `tests/algorithms/conftest.py`: Algorithm-specific test fixtures
- `tests/testdata/`: Pre-computed test data (models, trajectories)

## Naming Conventions

**Files:**
- `*.py`: Python modules
- `test_*.py`: Test modules (pytest discovery)
- `conftest.py`: Pytest configuration/fixtures
- Algorithm files use simple name: `bc.py` (not `behavioral_cloning.py`), `dagger.py`, `gail.py`

**Directories:**
- Lowercase snake_case: `src/imitation/algorithms/`, `src/imitation/rewards/`
- Subdirectories follow parent naming: `algorithms/adversarial/`, `scripts/config/`, `scripts/ingredients/`

**Classes:**
- PascalCase: `BehaviorCloningLossCalculator`, `AdversarialTrainer`, `RewardNet`, `DictObs`
- Algorithm classes extend base classes: `class BC(DemonstrationAlgorithm)`, `class GAIL(AdversarialTrainer)`

**Functions:**
- snake_case: `rollout()`, `make_vec_env()`, `flatten_trajectories()`, `make_sample_until()`
- Private functions prefixed with underscore: `_check_fixed_horizon()`, `_WrappedDataLoader`

**Type/Data Objects:**
- Named tuples and dataclasses use PascalCase: `Transitions`, `TrajectoryWithRew`, `DictObs`, `BCTrainingMetrics`

## Where to Add New Code

**New Algorithm:**
- Primary code: `src/imitation/algorithms/new_algorithm.py` or `src/imitation/algorithms/category/new_algorithm.py`
- Inherit from: `BaseImitationAlgorithm` or `DemonstrationAlgorithm` for demonstrated learning, or extend `AdversarialTrainer` for adversarial approaches
- Tests: `tests/algorithms/test_new_algorithm.py`
- Example/CLI: `src/imitation/scripts/train_new_algorithm.py` if user-facing

**New Reward Function:**
- Implementation: `src/imitation/rewards/reward_nets.py` (add class extending `RewardNet`)
- Serialization: Add serialization code to `src/imitation/rewards/serialize.py`
- Tests: `tests/rewards/test_reward_nets.py`

**New Data Type:**
- Definition: `src/imitation/data/types.py` (frozen dataclass or TypedDict)
- Conversion utilities: `src/imitation/data/types.py` or `src/imitation/data/rollout.py`
- Tests: `tests/data/test_types.py`

**New Utility Function:**
- General helpers: `src/imitation/util/util.py`
- Network building: `src/imitation/util/networks.py`
- Logging-related: `src/imitation/util/logger.py`
- Tests: Corresponding test file in `tests/util/`

**New Environment Wrapper:**
- Implementation: `src/imitation/data/wrappers.py` or `src/imitation/rewards/reward_wrapper.py`
- Tests: Appropriate test file in `tests/data/` or `tests/rewards/`

**New CLI Script:**
- Sacred configuration: `src/imitation/scripts/config/train_new_script.py`
- Sacred ingredients (if needed): `src/imitation/scripts/ingredients/new_ingredient.py`
- Main script: `src/imitation/scripts/train_new_script.py`
- Entry point: Add to `setup.py` console_scripts

## Special Directories

**src/imitation/scripts/config/tuned_hps/:**
- Purpose: Pre-tuned hyperparameter JSON configurations
- Generated: No (manually created and committed)
- Committed: Yes
- Usage: Referenced by Sacred configuration overrides

**tests/testdata/:**
- Purpose: Pre-computed test fixtures (saved policies, reward models, trajectories)
- Generated: No (committed once, reused)
- Committed: Yes
- Usage: Loaded by test modules to avoid training during test runs

**src/imitation/testing/:**
- Purpose: Testing utilities exported for users to test custom algorithms against standard tests
- Generated: No
- Committed: Yes
- Usage: Users import these utilities when implementing custom algorithms

**docs/:**
- Purpose: Sphinx documentation source
- Generated: No (source), Yes (built HTML/PDF)
- Committed: Source only (build artifacts excluded)
- Usage: `sphinx-build` generates documentation

---

*Structure analysis: 2026-03-19*
