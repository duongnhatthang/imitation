# Architecture

**Analysis Date:** 2026-03-19

## Pattern Overview

**Overall:** Modular, algorithm-centric architecture with clear separation of concerns. The codebase implements multiple imitation and reward learning algorithms as independent, composable modules that follow a common base class hierarchy and interface.

**Key Characteristics:**
- **Algorithm-first design**: Each algorithm (BC, GAIL, AIRL, DAgger, etc.) is self-contained in its own module
- **Base class abstraction**: `BaseImitationAlgorithm` and `DemonstrationAlgorithm` define common interface
- **Data-agnostic transitions**: Standardized data types (`types.Transitions`, `types.TrajectoryWithRew`) for algorithm compatibility
- **Reward network modularity**: Pluggable reward functions separate from algorithm logic
- **Environment wrapping**: Composable environment wrappers for rewards, logging, and state tracking

## Layers

**Algorithm Layer:**
- Purpose: Implements specific imitation/reward learning algorithms
- Location: `src/imitation/algorithms/`
- Contains: BC (behavioral cloning), DAgger, GAIL, AIRL, MCE-IRL, Density-based, SQIL, Preference Comparisons
- Depends on: Data types, rewards, policies, utilities, and Stable Baselines3
- Used by: Scripts (Sacred CLI), examples, and user code

**Data Layer:**
- Purpose: Handles trajectories, transitions, rollouts, and expert demonstrations
- Location: `src/imitation/data/`
- Contains: Type definitions (`types.py`), rollout collection (`rollout.py`), buffers, serialization, environment wrappers
- Depends on: Gymnasium, NumPy, PyTorch
- Used by: All algorithms, scripts, policies

**Reward Layer:**
- Purpose: Provides learnable and fixed reward functions
- Location: `src/imitation/rewards/`
- Contains: Reward networks (neural network modules), reward wrappers, serialization, reward function abstractions
- Depends on: PyTorch, Stable Baselines3, data types
- Used by: Adversarial algorithms, RL training, reward evaluation scripts

**Policy Layer:**
- Purpose: Policy management, serialization, and interactive utilities
- Location: `src/imitation/policies/`
- Contains: Policy serialization (load/save), exploration wrappers, replay buffer wrappers, interactive policy utilities
- Depends on: Stable Baselines3, data types
- Used by: Algorithms, scripts, examples

**Utility Layer:**
- Purpose: Cross-cutting concerns: logging, neural network building, configuration, type conversions
- Location: `src/imitation/util/`
- Contains: Hierarchical logger, network builders, Sacred configuration, registry, utility functions
- Depends on: Stable Baselines3, TensorBoard
- Used by: All other layers

**Regularization Layer:**
- Purpose: Applies regularization to reward/policy training
- Location: `src/imitation/regularization/`
- Contains: Regularizer implementations and updaters
- Depends on: PyTorch, utilities
- Used by: Reward learning algorithms (especially preference comparisons)

**Scripts/CLI Layer:**
- Purpose: Command-line interfaces and experiment orchestration using Sacred
- Location: `src/imitation/scripts/`
- Contains: Training scripts (train_rl, train_adversarial, train_imitation, train_preference_comparisons), evaluation, analysis, configuration, and ingredient definitions
- Depends on: All other layers, Sacred, Sacred ingredients
- Used by: End users via command line

## Data Flow

**Supervised Learning (BC) Flow:**

1. Expert demonstrations collected as `types.Transitions` or `types.Trajectory` objects
2. `bc.BC` algorithm accepts demonstrations via `set_demonstrations()`
3. Demonstrations converted to PyTorch DataLoader via `base.make_data_loader()`
4. Loss calculated via `BehaviorCloningLossCalculator` (supervised learning on (obs, action) pairs)
5. Policy optimized via gradient descent
6. Final policy available via `.policy` property

**Adversarial Learning (GAIL/AIRL) Flow:**

1. Expert demonstrations loaded into `types.TransitionMapping` format
2. Generator RL algorithm trained on environment with learned reward
3. Discriminator (reward network) trained to distinguish expert vs. generated transitions
4. Loop:
   - Generator collects rollouts: `rollout.rollout()` → `types.Trajectory` → `types.Transitions`
   - Discriminator batch: Expert batch + Generator batch of equal size
   - Compute discriminator logits and loss
   - Update discriminator (2 rounds per generator update by default)
   - Generator updates policy to maximize discriminator confusion (via learned reward)
5. `reward_train` and `reward_test` reward networks separated for evaluation

**Interactive Learning (DAgger) Flow:**

1. Start with expert demonstrations
2. Train behavioral cloning policy on expert data
3. Iterative rounds (controlled by `BetaSchedule`):
   - Sample from current policy: `β` fraction expert actions, `(1-β)` policy actions
   - Query expert for actions on policy-generated trajectories
   - Add new demonstrations to dataset
   - Retrain BC on combined data
   - `β` decreases from 1 to 0 over time

**Preference Comparisons Flow:**

1. TrajectoryDataset or live TrajectoryGenerator provides trajectory pairs
2. Human (or learned model) compares trajectory fragments, provides preferences
3. Reward network trained on preference labels (Bradley-Terry model)
4. Policy trained on learned reward
5. RL agent generates new trajectories for labeling
6. Iterative refinement

**State Management:**

- **Trajectory Accumulator** (`data.rollout.TrajectoryAccumulator`): Collects partial trajectories step-by-step from vectorized environments
- **Buffers** (`data.buffer.Buffer`): FIFO ring buffers for storing transitions with random sampling
- **Demonstrations**: Stored as transitions or trajectories, converted on-demand to appropriate format
- **Logger**: Hierarchical logger accumulates and logs statistics at training step intervals

## Key Abstractions

**BaseImitationAlgorithm:**
- Purpose: Base class for all learning algorithms
- Examples: `src/imitation/algorithms/base.py:BaseImitationAlgorithm`
- Pattern: Abstract base class with horizon validation, logger management, pickling support

**DemonstrationAlgorithm[TransitionKind]:**
- Purpose: Algorithms learning from expert demonstrations
- Examples: `src/imitation/algorithms/bc.BC`, `src/imitation/algorithms/adversarial/common.AdversarialTrainer`
- Pattern: Generic class handling various demonstration input formats (trajectories, transitions, data loaders)

**RewardNet:**
- Purpose: Pluggable neural network reward functions
- Examples: `src/imitation/rewards/reward_nets.BasicRewardNet`, `src/imitation/rewards/reward_nets.MLPRewardNet`
- Pattern: PyTorch `nn.Module` with standardized forward signature `(state, action, next_state, done) → reward`

**RewardVecEnvWrapper:**
- Purpose: Wraps VecEnv to use learned reward instead of environment reward
- Examples: `src/imitation/rewards/reward_wrapper.RewardVecEnvWrapper`
- Pattern: Environment wrapper returning learned rewards; separates `reward_train` and `reward_test`

**TrajectoryAccumulator & Buffer:**
- Purpose: Collect and store trajectories/transitions incrementally
- Examples: `src/imitation/data/rollout.TrajectoryAccumulator`, `src/imitation/data/buffer.Buffer`
- Pattern: Container classes for managing episode data during and after collection

**Types System:**
- Purpose: Define standardized data containers for transitions, trajectories, observations
- Examples: `src/imitation/data/types.Transitions`, `src/imitation/data/types.TrajectoryWithRew`, `src/imitation/data/types.DictObs`
- Pattern: Frozen dataclasses and TypedDicts for type safety and structure validation

## Entry Points

**CLI (Sacred):**
- Location: `src/imitation/scripts/train_rl.py`
- Triggers: `python -m imitation.scripts.train_rl` with config overrides
- Responsibilities: RL agent training, demonstration collection, policy/rollout saving

**CLI (Sacred):**
- Location: `src/imitation/scripts/train_adversarial.py`
- Triggers: `python -m imitation.scripts.train_adversarial gail|airl` with config overrides
- Responsibilities: GAIL/AIRL training, discriminator and policy saving

**CLI (Sacred):**
- Location: `src/imitation/scripts/train_imitation.py`
- Triggers: `python -m imitation.scripts.train_imitation bc|dagger` with config overrides
- Responsibilities: BC/DAgger training with interactive feedback

**CLI (Sacred):**
- Location: `src/imitation/scripts/train_preference_comparisons.py`
- Triggers: `python -m imitation.scripts.train_preference_comparisons` with config overrides
- Responsibilities: Preference comparison training for preference learning

**Python API:**
- Location: Examples in `examples/quickstart.py`
- Triggers: Direct algorithm instantiation and method calls
- Responsibilities: Flexible algorithmic control, custom training loops, experimentation

**Evaluation:**
- Location: `src/imitation/scripts/eval_policy.py`
- Triggers: `python -m imitation.scripts.eval_policy`
- Responsibilities: Policy evaluation on environment, statistics gathering

## Error Handling

**Strategy:** Fail-fast validation with informative error messages. Range checks, type validation, and state checks occur at algorithm initialization and training time.

**Patterns:**

- **Horizon validation**: `BaseImitationAlgorithm._check_fixed_horizon()` raises `ValueError` if episode lengths vary unexpectedly (configurable via `allow_variable_horizon`)
- **Batch size mismatch**: `_WrappedDataLoader` validates batch size matches expectation
- **Device placement**: `reward_nets.RewardNet.preprocess()` safely moves tensors to correct device
- **Pickle safety**: `BaseImitationAlgorithm.__getstate__/__setstate__` removes unpicklable logger before serialization
- **Trajectory format validation**: Algorithms verify transitions have required fields ("obs", "acts", etc.)

## Cross-Cutting Concerns

**Logging:** Hierarchical logger (`src/imitation/util/logger.HierarchicalLogger`) supports context-based logging, accumulates means, writes to multiple formats (stdout, log file, JSON, CSV, TensorBoard, Weights & Biases).

**Validation:** Core validation in `Transitions` and `Trajectory` type constructors; horizon checks in algorithms; device checks in reward networks.

**Authentication:** Sacred configuration system handles reproducibility and experiment tracking; Sacred ingredients encapsulate sub-configurations.

---

*Architecture analysis: 2026-03-19*
