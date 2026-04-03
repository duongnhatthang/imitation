# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**imitation** is a research-grade Python library providing clean implementations of imitation and reward learning algorithms, built on PyTorch and stable-baselines3. Core algorithms: BC, DAgger, GAIL, AIRL, MCE-IRL, SQIL, Density-based reward modeling, and deep RL from human preferences.

## Common Commands

### Installation
```bash
pip install -e ".[dev]"          # Full development install
pip install -e ".[test]"         # Test dependencies only
```

### Testing
```bash
pytest -n auto -vv tests/                          # Full parallel test suite
pytest tests/algorithms/test_bc.py -vv             # Single test file
pytest tests/algorithms/test_bc.py::TestBC::test_train -vv  # Single test
pytest -m "not expensive" tests/                   # Skip slow tests
```

### Linting & Formatting
```bash
black .                  # Auto-format
isort .                  # Sort imports
flake8                   # Static analysis
mypy --follow-imports=silent --show-error-codes    # Type checking
codespell --skip=*.pyc,tests/testdata/*,*.ipynb,*.csv --ignore-words-list=reacher,ith,iff
```

### Documentation
```bash
cd docs/ && make clean && make html    # Build docs
cd docs/ && make doctest               # Check doctests
```

## Architecture

### Source Layout (`src/imitation/`)

- **`algorithms/`**: All learning algorithms. Base classes in `base.py` (`BaseImitationAlgorithm`, `DemonstrationAlgorithm`). Adversarial methods (GAIL, AIRL) share code via `adversarial/common.py`. `preference_comparisons.py` is the largest module (deep RL from human preferences).
- **`data/`**: Data pipeline. `types.py` defines core structures (Transition, Trajectory, TransitionMapping). `rollout.py` handles trajectory collection. `buffer.py` provides replay buffers. `serialize.py` for persistence. `huggingface_utils.py` for HF Datasets integration.
- **`policies/`**: Policy classes extending SB3 (FeedForward32Policy, SAC1024Policy). Serialization, exploration wrappers, replay buffer wrappers.
- **`rewards/`**: Reward network architectures (`reward_nets.py`), environment reward wrapping (`reward_wrapper.py`), serialization.
- **`scripts/`**: Sacred-based CLI experiment entry points. `ingredients/` contains reusable Sacred config components. `config/` stores tuned hyperparameters as JSON.
- **`util/`**: Environment creation (`make_vec_env`), `HierarchicalLogger` with context-based metric accumulation, network builders (`build_mlp`, `build_cnn`), Sacred helpers.
- **`regularization/`**: Weight decay and Lp regularizers.
- **`testing/`**: Test utilities (expert trajectory generation, Hypothesis strategies, reward net helpers).

### Key Patterns

- **Data flow**: single-step Transitions → full-episode Trajectories → Buffers for training
- **Sacred configuration**: CLI experiments use Sacred ingredients for modular, reproducible configs. Sacred config files in `scripts/config/` use `F841` ignore (variables read by Sacred, not Python)
- **Vectorized environments**: All training uses SB3 `VecEnv` for parallelism
- **HierarchicalLogger**: Custom logger with context managers for structured metric accumulation and TensorBoard output

## Code Style

- **Formatter**: Black (line length 88)
- **Import sorting**: isort (multi-line mode 3, trailing commas, known first party: `imitation`)
- **Docstrings**: Google-style (enforced by darglint)
- **Type hints**: Throughout, checked by mypy. `py.typed` marker enabled.
- **Test markers**: `@pytest.mark.expensive` for slow tests
- **Environments**: Uses `gymnasium` (not old `gym`) and `seals` for test environments

## Dependencies

Core: `torch`, `stable-baselines3 ~= 2.2.1`, `gymnasium ~= 0.29`, `sacred >= 0.8.4`, `seals ~= 0.2.1`
