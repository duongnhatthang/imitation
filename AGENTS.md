# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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

## Execution host sync

Use the existing sync script for all code/results transfers. Do not write raw rsync commands:
```bash
./experiments/sync_results.sh push   # push code to server
./experiments/sync_results.sh pull   # pull plot images/PDFs only
```
The script is gitignored. Edit it if new exclusions are needed rather than inlining rsync flags.

## Dependencies

Core: `torch`, `stable-baselines3 ~= 2.2.1`, `gymnasium ~= 0.29`, `sacred >= 0.8.4`, `seals ~= 0.2.1`

## Experiment storage and result publication (required)

These rules apply to every experiment, feature, diagnostic, and worktree.

- Keep raw results, datasets, trajectories, checkpoints, tensor/NPZ caches, logs, and derived numerical analysis on the execution server. Run analysis and plotting there. Do not download these artifacts for local analysis.
- Publish only plot images/PDFs locally through the existing private sync wrapper. Its default pull is plots-only. Never use pull-selected to bypass this rule for experiment data. Small private control/review receipts may remain local; they are not result datasets.
- The user-facing plot destination is the MAIN repository's `experiments/plots/<run-id>/`, even when implementation uses a worktree. Follow the existing `classical/{learning_curves,coverage,runtime}/` and `atari/{learning_curves,coverage,runtime}/` layout. Do not hide new plots in `.local`, a worktree, or `experiments/agnostic/results`.
- Use a recognizable stable study/run ID. Label figures with environment, method, full observation or exact mask, seed or seed aggregation, metric, budget, and main versus diagnostic/pilot scope. Keep diagnostics and historical plots in named subfolders. Do not mix different protocols or silently overwrite older figures.
- Keep one current learning-curve location and coverage grouped by environment. Excluded diagnostics belong in `classical/diagnostics/` or `atari/diagnostics/`; older render revisions and pilots belong in the run's `archive/`. Do not publish `concise-*`, `reviewed-*`, or revision-number folders as current review locations. Export through `experiments/publish_plot_view.py` and use the existing plots-only sync wrapper. Choose current revisions explicitly in the private per-run layout policy, preserve old figure bytes, and use a new run ID for scientific protocol changes.
- Preserve the specified coverage design: t-SNE of the existing features, a heatmap colored by sample arrival time, a separate overlap/comparison map, and one plot per training seed. Reuse existing coverage implementations. Ask the user before changing these semantics.
- Extend existing experiment, plotting, and sync paths. Do not add permanent one-off baseline runners, queues, or analyzers merely to publish a new method.
- Before claiming an experiment is delivered, verify every expected plot exists in the main local plot mirror, check its labels and raw-result provenance on the server, and explicitly report missing figures or unresolved scientific qualification. Training completion alone is not delivery completion.
- Before removing an existing local raw artifact, preserve it remotely and verify its content hash. Keep server paths, connection details, and private receipts out of public files and commits.
