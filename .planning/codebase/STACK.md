# Technology Stack

**Analysis Date:** 2026-03-19

## Languages

**Primary:**
- Python 3.8+ (3.8, 3.9, 3.10 supported) - Core implementation language for RL/IL algorithms

## Runtime

**Environment:**
- Python 3.8 minimum (setuptools requires Python >= 3.8.0)
- PyPy support listed in classifiers

**Package Manager:**
- pip - Standard Python package manager
- Lockfile: Not enforced (flexible dependency versions for users)
- setuptools 66.1.1 (pinned for gym==0.21.0 compatibility)

## Frameworks

**Core RL/IL:**
- gymnasium ~0.29 - Environment API (successor to gym)
- stable-baselines3 ~2.2.1 - Pre-built RL algorithms and utilities
- torch >= 1.4.0 - Deep learning framework (PyTorch)

**Configuration & Experiment Management:**
- sacred >= 0.8.4 - Experiment configuration, logging, and reproducibility
  - Located in `src/imitation/util/sacred.py`
  - Used for managing hyperparameters and recording run metadata

**Hyperparameter Optimization:**
- optuna >= 3.0.1 - Bayesian hyperparameter optimization
  - Integrated with Ray Tune for parallel optimization (`src/imitation/scripts/tuning.py`)

**Data:**
- datasets >= 2.8.0 - HuggingFace datasets library for trajectory data management
  - Located in `src/imitation/data/huggingface_utils.py`
  - Used for serialization and conversion of imitation learning trajectories

**Monitoring & Logging:**
- tensorboard >= 1.14 - TensorBoard integration for training visualization
- wandb == 0.12.21 - Weights & Biases integration (optional logging format)
  - Integrated in `src/imitation/util/logger.py` as output format option
  - Used for experiment tracking and visualization

**Model Serialization:**
- huggingface_sb3 ~3.0 - HuggingFace model hub integration for stable-baselines3
  - Used for downloading pretrained policies and pushing trained models
  - Located in `src/imitation/policies/serialize.py`

**Scientific Computing:**
- numpy >= 1.15 - Numerical computing
- scipy ~1.9.0 - Scientific algorithms (in test dependencies)
- scikit-learn >= 0.21.2 - ML utilities

**Visualization & Media:**
- matplotlib - Plot generation
- tqdm - Progress bars
- rich - Terminal formatting
- moviepy ~1.0.3 - Video creation (test/documentation)

## Key Dependencies

**Critical:**
- torch >= 1.4.0 - Deep neural networks essential for reward/policy learning
- stable-baselines3 ~2.2.1 - Industry-standard RL algorithm implementations
- gymnasium ~0.29 - Standard RL environment API
- numpy >= 1.15 - Array operations for trajectory data

**Infrastructure:**
- sacred >= 0.8.4 - Experiment reproducibility and configuration management
- datasets >= 2.8.0 - Trajectory data serialization and version control
- huggingface_sb3 ~3.0 - Model hub integration for distributed training
- optuna >= 3.0.1 - Automated hyperparameter search

## Optional Dependencies

**Parallel Computing:**
- ray[debug,tune] ~2.0.0 - Distributed training and hyperparameter tuning
  - Required for `imitation-parallel` script and Ray Tune integration
  - Extras: `extras_require["parallel"]`

**Environment Support:**
- gymnasium[classic-control] - CartPole, MountainCar, Acrobot, etc.
- gymnasium[mujoco] - MuJoCo-based continuous control tasks (optional)
- seals ~0.2.1 - Additional benchmark environments

**Atari:**
- seals[atari] ~0.2.1 - Atari game environments

## Testing & Development

**Testing Framework:**
- pytest ~7.1.2 - Test runner
- pytest-cov ~3.0.0 - Coverage reporting
- pytest-xdist ~2.5.0 - Parallel test execution
- pytest-timeout ~2.1.0 - Timeout protection
- pytest-notebook == 0.8.0 - Jupyter notebook testing
- coverage ~6.4.2 - Coverage measurement
- filelock ~3.7.1 - Concurrent test file locking

**Code Quality:**
- black[jupyter] ~22.6.0 - Code formatter
- isort ~5.0 - Import sorting
- flake8 ~4.0.1 - Linter
  - flake8-blind-except == 0.2.1
  - flake8-builtins ~1.5.3
  - flake8-commas ~2.1.0
  - flake8-debugger ~4.1.2
  - flake8-docstrings ~1.6.0
  - flake8-isort ~4.1.2
  - darglint ~1.8.1
- mypy ~0.990 - Static type checking
- pytype == 2023.9.27 - Type analysis (non-Windows)
- codespell ~2.1.0 - Spell checking

**Type Checking:**
- py.typed - PEP 561 marker for type hints in `src/imitation/py.typed`

**Documentation:**
- sphinx ~5.1.1 - Documentation generation
- sphinx-autodoc-typehints ~1.19.1 - Type hint documentation
- sphinx-rtd-theme ~1.0.0 - ReadTheDocs theme
- sphinxcontrib-napoleon == 0.7 - Google/NumPy docstring support
- furo == 2022.6.21 - Alternative Sphinx theme
- sphinx-copybutton == 0.5.0 - Copy code button
- sphinx-github-changelog ~1.2.0 - Changelog integration
- myst-nb == 0.17.2 - Jupyter notebook in Sphinx
- sphinx-autobuild - Auto-rebuild on changes

**Miscellaneous:**
- hypothesis ~6.54.1 - Property-based testing
- codecov ~2.1.12 - CodeCov CI integration
- ipykernel ~6.15.1 - Jupyter kernel
- jupyter ~1.0.0 - Jupyter notebook
- jupyter-client ~6.1.12 - Jupyter client (specific version due to bug)
- ipdb - Interactive debugger
- autopep8 - PEP 8 auto-formatter
- pre-commit >= 2.20.0 - Git pre-commit hooks
- setuptools_scm ~7.0.5 - Version management from git

## Configuration

**Version Management:**
- setuptools_scm - Automatic versioning from git tags
  - Custom version schemes in `setup.py`: `get_version()` and `get_local_version()`
  - Local version uses commit hash instead of distance to avoid duplicates on multiple branches

**Build System:**
- setuptools - Package building
- build system configured in `pyproject.toml`

**Environment Variables:**
- No .env file required (standard Python package)
- GIT_LFS_SKIP_SMUDGE - For CI environments (skip large file download on checkout)
- SETUPTOOLS_SCM_PRETEND_VERSION - For Docker builds

**Development Configuration:**
- `pyproject.toml` - Black code formatter configuration (target Python 3.8)
- `pyproject.toml` - Pytype configuration

## Platform Requirements

**Development:**
- macOS: GNU utilities required (gnu-getopt, parallel, coreutils)
  - Installation: `brew install gnu-getopt parallel coreutils`
- Linux: Standard build tools (gcc, make, etc.)
- Windows: Python 3.9+ recommended (due to ray/pywinpty compatibility)

**Production:**
- Python 3.8+
- CUDA 11.8 + cuDNN 8 (Docker image uses nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04)
- Binary dependencies: libgl1-mesa, libosmesa, libglew, ffmpeg, xorg, patchelf
- Git LFS for large model checkpoints

## Deployment

**Containerization:**
- Docker - Multi-stage builds for optimization
  - Base stage: OS + binary dependencies
  - Python-req stage: Python venv + requirements
  - Full stage: Complete with source code
  - Entry point: pytest runner for CI/CD

**Package Distribution:**
- PyPI - Published as `imitation` package
- GitHub Releases - For version tags
- HuggingFace Model Hub - For pretrained policies (via huggingface_sb3)

**CI/CD:**
- CircleCI - Main CI platform
  - Executors: Linux (xlarge), macOS, Windows
  - Caching: Git LFS, Python dependencies
  - Coverage reporting via codecov orb
  - Pre-commit validation via circleci-config-validate

**Documentation:**
- Read the Docs - Hosted documentation
- Configuration in `.readthedocs.yml`
- Python 3.8 environment with docs extra dependencies

---

*Stack analysis: 2026-03-19*
