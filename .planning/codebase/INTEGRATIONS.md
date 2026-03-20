# External Integrations

**Analysis Date:** 2026-03-19

## APIs & External Services

**Model Hub (HuggingFace):**
- HuggingFace Model Hub - Repository of pretrained RL policies
  - SDK/Client: `huggingface_sb3` ~3.0
  - Auth: Requires HF token for private models (environment-based)
  - Usage: Load/push policies, download pretrained experts
  - Files: `src/imitation/policies/serialize.py`, `src/imitation/testing/expert_trajectories.py`

**Dataset Hub (HuggingFace):**
- HuggingFace Datasets Library - Cloud-hosted trajectory datasets
  - SDK/Client: `datasets` >= 2.8.0
  - Auth: Optional HF token for authentication
  - Usage: Download, cache, and version trajectory data
  - Files: `src/imitation/data/huggingface_utils.py`, `src/imitation/data/serialize.py`

## Data Storage

**Databases:**
- Local filesystem only - No relational or NoSQL database integration
- Trajectory storage:
  - Format: pickle files (`.pkl`, `.pkl.gz`) via `src/imitation/data/serialize.py`
  - Caching: Local directory-based with file locks for concurrent access
  - Format conversion: JSON via sacred experiment logs

**File Storage:**
- Local filesystem only
- Specific paths:
  - Log directory: Configurable via scripts (default: logs/)
  - Model checkpoints: Configurable via scripts
  - Sacred experiments: JSON metadata (config.json, run.json)
  - Sacred observers: File-based by default

**Caching:**
- Local filesystem cache with FileLock for concurrent access
- Lazy loading for HuggingFace datasets via custom transforms
- Implementation in `src/imitation/testing/expert_trajectories.py`

## Authentication & Identity

**Model/Dataset Access:**
- HuggingFace - Optional token authentication
  - Token location: `~/.huggingface/token` (standard HF location)
  - Usage: Accessing private models and datasets
  - Not required for public resources

**Git LFS:**
- Large model files via Git LFS
- Authentication: Git credentials (SSH or HTTPS)
- Used in CI/CD for caching large artifacts

## Monitoring & Observability

**Error Tracking:**
- None detected - Errors logged to files/console only

**Logs:**
- Multiple formats supported via stable-baselines3 logger:
  - stdout - Console output
  - log - Human-readable text files
  - csv - Comma-separated values
  - json - JSON format
  - tensorboard - TensorBoard event files
  - wandb - Weights & Biases (optional)

**Training Visualization:**
- TensorBoard - Via stable-baselines3.common.logger integration
  - Accessed at: logs/{exp_name}/tb/
  - Scalar logging: Loss, rewards, policy entropy, etc.

**Experiment Tracking:**
- Sacred - Experiment configuration and metadata logging
  - Observes: File-based observer (JSON dumps to disk)
  - Files: `src/imitation/util/sacred.py`
  - Metadata: config.json, run.json stored in sacred directory

**Weights & Biases (Optional):**
- Integration: `wandb == 0.12.21`
- Format: "wandb" output format in logger
- Implementation in `src/imitation/util/logger.py` (WandbOutputFormat class)
- Usage: Experiment tracking, visualization, and comparison
- Not installed by default - only in test requirements

## Hyperparameter Optimization

**Optuna Integration:**
- Framework: `optuna >= 3.0.1`
- Integration: Ray Tune search backend
- Files: `src/imitation/scripts/tuning.py`, `src/imitation/scripts/parallel.py`
- Usage: Bayesian optimization for hyperparameter search
- Requires: `ray[debug,tune]` optional dependency

**Ray Tune:**
- Framework: `ray[debug,tune] ~2.0.0`
- Purpose: Distributed hyperparameter tuning
- Integration: Works with Optuna for parallel search
- Scripts: `imitation-train-preference-comparisons`, `imitation-parallel`

## CI/CD & Deployment

**Hosting:**
- GitHub - Source code repository at github.com/HumanCompatibleAI/imitation
- PyPI - Package distribution
- Docker Hub - Pre-built images (humancompatibleai/imitation)

**CI Pipeline:**
- CircleCI - Main continuous integration
  - Config: `.circleci/config.yml`
  - Docker executors for Linux (xlarge)
  - macOS executors (py3.9 due to ray/python3.8 compatibility issue)
  - Windows executors (py3.9 for pywinpty)
  - Integration: codecov orb for coverage, shellcheck for bash, circleci-config-validate

**Code Quality Pipeline:**
- Pre-commit hooks (`.pre-commit-config.yaml`):
  - Black - Code formatting
  - isort - Import sorting
  - flake8 - Linting with multiple plugins
  - mypy - Type checking
  - pytype - Type analysis
  - Sphinx documentation building
  - Notebook cleaning and validation
  - Shellcheck for bash scripts
  - Codespell for spelling
  - Git-based validation (circleci-config-validate)

**Coverage Tracking:**
- Codecov - Coverage reports and analysis
  - Config: `.codecov.yml`
  - Thresholds:
    - Main source: Target coverage baseline
    - Auxiliary (examples/scripts): 0% requirement
    - Tests: 100% dead code requirement

**Documentation Hosting:**
- Read the Docs - Hosted at docs.imitation.org
  - Config: `.readthedocs.yml`
  - Build environment: Ubuntu 22.04, Python 3.8
  - Sphinx build with docs extra dependencies
  - All formats generated (HTML, PDF, ePub, etc.)

## Environment Configuration

**Required Environment Variables:**
- None mandated - Configuration via Sacred experiments and script arguments
- Optional: HF_TOKEN for private HuggingFace resources
- Docker/CI: GIT_LFS_SKIP_SMUDGE for caching control

**Configuration Methods:**
- Sacred experiments: YAML-based configuration files
  - Location: `src/imitation/scripts/config/`
  - Supports ingredient composition
- Script command-line arguments: Overrides Sacred configs
- Ingredient system: Modular configuration components
  - Files: `src/imitation/scripts/ingredients/`
  - Examples: environment, expert, demonstrations, logging

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- GitHub releases via setuptools_scm (automatic on tag)
- PyPI publication via CI workflow (`.github/workflows/publish-to-pypi.yml`)
- HuggingFace model uploads (manual via huggingface_sb3)

## Data Flow & Integration Architecture

**Expert Trajectory Pipeline:**
1. Query HuggingFace model hub (huggingface_sb3) for pretrained policies
2. Download policy to local cache
3. Roll out policy in environment (gymnasium)
4. Serialize trajectories to local disk (pickle format)
5. Optionally upload to HuggingFace datasets

**Training Pipeline:**
1. Load trajectories (local or HF datasets)
2. Configure with Sacred experiments
3. Train with stable-baselines3 algorithms + PyTorch
4. Log to multiple formats (TensorBoard, CSV, JSON, W&B)
5. Save checkpoints to local disk
6. Push best model to HuggingFace hub

**Hyperparameter Optimization Pipeline:**
1. Define search space via Optuna
2. Run parallel trials via Ray Tune
3. Log results to TensorBoard/CSV
4. Select best configuration
5. Train final model with best hyperparameters

## Version Pinning Strategy

**Strict Pinning (Tests & Docs):**
- Testing packages tightly pinned to known working versions
- Documentation packages tightly pinned for reproducibility
- Rationale: CI/CD stability

**Flexible Pinning (Installation):**
- Core packages: Tilde constraints (~=X.Y.Z) allow patch updates
- Secondary packages: No lower bound if compatible with recent versions
- Philosophy: Minimize dependency conflicts for end users

**Special Cases:**
- gym 0.21.0 support via setuptools 66.1.1 (last compatible version)
- wandb == 0.12.21 (exact version, specific to training configuration)
- jupyter-client ~6.1.12 (specific due to known bug in newer versions)

---

*Integration audit: 2026-03-19*
