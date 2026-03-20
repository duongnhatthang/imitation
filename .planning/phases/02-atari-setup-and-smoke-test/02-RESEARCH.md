# Phase 2: Atari Setup and Smoke Test - Research

**Researched:** 2026-03-19
**Domain:** Atari RL environment setup (seals, ALE, SB3), HuggingFace expert loading, random baseline collection, Sacred smoke test integration
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
None — all implementation choices are at Claude's discretion. Requirements are fully specified by REQUIREMENTS.md (ENV-01 through ENV-05, INFRA-05, INFRA-07).

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure/setup phase. Requirements are fully specified by REQUIREMENTS.md (ENV-01 through ENV-05, INFRA-05, INFRA-07).

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ENV-01 | 7 seals Atari games configured: Pong, Breakout, BeamRider, Enduro, Qbert, Seaquest, SpaceInvaders | seals[atari] installs ale-py 0.8.1 + shimmy[atari] + autorom; registers seals/PongNoFrameskip-v4 etc. |
| ENV-02 | Consistent Atari preprocessing: frame stacking (4), grayscale, 84x84 resize | SB3 AtariWrapper (grayscale, 84x84, clip_reward) + VecFrameStack(4) gives (4,84,84) obs space |
| ENV-03 | Expert policies loaded from HuggingFace sb3 org for all 7 games | All 7 sb3/ppo-*NoFrameskip-v4 repos confirmed to exist on HuggingFace (verified 2026-03) |
| ENV-04 | Expert observation space matches learner observation space | Learner venv must use SB3 make_atari_env + VecFrameStack(4); expert obs_space=(4,84,84) |
| ENV-05 | Random baseline scores computed and cached for all 7 games | HumanCompatibleAI has NO Atari datasets; must collect locally via rollout.rollout() with RandomPolicy |
| INFRA-05 | Quick smoke-test config: 1-2 games, 1 seed, 3-5 DAgger rounds | New experiment script with Sacred that runs BC, DAgger, FTRL with Atari config |
| INFRA-07 | Server setup script: create isolated Python env (venv or conda), install dependencies, clone repo | pip install -e ".[atari]" + autorom --accept-license to download ROMs |
</phase_requirements>

---

## Summary

Phase 2 establishes the Atari infrastructure for the empirical study. The primary work falls into three tracks: (1) environment setup — installing seals[atari] and verifying that all 7 game environments and their preprocessing pipelines produce the correct observation spaces; (2) expert loading — confirming all 7 HuggingFace sb3/ppo-*NoFrameskip-v4 models load without error; (3) random baseline collection — since HumanCompatibleAI publishes NO Atari datasets, random baselines must be collected locally and cached to disk.

The critical technical insight is the observation space compatibility problem (ENV-04). SB3 experts were trained with `AtariWrapper` (grayscale, 84x84, clip_reward) + `VecFrameStack(4)`, giving `Box(0,255,(4,84,84),uint8)`. The seals `make_atari_env` function adds only `AutoResetWrapper` and optionally `MaskScoreWrapper` — it does NOT apply preprocessing. Therefore, learner environments must be constructed with SB3's `make_atari_env + VecFrameStack(4)` (or equivalently, post_wrappers adding AtariWrapper + VecFrameStack) to match the expert's observation space.

The smoke test (INFRA-05) needs a new experiment script that runs all three methods (BC, DAgger, FTRL) on 2 Atari games with 1 seed and 3 DAgger rounds, producing Sacred output with normalized scores. This requires adding Sacred named configs for each Atari game and adding an `ftrl` Sacred command in the existing `train_imitation.py` or a new Atari-specific experiment script.

**Primary recommendation:** Use `SB3 make_atari_env + VecFrameStack(4)` for all learner environments. Collect random baselines locally with `RandomPolicy`. Install seals[atari] and ROM auto-download in the server setup script.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| seals[atari] | 0.2.1 | Register seals Atari env IDs; AutoReset + MaskScore wrappers | Project already uses seals 0.2.1; [atari] extra installs ale-py + shimmy |
| ale-py | ~0.8.1 | ALE (Arcade Learning Environment) backend | Required by seals[atari] |
| shimmy[atari] | >=0.1.0, <1.0 | gymnasium compatibility shim for ALE | Required by seals[atari] for gym API |
| autorom[accept-rom-license] | ~0.4.2 | Download Atari ROMs automatically | Required by seals[atari]; use autorom --accept-license |
| stable-baselines3 | ~2.2.1 | make_atari_env, VecFrameStack, AtariWrapper, PPO load | Already installed; provides Atari preprocessing chain |
| huggingface_sb3 | 3.0 | load_from_hub for expert policy loading | Already installed; existing serialize.py uses it |
| opencv-python | latest | Image processing for AtariWrapper | Required by seals[atari] |
| pillow | latest | Image processing support | Required by seals[atari] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sacred | >=0.8.4 | Experiment tracking, config management | Smoke test output; already installed |
| numpy | >=1.15 | Numerical ops, score caching (npz/pkl) | Random baseline collection |
| pickle / pathlib | stdlib | Cache random baselines to disk | Cache file for ENV-05 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| SB3 make_atari_env + VecFrameStack(4) | seals make_atari_env + manual wrappers | SB3's path exactly reproduces expert training setup; seals path requires manually matching wrapper chain |
| Local random baseline collection | HumanCompatibleAI HF datasets | HumanCompatibleAI has no Atari datasets (only MuJoCo); local collection is the only option |
| autorom --accept-license | Manual ROM download | autorom is already in seals[atari] deps and avoids interactive license prompts |

**Installation (server venv setup):**
```bash
# Create isolated venv
python3 -m venv .venv
source .venv/bin/activate

# Install imitation with Atari extras
pip install -e ".[atari]"

# Download Atari ROMs (accepts license automatically)
autorom --accept-license
```

---

## Architecture Patterns

### Recommended Project Structure
```
experiments/
├── atari_smoke.py           # New: Sacred smoke test script (BC, DAgger, FTRL on Atari)
├── collect_random_baselines.py   # New: Collect + cache random baseline scores
└── baselines/
    └── atari_random_scores.pkl   # Cached random baselines (dict: game -> mean_score)

src/imitation/scripts/config/
└── train_imitation.py       # Add: seals_pong, seals_breakout, ... named configs for Atari

setup/
└── setup_server.sh          # New: INFRA-07 server setup script
```

### Pattern 1: Atari Environment Construction (ENV-02, ENV-04)
**What:** Create learner venv with matching observation space to SB3 expert
**When to use:** Whenever creating venv for Atari DAgger/BC/FTRL training

```python
# Source: stable_baselines3 make_atari_env + VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

def make_atari_venv(game_env_id: str, n_envs: int = 8, seed: int = 0):
    """Create Atari venv matching SB3 expert observation space (4, 84, 84)."""
    # AtariWrapper: grayscale, 84x84, clip_reward, noop_max=30, frame_skip=4
    venv = make_atari_env(game_env_id, n_envs=n_envs, seed=seed)
    # VecFrameStack: stacks 4 frames -> obs shape (4, 84, 84)
    venv = VecFrameStack(venv, n_stack=4)
    return venv

# Usage: game_env_id = "PongNoFrameskip-v4" (raw ALE ID, not seals/)
venv = make_atari_venv("PongNoFrameskip-v4", n_envs=8, seed=0)
```

### Pattern 2: Expert Loading from HuggingFace (ENV-03)
**What:** Load pre-trained PPO expert from HuggingFace sb3 org
**When to use:** For all 7 games in ENV-03

```python
# Source: existing imitation/policies/serialize.py + huggingface_sb3
from imitation.policies.serialize import load_policy

# Existing infrastructure - just supply correct kwargs
expert_policy = load_policy(
    "ppo-huggingface",
    venv=venv,
    env_name="PongNoFrameskip-v4",   # must match sb3 repo naming
    organization="sb3",               # NOT "HumanCompatibleAI"
)
```

**HuggingFace repo ID format:** `sb3/ppo-{GameName}NoFrameskip-v4`

| Game | HF Repo ID | Verified |
|------|-----------|---------|
| Pong | sb3/ppo-PongNoFrameskip-v4 | Yes |
| Breakout | sb3/ppo-BreakoutNoFrameskip-v4 | Yes |
| BeamRider | sb3/ppo-BeamRiderNoFrameskip-v4 | Yes |
| Enduro | sb3/ppo-EnduroNoFrameskip-v4 | Yes |
| Qbert | sb3/ppo-QbertNoFrameskip-v4 | Yes |
| Seaquest | sb3/ppo-SeaquestNoFrameskip-v4 | Yes |
| SpaceInvaders | sb3/ppo-SpaceInvadersNoFrameskip-v4 | Yes |

### Pattern 3: Observation Space Assertion (ENV-04)
**What:** Assert learner venv obs space matches expert obs space
**When to use:** After constructing both venv and expert_policy

```python
# Source: ENV-04 requirement
assert venv.observation_space == expert_policy.observation_space, (
    f"Obs space mismatch: venv={venv.observation_space} "
    f"vs expert={expert_policy.observation_space}"
)
# Expected: both should be Box(0, 255, (4, 84, 84), uint8)
```

### Pattern 4: Random Baseline Collection (ENV-05)
**What:** Collect and cache random agent scores for all 7 games
**When to use:** Once per game; result cached to disk as a pickle file

```python
# Source: imitation rollout infrastructure
import pickle
import numpy as np
from imitation.data import rollout
from imitation.policies.base import RandomPolicy

def collect_random_baseline(game_env_id: str, n_episodes: int = 30, seed: int = 0):
    """Collect mean episode reward for a random policy on given Atari game."""
    rng = np.random.default_rng(seed)
    venv = make_atari_venv(game_env_id, n_envs=4, seed=seed)
    random_policy = RandomPolicy(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
    )
    trajs = rollout.rollout(
        random_policy,
        venv,
        rollout.make_sample_until(min_episodes=n_episodes),
        rng=rng,
    )
    venv.close()
    stats = rollout.rollout_stats(trajs)
    return stats["return_mean"]

# Cache structure: {"Pong": -20.1, "Breakout": 1.7, ...}
```

### Pattern 5: Smoke Test Script (INFRA-05)
**What:** Sacred experiment running BC, DAgger, FTRL on Atari with minimal config
**When to use:** Verification that all three methods work end-to-end on Atari

The smoke test extends the existing `train_imitation_ex` Sacred experiment by:
1. Adding Atari named configs (e.g., `seals_pong`) to `train_imitation.py`
2. Adding an `ftrl` Sacred command to `train_imitation.py` (analogous to existing `bc` and `dagger` commands)
3. Running with smoke-test config: `total_timesteps=50000`, 3 DAgger rounds, 1 seed

**Sacred command pattern for FTRL:**
```python
@train_imitation_ex.command
def ftrl(bc, dagger, _run, _rnd):
    """Runs FTRL-DAgger training."""
    from imitation.algorithms.ftrl import FTRLDAggerTrainer
    custom_logger, log_dir = logging_ingredient.setup_logging()
    with environment.make_venv() as venv:
        # ... construct FTRLDAggerTrainer and run
        # produce Sacred output with normalized score
```

### Pattern 6: Server Setup Script (INFRA-07)
```bash
#!/bin/bash
# setup_server.sh — INFRA-07
set -euo pipefail

REPO_URL="git@github.com:your-org/imitation.git"
VENV_DIR="$HOME/.venv/imitation"

# 1. Clone repo (skip if exists)
[ -d imitation ] || git clone "$REPO_URL" imitation
cd imitation

# 2. Create isolated venv
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# 3. Install with Atari extras
pip install --upgrade pip
pip install -e ".[atari]"

# 4. Download ROMs (non-interactive)
autorom --accept-license

# 5. Smoke test: verify ALE registers correctly
python3 -c "
import seals  # triggers seals Atari env registration
import gymnasium as gym
env = gym.make('PongNoFrameskip-v4')
print('ALE environment OK:', env.observation_space)
env.close()
"
```

### Anti-Patterns to Avoid

- **Using seals/PongNoFrameskip-v4 without preprocessing wrappers:** seals env obs space is raw (210,160,3) RGB; expert expects (4,84,84). Adding AtariWrapper + VecFrameStack is mandatory.
- **Loading expert with organization="HumanCompatibleAI":** The Atari PPO experts are in the `sb3` org, not `HumanCompatibleAI`. The existing `expert.py` ingredient defaults to `HumanCompatibleAI`; must override to `sb3`.
- **Using seals env IDs for expert loading:** Expert loading via `serialize.py` needs `env_name="PongNoFrameskip-v4"` (without "seals/") to form the correct HF repo ID `sb3/ppo-PongNoFrameskip-v4`.
- **Assuming HumanCompatibleAI has Atari datasets:** They do not. Only MuJoCo envs (Walker2d, Hopper, etc.) have random-seals-* datasets. Must collect locally.
- **clip_reward=True in evaluation:** AtariWrapper clips rewards to {-1,0,+1} during training; for score normalization, must evaluate with unclipped rewards. Use `clip_reward=False` in evaluation venv.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Atari preprocessing | Custom wrapper chain | SB3 AtariWrapper | Handles noop, frame skip, grayscale, resize, life-loss, fire-reset in correct order |
| Frame stacking | Custom frame buffer | SB3 VecFrameStack | Thread-safe, vectorized, correct channels_order |
| ALE backend | Custom ROM loader | ale-py + autorom | ALE handles ROM licensing, versioning, 57 games |
| Expert policy loading | Custom HF download | existing serialize.load_policy("ppo-huggingface", ...) | Already implemented in imitation; handles caching |
| Random rollout collection | Manual env loop | imitation rollout.rollout() + rollout_stats() | Handles vectorized envs, episode termination, stats |
| Sacred experiment config | Custom config system | Sacred named_config | Already used for DAgger; just add Atari named configs |

**Key insight:** The entire preprocessing pipeline (NoopReset + MaxAndSkip + EpisodicLife + FireReset + WarpFrame + ClipReward + VecFrameStack) is reproduced exactly by `SB3 make_atari_env + VecFrameStack(4)`. Hand-rolling this misses edge cases like `FireResetEnv` behavior.

---

## Common Pitfalls

### Pitfall 1: Organization Mismatch for HF Expert Loading
**What goes wrong:** `load_policy("ppo-huggingface", venv, env_name="PongNoFrameskip-v4")` uses default `organization="HumanCompatibleAI"` → looks for `HumanCompatibleAI/ppo-PongNoFrameskip-v4` which does not exist → 404 error.
**Why it happens:** The `expert.py` ingredient defaults to `HumanCompatibleAI`; Atari PPO experts are in the `sb3` org.
**How to avoid:** Always pass `organization="sb3"` explicitly when loading Atari experts.
**Warning signs:** `huggingface_hub.utils.HfHubHTTPError: 404 Client Error` during expert loading.

### Pitfall 2: Observation Space Mismatch (ENV-04 Assertion Fails)
**What goes wrong:** `assert venv.observation_space == expert_policy.observation_space` fails because venv has (210,160,3) or (84,84,1) while expert expects (4,84,84).
**Why it happens:** seals `make_atari_env` only adds AutoReset/MaskScore, not AtariWrapper or frame stacking. SB3 make_atari_env alone gives (84,84,1). VecFrameStack is needed to stack to (4,84,84).
**How to avoid:** Use `make_atari_env + VecFrameStack(4)` for learner venv construction.
**Warning signs:** Assertion error mentioning shape mismatch (n,) vs (4,84,84).

### Pitfall 3: ALE Not Registered Without seals[atari] Install
**What goes wrong:** `gym.make("PongNoFrameskip-v4")` raises `gymnasium.error.NameNotFound` because ale-py and shimmy are not installed.
**Why it happens:** Base seals install only includes `gymnasium + numpy`; Atari requires extras.
**How to avoid:** Run `pip install -e ".[atari]" && autorom --accept-license` on the server.
**Warning signs:** `gymnasium.error.NameNotFound: Environment PongNoFrameskip-v4 doesn't exist`.

### Pitfall 4: seals Atari Env IDs vs Raw ALE IDs
**What goes wrong:** Confusion between `seals/PongNoFrameskip-v4` (seals namespace) and `PongNoFrameskip-v4` (ALE/raw). Using seals ID for expert loading forms wrong HF repo name.
**Why it happens:** seals registers envs under `seals/` namespace; SB3 expert repos use raw ALE names.
**How to avoid:** Use raw ALE ID (`PongNoFrameskip-v4`) for expert loading; the seals env IDs are for the learner env (if using seals wrappers) or can be bypassed entirely by using SB3's `make_atari_env` directly with raw ALE IDs.
**Warning signs:** HF 404 error, or env creation error when using wrong ID.

### Pitfall 5: Reward Clipping During Score Evaluation
**What goes wrong:** Normalized scores are computed from episode returns; AtariWrapper clips rewards to {-1,0,+1} which changes episode return values, making normalized scores meaningless.
**Why it happens:** `clip_reward=True` is the default in AtariWrapper — intended for training, not evaluation.
**How to avoid:** Create a separate evaluation venv with `wrapper_kwargs={"clip_reward": False}` when evaluating policies for score normalization.
**Warning signs:** All episode returns look like small integers (-21 to +21 for Pong) instead of raw game scores.

### Pitfall 6: ROM Not Downloaded on Fresh Server
**What goes wrong:** `autorom` is installed but ROMs not downloaded; `gym.make` fails at env creation time.
**Why it happens:** `autorom` package only auto-downloads on first invocation of `autorom --accept-license`, not at import time.
**How to avoid:** Include `autorom --accept-license` as an explicit step in setup script and verify with a quick env test.
**Warning signs:** `ale_py.error.RuntimeError: ROM file not found`.

---

## Code Examples

### Full Atari Venv Construction Matching Expert Obs Space
```python
# Source: SB3 make_atari_env docs + VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

def make_atari_training_venv(game_id: str, n_envs: int = 8, seed: int = 0):
    """Training venv: (4, 84, 84) obs, clip_reward=True."""
    venv = make_atari_env(game_id, n_envs=n_envs, seed=seed)
    return VecFrameStack(venv, n_stack=4)

def make_atari_eval_venv(game_id: str, n_envs: int = 8, seed: int = 0):
    """Evaluation venv: (4, 84, 84) obs, clip_reward=False for true scores."""
    venv = make_atari_env(
        game_id, n_envs=n_envs, seed=seed,
        wrapper_kwargs={"clip_reward": False}
    )
    return VecFrameStack(venv, n_stack=4)
```

### Expert Loading for Atari (Correct Organization)
```python
# Source: imitation/policies/serialize.py + HF sb3 org
from imitation.policies.serialize import load_policy

def load_atari_expert(venv, game_id: str):
    """Load PPO expert from HuggingFace sb3 org."""
    return load_policy(
        "ppo-huggingface",
        venv=venv,
        env_name=game_id,        # e.g. "PongNoFrameskip-v4"
        organization="sb3",      # NOT "HumanCompatibleAI"
    )
```

### Random Baseline Collection and Caching
```python
# Source: imitation rollout utilities
import pickle
import numpy as np
from pathlib import Path
from imitation.data import rollout
from imitation.policies.base import RandomPolicy

GAMES = {
    "Pong": "PongNoFrameskip-v4",
    "Breakout": "BreakoutNoFrameskip-v4",
    "BeamRider": "BeamRiderNoFrameskip-v4",
    "Enduro": "EnduroNoFrameskip-v4",
    "Qbert": "QbertNoFrameskip-v4",
    "Seaquest": "SeaquestNoFrameskip-v4",
    "SpaceInvaders": "SpaceInvadersNoFrameskip-v4",
}

def collect_and_cache_random_baselines(cache_path: Path, n_episodes: int = 30, seed: int = 0):
    """Collect random baselines for all 7 games; cache to disk."""
    baselines = {}
    rng = np.random.default_rng(seed)
    for game_name, game_id in GAMES.items():
        venv = make_atari_eval_venv(game_id, n_envs=4, seed=seed)
        random_policy = RandomPolicy(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
        )
        trajs = rollout.rollout(
            random_policy, venv,
            rollout.make_sample_until(min_episodes=n_episodes),
            rng=rng,
        )
        venv.close()
        stats = rollout.rollout_stats(trajs)
        baselines[game_name] = {
            "mean": float(stats["return_mean"]),
            "std": float(stats["return_std"]),
        }
        print(f"{game_name}: random mean={baselines[game_name]['mean']:.2f}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(baselines, f)
    return baselines
```

### Smoke Test Normalized Score Computation
```python
# Source: CONTEXT.md established pattern
def compute_normalized_score(agent_score, random_score, expert_score):
    """Normalized score = (agent - random) / (expert - random)."""
    if expert_score == random_score:
        return 0.0
    return (agent_score - random_score) / (expert_score - random_score)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| gym (OpenAI) | gymnasium (Farama) | 2022-2023 | seals 0.2.1 uses gymnasium; shimmy bridges ALE |
| atari-py | ale-py + shimmy | 2021-2022 | ale-py 0.8.x is the current ALE backend; autorom handles ROMs |
| manual ROM download | autorom --accept-license | 2021+ | Automated, non-interactive ROM install |
| OpenAI gym AtariWrapper | SB3 AtariWrapper (same API) | SB3 v1+ | SB3 ships its own wrapper copy; same behavior |

**Deprecated/outdated:**
- `atari-py`: Replaced by `ale-py`. Do not install `atari-py`.
- `gym.make("PongNoFrameskip-v4")` without shimmy: Fails on gymnasium; requires `shimmy[atari]` for ALE env creation via gymnasium.
- `channels_last=True` in VecFrameStack for Atari: CnnPolicy expects channels-first (4,84,84); use default `channels_order=None` which auto-detects from obs space.

---

## Open Questions

1. **BeamRider expert policy performance**
   - What we know: `sb3/ppo-BeamRiderNoFrameskip-v4` exists on HF (confirmed via API)
   - What's unclear: Initial web search only found DQN/QR-DQN for BeamRider; the sb3 API listing confirmed PPO exists but it may have lower performance than the others
   - Recommendation: Load and verify during Phase 2; if performance is too low, note it but proceed (still useful as an expert)

2. **Seals env IDs needed for the study**
   - What we know: seals env IDs are `seals/PongNoFrameskip-v4` etc after `seals[atari]` install; the study uses raw ALE IDs directly via SB3 make_atari_env
   - What's unclear: Whether the CONTEXT.md note about "seals wrappers" means we must use seals IDs specifically, or if bypassing seals for learner env construction (using raw ALE IDs + SB3 make_atari_env) is acceptable
   - Recommendation: Use raw ALE IDs with SB3 make_atari_env. The CONTEXT.md says "Atari envs created via make_atari_env or seals wrappers" — the SB3 make_atari_env path is explicitly mentioned. seals wrappers would require manual preprocessing to match expert obs space.

3. **Number of episodes for random baseline stability**
   - What we know: Must cache to disk; standard practice varies by game
   - What's unclear: Whether 30 episodes per game is sufficient for stable estimates; Atari episodes can vary greatly in length
   - Recommendation: Use 30 episodes per game (conservative); document the variance; this can be increased later if normalization is noisy.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest ~7.1.2 |
| Config file | setup.cfg (`[tool:pytest]`) |
| Quick run command | `pytest tests/algorithms/test_atari_setup.py -x -v` |
| Full suite command | `pytest tests/ -x -v -m "not expensive"` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENV-01 | 7 Atari seals game IDs register without error after seals[atari] install | unit | `pytest tests/test_atari_setup.py::test_all_7_games_register -x` | Wave 0 |
| ENV-02 | make_atari_venv produces Box(0,255,(4,84,84),uint8) obs space | unit | `pytest tests/test_atari_setup.py::test_atari_obs_space -x` | Wave 0 |
| ENV-03 | All 7 SB3 experts load from HF without error | integration | `pytest tests/test_atari_setup.py::test_all_experts_load -x -m "not expensive"` | Wave 0 |
| ENV-04 | venv.observation_space == expert.observation_space for all 7 games | unit | `pytest tests/test_atari_setup.py::test_obs_space_assertion -x` | Wave 0 |
| ENV-05 | Random baselines cache file exists and has all 7 game entries | unit | `pytest tests/test_atari_setup.py::test_random_baselines_cached -x` | Wave 0 |
| INFRA-05 | Smoke test runs BC, DAgger, FTRL to completion on 2 games, 1 seed, 3 rounds | integration | `pytest tests/test_atari_setup.py::test_smoke_test -x -m "not expensive"` | Wave 0 |
| INFRA-07 | setup_server.sh runs without error (tested on server, not in CI) | manual | manual | N/A |

### Sampling Rate
- **Per task commit:** `pytest tests/test_atari_setup.py -x -v`
- **Per wave merge:** `pytest tests/ -x -v -m "not expensive"`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_atari_setup.py` — covers ENV-01 through ENV-05, INFRA-05
- [ ] `experiments/atari_smoke.py` — Sacred smoke test script with BC, DAgger, FTRL commands
- [ ] `experiments/collect_random_baselines.py` — standalone script for ENV-05 collection
- [ ] `setup/setup_server.sh` — INFRA-07 server setup script

---

## Sources

### Primary (HIGH confidence)
- Inspected `seals/atari.py` source at `/opt/miniconda3/lib/python3.8/site-packages/seals/atari.py` — confirmed `make_atari_env` only adds AutoReset/MaskScore (no AtariWrapper), all 7 game names in SCORE_REGIONS dict
- Inspected `seals-0.2.1.dist-info/METADATA` — confirmed `[atari]` extras: `ale-py ~=0.8.1`, `shimmy[atari] <1.0,>=0.1.0`, `autorom[accept-rom-license] ~=0.4.2`, `opencv-python`, `pillow`
- Inspected `imitation/policies/serialize.py` — confirmed `_load_stable_baselines_from_huggingface` uses `organization` param, forms repo_id as `{organization}/{algo}-{env_name}`
- Inspected `imitation/scripts/ingredients/expert.py` — confirmed default `organization = "HumanCompatibleAI"` must be overridden for sb3 Atari experts
- Inspected `stable_baselines3.common.env_util.make_atari_env` source — confirmed AtariWrapper applied; `VecFrameStack(4)` needed separately
- Inspected `stable_baselines3.common.atari_wrappers.AtariWrapper` — confirmed full preprocessing chain

### Secondary (MEDIUM confidence)
- HuggingFace API `huggingface.co/api/models?search=sb3/ppo` — confirmed all 7 `sb3/ppo-*NoFrameskip-v4` repos exist (Pong, Breakout, BeamRider, Enduro, Qbert, Seaquest, SpaceInvaders)
- HuggingFace `sb3/ppo-QbertNoFrameskip-v4` model page — confirmed `frame_stack: 4`, `CnnPolicy`, `AtariWrapper` hyperparameters
- HuggingFace `sb3/ppo-PongNoFrameskip-v4` model page — confirmed `frame_stack: 4`, `CnnPolicy`, `normalize: False`
- HuggingFace API `api/datasets?author=HumanCompatibleAI` — confirmed all 19 HumanCompatibleAI datasets are MuJoCo only; NO Atari datasets exist

### Tertiary (LOW confidence)
- None — all critical claims verified against primary or secondary sources

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — verified from installed packages and METADATA files
- Architecture: HIGH — based on inspected source code; HF API calls confirmed expert repos exist
- Pitfalls: HIGH — derived directly from source code inspection (serialize.py default org, seals atari.py wrapper chain)
- ENV-05 local collection: HIGH — HF API call proved no HumanCompatibleAI Atari datasets exist

**Research date:** 2026-03-19
**Valid until:** 2026-06-19 (stable libraries; HF model repos unlikely to change)
