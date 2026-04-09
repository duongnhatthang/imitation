# Expanded Environment Suite: Atari + Additional Classical MDPs

**Date**: 2026-04-09
**Status**: Approved
**Goal**: Verify FTL vs FTRL vs BC on harder environments where frozen-feature linear mode is non-trivial.

## Background

Current experiments test on 5 classical Gymnasium MDPs (CartPole, FrozenLake, CliffWalking, Acrobot, MountainCar). With frozen expert features and only `action_net` trainable, even BC beats FTL and FTRL — the tasks are too easy. We need harder environments to surface regularization differences.

## Scope

### New Classical MDPs (3)

| Environment | Obs Type | Action Space | Notes |
|---|---|---|---|
| Taxi-v3 | Discrete (500 states) | 6 | One-hot encoded, grid navigation |
| Blackjack-v1 | Tuple(Discrete(32), Discrete(11), Discrete(2)) | 2 | `FlattenTupleObsWrapper`: concatenate one-hot vectors → Box(45,) |
| LunarLander-v3 | Continuous (8D) | 4 | Harder control problem |

### Atari Games — 3 Tiers (16 total)

**Tier 1: HuggingFace model zoo** (no training needed, priority):
- Pong, Breakout, SpaceInvaders, BeamRider, Qbert, MsPacman, Enduro, Seaquest

**Tier 2: Fast self-trained** (~30 min each, ~1-2M steps):
- Freeway, Atlantis, DemonAttack, CrazyClimber

**Tier 3: Medium self-trained** (~2-5 hours each, ~5-10M steps):
- Asterix, Frostbite, Kangaroo, BankHeist

All Atari games use `NoFrameskip-v4` variant with standard Atari wrappers and `VecFrameStack(4)`.

## Architecture Decisions

### Policy mode: Linear only (freeze CNN + MLP, train `action_net`)

Same as current classical experiments. The expert's entire network (CNN features_extractor + mlp_extractor) is frozen. Only the final `action_net` linear layer is trainable. This tests whether L2 regularization (FTRL) helps the learner find a better linear mapping from rich features to actions.

### Clone-and-reset for linear policy creation

Instead of creating a new policy and copying weights (architecture-dependent), we:
1. Deep-copy the expert's full policy object
2. Freeze all parameters except `action_net`
3. Reinitialize `action_net` with Xavier uniform / zeros

This generalizes to any architecture (MLP, CNN, etc.) without knowing internals.

### Atari environment wrapping

Use SB3's `make_atari_env` + `VecFrameStack(n_stack=4)` + `RolloutInfoWrapper`. This matches the wrapper stack that HuggingFace models were trained with (frame skip, grayscale, resize to 84x84, frame stacking).

### Expert sourcing

- **Tier 1**: Download from HuggingFace via `huggingface_sb3.load_from_hub`, cache locally
- **Tier 2 & 3**: Train PPO with `CnnPolicy` on CC-server, cache locally
- **Classical**: Same as current — train PPO with `[64,64]` MLP, cache locally

## Files

### New: `src/imitation/experiments/ftrl/atari_utils.py`

Atari-specific utilities:

- `ATARI_CONFIGS`: Dict mapping game names to configs (tier, ppo_timesteps, hub_repo_id if available)
- `make_atari_env(env_name, n_envs, seed)`: Creates vectorized Atari env with standard wrappers + RolloutInfoWrapper
- `download_hub_expert(env_name, cache_dir)`: Downloads pre-trained model from HuggingFace `sb3/` org, saves to cache dir. Returns path to `.zip` file.
- `get_atari_env_id(game_name)`: Maps short name (e.g., "Pong") to full env ID ("PongNoFrameskip-v4")

### Modified: `src/imitation/experiments/ftrl/env_utils.py`

- Add `ENV_GROUPS` dict:
  - `"classical"`: current 5 + 3 new
  - `"atari-zoo"`: Tier 1 (8 games)
  - `"atari-fast"`: Tier 2 (4 games)
  - `"atari-medium"`: Tier 3 (4 games)
  - `"atari-all"`: Tier 1 + 2 + 3
  - `"all"`: classical + atari-all
- Add `is_atari(env_name)` helper
- Add `FlattenTupleObsWrapper` for Blackjack-v1
- Add Taxi-v3, Blackjack-v1, LunarLander-v3 to `ENV_CONFIGS`

### Modified: `src/imitation/experiments/ftrl/experts.py`

- `get_or_train_expert()`: Add branch — if Atari + Tier 1 game, call `download_hub_expert()`. If Atari + Tier 2/3, train PPO with `CnnPolicy` using `make_atari_env`. If classical, existing logic.
- Same caching pattern: `{cache_dir}/{env_name}/model.zip`

### Modified: `src/imitation/experiments/ftrl/policy_utils.py`

- Generalize `create_linear_policy()` to clone-and-reset approach:
  ```python
  import copy
  policy = copy.deepcopy(expert_policy)
  freeze_feature_layers(policy)
  reinitialize_action_net(policy)
  return policy
  ```
- Remove architecture-specific weight copying (no more `load_state_dict` calls)
- Keep `create_end_to_end_policy()` for backward compatibility

### Modified: `src/imitation/experiments/ftrl/run_experiment.py`

- Add `--env-group` CLI arg (resolved to env list via `ENV_GROUPS`)
- Route env creation: `is_atari()` → `make_atari_env()`, else → `make_env()`
- For Atari, enforce `linear` policy mode only

### Unchanged: `src/imitation/experiments/ftrl/plot_results.py`

Already reads generically from `experiments/results/{env_name}/`. No changes needed.

## Data Flow

```
User: python -m imitation.experiments.ftrl.run_experiment --env-group atari-zoo --algos ftl ftrl bc

For each game in atari-zoo:
  1. make_atari_env(game) → VecEnv (84x84 grayscale, 4-frame stack)
  2. get_or_train_expert(game):
     - Tier 1: download_hub_expert() → load PPO from HuggingFace
     - Tier 2/3: train PPO(CnnPolicy) → cache
  3. create_linear_policy(expert):
     - deepcopy(expert.policy)
     - freeze all except action_net
     - reinitialize action_net
  4. Run FTL/FTRL/BC rounds (same DAgger loop as classical)
  5. Save JSON → experiments/results/{env_name}/{algo}_linear_seed{N}.json

Plotting: python -m imitation.experiments.ftrl.plot_results → PNG per env
```

## Dependencies

- `huggingface_sb3` (already installed) — for downloading HuggingFace models
- `shimmy[atari]` or `ale-py` + `gymnasium[atari]` — for Atari environments
- `autorom` — for Atari ROMs (one-time setup)

## Open Questions / Future Work

- If a HuggingFace model doesn't load cleanly (version mismatch, etc.), fall back to self-training
- May need to tune PPO hyperparameters per Atari game for Tier 2/3 self-training — use SB3 Zoo's tuned configs as starting points
- If results show BC still dominates on Atari, consider option (B) from brainstorming: freeze only CNN, train MLP + action_net
