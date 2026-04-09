# Plot Metrics Enhancement & Expert Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add normalized expected return, on-policy disagreement rate, IQM plotting, expert quality validation, and remaining verification tasks from the Atari expansion.

**Architecture:** New `env_baselines.py` module for reference scores and baseline computation. Extend `run_experiment.py` with eval-interval metrics. Rewrite `plot_results.py` for 4-subplot IQM layout. Add expert quality validation tests.

**Tech Stack:** stable-baselines3, rliable, matplotlib, numpy, gymnasium

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/imitation/experiments/ftrl/env_baselines.py` | **Create** | Reference scores, baseline computation, expert quality validation |
| `src/imitation/experiments/ftrl/run_experiment.py` | **Modify** | Add `--eval-interval`, collect normalized return + disagreement rate |
| `src/imitation/experiments/ftrl/plot_results.py` | **Modify** | 4-subplot IQM layout via rliable |
| `src/imitation/experiments/ftrl/experts.py` | **Modify** | Compute and cache baselines alongside expert |
| `tests/experiments/test_env_baselines.py` | **Create** | Expert quality validation tests (fail hard) |
| `tests/experiments/test_run_experiment.py` | **Modify** | Smoke tests for new MDPs + eval interval |
| `tests/experiments/test_plot_results.py` | **Modify** | Tests for new plot layout |

---

### Task 1: Create `env_baselines.py` with reference scores and validation

**Files:**
- Create: `src/imitation/experiments/ftrl/env_baselines.py`
- Create: `tests/experiments/test_env_baselines.py`

- [ ] **Step 1: Write failing tests for reference baselines and validation**

```python
# tests/experiments/test_env_baselines.py
"""Tests for env_baselines: reference scores and expert quality validation."""

import numpy as np
import pytest

from imitation.experiments.ftrl.env_baselines import (
    REFERENCE_BASELINES,
    validate_expert_quality,
)


class TestReferenceBaselines:
    """Test that reference baselines are complete and sane."""

    def test_all_classical_envs_have_baselines(self):
        from imitation.experiments.ftrl.env_utils import ENV_CONFIGS
        for env_name in ENV_CONFIGS:
            assert env_name in REFERENCE_BASELINES, (
                f"{env_name} missing from REFERENCE_BASELINES"
            )

    def test_all_atari_envs_have_baselines(self):
        from imitation.experiments.ftrl.atari_utils import ATARI_CONFIGS
        for env_name in ATARI_CONFIGS:
            assert env_name in REFERENCE_BASELINES, (
                f"{env_name} missing from REFERENCE_BASELINES"
            )

    def test_expert_better_than_random(self):
        for env_name, bl in REFERENCE_BASELINES.items():
            assert bl["expert_score"] > bl["random_score"], (
                f"{env_name}: expert ({bl['expert_score']}) should be > "
                f"random ({bl['random_score']})"
            )


class TestValidateExpertQuality:
    """Test expert quality validation logic."""

    def test_good_expert_passes(self):
        is_ok, msg = validate_expert_quality("CartPole-v1", 490.0)
        assert is_ok, msg

    def test_bad_expert_fails(self):
        # 50.0 is way below 80% of (500 - 22) + 22 = 404.4
        is_ok, msg = validate_expert_quality("CartPole-v1", 50.0)
        assert not is_ok
        assert "below" in msg.lower() or "threshold" in msg.lower()

    def test_unknown_env_passes(self):
        # Unknown envs should pass (no reference to validate against)
        is_ok, msg = validate_expert_quality("UnknownEnv-v0", 100.0)
        assert is_ok
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/experiments/test_env_baselines.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create `env_baselines.py`**

```python
# src/imitation/experiments/ftrl/env_baselines.py
"""Reference baseline scores and expert quality validation.

Provides hardcoded reference scores (random and expert) for all supported
environments, used for normalized return computation and expert quality
validation.
"""

import json
import logging
import pathlib
from typing import Dict, Optional, Tuple

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)

# Reference baseline scores per environment.
# expert_score: expected return of a well-trained PPO expert.
# random_score: expected return of a uniform random policy.
# Sources: DQN Zoo (Atari random/human), CleanRL PPO benchmarks (Atari expert),
# Gymnasium docs and empirical measurement (classical MDPs).
REFERENCE_BASELINES: Dict[str, dict] = {
    # Classical MDPs
    "CartPole-v1": {"random_score": 22.0, "expert_score": 500.0},
    "FrozenLake-v1": {"random_score": 0.015, "expert_score": 1.0},
    "CliffWalking-v0": {"random_score": -56957.0, "expert_score": -13.0},
    "Acrobot-v1": {"random_score": -499.0, "expert_score": -85.0},
    "MountainCar-v0": {"random_score": -200.0, "expert_score": -110.0},
    "Taxi-v3": {"random_score": -763.0, "expert_score": 7.9},
    "Blackjack-v1": {"random_score": -0.40, "expert_score": -0.06},
    "LunarLander-v2": {"random_score": -176.0, "expert_score": 250.0},
    # Atari Tier 1 (HuggingFace model zoo)
    "PongNoFrameskip-v4": {"random_score": -20.7, "expert_score": 20.5},
    "BreakoutNoFrameskip-v4": {"random_score": 1.7, "expert_score": 405.7},
    "SpaceInvadersNoFrameskip-v4": {"random_score": 148.0, "expert_score": 1019.8},
    "BeamRiderNoFrameskip-v4": {"random_score": 363.9, "expert_score": 2835.7},
    "QbertNoFrameskip-v4": {"random_score": 163.9, "expert_score": 15228.3},
    "MsPacmanNoFrameskip-v4": {"random_score": 307.3, "expert_score": 2152.8},
    "EnduroNoFrameskip-v4": {"random_score": 0.0, "expert_score": 986.7},
    "SeaquestNoFrameskip-v4": {"random_score": 68.4, "expert_score": 1518.3},
    # Atari Tier 2 (fast self-trained)
    "FreewayNoFrameskip-v4": {"random_score": 0.0, "expert_score": 33.0},
    "AtlantisNoFrameskip-v4": {"random_score": 12850.0, "expert_score": 2036749.0},
    "DemonAttackNoFrameskip-v4": {"random_score": 152.1, "expert_score": 13788.4},
    "CrazyClimberNoFrameskip-v4": {"random_score": 10780.5, "expert_score": 119344.7},
    # Atari Tier 3 (medium self-trained)
    "AsterixNoFrameskip-v4": {"random_score": 210.0, "expert_score": 3738.5},
    "FrostbiteNoFrameskip-v4": {"random_score": 65.2, "expert_score": 933.6},
    "KangarooNoFrameskip-v4": {"random_score": 52.0, "expert_score": 5325.3},
    "BankHeistNoFrameskip-v4": {"random_score": 14.2, "expert_score": 1213.5},
}

# Expert must achieve at least this fraction of the (expert - random) range
# above random to be considered valid.
EXPERT_QUALITY_THRESHOLD = 0.80


def validate_expert_quality(
    env_name: str,
    measured_return: float,
) -> Tuple[bool, str]:
    """Validate that a trained expert achieves acceptable quality.

    Checks that the measured return is at least EXPERT_QUALITY_THRESHOLD
    of the way from random to reference expert score.

    Args:
        env_name: Environment name.
        measured_return: Mean return achieved by the expert.

    Returns:
        Tuple of (is_valid, message). is_valid is True if the expert
        meets the quality threshold or if no reference exists.
    """
    if env_name not in REFERENCE_BASELINES:
        return True, f"No reference baseline for {env_name}, skipping validation"

    ref = REFERENCE_BASELINES[env_name]
    score_range = ref["expert_score"] - ref["random_score"]
    threshold = ref["random_score"] + EXPERT_QUALITY_THRESHOLD * score_range

    if measured_return >= threshold:
        return True, (
            f"Expert quality OK for {env_name}: {measured_return:.1f} >= "
            f"{threshold:.1f} (threshold={EXPERT_QUALITY_THRESHOLD:.0%} of "
            f"[{ref['random_score']}, {ref['expert_score']}])"
        )
    else:
        return False, (
            f"Expert quality BELOW threshold for {env_name}: "
            f"{measured_return:.1f} < {threshold:.1f} "
            f"(threshold={EXPERT_QUALITY_THRESHOLD:.0%} of "
            f"[{ref['random_score']}, {ref['expert_score']}]). "
            f"Consider increasing PPO training steps."
        )


def compute_random_return(
    venv: VecEnv,
    n_episodes: int = 100,
) -> float:
    """Compute mean return of a uniform random policy.

    Args:
        venv: Vectorized environment (1 env).
        n_episodes: Number of episodes to average over.

    Returns:
        Mean undiscounted episode return.
    """
    returns = []
    obs = venv.reset()
    episode_return = 0.0
    while len(returns) < n_episodes:
        action = np.array([venv.action_space.sample()])
        obs, rewards, dones, infos = venv.step(action)
        episode_return += rewards[0]
        if dones[0]:
            returns.append(episode_return)
            episode_return = 0.0
    return float(np.mean(returns))


def compute_baselines(
    expert_policy: BasePolicy,
    venv: VecEnv,
    rng: np.random.Generator,
    n_expert_episodes: int = 20,
    n_random_episodes: int = 100,
) -> Dict[str, float]:
    """Compute expert and random baseline returns.

    Args:
        expert_policy: Trained expert policy.
        venv: Vectorized environment.
        rng: Random state.
        n_expert_episodes: Episodes for expert evaluation.
        n_random_episodes: Episodes for random evaluation.

    Returns:
        Dict with 'expert_return' and 'random_return' keys.
    """
    expert_return, _ = evaluate_policy(
        expert_policy, venv, n_eval_episodes=n_expert_episodes,
        deterministic=True,
    )
    random_return = compute_random_return(venv, n_episodes=n_random_episodes)

    logger.info(
        f"Baselines: expert={expert_return:.1f}, random={random_return:.1f}"
    )
    return {
        "expert_return": float(expert_return),
        "random_return": float(random_return),
    }


def load_or_compute_baselines(
    env_name: str,
    venv: VecEnv,
    expert_policy: BasePolicy,
    cache_dir: pathlib.Path,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Load cached baselines or compute and cache them.

    Args:
        env_name: Environment name.
        venv: Vectorized environment.
        expert_policy: Trained expert policy.
        cache_dir: Directory for caching (same as expert cache).
        rng: Random state.

    Returns:
        Dict with 'expert_return' and 'random_return'.
    """
    cache_path = cache_dir / env_name.replace("/", "_")
    baselines_file = cache_path / "baselines.json"

    if baselines_file.exists():
        with open(baselines_file) as f:
            baselines = json.load(f)
        logger.info(f"Loaded cached baselines from {baselines_file}")
        return baselines

    baselines = compute_baselines(expert_policy, venv, rng)

    cache_path.mkdir(parents=True, exist_ok=True)
    with open(baselines_file, "w") as f:
        json.dump(baselines, f, indent=2)
    logger.info(f"Cached baselines to {baselines_file}")

    return baselines
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/experiments/test_env_baselines.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/imitation/experiments/ftrl/env_baselines.py tests/experiments/test_env_baselines.py
git commit -m "feat: add env_baselines with reference scores and expert validation"
```

---

### Task 2: Add expert quality validation test (fail hard)

**Files:**
- Modify: `tests/experiments/test_env_baselines.py`

This test actually trains experts for classical MDPs and validates their quality against reference scores. Marked `@pytest.mark.expensive`.

- [ ] **Step 1: Write the expert quality validation test**

Append to `tests/experiments/test_env_baselines.py`:

```python
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.experiments.ftrl.env_baselines import validate_expert_quality
from imitation.experiments.ftrl.env_utils import ENV_CONFIGS, make_env
from imitation.experiments.ftrl.experts import get_or_train_expert


@pytest.mark.expensive
class TestExpertQualityHardFail:
    """Fail hard if any trained expert is degenerate.

    Catches issues like the MountainCar incident where PPO training
    steps were insufficient and the expert performed at random level.
    """

    @pytest.mark.parametrize("env_name", list(ENV_CONFIGS.keys()))
    def test_expert_meets_quality_threshold(self, env_name, tmp_path):
        """Train/load expert and validate against reference baseline."""
        rng = np.random.default_rng(42)
        venv = make_env(env_name, n_envs=1, rng=rng)
        expert = get_or_train_expert(
            env_name, venv, cache_dir=tmp_path, rng=rng, seed=42,
        )
        mean_return, _ = evaluate_policy(
            expert, venv, n_eval_episodes=20, deterministic=True,
        )
        venv.close()

        is_ok, msg = validate_expert_quality(env_name, mean_return)
        assert is_ok, msg
```

- [ ] **Step 2: Run test (expensive, takes minutes)**

Run: `pytest tests/experiments/test_env_baselines.py::TestExpertQualityHardFail::test_expert_meets_quality_threshold[CartPole-v1] -v`
Expected: PASS (CartPole is fast and reliable)

- [ ] **Step 3: Commit**

```bash
git add tests/experiments/test_env_baselines.py
git commit -m "test: add hard-fail expert quality validation for all classical MDPs"
```

---

### Task 3: Add eval metrics collection to `run_experiment.py`

**Files:**
- Modify: `src/imitation/experiments/ftrl/run_experiment.py`

- [ ] **Step 1: Add `eval_interval` to `ExperimentConfig`**

Add to the `ExperimentConfig` dataclass:

```python
    eval_interval: int  # evaluate return/disagreement every N rounds
```

- [ ] **Step 2: Create `_evaluate_learner_metrics` helper function**

Add this function to `run_experiment.py`:

```python
def _evaluate_learner_metrics(
    learner_policy: sb3_policies.ActorCriticPolicy,
    expert_policy: sb3_policies.ActorCriticPolicy,
    venv: VecEnv,
    baselines: Dict[str, float],
    n_episodes: int = 10,
) -> Dict[str, float]:
    """Evaluate normalized return and on-policy disagreement rate.

    Rolls out the learner policy, computing both metrics simultaneously.

    Args:
        learner_policy: The learner's current policy.
        expert_policy: The expert policy (for disagreement comparison).
        venv: Vectorized environment.
        baselines: Dict with 'expert_return' and 'random_return'.
        n_episodes: Number of episodes to roll out.

    Returns:
        Dict with 'normalized_return' and 'disagreement_rate'.
    """
    import torch as th

    learner_policy.eval()
    expert_policy.eval()

    episode_returns = []
    total_steps = 0
    total_disagreements = 0
    current_return = 0.0

    obs = venv.reset()
    while len(episode_returns) < n_episodes:
        # Get actions from both policies
        with th.no_grad():
            obs_tensor = th.as_tensor(obs, dtype=th.float32)
            learner_action = learner_policy.predict(obs, deterministic=True)[0]
            expert_action = expert_policy.predict(obs, deterministic=True)[0]

        # Count disagreements
        total_steps += 1
        if learner_action[0] != expert_action[0]:
            total_disagreements += 1

        # Step with learner action
        obs, rewards, dones, infos = venv.step(learner_action)
        current_return += rewards[0]

        if dones[0]:
            episode_returns.append(current_return)
            current_return = 0.0

    learner_policy.train()

    mean_return = float(np.mean(episode_returns))
    expert_ret = baselines["expert_return"]
    random_ret = baselines["random_return"]
    score_range = expert_ret - random_ret

    if abs(score_range) < 1e-8:
        normalized_return = 0.0
    else:
        normalized_return = (mean_return - random_ret) / score_range

    disagreement_rate = total_disagreements / max(total_steps, 1)

    return {
        "normalized_return": round(normalized_return, 6),
        "disagreement_rate": round(disagreement_rate, 6),
    }
```

- [ ] **Step 3: Add `--eval-interval` CLI arg**

In `main()` parser, add:

```python
    parser.add_argument("--eval-interval", type=int, default=5,
                        help="Evaluate return/disagreement every N rounds")
```

Update `build_configs` to pass it through.

- [ ] **Step 4: Update `_run_dagger_variant` to collect eval metrics**

After the training loop that extracts per-round metrics, add eval at intervals. Change the loop to:

```python
    per_round = []
    cum_obs = 0
    for m in trainer.get_metrics():
        demo_round = m.round_num - 1
        round_dir = trainer._demo_dir_path_for_round(demo_round)
        demo_paths = trainer._get_demo_paths(round_dir)
        round_demos = []
        for p in demo_paths:
            round_demos.extend(serialize.load(p))
        round_transitions = rollout.flatten_trajectories(round_demos)
        expert_ce = _evaluate_policy_cross_entropy(expert_policy, round_transitions)
        cum_obs += len(round_transitions)

        round_data = {
            "round": m.round_num,
            "n_observations": cum_obs,
            "cross_entropy": round(m.cross_entropy, 6),
            "l2_norm": round(m.l2_norm, 6),
            "total_loss": round(m.total_loss, 6),
            "expert_cross_entropy": round(expert_ce, 6),
            "normalized_return": None,
            "disagreement_rate": None,
        }

        # Evaluate at intervals: round 1, every eval_interval, and final round
        is_first = m.round_num == 1
        is_interval = m.round_num % config.eval_interval == 0
        is_final = m.round_num == len(trainer.get_metrics())
        if is_first or is_interval or is_final:
            eval_metrics = _evaluate_learner_metrics(
                bc_trainer.policy, expert_policy, venv, baselines,
            )
            round_data.update(eval_metrics)

        per_round.append(round_data)
```

Note: `baselines` must be loaded before the training loop. Add at the top of `_run_dagger_variant`:

```python
    from imitation.experiments.ftrl.env_baselines import load_or_compute_baselines
    baselines = load_or_compute_baselines(
        config.env_name, venv, expert_policy, config.expert_cache_dir, rng,
    )
```

- [ ] **Step 5: Update `_run_bc` similarly**

Add the same eval logic to `_run_bc`. Load baselines, then at eval intervals compute normalized_return and disagreement_rate.

- [ ] **Step 6: Add baselines to result JSON**

In `run_single`, after loading baselines, add to result dict:

```python
    result["baselines"] = baselines
```

This requires loading baselines in `run_single` before dispatching to `_run_dagger_variant` or `_run_bc`, then passing them as a parameter.

- [ ] **Step 7: Add expert quality warning in `run_single`**

After loading baselines, add the warning check:

```python
    from imitation.experiments.ftrl.env_baselines import validate_expert_quality
    is_ok, msg = validate_expert_quality(
        config.env_name, baselines["expert_return"],
    )
    if not is_ok:
        logger.warning(f"WARNING: {msg}")
```

- [ ] **Step 8: Run existing tests**

Run: `pytest tests/experiments/test_run_experiment.py -v`
Expected: All tests PASS (existing tests don't use eval metrics)

- [ ] **Step 9: Commit**

```bash
git add src/imitation/experiments/ftrl/run_experiment.py
git commit -m "feat: add eval-interval metrics (normalized return, disagreement rate)"
```

---

### Task 4: Rewrite `plot_results.py` with 4-subplot IQM layout

**Files:**
- Modify: `src/imitation/experiments/ftrl/plot_results.py`
- Modify: `tests/experiments/test_plot_results.py`

- [ ] **Step 1: Install rliable**

Run:
```bash
pip install rliable
```

- [ ] **Step 2: Write test for new plot layout**

Read existing `tests/experiments/test_plot_results.py` first. Then update/add tests:

```python
def test_plot_has_four_subplots(tmp_path):
    """Verify plot generates 4 subplots."""
    # Create minimal result JSON with all metrics
    result = {
        "algo": "ftl", "env": "CartPole-v1", "seed": 0,
        "policy_mode": "linear",
        "baselines": {"expert_return": 500.0, "random_return": 22.0},
        "per_round": [
            {
                "round": 1, "n_observations": 500,
                "cross_entropy": 0.5, "l2_norm": 1.0,
                "total_loss": 0.5, "expert_cross_entropy": 0.01,
                "normalized_return": 0.1, "disagreement_rate": 0.4,
            },
            {
                "round": 2, "n_observations": 1000,
                "cross_entropy": 0.3, "l2_norm": 0.8,
                "total_loss": 0.3, "expert_cross_entropy": 0.01,
                "normalized_return": None, "disagreement_rate": None,
            },
        ],
    }
    results_dir = tmp_path / "results" / "CartPole-v1"
    results_dir.mkdir(parents=True)
    with open(results_dir / "ftl_linear_seed0.json", "w") as f:
        json.dump(result, f)

    from imitation.experiments.ftrl.plot_results import plot_all
    paths = plot_all(tmp_path / "results", tmp_path / "plots")
    assert len(paths) == 1
    assert paths[0].exists()
```

- [ ] **Step 3: Rewrite `plot_results.py`**

Key changes to `plot_results.py`:

1. **Replace `_plot_metric` with IQM-based plotting** using `rliable`:

```python
def _compute_iqm_ci(values_per_seed: np.ndarray) -> Tuple[float, float, float]:
    """Compute IQM and 95% stratified bootstrap CI.

    Args:
        values_per_seed: Array of shape (n_seeds,).

    Returns:
        Tuple of (iqm, ci_low, ci_high).
    """
    from rliable import library as rly
    from rliable import metrics as rly_metrics

    # rliable expects shape (n_runs, n_tasks) — we have 1 task
    data = values_per_seed.reshape(-1, 1)
    iqm = lambda x: rly_metrics.aggregate_iqm(x)
    iqm_scores, iqm_cis = rly.get_interval_estimates(
        {"algo": data}, iqm, reps=2000,
    )
    return (
        float(iqm_scores["algo"]),
        float(iqm_cis["algo"][0]),
        float(iqm_cis["algo"][1]),
    )
```

2. **Update `_plot_metric` to use IQM + CI bands**:

Instead of mean +/- std, compute IQM at each round and plot with bootstrap CI.

3. **Change plot layout to 4 subplots**:

```python
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 13), sharex=True)
```

- Subplot 1: "Per-Round Imitation Loss" (log scale y)
- Subplot 2: "Normalized Expected Return" (0-1 scale)
- Subplot 3: "On-Policy Disagreement Rate" (0-1 scale)
- Subplot 4: "Cumulative Regret (vs Expert)"

4. **Handle null eval metrics**: For subplots 2 and 3, only plot at rounds where values are not None. Connect points with lines, skip nulls.

5. **Rename labels**: "cross_entropy" → "imitation_loss" in axis labels (column name stays same for backward compat).

6. **Load baselines from result JSON** for potential use in annotations.

- [ ] **Step 4: Run tests**

Run: `pytest tests/experiments/test_plot_results.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/imitation/experiments/ftrl/plot_results.py tests/experiments/test_plot_results.py
git commit -m "feat: 4-subplot IQM plot layout with imitation loss, return, disagreement, regret"
```

---

### Task 5: Integrate baseline computation into `experts.py`

**Files:**
- Modify: `src/imitation/experiments/ftrl/experts.py`

- [ ] **Step 1: Add baseline computation after expert training/loading**

In both `_get_atari_expert` and `_train_classical_expert`, after evaluating expert quality, also compute and cache baselines. Add after the `evaluate_policy` call in each function:

```python
    from .env_baselines import load_or_compute_baselines, validate_expert_quality

    baselines = load_or_compute_baselines(
        env_name, venv, model.policy, cache_dir, rng,
    )

    # Warn if expert quality is below reference
    is_ok, msg = validate_expert_quality(env_name, mean_reward)
    if not is_ok:
        logger.warning(f"WARNING: {msg}")
```

Note: `_get_atari_expert` doesn't have `rng` — pass `np.random.default_rng(seed)` instead.

- [ ] **Step 2: Run existing tests**

Run: `pytest tests/experiments/test_run_experiment.py tests/experiments/test_atari_utils.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/imitation/experiments/ftrl/experts.py
git commit -m "feat: compute and cache baselines alongside expert training"
```

---

### Task 6: Smoke tests for new classical MDPs (from original plan Task 7)

**Files:**
- Modify: `tests/experiments/test_run_experiment.py`

- [ ] **Step 1: Add smoke tests for LunarLander and Taxi**

Append to `tests/experiments/test_run_experiment.py`:

```python
def test_run_ftrl_lunarlander(tmp_path):
    """FTRL smoke test on LunarLander-v2 (new classical MDP)."""
    config = _make_config(
        "ftrl", tmp_path,
        env_name="LunarLander-v2",
        policy_mode="linear",
        n_rounds=2,
        samples_per_round=200,
    )
    result = run_single(config)
    assert result["algo"] == "ftrl"
    assert result["env"] == "LunarLander-v2"
    assert len(result["per_round"]) >= 1


def test_run_bc_taxi(tmp_path):
    """BC smoke test on Taxi-v3 (discrete obs, one-hot encoded)."""
    config = _make_config(
        "bc", tmp_path,
        env_name="Taxi-v3",
        policy_mode="linear",
        n_rounds=2,
        samples_per_round=200,
    )
    result = run_single(config)
    assert result["algo"] == "bc"
    assert result["env"] == "Taxi-v3"
    assert len(result["per_round"]) >= 1
```

Note: `_make_config` needs to include `eval_interval` now. Update the helper:

```python
def _make_config(algo, tmp_path, **overrides):
    defaults = dict(
        algo=algo,
        env_name="CartPole-v1",
        seed=0,
        policy_mode="end_to_end",
        n_rounds=3,
        samples_per_round=300,
        l2_lambda=1e-4,
        l2_decay=False,
        warm_start=True,
        beta_rampdown=2,
        bc_n_epochs=2,
        eval_interval=5,  # NEW
        output_dir=tmp_path / "results",
        expert_cache_dir=tmp_path / "experts",
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/experiments/test_run_experiment.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/experiments/test_run_experiment.py
git commit -m "test: add smoke tests for LunarLander and Taxi"
```

---

### Task 7: Verify Atari dependencies + integration test (from original plan Tasks 8-9)

**Files:**
- Modify: `tests/experiments/test_atari_utils.py`

- [ ] **Step 1: Check Atari dependencies**

Run:
```bash
pip install "gymnasium[atari]" ale-py autorom rliable
python -c "import ale_py; print('ale_py:', ale_py.__version__)"
python -c "from rliable import library; print('rliable OK')"
```

If ROMs aren't installed: `AutoROM --accept-license`

- [ ] **Step 2: Add Atari integration test (marked expensive)**

Append to `tests/experiments/test_atari_utils.py`:

```python
@pytest.mark.expensive
class TestAtariIntegration:
    """Integration tests that require Atari ROMs and network access."""

    def test_download_and_load_pong_expert(self, tmp_path):
        """Download Pong expert from HuggingFace and verify it loads."""
        from stable_baselines3 import PPO
        from imitation.experiments.ftrl.atari_utils import (
            download_hub_expert,
            make_atari_venv,
        )

        model_path = download_hub_expert("PongNoFrameskip-v4", tmp_path)
        assert model_path.exists()

        venv = make_atari_venv("PongNoFrameskip-v4", n_envs=1, seed=0)
        model = PPO.load(model_path, env=venv)
        obs = venv.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1,)
        venv.close()

    def test_pong_linear_policy_creation(self, tmp_path):
        """Create a linear policy from Pong expert using clone-and-reset."""
        from stable_baselines3 import PPO
        from imitation.experiments.ftrl.atari_utils import (
            download_hub_expert,
            make_atari_venv,
        )
        from imitation.experiments.ftrl.policy_utils import create_linear_policy

        model_path = download_hub_expert("PongNoFrameskip-v4", tmp_path)
        venv = make_atari_venv("PongNoFrameskip-v4", n_envs=1, seed=0)
        model = PPO.load(model_path, env=venv)

        linear_policy = create_linear_policy(model.policy)

        trainable = [
            name for name, p in linear_policy.named_parameters()
            if p.requires_grad
        ]
        frozen = [
            name for name, p in linear_policy.named_parameters()
            if not p.requires_grad
        ]
        assert all("action_net" in n for n in trainable)
        assert len(frozen) > 0
        assert not any("action_net" in n for n in frozen)
        venv.close()
```

- [ ] **Step 3: Commit**

```bash
git add tests/experiments/test_atari_utils.py
git commit -m "test: add Atari integration tests (download + linear policy)"
```

---

### Task 8: Final verification, lint, and push

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite (non-expensive)**

Run: `pytest tests/experiments/ -v -m "not expensive"`
Expected: All tests PASS

- [ ] **Step 2: Run linting**

Run:
```bash
black src/imitation/experiments/ftrl/ tests/experiments/
isort src/imitation/experiments/ftrl/ tests/experiments/
flake8 src/imitation/experiments/ftrl/ tests/experiments/
```

- [ ] **Step 3: Fix lint issues and commit**

```bash
git add -u
git commit -m "style: fix lint issues"
```

- [ ] **Step 4: Push branch**

```bash
git push -u origin feature/atari-expanded-mdps
```
