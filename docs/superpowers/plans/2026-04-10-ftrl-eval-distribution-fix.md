# FTRL Evaluation Distribution Fix & BC+DAgger Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify policy evaluation across FTL/FTRL/BC/Expert, add a BC+DAgger baseline whose data budget matches DAgger and whose loss is evaluated on an aggregated eval buffer, fix the Expert<BC bug via convergence-detecting expert training, and pin all invariants with regression tests.

**Architecture:** A single `eval_utils.eval_policy_rollout()` replaces every ad-hoc rollout loop. Two parallel buffers are maintained per dynamic run — `D_train^t` (algo-specific) and `D_eval^t` (aggregated fresh rollouts, shared definition) — and the plotted loss is the current policy's sampled-action CE on the entire aggregated `D_eval^t`. A chunked PPO trainer with return-plateau + residual-softmax-entropy stopping replaces the fixed-step classical expert trainer and raises on non-convergence.

**Tech Stack:** Python 3.10+, PyTorch, stable-baselines3 ~= 2.2.1, gymnasium ~= 0.29, pytest, numpy, matplotlib.

**Spec:** `docs/superpowers/specs/2026-04-10-ftrl-eval-distribution-fix.md`

---

## File Structure

**New files:**
- `src/imitation/experiments/ftrl/eval_utils.py` — `RolloutBatch`, `EvalResult`, `eval_policy_rollout`, `compute_sampled_action_ce`. Single policy-eval entry point for all call sites.
- `src/imitation/experiments/ftrl/expert_training.py` — `train_classical_expert_until_converged()` + `DEFAULT_CONVERGENCE` + `get_convergence_config()`.
- `tests/experiments/test_eval_utils.py` — SB3 equivalence + safety_step_cap tests.
- `tests/experiments/test_expert_quality.py` — `test_expert_converged`, `test_expert_beats_bc` (both `@pytest.mark.expensive`).

**Modified files:**
- `src/imitation/experiments/ftrl/env_utils.py` — add `convergence` sub-dicts for `MountainCar-v0`, `Taxi-v3`, `Blackjack-v1`.
- `src/imitation/experiments/ftrl/env_baselines.py` — route `compute_baselines`'s expert return through `eval_utils.eval_policy_rollout`.
- `src/imitation/experiments/ftrl/experts.py` — `_train_classical_expert` delegates to `expert_training.train_classical_expert_until_converged`.
- `src/imitation/experiments/ftrl/run_experiment.py` — delete `_evaluate_learner_metrics` & `_evaluate_policy_cross_entropy`, add `_run_bc_dagger`, extend `ALL_ALGOS = ["ftl", "ftrl", "bc", "bc_dagger"]`, add `D_eval^t` aggregation to `_run_dagger_variant`, rename JSON fields.
- `src/imitation/experiments/ftrl/plot_results.py` — new `ALGO_COLORS`/`ALGO_LABELS`/`ALGO_LINESTYLES` tables, loss metric switch to `rollout_cross_entropy`, `--show-expert-on-loss` flag, subtitle with sampled-action CE formula.

**Untouched:** `src/imitation/algorithms/ftrl.py`, Atari pipeline.

---

## Task 1: Create `eval_utils.py` with failing test for SB3 equivalence

**Files:**
- Create: `src/imitation/experiments/ftrl/eval_utils.py`
- Test: `tests/experiments/test_eval_utils.py`

- [ ] **Step 1: Write the failing equivalence test**

Create `tests/experiments/test_eval_utils.py`:

```python
"""Tests for unified policy evaluation in eval_utils."""

import numpy as np
import pytest
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.experiments.ftrl import env_utils, experts
from imitation.experiments.ftrl.eval_utils import (
    EvalResult,
    RolloutBatch,
    compute_sampled_action_ce,
    eval_policy_rollout,
)


def test_eval_matches_sb3_evaluate_policy_cartpole(tmp_path):
    """eval_policy_rollout must match SB3's evaluate_policy on CartPole."""
    rng = np.random.default_rng(0)
    venv = env_utils.make_env("CartPole-v1", n_envs=1, rng=rng)
    expert = experts.get_or_train_expert(
        "CartPole-v1", venv, cache_dir=tmp_path, rng=rng, seed=0
    )

    venv.reset()
    ours = eval_policy_rollout(expert, venv, n_episodes=20, deterministic=True)
    venv.reset()
    sb3_mean, _ = evaluate_policy(
        expert, venv, n_eval_episodes=20, deterministic=True
    )

    assert isinstance(ours, EvalResult)
    assert len(ours.episode_returns) == 20
    assert abs(ours.mean_return - sb3_mean) < 1e-4


def test_eval_raises_on_safety_cap():
    """Hitting safety_step_cap must raise RuntimeError, not silently truncate."""
    rng = np.random.default_rng(0)
    venv = env_utils.make_env("CartPole-v1", n_envs=1, rng=rng)

    class NullPolicy:
        def predict(self, obs, deterministic=True):
            # Action 0 keeps CartPole running for a while but hits cap
            return np.zeros(len(obs), dtype=np.int64), None

    with pytest.raises(RuntimeError, match="safety_step_cap"):
        eval_policy_rollout(
            NullPolicy(), venv, n_episodes=1000, safety_step_cap=50
        )


def test_rollout_batch_expert_actions_are_argmax(tmp_path):
    """When expert_policy is provided, rollout_batch.expert_actions must be
    the expert's deterministic argmax at every visited state."""
    rng = np.random.default_rng(0)
    venv = env_utils.make_env("CartPole-v1", n_envs=1, rng=rng)
    expert = experts.get_or_train_expert(
        "CartPole-v1", venv, cache_dir=tmp_path, rng=rng, seed=0
    )

    res = eval_policy_rollout(
        expert, venv, n_episodes=5, deterministic=True, expert_policy=expert
    )
    assert res.rollout_batch is not None
    # When learner == expert, disagreement must be zero.
    assert res.current_round_disagreement == 0.0
    # Shapes line up
    assert res.rollout_batch.obs.shape[0] == res.rollout_batch.expert_actions.shape[0]
    assert res.rollout_batch.obs.shape[0] == res.n_steps


def test_compute_sampled_action_ce_matches_bc_loss_calculator(tmp_path):
    """compute_sampled_action_ce must match bc.BehaviorCloningLossCalculator's
    neglogp on the same (obs, acts)."""
    import torch as th
    from imitation.algorithms.bc import BehaviorCloningLossCalculator

    rng = np.random.default_rng(0)
    venv = env_utils.make_env("CartPole-v1", n_envs=1, rng=rng)
    expert = experts.get_or_train_expert(
        "CartPole-v1", venv, cache_dir=tmp_path, rng=rng, seed=0
    )

    res = eval_policy_rollout(
        expert, venv, n_episodes=3, deterministic=True, expert_policy=expert
    )
    obs = res.rollout_batch.obs
    acts = res.rollout_batch.expert_actions

    ours = compute_sampled_action_ce(expert, obs, acts)

    calc = BehaviorCloningLossCalculator(ent_weight=0.0, l2_weight=0.0)
    with th.no_grad():
        metrics = calc(expert, obs, acts)
    theirs = metrics.neglogp.item()

    assert abs(ours - theirs) < 1e-4
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest tests/experiments/test_eval_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imitation.experiments.ftrl.eval_utils'`

- [ ] **Step 3: Implement `eval_utils.py`**

Create `src/imitation/experiments/ftrl/eval_utils.py`:

```python
"""Unified policy evaluation for FTRL experiments.

Single entry point for rolling out a policy, measuring its return, and
(optionally) collecting expert argmax actions at every visited state. The
caller is responsible for aggregating rollout batches across eval points
into a running D_eval buffer — see the spec
docs/superpowers/specs/2026-04-10-ftrl-eval-distribution-fix.md §3.4.
"""

import dataclasses
from typing import List, Optional

import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

from imitation.data import types
from imitation.util import util


@dataclasses.dataclass
class RolloutBatch:
    """Raw transitions collected during one eval call.

    Owned by the caller; aggregated across eval points into D_eval^t.
    """

    obs: np.ndarray  # shape (n_steps, *obs_shape)
    expert_actions: np.ndarray  # shape (n_steps,) — a*(s) = argmax π*(·|s)
    learner_actions: np.ndarray  # shape (n_steps,) — deterministic learner action


@dataclasses.dataclass
class EvalResult:
    """Return value from `eval_policy_rollout`."""

    mean_return: float
    episode_returns: List[float]
    n_steps: int
    rollout_batch: Optional[RolloutBatch]
    current_round_disagreement: Optional[float]
    current_round_ce: Optional[float]


def eval_policy_rollout(
    policy: BasePolicy,
    venv: VecEnv,
    n_episodes: int,
    deterministic: bool = True,
    expert_policy: Optional[BasePolicy] = None,
    safety_step_cap: int = 100_000,
) -> EvalResult:
    """Roll out `policy` for exactly `n_episodes` complete episodes.

    Args:
        policy: The policy to evaluate.
        venv: Single-env VecEnv (num_envs must be 1).
        n_episodes: Exact number of complete episodes to collect.
        deterministic: Pass to policy.predict.
        expert_policy: If provided, query at every step and store
            the expert's deterministic action in the returned RolloutBatch.
        safety_step_cap: Hard ceiling on environment steps. Exceeding
            raises RuntimeError.

    Returns:
        EvalResult with mean_return computed from Monitor episode_returns
        (matching SB3's evaluate_policy) when available.

    Raises:
        RuntimeError: If safety_step_cap is hit before n_episodes complete.
    """
    if venv.num_envs != 1:
        raise ValueError(
            f"eval_policy_rollout requires num_envs=1, got {venv.num_envs}"
        )

    obs_buf: List[np.ndarray] = []
    expert_act_buf: List[int] = []
    learner_act_buf: List[int] = []
    episode_returns: List[float] = []

    total_steps = 0
    current_return = 0.0

    obs = venv.reset()
    while len(episode_returns) < n_episodes:
        if total_steps >= safety_step_cap:
            raise RuntimeError(
                f"eval_policy_rollout hit safety_step_cap={safety_step_cap} "
                f"after collecting {len(episode_returns)}/{n_episodes} "
                f"episodes. Env may be non-terminating."
            )

        learner_action, _ = policy.predict(obs, deterministic=deterministic)
        if expert_policy is not None:
            expert_action, _ = expert_policy.predict(obs, deterministic=True)
            obs_buf.append(np.asarray(obs[0]))
            expert_act_buf.append(int(expert_action[0]))
            learner_act_buf.append(int(learner_action[0]))

        obs, rewards, dones, infos = venv.step(learner_action)
        total_steps += 1
        current_return += float(rewards[0])

        if dones[0]:
            info = infos[0]
            if "episode" in info and "r" in info["episode"]:
                episode_returns.append(float(info["episode"]["r"]))
            else:
                episode_returns.append(current_return)
            current_return = 0.0

    rollout_batch: Optional[RolloutBatch] = None
    current_round_disagreement: Optional[float] = None
    current_round_ce: Optional[float] = None
    if expert_policy is not None:
        obs_arr = np.stack(obs_buf, axis=0)
        expert_arr = np.asarray(expert_act_buf, dtype=np.int64)
        learner_arr = np.asarray(learner_act_buf, dtype=np.int64)
        rollout_batch = RolloutBatch(
            obs=obs_arr,
            expert_actions=expert_arr,
            learner_actions=learner_arr,
        )
        current_round_disagreement = float(
            np.mean(expert_arr != learner_arr)
        )
        current_round_ce = compute_sampled_action_ce(
            policy, obs_arr, expert_arr
        )

    return EvalResult(
        mean_return=float(np.mean(episode_returns)),
        episode_returns=episode_returns,
        n_steps=total_steps,
        rollout_batch=rollout_batch,
        current_round_disagreement=current_round_disagreement,
        current_round_ce=current_round_ce,
    )


def compute_sampled_action_ce(
    policy: BasePolicy,
    obs: np.ndarray,
    expert_actions: np.ndarray,
    batch_size: int = 2048,
) -> float:
    """Compute -1/|D| * sum log π(a*|s) on an arbitrary aggregated buffer.

    Single batched forward pass, no autograd. Uses policy.evaluate_actions
    to exactly match the loss form minimized by
    `imitation.algorithms.bc.BehaviorCloningLossCalculator`.

    Args:
        policy: Policy with `.evaluate_actions(obs, acts)`.
        obs: Aggregated state buffer, shape (N, *obs_shape).
        expert_actions: Aggregated expert argmax actions, shape (N,).
        batch_size: Mini-batch size for the forward pass.

    Returns:
        Mean sampled-action cross-entropy (scalar float).
    """
    was_training = policy.training
    policy.eval()
    device = policy.device
    total = 0.0
    n = int(obs.shape[0])
    try:
        with th.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_obs = obs[start:end]
                batch_acts = expert_actions[start:end]
                tensor_obs = types.map_maybe_dict(
                    lambda x: util.safe_to_tensor(x).to(device),
                    types.maybe_unwrap_dictobs(batch_obs),
                )
                tensor_acts = util.safe_to_tensor(batch_acts).to(device)
                _, log_prob, _ = policy.evaluate_actions(tensor_obs, tensor_acts)
                total += float(-log_prob.sum().item())
    finally:
        if was_training:
            policy.train()
    return total / max(n, 1)
```

- [ ] **Step 4: Run the equivalence test to verify it passes**

Run: `pytest tests/experiments/test_eval_utils.py::test_eval_matches_sb3_evaluate_policy_cartpole -v`
Expected: PASS.

- [ ] **Step 5: Run the remaining three tests**

Run: `pytest tests/experiments/test_eval_utils.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/imitation/experiments/ftrl/eval_utils.py tests/experiments/test_eval_utils.py
git commit -m "feat(ftrl): add unified eval_utils with SB3-equivalent rollout eval"
```

---

## Task 2: Route `env_baselines.compute_baselines` through `eval_utils`

**Files:**
- Modify: `src/imitation/experiments/ftrl/env_baselines.py:156-192`

- [ ] **Step 1: Write the replacement for `compute_baselines`**

Replace the function body (lines 156-192) so that it uses `eval_policy_rollout` and also stores the expert's self-CE (used by the plotter as the horizontal Expert line on the loss subplot):

```python
def compute_baselines(
    expert_policy,
    venv: VecEnv,
    rng: np.random.Generator,
    n_expert_episodes: int = 20,
    n_random_episodes: int = 100,
) -> Dict[str, float]:
    """Compute expert return, random return, and expert self-CE baselines.

    ``expert_self_ce`` is $-\\tfrac{1}{|D|}\\sum_s \\log \\pi^*(\\arg\\max \\pi^*(\\cdot|s)|s)$
    — the residual softmax entropy of the expert at its own argmax, a direct
    argmax-pathology gauge used as the Expert line on the loss subplot.

    Args:
        expert_policy: A policy with a ``predict(obs, deterministic=True)``
            method (e.g. an SB3 BaseAlgorithm).
        venv: Vectorized environment.
        rng: Random number generator (unused currently, reserved for future).
        n_expert_episodes: Number of expert rollout episodes.
        n_random_episodes: Number of random rollout episodes.

    Returns:
        Dict with keys ``"expert_return"``, ``"random_return"``,
        ``"expert_self_ce"``.
    """
    from imitation.experiments.ftrl.eval_utils import eval_policy_rollout

    expert_result = eval_policy_rollout(
        expert_policy,
        venv,
        n_episodes=n_expert_episodes,
        deterministic=True,
        expert_policy=expert_policy,  # enables self-CE computation
    )
    random_return = compute_random_return(venv, n_episodes=n_random_episodes)
    return {
        "expert_return": float(expert_result.mean_return),
        "random_return": random_return,
        "expert_self_ce": float(expert_result.current_round_ce),
    }
```

Note: `load_or_compute_baselines` may have cached old-schema `baselines.json` files without `expert_self_ce`. The cache wipe in Task 10 clears these, so no migration is needed. As a defensive guard, add to `load_or_compute_baselines` right after the JSON is loaded:

```python
        if "expert_self_ce" not in cached:
            logger.info(f"Baselines cache at {cache_path} missing expert_self_ce; recomputing")
        else:
            return cached
```

- [ ] **Step 2: Run existing baselines test to verify no regression**

Run: `pytest tests/experiments/test_env_baselines.py -v`
Expected: PASS. If a test pinned the old `evaluate_policy` return-tuple shape it may need a minor update — follow the failure message.

- [ ] **Step 3: Commit**

```bash
git add src/imitation/experiments/ftrl/env_baselines.py
git commit -m "refactor(ftrl): route env_baselines expert eval through eval_utils"
```

---

## Task 3: Add convergence config + `expert_training.py`

**Files:**
- Modify: `src/imitation/experiments/ftrl/env_utils.py`
- Create: `src/imitation/experiments/ftrl/expert_training.py`

- [ ] **Step 1: Add `convergence` defaults and overrides to `env_utils.py`**

Near the top of `env_utils.py`, right above `ENV_CONFIGS`, add:

```python
DEFAULT_CONVERGENCE: Dict[str, float] = {
    "chunk_timesteps": 25_000,
    "min_timesteps": 50_000,
    "max_timesteps": 5_000_000,
    "threshold": 0.95,
    "patience": 5,
    "self_ce_eps": 0.05,
}


def get_convergence_config(env_name: str) -> Dict[str, float]:
    """Return convergence config for an env, merging defaults with overrides."""
    cfg = dict(DEFAULT_CONVERGENCE)
    env_cfg = ENV_CONFIGS.get(env_name, {})
    cfg.update(env_cfg.get("convergence", {}))
    return cfg
```

Then add per-env overrides inside `ENV_CONFIGS`:

```python
# In ENV_CONFIGS["MountainCar-v0"]:
"convergence": {"max_timesteps": 6_000_000, "chunk_timesteps": 50_000},

# In ENV_CONFIGS["Taxi-v3"]:
"convergence": {"max_timesteps": 3_000_000, "chunk_timesteps": 50_000},

# In ENV_CONFIGS["Blackjack-v1"]:
"convergence": {
    "threshold": 0.85,
    "self_ce_eps": 0.15,
    "_note": "Blackjack stochastic optimum; 0.95/0.05 unreachable.",
},
```

- [ ] **Step 2: Write the failing test for `train_classical_expert_until_converged`**

Append to `tests/experiments/test_expert_pipeline.py` (or create `tests/experiments/test_expert_training.py` if preferred — use whichever file already imports the classical pipeline):

```python
def test_train_classical_expert_converges_cartpole(tmp_path):
    """CartPole expert trainer must return a converged policy within the cap."""
    import numpy as np
    from imitation.experiments.ftrl import env_utils
    from imitation.experiments.ftrl.expert_training import (
        train_classical_expert_until_converged,
    )
    from imitation.experiments.ftrl.eval_utils import eval_policy_rollout

    rng = np.random.default_rng(0)
    venv = env_utils.make_env("CartPole-v1", n_envs=1, rng=rng)
    policy = train_classical_expert_until_converged(
        env_name="CartPole-v1",
        cache_dir=tmp_path,
        rng=rng,
        seed=0,
    )
    res = eval_policy_rollout(
        policy, venv, n_episodes=20, deterministic=True, expert_policy=policy
    )
    cfg = env_utils.get_convergence_config("CartPole-v1")
    # CartPole expert_score=500, random_score=22
    normalized = (res.mean_return - 22.0) / (500.0 - 22.0)
    assert normalized >= cfg["threshold"] - 0.05
    assert res.current_round_ce <= cfg["self_ce_eps"] + 0.02


def test_train_classical_expert_raises_on_non_convergence(tmp_path):
    """Unreachable threshold must raise RuntimeError."""
    import numpy as np
    import pytest
    from imitation.experiments.ftrl.expert_training import (
        train_classical_expert_until_converged,
    )

    rng = np.random.default_rng(0)
    # Force an impossible convergence: patch env config with a tiny budget
    # AND a threshold that 500-step PPO can't hit. Use a monkeypatch or a
    # direct kwarg override if the function supports it.
    with pytest.raises(RuntimeError, match="failed to converge"):
        train_classical_expert_until_converged(
            env_name="CartPole-v1",
            cache_dir=tmp_path,
            rng=rng,
            seed=0,
            convergence_override={
                "chunk_timesteps": 500,
                "min_timesteps": 500,
                "max_timesteps": 1000,
                "threshold": 0.99,
                "self_ce_eps": 0.001,
                "patience": 1,
            },
        )
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/experiments/test_expert_pipeline.py::test_train_classical_expert_converges_cartpole tests/experiments/test_expert_pipeline.py::test_train_classical_expert_raises_on_non_convergence -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imitation.experiments.ftrl.expert_training'`.

- [ ] **Step 4: Implement `expert_training.py`**

Create `src/imitation/experiments/ftrl/expert_training.py`:

```python
"""Convergence-detecting classical PPO expert trainer.

Wraps PPO.learn in a chunked loop and stops when the normalized return has
plateaued above a threshold AND the policy's residual-softmax entropy
(self-CE) is below a tightness bound. Raises RuntimeError if the max step
budget is exhausted before convergence.
"""

import logging
import pathlib
from typing import Dict, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

from imitation.experiments.ftrl import env_utils
from imitation.experiments.ftrl.env_baselines import REFERENCE_BASELINES
from imitation.experiments.ftrl.eval_utils import eval_policy_rollout

logger = logging.getLogger(__name__)


def _normalize(mean_return: float, env_name: str) -> float:
    ref = REFERENCE_BASELINES[env_name]
    lo, hi = ref["random_score"], ref["expert_score"]
    if abs(hi - lo) < 1e-8:
        return 0.0
    return (mean_return - lo) / (hi - lo)


def train_classical_expert_until_converged(
    env_name: str,
    cache_dir: pathlib.Path,
    rng: np.random.Generator,
    seed: int,
    convergence_override: Optional[Dict[str, float]] = None,
) -> BasePolicy:
    """Train a PPO MlpPolicy expert with chunked learn + convergence detection.

    Args:
        env_name: Classical gymnasium env ID (must be in ENV_CONFIGS).
        cache_dir: Where to save the converged model.
        rng: Random state.
        seed: PPO seed.
        convergence_override: If provided, fully replaces the per-env config.
            Intended for tests that want to force a non-convergence path.

    Returns:
        The converged `model.policy`.

    Raises:
        RuntimeError: If the max_timesteps budget is exhausted before
            normalized return passes threshold and self-CE passes self_ce_eps
            for `patience` consecutive chunks.
    """
    env_cfg = env_utils.ENV_CONFIGS[env_name]
    ppo_kwargs = env_cfg.get("ppo_kwargs", {})
    ppo_n_envs = env_cfg.get("ppo_n_envs", None)

    if convergence_override is not None:
        conv = dict(convergence_override)
    else:
        conv = env_utils.get_convergence_config(env_name)

    chunk_size = int(conv["chunk_timesteps"])
    max_total = int(conv["max_timesteps"])
    min_total = int(conv["min_timesteps"])
    threshold = float(conv["threshold"])
    patience = int(conv["patience"])
    self_ce_eps = float(conv["self_ce_eps"])
    tolerance = 0.02  # plateau tolerance on normalized return

    if ppo_n_envs and ppo_n_envs > 1:
        train_venv = env_utils.make_env(env_name, n_envs=ppo_n_envs, rng=rng)
    else:
        train_venv = env_utils.make_env(env_name, n_envs=1, rng=rng)
    eval_venv = env_utils.make_env(env_name, n_envs=1, rng=rng)

    model = PPO(
        "MlpPolicy",
        train_venv,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=0,
        **ppo_kwargs,
    )

    best_norm = -float("inf")
    best_self_ce = float("inf")
    chunks_since_best = 0
    total_steps = 0
    last_norm = -float("inf")
    last_self_ce = float("inf")

    while total_steps < max_total:
        model.learn(chunk_size, reset_num_timesteps=False)
        total_steps += chunk_size

        eval_res = eval_policy_rollout(
            model.policy,
            eval_venv,
            n_episodes=20,
            deterministic=True,
            expert_policy=model.policy,
        )
        norm_ret = _normalize(eval_res.mean_return, env_name)
        self_ce = float(eval_res.current_round_ce)
        last_norm, last_self_ce = norm_ret, self_ce

        improved = (
            norm_ret > best_norm + tolerance
            or self_ce < best_self_ce - tolerance
        )
        if improved:
            best_norm = max(best_norm, norm_ret)
            best_self_ce = min(best_self_ce, self_ce)
            chunks_since_best = 0
        else:
            chunks_since_best += 1

        logger.info(
            f"[{env_name}] step={total_steps}/{max_total} "
            f"norm_ret={norm_ret:.3f} self_ce={self_ce:.3f} "
            f"(best norm={best_norm:.3f} ce={best_self_ce:.3f}, "
            f"patience {chunks_since_best}/{patience})"
        )

        if (
            total_steps >= min_total
            and norm_ret >= threshold
            and self_ce <= self_ce_eps
            and chunks_since_best >= patience
        ):
            break
    else:
        train_venv.close()
        eval_venv.close()
        raise RuntimeError(
            f"Expert for {env_name} failed to converge in {max_total} steps: "
            f"norm_return={last_norm:.3f} (threshold={threshold}), "
            f"self_ce={last_self_ce:.3f} (eps={self_ce_eps})"
        )

    cache_path = pathlib.Path(cache_dir) / env_name.replace("/", "_")
    cache_path.mkdir(parents=True, exist_ok=True)
    model.save(cache_path / "model.zip")
    logger.info(f"Saved converged expert to {cache_path / 'model.zip'}")

    if ppo_n_envs and ppo_n_envs > 1:
        train_venv.close()
    else:
        train_venv.close()
    eval_venv.close()

    return model.policy
```

- [ ] **Step 5: Run the two new tests**

Run: `pytest tests/experiments/test_expert_pipeline.py::test_train_classical_expert_converges_cartpole tests/experiments/test_expert_pipeline.py::test_train_classical_expert_raises_on_non_convergence -v`
Expected: both PASS. CartPole convergence should take well under a minute locally.

- [ ] **Step 6: Commit**

```bash
git add src/imitation/experiments/ftrl/env_utils.py \
        src/imitation/experiments/ftrl/expert_training.py \
        tests/experiments/test_expert_pipeline.py
git commit -m "feat(ftrl): add convergence-detecting classical expert trainer"
```

---

## Task 4: Delegate `experts._train_classical_expert` to the new trainer

**Files:**
- Modify: `src/imitation/experiments/ftrl/experts.py:138-201`

- [ ] **Step 1: Replace `_train_classical_expert` body**

```python
def _train_classical_expert(
    env_name: str,
    venv: VecEnv,
    cache_dir: pathlib.Path,
    rng: np.random.Generator,
    seed: int,
) -> BasePolicy:
    """Train a classical MDP expert via chunked PPO + convergence detection."""
    from .expert_training import train_classical_expert_until_converged

    policy = train_classical_expert_until_converged(
        env_name=env_name,
        cache_dir=cache_dir,
        rng=rng,
        seed=seed,
    )

    # Compute and cache baselines on the caller's venv so downstream code
    # sees a consistent cache.
    from .env_baselines import load_or_compute_baselines, validate_expert_quality
    from .eval_utils import eval_policy_rollout

    baselines = load_or_compute_baselines(env_name, venv, policy, cache_dir, rng)
    res = eval_policy_rollout(policy, venv, n_episodes=20, deterministic=True)
    logger.info(
        f"Expert quality for {env_name}: reward={res.mean_return:.1f}"
    )
    is_ok, msg = validate_expert_quality(env_name, res.mean_return)
    if not is_ok:
        logger.warning(f"WARNING: {msg}")
    return policy
```

- [ ] **Step 2: Run the existing expert pipeline tests**

Run: `pytest tests/experiments/test_expert_pipeline.py -v -m "not expensive"`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/imitation/experiments/ftrl/experts.py
git commit -m "refactor(ftrl): delegate classical expert training to expert_training"
```

---

## Task 5: Add regression tests `test_expert_quality.py`

**Files:**
- Create: `tests/experiments/test_expert_quality.py`

- [ ] **Step 1: Write the regression tests**

```python
"""Regression tests pinning expert quality invariants for classical MDPs.

Marked @pytest.mark.expensive — these retrain experts and take minutes.
"""

import numpy as np
import pytest

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.experiments.ftrl import env_utils, experts, policy_utils
from imitation.experiments.ftrl.eval_utils import eval_policy_rollout
from imitation.experiments.ftrl.env_baselines import REFERENCE_BASELINES

CLASSICAL_ENVS = [
    "CartPole-v1",
    "FrozenLake-v1",
    "CliffWalking-v0",
    "Acrobot-v1",
    "MountainCar-v0",
    "Taxi-v3",
    "Blackjack-v1",
    "LunarLander-v2",
]


def _normalize(mean_return: float, env_name: str) -> float:
    ref = REFERENCE_BASELINES[env_name]
    return (mean_return - ref["random_score"]) / (
        ref["expert_score"] - ref["random_score"]
    )


@pytest.mark.expensive
@pytest.mark.parametrize("env_name", CLASSICAL_ENVS)
def test_expert_converged(env_name, tmp_path):
    """Every classical expert must meet its convergence bar after retraining."""
    rng = np.random.default_rng(0)
    venv = env_utils.make_env(env_name, n_envs=1, rng=rng)
    expert = experts.get_or_train_expert(
        env_name, venv, cache_dir=tmp_path, rng=rng, seed=0
    )
    res = eval_policy_rollout(
        expert, venv, n_episodes=20, deterministic=True, expert_policy=expert
    )
    cfg = env_utils.get_convergence_config(env_name)
    normalized = _normalize(res.mean_return, env_name)
    assert normalized >= cfg["threshold"] - 0.05, (
        f"{env_name}: normalized return {normalized:.3f} < "
        f"threshold {cfg['threshold']}"
    )
    assert res.current_round_ce <= cfg["self_ce_eps"] + 0.02, (
        f"{env_name}: self-CE {res.current_round_ce:.3f} > "
        f"eps {cfg['self_ce_eps']} (argmax pathology risk)"
    )


@pytest.mark.expensive
@pytest.mark.parametrize("env_name", CLASSICAL_ENVS)
def test_expert_beats_bc(env_name, tmp_path):
    """Expert deterministic return must meet-or-exceed BC's deterministic return."""
    rng = np.random.default_rng(0)
    venv = env_utils.make_env(env_name, n_envs=1, rng=rng)
    expert = experts.get_or_train_expert(
        env_name, venv, cache_dir=tmp_path, rng=rng, seed=0
    )

    trajs = rollout.generate_trajectories(
        policy=expert,
        venv=venv,
        sample_until=rollout.make_sample_until(min_timesteps=4000, min_episodes=1),
        deterministic_policy=True,
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(list(trajs))

    bc_policy = policy_utils.create_end_to_end_policy(
        venv.observation_space, venv.action_space
    )
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        rng=rng,
        policy=bc_policy,
        demonstrations=transitions,
        batch_size=min(32, len(transitions)),
    )
    bc_trainer.train(n_epochs=20)

    expert_res = eval_policy_rollout(expert, venv, n_episodes=20, deterministic=True)
    bc_res = eval_policy_rollout(
        bc_trainer.policy, venv, n_episodes=20, deterministic=True
    )
    assert expert_res.mean_return >= bc_res.mean_return * 0.95, (
        f"{env_name}: expert {expert_res.mean_return:.2f} < "
        f"BC {bc_res.mean_return:.2f} (argmax pathology or undertraining)"
    )
```

- [ ] **Step 2: Smoke-run a single env to confirm the test file is valid**

Run: `pytest tests/experiments/test_expert_quality.py -m expensive -k CartPole -v`
Expected: PASS. (Full parametrization runs in the final validation phase — see Task 10.)

- [ ] **Step 3: Commit**

```bash
git add tests/experiments/test_expert_quality.py
git commit -m "test(ftrl): add expert convergence and expert>=BC regression tests"
```

---

## Task 6: Rewrite `_run_dagger_variant` eval path to use `eval_utils` + aggregated `D_eval^t`

**Files:**
- Modify: `src/imitation/experiments/ftrl/run_experiment.py:285-409`
- Delete: `_evaluate_learner_metrics` (lines 108-170), `_evaluate_policy_cross_entropy` (lines 88-105) — if no other caller remains after Task 7.

- [ ] **Step 1: Update `_run_dagger_variant` to maintain `D_eval^t` and compute `rollout_cross_entropy` via `compute_sampled_action_ce`**

Replace the per-round loop (lines 362-409) with:

```python
    from imitation.data import serialize
    from imitation.experiments.ftrl.eval_utils import (
        compute_sampled_action_ce,
        eval_policy_rollout,
    )

    metrics = list(trainer.get_metrics())
    total_rounds = len(metrics)

    d_eval_obs: List[np.ndarray] = []
    d_eval_expert_acts: List[np.ndarray] = []

    per_round: List[Dict[str, Any]] = []
    cum_obs = 0
    for m in metrics:
        demo_round = m.round_num - 1
        round_dir = trainer._demo_dir_path_for_round(demo_round)
        demo_paths = trainer._get_demo_paths(round_dir)
        round_demos = []
        for p in demo_paths:
            round_demos.extend(serialize.load(p))
        round_transitions = rollout.flatten_trajectories(round_demos)
        cum_obs += len(round_transitions)

        round_data: Dict[str, Any] = {
            "round": m.round_num,
            "n_observations": cum_obs,
            "train_cross_entropy": round(m.cross_entropy, 6),
            "l2_norm": round(m.l2_norm, 6),
            "total_loss": round(m.total_loss, 6),
            "rollout_cross_entropy": None,
            "normalized_return": None,
            "disagreement_rate": None,
            "d_eval_size": sum(a.shape[0] for a in d_eval_obs),
        }

        is_first = m.round_num == 1
        is_interval = m.round_num % config.eval_interval == 0
        is_final = m.round_num == total_rounds
        if is_first or is_interval or is_final:
            eval_res = eval_policy_rollout(
                bc_trainer.policy,
                venv,
                n_episodes=20,
                deterministic=True,
                expert_policy=expert_policy,
            )
            d_eval_obs.append(eval_res.rollout_batch.obs)
            d_eval_expert_acts.append(eval_res.rollout_batch.expert_actions)
            agg_obs = np.concatenate(d_eval_obs, axis=0)
            agg_acts = np.concatenate(d_eval_expert_acts, axis=0)
            round_data["rollout_cross_entropy"] = round(
                compute_sampled_action_ce(bc_trainer.policy, agg_obs, agg_acts),
                6,
            )
            expert_ret = baselines["expert_return"]
            random_ret = baselines["random_return"]
            score_range = expert_ret - random_ret
            if abs(score_range) < 1e-8:
                norm_ret = 0.0
            else:
                norm_ret = (eval_res.mean_return - random_ret) / score_range
            round_data["normalized_return"] = round(norm_ret, 6)
            round_data["disagreement_rate"] = round(
                eval_res.current_round_disagreement, 6
            )
            round_data["d_eval_size"] = int(agg_obs.shape[0])

        per_round.append(round_data)

    return per_round
```

- [ ] **Step 2: Delete the now-unused helpers**

Delete `_evaluate_learner_metrics` (run_experiment.py:108-170) and `_evaluate_policy_cross_entropy` (lines 88-105). (Task 7 will also remove their `_run_bc` usage.)

- [ ] **Step 3: Run the runner tests**

Run: `pytest tests/experiments/test_run_experiment.py -v`
Expected: PASS. If tests pinned old JSON field names (`cross_entropy`, `expert_cross_entropy`), update them to `train_cross_entropy` / drop `expert_cross_entropy` — spec §3.4 says the old name becomes `train_cross_entropy` in the JSON.

- [ ] **Step 4: Commit**

```bash
git add src/imitation/experiments/ftrl/run_experiment.py tests/experiments/test_run_experiment.py
git commit -m "refactor(ftrl): D_eval aggregation + rollout_cross_entropy in DAgger path"
```

---

## Task 7: Rewrite `_run_bc` to use `eval_utils` (no aggregation; fixed-BC is a reference only)

**Files:**
- Modify: `src/imitation/experiments/ftrl/run_experiment.py:412-513`

- [ ] **Step 1: Rewrite `_run_bc`**

```python
def _run_bc(
    config: ExperimentConfig,
    venv,
    expert_policy,
    rng: np.random.Generator,
    baselines: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Fixed BC baseline: train once on the full expert dataset, then eval.

    No D_eval aggregation — fixed BC is a static reference line on the
    return subplot and does NOT appear on the loss subplot.
    """
    from imitation.experiments.ftrl.eval_utils import eval_policy_rollout

    total_timesteps = config.n_rounds * config.samples_per_round

    sample_until = rollout.make_sample_until(
        min_timesteps=total_timesteps, min_episodes=1
    )
    trajs = rollout.generate_trajectories(
        policy=expert_policy,
        venv=venv,
        sample_until=sample_until,
        deterministic_policy=True,
        rng=rng,
    )
    all_transitions = rollout.flatten_trajectories(list(trajs))
    if len(all_transitions) > total_timesteps:
        all_transitions = all_transitions[:total_timesteps]

    if config.policy_mode == "linear":
        policy = policy_utils.create_linear_policy(expert_policy)
    else:
        policy = policy_utils.create_end_to_end_policy(
            venv.observation_space, venv.action_space
        )

    custom_logger = imit_logger.configure(
        str(config.output_dir / "tb" / f"bc_{config.env_name}_{config.seed}"),
        format_strs=[],
    )
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        rng=rng,
        policy=policy,
        demonstrations=all_transitions,
        batch_size=min(32, len(all_transitions)),
        custom_logger=custom_logger,
    )
    bc_trainer.train(n_epochs=config.bc_n_epochs)

    per_round: List[Dict[str, Any]] = []
    chunk_size = config.samples_per_round
    cum_obs = 0
    for round_num in range(config.n_rounds):
        start_idx = round_num * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_transitions))
        if start_idx >= len(all_transitions):
            break
        chunk_len = end_idx - start_idx
        cum_obs += chunk_len

        l2_norms = [
            th.sum(th.square(w)).item() for w in bc_trainer.policy.parameters()
        ]
        l2_norm = sum(l2_norms) / 2

        round_data: Dict[str, Any] = {
            "round": round_num + 1,
            "n_observations": cum_obs,
            "train_cross_entropy": None,  # not tracked for fixed BC
            "l2_norm": round(l2_norm, 6),
            "total_loss": None,
            "rollout_cross_entropy": None,  # fixed BC does not appear on loss subplot
            "normalized_return": None,
            "disagreement_rate": None,
        }

        is_first = round_num == 0
        is_interval = (round_num + 1) % config.eval_interval == 0
        is_final = round_num == config.n_rounds - 1
        if is_first or is_interval or is_final:
            eval_res = eval_policy_rollout(
                bc_trainer.policy,
                venv,
                n_episodes=20,
                deterministic=True,
                expert_policy=expert_policy,
            )
            expert_ret = baselines["expert_return"]
            random_ret = baselines["random_return"]
            score_range = expert_ret - random_ret
            if abs(score_range) < 1e-8:
                norm_ret = 0.0
            else:
                norm_ret = (eval_res.mean_return - random_ret) / score_range
            round_data["normalized_return"] = round(norm_ret, 6)
            round_data["disagreement_rate"] = round(
                eval_res.current_round_disagreement, 6
            )

        per_round.append(round_data)

    return per_round
```

- [ ] **Step 2: Run the runner tests**

Run: `pytest tests/experiments/test_run_experiment.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/imitation/experiments/ftrl/run_experiment.py
git commit -m "refactor(ftrl): fixed BC uses eval_utils; drop legacy eval helpers"
```

---

## Task 8: Add `_run_bc_dagger` + extend `ALL_ALGOS`

**Files:**
- Modify: `src/imitation/experiments/ftrl/run_experiment.py`

- [ ] **Step 1: Extend `ALL_ALGOS` and the algo dispatch**

Change line 32:

```python
ALL_ALGOS = ["ftl", "ftrl", "bc", "bc_dagger"]
```

Change the dispatch block (around line 259-270):

```python
    if config.algo in ("ftl", "ftrl"):
        result["per_round"] = _run_dagger_variant(
            config, venv, expert_policy, rng, baselines
        )
    elif config.algo == "bc":
        result["per_round"] = _run_bc(config, venv, expert_policy, rng, baselines)
    elif config.algo == "bc_dagger":
        result["per_round"] = _run_bc_dagger(
            config, venv, expert_policy, rng, baselines
        )
    else:
        raise ValueError(f"Unknown algo: {config.algo}")
```

- [ ] **Step 2: Implement `_run_bc_dagger`**

Append to `run_experiment.py`, after `_run_bc`:

```python
def _run_bc_dagger(
    config: ExperimentConfig,
    venv,
    expert_policy,
    rng: np.random.Generator,
    baselines: Dict[str, float],
) -> List[Dict[str, Any]]:
    """BC+DAgger baseline.

    Per-round ERM on a growing PREFIX of the expert dataset, sized to match
    DAgger's aggregated observation budget. Eval uses the same aggregated
    D_eval^t buffer construction as FTL/FTRL+DAgger (spec §3.4).
    """
    from imitation.experiments.ftrl.eval_utils import (
        compute_sampled_action_ce,
        eval_policy_rollout,
    )

    total_timesteps = config.n_rounds * config.samples_per_round

    sample_until = rollout.make_sample_until(
        min_timesteps=total_timesteps, min_episodes=1
    )
    trajs = rollout.generate_trajectories(
        policy=expert_policy,
        venv=venv,
        sample_until=sample_until,
        deterministic_policy=True,
        rng=rng,
    )
    all_transitions = rollout.flatten_trajectories(list(trajs))
    if len(all_transitions) < total_timesteps:
        raise RuntimeError(
            f"BC+DAgger: collected {len(all_transitions)} transitions, "
            f"need {total_timesteps}"
        )
    all_transitions = all_transitions[:total_timesteps]

    warm_start = env_utils.is_atari(config.env_name)
    policy = None

    d_eval_obs: List[np.ndarray] = []
    d_eval_expert_acts: List[np.ndarray] = []

    per_round: List[Dict[str, Any]] = []
    for round_num in range(1, config.n_rounds + 1):
        k = round_num * config.samples_per_round
        prefix = all_transitions[:k]

        if policy is None or not warm_start:
            if config.policy_mode == "linear":
                policy = policy_utils.create_linear_policy(expert_policy)
            else:
                policy = policy_utils.create_end_to_end_policy(
                    venv.observation_space, venv.action_space
                )

        custom_logger = imit_logger.configure(
            str(
                config.output_dir
                / "tb"
                / f"bc_dagger_{config.env_name}_{config.seed}_r{round_num}"
            ),
            format_strs=[],
        )
        bc_trainer = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            rng=rng,
            policy=policy,
            demonstrations=prefix,
            batch_size=min(32, len(prefix)),
            custom_logger=custom_logger,
        )
        bc_trainer.train(n_epochs=config.bc_n_epochs)
        policy = bc_trainer.policy  # warm-start reuses this next round

        l2_norms = [th.sum(th.square(w)).item() for w in policy.parameters()]
        l2_norm = sum(l2_norms) / 2

        round_data: Dict[str, Any] = {
            "round": round_num,
            "n_observations": k,
            "train_cross_entropy": None,
            "l2_norm": round(l2_norm, 6),
            "total_loss": None,
            "rollout_cross_entropy": None,
            "normalized_return": None,
            "disagreement_rate": None,
            "d_eval_size": sum(a.shape[0] for a in d_eval_obs),
        }

        is_first = round_num == 1
        is_interval = round_num % config.eval_interval == 0
        is_final = round_num == config.n_rounds
        if is_first or is_interval or is_final:
            eval_res = eval_policy_rollout(
                policy,
                venv,
                n_episodes=20,
                deterministic=True,
                expert_policy=expert_policy,
            )
            d_eval_obs.append(eval_res.rollout_batch.obs)
            d_eval_expert_acts.append(eval_res.rollout_batch.expert_actions)
            agg_obs = np.concatenate(d_eval_obs, axis=0)
            agg_acts = np.concatenate(d_eval_expert_acts, axis=0)
            round_data["rollout_cross_entropy"] = round(
                compute_sampled_action_ce(policy, agg_obs, agg_acts), 6
            )
            expert_ret = baselines["expert_return"]
            random_ret = baselines["random_return"]
            score_range = expert_ret - random_ret
            norm_ret = (
                0.0
                if abs(score_range) < 1e-8
                else (eval_res.mean_return - random_ret) / score_range
            )
            round_data["normalized_return"] = round(norm_ret, 6)
            round_data["disagreement_rate"] = round(
                eval_res.current_round_disagreement, 6
            )
            round_data["d_eval_size"] = int(agg_obs.shape[0])

        per_round.append(round_data)
    return per_round
```

- [ ] **Step 2b: Update the CLI `--algos` choices**

Line 619 of `run_experiment.py`:

```python
        choices=ALL_ALGOS,
```

(Already references `ALL_ALGOS`; no change needed — just confirm via grep.)

- [ ] **Step 3: Add a smoke test for `_run_bc_dagger`**

Append to `tests/experiments/test_run_experiment.py`:

```python
def test_run_bc_dagger_smoke(tmp_path):
    """bc_dagger runs end-to-end on CartPole with small budget."""
    import argparse
    from imitation.experiments.ftrl.run_experiment import (
        ExperimentConfig,
        run_single,
    )

    config = ExperimentConfig(
        algo="bc_dagger",
        env_name="CartPole-v1",
        seed=0,
        policy_mode="end_to_end",
        n_rounds=3,
        samples_per_round=200,
        l2_lambda=0.0,
        l2_decay=False,
        warm_start=False,
        beta_rampdown=1,
        bc_n_epochs=2,
        eval_interval=1,
        output_dir=tmp_path / "out",
        expert_cache_dir=tmp_path / "experts",
    )
    result = run_single(config)
    assert result["algo"] == "bc_dagger"
    assert len(result["per_round"]) == 3
    # Data budget invariant: round k has exactly k * samples_per_round observations
    for i, r in enumerate(result["per_round"]):
        assert r["n_observations"] == (i + 1) * 200
    # rollout_cross_entropy populated on every eval-point round (all of them here)
    for r in result["per_round"]:
        assert r["rollout_cross_entropy"] is not None
```

- [ ] **Step 4: Run the smoke test**

Run: `pytest tests/experiments/test_run_experiment.py::test_run_bc_dagger_smoke -v`
Expected: PASS (takes ~15-30 seconds, including CartPole expert training).

- [ ] **Step 5: Commit**

```bash
git add src/imitation/experiments/ftrl/run_experiment.py tests/experiments/test_run_experiment.py
git commit -m "feat(ftrl): add BC+DAgger baseline with aggregated D_eval buffer"
```

---

## Task 9: Update `plot_results.py` — relabel, colors, loss metric, subtitle, `--show-expert-on-loss`

**Files:**
- Modify: `src/imitation/experiments/ftrl/plot_results.py`

- [ ] **Step 1: Replace the algo tables at lines 35-45**

```python
ALGO_COLORS: Dict[str, str] = {
    "ftl": "#1f77b4",       # blue
    "ftrl": "#d62728",      # red
    "bc_dagger": "#2ca02c", # green
    "bc": "#17a663",        # dark green (dashed reference)
    "expert": "#555555",    # gray (dashed reference)
}

ALGO_LABELS: Dict[str, str] = {
    "ftl": "FTL+DAgger",
    "ftrl": "FTRL+DAgger",
    "bc_dagger": "BC+DAgger",
    "bc": "BC (fixed)",
    "expert": "Expert",
}

ALGO_LINESTYLES: Dict[str, str] = {
    "ftl": "-",
    "ftrl": "-",
    "bc_dagger": "-",
    "bc": "--",
    "expert": "--",
}

LOSS_SUBPLOT_ALGOS = {"ftl", "ftrl", "bc_dagger"}  # fixed BC excluded
```

- [ ] **Step 2: Update `load_results` to read new field names**

In `load_results` (lines 101-127), replace the per-round row construction with:

```python
        expert_self_ce = data.get("baselines", {}).get("expert_self_ce")
        for m in data.get("per_round", []):
            row = {
                "algo": data["algo"],
                "env": data["env"],
                "seed": data["seed"],
                "policy_mode": data.get("policy_mode", "unknown"),
                "round": m["round"],
                "n_observations": m.get("n_observations", 0),
                "train_cross_entropy": m.get("train_cross_entropy"),
                "rollout_cross_entropy": m.get("rollout_cross_entropy"),
                "l2_norm": m.get("l2_norm"),
                "total_loss": m.get("total_loss"),
                "normalized_return": (
                    m["normalized_return"]
                    if m.get("normalized_return") is not None
                    else np.nan
                ),
                "disagreement_rate": (
                    m["disagreement_rate"]
                    if m.get("disagreement_rate") is not None
                    else np.nan
                ),
                "expert_self_ce": expert_self_ce,
            }
            rows.append(row)
```

- [ ] **Step 3: Update `_plot_metric` to honor per-algo linestyle and skip algos not on the subplot**

Replace `_plot_metric` (lines 193-296):

```python
def _plot_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    log_scale: bool = False,
    allowed_algos: Optional[set] = None,
):
    algos = sorted(
        df["algo"].unique(),
        key=lambda a: list(ALGO_COLORS.keys()).index(a) if a in ALGO_COLORS else 99,
    )
    for algo in algos:
        if allowed_algos is not None and algo not in allowed_algos:
            continue
        algo_df = df[df["algo"] == algo]
        valid_df = algo_df.dropna(subset=[metric])
        if valid_df.empty:
            continue

        rounds = sorted(valid_df["round"].unique())
        x_vals, iqm_vals, ci_lows, ci_highs = [], [], [], []
        for rnd in rounds:
            rnd_df = valid_df[valid_df["round"] == rnd]
            values = rnd_df[metric].values
            if len(values) == 0:
                continue
            iqm, ci_lo, ci_hi = _compute_iqm_and_ci(values)
            x_vals.append(rnd_df["n_observations"].mean())
            iqm_vals.append(iqm)
            ci_lows.append(ci_lo)
            ci_highs.append(ci_hi)
        if not x_vals:
            continue
        color = ALGO_COLORS.get(algo, "#888888")
        linestyle = ALGO_LINESTYLES.get(algo, "-")
        label = ALGO_LABELS.get(algo, algo)
        ax.plot(
            x_vals,
            iqm_vals,
            color=color,
            linestyle=linestyle,
            label=label,
            linewidth=2,
            marker="o" if linestyle == "-" else None,
            markersize=3,
        )
        ax.fill_between(x_vals, ci_lows, ci_highs, color=color, alpha=0.12)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Number of Observations")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
```

- [ ] **Step 4: Update `plot_env` to use new metric, subtitle, and `show_expert_on_loss`**

Replace `plot_env` starting at line 299:

```python
LOSS_SUBTITLE = (
    r"Loss: $-\frac{1}{|D_{\mathrm{eval}}^t|}\sum_{(s,a^*)\in D_{\mathrm{eval}}^t}"
    r"\log\pi^t(a^*|s),\;a^*(s)=\arg\max_a \pi^*(a|s).$  "
    "$D_{\mathrm{eval}}^t$ = aggregated fresh rollouts of the current learner. "
    "BC+DAgger train set = expert-data prefix sized to DAgger's observation budget."
)


def plot_env(
    df: pd.DataFrame,
    env_name: str,
    output_path: pathlib.Path,
    show_expert_on_loss: bool = True,
) -> None:
    env_df = df[df["env"] == env_name]
    if env_df.empty:
        logger.warning(f"No data for {env_name}, skipping")
        return

    policy_modes = env_df["policy_mode"].unique()
    mode_str = policy_modes[0] if len(policy_modes) == 1 else "mixed"

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 15), sharex=True)
    fig.suptitle(f"{env_name}  ({mode_str})", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.955, LOSS_SUBTITLE, ha="center", fontsize=8, wrap=True)

    # Subplot 1: rollout_cross_entropy on the aggregated D_eval^t buffer
    _plot_metric(
        ax1,
        env_df,
        "rollout_cross_entropy",
        "Rollout CE on $D_{\\mathrm{eval}}^t$ (log)",
        log_scale=True,
        allowed_algos=LOSS_SUBPLOT_ALGOS,
    )
    if show_expert_on_loss:
        # Expert self-CE from baselines.json (stored by env_baselines.compute_baselines).
        # This is the direct argmax-pathology gauge: sharp expert → near 0.
        expert_self_ce_vals = env_df.get("expert_self_ce")
        if expert_self_ce_vals is not None and expert_self_ce_vals.notna().any():
            y = float(expert_self_ce_vals.dropna().iloc[0])
            ax1.axhline(
                y=y,
                color=ALGO_COLORS["expert"],
                linestyle=ALGO_LINESTYLES["expert"],
                linewidth=1.2,
                alpha=0.7,
                label=f"{ALGO_LABELS['expert']} self-CE = {y:.3f}",
            )
            ax1.legend(fontsize=9)

    # Subplot 2: Normalized expected return
    _plot_metric(
        ax2,
        env_df,
        "normalized_return",
        "Normalized Expected Return",
    )
    ax2.axhline(
        y=1.0,
        color=ALGO_COLORS["expert"],
        linestyle=ALGO_LINESTYLES["expert"],
        linewidth=1.2,
        alpha=0.7,
        label=ALGO_LABELS["expert"] + " (1.0)",
    )
    ax2.axhline(
        y=0.0, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Random (0.0)"
    )
    ax2.legend(fontsize=9)

    # Subplot 3: On-policy disagreement rate (no fixed BC or expert)
    _plot_metric(
        ax3,
        env_df,
        "disagreement_rate",
        "On-Policy Disagreement Rate",
        allowed_algos={"ftl", "ftrl", "bc_dagger", "bc"},
    )
    ax3.set_ylim(-0.05, 1.05)

    # Subplot 4: Cumulative regret vs expert (loss-subplot algos only)
    _plot_metric(
        ax4,
        env_df,
        "cum_regret",
        "Cumulative Regret (vs Expert)",
        allowed_algos=LOSS_SUBPLOT_ALGOS,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot: {output_path}")
```

Also update `compute_cumulative_loss` / `compute_cumulative_regret` (lines 141-190) to use `rollout_cross_entropy` instead of `cross_entropy`, and to filter to `LOSS_SUBPLOT_ALGOS`:

```python
def compute_cumulative_loss(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["algo", "env", "seed", "round"]).copy()
    df["cum_loss"] = (
        df.groupby(["algo", "env", "seed"])["rollout_cross_entropy"].cumsum()
    )
    return df


def compute_cumulative_regret(df: pd.DataFrame) -> pd.DataFrame:
    if "cum_loss" not in df.columns:
        df = compute_cumulative_loss(df)
    # Regret = cum_loss - min across dynamic algos at same (env, seed, round)
    loss_df = df[df["algo"].isin({"ftl", "ftrl", "bc_dagger"})]
    baseline = (
        loss_df.groupby(["env", "seed", "round"])["cum_loss"]
        .min()
        .rename("baseline_cum_loss")
    )
    df = df.merge(baseline, on=["env", "seed", "round"], how="left")
    df["cum_regret"] = df["cum_loss"] - df["baseline_cum_loss"]
    df.drop(columns=["baseline_cum_loss"], inplace=True)
    return df
```

- [ ] **Step 5: Add CLI flag `--show-expert-on-loss`**

In `main()` argparser (lines 421-442):

```python
    parser.add_argument(
        "--show-expert-on-loss",
        dest="show_expert_on_loss",
        action="store_true",
        default=True,
        help="Draw expert self-CE on the loss subplot (default on)",
    )
    parser.add_argument(
        "--hide-expert-on-loss",
        dest="show_expert_on_loss",
        action="store_false",
        help="Hide expert self-CE on the loss subplot",
    )
```

Pass `show_expert_on_loss=args.show_expert_on_loss` into `plot_all` → `plot_env`. Update `plot_all` signature accordingly.

- [ ] **Step 6: Run existing plot tests**

Run: `pytest tests/experiments/test_plot_results.py -v`
Expected: PASS. Update any fixtures that reference the old `cross_entropy` JSON field — they must now emit `rollout_cross_entropy`.

- [ ] **Step 7: Commit**

```bash
git add src/imitation/experiments/ftrl/plot_results.py tests/experiments/test_plot_results.py
git commit -m "feat(ftrl): new plot layout with BC+DAgger + aggregated rollout CE"
```

---

## Task 10: Cache wipe + Pass 1 on server + local plot

**Files:** (no code changes)

Execution target is the remote compute server. The gitignored
`./experiments/sync_results.sh` script handles all code push / result pull.
Do NOT hardcode server hostname, paths, or hardware details in any tracked
file — this is a public repo.

- [ ] **Step 1: Wipe classical caches and results locally**

```bash
cd /Users/thangduong/Desktop/imitation/.worktrees/ftrl-expert-fix
for env in CartPole-v1 FrozenLake-v1 CliffWalking-v0 Acrobot-v1 \
           MountainCar-v0 Taxi-v3 Blackjack-v1 LunarLander-v2; do
    rm -rf "experiments/expert_cache/$env" "experiments/results/$env"
done
```

Atari caches preserved (verify: `ls experiments/expert_cache/ | grep NoFrameskip`).

- [ ] **Step 2: Run full fast-path test suite locally to confirm green**

Run: `pytest -n auto tests/ -m "not expensive"`
Expected: PASS across the board.

- [ ] **Step 3: Push the branch to the server**

```bash
git push -u origin feature/ftrl-expert-fix-bcdagger
./experiments/sync_results.sh push
```

- [ ] **Step 4: On server — checkout branch, activate env, wipe classical caches**

Ask the user to paste this into an interactive shell on the server (the assistant should not drive remote shells directly for auth/tmux lifecycle reasons):

```bash
# In an ssh session on the compute server, inside tmux:
cd ~/imitation   # or wherever the repo lives on the server
git fetch
git checkout feature/ftrl-expert-fix-bcdagger
git pull
source ~/miniconda3/etc/profile.d/conda.sh
conda activate <env-name>
pip install -e ".[dev]"   # only if deps changed
for env in CartPole-v1 FrozenLake-v1 CliffWalking-v0 Acrobot-v1 \
           MountainCar-v0 Taxi-v3 Blackjack-v1 LunarLander-v2; do
    rm -rf "experiments/expert_cache/$env" "experiments/results/$env"
done
```

- [ ] **Step 5: On server — run expert_quality smoke (CartPole only) before the full pass**

```bash
pytest tests/experiments/test_expert_quality.py -m expensive -k CartPole -v
```
Expected: PASS.

- [ ] **Step 6: On server — Pass 1 (6 fast envs) in tmux**

```bash
tmux new -s ftrl_pass1
# inside tmux, after conda activate:
python -m imitation.experiments.ftrl.run_experiment \
  --envs CartPole-v1 FrozenLake-v1 CliffWalking-v0 Acrobot-v1 \
         Blackjack-v1 LunarLander-v2 \
  --algos ftl ftrl bc bc_dagger \
  --seeds 5 --n-workers 8 2>&1 | tee experiments/logs/pass1.log
# Ctrl-b d to detach.  tmux attach -t ftrl_pass1 to reattach.
```

Wait for the run to finish. Confirm exit code 0 and that every
`experiments/results/<env>/*.json` exists.

- [ ] **Step 7: Pull results back locally and generate Pass 1 plots**

```bash
# Local worktree:
./experiments/sync_results.sh pull
python -m imitation.experiments.ftrl.plot_results \
  --results-dir experiments/results/ \
  --envs CartPole-v1 FrozenLake-v1 CliffWalking-v0 Acrobot-v1 \
         Blackjack-v1 LunarLander-v2
```

Inspect `experiments/plots/*.png`. Acceptance criteria (spec §5 items 5-6):
- FTL+DAgger / FTRL+DAgger / BC+DAgger are solid lines on loss subplot.
- Fixed BC is dashed on return subplot ONLY (not loss).
- Expert is dashed on return subplot at y=1.0 AND on loss subplot (horizontal dashed near 0).
- Expert beats BC on the return subplot for all 6 fast envs.

- [ ] **Step 8: On server — run expensive expert_quality tests on all 6 fast envs**

```bash
pytest tests/experiments/test_expert_quality.py -m expensive \
  -k "CartPole or FrozenLake or CliffWalking or Acrobot or Blackjack or LunarLander" -v
```

Expected: all 12 parametrized cases PASS.

- [ ] **Step 9: Commit any tuning adjustments**

If any env fails convergence and requires tuning `ppo_kwargs` or `convergence` in `env_utils.py`:

```bash
git add src/imitation/experiments/ftrl/env_utils.py
git commit -m "tune(ftrl): adjust <env> PPO/convergence config to hit threshold"
```

---

## Task 11: Pass 2 — MountainCar + Taxi + final plot regeneration

**Files:** (no code changes)

- [ ] **Step 1: On server — run Pass 2 in tmux**

```bash
tmux new -s ftrl_pass2
# inside tmux, after conda activate:
python -m imitation.experiments.ftrl.run_experiment \
  --envs MountainCar-v0 Taxi-v3 \
  --algos ftl ftrl bc bc_dagger \
  --seeds 5 --n-workers 4 2>&1 | tee experiments/logs/pass2.log
```

- [ ] **Step 2: On server — run expensive tests on MountainCar + Taxi**

```bash
pytest tests/experiments/test_expert_quality.py -m expensive \
  -k "MountainCar or Taxi" -v
```

Expected: all 4 cases PASS.

- [ ] **Step 3: Pull results locally and regenerate plots for all 8 envs**

```bash
./experiments/sync_results.sh pull
python -m imitation.experiments.ftrl.plot_results \
  --results-dir experiments/results/ \
  --envs CartPole-v1 FrozenLake-v1 CliffWalking-v0 Acrobot-v1 \
         Blackjack-v1 LunarLander-v2 MountainCar-v0 Taxi-v3
```

- [ ] **Step 4: Final acceptance check**

Verify every item in spec §5 (acceptance criteria):

1. `pytest tests/experiments/test_eval_utils.py` → PASS.
2. `pytest -m expensive tests/experiments/test_expert_quality.py` → PASS on all 8 envs.
3. `baselines.json` for each env shows `expert_return` ≥ 0.90 of `REFERENCE_BASELINES`.
4. The full `run_experiment` invocation from §6 completed without errors.
5. Plots have correct labels, linestyles, colors, subtitle, and loss metric.
6. Every expert's self-CE ≤ `self_ce_eps`.
7. `pytest -n auto tests/ -m "not expensive"` → PASS.
8. Atari caches/JSONs untouched (`git status experiments/` shows only classical changes).

- [ ] **Step 5: Push branch**

```bash
git push -u origin feature/ftrl-expert-fix-bcdagger
```

---

## Notes on what NOT to change

- `src/imitation/algorithms/ftrl.py` (FTRLTrainer) — unchanged.
- Atari experts — caches preserved, `_get_atari_expert` unchanged.
- `compute_random_return` — kept as-is (random-action sampler, no policy).
- Disagreement rate definition — unchanged (current round only, from `eval_res.current_round_disagreement`).
- Normalized return formula — unchanged (`(ret - random) / (expert - random)`).

---

## Spec coverage matrix

| Spec section | Task |
|---|---|
| §3.1 eval_utils | Task 1 |
| §3.1 env_baselines route | Task 2 |
| §3.2 convergence trainer | Task 3 |
| §3.2 experts.py delegation | Task 4 |
| §3.3 regression tests | Task 5 |
| §3.4 DAgger D_eval aggregation | Task 6 |
| §3.4 fixed BC path uses eval_utils | Task 7 |
| §3.4 BC+DAgger algorithm | Task 8 |
| §3.5 plot relabeling + subtitle + debug flag | Task 9 |
| §6 local execution Pass 1 | Task 10 |
| §6 local execution Pass 2 | Task 11 |
