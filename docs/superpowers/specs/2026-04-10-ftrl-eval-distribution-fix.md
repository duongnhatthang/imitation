# FTRL Experiment: Evaluation Distribution Clarity, Expert Convergence Fix, and BC+DAgger Baseline

- **Date:** 2026-04-10
- **Branch:** `feature/ftrl-expert-fix-bcdagger` (off `feature/atari-expanded-mdps`)
- **Worktree:** `.worktrees/ftrl-expert-fix`
- **Related area:** `src/imitation/experiments/ftrl/`, `src/imitation/algorithms/ftrl.py`, `tests/experiments/`

---

## 1. Problem statement

Three problems surfaced in the current FTRL / FTL / BC classical-MDP experiment pipeline:

### P1. Evaluation distribution is opaque and inconsistent across algorithms

Learner return is computed under each learner's own rollout distribution by `_evaluate_learner_metrics` in `run_experiment.py`, while the **expert baseline return** is computed once by SB3's `evaluate_policy` on the expert's rollout distribution and cached as a single scalar in `baselines.json`. These are two different code paths with different episode counts (20 vs 10) and different Monitor-handling semantics. The cross-entropy column is worse: for BC it is computed on a slice of the pre-collected expert dataset, while for FTL/FTRL it is the BC trainer's loss over the DAgger-aggregated data — so the "same" column plots three different quantities.

### P2. No BC baseline exists that matches the DAgger data budget and is evaluated under its own rollout distribution

The current BC baseline trains once on the full expert dataset and stays fixed. There is no BC baseline that (a) is re-trained as more expert data becomes available, matching the per-round data budget of FTL+DAgger / FTRL+DAgger, and (b) is evaluated under its own rollout distribution so the loss/return comparison is apples-to-apples with DAgger variants.

### P3. Expert policy is systematically worse than BC on classical MDPs

Plots across the 8 classical MDPs show BC's normalized return exceeding the expert's normalized return (= 1.0) on most envs. This is theoretically impossible if BC converges to the same policy as the expert. Diagnosis indicates two compounding causes:

1. **Argmax pathology on undertrained PPO experts.** PPO optimizes a stochastic policy; its reported training return reflects the softmax-sampled rollouts, not the deterministic `argmax(logits)`. When the softmax has not sharpened enough, the deterministic mode is suboptimal at critical states (e.g., MountainCar's apex, Taxi pickups). BC distills stochastic expert samples into a sharper categorical whose argmax picks better actions than the expert's own argmax. Visible evidence: the current `expert_cross_entropy` at expert states is non-zero — that value is the residual entropy of the expert's softmax at its own argmax action, which is a direct sharpness gauge.
2. **Cache-first loading with no config re-check.** `experts.get_or_train_expert` loads `model.zip` unconditionally if it exists, ignoring any subsequent bumps to `ppo_timesteps` or hyperparameters in `ENV_CONFIGS`. Stale undertrained experts silently persist.

These interact: stale caches guarantee (1) stays bad.

---

## 2. Design overview

Five interconnected changes, all landing in one PR on one branch:

1. **Unified policy evaluation function** — a single custom `eval_policy_rollout()` used by every code path that measures rollout return or rollout cross-entropy. Pinned to SB3's `evaluate_policy` behavior by an equivalence test.
2. **Expert trainer with automatic convergence detection** — hybrid reference-threshold + plateau + residual-entropy signal, hard timestep ceiling with strict error on failure, per-env overrides, D1 fallback for genuinely-unconvergeable envs.
3. **Cache invalidation + retraining + automated regression tests** — delete stale classical expert caches, retrain via the new trainer, add CI-checkable tests that pin the invariants (`expert is converged`, `expert ≥ BC`, `custom eval ≈ SB3 eval`).
4. **BC+DAgger baseline** — new algorithm that matches FTL/FTRL's observation budget per round using expert data, evaluated under its own rollout distribution.
5. **Plot relabeling and new rollout-CE metric** — unified `rollout_cross_entropy` on the loss subplot, `BC` + `Expert` as dashed reference lines on the return subplot, clarified subtitles.

---

## 3. Component designs

### 3.1 Unified policy evaluation — `src/imitation/experiments/ftrl/eval_utils.py` (new)

**Function signature:**

```python
def eval_policy_rollout(
    policy: BasePolicy,
    venv: VecEnv,
    n_episodes: int,
    deterministic: bool = True,
    expert_policy: Optional[BasePolicy] = None,
    safety_step_cap: int = 100_000,
) -> EvalResult:
    """Roll out a policy for exactly n_episodes complete episodes and return metrics.

    If expert_policy is provided, at each step the expert's action is queried at the
    current state and used to compute disagreement_rate and rollout_cross_entropy
    (= -mean log prob of expert's action under the learner policy at the learner's
    own rollout states). No extra rollouts are performed for CE -- the same
    transitions that drive return are reused.

    Raises:
        RuntimeError: if safety_step_cap is hit before n_episodes complete.
            This is a failure signal, NOT a silent partial-episode accounting.
    """
```

**Returns:**

```python
@dataclasses.dataclass
class EvalResult:
    mean_return: float
    episode_returns: List[float]   # length == n_episodes exactly
    disagreement_rate: Optional[float]          # None if expert_policy is None
    rollout_cross_entropy: Optional[float]      # None if expert_policy is None
    n_steps: int                                # total env steps used
```

**Key properties:**

- Runs until **exactly** `n_episodes` complete episodes — no partial-episode fudge. The current `max_steps=10000` wart in `_evaluate_learner_metrics` that counted a partial episode as a full one (run_experiment.py:152-153) is removed.
- `safety_step_cap` is a hard ceiling to prevent infinite loops from buggy envs; hitting it raises `RuntimeError` with a diagnostic message. Default 100k, overridable.
- When a SB3 `Monitor` is present in the venv wrapper chain (which it always is via `util.make_vec_env`), the per-episode return is pulled from `info["episode"]["r"]` to exactly match SB3's convention. Otherwise falls back to accumulated `rewards[i]`. This is the same selection logic SB3 uses in `evaluation.py`.
- `deterministic=True` selects the policy's argmax (categorical) or mean (Gaussian) action, matching SB3.
- `rollout_cross_entropy` is computed from the same transitions collected during the rollout. No extra rollouts.

**Call sites that collapse to this function:**

| Old site | New |
|---|---|
| `_evaluate_learner_metrics` in `run_experiment.py` | `eval_policy_rollout(learner, venv, n_episodes=20, expert_policy=expert)` |
| `env_baselines.compute_baselines` — expert return | `eval_policy_rollout(expert, venv, n_episodes=20)` |
| `env_baselines.compute_random_return` | keep as-is (random action sampler, doesn't need a policy) OR swap for a `RandomPolicy` wrapper — pick whichever is smaller diff |
| `experts._train_classical_expert` — post-training check | `eval_policy_rollout(expert, venv, n_episodes=20)` |
| `experts._get_atari_expert` — quality check | `eval_policy_rollout(expert, venv, n_episodes=10)` (Atari eval is expensive) |

**Episode count unification:** all learner-side evals use `n_episodes=20`, up from 10. The expert baseline keeps 20. Noise on the plot drops by roughly √2; per-eval wall time grows by ~2× (dominated by env step cost, negligible for classical).

**Equivalence test** — `tests/experiments/test_eval_utils.py`:

```python
def test_eval_matches_sb3_evaluate_policy_cartpole():
    """eval_policy_rollout must produce the same mean_return as SB3's evaluate_policy
    on a deterministic env + seeded deterministic policy."""
    env = make_env("CartPole-v1", n_envs=1, rng=np.random.default_rng(0))
    expert = get_or_train_expert("CartPole-v1", env, cache_dir, rng, seed=0)
    ours = eval_policy_rollout(expert, env, n_episodes=20, deterministic=True)
    env.reset()  # reset so SB3 sees the same venv state
    sb3_mean, _ = evaluate_policy(expert, env, n_eval_episodes=20, deterministic=True)
    assert abs(ours.mean_return - sb3_mean) < 1e-4  # tight

def test_eval_raises_on_safety_cap():
    # construct a venv that never emits done within safety_step_cap
    with pytest.raises(RuntimeError, match="safety_step_cap"):
        eval_policy_rollout(policy, never_done_env, n_episodes=5, safety_step_cap=100)
```

The tight `1e-4` tolerance pins that our function uses Monitor-based episode returns identically to SB3 on a venv with no reward wrappers.

### 3.2 Expert trainer with convergence detection — `src/imitation/experiments/ftrl/expert_training.py` (new)

**Public entry point:** `train_classical_expert_until_converged(env_name, cache_dir, rng, seed) -> BasePolicy`. Replaces the body of `experts._train_classical_expert`.

**Algorithm (chunked training loop):**

```
chunk_size   = config["convergence"]["chunk_timesteps"]
max_total    = config["convergence"]["max_timesteps"]
min_total    = config["convergence"]["min_timesteps"]
threshold    = config["convergence"]["threshold"]          # default 0.95
patience     = config["convergence"]["patience"]           # default 5 chunks
tolerance    = 0.02                                        # normalized-return plateau tolerance
entropy_eps  = config["convergence"]["entropy_eps"]        # default 0.05, residual entropy cap

best_norm_return  = -inf
best_residual_ent = +inf
chunks_since_best = 0
total_steps       = 0

ppo = PPO("MlpPolicy", venv, **kwargs)
while total_steps < max_total:
    ppo.learn(chunk_size, reset_num_timesteps=False)
    total_steps += chunk_size

    eval_result = eval_policy_rollout(ppo.policy, eval_venv, n_episodes=20,
                                      deterministic=True, expert_policy=ppo.policy)
    norm_return  = normalize(eval_result.mean_return, ref_random, ref_expert)
    residual_ent = eval_result.rollout_cross_entropy   # using ppo as both learner and expert
                                                        # -> this is the residual entropy of
                                                        # ppo's softmax at its own argmax

    improved = norm_return > best_norm_return + tolerance \
            or residual_ent < best_residual_ent - tolerance
    if improved:
        best_norm_return, best_residual_ent = norm_return, residual_ent
        chunks_since_best = 0
    else:
        chunks_since_best += 1

    if total_steps >= min_total \
       and norm_return >= threshold \
       and residual_ent <= entropy_eps \
       and chunks_since_best >= patience:
        break   # converged

else:  # while-else: loop fell through without break
    raise RuntimeError(
        f"Expert for {env_name} failed to converge in {max_total} steps: "
        f"norm_return={norm_return:.3f} (threshold={threshold}), "
        f"residual_ent={residual_ent:.3f} (eps={entropy_eps})"
    )
```

**Convergence is hybrid AND strict:** normalized return must pass threshold, residual entropy (softmax sharpness) must pass entropy_eps, and both must have plateaued for `patience` chunks. Satisfying one is not enough. Hitting `max_total` without convergence **raises an error** — the whole point is that silent under-training is the original bug.

**New per-env config fields in `ENV_CONFIGS`** (all optional with module-level defaults):

```python
DEFAULT_CONVERGENCE = {
    "chunk_timesteps": 25_000,
    "min_timesteps":   50_000,
    "max_timesteps": 5_000_000,
    "threshold":          0.95,
    "patience":              5,
    "entropy_eps":        0.05,
}

ENV_CONFIGS["MountainCar-v0"] = {
    ...existing...
    "convergence": {"max_timesteps": 6_000_000, "chunk_timesteps": 50_000},
}

# D1 fallback: Blackjack has an intrinsic stochastic-optimal-strategy ceiling.
ENV_CONFIGS["Blackjack-v1"] = {
    ...existing...
    "convergence": {
        "threshold": 0.85,   # below default 0.95; reference expert_score is -0.06
        "entropy_eps": 0.15,  # Blackjack's optimal policy is multi-modal at edge states
        "_note": "Blackjack stochastic optimum; 0.95/0.05 physically unreachable.",
    },
}
```

**Per-env PPO kwargs**: already supported via `ppo_kwargs` in `ENV_CONFIGS`. During retraining, if specific envs still won't converge to the default threshold, I'll tune `ent_coef`, learning rate, and `n_steps` in that field. Every tuning goes into the config file (not code) so it's reproducible.

### 3.3 Cache invalidation + automated regression tests

**One-time cache wipe (manual, documented in §6 execution plan):**

```bash
rm -rf experiments/expert_cache/CartPole-v1/ \
       experiments/expert_cache/FrozenLake-v1/ \
       experiments/expert_cache/CliffWalking-v0/ \
       experiments/expert_cache/Acrobot-v1/ \
       experiments/expert_cache/MountainCar-v0/ \
       experiments/expert_cache/Taxi-v3/ \
       experiments/expert_cache/Blackjack-v1/ \
       experiments/expert_cache/LunarLander-v2/
rm -rf experiments/results/    # classical result JSONs will be regenerated
```

Atari caches are untouched (out of scope for this PR).

**New test file:** `tests/experiments/test_expert_quality.py`

```python
@pytest.mark.expensive
@pytest.mark.parametrize("env_name", CLASSICAL_ENVS)
def test_expert_converged(env_name, tmp_cache_dir):
    """Every classical expert must pass the same convergence bar it was trained to."""
    venv = make_env(env_name, n_envs=1, rng=np.random.default_rng(0))
    expert = get_or_train_expert(env_name, venv, tmp_cache_dir, rng, seed=0)
    res = eval_policy_rollout(expert, venv, n_episodes=20, deterministic=True,
                              expert_policy=expert)
    cfg = get_convergence_config(env_name)
    ref = REFERENCE_BASELINES[env_name]
    normalized = (res.mean_return - ref["random_score"]) / (ref["expert_score"] - ref["random_score"])
    assert normalized >= cfg["threshold"] - 0.05, (
        f"{env_name}: normalized return {normalized:.3f} < threshold {cfg['threshold']}"
    )
    assert res.rollout_cross_entropy <= cfg["entropy_eps"] + 0.02, (
        f"{env_name}: residual entropy {res.rollout_cross_entropy:.3f} too high "
        f"(softmax not sharp enough; argmax pathology risk)"
    )

@pytest.mark.expensive
@pytest.mark.parametrize("env_name", CLASSICAL_ENVS)
def test_expert_beats_bc(env_name, tmp_cache_dir):
    """After retraining, expert deterministic return must meet-or-exceed BC's
    deterministic return. Pins P3 in place so it can't silently regress."""
    venv = make_env(env_name, n_envs=1, rng=np.random.default_rng(0))
    expert = get_or_train_expert(env_name, venv, tmp_cache_dir, rng, seed=0)
    # Small BC run (1 seed, reduced rounds) to keep test time bounded.
    bc_policy = _train_small_bc(env_name, venv, expert, rng=np.random.default_rng(0))
    expert_res = eval_policy_rollout(expert, venv, n_episodes=20, deterministic=True)
    bc_res     = eval_policy_rollout(bc_policy, venv, n_episodes=20, deterministic=True)
    # Allow 5% noise band.
    assert expert_res.mean_return >= bc_res.mean_return * 0.95, (
        f"{env_name}: expert {expert_res.mean_return:.2f} < BC {bc_res.mean_return:.2f} "
        f"(argmax pathology or undertrained expert still present)"
    )
```

Tests are marked `@pytest.mark.expensive` so the fast CI loop doesn't pay for them on every commit. They run in the final validation phase before merge.

### 3.4 BC+DAgger baseline — `algo="bc_dagger"` in `run_experiment.py`

**Algorithm:**

```python
def _run_bc_dagger(config, venv, expert_policy, rng, baselines):
    total_timesteps = config.n_rounds * config.samples_per_round

    # Pre-collect the full expert dataset ONCE, identically to the fixed-BC branch.
    all_transitions = collect_expert_transitions(expert_policy, venv, total_timesteps, rng)
    assert len(all_transitions) >= total_timesteps

    # Persistent policy for warm-start mode; re-created per round for from-scratch.
    warm_start = should_warm_start_bc_dagger(config.env_name)   # Atari -> True, classical -> False
    policy = None

    per_round = []
    for round_num in range(1, config.n_rounds + 1):
        k = round_num * config.samples_per_round
        assert k == _expected_ftl_ftrl_data_budget(round_num, config), \
            "BC+DAgger data budget must exactly match DAgger aggregated obs count"
        prefix = all_transitions[:k]

        if not warm_start or policy is None:
            policy = create_policy(config.policy_mode, venv, expert_policy)

        bc_trainer = bc.BC(venv.observation_space, venv.action_space,
                           rng=rng, policy=policy, demonstrations=prefix,
                           batch_size=min(32, len(prefix)),
                           custom_logger=silent_logger)
        bc_trainer.train(n_epochs=config.bc_n_epochs)

        is_first = round_num == 1
        is_interval = round_num % config.eval_interval == 0
        is_final = round_num == config.n_rounds
        round_data = {
            "round": round_num,
            "n_observations": k,
            "train_cross_entropy": _compute_train_ce(policy, prefix),
            "l2_norm": _compute_l2_norm(policy),
            "total_loss": _compute_train_ce(policy, prefix),
            "rollout_cross_entropy": None,
            "normalized_return": None,
            "disagreement_rate": None,
        }
        if is_first or is_interval or is_final:
            eval_res = eval_policy_rollout(policy, venv, n_episodes=20,
                                           deterministic=True,
                                           expert_policy=expert_policy)
            round_data["normalized_return"] = normalize(eval_res.mean_return, baselines)
            round_data["disagreement_rate"] = eval_res.disagreement_rate
            round_data["rollout_cross_entropy"] = eval_res.rollout_cross_entropy
        per_round.append(round_data)
    return per_round
```

**Data budget invariant.** At round k, BC+DAgger has exactly `k × samples_per_round` training observations, which equals the total number of transitions aggregated by FTL+DAgger / FTRL+DAgger at the same round (since each DAgger round collects `samples_per_round` transitions and the aggregated dataset grows by exactly that amount). The `assert k == _expected_ftl_ftrl_data_budget(...)` line pins the invariant at runtime. A unit test asserts the same thing.

**Warm-start vs from-scratch decision:**

- **Classical envs**: from scratch at each round. Cheap (seconds per training), clean ERM on prefix, no path-dependence.
- **Atari envs**: warm-start (initialize from previous round's weights, continue training on the new prefix for `bc_n_epochs` epochs). Cheaper by roughly `n_eval_points`× on Atari where BC-CNN training is the expensive step.
- Controlled by `should_warm_start_bc_dagger()` which defaults to `is_atari(env_name)`. Can be overridden per env in `ENV_CONFIGS["<env>"]["bc_dagger_warm_start"]`. A CLI flag `--bc-dagger-warm-start / --bc-dagger-from-scratch` forces one mode for ad-hoc experiments.

**Cross-entropy storage.** Two CE columns in the JSON:
- `train_cross_entropy`: CE of learner on its own training data (expert data for BC/BC+DAgger, aggregated DAgger data for FTL/FTRL). Kept for debugging; not plotted.
- `rollout_cross_entropy`: CE of learner on fresh transitions from its own rollout, with expert's action as target. This is the plotted metric for all "dynamic" algos (FTL+DAgger, FTRL+DAgger, BC+DAgger).

### 3.5 Plot relabeling + new rollout-CE metric — `plot_results.py`

**Color / linestyle table:**

| JSON algo key | Legend label | Linestyle | Color | On loss subplot? | On return subplot? | On disagreement subplot? |
|---|---|---|---|---|---|---|
| `ftl` | FTL+DAgger | solid | #1f77b4 blue | ✓ | ✓ | ✓ |
| `ftrl` | FTRL+DAgger | solid | #d62728 red | ✓ | ✓ | ✓ |
| `bc_dagger` | BC+DAgger | solid | #2ca02c green | ✓ | ✓ | ✓ |
| `bc` | BC (fixed) | **dashed** | #17a663 dark green | — | ✓ (reference) | ✓ |
| `expert` | Expert | **dashed** | gray | — | ✓ (reference at y=1.0) | — |

**Loss subplot metric:** switch from `cross_entropy` (which was inconsistent across algos) to `rollout_cross_entropy` for all three plotted algos. The old `cross_entropy`/`total_loss` fields remain in the JSONs as `train_cross_entropy` for debugging.

**Figure subtitle** (added):

> *Loss subplot: cross-entropy of learner on its own rollout distribution (expert's action as target). Return subplot: mean deterministic rollout return, normalized to [random=0, expert=1]. BC trains on a fixed expert dataset; BC+DAgger trains on an expert-data prefix that grows with round to match DAgger's observation budget; FTL/FTRL+DAgger train on DAgger-aggregated rollout transitions. All returns are evaluated by rolling out the learner being plotted.*

Expert is NOT shown on the loss subplot (its rollout CE against itself is the same residual entropy already monitored in the convergence trainer; dashed horizontal line would be a misleading noisy constant). Fixed BC is also not shown on the loss subplot — it's a reference, not a subject of distribution-shift comparison.

---

## 4. Files touched

**New:**
- `src/imitation/experiments/ftrl/eval_utils.py`
- `src/imitation/experiments/ftrl/expert_training.py`
- `tests/experiments/test_eval_utils.py`
- `tests/experiments/test_expert_quality.py`
- `docs/superpowers/specs/2026-04-10-ftrl-eval-distribution-fix.md` (this file)

**Modified:**
- `src/imitation/experiments/ftrl/run_experiment.py` — replace `_evaluate_learner_metrics` and `_evaluate_policy_cross_entropy` with calls into `eval_utils`; add `_run_bc_dagger`; extend `ALL_ALGOS` with `"bc_dagger"`; ensure `_run_dagger_variant` records `rollout_cross_entropy` and keeps old `cross_entropy` as `train_cross_entropy`.
- `src/imitation/experiments/ftrl/experts.py` — `_train_classical_expert` delegates to `expert_training.train_classical_expert_until_converged`.
- `src/imitation/experiments/ftrl/env_utils.py` — add `convergence` sub-dicts per env; add `DEFAULT_CONVERGENCE`.
- `src/imitation/experiments/ftrl/env_baselines.py` — `compute_baselines` and `compute_random_return` call into `eval_utils` for the policy-rollout path; schema of `baselines.json` unchanged.
- `src/imitation/experiments/ftrl/plot_results.py` — label/dash/color table, loss-subplot metric rename, subtitle.

**Untouched:**
- `src/imitation/algorithms/ftrl.py` (FTRLTrainer internals unchanged; any new per-round metric is computed outside it)
- Atari expert pipeline (out of scope)
- All non-FTRL experiment code

---

## 5. Acceptance criteria

Before merge, all must hold:

1. `pytest tests/experiments/test_eval_utils.py` passes — eval function matches SB3 to 1e-4 on CartPole.
2. `pytest -m expensive tests/experiments/test_expert_quality.py` passes on all 8 classical envs — every expert converged and every expert ≥ 0.95 × BC return.
3. All classical experts' cached `baselines.json` shows `expert_return` within reach of `REFERENCE_BASELINES[env]["expert_score"]` (≥ 0.90).
4. `run_experiment.py --env-group classical --algos ftl ftrl bc bc_dagger --seeds 3` completes without errors.
5. `plot_results.py` generates per-env figures with: solid lines for FTL+DAgger / FTRL+DAgger / BC+DAgger, dashed lines for BC and Expert, correct legend, subtitle rendered, loss subplot using `rollout_cross_entropy`, expert line at y=1.0 on the return subplot.
6. Every classical env's expert_cross_entropy on its own rollouts (= residual entropy) is ≤ 0.05, ruling out argmax pathology visually.
7. Full test suite (`pytest -n auto tests/ -m "not expensive"`) green.
8. No regression on Atari results — Atari experts and JSONs unchanged; if anyone reruns Atari, results should be identical within noise.

---

## 6. Local execution plan

Server is down. We run classical retraining + experiments locally.

**Pre-run cleanup (from worktree root):**

```bash
cd /Users/thangduong/Desktop/imitation/.worktrees/ftrl-expert-fix
rm -rf experiments/expert_cache/CartPole-v1 \
       experiments/expert_cache/FrozenLake-v1 \
       experiments/expert_cache/CliffWalking-v0 \
       experiments/expert_cache/Acrobot-v1 \
       experiments/expert_cache/MountainCar-v0 \
       experiments/expert_cache/Taxi-v3 \
       experiments/expert_cache/Blackjack-v1 \
       experiments/expert_cache/LunarLander-v2
rm -rf experiments/results/CartPole-v1 \
       experiments/results/FrozenLake-v1 \
       experiments/results/CliffWalking-v0 \
       experiments/results/Acrobot-v1 \
       experiments/results/MountainCar-v0 \
       experiments/results/Taxi-v3 \
       experiments/results/Blackjack-v1 \
       experiments/results/LunarLander-v2
```

(Atari caches preserved.)

**Retrain experts (will happen automatically on first experiment run, but can be pre-warmed):**

```bash
python -m imitation.experiments.ftrl.run_experiment \
    --env-group classical --algos ftl --seeds 1 --n-rounds 2 --n-workers 1
```

This triggers expert training for all 8 classical envs sequentially via the pre-training pass in `main()`. Training may take hours on CPU for MountainCar / Taxi — run in tmux.

**Full experiment run:**

```bash
python -m imitation.experiments.ftrl.run_experiment \
    --env-group classical \
    --algos ftl ftrl bc bc_dagger \
    --seeds 5 \
    --n-workers 4
```

Local CPU count dictates `--n-workers`. Classical experts are fast enough after convergence that this should complete overnight.

**Plots:**

```bash
python -m imitation.experiments.ftrl.plot_results \
    --results-dir experiments/results/ \
    --envs CartPole-v1 FrozenLake-v1 CliffWalking-v0 Acrobot-v1 \
           MountainCar-v0 Taxi-v3 Blackjack-v1 LunarLander-v2
```

---

## 7. Server sync checklist (when CC-server is back)

1. `git push -u origin feature/ftrl-expert-fix-bcdagger` from the local worktree.
2. On the server (under `tmux`): `cd imitation && git fetch && git checkout feature/ftrl-expert-fix-bcdagger`.
3. `conda activate <env>`; `pip install -e ".[dev]"` if dependencies changed.
4. **Retrain experts on the server, do NOT ship local caches.** Reason: environment RNG / numerical details may differ; the test `test_expert_converged` should pass on the server independently.
   ```bash
   rm -rf experiments/expert_cache/<classical_envs> experiments/results/<classical_envs>
   ```
5. Run `pytest -m expensive tests/experiments/test_expert_quality.py` on the server first. Must pass before running full experiments.
6. Run full classical experiments at higher seed count (5 → 10) and, if time allows, Atari.
7. Pull results: `./experiments/sync_results.sh pull` (locally).
8. Regenerate plots locally.
9. **Do not commit server details or server-specific paths anywhere.** Global rule — this file already avoids them.

---

## 8. Out of scope

- Atari expert retraining / convergence detection (Atari experts come from HuggingFace hub or separate CnnPolicy training; this PR keeps them as-is).
- Refactoring `FTRLTrainer` internals.
- Adding new environments beyond the 8 classical MDPs.
- Changing the disagreement_rate metric definition.
- Changing the normalized-return normalization formula.
- Any modification to `src/imitation/algorithms/*` outside of what's strictly needed.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Expert retraining for some env (MountainCar, Taxi) doesn't hit 0.95 threshold even with tuned PPO kwargs | Medium | D1 fallback: lower that env's threshold explicitly in `ENV_CONFIGS` with a loud comment. Don't silently pass. |
| Local CPU too slow for expert retraining on MountainCar / Taxi | Medium | Bump `--n-workers`, run in tmux, split by env group. Document expected wall-clock in §6 as we measure it. |
| Equivalence test's 1e-4 tolerance too tight | Low | Relax to 1e-3 if Monitor handling has unexpected float drift. Document the number so drift is visible. |
| BC+DAgger warm-start mode gives visibly different curves than from-scratch on classical | Low | We run classical in from-scratch mode, so not a concern. If someone flips the flag, they see the difference. |
| Residual-entropy convergence signal fails for high-dim action spaces | Not applicable here | All classical envs are discrete action; Gaussian policies can be added later if needed. |
| BC+DAgger solid line overlaps with fixed BC's dashed line in some envs (visually confusing) | Low | Distinct shades of green; legend disambiguates. |
