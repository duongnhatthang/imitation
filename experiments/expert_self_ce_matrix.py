"""One-off experiment: measure the PPO/A2C self-CE floor on D_offline.

Answers the question "does PPO's ratio clipping (not just its entropy
bonus) cause the residual self-CE on classical MDPs?" by training four
expert variants and reporting each policy's cross-entropy on its own
argmax-rollout distribution.

Matrix (per env, per seed):
    row 1: PPO, clip_range=0.2,  ent_coef=0.01   (baseline, matches d3b8dd7)
    row 2: PPO, clip_range=0.2,  ent_coef=0.0    (d3b8dd7 repro)
    row 3: PPO, clip_range=1e9,  ent_coef=0.0    (clipping effectively off)
    row 4: A2C,                  ent_coef=0.0    (no PPO clip mechanism)

Envs: CartPole-v1, Acrobot-v1 (fast classical MDPs).
Seeds: 3.

This file lives in experiments/ because it is a one-off analysis, not
part of the main runner. See also experiments/ent_coef_zero_experiment.py
(the d3b8dd7 precursor).
"""

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np

from imitation.experiments.ftrl import env_utils, expert_training
from imitation.experiments.ftrl.eval_utils import eval_policy_rollout

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_ENVS = ["CartPole-v1", "Acrobot-v1"]
SEEDS = [0, 1, 2]
CACHE_ROOT = pathlib.Path("experiments/expert_cache_matrix")
OUTPUT_PATH = pathlib.Path("experiments/logs/expert_self_ce_matrix.json")

# Disable the self_ce_eps convergence gate for the matrix: we want to
# MEASURE the self_ce floor, not gate training on it. Only norm_return
# plateau decides convergence.
NO_SELF_CE_GATE: Dict[str, float] = {"self_ce_eps": 1.0e9}

# Row definitions. `kwargs` is merged into the trainer's kwargs.
ROWS: List[Dict[str, Any]] = [
    {
        "row": 1,
        "trainer": "PPO",
        "label": "ppo_clip0.2_ent0.01",
        "kwargs": {"clip_range": 0.2, "ent_coef": 0.01},
    },
    {
        "row": 2,
        "trainer": "PPO",
        "label": "ppo_clip0.2_ent0.0",
        "kwargs": {"clip_range": 0.2, "ent_coef": 0.0},
    },
    {
        "row": 3,
        "trainer": "PPO",
        "label": "ppo_clip1e9_ent0.0",
        "kwargs": {"clip_range": 1.0e9, "ent_coef": 0.0},
    },
    {
        "row": 4,
        "trainer": "A2C",
        "label": "a2c_ent0.0",
        "kwargs": {"ent_coef": 0.0},
    },
]



def _train_a2c_until_converged(
    env_name: str,
    rng: np.random.Generator,
    seed: int,
    ent_coef: float,
    convergence_override: Dict[str, float],
) -> Optional[object]:
    """A2C analog of train_classical_expert_until_converged.

    Chunked-learn + plateau convergence. Returns the trained
    `model.policy` or None on failure-to-converge.
    """
    import collections

    from stable_baselines3 import A2C

    from imitation.experiments.ftrl.env_baselines import REFERENCE_BASELINES

    conv = dict(convergence_override)
    chunk_size = int(conv["chunk_timesteps"])
    max_total = int(conv["max_timesteps"])
    min_total = int(conv["min_timesteps"])
    threshold = float(conv["threshold"])
    patience = int(conv["patience"])
    self_ce_eps = float(conv["self_ce_eps"])
    tolerance = 0.02

    train_venv = env_utils.make_env(env_name, n_envs=1, rng=rng)
    eval_venv = env_utils.make_env(env_name, n_envs=1, rng=rng)

    model = A2C(
        "MlpPolicy",
        train_venv,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=0,
        ent_coef=ent_coef,
    )

    ref = REFERENCE_BASELINES[env_name]
    score_range = ref["expert_score"] - ref["random_score"]

    def _norm(mean_return: float) -> float:
        if abs(score_range) < 1e-8:
            return 0.0
        return (mean_return - ref["random_score"]) / score_range

    best_norm = -float("inf")
    best_self_ce = float("inf")
    chunks_since_best = 0
    total_steps = 0
    return_window: "collections.deque[float]" = collections.deque(maxlen=patience)

    try:
        while total_steps < max_total:
            model.learn(chunk_size, reset_num_timesteps=False)
            total_steps += chunk_size

            eval_res = eval_policy_rollout(
                model.policy,
                eval_venv,
                n_episodes=50,
                deterministic=True,
                expert_policy=model.policy,
            )
            norm_ret = _norm(eval_res.mean_return)
            self_ce = float(eval_res.current_round_ce)
            return_window.append(norm_ret)

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

            window_min = min(return_window) if return_window else -float("inf")
            logger.info(
                f"[A2C {env_name}] step={total_steps}/{max_total} "
                f"norm_ret={norm_ret:.3f} self_ce={self_ce:.3f} "
                f"(best norm={best_norm:.3f} ce={best_self_ce:.3f}, "
                f"window_min={window_min:.3f}, "
                f"patience {chunks_since_best}/{patience})"
            )

            if (
                total_steps >= min_total
                and len(return_window) >= patience
                and window_min >= threshold
                and self_ce <= self_ce_eps
                and chunks_since_best >= patience
            ):
                return model.policy
    finally:
        train_venv.close()
        eval_venv.close()

    logger.warning(
        f"A2C {env_name} did not converge in {max_total} steps "
        f"(norm_ret={norm_ret:.3f} threshold={threshold})"
    )
    return None


def _run_ppo_cell(
    env_name: str,
    row: Dict[str, Any],
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Train one PPO cell and return {normalized_return, self_ce_on_Doffline}.

    Temporarily patches `env_utils.ENV_CONFIGS[env_name]['ppo_kwargs']`
    with the row's overrides, reuses
    `expert_training.train_classical_expert_until_converged` with the
    `NO_SELF_CE_GATE` convergence override, then restores the config.

    Returns None if the cell failed to converge (caller logs a warning
    and skips).
    """
    cfg = env_utils.ENV_CONFIGS[env_name]
    orig_ppo_kwargs = dict(cfg.get("ppo_kwargs", {}))
    new_ppo_kwargs = dict(orig_ppo_kwargs)
    new_ppo_kwargs.update(row["kwargs"])
    cfg["ppo_kwargs"] = new_ppo_kwargs

    # Build convergence override: inherit env defaults, then disable
    # the self_ce gate so we can MEASURE the floor without gating on it.
    conv = env_utils.get_convergence_config(env_name)
    conv.update(NO_SELF_CE_GATE)

    cache_dir = CACHE_ROOT / row["label"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        try:
            policy = expert_training.train_classical_expert_until_converged(
                env_name=env_name,
                cache_dir=cache_dir,
                rng=np.random.default_rng(seed),
                seed=seed,
                convergence_override=conv,
            )
        except RuntimeError as e:
            logger.warning(
                f"PPO cell failed to converge: "
                f"env={env_name} row={row['row']} seed={seed}: {e}"
            )
            return None

        venv = env_utils.make_env(
            env_name, n_envs=1, rng=np.random.default_rng(seed + 10_000)
        )
        try:
            res = eval_policy_rollout(
                policy,
                venv,
                n_episodes=50,
                deterministic=True,
                expert_policy=policy,
            )
        finally:
            venv.close()
    finally:
        cfg["ppo_kwargs"] = orig_ppo_kwargs

    from imitation.experiments.ftrl.env_baselines import REFERENCE_BASELINES
    ref = REFERENCE_BASELINES[env_name]
    score_range = ref["expert_score"] - ref["random_score"]
    norm_ret = (
        0.0
        if abs(score_range) < 1e-8
        else (res.mean_return - ref["random_score"]) / score_range
    )

    return {
        "normalized_return": float(norm_ret),
        "mean_return": float(res.mean_return),
        "self_ce_on_Doffline": float(res.current_round_ce),
    }


def _run_a2c_cell(
    env_name: str,
    row: Dict[str, Any],
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Train one A2C cell and return {normalized_return, self_ce_on_Doffline}."""
    conv = env_utils.get_convergence_config(env_name)
    conv.update(NO_SELF_CE_GATE)

    policy = _train_a2c_until_converged(
        env_name=env_name,
        rng=np.random.default_rng(seed),
        seed=seed,
        ent_coef=row["kwargs"]["ent_coef"],
        convergence_override=conv,
    )
    if policy is None:
        return None

    venv = env_utils.make_env(
        env_name, n_envs=1, rng=np.random.default_rng(seed + 10_000)
    )
    try:
        res = eval_policy_rollout(
            policy,
            venv,
            n_episodes=50,
            deterministic=True,
            expert_policy=policy,
        )
    finally:
        venv.close()

    from imitation.experiments.ftrl.env_baselines import REFERENCE_BASELINES
    ref = REFERENCE_BASELINES[env_name]
    score_range = ref["expert_score"] - ref["random_score"]
    norm_ret = (
        0.0
        if abs(score_range) < 1e-8
        else (res.mean_return - ref["random_score"]) / score_range
    )

    return {
        "normalized_return": float(norm_ret),
        "mean_return": float(res.mean_return),
        "self_ce_on_Doffline": float(res.current_round_ce),
    }


def main() -> None:
    logger.info("expert_self_ce_matrix: starting")
    results: List[Dict[str, Any]] = []
    for env_name in TARGET_ENVS:
        for row in ROWS:
            for seed in SEEDS:
                logger.info(
                    f"=== env={env_name} row={row['row']} "
                    f"trainer={row['trainer']} seed={seed} ==="
                )
                cell_key = {
                    "env": env_name,
                    "row": row["row"],
                    "seed": seed,
                    "trainer": row["trainer"],
                    "label": row["label"],
                    "clip_range": row["kwargs"].get("clip_range"),
                    "ent_coef": row["kwargs"].get("ent_coef"),
                }
                if row["trainer"] == "PPO":
                    metrics = _run_ppo_cell(env_name, row, seed)
                elif row["trainer"] == "A2C":
                    metrics = _run_a2c_cell(env_name, row, seed)
                else:
                    raise ValueError(f"unknown trainer {row['trainer']!r}")

                if metrics is None:
                    logger.warning(f"skipping cell {cell_key}")
                    continue

                results.append({**cell_key, **metrics})
                # Persist incrementally so a mid-run crash still keeps prior rows.
                OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(
                    f"  -> norm_ret={metrics['normalized_return']:.3f} "
                    f"self_ce={metrics['self_ce_on_Doffline']:.4f}"
                )

    logger.info(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
