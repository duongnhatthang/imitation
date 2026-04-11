"""One-off experiment: retrain the 3 high-self-CE experts with ent_coef=0.0.

Goal: verify that PPO's default ent_coef=0.01 entropy bonus is what keeps
self_ce away from zero on CartPole/LunarLander/Taxi. If we force ent_coef=0
the optimal PPO policy should become deterministic → self_ce → 0.

This file lives in experiments/ rather than src/ because it's a one-off
analysis, not part of the main runner.
"""

import json
import pathlib

import numpy as np

from imitation.experiments.ftrl import env_utils, expert_training
from imitation.experiments.ftrl.env_baselines import REFERENCE_BASELINES
from imitation.experiments.ftrl.eval_utils import eval_policy_rollout

TARGET_ENVS = ["CartPole-v1", "LunarLander-v2", "Taxi-v3"]
CACHE_DIR = pathlib.Path("experiments/expert_cache")
SEED = 0


def retrain_with_ent_coef_zero(env_name: str):
    """Temporarily patch ENV_CONFIGS[env_name]['ppo_kwargs']['ent_coef'] = 0.0,
    retrain via the convergence trainer, then restore the config.
    """
    cfg = env_utils.ENV_CONFIGS[env_name]
    orig_ppo_kwargs = dict(cfg.get("ppo_kwargs", {}))
    new_ppo_kwargs = dict(orig_ppo_kwargs)
    new_ppo_kwargs["ent_coef"] = 0.0
    cfg["ppo_kwargs"] = new_ppo_kwargs

    try:
        print(
            f"\n=== Retraining {env_name} with ent_coef=0.0 "
            f"(was {orig_ppo_kwargs.get('ent_coef', 0.01)}) ==="
        )
        policy = expert_training.train_classical_expert_until_converged(
            env_name=env_name,
            cache_dir=CACHE_DIR,
            rng=np.random.default_rng(SEED),
            seed=SEED,
        )

        venv = env_utils.make_env(
            env_name, n_envs=1, rng=np.random.default_rng(SEED + 1)
        )
        try:
            res = eval_policy_rollout(
                policy,
                venv,
                n_episodes=50,
                deterministic=True,
                expert_policy=policy,
            )
            ref = REFERENCE_BASELINES[env_name]
            norm = (res.mean_return - ref["random_score"]) / (
                ref["expert_score"] - ref["random_score"]
            )
            return {
                "env": env_name,
                "mean_return": res.mean_return,
                "normalized_return": norm,
                "self_ce": res.current_round_ce,
            }
        finally:
            venv.close()
    finally:
        cfg["ppo_kwargs"] = orig_ppo_kwargs


def load_archived_baselines(env_name: str):
    p = (
        pathlib.Path("experiments/expert_cache_ent_default")
        / env_name
        / "baselines.json"
    )
    with open(p) as f:
        return json.load(f)


def main():
    results = []
    for env_name in TARGET_ENVS:
        archived = load_archived_baselines(env_name)
        try:
            zero = retrain_with_ent_coef_zero(env_name)
        except Exception as e:
            print(f"FAILED {env_name}: {e}")
            zero = None
        results.append(
            {
                "env": env_name,
                "default_ent_coef_0.01": {
                    "expert_return": archived["expert_return"],
                    "expert_self_ce": archived["expert_self_ce"],
                },
                "ent_coef_0.0": zero,
            }
        )
        print(f"  {env_name} default : {archived}")
        print(f"  {env_name} ent_coef=0: {zero}")

    out = pathlib.Path("experiments/logs/ent_coef_zero_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    print("\n=== Summary ===")
    print(
        f"{'env':20} {'default_self_ce':>16} {'ent0_self_ce':>16} "
        f"{'default_nr':>12} {'ent0_nr':>12}"
    )
    for r in results:
        d = r["default_ent_coef_0.01"]
        z = r["ent_coef_0.0"] or {}
        ref = REFERENCE_BASELINES[r["env"]]
        d_nr = (d["expert_return"] - ref["random_score"]) / (
            ref["expert_score"] - ref["random_score"]
        )
        z_self = z.get("self_ce", float("nan"))
        z_nr = z.get("normalized_return", float("nan"))
        print(
            f'{r["env"]:20} {d["expert_self_ce"]:>16.4f} '
            f"{z_self:>16.4f} {d_nr:>12.3f} {z_nr:>12.3f}"
        )


if __name__ == "__main__":
    main()
