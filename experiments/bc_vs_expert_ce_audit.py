"""One-off experiment: compare expert vs BC cross-entropy on D_offline.

For each (env, seed) in {CartPole-v1, Acrobot-v1} x {0,1,2}:
  1. Load the cached row-1 default PPO expert.
  2. Build D_offline via rollout.generate_trajectories(expert,
     deterministic_policy=True) — same path as _run_fixed_bc.
  3. Train a fresh vanilla BC on D_offline using the same BC
     constructor args _run_fixed_bc uses.
  4. Roll out BC to collect D_eval (with expert argmax labels at each
     visited state).
  5. Measure four cross-entropies on shared state sets via
     compute_sampled_action_ce.

See docs/superpowers/specs/2026-04-11-bc-vs-expert-ce-audit-design.md
for the interpretation matrix.
"""

import json
import logging
import pathlib
from typing import Any, Dict, List

import numpy as np

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.experiments.ftrl import env_utils, expert_training, policy_utils
from imitation.experiments.ftrl.env_baselines import REFERENCE_BASELINES
from imitation.experiments.ftrl.eval_utils import (
    compute_sampled_action_ce,
    eval_policy_rollout,
)
from imitation.util import logger as imit_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_ENVS = ["CartPole-v1", "Acrobot-v1"]
SEEDS = [0, 1, 2]
N_DOFFLINE = 3_000  # matches main pipeline: n_rounds=60 * samples_per_round=50
N_EVAL_EPISODES = 50
BC_N_EPOCHS = 20  # matches --bc-n-epochs default in run_experiment.py
BC_BATCH_SIZE = 32
EXPERT_CACHE = pathlib.Path("experiments/expert_cache")
OUTPUT_PATH = pathlib.Path("experiments/logs/bc_vs_expert_ce_audit.json")


def _audit_one(env_name: str, seed: int) -> Dict[str, Any]:
    logger.info(f"=== env={env_name} seed={seed} ===")
    rng = np.random.default_rng(seed)

    # 1. Load the cached row-1 expert (trains it if not cached).
    expert_policy = expert_training.train_classical_expert_until_converged(
        env_name=env_name,
        cache_dir=EXPERT_CACHE,
        rng=np.random.default_rng(seed),
        seed=seed,
    )

    venv = env_utils.make_env(env_name, n_envs=1, rng=rng)
    try:
        # 2. Build D_offline — same code path as _run_fixed_bc.
        sample_until = rollout.make_sample_until(
            min_timesteps=N_DOFFLINE, min_episodes=1
        )
        trajs = rollout.generate_trajectories(
            policy=expert_policy,
            venv=venv,
            sample_until=sample_until,
            deterministic_policy=True,
            rng=rng,
        )
        all_transitions = rollout.flatten_trajectories(list(trajs))
        if len(all_transitions) < N_DOFFLINE:
            raise RuntimeError(
                f"D_offline: collected {len(all_transitions)} < {N_DOFFLINE}"
            )
        all_transitions = all_transitions[:N_DOFFLINE]

        d_offline_obs = np.asarray(all_transitions.obs)
        d_offline_acts = np.asarray(all_transitions.acts, dtype=np.int64)

        # 3. Train vanilla BC on D_offline — same constructor as _run_fixed_bc.
        fresh_policy = policy_utils.create_end_to_end_policy(
            venv.observation_space, venv.action_space
        )
        custom_logger = imit_logger.configure(
            str(pathlib.Path("experiments/logs/bc_audit_tb")
                / f"{env_name}_{seed}"),
            format_strs=[],
        )
        bc_trainer = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            rng=rng,
            policy=fresh_policy,
            demonstrations=all_transitions,
            batch_size=min(BC_BATCH_SIZE, N_DOFFLINE),
            custom_logger=custom_logger,
        )
        bc_trainer.train(n_epochs=BC_N_EPOCHS)

        # 4. Collect D_eval by rolling out the BC policy with expert labels.
        eval_res = eval_policy_rollout(
            bc_trainer.policy,
            venv,
            n_episodes=N_EVAL_EPISODES,
            deterministic=True,
            expert_policy=expert_policy,
        )
        d_eval_obs = eval_res.rollout_batch.obs
        d_eval_expert_acts = eval_res.rollout_batch.expert_actions

        # Also measure expert on its own rollout distribution for a
        # direct baseline return comparison.
        expert_eval_res = eval_policy_rollout(
            expert_policy,
            venv,
            n_episodes=N_EVAL_EPISODES,
            deterministic=True,
            expert_policy=expert_policy,
        )

        # 5. Four cross-entropies on shared state sets.
        expert_ce_on_Doffline = compute_sampled_action_ce(
            expert_policy, d_offline_obs, d_offline_acts
        )
        bc_ce_on_Doffline = compute_sampled_action_ce(
            bc_trainer.policy, d_offline_obs, d_offline_acts
        )
        expert_ce_on_Deval = compute_sampled_action_ce(
            expert_policy, d_eval_obs, d_eval_expert_acts
        )
        bc_ce_on_Deval = compute_sampled_action_ce(
            bc_trainer.policy, d_eval_obs, d_eval_expert_acts
        )

    finally:
        venv.close()

    ref = REFERENCE_BASELINES[env_name]
    score_range = ref["expert_score"] - ref["random_score"]

    def _norm(mean_return: float) -> float:
        if abs(score_range) < 1e-8:
            return 0.0
        return (mean_return - ref["random_score"]) / score_range

    record = {
        "env": env_name,
        "seed": seed,
        "n_Doffline": int(d_offline_obs.shape[0]),
        "n_Deval": int(d_eval_obs.shape[0]),
        "bc_normalized_return": float(_norm(eval_res.mean_return)),
        "expert_normalized_return": float(_norm(expert_eval_res.mean_return)),
        "expert_ce_on_Doffline": float(expert_ce_on_Doffline),
        "bc_ce_on_Doffline": float(bc_ce_on_Doffline),
        "expert_ce_on_Deval": float(expert_ce_on_Deval),
        "bc_ce_on_Deval": float(bc_ce_on_Deval),
    }
    logger.info(
        f"  expert_ce(Doffline)={record['expert_ce_on_Doffline']:.4f} "
        f"bc_ce(Doffline)={record['bc_ce_on_Doffline']:.4f} "
        f"expert_ce(Deval)={record['expert_ce_on_Deval']:.4f} "
        f"bc_ce(Deval)={record['bc_ce_on_Deval']:.4f}"
    )
    return record


def main() -> None:
    logger.info("bc_vs_expert_ce_audit: starting")
    results: List[Dict[str, Any]] = []
    for env_name in TARGET_ENVS:
        for seed in SEEDS:
            record = _audit_one(env_name, seed)
            results.append(record)
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2)
    logger.info(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
