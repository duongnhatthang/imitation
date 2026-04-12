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
                # Body added in Task 2.
                pass

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
