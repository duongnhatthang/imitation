"""E_rec: measure MDP recoverability and relate it to the FTRL advantage (H5).

Recoverability = how gracefully expert return degrades when the agent takes
occasional *wrong* (uniform-random) actions. In a recoverable MDP (e.g. a
long-horizon swing-up task) a wrong action is easily corrected, so return stays
near expert even under perturbation; in a brittle MDP a single mistake is
costly or terminal and return collapses toward random.

This tests the professor's hypothesis that Acrobot is special because it is
unusually *recoverable*: a DAgger learner can drift far off the expert's state
distribution (large rollout cross-entropy) without losing return, which is
exactly the regime where a hedged (FTRL) policy beats an over-confident (FTL)
one. Prediction: Acrobot has the highest recoverability score, and
recoverability correlates with FTRL's advantage over FTL.

Method (no environment state-setting required): roll out an "epsilon-perturbed
expert" that takes a uniform-random action with probability ``eps`` and the
expert action otherwise, for several ``eps`` values. The recoverability score
at ``eps`` is the fraction of expert-over-random performance retained:

    rec(eps) = (R_perturbed(eps) - R_random) / (R_expert - R_random)

rec close to 1 means perturbations barely hurt (recoverable); rec dropping fast
toward 0 means mistakes are costly (brittle).
"""
import argparse
import csv
import os
from typing import Dict, List, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class EpsilonExpert:
    """Expert wrapper that takes a uniform-random action w.p. ``eps``.

    Implements the minimal ``predict`` interface consumed by
    ``eval_utils.eval_policy_rollout`` (returns ``(actions, state)``).
    """

    def __init__(self, expert, action_space, eps: float, rng: np.random.Generator):
        """Builds an EpsilonExpert.

        Args:
            expert: Base policy with ``predict(obs, deterministic=True)``.
            action_space: Env action space used to sample random actions.
            eps: Probability of replacing the expert action with a random one.
            rng: Random generator for the perturbation coin and random actions.
        """
        self.expert = expert
        self.action_space = action_space
        self.eps = float(eps)
        self.rng = rng

    def predict(self, obs, deterministic: bool = True):
        """Return per-env actions, randomized with probability ``eps``."""
        act, _ = self.expert.predict(obs, deterministic=True)
        act = np.asarray(act)
        if self.eps > 0.0:
            mask = self.rng.random(act.shape[0]) < self.eps
            if mask.any():
                rand = np.array(
                    [self.action_space.sample() for _ in range(act.shape[0])],
                    dtype=act.dtype,
                )
                act = np.where(mask, rand, act)
        return act, None


def recoverability_scores(
    r_expert: float,
    r_random: float,
    r_perturbed: Dict[float, float],
) -> Dict[float, float]:
    """Normalize perturbed returns into recoverability fractions.

    Args:
        r_expert: Mean return of the unperturbed expert.
        r_random: Mean return of the uniform-random policy.
        r_perturbed: Map of eps -> mean return of the eps-perturbed expert.

    Returns:
        Map of eps -> (R_perturbed - R_random) / (R_expert - R_random). Returns
        NaN for every eps when the expert/random span is ~0 (undefined).
    """
    span = r_expert - r_random
    out: Dict[float, float] = {}
    for eps, r_p in r_perturbed.items():
        out[eps] = (r_p - r_random) / span if abs(span) > 1e-9 else float("nan")
    return out


def _mean_return(policy, venv, n_episodes: int) -> float:
    from imitation.experiments.ftrl.eval_utils import eval_policy_rollout

    return float(
        eval_policy_rollout(
            policy, venv, n_episodes=n_episodes, deterministic=True
        ).mean_return
    )


def profile_env(
    env_name: str,
    cache_dir: str,
    seed: int,
    eps_list: List[float],
    n_episodes: int,
) -> Dict[str, float]:
    """Compute the recoverability profile for one environment."""
    from imitation.experiments.ftrl import env_utils, experts

    rng = np.random.default_rng(seed)
    venv = env_utils.make_env(env_name, n_envs=1, rng=rng)
    try:
        expert = experts.get_or_train_expert(env_name, venv, cache_dir, rng, "cpu")
        r_expert = _mean_return(expert, venv, n_episodes)
        r_random = _mean_return(
            EpsilonExpert(expert, venv.action_space, 1.0, rng), venv, n_episodes
        )
        r_perturbed = {
            eps: _mean_return(
                EpsilonExpert(expert, venv.action_space, eps, rng), venv, n_episodes
            )
            for eps in eps_list
        }
        rec = recoverability_scores(r_expert, r_random, r_perturbed)
        row: Dict[str, float] = {
            "env": env_name,
            "R_expert": round(r_expert, 3),
            "R_random": round(r_random, 3),
        }
        for eps in eps_list:
            row[f"R_eps{eps}"] = round(r_perturbed[eps], 3)
            row[f"rec_eps{eps}"] = round(rec[eps], 4)
        # Area-under-curve recoverability: mean rec over the swept eps grid.
        row["rec_auc"] = round(
            float(np.nanmean([rec[eps] for eps in eps_list])), 4
        )
        return row
    finally:
        venv.close()


def main():
    parser = argparse.ArgumentParser(description="Recoverability profiling (E_rec)")
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--cache-dir", default="experiments/expert_cache")
    parser.add_argument("--output-dir", default="experiments/recoverability")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument(
        "--eps", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.4]
    )
    args = parser.parse_args()

    from imitation.experiments.ftrl import env_utils

    envs = args.envs or list(env_utils.ENV_GROUPS["classical"])
    os.makedirs(args.output_dir, exist_ok=True)

    rows = [
        profile_env(e, args.cache_dir, args.seed, args.eps, args.n_episodes)
        for e in envs
    ]

    fields = list(rows[0].keys())
    with open(os.path.join(args.output_dir, "recoverability.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    plt.figure(figsize=(9, 6))
    for r in rows:
        ys = [r[f"rec_eps{eps}"] for eps in args.eps]
        color = "crimson" if r["env"].startswith("Acrobot") else None
        lw = 3 if r["env"].startswith("Acrobot") else 1.5
        plt.plot(args.eps, ys, marker="o", label=r["env"], color=color, linewidth=lw)
    plt.axhline(1.0, color="grey", lw=0.6, ls="--")
    plt.xlabel("perturbation rate eps (fraction of random/'wrong' actions)")
    plt.ylabel("recoverability = retained expert performance")
    plt.title("MDP recoverability vs perturbation rate (Acrobot in red)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "recoverability.png"), dpi=130)
    plt.close()
    print(f"Wrote recoverability profiles to {args.output_dir}")
    for r in rows:
        print(f"  {r['env']:18} rec_auc={r['rec_auc']}  rec@0.1={r.get('rec_eps0.1')}")


if __name__ == "__main__":
    main()
