"""E_env: characterize each classical MDP to test H4 (near-tied optimal actions).

For each env, load the cached expert, roll it out, and measure the expert's
action-probability margin (top1 - top2) per state on the expert's own rollout
states and on a frozen-linear learner's rollout states. Acrobot is predicted to
have the largest fraction of small-margin (action-ambiguous) states.
"""

import argparse
import csv
import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch as th


def action_margins(policy, obs: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """Per-state top1-minus-top2 action probability under ``policy``.

    Args:
        policy: A policy with a ``device`` attribute and ``get_distribution``
            method returning an object whose ``.distribution.probs`` is a
            (batch, n_actions) tensor of action probabilities.
        obs: Observation array, shape (N, *obs_shape).
        batch_size: Number of observations to process per forward pass.

    Returns:
        Array of shape (N,) with the margin (top1_prob - top2_prob) per state.
        If there is only one action, the margin equals the top-1 probability.
    """
    device = policy.device
    out: List[np.ndarray] = []
    n = int(np.asarray(obs).shape[0])
    with th.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = th.as_tensor(np.asarray(obs)[start:end]).to(device)
            probs = policy.get_distribution(batch).distribution.probs
            top2 = th.topk(probs, k=min(2, probs.shape[1]), dim=1).values
            if top2.shape[1] == 1:
                margin = top2[:, 0]
            else:
                margin = top2[:, 0] - top2[:, 1]
            out.append(margin.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,), dtype=float)


def small_margin_fraction(margins: np.ndarray, threshold: float) -> float:
    """Fraction of states with action margin below ``threshold``.

    Args:
        margins: Array of per-state action margins, shape (N,).
        threshold: Upper bound (exclusive) for "small" margin.

    Returns:
        Fraction of states whose margin is strictly below ``threshold``.
        Returns ``float('nan')`` when ``margins`` is empty.
    """
    if margins.size == 0:
        return float("nan")
    return float(np.mean(margins < threshold))


def _profile_one_env(env_name: str, cache_dir: str, seed: int) -> dict:
    """Profile a single environment: compute expert action-margin distributions.

    Args:
        env_name: Gymnasium environment ID.
        cache_dir: Path to the expert model cache directory.
        seed: Random seed.

    Returns:
        Dictionary with taxonomy fields and computed margin statistics,
        plus a ``_margins_learner`` key for histogram plotting.
    """
    import pathlib

    import gymnasium as gym

    from imitation.experiments.ftrl import env_utils, eval_utils, experts, policy_utils

    rng = np.random.default_rng(seed)
    venv = env_utils.make_env(env_name, n_envs=1, rng=rng)

    expert_policy = experts.get_or_train_expert(
        env_name,
        venv,
        cache_dir=pathlib.Path(cache_dir),
        rng=rng,
        seed=seed,
    )

    # Expert-state rollout: pass expert_policy so rollout_batch.obs is populated.
    er = eval_utils.eval_policy_rollout(
        expert_policy,
        venv,
        n_episodes=50,
        deterministic=True,
        expert_policy=expert_policy,
    )
    expert_obs = er.rollout_batch.obs

    # Frozen-linear-learner rollout (untrained head -> exercises shifted states).
    learner = policy_utils.create_linear_policy(expert_policy)
    lr = eval_utils.eval_policy_rollout(
        learner,
        venv,
        n_episodes=50,
        deterministic=True,
        expert_policy=expert_policy,
    )
    learner_obs = lr.rollout_batch.obs

    m_exp = action_margins(expert_policy, expert_obs)
    m_learn = action_margins(expert_policy, learner_obs)

    space = venv.observation_space
    obs_type = "discrete" if isinstance(space, gym.spaces.Discrete) else "box"
    n_actions = int(venv.action_space.n)
    return {
        "env": env_name,
        "obs_type": obs_type,
        "obs_dim": int(np.prod(space.shape)) if space.shape else 0,
        "n_actions": n_actions,
        "small_margin_frac_expert_lt0.1": small_margin_fraction(m_exp, 0.1),
        "small_margin_frac_learner_lt0.1": small_margin_fraction(m_learn, 0.1),
        "small_margin_frac_learner_lt0.25": small_margin_fraction(m_learn, 0.25),
        "_margins_learner": m_learn,
    }


def main() -> None:
    """CLI entry point: profile envs and write CSV + histogram PNG.

    Writes:
        - ``<output_dir>/env_profiles.csv``: taxonomy and margin stats per env.
        - ``<output_dir>/action_margin_hist.png``: one subplot per env with
          Acrobot highlighted in crimson.
    """
    parser = argparse.ArgumentParser(description="Profile classical envs (E_env)")
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--cache-dir", default="experiments/expert_cache")
    parser.add_argument("--output-dir", default="experiments/env_profile")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from imitation.experiments.ftrl import env_utils

    envs = args.envs or list(env_utils.ENV_GROUPS["classical"])
    os.makedirs(args.output_dir, exist_ok=True)

    rows = [_profile_one_env(e, args.cache_dir, args.seed) for e in envs]

    fields = [k for k in rows[0] if not k.startswith("_")]
    csv_path = os.path.join(args.output_dir, "env_profiles.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5), squeeze=False)
    for ax, r in zip(axes[0], rows):
        color = "crimson" if r["env"].startswith("Acrobot") else "steelblue"
        ax.hist(r["_margins_learner"], bins=20, range=(0, 1), color=color)
        ax.set_title(r["env"], fontsize=8)
        ax.set_xlabel("expert action margin\n(learner-rollout states)", fontsize=7)
    fig.suptitle("Expert action-margin distributions (Acrobot in red)")
    fig.tight_layout()
    hist_path = os.path.join(args.output_dir, "action_margin_hist.png")
    fig.savefig(hist_path, dpi=130)
    plt.close(fig)
    print(f"Wrote env profiles to {args.output_dir}")


if __name__ == "__main__":
    main()
