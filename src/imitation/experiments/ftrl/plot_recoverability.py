"""Plot the recoverability-constant distribution mu(s) per environment.

mu is computed from a separately-trained DQN reference expert (NOT the PPO
imitation expert). The figure annotates this provenance and shows DQN vs PPO
return so both are visibly near-optimal; a reference line at the horizon return
marks the DAgger benefit threshold mu(s) << J.
"""

import argparse
import json
import pathlib
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from imitation.experiments.ftrl import coverage_data, recoverability  # noqa: E402


def expert_return_from_results(results_dir, env_name, seed) -> Optional[float]:
    """Read baselines.expert_return from any result JSON for this env."""
    env_dir = pathlib.Path(results_dir) / env_name.replace("/", "_")
    if not env_dir.is_dir():
        env_dir = pathlib.Path(results_dir) / env_name
    for jf in sorted(env_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
        except (ValueError, OSError):
            continue
        ret = data.get("baselines", {}).get("expert_return")
        if ret is not None:
            return float(ret)
    return None


def render_recoverability_figure(
    mu, out_path, env_name, horizon_return, dqn_return, ppo_return
) -> None:
    """Histogram of mu(s) with the horizon-return threshold and provenance text."""
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mu, bins=50, color="#4c72b0", alpha=0.85)
    ax.set_xlabel(r"$\mu(s) = \max_a Q^{\pi^E}(s,a) - \min_a Q^{\pi^E}(s,a)$")
    ax.set_ylabel("count (visited states)")
    if horizon_return is not None:
        ax.axvline(
            horizon_return,
            color="crimson",
            linestyle="--",
            label=f"expert / horizon return = {horizon_return:.0f}",
        )
        ax.legend()
    ax.set_title(f"Recoverability constant: {env_name}")
    provenance = (
        "mu from separately-trained DQN reference (not the PPO IL expert)\n"
        f"DQN return = {dqn_return:.0f}"
        + (
            f"   |   PPO expert return = {ppo_return:.0f}"
            if ppo_return is not None
            else ""
        )
        + "\nInteractive IL benefits when mu(s) << J for most s."
    )
    ax.text(
        0.98,
        0.97,
        provenance,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.5),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_and_plot(
    results_dir,
    env_name,
    seed,
    cache_dir,
    out_path,
    total_timesteps=None,
    horizon_return: Optional[float] = None,
) -> dict:
    """Train/load DQN reference, compute mu over visited states, render figure."""
    states = coverage_data.load_env_states(results_dir, env_name, seed)
    if not states:
        raise FileNotFoundError(
            f"No scratch demos for {env_name} seed {seed} under {results_dir}"
        )
    obs = coverage_data.pool(states).obs
    dqn = recoverability.get_or_train_dqn_reference(
        env_name, cache_dir, total_timesteps, seed
    )
    mu = recoverability.recoverability(dqn, obs)
    dqn_return = recoverability.reference_return(dqn, env_name)
    ppo_return = expert_return_from_results(results_dir, env_name, seed)
    line_value = horizon_return if horizon_return is not None else ppo_return
    render_recoverability_figure(
        mu, out_path, env_name, line_value, dqn_return, ppo_return
    )
    return {
        "mu_mean": float(mu.mean()),
        "mu_median": float(np.median(mu)),
        "dqn_return": dqn_return,
        "ppo_return": ppo_return,
    }


def main(argv: Optional[list] = None) -> None:
    """Entry point for the recoverability mu(s) distribution CLI."""
    parser = argparse.ArgumentParser(description="Recoverability mu(s) plots.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dqn-timesteps", type=int, default=None)
    parser.add_argument(
        "--horizon-return",
        type=float,
        default=None,
        help="Reference line value J; defaults to the PPO expert return from results.",
    )
    args = parser.parse_args(argv)
    out = pathlib.Path(args.output_dir) / (
        f"{args.env.replace('/', '_')}_recoverability.png"
    )
    info = build_and_plot(
        args.results_dir,
        args.env,
        args.seed,
        args.cache_dir,
        out,
        args.dqn_timesteps,
        horizon_return=args.horizon_return,
    )
    print(
        f"Wrote {out}; mu_median={info['mu_median']:.3f}, "
        f"DQN_return={info['dqn_return']:.0f}"
    )


if __name__ == "__main__":
    main()
