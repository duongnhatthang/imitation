"""Plot the recoverability-constant distribution mu(s) per environment.

mu is computed from a separately-trained DQN reference expert (NOT the PPO
imitation expert). The figure annotates this provenance and shows DQN vs PPO
return so both are visibly near-optimal. The horizon return J is undiscounted
and far larger than the discounted mu, so it is reported as a text note (with a
discounting caveat) rather than an on-axis line; the DAgger benefit threshold
mu(s) << J is thus read qualitatively.
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
    mu, out_path, env_name, horizon_return, dqn_return, ppo_return, dqn_return_std=None
) -> None:
    """Histogram of mu(s) with a median marker and provenance/threshold text.

    The x-axis is fit to the mu range so the (small, discounted) mu values are
    readable. The horizon return J is undiscounted and orders of magnitude larger
    than mu, so it is reported as a text note rather than an on-axis line that
    would crush the histogram.
    """
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mu = np.asarray(mu, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mu, bins=50, color="#4c72b0", alpha=0.85)
    ax.set_xlabel(r"$\mu(s) = \max_a Q^{\pi^E}(s,a) - \min_a Q^{\pi^E}(s,a)$")
    ax.set_ylabel("count (visited states)")

    # Fit the x-axis to the mu range so the bars fill the panel.
    mu_hi = float(mu.max()) if mu.size else 1.0
    ax.set_xlim(0.0, mu_hi * 1.08 if mu_hi > 0 else 1.0)

    # On-scale median marker: the "typical" recoverability of a visited state.
    if mu.size:
        mu_median = float(np.median(mu))
        ax.axvline(
            mu_median,
            color="#c44e52",
            linestyle="--",
            linewidth=1.5,
            label=f"median $\\mu$ = {mu_median:.3f}",
        )
        ax.legend(loc="upper right")

    ax.set_title(f"Recoverability constant: {env_name}")

    # J is undiscounted and off-scale vs mu -> report as a note, not an axvline.
    j_note = (
        f"horizon return J = {horizon_return:.0f} (undiscounted, off-scale $\\gg \\mu$)"
        if horizon_return is not None
        else "horizon return J: unavailable"
    )
    dqn_str = f"DQN return = {dqn_return:.0f}"
    if dqn_return_std is not None:
        dqn_str += f" +/- {dqn_return_std:.0f} (20 eps)"
    provenance = (
        "mu from separately-trained DQN reference (not the PPO IL expert)\n"
        f"{dqn_str}"
        + (
            f"   |   PPO expert return = {ppo_return:.0f}"
            if ppo_return is not None
            else ""
        )
        + f"\n{j_note}"
        + "\nmu uses discounted DQN Q-values; compare mu << J qualitatively."
        + "\nInteractive IL benefits when mu(s) << J for most s."
    )
    ax.text(
        0.5,
        -0.28,
        provenance,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.5),
    )
    fig.subplots_adjust(bottom=0.34)
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
    found = sorted(s.algo for s in states)
    expected = list(coverage_data.ALL_ALGOS)
    missing = [a for a in expected if a not in found]
    if missing:
        print(
            f"WARNING: no scratch demos for {missing} in {env_name}; "
            f"figure will only include {found}."
        )
    obs = coverage_data.pool(states).obs
    dqn = recoverability.get_or_train_dqn_reference(
        env_name, cache_dir, total_timesteps, seed
    )
    mu = recoverability.recoverability(dqn, obs)
    rets = recoverability.reference_returns(dqn, env_name)
    dqn_return = float(rets.mean())
    dqn_return_std = float(rets.std())
    ppo_return = expert_return_from_results(results_dir, env_name, seed)
    if ppo_return is not None and dqn_return < 0.9 * ppo_return:
        print(
            f"WARNING: DQN reference return ({dqn_return:.0f}) is well below the "
            f"PPO expert return ({ppo_return:.0f}) for {env_name}; the reference "
            f"may be under-trained, making mu(s) less reliable."
        )
    j_value = horizon_return if horizon_return is not None else ppo_return
    render_recoverability_figure(
        mu, out_path, env_name, j_value, dqn_return, ppo_return, dqn_return_std
    )
    return {
        "mu_mean": float(mu.mean()),
        "mu_median": float(np.median(mu)),
        "dqn_return": dqn_return,
        "dqn_return_std": dqn_return_std,
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
