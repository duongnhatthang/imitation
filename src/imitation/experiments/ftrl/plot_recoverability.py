"""Plot the recoverability-constant distribution mu(s) per environment.

mu is computed via the per-family dispatch in ``recoverability.recoverability_mu``:
- Exact tabular computation via env.P for toy-text (discrete obs) envs, using
  the PPO expert policy.
- Hub DQN (sb3/dqn-<env>) for Atari environments.
- Separately-trained DQN reference for continuous classical envs.
- Blackjack-v1 is skipped (no reliable env.P due to stochastic optimum).

The figure annotates provenance and shows DQN vs PPO return where applicable.
The horizon return J is undiscounted and far larger than the discounted mu, so
it is reported as a text note rather than an on-axis line.
"""

import argparse
import json
import pathlib
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from imitation.experiments.ftrl import (  # noqa: E402
    coverage_data,
    env_utils,
    recoverability,
)


def provenance_label(env_name: str) -> str:
    """Human-readable provenance string for the mu(s) computation backend.

    Args:
        env_name: Gymnasium environment id.

    Returns:
        A short string describing how mu(s) was computed for this env family:
        - ``"skipped (no env.P)"`` for Blackjack-v1.
        - ``"hub DQN (sb3/dqn-<env>)"`` for Atari environments.
        - ``"exact via env.P (PPO expert)"`` for toy-text discrete-obs envs.
        - ``"trained DQN reference"`` for continuous classical envs.
    """
    if env_name == "Blackjack-v1":
        return "skipped (no env.P)"
    if env_utils.is_atari(env_name):
        return f"hub DQN (sb3/dqn-{env_name})"
    if env_utils.ENV_CONFIGS.get(env_name, {}).get("obs_type") == "discrete":
        return "exact via env.P (PPO expert)"
    return "trained DQN reference"


def baseline_from_results(results_dir, env_name, seed, key: str) -> Optional[float]:
    """Read baselines[key] from any result JSON for this env.

    Args:
        results_dir: Root directory with per-env per-seed JSON result files.
        env_name: Gymnasium environment id.
        seed: Experiment seed (unused beyond directory lookup).
        key: Key within the ``baselines`` dict to retrieve (e.g.
            ``"expert_return"`` or ``"random_return"``).

    Returns:
        The float value of ``baselines[key]`` from the first matching JSON,
        or ``None`` if no file or key is found.
    """
    env_dir = pathlib.Path(results_dir) / env_name.replace("/", "_")
    if not env_dir.is_dir():
        env_dir = pathlib.Path(results_dir) / env_name
    for jf in sorted(env_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
        except (ValueError, OSError):
            continue
        ret = data.get("baselines", {}).get(key)
        if ret is not None:
            return float(ret)
    return None


def expert_return_from_results(results_dir, env_name, seed) -> Optional[float]:
    """Read baselines.expert_return from any result JSON for this env."""
    return baseline_from_results(results_dir, env_name, seed, "expert_return")


def dqn_gate_warning(
    dqn_return: float,
    random_return: Optional[float],
    expert_return: Optional[float],
    env_name: str,
) -> Optional[str]:
    """Return a warning string if the DQN reference fails the near-optimal gate.

    Uses the normalized return ``(dqn - random) / (expert - random)`` so that
    the threshold is scale-agnostic and correct for negative-return environments.

    Args:
        dqn_return: Mean return of the DQN reference policy.
        random_return: Mean return of a random policy (baseline).
        expert_return: Mean return of the expert policy (baseline).
        env_name: Environment id, used in the warning message.

    Returns:
        ``None`` if baselines are missing or the DQN passes the gate
        (normalized return ≥ 0.90); otherwise a human-readable warning string.
    """
    if random_return is None or expert_return is None:
        return None
    norm = recoverability.normalized_return(dqn_return, random_return, expert_return)
    if norm >= 0.9:
        return None
    return (
        f"DQN reference return ({dqn_return:.0f}) is below the near-optimal gate "
        f"(normalized {norm:.2f} < 0.90; random={random_return:.0f}, "
        f"expert={expert_return:.0f}) for {env_name}; mu less reliable."
    )


def render_recoverability_figure(
    mu,
    out_path,
    env_name,
    horizon_return,
    dqn_return,
    ppo_return,
    dqn_return_std=None,
    provenance_str: Optional[str] = None,
    show_dqn_return: bool = True,
) -> None:
    """Histogram of mu(s) with a median marker and provenance/threshold text.

    The x-axis is fit to the mu range so the (small, discounted) mu values are
    readable. The horizon return J is undiscounted and orders of magnitude larger
    than mu, so it is reported as a text note rather than an on-axis line that
    would crush the histogram.

    Args:
        mu: Array of recoverability values, shape ``[N]``.
        out_path: Output path for the PNG figure.
        env_name: Gymnasium environment id (used in the title).
        horizon_return: Undiscounted horizon return J; ``None`` if unavailable.
        dqn_return: DQN reference return; shown only when ``show_dqn_return``
            is True.
        ppo_return: PPO expert return; shown when not ``None``.
        dqn_return_std: Std-dev of DQN returns over evaluation episodes.
        provenance_str: Provenance label for the annotation. Defaults to the
            legacy ``"separately-trained DQN reference"`` string.
        show_dqn_return: Whether to include the DQN return line in the
            annotation. Set to ``False`` for toy-text envs where mu comes from
            exact tabular computation, not a DQN.
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
    prov_line = (
        provenance_str
        if provenance_str is not None
        else "separately-trained DQN reference"
    )
    annotation_parts = [f"mu from {prov_line}"]
    if show_dqn_return and dqn_return is not None:
        dqn_str = f"DQN return = {dqn_return:.0f}"
        if dqn_return_std is not None:
            dqn_str += f" +/- {dqn_return_std:.0f} (20 eps)"
        if ppo_return is not None:
            dqn_str += f"   |   PPO expert return = {ppo_return:.0f}"
        annotation_parts.append(dqn_str)
    elif ppo_return is not None:
        annotation_parts.append(f"PPO expert return = {ppo_return:.0f}")
    annotation_parts.append(j_note)
    annotation_parts.append(
        "mu uses discounted Q-values; compare mu << J qualitatively."
    )
    annotation_parts.append("Interactive IL benefits when mu(s) << J for most s.")
    provenance = "\n".join(annotation_parts)
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
    expert_cache: Optional[str] = None,
) -> dict:
    """Compute mu via per-family dispatch, render figure, return summary dict.

    Dispatches mu(s) computation to the appropriate backend via
    ``recoverability.recoverability_mu``:

    - Blackjack-v1: skipped (returns ``{"skipped": True}``).
    - Toy-text (discrete obs): exact tabular mu via env.P w.r.t. the PPO
      expert loaded from ``expert_cache``.
    - Atari: hub DQN from the HuggingFace model hub.
    - Continuous classical: trained/cached DQN reference.

    Args:
        results_dir: Root directory with per-env per-seed JSON result files.
        env_name: Gymnasium environment id.
        seed: Experiment seed.
        cache_dir: Directory for caching DQN models.
        out_path: Output PNG path.
        total_timesteps: DQN training budget override (continuous envs only).
        horizon_return: Reference J value; defaults to PPO expert return from
            results JSON.
        expert_cache: Directory of cached PPO expert policies; used only for
            toy-text (discrete-obs) environments.

    Returns:
        Dict with keys ``mu_mean``, ``mu_median``, ``ppo_return`` (and
        ``dqn_return``/``dqn_return_std`` for DQN backends), or
        ``{"skipped": True}`` for Blackjack-v1.
    """
    # Check skip first — before loading any demos.
    prov = provenance_label(env_name)
    if prov == "skipped (no env.P)":
        print(f"Skipping recoverability for {env_name}: {prov}")
        return {"skipped": True}

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

    # Load PPO expert only for toy-text (discrete-obs) envs.
    is_discrete = env_utils.ENV_CONFIGS.get(env_name, {}).get("obs_type") == "discrete"
    if expert_cache is None:
        expert_cache = "experiments/expert_cache"
    expert = None
    if is_discrete:
        from imitation.experiments.ftrl import coverage_features

        expert = coverage_features.load_expert_policy(env_name, expert_cache)

    mu = recoverability.recoverability_mu(
        env_name, obs, cache_dir, expert_policy=expert, expert_cache=expert_cache
    )
    # mu should not be None here (Blackjack was caught above), but guard anyway.
    if mu is None:
        print(f"Skipping recoverability for {env_name}: {prov}")
        return {"skipped": True}

    ppo_return = expert_return_from_results(results_dir, env_name, seed)
    j_value = horizon_return if horizon_return is not None else ppo_return

    # DQN return/± is only meaningful for DQN backends (Atari / continuous).
    is_dqn_backend = not is_discrete
    dqn_return: Optional[float] = None
    dqn_return_std: Optional[float] = None
    if is_dqn_backend:
        if env_utils.is_atari(env_name):
            # Evaluating an Atari hub DQN in-process is expensive (needs the
            # Atari env + rom); omit the DQN return line rather than blocking.
            pass
        else:
            dqn = recoverability.get_or_train_dqn_reference(
                env_name, cache_dir, total_timesteps, seed
            )
            rets = recoverability.reference_returns(dqn, env_name)
            dqn_return = float(rets.mean())
            dqn_return_std = float(rets.std())
            random_return = baseline_from_results(
                results_dir, env_name, seed, "random_return"
            )
            msg = dqn_gate_warning(dqn_return, random_return, ppo_return, env_name)
            if msg is not None:
                print(f"WARNING: {msg}")

    render_recoverability_figure(
        mu,
        out_path,
        env_name,
        j_value,
        dqn_return,
        ppo_return,
        dqn_return_std,
        provenance_str=prov,
        show_dqn_return=(is_dqn_backend and dqn_return is not None),
    )
    result: dict = {
        "mu_mean": float(mu.mean()),
        "mu_median": float(np.median(mu)),
        "ppo_return": ppo_return,
    }
    if dqn_return is not None:
        result["dqn_return"] = dqn_return
    if dqn_return_std is not None:
        result["dqn_return_std"] = dqn_return_std
    return result


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
    parser.add_argument(
        "--expert-cache",
        type=str,
        default="experiments/expert_cache",
        help=(
            "Directory of cached PPO expert policies; used only for toy-text "
            "(discrete-obs) environments."
        ),
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
        expert_cache=args.expert_cache,
    )
    if info.get("skipped"):
        print(f"Skipped {args.env} (no recoverability backend).")
    else:
        mu_median = info["mu_median"]
        dqn_ret = info.get("dqn_return")
        if dqn_ret is not None:
            print(
                f"Wrote {out}; mu_median={mu_median:.3f}, " f"DQN_return={dqn_ret:.0f}"
            )
        else:
            print(f"Wrote {out}; mu_median={mu_median:.3f}")


if __name__ == "__main__":
    main()
