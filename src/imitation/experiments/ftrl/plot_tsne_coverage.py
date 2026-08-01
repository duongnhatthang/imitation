"""Render per-environment t-SNE coverage grids colored by data-arrival round."""

import argparse
import pathlib
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from imitation.experiments.ftrl import coverage_data, tsne_coverage  # noqa: E402


def _auto_point_size(n_points: int) -> float:
    """Marker size that grows as a panel gets sparser, so few points stay visible.

    ~6 pt for dense panels (>=300 points), scaling up to ~45 pt for very sparse
    panels so a 30-point smoke run is still legible.
    """
    return float(np.clip(1800.0 / max(n_points, 1), 6.0, 45.0))


def _unique_positions(coords, decimals=2):
    """Count distinct 2-D positions (to size markers by overlap, not raw count)."""
    return int(len(np.unique(np.round(coords, decimals), axis=0)))


def render_coverage_figure(
    pooled,
    tsne_result,
    out_path,
    env_name,
    n_rounds,
    metrics_2d,
    metrics_hd,
    point_size=None,
    metrics_uniq=None,
) -> None:
    """One panel per algorithm on the shared embedding, colored by arrival round.

    Args:
        pooled: PooledStates with obs, algo, rounds fields.
        tsne_result: TSNEResult from fit_shared_tsne.
        out_path: Destination path for the PNG.
        env_name: Environment name for the figure title.
        n_rounds: Total number of data-arrival rounds (for colorbar range).
        metrics_2d: Per-algo dict from coverage_metrics_2d.
        metrics_hd: Per-algo dict from coverage_metrics_highdim.
        point_size: Fixed marker size; if None, sized per panel by point count.
    """
    algos = sorted(set(pooled.algo.tolist()))
    ncols = min(3, len(algos))
    nrows = int(np.ceil(len(algos) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False
    )
    emb = tsne_result.embedding
    norm = plt.Normalize(vmin=0, vmax=max(n_rounds, 1))
    xlim = (emb[:, 0].min(), emb[:, 0].max())
    ylim = (emb[:, 1].min(), emb[:, 1].max())
    xrange = max(xlim[1] - xlim[0], 1e-9)
    yrange = max(ylim[1] - ylim[0], 1e-9)
    jrng = np.random.default_rng(0)
    scatter = None
    for i, algo in enumerate(algos):
        ax = axes[i // ncols][i % ncols]
        mask = pooled.algo == algo
        n_pts = int(mask.sum())
        xs = emb[mask, 0].astype(float)
        ys = emb[mask, 1].astype(float)
        # Size by DISTINCT positions, not raw count: toy-text envs pile 1000
        # points onto ~6-18 states, so count-based sizing made them invisible.
        n_uniq = _unique_positions(np.column_stack([xs, ys]))
        size = point_size if point_size is not None else _auto_point_size(n_uniq)
        # When many points stack on few positions, jitter so the pile-up and its
        # arrival-round mix are visible instead of a single opaque dot.
        if n_pts > 2 * max(n_uniq, 1):
            xs = xs + jrng.normal(0, 0.008 * xrange, n_pts)
            ys = ys + jrng.normal(0, 0.008 * yrange, n_pts)
        scatter = ax.scatter(
            xs,
            ys,
            c=pooled.rounds[mask],
            cmap="coolwarm",
            norm=norm,
            s=size,
            alpha=0.6,
            edgecolors="none",
        )
        uniq = metrics_uniq.get(algo, n_uniq) if metrics_uniq else n_uniq
        ax.set_title(
            f"{algo}  (n={n_pts}, uniq={uniq}, " f"kNN={metrics_hd.get(algo, 0):.2f})"
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(len(algos), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.6)
        cbar.set_label("Data Arrival Rounds")
    fig.suptitle(
        f"t-SNE state coverage: {env_name}  "
        f"(perplexity={tsne_result.perplexity:.0f}, "
        f"trustworthiness={tsne_result.trustworthiness:.3f})"
    )
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_and_plot(
    results_dir,
    env_name,
    seed,
    out_path,
    perplexities=(15, 30, 50),
    seeds=(0,),
    cap=8000,
    point_size=None,
    expert_cache="experiments/expert_cache",
) -> dict:
    """Load states, fit the shared t-SNE, render, and cache the embedding.

    Args:
        results_dir: Root directory containing scratch demo files.
        env_name: Environment name to load states for.
        seed: Experiment seed.
        out_path: Destination path for the PNG.
        perplexities: Perplexity values to try; best by trustworthiness is kept.
        seeds: Random seeds for t-SNE.
        cap: Maximum points to subsample before fitting t-SNE.
        point_size: Fixed marker size; if None, sized per panel by point count.
        expert_cache: Directory containing cached PPO expert policies (used for
            Atari CNN feature extraction).

    Returns:
        Dict with keys ``coverage_2d``, ``coverage_highdim``, ``trustworthiness``.

    Raises:
        FileNotFoundError: If no scratch demos are found for the given env/seed.
    """
    from imitation.experiments.ftrl import coverage_features

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
    pooled = coverage_data.pool(states)
    if coverage_features.feature_mode(env_name) == "cnn":
        expert = coverage_features.load_expert_policy(env_name, expert_cache)
        feats_full = coverage_features.extract_features(
            pooled.obs, env_name, expert=expert
        )
    else:
        feats_full = coverage_features.extract_features(pooled.obs, env_name)
    feats, algo, rounds, _ = tsne_coverage.subsample(
        feats_full, pooled.algo, pooled.rounds, cap=cap, seed=seeds[0]
    )
    sub = coverage_data.PooledStates(obs=feats, algo=algo, rounds=rounds)
    result = tsne_coverage.fit_shared_tsne(feats, perplexities, seeds)
    metrics_2d = tsne_coverage.coverage_metrics_2d(result.embedding, algo)
    metrics_hd = tsne_coverage.coverage_metrics_highdim(feats, algo)
    metrics_uniq = tsne_coverage.coverage_metrics_unique(feats, algo)
    n_rounds = int(pooled.rounds.max()) if len(pooled.rounds) else 0
    render_coverage_figure(
        sub,
        result,
        out_path,
        env_name,
        n_rounds,
        metrics_2d,
        metrics_hd,
        point_size=point_size,
        metrics_uniq=metrics_uniq,
    )
    out_path = pathlib.Path(out_path)
    # Cache the pre-t-SNE features (plus algo/rounds) so the interactive tuner can
    # re-fit t-SNE at other perplexities/seeds without the raw scratch demos.
    np.savez(
        out_path.with_suffix(".npz"),
        embedding=result.embedding,
        features=feats,
        algo=algo.astype(str),
        rounds=rounds,
        perplexity=result.perplexity,
        trustworthiness=result.trustworthiness,
    )
    return {
        "coverage_2d": metrics_2d,
        "coverage_highdim": metrics_hd,
        "coverage_unique": metrics_uniq,
        "trustworthiness": result.trustworthiness,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for t-SNE state-coverage plots.

    Args:
        argv: Argument list (defaults to sys.argv if None).
    """
    parser = argparse.ArgumentParser(description="t-SNE state-coverage plots.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--perplexity", type=float, nargs="+", default=[15, 30, 50])
    parser.add_argument("--tsne-seed", type=int, nargs="+", default=[0])
    parser.add_argument("--cap", type=int, default=8000)
    parser.add_argument(
        "--point-size",
        type=float,
        default=None,
        help="Fixed scatter marker size; default auto-sizes per panel by count.",
    )
    parser.add_argument(
        "--expert-cache",
        default="experiments/expert_cache",
        help="Directory containing cached PPO expert policies for Atari CNN features.",
    )
    args = parser.parse_args(argv)
    out = pathlib.Path(args.output_dir) / f"{args.env.replace('/', '_')}.png"
    metrics = build_and_plot(
        args.results_dir,
        args.env,
        args.seed,
        out,
        tuple(args.perplexity),
        tuple(args.tsne_seed),
        args.cap,
        point_size=args.point_size,
        expert_cache=args.expert_cache,
    )
    print(f"Wrote {out}; trustworthiness={metrics['trustworthiness']:.3f}")


if __name__ == "__main__":
    main()
