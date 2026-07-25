"""Render per-environment t-SNE coverage grids colored by data-arrival round."""

import argparse
import pathlib
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from imitation.experiments.ftrl import coverage_data, tsne_coverage  # noqa: E402


def render_coverage_figure(
    pooled, tsne_result, out_path, env_name, n_rounds, metrics_2d, metrics_hd
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
    scatter = None
    for i, algo in enumerate(algos):
        ax = axes[i // ncols][i % ncols]
        mask = pooled.algo == algo
        scatter = ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            c=pooled.rounds[mask],
            cmap="coolwarm",
            norm=norm,
            s=4,
            alpha=0.6,
            edgecolors="none",
        )
        cov = metrics_2d.get(algo, {})
        ax.set_title(
            f"{algo}  (cells={cov.get('occupied_cells', 0)}, "
            f"kNN={metrics_hd.get(algo, 0):.2f})"
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

    Returns:
        Dict with keys ``coverage_2d``, ``coverage_highdim``, ``trustworthiness``.

    Raises:
        FileNotFoundError: If no scratch demos are found for the given env/seed.
    """
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
    feats_full = coverage_data.standardize_features(pooled.obs)
    feats, algo, rounds, _ = tsne_coverage.subsample(
        feats_full, pooled.algo, pooled.rounds, cap=cap, seed=seeds[0]
    )
    sub = coverage_data.PooledStates(obs=feats, algo=algo, rounds=rounds)
    result = tsne_coverage.fit_shared_tsne(feats, perplexities, seeds)
    metrics_2d = tsne_coverage.coverage_metrics_2d(result.embedding, algo)
    metrics_hd = tsne_coverage.coverage_metrics_highdim(feats, algo)
    n_rounds = int(pooled.rounds.max()) if len(pooled.rounds) else 0
    render_coverage_figure(
        sub, result, out_path, env_name, n_rounds, metrics_2d, metrics_hd
    )
    out_path = pathlib.Path(out_path)
    np.savez(
        out_path.with_suffix(".npz"),
        embedding=result.embedding,
        algo=algo.astype(str),
        rounds=rounds,
        perplexity=result.perplexity,
        trustworthiness=result.trustworthiness,
    )
    return {
        "coverage_2d": metrics_2d,
        "coverage_highdim": metrics_hd,
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
    )
    print(f"Wrote {out}; trustworthiness={metrics['trustworthiness']:.3f}")


if __name__ == "__main__":
    main()
