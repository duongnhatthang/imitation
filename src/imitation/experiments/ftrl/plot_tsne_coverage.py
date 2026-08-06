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


def _sized_jittered_scatter(xs, ys, xrange, yrange, point_size, rng):
    """Marker size (by distinct positions) + jitter for piled-up points.

    Toy-text envs pile many points onto a handful of distinct t-SNE positions,
    so sizing by *distinct* positions (not raw count) keeps sparse groups
    legible, and jittering a pile-up fans it out instead of collapsing to a
    single opaque dot. Shared by the per-algo panels and the coverage-diff
    overlay so both figures render the same points identically. Returns
    ``(xs, ys, size, n_uniq)`` with xs/ys jittered copies.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n_pts = len(xs)
    n_uniq = _unique_positions(np.column_stack([xs, ys]))
    size = point_size if point_size is not None else _auto_point_size(n_uniq)
    if n_pts > 2 * max(n_uniq, 1):
        xs = xs + rng.normal(0, 0.008 * xrange, n_pts)
        ys = ys + rng.normal(0, 0.008 * yrange, n_pts)
    return xs, ys, size, n_uniq


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
    # Square-ish grid (2x2 for the usual 4 algos) so panels are large and titles
    # don't collide; constrained_layout spaces titles/colorbar and trims whitespace.
    ncols = int(np.ceil(np.sqrt(len(algos))))
    nrows = int(np.ceil(len(algos) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 5 * nrows),
        squeeze=False,
        constrained_layout=True,
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
        # Size by DISTINCT positions + jitter piled-up points (shared with the
        # coverage-diff overlay so the two figures render points identically).
        xs, ys, size, n_uniq = _sized_jittered_scatter(
            emb[mask, 0], emb[mask, 1], xrange, yrange, point_size, jrng
        )
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
            f"{algo}  (n={n_pts}, uniq={uniq}, kNN={metrics_hd.get(algo, 0):.2f})",
            fontsize=11,
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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_coverage_diff(
    embedding, algo_labels, out_path, env_name, n_bins=40, both_gray=True
):
    """Region map on the shared embedding: where ftl/ftrl explore vs bc (offline).

    Grids the 2-D embedding and paints each cell by which family occupies it:
    interactive-only (ftl/ftrl but not bc), bc-only, both (light gray), or empty
    (white). Highlights what interaction discovers beyond the offline dataset.
    """
    import matplotlib.patches as mpatches

    x = embedding[:, 0].astype(float)
    y = embedding[:, 1].astype(float)
    xrange = max(x.max() - x.min(), 1e-9)
    yrange = max(y.max() - y.min(), 1e-9)
    xe = np.linspace(x.min(), x.max(), n_bins + 1)
    ye = np.linspace(y.min(), y.max(), n_bins + 1)

    def occupied(mask):
        hist, _, _ = np.histogram2d(x[mask], y[mask], bins=[xe, ye])
        return hist > 0

    bc = occupied(algo_labels == "bc")
    inter_mask = np.isin(algo_labels, ["ftl", "ftrl"])
    inter = occupied(inter_mask)
    both = bc & inter
    c_bc = (0.55, 0.68, 0.90)  # light blue
    c_int = (0.98, 0.70, 0.40)  # light orange
    c_both = (0.85, 0.85, 0.85)  # light gray
    img = np.ones((n_bins, n_bins, 3))
    img[bc & ~inter] = c_bc
    img[inter & ~bc] = c_int
    if both_gray:
        img[both] = c_both

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    ax.imshow(
        np.transpose(img, (1, 0, 2)),
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="auto",
        interpolation="nearest",
    )
    # Overlay the actual points using the SAME size + jitter as the per-algo
    # t-SNE panels (via the shared helper) so the two figures are visually
    # consistent for a reader comparing them. Alpha is kept lower than the
    # panels (0.6) so the region colors still read through the markers.
    jrng = np.random.default_rng(0)
    bc_mask = algo_labels == "bc"
    bx, by, bsize, _ = _sized_jittered_scatter(
        x[bc_mask], y[bc_mask], xrange, yrange, None, jrng
    )
    ax.scatter(bx, by, s=bsize, c="#26418f", alpha=0.35, edgecolors="none")
    ix, iy, isize, _ = _sized_jittered_scatter(
        x[inter_mask], y[inter_mask], xrange, yrange, None, jrng
    )
    ax.scatter(ix, iy, s=isize, c="#b25000", alpha=0.35, edgecolors="none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Coverage difference: {env_name}  (grid {n_bins}x{n_bins})")
    handles = [
        mpatches.Patch(color=c_int, label="ftl/ftrl only (interaction discovers)"),
        mpatches.Patch(color=c_bc, label="bc only (offline, not revisited)"),
        mpatches.Patch(color=c_both, label="both"),
        mpatches.Patch(facecolor="white", edgecolor="0.7", label="neither"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
    diff_path = out_path.parent / f"{out_path.stem}_coverage_diff.png"
    render_coverage_diff(result.embedding, algo, diff_path, env_name)
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
