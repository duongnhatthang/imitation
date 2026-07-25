"""Shared per-environment t-SNE embedding with quality-based selection.

All algorithms' pooled states are embedded with ONE t-SNE (the paper's "same
mapping for the same environment"), so per-algorithm panels are comparable.
A small perplexity/seed grid is swept and the embedding with the highest
trustworthiness is kept. Coverage metrics (2-D and high-dim) quantify the
sparse-offline / dense-interactive contrast independent of the layout.
"""

import dataclasses
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors


@dataclasses.dataclass
class TSNEResult:
    embedding: np.ndarray  # [N, 2]
    perplexity: float
    seed: int
    trustworthiness: float


def fit_shared_tsne(
    features: np.ndarray,
    perplexities: Sequence[float] = (15, 30, 50),
    seeds: Sequence[int] = (0,),
    n_neighbors: int = 10,
) -> TSNEResult:
    """Fit t-SNE across a small grid; keep the most trustworthy embedding."""
    n = len(features)
    best: Optional[TSNEResult] = None
    for perplexity in perplexities:
        # t-SNE requires perplexity < n_samples.
        perp = min(perplexity, max(5.0, (n - 1) / 3.0))
        for seed in seeds:
            emb = TSNE(
                n_components=2,
                perplexity=perp,
                init="pca",
                random_state=seed,
                n_iter=1000,
            ).fit_transform(features)
            tw = float(
                trustworthiness(features, emb, n_neighbors=min(n_neighbors, n - 1))
            )
            if best is None or tw > best.trustworthiness:
                best = TSNEResult(emb, float(perp), int(seed), tw)
    return best


def coverage_metrics_2d(
    embedding: np.ndarray, algo_labels: np.ndarray, n_bins: int = 50
) -> Dict[str, Dict[str, float]]:
    """Per-algo occupied-cell count and convex-hull area on the shared 2-D map."""
    x_edges = np.linspace(embedding[:, 0].min(), embedding[:, 0].max(), n_bins + 1)
    y_edges = np.linspace(embedding[:, 1].min(), embedding[:, 1].max(), n_bins + 1)
    out: Dict[str, Dict[str, float]] = {}
    for algo in sorted(set(algo_labels.tolist())):
        pts = embedding[algo_labels == algo]
        hist, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[x_edges, y_edges])
        occupied = int((hist > 0).sum())
        try:
            hull_area = float(ConvexHull(pts).volume)  # 'volume' == area in 2-D
        except Exception:
            hull_area = 0.0
        out[algo] = {"occupied_cells": occupied, "hull_area": hull_area}
    return out


def coverage_metrics_highdim(
    features: np.ndarray, algo_labels: np.ndarray, k: int = 5
) -> Dict[str, float]:
    """Per-algo mean distance to the k nearest neighbours in feature space."""
    out: Dict[str, float] = {}
    for algo in sorted(set(algo_labels.tolist())):
        pts = features[algo_labels == algo]
        kk = min(k + 1, len(pts))
        nn = NearestNeighbors(n_neighbors=kk).fit(pts)
        dists, _ = nn.kneighbors(pts)
        out[algo] = float(dists[:, 1:].mean())  # drop self-distance
    return out


def subsample(
    features: np.ndarray,
    algo_labels: np.ndarray,
    rounds: np.ndarray,
    cap: int = 8000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified subsample by algo x round so the arrival gradient survives."""
    n = len(features)
    if n <= cap:
        idx = np.arange(n)
        return features, algo_labels, rounds, idx
    rng = np.random.default_rng(seed)
    keep_frac = cap / n
    keep = []
    strata = {}
    for i in range(n):
        strata.setdefault((algo_labels[i], int(rounds[i])), []).append(i)
    for members in strata.values():
        m = max(1, int(round(len(members) * keep_frac)))
        keep.extend(rng.choice(members, size=min(m, len(members)), replace=False))
    idx = np.array(sorted(keep))[:cap]
    return features[idx], algo_labels[idx], rounds[idx], idx
