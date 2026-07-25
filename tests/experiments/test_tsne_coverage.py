import numpy as np

from imitation.experiments.ftrl import tsne_coverage


def _blob(n, center, spread, d=6, seed=0):
    rng = np.random.default_rng(seed)
    return center + spread * rng.standard_normal((n, d))


def test_fit_shared_tsne_deterministic_and_trustworthy():
    feats = np.vstack([_blob(60, 0.0, 1.0, seed=1), _blob(60, 8.0, 1.0, seed=2)])
    r1 = tsne_coverage.fit_shared_tsne(feats, perplexities=(15,), seeds=(0,))
    r2 = tsne_coverage.fit_shared_tsne(feats, perplexities=(15,), seeds=(0,))
    assert r1.embedding.shape == (120, 2)
    assert 0.0 <= r1.trustworthiness <= 1.0
    assert np.allclose(r1.embedding, r2.embedding)  # deterministic under fixed seed


def test_coverage_metric_monotonic_in_density():
    rng = np.random.default_rng(0)
    dense = rng.uniform(-5, 5, size=(400, 2))
    sparse = rng.uniform(-1, 1, size=(400, 2))
    emb = np.vstack([dense, sparse])
    labels = np.array(["dense"] * 400 + ["sparse"] * 400, dtype=object)
    m = tsne_coverage.coverage_metrics_2d(emb, labels, n_bins=40)
    assert m["dense"]["occupied_cells"] > m["sparse"]["occupied_cells"]
    assert m["dense"]["hull_area"] > m["sparse"]["hull_area"]


def test_highdim_metric_larger_for_more_spread():
    rng = np.random.default_rng(0)
    wide = rng.normal(0, 5, size=(200, 6))
    tight = rng.normal(0, 0.5, size=(200, 6))
    feats = np.vstack([wide, tight])
    labels = np.array(["wide"] * 200 + ["tight"] * 200, dtype=object)
    m = tsne_coverage.coverage_metrics_highdim(feats, labels, k=5)
    assert m["wide"] > m["tight"]


def test_subsample_respects_cap():
    feats = np.zeros((10000, 4))
    labels = np.array(["a"] * 5000 + ["b"] * 5000, dtype=object)
    rounds = np.zeros(10000, dtype=int)
    f, l, r, idx = tsne_coverage.subsample(feats, labels, rounds, cap=1000, seed=0)
    assert len(f) <= 1000 and len(f) == len(l) == len(r) == len(idx)
