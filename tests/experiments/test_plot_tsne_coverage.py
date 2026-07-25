import numpy as np

from imitation.data import serialize, types
from imitation.experiments.ftrl import coverage_data, plot_tsne_coverage


def _demo(root, algo, seed, rnd, n, d=4, val=0.0):
    obs = (val + np.random.RandomState(rnd).randn(n + 1, d)).astype(np.float32)
    traj = types.TrajectoryWithRew(
        obs=obs,
        acts=np.zeros(n, dtype=np.int64),
        infos=None,
        terminal=True,
        rews=np.zeros(n, dtype=np.float32),
    )
    dd = (
        coverage_data.scratch_demo_root(root, algo, "CartPole-v1", seed)
        / f"round-{rnd:03d}"
    )
    dd.mkdir(parents=True, exist_ok=True)
    serialize.save(dd / f"d-{rnd}.npz", [traj])


def test_build_and_plot_writes_png_and_cache(tmp_path):
    for rnd in range(3):
        _demo(tmp_path, "ftrl", 0, rnd, 40, val=float(rnd))
    _demo(tmp_path, "bc", 0, 0, 40, val=0.0)
    out = tmp_path / "plots" / "CartPole-v1.png"
    metrics = plot_tsne_coverage.build_and_plot(
        tmp_path,
        "CartPole-v1",
        0,
        out,
        perplexities=(15,),
        seeds=(0,),
        cap=8000,
    )
    assert out.exists()
    assert out.with_suffix(".npz").exists()  # embedding cache
    assert "ftrl" in metrics["coverage_2d"] and "bc" in metrics["coverage_2d"]
