import numpy as np

from imitation.experiments.ftrl import plot_recoverability


def test_render_writes_png_with_annotation(tmp_path):
    mu = np.abs(np.random.RandomState(0).randn(500)) * 10
    out = tmp_path / "CartPole-v1_recoverability.png"
    plot_recoverability.render_recoverability_figure(
        mu,
        out,
        "CartPole-v1",
        horizon_return=500.0,
        dqn_return=480.0,
        ppo_return=500.0,
    )
    assert out.exists()


def test_expert_return_from_results(tmp_path):
    import json

    d = tmp_path / "CartPole-v1"
    d.mkdir(parents=True)
    (d / "ftrl_end_to_end_seed0.json").write_text(
        json.dumps(
            {
                "algo": "ftrl",
                "env": "CartPole-v1",
                "seed": 0,
                "baselines": {"expert_return": 500.0},
                "per_round": [],
                "elapsed_seconds": 1.0,
            }
        )
    )
    assert (
        plot_recoverability.expert_return_from_results(tmp_path, "CartPole-v1", 0)
        == 500.0
    )
