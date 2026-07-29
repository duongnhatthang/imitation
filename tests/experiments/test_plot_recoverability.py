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


def test_render_with_zero_ppo_return_still_annotates(tmp_path):
    mu = np.abs(np.random.RandomState(1).randn(200)) * 5
    out = tmp_path / "CartPole-v1_recoverability_zero.png"
    plot_recoverability.render_recoverability_figure(
        mu,
        out,
        "CartPole-v1",
        horizon_return=500.0,
        dqn_return=480.0,
        ppo_return=0.0,
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


def test_provenance_label_per_family():
    assert "env.P" in plot_recoverability.provenance_label("FrozenLake-v1")
    assert "hub" in plot_recoverability.provenance_label("PongNoFrameskip-v4").lower()
    assert "trained" in plot_recoverability.provenance_label("Acrobot-v1").lower()


def test_build_and_plot_skips_blackjack(tmp_path):
    out = tmp_path / "Blackjack-v1_recoverability.png"
    result = plot_recoverability.build_and_plot(
        tmp_path, "Blackjack-v1", 0, tmp_path / "cache", out
    )
    assert result == {"skipped": True}
    assert not out.exists()  # skipped -> no figure written


def test_dqn_gate_warning_negative_return_env():
    # Acrobot-like: DQN better than expert -> normalized ~1.01 -> no warning.
    assert (
        plot_recoverability.dqn_gate_warning(-73.0, -500.0, -79.0, "Acrobot-v1") is None
    )

    # Genuinely under-trained: normalized well below 0.9 -> warning string.
    msg = plot_recoverability.dqn_gate_warning(-300.0, -500.0, -79.0, "Acrobot-v1")
    assert msg is not None
    assert "normalized" in msg

    # Missing random_return baseline -> no warning.
    assert (
        plot_recoverability.dqn_gate_warning(-73.0, None, -79.0, "Acrobot-v1") is None
    )
