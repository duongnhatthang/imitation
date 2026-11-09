"""Regression tests for sparse evaluation and training-loss stopping."""

import numpy as np
import pytest
import torch as th

from imitation.experiments.ftrl import run_experiment
from imitation.experiments.ftrl.run_experiment import ExperimentConfig, run_single


@pytest.fixture
def untrained_expert(monkeypatch):
    from stable_baselines3 import PPO

    expert = PPO("MlpPolicy", "CartPole-v1", seed=10, device="cpu")
    monkeypatch.setattr(
        run_experiment.experts, "get_or_train_expert", lambda *a, **kw: expert.policy
    )
    from imitation.experiments.ftrl import env_baselines

    monkeypatch.setattr(
        env_baselines,
        "load_or_compute_baselines",
        lambda *args, **kwargs: {"expert_return": 100.0, "random_return": 10.0},
    )
    monkeypatch.setattr(env_baselines, "validate_expert_quality", lambda *a: (True, ""))
    yield expert.policy
    expert.get_env().close()


def _config(algo, tmp_path, **overrides):
    defaults = dict(
        algo=algo,
        env_name="CartPole-v1",
        seed=0,
        policy_mode="linear",
        n_rounds=5,
        samples_per_round=1,
        l2_lambda=0.0,
        l2_decay=False,
        warm_start=False,
        beta_rampdown=1,
        bc_n_epochs=3,
        eval_interval=2,
        output_dir=tmp_path / "results",
        expert_cache_dir=tmp_path / "experts",
        inner_early_stop=False,
        outer_early_stop=False,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_eval_rounds_grid():
    assert run_experiment.eval_rounds(1000) == list(range(0, 1001, 50))
    assert run_experiment.eval_rounds(73, 50) == [0, 50, 73]
    assert run_experiment.eval_rounds(50, 50) == [0, 50]
    assert 1 not in run_experiment.eval_rounds(1000)


def test_config_defaults_and_validation(tmp_path):
    config = _config("ftl", tmp_path)
    assert config.eval_episodes == 1
    assert config.final_eval_episodes is None
    with pytest.raises(ValueError):
        _config("ftl", tmp_path, eval_episodes=0)
    with pytest.raises(ValueError):
        _config("ftl", tmp_path, final_eval_episodes=0)


def test_cli_defaults():
    import sys

    parsed = {}

    def capture(args):
        parsed["args"] = args
        raise SystemExit(0)

    old = sys.argv
    sys.argv = ["run_experiment", "--envs", "CartPole-v1"]
    original = run_experiment.build_configs
    run_experiment.build_configs = capture
    try:
        with pytest.raises(SystemExit):
            run_experiment.main()
    finally:
        run_experiment.build_configs = original
        sys.argv = old
    args = parsed["args"]
    assert args.eval_interval == 50
    assert args.eval_episodes == 1
    assert args.final_eval_episodes is None
    configs = original(args)
    assert configs and all(c.eval_episodes == 1 for c in configs)


@pytest.mark.parametrize("algo", ["ftl", "bc_iid"])
def test_round_loop_evaluates_grid_with_one_episode(tmp_path, untrained_expert, algo):
    config = _config(algo, tmp_path, n_rounds=5, eval_interval=2)
    result = run_single(config)
    rows = result["per_round"]
    evaluated = [r["round"] for r in rows if r.get("d_eval_size") is not None]
    assert evaluated == [0, 2, 4, 5] == run_experiment.eval_rounds(5, 2)
    for r in rows:
        if r.get("d_eval_size") is not None:
            assert len(r["episode_returns"]) == 1
            assert r["evaluation_episodes"] == 1


def test_final_eval_episodes_apply_to_final_row_only(tmp_path, untrained_expert):
    config = _config("ftl", tmp_path, n_rounds=4, eval_interval=2, final_eval_episodes=3)
    rows = run_single(config)["per_round"]
    counts = {r["round"]: len(r["episode_returns"]) for r in rows if r.get("d_eval_size")}
    assert counts == {0: 1, 2: 1, 4: 3}
    for row in rows[1:]:
        assert row["fit_train_examples"] == row["n_observations"]
        assert row["inner_es_val_nll_trajectory"] == []
        assert row["fit_rule"] == run_experiment.FIT_RULE_PLATEAU


def test_fixed_bc_single_final_eval(tmp_path, untrained_expert):
    config = _config("bc", tmp_path, n_rounds=40, final_eval_episodes=2)
    rows = run_single(config)["per_round"]
    assert len(rows) == 1 and len(rows[0]["episode_returns"]) == 2


def _bc_trainer(seed=3, n=96):
    from gymnasium import spaces

    from imitation.algorithms import bc
    from imitation.data import types

    rng = np.random.default_rng(seed)
    obs_space = spaces.Box(-1, 1, (4,), np.float32)
    act_space = spaces.Discrete(2)
    obs = rng.uniform(-1, 1, (n, 4)).astype(np.float32)
    acts = (obs[:, 0] + 0.3 * obs[:, 1] > 0).astype(np.int64)
    transitions = types.Transitions(
        obs=obs,
        acts=acts,
        infos=np.array([{}] * n),
        next_obs=obs.copy(),
        dones=np.zeros(n, dtype=bool),
    )
    th.manual_seed(seed)
    trainer = bc.BC(
        observation_space=obs_space,
        action_space=act_space,
        rng=np.random.default_rng(seed),
        demonstrations=transitions,
        batch_size=16,
        device="cpu",
    )
    return trainer, transitions


def test_plateau_stop_rule_on_a_learnable_dataset():
    trainer, data = _bc_trainer(n=96)

    class Cfg:
        fit_min_epochs = 10
        fit_plateau_window = 10
        fit_plateau_rel_tol = 0.01
        fit_nll_floor = 0.01
        fit_max_epochs = 2000
        fit_max_steps = 20000

    log = run_experiment._plateau_fit(trainer, Cfg, data)
    curve = [v for _, v in log["fit_train_nll_curve"]]
    assert log["fit_stop_reason"] in ("train_plateau", "nll_floor")
    assert log["fit_train_plateau"] and not log["fit_cap_without_plateau"]
    assert log["fit_epochs_run"] >= 11
    assert curve[-1] < curve[0]
    assert log["fit_gradient_steps"] == log["fit_epochs_run"] * 6
    assert trainer.batch_size == 16  # restored


def test_plateau_caps_are_flagged():
    trainer, data = _bc_trainer(n=96)

    class Cfg:
        fit_min_epochs = 10
        fit_plateau_window = 10
        fit_plateau_rel_tol = 0.0
        fit_nll_floor = 0.0
        fit_max_epochs = 2000
        fit_max_steps = 30

    log = run_experiment._plateau_fit(trainer, Cfg, data)
    assert log["fit_stop_reason"] == "max_steps"
    assert log["fit_gradient_steps"] == 30 and log["fit_epochs_run"] == 5
