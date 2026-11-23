"""Evaluation protocol eval-every-50-rounds-1-episode/1 and fit monitoring.

Runs against whichever source tree is first on PYTHONPATH (both lineages).
"""

import copy
import json

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
        beta_rampdown=0,
        bc_n_epochs=3,
        eval_interval=2,
        eval_initial_rounds=0,
        eval_episodes=1,
        save_every_round=False,
        output_dir=tmp_path / "results",
        expert_cache_dir=tmp_path / "experts",
        inner_early_stop=False,
        outer_early_stop=False,
        grow_batch_with_data=True,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


# --------------------------------------------------------------------------
# Evaluation protocol
# --------------------------------------------------------------------------


def test_protocol_constants():
    assert run_experiment.EVAL_INTERVAL == 20
    assert run_experiment.EVAL_EPISODES == 10
    assert run_experiment.EVAL_PROTOCOL == "dense-early-periodic-evaluation/2"


def test_eval_rounds_grid():
    assert run_experiment.eval_rounds(1000) == list(range(61)) + list(range(80, 1001, 20))
    assert run_experiment.eval_rounds(73, 50, 0) == [0, 50, 73]
    assert run_experiment.eval_rounds(50, 50, 0) == [0, 50]
    assert 1 in run_experiment.eval_rounds(1000)


def test_config_defaults_and_validation(tmp_path):
    config = _config("ftl", tmp_path)
    assert config.eval_episodes == 1
    assert config.final_eval_episodes is None
    with pytest.raises(ValueError):
        _config("ftl", tmp_path, eval_episodes=0)
    with pytest.raises(ValueError):
        _config("ftl", tmp_path, final_eval_episodes=0)


def test_cli_defaults():
    import argparse
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
    assert args.eval_interval == 20
    assert args.eval_initial_rounds == 60
    assert args.eval_episodes == 10
    assert args.final_eval_episodes is None
    configs = original(args)
    assert configs and all(c.eval_episodes == 10 for c in configs)


@pytest.mark.parametrize("algo", ["ftl", "bc_iid"])
def test_round_loop_evaluates_grid_with_one_episode(tmp_path, untrained_expert, algo):
    config = _config(algo, tmp_path, n_rounds=5, eval_interval=2)
    result = run_single(config)
    rows = result["per_round"]
    evaluated = [r["round"] for r in rows if r.get("d_eval_size") is not None]
    assert evaluated == [0, 2, 4, 5] == run_experiment.eval_rounds(5, 2, 0)
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


# --------------------------------------------------------------------------
# Fit monitoring
# --------------------------------------------------------------------------

MONITOR_KEYS = {
    "fit_monitor",
    "fit_train_examples",
    "fit_train_nll_initial",
    "fit_train_nll_curve",
    "fit_train_nll_final",
    "fit_train_accuracy_final",
    "fit_epochs_run",
    "fit_epoch_cap",
    "fit_gradient_steps",
    "fit_stop_reason",
    "fit_train_rel_drop_last_window",
    "fit_train_plateau",
    "fit_cap_without_plateau",
    "fit_best_val_epoch",
    "fit_val_improving_at_cap",
}


def _check_monitor(row, cap):
    assert MONITOR_KEYS <= set(row), sorted(MONITOR_KEYS - set(row))
    assert row["fit_monitor"] == run_experiment.FIT_MONITOR_PROTOCOL
    assert row["fit_train_nll_curve"][-1][0] == row["fit_epochs_run"]
    if "fit_train_nll_per_epoch" in row:
        assert row["fit_epochs_run"] == len(row["fit_train_nll_per_epoch"])
    assert 1 <= row["fit_epochs_run"] <= cap
    assert row["fit_epoch_cap"] == cap
    assert 0.0 <= row["fit_train_accuracy_final"] <= 1.0
    assert row["fit_train_nll_final"] >= 0.0
    assert row["fit_stop_reason"] in (
        "val_early_stop", "epoch_cap", "train_plateau", "nll_floor", "max_epochs", "max_steps"
    )
    if row["fit_cap_without_plateau"]:
        assert row["fit_stop_reason"] in ("epoch_cap", "max_epochs", "max_steps")
        assert not row["fit_train_plateau"]
    json.dumps(row)  # serializable


@pytest.mark.parametrize("early_stop", [False, True])
@pytest.mark.parametrize("algo", ["ftl", "bc_iid", "bc"])
def test_every_fit_is_monitored(tmp_path, untrained_expert, algo, early_stop):
    """Earlier epoch-cap rule, every fit (no skipping)."""
    overrides = dict(
        n_rounds=4,
        eval_interval=2,
        bc_n_epochs=4,
        fit_rule=run_experiment.FIT_RULE_EPOCHS,
        skip_unreported_fits=False,
    )
    if early_stop:
        overrides.update(
            inner_early_stop=True,
            inner_early_stop_min_val_size=1,
            inner_early_stop_val_frac=0.25,
            inner_early_stop_patience=1,
            inner_early_stop_min_epochs=1,
        )
    if algo == "bc":
        overrides["n_rounds"] = 64
    rows = run_single(_config(algo, tmp_path, **overrides))["per_round"]
    fits = [r for r in rows if r.get("inner_es_stop_epoch") is not None]
    assert fits
    for row in fits:
        _check_monitor(row, 4)


def test_monitor_measures_the_training_split(tmp_path, untrained_expert):
    config = _config(
        "bc",
        tmp_path,
        n_rounds=100,
        bc_n_epochs=5,
        inner_early_stop=True,
        inner_early_stop_val_frac=0.2,
        inner_early_stop_min_val_size=1,
        fit_rule=run_experiment.FIT_RULE_EPOCHS,
    )
    row = run_single(config)["per_round"][0]
    assert row["fit_train_examples"] == 80
    assert row["fit_best_val_epoch"] is not None


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


def test_monitor_does_not_change_training():
    """Same seed: identical weights with and without per-epoch measurements."""
    th.manual_seed(11)
    plain, _ = _bc_trainer()
    th.manual_seed(5)
    plain.train(n_epochs=6, progress_bar=False)

    th.manual_seed(11)
    watched, data = _bc_trainer()
    monitor = run_experiment._FitMonitor(watched.policy, data)
    th.manual_seed(5)
    monitor.start()
    watched.train(n_epochs=6, progress_bar=False, on_epoch_end=monitor.record)

    a, b = plain.policy.state_dict(), watched.policy.state_dict()
    assert a.keys() == b.keys()
    for k in a:
        assert th.equal(a[k], b[k]), k
    summary = monitor.summary(6, early_stopped=False, batches_per_epoch=6)
    assert summary["fit_epochs_run"] == 6
    assert summary["fit_gradient_steps"] == 36
    # Training on a learnable rule lowers the exact training loss.
    assert summary["fit_train_nll_final"] < summary["fit_train_nll_initial"]
    assert watched.policy.training  # mode restored


def test_plateau_flags():
    policy_trainer, data = _bc_trainer()
    monitor = run_experiment._FitMonitor(policy_trainer.policy, data)
    monitor.initial = 0.69
    monitor.epochs = [0.6, 0.5, 0.4, 0.3]
    still_falling = monitor.summary(4, early_stopped=False, batches_per_epoch=1)
    assert still_falling["fit_train_rel_drop_last_window"] == pytest.approx(0.5)
    # The final measurement is of the actual policy, so plateau depends on
    # the trajectory unless the final loss is at the floor.
    if still_falling["fit_train_nll_final"] > run_experiment.FIT_NLL_FLOOR:
        assert not still_falling["fit_train_plateau"]
        assert still_falling["fit_cap_without_plateau"]
    monitor.epochs = [0.3, 0.3, 0.3, 0.3]
    flat = monitor.summary(4, early_stopped=False, batches_per_epoch=1)
    assert flat["fit_train_plateau"] and not flat["fit_cap_without_plateau"]
    early = copy.copy(monitor)
    early.epochs = [0.6, 0.5, 0.4, 0.3]
    stopped = early.summary(4, early_stopped=True, batches_per_epoch=1)
    assert not stopped["fit_cap_without_plateau"]
    assert stopped["fit_stop_reason"] == "val_early_stop"


def test_fit_convergence_summary(tmp_path, untrained_expert, capsys):
    from imitation.experiments.ftrl import fit_convergence

    run_single(_config("ftl", tmp_path, n_rounds=4, eval_interval=2))
    rows = fit_convergence.collect([tmp_path / "results"])
    assert len(rows) == 4 and {r["method"] for r in rows} == {"ftl"}
    summary = fit_convergence.summarize(rows)
    assert [g["labels"] for g in summary] == [1, 2, 3, 4]
    assert all(0.0 <= g["cap_no_plateau"] <= 1.0 for g in summary)
    out = tmp_path / "fc.png"
    assert fit_convergence.main([str(tmp_path / "results"), "--plot", str(out)]) == 0
    assert out.exists() and "monitored fits" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Fit rule train-loss-plateau/1 and skipping unreported fits
# --------------------------------------------------------------------------


def test_plateau_rule_is_the_default(tmp_path):
    config = _config("ftl", tmp_path)
    assert config.fit_rule == run_experiment.FIT_RULE_PLATEAU
    assert not config.skip_unreported_fits


@pytest.mark.parametrize("algo", ["ftl", "bc_iid", "bc"])
def test_plateau_fits_train_on_all_labels_until_plateau(tmp_path, untrained_expert, algo):
    n_rounds = 64 if algo == "bc" else 4
    rows = run_single(_config(algo, tmp_path, n_rounds=n_rounds, eval_interval=2))["per_round"]
    fits = [r for r in rows if r.get("fit_monitor")]
    assert fits
    for row in fits:
        _check_monitor(row, run_experiment.FIT_MAX_EPOCHS)
        assert row["fit_rule"] == run_experiment.FIT_RULE_PLATEAU
        # No held-out split: every retained label is trained on.
        assert row["fit_train_examples"] == row["n_observations"]
        assert row["fit_stop_reason"] in ("train_plateau", "nll_floor", "max_epochs", "max_steps")
        assert row["fit_epochs_run"] >= 1


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


def test_compact_curve():
    assert run_experiment._compact_curve([0.5, 0.4]) == [[1, 0.5], [2, 0.4]]
    long = run_experiment._compact_curve([1.0 / (i + 1) for i in range(2000)])
    assert len(long) <= 66 and long[0][0] == 1 and long[-1][0] == 2000


def test_bc_iid_fits_only_evaluated_budgets(tmp_path, untrained_expert):
    rows = run_single(_config("bc_iid", tmp_path, n_rounds=5, eval_interval=2, skip_unreported_fits=True))["per_round"]
    fitted = [r["round"] for r in rows if r.get("fit_monitor")]
    skipped = [r["round"] for r in rows if r.get("fit_skipped")]
    assert fitted == [2, 4, 5] and skipped == [1, 3]
    # Skipped rounds still aggregate their data.
    assert [r["n_observations"] for r in rows] == [0, 1, 2, 3, 4, 5]
    assert all(r["train_cross_entropy"] is None for r in rows if r.get("fit_skipped"))


def test_ftl_fits_every_round(tmp_path, untrained_expert):
    rows = run_single(_config("ftl", tmp_path, n_rounds=3, eval_interval=2))["per_round"]
    assert [r["round"] for r in rows if r.get("fit_monitor")] == [1, 2, 3]


def test_paired_skip_is_exactly_equivalent(tmp_path, untrained_expert):
    """Paired cold fits are keyed by (seed, N): skipping unreported BC-iid fits
    leaves every evaluated policy bit-identical."""
    if "paired_cold_fits" not in ExperimentConfig.__dataclass_fields__:
        pytest.skip("lineage without paired cold fits")
    from imitation.experiments.ftrl.policy_utils import load_policy_checkpoint

    states = {}
    for skip in (True, False):
        cfg = _config(
            "bc_iid",
            tmp_path / f"skip{skip}",
            n_rounds=4,
            eval_interval=2,
            paired_cold_fits=True,
            skip_unreported_fits=skip,
        )
        rows = run_single(cfg)["per_round"]
        states[skip] = {
            r["round"]: load_policy_checkpoint(r["checkpoint"]).state_dict()
            for r in rows
            if r.get("checkpoint")
        }
        fits = [r for r in rows if r.get("fit_monitor")]
        assert all(r["fit_rule"] == run_experiment.FIT_RULE_PLATEAU for r in fits)
    assert states[True].keys() == states[False].keys() == {0, 2, 4}
    for rnd in states[True]:
        for k, v in states[True][rnd].items():
            assert th.equal(v, states[False][rnd][k]), (rnd, k)


def test_learning_rate_and_caps_defaults(tmp_path):
    config = _config("ftl", tmp_path)
    assert config.learning_rate == run_experiment.BC_LEARNING_RATE == 1e-2
    assert config.fit_max_steps == 20000 and config.fit_max_epochs == 20000
    assert config.fit_plateau_min_window_steps == 100
    assert config.fit_nll_floor == 0.05
