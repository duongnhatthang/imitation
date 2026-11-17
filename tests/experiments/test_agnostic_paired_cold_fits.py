"""Paired cold fits: opt-in protocol coupling offline BC and BC-prefix endpoints.

Under ``paired_cold_fits`` every cold fit on N retained labels draws its head
initialization, loader shuffle, NumPy trainer stream and held-out split from a
dedicated stream keyed by (seed, N), inside a scoped torch RNG. The offline
fixed-BC fit on N labels and the BC-prefix round-N fit therefore coincide
bitwise whatever ran before them, and evaluation at budget N uses a seed
shared by every method. The legacy behavior (flag off) is unchanged.
"""

import copy
import datetime
import json
import pathlib
import shutil
import types as pytypes

import numpy as np
import pytest
import torch as th

from imitation.data import types
from imitation.experiments.agnostic import classical, restriction
from imitation.experiments.agnostic import restriction_pilot as rp
from imitation.experiments.agnostic import task_audit, task_experts, task_pilot, tasks
from imitation.experiments.ftrl import (
    env_baselines,
    env_utils,
    eval_utils,
    policy_utils,
)
from imitation.experiments.ftrl import run_experiment as rex

UTC = datetime.timezone.utc
FAR = datetime.datetime(2026, 10, 7, 1, 0, tzinfo=UTC)
LIMIT = 3600
SEED = 300

# ---------------------------------------------------------------------------
# Synthetic CartPole fixture: an untrained expert, a fixed random dataset
# ---------------------------------------------------------------------------

N_SYNTH = 12
SYNTH_EVAL_INTERVAL = 6  # evaluations at budgets 0, 1, 6, 12


@pytest.fixture(scope="module")
def cartpole():
    from stable_baselines3 import PPO

    venv = env_utils.make_env("CartPole-v1", 1, np.random.default_rng(0))
    expert = PPO("MlpPolicy", venv, device="cpu", seed=0).policy
    data_rng = np.random.default_rng(7)
    obs = data_rng.normal(size=(N_SYNTH, 4)).astype(np.float32)
    transitions = types.Transitions(
        obs=obs,
        acts=data_rng.integers(0, 2, size=N_SYNTH).astype(np.int64),
        next_obs=obs + np.float32(0.01),
        dones=np.zeros(N_SYNTH, dtype=bool),
        infos=np.array([{} for _ in range(N_SYNTH)]),
    )
    baselines = {"expert_return": 500.0, "random_return": 20.0}
    yield pytypes.SimpleNamespace(
        venv=venv,
        expert=expert,
        transitions=transitions,
        baselines=baselines,
    )
    venv.close()


def _config(tmp_path, algo, n_rounds, **overrides):
    params = dict(
        algo=algo,
        env_name="CartPole-v1",
        seed=SEED,
        policy_mode="linear",
        n_rounds=n_rounds,
        samples_per_round=1,
        l2_lambda=0.0,
        l2_decay=False,
        warm_start=False,
        beta_rampdown=0,
        bc_n_epochs=6,
        eval_interval=SYNTH_EVAL_INTERVAL,
        output_dir=pathlib.Path(tmp_path),
        expert_cache_dir=pathlib.Path(tmp_path) / "unused",
        outer_early_stop=False,
        grow_batch_with_data=True,
        bc_batch_size=4,
        # Small enough that budgets >= 4 run the held-out split and budgets
        # below it take the fixed-epoch fallback.
        inner_early_stop_val_frac=0.25,
        inner_early_stop_min_val_size=1,
        inner_early_stop_patience=2,
        inner_early_stop_min_epochs=1,
        paired_cold_fits=True,
        fit_rule=rex.FIT_RULE_EPOCHS,
        eval_episodes=100,
    )
    params.update(overrides)
    return rex.ExperimentConfig(**params)


def _subset(transitions, n):
    return types.Transitions(
        obs=transitions.obs[:n].copy(),
        acts=transitions.acts[:n].copy(),
        next_obs=transitions.next_obs[:n].copy(),
        dones=transitions.dones[:n].copy(),
        infos=transitions.infos[:n].copy(),
    )


def _burn_rng(k):
    """An arbitrary, history-dependent consumption of every ambient stream."""
    th.manual_seed(1000 + k)
    th.rand(17 * k + 3)
    np.random.seed(2000 + k)
    np.random.rand(5 * k + 1)


def _assert_same_state(a, b):
    assert a.keys() == b.keys()
    for name in a:
        assert th.equal(a[name], b[name]), name


def _assert_same_optimizer(a, b):
    assert a["param_groups"] == b["param_groups"]
    assert a["state"].keys() == b["state"].keys()
    for idx in a["state"]:
        sa, sb = a["state"][idx], b["state"][idx]
        assert sa.keys() == sb.keys()
        for key in sa:
            if th.is_tensor(sa[key]):
                assert th.equal(sa[key], sb[key]), (idx, key)
            else:
                assert sa[key] == sb[key], (idx, key)


def _fixed_fit(cartpole, tmp_path, n, tag, policy=None):
    config = _config(tmp_path, "bc", n)
    if policy is None:
        policy = policy_utils.create_linear_policy(cartpole.expert)
    return rex._fit_fixed_bc(
        config,
        cartpole.venv,
        policy,
        _subset(cartpole.transitions, n),
        np.random.default_rng(SEED),
        device="cpu",
        tb_tag=tag,
    )


# ---------------------------------------------------------------------------
# Config: opt-in, versioned, legacy metadata unchanged
# ---------------------------------------------------------------------------


def test_flag_defaults_off_and_legacy_metadata_has_no_paired_keys(tmp_path):
    legacy = _config(tmp_path, "bc", 4, paired_cold_fits=False)
    default = rex.ExperimentConfig(
        **{
            k: v
            for k, v in vars(legacy).items()
            if k not in ("paired_cold_fits",) and not k.startswith("_")
        }
    )
    assert default.paired_cold_fits is False
    for config in (legacy, default):
        meta = rex._config_metadata(config)
        assert "paired_cold_fits" not in meta
        assert "paired_cold_fits_protocol" not in meta
        assert meta["protocol_version"] == 6

    paired = rex._config_metadata(_config(tmp_path, "bc", 4))
    assert paired["paired_cold_fits"] is True
    assert paired["paired_cold_fits_protocol"] == rex.PAIRED_COLD_FITS_PROTOCOL
    assert rex.PAIRED_COLD_FITS_PROTOCOL.endswith("/1")


def test_paired_protocol_rejects_warm_start_and_fixed_round_batches(tmp_path):
    with pytest.raises(ValueError, match="warm_start"):
        _config(tmp_path, "bc_prefix", 4, warm_start=True)
    with pytest.raises(ValueError, match="grow_batch_with_data"):
        _config(tmp_path, "bc_prefix", 4, grow_batch_with_data=False)
    # Legacy configs keep accepting both.
    _config(tmp_path, "ftrl", 4, warm_start=True, paired_cold_fits=False)
    _config(
        tmp_path,
        "bc_prefix",
        4,
        grow_batch_with_data=False,
        paired_cold_fits=False,
    )


def test_fit_and_eval_seeds_are_keyed_by_seed_and_label_count():
    a = rex.paired_fit_seeds(SEED, 6)
    assert a == rex.paired_fit_seeds(SEED, 6)
    assert a != rex.paired_fit_seeds(SEED, 7)
    assert a != rex.paired_fit_seeds(SEED + 1, 6)
    assert set(a) == {"init", "shuffle", "numpy"}
    assert rex.paired_eval_seed(SEED, 6) == rex.paired_eval_seed(SEED, 6)
    assert rex.paired_eval_seed(SEED, 6) != rex.paired_eval_seed(SEED, 7)
    assert rex.paired_eval_seed(SEED, 6) != rex.paired_eval_seed(SEED + 1, 6)


# ---------------------------------------------------------------------------
# Fitting: independent of RNG history, ambient streams restored
# ---------------------------------------------------------------------------


def test_paired_fixed_fit_ignores_rng_history_and_restores_ambient(
    cartpole,
    tmp_path,
):
    fits = []
    for k in (1, 2):
        _burn_rng(k)
        # Policy construction draws its own head from the ambient stream.
        policy = policy_utils.create_linear_policy(cartpole.expert)
        torch_before = th.get_rng_state()
        numpy_before = np.random.get_state()[1].copy()
        trainer, log = _fixed_fit(cartpole, tmp_path / f"h{k}", 8, f"h{k}", policy)
        assert th.equal(th.get_rng_state(), torch_before)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before)
        fits.append((trainer, log))

    (a, log_a), (b, log_b) = fits
    _assert_same_state(a.policy.state_dict(), b.policy.state_dict())
    _assert_same_optimizer(a.optimizer.state_dict(), b.optimizer.state_dict())
    assert log_a == log_b
    assert log_a["inner_es_fallback"] is None  # the held-out split ran
    assert log_a["fit_labels"] == 8
    assert log_a["fit_seed_key"] == [SEED, 8]
    assert log_a["fit_seeds"] == rex.paired_fit_seeds(SEED, 8)
    assert log_a["fit_protocol"] == rex.PAIRED_COLD_FITS_PROTOCOL
    assert log_a["fit_train_pairs_sha256"] != log_a["fit_val_pairs_sha256"]
    # The frozen features are the expert's, untouched by the cold reset.
    expert_state = cartpole.expert.state_dict()
    for name, value in a.policy.state_dict().items():
        if not name.startswith("action_net"):
            assert th.equal(value, expert_state[name]), name


def test_legacy_fixed_fit_still_consumes_the_ambient_stream(cartpole, tmp_path):
    """Sensitivity check: without the flag the fit depends on RNG history."""
    states = []
    for k in (1, 2):
        _burn_rng(k)
        policy = policy_utils.create_linear_policy(cartpole.expert)
        before = th.get_rng_state()
        config = _config(tmp_path / f"l{k}", "bc", 8, paired_cold_fits=False)
        trainer, log = rex._fit_fixed_bc(
            config,
            cartpole.venv,
            policy,
            _subset(cartpole.transitions, 8),
            np.random.default_rng(SEED),
            device="cpu",
            tb_tag=f"l{k}",
        )
        assert not th.equal(th.get_rng_state(), before)
        assert "fit_protocol" not in log
        states.append(trainer.policy.state_dict())
    assert not all(th.equal(states[0][n], states[1][n]) for n in states[0])


def test_cold_reset_clears_previous_optimizer_state(cartpole, tmp_path):
    """A trainer that already fitted other data refits N labels exactly cold."""
    fresh, _ = _fixed_fit(cartpole, tmp_path / "fresh", 8, "fresh")

    config = _config(tmp_path / "reused", "bc", 8)
    trainer = rex.bc.BC(
        observation_space=cartpole.venv.observation_space,
        action_space=cartpole.venv.action_space,
        rng=np.random.default_rng(1),
        policy=policy_utils.create_linear_policy(cartpole.expert),
        demonstrations=_subset(cartpole.transitions, 12),
        batch_size=4,
        optimizer_kwargs={"lr": config.learning_rate},
        device="cpu",
    )
    trainer.train(n_epochs=3, progress_bar=False)
    assert trainer.optimizer.state  # warm history to be discarded
    trainer.set_demonstrations(_subset(cartpole.transitions, 8))
    trainer.batch_size = trainer.minibatch_size = 4
    rex._inner_train(trainer, config, round_num=0, is_dagger=False, n_labels=8)
    _assert_same_state(trainer.policy.state_dict(), fresh.policy.state_dict())
    _assert_same_optimizer(
        trainer.optimizer.state_dict(),
        fresh.optimizer.state_dict(),
    )


def test_paired_fit_refuses_a_label_count_that_differs_from_the_data(
    cartpole,
    tmp_path,
):
    config = _config(tmp_path, "bc", 8)
    trainer = rex.bc.BC(
        observation_space=cartpole.venv.observation_space,
        action_space=cartpole.venv.action_space,
        rng=np.random.default_rng(1),
        policy=policy_utils.create_linear_policy(cartpole.expert),
        demonstrations=_subset(cartpole.transitions, 8),
        batch_size=4,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="labels"):
        rex._inner_train(trainer, config, round_num=0, is_dagger=False, n_labels=7)
    with pytest.raises(ValueError, match="n_labels"):
        rex._inner_train(trainer, config, round_num=0, is_dagger=False)


# ---------------------------------------------------------------------------
# Offline fixed BC vs BC-prefix: bitwise identical endpoints
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prefix_run(cartpole, tmp_path_factory):
    """A paired BC-prefix run with every fit's weights, optimizer and log."""
    out = tmp_path_factory.mktemp("prefix")
    config = _config(out, "bc_prefix", N_SYNTH, evaluate_first_round=True)
    snapshots = {}
    original = rex._inner_train

    def spy(trainer_or_bc, cfg, round_num, is_dagger, **kwargs):
        log = original(trainer_or_bc, cfg, round_num, is_dagger, **kwargs)
        bc_trainer = trainer_or_bc.bc_trainer
        snapshots[kwargs["n_labels"]] = (
            copy.deepcopy(bc_trainer.policy.state_dict()),
            copy.deepcopy(bc_trainer.optimizer.state_dict()),
            dict(log),
        )
        # Arbitrary RNG use between rounds must not reach later fits.
        _burn_rng(round_num)
        return log

    rex._inner_train = spy
    try:
        _burn_rng(9)
        rows = rex._run_dagger_variant(
            config,
            cartpole.venv,
            cartpole.expert,
            np.random.default_rng(SEED),
            cartpole.baselines,
            device="cpu",
            offline_data=cartpole.transitions,
        )
    finally:
        rex._inner_train = original
    return rows, snapshots


@pytest.mark.parametrize("n", [1, 4, 6, 12])
def test_offline_and_prefix_fits_are_bitwise_identical(
    cartpole,
    prefix_run,
    tmp_path,
    n,
):
    _, snapshots = prefix_run
    state, optimizer, log = snapshots[n]
    _burn_rng(50 + n)
    trainer, offline_log = _fixed_fit(cartpole, tmp_path, n, f"offline{n}")
    _assert_same_state(trainer.policy.state_dict(), state)
    _assert_same_optimizer(trainer.optimizer.state_dict(), optimizer)
    assert offline_log == log
    assert log["fit_data_pairs_sha256"] == rp.pairs_sha256(
        cartpole.transitions.obs[:n],
        cartpole.transitions.acts[:n],
    )
    if n >= 4:
        assert log["inner_es_fallback"] is None
        assert log["fit_val_pairs_sha256"] is not None
    else:
        assert log["inner_es_fallback"] == "dataset_too_small"
        assert log["fit_val_pairs_sha256"] is None


def test_prefix_checkpoints_and_evaluation_match_offline(
    cartpole,
    prefix_run,
    tmp_path,
):
    rows, _ = prefix_run
    evals = [r for r in rows if r.get("d_eval_size") is not None]
    assert [r["n_observations"] for r in evals] == [0, 1, 6, 12]
    for r in evals:
        assert r["eval_seed"] == rex.paired_eval_seed(SEED, r["n_observations"])
    final = evals[-1]
    trainer, _ = _fixed_fit(cartpole, tmp_path, N_SYNTH, "offline-final")
    saved = th.load(final["checkpoint"], map_location="cpu")
    _assert_same_state(saved["state_dict"], trainer.policy.state_dict())
    # Same policy and same budget seed: the same evaluation episodes.
    _burn_rng(77)
    offline = rex._compute_round_eval(
        trainer.policy,
        cartpole.expert,
        cartpole.venv,
        cartpole.baselines,
        n_episodes=100,
        eval_seed=rex.paired_eval_seed(SEED, N_SYNTH),
    )
    for key in (
        "episode_returns",
        "normalized_return",
        "disagreement_rate",
        "episode_lengths",
        "evaluation_episode_seeds",
        "initial_state_digests",
        "reset_accounting",
    ):
        assert offline[key] == final[key], key


# ---------------------------------------------------------------------------
# Evaluation: dedicated seed and optional episode outcome fields
# ---------------------------------------------------------------------------


def test_eval_seed_fixes_episodes_and_default_is_unchanged(cartpole):
    policy = policy_utils.create_linear_policy(cartpole.expert)
    runs = []
    for k in (1, 2):
        cartpole.venv.seed(k)
        cartpole.venv.reset()
        runs.append(
            rex._compute_round_eval(
                policy,
                cartpole.expert,
                cartpole.venv,
                cartpole.baselines,
                n_episodes=100,
                eval_seed=123,
            ),
        )
    assert runs[0]["eval_seed"] == 123
    assert runs[0]["episode_returns"] == runs[1]["episode_returns"]
    assert runs[0]["evaluation_episode_seeds"] == list(range(123, 223))
    legacy = rex._compute_round_eval(
        policy,
        cartpole.expert,
        cartpole.venv,
        cartpole.baselines,
    )
    assert "eval_seed" not in legacy
    assert "evaluation_episode_seeds" not in legacy
    assert len(legacy["episode_returns"]) == rex.EVAL_EPISODES == 1


def test_paired_episode_seeds_are_distinct_and_wrap_modulo_2_32():
    seeds = rex.paired_episode_seeds(7, 100)
    assert seeds == [7 + i for i in range(100)]
    top = rex.paired_episode_seeds(2**32 - 2, 100)
    assert top[:3] == [2**32 - 2, 2**32 - 1, 0]
    assert len(set(top)) == 100 and all(0 <= s < 2**32 for s in top)
    assert rex.paired_episode_seeds(rex.paired_eval_seed(SEED, 6)) == (
        rex.paired_episode_seeds(rex.paired_eval_seed(SEED, 6))
    )


def _constant_policy(expert, action):
    """Linear learner that always picks ``action`` (zero weight, one-hot bias)."""
    policy = policy_utils.create_linear_policy(expert)
    with th.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.zero_()
        policy.action_net.bias[action] = 5.0
    return policy


def test_seeded_eval_pairs_initial_states_across_policies_and_history():
    """Real CartPole: different episode lengths, same per-episode starts."""
    venv = rp.CountingVecEnv(
        env_utils.make_env("CartPole-v1", 1, np.random.default_rng(3)),
    )
    from stable_baselines3 import PPO

    expert = PPO("MlpPolicy", venv, device="cpu", seed=0).policy
    baselines = {"expert_return": 500.0, "random_return": 20.0}
    eval_seed = 2**32 - 40  # the list wraps past 2**32
    records = []
    for action, history in ((0, 5), (1, 37)):
        # Unrelated environment history before each evaluation.
        venv.seed(history)
        venv.reset()
        for _ in range(history):
            venv.step(np.array([history % 2]))
        resets_before, steps_before = venv.resets, venv.steps
        episodes_before = venv.completed_episodes
        record = rex._compute_round_eval(
            _constant_policy(expert, action),
            expert,
            venv,
            baselines,
            n_episodes=100,
            eval_seed=eval_seed,
        )
        # Counted by the wrapper, not inferred: one explicit reset per episode.
        assert venv.resets - resets_before == 100
        assert venv.steps - steps_before == record["d_eval_size"]
        assert venv.completed_episodes - episodes_before == 100
        assert record["reset_accounting"] == {
            "explicit_resets": 100,
            "auto_resets": 100,
            "discarded_auto_resets": 100,
        }
        records.append(record)
    venv.close()

    left, right = records
    expected = rex.paired_episode_seeds(eval_seed, 100)
    assert expected[40] == 0 and len(set(expected)) == 100
    for record in records:
        assert record["eval_seed"] == eval_seed
        assert record["evaluation_episode_seeds"] == expected
        assert len(record["initial_state_digests"]) == 100
        assert sum(record["episode_lengths"]) == record["d_eval_size"]
        json.dumps(record)
    assert left["episode_lengths"] != right["episode_lengths"]
    assert left["initial_state_digests"] == right["initial_state_digests"]
    assert len(set(left["initial_state_digests"])) == 100

    # Each digest is the state an independent environment resets to.
    fresh = env_utils.make_env("CartPole-v1", 1, np.random.default_rng(99))
    for i in (0, 40, 99):
        fresh.seed(expected[i])
        obs = fresh.reset()
        assert eval_utils.observation_digest(obs[0]) == left["initial_state_digests"][i]
    fresh.close()


def test_legacy_eval_keeps_the_original_rollout_call(cartpole, monkeypatch):
    calls = []

    def original_signature(
        policy,
        venv,
        n_episodes,
        deterministic=True,
        expert_policy=None,
        safety_step_cap=1_000_000,
    ):
        calls.append(n_episodes)
        return _fake_eval()

    monkeypatch.setattr(eval_utils, "eval_policy_rollout", original_signature)
    policy = policy_utils.create_linear_policy(cartpole.expert)
    record = rex._compute_round_eval(policy, cartpole.expert, None, cartpole.baselines)
    assert calls == [rex.EVAL_EPISODES]
    assert "eval_seed" not in record and "evaluation_episode_seeds" not in record
    assert record["evaluation_deterministic"] is True


@pytest.mark.parametrize("eval_seed", [None, 2**32 - 40])
def test_round_eval_declares_the_deterministic_rollout_it_ran(
    cartpole,
    monkeypatch,
    eval_seed,
):
    """The declaration is the argument the real rollout received."""
    real = eval_utils.eval_policy_rollout
    calls = []

    def spy(*args, **kwargs):
        calls.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(eval_utils, "eval_policy_rollout", spy)
    policy = policy_utils.create_linear_policy(cartpole.expert)
    cartpole.venv.seed(11)
    cartpole.venv.reset()
    record = rex._compute_round_eval(
        policy,
        cartpole.expert,
        cartpole.venv,
        cartpole.baselines,
        **({} if eval_seed is None else {"eval_seed": eval_seed}),
    )
    (call,) = calls
    assert call["deterministic"] is True
    assert call["n_episodes"] == rex.EVAL_EPISODES == 1
    assert record["evaluation_deterministic"] is call["deterministic"]
    assert len(record["episode_returns"]) == rex.EVAL_EPISODES
    if eval_seed is None:
        assert "episode_seeds" not in call
        assert "evaluation_episode_seeds" not in record
    else:
        seeds = rex.paired_episode_seeds(eval_seed)
        assert call["episode_seeds"] == seeds
        assert record["evaluation_episode_seeds"] == seeds
        assert record["eval_seed"] == eval_seed
    json.dumps(record)


def test_seeded_eval_refuses_a_rollout_without_the_requested_seeds(
    cartpole,
    monkeypatch,
):
    monkeypatch.setattr(
        eval_utils,
        "eval_policy_rollout",
        lambda *a, **k: _fake_eval(),
    )
    policy = policy_utils.create_linear_policy(cartpole.expert)
    with pytest.raises(RuntimeError, match="episode seeds"):
        rex._compute_round_eval(
            policy,
            cartpole.expert,
            None,
            cartpole.baselines,
            eval_seed=5,
        )


def test_evaluation_resets_sum_reports_and_never_impute():
    rows = [
        {"reset_accounting": {"explicit_resets": 100, "auto_resets": 100}},
        {"reset_accounting": {"explicit_resets": 1, "auto_resets": 100}},
    ]
    totals = rp._evaluation_resets(rows)
    assert totals["explicit_resets"] == 101 and totals["auto_resets"] == 200
    assert totals["discarded_auto_resets"] is None
    assert rp._evaluation_resets(rows + [{}])["explicit_resets"] is None
    assert rp._evaluation_resets([])["explicit_resets"] == 0


def _fake_eval(**extra):
    batch = eval_utils.RolloutBatch(
        obs=np.zeros((3, 4), dtype=np.float32),
        expert_actions=np.zeros(3, dtype=np.int64),
        learner_actions=np.zeros(3, dtype=np.int64),
    )
    result = eval_utils.EvalResult(
        mean_return=-150.0,
        episode_returns=[-100.0, -200.0],
        n_steps=3,
        rollout_batch=batch,
        current_round_disagreement=0.0,
        current_round_ce=0.1,
    )
    for key, value in extra.items():
        setattr(result, key, value)
    return result


def test_round_eval_persists_optional_episode_outcomes(cartpole, monkeypatch):
    policy = policy_utils.create_linear_policy(cartpole.expert)
    outcomes = dict(
        episode_lengths=[np.int64(2), 1],
        episode_terminated=[np.bool_(True), False],
        episode_truncated=[False, True],
        episode_successes=[None, np.bool_(False)],
    )
    monkeypatch.setattr(
        eval_utils,
        "eval_policy_rollout",
        lambda *a, **k: _fake_eval(**outcomes),
    )
    baselines = {"expert_return": 10.0, "random_return": 0.0}
    record = rex._compute_round_eval(policy, cartpole.expert, None, baselines)
    # Unclipped: far below the random baseline stays far below zero.
    assert record["normalized_return"] == -15.0
    assert record["episode_returns"] == [-100.0, -200.0]
    assert record["episode_lengths"] == [2, 1]
    assert record["episode_terminated"] == [True, False]
    assert record["episode_truncated"] == [False, True]
    assert record["episode_successes"] == [None, False]
    json.dumps(record)  # plain Python types only

    monkeypatch.setattr(
        eval_utils,
        "eval_policy_rollout",
        lambda *a, **k: _fake_eval(),
    )
    legacy = rex._compute_round_eval(policy, cartpole.expert, None, baselines)
    assert not set(outcomes) & set(legacy)
    assert legacy["normalized_return"] == -15.0


# ---------------------------------------------------------------------------
# Runner: the repaired /2 spec, all four methods, end to end on FrozenLake
# ---------------------------------------------------------------------------

ENV = "FrozenLake-v1"
LAKE_PATH = {0: 1, 4: 1, 8: 2, 9: 1, 13: 2, 14: 2}
BUDGET = 9
EVAL_INTERVAL = 3
CANDIDATE = "frozenlake_paired_test_candidate"


class Clock:
    """Deterministic clock that advances a fixed step per reading."""

    def __init__(self, step_seconds=0.001):
        self.now = datetime.datetime(2026, 10, 1, tzinfo=UTC)
        self.step = datetime.timedelta(seconds=step_seconds)

    def __call__(self):
        self.now += self.step
        return self.now


def _read(out):
    return json.loads((pathlib.Path(out) / rp.RESULT_FILE).read_text())


def _qualified_tabular_expert(tmp_path_factory, env_name, path, default=0):
    """A qualified one-hot expert acting ``path[s]`` in state ``s``."""
    from stable_baselines3 import PPO

    candidate = tmp_path_factory.mktemp("cache") / env_name / "model.zip"
    venv = env_utils.make_env(env_name, 1, np.random.default_rng(0))
    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(net_arch=dict(pi=[], vf=[])),
        device="cpu",
        seed=0,
    )
    with th.no_grad():
        weight = model.policy.action_net.weight
        weight.zero_()
        model.policy.action_net.bias.zero_()
        for s in range(weight.shape[1]):
            weight[path.get(s, default), s] = 10.0
    candidate.parent.mkdir(parents=True, exist_ok=True)
    model.save(candidate)
    venv.close()
    out = tmp_path_factory.mktemp(env_name) / "expert"
    code = task_experts.qualify_candidate(
        env_name,
        candidate,
        out,
        qualification_seed=1234,
        eval_episodes=3,
        deadline=FAR,
        clock=Clock(),
    )
    assert code == 0
    return out


@pytest.fixture(scope="module")
def lake_expert(tmp_path_factory):
    return _qualified_tabular_expert(tmp_path_factory, ENV, LAKE_PATH)


@pytest.fixture(scope="module")
def lake_data(lake_expert, tmp_path_factory):
    out = tmp_path_factory.mktemp("lake-data") / "data"
    code = task_pilot.prepare_data(
        out,
        lake_expert,
        env_name=ENV,
        seed=SEED,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        budget=BUDGET,
        baseline_episodes=2,
        clock=Clock(),
    )
    assert code == classical.EXIT_COMPLETE
    return out


def _run(expert, data, out, method, restriction_id="identity", **kwargs):
    params = dict(
        env_name=ENV,
        method=method,
        restriction_id=restriction_id,
        seed=SEED,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        n_rounds=BUDGET,
        eval_interval=EVAL_INTERVAL,
        clock=Clock(),
    )
    params.update(kwargs)
    return task_pilot.run_job(out, expert, data, **params)


@pytest.fixture(scope="module")
def paired_runs(lake_expert, lake_data, tmp_path_factory):
    root = tmp_path_factory.mktemp("paired")
    out = {}
    for method in task_pilot.RUN_METHODS:
        path = root / method
        code = _run(lake_expert, lake_data, path, method, paired_cold_fits=True)
        assert code == classical.EXIT_COMPLETE, method
        out[method] = (path, _read(path))
    return out


def test_paired_spec_names_protocol_2_and_runs_all_four_methods():
    legacy = task_pilot.runner_spec(ENV)
    paired = task_pilot.runner_spec(ENV, paired_cold_fits=True)
    assert task_pilot.PAIRED_PROTOCOL == "agnostic-classical-restriction/2"
    assert paired.protocol == task_pilot.PAIRED_PROTOCOL
    assert paired.paired_cold_fits is True and legacy.paired_cold_fits is False
    assert paired.methods == rp.METHODS + (rp.PREFIX_METHOD,)
    # Data jobs and their checks are shared with the legacy protocol.
    assert paired.data_protocol == legacy.data_protocol == task_pilot.PROTOCOL
    assert paired.source_files == legacy.source_files
    assert paired.run_restrictions == legacy.run_restrictions
    assert legacy == task_pilot.runner_spec(ENV, paired_cold_fits=False)
    assert legacy.protocol == task_pilot.PROTOCOL and legacy.methods == rp.METHODS
    with pytest.raises(ValueError, match="paired"):
        task_pilot.runner_spec(ENV, bc_prefix=True, paired_cold_fits=True)


def test_paired_runs_record_protocol_and_share_fixed_bc_endpoint(
    paired_runs,
    lake_data,
):
    pool = rp.ExpertData.load(lake_data / rp.POOL_FILE)
    for method, (_, record) in paired_runs.items():
        assert record["status"] == "complete", method
        assert record["protocol"] == task_pilot.PAIRED_PROTOCOL
        assert record["schema"] == task_pilot.PAIRED_PROTOCOL + "/run"
        assert record["data"]["acceptance"]["producer_protocol"] == task_pilot.PROTOCOL
        config = record["config"]
        assert config["paired_cold_fits"] == rp.PAIRED_COLD_FITS_RULE
        assert config["experiment"]["paired_cold_fits"] is True
        assert config["experiment"]["warm_start"] is False
        for row in record["records"]:
            if row.get("d_eval_size") is not None:
                assert row["eval_seed"] == rex.paired_eval_seed(
                    SEED,
                    row["n_observations"],
                )
            if row.get("inner_es_stop_epoch") is not None:
                assert row["fit_seed_key"] == [SEED, row["n_observations"]]

    bc_path, bc = paired_runs["bc"]
    prefix_path, prefix = paired_runs["bc_prefix"]
    (fixed,) = bc["records"]
    final = [r for r in prefix["records"] if r["n_observations"] == BUDGET][-1]
    assert fixed["fit_data_pairs_sha256"] == pool.pairs_sha256(BUDGET)
    for key in (
        "fit_data_pairs_sha256",
        "fit_train_pairs_sha256",
        "fit_val_pairs_sha256",
        "fit_seeds",
        "inner_es_stop_epoch",
        "inner_es_val_nll_trajectory",
        "inner_es_fallback",
        "train_batch_size",
        "train_examples",
        "eval_seed",
        "episode_returns",
        "normalized_return",
        "disagreement_rate",
        "evaluation_episode_seeds",
        "initial_state_digests",
        "episode_lengths",
    ):
        assert fixed[key] == final[key], key
    a = restriction.load_policy_checkpoint(bc_path / fixed["checkpoint"])
    b = restriction.load_policy_checkpoint(prefix_path / final["checkpoint"])
    _assert_same_state(a.state_dict(), b.state_dict())


def test_paired_runs_share_episode_seeds_per_budget_across_methods(paired_runs):
    by_budget = {}
    for method, (_, record) in paired_runs.items():
        evals = [r for r in record["records"] if r.get("d_eval_size") is not None]
        for row in evals:
            budget = row["n_observations"]
            assert row["evaluation_episode_seeds"] == rex.paired_episode_seeds(
                rex.paired_eval_seed(SEED, budget),
                record["config"]["eval_episodes"],
            )
            by_budget.setdefault(budget, []).append(
                (method, row["evaluation_episode_seeds"], row["initial_state_digests"]),
            )
            # Run metadata stays with the calibration adapter, not the rows.
            assert not {"final", "mask_sha256", "expert_sha256"} & set(row)
        resets = record["accounting"]["completed_records"]["evaluation"]["resets"]
        assert resets["explicit_resets"] == 100 * len(evals)
        assert (
            resets["explicit_resets"] <= record["accounting"]["observed"]["env_resets"]
        )
    assert {m for m, _, _ in by_budget[BUDGET]} == set(task_pilot.RUN_METHODS)
    for entries in by_budget.values():
        assert len({json.dumps([s, d]) for _, s, d in entries}) == 1


def test_every_runner_evaluation_declares_deterministic_rollouts(paired_runs):
    for _, record in paired_runs.values():
        assert record["config"]["eval_deterministic"] is True
        evals = [r for r in record["records"] if r.get("d_eval_size") is not None]
        assert evals
        for row in evals:
            assert row["evaluation_deterministic"] is True
            assert row["evaluation_episode_seeds"] == rex.paired_episode_seeds(
                row["eval_seed"],
                record["config"]["eval_episodes"],
            )


def test_legacy_runs_keep_their_records_without_paired_fields(
    lake_expert,
    lake_data,
    tmp_path,
):
    out = tmp_path / "legacy"
    assert _run(lake_expert, lake_data, out, "bc") == classical.EXIT_COMPLETE
    record = _read(out)
    assert record["protocol"] == task_pilot.PROTOCOL
    assert "paired_cold_fits" not in record["config"]
    assert "paired_cold_fits" not in record["config"]["experiment"]
    (row,) = record["records"]
    assert "eval_seed" not in row
    assert "fit_protocol" not in row


def test_legacy_spec_refuses_paired_only_combinations(
    lake_expert,
    lake_data,
    tmp_path,
):
    common = dict(
        restriction_id="identity",
        seed=SEED,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        n_rounds=BUDGET,
        eval_interval=EVAL_INTERVAL,
        clock=Clock(),
    )
    # The legacy prefix spec still runs BC-prefix only.
    out = tmp_path / "iid"
    code = rp.run_job(
        out,
        lake_expert,
        lake_data,
        spec=task_pilot.runner_spec(ENV, bc_prefix=True),
        method="bc_iid",
        **common,
    )
    assert code == classical.EXIT_REFUSED and not out.exists()


# ---------------------------------------------------------------------------
# Calibration candidates: opt-in extra masks bound into the spec
# ---------------------------------------------------------------------------


def _jsonable(value):
    return json.loads(json.dumps(value))


@pytest.fixture
def candidate(monkeypatch):
    monkeypatch.setitem(
        restriction.RESTRICTIONS,
        CANDIDATE,
        restriction.Restriction(
            restriction_id=CANDIDATE,
            zeroed_indices=(),
            obs_shape=None,
            description="Test-only FrozenLake calibration candidate.",
        ),
    )
    monkeypatch.setattr(
        tasks,
        "CALIBRATION_CANDIDATES",
        {ENV: (CANDIDATE,)},
        raising=False,
    )
    return CANDIDATE


def test_candidates_are_opt_in_and_extend_only_the_restrictions(candidate):
    task = tasks.get_task(ENV)
    legacy = task_pilot.runner_spec(ENV)
    assert candidate not in legacy.run_restrictions
    assert legacy.calibration_candidates == ()
    spec = task_pilot.runner_spec(ENV, calibration_candidates=True)
    assert spec.run_restrictions == task.restriction_ids + (candidate,)
    assert spec.calibration_candidates == (candidate,)
    assert spec.protocol == legacy.protocol and spec.methods == legacy.methods
    both = task_pilot.runner_spec(
        ENV,
        paired_cold_fits=True,
        calibration_candidates=True,
    )
    assert both.protocol == task_pilot.PAIRED_PROTOCOL
    assert both.run_restrictions == spec.run_restrictions
    # The task identity is unchanged: candidates are not part of the ladder.
    assert tasks.describe(task)["declared"]["ladder"] == list(task.ladder)


def test_candidates_must_be_declared_and_registered(monkeypatch):
    monkeypatch.setattr(tasks, "CALIBRATION_CANDIDATES", {}, raising=False)
    with pytest.raises(ValueError, match="calibration candidates"):
        task_pilot.runner_spec(ENV, calibration_candidates=True)
    monkeypatch.setattr(
        tasks,
        "CALIBRATION_CANDIDATES",
        {ENV: ("not_a_registered_mask",)},
        raising=False,
    )
    with pytest.raises(ValueError):
        task_pilot.runner_spec(ENV, calibration_candidates=True)


def test_candidate_run_is_refused_without_the_opt_in(
    candidate,
    lake_expert,
    lake_data,
    tmp_path,
):
    out = tmp_path / "plain"
    assert _run(lake_expert, lake_data, out, "bc", candidate) == (
        classical.EXIT_REFUSED
    )
    assert not out.exists()


def test_candidate_run_binds_candidate_identity(
    candidate,
    lake_expert,
    lake_data,
    tmp_path,
):
    out = tmp_path / "cand"
    code = _run(
        lake_expert,
        lake_data,
        out,
        "bc",
        candidate,
        calibration_candidates=True,
    )
    assert code == classical.EXIT_COMPLETE
    config = _read(out)["config"]
    described = _jsonable(restriction.describe_restriction(candidate))
    assert config["restriction"] == described
    assert config["restriction_sha256"] == rp.pilot.config_digest(described)
    assert config["calibration_candidates"] == {
        candidate: rp.pilot.config_digest(described),
    }


def test_audit_validates_against_the_spec_restrictions(
    candidate,
    lake_expert,
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_audit(expert, env_name, restriction_id, reset_seed, check):
        calls.append(restriction_id)
        return {"status": "complete", "expert_predict_calls": 0}

    monkeypatch.setattr(task_audit, "audit", fake_audit)
    common = dict(env_name=ENV, deadline=FAR, job_limit_seconds=LIMIT, clock=Clock())
    plain = tmp_path / "plain"
    code = task_pilot.run_audit(plain, lake_expert, restriction_id=candidate, **common)
    assert code == classical.EXIT_REFUSED and not plain.exists()
    out = tmp_path / "cand"
    code = task_pilot.run_audit(
        out,
        lake_expert,
        restriction_id=candidate,
        calibration_candidates=True,
        **common,
    )
    assert code == classical.EXIT_COMPLETE and calls == [candidate]
    record = _read(out)
    assert record["config"]["restriction_id"] == candidate
    assert record["config"]["restriction"] == _jsonable(
        restriction.describe_restriction(candidate),
    )
    # identity is a run restriction, never an audit target.
    ident = tmp_path / "identity"
    code = task_pilot.run_audit(
        ident,
        lake_expert,
        restriction_id="identity",
        calibration_candidates=True,
        **common,
    )
    assert code == classical.EXIT_REFUSED


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_run(tmp_path, monkeypatch, method, *flags):
    seen = {}

    def fake_run_job(output_dir, expert_dir, data_dir, *, spec, **kwargs):
        seen.update(spec=spec, kwargs=kwargs)
        pathlib.Path(output_dir).mkdir()
        (pathlib.Path(output_dir) / rp.RESULT_FILE).write_text('{"status": "complete"}')
        return 0

    monkeypatch.setattr(rp, "run_job", fake_run_job)
    out = tmp_path / f"cli-{method}-{len(flags)}"
    argv = [
        "run",
        "--env",
        ENV,
        "--expert-dir",
        "e",
        "--output-dir",
        str(out),
        "--deadline",
        "2026-10-07T00:00:00Z",
        "--job-limit-seconds",
        "60",
        "--data-dir",
        "d",
        "--seed",
        str(SEED),
        "--method",
        method,
        "--restriction",
        "identity",
        *flags,
    ]
    assert task_pilot.main(argv) == 0
    return seen


@pytest.mark.parametrize("method", task_pilot.RUN_METHODS)
def test_cli_paired_flag_selects_the_paired_spec_for_every_method(
    tmp_path,
    monkeypatch,
    method,
):
    seen = _cli_run(tmp_path, monkeypatch, method, "--paired-cold-fits")
    assert seen["spec"] == task_pilot.runner_spec(ENV, paired_cold_fits=True)
    assert seen["kwargs"]["method"] == method
    legacy = _cli_run(tmp_path, monkeypatch, method)
    assert legacy["spec"] == task_pilot.runner_spec(
        ENV,
        bc_prefix=method == rp.PREFIX_METHOD,
    )


def test_cli_passes_candidates_by_spec(tmp_path, monkeypatch, candidate):
    seen = _cli_run(
        tmp_path,
        monkeypatch,
        "bc",
        "--paired-cold-fits",
        "--calibration-candidates",
    )
    assert seen["spec"] == task_pilot.runner_spec(
        ENV,
        paired_cold_fits=True,
        calibration_candidates=True,
    )
    assert candidate in seen["spec"].run_restrictions


def test_cli_audit_accepts_the_candidate_flag(tmp_path, monkeypatch, candidate):
    seen = {}

    def fake_audit(output_dir, expert_dir, **kwargs):
        seen.update(kwargs)
        pathlib.Path(output_dir).mkdir()
        (pathlib.Path(output_dir) / rp.RESULT_FILE).write_text('{"status": "complete"}')
        return 0

    monkeypatch.setattr(task_pilot, "run_audit", fake_audit)
    argv = [
        "audit",
        "--env",
        ENV,
        "--expert-dir",
        "e",
        "--output-dir",
        str(tmp_path / "audit"),
        "--deadline",
        "2026-10-07T00:00:00Z",
        "--job-limit-seconds",
        "60",
        "--restriction",
        candidate,
        "--calibration-candidates",
    ]
    assert task_pilot.main(argv) == 0
    assert seen["calibration_candidates"] is True
    assert seen["restriction_id"] == candidate


# ---------------------------------------------------------------------------
# Every environment, CliffWalking included: uniform random lower reference
# ---------------------------------------------------------------------------

CLIFF = "CliffWalking-v0"
# Optimal path: up from the start 36, right along row 2, down from 35 to 47.
CLIFF_PATH = {36: 0, **{s: 1 for s in range(24, 35)}, 35: 2}
CLIFF_BUDGET = 6
STALLED_BASELINES = {
    "expert_return": -13.0,
    "random_return": -200.0,
    "expert_self_ce": 0.1,
    "measured_uniform_random_return": -5000.0,
    "normalization_reference_type": "stalled_policy",
    "stalled_return": -200.0,
}


def test_runner_has_no_stalled_reference_override():
    for name in (
        "STALLED_REFERENCE_ENVS",
        "STALLED_REFERENCE_RULE",
        "PRE_STALLED_RULE",
        "normalization_reference",
    ):
        assert not hasattr(rp, name)
    for env in ("CliffWalking-v0", "CliffWalkingVariedStart-v0", "CartPole-v1"):
        assert env_baselines.normalization_reference(env, -5000.0) == {}


def test_reference_descriptor_reads_old_baselines_as_recorded():
    old = {"expert_return": -13.0, "random_return": -5000.0, "expert_self_ce": 0.1}
    described = rp._reference_descriptor(old)
    assert described["type"] == "uniform_random"
    assert described["lower_return"] == -5000.0
    assert described["measured_uniform_random_return"] == -5000.0
    assert described["reference_type_recorded"] is False
    assert described["rule"] == rp.UNIFORM_RANDOM_REFERENCE_RULE
    assert "stalled" not in json.dumps(described)
    assert old == {
        "expert_return": -13.0,
        "random_return": -5000.0,
        "expert_self_ce": 0.1,
    }
    labelled = {**old, "normalization_reference_type": "uniform_random"}
    assert rp._reference_descriptor(labelled)["reference_type_recorded"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"normalization_reference_type": "uniform_random"},
        {"normalization_reference_type": "uniform_random", "stalled_return": None},
    ],
)
def test_stalled_or_inconsistent_baselines_are_refused_unchanged(changes):
    baselines = {**STALLED_BASELINES, **changes}
    before = json.dumps(baselines, sort_keys=True)
    assert not rp._uniform_random_baselines(baselines)
    with pytest.raises(classical.Refused, match="uniform random"):
        rp._reference_descriptor(baselines)
    assert json.dumps(baselines, sort_keys=True) == before


def test_uniform_random_metadata_must_match_the_random_return():
    old = {"expert_return": -13.0, "random_return": -5000.0, "expert_self_ce": 0.1}
    assert rp._uniform_random_baselines(old)
    assert rp._uniform_random_baselines(
        {**old, "measured_uniform_random_return": -5000.0},
    )
    assert not rp._uniform_random_baselines(
        {**old, "measured_uniform_random_return": -4000.0},
    )
    assert not rp._uniform_random_baselines({**old, "stalled_return": -200.0})
    assert not rp._uniform_random_baselines(None)


@pytest.fixture(scope="module")
def cliff_expert(tmp_path_factory):
    return _qualified_tabular_expert(tmp_path_factory, CLIFF, CLIFF_PATH)


def test_cliff_baselines_keep_the_measured_random_return(cliff_expert):
    spec = task_pilot.runner_spec(CLIFF)
    expert, _ = rp._load_expert(cliff_expert, spec)
    baselines, costs = rp.measure_baselines(expert, SEED, 2, env_name=CLIFF)
    assert set(baselines) == {"expert_return", "random_return", "expert_self_ce"}
    assert baselines["expert_return"] == -13.0
    # Uniform random actions with a 200-step cap: at best -200, the stall value
    # only if no step falls off the cliff; never replaced by it.
    assert baselines["random_return"] <= -200.0
    assert costs["random_episodes"] == 2
    assert rp._finite_baselines(baselines)
    assert rp._uniform_random_baselines(baselines)


@pytest.fixture(scope="module")
def cliff_data(cliff_expert, tmp_path_factory):
    data = tmp_path_factory.mktemp("cliff-data") / "data"
    code = task_pilot.prepare_data(
        data,
        cliff_expert,
        env_name=CLIFF,
        seed=SEED,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        budget=CLIFF_BUDGET,
        baseline_episodes=2,
        clock=Clock(),
    )
    assert code == classical.EXIT_COMPLETE
    return data


def _cliff_run(out, cliff_expert, data):
    return task_pilot.run_job(
        out,
        cliff_expert,
        data,
        env_name=CLIFF,
        method="bc",
        restriction_id="identity",
        seed=SEED,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        n_rounds=CLIFF_BUDGET,
        eval_interval=3,
        clock=Clock(),
        paired_cold_fits=True,
    )


def test_cliff_run_normalizes_with_the_uniform_random_reference(
    cliff_expert,
    cliff_data,
    tmp_path_factory,
):
    baselines = _read(cliff_data)["baselines"]
    assert set(baselines) == {"expert_return", "random_return", "expert_self_ce"}
    random_return = baselines["random_return"]
    assert random_return <= -200.0

    out = tmp_path_factory.mktemp("cliff-run") / "bc"
    assert _cliff_run(out, cliff_expert, cliff_data) == classical.EXIT_COMPLETE
    record = _read(out)
    chosen = record["config"]["normalization_reference"]
    assert chosen == {
        "type": "uniform_random",
        "lower_return": random_return,
        "expert_return": -13.0,
        "measured_uniform_random_return": random_return,
        "reference_type_recorded": False,
        "rule": rp.UNIFORM_RANDOM_REFERENCE_RULE,
    }
    (row,) = record["records"]
    assert row["evaluation_deterministic"] is True
    mean = float(np.mean(row["episode_returns"]))
    assert row["normalized_return"] == round(
        (mean - random_return) / (-13.0 - random_return),
        6,
    )


def test_cliff_data_with_stalled_baselines_is_refused_unmodified(
    cliff_expert,
    cliff_data,
    tmp_path_factory,
):
    historical = tmp_path_factory.mktemp("cliff-stalled") / "data"
    shutil.copytree(cliff_data, historical)
    path = historical / rp.RESULT_FILE
    record = json.loads(path.read_text())
    record["baselines"] = {
        **record["baselines"],
        "measured_uniform_random_return": record["baselines"]["random_return"],
        "random_return": -200.0,
        "normalization_reference_type": "stalled_policy",
        "stalled_return": -200.0,
    }
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    out = tmp_path_factory.mktemp("cliff-stalled-run") / "bc"
    assert _cliff_run(out, cliff_expert, historical) == classical.EXIT_REFUSED
    assert path.read_bytes() == before
    assert not (out / rp.RESULT_FILE).exists()


def test_non_cliff_runs_record_the_uniform_random_reference(paired_runs):
    for _, record in paired_runs.values():
        baselines = record["data"]["baselines"]
        assert set(baselines) == {"expert_return", "random_return", "expert_self_ce"}
        chosen = record["config"]["normalization_reference"]
        assert chosen["type"] == "uniform_random"
        assert chosen["lower_return"] == baselines["random_return"]
        assert chosen["measured_uniform_random_return"] == baselines["random_return"]


# ---------------------------------------------------------------------------
# Variant tasks: their ladder is already new, no calibration flag
# ---------------------------------------------------------------------------

VARIANT = "FrozenLakeVariedStart-v0"


def test_variant_spec_runs_its_ladder_without_the_candidate_flag():
    task = tasks.get_task(VARIANT)
    spec = task_pilot.runner_spec(VARIANT)
    assert spec.run_restrictions == task.restriction_ids
    assert spec.calibration_candidates == ()
    paired = task_pilot.runner_spec(VARIANT, paired_cold_fits=True)
    assert paired.run_restrictions == task.restriction_ids
    with pytest.raises(ValueError, match="calibration candidates"):
        task_pilot.runner_spec(VARIANT, calibration_candidates=True)


def test_variant_audit_refuses_only_the_candidate_opt_in(tmp_path, monkeypatch):
    task = tasks.get_task(VARIANT)
    out = tmp_path / "cand"
    code = task_pilot.run_audit(
        out,
        tmp_path / "no-expert",
        env_name=VARIANT,
        restriction_id=task.ladder[0],
        deadline=FAR,
        job_limit_seconds=LIMIT,
        clock=Clock(),
        calibration_candidates=True,
    )
    assert code == classical.EXIT_REFUSED and not out.exists()


@pytest.mark.parametrize("command", ["audit", "run", "data"])
def test_cli_accepts_variant_tasks_without_flags(tmp_path, monkeypatch, command):
    task = tasks.get_task(VARIANT)
    seen = {}

    def fake(output_dir, *args, **kwargs):
        seen.update(kwargs)
        pathlib.Path(output_dir).mkdir()
        (pathlib.Path(output_dir) / rp.RESULT_FILE).write_text('{"status": "complete"}')
        return 0

    target = {"audit": "run_audit", "run": "run_job", "data": "prepare_data"}
    monkeypatch.setattr(task_pilot, target[command], fake)
    argv = [
        command,
        "--env",
        VARIANT,
        "--expert-dir",
        "e",
        "--output-dir",
        str(tmp_path / command),
        "--deadline",
        "2026-10-07T00:00:00Z",
        "--job-limit-seconds",
        "60",
    ]
    if command == "audit":
        argv += ["--restriction", task.ladder[0]]
    if command == "data":
        argv += ["--seed", str(SEED)]
    if command == "run":
        argv += [
            "--data-dir",
            "d",
            "--seed",
            str(SEED),
            "--method",
            "bc",
            "--restriction",
            task.ladder[0],
        ]
    assert task_pilot.main(argv) == 0
    assert seen["env_name"] == VARIANT
    if command != "data":
        assert seen["calibration_candidates"] is False


# ---------------------------------------------------------------------------
# Generic runner integration: BC epoch cap, constructed grids, task union
# ---------------------------------------------------------------------------


def test_default_bc_epochs_keep_the_config_and_an_explicit_cap_replaces_it(
    lake_expert,
    lake_data,
    tmp_path,
):
    legacy, default = tmp_path / "legacy", tmp_path / "default"
    assert _run(lake_expert, lake_data, legacy, "bc") == classical.EXIT_COMPLETE
    code = _run(lake_expert, lake_data, default, "bc", bc_n_epochs=None)
    assert code == classical.EXIT_COMPLETE
    assert _read(default)["config"] == _read(legacy)["config"]
    assert _read(default)["config_sha256"] == _read(legacy)["config_sha256"]
    assert _read(legacy)["config"]["experiment"]["bc_n_epochs"] == rp.BC_EPOCHS

    capped = tmp_path / "capped"
    code = _run(lake_expert, lake_data, capped, "bc", bc_n_epochs=2)
    assert code == classical.EXIT_COMPLETE
    record = _read(capped)
    assert record["config"]["experiment"]["bc_n_epochs"] == 2
    expected = copy.deepcopy(_read(legacy)["config"])
    expected["experiment"]["bc_n_epochs"] = 2
    assert record["config"] == expected
    training = record["accounting"]["completed_records"]["training"]
    assert 0 < training["epochs"] <= 2 * training["fits"]


@pytest.mark.parametrize("bad", [True, False, 0, -1, 2.5, "3"])
def test_bad_bc_epochs_are_refused_before_the_output_claim(
    lake_expert,
    lake_data,
    tmp_path,
    bad,
):
    out = tmp_path / "bad"
    code = _run(lake_expert, lake_data, out, "bc", bc_n_epochs=bad)
    assert code == classical.EXIT_REFUSED and not out.exists()


def test_cartpole_pilot_spec_refuses_a_bc_epoch_override(tmp_path):
    out = tmp_path / "cartpole"
    code = rp.run_job(
        out,
        tmp_path / "no-expert",
        tmp_path / "no-data",
        method="bc",
        restriction_id="identity",
        seed=SEED,
        deadline=FAR,
        job_limit_seconds=LIMIT,
        clock=Clock(),
        bc_n_epochs=200,
    )
    assert code == classical.EXIT_REFUSED and not out.exists()


def test_cli_bc_epochs_is_run_only_and_defaults_to_none(tmp_path, monkeypatch):
    assert _cli_run(tmp_path, monkeypatch, "bc")["kwargs"]["bc_n_epochs"] is None
    seen = _cli_run(tmp_path, monkeypatch, "ftl", "--bc-epochs", "200")
    assert seen["kwargs"]["bc_n_epochs"] == 200
    for bad in ("0", "-3", "2.5", "abc"):
        with pytest.raises(SystemExit):
            _cli_run(tmp_path, monkeypatch, "bc", "--bc-epochs", bad)
    parser = task_pilot.build_parser()
    for command in ("audit", "data"):
        with pytest.raises(SystemExit):
            parser.parse_args([command, "--bc-epochs", "200"])


def test_audit_grids_resolve_candidates_and_keep_ladder_metadata():
    for rid, grids in task_audit.CONSTRUCTED_GRIDS.items():
        assert task_pilot._constructed_grids(rid) == grids
    for rid in getattr(task_audit, "CANDIDATE_GRIDS", {}):
        assert task_pilot._constructed_grids(rid) == task_audit.constructed_grids(rid)
        assert task_pilot._constructed_grids(rid)
    assert task_pilot._constructed_grids("identity") == ()
    assert task_pilot._constructed_grids(
        "cart_position_angular_velocity_zero",
    ) == task_audit.CONSTRUCTED_GRIDS.get("cart_position_angular_velocity_zero", ())


def test_every_declared_task_runs_legacy_by_default_and_paired_by_opt_in():
    declared = sorted(tasks.TASKS) + sorted(getattr(tasks, "VARIANT_TASKS", {}))
    run = task_pilot.build_parser()._subparsers._group_actions[0].choices["run"]
    (env_action,) = [a for a in run._actions if a.dest == "env"]
    assert list(env_action.choices) == declared
    for env_name in declared:
        legacy = task_pilot.runner_spec(env_name)
        assert legacy.paired_cold_fits is False
        assert legacy.protocol == task_pilot.PROTOCOL
        assert legacy.methods == rp.METHODS
        paired = task_pilot.runner_spec(env_name, paired_cold_fits=True)
        assert paired.protocol == task_pilot.PAIRED_PROTOCOL
        assert paired.run_restrictions == legacy.run_restrictions
