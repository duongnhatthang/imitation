"""Learner-only observation restriction: policy, checkpoint, and audit contracts.

The restriction must act on the learner's inputs only, before the frozen
features, on every policy path, without consuming RNG, and it must survive a
checkpoint round trip. The identity restriction must reproduce the previous
linear policy and fit mechanics exactly.
"""

import copy
import itertools

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO

from imitation.algorithms import bc, ftrl
from imitation.data import types
from imitation.experiments.agnostic import restriction
from imitation.experiments.ftrl import env_utils, policy_utils, run_experiment

BASELINES = {"expert_return": 500.0, "random_return": 22.0}
STRONG = "cart_position_angular_velocity_zero"
MASKS = ("cart_position_zero", STRONG)
# Observation coordinates each mask hides (x is 0, theta_dot is 3).
HIDDEN = {"cart_position_zero": (0,), STRONG: (0, 3)}


@pytest.fixture(scope="module")
def expert():
    """A CartPole PPO policy with the pipeline's [64, 64] architecture."""
    venv = env_utils.make_env("CartPole-v1", n_envs=1, rng=np.random.default_rng(0))
    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(net_arch=[64, 64]),
        device="cpu",
        seed=0,
    )
    venv.close()
    return model.policy


def _obs(n=64, seed=0):
    rng = np.random.default_rng(seed)
    scale = np.array([2.0, 1.5, 0.2, 1.5], dtype=np.float32)
    return (rng.uniform(-1, 1, size=(n, 4)) * scale).astype(np.float32)


def _with_x(obs, x):
    out = obs.copy()
    out[:, 0] = x
    return out


def _with(obs, indices, value):
    out = obs.copy()
    out[:, list(indices)] = value
    return out


def _outputs(policy, obs):
    """Deterministic actions and every tensor-valued policy path on ``obs``."""
    acts, _ = policy.predict(obs, deterministic=True)
    tensor = th.as_tensor(obs)
    probe = th.as_tensor(np.arange(len(obs)) % 2)
    with th.no_grad():
        values, log_prob, entropy = policy.evaluate_actions(tensor, probe)
        probs = policy.get_distribution(tensor).distribution.probs
        predicted_values = policy.predict_values(tensor)
        greedy, forward_values, _ = policy(tensor, deterministic=True)
    return {
        "acts": acts,
        "log_prob": log_prob.numpy(),
        "values": values.numpy(),
        "entropy": entropy.numpy(),
        "probs": probs.numpy(),
        "predict_values": predicted_values.numpy(),
        "forward_actions": greedy.numpy(),
        "forward_values": forward_values.numpy(),
    }


def _assert_same_outputs(a, b):
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def _assert_same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sorted(sa) == sorted(sb)
    for name in sa:
        assert th.equal(sa[name], sb[name]), name


def _trainable(policy):
    return sorted(n for n, p in policy.named_parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Identity reproduces the previous linear policy
# ---------------------------------------------------------------------------


def test_identity_factory_reproduces_previous_linear_policy(expert):
    th.manual_seed(7)
    previous = policy_utils.create_linear_policy(expert)
    rng_after_previous = th.get_rng_state()
    th.manual_seed(7)
    identity = restriction.make_linear_policy(expert, "identity")
    rng_after_identity = th.get_rng_state()

    assert th.equal(rng_after_previous, rng_after_identity)
    _assert_same_weights(previous, identity)
    assert _trainable(identity) == _trainable(previous)
    assert _trainable(identity) == ["action_net.bias", "action_net.weight"]
    obs = _obs()
    _assert_same_outputs(_outputs(previous, obs), _outputs(identity, obs))


@pytest.mark.parametrize("restriction_id", MASKS)
def test_head_initialization_is_identical_across_conditions(expert, restriction_id):
    th.manual_seed(3)
    full = restriction.make_linear_policy(expert, "identity")
    rng_full = th.get_rng_state()
    th.manual_seed(3)
    restricted = restriction.make_linear_policy(expert, restriction_id)
    rng_restricted = th.get_rng_state()

    assert th.equal(rng_full, rng_restricted)
    _assert_same_weights(full, restricted)


# ---------------------------------------------------------------------------
# Restriction placement
# ---------------------------------------------------------------------------


def test_stronger_mask_keeps_cart_velocity_and_pole_angle_only():
    mask = restriction.get_restriction(STRONG)
    obs = _obs(n=8, seed=2)
    masked = mask.apply(obs)

    assert mask.zeroed_indices == (0, 3) and mask.obs_shape == (4,)
    np.testing.assert_array_equal(masked[:, [0, 3]], 0.0)
    np.testing.assert_array_equal(masked[:, [1, 2]], obs[:, [1, 2]])
    with th.no_grad():
        torch_masked = restriction.ObservationMask(STRONG)(th.as_tensor(obs))
    np.testing.assert_array_equal(torch_masked.numpy(), masked)
    # The historical x-only mask stays registered so old checkpoints load.
    assert restriction.get_restriction("cart_position_zero").zeroed_indices == (0,)


@pytest.mark.parametrize("restriction_id", MASKS)
def test_restriction_zeroes_hidden_coordinates_before_frozen_features(
    expert,
    restriction_id,
):
    hidden = HIDDEN[restriction_id]
    th.manual_seed(11)
    full = restriction.make_linear_policy(expert, "identity")
    th.manual_seed(11)
    restricted = restriction.make_linear_policy(expert, restriction_id)
    obs = _obs()

    # Every path of the restricted learner equals the unrestricted twin
    # evaluated with the hidden coordinates set to 0 ...
    _assert_same_outputs(
        _outputs(restricted, obs),
        _outputs(full, _with(obs, hidden, 0.0)),
    )
    # ... so it cannot distinguish states that differ only in them.
    for value in (-1.8, 0.6, 2.3):
        for index in hidden:
            _assert_same_outputs(
                _outputs(restricted, obs),
                _outputs(restricted, _with(obs, (index,), value)),
            )


@pytest.mark.parametrize("restriction_id", MASKS)
def test_all_extractor_aliases_are_restricted_and_only_the_head_trains(
    expert,
    restriction_id,
):
    policy = restriction.make_linear_policy(expert, restriction_id)

    assert policy.pi_features_extractor is policy.features_extractor
    assert policy.vf_features_extractor is policy.features_extractor
    assert policy.restriction_id == restriction_id
    assert _trainable(policy) == ["action_net.bias", "action_net.weight"]
    for name, param in expert.named_parameters():
        frozen = dict(policy.named_parameters())[name]
        if not name.startswith("action_net"):
            assert th.equal(frozen, param), name


def test_masking_the_learner_never_alters_expert_inputs(expert):
    before = copy.deepcopy(expert)
    obs = _obs()
    shifted = _with(obs, (0, 3), 1.9)

    restriction.make_linear_policy(expert, STRONG)

    _assert_same_weights(expert, before)
    assert type(expert.features_extractor) is type(before.features_extractor)
    _assert_same_outputs(_outputs(expert, obs), _outputs(before, obs))
    _assert_same_outputs(_outputs(expert, shifted), _outputs(before, shifted))


def test_unknown_restriction_and_incompatible_observations_are_refused(expert):
    with pytest.raises(ValueError, match="restriction"):
        restriction.make_linear_policy(expert, "cart_velocity_zero")
    venv = env_utils.make_env("MountainCar-v0", 1, np.random.default_rng(0))
    other = PPO("MlpPolicy", venv, device="cpu", seed=0).policy
    venv.close()
    with pytest.raises(ValueError, match="observation"):
        restriction.make_linear_policy(other, "cart_position_zero")


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("restriction_id", ("identity",) + MASKS)
def test_checkpoint_round_trip_preserves_mask_frozen_layers_and_outputs(
    expert,
    tmp_path,
    restriction_id,
):
    th.manual_seed(5)
    policy = restriction.make_linear_policy(expert, restriction_id)
    with th.no_grad():
        policy.action_net.weight.add_(th.randn_like(policy.action_net.weight))
        policy.action_net.bias.add_(th.randn_like(policy.action_net.bias))
    path = tmp_path / "round-00010.pt"
    policy.save(path)

    loaded = restriction.load_policy_checkpoint(path)

    assert loaded.restriction_id == restriction_id
    _assert_same_weights(policy, loaded)
    assert _trainable(loaded) == ["action_net.bias", "action_net.weight"]
    obs = _obs(seed=4)
    _assert_same_outputs(_outputs(policy, obs), _outputs(loaded, obs))
    for index in HIDDEN.get(restriction_id, ()):
        _assert_same_outputs(
            _outputs(loaded, obs),
            _outputs(loaded, _with(obs, (index,), -2.0)),
        )


def test_plain_loader_refuses_restricted_checkpoints(expert, tmp_path):
    path = tmp_path / "restricted.pt"
    restriction.make_linear_policy(expert, "cart_position_zero").save(path)

    with pytest.raises(TypeError):
        policy_utils.load_policy_checkpoint(path)


# ---------------------------------------------------------------------------
# Identity fits reproduce the previous pipeline
# ---------------------------------------------------------------------------


def _config(tmp_path, algo, n_rounds, **overrides):
    params = dict(
        algo=algo,
        env_name="CartPole-v1",
        seed=4,
        policy_mode="linear",
        n_rounds=n_rounds,
        samples_per_round=1,
        l2_lambda=0.0,
        l2_decay=False,
        warm_start=False,
        beta_rampdown=0,
        bc_n_epochs=20,
        eval_interval=1,
        output_dir=tmp_path,
        expert_cache_dir=tmp_path / "unused",
        outer_early_stop=False,
    )
    params.update(overrides)
    return run_experiment.ExperimentConfig(**params)


def _run_ftl(tmp_path, expert, policy_factory):
    config = _config(tmp_path, "ftl", n_rounds=4)
    run_experiment._seed_everything(config.seed)
    rng = np.random.default_rng(config.seed)
    venv = env_utils.make_env("CartPole-v1", 1, rng)
    th.manual_seed(config.seed)
    kwargs = {} if policy_factory is None else {"policy_factory": policy_factory}
    records = run_experiment._run_dagger_variant(
        config,
        venv,
        expert,
        rng,
        BASELINES,
        **kwargs,
    )
    venv.close()
    return records


def _without_paths(records):
    return [{k: v for k, v in r.items() if k != "checkpoint"} for r in records]


def test_identity_ftl_run_matches_previous_pipeline(expert, tmp_path):
    previous = _run_ftl(tmp_path / "previous", expert, None)
    identity = _run_ftl(
        tmp_path / "identity",
        expert,
        lambda e: restriction.make_linear_policy(e, "identity"),
    )

    assert _without_paths(identity) == _without_paths(previous)
    for old, new in zip(previous, identity):
        if "checkpoint" in old:
            _assert_same_weights(
                policy_utils.load_policy_checkpoint(old["checkpoint"]),
                restriction.load_policy_checkpoint(new["checkpoint"]),
            )


def test_identity_fixed_bc_fit_matches_previous_fit(expert, tmp_path):
    config = _config(tmp_path, "bc", n_rounds=40)
    th.manual_seed(9)
    rng = np.random.default_rng(9)
    venv = env_utils.make_env("CartPole-v1", 1, rng)
    [previous] = run_experiment._run_bc(config, venv, expert, rng, BASELINES)
    data = run_experiment._shared_expert_data(config, venv, expert)

    th.manual_seed(9)
    policy = restriction.make_linear_policy(expert, "identity")
    trainer, inner_log = run_experiment._fit_fixed_bc(
        config,
        venv,
        policy,
        data,
        np.random.default_rng(9),
        device="cpu",
        tb_tag="identity",
    )
    venv.close()

    for key, value in inner_log.items():
        assert previous[key] == value, key
    _assert_same_weights(
        policy_utils.load_policy_checkpoint(previous["checkpoint"]),
        trainer.policy,
    )


# ---------------------------------------------------------------------------
# Online batch size: observed training minibatches
# ---------------------------------------------------------------------------


def _observe_fits(monkeypatch):
    """Record every gradient minibatch size, grouped by `_inner_train` call."""
    fits = []
    real_inner = run_experiment._inner_train

    def inner(trainer_or_bc, config, round_num, is_dagger):
        fits.append({"round": round_num, "batches": []})
        fits[-1]["log"] = real_inner(trainer_or_bc, config, round_num, is_dagger)
        return fits[-1]["log"]

    monkeypatch.setattr(run_experiment, "_inner_train", inner)
    for owner in (bc.BehaviorCloningLossCalculator, ftrl.TrainableParamsLossCalculator):
        real = owner.__call__

        def call(self, policy, obs, acts, _real=real, _owner=owner):
            # Validation NLL and round metrics run without gradients.
            if type(self) is _owner and th.is_grad_enabled():
                fits[-1]["batches"].append(len(acts))
            return _real(self, policy, obs, acts)

        monkeypatch.setattr(owner, "__call__", call)
    return fits


def _labelled_states(expert, n, seed=0):
    obs = _obs(n=n, seed=seed)
    acts, _ = expert.predict(obs, deterministic=True)
    return types.Transitions(
        obs=obs,
        acts=np.asarray(acts),
        next_obs=_obs(n=n, seed=seed + 1),
        dones=np.zeros(n, dtype=bool),
        infos=np.array([{} for _ in range(n)]),
    )


def _run_replay(tmp_path, expert, n_rounds, **overrides):
    config = _config(
        tmp_path,
        "bc_iid",
        n_rounds=n_rounds,
        eval_interval=10_000,
        # A small held-out threshold exercises the split path at tiny budgets.
        inner_early_stop_min_val_size=3,
        **overrides,
    )
    rng = np.random.default_rng(config.seed)
    venv = env_utils.make_env("CartPole-v1", 1, rng)
    th.manual_seed(config.seed)
    records = run_experiment._run_dagger_variant(
        config,
        venv,
        expert,
        rng,
        BASELINES,
        policy_factory=lambda e: restriction.make_linear_policy(e, STRONG),
        offline_data=_labelled_states(expert, n_rounds),
    )
    venv.close()
    return config, records


def _expected_split(t, config):
    """Training examples and batch size the original split rule implies."""
    n_val = int(config.inner_early_stop_val_frac * t)
    split = n_val >= config.inner_early_stop_min_val_size
    return (t - n_val if split else t), split


def test_online_batch_grows_with_accumulated_labels_up_to_the_cap(
    expert,
    tmp_path,
    monkeypatch,
):
    fits = _observe_fits(monkeypatch)
    n_rounds = 40
    config, records = _run_replay(
        tmp_path,
        expert,
        n_rounds,
        grow_batch_with_data=True,
    )

    assert [f["round"] for f in fits] == list(range(1, n_rounds + 1))
    seen = set()
    for fit, record in zip(fits, records[1:]):
        t = fit["round"]
        train, split = _expected_split(t, config)
        # min(32, t), shrunk only when the held-out split leaves fewer.
        batch = min(min(32, t), train)
        epochs = record["inner_es_stop_epoch"]
        assert set(fit["batches"]) == {batch}, t
        # drop_last: an incomplete final batch is skipped every epoch.
        assert len(fit["batches"]) == epochs * (train // batch), t
        assert record["train_batch_size"] == batch
        assert record["train_examples"] == train
        assert record["train_batches_per_epoch"] == train // batch
        assert record["train_drop_last"] is True
        assert (record["inner_es_fallback"] is None) == split
        seen.add((split, batch < min(32, t), batch == 32))
    # Rounds without a split, with a split that shrinks the batch, and with
    # the batch capped at 32 past 32 labels all occurred.
    assert {(False, False, False), (True, True, False), (True, False, True)} <= seen


def test_default_online_batch_keeps_the_original_minibatch_of_one(
    expert,
    tmp_path,
    monkeypatch,
):
    fits = _observe_fits(monkeypatch)
    config, records = _run_replay(tmp_path, expert, 33)

    assert config.grow_batch_with_data is False
    for fit, record in zip(fits, records[1:]):
        train, _ = _expected_split(fit["round"], config)
        assert set(fit["batches"]) == {1}
        assert len(fit["batches"]) == record["inner_es_stop_epoch"] * train
        assert record["train_batch_size"] == 1


def test_fixed_bc_fit_records_its_observed_batch_and_split(
    expert,
    tmp_path,
    monkeypatch,
):
    fits = _observe_fits(monkeypatch)
    config = _config(tmp_path, "bc", n_rounds=40, inner_early_stop_min_val_size=3)
    venv = env_utils.make_env("CartPole-v1", 1, np.random.default_rng(0))
    _, inner_log = run_experiment._fit_fixed_bc(
        config,
        venv,
        restriction.make_linear_policy(expert, STRONG),
        _labelled_states(expert, 40),
        np.random.default_rng(0),
        device="cpu",
        tb_tag="observed",
    )
    venv.close()

    [fit] = fits
    # 40 labels: 4 held out, 36 trained in minibatches of 32 (4 dropped).
    assert set(fit["batches"]) == {32}
    assert len(fit["batches"]) == inner_log["inner_es_stop_epoch"]
    assert inner_log["train_batch_size"] == 32
    assert inner_log["train_examples"] == 36
    assert inner_log["train_batches_per_epoch"] == 1


# ---------------------------------------------------------------------------
# Mask audit on the real CartPole simulator
# ---------------------------------------------------------------------------


class RuleExpert:
    """Deterministic expert given by a rule on the full observation."""

    def __init__(self, rule):
        self.rule = rule
        self.calls = 0

    def predict(self, obs, deterministic=True):
        assert deterministic
        self.calls += 1
        obs = np.asarray(obs, dtype=np.float32).reshape(-1, 4)
        return np.array([self.rule(o) for o in obs], dtype=np.int64), None


def _balancing_rule(o):
    # A position-aware linear controller that keeps CartPole up for 500 steps.
    return int(0.5 * o[0] + o[1] + 10.0 * o[2] + 10.0 * o[3] > 0)


def test_audit_finds_conflicts_when_the_expert_uses_cart_position():
    expert = RuleExpert(_balancing_rule)

    audit = restriction.audit_cartpole_position(
        expert,
        reset_seed=301,
        restriction_id="cart_position_zero",
    )

    assert audit["status"] == "conflict_found"
    assert audit["restriction_id"] == "cart_position_zero"
    assert audit["reset_seed"] == 301
    assert audit["grid"] == list(restriction.AUDIT_X_GRID)
    bases = audit["base_states"]
    assert [b["step"] for b in bases] == list(restriction.AUDIT_BASE_STEPS)
    # Base states are the independent diagnostic trajectory's own states.
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=301)
    trajectory = [obs]
    for _ in range(max(restriction.AUDIT_BASE_STEPS)):
        obs, _, terminated, truncated, _ = env.step(_balancing_rule(obs))
        assert not (terminated or truncated)
        trajectory.append(obs)
    for base in bases:
        np.testing.assert_allclose(
            base["full_state"],
            trajectory[base["step"]],
            rtol=0,
            atol=1e-6,
        )

    per_base = len(list(itertools.combinations(restriction.AUDIT_X_GRID, 2)))
    assert len(audit["pairs"]) == len(bases) * per_base
    states = {(s["base"], s["x"]): s for s in audit["states"]}
    x_threshold = env.unwrapped.x_threshold
    theta_threshold = env.unwrapped.theta_threshold_radians
    conflicts = 0
    for state in audit["states"]:
        full = np.array(state["observation"], dtype=np.float32)
        assert abs(full[0]) < x_threshold and abs(full[2]) < theta_threshold
        assert state["simulator_valid"] is True
        assert state["expert_action"] == _balancing_rule(full)
        np.testing.assert_array_equal(state["masked_observation"][0], 0.0)
        np.testing.assert_array_equal(state["masked_observation"][1:], full[1:])
    for pair in audit["pairs"]:
        a = states[(pair["base"], pair["x_a"])]
        b = states[(pair["base"], pair["x_b"])]
        assert a["masked_observation"] == b["masked_observation"]
        assert a["observation"][1:] == b["observation"][1:]
        assert a["observation"][0] != b["observation"][0]
        assert pair["simulator_consistent"] is True
        assert pair["conflict"] == (a["expert_action"] != b["expert_action"])
        conflicts += pair["conflict"]
    assert audit["n_conflicting_pairs"] == conflicts > 0
    assert "on-policy" in audit["claim"]


def test_stronger_mask_audit_reuses_the_same_x_witnesses():
    # The fixed x-grid pairs share x_dot, theta and theta_dot, so they also
    # collide under the coarser mask: same states, physics checks and labels.
    x_only = restriction.audit_cartpole_position(
        RuleExpert(_balancing_rule),
        reset_seed=301,
        restriction_id="cart_position_zero",
    )
    expert = RuleExpert(_balancing_rule)
    strong = restriction.audit_cartpole_position(
        expert,
        reset_seed=301,
        restriction_id=STRONG,
    )

    assert strong["restriction_id"] == STRONG
    assert strong["status"] == "conflict_found"
    assert strong["expert_predict_calls"] == expert.calls
    assert strong["base_states"] == x_only["base_states"]
    assert strong["n_conflicting_pairs"] == x_only["n_conflicting_pairs"] > 0
    drop = ("masked_observation",)
    for a, b in zip(strong["states"], x_only["states"]):
        assert {k: v for k, v in a.items() if k not in drop} == {
            k: v for k, v in b.items() if k not in drop
        }
        full = a["observation"]
        assert a["masked_observation"] == [0.0, full[1], full[2], 0.0]
    assert strong["pairs"] == x_only["pairs"]
    # Only cart position varies within a pair: nothing about theta_dot.
    for pair in strong["pairs"]:
        assert pair["x_a"] != pair["x_b"]
    assert "angular velocity" in strong["witness_scope"]


def test_audit_refuses_restrictions_that_keep_cart_position():
    with pytest.raises(ValueError, match="cart position"):
        restriction.audit_cartpole_position(
            RuleExpert(_balancing_rule),
            reset_seed=301,
            restriction_id="identity",
        )


def test_audit_without_conflicts_is_uncertified_and_keeps_every_pair():
    # Ignores cart position, and falls over quickly, so later bases are
    # unavailable and recorded as such.
    expert = RuleExpert(lambda o: 0)

    audit = restriction.audit_cartpole_position(
        expert,
        reset_seed=301,
        restriction_id=STRONG,
    )

    assert audit["status"] == "restriction_uncertified"
    assert audit["n_conflicting_pairs"] == 0
    used = [b["step"] for b in audit["base_states"]]
    assert used and set(used) | set(audit["unavailable_base_steps"]) == set(
        restriction.AUDIT_BASE_STEPS
    )
    assert audit["unavailable_base_steps"]
    per_base = len(list(itertools.combinations(restriction.AUDIT_X_GRID, 2)))
    assert len(audit["pairs"]) == len(used) * per_base
