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

from imitation.experiments.agnostic import restriction
from imitation.experiments.ftrl import env_utils, policy_utils, run_experiment

BASELINES = {"expert_return": 500.0, "random_return": 22.0}


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


def test_head_initialization_is_identical_across_conditions(expert):
    th.manual_seed(3)
    full = restriction.make_linear_policy(expert, "identity")
    rng_full = th.get_rng_state()
    th.manual_seed(3)
    restricted = restriction.make_linear_policy(expert, "cart_position_zero")
    rng_restricted = th.get_rng_state()

    assert th.equal(rng_full, rng_restricted)
    _assert_same_weights(full, restricted)


# ---------------------------------------------------------------------------
# Restriction placement
# ---------------------------------------------------------------------------


def test_restriction_zeroes_cart_position_before_frozen_features(expert):
    th.manual_seed(11)
    full = restriction.make_linear_policy(expert, "identity")
    th.manual_seed(11)
    restricted = restriction.make_linear_policy(expert, "cart_position_zero")
    obs = _obs()

    # Every path of the restricted learner sees exactly (0, x_dot, theta,
    # theta_dot): it equals the unrestricted twin evaluated at x = 0 ...
    _assert_same_outputs(
        _outputs(restricted, obs),
        _outputs(full, _with_x(obs, 0.0)),
    )
    # ... so it cannot distinguish states that differ only in x.
    for x in (-1.8, 0.6, 2.3):
        _assert_same_outputs(
            _outputs(restricted, obs),
            _outputs(restricted, _with_x(obs, x)),
        )


def test_all_extractor_aliases_are_restricted_and_only_the_head_trains(expert):
    policy = restriction.make_linear_policy(expert, "cart_position_zero")

    assert policy.pi_features_extractor is policy.features_extractor
    assert policy.vf_features_extractor is policy.features_extractor
    assert policy.restriction_id == "cart_position_zero"
    assert _trainable(policy) == ["action_net.bias", "action_net.weight"]
    for name, param in expert.named_parameters():
        frozen = dict(policy.named_parameters())[name]
        if not name.startswith("action_net"):
            assert th.equal(frozen, param), name


def test_masking_the_learner_never_alters_expert_inputs(expert):
    before = copy.deepcopy(expert)
    obs = _obs()
    shifted = _with_x(obs, 1.9)

    restriction.make_linear_policy(expert, "cart_position_zero")

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


@pytest.mark.parametrize("restriction_id", ["identity", "cart_position_zero"])
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
    if restriction_id == "cart_position_zero":
        _assert_same_outputs(
            _outputs(loaded, obs),
            _outputs(loaded, _with_x(obs, -2.0)),
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

    audit = restriction.audit_cartpole_position(expert, reset_seed=301)

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


def test_audit_without_conflicts_is_uncertified_and_keeps_every_pair():
    # Ignores cart position, and falls over quickly, so later bases are
    # unavailable and recorded as such.
    expert = RuleExpert(lambda o: 0)

    audit = restriction.audit_cartpole_position(expert, reset_seed=301)

    assert audit["status"] == "restriction_uncertified"
    assert audit["n_conflicting_pairs"] == 0
    used = [b["step"] for b in audit["base_states"]]
    assert used and set(used) | set(audit["unavailable_base_steps"]) == set(
        restriction.AUDIT_BASE_STEPS
    )
    assert audit["unavailable_base_steps"]
    per_base = len(list(itertools.combinations(restriction.AUDIT_X_GRID, 2)))
    assert len(audit["pairs"]) == len(used) * per_base
