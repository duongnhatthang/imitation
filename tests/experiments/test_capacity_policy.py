"""One-hidden-layer capacity students: checkpoints, freezing and cold refits.

Complements the root behavioral contract (`test_capacity_policy_contract.py`)
with the trusted checkpoint family, metadata and frozen-state tampering,
normalization buffers under ``policy.train()``, the BC optimizer and FTRL
cold-refit hooks, and the unchanged legacy paths. Every policy is a real SB3
``ActorCriticPolicy``; nothing is mocked.
"""

import copy
import hashlib

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from imitation.algorithms import bc, ftrl
from imitation.data import rollout, types
from imitation.experiments.agnostic import capacity_policy as cp
from imitation.experiments.ftrl import env_utils, policy_utils

OBS = th.tensor(
    [[0.1, 0.2, 0.3, 0.4], [-0.4, 0.2, -0.1, 0.3], [0.5, -0.5, 0.0, 0.2]],
)


class NormedEncoder(BaseFeaturesExtractor):
    """Small image encoder with batch-norm buffers that must stay frozen."""

    def __init__(self, observation_space, features_dim=8):
        super().__init__(observation_space, features_dim)
        channels = observation_space.shape[0]
        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(channels, 3, kernel_size=4, stride=4),
            th.nn.BatchNorm2d(3),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, *observation_space.shape)).shape[1]
        self.linear = th.nn.Sequential(th.nn.Linear(n_flat, features_dim), th.nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))


def _vector_teacher(hidden=(64,), n_actions=2, activation=th.nn.Tanh):
    return ActorCriticPolicy(
        spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
        spaces.Discrete(n_actions),
        lr_schedule=lambda _: 1e-3,
        net_arch={"pi": list(hidden), "vf": list(hidden)},
        activation_fn=activation,
    )


def _image_teacher():
    teacher = ActorCriticPolicy(
        spaces.Box(0, 255, shape=(1, 16, 16), dtype=np.uint8),
        spaces.Discrete(3),
        lr_schedule=lambda _: 1e-3,
        net_arch={"pi": [64], "vf": [64]},
        features_extractor_class=NormedEncoder,
        features_extractor_kwargs={"features_dim": 8},
    )
    # Move the running statistics away from their defaults, then freeze them.
    teacher.train()
    with th.no_grad():
        teacher.extract_features(_images(16, seed=1), teacher.features_extractor)
    teacher.eval()
    return teacher


def _images(n, seed=0):
    generator = th.Generator().manual_seed(seed)
    return th.randint(0, 256, (n, 1, 16, 16), dtype=th.uint8, generator=generator)


def _actor(policy):
    return {
        name: value.clone()
        for name, value in policy.state_dict().items()
        if name.startswith(("mlp_extractor.policy_net.", "action_net."))
    }


def _frozen(policy):
    return {
        name: value.clone()
        for name, value in policy.state_dict().items()
        if not name.startswith(("mlp_extractor.policy_net.", "action_net."))
    }


def _equal(a, b):
    return a.keys() == b.keys() and all(th.equal(a[k], b[k]) for k in a)


def _digests(path, policy):
    """Writer digests for `cp.load_checkpoint` of a file ``policy`` saved."""
    return dict(
        expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_state_sha256=cp.state_sha256(policy),
    )


def _step(policy, obs=OBS, acts=None):
    acts = th.tensor([0, 1, 1][: len(obs)]) if acts is None else acts
    optimizer = th.optim.Adam(policy.bc_optimizer_parameters(), lr=0.1)
    loss = -policy.get_distribution(obs).log_prob(acts).mean()
    loss.backward()
    optimizer.step()


# ---------------------------------------------------------------------------
# Architecture, spec and counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", cp.STUDENT_WIDTHS)
def test_spec_binds_teacher_encoder_and_actual_counts(width):
    teacher = _vector_teacher(n_actions=3)
    student = cp.make_capacity_policy(
        teacher, hidden_width=width, initialization_seed=7
    )
    spec = student.capacity_spec
    d, a = 4, 3
    head = d * width + width * a + width + a
    teacher_head = d * 64 + 64 * a + 64 + a
    assert spec["schema"] == cp.POLICY_SCHEMA
    assert spec["learner_family"] == cp.LEARNER_FAMILY == "capacity-one-hidden"
    assert spec["hidden_width"] == width and spec["teacher_hidden_width"] == 64
    assert spec["teacher"]["actor_hidden_widths"] == [64]
    assert spec["teacher"]["activation_fn"] == "torch.nn.modules.activation.Tanh"
    assert spec["teacher"]["policy_state_sha256"] == cp.state_sha256(teacher)
    assert spec["encoder"]["class"].endswith("FlattenExtractor")
    assert spec["encoder"]["features_dim"] == d
    assert spec["encoder"]["state_sha256"] == cp.encoder_sha256(teacher)
    counts = spec["parameters"]
    assert (
        counts["trainable"]
        == head
        == sum(p.numel() for p in student.parameters() if p.requires_grad)
    )
    assert counts["decision_head_formula"] == "d*w + w*a + w + a"
    assert counts["encoder"] == 0 and counts["total_excluding_critic"] == head
    assert counts["teacher_decision_head"] == teacher_head
    assert spec["width_ratio"] == width / 64
    assert spec["parameter_ratio"] == head / teacher_head != spec["width_ratio"]
    assert student.capacity_spec_sha256 == cp.spec_sha256(spec)
    assert isinstance(student.activation_fn(), th.nn.Tanh)
    # Exactly one hidden layer: Linear then the teacher's activation.
    kinds = [type(m) for m in student.mlp_extractor.policy_net]
    assert kinds == [th.nn.Linear, th.nn.Tanh]


def test_teacher_activation_is_preserved_and_mismatched_teachers_refuse():
    student = cp.make_capacity_policy(
        _vector_teacher(activation=th.nn.ReLU),
        hidden_width=8,
        initialization_seed=1,
    )
    assert [type(m) for m in student.mlp_extractor.policy_net] == [
        th.nn.Linear,
        th.nn.ReLU,
    ]
    with pytest.raises(ValueError, match="Discrete"):
        cp.make_capacity_policy(
            ActorCriticPolicy(
                spaces.Box(-1, 1, shape=(4,)),
                spaces.Box(-1, 1, shape=(2,)),
                lr_schedule=lambda _: 1e-3,
                net_arch=[64],
            ),
            hidden_width=8,
            initialization_seed=1,
        )
    restricted_like = cp.make_capacity_policy(
        _vector_teacher(), hidden_width=32, initialization_seed=1
    )
    with pytest.raises(ValueError, match="teacher"):
        cp.make_capacity_policy(restricted_like, hidden_width=4, initialization_seed=1)
    with pytest.raises(ValueError, match="teacher_hidden_width"):
        cp.make_capacity_policy(
            _vector_teacher(hidden=(128,)),
            hidden_width=4,
            initialization_seed=1,
            teacher_hidden_width=128,
        )


@pytest.mark.parametrize("seed", [True, -1, 1.5, "7", None])
def test_initialization_seed_must_be_a_plain_non_negative_integer(seed):
    with pytest.raises(ValueError, match="initialization_seed"):
        cp.make_capacity_policy(
            _vector_teacher(), hidden_width=4, initialization_seed=seed
        )


def test_student_never_aliases_teacher_tensors():
    teacher = _image_teacher()
    student = cp.make_capacity_policy(teacher, hidden_width=4, initialization_seed=3)
    teacher_ptrs = {t.data_ptr() for t in teacher.state_dict().values() if t.numel()}
    student_ptrs = {t.data_ptr() for t in student.state_dict().values() if t.numel()}
    assert not teacher_ptrs & student_ptrs


# ---------------------------------------------------------------------------
# Frozen encoder buffers and gradients
# ---------------------------------------------------------------------------


def test_encoder_buffers_stay_frozen_under_train_mode_and_bc_fit():
    teacher = _image_teacher()
    teacher_state = copy.deepcopy(teacher.state_dict())
    student = cp.make_capacity_policy(teacher, hidden_width=4, initialization_seed=3)
    frozen = _frozen(student)
    running = student.features_extractor.cnn[1].running_mean.clone()
    assert not th.equal(running, th.zeros(3))  # the teacher's statistics
    assert th.equal(running, teacher.features_extractor.cnn[1].running_mean)
    student.train()
    student.set_training_mode(True)
    assert student.training and not student.features_extractor.training
    assert all(not m.training for m in student.features_extractor.modules())
    obs = _images(6, seed=2).numpy()
    acts = np.array([0, 1, 2, 0, 1, 2])
    trainer = bc.BC(
        observation_space=student.observation_space,
        action_space=student.action_space,
        policy=student,
        demonstrations=types.Transitions(
            obs=obs,
            acts=acts,
            next_obs=obs,
            dones=np.zeros(6, dtype=bool),
            infos=np.array([{}] * 6),
        ),
        rng=np.random.default_rng(0),
        batch_size=3,
        device="cpu",
    )
    actor = _actor(student)
    trainer.train(n_epochs=3, progress_bar=False)
    assert not _equal(actor, _actor(student))
    assert _equal(frozen, _frozen(student))
    student.verify_frozen_state()
    assert _equal(teacher_state, teacher.state_dict())


def test_actor_gradients_reach_both_layers_and_nothing_else():
    student = cp.make_capacity_policy(
        _image_teacher(), hidden_width=8, initialization_seed=5
    )
    loss = -student.get_distribution(_images(3)).log_prob(th.tensor([0, 1, 2])).mean()
    loss.backward()
    with_grad = {n for n, p in student.named_parameters() if p.grad is not None}
    assert with_grad == {
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.policy_net.0.bias",
        "action_net.weight",
        "action_net.bias",
    }


# ---------------------------------------------------------------------------
# Optimizer membership: BC hook, legacy unchanged
# ---------------------------------------------------------------------------


def test_bc_optimizer_hook_owns_only_the_actor_and_legacy_owns_everything():
    student = cp.make_capacity_policy(
        _vector_teacher(), hidden_width=16, initialization_seed=2
    )
    actor = {
        id(p)
        for m in (student.mlp_extractor.policy_net, student.action_net)
        for p in m.parameters()
    }
    trainer = bc.BC(
        observation_space=student.observation_space,
        action_space=student.action_space,
        policy=student,
        rng=np.random.default_rng(0),
        device="cpu",
    )
    owned = [p for g in trainer.optimizer.param_groups for p in g["params"]]
    assert {id(p) for p in owned} == actor and len(owned) == len(actor)

    legacy = policy_utils.create_linear_policy(_vector_teacher(hidden=(64, 64)))
    assert not hasattr(legacy, "bc_optimizer_parameters")
    trainer = bc.BC(
        observation_space=legacy.observation_space,
        action_space=legacy.action_space,
        policy=legacy,
        rng=np.random.default_rng(0),
        device="cpu",
    )
    owned = [p for g in trainer.optimizer.param_groups for p in g["params"]]
    assert [id(p) for p in owned] == [id(p) for p in legacy.parameters()]


# ---------------------------------------------------------------------------
# Cold refits and RNG isolation
# ---------------------------------------------------------------------------


def test_cold_refit_draws_fresh_actor_from_named_stream_only():
    teacher = _vector_teacher()
    a = cp.make_capacity_policy(teacher, hidden_width=8, initialization_seed=11)
    b = cp.make_capacity_policy(teacher, hidden_width=8, initialization_seed=11)
    other = cp.make_capacity_policy(teacher, hidden_width=8, initialization_seed=12)
    draw0, frozen = _actor(a), _frozen(a)
    assert _equal(draw0, _actor(b)) and not _equal(draw0, _actor(other))
    # Different training histories, then one cold refit each.
    _step(a)
    _step(b, acts=th.tensor([1, 0, 0]))
    _step(b, acts=th.tensor([1, 0, 0]))
    th.manual_seed(5)
    np.random.seed(5)
    torch_state, numpy_state = th.get_rng_state(), np.random.get_state()
    a.cold_refit_reinitialize()
    b.cold_refit_reinitialize()
    assert th.equal(torch_state, th.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    draw1 = _actor(a)
    assert _equal(draw1, _actor(b)) and not _equal(draw1, draw0)
    assert a.initialization_draws == b.initialization_draws == 1
    assert _equal(frozen, _frozen(a))
    a.verify_frozen_state()
    # The same draw is reproducible from a fresh policy at the same index.
    fresh = cp.make_capacity_policy(teacher, hidden_width=8, initialization_seed=11)
    fresh.cold_refit_reinitialize()
    assert _equal(draw1, _actor(fresh))


def _ftrl_trainer(policy, tmp_path, rng_seed=0):
    venv = env_utils.make_env("CartPole-v1", 1, np.random.default_rng(0))
    teacher = _cartpole_policy()
    trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=policy,
        rng=np.random.default_rng(rng_seed),
        device="cpu",
        batch_size=2,
    )
    return venv, ftrl.FTRLTrainer(
        venv=venv,
        scratch_dir=tmp_path,
        expert_policy=teacher,
        bc_trainer=trainer,
        rng=np.random.default_rng(rng_seed),
    )


def _cartpole_policy(hidden=(64,)):
    venv = env_utils.make_env("CartPole-v1", 1, np.random.default_rng(0))
    try:
        return ActorCriticPolicy(
            venv.observation_space,
            venv.action_space,
            lr_schedule=lambda _: 1e-3,
            net_arch={"pi": list(hidden), "vf": list(hidden)},
        )
    finally:
        venv.close()


def test_ftrl_cold_reset_uses_the_policy_hook_and_clears_optimizer(tmp_path):
    student = cp.make_capacity_policy(
        _cartpole_policy(), hidden_width=4, initialization_seed=4100
    )
    frozen = _frozen(student)
    venv, trainer = _ftrl_trainer(student, tmp_path)
    try:
        optimizer = trainer.bc_trainer.optimizer
        loss = -student.get_distribution(OBS).log_prob(th.tensor([0, 1, 1])).mean()
        loss.backward()
        optimizer.step()
        assert optimizer.state
        expected = cp.make_capacity_policy(
            _cartpole_policy(), hidden_width=4, initialization_seed=4100
        )
        expected.cold_refit_reinitialize()
        th.manual_seed(9)
        before = th.get_rng_state()
        trainer._reinitialize_trainable_params()
        assert th.equal(before, th.get_rng_state())
        assert not optimizer.state
        assert all(p.grad is None for p in student.parameters())
        assert _equal(_actor(expected), _actor(student))
        assert _equal(frozen, _frozen(student))
    finally:
        venv.close()


def test_ftrl_legacy_reset_still_draws_xavier_from_global_rng(tmp_path):
    legacy = policy_utils.create_linear_policy(_cartpole_policy(hidden=(64, 64)))
    venv, trainer = _ftrl_trainer(legacy, tmp_path)
    try:
        th.manual_seed(9)
        trainer._reinitialize_trainable_params()
        first = legacy.action_net.weight.detach().clone()
        th.manual_seed(9)
        reference = th.empty_like(first)
        th.nn.init.xavier_uniform_(reference)
        assert th.equal(first, reference)
        assert th.equal(legacy.action_net.bias, th.zeros_like(legacy.action_net.bias))
    finally:
        venv.close()


def test_width_does_not_change_minibatch_order_or_global_streams():
    """BC on the same data draws identical batch orders for every width."""
    teacher = _cartpole_policy()
    venv = env_utils.make_env("CartPole-v1", 1, np.random.default_rng(0))
    try:
        trajs = rollout.generate_trajectories(
            teacher,
            venv,
            rollout.make_sample_until(min_timesteps=40),
            rng=np.random.default_rng(0),
            deterministic_policy=True,
        )
    finally:
        venv.close()
    data = rollout.flatten_trajectories(trajs)
    states = {}
    for width in cp.STUDENT_WIDTHS:
        th.manual_seed(4100)
        student = cp.make_capacity_policy(
            teacher, hidden_width=width, initialization_seed=4100
        )
        trainer = bc.BC(
            observation_space=student.observation_space,
            action_space=student.action_space,
            policy=student,
            demonstrations=data,
            rng=np.random.default_rng(4100),
            batch_size=8,
            device="cpu",
        )
        trainer.train(n_epochs=2, progress_bar=False)
        states[width] = th.get_rng_state()
    reference = states[cp.STUDENT_WIDTHS[0]]
    assert all(th.equal(reference, state) for state in states.values())


# ---------------------------------------------------------------------------
# Trusted checkpoint family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", cp.STUDENT_WIDTHS)
def test_image_checkpoint_roundtrip_keeps_outputs_buffers_and_optimizer(
    tmp_path,
    width,
):
    student = cp.make_capacity_policy(
        _image_teacher(), hidden_width=width, initialization_seed=4100
    )
    _step(student, obs=_images(3))
    path = tmp_path / "student.pt"
    student.save(path)
    for restored in (
        policy_utils.load_policy_checkpoint(path),
        cp.load_checkpoint(
            path,
            expected_spec_sha256=student.capacity_spec_sha256,
            **_digests(path, student),
        ),
    ):
        assert isinstance(restored, cp.CapacityActorCriticPolicy)
        assert restored.capacity_spec == student.capacity_spec
        images = _images(5, seed=4)
        restored.train()
        student.eval()
        assert th.equal(
            student.get_distribution(images).distribution.logits,
            restored.get_distribution(images).distribution.logits,
        )
        assert np.array_equal(
            student.predict(images.numpy(), deterministic=True)[0],
            restored.predict(images.numpy(), deterministic=True)[0],
        )
        assert _equal(student.state_dict(), restored.state_dict())
        assert not restored.features_extractor.training
        assert [p.requires_grad for p in student.parameters()] == [
            p.requires_grad for p in restored.parameters()
        ]
        trainer = bc.BC(
            observation_space=restored.observation_space,
            action_space=restored.action_space,
            policy=restored,
            rng=np.random.default_rng(0),
            device="cpu",
        )
        owned = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
        assert owned == {id(p) for p in restored.parameters() if p.requires_grad}


def _saved(tmp_path, width=4, image=False):
    teacher = _image_teacher() if image else _vector_teacher()
    student = cp.make_capacity_policy(
        teacher, hidden_width=width, initialization_seed=4100
    )
    path = tmp_path / "student.pt"
    student.save(path)
    return student, th.load(path, weights_only=False)


def _resave(tmp_path, saved):
    path = tmp_path / "mutated.pt"
    th.save(saved, path)
    return path


def _rehash(saved):
    saved["data"]["capacity_spec_sha256"] = cp.spec_sha256(
        saved["data"]["capacity_spec"],
    )


def _mutate_width(saved):
    saved["data"]["capacity_spec"]["hidden_width"] = 8


def _mutate_width_consistently(saved):
    spec = saved["data"]["capacity_spec"]
    spec["hidden_width"] = 8
    saved["data"]["net_arch"]["pi"] = [8]
    _rehash(saved)


def _mutate_to_width64(saved):
    spec = saved["data"]["capacity_spec"]
    spec["hidden_width"] = 64
    saved["data"]["net_arch"]["pi"] = [64]
    _rehash(saved)


def _mutate_schema(saved):
    saved["data"]["capacity_spec"]["schema"] = "agnostic-capacity-two-hidden/1"
    _rehash(saved)


def _mutate_family(saved):
    saved["data"]["capacity_spec"]["learner_family"] = "linear-head"
    _rehash(saved)


def _mutate_activation(saved):
    saved["data"]["activation_fn"] = th.nn.ReLU


def _mutate_hash(saved):
    saved["data"]["capacity_spec_sha256"] = "0" * 64


def _mutate_count(saved):
    saved["data"]["capacity_spec"]["parameters"]["trainable"] += 1
    _rehash(saved)


def _mutate_extra_layer(saved):
    saved["data"]["net_arch"]["pi"] = [4, 4]


def _mutate_encoder_claim(saved):
    saved["data"]["capacity_spec"]["encoder"]["state_sha256"] = "f" * 64
    _rehash(saved)


@pytest.mark.parametrize(
    "mutate",
    [
        _mutate_width,
        _mutate_width_consistently,
        _mutate_to_width64,
        _mutate_schema,
        _mutate_family,
        _mutate_activation,
        _mutate_hash,
        _mutate_count,
        _mutate_extra_layer,
        _mutate_encoder_claim,
    ],
)
def test_mutated_checkpoint_metadata_refuses(tmp_path, mutate):
    _, saved = _saved(tmp_path)
    mutate(saved)
    path = _resave(tmp_path, saved)
    with pytest.raises(ValueError):
        policy_utils.load_policy_checkpoint(path)


@pytest.mark.parametrize(
    "key",
    [
        "features_extractor.cnn.0.weight",
        "features_extractor.cnn.1.running_mean",
        "features_extractor.cnn.1.running_var",
        "features_extractor.cnn.1.num_batches_tracked",
        "pi_features_extractor.linear.0.bias",
        "value_net.weight",
        "mlp_extractor.value_net.0.bias",
    ],
)
def test_mutated_frozen_state_refuses(tmp_path, key):
    _, saved = _saved(tmp_path, image=True)
    saved["state_dict"][key] = saved["state_dict"][key] + 1
    path = _resave(tmp_path, saved)
    with pytest.raises(ValueError, match="frozen"):
        policy_utils.load_policy_checkpoint(path)


def test_external_binding_refuses_other_specs_and_stripped_family(tmp_path):
    student, saved = _saved(tmp_path, width=32)
    path = tmp_path / "student.pt"
    with pytest.raises(ValueError, match="spec"):
        cp.load_checkpoint(
            path,
            expected_spec_sha256="0" * 64,
            **_digests(path, student),
        )
    stripped = copy.deepcopy(saved)
    for key in ("capacity_spec", "capacity_spec_sha256", "share_features_extractor"):
        stripped["data"].pop(key)
    stripped_path = _resave(tmp_path, stripped)
    # The generic loader sees a plain policy; the capacity loader refuses it.
    assert type(policy_utils.load_policy_checkpoint(stripped_path)) is (
        ActorCriticPolicy
    )
    with pytest.raises(ValueError, match="capacity"):
        cp.load_checkpoint(
            stripped_path,
            expected_spec_sha256=student.capacity_spec_sha256,
            **_digests(stripped_path, student),
        )
    assert cp.load_checkpoint(
        path,
        expected_spec_sha256=student.capacity_spec_sha256,
        **_digests(path, student),
    ).capacity_spec_sha256 == (student.capacity_spec_sha256)


def test_restricted_checkpoints_still_refuse_the_plain_loader(tmp_path):
    from imitation.experiments.agnostic import restriction

    learner = restriction.make_linear_policy(
        _cartpole_policy(hidden=(64, 64)),
        "cart_position_angular_velocity_zero",
    )
    path = tmp_path / "restricted.pt"
    learner.save(path)
    with pytest.raises(TypeError):
        policy_utils.load_policy_checkpoint(path)
    assert restriction.load_policy_checkpoint(path).restriction_id == (
        "cart_position_angular_velocity_zero"
    )


def test_capacity_checkpoints_refuse_the_restricted_loader(tmp_path):
    from imitation.experiments.agnostic import restriction

    _saved(tmp_path)
    with pytest.raises(ValueError, match="restricted"):
        restriction.load_policy_checkpoint(tmp_path / "student.pt")


def test_state_digest_matches_the_preparation_identity_digest():
    from imitation.experiments.agnostic import pilot

    teacher = _image_teacher()
    assert cp.state_sha256(teacher) == pilot.policy_state_sha256(teacher)
