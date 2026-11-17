"""Behavioral contract for the approved shared-encoder capacity study."""

import copy
import importlib
import importlib.util

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import NatureCNN

from imitation.algorithms import bc
from imitation.experiments.ftrl.policy_utils import load_policy_checkpoint


def _factory():
    name = "imitation.experiments.agnostic.capacity_policy"
    assert importlib.util.find_spec(name) is not None, (
        "Missing approved one-hidden-layer capacity policy implementation"
    )
    return importlib.import_module(name).make_capacity_policy


def _teacher(hidden=(64,), image=False):
    observation = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
    kwargs = {}
    if image:
        observation = spaces.Box(0, 255, shape=(4, 84, 84), dtype=np.uint8)
        kwargs = dict(
            features_extractor_class=NatureCNN,
            features_extractor_kwargs={"features_dim": 32},
        )
    return ActorCriticPolicy(
        observation,
        spaces.Discrete(2),
        lr_schedule=lambda _: 1e-3,
        net_arch={"pi": list(hidden), "vf": list(hidden)},
        **kwargs,
    )


@pytest.mark.parametrize(
    "width,parameter_count", [(32, 226), (16, 114), (8, 58), (4, 30)]
)
def test_one_hidden_actor_capacity_and_actual_bc_optimizer(width, parameter_count):
    """Catch frozen hidden layers, extra layers, and optimizer-owned critics."""
    student = _factory()(_teacher(), hidden_width=width, initialization_seed=4100)
    layers = [m for m in student.mlp_extractor.policy_net if isinstance(m, th.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in layers] == [(4, width)]
    trainable = {id(p) for p in student.parameters() if p.requires_grad}
    assert sum(p.numel() for p in student.parameters() if p.requires_grad) == parameter_count
    assert all(not p.requires_grad for p in student.mlp_extractor.value_net.parameters())
    assert all(not p.requires_grad for p in student.value_net.parameters())
    trainer = bc.BC(
        observation_space=student.observation_space,
        action_space=student.action_space,
        policy=student,
        rng=np.random.default_rng(4100),
        batch_size=2,
        device="cpu",
    )
    owned = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert owned == trainable


def test_student_update_does_not_train_teacher_or_copy_its_actor():
    expert = _teacher()
    before = copy.deepcopy(expert.state_dict())
    student = _factory()(expert, hidden_width=32, initialization_seed=4100)
    assert not th.equal(
        student.mlp_extractor.policy_net[0].weight,
        expert.mlp_extractor.policy_net[0].weight[:32],
    )
    inputs = th.tensor([[0.1, 0.2, 0.3, 0.4], [-0.4, 0.2, -0.1, 0.3]])
    loss = -student.get_distribution(inputs).log_prob(th.tensor([0, 1])).mean()
    loss.backward()
    for name, parameter in student.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
    optimizer = th.optim.Adam([p for p in student.parameters() if p.requires_grad])
    optimizer.step()
    assert all(th.equal(value, expert.state_dict()[name]) for name, value in before.items())
    assert all(p.grad is None for p in expert.parameters())


def test_shared_visual_encoder_is_identical_and_immutable_under_training():
    expert = _teacher(image=True)
    student = _factory()(expert, hidden_width=4, initialization_seed=4100)
    before = copy.deepcopy(expert.features_extractor.state_dict())
    assert all(
        th.equal(value, student.features_extractor.state_dict()[name])
        for name, value in before.items()
    )
    student.train()
    assert student.features_extractor.training is False
    observations = th.randint(0, 256, (2, 4, 84, 84), dtype=th.uint8)
    loss = -student.get_distribution(observations).log_prob(th.tensor([0, 1])).mean()
    loss.backward()
    assert all(p.grad is None for p in student.features_extractor.parameters())
    assert all(
        th.equal(value, student.features_extractor.state_dict()[name])
        for name, value in before.items()
    )


def test_width_initialization_cannot_advance_other_rng_streams():
    expert = _teacher()
    th.manual_seed(123)
    before = th.get_rng_state().clone()
    small = _factory()(expert, hidden_width=4, initialization_seed=4100)
    assert th.equal(before, th.get_rng_state())
    _factory()(expert, hidden_width=32, initialization_seed=4100)
    assert th.equal(before, th.get_rng_state())
    repeated = _factory()(expert, hidden_width=4, initialization_seed=4100)
    assert all(
        th.equal(value, repeated.state_dict()[name])
        for name, value in small.state_dict().items()
    )


@pytest.mark.parametrize("hidden", [(), (64, 64), (32,)])
def test_incompatible_teacher_cannot_be_relabelled(hidden):
    with pytest.raises(ValueError):
        _factory()(_teacher(hidden), hidden_width=4, initialization_seed=4100)


@pytest.mark.parametrize("width", [True, 0, -1, 3, 4.5, 64])
def test_invalid_or_undeclared_student_capacity_refuses(width):
    with pytest.raises(ValueError):
        _factory()(_teacher(), hidden_width=width, initialization_seed=4100)


@pytest.mark.parametrize("width", [32, 4])
def test_checkpoint_roundtrip_keeps_actions_freeze_and_optimizer_contract(tmp_path, width):
    student = _factory()(_teacher(), hidden_width=width, initialization_seed=4100)
    observations = th.tensor([[0.1, 0.2, 0.3, 0.4], [-0.4, 0.2, -0.1, 0.3]])
    path = tmp_path / "capacity.pt"
    before = student.get_distribution(observations).distribution.logits.detach()
    student.save(path)
    restored = load_policy_checkpoint(path, device="cpu")
    after = restored.get_distribution(observations).distribution.logits.detach()
    assert th.equal(before, after)
    assert {
        name for name, p in student.named_parameters() if p.requires_grad
    } == {
        name for name, p in restored.named_parameters() if p.requires_grad
    }
    assert restored.features_extractor.training is False
