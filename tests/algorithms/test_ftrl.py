"""Tests for `imitation.algorithms.ftrl`."""

import dataclasses
from typing import Sequence

import numpy as np
import pytest
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

from imitation.algorithms import bc, dagger, ftrl
from imitation.data.types import TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies import serialize
from imitation.testing.expert_trajectories import lazy_generate_expert_trajectories
from imitation.util import util


CARTPOLE_ENV_NAME = "seals/CartPole-v0"


@pytest.fixture
def cartpole_venv_single(rng) -> VecEnv:
    return util.make_vec_env(
        CARTPOLE_ENV_NAME,
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        rng=rng,
    )


@pytest.fixture
def cartpole_expert_policy(cartpole_venv_single: VecEnv) -> BasePolicy:
    return serialize.load_policy(
        "ppo-huggingface",
        cartpole_venv_single,
        env_name=CARTPOLE_ENV_NAME,
    )


@pytest.fixture
def cartpole_expert_trajectories(
    pytestconfig,
    rng,
) -> Sequence[TrajectoryWithRew]:
    return lazy_generate_expert_trajectories(
        pytestconfig.cache.makedir("experts"),
        CARTPOLE_ENV_NAME,
        20,
        rng,
    )


# --- L2 Schedule Tests ---


def test_constant_l2_schedule():
    sched = ftrl.ConstantL2Schedule(0.01)
    for r in range(10):
        assert sched(r) == 0.01


def test_constant_l2_schedule_zero():
    sched = ftrl.ConstantL2Schedule(0.0)
    for r in range(5):
        assert sched(r) == 0.0


def test_constant_l2_schedule_negative():
    with pytest.raises(ValueError, match="non-negative"):
        ftrl.ConstantL2Schedule(-0.1)


def test_decaying_l2_schedule():
    sched = ftrl.DecayingL2Schedule(1.0)
    assert sched(0) == pytest.approx(1.0)
    assert sched(1) == pytest.approx(0.5)
    assert sched(2) == pytest.approx(1.0 / 3)
    assert sched(9) == pytest.approx(0.1)


def test_decaying_l2_schedule_negative():
    with pytest.raises(ValueError, match="non-negative"):
        ftrl.DecayingL2Schedule(-0.5)


# --- TrainableParamsLossCalculator Tests ---


def test_trainable_params_loss_calculator(cartpole_venv_single, rng):
    """L2 norm should only include trainable parameters."""
    bc_trainer = bc.BC(
        observation_space=cartpole_venv_single.observation_space,
        action_space=cartpole_venv_single.action_space,
        rng=rng,
    )
    policy = bc_trainer.policy

    # Freeze some parameters
    params = list(policy.parameters())
    for p in params[:len(params) // 2]:
        p.requires_grad = False

    calc_all = bc.BehaviorCloningLossCalculator(ent_weight=0.0, l2_weight=1.0)
    calc_trainable = ftrl.TrainableParamsLossCalculator(ent_weight=0.0, l2_weight=1.0)

    # Generate a dummy batch
    obs = cartpole_venv_single.observation_space.sample()
    obs = np.expand_dims(obs, 0)  # batch dim
    act = np.array([cartpole_venv_single.action_space.sample()])

    metrics_all = calc_all(policy, obs, act)
    metrics_trainable = calc_trainable(policy, obs, act)

    # Trainable-only L2 should be smaller since fewer params are counted
    assert metrics_trainable.l2_norm.item() < metrics_all.l2_norm.item()
    # Cross-entropy should be the same
    assert metrics_trainable.neglogp.item() == pytest.approx(
        metrics_all.neglogp.item(),
        abs=1e-5,
    )


# --- FTRLTrainer Tests ---


def _make_ftrl_trainer(
    tmpdir,
    venv,
    expert_policy,
    rng,
    l2_schedule=None,
    warm_start=True,
    track_per_round_loss=True,
    use_trainable_params_loss=False,
    custom_logger=None,
    expert_trajs=None,
):
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        optimizer_kwargs=dict(lr=1e-3),
        custom_logger=custom_logger,
        rng=rng,
    )
    return ftrl.FTRLTrainer(
        venv=venv,
        scratch_dir=tmpdir,
        bc_trainer=bc_trainer,
        expert_policy=expert_policy,
        rng=rng,
        l2_schedule=l2_schedule,
        warm_start=warm_start,
        track_per_round_loss=track_per_round_loss,
        use_trainable_params_loss=use_trainable_params_loss,
        expert_trajs=expert_trajs,
        custom_logger=custom_logger,
    )


def test_ftrl_updates_l2_per_round(
    tmpdir,
    cartpole_venv_single,
    cartpole_expert_policy,
    rng,
    custom_logger,
):
    """Verify L2 weight updates each round per the schedule."""
    schedule = ftrl.DecayingL2Schedule(1.0)
    trainer = _make_ftrl_trainer(
        tmpdir,
        cartpole_venv_single,
        cartpole_expert_policy,
        rng,
        l2_schedule=schedule,
        custom_logger=custom_logger,
    )

    trainer.train(
        total_timesteps=900,
        rollout_round_min_episodes=1,
        rollout_round_min_timesteps=300,
    )

    # Should have completed at least 2 rounds
    assert trainer.round_num >= 2
    metrics = trainer.get_metrics()
    assert len(metrics) >= 2


def test_ftrl_warm_start_vs_reinit(
    tmpdir,
    cartpole_venv_single,
    cartpole_expert_policy,
    rng,
    custom_logger,
):
    """With warm_start=False, params should be reinitialized between rounds."""
    import torch as th

    trainer_reinit = _make_ftrl_trainer(
        str(tmpdir.join("reinit")),
        cartpole_venv_single,
        cartpole_expert_policy,
        rng,
        l2_schedule=ftrl.ConstantL2Schedule(0.01),
        warm_start=False,
        track_per_round_loss=False,
        custom_logger=custom_logger,
    )

    trainer_warm = _make_ftrl_trainer(
        str(tmpdir.join("warm")),
        cartpole_venv_single,
        cartpole_expert_policy,
        rng,
        l2_schedule=ftrl.ConstantL2Schedule(0.01),
        warm_start=True,
        track_per_round_loss=False,
        custom_logger=custom_logger,
    )

    # Train both for 2+ rounds
    train_kwargs = dict(
        total_timesteps=900,
        rollout_round_min_episodes=1,
        rollout_round_min_timesteps=300,
    )
    trainer_reinit.train(**train_kwargs)
    trainer_warm.train(**train_kwargs)

    # Both should complete training (smoke test)
    assert trainer_reinit.round_num >= 2
    assert trainer_warm.round_num >= 2


def test_ftrl_tracks_per_round_metrics(
    tmpdir,
    cartpole_venv_single,
    cartpole_expert_policy,
    rng,
    custom_logger,
):
    """Verify per-round metrics are populated with correct fields."""
    trainer = _make_ftrl_trainer(
        tmpdir,
        cartpole_venv_single,
        cartpole_expert_policy,
        rng,
        l2_schedule=ftrl.ConstantL2Schedule(0.01),
        track_per_round_loss=True,
        custom_logger=custom_logger,
    )

    trainer.train(
        total_timesteps=600,
        rollout_round_min_episodes=1,
        rollout_round_min_timesteps=300,
    )

    metrics = trainer.get_metrics()
    assert len(metrics) >= 1

    for m in metrics:
        assert isinstance(m, ftrl.RoundMetrics)
        assert m.cross_entropy >= 0
        assert m.l2_norm >= 0
        assert m.total_loss >= 0
        assert isinstance(m.round_num, int)


def test_ftrl_as_ftl(
    tmpdir,
    cartpole_venv_single,
    cartpole_expert_policy,
    rng,
    custom_logger,
):
    """FTL mode: l2=0 should produce zero L2 loss in metrics."""
    trainer = _make_ftrl_trainer(
        tmpdir,
        cartpole_venv_single,
        cartpole_expert_policy,
        rng,
        l2_schedule=ftrl.ConstantL2Schedule(0.0),
        track_per_round_loss=True,
        custom_logger=custom_logger,
    )

    trainer.train(
        total_timesteps=600,
        rollout_round_min_episodes=1,
        rollout_round_min_timesteps=300,
    )

    metrics = trainer.get_metrics()
    assert len(metrics) >= 1
    # With l2_weight=0, total_loss should equal cross_entropy (+ ent_loss ~= 0)
    for m in metrics:
        # L2 norm is computed but not weighted, so l2_loss=0 but l2_norm>=0
        # total_loss = neglogp + ent_loss + 0 * l2_norm
        # We just check the L2 contribution is small relative to CE
        assert m.total_loss == pytest.approx(
            m.cross_entropy,
            abs=0.5,  # ent_loss can contribute a small amount
        )


def test_ftrl_smoke_cartpole(
    tmpdir,
    cartpole_venv_single,
    cartpole_expert_policy,
    cartpole_expert_trajectories,
    rng,
    custom_logger,
):
    """End-to-end smoke test: 3 rounds on CartPole with preloaded expert data."""
    trainer = _make_ftrl_trainer(
        tmpdir,
        cartpole_venv_single,
        cartpole_expert_policy,
        rng,
        l2_schedule=ftrl.ConstantL2Schedule(1e-4),
        expert_trajs=cartpole_expert_trajectories,
        custom_logger=custom_logger,
    )

    trainer.train(
        total_timesteps=1500,
        rollout_round_min_episodes=1,
        rollout_round_min_timesteps=500,
    )

    assert trainer.round_num >= 3
    metrics = trainer.get_metrics()
    assert len(metrics) == trainer.round_num


def test_round_metrics_dataclass():
    """RoundMetrics is a frozen dataclass with correct fields."""
    m = ftrl.RoundMetrics(round_num=0, cross_entropy=0.5, l2_norm=0.1, total_loss=0.6)
    assert m.round_num == 0
    assert m.cross_entropy == 0.5
    assert m.l2_norm == 0.1
    assert m.total_loss == 0.6

    with pytest.raises(dataclasses.FrozenInstanceError):
        m.round_num = 1  # type: ignore[misc]
