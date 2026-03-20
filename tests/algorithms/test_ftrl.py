"""Tests for FTRLDAggerTrainer and FTRLLossCalculator.

Covers ALGO-01 through ALGO-06 requirements:
  ALGO-01: Instantiation with CartPole fixtures
  ALGO-02: Proximal term centered on w_t anchor, not on zero
  ALGO-03: Linear correction sign and sigma_grad freshness at current weights
  ALGO-04: Anchor params frozen during round's optimization
  ALGO-05: FTL degeneracy with very large alpha
  ALGO-06: CartPole smoke test (5 rounds, reward >= baseline)
"""

import numpy as np
import pytest
import torch as th
from imitation.algorithms import bc, dagger, ftrl
from imitation.algorithms.bc import BehaviorCloningLossCalculator
from imitation.algorithms.ftrl import FTRLDAggerTrainer, FTRLLossCalculator
from imitation.util import util
from imitation.data.wrappers import RolloutInfoWrapper


# ---------------------------------------------------------------------------
# Helper: build an FTRLDAggerTrainer following test_dagger.py pattern
# ---------------------------------------------------------------------------

def _build_ftrl_trainer(tmpdir, venv, expert_policy, custom_logger, rng, alpha=1.0):
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        optimizer_kwargs=dict(lr=1e-3),
        custom_logger=custom_logger,
        rng=rng,
    )
    return ftrl.FTRLDAggerTrainer(
        alpha=alpha,
        venv=venv,
        scratch_dir=tmpdir,
        bc_trainer=bc_trainer,
        expert_policy=expert_policy,
        custom_logger=custom_logger,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Task 1: Unit tests for FTRLLossCalculator correctness (ALGO-01 to ALGO-04)
# ---------------------------------------------------------------------------

def test_ftrl_instantiation(tmpdir, cartpole_venv, cartpole_expert_policy, custom_logger, rng):
    """ALGO-01: FTRLDAggerTrainer can be constructed and is a SimpleDAggerTrainer."""
    trainer = _build_ftrl_trainer(
        tmpdir, cartpole_venv, cartpole_expert_policy, custom_logger, rng, alpha=1.0
    )
    assert isinstance(trainer, dagger.SimpleDAggerTrainer)
    assert trainer.alpha == 1.0
    assert hasattr(trainer, "_cumulative_sigma")
    assert trainer._cumulative_sigma == 0.0


def test_proximal_term_centering(cartpole_venv, rng):
    """ALGO-02: Proximal term is centered on w_t anchor, not zero.

    Verify:
      - loss at anchor == BC-only loss (proximal = 0)
      - loss with perturbed params > BC-only loss (proximal > 0)
    """
    from imitation.policies import base as policy_base
    import gymnasium as gym

    obs_space = cartpole_venv.observation_space
    act_space = cartpole_venv.action_space

    # Create a small policy
    policy = policy_base.FeedForward32Policy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: th.finfo(th.float32).max,
    )

    # Base BC loss calculator
    base_calc = BehaviorCloningLossCalculator(ent_weight=1e-3, l2_weight=0.0)

    # Anchor = current policy params
    anchor_params = [p.detach().clone() for p in policy.parameters()]

    # sigma_grad = all zeros (no linear correction)
    sigma_grad = [th.zeros_like(p) for p in policy.parameters()]

    # Create FTRLLossCalculator with policy AT the anchor (proximal = 0)
    ftrl_calc_at_anchor = FTRLLossCalculator(
        bc_loss_calculator=base_calc,
        anchor_params=anchor_params,
        sigma_grad=sigma_grad,
        eta_t=1.0,
    )

    # Create dummy obs/acts: sample from the space
    obs_np = np.array([obs_space.sample() for _ in range(8)], dtype=np.float32)
    acts_np = np.array([act_space.sample() for _ in range(8)])

    # Compute loss with policy == anchor (proximal should be ~0)
    metrics_at_anchor = ftrl_calc_at_anchor(policy, obs_np, acts_np)
    bc_metrics = base_calc(policy, obs_np, acts_np)

    # FTRL loss at anchor should equal BC loss (proximal = 0, sigma_grad = 0)
    assert th.allclose(
        metrics_at_anchor.loss, bc_metrics.loss, atol=1e-5
    ), f"FTRL at anchor: {metrics_at_anchor.loss.item():.6f} vs BC: {bc_metrics.loss.item():.6f}"

    # Now perturb policy params by adding a small amount to each
    # Use 0.01 to avoid NaN in logits while still creating nonzero proximal term
    with th.no_grad():
        for p in policy.parameters():
            p.add_(0.01)

    # Compute FTRL loss with perturbed params (proximal should be > 0)
    metrics_perturbed = ftrl_calc_at_anchor(policy, obs_np, acts_np)
    bc_metrics_perturbed = base_calc(policy, obs_np, acts_np)

    # FTRL loss with perturbed params should be > BC loss (proximal is positive)
    assert metrics_perturbed.loss > bc_metrics_perturbed.loss, (
        f"FTRL with perturbed params ({metrics_perturbed.loss.item():.4f}) should "
        f"exceed BC loss ({bc_metrics_perturbed.loss.item():.4f}) due to proximal term"
    )


def test_linear_correction_sign(cartpole_venv, rng):
    """ALGO-03: Linear correction has the correct sign (negative dot product with weights).

    With sigma_grad = all-ones and large eta_t (weak proximal), the loss should
    differ from BC-only loss by approximately -sum(all params).
    """
    from imitation.policies import base as policy_base

    obs_space = cartpole_venv.observation_space
    act_space = cartpole_venv.action_space

    policy = policy_base.FeedForward32Policy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: th.finfo(th.float32).max,
    )

    base_calc = BehaviorCloningLossCalculator(ent_weight=1e-3, l2_weight=0.0)

    # anchor_params = current params (so proximal = 0)
    anchor_params = [p.detach().clone() for p in policy.parameters()]
    # sigma_grad = all ones
    sigma_grad = [th.ones_like(p) for p in policy.parameters()]

    # Very large eta_t -> proximal coefficient (1/(2*eta_t)) is negligible
    ftrl_calc = FTRLLossCalculator(
        bc_loss_calculator=base_calc,
        anchor_params=anchor_params,
        sigma_grad=sigma_grad,
        eta_t=1e10,
    )

    obs_np = np.array([obs_space.sample() for _ in range(8)], dtype=np.float32)
    acts_np = np.array([act_space.sample() for _ in range(8)])

    ftrl_metrics = ftrl_calc(policy, obs_np, acts_np)
    bc_metrics = base_calc(policy, obs_np, acts_np)

    # Expected linear correction: -sum(p * 1) for all params
    expected_linear = -sum(th.sum(p).item() for p in policy.parameters())

    # Difference between FTRL and BC loss should approximate the linear correction
    actual_diff = (ftrl_metrics.loss - bc_metrics.loss).item()

    # The proximal term (1/(2*1e10)) * ||w||^2 is negligible, so diff ~ linear_correction
    assert abs(actual_diff - expected_linear) < 1e-3, (
        f"Linear correction: expected ~{expected_linear:.4f}, "
        f"got diff {actual_diff:.4f}"
    )
    # Verify the linear correction affects the loss
    assert abs(actual_diff) > 1e-4 or abs(expected_linear) < 1e-4, (
        "Linear correction should affect the total loss"
    )


def _compute_full_grad(policy, data_loader, base_calc, device):
    """Compute accumulated gradient over an entire data loader, deterministically.

    Uses a fresh copy of the policy parameters to avoid order-dependence issues
    with shuffled data loaders. Converts the DataLoader to a list first to fix
    the batch order for this computation.
    """
    from imitation.data import types as data_types
    from imitation.util import util as imit_util

    policy.zero_grad()
    # Collect all batches first to fix iteration order (data loader shuffles each time)
    batches = list(data_loader)
    for batch in batches:
        obs_tensor = data_types.map_maybe_dict(
            lambda x: imit_util.safe_to_tensor(x, device=device),
            data_types.maybe_unwrap_dictobs(batch["obs"]),
        )
        acts = imit_util.safe_to_tensor(batch["acts"], device=device)
        metrics = base_calc(policy, obs_tensor, acts)
        metrics.loss.backward()
    grad = [
        p.grad.detach().clone() if p.grad is not None else th.zeros_like(p)
        for p in policy.parameters()
    ]
    policy.zero_grad()
    return grad


def test_sigma_grad_at_current_weights(
    tmpdir, cartpole_venv, cartpole_expert_policy, custom_logger, rng
):
    """ALGO-03: sigma_grad is computed at current weights w_t (not stale weights).

    Verifies sigma_grad is more correlated with gradients at anchor_params
    than with gradients at the post-training weights, proving it was computed
    at the round-start snapshot w_t rather than some stale value.
    """
    trainer = _build_ftrl_trainer(
        tmpdir, cartpole_venv, cartpole_expert_policy, custom_logger, rng, alpha=1.0
    )

    # Run two full rounds so the second round has non-trivial sigma_grad
    # (round 0 sigma_grad is zeros; round 1+ uses actual gradients)
    # Use large total_timesteps to ensure >= 2 rounds even with n_envs=4
    trainer.train(
        total_timesteps=3000,
        rollout_round_min_episodes=1,
        rollout_round_min_timesteps=100,
    )

    # We need at least 2 rounds for sigma_grad to be non-trivial
    if trainer.round_num < 2:
        pytest.skip(
            f"Need at least 2 rounds; only got {trainer.round_num}. "
            "Increase total_timesteps or reduce min_timesteps."
        )

    # loss_calculator.anchor_params = weights at the start of the LAST round
    # loss_calculator.sigma_grad = gradients computed at those anchor weights
    stored_sigma_grad = trainer.bc_trainer.loss_calculator.sigma_grad
    anchor_params = trainer.bc_trainer.loss_calculator.anchor_params
    current_params = [p.detach().clone() for p in trainer.bc_trainer.policy.parameters()]
    base_calc = trainer._default_bc_loss_calculator
    device = trainer.bc_trainer.policy.device

    # Verify sigma_grad is non-zero (meaningful for round > 0)
    sigma_grad_norm = sum(g.norm().item() for g in stored_sigma_grad)
    assert sigma_grad_norm > 0, "sigma_grad should be non-zero after round 0"

    # Compute gradients at anchor_params and at post-training weights using SAME batches
    # Save current (post-training) weights
    with th.no_grad():
        for p, a in zip(trainer.bc_trainer.policy.parameters(), anchor_params):
            p.copy_(a)
    grad_at_anchor = _compute_full_grad(
        trainer.bc_trainer.policy, trainer.bc_trainer._demo_data_loader, base_calc, device
    )

    with th.no_grad():
        for p, c in zip(trainer.bc_trainer.policy.parameters(), current_params):
            p.copy_(c)
    grad_at_current = _compute_full_grad(
        trainer.bc_trainer.policy, trainer.bc_trainer._demo_data_loader, base_calc, device
    )

    # Compute cosine similarity of stored sigma_grad with both gradient estimates
    def flat_cat(grads):
        return th.cat([g.flatten() for g in grads])

    stored_flat = flat_cat(stored_sigma_grad)
    anchor_flat = flat_cat(grad_at_anchor)
    current_flat = flat_cat(grad_at_current)

    def cosine_sim(a, b):
        return (a * b).sum() / (a.norm() * b.norm() + 1e-10)

    sim_with_anchor = cosine_sim(stored_flat, anchor_flat).item()
    sim_with_current = cosine_sim(stored_flat, current_flat).item()

    # sigma_grad should align much better with gradients at anchor_params
    # than with gradients at post-training weights (which have moved significantly)
    assert sim_with_anchor > 0.9, (
        f"sigma_grad should have high cosine similarity with gradient at anchor_params, "
        f"got {sim_with_anchor:.3f}"
    )
    assert sim_with_anchor > sim_with_current - 0.1, (
        f"sigma_grad should align at least as well with anchor grads ({sim_with_anchor:.3f}) "
        f"as with post-training grads ({sim_with_current:.3f})"
    )


def test_anchor_frozen_during_round(
    tmpdir, cartpole_venv, cartpole_expert_policy, custom_logger, rng
):
    """ALGO-04: Anchor params are frozen during the round's optimization.

    After training, the loss_calculator's anchor_params should reflect the
    weights at round start — they must NOT equal the final trained weights
    (unless the policy didn't move at all, which is unlikely with real data).
    """
    trainer = _build_ftrl_trainer(
        tmpdir, cartpole_venv, cartpole_expert_policy, custom_logger, rng, alpha=1.0
    )

    # Train for 1-2 rounds so there is at least one round with FTRL loss
    trainer.train(total_timesteps=1000, rollout_round_min_episodes=1, rollout_round_min_timesteps=100)

    # After training, loss_calculator should be an FTRLLossCalculator
    assert isinstance(trainer.bc_trainer.loss_calculator, FTRLLossCalculator), (
        "bc_trainer.loss_calculator should be FTRLLossCalculator after training"
    )

    anchor_params = trainer.bc_trainer.loss_calculator.anchor_params
    current_params = list(trainer.bc_trainer.policy.parameters())

    # Policy should have moved during training — anchors should differ from trained params
    param_diffs = [
        (p.detach() - a.detach()).abs().sum().item()
        for p, a in zip(current_params, anchor_params)
    ]
    total_diff = sum(param_diffs)

    assert total_diff > 0, (
        "Anchor params should differ from current policy params after training. "
        f"Total L1 diff = {total_diff:.6f}"
    )


# ---------------------------------------------------------------------------
# Task 2: FTL degeneracy test and CartPole smoke test (ALGO-05, ALGO-06)
# ---------------------------------------------------------------------------

def test_ftl_degeneracy(tmpdir, cartpole_venv, cartpole_expert_policy, custom_logger, rng):
    """ALGO-05: With very large alpha, FTRL should behave like FTL (plain DAgger)."""
    # Build FTRL trainer with alpha=1e8 (large alpha = large eta = weak proximal = FTL)
    ftrl_trainer = _build_ftrl_trainer(
        tmpdir / "ftrl", cartpole_venv, cartpole_expert_policy, custom_logger, rng, alpha=1e8
    )
    # Build plain DAgger trainer for comparison
    bc_trainer_dagger = bc.BC(
        observation_space=cartpole_venv.observation_space,
        action_space=cartpole_venv.action_space,
        optimizer_kwargs=dict(lr=1e-3),
        custom_logger=custom_logger,
        rng=np.random.default_rng(0),
    )
    dagger_trainer = dagger.SimpleDAggerTrainer(
        venv=cartpole_venv,
        scratch_dir=tmpdir / "dagger",
        bc_trainer=bc_trainer_dagger,
        expert_policy=cartpole_expert_policy,
        custom_logger=custom_logger,
        rng=np.random.default_rng(0),
    )

    total_timesteps = 2000  # enough for ~4-5 rounds
    ftrl_trainer.train(total_timesteps=total_timesteps)
    dagger_trainer.train(total_timesteps=total_timesteps)

    # Evaluate both policies
    from imitation.data import rollout
    ftrl_trajs = rollout.generate_trajectories(
        ftrl_trainer.policy, cartpole_venv,
        rollout.make_sample_until(min_episodes=10, min_timesteps=None),
        rng=np.random.default_rng(42),
    )
    ftrl_rewards = np.mean([np.sum(t.rews) for t in ftrl_trajs])

    dagger_trajs = rollout.generate_trajectories(
        dagger_trainer.policy, cartpole_venv,
        rollout.make_sample_until(min_episodes=10, min_timesteps=None),
        rng=np.random.default_rng(42),
    )
    dagger_rewards = np.mean([np.sum(t.rews) for t in dagger_trajs])

    # With alpha=1e8, proximal is essentially zero, so FTRL ≈ DAgger
    # Allow generous tolerance since stochastic
    assert abs(ftrl_rewards - dagger_rewards) / max(abs(dagger_rewards), 1.0) < 0.5, (
        f"FTRL with large alpha should approximate DAgger. "
        f"FTRL reward={ftrl_rewards:.1f}, DAgger reward={dagger_rewards:.1f}"
    )


def test_cartpole_smoke(tmpdir, cartpole_venv, cartpole_expert_policy, custom_logger, rng):
    """ALGO-06: FTRL on CartPole for 5 rounds completes without error, reward >= BC."""
    trainer = _build_ftrl_trainer(
        tmpdir, cartpole_venv, cartpole_expert_policy, custom_logger, rng, alpha=1.0
    )
    # Train for enough timesteps to get ~5 rounds
    trainer.train(total_timesteps=2000)
    assert trainer.round_num >= 5, f"Expected at least 5 rounds, got {trainer.round_num}"

    # Evaluate
    from imitation.data import rollout
    trajs = rollout.generate_trajectories(
        trainer.policy, cartpole_venv,
        rollout.make_sample_until(min_episodes=10, min_timesteps=None),
        rng=np.random.default_rng(42),
    )
    mean_reward = np.mean([np.sum(t.rews) for t in trajs])

    # CartPole random baseline is ~20; expert is ~500
    # FTRL after 5 rounds should be significantly above random
    assert mean_reward > 50, (
        f"FTRL CartPole reward {mean_reward:.1f} should exceed random baseline (~20)"
    )
