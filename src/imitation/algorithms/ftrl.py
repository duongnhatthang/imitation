"""Follow The Regularized Leader (FTRL) for imitation learning.

Extends DAgger with configurable L2 regularization schedules, optional
warm-starting, and per-round loss tracking. FTL (Follow The Leader) is the
special case where the L2 weight is zero.
"""

import abc
import dataclasses
import logging
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
import torch as th
from stable_baselines3.common import policies, vec_env

from imitation.algorithms import bc, dagger
from imitation.data import rollout, types
from imitation.util import logger as imit_logger


class L2Schedule(abc.ABC):
    """Abstract base class for L2 regularization weight schedules."""

    @abc.abstractmethod
    def __call__(self, round_num: int) -> float:
        """Return the L2 weight for the given round number.

        Args:
            round_num: The current DAgger round (0-indexed).

        Returns:
            Non-negative L2 regularization weight.
        """


class ConstantL2Schedule(L2Schedule):
    """Returns a fixed L2 weight regardless of round number."""

    def __init__(self, lambda_: float) -> None:
        """Builds ConstantL2Schedule.

        Args:
            lambda_: The constant L2 weight. Must be non-negative.

        Raises:
            ValueError: If lambda_ is negative.
        """
        if lambda_ < 0:
            raise ValueError(f"lambda_ must be non-negative, got {lambda_}")
        self.lambda_ = lambda_

    def __call__(self, round_num: int) -> float:
        return self.lambda_


class DecayingL2Schedule(L2Schedule):
    """Returns L2 weight that decays as lambda_ / (round_num + 1)."""

    def __init__(self, lambda_: float) -> None:
        """Builds DecayingL2Schedule.

        Args:
            lambda_: The initial L2 weight. Must be non-negative.

        Raises:
            ValueError: If lambda_ is negative.
        """
        if lambda_ < 0:
            raise ValueError(f"lambda_ must be non-negative, got {lambda_}")
        self.lambda_ = lambda_

    def __call__(self, round_num: int) -> float:
        return self.lambda_ / (round_num + 1)


@dataclasses.dataclass(frozen=True)
class TrainableParamsLossCalculator(bc.BehaviorCloningLossCalculator):
    """Like BehaviorCloningLossCalculator but only penalizes trainable params."""

    def __call__(
        self,
        policy,
        obs,
        acts,
    ) -> bc.BCTrainingMetrics:
        """Calculate BC loss with L2 only on parameters with requires_grad=True.

        Args:
            policy: The actor-critic policy whose loss is being computed.
            obs: The observations seen by the expert.
            acts: The actions taken by the expert.

        Returns:
            A BCTrainingMetrics object.
        """
        from imitation.util import util

        device = policy.device
        tensor_obs = types.map_maybe_dict(
            lambda x: util.safe_to_tensor(x).to(device),
            types.maybe_unwrap_dictobs(obs),
        )
        acts = util.safe_to_tensor(acts).to(device)

        (_, log_prob, entropy) = policy.evaluate_actions(
            tensor_obs,
            acts,
        )
        prob_true_act = th.exp(log_prob).mean()
        log_prob = log_prob.mean()
        entropy = entropy.mean() if entropy is not None else None

        # Only penalize trainable parameters
        l2_norms = [
            th.sum(th.square(w)) for w in policy.parameters() if w.requires_grad
        ]
        l2_norm = sum(l2_norms) / 2
        assert isinstance(l2_norm, th.Tensor)

        ent_loss = -self.ent_weight * (entropy if entropy is not None else th.zeros(1))
        neglogp = -log_prob
        l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss + l2_loss

        return bc.BCTrainingMetrics(
            neglogp=neglogp,
            entropy=entropy,
            ent_loss=ent_loss,
            prob_true_act=prob_true_act,
            l2_norm=l2_norm,
            l2_loss=l2_loss,
            loss=loss,
        )


@dataclasses.dataclass(frozen=True)
class RoundMetrics:
    """Metrics collected for a single FTRL training round."""

    round_num: int
    cross_entropy: float
    l2_norm: float
    total_loss: float


class FTRLTrainer(dagger.SimpleDAggerTrainer):
    """DAgger trainer with configurable L2 regularization schedule.

    Extends SimpleDAggerTrainer to support:
    - Per-round L2 weight updates via an L2Schedule
    - Optional warm-starting (vs reinitializing trainable params each round)
    - Per-round metric tracking (cross-entropy on current round's data)

    FTL (Follow The Leader) is the special case with l2_schedule returning 0.
    """

    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        scratch_dir: types.AnyPath,
        expert_policy: policies.BasePolicy,
        rng: np.random.Generator,
        l2_schedule: Optional[L2Schedule] = None,
        warm_start: bool = True,
        track_per_round_loss: bool = True,
        use_trainable_params_loss: bool = False,
        expert_trajs: Optional[Sequence[types.Trajectory]] = None,
        **dagger_trainer_kwargs,
    ):
        """Builds FTRLTrainer.

        Args:
            venv: Vectorized training environment.
            scratch_dir: Directory for intermediate training data.
            expert_policy: Expert policy for synthetic demonstrations.
            rng: Random state for random number generation.
            l2_schedule: Schedule returning L2 weight per round. If None,
                uses ConstantL2Schedule(0.0) (equivalent to FTL/plain DAgger).
            warm_start: If True (default), keep policy weights between rounds.
                If False, reinitialize trainable parameters each round.
            track_per_round_loss: If True (default), evaluate cross-entropy on
                the current round's data after each training step.
            use_trainable_params_loss: If True, use TrainableParamsLossCalculator
                (L2 only on requires_grad params). Useful for linear policy mode.
            expert_trajs: Optional starting dataset inserted into round 0.
            dagger_trainer_kwargs: Passed to SimpleDAggerTrainer.__init__.
        """
        super().__init__(
            venv=venv,
            scratch_dir=scratch_dir,
            expert_policy=expert_policy,
            rng=rng,
            expert_trajs=expert_trajs,
            **dagger_trainer_kwargs,
        )

        if l2_schedule is None:
            l2_schedule = ConstantL2Schedule(0.0)
        self.l2_schedule = l2_schedule
        self.warm_start = warm_start
        self.track_per_round_loss = track_per_round_loss
        self.use_trainable_params_loss = use_trainable_params_loss
        self.per_round_metrics: List[RoundMetrics] = []

    def _update_l2_weight(self, round_num: int) -> None:
        """Update the BC trainer's loss calculator with the current L2 weight."""
        l2_weight = self.l2_schedule(round_num)
        ent_weight = self.bc_trainer.loss_calculator.ent_weight

        if self.use_trainable_params_loss:
            self.bc_trainer.loss_calculator = TrainableParamsLossCalculator(
                ent_weight=ent_weight,
                l2_weight=l2_weight,
            )
        else:
            self.bc_trainer.loss_calculator = bc.BehaviorCloningLossCalculator(
                ent_weight=ent_weight,
                l2_weight=l2_weight,
            )

    def _reinitialize_trainable_params(self) -> None:
        """Reinitialize all trainable parameters with Xavier uniform."""
        for p in self.policy.parameters():
            if p.requires_grad and p.dim() >= 2:
                th.nn.init.xavier_uniform_(p)
            elif p.requires_grad and p.dim() == 1:
                th.nn.init.zeros_(p)

    def _compute_round_loss(
        self,
        round_transitions: types.Transitions,
        batch_size: int = 1024,
    ) -> RoundMetrics:
        """Evaluate the current policy on a set of transitions.

        Processes ``round_transitions`` in ``batch_size`` chunks so that
        Atari rounds with thousands of frame-stacked 84x84 observations do
        not OOM the GPU on a single forward pass through a CNN feature
        extractor. Classical MDP rounds typically have a few hundred low-
        dimensional observations and fit in a single chunk, so this is a
        no-op for classical runs (same numeric result as one big pass).

        Args:
            round_transitions: Transitions from the current round.
            batch_size: Maximum number of transitions per forward pass.
                Defaults to 1024; any value that fits a CNN forward pass
                in GPU memory is fine.

        Returns:
            RoundMetrics with cross-entropy and L2 norm on the given data.
        """
        from imitation.util import util

        device = self.policy.device
        obs_all = types.maybe_unwrap_dictobs(round_transitions.obs)
        acts_all = round_transitions.acts
        total_n = len(acts_all)

        loss_calc = self.bc_trainer.loss_calculator
        l2_weight = loss_calc.l2_weight
        ent_weight = loss_calc.ent_weight

        # L2 norm is data-independent; compute once. Match the loss
        # calculator's choice of trainable vs all parameters.
        if isinstance(loss_calc, TrainableParamsLossCalculator):
            l2_norms = [
                th.sum(th.square(w))
                for w in self.policy.parameters()
                if w.requires_grad
            ]
        else:
            l2_norms = [th.sum(th.square(w)) for w in self.policy.parameters()]
        l2_norm_t = sum(l2_norms) / 2
        assert isinstance(l2_norm_t, th.Tensor)

        # Aggregate log_prob and entropy as sums, convert to means at end.
        log_prob_sum = 0.0
        entropy_sum = 0.0
        entropy_seen = False

        self.policy.eval()
        with th.no_grad():
            for start in range(0, total_n, batch_size):
                end = min(start + batch_size, total_n)
                obs_chunk = types.map_maybe_dict(lambda x: x[start:end], obs_all)
                tensor_obs = types.map_maybe_dict(
                    lambda x: util.safe_to_tensor(x).to(device),
                    obs_chunk,
                )
                acts_chunk = util.safe_to_tensor(acts_all[start:end]).to(device)
                _, log_prob, entropy = self.policy.evaluate_actions(
                    tensor_obs, acts_chunk
                )
                log_prob_sum += float(log_prob.sum().item())
                if entropy is not None:
                    entropy_sum += float(entropy.sum().item())
                    entropy_seen = True
                del tensor_obs, acts_chunk, log_prob, entropy
        self.policy.train()

        denom = max(total_n, 1)
        neglogp_val = -log_prob_sum / denom
        if entropy_seen:
            mean_entropy = entropy_sum / denom
            ent_loss_val = -ent_weight * mean_entropy
        else:
            ent_loss_val = 0.0
        l2_norm_val = float(l2_norm_t.item())
        l2_loss_val = l2_weight * l2_norm_val
        total_loss_val = neglogp_val + ent_loss_val + l2_loss_val

        return RoundMetrics(
            round_num=self.round_num,
            cross_entropy=neglogp_val,
            l2_norm=l2_norm_val,
            total_loss=total_loss_val,
        )

    def extend_and_update(
        self,
        bc_train_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Extend dataset and train with updated L2 regularization.

        1. Update L2 weight from schedule
        2. Optionally reinitialize trainable parameters
        3. Train via parent's extend_and_update
        4. Optionally compute per-round metrics

        Args:
            bc_train_kwargs: Keyword arguments for BC.train().

        Returns:
            New round number after advancing the round counter.
        """
        current_round = self.round_num

        # Update L2 weight for this round
        self._update_l2_weight(current_round)
        logging.info(
            f"FTRL round {current_round}: "
            f"l2_weight={self.bc_trainer.loss_calculator.l2_weight}",
        )

        # Optionally reinitialize trainable parameters
        if not self.warm_start and current_round > 0:
            self._reinitialize_trainable_params()
            logging.info(f"Reinitialized trainable params for round {current_round}")

        # Train (this increments self.round_num)
        new_round = super().extend_and_update(bc_train_kwargs)

        # Compute per-round metrics on current round's data
        if self.track_per_round_loss:
            round_dir = self._demo_dir_path_for_round(current_round)
            demo_paths = self._get_demo_paths(round_dir)
            round_demos = []
            from imitation.data import serialize

            for p in demo_paths:
                trajs = serialize.load(p)
                round_demos.extend(trajs)
            round_transitions = rollout.flatten_trajectories(round_demos)

            round_metrics = self._compute_round_loss(round_transitions)
            self.per_round_metrics.append(round_metrics)
            logging.info(
                f"Round {current_round} metrics: "
                f"CE={round_metrics.cross_entropy:.4f}, "
                f"L2={round_metrics.l2_norm:.4f}",
            )

        return new_round

    def get_metrics(self) -> List[RoundMetrics]:
        """Return per-round metrics collected during training.

        Returns:
            List of RoundMetrics, one per completed round.
        """
        return list(self.per_round_metrics)
