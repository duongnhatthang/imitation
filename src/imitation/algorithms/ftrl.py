"""Follow-The-Regularized-Leader (FTRL) algorithm for online imitation learning.

Implements Lavington et al. (2022) "Improved Policy Optimization for Online
Imitation Learning" Eq. 6, wrapping the existing DAgger/BC infrastructure.

The core idea: each round of DAgger training augments the BC loss with:
  1. A proximal term: (1/(2*eta_t)) * ||w - w_t||^2   (anchored at w_t)
  2. A linear correction: -<w, Sigma_{i<t} grad l_i(w_t)>

Both terms are computed at the boundary between rounds, before the BC inner loop.
"""

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch as th
from stable_baselines3.common import policies, utils, vec_env

from imitation.algorithms import bc, dagger
from imitation.algorithms.bc import BCTrainingMetrics, BehaviorCloningLossCalculator
from imitation.data import types
from imitation.util import util


@dataclasses.dataclass(frozen=True)
class FTRLLossCalculator:
    """Loss calculator implementing the FTRL regularized objective (Eq. 6).

    Wraps a base BehaviorCloningLossCalculator and adds:
      - Proximal term: (1/(2*eta_t)) * ||w - w_t||^2
      - Linear correction: -<w, sigma_grad>

    Both anchor_params and sigma_grad are frozen (detached) snapshots from the
    start of the current round. A new FTRLLossCalculator is created each round.
    """

    bc_loss_calculator: BehaviorCloningLossCalculator
    """The base BC loss calculator used to compute the standard imitation loss."""

    anchor_params: List[th.Tensor]
    """Frozen snapshot of policy weights w_t at the start of the current round."""

    sigma_grad: List[th.Tensor]
    """Accumulated sum of gradients of past losses evaluated at w_t."""

    eta_t: float
    """FTRL learning rate for this round. Large eta_t = weak proximal regularization."""

    def __call__(
        self,
        policy: policies.ActorCriticPolicy,
        obs: Union[
            types.AnyTensor,
            types.DictObs,
            Dict[str, np.ndarray],
            Dict[str, th.Tensor],
        ],
        acts: Union[th.Tensor, np.ndarray],
    ) -> BCTrainingMetrics:
        """Compute FTRL loss = BC loss + proximal term + linear correction.

        Args:
            policy: The actor-critic policy whose loss is being computed.
            obs: The observations seen by the expert.
            acts: The actions taken by the expert.

        Returns:
            A BCTrainingMetrics object with the combined FTRL loss.
        """
        # Step 1: compute base BC loss
        bc_metrics = self.bc_loss_calculator(policy, obs, acts)

        # Step 2: proximal term = (1/(2*eta_t)) * ||w - w_t||^2
        # anchor_params already detached but use .detach() defensively
        proximal_term = sum(
            th.sum((p - a.detach()) ** 2)
            for p, a in zip(policy.parameters(), self.anchor_params)
        ) / (2.0 * self.eta_t)

        # Step 3: linear correction = -<w, sigma_grad>  (negative sign per Eq. 6)
        linear_correction = -sum(
            th.sum(p * g.detach())
            for p, g in zip(policy.parameters(), self.sigma_grad)
        )

        # Step 4: combine into total FTRL loss
        total_loss = bc_metrics.loss + proximal_term + linear_correction

        # Step 5: return BCTrainingMetrics with updated loss (frozen dataclass, use replace)
        return dataclasses.replace(bc_metrics, loss=total_loss)


class FTRLDAggerTrainer(dagger.SimpleDAggerTrainer):
    """DAgger trainer with FTRL regularization (Lavington et al. 2022, Eq. 6).

    At the boundary between each round:
      1. Snapshots the current policy weights as anchor w_t.
      2. Computes sigma_grad = sum of gradients of all past losses evaluated at w_t.
      3. Computes eta_t = alpha / cumulative_sigma.
      4. Injects a FTRLLossCalculator into bc_trainer for this round.

    Parameterization: eta_t = alpha / cumulative_sigma, so:
      - Large alpha => large eta_t => small proximal coefficient (1/(2*eta_t)) => weak
        regularization => algorithm degenerates toward FTL (plain DAgger) as alpha -> inf.
      - Small alpha => small eta_t => large proximal coefficient => strong proximal pull
        toward anchor w_t.
    """

    def __init__(self, *, alpha: float = 1.0, **kwargs):
        """Build FTRLDAggerTrainer.

        Args:
            alpha: FTRL step-size scale. Large alpha produces weak proximal regularization
                (FTL degeneracy); small alpha produces strong proximal pull. Defaults to 1.0.
            **kwargs: All other arguments forwarded to SimpleDAggerTrainer.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        # Accumulated sum of per-round sigma_i values (sigma_i = 1.0 constant)
        self._cumulative_sigma: float = 0.0
        # Save original BC loss calculator to use inside FTRLLossCalculator
        self._default_bc_loss_calculator: BehaviorCloningLossCalculator = (
            self.bc_trainer.loss_calculator
        )

    def extend_and_update(
        self,
        bc_train_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Extend data, inject FTRL loss, train BC, and advance round counter.

        Overrides DAggerTrainer.extend_and_update to insert pre-round FTRL setup:
          1. Load demos (populates bc_trainer._demo_data_loader)
          2. Snapshot anchor w_t and compute sigma_grad at current weights
          3. Compute eta_t and inject FTRLLossCalculator into bc_trainer
          4. Train BC and increment round counter

        Args:
            bc_train_kwargs: Keyword arguments for BC.train(). Defaults follow parent class.

        Returns:
            New round number after advancing the round counter.
        """
        # Step 1: normalize bc_train_kwargs (mirror parent's defaulting logic)
        if bc_train_kwargs is None:
            bc_train_kwargs = {}
        else:
            bc_train_kwargs = dict(bc_train_kwargs)
        if "log_rollouts_venv" not in bc_train_kwargs:
            bc_train_kwargs["log_rollouts_venv"] = self.venv
        if "n_epochs" not in bc_train_kwargs and "n_batches" not in bc_train_kwargs:
            bc_train_kwargs["n_epochs"] = self.DEFAULT_N_EPOCHS

        # Step 2: load demos so data loader is populated for sigma_grad computation
        logging.info("Loading demonstrations")
        self._try_load_demos()

        # Step 3: snapshot anchor weights w_t (detached, no graph connection)
        anchor_params = [
            p.detach().clone() for p in self.bc_trainer.policy.parameters()
        ]

        # Step 4: compute sigma_grad = sum of gradients of past losses at w_t
        if self.round_num == 0:
            # No past data on round 0; sigma_grad is all zeros
            sigma_grad = [
                th.zeros_like(p) for p in self.bc_trainer.policy.parameters()
            ]
        else:
            # Accumulate gradients over the full dataset at current weights w_t.
            # Normalize by the number of batches so sigma_grad is a per-batch-average
            # gradient, keeping it on the same scale as the BC loss gradient during
            # the inner optimization loop. Without this, sigma_grad grows with the
            # dataset size and can overwhelm the BC loss in later rounds.
            self.bc_trainer.policy.zero_grad()
            base_calc = self._default_bc_loss_calculator
            n_batches = 0
            for batch in self.bc_trainer._demo_data_loader:
                obs_tensor = types.map_maybe_dict(
                    lambda x: util.safe_to_tensor(x, device=self.bc_trainer.policy.device),
                    types.maybe_unwrap_dictobs(batch["obs"]),
                )
                acts = util.safe_to_tensor(
                    batch["acts"], device=self.bc_trainer.policy.device
                )
                metrics = base_calc(self.bc_trainer.policy, obs_tensor, acts)
                metrics.loss.backward()
                n_batches += 1
            # Normalize by number of batches so gradient is on per-batch scale
            n_batches = max(n_batches, 1)
            sigma_grad = [
                (p.grad / n_batches).detach().clone() if p.grad is not None else th.zeros_like(p)
                for p in self.bc_trainer.policy.parameters()
            ]
            # Clear gradients so BC.train() starts from zero (CRITICAL: prevents stale grads)
            self.bc_trainer.policy.zero_grad()

        # Step 5: compute eta_t using constant sigma_i = 1.0
        sigma_i = 1.0
        self._cumulative_sigma += sigma_i
        eta_t = self.alpha / self._cumulative_sigma

        # Step 6: inject FTRLLossCalculator into bc_trainer for this round
        self.bc_trainer.loss_calculator = FTRLLossCalculator(
            bc_loss_calculator=self._default_bc_loss_calculator,
            anchor_params=anchor_params,
            sigma_grad=sigma_grad,
            eta_t=eta_t,
        )

        # Step 7: train and advance round counter
        logging.info(f"FTRL training at round {self.round_num} with eta_t={eta_t:.6f}")
        self.bc_trainer.train(**bc_train_kwargs)
        self.round_num += 1
        logging.info(f"FTRL new round number is {self.round_num}")
        return self.round_num
