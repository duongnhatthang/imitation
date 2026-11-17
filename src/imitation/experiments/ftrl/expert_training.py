"""Convergence-detecting classical PPO expert trainer.

Wraps PPO.learn in a chunked loop and stops when the normalized return has
plateaued above a threshold AND the policy's self-CE (argmax-sharpness
gauge) is below a tightness bound. Raises RuntimeError if the max step
budget is exhausted before convergence.
"""

import collections
import logging
import pathlib
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

from imitation.experiments.ftrl import env_utils
from imitation.experiments.ftrl.env_baselines import REFERENCE_BASELINES
from imitation.experiments.ftrl.eval_utils import eval_policy_rollout

logger = logging.getLogger(__name__)


DEFAULT_NET_ARCH = (64, 64)


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def policy_architecture(policy) -> Dict[str, Any]:
    """Actual architecture of an SB3 actor-critic policy, read from its modules.

    Args:
        policy: An SB3 ``ActorCriticPolicy`` (for example ``model.policy``).

    Returns:
        Hidden widths of the actor and critic MLPs, their activation classes,
        the features extractor class, features and action dimensions.
    """

    def describe(net):
        widths = [m.out_features for m in net if isinstance(m, th.nn.Linear)]
        others = sorted(
            {_qualified(type(m)) for m in net} - {"torch.nn.modules.linear.Linear"}
        )
        return widths, others

    pi, pi_activations = describe(policy.mlp_extractor.policy_net)
    vf, vf_activations = describe(policy.mlp_extractor.value_net)
    head = policy.action_net
    return {
        "pi": pi,
        "vf": vf,
        "activation_fn": _qualified(policy.activation_fn),
        "module_activations": sorted(set(pi_activations) | set(vf_activations)),
        "features_extractor": _qualified(type(policy.features_extractor)),
        "features_dim": int(policy.features_dim),
        "n_actions": int(getattr(head, "out_features", -1)),
    }


def _normalize(mean_return: float, env_name: str) -> float:
    ref = REFERENCE_BASELINES[env_name]
    lo, hi = ref["random_score"], ref["expert_score"]
    if abs(hi - lo) < 1e-8:
        return 0.0
    return (mean_return - lo) / (hi - lo)


def train_classical_expert_until_converged(
    env_name: str,
    cache_dir: pathlib.Path,
    rng: np.random.Generator,
    seed: int,
    convergence_override: Optional[Dict[str, float]] = None,
    net_arch: Optional[Sequence[int]] = None,
) -> BasePolicy:
    """Train a PPO MlpPolicy expert with chunked learn + convergence detection.

    Args:
        env_name: Classical gymnasium env ID (must be in ENV_CONFIGS).
        cache_dir: Where to save the converged model.
        rng: Random state.
        seed: PPO seed.
        convergence_override: If provided, fully replaces the per-env config.
            Intended for tests that want to force a non-convergence path.
        net_arch: Hidden widths shared by the actor and critic MLPs; None
            keeps the original `DEFAULT_NET_ARCH`.

    Returns:
        The converged `model.policy`.

    Raises:
        RuntimeError: If the max_timesteps budget is exhausted before
            normalized return passes threshold and self-CE passes
            self_ce_eps for `patience` consecutive chunks.
    """
    env_cfg = env_utils.ENV_CONFIGS[env_name]
    ppo_kwargs = env_cfg.get("ppo_kwargs", {})
    ppo_n_envs = env_cfg.get("ppo_n_envs", None)

    if convergence_override is not None:
        conv = dict(convergence_override)
    else:
        conv = env_utils.get_convergence_config(env_name)

    chunk_size = int(conv["chunk_timesteps"])
    max_total = int(conv["max_timesteps"])
    min_total = int(conv["min_timesteps"])
    threshold = float(conv["threshold"])
    patience = int(conv["patience"])
    self_ce_eps = float(conv["self_ce_eps"])
    tolerance = 0.02  # plateau tolerance on normalized return

    if ppo_n_envs and ppo_n_envs > 1:
        train_venv = env_utils.make_env(env_name, n_envs=ppo_n_envs, rng=rng)
    else:
        train_venv = env_utils.make_env(env_name, n_envs=1, rng=rng)
    eval_venv = env_utils.make_env(env_name, n_envs=1, rng=rng)

    model = PPO(
        "MlpPolicy",
        train_venv,
        policy_kwargs=dict(
            net_arch=list(DEFAULT_NET_ARCH if net_arch is None else net_arch),
        ),
        seed=seed,
        verbose=0,
        # Last, so no per-env config can move this MLP off the CPU: SB3's
        # "auto" would pick CUDA whenever it is available.
        **{**ppo_kwargs, "device": "cpu"},
    )

    best_norm = -float("inf")
    best_self_ce = float("inf")
    chunks_since_best = 0
    total_steps = 0
    last_norm = -float("inf")
    last_self_ce = float("inf")
    # Window of recent normalized returns. We require EVERY entry in the
    # window to pass `threshold` before converging — this guards against
    # a single lucky noisy chunk false-converging at the boundary.
    return_window: "collections.deque[float]" = collections.deque(maxlen=patience)

    try:
        while total_steps < max_total:
            model.learn(chunk_size, reset_num_timesteps=False)
            total_steps += chunk_size

            # 50 episodes (vs 20 default) tightens convergence stderr by
            # ~sqrt(2.5). PPO chunk time dominates eval time, so ~negligible.
            eval_res = eval_policy_rollout(
                model.policy,
                eval_venv,
                n_episodes=50,
                deterministic=True,
                expert_policy=model.policy,
            )
            norm_ret = _normalize(eval_res.mean_return, env_name)
            self_ce = float(eval_res.current_round_ce)
            last_norm, last_self_ce = norm_ret, self_ce
            return_window.append(norm_ret)

            improved = (
                norm_ret > best_norm + tolerance or self_ce < best_self_ce - tolerance
            )
            if improved:
                best_norm = max(best_norm, norm_ret)
                best_self_ce = min(best_self_ce, self_ce)
                chunks_since_best = 0
            else:
                chunks_since_best += 1

            window_min = min(return_window) if return_window else -float("inf")
            logger.info(
                f"[{env_name}] step={total_steps}/{max_total} "
                f"norm_ret={norm_ret:.3f} self_ce={self_ce:.3f} "
                f"(best norm={best_norm:.3f} ce={best_self_ce:.3f}, "
                f"window_min={window_min:.3f}, "
                f"patience {chunks_since_best}/{patience})"
            )

            # Strict convergence: require EVERY chunk in the patience window
            # to clear the return threshold, not just the current one. This
            # closes the gap where noisy PPO returns oscillate across the
            # threshold and a lucky final chunk triggers early termination.
            if (
                total_steps >= min_total
                and len(return_window) >= patience
                and window_min >= threshold
                and self_ce <= self_ce_eps
                and chunks_since_best >= patience
            ):
                break
        else:
            raise RuntimeError(
                f"Expert for {env_name} failed to converge in {max_total} "
                f"steps: norm_return={last_norm:.3f} (threshold={threshold}), "
                f"self_ce={last_self_ce:.3f} (eps={self_ce_eps})"
            )
    finally:
        train_venv.close()
        eval_venv.close()

    cache_path = pathlib.Path(cache_dir) / env_name.replace("/", "_")
    cache_path.mkdir(parents=True, exist_ok=True)
    model.save(cache_path / "model.zip")
    logger.info(f"Saved converged expert to {cache_path / 'model.zip'}")
    return model.policy
