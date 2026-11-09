"""Learner-only observation restrictions for the approved CartPole pilot.

A restriction ``M`` is a fixed, parameter-free, dimension-preserving map on the
learner's observation. The linear learner of the previous pipeline clones the
expert, freezes everything except ``action_net`` and reinitializes that head;
here ``M`` is applied before the frozen features, so the learner computes
``head(phi_E(M(x)))`` on every path (training loss, collection, prediction,
evaluation and cross-entropy). The expert is a separate object and keeps
reading the full observation.

Three restrictions exist: ``identity``, ``cart_position_zero`` (CartPole
``x := 0``; the historical x-only pilot, kept so its checkpoints still load)
and ``cart_position_angular_velocity_zero`` (``x := 0`` and
``theta_dot := 0``; ``x_dot`` and ``theta`` unchanged).

`audit_cartpole_position` checks, on simulator-valid paired states that differ
only in cart position, whether the expert gives different labels to states
that the mask makes identical. That proves only that conflicting expert labels
exist under the mask, never a positive on-policy error floor.
"""

import dataclasses
import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from imitation.experiments.ftrl import policy_utils


@dataclasses.dataclass(frozen=True)
class Restriction:
    """A fixed coordinate-zeroing map on the learner's observation."""

    restriction_id: str
    zeroed_indices: Tuple[int, ...]
    obs_shape: Optional[Tuple[int, ...]]
    description: str

    def apply(self, obs: np.ndarray) -> np.ndarray:
        """Return a masked copy of ``obs`` (last axis is the feature axis)."""
        masked = np.array(obs, copy=True)
        if self.zeroed_indices:
            masked[..., list(self.zeroed_indices)] = 0
        return masked


RESTRICTIONS: Dict[str, Restriction] = {
    "identity": Restriction(
        restriction_id="identity",
        zeroed_indices=(),
        obs_shape=None,
        description="Full observation, unchanged.",
    ),
    "cart_position_zero": Restriction(
        restriction_id="cart_position_zero",
        zeroed_indices=(0,),
        obs_shape=(4,),
        description=(
            "CartPole cart position x := 0; x_dot, theta and theta_dot are kept."
        ),
    ),
    "cart_position_angular_velocity_zero": Restriction(
        restriction_id="cart_position_angular_velocity_zero",
        zeroed_indices=(0, 3),
        obs_shape=(4,),
        description=(
            "CartPole cart position x := 0 and pole angular velocity "
            "theta_dot := 0; x_dot and theta are kept."
        ),
    ),
}


def get_restriction(restriction_id: str) -> Restriction:
    """Look up a registered restriction.

    Raises:
        ValueError: If ``restriction_id`` is not registered.
    """
    try:
        return RESTRICTIONS[restriction_id]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown restriction {restriction_id!r}; "
            f"expected one of {sorted(RESTRICTIONS)}",
        ) from None


class ObservationMask(th.nn.Module):
    """Parameter-free torch form of a `Restriction`. Draws no random numbers."""

    def __init__(self, restriction_id: str):
        super().__init__()
        self.restriction = get_restriction(restriction_id)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        indices = self.restriction.zeroed_indices
        if not indices:
            return obs
        masked = obs.clone()
        masked[..., list(indices)] = 0
        return masked

    def extra_repr(self) -> str:
        return f"restriction_id={self.restriction.restriction_id!r}"


class MaskedFeaturesExtractor(BaseFeaturesExtractor):
    """Apply an `ObservationMask` before an existing features extractor."""

    def __init__(self, extractor: BaseFeaturesExtractor, restriction_id: str):
        super().__init__(extractor._observation_space, extractor.features_dim)
        self.mask = ObservationMask(restriction_id)
        self.extractor = extractor

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.extractor(self.mask(observations))


# State-dict prefixes of the (possibly aliased) extractors of an SB3 policy.
_EXTRACTOR_PREFIXES = (
    "features_extractor.",
    "pi_features_extractor.",
    "vf_features_extractor.",
)


def _restricted_key(name: str) -> str:
    for prefix in _EXTRACTOR_PREFIXES:
        if name.startswith(prefix):
            return prefix + "extractor." + name[len(prefix) :]
    return name


class RestrictedActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy whose every features extractor is masked.

    SB3 routes ``forward``, ``predict``, ``evaluate_actions``,
    ``get_distribution`` and ``predict_values`` through the features
    extractors, so wrapping them (including the ``pi_``/``vf_`` aliases)
    restricts every path. ``restriction_id``, ``freeze_features`` and
    ``share_features_extractor`` are saved as constructor data, so
    `load_policy_checkpoint` rebuilds the same mask and frozen layers. The
    plain `policy_utils.load_policy_checkpoint` rejects these checkpoints with
    a ``TypeError`` instead of silently dropping the mask.
    """

    def __init__(
        self,
        *args: Any,
        restriction_id: str = "identity",
        freeze_features: bool = False,
        **kwargs: Any,
    ):
        restriction = get_restriction(restriction_id)
        self.restriction_id = restriction.restriction_id
        self.freeze_features = bool(freeze_features)
        super().__init__(*args, **kwargs)
        shape = tuple(self.observation_space.shape or ())
        if restriction.obs_shape is not None and shape != restriction.obs_shape:
            raise ValueError(
                f"Restriction {restriction_id!r} needs observation shape "
                f"{restriction.obs_shape}, got {shape}",
            )
        if self.freeze_features:
            policy_utils.freeze_feature_layers(self)

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        return MaskedFeaturesExtractor(
            super().make_features_extractor(),
            self.restriction_id,
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            share_features_extractor=self.share_features_extractor,
            restriction_id=self.restriction_id,
            freeze_features=self.freeze_features,
        )
        return data


def make_linear_policy(
    expert: ActorCriticPolicy,
    restriction_id: str,
) -> RestrictedActorCriticPolicy:
    """Restricted counterpart of `policy_utils.create_linear_policy`.

    Rebuilds the expert's architecture with masked extractors, copies every
    expert weight, freezes everything except ``action_net`` and reinitializes
    the head exactly as `policy_utils.create_linear_policy` does. Construction
    runs inside a forked torch RNG, so the only draws are the head
    initialization, identical for every restriction under the same seed. The
    expert object is not modified.

    Args:
        expert: Trained expert policy (a plain ``ActorCriticPolicy``).
        restriction_id: A key of `RESTRICTIONS`.

    Returns:
        The restricted linear learner.

    Raises:
        ValueError: For an unknown restriction, an incompatible observation
            space, or an expert that is itself restricted.
    """
    restriction = get_restriction(restriction_id)
    if isinstance(expert, RestrictedActorCriticPolicy):
        raise ValueError("The expert must be an unrestricted policy")
    params = expert._get_constructor_parameters()
    params["share_features_extractor"] = expert.share_features_extractor
    with th.random.fork_rng(devices=[]):
        policy = RestrictedActorCriticPolicy(
            **params,
            restriction_id=restriction.restriction_id,
            freeze_features=True,
        )
    policy.load_state_dict(
        {_restricted_key(k): v for k, v in expert.state_dict().items()},
    )
    policy = policy.to(expert.device)
    policy.train(expert.training)
    policy_utils.reinitialize_action_net(policy)
    return policy


def load_policy_checkpoint(path, device: str = "cpu") -> RestrictedActorCriticPolicy:
    """Load a checkpoint written by ``RestrictedActorCriticPolicy.save``.

    Args:
        path: Checkpoint path (trusted experiment artifact).
        device: Torch device for the loaded policy.

    Returns:
        The policy with its mask, frozen layers and weights restored.

    Raises:
        ValueError: If the checkpoint has no restriction metadata.
    """
    saved = th.load(path, map_location=device, weights_only=False)
    if "restriction_id" not in saved["data"]:
        raise ValueError(f"{path} is not a restricted policy checkpoint")
    policy = RestrictedActorCriticPolicy(**saved["data"])
    policy.load_state_dict(saved["state_dict"])
    return policy.to(device)


# ---------------------------------------------------------------------------
# CartPole paired-state mask audit
# ---------------------------------------------------------------------------

AUDIT_ENV = "CartPole-v1"
# Masks the fixed x-grid audit applies to: both hide cart position, so every
# pair of grid states on one base collides under them.
AUDIT_RESTRICTIONS: Tuple[str, ...] = (
    "cart_position_zero",
    "cart_position_angular_velocity_zero",
)
# Predeclared nonterminal cart positions, well inside |x| < 2.4.
AUDIT_X_GRID: Tuple[float, ...] = (-1.8, -0.6, 0.6, 1.8)
# Base states: the diagnostic expert trajectory at these time steps.
AUDIT_BASE_STEPS: Tuple[int, ...] = tuple(range(0, 500, 25))
AUDIT_CLAIM = (
    "A conflicting pair has identical masked observations and different "
    "expert labels on the full state, so no deterministic policy of the "
    "masked observation matches the expert on both states. This proves only "
    "that conflicting expert labels exist under the mask. It does not prove "
    "a positive error under the expert's or the learner's on-policy state "
    "distribution, and no classifier-based Bayes error is claimed."
)
AUDIT_WITNESS_SCOPE = (
    "Every checked pair differs only in cart position x and shares x_dot, "
    "theta and theta_dot, so it collides under any mask that zeroes x. Under "
    "a mask that also zeroes theta_dot these are the same pairs; no pair that "
    "differs in angular velocity is checked, so the audit says nothing about "
    "conflicts that hiding angular velocity adds."
)


def _expert_action(expert, obs: np.ndarray) -> int:
    action, _ = expert.predict(obs[None].astype(np.float32), deterministic=True)
    return int(np.asarray(action).reshape(-1)[0])


def audit_cartpole_position(
    expert,
    reset_seed: int,
    *,
    restriction_id: str,
    grid: Sequence[float] = AUDIT_X_GRID,
    base_steps: Sequence[int] = AUDIT_BASE_STEPS,
    check: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Paired-state x-grid audit of a cart-position mask on the CartPole simulator.

    Base states come from one deterministic expert trajectory started with
    ``env.reset(seed=reset_seed)``; a base is used only if the trajectory
    reaches that step without terminating. For every base and every grid
    position, the simulator state ``(x, x_dot, theta, theta_dot)`` is set
    directly, and the observation is CartPole's own map of that state (the
    float32 state, verified against ``step`` output). A state is valid if it
    is inside the env's own nonterminal thresholds and the observation map
    check holds. A pair of positions on one base is simulator consistent if
    one physics step under each action moves both states identically except
    for the position offset. The expert labels full observations only; the
    labels are diagnostic and never used for training. States, physics checks
    and labels do not depend on the mask; only the masked observations and
    the collision test do.

    Args:
        expert: Object with ``predict(obs, deterministic=True)``.
        reset_seed: Diagnostic reset seed, disjoint from the pilot seed.
        restriction_id: One of `AUDIT_RESTRICTIONS`.
        grid: Cart positions substituted into every base.
        base_steps: Trajectory steps used as base states.
        check: Optional callable run before every simulator or expert call
            (for example a deadline guard).

    Returns:
        A JSON-serializable record with every checked state and pair, and
        ``status`` ``"conflict_found"`` or ``"restriction_uncertified"``.

    Raises:
        ValueError: If the restriction does not hide cart position.
    """
    check = check or (lambda what: None)
    restriction = get_restriction(restriction_id)
    if restriction.restriction_id not in AUDIT_RESTRICTIONS:
        raise ValueError(
            f"The x-grid audit needs a mask that hides cart position, one of "
            f"{AUDIT_RESTRICTIONS}; got {restriction_id!r}",
        )
    mask = ObservationMask(restriction.restriction_id)
    grid = [float(x) for x in grid]
    wanted = sorted({int(s) for s in base_steps})
    expert_calls = 0

    env = gym.make(AUDIT_ENV)
    sim = env.unwrapped
    x_threshold = float(sim.x_threshold)
    theta_threshold = float(sim.theta_threshold_radians)

    check("diagnostic reset")
    obs, _ = env.reset(seed=int(reset_seed))
    bases: List[Dict[str, Any]] = []
    step = 0
    while True:
        if step in wanted:
            bases.append(
                {
                    "base": len(bases),
                    "step": step,
                    "full_state": [float(v) for v in obs],
                    "simulator_state": [float(v) for v in sim.state],
                },
            )
        if step >= wanted[-1]:
            break
        check("diagnostic expert query")
        action = _expert_action(expert, obs)
        expert_calls += 1
        check("diagnostic env step")
        obs, _, terminated, truncated, _ = env.step(action)
        step += 1
        if terminated or truncated:
            break
    env.close()
    used = {b["step"] for b in bases}

    probe = gym.make(AUDIT_ENV).unwrapped
    probe.reset(seed=int(reset_seed))
    states: List[Dict[str, Any]] = []
    next_states: Dict[Tuple[int, float], Dict[int, np.ndarray]] = {}
    for base in bases:
        _, x_dot, theta, theta_dot = base["simulator_state"]
        for x in grid:
            state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
            observation = np.array(state, dtype=np.float32)
            nonterminal = abs(x) < x_threshold and abs(theta) < theta_threshold
            map_ok = True
            successors = {}
            for action in (0, 1):
                check("audit simulator step")
                probe.state = state.copy()
                probe.steps_beyond_terminated = None
                stepped_obs, _, _, _, _ = probe.step(action)
                successor = np.array(probe.state, dtype=np.float64)
                map_ok &= bool(
                    np.array_equal(stepped_obs, np.array(successor, dtype=np.float32))
                )
                successors[action] = successor
            next_states[(base["base"], x)] = successors
            check("audit expert query")
            label = _expert_action(expert, observation)
            expert_calls += 1
            with th.no_grad():
                masked = mask(th.as_tensor(observation[None]))[0].numpy()
            if not np.array_equal(masked, restriction.apply(observation)):
                raise RuntimeError("Torch and numpy masks disagree")
            states.append(
                {
                    "base": base["base"],
                    "x": x,
                    "simulator_state": [float(v) for v in state],
                    "observation": [float(v) for v in observation],
                    "masked_observation": [float(v) for v in masked],
                    "expert_action": label,
                    "nonterminal": bool(nonterminal),
                    "observation_map_verified": bool(map_ok),
                    "simulator_valid": bool(nonterminal and map_ok),
                },
            )
    probe.close()

    by_key = {(s["base"], s["x"]): s for s in states}
    pairs = []
    for base in bases:
        for x_a, x_b in itertools.combinations(grid, 2):
            a = by_key[(base["base"], x_a)]
            b = by_key[(base["base"], x_b)]
            na = next_states[(base["base"], x_a)]
            nb = next_states[(base["base"], x_b)]
            consistent = all(
                np.array_equal(na[act][1:], nb[act][1:])
                and np.isclose(na[act][0] - nb[act][0], x_a - x_b, rtol=0, atol=1e-9)
                for act in (0, 1)
            )
            masked_equal = a["masked_observation"] == b["masked_observation"]
            valid = a["simulator_valid"] and b["simulator_valid"]
            pairs.append(
                {
                    "base": base["base"],
                    "x_a": x_a,
                    "x_b": x_b,
                    "expert_action_a": a["expert_action"],
                    "expert_action_b": b["expert_action"],
                    "masked_equal": bool(masked_equal),
                    "simulator_consistent": bool(consistent),
                    "both_valid": bool(valid),
                    "conflict": bool(
                        valid
                        and consistent
                        and masked_equal
                        and a["expert_action"] != b["expert_action"]
                    ),
                },
            )
    n_conflicts = sum(p["conflict"] for p in pairs)
    return {
        "status": "conflict_found" if n_conflicts else "restriction_uncertified",
        "claim": AUDIT_CLAIM,
        "witness_scope": AUDIT_WITNESS_SCOPE,
        "env_name": AUDIT_ENV,
        "restriction_id": restriction.restriction_id,
        "zeroed_indices": list(restriction.zeroed_indices),
        "reset_seed": int(reset_seed),
        "grid": grid,
        "requested_base_steps": wanted,
        "unavailable_base_steps": [s for s in wanted if s not in used],
        "thresholds": {"x": x_threshold, "theta_radians": theta_threshold},
        "base_states": bases,
        "states": states,
        "pairs": pairs,
        "n_checked_pairs": len(pairs),
        "n_invalid_pairs": sum(
            not (p["both_valid"] and p["simulator_consistent"] and p["masked_equal"])
            for p in pairs
        ),
        "n_conflicting_pairs": int(n_conflicts),
        "expert_predict_calls": expert_calls,
        "labels_used_for_training": False,
    }
