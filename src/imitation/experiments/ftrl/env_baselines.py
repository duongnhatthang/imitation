"""Reference baselines and expert quality validation for FTRL experiments.

Reference baselines serve two purposes:

1. **Expert quality validation**: detect degenerate experts early by checking
   that measured expert return exceeds a threshold between random and known-good
   expert performance.

2. **Normalized return computation**: map raw returns to a 0-1 scale where
   0 = random policy and 1 = expert policy, enabling cross-environment
   comparison.

Adding a new MDP requires adding its reference baseline here -- the tests in
``tests/experiments/test_env_baselines.py`` enforce that every environment in
``ENV_CONFIGS`` and ``ATARI_CONFIGS`` has a corresponding entry.

Random scores come from uniform-random rollouts (100 episodes), except for
the variant tasks, whose random score is the exact expected return of the
uniform-random policy (`finite_horizon_value` on the transition table, the
reset distribution and the 100-step cap). Expert scores come from well-trained
PPO agents or published benchmarks, except ``FrozenLakeSlippery-v0``, whose
expert score is the exact finite-horizon optimum.

Every environment, CliffWalking included, normalizes against its uniform-random
lower reference. The CliffWalking stalled-policy return (-200, -1 per step
until the 200-step cap) is only a diagnostic horizontal benchmark
(`STALL_BENCHMARK_RETURNS`), never a lower normalization reference.
"""

import json
import logging
import pathlib
from fractions import Fraction
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference baselines: {env_name: {"random_score": float, "expert_score": float}}
# ---------------------------------------------------------------------------
REFERENCE_BASELINES: Dict[str, Dict[str, float]] = {
    # ---- Classical MDPs ----
    "CartPole-v1": {"random_score": 22.0, "expert_score": 500.0},
    "FrozenLake-v1": {"random_score": 0.015, "expert_score": 1.0},
    "CliffWalking-v0": {"random_score": -56957.0, "expert_score": -13.0},
    "Acrobot-v1": {"random_score": -499.0, "expert_score": -85.0},
    "MountainCar-v0": {"random_score": -200.0, "expert_score": -110.0},
    "Taxi-v3": {"random_score": -763.0, "expert_score": 7.9},
    "Blackjack-v1": {"random_score": -0.40, "expert_score": -0.06},
    "LunarLander-v2": {"random_score": -176.0, "expert_score": 250.0},
    # ---- Variant tasks ----
    # Exact uniform-random value, float of the rational from
    # finite_horizon_value(P, uniform reset over the 11 safe states, 100, 4).
    # The optimal policy succeeds from every reset state.
    "FrozenLakeVariedStart-v0": {
        "random_score": 0.09037647670287287,
        "expert_score": 1.0,
    },
    # Floats of the exact rationals in REFERENCE_METADATA: uniform-random and
    # optimal finite_horizon_value from state 0 with the intended 1/3 slips
    # (rational_transitions), 100 actions.
    "FrozenLakeSlippery-v0": {
        "random_score": 0.013939795959171435,
        "expert_score": 0.7441902878292673,
    },
    # ---- Atari Tier 1 ----
    "PongNoFrameskip-v4": {"random_score": -20.7, "expert_score": 20.5},
    "BreakoutNoFrameskip-v4": {"random_score": 1.7, "expert_score": 405.7},
    "SpaceInvadersNoFrameskip-v4": {
        "random_score": 148.0,
        "expert_score": 1019.8,
    },
    "BeamRiderNoFrameskip-v4": {"random_score": 363.9, "expert_score": 2835.7},
    "QbertNoFrameskip-v4": {"random_score": 163.9, "expert_score": 15228.3},
    "MsPacmanNoFrameskip-v4": {"random_score": 307.3, "expert_score": 2152.8},
    "EnduroNoFrameskip-v4": {"random_score": 0.0, "expert_score": 986.7},
    "SeaquestNoFrameskip-v4": {"random_score": 68.4, "expert_score": 1518.3},
    # ---- Atari Tier 2 ----
    "FreewayNoFrameskip-v4": {"random_score": 0.0, "expert_score": 33.0},
    "AtlantisNoFrameskip-v4": {
        "random_score": 12850.0,
        "expert_score": 2036749.0,
    },
    "DemonAttackNoFrameskip-v4": {
        "random_score": 152.1,
        "expert_score": 13788.4,
    },
    "CrazyClimberNoFrameskip-v4": {
        "random_score": 10780.5,
        "expert_score": 119344.7,
    },
    # ---- Atari Tier 3 ----
    "AsterixNoFrameskip-v4": {"random_score": 210.0, "expert_score": 3738.5},
    "FrostbiteNoFrameskip-v4": {"random_score": 65.2, "expert_score": 933.6},
    "KangarooNoFrameskip-v4": {"random_score": 52.0, "expert_score": 5325.3},
    "BankHeistNoFrameskip-v4": {"random_score": 14.2, "expert_score": 1213.5},
}

# Minimum fraction of (expert - random) that a measured expert must achieve
# to be considered acceptable.
EXPERT_QUALITY_THRESHOLD: float = 0.80

# Diagnostic horizontal benchmarks at the stalled-policy return (CliffWalking
# only): -1 per step until the 200-step cap. Never a normalization reference.
STALL_BENCHMARK_RETURNS: Dict[str, float] = {"CliffWalking-v0": -200.0}
STALL_BENCHMARK_RULE = (
    "CliffWalking only: diagnostic horizontal benchmark at the stalled-policy "
    "return -200 (-1 per step until the 200-step cap). It is never a lower "
    "normalization reference; normalization uses the uniform-random "
    "random_score."
)

# Provenance of reference scores that are not plain uniform-random rollouts
# and published or PPO expert scores. Read through `reference_metadata`.
REFERENCE_METADATA: Dict[str, Dict[str, Any]] = {
    "CliffWalking-v0": {
        "normalization_reference_type": "uniform_random",
        "random_score_rule": "historic uniform-random score carried over "
        "unchanged; not measured under the current 200-step cap (it lies below "
        "-20000, the worst return of any 200-step episode)",
        "stall_benchmark_return": STALL_BENCHMARK_RETURNS["CliffWalking-v0"],
        "stall_benchmark_rule": STALL_BENCHMARK_RULE,
    },
    "FrozenLakeSlippery-v0": {
        "normalization_reference_type": "uniform_random",
        "random_score_exact": (
            "700012139250010050149443607275333445793063125491190300891/"
            "50216813883093446110686315385661331328818843555712276103168"
        ),
        "expert_score_exact": (
            "127846315164763240597097944751513986598202943912/"
            "171792506910670443678820376588540424234035840667"
        ),
        "random_score_rule": "exact uniform-random finite_horizon_value from "
        "state 0, 100 actions, intended 1/3 slips (rational_transitions)",
        "expert_score_rule": "exact finite-horizon optimum (a nonstationary "
        "policy, the upper benchmark); a stationary expert can fall short of "
        "it and is qualified against the usual threshold",
    },
}


def reference_metadata(env_name: str) -> Dict[str, Any]:
    """How the reference scores of ``env_name`` were obtained.

    Every environment has the uniform-random lower reference; entries in
    `REFERENCE_METADATA` add provenance and diagnostic benchmarks.
    """
    meta = {"normalization_reference_type": "uniform_random"}
    meta.update(REFERENCE_METADATA.get(env_name, {}))
    return meta


def normalization_reference(
    env_name: Optional[str],
    measured_uniform_random_return: float,
) -> Dict[str, Any]:
    """Overrides of the measured lower normalization reference: always empty.

    Every environment, CliffWalking included, keeps its measured uniform
    random return as ``random_return``; `STALL_BENCHMARK_RETURNS` are
    diagnostic benchmarks only.
    """
    del env_name, measured_uniform_random_return
    return {}


def validate_expert_quality(
    env_name: str,
    measured_return: float,
) -> Tuple[bool, str]:
    """Check whether a measured expert return meets quality expectations.

    The expert is considered acceptable if::

        measured_return >= random_score + threshold * (expert_score - random_score)

    For unknown environments (not in ``REFERENCE_BASELINES``), validation
    is skipped and the expert is assumed OK.

    Args:
        env_name: Environment ID.
        measured_return: Mean return measured from the expert policy.

    Returns:
        A ``(is_ok, message)`` tuple. ``is_ok`` is True if the expert passes
        or the environment is unknown; ``message`` explains the outcome.
    """
    if env_name not in REFERENCE_BASELINES:
        return True, f"No reference baseline for {env_name}; skipping validation."

    bl = REFERENCE_BASELINES[env_name]
    random_score = bl["random_score"]
    expert_score = bl["expert_score"]
    threshold_value = random_score + EXPERT_QUALITY_THRESHOLD * (
        expert_score - random_score
    )

    if measured_return >= threshold_value:
        return True, (
            f"{env_name}: expert return {measured_return:.1f} >= "
            f"threshold {threshold_value:.1f} "
            f"({EXPERT_QUALITY_THRESHOLD:.0%} of [{random_score}, {expert_score}])."
        )
    else:
        return False, (
            f"{env_name}: expert return {measured_return:.1f} is below "
            f"threshold {threshold_value:.1f} "
            f"({EXPERT_QUALITY_THRESHOLD:.0%} of [{random_score}, {expert_score}]). "
            f"The expert may be degenerate."
        )


def rational_transitions(
    P: Mapping[int, Mapping[int, Sequence[Tuple[Any, int, Any, bool]]]],
    max_denominator: int = 1000,
) -> Dict[int, Dict[int, list]]:
    """Copy of ``P`` with each float probability as its intended rational.

    Toy-text tables store intended probabilities such as 1/3 as floats, whose
    exact value ``Fraction(x)`` is not 1/3. Each probability becomes the
    closest rational with denominator at most ``max_denominator``, accepted
    only if it rounds back to the same float.

    Raises:
        ValueError: If no such rational rounds back to a stored probability.
    """
    table: Dict[int, Dict[int, list]] = {}
    for s, actions in P.items():
        table[s] = {}
        for a, outcomes in actions.items():
            rows = []
            for p, nxt, r, done in outcomes:
                exact = Fraction(p).limit_denominator(max_denominator)
                if float(exact) != float(p):
                    raise ValueError(
                        f"P[{s}][{a}]: probability {p!r} is no rational with "
                        f"denominator <= {max_denominator}",
                    )
                rows.append((exact, nxt, r, done))
            table[s][a] = rows
    return table


def finite_horizon_value(
    P: Mapping[int, Mapping[int, Sequence[Tuple[Any, int, Any, bool]]]],
    initial_distribution: Sequence[Any],
    horizon: int,
    n_actions: int,
    *,
    optimal: bool = False,
    policy: Optional[Mapping[int, int]] = None,
) -> Fraction:
    """Exact expected return of at most ``horizon`` actions on a tabular MDP.

    Backward induction over ``P[s][a] = [(prob, next, reward, done), ...]``
    (the gymnasium toy-text layout) in exact rational arithmetic: floats are
    converted with ``Fraction(x)``, which is exact. A ``done`` transition ends
    the episode; running out of actions truncates it with no further reward.

    Args:
        P: Transition table.
        initial_distribution: Reset weight indexed by state (exact if given
            as ``Fraction``).
        horizon: Maximum number of actions (the episode cap).
        n_actions: Number of actions.
        optimal: If True, the best action in every state and step; otherwise
            the uniform-random policy.
        policy: If given, the deterministic stationary policy ``policy[s]``
            instead of the uniform-random one.

    Returns:
        ``sum_s initial_distribution[s] * V_horizon(s)`` as a ``Fraction``.

    Raises:
        ValueError: If both ``optimal`` and ``policy`` are given.
    """
    if optimal and policy is not None:
        raise ValueError("Give either optimal=True or a policy, not both")
    states = sorted(P)
    value = {s: Fraction(0) for s in states}
    for _ in range(int(horizon)):
        q = {
            s: [
                sum(
                    Fraction(p) * (Fraction(r) + (0 if done else value[nxt]))
                    for p, nxt, r, done in P[s][a]
                )
                for a in range(n_actions)
            ]
            for s in states
        }
        if policy is not None:
            value = {s: q[s][policy[s]] for s in states}
        else:
            value = {s: max(q[s]) if optimal else sum(q[s]) / n_actions for s in states}
    return sum(
        (Fraction(initial_distribution[s]) * value[s] for s in states),
        Fraction(0),
    )


def compute_random_return(venv: VecEnv, n_episodes: int = 100) -> float:
    """Estimate the mean return of a uniform-random policy.

    Args:
        venv: Vectorized environment (reset before calling).
        n_episodes: Number of episodes to average over.

    Returns:
        Mean episodic return across ``n_episodes`` episodes.
    """
    obs = venv.reset()
    episode_returns: list = []
    running_returns = np.zeros(venv.num_envs)

    while len(episode_returns) < n_episodes:
        actions = np.array([venv.action_space.sample() for _ in range(venv.num_envs)])
        obs, rewards, dones, infos = venv.step(actions)
        running_returns += rewards
        for i, done in enumerate(dones):
            if done:
                episode_returns.append(running_returns[i])
                running_returns[i] = 0.0
                if len(episode_returns) >= n_episodes:
                    break

    return float(np.mean(episode_returns[:n_episodes]))


def compute_baselines(
    expert_policy,
    venv: VecEnv,
    rng: np.random.Generator,
    n_expert_episodes: int = 500,
    n_random_episodes: int = 500,
) -> Dict[str, float]:
    r"""Compute expert return, random return, and expert self-CE baselines.

    ``expert_self_ce`` is
    $-\tfrac{1}{|D|}\sum_s \log \pi^*(\arg\max_a \pi^*(a|s)|s)$
    -- the expert's own negative log-probability at its greedy action,
    averaged over states visited in the expert rollout. Sharp softmax
    -> near zero; diffuse softmax -> positive. Direct gauge of the
    argmax pathology, used as the Expert line on the loss subplot.

    Args:
        expert_policy: A policy with a ``predict(obs, deterministic=True)``
            method (e.g. an SB3 BaseAlgorithm).
        venv: Vectorized environment.
        rng: Random number generator (unused currently, reserved for future).
        n_expert_episodes: Number of expert rollout episodes.
        n_random_episodes: Number of random rollout episodes.

    Returns:
        Dict with keys ``"expert_return"``, ``"random_return"``,
        ``"expert_self_ce"``.
    """
    from imitation.experiments.ftrl.eval_utils import eval_policy_rollout

    expert_result = eval_policy_rollout(
        expert_policy,
        venv,
        n_episodes=n_expert_episodes,
        deterministic=True,
        expert_policy=expert_policy,  # enables self-CE computation
    )

    # Random return
    random_return = compute_random_return(venv, n_episodes=n_random_episodes)

    return {
        "expert_return": float(expert_result.mean_return),
        "random_return": random_return,
        "expert_self_ce": float(expert_result.current_round_ce),
    }


def load_or_compute_baselines(
    env_name: str,
    venv: VecEnv,
    expert_policy,
    cache_dir: pathlib.Path,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Load cached baselines or compute and cache them.

    Baselines are stored at ``{cache_dir}/{env_name}/baselines.json``.

    Args:
        env_name: Environment ID (used for cache path).
        venv: Vectorized environment.
        expert_policy: Policy with ``predict(obs, deterministic=True)``.
        cache_dir: Root cache directory.
        rng: Random number generator.

    Returns:
        Dict with keys ``"expert_return"``, ``"random_return"``, and
        ``"expert_self_ce"``.
    """
    cache_path = cache_dir / env_name / "baselines.json"

    if cache_path.exists():
        logger.info(f"Loading cached baselines from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        if "expert_self_ce" in cached:
            return cached
        logger.info(
            f"Baselines cache at {cache_path} missing expert_self_ce; "
            f"recomputing"
        )

    logger.info(f"Computing baselines for {env_name}...")
    baselines = compute_baselines(expert_policy, venv, rng)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(baselines, f, indent=2)
    logger.info(f"Cached baselines to {cache_path}")

    return baselines
