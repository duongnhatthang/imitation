"""Immutable adapters for the eight classical restriction tasks.

A `Task` declares what the restriction runner needs to know about one of the
original classical environments: its effective episode cap, the layout of the
float observation that ``env_utils.make_env`` emits, its action count, and the
fixed ladder of learner restrictions predeclared for it. Nothing here changes
the environment: `describe` builds the original factory and refuses if the
declared values disagree with it, and records the registry kwargs, configured
kwargs, raw and effective caps, wrapper chain and, for tabular tasks, a digest
of the transition table, so records bind the exact environment they used.

Categorical restrictions are fixed maps from a state index to a representative
state index, then re-encoded (`alias_table`). The classes of a rung are given
by the visible factors; the representative of a class is its smallest-index
reachable nonterminal state, and a class without such a state keeps identity
because the learner is never queried there. On reachable nonterminal inputs
the map is therefore idempotent, keeps every visible factor, collapses the
hidden ones and always outputs a valid encoding. Reachability is a
breadth-first search over ``env.P`` from the reset support for FrozenLake,
CliffWalking and Taxi, and an exact enumeration of the draw process for
Blackjack (`reachable_nonterminal`). It means reachable under some action
sequence, not under any particular policy. Each table is pinned by
`FROZEN_TABLE_SHA256`, so a changed environment or rule is refused rather
than silently producing a different mask.

`TASKS` holds exactly the eight original tasks and their ladders. Variant
tasks (`VARIANT_TASKS`: ``FrozenLakeVariedStart-v0`` and the standard
stochastic ``FrozenLakeSlippery-v0``) are separate, so iterating `TASKS`
never picks them up; `get_task` and `describe` accept both, and a variant's
description also pins its declared reset distribution.
`CALIBRATION_CANDIDATES` lists opt-in replacement restrictions per original
environment. They are alternatives with no severity order and are never part
of an original ladder, task description or task digest. Grid candidates are
regular row and column bins (`_grid_bins`), which generalize hiding a factor
(one bin over the whole axis).

Random state binning (`RANDOM_ALIAS_SPECS`, listed per environment in
`RANDOM_ALIAS_CANDIDATES` and appended to `CALIBRATION_CANDIDATES`) is a
second kind of categorical candidate. Its classes ignore the factors: the
sorted reachable nonterminal states are shuffled once by a local
``np.random.RandomState(seed)`` and split by ``np.array_split`` into exactly
``K`` balanced nonempty bins (`random_state_partition`); the representative
of a bin is its first shuffled member, and every other state keeps identity.
The partition depends on nothing but the reachable set, ``K`` and the seed:
no data, expert action or reward. One permutation serves every ``K`` of an
environment; ``array_split`` bins of one ``K`` refine those of another only
when the chunk boundaries happen to line up (37 states: ``K=4`` splits at
10, 19, 28 and refines ``K=2``, which splits at 19), so refinement is not
guaranteed and the counts are alternatives. Each candidate is pinned by a static table
digest (`FROZEN_TABLE_SHA256`) and a static reachable-set digest
(`FROZEN_REACHABLE_SHA256`), and `random_partition_descriptor` rebuilds its
descriptor without building an environment.

Random-offset grids (`RANDOM_GRID_SPECS`, listed per environment in
`RANDOM_GRID_CANDIDATES`) are the continuous counterpart for CartPole,
Acrobot, MountainCar and LunarLander: one seed, the per-axis counts
`RANDOM_GRID_COUNTS`, and the map itself in `random_grid`. They are appended
to `CALIBRATION_CANDIDATES` after every earlier candidate and do not change
`describe`: their declared Box bounds are checked by `random_grid`, only for
grid ids.
"""

import collections
import dataclasses
import functools
import hashlib
import json
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from imitation.experiments.ftrl import env_utils


class TaskMismatch(ValueError):
    """The environment factory disagrees with the declared task."""


@dataclasses.dataclass(frozen=True)
class Task:
    """Declared, immutable facts about one classical restriction task."""

    env_name: str
    episode_cap: int
    # "continuous" (named float features) or "one_hot" (concatenated blocks).
    observation: str
    feature_names: Tuple[str, ...]
    # One-hot block sizes; empty for continuous observations.
    blocks: Tuple[int, ...]
    n_actions: int
    # Predeclared learner restrictions, screened in this order.
    ladder: Tuple[str, ...]
    # "nested": each rung hides more; "alternatives": no severity order.
    ladder_rule: str

    @property
    def obs_dim(self) -> int:
        return int(sum(self.blocks)) if self.blocks else len(self.feature_names)

    @property
    def restriction_ids(self) -> Tuple[str, ...]:
        """Restrictions a run of this task may use: the full view and the ladder."""
        return ("identity",) + self.ladder


TASKS: Dict[str, Task] = {
    task.env_name: task
    for task in (
        Task(
            env_name="CartPole-v1",
            episode_cap=500,
            observation="continuous",
            feature_names=("x", "x_dot", "theta", "theta_dot"),
            blocks=(),
            n_actions=2,
            ladder=(
                "cart_position_angular_velocity_zero",
                "cart_position_velocity_angular_velocity_zero",
            ),
            ladder_rule="nested",
        ),
        Task(
            env_name="FrozenLake-v1",
            episode_cap=100,
            observation="one_hot",
            feature_names=("state",),
            blocks=(16,),
            n_actions=4,
            ladder=(
                "frozenlake_column_pairs",
                "frozenlake_column_hidden",
                "frozenlake_column_hidden_row_pairs",
            ),
            ladder_rule="nested",
        ),
        Task(
            env_name="CliffWalking-v0",
            episode_cap=200,
            observation="one_hot",
            feature_names=("state",),
            blocks=(48,),
            n_actions=4,
            ladder=(
                "cliffwalking_column_pairs",
                "cliffwalking_column_hidden",
                "cliffwalking_column_hidden_row_pairs",
            ),
            ladder_rule="nested",
        ),
        Task(
            env_name="Acrobot-v1",
            episode_cap=500,
            observation="continuous",
            feature_names=(
                "cos_theta1",
                "sin_theta1",
                "cos_theta2",
                "sin_theta2",
                "theta1_dot",
                "theta2_dot",
            ),
            blocks=(),
            n_actions=3,
            ladder=(
                "acrobot_theta2_dot_zero",
                "acrobot_theta_dots_zero",
                "acrobot_theta2_and_dots_zero",
            ),
            ladder_rule="nested",
        ),
        Task(
            env_name="MountainCar-v0",
            episode_cap=200,
            observation="continuous",
            feature_names=("position", "velocity"),
            blocks=(),
            n_actions=3,
            ladder=("mountaincar_velocity_zero", "mountaincar_position_zero"),
            ladder_rule="alternatives",
        ),
        Task(
            env_name="Taxi-v3",
            episode_cap=200,
            observation="one_hot",
            feature_names=("state",),
            blocks=(500,),
            n_actions=6,
            ladder=("taxi_destination_hidden", "taxi_destination_passenger_hidden"),
            ladder_rule="nested",
        ),
        Task(
            env_name="Blackjack-v1",
            episode_cap=20,
            observation="one_hot",
            feature_names=("player_sum", "dealer_card", "usable_ace"),
            blocks=(32, 11, 2),
            n_actions=2,
            ladder=(
                "blackjack_usable_ace_hidden",
                "blackjack_usable_ace_dealer_hidden",
            ),
            ladder_rule="nested",
        ),
        Task(
            env_name="LunarLander-v2",
            episode_cap=1000,
            observation="continuous",
            feature_names=(
                "x",
                "y",
                "vx",
                "vy",
                "angle",
                "angular_velocity",
                "left_leg_contact",
                "right_leg_contact",
            ),
            blocks=(),
            n_actions=4,
            ladder=(
                "lunarlander_angular_velocity_zero",
                "lunarlander_velocities_zero",
                "lunarlander_velocities_angle_zero",
            ),
            ladder_rule="nested",
        ),
    )
}


# Variant tasks: declared like `TASKS` but never iterated with them.
VARIANT_TASKS: Dict[str, Task] = {
    task.env_name: task
    for task in (
        Task(
            env_name=env_utils.FROZENLAKE_VARIED_START,
            episode_cap=100,
            observation="one_hot",
            feature_names=("state",),
            blocks=(16,),
            n_actions=4,
            ladder=(
                "frozenlake_varied_start_column_pairs",
                "frozenlake_varied_start_column_hidden",
                "frozenlake_varied_start_column_hidden_row_pairs",
            ),
            ladder_rule="nested",
        ),
        Task(
            env_name=env_utils.FROZENLAKE_SLIPPERY,
            episode_cap=100,
            observation="one_hot",
            feature_names=("state",),
            blocks=(16,),
            n_actions=4,
            ladder=(
                "frozenlake_slippery_column_pairs",
                "frozenlake_slippery_column_hidden",
                "frozenlake_slippery_column_hidden_row_pairs",
            ),
            ladder_rule="nested",
        ),
    )
}

# Declared reset support of the variant tasks; the reset is uniform over it.
DECLARED_RESET_SUPPORT: Dict[str, Tuple[int, ...]] = {
    env_utils.FROZENLAKE_VARIED_START: (0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14),
    env_utils.FROZENLAKE_SLIPPERY: (0,),
}
_RESET_RULES: Dict[str, str] = {
    env_utils.FROZENLAKE_VARIED_START: "uniform over the safe nonterminal states",
    env_utils.FROZENLAKE_SLIPPERY: (
        "fixed start state 0 (the map's S cell), as in FrozenLake-v1"
    ),
}

# Opt-in replacement restrictions per original environment, not on any ladder.
CALIBRATION_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "MountainCar-v0": (
        "mountaincar_bins_pos005_vel005",
        "mountaincar_bins_pos005_vel015",
        "mountaincar_bins_pos010_vel005",
        "mountaincar_bins_pos010_vel015",
    ),
    "Blackjack-v1": (
        "blackjack_player_sum_bins2_dealer_ace_hidden",
        "blackjack_player_sum_bins3_dealer_ace_hidden",
        "blackjack_player_sum_bins4_dealer_ace_hidden",
        # Wider bins, predeclared together before any selection run.
        "blackjack_player_sum_bins6_dealer_ace_hidden",
        "blackjack_player_sum_bins8_dealer_ace_hidden",
        "blackjack_player_sum_bins10_dealer_ace_hidden",
    ),
    # Regular grid bins; cliffwalking_column_pairs is already on the ladder.
    "CliffWalking-v0": (
        "cliffwalking_row_pairs",
        "cliffwalking_column_triples",
    ),
}

# Random state binning (`RandomAliasSpec`): one predeclared partition seed and
# exact bin counts per environment, frozen before any calibration run. Counts
# of one environment are alternatives: refinement across counts is not
# guaranteed, so they do not form a nested ladder.
RANDOM_BINNING_SEED = 20261003
RANDOM_BINNING_COUNTS: Dict[str, Tuple[int, ...]] = {
    "CliffWalking-v0": (32, 28, 24, 20, 16, 12, 8, 4),
    env_utils.FROZENLAKE_SLIPPERY: (9, 8, 7, 6, 5, 4, 3, 2),
    "Taxi-v3": (360, 320, 280, 240, 200, 160, 120, 80, 40, 20, 10),
    "Blackjack-v1": (252, 224, 196, 168, 140, 112, 84, 56, 28, 14, 7),
}
_RANDOM_ID_PREFIX: Dict[str, str] = {
    "CliffWalking-v0": "cliffwalking",
    env_utils.FROZENLAKE_SLIPPERY: "frozenlake_slippery",
    "Taxi-v3": "taxi",
    "Blackjack-v1": "blackjack",
}


_UINT32_MAX = 2**32 - 1


def _strict_int(value: Any, what: str) -> int:
    """``value`` as an int; refuses bools, floats and every non-integer type."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    return int(value)


def random_binning_id(env_name: str, n_bins: int, seed: int) -> str:
    """Restriction id of a random binning, e.g. ``taxi_random_bins40_seed20261003``.

    The id names the exact bin count and seed, so neither is coerced. It may
    name an unregistered bin count; `get_restriction` refuses such ids.

    Raises:
        ValueError: If ``env_name`` has no random binning, ``n_bins`` is not a
            positive integer or ``seed`` is not an integer in ``0..2**32 - 1``
            (bools and floats are refused).
    """
    if not isinstance(env_name, str) or env_name not in _RANDOM_ID_PREFIX:
        raise ValueError(f"{env_name!r} has no random binning")
    n_bins = _strict_int(n_bins, "n_bins")
    if n_bins < 1:
        raise ValueError(f"n_bins {n_bins} is not positive")
    seed = _strict_int(seed, "seed")
    if not 0 <= seed <= _UINT32_MAX:
        raise ValueError(f"seed {seed} is outside 0..{_UINT32_MAX}")
    return f"{_RANDOM_ID_PREFIX[env_name]}_random_bins{n_bins}_seed{seed}"


# Random binning candidates per environment, kept apart from the strategic
# candidates so a controller can select only these. They are appended to
# `CALIBRATION_CANDIDATES`, after every strategic candidate.
RANDOM_ALIAS_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    env_name: tuple(random_binning_id(env_name, k, RANDOM_BINNING_SEED) for k in ks)
    for env_name, ks in RANDOM_BINNING_COUNTS.items()
}
CALIBRATION_CANDIDATES.update(
    {
        env_name: CALIBRATION_CANDIDATES.get(env_name, ()) + ids
        for env_name, ids in RANDOM_ALIAS_CANDIDATES.items()
    },
)


# Random-offset grids (`RandomGridSpec`, built by `random_grid`) on the four
# continuous original tasks: one predeclared seed and one shared per-axis bin
# count ladder, frozen before any calibration run. They are appended to
# `CALIBRATION_CANDIDATES` (opt-in), and never touch a ladder, a task
# description or a task digest.
RANDOM_GRID_SEED = 20261003
RANDOM_GRID_COUNTS: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256)
# Version pin of `random_grid`; change it with any change of rule.
RANDOM_GRID_GENERATOR = "random_offset_grid/1"
_GRID_ID_PREFIX: Dict[str, str] = {
    "CartPole-v1": "cartpole",
    "Acrobot-v1": "acrobot",
    "MountainCar-v0": "mountaincar",
    "LunarLander-v2": "lunarlander",
}


def random_grid_id(env_name: str, n_bins: int, seed: int) -> str:
    """Restriction id of a random-offset grid.

    For example ``acrobot_random_grid8_seed20261003``.

    ``n_bins`` is the per-axis count ``K``. Like `random_binning_id`, the id may
    name an unregistered count or seed; `get_restriction` refuses such ids.

    Raises:
        ValueError: If ``env_name`` has no random grid, ``n_bins`` is not a
            positive integer or ``seed`` is not an integer in ``0..2**32 - 1``
            (bools and floats are refused).
    """
    if not isinstance(env_name, str) or env_name not in _GRID_ID_PREFIX:
        raise ValueError(f"{env_name!r} has no random grid")
    n_bins = _strict_int(n_bins, "n_bins")
    if n_bins < 1:
        raise ValueError(f"n_bins {n_bins} is not positive")
    seed = _strict_int(seed, "seed")
    if not 0 <= seed <= _UINT32_MAX:
        raise ValueError(f"seed {seed} is outside 0..{_UINT32_MAX}")
    return f"{_GRID_ID_PREFIX[env_name]}_random_grid{n_bins}_seed{seed}"


@dataclasses.dataclass(frozen=True)
class RandomGridSpec:
    """A random-offset grid of a continuous task's declared observation Box."""

    restriction_id: str
    env_name: str
    n_bins: int
    seed: int
    generator: str = RANDOM_GRID_GENERATOR


RANDOM_GRID_SPECS: Dict[str, RandomGridSpec] = {
    random_grid_id(env_name, k, RANDOM_GRID_SEED): RandomGridSpec(
        restriction_id=random_grid_id(env_name, k, RANDOM_GRID_SEED),
        env_name=env_name,
        n_bins=k,
        seed=RANDOM_GRID_SEED,
    )
    for env_name in _GRID_ID_PREFIX
    for k in RANDOM_GRID_COUNTS
}
# Grid candidates per environment, in ladder order (alternatives, not nested).
RANDOM_GRID_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    env_name: tuple(
        random_grid_id(env_name, k, RANDOM_GRID_SEED) for k in RANDOM_GRID_COUNTS
    )
    for env_name in _GRID_ID_PREFIX
}
# Opt-in like every candidate: appended after the strategic and random binning
# candidates, so they run only with ``--calibration-candidates``.
CALIBRATION_CANDIDATES.update(
    {
        env_name: CALIBRATION_CANDIDATES.get(env_name, ()) + ids
        for env_name, ids in RANDOM_GRID_CANDIDATES.items()
    },
)


def calibration_candidates(env_name: str) -> Tuple[str, ...]:
    """Opt-in replacement restrictions of an environment (alternatives)."""
    return CALIBRATION_CANDIDATES.get(env_name, ())


def get_task(env_name: str) -> Task:
    """Look up a declared task (original or variant).

    Raises:
        ValueError: If ``env_name`` is not a declared task.
    """
    try:
        return TASKS[env_name]
    except (KeyError, TypeError):
        pass
    try:
        return VARIANT_TASKS[env_name]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown task {env_name!r}; expected one of "
            f"{sorted(TASKS) + sorted(VARIANT_TASKS)}",
        ) from None


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Effective episode cap and environment description
# ---------------------------------------------------------------------------


def _wrapper_chain(env: gym.Env) -> List[gym.Env]:
    chain = [env]
    while isinstance(chain[-1], gym.Wrapper):
        chain.append(chain[-1].env)
    return chain


def venv_episode_cap(venv) -> int:
    """The one TimeLimit cap of a single-env DummyVecEnv, wrappers included.

    Raises:
        TaskMismatch: Unless there is exactly one environment with exactly one
            TimeLimit, and the env's spec reports the same cap.
    """
    base = venv
    while hasattr(base, "venv"):
        base = base.venv
    envs = getattr(base, "envs", None)
    if envs is None or len(envs) != 1:
        raise TaskMismatch("Expected a single-environment DummyVecEnv")
    limits = [
        e._max_episode_steps
        for e in _wrapper_chain(envs[0])
        if isinstance(e, gym.wrappers.TimeLimit)
    ]
    if len(limits) != 1:
        raise TaskMismatch(f"Expected exactly one TimeLimit, found caps {limits}")
    cap = int(limits[0])
    spec = envs[0].spec
    if spec is None or spec.max_episode_steps != cap:
        raise TaskMismatch(
            f"Wrapped spec cap {getattr(spec, 'max_episode_steps', None)} differs "
            f"from the TimeLimit cap {cap}",
        )
    return cap


def effective_episode_cap(env_name: str) -> int:
    """Episode cap of the original factory ``env_utils.make_env``.

    The raw registered cap can be None (CliffWalking, Blackjack); the factory
    then applies its configured fallback TimeLimit, and that is the cap
    episodes actually have.
    """
    venv = env_utils.make_env(env_name, 1, np.random.default_rng(0))
    try:
        return venv_episode_cap(venv)
    finally:
        venv.close()


def _tabular_dynamics_sha256(unwrapped) -> Optional[str]:
    table = getattr(unwrapped, "P", None)
    if table is None:
        return None
    rows = {
        str(s): {
            str(a): [[float(p), int(n), float(r), bool(d)] for p, n, r, d in outcomes]
            for a, outcomes in sorted(actions.items())
        }
        for s, actions in sorted(table.items())
    }
    return _digest(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


@functools.lru_cache(maxsize=None)
def _describe(env_name: str) -> str:
    task = get_task(env_name)
    config = env_utils.ENV_CONFIGS[env_name]
    spec = gym.spec(env_name)
    venv = env_utils.make_env(env_name, 1, np.random.default_rng(0))
    try:
        env = venv.envs[0]
        cap = venv_episode_cap(venv)
        obs_space, act_space = venv.observation_space, venv.action_space
        unwrapped = env.unwrapped
        flags = {
            name: _jsonable(getattr(unwrapped, name))
            for name in ("sab", "natural", "desc", "shape", "nrow", "ncol")
            if hasattr(unwrapped, name)
        }
        description = {
            "env_name": env_name,
            "entry_point": str(spec.entry_point),
            "registered_kwargs": _jsonable(dict(spec.kwargs)),
            "configured_env_kwargs": _jsonable(dict(config.get("env_kwargs", {}))),
            "registered_max_episode_steps": spec.max_episode_steps,
            "episode_cap": cap,
            "episode_cap_rule": (
                "registered TimeLimit"
                if spec.max_episode_steps is not None
                else "env_utils fallback TimeLimit (fallback_max_episode_steps)"
            ),
            "wrappers": [type(e).__name__ for e in _wrapper_chain(env)],
            "observation_space": {
                "type": type(obs_space).__name__,
                "shape": list(obs_space.shape),
                "dtype": str(obs_space.dtype),
            },
            "action_space": {"type": type(act_space).__name__, "n": int(act_space.n)},
            "unwrapped_attributes": flags,
            "tabular_dynamics_sha256": _tabular_dynamics_sha256(unwrapped),
            "declared": {
                "observation": task.observation,
                "feature_names": list(task.feature_names),
                "blocks": list(task.blocks),
                "ladder": list(task.ladder),
                "ladder_rule": task.ladder_rule,
            },
        }
        reset = getattr(unwrapped, "initial_state_distrib", None)
        if reset is not None:
            reset = [float(p) for p in reset]
    finally:
        venv.close()
    problems = []
    if env_name in DECLARED_RESET_SUPPORT:
        support = DECLARED_RESET_SUPPORT[env_name]
        declared = [
            1.0 / len(support) if s in support else 0.0 for s in range(task.obs_dim)
        ]
        if reset != declared:
            problems.append(f"reset distribution {reset} != declared {declared}")
        description["reset_distribution"] = {
            "support": list(support),
            "probabilities": declared,
            "rule": _RESET_RULES[env_name],
        }
    if cap != task.episode_cap:
        problems.append(f"episode cap {cap} != declared {task.episode_cap}")
    if not isinstance(obs_space, gym.spaces.Box) or obs_space.dtype != np.float32:
        problems.append(f"observation space {obs_space} is not a float32 Box")
    elif tuple(obs_space.shape) != (task.obs_dim,):
        problems.append(f"observation shape {obs_space.shape} != ({task.obs_dim},)")
    if not isinstance(act_space, gym.spaces.Discrete) or (
        int(act_space.n) != task.n_actions
    ):
        problems.append(f"action space {act_space} != Discrete({task.n_actions})")
    if problems:
        raise TaskMismatch(f"{env_name}: {'; '.join(problems)}")
    return json.dumps(description, sort_keys=True)


def describe(task: Task) -> Dict[str, Any]:
    """Verified description of the task's original environment factory.

    Raises:
        TaskMismatch: If the factory disagrees with the declared task.
    """
    declared = TASKS.get(task.env_name) or VARIANT_TASKS.get(task.env_name)
    if declared != task:
        raise TaskMismatch(f"{task.env_name}: not the registered task declaration")
    return json.loads(_describe(task.env_name))


def task_sha256(task: Task) -> str:
    return _digest(describe(task))


# ---------------------------------------------------------------------------
# Reachable nonterminal states
# ---------------------------------------------------------------------------

_TABULAR_ENVS = (
    "FrozenLake-v1",
    "CliffWalking-v0",
    "Taxi-v3",
    env_utils.FROZENLAKE_VARIED_START,
    env_utils.FROZENLAKE_SLIPPERY,
)
BLACKJACK_SIZES = (32, 11, 2)


def _raw_env(env_name: str) -> gym.Env:
    kwargs = dict(env_utils.ENV_CONFIGS[env_name].get("env_kwargs", {}))
    return gym.make(env_name, **kwargs)


def _tabular_reachable(env_name: str) -> Tuple[List[int], Dict[str, Any]]:
    env = _raw_env(env_name)
    try:
        unwrapped = env.unwrapped
        table = unwrapped.P
        starts = [int(s) for s in np.flatnonzero(unwrapped.initial_state_distrib > 0)]
        dynamics = _tabular_dynamics_sha256(unwrapped)
    finally:
        env.close()
    seen = set(starts)
    queue = collections.deque(starts)
    terminal_flags: Dict[int, set] = collections.defaultdict(set)
    while queue:
        s = queue.popleft()
        for outcomes in table[s].values():
            for p, nxt, _, done in outcomes:
                if p <= 0:
                    continue
                nxt = int(nxt)
                terminal_flags[nxt].add(bool(done))
                if not done and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    mixed = sorted(s for s, flags in terminal_flags.items() if len(flags) > 1)
    if mixed:
        raise RuntimeError(f"{env_name}: termination is not a function of {mixed}")
    if any(True in terminal_flags.get(s, ()) for s in starts):
        raise RuntimeError(f"{env_name}: a reset state is also terminal")
    terminal = sorted(s for s, flags in terminal_flags.items() if flags == {True})
    evidence = {
        "type": "breadth-first search over env.P from the reset support",
        "reset_support": starts,
        "terminal_states_reached": terminal,
        "tabular_dynamics_sha256": dynamics,
        "limits": "reachable under some action sequence from a reset state; "
        "not a statement about any policy's state distribution",
    }
    return sorted(seen), evidence


def blackjack_index(player_sum: int, dealer_card: int, usable_ace: int) -> int:
    """Joint index of a Blackjack observation, usable ace varying fastest."""
    return (int(player_sum) * 11 + int(dealer_card)) * 2 + int(usable_ace)


def blackjack_factors(index: int) -> Tuple[int, int, int]:
    rest, ace = divmod(int(index), 2)
    player_sum, dealer = divmod(rest, 11)
    return player_sum, dealer, ace


def _blackjack_reachable() -> Tuple[List[int], Dict[str, Any]]:
    from gymnasium.envs.toy_text import blackjack as bj

    env = _raw_env("Blackjack-v1")
    try:
        flags = {"sab": bool(env.unwrapped.sab), "natural": bool(env.unwrapped.natural)}
    finally:
        env.close()
    cards = sorted(set(int(c) for c in bj.deck))

    def hand(raw: int, has_ace: bool) -> List[int]:
        # Same total and ace content as any real hand of this class.
        return [1, raw - 1] if has_ace else [raw]

    # A hand is summarized by (sum counting aces as 1, holds an ace): the
    # observation depends on nothing else, and drawing adds one card.
    start = {
        (a + b, a == 1 or b == 1) for a in cards for b in cards
    }  # two-card deals, never terminal at reset
    seen = set(start)
    queue = collections.deque(sorted(start))
    while queue:
        raw, has_ace = queue.popleft()
        for card in cards:
            nxt = (raw + card, has_ace or card == 1)
            if bj.is_bust(hand(*nxt)) or nxt in seen:
                continue  # a bust ends the episode; its observation is terminal
            seen.add(nxt)
            queue.append(nxt)
    player = sorted(
        {(bj.sum_hand(hand(*h)), int(bj.usable_ace(hand(*h)))) for h in seen}
    )
    states = sorted(
        blackjack_index(s, dealer, ace) for s, ace in player for dealer in cards
    )
    evidence = {
        "type": "exact enumeration of the draw process",
        "deck": [int(c) for c in bj.deck],
        "unwrapped_flags": flags,
        "rule": "reset deals two cards to the player and shows one dealer card "
        "from the deck with replacement and never terminates; hit draws one "
        "card and terminates only on a bust; stick always terminates",
        "limits": "reachable under some action sequence; sab and natural change "
        "payouts only, not which pre-action observations occur",
    }
    return states, evidence


@functools.lru_cache(maxsize=None)
def _reachable(env_name: str) -> str:
    if env_name in _TABULAR_ENVS:
        states, evidence = _tabular_reachable(env_name)
    elif env_name == "Blackjack-v1":
        states, evidence = _blackjack_reachable()
    else:
        raise ValueError(f"{env_name} has no enumerable state set")
    return json.dumps({"states": states, "evidence": evidence}, sort_keys=True)


def reachable_nonterminal(env_name: str) -> Tuple[List[int], Dict[str, Any]]:
    """Sorted reachable nonterminal state indices and their evidence record."""
    data = json.loads(_reachable(env_name))
    return data["states"], data["evidence"]


# ---------------------------------------------------------------------------
# Categorical alias tables
# ---------------------------------------------------------------------------


def _grid_key(n_cols: int, rule: str) -> Callable[[int], Hashable]:
    def key(s: int) -> Hashable:
        r, c = divmod(int(s), n_cols)
        if rule == "column_pairs":
            return (r, c // 2)
        if rule == "column_hidden":
            return (r,)
        return (r // 2,)  # column hidden, rows merged in pairs

    return key


def _grid_bins(
    n_cols: int, row_width: int, col_width: int
) -> Callable[[int], Hashable]:
    """Regular bins: rows in blocks of ``row_width``, columns of ``col_width``."""

    def key(s: int) -> Hashable:
        r, c = divmod(int(s), n_cols)
        return (r // row_width, c // col_width)

    return key


def _blackjack_player_bins(width: int) -> Callable[[int], Hashable]:
    """Player sums below 12 form one class; from 12 up, bins of ``width``."""

    def key(s: int) -> Hashable:
        player_sum = blackjack_factors(s)[0]
        return ("low",) if player_sum < 12 else ("high", (player_sum - 12) // width)

    return key


def _taxi_factors(s: int) -> Tuple[int, int, int, int]:
    rest, dest = divmod(int(s), 4)
    rest, passenger = divmod(rest, 5)
    row, col = divmod(rest, 5)
    return row, col, passenger, dest


@dataclasses.dataclass(frozen=True)
class AliasRung:
    restriction_id: str
    env_name: str
    hidden: str
    key: Callable[[int], Hashable]


_ALIAS_RUNGS = (
    AliasRung(
        "frozenlake_column_pairs",
        "FrozenLake-v1",
        "column within pairs {0,1}, {2,3}",
        _grid_key(4, "column_pairs"),
    ),
    AliasRung(
        "frozenlake_column_hidden",
        "FrozenLake-v1",
        "column",
        _grid_key(4, "column_hidden"),
    ),
    AliasRung(
        "frozenlake_column_hidden_row_pairs",
        "FrozenLake-v1",
        "column, and row within pairs {0,1}, {2,3}",
        _grid_key(4, "row_pairs"),
    ),
    AliasRung(
        "cliffwalking_column_pairs",
        "CliffWalking-v0",
        "column within pairs {2k, 2k+1}",
        _grid_key(12, "column_pairs"),
    ),
    AliasRung(
        "cliffwalking_column_hidden",
        "CliffWalking-v0",
        "column",
        _grid_key(12, "column_hidden"),
    ),
    AliasRung(
        "cliffwalking_column_hidden_row_pairs",
        "CliffWalking-v0",
        "column, and row within pairs {0,1}, {2,3}",
        _grid_key(12, "row_pairs"),
    ),
    AliasRung(
        "taxi_destination_hidden",
        "Taxi-v3",
        "destination",
        lambda s: _taxi_factors(s)[:3],
    ),
    AliasRung(
        "taxi_destination_passenger_hidden",
        "Taxi-v3",
        "destination and passenger location (including in taxi)",
        lambda s: _taxi_factors(s)[:2],
    ),
    AliasRung(
        "blackjack_usable_ace_hidden",
        "Blackjack-v1",
        "usable ace",
        lambda s: blackjack_factors(s)[:2],
    ),
    AliasRung(
        "blackjack_usable_ace_dealer_hidden",
        "Blackjack-v1",
        "usable ace and dealer card",
        lambda s: blackjack_factors(s)[:1],
    ),
    # Calibration candidates (`CALIBRATION_CANDIDATES`), not on a ladder.
    *(
        AliasRung(
            f"blackjack_player_sum_bins{width}_dealer_ace_hidden",
            "Blackjack-v1",
            f"usable ace, dealer card, and player sum except the class: sums "
            f"below 12 are one class, from 12 up (sum - 12) // {width}",
            _blackjack_player_bins(width),
        )
        for width in (2, 3, 4, 6, 8, 10)
    ),
    AliasRung(
        "cliffwalking_row_pairs",
        "CliffWalking-v0",
        "row within pairs {0,1}, {2,3}",
        _grid_bins(12, 2, 1),
    ),
    AliasRung(
        "cliffwalking_column_triples",
        "CliffWalking-v0",
        "column within triples {3k, 3k+1, 3k+2}",
        _grid_bins(12, 1, 3),
    ),
    # FrozenLakeVariedStart-v0 ladder: the FrozenLake keys on the new reset.
    AliasRung(
        "frozenlake_varied_start_column_pairs",
        env_utils.FROZENLAKE_VARIED_START,
        "column within pairs {0,1}, {2,3}",
        _grid_key(4, "column_pairs"),
    ),
    AliasRung(
        "frozenlake_varied_start_column_hidden",
        env_utils.FROZENLAKE_VARIED_START,
        "column",
        _grid_key(4, "column_hidden"),
    ),
    AliasRung(
        "frozenlake_varied_start_column_hidden_row_pairs",
        env_utils.FROZENLAKE_VARIED_START,
        "column, and row within pairs {0,1}, {2,3}",
        _grid_key(4, "row_pairs"),
    ),
    # FrozenLakeSlippery-v0 ladder: the FrozenLake keys on the slippery lake.
    AliasRung(
        "frozenlake_slippery_column_pairs",
        env_utils.FROZENLAKE_SLIPPERY,
        "column within pairs {0,1}, {2,3}",
        _grid_key(4, "column_pairs"),
    ),
    AliasRung(
        "frozenlake_slippery_column_hidden",
        env_utils.FROZENLAKE_SLIPPERY,
        "column",
        _grid_key(4, "column_hidden"),
    ),
    AliasRung(
        "frozenlake_slippery_column_hidden_row_pairs",
        env_utils.FROZENLAKE_SLIPPERY,
        "column, and row within pairs {0,1}, {2,3}",
        _grid_key(4, "row_pairs"),
    ),
)
ALIAS_RUNGS: Dict[str, AliasRung] = {r.restriction_id: r for r in _ALIAS_RUNGS}

# Version pin of `random_state_partition`; change it with any change of rule.
RANDOM_BINNING_GENERATOR = "random_state_binning/1"
RANDOM_BINNING_RULE = (
    "the sorted unique reachable nonterminal states are permuted by a local "
    "numpy.random.RandomState(seed).permutation (legacy MT19937 stream) and "
    "split in permuted order by numpy.array_split into n_bins balanced "
    "nonempty bins; each bin maps to its first permuted member and every "
    "other state keeps identity; partitions of different n_bins share the "
    "permutation; refinement is not guaranteed across bin counts"
)


@dataclasses.dataclass(frozen=True)
class RandomAliasSpec:
    """A random state binning of an environment's reachable nonterminal states."""

    restriction_id: str
    env_name: str
    n_bins: int
    seed: int
    generator: str = RANDOM_BINNING_GENERATOR

    @property
    def hidden(self) -> str:
        return (
            f"every factor: {self.n_bins} balanced random bins of the reachable "
            f"nonterminal states (partition seed {self.seed}), each seen as its "
            f"first shuffled member"
        )


RANDOM_ALIAS_SPECS: Dict[str, RandomAliasSpec] = {
    random_binning_id(env_name, k, RANDOM_BINNING_SEED): RandomAliasSpec(
        restriction_id=random_binning_id(env_name, k, RANDOM_BINNING_SEED),
        env_name=env_name,
        n_bins=k,
        seed=RANDOM_BINNING_SEED,
    )
    for env_name, ks in RANDOM_BINNING_COUNTS.items()
    for k in ks
}

# Digests of the frozen tables (`table_sha256`), fixed before any screening.
FROZEN_TABLE_SHA256: Dict[str, str] = {
    "frozenlake_column_pairs": (
        "ceeaea50bb590eb134d3a282573a8fbc2b08e9a9dd27b5d558d5f44a802f3b10"
    ),
    "frozenlake_column_hidden": (
        "e14eafe001fea5acee0c66e01bcd5ed223461994ba130de9c2e8c02164c6f1ea"
    ),
    "frozenlake_column_hidden_row_pairs": (
        "3cb6422a91a104ef9de2eda9c7ac4bd60391459523d293f3b02db247415c9cdb"
    ),
    "cliffwalking_column_pairs": (
        "5edeb6077203af829b57f44912fe12af786fba1e774667fd5b0fa3e3f69cc639"
    ),
    "cliffwalking_column_hidden": (
        "741b118c6dd6c254a5f6a065b07d6cb8fe7bea2e309115ceafe6035767334cb4"
    ),
    "cliffwalking_column_hidden_row_pairs": (
        "56af4e8f488cf5bd0571df4e8b17c127d020c2eb304ab7a661e6c07b5c141bfe"
    ),
    "taxi_destination_hidden": (
        "949469d8309e8871b1c427d86ec1ddc63a0e5739483227c46fba4465f1046c1a"
    ),
    "taxi_destination_passenger_hidden": (
        "a225e30160f5710a3886249405154138e788f955fd0e03ee7761fa4dec22e632"
    ),
    "blackjack_usable_ace_hidden": (
        "da0225fe2e322c7c17650ee6064ade7039266077cba075cb3d306f28a544a4bb"
    ),
    "blackjack_usable_ace_dealer_hidden": (
        "ee81f71455823860f5f7be9db0029000fc04ff68373e465f617a3688120af3d7"
    ),
    # Calibration candidates, frozen before any calibration run.
    "blackjack_player_sum_bins2_dealer_ace_hidden": (
        "c66418ab0fbaa47ddaa4a0ecf847284c7b20e36ad9b083ae8e28241d1bcc7d0f"
    ),
    "blackjack_player_sum_bins3_dealer_ace_hidden": (
        "5bb2b8d23c22c33522c2680a294b29da2eb45141519e4ed3999e93b3a7d81128"
    ),
    "blackjack_player_sum_bins4_dealer_ace_hidden": (
        "b4e5577555d4c25425d1d33900e6726290ae7547c88e75b81a6504f613036c11"
    ),
    # FrozenLakeVariedStart-v0: every safe state is reachable from the
    # original reset too, so the tables equal the FrozenLake-v1 ones.
    "frozenlake_varied_start_column_pairs": (
        "ceeaea50bb590eb134d3a282573a8fbc2b08e9a9dd27b5d558d5f44a802f3b10"
    ),
    "frozenlake_varied_start_column_hidden": (
        "e14eafe001fea5acee0c66e01bcd5ed223461994ba130de9c2e8c02164c6f1ea"
    ),
    "frozenlake_varied_start_column_hidden_row_pairs": (
        "3cb6422a91a104ef9de2eda9c7ac4bd60391459523d293f3b02db247415c9cdb"
    ),
    # Wider Blackjack bins, frozen together before any selection run.
    "blackjack_player_sum_bins6_dealer_ace_hidden": (
        "d112edab4ca43f67e204df5fded9dc6a410a87abc3f00a1641639118d309f34a"
    ),
    "blackjack_player_sum_bins8_dealer_ace_hidden": (
        "8579086117518528767d40ba45abc9ffeb382589c5bbb798ead6d49421b2f210"
    ),
    "blackjack_player_sum_bins10_dealer_ace_hidden": (
        "43ea38ffd3d5043b1d035b675565a32e45308ed99053874d49c6a5273f863322"
    ),
    # CliffWalking regular grid bins, frozen before any calibration run.
    "cliffwalking_row_pairs": (
        "37d71edc9db4c5736f59aadd934ebe0d84945ea063bde3607b007f7a06cc3976"
    ),
    "cliffwalking_column_triples": (
        "97c48e710daf86fe5027a06c7cf982f8bbe14d64e5d39bb62319679a059a5d03"
    ),
    # FrozenLakeSlippery-v0: the reachable set from state 0 is the
    # deterministic lake's, so the tables equal the FrozenLake-v1 ones.
    "frozenlake_slippery_column_pairs": (
        "ceeaea50bb590eb134d3a282573a8fbc2b08e9a9dd27b5d558d5f44a802f3b10"
    ),
    "frozenlake_slippery_column_hidden": (
        "e14eafe001fea5acee0c66e01bcd5ed223461994ba130de9c2e8c02164c6f1ea"
    ),
    "frozenlake_slippery_column_hidden_row_pairs": (
        "3cb6422a91a104ef9de2eda9c7ac4bd60391459523d293f3b02db247415c9cdb"
    ),
    # Random state binnings (`RANDOM_ALIAS_SPECS`), derived from the pinned
    # generator and reachable sets (`FROZEN_REACHABLE_SHA256`) before any
    # calibration run.
    "cliffwalking_random_bins32_seed20261003": (
        "72833be47d149afaf7a56ff02608a7eefa07e181f69c6403159a54f18553c184"
    ),
    "cliffwalking_random_bins28_seed20261003": (
        "4a4704347bc453a13db60f3b96487e09a13a473ec6644279cbfeeef2f0b8f027"
    ),
    "cliffwalking_random_bins24_seed20261003": (
        "fa36ac6234e9b89f00448b35f0a3231b5ba4b2acf67bbf43431408558b93eb44"
    ),
    "cliffwalking_random_bins20_seed20261003": (
        "72667ae39f08062c81704bd9719bc4134cca29633f9c89f11b2768d44fadcb71"
    ),
    "cliffwalking_random_bins16_seed20261003": (
        "06fdb2dfe71ce12f79bd7c8b9804cfa30ba0a3ffd1ccc56a8e36e14fd43d03a9"
    ),
    "cliffwalking_random_bins12_seed20261003": (
        "86dba59bcddff0d03757a79213c39b566f20c5b4dfa40ba0e67acdbf985e7700"
    ),
    "cliffwalking_random_bins8_seed20261003": (
        "164768a60098c5a763df4a59f91ad51015a25fa6f161343970224992c75fe698"
    ),
    "cliffwalking_random_bins4_seed20261003": (
        "f3884cab7ca2aab5f65754d78d187035050f580e52183ae16721028cc028a9fd"
    ),
    "frozenlake_slippery_random_bins9_seed20261003": (
        "4df0feea2c6ea08f4104e7d30686dd7fbce4ea09a6c65fdeff6e3681d617968f"
    ),
    "frozenlake_slippery_random_bins8_seed20261003": (
        "29639b9826c472d672360c24a2bdced3437ebd9af85e4fc2e94eb094f8506219"
    ),
    "frozenlake_slippery_random_bins7_seed20261003": (
        "d2e20f6e1cd7eb803f9ceea473b38b858ac1f52ff46711e0c7302ce7e1f9fffc"
    ),
    "frozenlake_slippery_random_bins6_seed20261003": (
        "f9510804cb48643709c9c2b5bb7be8830ce581f3174cbf648d9ead959ca17274"
    ),
    "frozenlake_slippery_random_bins5_seed20261003": (
        "9d5b9ead16e3d9844f03debf7bc5904147c456cf40f150fcfdc8366234dbd626"
    ),
    "frozenlake_slippery_random_bins4_seed20261003": (
        "eaf8d4a72bf6c7a50b808358a4bf4aaf6142a138fde1ab44a372735621ebb835"
    ),
    "frozenlake_slippery_random_bins3_seed20261003": (
        "357766baebf80f9479a4001febc02273a0acf1f8094ee1b83410c78e9fe5a8aa"
    ),
    "frozenlake_slippery_random_bins2_seed20261003": (
        "58cf1dfe9e2a705cc3f2f132700bc9e9a0df1514e12adf27d4e48622f0f1baa4"
    ),
    "taxi_random_bins360_seed20261003": (
        "2fbff26d6bdb806c7f32ad823df5a4b6158ac729c88aceeb8d76044115a06d20"
    ),
    "taxi_random_bins320_seed20261003": (
        "51ef6020801b1cc8f115e4c202b51e68244f9ee55260f4fa1d98e443a815873c"
    ),
    "taxi_random_bins280_seed20261003": (
        "94a2fa5d515716f748cc1f546a4f5badb63196ee6914b8d7f1ce44d2053b2838"
    ),
    "taxi_random_bins240_seed20261003": (
        "1ab225c77844ac67cc260b1a57e05fdbf26ba98a41b8a50ce4994740e0bf44d4"
    ),
    "taxi_random_bins200_seed20261003": (
        "cf1ddcdf2d083ae02889ca2ae119c68c72cb21f59edc61d4e5b663d556bc134c"
    ),
    "taxi_random_bins160_seed20261003": (
        "7bd15e21219b5d86a19c1a62b62efaa899a7b8638cee1ea57ada3a23c292e32b"
    ),
    "taxi_random_bins120_seed20261003": (
        "ca0f907d7fe87fb0b1452e32c0208c338a2b9c1d63fc87bc62e74c518f3cf9ef"
    ),
    "taxi_random_bins80_seed20261003": (
        "74715ec72e402c8b6b39d542e45991c03d2686b6e02eb8e70b4510ba2add6aa1"
    ),
    "taxi_random_bins40_seed20261003": (
        "ff212052e179721f87f9c7139141d309f53323fe2281e5c894d10380e0b60162"
    ),
    "taxi_random_bins20_seed20261003": (
        "fd0d2704748f15e29551bb0c5859c1c38a1091f918cdb63b71a22b84446d523f"
    ),
    "taxi_random_bins10_seed20261003": (
        "ea363b805aae1a91955a3880e93bd33014c69f7d5988f9f39b4417483aeb52ba"
    ),
    "blackjack_random_bins252_seed20261003": (
        "be4e7979ca3d738a0bc1220f4be1bfcaef0861b8744965da00cad1393e5fc5aa"
    ),
    "blackjack_random_bins224_seed20261003": (
        "98159592277d22bfe4a7200a80fceedf4acd0062fe12118a6f8021a8706bb4b6"
    ),
    "blackjack_random_bins196_seed20261003": (
        "8efc0c8dcf5b13e11bcfe554fe90a42f325ce932298180ce1579031dbcc34618"
    ),
    "blackjack_random_bins168_seed20261003": (
        "e16cf1601f5177b9cc9271d9b2f3b5ccadc02d6f6fa76d0114a3b77d56da304c"
    ),
    "blackjack_random_bins140_seed20261003": (
        "ae7eb5a92c3dc3d4338ed1e98700a91b0a677dc8d46881e291a0162956504c7a"
    ),
    "blackjack_random_bins112_seed20261003": (
        "71e22cfce62f63e813c39ed9dc73fe8a12c8e7fa9a546102f03792bce1a8022a"
    ),
    "blackjack_random_bins84_seed20261003": (
        "96f52f692fb26a1923cf96530f342a64417b3a237b57decec9f77348eb5d357f"
    ),
    "blackjack_random_bins56_seed20261003": (
        "833b04e6a38598bb8044fa40329c73738109f4ed10c613b223ad710b6a3581e8"
    ),
    "blackjack_random_bins28_seed20261003": (
        "cbbe132d3c8097bb1f42304d00bbdcb6817a18d6a01715a25ff611ba716a1491"
    ),
    "blackjack_random_bins14_seed20261003": (
        "74df0b7d6092bef370f3c701d50d22a64aa6ddc69d4861dcd589027b3583e126"
    ),
    "blackjack_random_bins7_seed20261003": (
        "701a89eec750bf22c7e37ecc021ef683e3cfbcd909b3e5ad6672359fd8222f62"
    ),
}


def table_sha256(blocks: Tuple[int, ...], table: Tuple[int, ...]) -> str:
    return _digest({"blocks": list(blocks), "representative": list(table)})


# ---------------------------------------------------------------------------
# Random state binning
# ---------------------------------------------------------------------------

# Reachable nonterminal sets the random binnings were frozen on: the count, the
# digest of the sorted state list and of the reachability evidence record.
FROZEN_REACHABLE_SHA256: Dict[str, Dict[str, Any]] = {
    "CliffWalking-v0": {
        "n_states": 37,
        "states_sha256": (
            "2dc009c14ef334c834b0f1d3f5c286d209e72fe743d797b1af008acaad7c2b7c"
        ),
        "evidence_sha256": (
            "e0f6025773eaf3853673a5b9a5b14270dd324c2a69301517c016af87d9616635"
        ),
    },
    env_utils.FROZENLAKE_SLIPPERY: {
        "n_states": 11,
        "states_sha256": (
            "bddd533c2f0960bc18030394aebf6b2a4da5b13325fad622a92018077cd32c9d"
        ),
        "evidence_sha256": (
            "66d4fe125bc59d2fe81ef643312c26a3bfc5f7e74e030757511652c026c443aa"
        ),
    },
    "Taxi-v3": {
        "n_states": 400,
        "states_sha256": (
            "df52ebb3e452304deee1b2ebd2f6fc51764eeb8d2f92ed5e639fbe233fee9e0e"
        ),
        "evidence_sha256": (
            "3881a6349e32fd951e38458cea65a6eeaa1fffda53b200524ac3517a2f72dbce"
        ),
    },
    "Blackjack-v1": {
        "n_states": 280,
        "states_sha256": (
            "9f0f4e37f5f6d704f03ebb8eda5ca1f45dc97882956f616861685886557a0b73"
        ),
        "evidence_sha256": (
            "bbb207e3113a284b80cdf24b26637d61d16efd622dd96c958d93e703c1bbd72c"
        ),
    },
}


def random_state_partition(
    states: Any,
    n_bins: int,
    seed: int,
    *,
    n_states: Optional[int] = None,
) -> Tuple[Tuple[int, ...], ...]:
    """Partition of ``states`` into ``n_bins`` balanced random bins.

    The sorted states are permuted by a local ``np.random.RandomState(seed)``
    and split in permuted order by ``np.array_split``; bin sizes differ by at
    most one and the first bin of each is its representative. No global numpy
    or torch random state is read or advanced.

    Args:
        states: Distinct nonnegative integer state indices, in any order.
        n_bins: Number of bins, ``1 <= n_bins <= len(states)``.
        seed: Partition seed, ``0 <= seed < 2**32``.
        n_states: If given, every state must be below it.

    Returns:
        The bins, each a tuple of states in permuted order.

    Raises:
        ValueError: For a non-integer (bool and float included), duplicate,
            negative or out-of-range state, an empty or non 1-D ``states``, or
            ``n_bins``, ``seed`` or ``n_states`` out of range or not integers.
    """
    if isinstance(states, (str, bytes)):
        raise ValueError("states must be a sequence of integers")
    if isinstance(states, np.ndarray) and states.ndim != 1:
        raise ValueError(f"states must be 1-D, got shape {states.shape}")
    values = [_strict_int(s, "state") for s in states]
    if not values:
        raise ValueError("states must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("states must not repeat")
    if min(values) < 0:
        raise ValueError(f"state {min(values)} is negative")
    if n_states is not None:
        n_states = _strict_int(n_states, "n_states")
        if max(values) >= n_states:
            raise ValueError(f"state {max(values)} is not below n_states {n_states}")
    n_bins = _strict_int(n_bins, "n_bins")
    if not 1 <= n_bins <= len(values):
        raise ValueError(f"n_bins {n_bins} is outside 1..{len(values)}")
    seed = _strict_int(seed, "seed")
    if not 0 <= seed <= _UINT32_MAX:
        raise ValueError(f"seed {seed} is outside 0..{_UINT32_MAX}")
    ordered = np.array(sorted(values), dtype=np.int64)
    permuted = np.random.RandomState(seed).permutation(ordered)
    return tuple(
        tuple(int(s) for s in part) for part in np.array_split(permuted, n_bins)
    )


def random_alias_table(
    n_states: int,
    states: Any,
    n_bins: int,
    seed: int,
) -> Tuple[int, ...]:
    """Representative table of `random_state_partition` over ``range(n_states)``.

    Every state of a bin maps to the bin's first member; states outside
    ``states`` keep identity.
    """
    n_states = _strict_int(n_states, "n_states")
    table = list(range(n_states))
    for part in random_state_partition(states, n_bins, seed, n_states=n_states):
        for s in part:
            table[s] = part[0]
    return tuple(table)


def reachable_digests(states: List[int], evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Count and digests of a reachable set, as pinned in `FROZEN_REACHABLE_SHA256`."""
    return {
        "n_states": len(states),
        "states_sha256": _digest([int(s) for s in states]),
        "evidence_sha256": _digest(evidence),
    }


def random_partition_descriptor(restriction_id: str) -> Dict[str, Any]:
    """Static descriptor of a random binning; builds no environment.

    It names the generator, rule, seed, bin count, task blocks, the pinned
    reachable set and the frozen table digest, so a record's ``partition``
    can be validated against the static registry.

    Raises:
        ValueError: If ``restriction_id`` is not a random binning.
    """
    spec = None
    if isinstance(restriction_id, str):
        spec = RANDOM_ALIAS_SPECS.get(restriction_id)
    if spec is None:
        raise ValueError(f"{restriction_id!r} is not a random binning")
    reach = FROZEN_REACHABLE_SHA256[spec.env_name]
    return {
        "kind": "random_state_binning",
        "generator": spec.generator,
        "rule": RANDOM_BINNING_RULE,
        "representative": "first member of each bin in permuted order",
        "restriction_id": spec.restriction_id,
        "env_name": spec.env_name,
        "blocks": list(get_task(spec.env_name).blocks),
        "seed": spec.seed,
        "n_bins": spec.n_bins,
        "n_reachable_nonterminal": reach["n_states"],
        "reachable_nonterminal_sha256": reach["states_sha256"],
        "reachability_evidence_sha256": reach["evidence_sha256"],
        "frozen_table_sha256": FROZEN_TABLE_SHA256[spec.restriction_id],
        # No guaranteed refinement across counts; some pairs nest by chance.
        "nested_across_bin_counts": False,
        "inputs": "reachable set, n_bins and seed only; no data, action or reward",
    }


def random_partition_sha256(restriction_id: str) -> str:
    """Digest of `random_partition_descriptor`."""
    return _digest(random_partition_descriptor(restriction_id))


def _check_encodings(env_name: str) -> None:
    """The joint index must agree with the environment's own encoding."""
    if env_name == "Taxi-v3":
        env = _raw_env(env_name)
        try:
            for s in range(500):
                if env.unwrapped.encode(*_taxi_factors(s)) != s:
                    raise RuntimeError(f"Taxi encoding disagrees at state {s}")
        finally:
            env.close()


@functools.lru_cache(maxsize=None)
def _alias(restriction_id: str) -> str:
    rung = ALIAS_RUNGS[restriction_id]
    task = get_task(rung.env_name)
    _check_encodings(rung.env_name)
    reachable, evidence = reachable_nonterminal(rung.env_name)
    n = int(np.prod(task.blocks))
    members: Dict[Hashable, List[int]] = collections.defaultdict(list)
    for s in range(n):
        members[rung.key(s)].append(s)
    reachable_set = set(reachable)
    table = list(range(n))
    identity_classes = 0
    for states in members.values():
        candidates = [s for s in states if s in reachable_set]
        if not candidates:
            identity_classes += 1
            continue
        for s in states:
            table[s] = min(candidates)
    table = tuple(table)
    digest = table_sha256(task.blocks, table)
    frozen = FROZEN_TABLE_SHA256.get(restriction_id)
    if digest != frozen:
        raise RuntimeError(
            f"{restriction_id}: alias table {digest} differs from the frozen "
            f"{frozen}; the environment or the rule changed",
        )
    return json.dumps(
        {
            "restriction_id": restriction_id,
            "env_name": rung.env_name,
            "blocks": list(task.blocks),
            "hidden": rung.hidden,
            "representative": list(table),
            "table_sha256": digest,
            "frozen_table_sha256": frozen,
            "reachable_nonterminal_states": len(reachable),
            "reachable_classes": len({table[s] for s in reachable}),
            "identity_classes_without_reachable_state": identity_classes,
            "reachability": evidence,
            "rule": "classes are the visible factors; each class maps to its "
            "smallest-index reachable nonterminal state, and a class with none "
            "keeps identity",
        },
        sort_keys=True,
    )


@functools.lru_cache(maxsize=None)
def _random_alias(restriction_id: str) -> str:
    spec = RANDOM_ALIAS_SPECS[restriction_id]
    task = get_task(spec.env_name)
    _check_encodings(spec.env_name)
    reachable, evidence = reachable_nonterminal(spec.env_name)
    reach = reachable_digests(reachable, evidence)
    pinned = FROZEN_REACHABLE_SHA256.get(spec.env_name)
    if reach != pinned:
        raise RuntimeError(
            f"{restriction_id}: reachable set {reach} differs from the frozen "
            f"{pinned}; the environment or the enumeration changed",
        )
    n = int(np.prod(task.blocks))
    groups = random_state_partition(reachable, spec.n_bins, spec.seed, n_states=n)
    table = random_alias_table(n, reachable, spec.n_bins, spec.seed)
    digest = table_sha256(task.blocks, table)
    frozen = FROZEN_TABLE_SHA256.get(restriction_id)
    if digest != frozen:
        raise RuntimeError(
            f"{restriction_id}: alias table {digest} differs from the frozen "
            f"{frozen}; the environment or the rule changed",
        )
    return json.dumps(
        {
            "restriction_id": restriction_id,
            "env_name": spec.env_name,
            "blocks": list(task.blocks),
            "hidden": spec.hidden,
            "representative": list(table),
            "table_sha256": digest,
            "frozen_table_sha256": frozen,
            "reachable_nonterminal_states": len(reachable),
            "reachable_classes": len(groups),
            "identity_classes_without_reachable_state": n - len(reachable),
            "reachability": evidence,
            "rule": RANDOM_BINNING_RULE,
            "partition": random_partition_descriptor(restriction_id),
            "partition_sha256": random_partition_sha256(restriction_id),
            "groups": [list(g) for g in groups],
        },
        sort_keys=True,
    )


def alias_table(restriction_id: str) -> Dict[str, Any]:
    """The frozen representative table of a categorical rung and its record.

    Random binnings (`RANDOM_ALIAS_SPECS`) add their static ``partition``
    descriptor, its digest and the full bin membership ``groups``.

    Raises:
        ValueError: If ``restriction_id`` is not a categorical rung.
        RuntimeError: If the rebuilt table (or, for a random binning, the
            reachable set) differs from the frozen digest.
    """
    if isinstance(restriction_id, str) and restriction_id in RANDOM_ALIAS_SPECS:
        return json.loads(_random_alias(restriction_id))
    if restriction_id not in ALIAS_RUNGS:
        raise ValueError(f"{restriction_id!r} is not a categorical restriction")
    return json.loads(_alias(restriction_id))
