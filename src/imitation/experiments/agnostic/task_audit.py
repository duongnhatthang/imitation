"""Witness audits of learner restrictions for the classical restriction tasks.

A witness is two states that share a masked encoding and receive different
deterministic expert labels on their full observations. It proves only that
no deterministic policy of the masked observation matches the expert on both
states; it is not an error floor on any on-policy distribution, and low BC
return is not a substitute for it. Every audit reports its evidence type and
limits separately:

``reachable_enumeration`` (FrozenLake, CliffWalking, Taxi, Blackjack,
FrozenLakeVariedStart, FrozenLakeSlippery)
    Every reachable nonterminal state (`tasks.reachable_nonterminal`) is
    encoded with the original observation wrapper, labelled by the expert and
    grouped by its masked encoding. Exact for the enumerated state set. On
    the stochastic FrozenLakeSlippery a state counts as reachable under some
    action sequence and slip outcome.
``constructed_unwrapped_state`` (Acrobot, MountainCar)
    Base states from one expert trajectory on the audit reset seed; one
    simulator coordinate that feeds only hidden features is replaced by
    predeclared grid values. A state is valid if it is inside the simulator's
    bounds, nonterminal, and the observation map agrees with the simulator's
    own step output. Valid states are not proven reachable.
``cartpole_x_grid`` (CartPole)
    `restriction.audit_cartpole_position`, unchanged.
``constructed_box2d_state`` (LunarLander)
    Base states are real post-step Box2D worlds of one expert trajectory on
    the audit reset seed, rebuilt by replaying its actions. Only body
    velocities are edited, by one rigid increment over the lander and both
    legs, so positions, angles and contacts are unchanged. The observation
    comes from the unmodified ``LunarLander.step`` source (its hash is pinned)
    with a zero-length world step, which refreshes contacts but does not
    integrate. Bases are kept only if that refresh reproduces the recorded
    observation bitwise and the lander is airborne (no touching contact, no
    leg contact). A conflict member also has to survive one real step under
    every action. Edited states are valid constructed simulator states, not
    proven reachable, and no error or return bound is claimed.
``unverified``
    Any other environment. No simulator call is made and no agnostic claim
    is supported. Before ``constructed_box2d_state`` this was LunarLander's
    evidence type.
``constructed_grid_cell_state`` (random-offset grids of CartPole, Acrobot,
MountainCar)
    `audit_grid_constructed`. Base states every `CONSTRUCTED_BASE_STRIDE`
    steps of one expert trajectory on the audit reset seed. For every grid
    coordinate independently, the base's own frozen grid cell gives two
    constructed states at the fixed interior fractions `GRID_FRACTIONS`:
    scalar cells use the grid's raw edges intersected with the simulator's
    nonterminal and physical bounds (unbounded tails cut at the documented
    finite `GRID_UNBOUNDED_LIMIT`); Acrobot angles use the frozen arc
    boundaries on the simulator's own angle, so the observation stays a unit
    ``(cos, sin)`` pair. Validity and the observation map check are those of
    ``constructed_unwrapped_state``, every member is assigned to its cell by
    the frozen map itself, and a conflict needs bitwise identical ``M(x)``.

Random-offset grids of LunarLander keep ``constructed_box2d_state``: per
valid base, the two interior fractions of the base's cell on each velocity
axis (``vx``, ``vy``, angular velocity; cells clamped to the declared Box)
become float32 physical velocities applied by the same rigid edit, with every
existing check. Pose, angle and leg contacts are never edited, so position and
angle cells are not searched.

`audit` accepts a task's ladder restrictions and, opt-in, its calibration
candidates (`tasks.CALIBRATION_CANDIDATES`), with the same evidence type, and
its random-offset grids (`tasks.RANDOM_GRID_CANDIDATES`). A grid is frozen
before any label is drawn; the audit only searches witnesses on it.
"""

import copy
import hashlib
import inspect
import itertools
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch as th

from imitation.experiments.agnostic import random_grid, restriction, tasks
from imitation.experiments.ftrl import env_utils

CLAIM = (
    "A conflict is two states with identical masked observations and different "
    "expert labels on their full observations, so no deterministic policy of "
    "the masked observation matches the expert on both. It does not prove a "
    "positive error under the expert's or the learner's on-policy state "
    "distribution, and no return upper bound is claimed."
)

# Simulator coordinates replaced on a base state, each feeding only hidden
# features, with predeclared grid values inside the simulator's bounds.
CONSTRUCTED_GRIDS: Dict[str, Tuple[Tuple[int, str, Tuple[float, ...]], ...]] = {
    "acrobot_theta2_dot_zero": ((3, "theta2_dot", (-8.0, -3.0, 3.0, 8.0)),),
    "acrobot_theta_dots_zero": (
        (2, "theta1_dot", (-4.0, -1.5, 1.5, 4.0)),
        (3, "theta2_dot", (-8.0, -3.0, 3.0, 8.0)),
    ),
    "acrobot_theta2_and_dots_zero": (
        (1, "theta2", (-2.5, -0.8, 0.8, 2.5)),
        (2, "theta1_dot", (-4.0, -1.5, 1.5, 4.0)),
        (3, "theta2_dot", (-8.0, -3.0, 3.0, 8.0)),
    ),
    "mountaincar_velocity_zero": ((1, "velocity", (-0.05, -0.02, 0.02, 0.05)),),
    "mountaincar_position_zero": ((0, "position", (-1.1, -0.8, -0.5, -0.2)),),
    # LunarLander: the index is the observation feature the edited body
    # velocity feeds; values are physical units, each grid maps to the
    # observation values -1.0, -0.4, 0.4 and 1.0. The angle is never edited.
    "lunarlander_angular_velocity_zero": (
        (5, "lander_angular_velocity", (-2.5, -1.0, 1.0, 2.5)),
    ),
    "lunarlander_velocities_zero": (
        (2, "lander_vx", (-5.0, -2.0, 2.0, 5.0)),
        (3, "lander_vy", (-7.5, -3.0, 3.0, 7.5)),
        (5, "lander_angular_velocity", (-2.5, -1.0, 1.0, 2.5)),
    ),
    "lunarlander_velocities_angle_zero": (
        (2, "lander_vx", (-5.0, -2.0, 2.0, 5.0)),
        (3, "lander_vy", (-7.5, -3.0, 3.0, 7.5)),
        (5, "lander_angular_velocity", (-2.5, -1.0, 1.0, 2.5)),
    ),
}
CONSTRUCTED_BASE_STRIDE = 10

# Grids of the opt-in calibration candidates, kept apart from the ladder
# grids above. MountainCar quantized candidates: every grid stays inside one
# bin (the velocity dead zone; the position bin around -0.5 for both steps).
CANDIDATE_GRIDS: Dict[str, Tuple[Tuple[int, str, Tuple[float, ...]], ...]] = {
    f"mountaincar_bins_pos{pos}_vel{vel}": (
        (0, "position", (-0.52, -0.505, -0.495, -0.48)),
        (1, "velocity", tuple(v * zone for v in (-1.0, -0.5, 0.5, 1.0))),
    )
    for pos in ("005", "010")
    for vel, zone in (("005", 0.005), ("015", 0.015))
}


def constructed_grids(
    restriction_id: str,
) -> Tuple[Tuple[int, str, Tuple[float, ...]], ...]:
    """Constructed-state grids of a ladder restriction or calibration candidate."""
    if restriction_id in CONSTRUCTED_GRIDS:
        return CONSTRUCTED_GRIDS[restriction_id]
    return CANDIDATE_GRIDS[restriction_id]


# LunarLander constructed simulator states (`audit_box2d`).
LUNAR_ENV = "LunarLander-v2"
# sha256 of inspect.getsource(LunarLander.step), gymnasium 0.29.1.
LUNAR_STEP_SOURCE_SHA256 = (
    "9aca2d0cb16c6cc8cf5d489d960e5b082be76664ba846105114a430dcc357987"
)
LUNAR_WRAPPERS = ("TimeLimit", "OrderEnforcing", "PassiveEnvChecker")
LUNAR_WORLD_STEP = (1.0 / 50, 6 * 30, 2 * 30)
LUNAR_JOINT_TOLERANCE = 1e-5
LUNAR_CLAIM = (
    "Valid means: only body velocities of a real post-step Box2D world from "
    "the expert's trajectory on the audit reset seed were edited, by one rigid "
    "increment over the lander and both legs; positions, angles and contacts "
    "are unchanged; the observation comes from the unmodified LunarLander.step "
    "source with a zero-length world step, which refreshes contacts but does "
    "not integrate; and one real step under every action executes with finite "
    "output. Not claimed: reachability of edited states, physical plausibility "
    "beyond these checks, or any error or return bound. The unmodified member "
    "of a pair is the expert's own visited state."
)

# Random-offset grids (`random_grid`): constructed states inside frozen cells.
GRID_EVIDENCE = "constructed_grid_cell_state"
GRID_ENVS = ("CartPole-v1", "Acrobot-v1", "MountainCar-v0")
# Fixed interior fractions of a cell, the same for every grid, base and axis.
GRID_FRACTIONS: Tuple[float, ...] = (0.25, 0.75)
# Finite construction range of an unbounded coordinate (CartPole velocities):
# 10 times the grid's arctan scale, in raw units. Tail cells are cut here.
GRID_UNBOUNDED_LIMIT = 10.0 * random_grid.UNBOUNDED_SCALE
# Simulator state index of each Acrobot grid coordinate (angles are pairs).
GRID_STATE_INDEX: Dict[str, Dict[str, int]] = {
    "Acrobot-v1": {"theta1": 0, "theta2": 1, "theta1_dot": 2, "theta2_dot": 3},
}
# LunarLander velocity axes edited by the rigid increment (observation index).
LUNAR_GRID_AXES: Tuple[Tuple[int, str], ...] = (
    (2, "lander_vx"),
    (3, "lander_vy"),
    (5, "lander_angular_velocity"),
)
GRID_CLAIM = (
    "Valid means: a simulator state built from a base state of the expert's "
    "trajectory on the audit reset seed by moving one grid coordinate to a "
    "fixed interior fraction of the base's own frozen grid cell, inside the "
    "simulator's bounds and nonterminal, whose observation map agrees "
    "bitwise with the simulator's step output under every action, and that "
    "the frozen grid assigns to that cell. A conflict needs bitwise identical "
    "grid observations M(x) and different expert labels. Not claimed: "
    "reachability of constructed states, an on-policy error floor or any "
    "return bound. The grid is fixed before any label and never changed; "
    "nearby states are never called aliases."
)


def evidence_type(env_name: str, restriction_id: Optional[str] = None) -> str:
    """Evidence type of the audit of ``env_name`` (of a grid if one is named)."""
    if random_grid.is_grid_id(restriction_id) and env_name in GRID_ENVS:
        return GRID_EVIDENCE
    if env_name in (
        "FrozenLake-v1",
        "CliffWalking-v0",
        "Taxi-v3",
        "Blackjack-v1",
        env_utils.FROZENLAKE_VARIED_START,
        env_utils.FROZENLAKE_SLIPPERY,
    ):
        return "reachable_enumeration"
    if env_name in ("Acrobot-v1", "MountainCar-v0"):
        return "constructed_unwrapped_state"
    if env_name == "CartPole-v1":
        return "cartpole_x_grid"
    if env_name == LUNAR_ENV:
        return "constructed_box2d_state"
    return "unverified"


def _labels(expert, obs: np.ndarray) -> np.ndarray:
    actions, _ = expert.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
    return np.asarray(actions).reshape(-1).astype(np.int64)


def _masked(restriction_id: str, obs: np.ndarray) -> np.ndarray:
    found = restriction.get_restriction(restriction_id)
    masked = found.apply(obs)
    with th.no_grad():
        torch_masked = restriction.ObservationMask(restriction_id)(
            th.as_tensor(np.asarray(obs, dtype=np.float32)),
        ).numpy()
    if not np.array_equal(masked, torch_masked):
        raise RuntimeError("Torch and numpy masks disagree")
    return masked


def _encode_states(env_name: str, states: Sequence[int]) -> np.ndarray:
    """Original wrapper encodings of joint state indices."""
    raw = gym.make(env_name, **env_utils.ENV_CONFIGS[env_name].get("env_kwargs", {}))
    try:
        if env_name == "Blackjack-v1":
            wrapper = env_utils.FlattenTupleObsWrapper(raw)
            return np.stack(
                [wrapper.observation(tasks.blackjack_factors(s)) for s in states],
            )
        wrapper = env_utils.OneHotObsWrapper(raw)
        return np.stack([wrapper.observation(int(s)) for s in states])
    finally:
        raw.close()


def audit_enumerated(expert, restriction_id: str, check) -> Dict[str, Any]:
    found = restriction.get_restriction(restriction_id)
    env_name = found.env_name
    states, evidence = tasks.reachable_nonterminal(env_name)
    check("enumerated audit expert labels")
    obs = _encode_states(env_name, states)
    if not np.array_equal(found.decode(obs), np.asarray(states)):
        raise RuntimeError("Wrapper encodings disagree with the joint index")
    labels = _labels(expert, obs)
    masked = _masked(restriction_id, obs)
    groups: Dict[bytes, List[int]] = {}
    for i, row in enumerate(masked):
        groups.setdefault(row.tobytes(), []).append(i)
    classes = []
    for members in groups.values():
        actions = sorted({int(labels[i]) for i in members})
        classes.append(
            {
                "representative": int(found.decode(masked[members[0]])),
                "states": [int(states[i]) for i in members],
                "expert_actions": [int(labels[i]) for i in members],
                "distinct_actions": actions,
                "conflict": len(actions) > 1,
            },
        )
    classes.sort(key=lambda c: c["representative"])
    n_conflicts = sum(c["conflict"] for c in classes)
    return {
        "status": "conflict_found" if n_conflicts else "no_conflict_on_reachable",
        "evidence": evidence,
        "states": [int(s) for s in states],
        "expert_actions": labels.tolist(),
        "classes": classes,
        "n_states": len(states),
        "n_classes": len(classes),
        "n_conflicting_classes": int(n_conflicts),
        "expert_predict_calls": 1,
        "expert_action_entries": len(states),
        "scope": "every reachable nonterminal state; a class without conflict "
        "has one expert action on all of its reachable states",
    }


def _obs_map(env_name: str, state: np.ndarray) -> np.ndarray:
    if env_name == "Acrobot-v1":
        th1, th2, d1, d2 = state
        # The same numpy functions AcrobotEnv._get_ob uses.
        return np.array(
            [np.cos(th1), np.sin(th1), np.cos(th2), np.sin(th2), d1, d2],
            dtype=np.float32,
        )
    return np.array(state, dtype=np.float32)


def _valid(env_name: str, sim, state: np.ndarray) -> bool:
    if env_name == "Acrobot-v1":
        th1, th2, d1, d2 = state
        inside = (
            -math.pi <= th2 <= math.pi
            and abs(d1) <= sim.MAX_VEL_1
            and abs(d2) <= sim.MAX_VEL_2
        )
        return bool(inside and not (-math.cos(th1) - math.cos(th2 + th1) > 1.0))
    position, velocity = state
    inside = (
        sim.min_position <= position <= sim.max_position
        and abs(velocity) <= sim.max_speed
    )
    goal = position >= sim.goal_position and velocity >= sim.goal_velocity
    return bool(inside and not goal)


def audit_constructed(
    expert,
    restriction_id: str,
    reset_seed: int,
    check,
) -> Dict[str, Any]:
    found = restriction.get_restriction(restriction_id)
    grids = constructed_grids(restriction_id)
    env_name = next(
        name
        for name, task in tasks.TASKS.items()
        if restriction_id in task.ladder + tasks.calibration_candidates(name)
    )
    task = tasks.get_task(env_name)
    expert_calls = 0
    env = gym.make(env_name)
    sim = env.unwrapped
    check("diagnostic reset")
    obs, _ = env.reset(seed=int(reset_seed))
    bases = []
    step = 0
    while True:
        if step % CONSTRUCTED_BASE_STRIDE == 0:
            bases.append(
                {
                    "base": len(bases),
                    "step": step,
                    "simulator_state": [float(v) for v in sim.state],
                },
            )
        check("diagnostic expert query")
        action = int(_labels(expert, obs[None])[0])
        expert_calls += 1
        check("diagnostic env step")
        obs, _, terminated, truncated, _ = env.step(action)
        step += 1
        if terminated or truncated:
            break
    env.close()

    probe = gym.make(env_name).unwrapped
    probe.reset(seed=int(reset_seed))
    states = []
    for base in bases:
        for index, name, grid in grids:
            for value in grid:
                state = np.array(base["simulator_state"], dtype=np.float64)
                state[index] = float(value)
                observation = _obs_map(env_name, state)
                map_ok = True
                for act in range(task.n_actions):
                    check("audit simulator step")
                    probe.state = state.copy()
                    stepped, _, _, _, _ = probe.step(act)
                    after = np.array(probe.state, dtype=np.float64)
                    map_ok &= bool(np.array_equal(stepped, _obs_map(env_name, after)))
                check("audit expert query")
                label = int(_labels(expert, observation[None])[0])
                expert_calls += 1
                masked = _masked(restriction_id, observation[None])[0]
                valid = _valid(env_name, probe, state)
                states.append(
                    {
                        "base": base["base"],
                        "coordinate": name,
                        "value": float(value),
                        "simulator_state": [float(v) for v in state],
                        "observation": [float(v) for v in observation],
                        "masked_observation": [float(v) for v in masked],
                        "expert_action": label,
                        "inside_bounds_nonterminal": valid,
                        "observation_map_verified": map_ok,
                        "simulator_valid": bool(valid and map_ok),
                    },
                )
    probe.close()
    pairs = []
    for (base, name), group in itertools.groupby(
        states,
        key=lambda s: (s["base"], s["coordinate"]),
    ):
        for a, b in itertools.combinations(list(group), 2):
            equal = a["masked_observation"] == b["masked_observation"]
            valid = a["simulator_valid"] and b["simulator_valid"]
            pairs.append(
                {
                    "base": base,
                    "coordinate": name,
                    "value_a": a["value"],
                    "value_b": b["value"],
                    "expert_action_a": a["expert_action"],
                    "expert_action_b": b["expert_action"],
                    "masked_equal": bool(equal),
                    "both_valid": bool(valid),
                    "conflict": bool(
                        valid and equal and a["expert_action"] != b["expert_action"]
                    ),
                },
            )
    n_conflicts = sum(p["conflict"] for p in pairs)
    return {
        "status": "conflict_found" if n_conflicts else "restriction_uncertified",
        "evidence": {
            "type": "constructed through the unwrapped simulator state",
            "limits": "states are inside the simulator's bounds and nonterminal "
            "and their observation map is verified against step output; they "
            "are valid simulator states, not proven reachable, and nearby "
            "states are never called aliases",
        },
        "grids": [[i, n, list(g)] for i, n, g in grids],
        "base_stride": CONSTRUCTED_BASE_STRIDE,
        "base_states": bases,
        "states": states,
        "pairs": pairs,
        "n_checked_pairs": len(pairs),
        "n_invalid_pairs": sum(
            not (p["both_valid"] and p["masked_equal"]) for p in pairs
        ),
        "n_conflicting_pairs": int(n_conflicts),
        "expert_predict_calls": expert_calls,
        # Quantized candidates zero nothing; their map is in "restriction".
        **(
            {"zeroed_indices": list(found.zeroed_indices)}
            if isinstance(found, restriction.Restriction)
            else {}
        ),
    }


def _grid_cell(axis: random_grid.GridAxis, masked: np.ndarray) -> int:
    """Cell of ``axis`` that the frozen map chose, read from its output ``M(x)``.

    Raises:
        RuntimeError: If the output is not exactly one representative.
    """
    if axis.kind == "circular":
        ic, is_ = axis.indices
        hits = [
            j
            for j, (c, s) in enumerate(zip(axis.center_cos, axis.center_sin))
            if c == float(masked[ic]) and s == float(masked[is_])
        ]
    else:
        value = float(masked[axis.indices[0]])
        hits = [j for j, c in enumerate(axis.centers) if c == value]
    if len(hits) != 1:
        raise RuntimeError(f"{axis.name}: grid output is not one representative")
    return hits[0]


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) else None


def _scalar_cell(axis: random_grid.GridAxis, cell: int, bounds) -> Dict[str, Any]:
    """Raw edges of a scalar cell and their intersection with ``bounds``."""
    edges = axis.edges
    left = edges[cell - 1] if cell > 0 else -math.inf
    right = edges[cell] if cell < len(edges) else math.inf
    low, high = max(left, bounds[0]), min(right, bounds[1])
    return {
        "cell": cell,
        "cell_left": _finite_or_none(left),
        "cell_right": _finite_or_none(right),
        "construction_bounds": [float(bounds[0]), float(bounds[1])],
        "low": float(low),
        "high": float(high),
        "empty": not low < high,
    }


def _arc_cell(axis: random_grid.GridAxis, cell: int) -> Dict[str, Any]:
    """Start angle and width of a frozen arc (``atan2`` of its edge vectors)."""
    k = len(axis.edge_cos)
    start = math.atan2(axis.edge_sin[cell], axis.edge_cos[cell])
    end = math.atan2(axis.edge_sin[(cell + 1) % k], axis.edge_cos[(cell + 1) % k])
    width = (end - start) % (2.0 * math.pi)
    return {"cell": cell, "start": start, "width": width, "empty": not width > 0}


def _grid_value(cell: Mapping[str, Any], fraction: float) -> float:
    if "width" in cell:
        angle = cell["start"] + fraction * cell["width"]
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
    return cell["low"] + fraction * (cell["high"] - cell["low"])


def _grid_bounds(env_name: str, sim) -> Dict[str, Tuple[float, float]]:
    """Nonterminal and physical bounds of every scalar grid coordinate."""
    if env_name == "CartPole-v1":
        x, theta = float(sim.x_threshold), float(sim.theta_threshold_radians)
        v = GRID_UNBOUNDED_LIMIT
        return {
            "x": (-x, x),
            "x_dot": (-v, v),
            "theta": (-theta, theta),
            "theta_dot": (-v, v),
        }
    if env_name == "Acrobot-v1":
        v1, v2 = float(sim.MAX_VEL_1), float(sim.MAX_VEL_2)
        return {"theta1_dot": (-v1, v1), "theta2_dot": (-v2, v2)}
    speed = float(sim.max_speed)
    return {
        "position": (float(sim.min_position), float(sim.max_position)),
        "velocity": (-speed, speed),
    }


def _grid_validity(env_name: str, sim, state: np.ndarray) -> List[str]:
    """Reasons a constructed state is outside the bounds or terminal (none: valid).

    Acrobot and MountainCar apply the same tests as `_valid`; CartPole needs
    ``|x| < x_threshold`` and ``|theta| < theta_threshold``.
    """
    reasons = []
    if env_name == "CartPole-v1":
        x, _, theta, _ = state
        if not abs(x) < sim.x_threshold:
            reasons.append("terminal_cart_position")
        if not abs(theta) < sim.theta_threshold_radians:
            reasons.append("terminal_pole_angle")
        return reasons
    if env_name == "Acrobot-v1":
        th1, th2, d1, d2 = state
        if not (-math.pi <= th2 <= math.pi):
            reasons.append("theta2_outside_simulator_range")
        if not (abs(d1) <= sim.MAX_VEL_1 and abs(d2) <= sim.MAX_VEL_2):
            reasons.append("velocity_outside_simulator_bounds")
        if -math.cos(th1) - math.cos(th2 + th1) > 1.0:
            reasons.append("terminal_height")
        return reasons
    position, velocity = state
    if not (
        sim.min_position <= position <= sim.max_position
        and abs(velocity) <= sim.max_speed
    ):
        reasons.append("outside_simulator_bounds")
    if position >= sim.goal_position and velocity >= sim.goal_velocity:
        reasons.append("terminal_goal")
    return reasons


def audit_grid_constructed(
    expert,
    restriction_id: str,
    reset_seed: int,
    check,
) -> Dict[str, Any]:
    """Constructed-state witness audit of a random-offset grid (CartPole,
    Acrobot, MountainCar).

    Every member of a pair is evaluated alone: its observation is the
    simulator's own map of its state (checked bitwise against step output
    under every action), its validity is the simulator's bounds and
    nonterminal test, its cell is what the frozen map returns, and its label
    is one deterministic expert query on the full observation. A pair is a
    conflict only if both members are valid, their ``M(x)`` are bitwise equal
    and their labels differ; every other checked pair is kept with reasons.

    Raises:
        RuntimeError: If the trajectory's observations disagree with the
            observation map, or the map returns no single representative.
        tasks.TaskMismatch: If the environment's Box differs from the grid's.
    """
    grid = restriction.get_restriction(restriction_id)
    env_name = grid.env_name
    task = tasks.get_task(env_name)
    state_index = GRID_STATE_INDEX.get(env_name, {})
    axes = [a for a in grid.axes if a.kind != "binary"]
    expert_calls = simulator_steps = 0
    env = gym.make(env_name)
    grid.check_box(env.observation_space)
    sim = env.unwrapped
    check("diagnostic reset")
    obs, _ = env.reset(seed=int(reset_seed))
    bases = []
    step = 0
    while True:
        if step % CONSTRUCTED_BASE_STRIDE == 0:
            state = np.array(sim.state, dtype=np.float64)
            if _bits(_obs_map(env_name, state)) != _bits(obs):
                env.close()
                raise RuntimeError(f"observation map differs at trajectory step {step}")
            bases.append(
                {
                    "base": len(bases),
                    "step": step,
                    "simulator_state": [float(v) for v in state],
                    "observation": [float(v) for v in obs],
                },
            )
        check("diagnostic expert query")
        action = int(_labels(expert, obs[None])[0])
        expert_calls += 1
        check("diagnostic env step")
        obs, _, terminated, truncated, _ = env.step(action)
        step += 1
        if terminated or truncated:
            break
    env.close()

    probe = gym.make(env_name).unwrapped
    probe.reset(seed=int(reset_seed))
    bounds = _grid_bounds(env_name, probe)

    def coordinate(axis) -> int:
        return state_index.get(axis.name, axis.indices[0])

    def evaluate(base, axis, fraction, cell, state):
        nonlocal expert_calls, simulator_steps
        observation = _obs_map(env_name, state)
        map_ok = True
        for act in range(task.n_actions):
            check("audit simulator step")
            probe.state = state.copy()
            if env_name == "CartPole-v1":
                probe.steps_beyond_terminated = None
            stepped, _, _, _, _ = probe.step(act)
            simulator_steps += 1
            after = np.array(probe.state, dtype=np.float64)
            map_ok &= _bits(stepped) == _bits(_obs_map(env_name, after))
        check("audit expert query")
        label = int(_labels(expert, observation[None])[0])
        expert_calls += 1
        masked = _masked(restriction_id, observation[None])[0]
        cells = {a.name: _grid_cell(a, masked) for a in axes}
        outside = _grid_validity(env_name, probe, state)
        in_cell = axis is None or cells[axis.name] == cell
        reasons = (
            outside
            + ([] if map_ok else ["observation_map_differs_from_step_output"])
            + ([] if in_cell else ["frozen_map_assigns_another_cell"])
        )
        return {
            "base": base["base"],
            "coordinate": None if axis is None else axis.name,
            "fraction": fraction,
            "value": None if axis is None else float(state[coordinate(axis)]),
            "simulator_state": [float(v) for v in state],
            "observation": [float(v) for v in observation],
            "masked_observation": [float(v) for v in masked],
            "masked_bits": _bits(masked).hex(),
            "grid_cells": cells,
            "expert_action": label,
            "checks": {
                "inside_bounds_nonterminal": not outside,
                "observation_map_verified": bool(map_ok),
                "in_declared_cell": bool(in_cell),
            },
            "invalid_reasons": reasons,
            "simulator_valid": not reasons,
        }

    states: List[Dict[str, Any]] = []
    groups: List[Tuple[int, str, List[Dict[str, Any]]]] = []
    excluded: List[Dict[str, Any]] = []
    for base in bases:
        state0 = np.array(base["simulator_state"], dtype=np.float64)
        member0 = evaluate(base, None, None, None, state0)
        states.append(member0)
        base["grid_cells"] = {}
        for axis in axes:
            cell_index = member0["grid_cells"][axis.name]
            if axis.kind == "circular":
                cell = _arc_cell(axis, cell_index)
            else:
                cell = _scalar_cell(axis, cell_index, bounds[axis.name])
            base["grid_cells"][axis.name] = cell
            if cell["empty"]:
                excluded.append(
                    {
                        "base": base["base"],
                        "coordinate": axis.name,
                        "reason": "cell_outside_construction_bounds",
                    },
                )
                continue
            group = [member0]
            for fraction in GRID_FRACTIONS:
                state = state0.copy()
                state[coordinate(axis)] = _grid_value(cell, fraction)
                member = evaluate(base, axis, fraction, cell_index, state)
                states.append(member)
                group.append(member)
            groups.append((base["base"], axis.name, group))
    probe.close()

    pairs = []
    for base_index, name, group in groups:
        for a, b in itertools.combinations(group, 2):
            equal = a["masked_bits"] == b["masked_bits"]
            valid = a["simulator_valid"] and b["simulator_valid"]
            reasons = [f"member_a: {r}" for r in a["invalid_reasons"]] + [
                f"member_b: {r}" for r in b["invalid_reasons"]
            ]
            if not equal:
                reasons.append("masked_observations_differ")
            if valid and equal and a["expert_action"] == b["expert_action"]:
                reasons.append("same_expert_action")
            pairs.append(
                {
                    "base": base_index,
                    "coordinate": name,
                    "fraction_a": a["fraction"],
                    "fraction_b": b["fraction"],
                    "value_a": a["value"],
                    "value_b": b["value"],
                    "expert_action_a": a["expert_action"],
                    "expert_action_b": b["expert_action"],
                    "masked_equal": bool(equal),
                    "both_valid": bool(valid),
                    "conflict": bool(
                        valid and equal and a["expert_action"] != b["expert_action"]
                    ),
                    "reasons": reasons,
                },
            )
    n_conflicts = sum(p["conflict"] for p in pairs)
    reason_counts: Dict[str, int] = {}
    for p in pairs:
        for r in p["reasons"]:
            key = r.split(": ", 1)[-1]
            reason_counts[key] = reason_counts.get(key, 0) + 1
    return {
        "status": "conflict_found" if n_conflicts else "restriction_uncertified",
        "evidence": {
            "type": "constructed simulator states at fixed interior fractions "
            "of the frozen grid cells of expert trajectory base states",
            "limits": GRID_CLAIM,
        },
        "grid": {
            "restriction_id": grid.restriction_id,
            "n_bins": grid.n_bins,
            "seed": grid.seed,
            "generator": grid.generator,
            "descriptor_sha256": grid.descriptor_sha256,
        },
        "fractions": list(GRID_FRACTIONS),
        "construction_bounds": {k: [float(v) for v in b] for k, b in bounds.items()},
        "unbounded_limit": GRID_UNBOUNDED_LIMIT,
        "construction_rule": "per base and grid coordinate independently: the "
        "base's own cell on that coordinate (raw edges intersected with the "
        "construction bounds; arcs from atan2 of the frozen edge vectors on "
        "the simulator's angle), the other coordinates kept; the members are "
        "the base and the states at each fraction",
        "base_stride": CONSTRUCTED_BASE_STRIDE,
        "base_states": bases,
        "excluded_coordinates": excluded,
        "states": states,
        "pairs": pairs,
        "n_states": len(states),
        "n_valid_states": sum(s["simulator_valid"] for s in states),
        "n_checked_pairs": len(pairs),
        "n_invalid_pairs": sum(
            not (p["both_valid"] and p["masked_equal"]) for p in pairs
        ),
        "n_conflicting_pairs": int(n_conflicts),
        "pair_reason_counts": dict(sorted(reason_counts.items())),
        "expert_predict_calls": expert_calls,
        "simulator_steps": simulator_steps,
    }


def _lunar_module():
    from gymnasium.envs.box2d import lunar_lander

    return lunar_lander


def _source_sha256(obj) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode("utf-8")).hexdigest()


def _lunar_simulator_pins(env) -> Dict[str, Any]:
    """Pins of the simulator the audit runs; refuses anything else.

    Raises:
        RuntimeError: If the step source, class, wrappers, factory kwargs,
            wind, continuous mode or world step differ from the pins.
    """
    import Box2D

    ll = _lunar_module()
    sim = env.unwrapped
    wrappers = []
    layer = env
    while layer is not sim:
        wrappers.append(type(layer).__name__)
        layer = layer.env
    step_sha = _source_sha256(type(sim).step)
    problems = [
        what
        for what, ok in (
            ("simulator class", type(sim) is ll.LunarLander),
            ("step source", step_sha == LUNAR_STEP_SOURCE_SHA256),
            ("continuous", sim.continuous is False),
            ("wind", sim.enable_wind is False),
            (
                "factory kwargs",
                env_utils.ENV_CONFIGS[LUNAR_ENV].get("env_kwargs") == {},
            ),
            (
                "wrappers",
                len(set(wrappers)) == len(wrappers)
                and set(wrappers) <= set(LUNAR_WRAPPERS),
            ),
            ("world step", (1.0 / ll.FPS, 6 * 30, 2 * 30) == LUNAR_WORLD_STEP),
        )
        if not ok
    ]
    if problems:
        raise RuntimeError(f"refused: LunarLander differs from the pins: {problems}")
    return {
        "step_source_sha256": step_sha,
        "contact_detector_source_sha256": _source_sha256(ll.ContactDetector),
        "box2d_version": Box2D.__version__,
        "gymnasium_version": gym.__version__,
        "wrappers": wrappers,
        "continuous": False,
        "enable_wind": False,
        "env_kwargs": {},
        "world_step": list(LUNAR_WORLD_STEP),
    }


def _box2d_obs_map(sim) -> np.ndarray:
    """An independent copy of the observation formula in ``LunarLander.step``."""
    ll = _lunar_module()
    pos, vel = sim.lander.position, sim.lander.linearVelocity
    half_w, half_h = ll.VIEWPORT_W / ll.SCALE / 2, ll.VIEWPORT_H / ll.SCALE / 2
    return np.array(
        [
            (pos.x - half_w) / half_w,
            (pos.y - (sim.helipad_y + ll.LEG_DOWN / ll.SCALE)) / half_h,
            vel.x * half_w / ll.FPS,
            vel.y * half_h / ll.FPS,
            sim.lander.angle,
            20.0 * sim.lander.angularVelocity / ll.FPS,
            1.0 if sim.legs[0].ground_contact else 0.0,
            1.0 if sim.legs[1].ground_contact else 0.0,
        ],
        dtype=np.float32,
    )


def _lunar_feature(index: int, value: float) -> np.float32:
    """The observation value a lander velocity ``value`` maps to."""
    ll = _lunar_module()
    if index == 2:
        return np.float32(value * (ll.VIEWPORT_W / ll.SCALE / 2) / ll.FPS)
    if index == 3:
        return np.float32(value * (ll.VIEWPORT_H / ll.SCALE / 2) / ll.FPS)
    if index == 5:
        return np.float32(20.0 * value / ll.FPS)
    raise ValueError(f"no LunarLander velocity feature {index}")


def _lunar_physical(index: int, feature: float) -> float:
    """A float32 lander velocity whose observation is near ``feature``.

    Rounding to float32 first makes the rigid edit set it exactly (Box2D
    stores float32), so `_lunar_feature` of it is what the step source shows.
    """
    ll = _lunar_module()
    if index == 2:
        value = feature * ll.FPS / (ll.VIEWPORT_W / ll.SCALE / 2)
    elif index == 3:
        value = feature * ll.FPS / (ll.VIEWPORT_H / ll.SCALE / 2)
    elif index == 5:
        value = feature * ll.FPS / 20.0
    else:
        raise ValueError(f"no LunarLander velocity feature {index}")
    return float(np.float32(value))


def _lunar_grid_cells(grid, restriction_id: str, base_obs: np.ndarray):
    """Per-base edit values of a grid: interior fractions of the base's cells.

    For each velocity axis (`LUNAR_GRID_AXES`) the cell that the frozen map
    gives the base observation is intersected with the declared Box (end
    cells extend to infinity and clamp), and each of `GRID_FRACTIONS` of that
    interval becomes a float32 physical velocity. The cell an edited state
    lands in is checked later on its actual observation, never assumed.

    Returns:
        ``(grids, cells)``: rows ``(index, name, values)`` in the format of
        `CONSTRUCTED_GRIDS`, and a JSON record of every cell.
    """
    masked = _masked(restriction_id, base_obs[None])[0]
    by_index = {a.indices[0]: a for a in grid.axes if a.kind == "linear"}
    rows, cells = [], []
    for index, name in LUNAR_GRID_AXES:
        axis = by_index[index]
        cell = _scalar_cell(
            axis,
            _grid_cell(axis, masked),
            (axis.low, axis.high),
        )
        targets = [_grid_value(cell, f) for f in GRID_FRACTIONS]
        values = tuple(_lunar_physical(index, t) for t in targets)
        cell.update(
            index=index,
            coordinate=name,
            base_feature=float(base_obs[index]),
            base_outside_declared_box=not axis.low <= base_obs[index] <= axis.high,
            fractions=list(GRID_FRACTIONS),
            target_features=targets,
            physical_values=list(values),
        )
        cells.append(cell)
        if not cell["empty"] and len(set(values)) == len(values):
            rows.append((index, name, values))
    return tuple(rows), cells


def _bodies(sim) -> List[Any]:
    return [sim.lander, *sim.legs]


def _pose_bits(sim) -> bytes:
    """Positions, angles and awake flags of every world body, as bytes."""
    rows = [
        (b.position.x, b.position.y, b.angle, float(b.awake)) for b in sim.world.bodies
    ]
    return np.asarray(rows, dtype=np.float64).tobytes()


def _body_bits(sim) -> bytes:
    """Positions, angles, velocities and awake flags of every world body."""
    rows = [
        (
            b.position.x,
            b.position.y,
            b.angle,
            b.linearVelocity.x,
            b.linearVelocity.y,
            b.angularVelocity,
            float(b.awake),
        )
        for b in sim.world.bodies
    ]
    return np.asarray(rows, dtype=np.float64).tobytes()


def _velocities(sim) -> List[Tuple[float, float, float]]:
    return [
        (b.linearVelocity.x, b.linearVelocity.y, b.angularVelocity)
        for b in _bodies(sim)
    ]


def _set_velocities(sim, rows) -> None:
    for body, (vx, vy, w) in zip(_bodies(sim), rows):
        body.linearVelocity = (vx, vy)
        body.angularVelocity = w


def _joint_rows(sim) -> List[Tuple[float, ...]]:
    """Per leg joint: anchor velocity mismatch, joint speed, anchor offset."""
    rows = []
    for joint in sim.world.joints:
        va = joint.bodyA.GetLinearVelocityFromWorldPoint(joint.anchorA)
        vb = joint.bodyB.GetLinearVelocityFromWorldPoint(joint.anchorB)
        rows.append(
            (
                va.x - vb.x,
                va.y - vb.y,
                joint.speed,
                joint.anchorA.x - joint.anchorB.x,
                joint.anchorA.y - joint.anchorB.y,
            ),
        )
    return rows


def _joint_residual(before, after, delta: float, angular: bool) -> float:
    """Largest joint change beyond the exact rigid-motion term.

    A rigid increment changes a joint's anchor velocity mismatch by exactly
    ``delta x (anchorA - anchorB)`` (zero for a linear increment) and leaves
    its speed unchanged; what remains is float32 rounding.
    """
    worst = 0.0
    for b, a in zip(before, after):
        sx, sy = b[3], b[4]
        expect = (-delta * sy, delta * sx) if angular else (0.0, 0.0)
        worst = max(
            worst,
            abs((a[0] - b[0]) - expect[0]),
            abs((a[1] - b[1]) - expect[1]),
            abs(a[2] - b[2]),
        )
    return worst


def _anchor_separation(sim) -> float:
    return max(
        math.hypot(j.anchorA.x - j.anchorB.x, j.anchorA.y - j.anchorB.y)
        for j in sim.world.joints
    )


def _touching_contacts(sim) -> int:
    ours = _bodies(sim)
    return sum(
        1
        for c in sim.world.contacts
        if c.touching and (c.fixtureA.body in ours or c.fixtureB.body in ours)
    )


def _apply_velocity_edit(sim, name: str, value: float) -> float:
    """Rigid velocity increment over the lander and both legs; returns it.

    The edited lander component is set to ``value`` exactly. Each leg gets the
    same increment: for the angular velocity also the linear velocity of the
    rotation about the lander's centre of mass, so the joints stay consistent.
    """
    lander = sim.lander
    value = float(value)
    if name == "lander_angular_velocity":
        delta = value - lander.angularVelocity
        cx, cy = lander.worldCenter.x, lander.worldCenter.y
        lander.angularVelocity = value
        for leg in sim.legs:
            rx, ry = leg.worldCenter.x - cx, leg.worldCenter.y - cy
            v = leg.linearVelocity
            leg.linearVelocity = (v.x - delta * ry, v.y + delta * rx)
            leg.angularVelocity = leg.angularVelocity + delta
        return delta
    axis = {"lander_vx": 0, "lander_vy": 1}[name]
    v = [lander.linearVelocity.x, lander.linearVelocity.y]
    delta = value - v[axis]
    v[axis] = value
    lander.linearVelocity = tuple(v)
    for leg in sim.legs:
        w = [leg.linearVelocity.x, leg.linearVelocity.y]
        w[axis] += delta
        leg.linearVelocity = tuple(w)
    return delta


class _ZeroLengthWorld:
    """Stand-in for ``world`` during one ``step``: ``Step`` runs with dt 0."""

    def __init__(self, world) -> None:
        self._world = world
        self.calls: List[List[Any]] = []

    def Step(self, dt, velocity_iterations, position_iterations) -> None:  # noqa: N802
        self.calls.append([dt, velocity_iterations, position_iterations])
        self._world.Step(0.0, velocity_iterations, position_iterations)

    def __getattr__(self, name):
        return getattr(self._world, name)


def _zero_dt_step(sim) -> Dict[str, Any]:
    """The unmodified ``step`` (action 0) with a zero-length world step.

    ``world``, the random stream, ``prev_shaping`` and ``game_over`` are
    restored afterwards, also if the call raises. Leg contact flags are not:
    they are what the contact refresh found at the unchanged positions.
    """
    rng = sim.np_random.bit_generator
    rng_state = copy.deepcopy(rng.state)
    prev_shaping, game_over, world = sim.prev_shaping, sim.game_over, sim.world
    before = _body_bits(sim)
    proxy = _ZeroLengthWorld(world)
    sim.world = proxy
    try:
        obs, _, terminated, truncated, _ = type(sim).step(sim, 0)
        refreshed_game_over = bool(sim.game_over)
    finally:
        sim.world = world
        rng.state = rng_state
        sim.prev_shaping = prev_shaping
        sim.game_over = game_over
    return {
        "obs": np.asarray(obs, dtype=np.float32),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "game_over": refreshed_game_over,
        "world_step_calls": proxy.calls,
        "bodies_unchanged": _body_bits(sim) == before,
        "restored": sim.world is world
        and rng.state == rng_state
        and sim.prev_shaping == prev_shaping
        and sim.game_over == game_over,
    }


def _refresh_problems(refresh: Mapping[str, Any]) -> List[str]:
    """Failures of the zero-length step itself (never a property of a state)."""
    return [
        what
        for what, ok in (
            (
                "world step called once",
                refresh["world_step_calls"] == [list(LUNAR_WORLD_STEP)],
            ),
            ("bodies unchanged", refresh["bodies_unchanged"]),
            ("state restored", refresh["restored"]),
            ("not truncated", not refresh["truncated"]),
        )
        if not ok
    ]


def _airborne(sim, refresh: Mapping[str, Any]) -> bool:
    return bool(
        not sim.legs[0].ground_contact
        and not sim.legs[1].ground_contact
        and not refresh["game_over"]
        and _touching_contacts(sim) == 0
    )


def _bits(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array, dtype=np.float32).tobytes()


def _lunar_replay(reset_seed: int, actions: Sequence[int], recorded, k: int, check):
    """A fresh environment replayed to base step ``k``, checked bitwise.

    Raises:
        RuntimeError: If the replay does not reproduce the recorded observation.
    """
    check("audit replay")
    env = gym.make(LUNAR_ENV)
    obs, _ = env.reset(seed=int(reset_seed))
    for action in actions[:k]:
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            env.close()
            raise RuntimeError(f"replay to base step {k} ended early")
    if _bits(obs) != _bits(recorded[k]):
        env.close()
        raise RuntimeError(
            f"replay to base step {k} differs from the recorded observation"
        )
    return env


def _one_real_step(reset_seed, actions, recorded, k, edit, n_actions, check):
    """One real step from the (edited) base under every action."""
    rows = []
    for action in range(n_actions):
        env = _lunar_replay(reset_seed, actions, recorded, k, check)
        sim = env.unwrapped
        if edit is not None:
            _apply_velocity_edit(sim, *edit)
        check("audit one real step")
        try:
            obs, reward, terminated, _, _ = env.step(action)
            rows.append(
                {
                    "action": action,
                    "executes": True,
                    "finite": bool(
                        np.all(np.isfinite(obs)) and math.isfinite(float(reward)),
                    ),
                    "terminated": bool(terminated),
                    "anchor_separation_max": float(_anchor_separation(sim)),
                },
            )
        except Exception as exc:  # noqa: BLE001 - recorded; the member fails
            rows.append({"action": action, "executes": False, "error": repr(exc)})
        finally:
            env.close()
    return {
        "actions": rows,
        "one_real_step_executes": all(r["executes"] and r["finite"] for r in rows),
        "note": "a real step observes the next state, not the edited one; it "
        "shows only that Box2D accepts the edited state",
    }


def audit_box2d(
    expert,
    restriction_id: str,
    reset_seed: int,
    check,
) -> Dict[str, Any]:
    """Constructed-state witness audit of one LunarLander ladder restriction.

    A random-offset grid takes its edit values per valid base from the base's
    own grid cells (`_lunar_grid_cells`); everything else, records included,
    is the ladder workflow, plus the check that an edited state lands in the
    base's cell.

    Raises:
        RuntimeError: If the simulator differs from the pins, a replay or the
            zero-length step misbehaves, or the observation copy disagrees with
            the step source on a valid state. These are job failures, never
            "no conflict".
    """
    found = restriction.get_restriction(restriction_id)
    is_grid = isinstance(found, random_grid.RandomGridRestriction)
    grids = () if is_grid else CONSTRUCTED_GRIDS[restriction_id]
    grids_of: Dict[int, Tuple[Tuple[int, str, Tuple[float, ...]], ...]] = {}
    n_actions = tasks.get_task(LUNAR_ENV).n_actions
    expert_calls = 0
    env = gym.make(LUNAR_ENV)
    pins = _lunar_simulator_pins(env)
    if is_grid:
        found.check_box(env.observation_space)
    check("diagnostic reset")
    obs, _ = env.reset(seed=int(reset_seed))
    recorded, actions = [np.asarray(obs, dtype=np.float32)], []
    terminated = truncated = False
    while not (terminated or truncated):
        check("diagnostic expert query")
        action = int(_labels(expert, obs[None])[0])
        expert_calls += 1
        actions.append(action)
        check("diagnostic env step")
        obs, _, terminated, truncated, _ = env.step(action)
        recorded.append(np.asarray(obs, dtype=np.float32))
    env.close()

    bases, states, members = [], [], {}
    for k in range(0, len(actions), CONSTRUCTED_BASE_STRIDE):
        env = _lunar_replay(reset_seed, actions, recorded, k, check)
        sim = env.unwrapped
        base_obs = recorded[k]
        check("audit zero-length step")
        refresh = _zero_dt_step(sim)
        problems = _refresh_problems(refresh)
        if problems:
            raise RuntimeError(f"zero-length step at base step {k}: {problems}")
        reproduced = _bits(refresh["obs"]) == _bits(base_obs)
        flags_only = bool(
            not reproduced and _bits(refresh["obs"][:6]) == _bits(base_obs[:6])
        )
        if not reproduced and not flags_only:
            raise RuntimeError(f"zero-length step at base step {k} moved a feature")
        airborne = _airborne(sim, refresh)
        mirror = _bits(_box2d_obs_map(sim)) == _bits(refresh["obs"])
        checks = {
            "replay_bitwise": True,
            "world_step_called_once_with_dt_zero": True,
            "bodies_unchanged": True,
            "rng_prev_shaping_game_over_world_restored": True,
            "source_map_reproduces_recorded": reproduced,
            "nonterminal": not refresh["terminated"],
            "airborne": airborne,
            "observation_copy_equals_source": mirror,
        }
        reason = None
        if not reproduced:
            reason = "contact_transition"
        elif refresh["terminated"]:
            reason = "terminal_after_refresh"
        elif not airborne:
            reason = "not_airborne"
        elif not mirror:
            raise RuntimeError(
                f"observation copy differs from the source at base step {k}"
            )
        base = {
            "base": len(bases),
            "step": k,
            "observation": [float(v) for v in base_obs],
            "trajectory_action": actions[k],
            "checks": checks,
            "excluded_reason": reason,
        }
        bases.append(base)
        if reason is not None:
            env.close()
            continue
        base_grids = grids
        if is_grid:
            base_grids, base["grid_cells"] = _lunar_grid_cells(
                found,
                restriction_id,
                base_obs,
            )
            base_masked = _bits(_masked(restriction_id, base_obs[None])[0])
        grids_of[base["base"]] = base_grids
        group = [
            {
                "base": base["base"],
                "coordinate": None,
                "index": None,
                "value": None,
                "delta": 0.0,
                "observation": refresh["obs"],
                "checks": {"unmodified_base_state": True},
                "simulator_valid": True,
            },
        ]
        for index, name, grid in base_grids:
            for value in grid:
                pose, velocities, joints = (
                    _pose_bits(sim),
                    _velocities(sim),
                    _joint_rows(sim),
                )
                delta = _apply_velocity_edit(sim, name, value)
                edited, pose_kept = _velocities(sim), _pose_bits(sim) == pose
                keep = {
                    "lander_angular_velocity": (0, 1),
                    "lander_vx": (1, 2),
                    "lander_vy": (0, 2),
                }[name]
                joint_change = _joint_residual(
                    joints,
                    _joint_rows(sim),
                    delta,
                    name == "lander_angular_velocity",
                )
                check("audit zero-length step")
                refresh = _zero_dt_step(sim)
                problems = _refresh_problems(refresh)
                if problems:
                    raise RuntimeError(f"zero-length step at base step {k}: {problems}")
                state_obs = refresh["obs"]
                mirror = _bits(_box2d_obs_map(sim)) == _bits(state_obs)
                airborne = _airborne(sim, refresh)
                _set_velocities(sim, velocities)
                if _velocities(sim) != velocities or _pose_bits(sim) != pose:
                    raise RuntimeError(f"velocity restore failed at base step {k}")
                state_checks = {
                    "positions_angles_awake_unchanged": pose_kept,
                    "untouched_lander_velocities_unchanged": all(
                        edited[0][i] == velocities[0][i] for i in keep
                    ),
                    "joint_kinematics_within_tolerance": joint_change
                    <= LUNAR_JOINT_TOLERANCE,
                    "other_features_bitwise_equal": _bits(np.delete(state_obs, index))
                    == _bits(np.delete(base_obs, index)),
                    "edited_feature_equals_target": _bits(state_obs[index : index + 1])
                    == _bits(np.array([_lunar_feature(index, value)])),
                    "nonterminal": not refresh["terminated"],
                    "airborne": airborne,
                    "observation_copy_equals_source": mirror,
                    "restored_bitwise": True,
                }
                if is_grid:
                    # The frozen map of the actual observation, never assumed.
                    state_checks["in_base_grid_cell"] = (
                        _bits(_masked(restriction_id, state_obs[None])[0])
                        == base_masked
                    )
                group.append(
                    {
                        "base": base["base"],
                        "coordinate": name,
                        "index": index,
                        "value": float(value),
                        "delta": float(delta),
                        "observation": state_obs,
                        "checks": state_checks,
                        "joint_residual_max": float(joint_change),
                        "anchor_separation_max": float(
                            max(math.hypot(r[3], r[4]) for r in joints),
                        ),
                        "inside_observation_space_diagnostic": bool(
                            sim.observation_space.contains(state_obs),
                        ),
                        "simulator_valid": all(state_checks.values()),
                    },
                )
        env.close()
        stacked = np.stack([s["observation"] for s in group])
        check("audit expert query")
        batched = _labels(expert, stacked)
        expert_calls += 1
        for i, s in enumerate(group):
            check("audit expert query")
            single = int(_labels(expert, s["observation"][None])[0])
            expert_calls += 1
            s.update(
                expert_action=single,
                expert_action_batched=int(batched[i]),
                label_stable=single == int(batched[i]),
                masked_observation=[
                    float(v) for v in _masked(restriction_id, s["observation"][None])[0]
                ],
                observation=[float(v) for v in s["observation"]],
                one_real_step=None,
            )
            if s["coordinate"] is None:
                s["label_stable"] &= single == actions[k]
            members[(s["base"], s["coordinate"], s["value"])] = s
            states.append(s)

    def one_step(member):
        if member["one_real_step"] is None:
            k = bases[member["base"]]["step"]
            edit = (
                None
                if member["coordinate"] is None
                else (member["coordinate"], member["value"])
            )
            member["one_real_step"] = _one_real_step(
                reset_seed, actions, recorded, k, edit, n_actions, check
            )
        return member["one_real_step"]["one_real_step_executes"]

    pairs = []
    for b in bases:
        if b["excluded_reason"] is not None:
            continue
        base_member = members[(b["base"], None, None)]
        for _, name, grid in grids_of[b["base"]]:
            group = [base_member] + [members[(b["base"], name, float(v))] for v in grid]
            for a, c in itertools.combinations(group, 2):
                equal = a["masked_observation"] == c["masked_observation"]
                valid = a["simulator_valid"] and c["simulator_valid"]
                stable = a["label_stable"] and c["label_stable"]
                candidate = bool(
                    valid
                    and stable
                    and equal
                    and a["expert_action"] != c["expert_action"]
                )
                stepped = bool(candidate and one_step(a) and one_step(c))
                pairs.append(
                    {
                        "base": b["base"],
                        "coordinate": name,
                        "value_a": a["value"],
                        "value_b": c["value"],
                        "expert_action_a": a["expert_action"],
                        "expert_action_b": c["expert_action"],
                        "masked_equal": bool(equal),
                        "both_valid": bool(valid),
                        "both_labels_stable": bool(stable),
                        "both_one_real_step_execute": stepped if candidate else None,
                        "conflict": stepped,
                    },
                )
    n_conflicts = sum(p["conflict"] for p in pairs)
    excluded: Dict[str, int] = {}
    for b in bases:
        if b["excluded_reason"] is not None:
            excluded[b["excluded_reason"]] = excluded.get(b["excluded_reason"], 0) + 1
    if is_grid:
        grid_fields = {
            "grids": None,
            "grid": {
                "restriction_id": found.restriction_id,
                "n_bins": found.n_bins,
                "seed": found.seed,
                "generator": found.generator,
                "descriptor_sha256": found.descriptor_sha256,
            },
            "fractions": list(GRID_FRACTIONS),
            "construction_rule": "per valid base and velocity axis (vx, vy, "
            "angular velocity) independently: the base's own grid cell clamped "
            "to the declared Box, its interior fractions as float32 physical "
            "velocities through the rigid edit; position, angle and leg "
            "contacts are never edited, so their cells are not searched",
            "grid_claim": GRID_CLAIM,
        }
    else:
        grid_fields = {
            "grids": [[i, n, list(g)] for i, n, g in grids],
            "zeroed_indices": list(found.zeroed_indices),
        }
    return {
        "status": "conflict_found" if n_conflicts else "restriction_uncertified",
        "evidence": {
            "type": "constructed Box2D simulator state, observed through the "
            "pinned step source with a zero-length world step",
            "limits": LUNAR_CLAIM,
        },
        "simulator": pins,
        "grids": grid_fields.pop("grids"),
        "base_stride": CONSTRUCTED_BASE_STRIDE,
        "trajectory": {
            "steps": len(actions),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "actions": actions,
        },
        "base_states": bases,
        "n_bases": len(bases),
        "n_valid_bases": sum(b["excluded_reason"] is None for b in bases),
        "n_excluded_bases": excluded,
        "states": states,
        "pairs": pairs,
        "n_checked_pairs": len(pairs),
        "n_invalid_pairs": sum(
            not (p["both_valid"] and p["masked_equal"]) for p in pairs
        ),
        "n_conflicting_pairs": int(n_conflicts),
        "n_one_real_step_members": sum(s["one_real_step"] is not None for s in states),
        "joint_tolerance": LUNAR_JOINT_TOLERANCE,
        "joint_rule": "per leg joint, the anchor velocity mismatch changes by "
        "the exact rigid-motion term delta x (anchorA - anchorB) and the joint "
        "speed is unchanged, each within the tolerance (float32 rounding); the "
        "existing anchor separation is reported, not gated",
        "expert_predict_calls": expert_calls,
        **grid_fields,
    }


def audit(
    expert,
    env_name: str,
    restriction_id: str,
    reset_seed: int,
    *,
    check: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the audit appropriate to ``env_name`` for one ladder restriction.

    Calibration candidates of the task (`tasks.calibration_candidates`) are
    accepted too, opt-in, with the same evidence type, and so are its
    random-offset grids (`tasks.RANDOM_GRID_CANDIDATES`; `audit_grid_constructed`
    or, on LunarLander, `audit_box2d`).

    Returns:
        A JSON-serializable record with ``status``, ``evidence_type``, the
        claim, and every checked state or pair. ``unverified`` audits make no
        simulator or expert call.

    Raises:
        ValueError: If the restriction is neither on the task's ladder nor one
            of its calibration candidates or grids.
    """
    check = check or (lambda what: None)
    task = tasks.get_task(env_name)
    accepted = (
        task.ladder
        + tasks.calibration_candidates(env_name)
        + tasks.RANDOM_GRID_CANDIDATES.get(env_name, ())
    )
    if restriction_id not in accepted:
        raise ValueError(
            f"{restriction_id!r} is not on the {env_name} ladder or among its "
            f"calibration candidates or grids",
        )
    kind = evidence_type(env_name, restriction_id)
    base = {
        "env_name": env_name,
        "restriction_id": restriction_id,
        "restriction": restriction.describe_restriction(restriction_id),
        "evidence_type": kind,
        "claim": CLAIM,
        "reset_seed": int(reset_seed),
        "labels_used_for_training": False,
    }
    if kind == "unverified":
        return dict(
            base,
            status="unverified",
            reason="LunarLander forms the observation after world.Step and "
            "takes leg contacts from a contact listener; constructing a state "
            "with a chosen observation is not validated here, so no witness is "
            "attempted and the agnostic claim is held",
            expert_predict_calls=0,
        )
    if kind == "reachable_enumeration":
        result = audit_enumerated(expert, restriction_id, check)
        base["reset_seed_used"] = False
    elif kind == "constructed_unwrapped_state":
        result = audit_constructed(expert, restriction_id, reset_seed, check)
    elif kind == "constructed_box2d_state":
        result = audit_box2d(expert, restriction_id, reset_seed, check)
    elif kind == GRID_EVIDENCE:
        result = audit_grid_constructed(expert, restriction_id, reset_seed, check)
    else:
        if restriction_id not in restriction.AUDIT_RESTRICTIONS:
            raise ValueError(f"No CartPole audit for {restriction_id!r}")
        result = restriction.audit_cartpole_position(
            expert,
            reset_seed,
            restriction_id=restriction_id,
            check=check,
        )
        result["evidence"] = {
            "type": "constructed through the unwrapped simulator state",
            "limits": "pairs differ only in cart position x; conflicts that "
            "hiding x_dot or theta_dot adds are not checked",
        }
    return dict(base, **result)
