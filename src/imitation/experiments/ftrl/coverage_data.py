"""Load states + arrival-round from retained DAgger/BC scratch demos.

States collected/queried during a run are serialized as imitation trajectories
under ``{results_dir}/scratch/{algo}_{env}_seed{seed}/demos/round-{NNN}/``. The
arrival round is encoded in the directory name; this module reconstructs, per
algorithm, the visited states and the round at which each was collected.
"""

import dataclasses
import pathlib
import re
from typing import List, Optional, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from imitation.data import serialize

ALL_ALGOS = ["ftl", "ftrl", "bc", "bc_dagger"]
_ROUND_RE = re.compile(r"round-(\d+)")


@dataclasses.dataclass
class CoverageStates:
    """Visited states for one algorithm with per-state arrival round."""

    algo: str
    obs: (
        np.ndarray
    )  # [N, ...] native dtype: [N, D] for vector obs, [N, H, W, C] for Atari image obs
    rounds: np.ndarray  # [N] int


@dataclasses.dataclass
class PooledStates:
    """All algorithms' states pooled for a shared embedding."""

    obs: np.ndarray  # [Ntot, D]
    algo: np.ndarray  # [Ntot] object
    rounds: np.ndarray  # [Ntot] int


def scratch_demo_root(results_dir, algo: str, env_name: str, seed: int) -> pathlib.Path:
    """Return the demos root for one run cell."""
    cell = f"{algo}_{env_name}_seed{seed}"
    return pathlib.Path(results_dir) / "scratch" / cell / "demos"


def load_algo_states(
    results_dir, env_name: str, algo: str, seed: int
) -> Optional[CoverageStates]:
    """Load pooled states + arrival rounds for one algorithm, or None if absent."""
    root = scratch_demo_root(results_dir, algo, env_name, seed)
    if not root.is_dir():
        return None
    obs_chunks: List[np.ndarray] = []
    round_chunks: List[np.ndarray] = []
    for round_dir in sorted(root.glob("round-*")):
        match = _ROUND_RE.search(round_dir.name)
        if match is None:
            continue
        round_num = int(match.group(1))
        for npz in sorted(round_dir.glob("*.npz")):
            traj = serialize.load(npz)[0]
            obs = np.asarray(traj.obs[:-1])
            if obs.ndim == 1:
                obs = obs.reshape(obs.shape[0], -1)
            obs_chunks.append(obs)
            round_chunks.append(np.full(len(obs), round_num, dtype=int))
    if not obs_chunks:
        return None
    return CoverageStates(
        algo=algo,
        obs=np.concatenate(obs_chunks, axis=0),
        rounds=np.concatenate(round_chunks, axis=0),
    )


def load_env_states(
    results_dir, env_name: str, seed: int, algos: Sequence[str] = tuple(ALL_ALGOS)
) -> List[CoverageStates]:
    """Load states for every available algorithm in one environment."""
    out = []
    for algo in algos:
        cs = load_algo_states(results_dir, env_name, algo, seed)
        if cs is not None:
            out.append(cs)
    return out


def pool(states_list: Sequence[CoverageStates]) -> PooledStates:
    """Concatenate per-algorithm states into a single pooled array."""
    obs = np.concatenate([s.obs for s in states_list], axis=0)
    algo = np.concatenate(
        [np.full(len(s.obs), s.algo, dtype=object) for s in states_list]
    )
    rounds = np.concatenate([s.rounds for s in states_list])
    return PooledStates(obs=obs, algo=algo, rounds=rounds)


def standardize_features(obs: np.ndarray) -> np.ndarray:
    """Zero-mean/unit-variance features for t-SNE (fit on the pooled set)."""
    return StandardScaler().fit_transform(obs)
