"""Predeclared observation quantizers and the exact majority-table learner.

Every quantizer is fixed before any data is seen. A feature with edges
``e_1 < ... < e_m`` has ``m + 1`` bins: ``(-inf, e_1)``, ``[e_1, e_2)``, ...,
``[e_m, inf)``. Intervals are closed on the left, so a value exactly on an edge
falls in the bin to its right; for the velocity-sign bins (edge 0) a zero
velocity is in the non-negative bin. Angles are ``atan2(sin, cos)`` mapped to
``(-pi, pi]``, so ``-pi`` and ``pi`` (the same physical angle) share a bin.
The bin id is mixed radix with the first listed feature most significant.
``n_bins`` counts every possible bin, observed or not.

The learner class is every table from bin id to action. Fitting is exact 0-1
empirical risk minimization from cumulative ``(K, A)`` label counts: each bin
takes its majority label, ties go to the lowest action id, and a bin without
data takes action 0. Each fit returns a new table and keeps no other state.
"""

import dataclasses
import hashlib
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

REPRESENTATIONS = ("mild", "severe")
_ANGLE_EDGES = (-math.pi / 3, math.pi / 3)
_SIGN_EDGES = (0.0,)


@dataclasses.dataclass(frozen=True)
class Feature:
    """One scalar feature of an observation and its bin edges.

    ``index`` reads ``obs[index]``. If ``angle_of`` is set to ``(sin_i, cos_i)``,
    the feature is instead ``atan2(obs[sin_i], obs[cos_i])`` in ``(-pi, pi]``.
    """

    name: str
    edges: Tuple[float, ...]
    index: Optional[int] = None
    angle_of: Optional[Tuple[int, int]] = None

    def value(self, obs: np.ndarray) -> float:
        if self.angle_of is None:
            return float(obs[self.index])
        sin_i, cos_i = self.angle_of
        angle = math.atan2(float(obs[sin_i]), float(obs[cos_i]))
        return math.pi if angle == -math.pi else angle

    def bin(self, obs: np.ndarray) -> int:
        return int(np.searchsorted(self.edges, self.value(obs), side="right"))


@dataclasses.dataclass(frozen=True)
class Quantizer:
    """Data-independent map from a raw observation to a bin id."""

    env_name: str
    representation: str
    obs_dim: int
    features: Tuple[Feature, ...]

    @property
    def n_bins(self) -> int:
        return int(np.prod([len(f.edges) + 1 for f in self.features]))

    def __call__(self, obs) -> int:
        """Return the bin id of one observation.

        Args:
            obs: Raw observation of shape ``(obs_dim,)``.

        Returns:
            Bin id in ``[0, n_bins)``.

        Raises:
            ValueError: On a wrong shape or a non-finite entry.
        """
        obs = np.asarray(obs, dtype=np.float64)
        if obs.shape != (self.obs_dim,):
            raise ValueError(
                f"{self.env_name} observation must have shape ({self.obs_dim},), "
                f"got {obs.shape}",
            )
        if not np.all(np.isfinite(obs)):
            raise ValueError(f"Non-finite observation {obs.tolist()}")
        bin_id = 0
        for feature in self.features:
            bin_id = bin_id * (len(feature.edges) + 1) + feature.bin(obs)
        return bin_id

    def describe(self) -> Dict[str, object]:
        """Return a JSON-compatible specification of this quantizer."""
        return {
            "env_name": self.env_name,
            "representation": self.representation,
            "obs_dim": self.obs_dim,
            "n_bins": self.n_bins,
            "bin_convention": (
                "intervals closed on the left: (-inf,e1), [e1,e2), ..., [em,inf); "
                "mixed radix, first feature most significant"
            ),
            "angle_convention": "atan2(sin, cos) mapped to (-pi, pi]",
            "features": [
                {
                    "name": f.name,
                    "edges": list(f.edges),
                    "index": f.index,
                    "angle_of_sin_cos": list(f.angle_of) if f.angle_of else None,
                }
                for f in self.features
            ],
        }


def _build() -> Dict[Tuple[str, str], Quantizer]:
    cart_severe = (
        Feature("cart_position", (-0.4, 0.4), index=0),
        Feature("pole_angle", (-0.05, 0.05), index=2),
    )
    cart_mild = cart_severe + (
        Feature("cart_velocity_sign", _SIGN_EDGES, index=1),
        Feature("pole_angular_velocity_sign", _SIGN_EDGES, index=3),
    )
    acro_severe = (
        Feature("theta1", _ANGLE_EDGES, angle_of=(1, 0)),
        Feature("theta2", _ANGLE_EDGES, angle_of=(3, 2)),
    )
    acro_mild = acro_severe + (
        Feature("theta1_velocity_sign", _SIGN_EDGES, index=4),
        Feature("theta2_velocity_sign", _SIGN_EDGES, index=5),
    )
    car_severe = (Feature("position", (-0.9, -0.5, -0.1), index=0),)
    car_mild = car_severe + (Feature("velocity", (-0.01, 0.01), index=1),)
    specs = {
        ("CartPole-v1", "severe", 4): cart_severe,
        ("CartPole-v1", "mild", 4): cart_mild,
        ("Acrobot-v1", "severe", 6): acro_severe,
        ("Acrobot-v1", "mild", 6): acro_mild,
        ("MountainCar-v0", "severe", 2): car_severe,
        ("MountainCar-v0", "mild", 2): car_mild,
    }
    return {
        (env, rep): Quantizer(env, rep, dim, feats)
        for (env, rep, dim), feats in specs.items()
    }


QUANTIZERS = _build()


def get_quantizer(env_name: str, representation: str) -> Quantizer:
    """Return the predeclared quantizer for an env and representation.

    Raises:
        ValueError: If the pair is not predeclared.
    """
    try:
        return QUANTIZERS[(env_name, representation)]
    except KeyError:
        raise ValueError(
            f"No quantizer for {env_name!r} / {representation!r}",
        ) from None


def label_counts(
    bins: Sequence[int],
    labels: Sequence[int],
    n_bins: int,
    n_actions: int,
) -> np.ndarray:
    """Return the ``(n_bins, n_actions)`` count matrix of (bin, label) pairs."""
    counts = np.zeros((n_bins, n_actions), dtype=np.int64)
    np.add.at(
        counts,
        (np.asarray(bins, dtype=np.int64), np.asarray(labels, dtype=np.int64)),
        1,
    )
    return counts


def fit_table(counts: np.ndarray) -> np.ndarray:
    """Exact 0-1 ERM: a new majority table from cumulative counts.

    ``np.argmax`` returns the first maximum, which is the lowest action id on
    ties and action 0 for a bin with no data.
    """
    return np.argmax(np.asarray(counts), axis=1).astype(np.int64)


def empirical_risk(table: np.ndarray, counts: np.ndarray) -> int:
    """Number of counted labels that ``table`` disagrees with."""
    counts = np.asarray(counts)
    return int(counts.sum() - counts[np.arange(len(table)), table].sum())


def table_sha256(table: np.ndarray) -> str:
    """Digest of a table's length and int64 actions."""
    table = np.asarray(table, dtype=np.int64)
    return hashlib.sha256(
        str(table.shape[0]).encode() + b":" + table.tobytes(),
    ).hexdigest()


def data_sha256(bins: Sequence[int], labels: Sequence[int]) -> str:
    """Order-sensitive digest of a retained (bin, label) sequence."""
    digest = hashlib.sha256()
    digest.update(np.asarray(bins, dtype=np.int64).tobytes())
    digest.update(b"|")
    digest.update(np.asarray(labels, dtype=np.int64).tobytes())
    return digest.hexdigest()


def alias_lower_bound(counts: np.ndarray, delta: float) -> Dict[str, object]:
    """Uniform lower bound on the reference disagreement of every table.

    With ``n`` iid reference samples, Hoeffding plus a union bound over all
    ``A**K`` tables gives, with probability at least ``1 - delta``, a lower
    bound ``max(0, empirical_min - sqrt((K log A + log(1/delta)) / (2 n)))``
    on the minimum population disagreement. ``K`` includes unobserved bins.

    Args:
        counts: ``(K, A)`` counts of reference samples.
        delta: Failure probability in ``(0, 1)``.

    Returns:
        JSON-compatible summary; ``positive_certificate`` is True only for a
        strictly positive bound.
    """
    counts = np.asarray(counts, dtype=np.int64)
    n_bins, n_actions = counts.shape
    n = int(counts.sum())
    summary: Dict[str, object] = {
        "n": n,
        "K": int(n_bins),
        "A": int(n_actions),
        "delta": float(delta),
        "observed_bins": int(np.count_nonzero(counts.sum(axis=1))),
        "empirical_min_disagreement": None,
        "slack": None,
        "lower_bound": None,
        "positive_certificate": False,
    }
    if n == 0:
        return summary
    empirical = (n - int(counts.max(axis=1).sum())) / n
    slack = math.sqrt(
        (n_bins * math.log(n_actions) + math.log(1.0 / delta)) / (2.0 * n),
    )
    bound = max(0.0, empirical - slack)
    summary.update(
        empirical_min_disagreement=empirical,
        slack=slack,
        lower_bound=bound,
        positive_certificate=bound > 0.0,
    )
    return summary
