"""Shared provenance, status, and threshold helpers for agnostic pilots.

These helpers keep evidence honest: status files are replaced atomically,
configs and checkpoints are identified by SHA256 digests, and the expert
acceptance threshold is frozen from the inherited convergence config before
any training starts.
"""

import hashlib
import json
import math
import numbers
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional

from imitation.experiments.ftrl import env_baselines, env_utils

PREPARATION_FILE = "preparation.json"
PREPARATION_SCHEMA = "agnostic-expert-preparation/1"
# Status of a qualified, approved preparation, as the campaign queue expects.
COMPLETE_STATUS = "complete"

# Budgets the inherited trainer casts with int(); each must be a positive
# integral finite number so the trainer's loop is finite and runs at least once.
_COUNT_KEYS = ("chunk_timesteps", "min_timesteps", "max_timesteps", "patience")

_PACKAGES = (
    "numpy",
    "torch",
    "stable_baselines3",
    "gymnasium",
    "imitation",
)

# Files whose content determines expert preparation behavior.
_SOURCE_FILES = (
    "experiments/agnostic/pilot.py",
    "experiments/agnostic/rollouts.py",
    "experiments/agnostic/run_classical.py",
    "experiments/ftrl/expert_training.py",
    "experiments/ftrl/env_utils.py",
    "experiments/ftrl/env_baselines.py",
    "experiments/ftrl/eval_utils.py",
    "util/util.py",
)
_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]


def merged_convergence_config(
    env_name: str,
    override: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the per-env convergence config with ``override`` merged on top.

    The inherited trainer treats an override as a full replacement, so it
    must always receive this merged result, never a partial dict.

    Args:
        env_name: Classical env ID.
        override: Optional subset of convergence keys.

    Returns:
        The complete merged config.

    Raises:
        ValueError: If ``override`` is not a mapping, has an unknown key, or
            the merged config is out of bounds (see `_validate_convergence`).
    """
    config = env_utils.get_convergence_config(env_name)
    if override is not None:
        if not isinstance(override, Mapping):
            raise ValueError(f"Convergence override must be an object: {override!r}")
        unknown = set(override) - set(env_utils.DEFAULT_CONVERGENCE)
        if unknown:
            raise ValueError(f"Unknown convergence keys: {sorted(unknown)}")
        config.update(override)
    return _validate_convergence(config)


def _finite_number(config: Mapping[str, Any], key: str) -> float:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"Convergence {key} must be a number, got {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"Convergence {key} must be finite, got {value!r}")
    return float(value)


def _validate_convergence(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check trainer budgets and gates; normalize integral budgets to int.

    Args:
        config: Complete merged convergence config.

    Returns:
        ``config`` with every count key as an int.

    Raises:
        ValueError: On a non-finite, non-positive, or non-integral budget or
            patience, ``min_timesteps > max_timesteps``, a threshold outside
            [0, 1], or a negative ``self_ce_eps``.
    """
    for key in _COUNT_KEYS:
        value = _finite_number(config, key)
        if value <= 0 or value != int(value):
            raise ValueError(
                f"Convergence {key} must be a positive integer, got {config[key]!r}",
            )
        config[key] = int(value)
    if config["min_timesteps"] > config["max_timesteps"]:
        raise ValueError(
            f"Convergence min_timesteps {config['min_timesteps']} exceeds "
            f"max_timesteps {config['max_timesteps']}",
        )
    threshold = _finite_number(config, "threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Convergence threshold must be in [0, 1], got {threshold}")
    config["threshold"] = threshold
    self_ce_eps = _finite_number(config, "self_ce_eps")
    if self_ce_eps < 0.0:
        raise ValueError(f"Convergence self_ce_eps must be >= 0, got {self_ce_eps}")
    config["self_ce_eps"] = self_ce_eps
    return config


def frozen_expert_threshold(
    env_name: str,
    convergence_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Freeze the raw-return acceptance threshold for an expert.

    ``raw = random + convergence_threshold * (expert - random)``, using the
    reference baselines. This is stricter than the generic
    ``EXPERT_QUALITY_THRESHOLD`` and matches what the trainer converged to.

    Args:
        env_name: Env ID present in ``REFERENCE_BASELINES``.
        convergence_config: Merged config; defaults to the env's own.

    Returns:
        Dict with the inputs, the formula, and ``raw_threshold``.
    """
    if convergence_config is None:
        convergence_config = merged_convergence_config(env_name)
    ref = env_baselines.REFERENCE_BASELINES[env_name]
    lo = float(ref["random_score"])
    hi = float(ref["expert_score"])
    frac = float(convergence_config["threshold"])
    return {
        "random_score": lo,
        "expert_score": hi,
        "convergence_threshold": frac,
        "raw_threshold": lo + frac * (hi - lo),
        "comparison": "mean_return >= raw_threshold",
        "formula": "random_score + convergence_threshold * "
        "(expert_score - random_score)",
    }


def canonical_json(value: Any) -> str:
    """Return a canonical JSON encoding used for digests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def config_digest(config: Mapping[str, Any]) -> str:
    """Return the SHA256 of a config's canonical JSON."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    """Return the SHA256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_state_sha256(policy) -> str:
    """Digest a torch module's parameters and buffers in key order.

    Args:
        policy: A torch module, for example an SB3 policy.

    Returns:
        SHA256 hex digest over names, dtypes, shapes, and raw values.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_write_json(path: pathlib.Path, payload: Any) -> None:
    """Write JSON to ``path`` via a temp file and atomic rename.

    Args:
        path: Destination file.
        payload: JSON-serializable value.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            # Strict JSON: the campaign queue rejects NaN and Infinity.
            json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(path: pathlib.Path) -> Any:
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)


def package_versions(packages: Iterable[str] = _PACKAGES) -> Dict[str, str]:
    """Return installed versions of relevant packages and Python.

    Args:
        packages: Import names to query.

    Returns:
        Mapping of package name to version string or ``"unavailable"``.
    """
    versions = {"python": sys.version.split()[0]}
    for name in packages:
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unavailable"))
        except Exception:  # pragma: no cover - depends on the environment
            versions[name] = "unavailable"
    return versions


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(_PACKAGE_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except Exception:
        return None
    return out.stdout.strip()


def source_fingerprint() -> Dict[str, Any]:
    """Fingerprint the source files that determine preparation behavior.

    Returns:
        Per-file SHA256 digests, their combined digest, and the git revision
        plus a dirty flag when git is available.
    """
    files = {}
    for rel in _SOURCE_FILES:
        path = _PACKAGE_ROOT / rel
        files[rel] = sha256_file(path) if path.is_file() else "missing"
    status = _git("status", "--porcelain")
    return {
        "files": files,
        "combined_sha256": config_digest(files),
        "git_revision": _git("rev-parse", "HEAD") or "unavailable",
        "git_dirty": (bool(status) if status is not None else "unavailable"),
    }


def verify_qualified_preparation(
    output_dir: pathlib.Path,
    config: Mapping[str, Any],
) -> bool:
    """Check that ``output_dir`` holds a qualified expert for ``config``.

    Later confirmatory cells should call this before using an expert. It
    requires ``status == "complete"`` with explicit ``qualified`` and
    ``approved`` both True, an identical config and config digest, and a
    checkpoint inside ``output_dir`` whose bytes still match both the
    recorded ``checkpoint.sha256`` and its top-level ``expert_sha256`` alias.

    Args:
        output_dir: Preparation directory.
        config: Expected preparation config.

    Returns:
        True only if every check passes.
    """
    output_dir = pathlib.Path(output_dir)
    try:
        record = read_json(output_dir / PREPARATION_FILE)
        checkpoint = (output_dir / record["checkpoint"]["path"]).resolve()
        checkpoint.relative_to(output_dir.resolve())
        return (
            record.get("status") == COMPLETE_STATUS
            and record.get("qualified") is True
            and record.get("approved") is True
            and record.get("config") == dict(config)
            and record.get("config_sha256") == config_digest(config)
            and record.get("expert_sha256") == record["checkpoint"]["sha256"]
            and checkpoint.is_file()
            and sha256_file(checkpoint) == record["checkpoint"]["sha256"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
