"""Immutable, ordered expert datasets for fixed BC."""

import fcntl
import hashlib
import json
import os
import pathlib
import tempfile

import numpy as np

from imitation.data import types


def policy_digest(policy):
    """Identify the exact expert weights used for collection."""
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_or_collect(root, metadata, collect):
    """Atomically create or load ordered transitions, including provenance.

    A filesystem lock permits concurrent workers to reuse the same fixed BC
    dataset without duplicate collection. The callback is called once
    per metadata key. Only numeric arrays are stored, never pickled objects.
    """
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(metadata, sort_keys=True)
    key = hashlib.sha256(encoded.encode()).hexdigest()
    path = root / f"{key}.npz"
    with (root / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            collected = collect()
            transitions, stats = (
                collected if isinstance(collected, tuple) else (collected, {})
            )
            if len(transitions) != metadata["budget"]:
                raise ValueError("Expert dataset does not match requested budget")
            with tempfile.NamedTemporaryFile(
                dir=root, suffix=".npz", delete=False
            ) as tmp:
                temporary = pathlib.Path(tmp.name)
                try:
                    np.savez_compressed(
                        tmp,
                        obs=transitions.obs,
                        acts=transitions.acts,
                        next_obs=transitions.next_obs,
                        dones=transitions.dones,
                        metadata=np.array(encoded),
                        stats=np.array(json.dumps(stats)),
                    )
                    tmp.flush()
                    os.fsync(tmp.fileno())
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["metadata"]) != encoded:
                raise ValueError("Shared expert dataset metadata mismatch")
            transitions = types.Transitions(
                obs=saved["obs"],
                acts=saved["acts"],
                next_obs=saved["next_obs"],
                dones=saved["dones"],
                infos=np.array([{} for _ in saved["acts"]]),
            )
            stats = json.loads(str(saved["stats"]))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return transitions, {
        **metadata,
        **stats,
        "path": str(path),
        "sha256": digest.hexdigest(),
    }
