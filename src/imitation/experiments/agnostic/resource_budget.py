"""Shared worker-slot budget for cooperating controllers on one machine.

A budget is a state directory holding a lock file and one JSON lease per slot
held by a cooperating controller, plus a config that sets the combined slot
limit and lists external (non-cooperating) workloads. External workloads are
charged through a caller-supplied probe, so this module knows nothing about
their state format. Every read and write happens under one exclusive ``flock``,
so cooperating controllers never claim more than the limit between them.

The state directory is bound to one canonical config on first use (written
under the lock). Controllers whose canonical config differs, in limit or in the
set of external workloads, are refused, since mixing limits would let one
controller exceed another's. Equal configs stored at different paths cooperate.
A missing binding beside existing leases, or a malformed binding, fails closed.

Charging is conservative. Anything not verified as finished stays charged:

* A lease with no recorded child stays charged even after its owner died, since
  the owner may have crashed between fork and recording the child. An operator
  must verify and delete such a lease by hand.
* A lease with a recorded child stays charged while its owner is alive and has
  not released it, or while the child's process group still exists.
* A lease is pruned only when its owner is dead and its recorded group is gone.
* A probe that raises (missing, malformed, or mismatched external state) blocks
  every acquisition instead of counting as zero.

Liveness checks use signal 0, which delivers nothing. This module never signals
or terminates a process.

Config format::

    {"version": 1, "limit": 8,
     "external_campaigns": [{"inventory_dir": "/abs", "manifest_sha256": "..."}]}
"""

import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import re
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

CONFIG_VERSION = 1
LEASE_VERSION = 1
BINDING_VERSION = 1
CONFIG_KEYS = {"version", "limit", "external_campaigns"}
BINDING_KEYS = {"version", "config_sha256", "config"}
EXTERNAL_KEYS = {"inventory_dir", "manifest_sha256"}
LEASE_KEYS = {
    "version",
    "lease_id",
    "owner_pid",
    "created_at",
    "label",
    "child_pid",
    "child_pgid",
    "owner_released",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LEASE_ID_RE = re.compile(r"^[0-9a-f]{32}$")

ExternalProbe = Callable[[Dict[str, str]], Dict[str, Any]]


class CapacityError(Exception):
    """Invalid capacity config or state. Acquisition fails closed."""


def pid_alive(pid: int) -> bool:
    """Conservative liveness: a pid owned by another user counts as alive."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def group_alive(pgid: int) -> bool:
    """Conservative group liveness: a foreign group counts as alive."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _load_json(path: str) -> Any:
    def reject(name: str) -> Any:
        raise ValueError(f"non-finite JSON constant {name}")

    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read(), parse_constant=reject)


def _sha256_json(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_write_json(path: str, obj: Any) -> None:
    tmp = os.path.join(os.path.dirname(path), f".{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def load_config(path: str, max_limit: int) -> Dict[str, Any]:
    """Load and validate a capacity config.

    Args:
        path: Config JSON path.
        max_limit: Largest allowed ``limit``.

    Returns:
        Normalized config with ``limit``, ``external_campaigns``, ``path``,
        ``canonical`` (path- and order-independent content), and its ``sha256``.

    Raises:
        CapacityError: If the config is unreadable or invalid.
    """
    try:
        raw = _load_json(path)
    except (OSError, ValueError) as exc:
        raise CapacityError(f"cannot read capacity config {path!r}: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != CONFIG_KEYS:
        raise CapacityError(f"capacity config keys must be {sorted(CONFIG_KEYS)}")
    if type(raw["version"]) is not int or raw["version"] != CONFIG_VERSION:
        raise CapacityError(f"unsupported capacity config version {raw['version']!r}")
    limit = raw["limit"]
    if not _positive_int(limit) or limit > max_limit:
        raise CapacityError(f"capacity limit must be an integer in [1, {max_limit}]")
    externals = raw["external_campaigns"]
    if not isinstance(externals, list):
        raise CapacityError("external_campaigns must be a list")
    normalized: List[Dict[str, str]] = []
    for i, entry in enumerate(externals):
        if not isinstance(entry, dict) or set(entry) != EXTERNAL_KEYS:
            raise CapacityError(f"external_campaigns[{i}] keys must be {EXTERNAL_KEYS}")
        directory, sha = entry["inventory_dir"], entry["manifest_sha256"]
        if not isinstance(directory, str) or not os.path.isabs(directory):
            raise CapacityError(f"external_campaigns[{i}].inventory_dir not absolute")
        if not isinstance(sha, str) or not _SHA256_RE.match(sha):
            raise CapacityError(f"external_campaigns[{i}].manifest_sha256 malformed")
        normalized.append(
            {"inventory_dir": os.path.normpath(directory), "manifest_sha256": sha},
        )
    dirs = [e["inventory_dir"] for e in normalized]
    if len(set(dirs)) != len(dirs):
        raise CapacityError("duplicate external inventory_dir")
    canonical = {  # external order is irrelevant to charging, so sort it
        "version": CONFIG_VERSION,
        "limit": limit,
        "external_campaigns": sorted(normalized, key=lambda e: e["inventory_dir"]),
    }
    return {
        "path": os.path.abspath(path),
        "canonical": canonical,
        "sha256": _sha256_json(canonical),
        "limit": limit,
        "external_campaigns": normalized,
    }


def lease_charged(lease: Dict[str, Any]) -> bool:
    """Whether a valid lease still occupies a slot (see module docstring)."""
    if lease["child_pgid"] is None:
        return True
    if pid_alive(lease["owner_pid"]) and not lease["owner_released"]:
        return True
    return group_alive(lease["child_pgid"])


def _check_lease(lease: Any, name: str) -> Dict[str, Any]:
    ok = (
        isinstance(lease, dict)
        and set(lease) == LEASE_KEYS
        and type(lease["version"]) is int
        and lease["version"] == LEASE_VERSION
        and f"{lease['lease_id']}.json" == name
        and _positive_int(lease["owner_pid"])
        and isinstance(lease["label"], dict)
        and isinstance(lease["owner_released"], bool)
        and all(
            lease[k] is None or _positive_int(lease[k])
            for k in ("child_pid", "child_pgid")
        )
        and (lease["child_pid"] is None) == (lease["child_pgid"] is None)
        and not (lease["owner_released"] and lease["child_pgid"] is None)
    )
    if not ok:
        raise CapacityError(f"malformed lease file {name!r}")
    return lease


class CapacityBudget:
    """Slot leases shared by every controller using the same state directory."""

    def __init__(
        self,
        config: Dict[str, Any],
        state_dir: str,
        external_probe: ExternalProbe,
    ):
        if not isinstance(state_dir, str) or not os.path.isabs(state_dir):
            raise CapacityError("capacity state dir must be an absolute path")
        self.config = config
        self.limit = config["limit"]
        self.state_dir = os.path.normpath(state_dir)
        self.lease_dir = os.path.join(self.state_dir, "leases")
        self.binding_path = os.path.join(self.state_dir, "config.json")
        self.external_probe = external_probe
        os.makedirs(self.lease_dir, exist_ok=True)
        for path in (self.state_dir, self.lease_dir):
            if os.path.islink(path) or not os.path.isdir(path):
                raise CapacityError(f"capacity state path is not a directory: {path}")
        with self._locked():
            self._check_binding()  # fail at startup, not at first launch

    def describe(self) -> Dict[str, Any]:
        """Static facts for audit records."""
        return {
            "config_path": self.config["path"],
            "config_sha256": self.config["sha256"],
            "state_dir": self.state_dir,
            "limit": self.limit,
        }

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        fd = os.open(
            os.path.join(self.state_dir, "budget.lock"), os.O_RDWR | os.O_CREAT
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _check_binding(self) -> None:
        """Bind state to this config if unbound, else require it. Lock held."""
        path = self.binding_path
        if not os.path.lexists(path):
            if any(not n.startswith(".") for n in os.listdir(self.lease_dir)):
                raise CapacityError(f"capacity state has leases but no {path}")
            _atomic_write_json(
                path,
                {
                    "version": BINDING_VERSION,
                    "config_sha256": self.config["sha256"],
                    "config": self.config["canonical"],
                },
            )
            return
        try:
            if os.path.islink(path):
                raise ValueError("symlink")
            bound = _load_json(path)
        except (OSError, ValueError) as exc:
            raise CapacityError(f"unreadable capacity binding {path}: {exc}") from exc
        ok = (
            isinstance(bound, dict)
            and set(bound) == BINDING_KEYS
            and type(bound["version"]) is int
            and bound["version"] == BINDING_VERSION
            and isinstance(bound["config"], dict)
            and bound["config_sha256"] == _sha256_json(bound["config"])
        )
        if not ok:
            raise CapacityError(f"malformed capacity binding {path}")
        if bound["config_sha256"] != self.config["sha256"]:
            raise CapacityError(
                f"capacity state {self.state_dir} is bound to config "
                f"{bound['config_sha256']} (limit {bound['config'].get('limit')!r}),"
                f" not {self.config['sha256']} from {self.config['path']}",
            )

    def _path(self, lease_id: str) -> str:
        if not isinstance(lease_id, str) or not _LEASE_ID_RE.match(lease_id):
            raise CapacityError(f"invalid lease id {lease_id!r}")
        return os.path.join(self.lease_dir, f"{lease_id}.json")

    def _write(self, lease: Dict[str, Any]) -> None:
        _atomic_write_json(self._path(lease["lease_id"]), lease)

    def _read(self, name: str) -> Dict[str, Any]:
        try:
            lease = _load_json(os.path.join(self.lease_dir, name))
        except (OSError, ValueError) as exc:
            raise CapacityError(f"unreadable lease file {name!r}: {exc}") from exc
        return _check_lease(lease, name)

    def _usage(self) -> Dict[str, Any]:
        """Count charged slots and prune finished leases. Caller holds the lock."""
        charged, pruned = [], []
        for name in sorted(os.listdir(self.lease_dir)):
            if name.startswith("."):
                continue  # temporary file of an interrupted write
            lease = self._read(name)
            if lease_charged(lease):
                charged.append(lease)
            else:
                os.unlink(os.path.join(self.lease_dir, name))
                pruned.append(lease["lease_id"])
        external = []
        for entry in self.config["external_campaigns"]:
            report = self.external_probe(entry)
            if not isinstance(report, dict) or type(report.get("charge")) is not int:
                raise CapacityError(f"external probe gave no charge for {entry}")
            if report["charge"] < 0:
                raise CapacityError(f"negative external charge for {entry}")
            external.append(dict(report, inventory_dir=entry["inventory_dir"]))
        own = sum(1 for lease in charged if lease["owner_pid"] == os.getpid())
        used = len(charged) + sum(report["charge"] for report in external)
        return {
            "limit": self.limit,
            "used": used,
            "leases_charged": len(charged),
            "leases_owned_by_this_process": own,
            "leases_without_child": sum(1 for x in charged if x["child_pgid"] is None),
            "leases_pruned": pruned,
            "external": external,
        }

    def try_acquire(self, label: Dict[str, Any]) -> Tuple[Optional[str], Dict]:
        """Claim one slot if the combined usage is below the limit.

        Args:
            label: JSON-serializable audit label stored in the lease.

        Returns:
            ``(lease_id, usage)`` on success, ``(None, usage)`` when full, or
            ``(None, {"refused": reason})`` when state is invalid.
        """
        with self._locked():
            try:
                self._check_binding()
                usage = self._usage()
            except (CapacityError, OSError) as exc:
                return None, {"refused": str(exc)}
            if usage["used"] >= self.limit:
                return None, usage
            lease = {
                "version": LEASE_VERSION,
                "lease_id": uuid.uuid4().hex,
                "owner_pid": os.getpid(),
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "label": label,
                "child_pid": None,
                "child_pgid": None,
                "owner_released": False,
            }
            self._write(lease)
            return lease["lease_id"], usage

    def bind_child(self, lease_id: str, pid: int, pgid: int) -> None:
        """Record the spawned child so the slot can outlive a crashed owner."""
        with self._locked():
            lease = self._read(os.path.basename(self._path(lease_id)))
            lease.update(child_pid=pid, child_pgid=pgid)
            self._write(_check_lease(lease, f"{lease_id}.json"))

    def release(self, lease_id: str, reaped: bool) -> None:
        """Release a slot owned by this process.

        Args:
            lease_id: Lease to release.
            reaped: True only when no child exists or its group was verified
                gone. Otherwise the lease stays charged until the group is gone.
        """
        with self._locked():
            path = self._path(lease_id)
            if reaped:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(path)
                return
            lease = self._read(os.path.basename(path))
            if lease["child_pgid"] is not None:
                lease["owner_released"] = True
                self._write(lease)
