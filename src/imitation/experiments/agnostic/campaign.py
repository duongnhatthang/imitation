"""Finite manifest runner for authorized agnostic DAgger campaign stages.

This is a small stdlib POSIX queue runner, not a production scheduler. It runs a
fixed list of independent jobs (one stage manifest, no job dependencies) with
bounded concurrency, per-job timeouts, a stage deadline, and an absolute hard
cap, and it records every attempt in a durable JSON inventory.

Usage::

    python -m imitation.experiments.agnostic.campaign \\
        --manifest stage.json --inventory-dir /absolute/state \\
        [--workers 4] [--retry-infrastructure JOB_ID] [--kill-grace 30]
    python -m imitation.experiments.agnostic.campaign --hash-source /abs/src

Assumptions (not enforced, the operator must provide them):

* ``source_root`` is an immutable snapshot. The runner hashes it at startup,
  before every launch, and before accepting each result. That detects mutation
  (and then stops the stage) but cannot prevent it.
* The controller itself is protected externally (for example it runs in tmux
  on a machine nobody else administers). A SIGKILL of the controller cannot be
  made safe: running attempts are recorded durably before launch, and a
  restart marks them ``interrupted``. It never marks them successful and never
  signals a recorded PID or process group, since those may have been reused.
* Jobs stay in the process group created for them. A descendant that calls
  ``setsid``/``setpgid`` escapes group cleanup.

Stale artifact rule: before every attempt, an existing file at
``expected_result`` is renamed (same directory, so the rename is atomic) to
``<result>.campaign-prior.<attempt_id>`` and recorded in the attempt. The
attempt can therefore only succeed by writing a fresh result, and prior or
partial outputs are preserved rather than deleted.

Retry rule: failed, timed-out, cancelled, and interrupted rows are never rerun
automatically. ``--retry-infrastructure JOB_ID`` allows one extra attempt per
job (two attempts total) for infrastructure outcomes only: timeout, deadline,
signal, controller crash, launch error, or a nonzero exit whose result file is
missing, malformed, or does not report a failure status. Quality failures are
never retryable here: exit 0 with a result that fails validation, or a nonzero
exit whose result JSON reports ``status`` in ``QUALITY_FAILURE_STATUSES``
(the producer contract for a scientific or quality gate failure). The failed
artifact stays in place for inspection.

Exit codes: 0 when every job has a validated complete result, 1 when any job
does not, 2 for refused input (invalid manifest, incompatible inventory,
source mismatch, ineligible retry, lock held), 128+N after signal N.
"""

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Sequence

MANIFEST_VERSION = 1
INVENTORY_VERSION = 1
HARD_CAP = "2026-10-07T01:01:00Z"
DEFAULT_WORKERS = 4
MAX_WORKERS = 64
DEFAULT_TIMEOUT_SECONDS = 7200.0
DEFAULT_KILL_GRACE_SECONDS = 30.0
MAX_KILL_GRACE_SECONDS = 60.0
# Bound on waiting for killed group members to disappear (for example zombies
# in a container whose init never reaps). The attempt is finalized with a note.
UNREAPED_GROUP_TIMEOUT_SECONDS = 10.0
MAX_RESULT_BYTES = 16 * 1024 * 1024
POLL_INTERVAL_SECONDS = 0.1

JOB_KINDS = ("expert", "learner")
JOB_STATES = (
    "pending",
    "running",
    "complete",
    "failed",
    "timed_out",
    "cancelled",
    "interrupted",
)
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
MANIFEST_KEYS = {
    "version",
    "campaign_id",
    "source_root",
    "source_sha256",
    "stage_deadline",
    "overall_deadline",
    "jobs",
}
JOB_REQUIRED_KEYS = {"id", "kind", "argv", "cwd", "expected_result"}
JOB_OPTIONAL_KEYS = {"expected_fields", "timeout_seconds"}
# Failure kinds that an operator may retry once. Quality failures
# ("validation", "reported_failure") and "source_changed" are absent.
RETRYABLE_FAILURE_KINDS = {"exit_nonzero", "exit_signal", "launch_error"}
# Result statuses a producer writes to declare a scientific/quality failure.
# A missing or malformed result (including a non-string status) after a
# nonzero exit stays an operator-retryable infrastructure outcome.
QUALITY_FAILURE_STATUSES = {"failed", "quality_failed", "rejected", "not_qualified"}
RETRYABLE_STATES = {"timed_out", "cancelled", "interrupted"}

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CampaignError(Exception):
    """Refused input or state. The runner exits 2 without launching jobs."""


# ---------------------------------------------------------------------------
# Hashing and small helpers
# ---------------------------------------------------------------------------


def hash_source(root: str) -> str:
    """Deterministic hash of the sorted ``.py`` files below ``root``.

    Each file contributes its POSIX relative path and the SHA-256 of its
    content. ``__pycache__`` directories are skipped. Symlinked directories or
    ``.py`` files are rejected so the hash cannot silently omit or alias code.

    Args:
        root: Absolute path of the source tree.

    Returns:
        Hex SHA-256 digest.

    Raises:
        CampaignError: If root is not an absolute directory, contains symlinks
            relevant to the hash, or contains no ``.py`` files.
    """
    if not os.path.isabs(root) or not os.path.isdir(root):
        raise CampaignError(f"source root must be an absolute directory: {root!r}")
    entries = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            if os.path.islink(os.path.join(dirpath, name)):
                raise CampaignError(f"symlinked directory in source tree: {name!r}")
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            if os.path.islink(path) or not os.path.isfile(path):
                raise CampaignError(f"non-regular .py file in source tree: {path!r}")
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            entries.append((rel, _file_sha256(path)))
    if not entries:
        raise CampaignError(f"source tree has no .py files: {root!r}")
    digest = hashlib.sha256()
    for rel, content_hash in sorted(entries):
        digest.update(rel.encode("utf-8", "surrogateescape"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name}")


def _load_json_strict(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_constant)


def parse_deadline(value: Any, name: str) -> datetime.datetime:
    """Parse an ISO 8601 timestamp with an explicit UTC offset.

    Python 3.8 ``fromisoformat`` does not accept a trailing ``Z``, so it is
    rewritten to ``+00:00`` first. Naive timestamps are rejected.

    Args:
        value: Timestamp string such as ``2026-10-02T01:01:00Z``.
        name: Field name used in error messages.

    Returns:
        Timezone-aware datetime in UTC.

    Raises:
        CampaignError: If the value is malformed or has no offset.
    """
    if not isinstance(value, str) or "T" not in value:
        raise CampaignError(f"{name} must be an ISO 8601 date-time string")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError as exc:
        raise CampaignError(f"{name} is malformed: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CampaignError(f"{name} must include a UTC offset or Z: {value!r}")
    return parsed.astimezone(datetime.timezone.utc)


def _iso(ts: datetime.datetime) -> str:
    return ts.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _json_equal(a: Any, b: Any) -> bool:
    """Exact JSON equality that does not conflate bool, int, and float."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return a == b


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _require_abs(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise CampaignError(f"{name} must be a nonempty string")
    if not os.path.isabs(value):
        raise CampaignError(f"{name} must be an absolute path: {value!r}")
    return os.path.normpath(value)


def _validate_timeout(value: Any, job_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignError(f"job {job_id}: timeout_seconds must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise CampaignError(f"job {job_id}: timeout_seconds must be finite and > 0")
    return value


def _validate_job(raw: Any, index: int) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise CampaignError(f"jobs[{index}] must be an object")
    keys = set(raw)
    missing = JOB_REQUIRED_KEYS - keys
    unknown = keys - JOB_REQUIRED_KEYS - JOB_OPTIONAL_KEYS
    if missing or unknown:
        raise CampaignError(
            f"jobs[{index}]: missing keys {sorted(missing)}, "
            f"unknown keys {sorted(unknown)}",
        )
    job_id = raw["id"]
    if not isinstance(job_id, str) or not _ID_RE.match(job_id):
        raise CampaignError(f"jobs[{index}]: invalid job id {job_id!r}")
    if raw["kind"] not in JOB_KINDS:
        raise CampaignError(f"job {job_id}: kind must be one of {JOB_KINDS}")
    argv = raw["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(a, str) and a and "\0" not in a for a in argv)
    ):
        raise CampaignError(f"job {job_id}: argv must be a nonempty list of strings")
    if not os.path.isabs(argv[0]):
        raise CampaignError(f"job {job_id}: argv[0] must be an absolute executable")
    cwd = _require_abs(raw["cwd"], f"job {job_id} cwd")
    if not os.path.isdir(cwd):
        raise CampaignError(f"job {job_id}: cwd is not a directory: {cwd!r}")
    result = _require_abs(raw["expected_result"], f"job {job_id} expected_result")
    if os.path.isdir(result):
        raise CampaignError(f"job {job_id}: expected_result is a directory: {result!r}")
    fields = raw.get("expected_fields", {})
    if not isinstance(fields, dict) or not all(isinstance(k, str) for k in fields):
        raise CampaignError(f"job {job_id}: expected_fields must be an object")
    fields = dict(fields)
    if fields.setdefault("status", "complete") != "complete":
        raise CampaignError(f"job {job_id}: expected_fields.status must be complete")
    timeout = _validate_timeout(
        raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
        job_id,
    )
    return {
        "id": job_id,
        "kind": raw["kind"],
        "argv": list(argv),
        "cwd": cwd,
        "expected_result": result,
        "expected_fields": fields,
        "timeout_seconds": timeout,
    }


def load_manifest(path: str) -> Dict[str, Any]:
    """Load and validate a stage manifest.

    Args:
        path: Manifest JSON path.

    Returns:
        Normalized manifest with ``manifest_sha256`` and effective deadlines.

    Raises:
        CampaignError: If the manifest is invalid.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = _load_json_strict(f.read())
    except (OSError, ValueError) as exc:
        raise CampaignError(f"cannot read manifest {path!r}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CampaignError("manifest must be a JSON object")
    if set(raw) != MANIFEST_KEYS:
        raise CampaignError(
            f"manifest keys must be exactly {sorted(MANIFEST_KEYS)}, got {sorted(raw)}",
        )
    if isinstance(raw["version"], bool) or raw["version"] != MANIFEST_VERSION:
        raise CampaignError(f"unsupported manifest version {raw['version']!r}")
    campaign_id = raw["campaign_id"]
    if not isinstance(campaign_id, str) or not _ID_RE.match(campaign_id):
        raise CampaignError(f"invalid campaign_id {campaign_id!r}")
    source_root = _require_abs(raw["source_root"], "source_root")
    source_sha = raw["source_sha256"]
    if not isinstance(source_sha, str) or not _SHA256_RE.match(source_sha):
        raise CampaignError("source_sha256 must be 64 lowercase hex characters")
    hard_cap = parse_deadline(HARD_CAP, "HARD_CAP")
    overall = min(parse_deadline(raw["overall_deadline"], "overall_deadline"), hard_cap)
    stage = min(parse_deadline(raw["stage_deadline"], "stage_deadline"), overall)
    if not isinstance(raw["jobs"], list) or not raw["jobs"]:
        raise CampaignError("jobs must be a nonempty list")
    jobs = [_validate_job(job, i) for i, job in enumerate(raw["jobs"])]
    ids = [job["id"] for job in jobs]
    if len(set(ids)) != len(ids):
        raise CampaignError("duplicate job ids")
    results = [job["expected_result"] for job in jobs]
    if len(set(results)) != len(results):
        raise CampaignError("duplicate expected_result paths")
    canonical = canonical_json(raw)
    return {
        "campaign_id": campaign_id,
        "source_root": source_root,
        "source_sha256": source_sha,
        "canonical": canonical,
        "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "stage_deadline": stage,
        "overall_deadline": overall,
        "jobs": jobs,
    }


# ---------------------------------------------------------------------------
# Inventory persistence
# ---------------------------------------------------------------------------


def _atomic_write_json(path: str, obj: Any) -> None:
    directory = os.path.dirname(path)
    tmp = os.path.join(
        directory,
        f".{os.path.basename(path)}.{os.getpid()}.{uuid.uuid4().hex}.tmp",
    )
    data = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


@contextlib.contextmanager
def _exclusive_lock(inventory_dir: str) -> Iterator[None]:
    fd = os.open(os.path.join(inventory_dir, "inventory.lock"), os.O_RDWR | os.O_CREAT)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise CampaignError("another runner holds the inventory lock") from exc
        yield
    finally:
        os.close(fd)


def _new_inventory(manifest: Dict[str, Any]) -> Dict[str, Any]:
    now = _iso(_utcnow())
    return {
        "version": INVENTORY_VERSION,
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "source_root": manifest["source_root"],
        "source_sha256": manifest["source_sha256"],
        "hard_cap": HARD_CAP,
        "effective_stage_deadline": _iso(manifest["stage_deadline"]),
        "effective_overall_deadline": _iso(manifest["overall_deadline"]),
        "created_at": now,
        "updated_at": now,
        "controller_runs": [],
        "jobs": {
            job["id"]: {
                "kind": job["kind"],
                "expected_result": job["expected_result"],
                "state": "pending",
                "state_reason": None,
                "result_sha256": None,
                "attempts": [],
            }
            for job in manifest["jobs"]
        },
    }


def _check_inventory_compatible(inv: Any, manifest: Dict[str, Any]) -> None:
    if not isinstance(inv, dict) or inv.get("version") != INVENTORY_VERSION:
        raise CampaignError("existing inventory has an unsupported format")
    for key in ("campaign_id", "manifest_sha256", "source_root", "source_sha256"):
        if inv.get(key) != manifest[key]:
            raise CampaignError(f"existing inventory {key} does not match manifest")
    jobs = inv.get("jobs")
    if not isinstance(jobs, dict) or set(jobs) != {j["id"] for j in manifest["jobs"]}:
        raise CampaignError("existing inventory job set does not match manifest")
    for job_id, row in jobs.items():
        if not isinstance(row, dict) or row.get("state") not in JOB_STATES:
            raise CampaignError(f"existing inventory row {job_id} is malformed")


# ---------------------------------------------------------------------------
# Result validation
# ---------------------------------------------------------------------------


def _read_result(path: str, details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the result JSON object, or None with ``details["error"]`` set."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        details["error"] = "result file missing"
        return None
    if not stat.S_ISREG(st.st_mode):
        details["error"] = "result path is not a regular file"
        return None
    if st.st_size > MAX_RESULT_BYTES:
        details["error"] = "result file too large"
        return None
    with open(path, "rb") as f:
        data = f.read()
    details["sha256"] = hashlib.sha256(data).hexdigest()
    try:
        payload = _load_json_strict(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        details["error"] = f"result is not valid JSON: {exc}"
        return None
    if not isinstance(payload, dict):
        details["error"] = "result JSON is not an object"
        return None
    return payload


def validate_result(job: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the result file of a job against its expected fields.

    Args:
        job: Normalized manifest job.

    Returns:
        Details dict with ``ok`` and either ``sha256`` or ``error``.
    """
    details: Dict[str, Any] = {"path": job["expected_result"], "ok": False}
    payload = _read_result(job["expected_result"], details)
    if payload is None:
        return details
    mismatched = [
        key
        for key, expected in sorted(job["expected_fields"].items())
        if key not in payload or not _json_equal(payload[key], expected)
    ]
    if mismatched:
        details["error"] = f"expected fields mismatched: {mismatched}"
        details["mismatched_fields"] = mismatched
        return details
    details["ok"] = True
    return details


# ---------------------------------------------------------------------------
# Process group handling
# ---------------------------------------------------------------------------


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The id now belongs to someone else's group. Never signal it.
        return False
    return True


def _signal_group(pgid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


class _Running:
    """Controller-side state of one live attempt."""

    def __init__(self, job, attempt, proc, started_mono, deadline, deadline_reason):
        self.job = job
        self.attempt = attempt
        self.proc = proc
        # start_new_session=True makes the leader's pid the group id.
        self.pgid = proc.pid
        self.started_mono = started_mono
        self.deadline = deadline
        self.deadline_reason = deadline_reason
        self.leader_exit: Optional[int] = None
        self.stop_reason: Optional[str] = None
        self.cleanup_reason: Optional[str] = None
        self.kill_at: Optional[float] = None
        self.kill_sent_mono: Optional[float] = None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class CampaignRunner:
    """Runs one stage manifest against one inventory directory."""

    def __init__(
        self,
        manifest: Dict[str, Any],
        inventory_dir: str,
        workers: int = DEFAULT_WORKERS,
        retry_ids: Sequence[str] = (),
        kill_grace: float = DEFAULT_KILL_GRACE_SECONDS,
        poll_interval: Optional[float] = None,
    ):
        if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
            raise CampaignError(f"workers must be an integer in [1, {MAX_WORKERS}]")
        if not (math.isfinite(kill_grace) and 0 < kill_grace <= MAX_KILL_GRACE_SECONDS):
            raise CampaignError(f"kill grace must be in (0, {MAX_KILL_GRACE_SECONDS}]")
        if not os.path.isabs(inventory_dir):
            raise CampaignError("inventory dir must be an absolute path")
        self.manifest = manifest
        self.jobs = {job["id"]: job for job in manifest["jobs"]}
        self.order = [job["id"] for job in manifest["jobs"]]
        self.inventory_dir = os.path.normpath(inventory_dir)
        self.inventory_path = os.path.join(self.inventory_dir, "inventory.json")
        self.workers = workers
        self.retry_ids = list(dict.fromkeys(retry_ids))
        self.kill_grace = float(kill_grace)
        self.poll_interval = (
            POLL_INTERVAL_SECONDS if poll_interval is None else poll_interval
        )
        self.inv: Dict[str, Any] = {}
        self.running: Dict[str, _Running] = {}
        self.stop_signal: Optional[int] = None
        self.abort_reason: Optional[str] = None
        self.run_record: Dict[str, Any] = {}
        # Deadlines are mapped to the monotonic clock once, at startup.
        wall, mono = time.time(), time.monotonic()
        self.stage_deadline = mono + (manifest["stage_deadline"].timestamp() - wall)
        self.overall_deadline = mono + (manifest["overall_deadline"].timestamp() - wall)

    # -- inventory ----------------------------------------------------------

    def _save(self) -> None:
        self.inv["updated_at"] = _iso(_utcnow())
        _atomic_write_json(self.inventory_path, self.inv)

    def _row(self, job_id: str) -> Dict[str, Any]:
        return self.inv["jobs"][job_id]

    def _set_state(self, job_id: str, state: str, reason: Optional[str]) -> None:
        row = self._row(job_id)
        row["state"] = state
        row["state_reason"] = reason

    def _load_or_create_inventory(self) -> None:
        if os.path.lexists(self.inventory_path):
            try:
                with open(self.inventory_path, "r", encoding="utf-8") as f:
                    inv = _load_json_strict(f.read())
            except (OSError, ValueError) as exc:
                raise CampaignError(f"cannot read existing inventory: {exc}") from exc
            _check_inventory_compatible(inv, self.manifest)
            self.inv = inv
        else:
            self.inv = _new_inventory(self.manifest)
            # Keep the exact manifest the inventory hash refers to.
            _atomic_write_json(
                os.path.join(self.inventory_dir, "manifest.canonical.json"),
                json.loads(self.manifest["canonical"]),
            )

    def _recover_and_revalidate(self) -> None:
        """Mark crashed attempts interrupted and recheck completed results."""
        for job_id in self.order:
            row = self._row(job_id)
            if row["state"] == "running":
                attempt = row["attempts"][-1] if row["attempts"] else None
                note = "controller ended while attempt was running"
                if attempt is not None:
                    attempt["state"] = "interrupted"
                    attempt["ended_at"] = None
                    attempt["interrupted_note"] = note
                    pgid = attempt.get("pgid")
                    attempt["recorded_group_present_at_recovery"] = (
                        _group_alive(pgid) if isinstance(pgid, int) else None
                    )
                self._set_state(job_id, "interrupted", note)
                print(f"[campaign] {job_id}: marked interrupted (controller crash)")
            elif row["state"] == "cancelled" and not row["attempts"]:
                # Never started, so rescheduling cannot duplicate any work.
                self._set_state(job_id, "pending", None)
            elif row["state"] == "complete":
                details = validate_result(self.jobs[job_id])
                if not details["ok"] or details.get("sha256") != row["result_sha256"]:
                    reason = "completed result changed or invalid at resume"
                    row["resume_check"] = details
                    self._set_state(job_id, "failed", reason)
                    print(f"[campaign] {job_id}: {reason}; failing closed")

    def _apply_retries(self) -> None:
        for job_id in self.retry_ids:
            if job_id not in self.jobs:
                raise CampaignError(f"retry requested for unknown job {job_id!r}")
            row = self._row(job_id)
            attempts = row["attempts"]
            if len(attempts) >= 2:
                raise CampaignError(f"job {job_id} already used its one retry")
            if not attempts:
                raise CampaignError(f"job {job_id} has no attempt to retry")
            last = attempts[-1]
            eligible = row["state"] in RETRYABLE_STATES or (
                row["state"] == "failed"
                and last.get("failure_kind") in RETRYABLE_FAILURE_KINDS
            )
            if not eligible:
                raise CampaignError(
                    f"job {job_id} in state {row['state']} "
                    f"({last.get('failure_kind')}) is not an infrastructure outcome",
                )
            pgid = last.get("pgid")
            if isinstance(pgid, int) and _group_alive(pgid):
                raise CampaignError(
                    f"job {job_id}: recorded process group {pgid} still exists; "
                    "verify and stop it manually before retrying",
                )
        for job_id in self.retry_ids:
            self._set_state(job_id, "pending", "infrastructure retry requested")

    # -- launching ----------------------------------------------------------

    def _job_env(self, job: Dict[str, Any], attempt_id: str) -> Dict[str, str]:
        env = dict(os.environ)
        for name in THREAD_ENV_VARS:
            env[name] = "1"
        env["CAMPAIGN_ID"] = self.manifest["campaign_id"]
        env["CAMPAIGN_JOB_ID"] = job["id"]
        env["CAMPAIGN_ATTEMPT_ID"] = attempt_id
        return env

    def _archive_prior_result(self, job, attempt_id) -> Optional[Dict[str, Any]]:
        path = job["expected_result"]
        if not os.path.lexists(path):
            return None
        if os.path.isdir(path) and not os.path.islink(path):
            raise OSError(f"expected_result became a directory: {path!r}")
        archived = f"{path}.campaign-prior.{attempt_id}"
        regular = stat.S_ISREG(os.lstat(path).st_mode)
        sha = _file_sha256(path) if regular else None
        os.replace(path, archived)
        return {"from": path, "to": archived, "sha256": sha}

    def _launch(self, job_id: str) -> None:
        job = self.jobs[job_id]
        row = self._row(job_id)
        number = len(row["attempts"]) + 1
        attempt_id = f"{job_id}-a{number}-{uuid.uuid4().hex[:12]}"
        log_dir = os.path.join(self.inventory_dir, "logs", job_id)
        os.makedirs(log_dir, exist_ok=True)
        stdout_log = os.path.join(log_dir, f"{attempt_id}.stdout.log")
        stderr_log = os.path.join(log_dir, f"{attempt_id}.stderr.log")
        start_mono = time.monotonic()
        timeout_deadline = start_mono + job["timeout_seconds"]
        deadline, reason = min(
            (timeout_deadline, "job_timeout"),
            (self.stage_deadline, "stage_deadline"),
            (self.overall_deadline, "overall_deadline"),
        )
        attempt: Dict[str, Any] = {
            "attempt_id": attempt_id,
            "number": number,
            "infrastructure_retry": number > 1,
            "state": "running",
            "started_at": _iso(_utcnow()),
            "ended_at": None,
            "elapsed_seconds": None,
            "controller_pid": os.getpid(),
            "pid": None,
            "pgid": None,
            "argv": job["argv"],
            "cwd": job["cwd"],
            "stdout_log": stdout_log,
            "stderr_log": stderr_log,
            "exit_code": None,
            "failure_kind": None,
            "termination": None,
            "archived_prior_result": None,
            "validation": None,
        }
        row["attempts"].append(attempt)
        self._set_state(job_id, "running", None)
        try:
            attempt["archived_prior_result"] = self._archive_prior_result(
                job,
                attempt_id,
            )
            # Persist before Popen so a crash leaves a durable running row.
            self._save()
            with open(stdout_log, "wb") as out, open(stderr_log, "wb") as err:
                proc = subprocess.Popen(
                    job["argv"],
                    cwd=job["cwd"],
                    env=self._job_env(job, attempt_id),
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    shell=False,
                    start_new_session=True,
                    close_fds=True,
                )
        except OSError as exc:
            attempt.update(
                state="failed",
                failure_kind="launch_error",
                ended_at=_iso(_utcnow()),
                elapsed_seconds=time.monotonic() - start_mono,
                validation={"ok": False, "error": f"launch failed: {exc}"},
            )
            self._set_state(job_id, "failed", "launch_error")
            self._save()
            print(f"[campaign] {job_id}: launch failed: {exc}")
            return
        # Track the live group before any fallible bookkeeping, so an error
        # below still reaches _kill_owned_groups.
        self.running[job_id] = _Running(
            job,
            attempt,
            proc,
            start_mono,
            deadline,
            reason,
        )
        attempt["pid"] = attempt["pgid"] = proc.pid
        self._save()
        print(f"[campaign] {job_id}: launched {attempt_id} pid={proc.pid}")

    # -- monitoring ---------------------------------------------------------

    def _grace_deadline(self, now: float) -> float:
        remaining = max(0.0, min(self.stage_deadline, self.overall_deadline) - now)
        return now + min(self.kill_grace, remaining)

    def _begin_stop(self, entry: _Running, reason: str, now: float) -> None:
        if entry.stop_reason is not None or entry.cleanup_reason is not None:
            # Already terminating: a later deadline or signal can only shorten.
            entry.kill_at = min(entry.kill_at, self._grace_deadline(now))
            return
        entry.stop_reason = reason
        entry.kill_at = self._grace_deadline(now)
        _signal_group(entry.pgid, signal.SIGTERM)

    def _tick(self, entry: _Running, now: float) -> bool:
        """Advance one attempt. Returns True when it has been finalized."""
        if entry.leader_exit is None:
            code = entry.proc.poll()
            if code is not None:
                entry.leader_exit = code
        if entry.leader_exit is None and now >= entry.deadline:
            self._begin_stop(entry, entry.deadline_reason, now)
        group_alive = _group_alive(entry.pgid)
        if (
            entry.leader_exit is not None
            and group_alive
            and entry.stop_reason is None
            and entry.cleanup_reason is None
        ):
            # Leader exited but descendants remain: clean before accepting.
            entry.cleanup_reason = "leader_exited_with_live_group"
            entry.kill_at = self._grace_deadline(now)
            _signal_group(entry.pgid, signal.SIGTERM)
        if group_alive and entry.kill_at is not None and now >= entry.kill_at:
            if entry.kill_sent_mono is None:
                entry.kill_sent_mono = now
            _signal_group(entry.pgid, signal.SIGKILL)
        if entry.leader_exit is None:
            return False
        unreaped = False
        if group_alive:
            if entry.kill_sent_mono is None:
                return False
            if now - entry.kill_sent_mono < UNREAPED_GROUP_TIMEOUT_SECONDS:
                return False
            unreaped = True
        self._finalize(entry, unreaped)
        return True

    def _finalize(self, entry: _Running, unreaped: bool) -> None:
        job, attempt = entry.job, entry.attempt
        job_id = job["id"]
        code = entry.leader_exit
        attempt["ended_at"] = _iso(_utcnow())
        attempt["elapsed_seconds"] = time.monotonic() - entry.started_mono
        attempt["exit_code"] = code
        attempt["termination"] = {
            "stop_reason": entry.stop_reason,
            "cleanup_reason": entry.cleanup_reason,
            "term_sent": entry.kill_at is not None,
            "kill_sent": entry.kill_sent_mono is not None,
            "group_members_unreaped": unreaped,
        }
        if entry.stop_reason is not None:
            state = (
                "cancelled" if entry.stop_reason.startswith("cancel") else "timed_out"
            )
            attempt["state"] = state
            self._set_state(job_id, state, entry.stop_reason)
        elif code != 0:
            kind = "exit_signal" if code < 0 else "exit_nonzero"
            payload = _read_result(job["expected_result"], {})
            reported = payload.get("status") if payload is not None else None
            if not isinstance(reported, str):
                reported = None  # malformed status: treated like a malformed result
            if reported in QUALITY_FAILURE_STATUSES:
                kind = "reported_failure"
            attempt.update(state="failed", failure_kind=kind, reported_status=reported)
            self._set_state(job_id, "failed", kind)
        else:
            self._accept_or_reject(job, attempt)
        self._save()
        print(f"[campaign] {job_id}: {attempt['state']} ({attempt['attempt_id']})")

    def _accept_or_reject(self, job: Dict[str, Any], attempt: Dict[str, Any]) -> None:
        job_id = job["id"]
        if not self._source_intact():
            attempt.update(
                state="failed",
                failure_kind="source_changed",
                validation={"ok": False, "error": "source tree changed during run"},
            )
            self._set_state(job_id, "failed", "source_changed")
            self.abort_reason = "cancelled_source_changed"
            return
        details = validate_result(job)
        attempt["validation"] = details
        if details["ok"]:
            attempt["state"] = "complete"
            self._row(job_id)["result_sha256"] = details["sha256"]
            self._set_state(job_id, "complete", None)
        else:
            attempt.update(state="failed", failure_kind="validation")
            self._set_state(job_id, "failed", "validation")

    def _cancel_pending(self, pending: List[str], reason: str) -> None:
        for job_id in pending:
            self._set_state(job_id, "cancelled", reason)
            print(f"[campaign] {job_id}: cancelled before launch ({reason})")
        pending.clear()
        self._save()

    def _handle_signal(self, signum, frame) -> None:
        if self.stop_signal is None:
            self.stop_signal = signum

    def _source_intact(self) -> bool:
        try:
            current = hash_source(self.manifest["source_root"])
        except CampaignError:
            return False
        return current == self.manifest["source_sha256"]

    def _enforce_stop(self, pending: List[str], check_source: bool = False) -> bool:
        """Stop everything if a signal, abort, or deadline applies.

        Args:
            pending: Jobs not yet launched; cancelled in place on stop.
            check_source: Also rehash the source tree (done before each launch).

        Returns:
            True if the stage is stopping and nothing more may launch.
        """
        if self.abort_reason is None and check_source and not self._source_intact():
            self.abort_reason = "cancelled_source_changed"
        # Read the clock after the (possibly slow) hash, so deadlines are fresh.
        now = time.monotonic()
        if self.stop_signal is not None:
            reason = f"cancelled_signal_{self.stop_signal}"
        elif self.abort_reason is not None:
            reason = self.abort_reason
        elif now >= self.overall_deadline:
            reason = "overall_deadline"
        elif now >= self.stage_deadline:
            reason = "stage_deadline"
        else:
            return False
        if pending:
            self._cancel_pending(pending, reason)
        for entry in self.running.values():
            self._begin_stop(entry, reason, now)
        return True

    def _loop(self) -> None:
        pending = [j for j in self.order if self._row(j)["state"] == "pending"]
        while pending or self.running:
            self._enforce_stop(pending)
            now = time.monotonic()
            for job_id in list(self.running):
                if self._tick(self.running[job_id], now):
                    del self.running[job_id]
            # Re-check right before each launch: a tick above may have found
            # a source change, and signals or deadlines may have arrived.
            while pending and len(self.running) < self.workers:
                if self._enforce_stop(pending, check_source=True):
                    break
                self._launch(pending.pop(0))
            if pending or self.running:
                time.sleep(self.poll_interval)

    def _kill_owned_groups(self) -> None:
        """Bounded best-effort cleanup after an unexpected controller error."""
        entries = list(self.running.values())
        for entry in entries:
            _signal_group(entry.pgid, signal.SIGTERM)
        kill_at = self._grace_deadline(time.monotonic())
        while time.monotonic() < kill_at and any(
            e.proc.poll() is None or _group_alive(e.pgid) for e in entries
        ):
            time.sleep(0.05)
        for entry in entries:
            _signal_group(entry.pgid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                entry.proc.wait(timeout=UNREAPED_GROUP_TIMEOUT_SECONDS)

    def run(self) -> int:
        """Run the stage. Returns the process exit code."""
        os.makedirs(self.inventory_dir, exist_ok=True)
        with _exclusive_lock(self.inventory_dir):
            actual = hash_source(self.manifest["source_root"])
            if actual != self.manifest["source_sha256"]:
                raise CampaignError(
                    f"source hash mismatch: manifest {self.manifest['source_sha256']}"
                    f" but tree hashes to {actual}",
                )
            self._load_or_create_inventory()
            self._recover_and_revalidate()
            self._apply_retries()
            self.run_record = {
                "controller_pid": os.getpid(),
                "started_at": _iso(_utcnow()),
                "ended_at": None,
                "workers": self.workers,
                "retry_requests": self.retry_ids,
                "exit_code": None,
            }
            self.inv["controller_runs"].append(self.run_record)
            self._save()
            handled = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
            previous = {s: signal.signal(s, self._handle_signal) for s in handled}
            try:
                self._loop()
            except BaseException:
                # Rows stay "running" on disk and become interrupted on restart.
                self._kill_owned_groups()
                raise
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            incomplete = [j for j in self.order if self._row(j)["state"] != "complete"]
            if self.stop_signal is not None:
                code = 128 + self.stop_signal
            else:
                code = 1 if incomplete else 0
            self.run_record["ended_at"] = _iso(_utcnow())
            self.run_record["exit_code"] = code
            self.run_record["incomplete_jobs"] = incomplete
            self._save()
            print(f"[campaign] finished: {len(incomplete)} incomplete, exit {code}")
            return code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    # Range checks live in CampaignRunner, so bad values exit 2 as refusals.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", help="stage manifest JSON")
    mode.add_argument("--hash-source", metavar="ROOT", help="print source hash")
    parser.add_argument("--inventory-dir", help="absolute inventory directory")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--retry-infrastructure",
        metavar="JOB_ID",
        action="append",
        default=[],
        help="allow one retry of an infrastructure outcome (repeatable)",
    )
    parser.add_argument(
        "--kill-grace",
        type=float,
        default=DEFAULT_KILL_GRACE_SECONDS,
        help="seconds between TERM and KILL of a process group (<= 60)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.hash_source is not None:
            print(hash_source(os.path.normpath(args.hash_source)))
            return 0
        if not args.inventory_dir:
            raise CampaignError("--inventory-dir is required with --manifest")
        runner = CampaignRunner(
            load_manifest(args.manifest),
            args.inventory_dir,
            workers=args.workers,
            retry_ids=args.retry_infrastructure,
            kill_grace=args.kill_grace,
        )
        return runner.run()
    except CampaignError as exc:
        print(f"[campaign] refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
