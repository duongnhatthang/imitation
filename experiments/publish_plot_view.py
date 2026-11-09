"""Export a stable plots-only view without changing rendered figures.

Renderers may retain immutable revision directories on the execution machine.
A publication policy explicitly chooses current figures and routes other
revisions to archive/. With no policy, existing paths are preserved, except
flat coverage images are grouped by environment. Numerical artifacts never
enter the view. Archived revisions with different bytes are refused. Refreshing
an active figure preserves its previous bytes in the archive.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

MEDIA_SUFFIXES = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".webp"}


def sha256(path):
    """Digest the bytes of one figure."""
    with path.open("rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def relative(value):
    """Validate an explicitly relative publication path or prefix."""
    if not isinstance(value, str) or value.startswith("/"):
        raise ValueError("publication paths must be relative")
    parts = value.rstrip("/").split("/") if value else []
    if "//" in value or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("publication paths cannot traverse directories")
    return value


def destination_name(name, policy=None):
    """Resolve an input figure using explicit longest-prefix routes."""
    policy = policy or {}
    fallback = policy.get("fallback", "identity")
    if fallback not in {"identity", "archive"}:
        raise ValueError("unknown publication fallback")
    routes = policy.get("routes", {})
    if not isinstance(routes, dict):
        raise ValueError("routes must be a prefix mapping")
    for prefix, target in routes.items():
        relative(prefix)
        relative(target)
        if prefix and not prefix.endswith("/"):
            raise ValueError("route prefixes must end with a slash")
        if target and not target.endswith("/"):
            raise ValueError("route targets must end with a slash")
    for prefix in sorted(routes, key=len, reverse=True):
        if name.startswith(prefix):
            return relative(routes[prefix] + name[len(prefix) :])
    if fallback == "archive":
        return "archive/" + name
    match = re.fullmatch(r"((?:classical|atari)/coverage)/([^/]+)_seed\d+[^/]*", name)
    if match:
        return match[1] + "/" + match[2] + "/" + Path(name).name
    return name


def check_path(path):
    """Refuse symlinks and non-directory ancestors before publication."""
    if path.is_symlink():
        raise ValueError("publication path is a symlink: " + str(path))
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError("publication ancestor is a symlink: " + str(parent))
        if parent.exists() and not parent.is_dir():
            raise ValueError("publication ancestor is not a directory: " + str(parent))


def publish(source, destination, policy=None):
    """Validate a complete export, then atomically copy each figure.

    Returns the input-relative to output-relative path mapping. Repeating the
    same export is idempotent. Conflicting paths or bytes fail before writes.
    """
    source, destination = Path(source).absolute(), Path(destination).absolute()
    check_path(source)
    check_path(destination)
    if not source.is_dir():
        raise ValueError("publication source does not exist")
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("publication trees must not overlap")
    # Validate even an empty source's policy before any writes.
    destination_name("", policy)
    mapping, planned, targets = {}, [], set()
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("publication source contains a symlink: " + str(path))
        if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        name = path.relative_to(source).as_posix()
        target_name = destination_name(name, policy)
        if target_name in targets:
            raise ValueError("ambiguous publication target: " + target_name)
        targets.add(target_name)
        target = destination / target_name
        check_path(target)
        digest = sha256(path)
        if target.exists() and (not target.is_file() or sha256(target) != digest):
            raise ValueError("publication target has different bytes: " + str(target))
        mapping[name] = target_name
        planned.append((path, target, digest))
    for path, target, digest in planned:
        check_path(target)
        if target.exists():
            if not target.is_file() or sha256(target) != digest:
                raise ValueError("publication target changed: " + str(target))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(
            prefix=".figure-", dir=str(target.parent)
        )
        os.close(descriptor)
        temporary = Path(filename)
        try:
            shutil.copy2(path, temporary)
            if sha256(temporary) != digest or sha256(path) != digest:
                raise ValueError("publication source changed: " + str(path))
            # A hard link publishes without overwriting a concurrent writer.
            os.link(str(temporary), str(target))
        finally:
            temporary.unlink()
    return mapping


def refresh(source, destination, policy=None):
    """Refresh under an exclusive lock shared by all publishers of this view."""
    destination = Path(destination).absolute()
    check_path(destination)
    destination_name("", policy)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / (".plot-view-lock-" + destination.name)
    check_path(lock)
    descriptor = os.open(str(lock), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("publication view is busy: " + str(destination)) from exc
        # Keep the lock inode after unlocking. Unlinking it lets a later writer
        # open a different inode while another process still holds the old one.
        return _refresh_locked(source, destination, policy)


def _refresh_locked(source, destination, policy=None):
    """Replace a complete view while archiving superseded figure bytes.

    Existing archive figures keep their paths. Removed or changed active
    figures move into archive/previous-publications/<sha256>/. Review Markdown
    stays local. A staged export prevents stale version directories surviving
    a policy change; publication is rolled back if the final rename fails.
    """
    source, destination = Path(source).absolute(), Path(destination).absolute()
    check_path(source)
    check_path(destination)
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("publication trees must not overlap")
    destination_name("", policy)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".plot-view-", dir=str(destination.parent)))
    backup = None
    try:
        mapping = publish(source, staging, policy)
        before = {}
        if destination.exists():
            for old in sorted(destination.rglob("*")):
                check_path(old)
                if not old.is_file():
                    continue
                name = old.relative_to(destination).as_posix()
                before[name] = sha256(old)
                if old.name == ".DS_Store":
                    continue
                if old.suffix.lower() not in MEDIA_SUFFIXES | {".md"}:
                    raise ValueError("unexpected non-plot file in publication: " + name)
                target = staging / name
                if target.exists() and sha256(target) == before[name]:
                    continue
                if name.startswith("archive/") and target.exists():
                    raise ValueError(
                        "archived publication has different bytes: " + name
                    )
                if old.suffix.lower() in MEDIA_SUFFIXES and not name.startswith(
                    "archive/"
                ):
                    target = (
                        staging / "archive/previous-publications" / before[name] / name
                    )
                if target.exists():
                    if sha256(target) != before[name]:
                        raise ValueError("conflicting archive bytes: " + name)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old, target)
                if sha256(target) != before[name] or sha256(old) != before[name]:
                    raise ValueError("existing publication changed during export")
            after = {
                p.relative_to(destination).as_posix(): sha256(p)
                for p in destination.rglob("*")
                if p.is_file()
            }
            if before != after:
                raise ValueError("existing publication changed during export")
            backup = Path(
                tempfile.mkdtemp(
                    prefix=".plot-view-backup-", dir=str(destination.parent)
                )
            )
            backup.rmdir()
            os.rename(str(destination), str(backup))
        try:
            os.rename(str(staging), str(destination))
        except BaseException:
            if backup is not None:
                os.rename(str(backup), str(destination))
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return mapping
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    """Export figures selected by an optional private publication policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text()) if args.policy else None
    exporter = refresh if args.refresh else publish
    mapping = exporter(args.source, args.destination, policy)
    print(json.dumps({"figures": len(mapping), "destination": str(args.destination)}))


if __name__ == "__main__":
    main()
