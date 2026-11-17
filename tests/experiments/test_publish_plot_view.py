"""Publication paths must stay clean without losing older figures."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_publisher():
    spec = importlib.util.spec_from_file_location(
        "publish_plot_view", ROOT / "experiments/publish_plot_view.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(root, name, content=b"figure"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_routes_current_figures_archives_revisions_and_excludes_raw(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "view"
    current = "classical/learning_curves/concise-v2/CartPole-v1.png"
    put(source, current, b"current")
    put(source, "classical/learning_curves/concise-v1/CartPole-v1.png", b"old")
    put(source, "classical/coverage/reviewed-v2/CartPole-v1/seed3100.png")
    put(source, "classical/coverage/reviewed-v2/metrics.json", b"raw")
    put(source, "classical/coverage/reviewed-v2/cache.npz", b"raw")
    policy = {
        "fallback": "archive",
        "routes": {
            "classical/learning_curves/concise-v2/": "classical/learning_curves/",
            "classical/coverage/reviewed-v2/": "classical/coverage/",
        },
    }
    publisher = load_publisher()
    mapping = publisher.publish(source, destination, policy)
    assert (
        destination / "classical/learning_curves/CartPole-v1.png"
    ).read_bytes() == b"current"
    assert (
        destination / "archive/classical/learning_curves/concise-v1/CartPole-v1.png"
    ).read_bytes() == b"old"
    assert (destination / "classical/coverage/CartPole-v1/seed3100.png").exists()
    assert len(mapping) == 3
    assert not list(destination.rglob("*.json"))
    assert not list(destination.rglob("*.npz"))
    assert publisher.publish(source, destination, policy) == mapping


def test_native_coverage_groups_by_environment(tmp_path):
    source = tmp_path / "source"
    put(source, "classical/coverage/CartPole-v1_seed3100.png")
    put(source, "classical/coverage/CartPole-v1_seed3100_coverage_diff.png")
    destination = tmp_path / "view"
    load_publisher().publish(source, destination)
    assert (
        len(list((destination / "classical/coverage/CartPole-v1").glob("*.png"))) == 2
    )
    assert not list((destination / "classical/coverage").glob("*.png"))


def test_conflict_is_detected_before_any_publication(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "view"
    put(source, "a.png", b"new")
    put(source, "b.png", b"second")
    put(destination, "a.png", b"existing")
    with pytest.raises(ValueError, match="different bytes"):
        load_publisher().publish(source, destination)
    assert (destination / "a.png").read_bytes() == b"existing"
    assert not (destination / "b.png").exists()


@pytest.mark.parametrize("prefix", ["../escape/", "/absolute/"])
def test_policy_cannot_escape_destination(tmp_path, prefix):
    source = tmp_path / "source"
    put(source, "a.png")
    with pytest.raises(ValueError):
        load_publisher().publish(source, tmp_path / "view", {"routes": {"": prefix}})
    assert not (tmp_path / "view").exists()


def test_destination_symlink_is_refused(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "view"
    put(source, "classical/learning_curves/a.png")
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "classical").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_publisher().publish(source, destination)
    assert not list(outside.iterdir())


def test_repeated_pull_does_not_recreate_version_directories(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "view"
    put(source, "classical/learning_curves/concise-v2/a.png")
    policy = {
        "routes": {
            "classical/learning_curves/concise-v2/": "classical/learning_curves/"
        }
    }
    publisher = load_publisher()
    publisher.publish(source, destination, policy)
    publisher.publish(source, destination, policy)
    assert sorted(
        p.name for p in (destination / "classical/learning_curves").iterdir()
    ) == ["a.png"]


def test_policy_path_alias_is_rejected_before_any_write(tmp_path):
    source = tmp_path / "source"
    put(source, "one/a.png", b"one")
    put(source, "two/a.png", b"two")
    destination = tmp_path / "view"
    with pytest.raises(ValueError):
        load_publisher().publish(
            source, destination, {"routes": {"one/": "same/", "two/": "same//"}}
        )
    assert not destination.exists()


def test_refresh_changes_current_view_without_losing_old_bytes(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "view"
    put(source, "revision1/a.png", b"old")
    put(source, "revision2/a.png", b"new")
    publisher = load_publisher()
    publisher.publish(source, destination)
    publisher.refresh(
        source,
        destination,
        {"fallback": "archive", "routes": {"revision2/": "classical/learning_curves/"}},
    )
    assert (destination / "classical/learning_curves/a.png").read_bytes() == b"new"
    assert not (destination / "revision1").exists()
    assert not (destination / "revision2").exists()
    assert b"old" in [p.read_bytes() for p in (destination / "archive").rglob("*.png")]
    before = {
        str(p.relative_to(destination)): p.read_bytes()
        for p in destination.rglob("*.png")
    }
    publisher.refresh(
        source,
        destination,
        {"fallback": "archive", "routes": {"revision2/": "classical/learning_curves/"}},
    )
    assert before == {
        str(p.relative_to(destination)): p.read_bytes()
        for p in destination.rglob("*.png")
    }


def test_refresh_preserves_local_archive_and_review_index(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "view"
    put(source, "classical/learning_curves/a.png", b"new")
    put(destination, "classical/learning_curves/a.png", b"old")
    put(destination, "archive/local-only.pdf", b"historical")
    put(destination, "README.md", b"review links")
    load_publisher().refresh(source, destination)
    assert (destination / "README.md").read_bytes() == b"review links"
    assert (destination / "archive/local-only.pdf").read_bytes() == b"historical"
    assert (destination / "classical/learning_curves/a.png").read_bytes() == b"new"
    assert b"old" in [p.read_bytes() for p in (destination / "archive").rglob("*.png")]


def test_refresh_refuses_to_rewrite_an_archived_revision(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "view"
    put(source, "archive/reviewed-v1/a.png", b"changed")
    put(source, "classical/learning_curves/a.png", b"new")
    put(destination, "archive/reviewed-v1/a.png", b"immutable")
    put(destination, "classical/learning_curves/a.png", b"current")
    with pytest.raises(ValueError, match="archive"):
        load_publisher().refresh(source, destination)
    assert (destination / "archive/reviewed-v1/a.png").read_bytes() == b"immutable"
    assert (destination / "classical/learning_curves/a.png").read_bytes() == b"current"
    assert not [
        p
        for p in tmp_path.glob(".plot-view-*")
        if not p.name.startswith(".plot-view-lock-")
    ]


def test_concurrent_refresh_fails_closed_without_losing_first_publication(
    tmp_path, monkeypatch
):
    first, second, destination = (
        tmp_path / "first",
        tmp_path / "second",
        tmp_path / "view",
    )
    put(first, "classical/learning_curves/a.png", b"first")
    put(second, "classical/learning_curves/b.png", b"second")
    publisher = load_publisher()
    actual_publish = publisher.publish
    entered, release = threading.Event(), threading.Event()
    errors = []

    def pause_after_real_export(*args, **kwargs):
        result = actual_publish(*args, **kwargs)
        if not entered.is_set():
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test publication barrier timed out")
        return result

    monkeypatch.setattr(publisher, "publish", pause_after_real_export)

    def run_first():
        try:
            publisher.refresh(first, destination)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run_first)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(ValueError, match="busy"):
            publisher.refresh(second, destination)
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    assert not errors
    assert (destination / "classical/learning_curves/a.png").read_bytes() == b"first"


@pytest.mark.skipif(
    not (ROOT / "experiments/sync_results.sh").is_file(),
    reason="requires the private plots-only sync wrapper",
)
def test_sync_wrapper_exports_and_repeats_without_revision_or_raw_leaks(tmp_path):
    """Exercise the real wrapper, exporter and rsync against a local fake SSH."""
    project, remote = tmp_path / "project", tmp_path / "remote"
    for root in [project, remote]:
        (root / "experiments").mkdir(parents=True)
        shutil.copy2(ROOT / "experiments/publish_plot_view.py", root / "experiments")
    shutil.copy2(ROOT / "experiments/sync_results.sh", project / "experiments")
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    source = remote / "experiments/runs/trial/plots"
    put(source, "classical/learning_curves/concise-v2/a.png")
    put(source, "classical/learning_curves/old/a.png", b"old")
    put(source, "classical/learning_curves/concise-v2/result.json", b"raw")
    policy = remote / "experiments/.local/plot-layouts/trial.json"
    policy.parent.mkdir(parents=True)
    routes = {"classical/learning_curves/concise-v2/": "classical/learning_curves/"}
    policy.write_text(
        json.dumps(
            {
                "fallback": "archive",
                "routes": routes,
            }
        )
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(
        "#!/usr/bin/env python3\nimport os,sys\na=sys.argv[1:]\n"
        "while a and a[0]=='-o': a=a[2:]\n"
        "os.execv('/bin/bash',['bash','-c',a[1]])\n"
    )
    ssh.chmod(0o755)
    rsync = bin_dir / "rsync"
    real_rsync = shutil.which("rsync")
    rsync.write_text(
        "#!/usr/bin/env python3\nimport os,sys\n"
        "a=[x.split(':',1)[1] if x.startswith('local:') else x for x in sys.argv[1:]]\n"
        f"os.execv({real_rsync!r},[{real_rsync!r}]+a)\n"
    )
    rsync.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        SYNC_SERVER="local",
        SYNC_REMOTE_DIR=str(remote),
        SYNC_REMOTE_PYTHON=sys.executable,
    )
    for _ in range(2):
        result = subprocess.run(
            ["bash", str(project / "experiments/sync_results.sh"), "pull", "trial"],
            env=env,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    view = project / "experiments/plots/trial"
    assert (view / "classical/learning_curves/a.png").read_bytes() == b"figure"
    assert (view / "archive/classical/learning_curves/old/a.png").read_bytes() == b"old"
    assert not list(view.rglob("*.json"))
    assert not (view / "classical/learning_curves/concise-v2").exists()
