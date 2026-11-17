"""Tests for the finite agnostic campaign queue runner.

Every job is a tiny local Python subprocess. Nothing here contacts a remote
machine. The runner module is loaded from its file path so that these tests do
not depend on heavier sibling agnostic modules; its only sibling import is the
stdlib ``resource_budget`` module, so ``src`` must be importable.
"""

import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import textwrap
import time

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CAMPAIGN_PATH = (
    REPO_ROOT / "src" / "imitation" / "experiments" / "agnostic" / "campaign.py"
)
SYNC_SCRIPT = REPO_ROOT / "experiments" / "sync_results.sh"
FAR_FUTURE = "2099-01-01T00:00:00Z"

_spec = importlib.util.spec_from_file_location(
    "agnostic_campaign_under_test", CAMPAIGN_PATH
)
campaign = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(campaign)

PRODUCER = textwrap.dedent(
    """
    import json, os, signal, subprocess, sys, time

    mode, out = sys.argv[1], sys.argv[2]
    extra = sys.argv[3:]

    def write(payload):
        with open(out, "w") as f:
            json.dump(payload, f)

    def ok_payload():
        return {
            "status": "complete",
            "protocol": "p1",
            "attempt": os.environ["CAMPAIGN_ATTEMPT_ID"],
            "threads": [os.environ[k] for k in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")],
        }

    def stubborn_child(pid_file):
        code = (
            "import os, signal, sys, time\\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
            "open(sys.argv[1] + '.tmp', 'w').write(str(os.getpid()))\\n"
            "os.rename(sys.argv[1] + '.tmp', sys.argv[1])\\n"
            "time.sleep(60)\\n"
        )
        subprocess.Popen([sys.executable, "-c", code, pid_file])
        while not os.path.exists(pid_file):
            time.sleep(0.01)

    if mode == "ok":
        write(ok_payload())
    elif mode == "track":
        log = extra[0]
        with open(log, "a") as f:
            f.write("start %s %.6f\\n" % (os.environ["CAMPAIGN_JOB_ID"], time.time()))
        time.sleep(0.3)
        with open(log, "a") as f:
            f.write("end %s %.6f\\n" % (os.environ["CAMPAIGN_JOB_ID"], time.time()))
        write(ok_payload())
    elif mode == "missing":
        pass
    elif mode == "malformed":
        with open(out, "w") as f:
            f.write("{not json")
    elif mode == "failed_status":
        write({"status": "failed", "protocol": "p1"})
    elif mode == "mismatch":
        payload = ok_payload()
        payload["protocol"] = "other"
        write(payload)
    elif mode == "exit1":
        write(ok_payload())
        sys.exit(1)
    elif mode == "exit1_status":
        write({"status": json.loads(extra[0]), "gate": "expert_quality"})
        sys.exit(1)
    elif mode == "exit1_missing":
        sys.exit(1)
    elif mode == "stubborn":
        stubborn_child(extra[0])
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        write({"status": "partial"})
        time.sleep(60)
    elif mode == "fork_and_exit":
        stubborn_child(extra[0])
        write(ok_payload())
    elif mode == "sleep_unless_flag":
        if not os.path.exists(extra[0]):
            write({"status": "partial"})
            time.sleep(60)
        write(ok_payload())
    elif mode in ("mutate_source", "mutate_source_exit1"):
        with open(extra[0], "a") as f:
            f.write("# mutated\\n")
        write(ok_payload())
        sys.exit(1 if mode.endswith("exit1") else 0)
    elif mode == "count_leases":
        leases = [n for n in os.listdir(extra[0]) if not n.startswith(".")]
        with open(extra[1], "a") as f:
            f.write("%s %d\\n" % (os.environ["CAMPAIGN_JOB_ID"], len(leases)))
        time.sleep(0.2)
        write(ok_payload())
    else:
        sys.exit(9)
    """,
)


def _iso(dt):
    return dt.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _in(seconds):
    return _iso(
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(seconds=seconds)
    )


def _pid_gone(pid, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def _read_pid(path, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if os.path.exists(path):
            return int(pathlib.Path(path).read_text())
        time.sleep(0.02)
    raise AssertionError(f"pid file never appeared: {path}")


@pytest.fixture(autouse=True)
def _far_hard_cap(monkeypatch):
    # The real cap is exercised in test_hard_cap_always_applies.
    monkeypatch.setattr(campaign, "HARD_CAP", FAR_FUTURE)


class Env:
    """Tiny immutable-source fixture plus manifest and runner helpers."""

    def __init__(self, root):
        self.root = root
        self.src = root / "src"
        (self.src / "pkg" / "__pycache__").mkdir(parents=True)
        (self.src / "pkg" / "__init__.py").write_text("")
        (self.src / "pkg" / "mod.py").write_text("X = 1\n")
        (self.src / "pkg" / "__pycache__" / "mod.cpython-38.pyc").write_bytes(b"junk")
        self.producer = root / "producer.py"
        self.producer.write_text(PRODUCER)
        self.results = root / "results"
        self.results.mkdir()
        self.inventory = root / "state"
        self.manifest_path = root / "stage.json"

    def job(self, job_id, mode, *extra, timeout=30, fields=None, kind="learner"):
        return {
            "id": job_id,
            "kind": kind,
            "argv": [sys.executable, str(self.producer), mode, self.result(job_id)]
            + [str(e) for e in extra],
            "cwd": str(self.root),
            "expected_result": self.result(job_id),
            "expected_fields": (
                {"status": "complete", "protocol": "p1"} if fields is None else fields
            ),
            "timeout_seconds": timeout,
        }

    def result(self, job_id):
        return str(self.results / f"{job_id}.json")

    def write_manifest(self, jobs, stage=FAR_FUTURE, overall=FAR_FUTURE, **overrides):
        manifest = {
            "version": 1,
            "campaign_id": "pilot-test",
            "source_root": str(self.src),
            "source_sha256": campaign.hash_source(str(self.src)),
            "stage_deadline": stage,
            "overall_deadline": overall,
            "jobs": jobs,
        }
        manifest.update(overrides)
        self.manifest_path.write_text(json.dumps(manifest))
        return manifest

    def run(self, *extra_args, workers=2, grace="0.5"):
        argv = [
            "--manifest",
            str(self.manifest_path),
            "--inventory-dir",
            str(self.inventory),
            "--workers",
            str(workers),
            "--kill-grace",
            grace,
        ] + list(extra_args)
        return campaign.main(argv)

    def inventory_json(self):
        return json.loads((self.inventory / "inventory.json").read_text())

    def spawn_runner(self, *extra_args):
        """Run the controller in a separate process (for signal tests)."""
        code = (
            "import importlib.util, sys\n"
            "spec = importlib.util.spec_from_file_location('c', sys.argv[1])\n"
            "m = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(m)\n"
            f"m.HARD_CAP = {FAR_FUTURE!r}\n"
            "sys.exit(m.main(sys.argv[2:]))\n"
        )
        argv = [
            sys.executable,
            "-c",
            code,
            str(CAMPAIGN_PATH),
            "--manifest",
            str(self.manifest_path),
            "--inventory-dir",
            str(self.inventory),
            "--kill-grace",
            "0.5",
        ] + list(extra_args)
        return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SECONDS", 0.02)
    return Env(tmp_path)


# ---------------------------------------------------------------------------
# Source hashing
# ---------------------------------------------------------------------------


def test_hash_source_deterministic_and_sensitive(env):
    first = campaign.hash_source(str(env.src))
    assert first == campaign.hash_source(str(env.src))
    (env.src / "pkg" / "__pycache__" / "other.py").write_text("ignored = True\n")
    (env.src / "pkg" / "notes.txt").write_text("not python\n")
    assert campaign.hash_source(str(env.src)) == first
    (env.src / "pkg" / "mod.py").write_text("X = 2\n")
    changed = campaign.hash_source(str(env.src))
    assert changed != first
    (env.src / "pkg" / "mod.py").rename(env.src / "pkg" / "mod2.py")
    assert campaign.hash_source(str(env.src)) not in (first, changed)


def test_hash_source_cli_and_rejections(env, capsys):
    assert campaign.main(["--hash-source", str(env.src)]) == 0
    assert capsys.readouterr().out.strip() == campaign.hash_source(str(env.src))
    empty = env.root / "empty"
    (empty / "__pycache__").mkdir(parents=True)
    (empty / "__pycache__" / "x.py").write_text("")
    assert campaign.main(["--hash-source", str(empty)]) == 2
    with pytest.raises(campaign.CampaignError):
        campaign.hash_source("relative/path")


# ---------------------------------------------------------------------------
# Manifest and CLI validation
# ---------------------------------------------------------------------------


def test_workers_default_and_positive_validation(env):
    args = campaign.build_parser().parse_args(
        ["--manifest", "m", "--inventory-dir", "/x"]
    )
    assert args.workers == 4
    assert args.kill_grace <= 60
    env.write_manifest([env.job("a", "ok")])
    for flag, bad in [("--workers", w) for w in ("0", "-1", "1000")] + [
        ("--kill-grace", g) for g in ("0", "61", "nan", "inf")
    ]:
        assert env.run(flag, bad) == 2, (flag, bad)
    assert not env.inventory.exists()
    for bad in ("abc", "1.5"):
        with pytest.raises(SystemExit):
            env.run("--workers", bad)


def test_default_timeout_and_status_mandatory(env):
    job = env.job("a", "ok", fields={"protocol": "p1"})
    del job["timeout_seconds"]
    env.write_manifest([job])
    manifest = campaign.load_manifest(str(env.manifest_path))
    assert manifest["jobs"][0]["timeout_seconds"] == 7200
    assert manifest["jobs"][0]["expected_fields"]["status"] == "complete"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m, e: m["jobs"].append(
            dict(m["jobs"][0], expected_result=e.result("z"))
        ),
        lambda m, e: m["jobs"][0].update(id="../escape"),
        lambda m, e: m["jobs"][0].update(argv=[]),
        lambda m, e: m["jobs"][0].update(argv=["python", "x.py"]),
        lambda m, e: m["jobs"][0].update(argv="/bin/echo hi"),
        lambda m, e: m["jobs"][0].update(timeout_seconds=float("inf")),
        lambda m, e: m["jobs"][0].update(timeout_seconds=0),
        lambda m, e: m["jobs"][0].update(timeout_seconds="10"),
        lambda m, e: m["jobs"][0].update(timeout_seconds=True),
        lambda m, e: m["jobs"][0].update(expected_result=str(e.results)),
        lambda m, e: m["jobs"][0].update(expected_result="rel.json"),
        lambda m, e: m["jobs"][0].update(expected_fields={"status": "failed"}),
        lambda m, e: m["jobs"][0].update(kind="other"),
        lambda m, e: m["jobs"][0].update(cwd="relative"),
        lambda m, e: m["jobs"][0].update(unexpected=1),
        lambda m, e: m["jobs"].append(dict(m["jobs"][0], id="b")),
        lambda m, e: m.update(stage_deadline="2026-10-02"),
        lambda m, e: m.update(stage_deadline="2026-10-02T01:01:00"),
        lambda m, e: m.update(overall_deadline="soon"),
        lambda m, e: m.update(version=2),
        lambda m, e: m.update(source_sha256="abc"),
        lambda m, e: m.update(extra_key=True),
        lambda m, e: m.update(jobs=[]),
    ],
)
def test_invalid_manifests_rejected(env, mutate):
    manifest = env.write_manifest([env.job("a", "ok")])
    mutate(manifest, env)
    env.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(campaign.CampaignError):
        campaign.load_manifest(str(env.manifest_path))
    assert env.run() == 2
    assert not env.inventory.exists()


def test_hard_cap_always_applies(env, monkeypatch):
    monkeypatch.setattr(campaign, "HARD_CAP", "2026-10-07T01:01:00Z")
    cap = datetime.datetime(2026, 10, 7, 1, 1, tzinfo=datetime.timezone.utc)
    env.write_manifest(
        [env.job("a", "ok")],
        stage="2030-01-01T00:00:00Z",
        overall="2030-01-01T00:00:00Z",
    )
    manifest = campaign.load_manifest(str(env.manifest_path))
    assert manifest["overall_deadline"] == cap
    assert manifest["stage_deadline"] == cap
    env.write_manifest(
        [env.job("a", "ok")],
        stage="2026-10-02T01:01:00Z",
        overall="2026-10-09T00:00:00+02:00",
    )
    manifest = campaign.load_manifest(str(env.manifest_path))
    assert manifest["overall_deadline"] == cap
    assert manifest["stage_deadline"] == datetime.datetime(
        2026, 10, 2, 1, 1, tzinfo=datetime.timezone.utc
    )


# ---------------------------------------------------------------------------
# Running jobs
# ---------------------------------------------------------------------------


def test_multiworker_queue_completes(env):
    log = env.root / "track.log"
    jobs = [env.job(f"job-{i:03d}", "track", log, kind="expert") for i in range(5)]
    env.write_manifest(jobs)
    assert env.run(workers=2) == 0
    inv = env.inventory_json()
    assert inv["source_sha256"] == campaign.hash_source(str(env.src))
    assert len(inv["manifest_sha256"]) == 64
    attempt_ids = set()
    for i in range(5):
        row = inv["jobs"][f"job-{i:03d}"]
        assert row["state"] == "complete"
        assert len(row["attempts"]) == 1
        attempt = row["attempts"][0]
        attempt_ids.add(attempt["attempt_id"])
        assert attempt["exit_code"] == 0
        assert attempt["validation"]["ok"]
        assert attempt["elapsed_seconds"] > 0
        payload = json.loads(pathlib.Path(env.result(f"job-{i:03d}")).read_text())
        assert payload["attempt"] == attempt["attempt_id"]
        assert payload["threads"] == ["1", "1", "1", "1"]
        assert os.path.exists(attempt["stdout_log"])
        assert os.path.exists(attempt["stderr_log"])
    assert len(attempt_ids) == 5
    # Hand-count concurrency from the start/end events.
    events = sorted(
        (float(t), kind)
        for kind, _, t in (line.split() for line in log.read_text().splitlines())
    )
    live = peak = 0
    for _, kind in events:
        live += 1 if kind == "start" else -1
        peak = max(peak, live)
    assert peak == 2
    assert len(events) == 10
    # Resuming a fully complete campaign launches nothing and succeeds.
    assert env.run() == 0
    assert all(len(r["attempts"]) == 1 for r in env.inventory_json()["jobs"].values())


@pytest.mark.parametrize("mode", ["missing", "malformed", "failed_status", "mismatch"])
def test_exit_zero_without_valid_result_is_not_complete(env, mode):
    env.write_manifest([env.job("a", mode)])
    assert env.run() == 1
    row = env.inventory_json()["jobs"]["a"]
    assert row["state"] == "failed"
    assert row["attempts"][0]["exit_code"] == 0
    assert row["attempts"][0]["failure_kind"] == "validation"
    assert row["result_sha256"] is None


@pytest.mark.parametrize(
    "mode, extra, kind, retry_code",
    [
        ("exit1", [], "exit_nonzero", 1),  # retried, fails again
        ("exit1_missing", [], "exit_nonzero", 1),
        ("exit1_status", ['["failed"]'], "exit_nonzero", 1),  # malformed status
        ("exit1_status", ['"failed"'], "reported_failure", 2),  # quality: refused
        ("exit1_status", ['"not_qualified"'], "reported_failure", 2),
    ],
)
def test_nonzero_exit_retry_contract(env, mode, extra, kind, retry_code):
    env.write_manifest([env.job("a", mode, *extra), env.job("b", "ok")])
    assert env.run() == 1
    inv = env.inventory_json()
    assert inv["jobs"]["b"]["state"] == "complete"  # controller kept going
    row = inv["jobs"]["a"]
    assert row["state"] == "failed"
    assert row["attempts"][0]["failure_kind"] == kind
    before = (
        pathlib.Path(env.result("a")).read_bytes() if mode != "exit1_missing" else None
    )
    assert env.run("--retry-infrastructure", "a") == retry_code
    attempts = env.inventory_json()["jobs"]["a"]["attempts"]
    assert len(attempts) == (2 if retry_code == 1 else 1)
    if kind == "reported_failure":
        assert attempts[0]["reported_status"] == json.loads(extra[0])
        assert pathlib.Path(env.result("a")).read_bytes() == before


def test_prior_artifact_not_trusted_and_preserved(env):
    prior = {"status": "complete", "protocol": "p1"}
    pathlib.Path(env.result("a")).write_text(json.dumps(prior))
    env.write_manifest([env.job("a", "missing")])
    assert env.run() == 1
    attempt = env.inventory_json()["jobs"]["a"]["attempts"][0]
    assert attempt["failure_kind"] == "validation"
    archived = attempt["archived_prior_result"]
    assert archived["from"] == env.result("a")
    assert json.loads(pathlib.Path(archived["to"]).read_text()) == prior
    assert not os.path.exists(env.result("a"))


def test_resume_refuses_manifest_or_source_mismatch(env):
    env.write_manifest([env.job("a", "ok")])
    assert env.run() == 0
    before = (env.inventory / "inventory.json").read_text()
    env.write_manifest([env.job("a", "ok", timeout=31)])
    assert env.run() == 2
    (env.src / "pkg" / "mod.py").write_text("X = 3\n")
    env.write_manifest([env.job("a", "ok")])  # new source hash, new manifest hash
    assert env.run() == 2
    manifest = json.loads(env.manifest_path.read_text())
    manifest["source_sha256"] = "0" * 64
    env.manifest_path.write_text(json.dumps(manifest))
    assert env.run() == 2
    assert (env.inventory / "inventory.json").read_text() == before


def test_completed_artifact_changed_on_resume_fails_closed(env):
    env.write_manifest([env.job("a", "ok")])
    assert env.run() == 0
    payload = json.loads(pathlib.Path(env.result("a")).read_text())
    payload["extra"] = 1
    pathlib.Path(env.result("a")).write_text(json.dumps(payload))
    tampered = pathlib.Path(env.result("a")).read_text()
    assert env.run() == 1
    row = env.inventory_json()["jobs"]["a"]
    assert row["state"] == "failed"
    assert "changed" in row["state_reason"]
    assert len(row["attempts"]) == 1
    assert pathlib.Path(env.result("a")).read_text() == tampered
    assert env.run("--retry-infrastructure", "a") == 2


def test_timeout_kills_stubborn_descendant(env):
    pid_file = env.root / "grandchild.pid"
    env.write_manifest([env.job("a", "stubborn", pid_file, timeout=1)])
    start = time.monotonic()
    assert env.run() == 1
    assert time.monotonic() - start < 15
    grandchild = _read_pid(pid_file)
    assert _pid_gone(grandchild)
    row = env.inventory_json()["jobs"]["a"]
    attempt = row["attempts"][0]
    assert row["state"] == "timed_out"
    assert attempt["termination"]["stop_reason"] == "job_timeout"
    assert attempt["termination"]["kill_sent"]
    assert json.loads(pathlib.Path(env.result("a")).read_text()) == {
        "status": "partial"
    }


def test_leader_exit_cleans_group_before_accepting(env):
    pid_file = env.root / "grandchild.pid"
    env.write_manifest([env.job("a", "fork_and_exit", pid_file)])
    assert env.run() == 0
    assert _pid_gone(_read_pid(pid_file))
    attempt = env.inventory_json()["jobs"]["a"]["attempts"][0]
    assert attempt["state"] == "complete"
    assert attempt["termination"]["cleanup_reason"] == "leader_exited_with_live_group"
    assert attempt["termination"]["kill_sent"]


def test_stage_deadline_during_job_kills_group(env):
    pid_file = env.root / "grandchild.pid"
    env.write_manifest(
        [env.job("a", "stubborn", pid_file, timeout=7200), env.job("b", "ok")],
        stage=_in(2.5),
    )
    assert env.run(workers=1) == 1
    assert _pid_gone(_read_pid(pid_file))
    inv = env.inventory_json()
    assert inv["jobs"]["a"]["state"] == "timed_out"
    assert inv["jobs"]["a"]["state_reason"] == "stage_deadline"
    assert inv["jobs"]["b"]["state"] == "cancelled"
    assert inv["jobs"]["b"]["attempts"] == []


def test_expired_deadline_launches_nothing(env):
    marker = env.root / "never.pid"
    env.write_manifest(
        [env.job("a", "fork_and_exit", marker)], stage="2020-01-01T00:00:00Z"
    )
    assert env.run() == 1
    row = env.inventory_json()["jobs"]["a"]
    assert row["state"] == "cancelled"
    assert row["attempts"] == []
    assert not marker.exists()
    assert not (env.inventory / "logs").exists()


def test_deadline_passing_during_prelaunch_hash_launches_nothing(env, monkeypatch):
    env.write_manifest([env.job("a", "ok")], stage=_in(1.0))
    runner = campaign.CampaignRunner(
        campaign.load_manifest(str(env.manifest_path)),
        str(env.inventory),
    )

    def slow_intact():
        # The deadline expires while the pre-launch hash is running.
        time.sleep(max(0.0, runner.stage_deadline - time.monotonic()) + 0.05)
        return True

    monkeypatch.setattr(runner, "_source_intact", slow_intact)
    assert runner.run() == 1
    row = env.inventory_json()["jobs"]["a"]
    assert (row["state"], row["state_reason"]) == ("cancelled", "stage_deadline")
    assert row["attempts"] == []


def test_sigterm_cancels_job_group(env):
    pid_file = env.root / "grandchild.pid"
    env.write_manifest(
        [env.job("a", "stubborn", pid_file, timeout=7200), env.job("b", "ok")]
    )
    proc = env.spawn_runner("--workers", "1")
    try:
        grandchild = _read_pid(pid_file)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=20) == 128 + signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
    assert _pid_gone(grandchild)
    inv = env.inventory_json()
    leader = inv["jobs"]["a"]["attempts"][0]["pid"]
    assert _pid_gone(leader)
    assert inv["jobs"]["a"]["state"] == "cancelled"
    assert inv["jobs"]["b"]["state"] == "cancelled"
    assert inv["jobs"]["b"]["attempts"] == []


def test_controller_crash_marks_interrupted_then_single_retry(env):
    flag = env.root / "flag"
    env.write_manifest([env.job("a", "sleep_unless_flag", flag, timeout=7200)])
    proc = env.spawn_runner()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            inv_path = env.inventory / "inventory.json"
            if inv_path.exists():
                attempts = json.loads(inv_path.read_text())["jobs"]["a"]["attempts"]
                if attempts and attempts[0]["pid"] and os.path.exists(env.result("a")):
                    break
            time.sleep(0.02)
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        proc.stdout.close()
    pgid = json.loads((env.inventory / "inventory.json").read_text())["jobs"]["a"][
        "attempts"
    ][0]["pgid"]
    try:
        # Restart: the orphan is left alone and the row becomes interrupted.
        assert env.run() == 1
        row = env.inventory_json()["jobs"]["a"]
        assert row["state"] == "interrupted"
        assert row["attempts"][0]["recorded_group_present_at_recovery"] is True
        os.killpg(pgid, 0)  # still alive: the runner did not kill a recorded pgid
        # Retry is refused while the recorded group still exists.
        assert env.run("--retry-infrastructure", "a") == 2
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    assert _pid_gone(pgid)
    flag.write_text("")
    assert env.run("--retry-infrastructure", "a") == 0
    row = env.inventory_json()["jobs"]["a"]
    assert row["state"] == "complete"
    first, second = row["attempts"]
    assert first["state"] == "interrupted"
    assert second["infrastructure_retry"] is True
    assert first["attempt_id"] != second["attempt_id"]
    partial = pathlib.Path(second["archived_prior_result"]["to"])
    assert json.loads(partial.read_text()) == {"status": "partial"}
    assert os.path.exists(first["stdout_log"]) and os.path.exists(second["stdout_log"])


def test_retry_cap_and_no_automatic_rerun(env):
    flag = env.root / "flag"
    env.write_manifest([env.job("a", "sleep_unless_flag", flag, timeout=0.5)])
    assert env.run() == 1
    assert env.inventory_json()["jobs"]["a"]["state"] == "timed_out"
    assert env.run() == 1  # no automatic rerun
    assert len(env.inventory_json()["jobs"]["a"]["attempts"]) == 1
    assert env.run("--retry-infrastructure", "a") == 1
    row = env.inventory_json()["jobs"]["a"]
    assert [a["state"] for a in row["attempts"]] == ["timed_out", "timed_out"]
    assert env.run("--retry-infrastructure", "a") == 2
    assert env.run("--retry-infrastructure", "unknown") == 2
    assert len(env.inventory_json()["jobs"]["a"]["attempts"]) == 2
    logs = sorted((env.inventory / "logs" / "a").iterdir())
    assert len(logs) == 4


def test_quality_failure_cannot_be_retried(env):
    env.write_manifest([env.job("a", "mismatch"), env.job("b", "ok")])
    assert env.run() == 1
    assert env.run("--retry-infrastructure", "a") == 2
    assert env.run("--retry-infrastructure", "b") == 2
    assert len(env.inventory_json()["jobs"]["a"]["attempts"]) == 1


@pytest.mark.parametrize(
    "mode, kind",
    [
        ("mutate_source", "source_changed"),  # caught at acceptance
        ("mutate_source_exit1", "exit_nonzero"),  # caught by the pre-launch hash
    ],
)
def test_source_mutation_launches_nothing_more(env, mode, kind):
    target = env.src / "pkg" / "mod.py"
    marker = env.root / "b_started.pid"
    env.write_manifest(
        [env.job("a", mode, target), env.job("b", "fork_and_exit", marker)]
    )
    assert env.run(workers=1) == 1
    inv = env.inventory_json()
    assert inv["jobs"]["a"]["attempts"][0]["failure_kind"] == kind
    b = inv["jobs"]["b"]
    assert (b["state"], b["state_reason"]) == ("cancelled", "cancelled_source_changed")
    assert b["attempts"] == []
    assert not marker.exists()
    assert not (env.inventory / "logs" / "b").exists()
    assert env.run() == 2  # source no longer matches the manifest


def test_save_failure_after_popen_kills_child(env, monkeypatch):
    env.write_manifest([env.job("a", "stubborn", env.root / "gc.pid", timeout=7200)])
    runner = campaign.CampaignRunner(
        campaign.load_manifest(str(env.manifest_path)),
        str(env.inventory),
        kill_grace=0.5,
    )
    real_save, seen = runner._save, {}

    def failing_save():
        attempts = runner.inv["jobs"]["a"]["attempts"]
        if attempts and attempts[-1]["pid"] is not None:
            seen["pid"] = attempts[-1]["pid"]
            raise OSError("injected save failure")
        real_save()

    monkeypatch.setattr(runner, "_save", failing_save)
    with pytest.raises(OSError, match="injected"):
        runner.run()
    # _pid_gone also proves the leader was reaped: a zombie still answers kill 0.
    assert _pid_gone(seen["pid"], timeout=1.0)
    if (env.root / "gc.pid").exists():
        assert _pid_gone(_read_pid(env.root / "gc.pid"))
    assert env.run() == 1
    assert env.inventory_json()["jobs"]["a"]["state"] == "interrupted"


def test_lock_prevents_second_controller(env):
    env.write_manifest([env.job("a", "ok")])
    env.inventory.mkdir()
    with campaign._exclusive_lock(str(env.inventory)):
        assert env.run() == 2


# ---------------------------------------------------------------------------
# Shared capacity (opt-in)
# ---------------------------------------------------------------------------

budget_mod = campaign.resource_budget


def _peak_concurrency(log):
    events = sorted(
        (float(t), kind)
        for kind, _, t in (line.split() for line in log.read_text().splitlines())
    )
    live = peak = 0
    for _, kind in events:
        live += 1 if kind == "start" else -1
        peak = max(peak, live)
    return peak


def _capacity_config(path, limit, externals=()):
    path.write_text(
        json.dumps(
            {"version": 1, "limit": limit, "external_campaigns": list(externals)},
        ),
    )
    return str(path)


def _budget(tmp_path, limit, externals=()):
    config = budget_mod.load_config(
        _capacity_config(tmp_path / "capacity.json", limit, externals),
        campaign.MAX_WORKERS,
    )
    return budget_mod.CapacityBudget(
        config,
        str(tmp_path / "budget"),
        campaign.inventory_capacity_charge,
    )


def _external(root, name, workers, states, controller_pid=None, ended=False, pgids=()):
    """Fabricate another campaign's inventory directory (read-only for budget)."""
    directory = root / name
    directory.mkdir()
    manifest = {
        "campaign_id": name,
        "jobs": [{"id": f"j{i}"} for i in range(len(states))],
    }
    (directory / "manifest.canonical.json").write_text(json.dumps(manifest))
    sha = hashlib.sha256(campaign.canonical_json(manifest).encode()).hexdigest()
    pgids = dict(pgids)
    jobs = {
        f"j{i}": {
            "state": state,
            "attempts": (
                []
                if state == "pending"
                else [{"state": state, "pgid": pgids.get(f"j{i}")}]
            ),
        }
        for i, state in enumerate(states)
    }
    run = {
        "controller_pid": controller_pid or os.getpid(),
        "workers": workers,
        "ended_at": _in(0) if ended else None,
    }
    inv = {"version": 1, "manifest_sha256": sha, "controller_runs": [run], "jobs": jobs}
    (directory / "inventory.json").write_text(json.dumps(inv))
    return {"inventory_dir": str(directory), "manifest_sha256": sha}


def _states(complete, running, pending):
    return ["complete"] * complete + ["running"] * running + ["pending"] * pending


def _dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_active_external_campaigns_reserve_min_workers_open_jobs(tmp_path):
    externals = [
        _external(tmp_path, "taxi", 2, _states(5, 2, 3)),
        _external(tmp_path, "lunar", 1, _states(2, 1, 27)),
        _external(tmp_path, "generic", 1, _states(10, 1, 14)),
    ]
    budget = _budget(tmp_path, 8, externals)
    grants = []
    while True:
        lease_id, usage = budget.try_acquire({})
        if lease_id is None:
            break
        grants.append(lease_id)
    assert len(grants) == 4
    assert [r["charge"] for r in usage["external"]] == [2, 1, 1]
    assert usage["used"] == 8
    # Tail of a two-worker campaign: one running, none pending, charges one.
    tail = _external(tmp_path, "tail", 2, _states(9, 1, 0))
    assert campaign.inventory_capacity_charge(tail)["charge"] == 1
    # A finished campaign whose controller ended releases its reservation.
    done = _external(tmp_path, "done", 2, _states(10, 0, 0), ended=True)
    assert campaign.inventory_capacity_charge(done)["charge"] == 0


def test_dead_external_controller_orphans_stay_charged(tmp_path):
    sleeper = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        entry = _external(
            tmp_path,
            "crashed",
            2,
            ["running", "interrupted", "running", "complete", "pending"],
            controller_pid=_dead_pid(),
            pgids={"j0": sleeper.pid, "j1": _dead_pid()},
        )
        report = campaign.inventory_capacity_charge(entry)
        assert not report["controller_active"] and report["reserved"] == 0
        # j0: live orphan group. j2: running with no recorded group, so a
        # crash between fork and record cannot be ruled out.
        assert (report["orphan_groups"], report["unverifiable_attempts"]) == (1, 1)
        assert report["charge"] == 2
    finally:
        os.killpg(sleeper.pid, signal.SIGKILL)
        sleeper.wait()
    assert campaign.inventory_capacity_charge(entry)["charge"] == 1


def test_missing_or_mismatched_external_state_fails_closed(tmp_path):
    good = _external(tmp_path, "good", 1, _states(0, 1, 1))
    cases = {
        "wrong_hash": dict(good, manifest_sha256="0" * 64),
        "missing": dict(good, inventory_dir=str(tmp_path / "absent")),
    }
    malformed = _external(tmp_path, "malformed", 1, _states(0, 1, 1))
    (pathlib.Path(malformed["inventory_dir"]) / "inventory.json").write_text("{")
    cases["malformed"] = malformed
    rebound = _external(tmp_path, "rebound", 1, _states(0, 1, 1))
    inv_path = pathlib.Path(rebound["inventory_dir"]) / "inventory.json"
    inv = json.loads(inv_path.read_text())
    inv["manifest_sha256"] = "f" * 64
    inv_path.write_text(json.dumps(inv))
    cases["rebound"] = rebound
    for name, entry in cases.items():
        with pytest.raises(budget_mod.CapacityError):
            campaign.inventory_capacity_charge(entry)
        root = tmp_path / f"budget-{name}"
        root.mkdir()
        lease_id, usage = _budget(root, 8, [entry]).try_acquire({})
        assert lease_id is None and "refused" in usage, name


def test_runner_holds_lease_from_launch_until_reaped(env, monkeypatch):
    monkeypatch.setattr(campaign, "CAPACITY_RETRY_SECONDS", 0.02)
    leases, log = env.root / "budget" / "leases", env.root / "leases.log"
    jobs = [env.job(j, "count_leases", leases, log) for j in ("a", "b")]
    # A bounded stage turns a leaked lease into a failure instead of a hang.
    env.write_manifest(jobs, stage=_in(30))
    config = _capacity_config(env.root / "capacity.json", 1)
    flags = (
        "--capacity-config",
        config,
        "--capacity-state-dir",
        str(env.root / "budget"),
    )
    assert env.run(*flags, workers=2) == 0
    # Each job saw exactly its own lease: the limit, not workers, bounded it.
    assert sorted(log.read_text().split("\n")[:-1]) == ["a 1", "b 1"]
    assert os.listdir(leases) == []
    inv = env.inventory_json()
    assert inv["controller_runs"][-1]["capacity"]["limit"] == 1
    grants = [inv["jobs"][j]["attempts"][0]["capacity"] for j in ("a", "b")]
    assert grants[0]["lease_id"] != grants[1]["lease_id"]
    assert grants[1]["waited_seconds"] > 0


def test_capacity_wait_respects_deadline_without_marking_running(env, monkeypatch):
    monkeypatch.setattr(campaign, "CAPACITY_RETRY_SECONDS", 0.02)
    budget = _budget(env.root, 1)
    held, _ = budget.try_acquire({"holder": "test"})
    seen = []
    original = campaign.CampaignRunner._save

    def recording_save(self):
        seen.append(self.inv["jobs"]["a"]["state"])
        original(self)

    monkeypatch.setattr(campaign.CampaignRunner, "_save", recording_save)
    env.write_manifest([env.job("a", "ok")], stage=_in(1.0))
    flags = ("--capacity-config", budget.config["path"])
    assert env.run(*flags, "--capacity-state-dir", budget.state_dir) == 1
    row = env.inventory_json()["jobs"]["a"]
    assert (row["state"], row["state_reason"]) == ("cancelled", "stage_deadline")
    assert row["attempts"] == [] and "running" not in seen
    last_wait = env.inventory_json()["controller_runs"][-1]["capacity"]["last_wait"]
    assert last_wait["summary"] == "1/1 used"
    assert os.listdir(budget.lease_dir) == [f"{held}.json"]


def test_two_controllers_share_one_limit(tmp_path):
    log = tmp_path / "track.log"
    envs = [Env(tmp_path / name) for name in ("first", "second")]
    config = _capacity_config(tmp_path / "capacity.json", 2)
    procs = []
    for e in envs:
        e.write_manifest([e.job(f"j{i}", "track", log) for i in range(3)])
        procs.append(
            e.spawn_runner(
                "--workers",
                "2",
                "--capacity-config",
                config,
                "--capacity-state-dir",
                str(tmp_path / "budget"),
            ),
        )
    for proc in procs:
        out, _ = proc.communicate(timeout=120)
        assert proc.returncode == 0, out.decode()
    assert len(log.read_text().splitlines()) == 12
    assert _peak_concurrency(log) == 2
    assert os.listdir(tmp_path / "budget" / "leases") == []


def test_capacity_flags_optional_and_default_unchanged(env):
    env.write_manifest([env.job("a", "ok")])
    config = _capacity_config(env.root / "capacity.json", 2)
    bad = _capacity_config(env.root / "bad.json", campaign.MAX_WORKERS + 1)
    state = str(env.root / "budget")
    assert env.run("--capacity-config", config) == 2
    assert env.run("--capacity-state-dir", state) == 2
    assert env.run("--capacity-config", bad, "--capacity-state-dir", state) == 2
    assert not env.inventory.exists()
    assert env.run() == 0
    inv = env.inventory_json()
    assert "capacity" not in inv["controller_runs"][-1]
    assert "capacity" not in inv["jobs"]["a"]["attempts"][0]
    assert not os.path.exists(state)


# ---------------------------------------------------------------------------
# Private sync wrapper (ignored file, fake transport only)
# ---------------------------------------------------------------------------


@pytest.fixture
def sync(tmp_path):
    if not SYNC_SCRIPT.exists():
        pytest.skip("private sync wrapper not present")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls.log"
    for tool in ("rsync", "ssh"):
        stub = fake_bin / tool
        stub.write_text(
            "#!/bin/sh\n"
            f'echo "{tool}" >> "{calls}"\n'
            f'for a in "$@"; do printf "%s\\n" "$a" >> "{calls}"; '
            'case "$a" in --files-from=*) cat "${a#--files-from=}" '
            f'| sed "s/^/LIST:/" >> "{calls}";; esac; done\n',
        )
        stub.chmod(0o755)
    local = tmp_path / "local"
    local.mkdir()
    environ = dict(
        os.environ,
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        SYNC_SERVER="fake-host",
        SYNC_REMOTE_DIR="/remote/root",
        SYNC_LOCAL_DIR=str(local),
    )

    def run(*entries, raw=None):
        listing = tmp_path / "list.txt"
        listing.write_text(
            raw if raw is not None else "".join(e + "\n" for e in entries)
        )
        return subprocess.run(
            ["bash", str(SYNC_SCRIPT), "pull-selected", str(listing)],
            env=environ,
            capture_output=True,
            text=True,
        )

    run.calls = calls
    run.local = local
    run.tmp = tmp_path
    return run


def test_sync_pull_selected_direction_and_roots(sync):
    result = sync("experiments/runs/r1/result.json", "a/b_c-1.2/x.json")
    assert result.returncode == 0, result.stderr
    lines = sync.calls.read_text().splitlines()
    assert lines[0] == "rsync"
    assert lines.count("rsync") == 1 and "ssh" not in lines
    assert lines[-2:] == ["fake-host:/remote/root/", f"{sync.local}/"]
    assert "--no-links" in lines and "-tz" in lines
    assert not any(x in lines for x in ("-r", "-a", "-rtz", "--recursive"))
    assert [x for x in lines if x.startswith("LIST:")] == [
        "LIST:experiments/runs/r1/result.json",
        "LIST:a/b_c-1.2/x.json",
    ]


def test_sync_pull_selected_real_rsync_copies_only_listed_files(sync):
    real = shutil.which("rsync")
    if real is None:
        pytest.skip("rsync not installed")
    remote = sync.tmp / "remote"
    (remote / "dir").mkdir(parents=True)
    (remote / "sel.json").write_text("{}")
    (remote / "dir" / "child.json").write_text("{}")
    (sync.tmp / "secret").write_text("outside")
    (remote / "link.json").symlink_to(sync.tmp / "secret")
    # Transport that only rewrites the remote endpoint to a local directory.
    (sync.tmp / "bin" / "rsync").write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        f"args = [{str(remote) + '/'!r} if a == 'fake-host:/remote/root/' else a"
        " for a in sys.argv[1:]]\n"
        f"os.execv({real!r}, [{real!r}] + args)\n",
    )
    result = sync("sel.json", "dir", "link.json")
    assert result.returncode == 0, result.stderr
    assert (sync.local / "sel.json").read_text() == "{}"
    assert not (sync.local / "dir" / "child.json").exists()
    assert not os.path.lexists(sync.local / "link.json")


@pytest.mark.parametrize(
    "raw",
    [
        "/etc/passwd\n",
        "../outside.json\n",
        "a/../../outside.json\n",
        "a/./b.json\n",
        "-e sh\n",
        "--rsh=evil\n",
        "\n",
        "ok.json\n\n",
        "a//b.json\n",
        "dir/\n",
        "#comment\n",
        "a b.json\n",
        "a.json\r\n",
        "",
    ],
)
def test_sync_pull_selected_rejects_unsafe_entries(sync, raw):
    result = sync(raw=raw)
    assert result.returncode != 0
    assert not sync.calls.exists()


def test_sync_pull_selected_rejects_symlink_escape(sync):
    outside = sync.tmp / "outside"
    outside.mkdir()
    (sync.local / "linked").symlink_to(outside)
    (sync.local / "file_link.json").symlink_to(outside / "t.json")
    (sync.local / "plain").write_text("")
    for entry in ("linked/result.json", "file_link.json", "plain/child.json"):
        result = sync(entry)
        assert result.returncode != 0, entry
        assert not sync.calls.exists()
    assert list(outside.iterdir()) == []
