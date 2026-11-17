"""Tests for the shared worker-slot budget.

The module is stdlib only and loaded from its file path. Processes used here
are tiny local sleepers; nothing contacts a remote machine.
"""

import importlib.util
import json
import os
import pathlib
import signal
import subprocess
import sys
import textwrap
import time

import pytest

BUDGET_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "imitation"
    / "experiments"
    / "agnostic"
    / "resource_budget.py"
)
_spec = importlib.util.spec_from_file_location(
    "resource_budget_under_test", BUDGET_PATH
)
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)


def _config(tmp_path, limit=2, externals=()):
    path = tmp_path / "capacity.json"
    path.write_text(
        json.dumps(
            {"version": 1, "limit": limit, "external_campaigns": list(externals)},
        ),
    )
    return rb.load_config(str(path), max_limit=64)


def _budget(tmp_path, limit=2, probe=None, externals=()):
    config = _config(tmp_path, limit, externals)
    return rb.CapacityBudget(config, str(tmp_path / "budget"), probe or (lambda e: {}))


def _dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _sleeper():
    return subprocess.Popen(["sleep", "60"], start_new_session=True)


def _kill(proc):
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def _write_lease(budget, owner_pid, child_pgid=None, released=False):
    lease_id = os.urandom(16).hex()
    lease = {
        "version": 1,
        "lease_id": lease_id,
        "owner_pid": owner_pid,
        "created_at": "2026-01-01T00:00:00+00:00",
        "label": {"test": True},
        "child_pid": child_pgid,
        "child_pgid": child_pgid,
        "owner_released": released,
    }
    pathlib.Path(budget.lease_dir, f"{lease_id}.json").write_text(json.dumps(lease))
    return lease_id


def _leases(budget):
    return sorted(n for n in os.listdir(budget.lease_dir) if not n.startswith("."))


CONTENDER = textwrap.dedent(
    """
    import importlib.util, json, os, sys, time
    spec = importlib.util.spec_from_file_location("rb", sys.argv[1])
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    config, state, go, out, hold = sys.argv[2:7]
    config = rb.load_config(config, 64)
    while not os.path.exists(go):
        time.sleep(0.001)
    try:
        budget = rb.CapacityBudget(config, state, lambda e: {})
        lease_id, error = budget.try_acquire({"pid": os.getpid()})[0], None
    except rb.CapacityError as exc:
        lease_id, error = None, str(exc)
    with open(out + ".tmp", "w") as f:
        json.dump({"lease_id": lease_id, "limit": config["limit"], "error": error}, f)
    os.rename(out + ".tmp", out)
    while not os.path.exists(hold):
        time.sleep(0.01)
    """,
)


def _race(tmp_path, configs, state_dir):
    """Start one contender per config path at once; return their reports."""
    go, hold = tmp_path / "go", tmp_path / "hold"
    outs = [tmp_path / f"out{i}.json" for i in range(len(configs))]
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", CONTENDER, str(BUDGET_PATH)]
            + [config, state_dir, str(go), str(out), str(hold)],
        )
        for config, out in zip(configs, outs)
    ]
    try:
        go.write_text("")
        end = time.monotonic() + 30
        while not all(o.exists() for o in outs) and time.monotonic() < end:
            time.sleep(0.02)
        return [json.loads(o.read_text()) for o in outs]
    finally:
        hold.write_text("")
        for proc in procs:
            proc.wait(timeout=30)


def test_cooperating_controllers_never_overclaim_under_race(tmp_path):
    budget = _budget(tmp_path, limit=3)
    reports = _race(tmp_path, [budget.config["path"]] * 8, budget.state_dir)
    granted = [r["lease_id"] for r in reports]
    assert sum(g is not None for g in granted) == 3
    assert len(_leases(budget)) == 3
    assert all(r["error"] is None for r in reports)
    # Owners died before recording a child: a crash between fork and record
    # cannot be ruled out, so the slots stay charged rather than pruned.
    lease_id, usage = budget.try_acquire({})
    assert lease_id is None
    assert usage["used"] == 3 and usage["leases_without_child"] == 3


def _write_config(path, limit, externals=()):
    path.write_text(
        json.dumps(
            {"external_campaigns": list(externals), "limit": limit, "version": 1}
        ),
    )
    return str(path)


def test_controllers_with_different_configs_cannot_share_state(tmp_path):
    a = {"inventory_dir": "/ext/a", "manifest_sha256": "a" * 64}
    b = {"inventory_dir": "/ext/b", "manifest_sha256": "b" * 64}
    budget = _budget(tmp_path, limit=1, probe=lambda e: {"charge": 0}, externals=[a, b])
    probe = budget.external_probe
    state = budget.state_dir
    rejected = {
        "limit": _write_config(tmp_path / "limit2.json", 2, [a, b]),
        "fewer_externals": _write_config(tmp_path / "subset.json", 1, [a]),
        "other_manifest": _write_config(
            tmp_path / "manifest.json",
            1,
            [a, dict(b, manifest_sha256="c" * 64)],
        ),
    }
    for name, path in rejected.items():
        with pytest.raises(rb.CapacityError, match="bound to config"):
            rb.CapacityBudget(rb.load_config(path, 64), state, probe)
    # Same canonical content at another path, with reordered keys, reordered
    # externals and an unnormalized directory, cooperates under the one limit.
    (tmp_path / "copy").mkdir()
    same = _write_config(
        tmp_path / "copy" / "capacity.json",
        1,
        [b, dict(a, inventory_dir="/ext/x/../a")],
    )
    twin = rb.CapacityBudget(rb.load_config(same, 64), state, probe)
    assert twin.config["sha256"] == budget.config["sha256"]
    assert twin.try_acquire({})[0] is not None
    assert budget.try_acquire({})[0] is None


def test_race_of_different_configs_binds_exactly_one(tmp_path):
    configs = [
        _write_config(tmp_path / f"limit{limit}.json", limit)
        for limit in (1, 2)
        for _ in range(4)
    ]
    state = str(tmp_path / "budget")
    reports = _race(tmp_path, configs, state)
    bound = json.loads(pathlib.Path(state, "config.json").read_text())
    limit = bound["config"]["limit"]
    winners = [r for r in reports if r["limit"] == limit]
    losers = [r for r in reports if r["limit"] != limit]
    assert all(r["error"] is None for r in winners)
    assert all("bound to config" in r["error"] for r in losers)
    assert all(r["lease_id"] is None for r in losers)
    assert sum(r["lease_id"] is not None for r in winners) == limit
    assert len(os.listdir(pathlib.Path(state, "leases"))) == limit


@pytest.mark.parametrize(
    "corrupt",
    ["truncated", "extra_key", "tampered_limit", "bad_version", "list_config"],
)
def test_malformed_binding_fails_closed(tmp_path, corrupt):
    budget = _budget(tmp_path, limit=2)
    path = pathlib.Path(budget.binding_path)
    bound = json.loads(path.read_text())
    if corrupt == "truncated":
        path.write_text(path.read_text()[:-1])
    else:
        if corrupt == "extra_key":
            bound["note"] = 1
        elif corrupt == "tampered_limit":
            bound["config"]["limit"] = 8
        elif corrupt == "bad_version":
            bound["version"] = True
        else:
            bound["config"] = []
            bound["config_sha256"] = rb._sha256_json([])
        path.write_text(json.dumps(bound))
    lease_id, usage = budget.try_acquire({})
    assert lease_id is None and "binding" in usage["refused"]
    with pytest.raises(rb.CapacityError, match="binding"):
        rb.CapacityBudget(budget.config, budget.state_dir, budget.external_probe)


def test_missing_binding_beside_leases_fails_closed(tmp_path):
    budget = _budget(tmp_path, limit=2)
    held, _ = budget.try_acquire({})
    os.unlink(budget.binding_path)
    lease_id, usage = budget.try_acquire({})
    assert lease_id is None and "leases but no" in usage["refused"]
    with pytest.raises(rb.CapacityError):
        rb.CapacityBudget(budget.config, budget.state_dir, budget.external_probe)
    assert _leases(budget) == [f"{held}.json"]
    assert not os.path.exists(budget.binding_path)


def test_child_lease_released_only_after_reap(tmp_path):
    budget = _budget(tmp_path, limit=1)
    lease_id, usage = budget.try_acquire({"job": "a"})
    assert lease_id is not None and usage["used"] == 0
    child = _sleeper()
    try:
        budget.bind_child(lease_id, child.pid, child.pid)
        # The owner gives up but the group is not verified gone: still charged.
        budget.release(lease_id, reaped=False)
        refused, usage = budget.try_acquire({})
        assert refused is None and usage["used"] == 1
    finally:
        _kill(child)
    granted, usage = budget.try_acquire({})
    assert granted is not None and usage["leases_pruned"] == [lease_id]
    budget.release(granted, reaped=True)
    assert _leases(budget) == []


def test_dead_owner_leases_pruned_only_when_child_verified_gone(tmp_path):
    budget = _budget(tmp_path, limit=2)
    dead = _dead_pid()
    unknown = _write_lease(budget, dead)
    child = _sleeper()
    try:
        orphan = _write_lease(budget, dead, child_pgid=child.pid)
        finished = _write_lease(budget, dead, child_pgid=_dead_pid())
        lease_id, usage = budget.try_acquire({})
        assert lease_id is None
        assert usage["used"] == 2 and usage["leases_pruned"] == [finished]
    finally:
        _kill(child)
    lease_id, usage = budget.try_acquire({})
    assert lease_id is not None and usage["leases_pruned"] == [orphan]
    assert unknown + ".json" in _leases(budget)


def test_external_charge_counts_and_invalid_state_fails_closed(tmp_path):
    entry = {"inventory_dir": "/ext/a", "manifest_sha256": "a" * 64}
    reports = {"charge": 3}
    budget = _budget(tmp_path, limit=4, probe=lambda e: reports, externals=[entry])
    first, usage = budget.try_acquire({})
    assert first is not None and usage["external"][0]["charge"] == 3
    assert budget.try_acquire({})[0] is None

    def broken(entry):
        raise rb.CapacityError("inventory malformed")

    budget.external_probe = broken
    budget.release(first, reaped=True)
    lease_id, usage = budget.try_acquire({})
    assert lease_id is None and "malformed" in usage["refused"]

    budget.external_probe = lambda e: {"charge": 0}
    pathlib.Path(budget.lease_dir, "junk.json").write_text("{")
    lease_id, usage = budget.try_acquire({})
    assert lease_id is None and "junk.json" in usage["refused"]


@pytest.mark.parametrize(
    "raw",
    [
        {"version": 1, "limit": 65, "external_campaigns": []},
        {"version": 1, "limit": 0, "external_campaigns": []},
        {"version": True, "limit": 2, "external_campaigns": []},
        {"version": 1, "limit": 2},
        {
            "version": 1,
            "limit": 2,
            "external_campaigns": [
                {"inventory_dir": "rel", "manifest_sha256": "a" * 64}
            ],
        },
        {
            "version": 1,
            "limit": 2,
            "external_campaigns": [{"inventory_dir": "/x", "manifest_sha256": "bad"}],
        },
    ],
)
def test_invalid_config_rejected(tmp_path, raw):
    path = tmp_path / "capacity.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(rb.CapacityError):
        rb.load_config(str(path), max_limit=64)
