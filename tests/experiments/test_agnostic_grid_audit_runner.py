"""`task_pilot.run_audit` records the evidence type its audit actually uses.

The expert load and the witness audit are stubbed (no rollout, no fit); the
runner spec, its restriction dispatch and the recorded config are real. The
stub reports the kind exactly as `task_audit.audit` derives it.
"""

import datetime
import json

import pytest

from imitation.experiments.agnostic import restriction_pilot as rp
from imitation.experiments.agnostic import task_audit, task_pilot, tasks

UTC = datetime.timezone.utc
FAR = datetime.datetime(2026, 10, 7, 1, 0, tzinfo=UTC)


class Clock:
    """Deterministic clock that advances a fixed step per reading."""

    def __init__(self, step_seconds=0.001):
        self.now = datetime.datetime(2026, 10, 1, tzinfo=UTC)
        self.step = datetime.timedelta(seconds=step_seconds)

    def __call__(self):
        self.now += self.step
        return self.now


def _ladder_rung(env_name):
    return next(r for r in tasks.get_task(env_name).restriction_ids if r != "identity")


GRID_CASES = [
    (env_name, tasks.RANDOM_GRID_CANDIDATES[env_name][0])
    for env_name in task_audit.GRID_ENVS
]
OLD_CASES = [(env_name, _ladder_rung(env_name)) for env_name in task_audit.GRID_ENVS]
OLD_CASES.append(("MountainCar-v0", "mountaincar_bins_pos005_vel005"))


@pytest.fixture
def stubbed(monkeypatch):
    calls = []

    def load_expert(preparation_dir, spec):
        return object(), {"expert_sha256": "0" * 64, "preparation_dir_name": "stub"}

    def audit(expert, env_name, restriction_id, reset_seed, check):
        calls.append((env_name, restriction_id))
        return {
            "status": "stub",
            "evidence_type": task_audit.evidence_type(env_name, restriction_id),
            "expert_predict_calls": 0,
        }

    monkeypatch.setattr(rp, "_load_expert", load_expert)
    monkeypatch.setattr(task_audit, "audit", audit)
    return calls


def _run(tmp_path, env_name, restriction_id):
    out = tmp_path / "audit"
    code = task_pilot.run_audit(
        out,
        tmp_path / "expert",
        env_name=env_name,
        restriction_id=restriction_id,
        deadline=FAR,
        job_limit_seconds=60,
        clock=Clock(),
        calibration_candidates=True,
    )
    assert code == 0
    return json.loads((out / rp.RESULT_FILE).read_text())


@pytest.mark.parametrize("env_name,restriction_id", GRID_CASES)
def test_grid_audit_config_names_the_grid_evidence(
    stubbed,
    tmp_path,
    env_name,
    restriction_id,
):
    record = _run(tmp_path, env_name, restriction_id)
    assert stubbed == [(env_name, restriction_id)]
    assert record["status"] == "complete"
    assert record["audit"]["evidence_type"] == task_audit.GRID_EVIDENCE
    assert record["config"]["evidence_type"] == record["audit"]["evidence_type"]
    assert record["config_sha256"] == rp.pilot.config_digest(record["config"])


@pytest.mark.parametrize("env_name,restriction_id", OLD_CASES)
def test_old_audit_config_keeps_the_environment_evidence(
    stubbed,
    tmp_path,
    env_name,
    restriction_id,
):
    record = _run(tmp_path, env_name, restriction_id)
    assert stubbed == [(env_name, restriction_id)]
    assert record["audit"]["evidence_type"] == task_audit.evidence_type(env_name)
    assert record["config"]["evidence_type"] == record["audit"]["evidence_type"]
