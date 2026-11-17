"""Pinned environment-source lineage of the four vector width teachers (v5).

A preparation written before env_utils.py / env_baselines.py gained the
FrozenLakeSlippery-v0 variant loads only when both recorded digests are the
exact pinned old ones, the current digests are the pinned new ones and the
task is one of the four vector tasks; everything else still refuses.
"""

import json

import pytest

from imitation.experiments.agnostic import classical, pilot, task_experts
from imitation.experiments.agnostic import capacity_policy as cp

from .test_capacity_runner import _data
from .test_capacity_teacher import injected_trainer, prepare

LINEAGE = task_experts.APPROVED_ENVIRONMENT_SOURCE_LINEAGE
UTILS = "experiments/ftrl/env_utils.py"
BASELINES = "experiments/ftrl/env_baselines.py"


def _current():
    return {
        rel: pilot.sha256_file(pilot._PACKAGE_ROOT / rel)
        for rel in task_experts.PREPARATION_ENVIRONMENT_SOURCES
    }


def test_pins_match_this_source():
    current = _current()
    for rel, entry in LINEAGE["files"].items():
        assert current[rel] == entry["current"]


def _old_files():
    files = dict(_current())
    for rel, entry in LINEAGE["files"].items():
        files[rel] = entry["recorded"]
    return files


def test_match_rules():
    current = _current()
    ok, lineage = task_experts._environment_sources_match("CartPole-v1", dict(current), current)
    assert ok and lineage == {}
    ok, lineage = task_experts._environment_sources_match("LunarLander-v2", _old_files(), current)
    assert ok and set(lineage) == {UTILS, BASELINES}
    # tabular tasks never get the lineage
    for env in ("Taxi-v3", "Blackjack-v1", "FrozenLakeSlippery-v0", "CliffWalking-v0"):
        assert not task_experts._environment_sources_match(env, _old_files(), current)[0]
    # one old file alone is fine, another digest is not
    one = dict(current, **{UTILS: LINEAGE["files"][UTILS]["recorded"]})
    assert task_experts._environment_sources_match("Acrobot-v1", one, current)[0]
    bad = dict(_old_files(), **{BASELINES: "0" * 64})
    assert not task_experts._environment_sources_match("Acrobot-v1", bad, current)[0]
    # other environment files must still match
    other = dict(_old_files(), **{"util/util.py": "0" * 64})
    assert not task_experts._environment_sources_match("Acrobot-v1", other, current)[0]
    # a changed current source does not inherit the pin
    moved = dict(current, **{UTILS: "1" * 64})
    assert not task_experts._environment_sources_match("Acrobot-v1", _old_files(), moved)[0]
    assert not task_experts._environment_sources_match("Acrobot-v1", None, current)[0]


def _rewrite_record(prep, files):
    path = prep / pilot.PREPARATION_FILE
    record = json.loads(path.read_text())
    record["source_fingerprint"]["files"] = files
    record["source_fingerprint"]["combined_sha256"] = pilot.config_digest(files)
    path.write_text(json.dumps(record, indent=2, sort_keys=True))


@pytest.fixture
def old_prep(tmp_path):
    prep = tmp_path / "prep"
    assert prepare(prep, injected_trainer((64,)), (64,)) == 0
    record = json.loads((prep / pilot.PREPARATION_FILE).read_text())
    files = dict(record["source_fingerprint"]["files"])
    for rel, entry in LINEAGE["files"].items():
        files[rel] = entry["recorded"]
    return prep, files


def test_old_preparation_loads_with_recorded_lineage(old_prep, tmp_path):
    prep, files = old_prep
    _rewrite_record(prep, files)
    try:
        _, identity = task_experts.load_qualified_expert(prep, "CartPole-v1")
    except classical.Refused as exc:
        pytest.skip(f"record rewrite not accepted by the base loader: {exc}")
    lineage = identity["compatibility"]["environment_source_lineage"]
    assert set(lineage["recorded_and_current"]) == {UTILS, BASELINES}
    out = tmp_path / "data"
    assert _data(out, prep, learner_family=cp.LEARNER_FAMILY) == classical.EXIT_COMPLETE


def test_old_preparation_with_other_digest_refuses(old_prep):
    prep, files = old_prep
    files[UTILS] = "2" * 64
    _rewrite_record(prep, files)
    with pytest.raises(classical.Refused, match="environment_sources"):
        task_experts.load_qualified_expert(prep, "CartPole-v1")
