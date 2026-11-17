"""Synthetic checks of the expert recovery runner; nothing here is evidence.

No environment is created or rolled out, no model is trained or loaded, and
no remote machine is contacted. Tabular dynamics, models, environments and
qualification records below are small hand-made stand-ins.
"""

import datetime
import hashlib
import json
import os
import pathlib
import sys
import types

import numpy as np
import pytest
import torch as th

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "experiments" / "agnostic" / "recover_expert.py"
EVIDENCE = (
    REPO_ROOT
    / "experiments"
    / ".local"
    / "review-evidence"
    / "remaining-classical-20261002"
)
UTC = datetime.timezone.utc
FUTURE = datetime.datetime(2100, 1, 1, tzinfo=UTC)


def _load():
    import importlib.util

    spec = importlib.util.spec_from_file_location("recover_expert", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rx = _load()
REWARDS = {"delivery": 20, "step": -1, "illegal": -10}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clock_at(when):
    return lambda: when


# ---------------------------------------------------------------------------
# Predeclared configs
# ---------------------------------------------------------------------------


def test_lunar_budget_rounds_down_to_whole_rollouts():
    assert rx.usable_budget(4_000_000, 16384) == (3_997_696, 244)
    with pytest.raises(ValueError):
        rx.usable_budget(16383, 16384)
    targets = rx.check_targets(3_997_696, 16384, 4)
    assert len(targets) == 61 and targets[0] == 65536 and targets[-1] == 3_997_696
    # A budget that is not a multiple of the check interval gets a final check.
    assert rx.check_targets(5 * 8, 8, 2) == [16, 32, 40]
    config = rx.build_config("lunar-A")["budget"]
    assert config["usable_max_env_transitions"] == 3_997_696
    assert config["check_env_transitions"] == 65536


def test_candidates_differ_only_where_predeclared():
    a, b = rx.build_config("lunar-A"), rx.build_config("lunar-B")
    assert a["ppo"]["net_arch"] == [64, 64] and b["ppo"]["net_arch"] == [256, 256]
    assert (a["qualification"]["seed"], b["qualification"]["seed"]) == (900112, 900113)
    assert a["selection"] == b["selection"]
    assert a["selection"]["self_ce_max"] == 0.4
    assert a["selection"]["normalized_threshold"] == 0.8
    ta, tb = rx.build_config("taxi-A"), rx.build_config("taxi-B")
    assert ta["distillation"]["net_arch"] == [64, 64]
    assert tb["distillation"]["net_arch"] == [256, 256]
    assert ta["distillation"]["stop_rule"]["self_ce_max"] == 0.7
    assert ta["qualification"] == {"episodes": 100, "seed": 900111}


def test_load_config_refuses_relaxed_gate_and_wrong_command(tmp_path):
    config = rx.build_config("lunar-A")
    config["selection"]["self_ce_max"] = 0.5
    path = tmp_path / "relaxed.json"
    path.write_text(json.dumps(config))
    with pytest.raises(rx.Refused, match="selection"):
        rx.load_config(path, "lunar")
    path.write_text(json.dumps(rx.build_config("lunar-A")))
    with pytest.raises(rx.Refused, match="run by"):
        rx.load_config(path, "taxi")
    assert rx.load_config(path, "lunar")[0] == rx.build_config("lunar-A")


@pytest.mark.parametrize(
    "candidate,prefix",
    [
        ("taxi-A", "expert3"),
        ("taxi-B", "expert3"),
        ("lunar-A", "expert3"),
        ("lunar-B", "expert3"),
        ("lunar-C", "expert4"),
    ],
)
def test_private_config_files_equal_builder(candidate, prefix):
    path = EVIDENCE / f"{prefix}-{candidate}.json"
    if not path.is_file():
        pytest.skip("private evidence directory not present")
    command = rx.build_config(candidate)["command"]
    assert rx.load_config(path, command)[0] == rx.build_config(candidate)


# Digests of the A and B configs before lunar-C was added; they must not move.
FROZEN_CONFIG_SHA256 = {
    "taxi-A": "152e3a8405e06d13f846f85f827fde9cc387aca4d04a350e25c7c76adea43862",
    "taxi-B": "8093423b293628b400e54cac9bf29c929fc1ec9f6673d3c5d57bcc51d88f6a3b",
    "lunar-A": "125c1b56cd29b41170376721c74df2d7fe1f33abded7755d6b033d7cad7d1699",
    "lunar-B": "0420b92e38ff8f47b14615989a83dea4991dad2ad50dc70776af8c6eb84d8f6d",
}


def test_a_and_b_configs_are_unchanged():
    for candidate, digest in FROZEN_CONFIG_SHA256.items():
        assert rx.pilot.config_digest(rx.build_config(candidate)) == digest


def test_lunar_c_is_a_with_an_8m_bound_and_its_own_heldout_seed():
    a, c = rx.build_config("lunar-A"), rx.build_config("lunar-C")
    assert c["candidate_id"] == "lunar-C" and c["command"] == "lunar"
    assert rx.usable_budget(8_000_000, 16384) == (7_995_392, 488)
    assert c["budget"]["requested_max_env_transitions"] == 8_000_000
    assert c["budget"]["usable_max_env_transitions"] == 7_995_392
    assert c["budget"]["rollouts"] == 488
    targets = rx.check_targets(7_995_392, 16384, 4)
    assert len(targets) == 122 and targets[-1] == 7_995_392
    assert c["qualification"] == {
        "episodes": 100,
        "seed": 900114,
        "raw_mean_min": 200.0,
    }
    others = [k for k in rx.CANDIDATES if k != "lunar-C"]
    assert 900114 not in {rx.build_config(k)["qualification"]["seed"] for k in others}
    # Same architecture, seed, PPO settings, selection gates, quality gates.
    for key in ("ppo", "selection", "gates", "training_method", "fallback_rule"):
        assert c[key] == a[key], key
    assert c["ppo"]["net_arch"] == [64, 64] and c["ppo"]["seed"] == 1
    budget_keys = {"requested_max_env_transitions", "usable_max_env_transitions"}
    budget_keys.add("rollouts")
    assert {k for k in a["budget"] if a["budget"][k] != c["budget"][k]} == budget_keys
    assert {k for k in a if a[k] != c[k]} == {"candidate_id", "budget", "qualification"}


def test_existing_a_and_b_tokens_permit_one_distinct_c_claim(tmp_path, monkeypatch):
    no_science(monkeypatch)
    ledger = new_ledger(tmp_path)
    kept = {}
    for cid in ("lunar-A", "lunar-B"):
        for kind in rx.LEDGER_KINDS:
            ledger.claim(cid, kind, {"output_dir": f"failed {cid}"})
            kept[cid, kind] = ledger.token_path(cid, kind).read_bytes()
    config = rx.build_config("lunar-C")
    raw = json.dumps(config).encode()
    attempt = rx.Attempt(tmp_path / "c", config, raw, FUTURE, ledger=ledger)
    claim = attempt.record["candidate_ledger"]["execution_claim"]
    assert claim["file"] == "lunar-C.execution.json"
    assert claim["contents"]["config_sha256"] == rx.pilot.config_digest(config)
    with pytest.raises(rx.Refused, match="already has"):
        rx.run_lunar(config, raw, tmp_path / "again", FUTURE, ledger=ledger)
    assert not (tmp_path / "again").exists()
    for (cid, kind), data in kept.items():
        assert ledger.token_path(cid, kind).read_bytes() == data
    assert sorted(p.name for p in ledger.path.iterdir()) == sorted(
        [f"{cid}.{kind}.json" for cid, kind in kept] + ["lunar-C.execution.json"]
    )


# ---------------------------------------------------------------------------
# Taxi planner and DP on a synthetic table
# ---------------------------------------------------------------------------


def raw_table():
    """Three live states; 9 is entered only by terminal edges.

    State 9 loops with a large reward. It must never be reached or
    bootstrapped. State 0 has two shortest actions (0 and 2).
    """
    return {
        0: {
            0: [(1.0, 1, -1, False)],
            1: [(1.0, 0, -10, False)],
            2: [(1.0, 2, -1, False)],
        },
        1: {
            0: [(1.0, 9, 20, True)],
            1: [(1.0, 0, -1, False)],
            2: [(1.0, 1, -10, False)],
        },
        2: {
            0: [(1.0, 1, -1, False)],
            1: [(1.0, 9, 20, True)],
            2: [(1.0, 2, -10, False)],
        },
        9: {a: [(1.0, 9, 100, False)] for a in range(3)},
    }


def teacher_config(horizon=200):
    return {
        "teacher": {
            "rewards": REWARDS,
            "dp_horizon": horizon,
            "tie_break": "lowest action index",
        }
    }


def test_planner_lowest_index_tie_and_terminal_not_bootstrapped():
    teacher = rx.build_teacher(raw_table(), [0], teacher_config())
    assert teacher["reachable_states"] == [0, 1, 2]
    assert teacher["distance"] == {"0": 2, "1": 1, "2": 1}
    assert teacher["labels"] == {"0": 0, "1": 0, "2": 1}
    dp = teacher["dp"]
    assert dp["equal_all_horizons"] and dp["policy_in_optimal_set"]
    assert dp["optimal_action_sets"]["0"] == [0, 2]
    # 20 on delivery, -1 per ordinary step, nothing after the terminal edge.
    assert dp["optimal_value"] == {"0": 19, "1": 20, "2": 20}


def test_short_horizon_values_equal_without_delivery():
    table = rx.deterministic_table(raw_table())
    states = rx.reachable_states(table, [0])
    policy = rx.planner_policy(table, rx.shortest_delivery(table, states))
    dp = rx.finite_horizon_check(table, states, policy, 1)
    assert dp["equal_all_horizons"]
    assert dp["optimal_value"]["0"] == -1


def test_dp_flags_a_suboptimal_policy():
    table = rx.deterministic_table(raw_table())
    states = rx.reachable_states(table, [0])
    dp = rx.finite_horizon_check(table, states, {0: 1, 1: 0, 2: 1}, 200)
    assert not dp["equal_all_horizons"]
    assert not dp["policy_in_optimal_set"]


def test_impossible_and_malformed_tables_fail():
    trapped = raw_table()
    trapped[0][1] = [(1.0, 3, -1, False)]
    trapped[3] = {a: [(1.0, 3, -1, False)] for a in range(3)}
    with pytest.raises(rx.RecoveryFailure, match="cannot deliver"):
        rx.build_teacher(trapped, [0], teacher_config())
    stochastic = raw_table()
    stochastic[0][0] = [(0.5, 1, -1, False), (0.5, 2, -1, False)]
    with pytest.raises(rx.RecoveryFailure, match="not deterministic"):
        rx.build_teacher(stochastic, [0], teacher_config())
    wrong_reward = raw_table()
    wrong_reward[1][0] = [(1.0, 9, 5, True)]
    with pytest.raises(rx.RecoveryFailure, match="Reward semantics"):
        rx.build_teacher(wrong_reward, [0], teacher_config())
    terminal_start = raw_table()
    terminal_start[1][0] = [(1.0, 0, 20, True)]
    with pytest.raises(rx.RecoveryFailure, match="reset state"):
        rx.build_teacher(terminal_start, [0], teacher_config())


class FakeTaxiEnv:
    """Seed s starts at state s % 2 of a two-state task; delivery after k steps."""

    def __init__(self, illegal_from=None):
        self.illegal_from = illegal_from

    def _obs(self):
        obs = np.zeros(4, dtype=np.float32)
        obs[self.state] = 1.0
        return obs

    def reset(self, seed):
        self.state, self.t = seed % 2, 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        if self.state == self.illegal_from and self.t == 1:
            return self._obs(), -10, False, False, {}
        done = self.t >= self.state + 1
        return self._obs(), (20 if done else -1), done, False, {}


def test_reset_support_verification_covers_and_flags():
    expected = {0: (1, 20), 1: (2, 19)}
    before = FUTURE - datetime.timedelta(days=1)
    kwargs = dict(seed_start=0, max_resets=10, deadline=FUTURE, clock=clock_at(before))
    good = rx.verify_reset_support(
        FakeTaxiEnv(), lambda o: 0, expected, REWARDS, **kwargs
    )
    assert all(good["checks"].values()) and good["resets"] == 2
    assert good["env_transitions"] == good["policy_queries"] == 3
    bad = rx.verify_reset_support(
        FakeTaxiEnv(illegal_from=1), lambda o: 0, expected, REWARDS, **kwargs
    )
    assert not bad["checks"]["no_illegal_actions"]
    assert not bad["checks"]["optimal_length_and_return"]
    short = rx.verify_reset_support(
        FakeTaxiEnv(), lambda o: 0, expected, REWARDS, **dict(kwargs, max_resets=1)
    )
    assert short["stop_reason"] == "max_resets"
    assert not short["checks"]["reset_support_covered"]


# ---------------------------------------------------------------------------
# LunarLander segments, selection and checkpoints with a fake model
# ---------------------------------------------------------------------------

R = 8
RULE = {
    "plateau_tolerance": 0.02,
    "patience": 2,
    "min_env_transitions": 0,
    "normalized_threshold": 0.8,
    "self_ce_max": 0.4,
}


def mean_for(norm):
    return -176.0 + norm * 426.0


class FakeModel:
    def __init__(self, extra=0, fail_at=None):
        self.num_timesteps, self.extra, self.fail_at = 0, extra, fail_at
        self.requests = []

    def learn(self, total_timesteps, reset_num_timesteps):
        assert reset_num_timesteps is False
        self.requests.append(total_timesteps)
        if self.fail_at is not None and len(self.requests) == self.fail_at:
            raise RuntimeError("synthetic learn failure")
        goal = self.num_timesteps + total_timesteps
        while self.num_timesteps < goal:  # SB3 collects whole rollouts
            self.num_timesteps += R
        self.num_timesteps += self.extra

    def save(self, path):
        pathlib.Path(path).write_bytes(f"model-{self.num_timesteps}".encode())


def fake_eval(norms, ces):
    def evaluate(model, check):
        i = check - 1
        return {
            "mean_return": mean_for(norms[i]),
            "self_ce": ces[i],
            "episode_returns": [],
            "env_transitions": 0,
            "policy_queries": 0,
        }

    return evaluate


def run(model, tmp_path, norms, ces, targets=(16, 32, 40), now=None):
    store = rx.CheckpointStore(tmp_path, rng_state=lambda: b"rng")
    return rx.train_with_selection(
        model,
        targets=list(targets),
        usable=targets[-1],
        rollout=R,
        rule=RULE,
        state=rx.SelectionState(),
        first_check=1,
        evaluate=fake_eval(norms, ces),
        store=store,
        deadline=FUTURE,
        clock=clock_at(now or rx.rollouts.utc_now()),
    )


def test_exhausted_budget_keeps_latest_and_best(tmp_path):
    model = FakeModel()
    result = run(model, tmp_path, [0.5, 0.7, 0.6], [0.3, 0.3, 0.3])
    assert result["stop"] == "budget_exhausted" and result["selected"] is None
    assert model.requests == [16, 16, 8] and model.num_timesteps == 40
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["name"] == "check-0003" and latest["num_timesteps"] == 40
    assert result["best"]["check"] == 2
    best_zip = tmp_path / result["best"]["checkpoint"]["dir"] / "model.zip"
    assert best_zip.read_bytes() == b"model-32"
    for n in (1, 2, 3):
        assert (tmp_path / "checkpoints" / f"check-{n:04d}" / "progress.json").is_file()


def test_first_passing_check_is_selected_and_training_stops(tmp_path):
    model = FakeModel()
    targets = (8, 16, 24, 32, 40)
    result = run(model, tmp_path, [0.9] * 5, [0.3] * 5, targets=targets)
    # Check 1 improves; checks 2 and 3 do not, so the plateau count reaches 2.
    assert result["stop"] == "selected" and result["selected"]["check"] == 3
    assert model.num_timesteps == 24 and len(model.requests) == 3
    assert result["latest"]["model_sha256"] == result["selected"]["model_sha256"]


def test_overshoot_fails_and_saves_model(tmp_path):
    result = run(FakeModel(extra=1), tmp_path, [0.5] * 3, [0.3] * 3)
    assert result["stop"] == "failed"
    assert result["error"]["type"] == "BudgetViolation"
    saved = tmp_path / result["failure_checkpoint"]["dir"] / "model.zip"
    assert saved.read_bytes() == b"model-17"


def test_learn_failure_saves_in_memory_model(tmp_path):
    result = run(FakeModel(fail_at=2), tmp_path, [0.5] * 3, [0.3] * 3)
    assert result["stop"] == "failed" and result["latest"]["name"] == "check-0001"
    saved = tmp_path / result["failure_checkpoint"]["dir"] / "model.zip"
    assert saved.read_bytes() == b"model-16"


def test_deadline_stops_before_learning(tmp_path):
    model = FakeModel()
    result = run(model, tmp_path, [0.5] * 3, [0.3] * 3, now=FUTURE)
    assert result["stop"] == "deadline" and model.requests == []


def test_selection_plateau_rule_is_inherited():
    state = rx.SelectionState()
    assert state.update(0.80, 0.30, 10, RULE)["improved"]
    assert not state.update(0.81, 0.30, 10, RULE)["improved"]  # within tolerance
    assert state.best_norm == 0.80 and state.since_best == 1
    out = state.update(0.85, 0.30, 10, RULE)  # beats best by more than 0.02
    assert out["improved"] and state.since_best == 0 and not out["passed"]
    out = state.update(0.85, 0.50, 10, RULE)
    assert not out["gates"]["self_ce"]
    with pytest.raises(rx.RecoveryFailure):
        state.update(float("nan"), 0.3, 10, RULE)


def test_episode_recorder_marks_time_limit():
    class Venv:
        num_envs = 1

        def reset(self):
            return np.zeros((1, 2))

        def step(self, actions):
            info = {"TimeLimit.truncated": True}
            return np.zeros((1, 2)), np.array([-1.0]), np.array([True]), [info]

    recorder = rx.EpisodeRecorder(Venv())
    recorder.reset()
    recorder.step(np.array([0]))
    assert recorder.episodes[0]["end"] == "time_limit"
    assert recorder.episodes[0]["landing_or_crash"].startswith("unknown")


# ---------------------------------------------------------------------------
# Acceptance sidecar
# ---------------------------------------------------------------------------


def lunar_attempt(root: pathlib.Path, returns=(210.0,) * 100, status="complete"):
    config = rx.build_config("lunar-A")
    config_bytes = json.dumps(config, indent=2, sort_keys=True).encode()
    root.mkdir()
    (root / "config.json").write_bytes(config_bytes)
    (root / "config.src.json").write_bytes(config_bytes)
    model = b"selected-model"
    for rel in ("selected/model.zip", "checkpoints/check-0001/model.zip"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(model)
    (root / "selection.json").write_text("{}")
    record = {
        "status": "complete",
        "qualified": True,
        "env_name": "LunarLander-v2",
        "expert_sha256": sha(model),
        "config": {
            "qualification_seed": 900112,
            "eval_episodes": 100,
            "threshold": config["gates"]["per_env_threshold"],
        },
        "qualification": {
            "episode_returns": list(returns),
            "episode_lengths": [300] * len(returns),
        },
    }
    (root / "qualification").mkdir()
    (root / rx.QUALIFICATION_RECORD).write_text(json.dumps(record))
    attempt = {
        "status": status,
        "config_sha256": rx.pilot.config_digest(config),
        "checkpoint_sha256": sha(model),
        "env_configs_sha256_start": "e",
        "env_configs_sha256_end": "e",
        "source": {"runner": {"sha256": rx.pilot.sha256_file(rx.RUNNER_PATH)}},
        "candidate_ledger": ledger_record(
            ledger_for(root), config, sha(model), root.resolve()
        ),
    }
    (root / "attempt.json").write_text(json.dumps(attempt))
    bindings = {
        "config": "config.json",
        "attempt": "attempt.json",
        "checkpoint": "selected/model.zip",
        "selection": "selection.json",
        "latest_checkpoint": "checkpoints/check-0001/model.zip",
        "qualification_record": rx.QUALIFICATION_RECORD,
    }
    summary = rx.qualification_summary(record, config, sha(model))
    checks = {name: True for name in rx.LUNAR_CHECKS}
    checks.update(summary["checks"])
    checks["qualification_loader"] = True
    return config, checks, bindings


def ledger_for(root: pathlib.Path):
    path = root.parent / f"{root.name}-ledger"
    path.mkdir(exist_ok=True)
    return rx.CandidateLedger(path)


def ledger_record(ledger, config, checkpoint_sha, output_dir):
    """Real create-once tokens, recorded as `Attempt` records them."""
    cid = config["candidate_id"]
    claim = ledger.claim(
        cid,
        "execution",
        {
            "config_sha256": rx.pilot.config_digest(config),
            "output_dir": str(output_dir),
        },
    )
    held = ledger.claim(
        cid,
        "heldout",
        {
            "execution_claim_sha256": claim["sha256"],
            "checkpoint_sha256": checkpoint_sha,
        },
    )
    return {
        "path": str(ledger.path),
        "execution_claim": claim,
        "heldout_token": held,
    }


def accept(root, status="complete", drop=None, **kwargs):
    config, checks, bindings = lunar_attempt(root, status=status, **kwargs)
    if drop:
        checks[drop] = False
    return rx.write_acceptance(root, config, status, checks, bindings, {})


def test_complete_acceptance_verifies(tmp_path):
    payload = accept(tmp_path / "a")
    assert payload["release_allowed"] and payload["failed_checks"] == []
    assert payload["candidate_b_rule"] == rx.CANDIDATE_B_RULE
    assert payload["limitations"] == list(rx.LIMITATIONS)
    out = rx.verify_acceptance(tmp_path / "a", tmp_path / "a" / "config.src.json")
    assert out["recomputed_mean_return"] == 210.0
    assert not out["ledger_files_checked"]
    ledger = rx.CandidateLedger(tmp_path / "a-ledger")
    out = rx.verify_acceptance(
        tmp_path / "a", tmp_path / "a" / "config.src.json", ledger=ledger
    )
    assert out["ledger_files_checked"]


@pytest.mark.parametrize("status", ["failed", "partial"])
def test_failed_or_partial_acceptance_is_refused(tmp_path, status):
    payload = accept(tmp_path / "a", status=status)
    assert not payload["release_allowed"]
    with pytest.raises(rx.Refused, match="status"):
        rx.verify_acceptance(tmp_path / "a", tmp_path / "a" / "config.src.json")


def test_qualification_alone_cannot_release(tmp_path):
    payload = accept(tmp_path / "a", drop="selection_passed")
    assert not payload["release_allowed"]
    assert payload["checks"]["qualification_complete"]


def test_low_heldout_mean_blocks_release(tmp_path):
    payload = accept(tmp_path / "a", returns=(190.0,) * 100)
    assert not payload["release_allowed"]
    assert payload["failed_checks"] == ["raw_mean_min"]


def edit_json(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "forge",
    [
        "flip_release",
        "drop_check",
        "checkpoint_bytes",
        "qualification_sha",
        "qualification_mean",
        "other_config",
        "no_ledger_record",
        "heldout_other_checkpoint",
        "other_ledger",
        "ledger_token_file",
    ],
)
def test_forged_bindings_are_refused(tmp_path, forge):
    root = tmp_path / "a"
    accept(root, drop="raw_mean_min" if forge == "flip_release" else None)
    config_path = root / "config.src.json"
    acc = root / rx.ACCEPTANCE_FILE
    if forge == "flip_release":
        edit_json(acc, lambda d: d.update(release_allowed=True))
    elif forge == "drop_check":
        edit_json(acc, lambda d: d["checks"].pop("selection_passed"))
    elif forge == "checkpoint_bytes":
        (root / "selected" / "model.zip").write_bytes(b"other-model")
    elif forge == "qualification_sha":
        edit_json(
            root / rx.QUALIFICATION_RECORD, lambda d: d.update(expert_sha256="0" * 64)
        )
    elif forge == "qualification_mean":
        edit_json(
            root / rx.QUALIFICATION_RECORD,
            lambda d: d["qualification"].update(episode_returns=[150.0] * 100),
        )
    elif forge == "other_config":
        config_path = tmp_path / "b.json"
        config_path.write_text(json.dumps(rx.build_config("lunar-B")))
    elif forge == "no_ledger_record":
        edit_json(root / "attempt.json", lambda d: d.pop("candidate_ledger"))
    elif forge == "heldout_other_checkpoint":

        def change(d):
            held = d["candidate_ledger"]["heldout_token"]
            held["contents"]["checkpoint_sha256"] = "0" * 64
            held["sha256"] = sha(rx.token_bytes(held["contents"]))

        edit_json(root / "attempt.json", change)
    ledger = None
    if forge == "other_ledger":
        (tmp_path / "other").mkdir()
        ledger = rx.CandidateLedger(tmp_path / "other")
    elif forge == "ledger_token_file":
        ledger = rx.CandidateLedger(tmp_path / "a-ledger")
        token = ledger.token_path("lunar-A", "heldout")
        os.chmod(token, 0o644)
        token.write_text("{}")
    with pytest.raises(rx.Refused):
        rx.verify_acceptance(root, config_path, ledger=ledger)


# ---------------------------------------------------------------------------
# Resume is refused at every boundary
# ---------------------------------------------------------------------------

DEADLINE_TEXT = "2100-01-01T00:00:00Z"


def no_science(monkeypatch):
    """Any environment, model or qualification start fails the test."""

    def started(*args, **kwargs):
        raise AssertionError("science started")

    monkeypatch.setattr(rx.env_utils, "make_env", started)
    monkeypatch.setattr(rx, "_make_lunar_model", started)
    monkeypatch.setattr(rx.task_experts, "qualify_candidate", started)


def rollback_prior(root: pathlib.Path):
    """The reproduced case: 131072 transitions recorded, latest checkpoint 65536."""
    config = rx.build_config("lunar-A")
    ckpt = root / "checkpoints" / "check-0001"
    ckpt.mkdir(parents=True)
    (ckpt / "model.zip").write_bytes(b"m")
    (ckpt / "rng_state.pkl").write_bytes(b"r")
    ident = {
        "name": "check-0001",
        "dir": "checkpoints/check-0001",
        "model_sha256": sha(b"m"),
        "rng_state_sha256": sha(b"r"),
        "num_timesteps": 65536,
    }
    progress = {
        "check": 1,
        "num_timesteps": 65536,
        "decision": {"passed": False},
        "checkpoint": ident,
        "selection_state": rx.SelectionState().to_dict(),
    }
    (ckpt / "progress.json").write_text(json.dumps(progress))
    latest = dict(ident, progress_sha256=sha((ckpt / "progress.json").read_bytes()))
    (root / "latest.json").write_text(json.dumps(latest))
    attempt = {
        "schema": rx.ATTEMPT_SCHEMA,
        "env_name": "LunarLander-v2",
        "candidate_id": "lunar-A",
        "status": "failed",
        "config_sha256": rx.pilot.config_digest(config),
        "source": rx.source_identity(),
        "package_versions": rx.pilot.package_versions(),
        "task": {"task_sha256": "task"},
        "costs": {"model_num_timesteps": 131072, "training_env_transitions": 131072},
    }
    (root / "attempt.json").write_text(json.dumps(attempt))
    return config


def snapshot(root: pathlib.Path):
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def write_config(tmp_path, candidate):
    path = tmp_path / f"{candidate}.json"
    path.write_text(json.dumps(rx.build_config(candidate)))
    return path


def new_ledger(tmp_path, name="ledger"):
    (tmp_path / name).mkdir()
    return rx.CandidateLedger(tmp_path / name)


def test_resume_is_refused_at_every_boundary(tmp_path, monkeypatch):
    no_science(monkeypatch)
    prior = tmp_path / "prior"
    config = rollback_prior(prior)
    before = snapshot(prior)
    ledger = new_ledger(tmp_path)
    with pytest.raises(rx.Refused, match="Resume is not supported"):
        rx.verify_resume_source(prior, config, {"combined_sha256": "src"}, "task")
    raw = json.dumps(config).encode()
    with pytest.raises(rx.Refused, match="Resume is not supported"):
        rx.run_lunar(config, raw, tmp_path / "direct", FUTURE, prior, ledger=ledger)
    with pytest.raises(rx.Refused, match="Resume is not supported"):
        rx.run_lunar(config, raw, tmp_path / "direct", FUTURE, prior_attempt=prior)
    config_path = write_config(tmp_path, "lunar-A")
    common = [
        "--config",
        str(config_path),
        "--output-dir",
        str(tmp_path / "cli"),
        "--candidate-ledger",
        str(ledger.path),
        "--deadline",
        DEADLINE_TEXT,
    ]
    resume = ["lunar-resume", "--prior-attempt", str(prior)] + common
    assert rx.main(resume) == rx.EXIT_REFUSED
    assert rx.main(["lunar-resume"]) == rx.EXIT_REFUSED
    # The fresh command has no prior-attempt option.
    assert rx.main(["lunar"] + common + ["--prior-attempt", str(prior)]) == 2
    assert not (tmp_path / "direct").exists() and not (tmp_path / "cli").exists()
    assert list(ledger.path.iterdir()) == []
    assert snapshot(prior) == before
    assert not hasattr(rx, "restore_rng_state")


# ---------------------------------------------------------------------------
# Candidate ledger: one execution and one held-out use per candidate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate", ["taxi-A", "lunar-A"])
def test_science_commands_require_an_existing_ledger(tmp_path, monkeypatch, candidate):
    no_science(monkeypatch)
    config = rx.build_config(candidate)
    path = write_config(tmp_path, candidate)
    base = [
        config["command"],
        "--config",
        str(path),
        "--output-dir",
        str(tmp_path / "out"),
        "--deadline",
        DEADLINE_TEXT,
    ]
    assert rx.main(base) == rx.EXIT_USAGE
    missing = base + ["--candidate-ledger", str(tmp_path / "missing")]
    assert rx.main(missing) == rx.EXIT_REFUSED
    run = rx.run_taxi if config["command"] == "taxi" else rx.run_lunar
    with pytest.raises(rx.Refused, match="ledger"):
        run(config, path.read_bytes(), tmp_path / "out", FUTURE)
    with pytest.raises(rx.Refused, match="ledger"):
        rx.Attempt(tmp_path / "out", config, path.read_bytes(), FUTURE, ledger=None)
    assert not (tmp_path / "out").exists() and not (tmp_path / "missing").exists()


def test_ledger_tokens_are_create_once_and_kept(tmp_path):
    ledger = rx.CandidateLedger(tmp_path)
    token = ledger.claim("taxi-A", "execution", {"output_dir": "x"})
    path = tmp_path / "taxi-A.execution.json"
    data = path.read_bytes()
    assert sha(data) == token["sha256"] == sha(rx.token_bytes(token["contents"]))
    assert json.loads(data) == token["contents"]
    assert token["contents"]["ledger"] == str(tmp_path.resolve())
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(rx.Refused, match="already has"):
        ledger.claim("taxi-A", "execution", {"output_dir": "y"})
    with pytest.raises(rx.Refused, match="already has"):
        ledger.require_unclaimed("taxi-A", "execution")
    ledger.require_unclaimed("taxi-B", "execution")
    ledger.require_unclaimed("taxi-A", "heldout")
    assert path.read_bytes() == data
    with pytest.raises(ValueError):
        ledger.token_path("taxi-C", "execution")


def test_lunar_candidate_cannot_run_again_from_another_output_dir(
    tmp_path,
    monkeypatch,
):
    no_science(monkeypatch)
    config = rx.build_config("lunar-A")
    ledger = new_ledger(tmp_path)
    ledger.claim("lunar-A", "execution", {"output_dir": "earlier attempt"})
    with pytest.raises(rx.Refused, match="already has"):
        rx.run_lunar(
            config, json.dumps(config).encode(), tmp_path / "b", FUTURE, ledger=ledger
        )
    path = write_config(tmp_path, "lunar-A")
    argv = [
        "lunar",
        "--config",
        str(path),
        "--output-dir",
        str(tmp_path / "c"),
        "--candidate-ledger",
        str(ledger.path),
        "--deadline",
        DEADLINE_TEXT,
    ]
    assert rx.main(argv) == rx.EXIT_REFUSED
    assert not (tmp_path / "b").exists() and not (tmp_path / "c").exists()


def test_claim_race_leaves_a_refused_record_and_no_acceptance(tmp_path):
    config = rx.build_config("lunar-A")
    raw = json.dumps(config).encode()
    ledger = new_ledger(tmp_path)
    first = rx.Attempt(tmp_path / "one", config, raw, FUTURE, ledger=ledger)
    claim = first.record["candidate_ledger"]["execution_claim"]
    assert claim["contents"]["output_dir"] == str((tmp_path / "one").resolve())
    assert claim["contents"]["config_sha256"] == rx.pilot.config_digest(config)
    on_disk = json.loads((tmp_path / "one" / "attempt.json").read_text())
    assert on_disk["candidate_ledger"]["execution_claim"] == claim
    with pytest.raises(rx.Refused, match="already has"):
        rx.Attempt(tmp_path / "two", config, raw, FUTURE, ledger=ledger)
    record = json.loads((tmp_path / "two" / "attempt.json").read_text())
    assert record["status"] == "refused"
    assert "execution_claim" not in record["candidate_ledger"]
    assert not (tmp_path / "two" / rx.ACCEPTANCE_FILE).exists()
    assert ledger.token_path("lunar-A", "execution").read_bytes() == rx.token_bytes(
        claim["contents"]
    )


def fake_qualification(monkeypatch, config, mean, calls):
    """Stand-ins for the unchanged qualification and loader; no model is read."""

    def qualify(env_name, candidate_path, output_dir, **kwargs):
        calls.append(pathlib.Path(output_dir))
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir()
        n = kwargs["eval_episodes"]
        record = {
            "status": "complete",
            "qualified": True,
            "env_name": env_name,
            "expert_sha256": sha(pathlib.Path(candidate_path).read_bytes()),
            "config": {
                "qualification_seed": kwargs["qualification_seed"],
                "eval_episodes": n,
                "threshold": config["gates"]["per_env_threshold"],
            },
            "qualification": {
                "episode_returns": [mean] * n,
                "episode_lengths": [10] * n,
            },
        }
        (output_dir / rx.task_experts.CANDIDATE_FILE).write_text(json.dumps(record))
        return 0

    def load(qdir, env_name):
        path = pathlib.Path(qdir) / rx.task_experts.CANDIDATE_FILE
        record = json.loads(path.read_text())
        identity = dict.fromkeys(
            ("expert_source", "record_sha256", "config_sha256", "policy_state_sha256"),
            "synthetic",
        )
        identity.update(
            expert_sha256=record["expert_sha256"],
            qualification_mean_return=mean,
            quality={},
        )
        return None, identity

    monkeypatch.setattr(rx.task_experts, "qualify_candidate", qualify)
    monkeypatch.setattr(rx.task_experts, "load_qualified_expert", load)


def test_heldout_games_run_at_most_once_per_candidate(tmp_path, monkeypatch):
    config = rx.build_config("lunar-A")
    calls = []
    fake_qualification(monkeypatch, config, 210.0, calls)
    ledger = new_ledger(tmp_path)
    raw = json.dumps(config).encode()
    attempt = rx.Attempt(tmp_path / "a", config, raw, FUTURE, ledger=ledger)
    model = tmp_path / "a" / "selected" / "model.zip"
    model.parent.mkdir()
    model.write_bytes(b"selected")
    qual = rx.run_qualification(config, model, attempt, sha(b"selected"), FUTURE)
    assert len(calls) == 1 and all(qual["checks"].values())
    held = attempt.record["candidate_ledger"]["heldout_token"]
    assert held["contents"]["checkpoint_sha256"] == sha(b"selected")
    assert held["contents"]["qualification_seed"] == 900112
    assert held["contents"]["execution_claim_sha256"] == (
        attempt.record["candidate_ledger"]["execution_claim"]["sha256"]
    )
    with pytest.raises(rx.Refused, match="heldout token"):
        rx.run_qualification(config, model, attempt, sha(b"selected"), FUTURE)
    assert len(calls) == 1
    assert rx.ledger_problems(attempt.record, config, sha(b"selected"), ledger) == []


# ---------------------------------------------------------------------------
# Taxi end to end with synthetic stand-ins (no env, no SB3 model)
# ---------------------------------------------------------------------------

LABELS = {0: 0, 1: 0, 2: 1}  # the planner labels of raw_table() from state 0


class TinyPolicy(th.nn.Module):
    """Linear stand-in for an SB3 policy: actor ``action_net``, critic ``value_net``."""

    def __init__(self, labels=None, fail_after=None):
        super().__init__()
        self.action_net = th.nn.Linear(500, 3)
        self.value_net = th.nn.Linear(500, 1)
        with th.no_grad():
            self.action_net.weight.zero_()
            self.action_net.bias.zero_()
            for state, action in (labels or {}).items():
                self.action_net.weight[action, state] = 5.0
        self.fail_after, self.losses = fail_after, 0

    def set_training_mode(self, mode):
        self.train(mode)

    def get_distribution(self, obs):
        self.losses += 1
        if self.fail_after is not None and self.losses > self.fail_after:
            raise RuntimeError("synthetic update failure")
        return th.distributions.Categorical(logits=self.action_net(obs))

    def predict(self, obs, deterministic=True):
        with th.no_grad():
            obs = th.as_tensor(np.asarray(obs), dtype=th.float32)
            return self.action_net(obs).argmax(-1).numpy(), None


def tiny_ce(policy, obs, actions):
    with th.no_grad():
        logits = policy.action_net(th.as_tensor(obs, dtype=th.float32))
        logp = th.log_softmax(logits, -1)
    rows = th.arange(len(actions))
    return float(-logp[rows, th.as_tensor(actions, dtype=th.long)].mean())


class TableEnv:
    """Steps raw_table() from state 0; observations are one-hot of size 500."""

    def __init__(self):
        self.unwrapped = types.SimpleNamespace(
            P=raw_table(), initial_state_distrib=np.array([1.0, 0.0, 0.0, 0.0])
        )

    def _obs(self):
        return np.eye(500, dtype=np.float32)[self.state]

    def reset(self, seed):
        self.state = 0
        return self._obs(), {}

    def step(self, action):
        _, nxt, reward, done = self.unwrapped.P[self.state][int(action)][0]
        self.state = nxt
        return self._obs(), reward, done, False, {}


def taxi_world(monkeypatch, policy=None, save_fails=False, load_fails=False):
    """Patch the tasks, env factory, CE and SB3 PPO with synthetic stand-ins."""
    world = types.SimpleNamespace(built=[], envs=[])

    class FakePPO:
        def __init__(self, policy_name, venv, policy_kwargs, seed, device, verbose):
            self.policy = TinyPolicy(**(policy or {}))
            world.built.append(self)

        def predict(self, obs, deterministic=True):
            return self.policy.predict(obs, deterministic)

        def save(self, path):
            if save_fails:
                raise OSError("synthetic save failure")
            state = {k: v.tolist() for k, v in self.policy.state_dict().items()}
            pathlib.Path(path).write_text(json.dumps(state))

        @classmethod
        def load(cls, path, device):
            if load_fails:
                raise RuntimeError("synthetic reload failure")
            model = cls.__new__(cls)
            model.policy = TinyPolicy()
            state = json.loads(pathlib.Path(path).read_text())
            model.policy.load_state_dict({k: th.tensor(v) for k, v in state.items()})
            return model

    class FakeVenv:
        def __init__(self):
            self.envs = [TableEnv()]
            world.envs.append(self)

        def close(self):
            pass

    digest = rx.dynamics_sha256(raw_table())
    monkeypatch.setattr(rx.tasks, "get_task", lambda name: types.SimpleNamespace())
    monkeypatch.setattr(
        rx.tasks,
        "describe",
        lambda task: {"episode_cap": 200, "tabular_dynamics_sha256": digest},
    )
    monkeypatch.setattr(rx.tasks, "task_sha256", lambda task: "task")
    monkeypatch.setattr(
        rx.tasks,
        "reachable_nonterminal",
        lambda name: ([0, 1, 2], {"reset_support": [0]}),
    )
    monkeypatch.setattr(rx.tasks, "venv_episode_cap", lambda venv: 200)
    monkeypatch.setattr(
        rx,
        "_mask_representatives",
        lambda env, reach: {"r": {"representatives": [0], "all_reachable": True}},
    )
    monkeypatch.setattr(rx.env_utils, "make_env", lambda *a, **k: FakeVenv())
    monkeypatch.setattr(rx.eval_utils, "compute_sampled_action_ce", tiny_ce)
    sb3 = types.ModuleType("stable_baselines3")
    sb3.PPO = FakePPO
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
    return world


def test_taxi_candidate_and_heldout_run_once_across_output_dirs(
    tmp_path,
    monkeypatch,
):
    world = taxi_world(monkeypatch, policy={"labels": LABELS})
    config = rx.build_config("taxi-A")
    calls = []
    fake_qualification(monkeypatch, config, 10.0, calls)
    path = write_config(tmp_path, "taxi-A")
    ledger = new_ledger(tmp_path)

    def argv(out):
        return [
            "taxi",
            "--config",
            str(path),
            "--output-dir",
            str(tmp_path / out),
            "--candidate-ledger",
            str(ledger.path),
            "--deadline",
            DEADLINE_TEXT,
        ]

    assert rx.main(argv("first")) == rx.EXIT_RELEASED
    first = tmp_path / "first"
    acc = json.loads((first / rx.ACCEPTANCE_FILE).read_text())
    assert acc["release_allowed"] and acc["failed_checks"] == []
    assert acc["training_method"] == rx.TAXI_METHOD
    assert set(acc["files"]) >= {"distillation", "reload", "checkpoint"}
    tokens = sorted(p.name for p in ledger.path.iterdir())
    assert tokens == ["taxi-A.execution.json", "taxi-A.heldout.json"]
    out = rx.verify_acceptance(first, path, ledger=ledger)
    assert out["candidate_ledger"] == str(ledger.path)
    # A second invocation into a new output directory runs nothing.
    assert rx.main(argv("second")) == rx.EXIT_REFUSED
    assert not (tmp_path / "second").exists()
    assert len(world.built) == 1 and len(calls) == 1


def test_distill_actor_counts_and_keeps_partial_fit(monkeypatch):
    monkeypatch.setattr(rx.eval_utils, "compute_sampled_action_ce", tiny_ce)
    obs = np.eye(500, dtype=np.float32)[:3]
    labels = np.array([0, 0, 1])
    sets = [{0}, {0}, {1}]
    cfg = {
        "learning_rate": 0.001,
        "max_updates": 3,
        "stop_rule": {"canonical_agreement": 1.0, "self_ce_max": 0.7},
    }
    now = rx.rollouts.utc_now
    fit = rx.distill_actor(TinyPolicy(), obs, labels, sets, cfg, FUTURE, now)
    assert fit["stop_reason"] == "max_updates" and "error" not in fit
    assert fit["updates"] == fit["final"]["updates"] == 3
    assert fit["forward_state_evaluations"] == (3 * 3 + 2) * 3
    broken = rx.distill_actor(
        TinyPolicy(fail_after=1), obs, labels, sets, cfg, FUTURE, now
    )
    assert broken["stop_reason"] == "error"
    assert broken["error"]["message"] == "synthetic update failure"
    assert broken["updates"] == broken["final"]["updates"] == 1
    assert broken["forward_state_evaluations"] == (2 * 2 + 1) * 3


@pytest.mark.parametrize(
    "failure",
    ["update_raises", "reload_raises", "save_raises", "fit_write_raises"],
)
def test_taxi_failure_paths_keep_model_and_fit(tmp_path, monkeypatch, failure):
    converged = {"labels": LABELS}
    world = taxi_world(
        monkeypatch,
        policy={"fail_after": 2} if failure == "update_raises" else converged,
        save_fails=failure == "save_raises",
        load_fails=failure == "reload_raises",
    )
    if failure == "fit_write_raises":
        original = rx._write_new_json

        def write(path, payload):
            if pathlib.Path(path).name == "distillation.json":
                raise OSError("synthetic fit write failure")
            return original(path, payload)

        monkeypatch.setattr(rx, "_write_new_json", write)
    config = rx.build_config("taxi-A")
    calls = []
    fake_qualification(monkeypatch, config, 10.0, calls)
    ledger = new_ledger(tmp_path)
    raw = json.dumps(config).encode()
    out = tmp_path / "a"
    assert rx.run_taxi(config, raw, out, FUTURE, ledger=ledger) == 1
    assert calls == [] and len(world.built) == 1
    attempt = json.loads((out / "attempt.json").read_text())
    acc = json.loads((out / rx.ACCEPTANCE_FILE).read_text())
    assert attempt["status"] == acc["status"] == "failed"
    assert not acc["release_allowed"]
    assert "heldout_token" not in attempt["candidate_ledger"]
    reasons = " ".join(attempt["failure_reasons"])
    fit_path = out / "distill" / "distillation.json"
    zip_path = out / "distill" / "model.zip"
    if failure != "fit_write_raises":
        fit = json.loads(fit_path.read_text())
        assert acc["files"]["distillation"]["sha256"] == rx.pilot.sha256_file(fit_path)
        assert attempt["distillation"]["file_sha256"] == rx.pilot.sha256_file(fit_path)
    if failure == "update_raises":
        assert fit["stop_reason"] == "error"
        assert fit["error"]["message"] == "synthetic update failure"
        assert fit["updates"] == fit["final"]["updates"] == 2
        assert fit["forward_state_evaluations"] == (2 * 3 + 2) * 3
        assert attempt["costs"]["distillation_updates"] == 2
        assert attempt["distillation"]["error"] == fit["error"]
        assert fit["checkpoint"]["sha256"] == rx.pilot.sha256_file(zip_path)
        assert attempt["checkpoint_sha256"] == fit["checkpoint"]["sha256"]
        assert not (out / "distill" / "reload.json").exists()
        assert "model and partial fit kept" in reasons
    elif failure == "reload_raises":
        assert fit["stop_reason"] == "stop_rule_met" and "reload" not in fit
        assert fit["final"]["canonical_agreement_count"] == 3
        assert fit["policy_state_sha256"] is not None
        reload_ev = json.loads((out / "distill" / "reload.json").read_text())
        assert "synthetic reload failure" in reload_ev["error"]
        assert acc["files"]["reload"]["sha256"] is not None
        assert acc["checks"]["distill_canonical_agreement"]
        assert not acc["checks"]["reload_canonical_agreement"]
    elif failure == "save_raises":
        assert fit["stop_reason"] == "stop_rule_met" and not zip_path.exists()
        assert fit["preservation_errors"] == [
            "model save: OSError: synthetic save failure"
        ]
        assert "checkpoint" not in acc["files"]
        assert "preservation failed: model save" in reasons
        assert acc["checks"]["distill_canonical_agreement"]  # yet not released
    else:
        assert not fit_path.exists() and zip_path.is_file()
        assert attempt["distillation"]["stop_reason"] == "stop_rule_met"
        assert "preservation failed: fit evidence" in reasons
        assert "distillation" not in acc["files"]
    # The candidate is used up whatever the failure; nothing is retried.
    assert ledger.token_path("taxi-A", "execution").is_file()
    with pytest.raises(rx.Refused, match="already has"):
        rx.run_taxi(config, raw, tmp_path / "retry", FUTURE, ledger=ledger)
    assert len(world.built) == 1


def test_runner_and_tests_have_no_em_dash():
    dash = chr(0x2014)
    assert dash not in RUNNER.read_text()
    assert dash not in pathlib.Path(__file__).read_text()
