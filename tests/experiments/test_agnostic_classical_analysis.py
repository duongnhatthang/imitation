"""Tests for the fail-closed classical agnostic analysis.

Inputs are produced by the real ``classical.audit`` and ``classical.run_cell``
from a synthetic hand-set linear expert passed through the real stage 1
preparation (only the PPO trainer is replaced), on a tiny fake env with
CartPole-shaped observations and a fake clock before the hard cap. The
fixtures support no scientific or public-benchmark claim.
"""

import datetime
import itertools
import json
import pathlib
import shutil

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO

from imitation.experiments.agnostic import analyze_classical as ac
from imitation.experiments.agnostic import classical, pilot, quantized, run_classical
from imitation.experiments.ftrl import env_utils

UTC = datetime.timezone.utc
START = datetime.datetime(2026, 1, 1, tzinfo=UTC)
DEADLINE = datetime.datetime(2026, 6, 1, tzinfo=UTC)
ENV = "CartPole-v1"
REPS = ["mild", "severe"]
SEEDS = [1000, 1001, 1002]
AUDIT_SEED = 500


class FakeCartEnv(gym.Env):
    """CartPole-shaped observations; a non-expert action may end the episode."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def _draw(self):
        scale = np.array([0.6, 1.0, 0.08, 1.0])
        self.obs = (self.np_random.uniform(-1, 1, 4) * scale).astype(np.float32)
        return self.obs

    def _expert(self):
        # Linear expert below: action 1 iff obs0 + obs1 + 10 obs2 + 10 obs3 > 0.
        o = self.obs
        return int(o[0] + o[1] + 10 * o[2] + 10 * o[3] > 0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._draw(), {}

    def step(self, action):
        reward = float(1 + self.obs[0])
        terminated = int(action) != self._expert() and self.np_random.random() < 0.3
        return self._draw(), reward, terminated, False, {}


class FakeAcroEnv(FakeCartEnv):
    """Acrobot-shaped observations (cos, sin of two angles, two velocities)."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), np.float32)
    action_space = gym.spaces.Discrete(3)

    def _draw(self):
        t1, t2 = self.np_random.uniform(-np.pi, np.pi, 2)
        v1, v2 = self.np_random.uniform(-1, 1, 2)
        self.obs = np.array(
            [np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2), v1, v2], np.float32
        )
        return self.obs

    def _expert(self):
        return 2 if self.obs[4] < 0 else 0


FAKE_ENVS = {
    "CartPole-v1": ("AgnosticAnalysisFakeCart-v0", FakeCartEnv),
    "Acrobot-v1": ("AgnosticAnalysisFakeAcro-v0", FakeAcroEnv),
}
for _fake_id, _entry in FAKE_ENVS.values():
    if _fake_id not in gym.registry:
        gym.register(id=_fake_id, entry_point=_entry, max_episode_steps=10)

# Hand-set linear experts, one logit row per action. The Acrobot rule (action 2
# iff the first joint's velocity is negative) also qualifies on the real Acrobot,
# which stage 1 preparation requires.
EXPERT_LOGITS = {
    "CartPole-v1": {1: [1.0, 1.0, 10.0, 10.0]},
    "Acrobot-v1": {2: [0, 0, 0, 0, -10.0, 0], 0: [0, 0, 0, 0, 10.0, 0]},
}


class Clock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return START + datetime.timedelta(seconds=self.calls)


def _save_linear_expert(path, env_name):
    env = env_utils.make_env(env_name, n_envs=1, rng=np.random.default_rng(0))
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[], vf=[])),
        device="cpu",
        seed=0,
    )
    with th.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        for action, row in EXPERT_LOGITS[env_name].items():
            model.policy.action_net.weight[action] = th.tensor(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    env.close()


def _produce_env(base, env_name, n_actions):
    """Prepare, audit and run every cell of one env; return its manifest entry."""
    prep = base / "prep" / env_name
    fake_id = FAKE_ENVS[env_name][0]

    def trainer(env_name, cache_dir, rng, seed, convergence_override=None):
        _save_linear_expert(pathlib.Path(cache_dir) / env_name / "model.zip", env_name)

    assert (
        run_classical.prepare_expert(
            env_name,
            prep,
            seed=0,
            qualification_seed=1234,
            eval_episodes=3,
            deadline=datetime.datetime(2100, 1, 1, tzinfo=UTC),
            trainer=trainer,
        )
        == 0
    )
    audit_dir = base / "audits" / env_name
    assert (
        classical.audit(
            env_name,
            prep,
            audit_dir,
            seed=AUDIT_SEED,
            deadline=DEADLINE,
            episodes=200,
            clock=Clock(),
            env_factory=lambda: gym.make(fake_id),
        )
        == 0
    )
    for rep, seed in itertools.product(REPS, SEEDS):
        assert (
            classical.run_cell(
                env_name,
                prep,
                rep,
                base / "results" / env_name / rep / "seed-{}".format(seed),
                seed=seed,
                deadline=DEADLINE,
                budget=32,
                batch=16,
                checkpoints=[16, 32],
                eval_episodes=4,
                clock=Clock(),
                env_factory=lambda: gym.make(fake_id),
            )
            == 0
        )
    record = json.loads((prep / pilot.PREPARATION_FILE).read_text())
    return {
        "env_name": env_name,
        "n_actions": n_actions,
        "expert_sha256": record["checkpoint"]["sha256"],
        "preparation_config_sha256": record["config_sha256"],
        "quantizer_sha256": {r: ac.quantizer_sha256(env_name, r) for r in REPS},
        "audit": {
            "result_sha256": pilot.sha256_file(audit_dir / "result.json"),
            "samples_sha256": pilot.sha256_file(audit_dir / classical.AUDIT_DATA_FILE),
            "seed": AUDIT_SEED,
            "episodes": 200,
            "delta": classical.DEFAULT_DELTA,
        },
    }


def _manifest(envs):
    return {
        "schema": ac.MANIFEST_SCHEMA,
        "study": "synthetic fixture",
        "cell_protocol": classical.CELL_SCHEMA,
        "audit_protocol": classical.AUDIT_SCHEMA,
        "source_stage2_files": classical.source_identity()["stage2_files"],
        "source_stage1_sha256": pilot.source_fingerprint()["combined_sha256"],
        "package_versions": pilot.package_versions(),
        "candidate_env_order": [ENV, "Acrobot-v1", "MountainCar-v0"],
        "selection_rule": "first two qualified candidates with a positive bound",
        "envs": envs,
        "representations": REPS,
        "seeds": SEEDS,
        "excluded_pilot_seeds": [0, 1, 2],
        "budget": 32,
        "batch": 16,
        "checkpoints": [16, 32],
        "eval_episodes": 4,
        "analysis": {
            "confidence_level": 0.95,
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 7,
            "auc_budget_range": [16, 32],
        },
    }


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    base = tmp_path_factory.mktemp("campaign")
    entry = _produce_env(base, ENV, 2)
    return {
        "results": base / "results",
        "audits": base / "audits",
        "manifest": _manifest([entry]),
    }


@pytest.fixture
def work(campaign, tmp_path):
    shutil.copytree(campaign["results"], tmp_path / "results")
    shutil.copytree(campaign["audits"], tmp_path / "audits")
    return {
        "root": tmp_path,
        "results": tmp_path / "results",
        "audits": tmp_path / "audits",
        "manifest": json.loads(json.dumps(campaign["manifest"])),
    }


def _run(w, manifest=None):
    path = w["root"] / "manifest.json"
    path.write_text(json.dumps(manifest or w["manifest"]))
    out = w["root"] / "out"
    code = ac.main(
        [
            "--manifest",
            str(path),
            "--results-root",
            str(w["results"]),
            "--audit-root",
            str(w["audits"]),
            "--output-dir",
            str(out),
        ]
    )
    written = out / "summary.json"
    summary = json.loads(written.read_text()) if written.exists() else None
    return code, summary, out


def _cell_path(w, rep="mild", seed=1001):
    return w["results"] / ENV / rep / "seed-{}".format(seed) / "result.json"


def _edit(path, fn):
    rec = json.loads(path.read_text())
    fn(rec)
    path.write_text(json.dumps(rec))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_complete_tiny_analysis_recomputes_estimates_and_writes_figures(work):
    code, summary, out = _run(work)
    assert code == ac.EXIT_OK and summary["analysis_status"] == "complete"
    assert summary["inventory"]["cells_complete"] == 6
    audit = json.loads((work["audits"] / ENV / "result.json").read_text())
    for c in summary["results"]["conditions"]:
        rep = c["representation"]
        recs = [json.loads(_cell_path(work, rep, s).read_text()) for s in SEEDS]
        final = [r["checkpoints"][-1] for r in recs]
        diffs = [
            np.mean(f["ftl_final"]["eval"]["returns"])
            - np.mean(f["bc_iid_final"]["eval"]["returns"])
            for f in final
        ]
        primary = c["primary"]
        assert primary["n_seeds"] == 3 and primary["estimate"] == pytest.approx(
            np.mean(diffs)
        )
        assert set(primary["intervals"]) == {"0.95", "0.975"}
        # Eligibility is the audit's own certificate, not an analysis choice.
        assert (
            c["eligibility"]["positive_bound"]
            is audit["representations"][rep]["bound"]["positive_certificate"]
        )
        floor = c["eligibility"]["reference_floor"]["empirical_min_disagreement"]
        arms = c["curves"]["arms"]
        for arm in ac.ARMS:
            for est in arms[arm]["reference_disagreement"]:
                assert est["estimate"] >= floor - 1e-12
        # Oracle x-axis: FTL spends B entries; BC-iid spends its driving steps.
        assert arms["ftl_final"]["training_expert_action_entries"][-1]["mean"] == 32
        bc_steps = np.mean([sum(r["retained"]["bc_iid"]["lengths"]) for r in recs])
        assert arms["bc_iid_final"]["training_expert_action_entries"][-1][
            "mean"
        ] == pytest.approx(bc_steps)
    ledger = summary["accounting"]["per_condition"][0]
    assert ledger["physical_training"]["retained_labels"] == 2 * 32 * 3
    assert ledger["physical_training"]["fits"] == 3 * (2 + 2 + 2)
    assert "unavailable" in summary["accounting"]["expert_preparation"]
    names = sorted(f["path"] for f in summary["figures"])
    assert names == [
        "figures/return_vs_retained_labels.pdf",
        "figures/return_vs_retained_labels.png",
        "figures/return_vs_training_expert_entries.pdf",
        "figures/return_vs_training_expert_entries.png",
    ]
    assert all((out / n).stat().st_size > 0 for n in names)
    report = (out / "report.md").read_text()
    for phrase in (
        "Primary paired estimates",
        "uniform slack",
        "Mixture versus final",
        "No agnostic guarantee",
        "Shared provenance",
        "operator must verify",
    ):
        assert phrase in report
    text = (out / "summary.json").read_text() + report
    assert str(work["root"]) not in text and chr(0x2014) not in text


# ---------------------------------------------------------------------------
# Statistics units
# ---------------------------------------------------------------------------


def test_bootstrap_resamples_training_seeds_not_evaluation_episodes():
    # Episodes vary widely within each run, but every seed has difference 0.
    ftl = [[0.0, 10.0, 0.0, 10.0]] * 3
    bc = [[5.0, 5.0, 5.0, 5.0]] * 3
    est = ac.paired_estimate(ftl, bc, ac.seed_index(3, 500, 0), [0.95])
    assert est["n_seeds"] == 3 and est["estimate"] == 0.0
    assert est["intervals"]["0.95"] == [0.0, 0.0] and est["degenerate"]
    assert "not a universal result" in est["interval_note"]
    # Within-run precision is reported separately, never folded in.
    assert est["episode_mc_se_of_seed_mean"] > 0
    single = ac.paired_estimate(ftl[:1], bc[:1], ac.seed_index(1, 500, 0), [0.95])
    assert single["intervals"] is None and single["seed_sd"] is None


def test_reference_floor_counts_every_bin_and_bounds_every_table():
    counts = np.zeros((9, 2), np.int64)
    counts[0] = [30, 10]
    counts[4] = [5, 15]
    floor = ac.reference_floor(counts, 0.01)
    expected = quantized.alias_lower_bound(counts, 0.01)
    assert floor["K"] == 9 and floor["observed_bins"] == 2
    for key in ("empirical_min_disagreement", "slack", "lower_bound"):
        assert floor[key] == pytest.approx(expected[key])
    for table in itertools.product([0, 1], repeat=9):
        assert ac.reference_disagreement(table, counts) >= (
            floor["empirical_min_disagreement"] - 1e-12
        )


# ---------------------------------------------------------------------------
# Fail-closed checks
# ---------------------------------------------------------------------------


def _final(rec):
    return rec["checkpoints"][-1]


TAMPER = {
    "fixed_bc_evaluated": (
        lambda r: _final(r)["fixed_bc"].update(eval=_final(r)["bc_iid_final"]["eval"]),
        "alias",
    ),
    "shared_data_counted_twice": (
        lambda r: r["physical_costs"]["training"].update(retained_labels=96),
        "physical",
    ),
    "ftl_oracle_counter": (
        lambda r: _final(r)["train_costs_at_checkpoint"]["ftl"].update(
            expert_action_entries=33
        ),
        "FTL counters",
    ),
    "initial_history": (
        lambda r: r["ftl_behavior_tables"][0].__setitem__(0, 1),
        "initial",
    ),
    "mean_return": (
        lambda r: _final(r)["ftl_final"]["eval"].update(
            mean_return=_final(r)["ftl_final"]["eval"]["mean_return"] + 1
        ),
        "mean_return",
    ),
    "mixture_index": (
        lambda r: _final(r)["ftl_mixture"]["mixture_indices"].__setitem__(0, 2),
        "mixture",
    ),
    "counts": (
        lambda r: r["retained"]["bc_iid"]["counts"][0].__setitem__(0, 99),
        "counts differ",
    ),
    "eval_seeds": (
        lambda r: _final(r)["bc_iid_final"]["eval"]["reset_seeds"].reverse(),
        "shared seeds",
    ),
    "source": (
        lambda r: r["source"]["stage2_files"].update({"classical.py": "0" * 64}),
        "provenance mismatch: source",
    ),
    "expert": (lambda r: r.update(expert_sha256="0" * 64), "expert_sha256"),
}


@pytest.mark.parametrize("case", sorted(TAMPER))
def test_tampered_cell_fails_closed_without_estimates(work, case):
    fn, needle = TAMPER[case]
    _edit(_cell_path(work), fn)
    code, summary, out = _run(work)
    assert code == ac.EXIT_FAILED and summary["analysis_status"] == "failed"
    assert summary["results"] is None and not (out / "figures").exists()
    (bad,) = [f for f in summary["failures"] if "seed-1001" in f["where"]]
    assert needle in bad["detail"]


@pytest.mark.parametrize("field", ["expert", "audit_result", "audit_seed"])
def test_manifest_identity_mismatch_fails_closed(work, field):
    m = work["manifest"]
    env = m["envs"][0]
    if field == "expert":
        env["expert_sha256"] = "1" * 64
    elif field == "audit_result":
        env["audit"]["result_sha256"] = "1" * 64
    else:
        env["audit"]["seed"] = AUDIT_SEED + 1
    code, summary, _ = _run(work, m)
    assert code == ac.EXIT_FAILED and summary["results"] is None
    assert summary["failure_count"] >= 1


def test_audit_bound_is_recomputed_from_samples(work):
    path = work["audits"] / ENV / "result.json"
    _edit(path, lambda r: r["representations"]["severe"]["bound"].update(slack=0.0))
    work["manifest"]["envs"][0]["audit"]["result_sha256"] = pilot.sha256_file(path)
    code, summary, _ = _run(work)
    assert code == ac.EXIT_FAILED
    assert "differs from recomputation" in summary["failures"][0]["detail"]


@pytest.mark.parametrize("problem", ["missing_cell", "partial_cell", "missing_audit"])
def test_incomplete_inputs_keep_inventory_and_give_no_estimates(work, problem):
    if problem == "missing_cell":
        shutil.rmtree(_cell_path(work).parent)
    elif problem == "partial_cell":
        _edit(_cell_path(work), lambda r: r.update(status="partial"))
    else:
        shutil.rmtree(work["audits"] / ENV)
    code, summary, out = _run(work)
    assert code == ac.EXIT_INCOMPLETE and summary["analysis_status"] == "incomplete"
    assert summary["results"] is None and summary["figures"] == []
    assert not (out / "figures").exists()
    assert len(summary["inventory"]["cells"]) == 6
    statuses = [
        a["status"]
        for a in summary["inventory"]["cells"] + summary["inventory"]["audits"]
    ]
    assert statuses.count("complete") == 6
    assert "no primary estimates" in (out / "report.md").read_text().lower()


@pytest.mark.parametrize("name", ["seed-0", "seed-01001"])
def test_pilot_or_duplicate_seed_directory_fails(work, name):
    src = _cell_path(work).parent
    shutil.copytree(src, src.parent / name)
    code, summary, _ = _run(work)
    assert code == ac.EXIT_FAILED
    assert any(f["where"] == "results root" for f in summary["failures"])


def test_invalid_invocations_write_nothing(work):
    bad = json.loads(json.dumps(work["manifest"]))
    bad["seeds"] = [0] + SEEDS
    code, summary, out = _run(work, bad)
    assert code == ac.EXIT_INVALID and not out.exists()
    bad = json.loads(json.dumps(work["manifest"]))
    bad["envs"][0]["quantizer_sha256"]["mild"] = "2" * 64
    assert _run(work, bad)[0] == ac.EXIT_INVALID and not out.exists()
    out.mkdir()
    assert _run(work)[0] == ac.EXIT_INVALID
    assert list(out.iterdir()) == []


# ---------------------------------------------------------------------------
# Audit samples, producer status, timestamps and malformed records
# ---------------------------------------------------------------------------


def _audit_path(w):
    return w["audits"] / ENV / "result.json"


def _repin_audit(w):
    """Re-pin the manifest to the (tampered) audit files, as a forger would."""
    spec = w["manifest"]["envs"][0]["audit"]
    spec["samples_sha256"] = pilot.sha256_file(w["audits"] / ENV / AUDIT_NPZ)
    spec["result_sha256"] = pilot.sha256_file(_audit_path(w))


AUDIT_NPZ = classical.AUDIT_DATA_FILE


def _edit_samples(w, fn):
    path = w["audits"] / ENV / AUDIT_NPZ
    with np.load(path) as z:
        arrays = dict(z)
    fn(arrays)
    np.savez(path, **arrays)
    digest = pilot.sha256_file(path)
    _edit(_audit_path(w), lambda r: r["samples_file"].update(sha256=digest))
    _repin_audit(w)


def _failure(summary, where):
    (bad,) = [f for f in summary["failures"] if where in f["where"]]
    return bad["detail"]


SAMPLE_TAMPER = {
    "behavior_stream": (
        lambda z: z.update(expert_behavior=~z["expert_behavior"]),
        "choice stream",
    ),
    "sample_index": (lambda z: z["index"].fill(0), "select_index(u, length)"),
    "label_dtype": (lambda z: z.update(label=z["label"].astype(float)), "dtype"),
    "label_domain": (lambda z: z["label"].__setitem__(0, 2), "labels outside"),
    "obs_shape": (lambda z: z.update(obs=z["obs"][:, :3]), "shape"),
    "extra_array": (lambda z: z.update(extra=z["index"]), "exactly"),
    "reset_seed": (lambda z: z["reset_seed"].__setitem__(0, 0), "reference stream"),
}


@pytest.mark.parametrize("case", sorted(SAMPLE_TAMPER))
def test_repinned_audit_samples_must_follow_frozen_streams(work, case):
    fn, needle = SAMPLE_TAMPER[case]
    _edit_samples(work, fn)
    code, summary, out = _run(work)
    assert code == ac.EXIT_FAILED and summary["results"] is None
    assert not (out / "figures").exists()
    assert needle in _failure(summary, "CartPole-v1/result.json")


def test_audit_counters_are_exact_per_behavior(work):
    # Moving one episode between behaviors keeps the old total-only check happy.
    def shift(r):
        r["costs"]["expert_episodes"]["completed_episodes"] += 1
        r["costs"]["random_episodes"]["completed_episodes"] -= 1

    _edit(_audit_path(work), shift)
    _repin_audit(work)
    code, summary, _ = _run(work)
    assert code == ac.EXIT_FAILED
    assert "expert_episodes counters" in _failure(summary, "CartPole-v1/result.json")


def test_complete_audit_with_error_or_interruption_fails(work):
    _edit(
        _audit_path(work),
        lambda r: r.update(error={"type": "RuntimeError"}, interruption={}),
    )
    _repin_audit(work)
    code, summary, _ = _run(work)
    assert code == ac.EXIT_FAILED
    assert "interruption or error" in _failure(summary, "CartPole-v1/result.json")


def test_mixture_indices_are_regenerated_from_the_frozen_stream(work):
    # In range for R = 2 policies, but not the producer's draw.
    _edit(
        _cell_path(work),
        lambda r: _final(r)["ftl_mixture"].update(
            mixture_indices=[1 - i for i in _final(r)["ftl_mixture"]["mixture_indices"]]
        ),
    )
    code, summary, _ = _run(work)
    assert code == ac.EXIT_FAILED
    assert "frozen mixture stream" in _failure(summary, "seed-1001")


def _set_status(status):
    def fn(r):
        r.update(status=status, error={"type": "RuntimeError"}, interruption={})
        if status == "running":
            r.update(finished_at_utc=None, error=None)

    return fn


@pytest.mark.parametrize("kind", ["cell", "audit"])
@pytest.mark.parametrize(
    "status, code, art_status",
    [
        ("failed", ac.EXIT_FAILED, "failed"),
        ("partial", ac.EXIT_INCOMPLETE, "incomplete"),
        ("running", ac.EXIT_INCOMPLETE, "incomplete"),
        ("done", ac.EXIT_FAILED, "invalid"),
    ],
)
def test_producer_status_is_failed_or_incomplete(work, kind, status, code, art_status):
    path = _cell_path(work) if kind == "cell" else _audit_path(work)
    _edit(path, _set_status(status))
    if kind == "audit":
        _repin_audit(work)
    got, summary, out = _run(work)
    assert got == code and summary["results"] is None
    assert not (out / "figures").exists()
    inv = summary["inventory"]
    assert len(inv["cells"]) == 6 and len(inv["audits"]) == 1
    (art,) = [a for a in inv["cells"] + inv["audits"] if a["status"] != "complete"]
    assert art["status"] == art_status and art["producer_status"] == status
    failed = [f["where"] for f in summary["failures"]]
    assert failed == ([art["path"]] if code == ac.EXIT_FAILED else [])


TIME_TAMPER = {
    "late_complete": (
        lambda r: r.update(finished_at_utc="2100-01-01T00:00:00+00:00"),
        "after its effective deadline",
    ),
    "naive": (
        lambda r: r.update(started_at_utc=r["started_at_utc"][:19]),
        "not an aware UTC",
    ),
    "missing": (lambda r: r.pop("finished_at_utc"), "not an aware UTC"),
    "not_utc": (
        lambda r: r.update(started_at_utc="2026-01-01T09:00:00+09:00"),
        "not an aware UTC",
    ),
    "uncapped": (
        lambda r: r.update(
            requested_deadline_utc="2100-01-01T00:00:00+00:00",
            effective_deadline_utc="2100-01-01T00:00:00+00:00",
        ),
        "clamped to the hard cap",
    ),
    "finished_before_start": (
        lambda r: r.update(finished_at_utc=r["claim"]["claimed_at_utc"]),
        "finished before it started",
    ),
}


@pytest.mark.parametrize("kind", ["cell", "audit"])
@pytest.mark.parametrize("case", sorted(TIME_TAMPER))
def test_timestamps_fail_closed(work, kind, case):
    fn, needle = TIME_TAMPER[case]
    path = _cell_path(work) if kind == "cell" else _audit_path(work)
    _edit(path, fn)
    if kind == "audit":
        _repin_audit(work)
    code, summary, _ = _run(work)
    assert code == ac.EXIT_FAILED and summary["results"] is None
    where = "seed-1001" if kind == "cell" else "CartPole-v1/result.json"
    assert needle in _failure(summary, where)


def test_partial_record_may_finish_after_its_deadline(work):
    # A deadline stop legitimately ends late; it is incomplete, not failed.
    _edit(
        _cell_path(work),
        lambda r: r.update(
            status="partial", finished_at_utc="2026-07-01T00:00:00+00:00"
        ),
    )
    assert _run(work)[0] == ac.EXIT_INCOMPLETE


@pytest.mark.parametrize(
    "kind, fn",
    [
        ("cell", lambda r: r.update(source="x")),
        ("cell", lambda r: r.update(retained="x")),
        ("cell", lambda r: r.update(seeds=[])),
        ("cell", lambda r: r["checkpoints"].__setitem__(0, "x")),
        ("cell", lambda r: _final(r).update(train_costs_at_checkpoint=[])),
        ("audit", lambda r: r.update(representations="x")),
        ("audit", lambda r: r.update(samples_file="x")),
    ],
)
def test_malformed_records_are_invalid_artifacts(work, kind, fn):
    path = _cell_path(work) if kind == "cell" else _audit_path(work)
    _edit(path, fn)
    if kind == "audit":
        _repin_audit(work)
    code, summary, out = _run(work)
    assert code == ac.EXIT_FAILED and summary["results"] is None
    assert len(summary["inventory"]["cells"]) == 6
    (art,) = [
        a
        for a in summary["inventory"]["cells"] + summary["inventory"]["audits"]
        if a["status"] != "complete"
    ]
    assert art["status"] in ("invalid", "failed_checks")
    assert (out / "report.md").exists()


@pytest.mark.parametrize(
    "fn",
    [
        lambda m: m["envs"][0].update(env_name=[ENV]),
        lambda m: m.update(representations=[["mild"]]),
        lambda m: m.update(candidate_env_order=[[ENV]]),
        lambda m: m["analysis"].update(auc_budget_range=[16.0, 32]),
    ],
)
def test_malformed_manifest_types_are_invalid(work, fn):
    bad = json.loads(json.dumps(work["manifest"]))
    fn(bad)
    code, _, out = _run(work, bad)
    assert code == ac.EXIT_INVALID and not out.exists()


def test_manifest_source_must_match_the_imported_producer_code(work):
    # Consistent records and manifest, but not the code that regenerates them.
    forged = {"classical.py": "0" * 64}
    for p in list(work["results"].rglob("result.json")) + [_audit_path(work)]:
        _edit(p, lambda r: r["source"]["stage2_files"].update(forged))
    work["manifest"]["source_stage2_files"].update(forged)
    _repin_audit(work)
    code, _, out = _run(work)
    assert code == ac.EXIT_INVALID and not out.exists()


PROVENANCE_TAMPER = {
    "library_version": (
        lambda r: r["package_versions"].update(gymnasium="0.0.0-other"),
        "package_versions",
    ),
    "stage1_digest": (
        lambda r: r["source"]["stage1"].update(combined_sha256="0" * 64),
        "source_stage1",
    ),
    "stage1_file_map": (
        lambda r: r["source"]["stage1"]["files"].update({"util/util.py": "0" * 64}),
        "source_stage1",
    ),
}


@pytest.mark.parametrize("kind", ["cell", "audit"])
@pytest.mark.parametrize("case", sorted(PROVENANCE_TAMPER))
def test_records_must_match_manifest_version_and_stage1_pins(work, kind, case):
    fn, needle = PROVENANCE_TAMPER[case]
    path = _cell_path(work) if kind == "cell" else _audit_path(work)
    _edit(path, fn)
    if kind == "audit":
        _repin_audit(work)
    code, summary, _ = _run(work)
    assert code == ac.EXIT_FAILED and summary["results"] is None
    where = "seed-1001" if kind == "cell" else "CartPole-v1/result.json"
    assert "provenance mismatch: " + needle in _failure(summary, where)


@pytest.mark.parametrize(
    "fn",
    [
        lambda m: m["package_versions"].pop("torch"),
        lambda m: m["package_versions"].update(torch="unavailable"),
        lambda m: m.update(source_stage1_sha256="0" * 64),
    ],
)
def test_manifest_version_and_stage1_pins_are_required(work, fn):
    bad = json.loads(json.dumps(work["manifest"]))
    fn(bad)
    code, _, out = _run(work, bad)
    assert code == ac.EXIT_INVALID and not out.exists()


def test_summary_hashes_every_direct_analysis_source(work):
    _, summary, _ = _run(work)
    here = pathlib.Path(ac.__file__).parent
    assert summary["analysis_sources_sha256"] == {
        name: pilot.sha256_file(here / name)
        for name in (
            "analyze_classical.py",
            "analyze_toy.py",
            "pilot.py",
            "rollouts.py",
            "classical.py",
            "quantized.py",
        )
    }


# ---------------------------------------------------------------------------
# Two environments by both representations: the production family of 4
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_env(campaign, tmp_path_factory):
    base = tmp_path_factory.mktemp("two_env")
    shutil.copytree(campaign["results"], base / "results")
    shutil.copytree(campaign["audits"], base / "audits")
    acro = _produce_env(base, "Acrobot-v1", 3)
    return {
        "results": base / "results",
        "audits": base / "audits",
        "manifest": _manifest(campaign["manifest"]["envs"] + [acro]),
    }


def test_two_env_both_rep_analysis_uses_family_of_four(two_env, tmp_path):
    shutil.copytree(two_env["results"], tmp_path / "results")
    shutil.copytree(two_env["audits"], tmp_path / "audits")
    w = {
        "root": tmp_path,
        "results": tmp_path / "results",
        "audits": tmp_path / "audits",
        "manifest": two_env["manifest"],
    }
    code, summary, out = _run(w)
    assert code == ac.EXIT_OK and summary["analysis_status"] == "complete"
    assert summary["inventory"]["cells_complete"] == 12
    conditions = summary["results"]["conditions"]
    assert [(c["env"], c["representation"]) for c in conditions] == [
        (e, r) for e in (ENV, "Acrobot-v1") for r in REPS
    ]
    for c in conditions:
        primary = c["primary"]
        assert primary["family_size"] == 4
        assert primary["family_level"] == pytest.approx(0.9875)
        assert set(primary["intervals"]) == {"0.95", "0.9875"}
        final = [
            json.loads(
                (w["results"] / c["env"] / c["representation"] / "seed-{}".format(s))
                .joinpath("result.json")
                .read_text()
            )["checkpoints"][-1]
            for s in SEEDS
        ]
        diffs = [
            np.mean(f["ftl_final"]["eval"]["returns"])
            - np.mean(f["bc_iid_final"]["eval"]["returns"])
            for f in final
        ]
        assert primary["per_seed_difference"] == pytest.approx(diffs)
        lo, hi = primary["intervals"]["0.95"]
        flo, fhi = primary["intervals"]["0.9875"]
        assert flo <= lo and hi <= fhi
    report = (out / "report.md").read_text()
    assert "Bonferroni over 4 primary tests" in report
    assert "family 0.9875" in report
    assert len(summary["figures"]) == 4
