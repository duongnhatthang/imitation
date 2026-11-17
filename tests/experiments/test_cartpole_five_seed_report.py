"""Behavior of the five-seed CartPole results adapter on synthetic inputs only.

One coherent synthetic bundle (frozen design, manifests, inventories, 5 data
jobs, 30 runs and placeholder checkpoints) is built once per module, copied per
test and broken in exactly one place. No real experiment record is read, and
nothing produced here is scientific evidence.
"""

import ast
import hashlib
import json
import math
import pathlib
import shutil

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ANALYZER = REPO_ROOT / "experiments" / "agnostic" / "analyze_cartpole_five_seed.py"
SRC = REPO_ROOT / "src" / "imitation" / "experiments"


def _load():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cartpole_five_seed", ANALYZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fs = _load()
SEEDS = fs.SEEDS
MASK = fs.MASK
REMOTE = "/home/alice/study/runs/five-seed"
SPAN = 500.0 - 22.95


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, indent=1, sort_keys=True).encode()
    path.write_bytes(raw)
    return sha(raw)


# ----------------------------------------------------------- synthetic bundle

FILES = {f: sha(f.encode()) for f in fs.SOURCE_FILES}
SOURCE = {
    "files": FILES,
    "combined_sha256": fs.config_digest(FILES),
    "git_revision": None,
    "git_dirty": None,
}
TASK = {"env_name": "CartPole-v1", "episode_cap": 500, "declared": {"ladder": [MASK]}}
EXPERT = sha(b"expert")
IDENTITY = {
    "expert_sha256": EXPERT,
    "policy_state_sha256": sha(b"policy"),
    "record_sha256": sha(b"prep"),
    "preparation_dir_name": "CartPole-v1",
}
REFERENCE = {"expert_return": 500.0, "random_return": 22.95, "expert_self_ce": 0.2}
DESIGN = {
    "version": 1,
    "env_name": "CartPole-v1",
    "seeds": list(SEEDS),
    "excluded_seeds": [300, 301],
    "methods": ["bc", "bc_iid", "ftl"],
    "restrictions": ["identity", MASK],
    "labels": 1000,
    "eval_episodes": 100,
    "episode_cap": 500,
    "bc_epochs": 20,
    "learning_rate": 0.001,
    "entropy_weight": 0.001,
    "warm_start": False,
    "ftl_beta": 0,
    "mixture": False,
    "bc_prefix": False,
    "reference_data_sha256": sha(b"reference"),
    "reference": REFERENCE,
    "expert_sha256": EXPERT,
    "source_sha256": sha(b"campaign"),
    "expert_identity": IDENTITY,
    "task_sha256": fs.config_digest(TASK),
}
DESIGN_RAW = json.dumps(DESIGN, indent=2).encode()
EXPECTED = fs.Expected(
    design_sha256=sha(DESIGN_RAW),
    scientific_source_combined_sha256=SOURCE["combined_sha256"],
)
PKGS = {"python": "3.10.0", "numpy": "1.26.4"}


def base_record(schema):
    return {
        "protocol": fs.PROTOCOL,
        "schema": schema,
        "env_name": "CartPole-v1",
        "episode_cap": 500,
        "expert_sha256": EXPERT,
        "preparation": IDENTITY,
        "deadline": {"job_limit_seconds": 7200.0},
        "source": SOURCE,
        "package_versions": PKGS,
        "task": TASK,
        "task_sha256": DESIGN["task_sha256"],
        "claim": {"pid": 1},
        "status": "complete",
        "error": None,
        "interruption": None,
        "snapshot": {"final": True},
    }


def make_data(root, s):
    rng = np.random.default_rng(s)
    out = root / fs.data_id(s)
    out.mkdir(parents=True)
    pool_len = [400, 400, 300]
    stream_len = rng.integers(10, 500, size=1000).tolist()
    datasets = {}
    for name, lengths in (("pool", pool_len), ("stream", stream_len)):
        obs = rng.normal(size=(1000, 4)).astype(np.float32)
        acts = rng.integers(0, 2, size=1000).astype(np.int64)
        step = np.array([n // 2 for n in lengths] if name == "stream" else range(1000))
        path = out / fs.DATA_FILES[name]
        np.savez_compressed(
            path, obs=obs, acts=acts, next_obs=obs, dones=np.zeros(1000, bool),
            episode_index=np.arange(1000), step_index=step,
        )
        stats = {
            "episodes": len(lengths),
            "env_steps": sum(lengths),
            "expert_predict_calls": sum(lengths),
            "retained_labels": 1000,
            "episode_lengths": lengths,
        }
        if name == "pool":
            stats["overshoot_transitions"] = sum(lengths) - 1000
        datasets[name] = {
            "file": fs.DATA_FILES[name],
            "file_sha256": sha(path.read_bytes()),
            "pairs_sha256": fs.pairs_sha256(obs, acts),
            "n": 1000,
            "stats": stats,
        }
    cfg = {
        "seed": s, "budget": 1000, "baseline_episodes": 500, "pool": fs.POOL_RULE,
        "stream": fs.STREAM_RULE, "expert_action_rule": fs.ACTION_RULE,
        "allow_relaxed_expert": False,
    }
    rec = dict(
        base_record(fs.DATA_SCHEMA), seed=s, config=cfg,
        config_sha256=fs.config_digest(cfg), datasets=datasets,
        # Each data job's own baselines differ from the common reference.
        baselines={"expert_return": 500.0, "random_return": 21.0, "expert_self_ce": 0.2},
        baselines_costs={
            "expert_episodes": 500, "random_episodes": 500, "expert_env_steps": 250000,
            "random_env_steps": 11000, "env_steps_counted": 261000,
            "elapsed_seconds": 30.0,
        },
    )
    return rec, write_json(out / fs.RESULT_FILE, rec)


def returns_for(method, restriction, seed, n):
    """Synthetic returns: method and budget effects plus a shared seed effect."""
    level = {"ftl": 0.9, "bc_iid": 0.7, "bc": 0.6}[method] - 0.1 * (restriction == MASK)
    level = level * min(1.0, 0.2 + n / 300) + 0.02 * (seed - 3102)
    rng = np.random.default_rng(seed * 7 + n)
    mean = 22.95 + level * SPAN
    return np.clip(np.round(rng.normal(mean, 40.0, 100)), 1, 500).tolist()


def evaluation(method, restriction, seed, n, run_dir):
    ret = returns_for(method, restriction, seed, n)
    ckpt = f"checkpoints/{method}_CartPole-v1_seed{seed}/round-{n:05d}.pt"
    (run_dir / ckpt).parent.mkdir(parents=True, exist_ok=True)
    (run_dir / ckpt).write_bytes(f"{method}{restriction}{seed}{n}".encode())
    frac = 0.3 / (1 + n / 50) + 0.01 * (seed - 3100)
    return {
        "episode_returns": [float(v) for v in ret],
        "normalized_return": round((np.mean(ret) - 22.95) / SPAN, 6),
        "disagreement_rate": round(frac, 6),
        "rollout_cross_entropy": round(2 * frac + 0.05, 6),
        "expert_rollout_cross_entropy": 0.2,
        "d_eval_size": int(sum(ret)),
        "checkpoint": ckpt,
    }


def fit_fields(n):
    n_val = int(0.1 * n)
    if n_val < 32:
        examples, inner = n, {
            "inner_es_stop_epoch": 20, "inner_es_val_nll_trajectory": [],
            "inner_es_val_nll_best": None, "inner_es_fallback": "dataset_too_small",
        }
    else:
        traj = [0.9, 0.7, 0.6, 0.65, 0.66]
        examples, inner = n - n_val, {
            "inner_es_stop_epoch": 5, "inner_es_val_nll_trajectory": traj,
            "inner_es_val_nll_best": 0.6, "inner_es_fallback": None,
        }
    batch = min(32, n, examples)
    return dict(
        inner, train_examples=examples, train_batch_size=batch,
        train_batches_per_epoch=examples // batch, train_drop_last=True,
    )


def make_run(root, method, restriction, seed, data_rec, data_sha):
    job = fs.run_id(method, restriction, seed)
    run_dir = root / job
    run_dir.mkdir(parents=True)
    pool, stream = data_rec["datasets"]["pool"], data_rec["datasets"]["stream"]
    lengths = stream["stats"]["episode_lengths"]
    with np.load(root / fs.data_id(seed) / fs.DATA_FILES["stream"]) as z:
        s_obs, s_acts = z["obs"], z["acts"]
    records = []
    if method == "bc":
        records.append(dict(
            round=0, n_observations=1000, reference="flat",
            trained_pairs_sha256=pool["pairs_sha256"],
            logical=fs.pool_logical_cost(pool["stats"]["episode_lengths"], 1000),
            wall_seconds={"fit": 1.0, "eval": 1.0},
            **fit_fields(1000), **evaluation("bc", restriction, seed, 1000, run_dir),
        ))
        trained = pool["pairs_sha256"]
    else:
        schedule = set(fs.ONLINE_SCHEDULE)
        for t in range(1001):
            r = {"round": t, "n_observations": t, "wall_seconds": {}}
            if t:
                r.update(
                    collection_steps=20 * t if method == "ftl" else stream["stats"]["env_steps"],
                    trajectories_collected_this_round=1 if method == "ftl" else 0,
                    train_cross_entropy=0.5, **fit_fields(t),
                )
                if method == "bc_iid":
                    r["logical"] = {"labels": t, "episodes": t, "env_steps": sum(lengths[:t])}
            else:
                r.update(train_cross_entropy=None)
            if t in schedule:
                r.update(evaluation(method, restriction, seed, t, run_dir))
                if method == "bc_iid" and t:
                    r["prefix_pairs_sha256"] = fs.pairs_sha256(s_obs[:t], s_acts[:t])
            else:
                r.update({k: None for k in fs.METRICS}, d_eval_size=None)
            records.append(r)
        trained = stream["pairs_sha256"] if method == "bc_iid" else sha(job.encode())
    evals = [r for r in records if r.get("d_eval_size") is not None]
    eval_steps = sum(r["d_eval_size"] for r in evals)
    fits = [r for r in records if r.get("inner_es_stop_epoch") is not None]
    steps = records[-1]["collection_steps"] if method == "ftl" else 0
    episodes = 1000 if method == "ftl" else 0
    q = {k: eval_steps for k in ("expert_predict_calls", "expert_action_entries",
                                 "learner_predict_calls")}
    qc = {k: steps for k in q}
    total = steps + eval_steps
    kind = fs.SHARED_DATASET[method]
    accounting = {
        "observed": {"env_steps": total},
        "completed_records": {
            "collection": dict(qc, env_steps=steps, episodes=episodes),
            "evaluation": dict(q, evaluations=len(evals), episodes=100 * len(evals),
                               env_steps=eval_steps),
            "retained_labels": 1000,
            "training": {
                "fits": len(fits),
                "epochs": sum(r["inner_es_stop_epoch"] for r in fits),
                "early_stopped_fits": sum(r["inner_es_stop_epoch"] < 20 for r in fits),
                "fixed_budget_fallback_fits": sum(
                    r["inner_es_fallback"] is not None for r in fits
                ),
            },
        },
        "in_flight": {"env_steps": 0},
        "totals": {"exact": True, "env_steps": total, **{k: total for k in q}},
        "reconciled": True,
        "shared_acquisition": None if kind is None else {
            "dataset": kind, "pairs_sha256": data_rec["datasets"][kind]["pairs_sha256"],
            "charged_to_this_run": False,
        },
        "wall_seconds": {"total": 10.0},
    }
    restriction_rec = fs.RESTRICTION_RECORDS[restriction]
    experiment = dict(
        fs.EXPERIMENT_PINS, algo=method, seed=seed,
        subsample_strategy=fs.SUBSAMPLE[method], outer_early_stop_patience=5,
    )
    cfg = {
        "seed": seed, "method": method, "restriction_id": restriction,
        "restriction": restriction_rec,
        "restriction_sha256": fs.config_digest(restriction_rec),
        "policy": fs.POLICY_RULE, "beta": fs.BETA[method], "eval_episodes": 100,
        "eval_deterministic": True,
        "eval_budgets": [1000] if method == "bc" else fs.ONLINE_SCHEDULE,
        "fixed_bc": fs.FIXED_BC_RULE if method == "bc" else None,
        "batch_rule": fs.BATCH_RULE, "mixture": False, "experiment": experiment,
        "task_sha256": DESIGN["task_sha256"], "allow_relaxed_expert": False,
        "distinct_observations": {"pool": 1000, "masked_pool": 1000},
        "reference": {
            "source": "pinned reference data job",
            "data_result_sha256": DESIGN["reference_data_sha256"],
            "expected_sha256": DESIGN["reference_data_sha256"],
            "producer_seed": 300,
            "baselines": REFERENCE,
            "baselines_costs": {"env_steps_counted": 261000, "elapsed_seconds": 30.0},
        },
    }
    rec = dict(
        base_record(fs.RUN_SCHEMA), seed=seed, method=method,
        restriction_id=restriction, config=cfg, config_sha256=fs.config_digest(cfg),
        data={
            "data_result_sha256": data_sha,
            "pool_pairs_sha256": pool["pairs_sha256"],
            "stream_pairs_sha256": stream["pairs_sha256"],
            "baselines": data_rec["baselines"],
            "acceptance": {
                "rule": "same_source", "expected_data_sha256": None,
                "data_result_sha256": data_sha, "checks": fs.ACCEPTANCE_CHECKS,
                "producer_protocol": fs.PROTOCOL, "producer_schema": fs.DATA_SCHEMA,
                "producer_source": SOURCE,
                "current_source_combined_sha256": SOURCE["combined_sha256"],
            },
        },
        trained_pairs_sha256=trained, records=records, accounting=accounting,
    )
    return write_json(run_dir / fs.RESULT_FILE, rec)


def manifest(campaign, jobs):
    return {
        "version": 1, "campaign_id": campaign, "source_root": f"{REMOTE}/snapshot/src",
        "source_sha256": DESIGN["source_sha256"], "stage_deadline": "2026-10-02T12:00:00Z",
        "overall_deadline": "2026-10-07T01:01:00Z", "jobs": jobs,
    }


def job_entry(job_id, command, extra, fields):
    out = f"{REMOTE}/{job_id}"
    argv = ["/home/alice/miniconda3/bin/python", "-m", fs.PRODUCER_MODULE, command,
            "--env", "CartPole-v1", "--expert-dir", f"{REMOTE}/expert",
            "--output-dir", out, "--deadline", "2026-10-02T12:00:00Z",
            "--job-limit-seconds", "7200"] + extra
    return {
        "id": job_id, "kind": "learner", "argv": argv, "cwd": f"{REMOTE}/snapshot",
        "expected_result": f"{out}/result.json", "timeout_seconds": 7200,
        "expected_fields": dict(fields, status="complete", protocol=fs.PROTOCOL),
    }


def inventory(m, shas):
    canonical = json.dumps(m, sort_keys=True, separators=(",", ":"))
    jobs = {}
    for job in m["jobs"]:
        jobs[job["id"]] = {
            "kind": job["kind"], "expected_result": job["expected_result"],
            "state": "complete", "state_reason": None, "result_sha256": shas[job["id"]],
            "attempts": [{
                "number": 1, "infrastructure_retry": False, "state": "complete",
                "started_at": "2026-10-01T10:00:00Z", "ended_at": "2026-10-01T10:30:00Z",
                "elapsed_seconds": 1800.0, "argv": job["argv"], "cwd": job["cwd"],
                "stdout_log": f"{REMOTE}/logs/{job['id']}.log", "exit_code": 0,
                "failure_kind": None, "termination": {"stop_reason": None},
                "archived_prior_result": None,
                "validation": {"ok": True, "sha256": shas[job["id"]],
                               "path": job["expected_result"]},
            }],
        }
    return {
        "version": 1, "campaign_id": m["campaign_id"],
        "manifest_sha256": sha(canonical.encode()), "source_root": m["source_root"],
        "source_sha256": m["source_sha256"],
        "controller_runs": [{"started_at": "2026-10-01T10:00:00Z",
                             "ended_at": "2026-10-01T20:00:00Z", "exit_code": 0,
                             "retry_requests": []}],
        "jobs": jobs,
    }


def build(root):
    root.mkdir(parents=True)
    (root / "control").mkdir()
    (root / "control" / "frozen-design.json").write_bytes(DESIGN_RAW)
    shas, data_jobs, run_jobs = {}, [], []
    for s in SEEDS:
        rec, shas[fs.data_id(s)] = make_data(root, s)
        data_jobs.append(job_entry(fs.data_id(s), "data", ["--seed", str(s)], {
            "schema": fs.DATA_SCHEMA, "seed": s, "config_sha256": rec["config_sha256"],
        }))
        for m in fs.METHODS:
            for r in fs.RESTRICTIONS:
                job = fs.run_id(m, r, s)
                shas[job] = make_run(root, m, r, s, rec, shas[fs.data_id(s)])
                run_jobs.append(job_entry(job, "run", [
                    "--data-dir", f"{REMOTE}/{fs.data_id(s)}", "--seed", str(s),
                    "--method", m, "--restriction", r,
                    "--reference-data-dir", f"{REMOTE}/reference",
                    "--expected-reference-sha256", DESIGN["reference_data_sha256"],
                ], {"schema": fs.RUN_SCHEMA, "seed": s}))
    for name, jobs in (("data", data_jobs), ("run", run_jobs)):
        m = manifest(f"five-seed-{name}", jobs)
        write_json(root / "control" / f"{name}-manifest.json", m)
        write_json(root / "control" / f"{name}-inventory.json", inventory(m, shas))


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    root = tmp_path_factory.mktemp("template") / "bundle"
    build(root)
    return root


@pytest.fixture
def bundle(template, tmp_path):
    root = tmp_path / "bundle"
    shutil.copytree(template, root)
    (tmp_path / "design.json").write_bytes(DESIGN_RAW)
    return root


def run(bundle, extra=()):
    out = bundle.parent / "out"
    code = fs.main(
        ["--bundle-root", str(bundle), "--frozen-design", str(bundle.parent / "design.json"),
         "--output-dir", str(out), *extra],
        expected=EXPECTED,
    )
    return code, out


def rewrite(bundle, job_id, mutate):
    """Change a record; the controller is made to have validated the new bytes."""
    path = bundle / job_id / fs.RESULT_FILE
    rec = json.loads(path.read_text())
    mutate(rec)
    new = write_json(path, rec)
    kind = "data" if job_id.startswith("data-") else "run"
    inv_path = bundle / "control" / f"{kind}-inventory.json"
    inv = json.loads(inv_path.read_text())
    row = inv["jobs"][job_id]
    row["result_sha256"] = row["attempts"][0]["validation"]["sha256"] = new
    write_json(inv_path, inv)


def inventory_rows(out):
    inv = json.loads((out / fs.INVENTORY).read_text())
    return {row["job"]: row for row in inv["jobs"]}


# -------------------------------------------------------------------- tests


def test_band_is_training_seed_se_not_episode_se():
    seed_values = [0.50, 0.52, 0.48, 0.51, 0.49]
    curves = {}
    for m in fs.ONLINE:
        for r in fs.RESTRICTIONS:
            curves[(m, r)] = {
                s: [{
                    "labels": 1000, "normalized_return": v, "mean_return": v,
                    "disagreement_rate": 0.1, "rollout_cross_entropy": 0.3,
                    "expert_rollout_cross_entropy": 0.2,
                    # Huge within-evaluation spread must not leak into the band.
                    "within_eval_episode_se_normalized": 5.0,
                }] for s, v in zip(SEEDS, seed_values)
            }
    bc = {r: {s: curves[("ftl", r)][s][0] for s in SEEDS} for r in fs.RESTRICTIONS}
    online, flat = fs.aggregate(curves, bc)
    stat = online[("ftl", "identity")][0]["normalized_return"]
    want = float(np.std(seed_values, ddof=1) / math.sqrt(5))
    assert stat["se"] == pytest.approx(want)
    assert stat["mean"] == pytest.approx(np.mean(seed_values))
    assert stat["se"] < 0.01  # an episode SE of 5.0 or pooled episodes would be far larger
    assert online[("ftl", "identity")][0]["disagreement_rate"]["se"] == 0.0
    assert flat["identity"]["labels"] == 1000


def test_paired_difference_keeps_seed_covariance():
    base = [0.1, 0.9, 0.3, 0.7, 0.5]  # large shared seed effect

    def pt(v):
        return {"labels": 1000, "normalized_return": v, "disagreement_rate": v,
                "rollout_cross_entropy": v, "expert_rollout_cross_entropy": v}
    curves = {
        ("ftl", r): {s: [pt(b + 0.05)] for s, b in zip(SEEDS, base)}
        for r in fs.RESTRICTIONS
    }
    curves.update({
        ("bc_iid", r): {s: [pt(b)] for s, b in zip(SEEDS, base)} for r in fs.RESTRICTIONS
    })
    bc = {r: {s: pt(b - 0.1) for s, b in zip(SEEDS, base)} for r in fs.RESTRICTIONS}
    paired = fs.paired_differences(curves, bc)["identity"]
    d_iid = paired["ftl_minus_bc_iid"][0]["normalized_return"]
    d_bc = paired["ftl_minus_fixed_bc"][0]["normalized_return"]
    assert d_iid["mean"] == pytest.approx(0.05) and d_iid["se"] == pytest.approx(0, abs=1e-12)
    assert d_bc["mean"] == pytest.approx(0.15) and d_bc["se"] == pytest.approx(0, abs=1e-12)
    unpaired = math.sqrt(2) * np.std(base, ddof=1) / math.sqrt(5)
    assert unpaired > 0.1  # what an unpaired SE would wrongly report


def test_complete_grid_flat_bc_and_private_outputs(bundle):
    code, out = run(bundle)
    assert code == fs.EXIT_OK
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [fs.FIG_CURVES, fs.SUMMARY, fs.POINTS, fs.REPORT]
    )
    points = json.loads((out / fs.POINTS).read_text())
    ftl = points["online"]["ftl/identity"][-1]["normalized_return"]
    per_seed = [
        (np.mean(returns_for("ftl", "identity", s, 1000)) - 22.95) / SPAN for s in SEEDS
    ]
    assert ftl["per_seed"] == pytest.approx(per_seed, abs=1e-6)
    assert ftl["se"] == pytest.approx(np.std(ftl["per_seed"], ddof=1) / math.sqrt(5))
    assert set(points["fixed_bc_flat"]) == set(fs.RESTRICTIONS)
    assert points["fixed_bc_flat"][MASK]["labels"] == 1000

    online = {tuple(k.split("/")): rows for k, rows in points["online"].items()}
    fig = fs.curves_figure(online, points["fixed_bc_flat"])
    bc_lines = [
        ln for ln in fig.axes[0].lines
        if ln.get_color() == fs.COLOR["bc"] and len(ln.get_xdata()) == 2
    ]
    assert len(bc_lines) == 2
    for ln in bc_lines:
        assert list(ln.get_xdata()) == [0, 1000]
        assert ln.get_ydata()[0] == ln.get_ydata()[1]
    legend = " ".join(t.get_text() for t in fig.legends[0].get_texts())
    assert "Fixed offline BC, 1,000 training labels" in legend
    # Shared fixed view on the normalized panel; plotted means are unchanged.
    ret_ax = fig.axes[0]
    assert fs.NORMALIZED_RETURN_YLIM == (-0.5, 1.5)
    assert ret_ax.get_yscale() == "linear"
    assert ret_ax.get_ylim() == fs.NORMALIZED_RETURN_YLIM
    means = [p["normalized_return"]["mean"] for p in online[("ftl", "identity")]]
    assert any(list(ln.get_ydata()) == means for ln in ret_ax.lines)
    fs.plt.close(fig)

    summary = json.loads((out / fs.SUMMARY).read_text())
    assert summary["n_seeds"] == 5 and summary["excluded_seeds"] == [300, 301]
    assert summary["costs"]["reused_expert_preparation"]["cost"] is None
    for path in out.iterdir():
        data = path.read_bytes()
        for token in (b"/home/", b"alice", b"miniconda", str(bundle.parent).encode(),
                      chr(0x2014).encode()):
            assert token not in data, (path.name, token)


def test_below_random_returns_are_kept_unchanged(tmp_path, monkeypatch):
    """Masked fixed BC and BC-iid below the random reference are valid outcomes."""
    honest = returns_for

    def below_random(method, restriction, seed, n):
        if method == "ftl" or restriction != MASK:
            return honest(method, restriction, seed, n)
        rng = np.random.default_rng(seed * 7 + n)
        return np.clip(np.round(rng.normal(15.0, 3.0, 100)), 1, 500).tolist()

    monkeypatch.setitem(globals(), "returns_for", below_random)
    bundle = tmp_path / "bundle"
    build(bundle)
    (tmp_path / "design.json").write_bytes(DESIGN_RAW)
    code, out = run(bundle)
    assert code == fs.EXIT_OK
    summary = json.loads((out / fs.SUMMARY).read_text())
    assert len(summary["inventory"]) == 35
    assert all(row["usable"] for row in summary["inventory"])

    def stored(method, seed):
        return round((np.mean(below_random(method, MASK, seed, 1000)) - 22.95) / SPAN, 6)

    want = {m: [stored(m, s) for s in SEEDS] for m in ("bc", "bc_iid")}
    assert max(want["bc"] + want["bc_iid"]) < 0
    per_seed = {
        (row["method"], row["seed"]): row["normalized_return"]
        for row in summary["per_seed_final"] if row["restriction"] == MASK
    }
    for m in ("bc", "bc_iid"):
        assert [per_seed[(m, s)] for s in SEEDS] == want[m]

    points = json.loads((out / fs.POINTS).read_text())
    iid_curve = points["online"][f"bc_iid/{MASK}"]
    assert all(row["normalized_return"]["mean"] < 0 for row in iid_curve)
    for m, stat in (("bc", points["fixed_bc_flat"][MASK]["normalized_return"]),
                    ("bc_iid", iid_curve[-1]["normalized_return"])):
        assert stat["per_seed"] == want[m]
        assert stat["mean"] == pytest.approx(np.mean(want[m])) and stat["mean"] < 0
        assert summary["final"][f"{m}/{MASK}"]["normalized_return"] == stat

    ftl = points["online"][f"ftl/{MASK}"][-1]["normalized_return"]["per_seed"]
    for name, m in (("ftl_minus_bc_iid", "bc_iid"), ("ftl_minus_fixed_bc", "bc")):
        diff = summary["paired_final"][MASK][name]["normalized_return"]
        assert diff["per_seed"] == pytest.approx(np.subtract(ftl, want[m]))


def _timed_out(bundle):
    job = fs.run_id("ftl", MASK, 3103)
    rewrite(bundle, job, lambda r: r.update(status="partial", snapshot={"final": True}))
    inv_path = bundle / "control" / "run-inventory.json"
    inv = json.loads(inv_path.read_text())
    row = inv["jobs"][job]
    row.update(state="timed_out", state_reason="job_timeout", result_sha256=None)
    row["attempts"][0].update(state="timed_out", exit_code=None, validation=None,
                              termination={"stop_reason": "job_timeout"})
    write_json(inv_path, inv)
    return job


def test_incomplete_grid_keeps_inventory_and_withholds_aggregates(bundle):
    job = _timed_out(bundle)
    code, out = run(bundle)
    assert code == fs.EXIT_INCOMPLETE
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [fs.INVENTORY, fs.INVENTORY_REPORT]
    )
    rows = inventory_rows(out)
    assert rows[job]["usable"] is False and rows[job]["record_status"] == "partial"
    assert any("timed_out" in p for p in rows[job]["problems"])
    assert sum(r["usable"] for r in rows.values()) == 34
    inv = json.loads((out / fs.INVENTORY).read_text())
    assert inv["aggregates"] is None


@pytest.mark.parametrize("breakage", ["duplicate_key", "stray_pilot_dir", "retry"])
def test_duplicate_or_foreign_rows_refuse_everything(bundle, breakage):
    if breakage == "duplicate_key":
        path = bundle / "control" / "run-inventory.json"
        text = path.read_text()
        path.write_text(text.replace('"version": 1', '"version": 1, "version": 1', 1))
    elif breakage == "stray_pilot_dir":
        (bundle / "run-ftl-identity-seed300").mkdir()
    else:
        path = bundle / "control" / "data-inventory.json"
        inv = json.loads(path.read_text())
        row = inv["jobs"][fs.data_id(3100)]
        row["attempts"].append(dict(row["attempts"][0], number=2))
        write_json(path, inv)
    code, out = run(bundle)
    assert code == fs.EXIT_REFUSED
    assert not out.exists()


def _wrong_mask(r):
    r["config"]["restriction"] = dict(r["config"]["restriction"], zeroed_indices=[0])


def _other_seed_data(r):
    r["data"]["data_result_sha256"] = sha(b"another seed")


def _bc_prefix(r):
    r["records"].append(dict(r["records"][0], round=1, n_observations=500))


def _own_reference(r):
    r["config"]["reference"] = {"source": "own data job",
                                "data_result_sha256": r["data"]["data_result_sha256"],
                                "baselines": r["data"]["baselines"]}


def _wrong_baseline(r):
    r["config"]["reference"]["baselines"] = dict(REFERENCE, random_return=22.0)


def _alias_restriction(r):
    r["config"]["restriction"] = dict(r["config"]["restriction"], kind="alias")


def _negative_cross_entropy(r):
    r["records"][-1]["rollout_cross_entropy"] = -0.01


@pytest.mark.parametrize(
    "job, mutate, phrase",
    [
        (fs.run_id("ftl", MASK, 3101), _wrong_mask, "restriction"),
        (fs.run_id("bc_iid", "identity", 3102), _other_seed_data, "same-seed data"),
        (fs.run_id("bc", MASK, 3104), _bc_prefix, "prefix fits are not accepted"),
        (fs.run_id("ftl", "identity", 3100), _own_reference, "pinned common reference"),
        (fs.run_id("bc", "identity", 3103), _wrong_baseline, "expert 500 / random 22.95"),
        (fs.run_id("bc_iid", MASK, 3100), _alias_restriction, "restriction"),
        (fs.run_id("ftl", "identity", 3101), _negative_cross_entropy, "invalid metrics"),
    ],
)
def test_mismatched_record_or_reference_is_never_aggregated(bundle, job, mutate, phrase):
    def apply(r):
        mutate(r)
        r["config_sha256"] = fs.config_digest(r["config"])

    rewrite(bundle, job, apply)
    code, out = run(bundle)
    assert code == fs.EXIT_INCOMPLETE
    assert not (out / fs.SUMMARY).exists() and not (out / fs.FIG_CURVES).exists()
    rows = inventory_rows(out)
    assert rows[job]["usable"] is False
    assert any(phrase in p for p in rows[job]["problems"]), rows[job]["problems"]


def test_wrong_design_or_tampered_artifact_refused(bundle):
    lines = [
        f"{sha(p.read_bytes())}  {p.relative_to(bundle).as_posix()}"
        for p in sorted(bundle.rglob("*")) if p.is_file()
    ]
    listing = bundle.parent / "SHA256SUMS"
    listing.write_text("\n".join(lines) + "\n")
    ckpt = bundle / fs.run_id("ftl", MASK, 3104) / "checkpoints" / \
        "ftl_CartPole-v1_seed3104" / "round-01000.pt"
    ckpt.write_bytes(b"tampered")
    code, out = run(bundle, ["--artifact-manifest", str(listing)])
    assert code == fs.EXIT_REFUSED and not out.exists()

    (bundle.parent / "design.json").write_bytes(DESIGN_RAW.replace(b"22.95", b"22.0"))
    code, out = run(bundle)
    assert code == fs.EXIT_REFUSED and not out.exists()


def test_rule_strings_match_the_frozen_producer_source():
    def assigned(path, name):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            ):
                return node.value
        raise AssertionError(f"{name} not found in {path.name}")

    rp = SRC / "agnostic" / "restriction_pilot.py"
    for name in ("POOL_RULE", "STREAM_RULE", "BATCH_RULE", "FIXED_BC_RULE"):
        assert ast.literal_eval(assigned(rp, name)) == getattr(fs, name)
    assert ast.literal_eval(assigned(SRC / "agnostic" / "classical.py", "ACTION_RULE")) \
        == fs.ACTION_RULE
    base = ast.literal_eval(assigned(rp, "_SOURCE_FILES"))
    extra = ast.literal_eval(assigned(SRC / "agnostic" / "task_pilot.py", "SOURCE_FILES").right)
    assert tuple(base) + tuple(extra) == fs.SOURCE_FILES
    assert fs.POLICY_RULE.split(";")[0] in rp.read_text()
