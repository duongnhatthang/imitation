"""Behavior of the stronger-mask CartPole pilot report on synthetic inputs only.

One coherent synthetic pilot (manifests, inventories, audit, six runs, the pinned
reused acquisition, preparation and trusted evidence) is built per test and then
broken in exactly one place. No real pilot record is read here.
"""

import copy
import dataclasses
import hashlib
import importlib.util
import json
import pathlib

import matplotlib.colors as mcolors
import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ANALYZER = REPO_ROOT / "experiments" / "agnostic" / "analyze_cartpole_stronger_pilot.py"
_spec = importlib.util.spec_from_file_location("cartpole_stronger_report", ANALYZER)
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

MASK = sp.MASK
PIN_PLACEHOLDER = None
PACKAGES = {
    "python": "3.10.21",
    "numpy": "1.26.4",
    "torch": "2.4.1+cu121",
    "stable_baselines3": "2.2.1",
    "gymnasium": "0.29.1",
    "imitation": "1.0.1.dev0",
}
RUN_FILES = {f"experiments/agnostic/f{i}.py": f"{i:064x}" for i in range(3)}
DATA_FILES = {k: v[::-1] for k, v in RUN_FILES.items()}
BASELINES = {"expert_return": 500.0, "random_return": 20.0, "expert_self_ce": 0.01}
DEADLINE = "2026-10-01T10:28:10.169175+00:00"
PRIVATE = "/home/opuser/study"
CAMPAIGN = "c" * 64
POOL_LENGTHS = [300, 300, 300, 250]
STREAM_LENGTHS = [20 + i % 50 for i in range(1000)]
ELAPSED = {
    "audit": 4.0,
    "run-bc-identity": 30.0,
    f"run-bc-{MASK}": 6.0,
    "run-ftl-identity": 3000.0,
    f"run-ftl-{MASK}": 2500.0,
    "run-bc_iid-identity": 2000.0,
    f"run-bc_iid-{MASK}": 900.0,
}
ACQ_ELAPSED = 77.0


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _dump(obj):
    return (json.dumps(obj, indent=2) + "\n").encode("utf-8")


def _norm(returns):
    mean = sum(returns) / len(returns)
    return round((mean - 20.0) / 480.0, 6)


def _returns(seed, t):
    return [float(min(500, 20 + (t * 7 + seed * 13 + i) % 481)) for i in range(100)]


def _evaluation(seed, t):
    ret = _returns(seed, t)
    return {
        "rollout_cross_entropy": round(0.7 / (t + 1) + seed * 1e-3, 6),
        "expert_rollout_cross_entropy": round(0.01 + seed * 1e-4, 6),
        "normalized_return": _norm(ret),
        "disagreement_rate": round(0.5 / (t + 1), 6),
        "d_eval_size": int(sum(ret)),
        "episode_returns": ret,
        "checkpoint": f"checkpoints/round-{t:05d}.pt",
    }


def _fit(n, early=False):
    n_val = int(0.1 * n)
    if n_val < 32:
        b = min(32, n)
        return {
            "inner_es_stop_epoch": 20,
            "inner_es_val_nll_best": None,
            "inner_es_val_nll_trajectory": [],
            "inner_es_fallback": "dataset_too_small",
            "train_batch_size": b,
            "train_examples": n,
            "train_batches_per_epoch": n // b,
            "train_drop_last": True,
        }
    stop = 6 if early else 20
    traj = [1.0 / (e + 1) for e in range(stop)]
    if early:
        traj = [0.5, 0.4, 0.41, 0.42, 0.43, 0.44]
    ex = n - n_val
    return {
        "inner_es_stop_epoch": stop,
        "inner_es_val_nll_best": min(traj),
        "inner_es_val_nll_trajectory": traj,
        "inner_es_fallback": None,
        "train_batch_size": 32,
        "train_examples": ex,
        "train_batches_per_epoch": ex // 32,
        "train_drop_last": True,
    }


def _account(rec, method):
    """The producer's accounting (run_job.accounting at 796e8a4) of saved records.

    An unfinished record has 17 observed steps of unattributed in-flight work.
    """
    records = rec["records"]
    evals = [r for r in records if r.get("d_eval_size") is not None]
    eval_steps = sum(r["d_eval_size"] for r in evals)
    fits = [r for r in records if r.get("inner_es_stop_epoch") is not None]
    steps = records[-1].get("collection_steps", 0) if method == "ftl" and records else 0
    complete = rec["status"] == "complete"
    counted = steps + eval_steps + (0 if complete else 17)
    exact = complete and counted == steps + eval_steps
    queries = steps + eval_steps if exact else None
    rec["accounting"] = {
        "observed": {
            "env_steps": counted,
            "env_resets": len(evals),
            "finished_episodes": len(evals) * 100,
            "rule": "r",
        },
        "completed_records": {
            "collection": {
                "env_steps": steps,
                "episodes": (
                    sum(r.get("trajectories_collected_this_round", 0) for r in records)
                    if method == "ftl"
                    else 0
                ),
                "expert_predict_calls": steps,
                "expert_action_entries": steps,
                "learner_predict_calls": steps,
            },
            "evaluation": {
                "evaluations": len(evals),
                "episodes": sum(len(r["episode_returns"]) for r in evals),
                "env_steps": eval_steps,
                "expert_predict_calls": eval_steps,
                "expert_action_entries": eval_steps,
                "learner_predict_calls": eval_steps,
            },
            "retained_labels": records[-1]["n_observations"] if records else 0,
            "training": {
                "fits": len(fits),
                "epochs": sum(r["inner_es_stop_epoch"] for r in fits),
                "early_stopped_fits": sum(r["inner_es_stop_epoch"] < 20 for r in fits),
                "fixed_budget_fallback_fits": sum(
                    r["inner_es_fallback"] is not None for r in fits
                ),
            },
        },
        "in_flight": {
            "operation": None if complete else "round",
            "env_steps": counted - steps - eval_steps,
            "expert_predict_calls": 0 if exact else None,
            "learner_predict_calls": 0 if exact else None,
        },
        "totals": {
            "exact": exact,
            "env_steps": counted,
            "expert_predict_calls": queries,
            "expert_action_entries": queries,
            "learner_predict_calls": queries,
        },
        "reconciled": counted == steps + eval_steps,
        "shared_acquisition": None,
        "wall_seconds": {"setup": 1.0, "total": 10.0},
    }


class Pilot:
    """A coherent synthetic pilot; edit attributes, then `write`."""

    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.inputs = self.root / "inputs"
        self.data_dir = self.inputs / "data"
        self.data_dir.mkdir(parents=True)
        prep = {
            "status": "complete",
            "qualified": True,
            "approved": True,
            "env_name": "CartPole-v1",
            "protocol": "agnostic-expert-preparation/1",
            "config_sha256": "1" * 64,
            "qualification": {"mean_return": 500.0},
            "checkpoint": {
                "path": "expert.zip",
                "sha256": "2" * 64,
                "policy_state_sha256": "3" * 64,
            },
        }
        raw = _dump(prep)
        (self.inputs / "preparation.json").write_bytes(raw)
        self.identity = {
            "record_sha256": _sha(raw),
            "protocol": prep["protocol"],
            "config_sha256": "1" * 64,
            "expert_sha256": "2" * 64,
            "policy_state_sha256": "3" * 64,
            "qualification_mean_return": 500.0,
            "preparation_dir_name": "CartPole-v1",
            "action_rule": "a",
        }
        self._write_data()
        self.results = {"audit": self._audit()}
        for job_id in sp.RUN_IDS:
            m, r = sp.split(job_id)
            self.results[job_id] = self._run(m, r)
        self.states = {j: "complete" for j in self.results}
        self.missing = set()
        self.never_launched = set()
        self.failure_kind = {}  # job id -> controller failure kind with exit 0
        self.resumed_failed = set()  # complete rows failed at controller resume
        self.evidence_note = "Operator compared the deployed tree to the commit."
        self.expected = sp.Expected(
            execution_source_commit="e" * 40,
            campaign_source_sha256=CAMPAIGN,
            scientific_source_combined_sha256=sp.config_digest(RUN_FILES),
            data_result_sha256=self.pin,
        )

    # -- builders ---------------------------------------------------------

    def _write_data(self):
        rng = np.random.default_rng(0)
        datasets = {}
        self.arrays = {}
        for name, lengths in (("pool", POOL_LENGTHS), ("stream", STREAM_LENGTHS)):
            obs = rng.normal(size=(1000, 4)).astype(np.float32)
            acts = rng.integers(0, 2, size=1000).astype(np.int64)
            path = self.data_dir / sp.DATA_FILES[name]
            np.savez_compressed(path, obs=obs, acts=acts)
            self.arrays[name] = (obs, acts)
            stats = {
                "elapsed_seconds": 5.0,
                "env_steps": sum(lengths),
                "episodes": len(lengths),
                "env_resets": len(lengths),
                "expert_predict_calls": sum(lengths),
                "retained_labels": 1000,
                "episode_lengths": lengths,
            }
            if name == "pool":
                stats["overshoot_transitions"] = sum(lengths) - 1000
            datasets[name] = {
                "file": path.name,
                "file_sha256": _sha(path.read_bytes()),
                "pairs_sha256": sp.pairs_sha256(obs, acts),
                "n": 1000,
                "stats": stats,
            }
        config = {
            "seed": 300,
            "budget": 1000,
            "baseline_episodes": 500,
            "pool": "p",
            "stream": "s",
            "expert_action_rule": "a",
        }
        self.data = {
            "protocol": sp.DATA_PROTOCOL,
            "schema": sp.DATA_SCHEMA,
            "status": "complete",
            "snapshot": {"final": True},
            "env_name": "CartPole-v1",
            "episode_cap": 500,
            "seed": 300,
            "config": config,
            "config_sha256": sp.config_digest(config),
            "source": {
                "files": DATA_FILES,
                "combined_sha256": sp.config_digest(DATA_FILES),
            },
            "package_versions": PACKAGES,
            "expert_sha256": "2" * 64,
            "preparation": self.identity,
            "baselines": BASELINES,
            "baselines_costs": {
                "elapsed_seconds": 9.0,
                "env_steps_counted": 260000,
                "expert_episodes": 500,
                "random_episodes": 500,
                "expert_predict_calls": 500000,
            },
            "datasets": datasets,
        }
        raw = _dump(self.data)
        (self.data_dir / "result.json").write_bytes(raw)
        self.pin = _sha(raw)

    def _common(self, schema):
        return {
            "protocol": sp.PROTOCOL,
            "schema": schema,
            "env_name": "CartPole-v1",
            "episode_cap": 500,
            "expert_sha256": "2" * 64,
            "preparation": self.identity,
            "source": {
                "files": RUN_FILES,
                "combined_sha256": sp.config_digest(RUN_FILES),
                "git_revision": None,
                "git_dirty": None,
            },
            "package_versions": PACKAGES,
            "status": "complete",
            "error": None,
            "snapshot": {"final": True, "written_at_utc": "2026-10-01T06:00:00+00:00"},
            "deadline": {"job_limit_seconds": 6600.0},
        }

    def _audit(self):
        rec = self._common(sp.AUDIT_SCHEMA)
        rec["deadline"] = {"job_limit_seconds": 570.0}
        rec.update(
            config={
                "restriction_id": MASK,
                "reset_seed": 301,
                "grid": [-1.8, -0.6, 0.6, 1.8],
                "witness_scope": "w",
            },
            audit_status="conflict_found",
            expert_predict_calls=120,
            audit={
                "status": "conflict_found",
                "n_checked_pairs": 120,
                "n_conflicting_pairs": 80,
                "n_invalid_pairs": 0,
                "claim": "c",
                "labels_used_for_training": False,
            },
        )
        return rec

    def _run(self, method, restriction):
        seed = sp.RUN_IDS.index(f"run-{method}-{restriction}")
        experiment = {
            **sp.EXPERIMENT_PINS,
            "algo": method,
            "l2_lambda": 0.0,
            "subsample_strategy": "prefix" if method == "bc" else "uniform",
        }
        config = {
            "seed": 300,
            "method": method,
            "restriction_id": restriction,
            "restriction": {
                "restriction_id": restriction,
                "zeroed_indices": sp.ZEROED[restriction],
                "obs_shape": [4],
                "description": "d",
            },
            "beta": sp.BETA[method],
            "eval_episodes": 100,
            "eval_deterministic": True,
            "eval_budgets": [1000] if method == "bc" else sp.ONLINE_SCHEDULE,
            "fixed_bc": "one fit" if method == "bc" else None,
            "batch_rule": "b",
            "mixture": False,
            "experiment": experiment,
        }
        ds = self.data["datasets"]
        rec = self._common(sp.RUN_SCHEMA)
        acceptance = {
            "rule": "pinned_different_source",
            "expected_data_sha256": self.pin,
            "data_result_sha256": self.pin,
            "checks": sp.ACCEPTANCE_CHECKS,
            "producer_protocol": sp.DATA_PROTOCOL,
            "producer_schema": sp.DATA_SCHEMA,
            "producer_source": self.data["source"],
            "producer_package_versions": PACKAGES,
            "current_source_combined_sha256": sp.config_digest(RUN_FILES),
        }
        rec.update(
            seed=300,
            method=method,
            restriction_id=restriction,
            config=config,
            config_sha256=sp.config_digest(config),
            data={
                "data_result_sha256": self.pin,
                "pool_pairs_sha256": ds["pool"]["pairs_sha256"],
                "stream_pairs_sha256": ds["stream"]["pairs_sha256"],
                "baselines": BASELINES,
                "acceptance": acceptance,
            },
        )
        if method == "bc":
            rec["trained_pairs_sha256"] = ds["pool"]["pairs_sha256"]
            rec["records"] = [
                {
                    "round": 0,
                    "n_observations": 1000,
                    "reference": "flat",
                    "trained_pairs_sha256": ds["pool"]["pairs_sha256"],
                    "logical": {
                        "labels": 1000,
                        "episodes": 4,
                        "env_steps": 1150,
                        "overshoot_transitions": 150,
                    },
                    **_fit(1000),
                    **_evaluation(seed, 1000),
                    "wall_seconds": {"fit": 1.0, "eval": 2.0},
                }
            ]
            return rec
        records = [
            {
                "round": 0,
                "n_observations": 0,
                "train_cross_entropy": None,
                "l2_norm": None,
                "total_loss": None,
                **_evaluation(seed, 0),
                "wall_seconds": {"collect_train_seconds": 0.0, "eval_seconds": 1.0},
            }
        ]
        s_obs, s_acts = self.arrays["stream"]
        for t in range(1, 1001):
            r = {
                "round": t,
                "n_observations": t,
                "collection_steps": 30 * t if method == "ftl" else sum(STREAM_LENGTHS),
                "trajectories_collected_this_round": 1 if method == "ftl" else 0,
                "train_cross_entropy": 0.3,
                "l2_norm": 1.0,
                "total_loss": 0.3,
                "rollout_cross_entropy": None,
                "expert_rollout_cross_entropy": None,
                "normalized_return": None,
                "disagreement_rate": None,
                "d_eval_size": None,
                **_fit(t, early=(t == 333)),
                "wall_seconds": {"collect_train_seconds": 0.5, "eval_seconds": 0.0},
            }
            if method == "bc_iid":
                r["logical"] = {
                    "labels": t,
                    "episodes": t,
                    "env_steps": sum(STREAM_LENGTHS[:t]),
                }
            if t in sp.ONLINE_SCHEDULE:
                r.update(_evaluation(seed, t))
                if method == "bc_iid":
                    r["prefix_pairs_sha256"] = sp.pairs_sha256(s_obs[:t], s_acts[:t])
            records.append(r)
        rec["records"] = records
        rec["trained_pairs_sha256"] = (
            ds["stream"]["pairs_sha256"] if method == "bc_iid" else "9" * 64
        )
        return rec

    def truncate(self, job_id, last_round, status="running"):
        """Turn a run into a killed or failed snapshot ending at ``last_round``."""
        rec = self.results[job_id]
        rec["records"] = rec["records"][: last_round + 1]
        rec["status"] = status
        rec["snapshot"]["final"] = status != "running"

    # -- writing ----------------------------------------------------------

    def _job(self, stage, job_id):
        out = f"{PRIVATE}/runs/{job_id}"
        argv = [f"{PRIVATE}/env/bin/python", "-m", sp.PRODUCER_MODULE]
        common = [
            "--preparation-dir",
            f"{PRIVATE}/prep/CartPole-v1",
            "--output-dir",
            out,
            "--deadline",
            DEADLINE,
        ]
        fields = {
            "status": "complete",
            "protocol": sp.PROTOCOL,
            "expert_sha256": "2" * 64,
            "package_versions": PACKAGES,
        }
        if stage == "audit":
            argv += ["audit", *common, "--job-limit-seconds", "570"]
            fields.update(
                schema=sp.AUDIT_SCHEMA,
                audit_status="conflict_found",
                config=self.results["audit"]["config"],
            )
        else:
            m, r = sp.split(job_id)
            argv += [
                "run",
                *common,
                "--job-limit-seconds",
                "6600",
                "--data-dir",
                f"{PRIVATE}/data",
                "--seed",
                "300",
                "--method",
                m,
                "--restriction",
                r,
                "--expected-data-sha256",
                self.pin,
            ]
            fields.update(schema=sp.RUN_SCHEMA, seed=300, method=m, restriction_id=r)
        return {
            "id": job_id,
            "kind": "learner",
            "argv": argv,
            "cwd": f"{PRIVATE}/snapshots/796e8a4",
            "expected_result": f"{out}/result.json",
            "expected_fields": fields,
            "timeout_seconds": 7100,
        }

    def manifest(self, stage, job_ids=None):
        ids = sp.STAGE_JOBS[stage] if job_ids is None else job_ids
        return {
            "version": 1,
            "campaign_id": f"synthetic-{stage}",
            "source_root": f"{PRIVATE}/snapshots/796e8a4/src",
            "source_sha256": CAMPAIGN,
            "stage_deadline": DEADLINE,
            "overall_deadline": "2026-10-07T01:01:00Z",
            "jobs": [self._job(stage, j) for j in ids],
        }

    def _row(self, job_id, result_sha):
        state = self.states[job_id]
        if job_id in self.never_launched:
            return {
                "kind": "learner",
                "state": state,
                "state_reason": "stage_deadline",
                "result_sha256": None,
                "attempts": [],
                "expected_result": f"{PRIVATE}/runs/{job_id}/result.json",
            }
        stage = "audit" if job_id == "audit" else "x"
        job = self._job(stage, job_id)
        att = {
            "attempt_id": f"{job_id}-a1-0",
            "number": 1,
            "infrastructure_retry": False,
            "state": state,
            "started_at": "2026-10-01T04:40:00.000000Z",
            "ended_at": "2026-10-01T05:40:00.000000Z",
            "elapsed_seconds": ELAPSED[job_id],
            "argv": job["argv"],
            "cwd": job["cwd"],
            "exit_code": 0,
            "failure_kind": None,
            "termination": None,
            "archived_prior_result": None,
            "validation": None,
        }
        row = {
            "kind": "learner",
            "state": state,
            "state_reason": None,
            "result_sha256": None,
            "attempts": [att],
            "expected_result": job["expected_result"],
        }
        if state == "complete":
            att["validation"] = {"ok": True, "sha256": result_sha}
            row["result_sha256"] = result_sha
        elif state == "timed_out":
            att.update(exit_code=-9, termination={"stop_reason": "job_timeout"})
            row["state_reason"] = "job_timeout"
        elif job_id in self.resumed_failed:
            att.update(state="complete", validation={"ok": True, "sha256": result_sha})
            row.update(
                state_reason="completed result changed or invalid at resume",
                result_sha256=result_sha,
                resume_check={"ok": False, "error": "result changed"},
            )
        elif job_id in self.failure_kind:
            kind = self.failure_kind[job_id]
            att.update(failure_kind=kind, validation={"ok": False, "error": kind})
            row["state_reason"] = kind
        elif state == "failed":
            att.update(exit_code=1, failure_kind="exit_nonzero")
            row["state_reason"] = "exit_nonzero"
        return row

    def write(self, manifests=None):
        """Write every input; returns the CLI argv (output dir not created)."""
        manifests = manifests or {s: self.manifest(s) for s in sp.STAGE_JOBS}
        args, file_sha = [], {}
        for stage, man in manifests.items():
            root = self.inputs / stage
            root.mkdir(exist_ok=True)
            shas = {}
            for job in man["jobs"]:
                job_id = job["id"]
                rec = self.results[job_id]
                if job_id != "audit":
                    _account(rec, sp.split(job_id)[0])
                    if sp.split(job_id)[0] != "ftl":
                        kind = sp.SHARED_DATASET[sp.split(job_id)[0]]
                        rec["accounting"]["shared_acquisition"] = {
                            "dataset": kind,
                            "charged_to_this_run": False,
                            "pairs_sha256": self.data["datasets"][kind]["pairs_sha256"],
                        }
                shas[job_id] = None
                if job_id not in self.missing and job_id not in self.never_launched:
                    (root / job_id).mkdir(exist_ok=True)
                    raw = _dump(rec)
                    (root / job_id / "result.json").write_bytes(raw)
                    shas[job_id] = _sha(raw)
            raw_m = _dump(man)
            (root / "manifest.json").write_bytes(raw_m)
            file_sha[stage] = _sha(raw_m)
            inv = {
                "version": 1,
                "campaign_id": man["campaign_id"],
                "manifest_sha256": sp.config_digest(man),
                "source_root": man["source_root"],
                "source_sha256": CAMPAIGN,
                "controller_runs": [
                    {
                        "started_at": "2026-10-01T04:39:00Z",
                        "ended_at": "2026-10-01T06:00:00Z",
                        "exit_code": 0,
                        "retry_requests": [],
                    }
                ],
                "jobs": {
                    j["id"]: self._row(j["id"], shas[j["id"]]) for j in man["jobs"]
                },
            }
            (root / "inventory.json").write_bytes(_dump(inv))
            opt = sp.STAGE_OPTION[stage]
            args += [
                f"--{opt}-manifest",
                str(root / "manifest.json"),
                f"--{opt}-inventory",
                str(root / "inventory.json"),
                f"--{opt}-root",
                str(root),
            ]
        acq = {
            "version": 1,
            "campaign_id": "historical-data",
            "source_sha256": "d" * 64,
            "jobs": {
                "data": {
                    "state": "complete",
                    "result_sha256": self.pin,
                    "attempts": [
                        {
                            "exit_code": 0,
                            "elapsed_seconds": ACQ_ELAPSED,
                            "validation": {"ok": True, "sha256": self.pin},
                        }
                    ],
                }
            },
        }
        (self.inputs / "acquisition-inventory.json").write_bytes(_dump(acq))
        evidence = {
            "schema": sp.EVIDENCE_SCHEMA,
            **dataclasses.asdict(self.expected),
            "manifest_file_sha256": file_sha,
            "source_snapshot_verified": True,
            "verification_note": self.evidence_note,
        }
        (self.inputs / "evidence.json").write_bytes(_dump(evidence))
        return args + [
            "--trusted-evidence",
            str(self.inputs / "evidence.json"),
            "--data-dir",
            str(self.data_dir),
            "--acquisition-inventory",
            str(self.inputs / "acquisition-inventory.json"),
            "--preparation-record",
            str(self.inputs / "preparation.json"),
            "--output-dir",
            str(self.root / "out"),
        ]

    def run(self, argv):
        return sp.main(argv, expected=self.expected)

    def analyze(self, argv):
        return sp.analyze(sp.build_parser().parse_args(argv), self.expected)

    def refuses(self, argv, match):
        with pytest.raises(sp.PilotInputError, match=match):
            self.run(argv)
        assert not (self.root / "out").exists()


@pytest.fixture
def pilot(tmp_path):
    return Pilot(tmp_path)


def _tree_hashes(root):
    return {
        p: _sha(p.read_bytes())
        for p in sorted(pathlib.Path(root).rglob("*"))
        if p.is_file()
    }


# --------------------------------------------------------------- valid pilot


def test_valid_pilot_writes_new_outputs_and_preserves_inputs(pilot):
    argv = pilot.write()
    before = _tree_hashes(pilot.inputs)
    assert pilot.run(argv) == 0
    assert _tree_hashes(pilot.inputs) == before
    out = pilot.root / "out"
    names = {p.name for p in out.iterdir()}
    assert names == {sp.REPORT, sp.SUMMARY, sp.POINTS, sp.FIG_CURVES, sp.FIG_COSTS}
    historical = {
        sp._hist.REPORT,
        sp._hist.SUMMARY,
        sp._hist.POINTS,
        sp._hist.FIG_CURVES,
        sp._hist.FIG_COSTS,
    }
    assert not names & historical
    report = (out / sp.REPORT).read_text()
    for text in (report, (out / sp.SUMMARY).read_text(), (out / sp.POINTS).read_text()):
        assert chr(0x2014) not in text and PRIVATE not in text
        assert "/home/" not in text and str(pilot.root) not in text
    assert f"]({sp.FIG_CURVES})" in report and f"]({sp.FIG_COSTS})" in report
    assert "Largest label count evaluated by all four online curves: 1,000" in report
    assert "This comparison is descriptive only" in report
    assert "unfinished run" not in report
    assert "does not certify any additional conflict" in report
    assert "Recorded witness scope: w" in report
    limitations = report.split("## Limitations", 1)[1]
    assert "whether any learning curve has reached a plateau, remain unresolved" in (
        " ".join(limitations.split())
    )
    assert "inner epoch cap does not establish convergence" in limitations


def test_metrics_are_preserved_exactly(pilot):
    result = pilot.analyze(pilot.write())
    points = result["points"]
    for job_id in sp.ONLINE_IDS:
        stored = [
            r
            for r in pilot.results[job_id]["records"]
            if r.get("d_eval_size") is not None
        ]
        got = points["online"][job_id]["points"]
        assert [p["labels"] for p in got] == sp.ONLINE_SCHEDULE
        assert len(got) == 102
        for p, r in zip(got, stored):
            for key in ("normalized_return",) + sp.METRICS:
                assert p[key] == r[key]
            assert p["episode_returns"] == r["episode_returns"]
        assert len(points["online"][job_id]["fits"]) == 1000
    for job_id in sp.BC_IDS:
        ref = points["offline_bc_reference"][job_id]["point"]
        r = pilot.results[job_id]["records"][0]
        assert ref["labels"] == 1000 and ref["episode_returns"] == r["episode_returns"]
        assert ref["normalized_return"] == r["normalized_return"]
        assert ref["fit"]["train_batch_size"] == 32
        assert ref["fit"]["train_examples"] == 900
    opt = result["summary"]["optimizer"]
    assert opt["run-ftl-identity"]["fits"] == 1000
    assert opt["run-ftl-identity"]["fixed_budget_fallback_fits"] == 319
    assert opt["run-ftl-identity"]["early_stopped_rounds"] == [333]
    assert opt["run-ftl-identity"]["epochs"] == 999 * 20 + 6
    assert opt["run-bc-identity"]["fits"] == 1


def test_curve_figure_style_contract(pilot):
    result = pilot.analyze(pilot.write())
    s = result["summary"]
    states = {j: "complete" for j in sp.RUN_IDS}
    fig = sp.curves_figure(
        s["curves"], s["offline_bc_reference"], states, s["baselines"]
    )
    for ax in fig.axes[:4]:
        lines = {ln.get_label(): ln for ln in ax.get_lines()}
        for job_id in sp.ONLINE_IDS + sp.BC_IDS:
            m, r = sp.split(job_id)
            ln = lines[job_id]
            assert mcolors.to_hex(ln.get_color()) == sp.COLOR[m]
            assert ln.get_linestyle() == sp.STYLE[r]
        for job_id in sp.BC_IDS:
            xs, ys = lines[job_id].get_data()
            assert list(xs) == [0, 1] and ys[0] == ys[1]  # axhline: whole axis
        assert len(lines["run-ftl-identity"].get_xdata()) == 102
    # Shared fixed view on the normalized panel; stored values unchanged.
    ret_ax = fig.axes[0]
    assert sp.NORMALIZED_RETURN_YLIM == (-0.5, 1.5)
    assert ret_ax.get_yscale() == "linear"
    assert ret_ax.get_ylim() == sp.NORMALIZED_RETURN_YLIM
    ftl = {ln.get_label(): ln for ln in ret_ax.get_lines()}["run-ftl-identity"]
    stored = [p["normalized_return"] for p in s["curves"]["run-ftl-identity"]]
    assert list(ftl.get_ydata()) == stored
    assert sp.STYLE == {"identity": "--", MASK: "-"}


def test_new_costs_exclude_the_reused_acquisition(pilot):
    s = pilot.analyze(pilot.write())["summary"]
    new, acq = s["costs"]["new_execution"], s["costs"]["reused_acquisition"]
    assert new["controller_elapsed_seconds"]["sum"] == sum(ELAPSED.values())
    assert new["controller_elapsed_seconds"]["label"].startswith("exact")
    assert acq["controller_elapsed_seconds"] == ACQ_ELAPSED
    ftl_steps = sum(
        s["outcomes"][j]["in_run_collection"]["env_steps"]
        for j in sp.ONLINE_IDS
        if j.startswith("run-ftl")
    )
    assert new["ftl_in_run_collection_env_steps"]["sum"] == ftl_steps == 2 * 30000
    assert acq["bc_iid_stream"]["env_steps"] == sum(STREAM_LENGTHS)
    assert s["outcomes"]["run-bc_iid-identity"]["in_run_collection"]["env_steps"] == 0


# ------------------------------------------------------- provenance contracts


def test_pinned_data_is_the_only_mixed_protocol_input(pilot):
    pilot.results["run-ftl-identity"]["protocol"] = sp.DATA_PROTOCOL
    pilot.refuses(pilot.write(), "protocol")


def test_unpinned_data_bytes_are_refused(pilot):
    argv = pilot.write()
    path = pilot.data_dir / "result.json"
    path.write_bytes(path.read_bytes() + b" ")
    pilot.refuses(argv, "approved pinned acquisition")


def test_other_execution_source_is_refused(pilot):
    files = dict(RUN_FILES, extra="0" * 64)
    for job_id in pilot.results:
        pilot.results[job_id]["source"] = {
            "files": files,
            "combined_sha256": sp.config_digest(files),
        }
    pilot.refuses(pilot.write(), "expected scientific execution source")


def test_trusted_evidence_must_match_public_identity(pilot):
    argv = pilot.write()
    pilot.expected = dataclasses.replace(
        pilot.expected, campaign_source_sha256="0" * 64
    )
    pilot.refuses(argv, "campaign_source_sha256")


@pytest.mark.parametrize("case", ["duplicate", "missing", "archived_six"])
def test_stage_job_sets(pilot, case):
    manifests = {s: pilot.manifest(s) for s in sp.STAGE_JOBS}
    if case == "duplicate":
        manifests["online"]["jobs"].append(
            copy.deepcopy(manifests["offline_bc"]["jobs"][0])
        )
    elif case == "missing":
        manifests["online"]["jobs"].pop()
    else:
        manifests["online"] = pilot.manifest("online", sp.RUN_IDS)
    pilot.refuses(pilot.write(manifests), "job set")


def test_complete_result_must_match_controller_hash(pilot):
    argv = pilot.write()
    path = pilot.inputs / "online" / "run-ftl-identity" / "result.json"
    path.write_bytes(path.read_bytes() + b"\n")
    pilot.refuses(argv, "controller's hash")


def test_config_digest_is_recomputed(pilot):
    pilot.results["run-bc_iid-identity"]["config"]["eval_episodes"] = 99
    pilot.refuses(pilot.write(), "config_sha256")


# ------------------------------------------------------------- offline BC


@pytest.mark.parametrize(
    "case,match",
    [
        ("prefix_fits", "one fit and one evaluation"),
        ("prefix_digest", "prefix digest"),
        ("other_pairs", "whole fixed pool"),
        ("not_round0", "flat reference"),
    ],
)
def test_offline_bc_is_one_full_pool_fit(pilot, case, match):
    rec = pilot.results[f"run-bc-{MASK}"]
    r = rec["records"][0]
    if case == "prefix_fits":
        rec["records"].insert(0, dict(r, n_observations=10))
    elif case == "prefix_digest":
        r["prefix_pairs_sha256"] = r["trained_pairs_sha256"]
    elif case == "other_pairs":
        r["trained_pairs_sha256"] = rec["trained_pairs_sha256"] = "8" * 64
    else:
        r["round"] = 1000
    pilot.refuses(pilot.write(), match)


# -------------------------------------------------------- online schedules


@pytest.mark.parametrize(
    "case,match",
    [
        ("drop_round", "every round"),
        ("extra_eval", "schedule"),
        ("missing_eval", "schedule"),
        ("null_metric", "non-finite"),
        ("value_off_schedule", "carries evaluation values"),
        ("batch", "trained loader"),
        ("fallback", "fixed-budget fallback"),
        ("early_end", "every round to 1000"),
    ],
)
def test_online_round_contract(pilot, case, match):
    rec = pilot.results["run-ftl-identity"]
    records = rec["records"]
    if case == "drop_round":
        del records[5]
    elif case == "extra_eval":
        records[5].update(_evaluation(0, 5))
    elif case == "missing_eval":
        for k in (
            "d_eval_size",
            "normalized_return",
            "rollout_cross_entropy",
            "expert_rollout_cross_entropy",
            "disagreement_rate",
            "episode_returns",
            "checkpoint",
        ):
            records[10][k] = None
    elif case == "null_metric":
        records[20]["disagreement_rate"] = None
    elif case == "value_off_schedule":
        records[7]["disagreement_rate"] = 0.1
    elif case == "batch":
        records[5]["train_batch_size"] = 1
    elif case == "fallback":
        records[100].update(_fit(1000))
    else:
        del records[641:]
    pilot.refuses(pilot.write(), match)


# ------------------------------------------------------ incomplete outcomes


def test_timed_out_run_stays_visible_and_sets_matched_budget(pilot):
    job = f"run-ftl-{MASK}"
    pilot.truncate(job, 645)
    pilot.states[job] = "timed_out"
    result = pilot.analyze(pilot.write())
    s = result["summary"]
    assert s["matched_budget"]["labels"] == 640
    row = s["outcomes"][job]
    assert (
        row["controller_state"] == "timed_out" and row["stop_reason"] == "job_timeout"
    )
    assert row["last_recorded_round"] == 645 and row["last_evaluated_labels"] == 640
    assert row["counts_are"].startswith("saved counts")
    assert s["optimizer"][job]["fits"] == 645
    new = s["costs"]["new_execution"]
    assert new["controller_elapsed_seconds"]["label"].startswith("exact")
    assert new["run_observed_env_steps"]["includes_lower_bounds"] is True
    assert not new["run_observed_env_steps"]["label"].startswith("exact")
    report = result["texts"][sp.REPORT]
    assert "timed_out" in report and "1,000" in report
    assert "not a total" in report
    assert "saved evaluations of an unfinished run (FTL, cart position" in report


def test_failed_and_missing_records_are_accounted_not_imputed(pilot):
    bc = f"run-bc-{MASK}"
    pilot.truncate(bc, -1, status="failed")
    pilot.states[bc] = "failed"
    pilot.states["run-bc_iid-identity"] = "failed"
    pilot.missing.add("run-bc_iid-identity")
    pilot.states["run-ftl-identity"] = "cancelled"
    pilot.never_launched.add("run-ftl-identity")
    result = pilot.analyze(pilot.write())
    s = result["summary"]
    assert s["offline_bc_reference"][bc] is None
    assert s["offline_bc_reference"]["run-bc-identity"] is not None
    assert s["matched_budget"]["labels"] is None
    assert s["outcomes"]["run-bc_iid-identity"]["result"] == "missing"
    assert s["outcomes"]["run-ftl-identity"]["attempts"] == 0
    new = s["costs"]["new_execution"]
    assert new["not_launched"] == ["run-ftl-identity"]
    assert new["controller_elapsed_seconds"]["sum"] == (
        sum(ELAPSED.values()) - ELAPSED["run-ftl-identity"]
    )
    assert new["run_observed_env_steps"]["unknown"] == 1
    assert new["run_observed_env_steps"]["label"].startswith("known subtotal")
    report = result["texts"][sp.REPORT]
    assert "Unavailable" in report and "no saved evaluation, no reference" in report


def _negative_observed(acc):
    acc["observed"]["env_steps"] = acc["totals"]["env_steps"] = -5000


def _phantom_evaluation_steps(acc):
    acc["completed_records"]["evaluation"]["env_steps"] = 123456


@pytest.mark.parametrize(
    "edit,match",
    [
        (_negative_observed, "observed steps do not equal"),
        (_phantom_evaluation_steps, "evaluation counters differ"),
    ],
)
def test_empty_offline_bc_accounting_is_recomputed(pilot, edit, match):
    bc = f"run-bc-{MASK}"
    pilot.truncate(bc, -1, status="failed")
    pilot.states[bc] = "failed"
    argv = pilot.write()
    path = pilot.inputs / "offline_bc" / bc / "result.json"
    rec = json.loads(path.read_text())
    edit(rec["accounting"])
    path.write_bytes(_dump(rec))  # a failed job has no controller hash to keep
    pilot.refuses(argv, match)


def test_complete_job_without_result_is_refused(pilot):
    argv = pilot.write()
    (pilot.inputs / "offline_bc" / "run-bc-identity" / "result.json").unlink()
    pilot.refuses(argv, "complete job without its result")


# ------------------------------------------------------------ output safety


def test_private_text_is_never_written(pilot):
    pilot.evidence_note = f"checked {PRIVATE}/snapshots/796e8a4"
    pilot.refuses(pilot.write(), "private or banned text")


def test_existing_output_dir_is_refused(pilot):
    argv = pilot.write()
    (pilot.root / "out").mkdir()
    with pytest.raises(sp.PilotInputError, match="already exists"):
        pilot.run(argv)


def test_relative_output_dir_is_not_a_private_token(pilot, monkeypatch):
    argv = pilot.write()
    argv[argv.index("--output-dir") + 1] = "out"
    monkeypatch.chdir(pilot.root)
    assert pilot.run(argv) == 0


# ------------------------------------------------- accounting conservation


def _rewrite_complete(pilot, stage, job_id, edit):
    """Edit a complete result after writing and keep the controller hash valid."""
    path = pilot.inputs / stage / job_id / "result.json"
    rec = json.loads(path.read_text())
    edit(rec)
    raw = _dump(rec)
    path.write_bytes(raw)
    inv_path = pilot.inputs / stage / "inventory.json"
    inv = json.loads(inv_path.read_text())
    inv["jobs"][job_id]["result_sha256"] = _sha(raw)
    inv["jobs"][job_id]["attempts"][0]["validation"]["sha256"] = _sha(raw)
    inv_path.write_bytes(_dump(inv))


def _unconserved(rec):
    acc = rec["accounting"]
    acc["observed"]["env_steps"] = acc["totals"]["env_steps"] = 0
    acc["in_flight"]["env_steps"] = 99


def _queries_off(rec):
    rec["accounting"]["totals"]["expert_predict_calls"] += 1


def _stream_as_run_work(rec):
    steps = sum(STREAM_LENGTHS)
    rec["accounting"]["completed_records"]["collection"]["env_steps"] = steps
    rec["accounting"]["observed"]["env_steps"] += steps
    rec["accounting"]["totals"]["env_steps"] += steps


@pytest.mark.parametrize(
    "job_id,edit,match",
    [
        ("run-ftl-identity", _unconserved, "observed steps do not equal"),
        ("run-ftl-identity", _queries_off, "observed steps do not equal"),
        ("run-bc_iid-identity", _stream_as_run_work, "collection or label"),
    ],
)
def test_exact_totals_are_recomputed_not_trusted(pilot, job_id, edit, match):
    argv = pilot.write()
    _rewrite_complete(pilot, "online", job_id, edit)
    pilot.refuses(argv, match)


# ------------------------------------------------- outcome preservation


def test_all_learners_missing_keeps_every_outcome(pilot):
    for job_id in sp.RUN_IDS:
        pilot.states[job_id] = "failed"
        pilot.missing.add(job_id)
    result = pilot.analyze(pilot.write())
    s = result["summary"]
    assert set(s["outcomes"]) == {"audit", *sp.RUN_IDS}
    assert all(s["outcomes"][j]["result"] == "missing" for j in sp.RUN_IDS)
    assert s["costs"]["new_execution"]["controller_elapsed_seconds"]["sum"] == sum(
        ELAPSED.values()
    )
    assert s["matched_budget"]["labels"] is None and s["curves"] == {}
    assert all(v is None for v in s["offline_bc_reference"].values())
    assert "Unavailable" in result["texts"][sp.REPORT]


@pytest.mark.parametrize("case", ["validation", "source_changed", "late", "resume"])
def test_unaccepted_result_is_inventory_only(pilot, case):
    job = "run-bc_iid-identity"
    if case == "late":
        pilot.states[job] = "timed_out"  # killed after writing a complete record
    else:
        pilot.states[job] = "failed"
        if case == "resume":
            pilot.resumed_failed.add(job)
        else:
            pilot.failure_kind[job] = case
    result = pilot.analyze(pilot.write())
    s = result["summary"]
    row = s["outcomes"][job]
    assert row["result"].startswith("rejected: ")
    assert row["controller_state"] == pilot.states[job]
    assert row["controller_elapsed_seconds"] == ELAPSED[job]
    assert row["counts_are"].startswith("unknown") and "training" not in row
    assert job not in s["curves"] and s["optimizer"][job] is None
    assert s["matched_budget"]["labels"] is None
    assert "rejected" in result["texts"][sp.REPORT]


def test_unaccepted_result_does_not_hide_corruption(pilot):
    job = "run-bc_iid-identity"
    pilot.states[job] = "failed"
    pilot.truncate(job, 300, status="partial")
    pilot.results[job]["source"]["combined_sha256"] = "0" * 64
    pilot.refuses(pilot.write(), "source differs|scientific execution source")


# ------------------------------------------------------ reused helper bytes


def _copy_adapter(tmp_path, helper_bytes):
    root = tmp_path / "copy" / "experiments" / "agnostic"
    root.mkdir(parents=True)
    (root / ANALYZER.name).write_bytes(ANALYZER.read_bytes())
    (root / sp._HIST_PATH.name).write_bytes(helper_bytes)
    return root / ANALYZER.name


def _load(path):
    spec = importlib.util.spec_from_file_location("stronger_copy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_helper_bytes_are_checked_before_they_run(tmp_path):
    sentinel = tmp_path / "helper-ran"
    edited = (
        f"open({str(sentinel)!r}, 'w').close()\n".encode() + sp._HIST_PATH.read_bytes()
    )
    with pytest.raises(ImportError, match="reviewed helper bytes"):
        _load(_copy_adapter(tmp_path, edited))
    assert not sentinel.exists()


def test_loaded_helper_ignores_later_file_changes(tmp_path, monkeypatch):
    path = _copy_adapter(tmp_path, sp._HIST_PATH.read_bytes())
    mod = _load(path)
    path.with_name(sp._HIST_PATH.name).write_bytes(b"raise SystemExit(3)\n")
    monkeypatch.setitem(globals(), "sp", mod)
    pilot = Pilot(tmp_path / "pilot")
    s = pilot.analyze(pilot.write())["summary"]
    assert s["provenance"]["reused_helper_script_sha256"] == sp.HIST_SHA256
    assert s["provenance"]["reused_helper_script_sha256"] == _sha(
        ANALYZER.with_name(sp._HIST_PATH.name).read_bytes()
    )
