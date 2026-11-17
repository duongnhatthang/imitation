"""Report adapter for the stronger-mask CartPole pilot (producer protocol /2).

Analysis only. Reads explicit local copies of the three executed stage manifests
and their controller inventories (audit, the two-job offline BC stage and the
four-job online stage), the audit and six run ``result.json`` files, the reused
protocol /1 data acquisition, the expert preparation record and a trusted
execution evidence file. It never launches jobs, rolls out environments or loads
checkpoints. Every check runs before anything is written, and outputs go into a
new directory only:

    python experiments/agnostic/analyze_cartpole_stronger_pilot.py \
        --trusted-evidence EVIDENCE.json \
        --audit-manifest M --audit-inventory I --audit-root DIR \
        --offline-bc-manifest M --offline-bc-inventory I --offline-bc-root DIR \
        --online-manifest M --online-inventory I --online-root DIR \
        --data-dir DATA --acquisition-inventory I \
        --preparation-record preparation.json --output-dir NEW_DIR

Each stage root holds ``<job id>/result.json``. The data directory holds the
pinned acquisition ``result.json`` and its two datasets.

This is not the historical analyzer: ``analyze_cartpole_pilot.py`` reads protocol
/1 runs (its own analysis schema suffix /2 is unrelated to producer protocols) and
is neither run nor changed here. Only its stateless helpers are reused.

Inputs are bound three ways. Public protocol constants (`PUBLIC_EXPECTED`) name
the execution source and the approved data pin. The trusted evidence file, written
by the operator who verified the deployed snapshot, must repeat them and gives
the exact bytes of the three executed manifests; results alone are never trusted
to identify their own source. Inventories, results and the acquisition are then
checked against the manifests and each other. The archived six-job manifest was
never executed and is refused by the per-stage job sets.

Offline BC is one fit on the whole 1,000-label pool and one evaluation per
observation condition, drawn as a flat reference. FTL and BC-iid record every
round 0..1000 and evaluate rounds 0, 1, 10, ..., 1000; only those four curves set
the matched label budget. Failed, partial, cancelled and missing outcomes stay
visible; unknown quantities stay null.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import pathlib
import re
import sys
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from imitation.experiments.ftrl.plot_results import (  # noqa: E402, F401
    NORMALIZED_RETURN_YLIM,
    set_normalized_return_axis,
)

_HIST_PATH = pathlib.Path(__file__).with_name("analyze_cartpole_pilot.py")
# SHA-256 of the reviewed historical analyzer. Re-pinned from ef7c7dca... after
# the shared normalized-return view change, which touched only its import block
# and curves_figure, not the helpers used here. Its bytes are checked before
# anything in them runs, and the module is built from exactly those bytes, so
# later edits to the file cannot change this analysis.
HIST_SHA256 = "4aaa48afed1aefe2c1c8353ae1419c9fbab85b147a653550dd3b76dd5e433a2a"
_hist_bytes = _HIST_PATH.read_bytes()
if hashlib.sha256(_hist_bytes).hexdigest() != HIST_SHA256:
    raise ImportError(
        f"{_HIST_PATH.name} differs from the reviewed helper bytes; refusing to load"
    )
_hist = types.ModuleType("_cartpole_pilot_helpers")
_hist.__file__ = str(_HIST_PATH)
exec(compile(_hist_bytes, _HIST_PATH.name, "exec"), _hist.__dict__)
del _hist_bytes

# Stateless helpers of the historical analyzer whose contracts fit unchanged.
PilotInputError = _hist.PilotInputError
require = _hist.require
config_digest = _hist.config_digest  # producer canonical JSON digest
evaluation_point = _hist.evaluation_point  # checks normalized_return, keeps stored
png = _hist.png
num = _hist.num
METRICS = _hist.METRICS
METHOD_LABEL = _hist.METHOD_LABEL
COLOR = _hist.COLOR

# ------------------------------------------------------------ protocol constants

PROTOCOL = "agnostic-cartpole-restriction-pilot/2"
RUN_SCHEMA = PROTOCOL + "/run"
AUDIT_SCHEMA = PROTOCOL + "/audit"
# The reused acquisition keeps the unchanged /1 data producer.
DATA_PROTOCOL = "agnostic-cartpole-restriction-pilot/1"
DATA_SCHEMA = DATA_PROTOCOL + "/data"
PRODUCER_MODULE = "imitation.experiments.agnostic.restriction_pilot"
ENV_NAME = "CartPole-v1"
CAP = 500
SEED = 300
AUDIT_SEED = 301
MASK = "cart_position_angular_velocity_zero"
RESTRICTIONS = ("identity", MASK)
ZEROED = {"identity": [], MASK: [0, 3]}
OBS_LABEL = {
    "identity": "full observation",
    MASK: "cart position and pole angular velocity zeroed",
}
STYLE = {"identity": "--", MASK: "-"}
METHODS = ("ftl", "bc", "bc_iid")
ONLINE = ("ftl", "bc_iid")
BC_IDS = tuple(f"run-bc-{r}" for r in RESTRICTIONS)
ONLINE_IDS = tuple(f"run-{m}-{r}" for m in ONLINE for r in RESTRICTIONS)
RUN_IDS = BC_IDS + ONLINE_IDS
STAGE_JOBS = {"audit": ("audit",), "offline_bc": BC_IDS, "online": ONLINE_IDS}
STAGE_OPTION = {"audit": "audit", "offline_bc": "offline-bc", "online": "online"}
BETA = {"ftl": 0.0, "bc_iid": 1.0, "bc": None}
SHARED_DATASET = {"ftl": None, "bc": "pool", "bc_iid": "stream"}
DATA_FILES = {"pool": "fixed_bc_pool.npz", "stream": "bc_iid_stream.npz"}
N_LABELS = 1000
EVAL_EPISODES = 100
BASELINE_EPISODES = 500
# Round 0, round 1, every 10th round and the last (restriction_pilot.eval_budgets).
ONLINE_SCHEDULE = [0, 1] + list(range(10, N_LABELS + 1, 10))
# Experiment settings shared by all six runs (restriction_pilot._experiment_config
# and the ExperimentConfig defaults it inherits at the execution snapshot).
EXPERIMENT_PINS = {
    "env_name": ENV_NAME,
    "seed": SEED,
    "policy_mode": "linear",
    "n_rounds": N_LABELS,
    "samples_per_round": 1,
    "bc_n_epochs": 20,
    "bc_batch_size": 32,
    "eval_interval": 10,
    "learning_rate": 1e-3,
    "grow_batch_with_data": True,
    "inner_early_stop": True,
    "inner_early_stop_val_frac": 0.1,
    "inner_early_stop_min_val_size": 32,
    "outer_early_stop": False,
}
METHOD_KEYS = {"algo", "subsample_strategy"}
# Checks the producer's _verify_data ran before accepting the reused data.
ACCEPTANCE_CHECKS = sorted(
    [
        "schema",
        "protocol",
        "status_complete",
        "env_name",
        "episode_cap",
        "seed",
        "config_sha256",
        "config_seed",
        "config_budget",
        "config_rules",
        "source_combined_sha256",
        "source_file_set",
        "package_versions",
        "expert_sha256",
        "preparation_identity",
        "baselines",
    ]
)
RUN_FLAGS = {
    "--preparation-dir",
    "--output-dir",
    "--deadline",
    "--job-limit-seconds",
    "--data-dir",
    "--seed",
    "--method",
    "--restriction",
    "--expected-data-sha256",
}
AUDIT_FLAGS = {"--preparation-dir", "--output-dir", "--deadline", "--job-limit-seconds"}
MANIFEST_KEYS = {
    "version",
    "campaign_id",
    "source_root",
    "source_sha256",
    "stage_deadline",
    "overall_deadline",
    "jobs",
}
TERMINAL_STATES = {"complete", "failed", "timed_out", "cancelled", "interrupted"}
STOP_REASONS = {"job_timeout", "stage_deadline", "overall_deadline"}
RESULT_FILE = "result.json"
EVIDENCE_SCHEMA = "cartpole-stronger-pilot-execution-evidence/1"
EVIDENCE_KEYS = {
    "schema",
    "execution_source_commit",
    "campaign_source_sha256",
    "scientific_source_combined_sha256",
    "data_result_sha256",
    "manifest_file_sha256",
    "source_snapshot_verified",
    "verification_note",
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

FIG_CURVES = "cartpole_stronger_pilot_learning_curves.png"
FIG_COSTS = "cartpole_stronger_pilot_costs.png"
REPORT = "CARTPOLE_STRONGER_PILOT_RESULTS.md"
SUMMARY = "cartpole_stronger_pilot_summary.json"
POINTS = "cartpole_stronger_pilot_curve_points.json"
SUMMARY_SCHEMA = "cartpole-stronger-pilot-analysis/1"
POINTS_SCHEMA = "cartpole-stronger-pilot-curve-points/1"


@dataclasses.dataclass(frozen=True)
class Expected:
    """Execution identities every input must agree with."""

    # Frozen source commit of the stronger-mask pilot.
    execution_source_commit: str = "796e8a405b0ca868925a9e77ca854c09ad376617"
    # campaign.hash_source of the deployed snapshot's src tree; manifests and
    # inventories carry it.
    campaign_source_sha256: str = (
        "1433110845993f2b17fde6fb4b3a83dd8cd5394b347314f141d566eda897b8fd"
    )
    # restriction_pilot.source_identity()["combined_sha256"] of that snapshot;
    # audit and run records carry it.
    scientific_source_combined_sha256: str = (
        "c680c12139b1d83c301d3367813acc65252b5ef6f7666308db61f033cea35bb7"
    )
    # SHA-256 of the approved reused /1 data result.json bytes, as pinned by
    # every run's --expected-data-sha256.
    data_result_sha256: str = (
        "2588f231c5d98b9c22f321231dc6f61d36326c1843330aa5a552fa502c4abc8c"
    )


PUBLIC_EXPECTED = Expected()


# ------------------------------------------------------------------ small helpers


def _reject_constant(name):
    raise ValueError(f"non-finite JSON constant {name}")


class Inputs:
    """Reads input files once, remembering each file's SHA-256 under a logical name.

    The files are re-hashed before anything is written so that inputs changing
    during the analysis are refused rather than silently mixed.
    """

    def __init__(self):
        self.files = {}

    def read(self, name, path):
        path = pathlib.Path(path)
        require(path.is_file(), f"{name}: input file is missing")
        raw = path.read_bytes()
        self.files[name] = (path, hashlib.sha256(raw).hexdigest())
        return raw

    def json(self, name, path):
        raw = self.read(name, path)
        try:
            obj = json.loads(raw, parse_constant=_reject_constant)
        except ValueError as exc:
            raise PilotInputError(f"{name}: not valid JSON: {exc}")
        return obj, self.files[name][1]

    def recheck(self):
        for name, (path, sha) in self.files.items():
            require(
                path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha,
                f"{name}: input changed during the analysis",
            )

    def hashes(self):
        return {name: sha for name, (_, sha) in sorted(self.files.items())}


def _is_sha(v):
    return isinstance(v, str) and bool(_SHA_RE.match(v))


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _json_equal(a, b):
    """Exact JSON equality that does not conflate bool, int and float."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return a == b


def pairs_sha256(obs, acts):
    """restriction_pilot.pairs_sha256: float32 obs and int64 acts with shapes."""
    obs = np.ascontiguousarray(obs, dtype=np.float32)
    acts = np.ascontiguousarray(acts, dtype=np.int64)
    digest = hashlib.sha256()
    for name, array in (("obs", obs), ("acts", acts)):
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def pool_logical_cost(lengths, budget):
    """restriction_pilot.pool_logical_cost of the pool's episode lengths."""
    total = 0
    for i, length in enumerate(lengths):
        total += int(length)
        if total >= budget:
            return {
                "labels": budget,
                "episodes": i + 1,
                "env_steps": total,
                "overshoot_transitions": total - budget,
            }
    raise PilotInputError(f"pool episodes cover fewer than {budget} transitions")


def split(job_id):
    _, method, restriction = job_id.split("-", 2)
    return method, restriction


def label(job_id):
    m, r = split(job_id)
    return f"{METHOD_LABEL[m]}, {OBS_LABEL[r]}"


# -------------------------------------------------------- evidence and manifests


def check_evidence(ev, expected):
    """The operator's evidence must repeat the public identities exactly."""
    require(
        isinstance(ev, dict) and set(ev) == EVIDENCE_KEYS,
        f"trusted evidence keys must be exactly {sorted(EVIDENCE_KEYS)}",
    )
    require(ev["schema"] == EVIDENCE_SCHEMA, "trusted evidence: unexpected schema")
    for key in (
        "execution_source_commit",
        "campaign_source_sha256",
        "scientific_source_combined_sha256",
        "data_result_sha256",
    ):
        require(
            ev[key] == getattr(expected, key),
            f"trusted evidence {key} differs from the expected identity",
        )
    hashes = ev["manifest_file_sha256"]
    require(
        isinstance(hashes, dict)
        and set(hashes) == set(STAGE_JOBS)
        and all(_is_sha(h) for h in hashes.values())
        and len(set(hashes.values())) == len(hashes),
        "trusted evidence must give three distinct manifest SHA-256 values",
    )
    require(
        ev["source_snapshot_verified"] is True,
        "trusted evidence does not attest that the deployed snapshot was verified",
    )
    require(
        isinstance(ev["verification_note"], str) and ev["verification_note"],
        "trusted evidence needs a verification note",
    )
    return hashes


def parse_argv(job_id, argv):
    """Producer command and its ``--flag value`` pairs from a manifest argv."""
    require(
        isinstance(argv, list)
        and all(isinstance(a, str) for a in argv)
        and "-m" in argv,
        f"{job_id}: argv does not run a module",
    )
    i = argv.index("-m")
    require(
        argv[i + 1 : i + 2] == [PRODUCER_MODULE],
        f"{job_id}: argv does not run {PRODUCER_MODULE}",
    )
    command, rest = argv[i + 2], argv[i + 3 :]
    require(len(rest) % 2 == 0, f"{job_id}: argv flags are not flag/value pairs")
    flags = {}
    for flag, value in zip(rest[::2], rest[1::2]):
        require(
            flag.startswith("--") and flag not in flags,
            f"{job_id}: malformed or repeated argv flag {flag}",
        )
        flags[flag] = value
    return command, flags


def check_manifest(stage, m, expected):
    """One executed stage manifest: exact job set and producer arguments."""
    require(
        isinstance(m, dict) and set(m) == MANIFEST_KEYS and m["version"] == 1,
        f"{stage} manifest: unexpected format",
    )
    require(
        m["source_sha256"] == expected.campaign_source_sha256,
        f"{stage} manifest: campaign source hash differs from the expected one",
    )
    ids = [j.get("id") for j in m["jobs"]]
    require(len(ids) == len(set(ids)), f"{stage} manifest: duplicate job ids")
    require(
        sorted(ids) == sorted(STAGE_JOBS[stage]),
        f"{stage} manifest: job set {sorted(ids)} is not the authorized "
        f"{sorted(STAGE_JOBS[stage])}",
    )
    jobs = {}
    for job in m["jobs"]:
        job_id = job["id"]
        require(job.get("kind") == "learner", f"{job_id}: unexpected job kind")
        command, flags = parse_argv(job_id, job["argv"])
        want = ("audit", AUDIT_FLAGS) if stage == "audit" else ("run", RUN_FLAGS)
        require(
            command == want[0] and set(flags) == want[1],
            f"{job_id}: producer command or flags differ from the protocol",
        )
        out = flags["--output-dir"]
        require(
            out.rstrip("/").split("/")[-1] == job_id
            and job["expected_result"] == out.rstrip("/") + "/" + RESULT_FILE,
            f"{job_id}: expected result is not <output dir>/{job_id}/result.json",
        )
        require(
            flags["--deadline"] == m["stage_deadline"],
            f"{job_id}: producer deadline differs from the stage deadline",
        )
        if stage != "audit":
            method, restriction = split(job_id)
            require(
                flags["--method"] == method
                and flags["--restriction"] == restriction
                and flags["--seed"] == str(SEED),
                f"{job_id}: argv method, restriction or seed differ from job id",
            )
            require(
                flags["--expected-data-sha256"] == expected.data_result_sha256,
                f"{job_id}: argv does not pin the approved data result",
            )
        fields = job.get("expected_fields") or {}
        require(
            fields.get("status") == "complete" and fields.get("protocol") == PROTOCOL,
            f"{job_id}: manifest expected fields do not require protocol /2",
        )
        jobs[job_id] = dict(job, flags=flags)
    return jobs


def check_stage_union(manifests, jobs):
    """The three stages share one snapshot and clock and cover each job once."""
    all_ids = [j for s in STAGE_JOBS for j in jobs[s]]
    require(len(all_ids) == len(set(all_ids)), "a job appears in two stages")
    run_ids = [j for s in ("offline_bc", "online") for j in jobs[s]]
    require(
        sorted(run_ids) == sorted(RUN_IDS),
        "offline BC and online stages do not cover exactly the six runs",
    )
    for key in ("source_root", "source_sha256", "stage_deadline", "overall_deadline"):
        require(
            len({manifests[s][key] for s in manifests}) == 1,
            f"stage manifests differ in {key}",
        )
    require(
        len({manifests[s]["campaign_id"] for s in manifests}) == 3,
        "stage manifests reuse a campaign id",
    )
    flat = {j: job for s in jobs for j, job in jobs[s].items()}
    results = [job["expected_result"] for job in flat.values()]
    require(len(set(results)) == len(results), "two jobs share an expected result")
    require(len({job["cwd"] for job in flat.values()}) == 1, "jobs differ in cwd")
    require(
        len({job["argv"][0] for job in flat.values()}) == 1,
        "jobs use different interpreters",
    )
    require(
        len({job["flags"]["--preparation-dir"] for job in flat.values()}) == 1,
        "jobs use different preparation directories",
    )
    require(
        len({flat[j]["flags"]["--data-dir"] for j in RUN_IDS}) == 1,
        "runs use different data directories",
    )
    return flat


def check_inventory(stage, inv, manifest, canonical_sha, mjobs, expected):
    """A terminal controller inventory; returns one outcome per job."""
    require(
        isinstance(inv, dict) and inv.get("version") == 1,
        f"{stage} inventory: unexpected version",
    )
    require(
        inv.get("campaign_id") == manifest["campaign_id"]
        and inv.get("manifest_sha256") == canonical_sha
        and inv.get("source_root") == manifest["source_root"]
        and inv.get("source_sha256") == expected.campaign_source_sha256,
        f"{stage} inventory: does not belong to the executed manifest",
    )
    runs = inv.get("controller_runs")
    require(isinstance(runs, list) and runs, f"{stage} inventory: no controller run")
    for run in runs:
        require(not run.get("retry_requests"), f"{stage}: retry requests recorded")
    require(
        runs[-1].get("ended_at") is not None and runs[-1].get("exit_code") is not None,
        f"{stage} inventory: the controller has not finished; not terminal",
    )
    require(
        isinstance(inv.get("jobs"), dict) and set(inv["jobs"]) == set(mjobs),
        f"{stage} inventory: job set differs from its manifest",
    )
    outcomes = {}
    for job_id, mj in mjobs.items():
        row = inv["jobs"][job_id]
        state = row.get("state")
        require(state in TERMINAL_STATES, f"{job_id}: job is not terminal ({state})")
        require(
            row.get("expected_result") == mj["expected_result"],
            f"{job_id}: inventory expected result differs from manifest",
        )
        attempts = row.get("attempts")
        require(isinstance(attempts, list), f"{job_id}: malformed attempts")
        att, resume_failed = None, False
        if not attempts:
            require(state == "cancelled", f"{job_id}: {state} without an attempt")
        else:
            require(
                len(attempts) == 1,
                f"{job_id}: {len(attempts)} attempts; no retries were authorized",
            )
            att = attempts[0]
            require(
                att.get("number") == 1
                and att.get("infrastructure_retry") is False
                and att.get("archived_prior_result") is None,
                f"{job_id}: attempt is a retry or replaced a prior result",
            )
            # A resumed controller fails a complete row whose result changed or no
            # longer validates, leaving the attempt itself marked complete.
            resume_failed = (
                state == "failed"
                and att.get("state") == "complete"
                and row.get("resume_check") is not None
            )
            require(
                att.get("state") == state or resume_failed,
                f"{job_id}: attempt and job state differ",
            )
            require(
                att.get("argv") == mj["argv"] and att.get("cwd") == mj["cwd"],
                f"{job_id}: attempt ran a different command than the manifest",
            )
            require(
                att.get("started_at") is not None,
                f"{job_id}: attempt has no start time",
            )
            if state != "interrupted":
                require(
                    att.get("ended_at") is not None
                    and _finite(att.get("elapsed_seconds")),
                    f"{job_id}: terminal attempt without controller timing",
                )
        if state == "complete":
            sha = row.get("result_sha256")
            val = att.get("validation") or {}
            require(
                att.get("exit_code") == 0
                and _is_sha(sha)
                and val.get("ok") is True
                and val.get("sha256") == sha,
                f"{job_id}: complete without a matching controller validation",
            )
        if state == "timed_out":
            require(
                ((att or {}).get("termination") or {}).get("stop_reason")
                in STOP_REASONS,
                f"{job_id}: timed out without a stop reason",
            )
        outcomes[job_id] = {
            "stage": stage,
            "state": state,
            "resume_failed": resume_failed,
            "state_reason": row.get("state_reason"),
            "controller_result_sha256": row.get("result_sha256"),
            "attempt": att,
            "timeout_seconds": mj.get("timeout_seconds"),
            "job_limit_seconds": float(mj["flags"]["--job-limit-seconds"]),
        }
    return outcomes


def stage_span(inv):
    began = inv["controller_runs"][0]["started_at"]
    ended = inv["controller_runs"][-1]["ended_at"]
    return {
        "campaign_id": inv["campaign_id"],
        "controller_runs": len(inv["controller_runs"]),
        "controller_started_at_utc": began,
        "controller_ended_at_utc": ended,
        "controller_exit_code": inv["controller_runs"][-1]["exit_code"],
    }


# ---------------------------------------------------------------------- results


def read_result(inputs, job_id, outcome, root):
    """The job's result, or None with a visible reason when there is none."""
    path = pathlib.Path(root) / job_id / RESULT_FILE
    state = outcome["state"]
    if not path.is_file():
        require(state != "complete", f"{job_id}: complete job without its result")
        return None, None, "missing" if outcome["attempt"] else "none, never launched"
    require(
        outcome["attempt"] is not None,
        f"{job_id}: a result exists for a job that never launched",
    )
    try:
        rec, sha = inputs.json(f"{job_id}/{RESULT_FILE}", path)
    except PilotInputError:
        require(state != "complete", f"{job_id}: complete job with unreadable result")
        return None, inputs.files[f"{job_id}/{RESULT_FILE}"][1], "unreadable"
    require(isinstance(rec, dict), f"{job_id}: result is not an object")
    if state == "complete":
        require(
            sha == outcome["controller_result_sha256"],
            f"{job_id}: result hash differs from the controller's hash",
        )
    return rec, sha, "read"


# Controller failure kinds that reject the result itself (campaign._accept_or_reject).
REJECTING_FAILURES = {"validation", "source_changed"}


def rejection_reason(outcome, rec):
    """Why a result the controller did not accept is kept out of the science.

    Such a job keeps its controller outcome and wall time as an inventory-only
    row; its record is neither validated as a run nor used for metrics or counts.
    """
    if outcome["state"] == "complete":
        return None
    kind = (outcome["attempt"] or {}).get("failure_kind")
    if outcome["resume_failed"]:
        return "the resumed controller found the completed result changed or invalid"
    if kind in REJECTING_FAILURES:
        return f"the controller rejected the result ({kind})"
    if rec.get("status") == "complete":
        return (
            f"the record reports complete but the controller state is "
            f"{outcome['state']}, so it was never accepted"
        )
    return None


def check_status(job_id, rec, state):
    status = rec.get("status")
    final = (rec.get("snapshot") or {}).get("final")
    if state == "complete":
        require(
            status == "complete" and final is True and rec.get("error") is None,
            f"{job_id}: controller says complete but the record is not final",
        )
    else:
        require(
            status in ("running", "partial", "failed"),
            f"{job_id}: controller state {state} with record status {status}",
        )
        require(
            final is (status != "running"),
            f"{job_id}: snapshot finality contradicts status {status}",
        )


def reference_identity(prep, prep_sha, records, expected):
    """Expert, preparation, source and packages shared by every record."""
    ref = next(iter(records.values()))
    for key in ("expert_sha256", "preparation", "source", "package_versions"):
        for job_id, rec in records.items():
            require(
                _json_equal(rec.get(key), ref.get(key)),
                f"{job_id}: {key} differs between records",
            )
    src = ref["source"]
    require(
        isinstance(src.get("files"), dict)
        and src.get("combined_sha256")
        == config_digest(src["files"])
        == expected.scientific_source_combined_sha256,
        "records do not come from the expected scientific execution source",
    )
    p = ref["preparation"]
    ck = prep.get("checkpoint") or {}
    require(
        prep.get("status") == "complete"
        and prep.get("qualified") is True
        and prep.get("approved") is True
        and prep.get("env_name") == ENV_NAME,
        "preparation record is not a qualified, approved CartPole expert",
    )
    require(
        p.get("record_sha256") == prep_sha
        and p.get("protocol") == prep.get("protocol")
        and p.get("config_sha256") == prep.get("config_sha256")
        and p.get("expert_sha256") == ck.get("sha256") == ref["expert_sha256"]
        and p.get("policy_state_sha256") == ck.get("policy_state_sha256")
        and p.get("qualification_mean_return")
        == (prep.get("qualification") or {}).get("mean_return"),
        "records cite a preparation other than the supplied record",
    )
    return ref


# Expected fields that describe a finished outcome rather than identity.
OUTCOME_FIELDS = {"status", "audit_status"}


def check_common(job_id, rec, schema, ref, mj, complete):
    require(
        rec.get("protocol") == PROTOCOL,
        f"{job_id}: protocol {rec.get('protocol')!r} is not {PROTOCOL}",
    )
    require(rec.get("schema") == schema, f"{job_id}: unexpected schema")
    require(
        rec.get("env_name") == ENV_NAME and rec.get("episode_cap") == CAP,
        f"{job_id}: unexpected environment or step cap",
    )
    for key in ("expert_sha256", "preparation", "source", "package_versions"):
        require(_json_equal(rec.get(key), ref[key]), f"{job_id}: {key} differs")
    for key, value in mj["expected_fields"].items():
        if key != "status" and (complete or key not in OUTCOME_FIELDS):
            require(
                _json_equal(rec.get(key), value),
                f"{job_id}: {key} differs from the manifest's expected field",
            )


def check_data(inputs, data_dir, acq_inv, expected):
    """The reused /1 acquisition, accepted only as the exact approved bytes."""
    data_dir = pathlib.Path(data_dir)
    d, sha = inputs.json("acquisition/result.json", data_dir / RESULT_FILE)
    require(
        sha == expected.data_result_sha256,
        "data result is not the approved pinned acquisition",
    )
    require(
        d.get("protocol") == DATA_PROTOCOL and d.get("schema") == DATA_SCHEMA,
        "pinned data does not carry the /1 data producer protocol",
    )
    require(
        d.get("status") == "complete"
        and d["snapshot"]["final"] is True
        and d.get("env_name") == ENV_NAME
        and d.get("episode_cap") == CAP
        and d.get("seed") == SEED,
        "pinned data record is not a complete record",
    )
    cfg = d["config"]
    require(
        config_digest(cfg) == d["config_sha256"]
        and cfg["seed"] == SEED
        and cfg["budget"] == N_LABELS
        and cfg["baseline_episodes"] == BASELINE_EPISODES,
        "pinned data config differs from the protocol",
    )
    src = d["source"]
    require(
        src["combined_sha256"] == config_digest(src["files"]),
        "pinned data source fingerprint is inconsistent",
    )
    b = d["baselines"]
    require(
        all(
            _finite(b.get(k))
            for k in ("expert_return", "random_return", "expert_self_ce")
        )
        and abs(b["expert_return"] - b["random_return"]) >= 1e-8,
        "pinned data baselines are not finite and distinct",
    )
    arrays = {}
    for name, file_name in DATA_FILES.items():
        entry = d["datasets"][name]
        raw = inputs.read(f"acquisition/{file_name}", data_dir / file_name)
        require(
            entry["file"] == file_name
            and hashlib.sha256(raw).hexdigest() == entry["file_sha256"],
            f"{file_name} does not match the pinned data record",
        )
        with np.load(data_dir / file_name, allow_pickle=False) as saved:
            obs, acts = saved["obs"], saved["acts"]
        require(
            pairs_sha256(obs, acts) == entry["pairs_sha256"]
            and entry["n"] == len(acts) == N_LABELS,
            f"{file_name} content does not match the pinned data record",
        )
        arrays[name] = (obs, acts)
    pool_lengths = d["datasets"]["pool"]["stats"]["episode_lengths"]
    require(
        len(d["datasets"]["stream"]["stats"]["episode_lengths"]) == N_LABELS,
        "stream does not hold one episode per label",
    )
    pool_logical_cost(pool_lengths, N_LABELS)
    rows = [
        (j, row)
        for j, row in (acq_inv.get("jobs") or {}).items()
        if row.get("result_sha256") == sha
    ]
    require(len(rows) == 1, "acquisition inventory does not record the pinned result")
    job_id, row = rows[0]
    atts = row.get("attempts") or []
    require(
        row.get("state") == "complete"
        and len(atts) == 1
        and atts[0].get("exit_code") == 0
        and (atts[0].get("validation") or {}).get("sha256") == sha,
        "acquisition inventory does not show one validated complete attempt",
    )
    return {
        "record": d,
        "sha256": sha,
        "arrays": arrays,
        "inventory_job": job_id,
        "inventory": acq_inv,
        "attempt": atts[0],
    }


def check_run(job_id, rec, data, expected, job_limit):
    method, restriction = split(job_id)
    d = data["record"]
    require(
        rec.get("method") == method
        and rec.get("restriction_id") == restriction
        and rec.get("seed") == SEED,
        f"{job_id}: method, restriction or seed",
    )
    cfg = rec["config"]
    require(
        config_digest(cfg) == rec.get("config_sha256"),
        f"{job_id}: config_sha256 does not match the producer canonical digest",
    )
    require(
        cfg["seed"] == SEED
        and cfg["method"] == method
        and cfg["restriction_id"] == restriction
        and cfg["restriction"]["restriction_id"] == restriction
        and cfg["restriction"]["zeroed_indices"] == ZEROED[restriction],
        f"{job_id}: config restriction differs from the job",
    )
    require(
        _json_equal(cfg["beta"], BETA[method])
        and cfg["mixture"] is False
        and cfg["eval_episodes"] == EVAL_EPISODES
        and cfg["eval_deterministic"] is True,
        f"{job_id}: beta, mixture or evaluation settings differ",
    )
    schedule = [N_LABELS] if method == "bc" else ONLINE_SCHEDULE
    require(
        cfg["eval_budgets"] == schedule,
        f"{job_id}: evaluation schedule differs from the protocol",
    )
    require(
        (cfg["fixed_bc"] is not None) == (method == "bc"),
        f"{job_id}: fixed BC rule present on the wrong method",
    )
    exp = cfg["experiment"]
    require(
        exp["algo"] == method
        and all(_json_equal(exp.get(k), v) for k, v in EXPERIMENT_PINS.items()),
        f"{job_id}: experiment settings differ from the protocol",
    )
    rd = rec["data"]
    pool, stream = (d["datasets"][k]["pairs_sha256"] for k in ("pool", "stream"))
    require(
        rd["data_result_sha256"] == data["sha256"]
        and rd["pool_pairs_sha256"] == pool
        and rd["stream_pairs_sha256"] == stream
        and _json_equal(rd["baselines"], d["baselines"]),
        f"{job_id}: data digests or baselines differ from the pinned data",
    )
    acc = rd["acceptance"]
    require(
        acc["rule"] == "pinned_different_source"
        and acc["expected_data_sha256"]
        == acc["data_result_sha256"]
        == expected.data_result_sha256
        and acc["checks"] in (ACCEPTANCE_CHECKS, sorted([*ACCEPTANCE_CHECKS, "baseline_reference"]))
        and acc["producer_protocol"] == DATA_PROTOCOL
        and acc["producer_schema"] == DATA_SCHEMA
        and _json_equal(acc["producer_source"], d["source"])
        and _json_equal(acc["producer_package_versions"], d["package_versions"])
        and acc["current_source_combined_sha256"]
        == expected.scientific_source_combined_sha256,
        f"{job_id}: data acceptance is not the pinned reuse of the /1 data",
    )
    require(
        sorted(d["source"]["files"]) == sorted(rec["source"]["files"])
        and _json_equal(d["package_versions"], rec["package_versions"])
        and d["expert_sha256"] == rec["expert_sha256"]
        and _json_equal(d["preparation"], rec["preparation"]),
        f"{job_id}: pinned data was made for another expert or environment",
    )
    shared = rec["accounting"]["shared_acquisition"]
    kind = SHARED_DATASET[method]
    if kind is None:
        require(shared is None, f"{job_id}: FTL records shared acquisition")
    else:
        require(
            shared["dataset"] == kind
            and shared["charged_to_this_run"] is False
            and shared["pairs_sha256"] == d["datasets"][kind]["pairs_sha256"],
            f"{job_id}: shared acquisition does not match the pinned data",
        )
    require(
        rec["deadline"]["job_limit_seconds"] == job_limit,
        f"{job_id}: job limit differs from the manifest",
    )
    return exp


def check_configs(runs):
    """Runs share one experiment config apart from method-specific keys."""

    def common(rec):
        return {
            k: v for k, v in rec["config"]["experiment"].items() if k not in METHOD_KEYS
        }

    if not runs:
        return
    ref = common(next(iter(runs.values())))
    for job_id, rec in runs.items():
        require(
            _json_equal(common(rec), ref),
            f"{job_id}: experiment config differs from the other runs",
        )
    for m in METHODS:
        pair = [runs[f"run-{m}-{r}"] for r in RESTRICTIONS if f"run-{m}-{r}" in runs]
        if len(pair) == 2:
            require(
                _json_equal(
                    pair[0]["config"]["experiment"], pair[1]["config"]["experiment"]
                ),
                f"{m}: full and masked runs differ beyond the restriction",
            )


# ------------------------------------------------------------------ extraction


def check_fit(job_id, r, n, exp):
    """One recorded fit: the batch and held-out split the trainer actually used."""
    max_epochs = exp["bc_n_epochs"]
    stop, traj = r.get("inner_es_stop_epoch"), r.get("inner_es_val_nll_trajectory")
    fallback, best = r.get("inner_es_fallback"), r.get("inner_es_val_nll_best")
    where = f"{job_id}: round {r['round']}"
    require(
        isinstance(stop, int) and 1 <= stop <= max_epochs and isinstance(traj, list),
        f"{where}: malformed optimizer record",
    )
    n_val = int(exp["inner_early_stop_val_frac"] * n)
    if n_val < exp["inner_early_stop_min_val_size"]:
        require(
            fallback == "dataset_too_small"
            and stop == max_epochs
            and traj == []
            and best is None,
            f"{where}: {n} labels must use the fixed-budget fallback",
        )
        examples = n
    else:
        require(
            fallback is None
            and len(traj) == stop
            and all(_finite(v) for v in traj)
            and best == min(traj),
            f"{where}: held-out split record is inconsistent",
        )
        examples = n - n_val
    batch = min(exp["bc_batch_size"], n, examples)
    require(
        r.get("train_examples") == examples
        and r.get("train_batch_size") == batch
        and r.get("train_batches_per_epoch") == examples // batch
        and r.get("train_drop_last") is True,
        f"{where}: trained loader differs from batch min({exp['bc_batch_size']}, "
        f"labels) on the training split",
    )
    return {
        "round": r["round"],
        "labels": n,
        "stop_epoch": stop,
        "fallback": fallback,
        "train_batch_size": batch,
        "train_examples": examples,
        "val_nll_best": best,
        "val_nll_last": traj[-1] if traj else None,
        "lowest_val_nll_at_last_epoch": (
            traj.index(min(traj)) == len(traj) - 1 if traj else None
        ),
    }


def point(job_id, r, baselines):
    """A saved evaluation: stored metrics as recorded, plus the raw returns."""
    p = evaluation_point(job_id, r, EVAL_EPISODES, CAP, baselines)
    ret = r["episode_returns"]
    require(
        all(_finite(v) and 0 <= v <= CAP for v in ret)
        and all(_finite(r[k]) for k in METRICS)
        and 0 <= r["disagreement_rate"] <= 1,
        f"{job_id}: round {r['round']} has non-finite or invalid metrics",
    )
    require(
        isinstance(r.get("checkpoint"), str) and r["checkpoint"],
        f"{job_id}: round {r['round']} evaluation has no checkpoint",
    )
    return {**p, "episode_returns": list(ret), "checkpoint": r["checkpoint"]}


def check_accounting(job_id, rec, complete, fits, evals, method):
    """Recompute the producer's accounting (run_job.accounting) from the rounds.

    Flags are never trusted on their own: observed steps must equal completed
    collection plus evaluation steps plus unfinished work, and exact totals are
    accepted only when the record finished with nothing unattributed. Only FTL
    collects in the run; BC-iid ``collection_steps`` repeat the reused stream and
    are not run work.
    """
    acc = rec["accounting"]
    done = acc["completed_records"]
    records = rec["records"]
    epochs = [f["stop_epoch"] for f in fits]
    want = {
        "fits": len(fits),
        "epochs": sum(epochs),
        "early_stopped_fits": sum(e < EXPERIMENT_PINS["bc_n_epochs"] for e in epochs),
        "fixed_budget_fallback_fits": sum(f["fallback"] is not None for f in fits),
    }
    require(done["training"] == want, f"{job_id}: training counters differ from rounds")
    eval_steps = sum(p["eval_env_steps"] for p in evals)
    if method == "ftl" and records:
        steps = int(records[-1].get("collection_steps", 0))
        episodes = sum(
            int(r.get("trajectories_collected_this_round", 0)) for r in records
        )
    else:
        steps = episodes = 0
    queries = ("expert_predict_calls", "expert_action_entries", "learner_predict_calls")
    ev, col = done["evaluation"], done["collection"]
    require(
        ev["evaluations"] == len(evals)
        and ev["episodes"] == sum(p["episodes"] for p in evals)
        and ev["env_steps"] == eval_steps
        and all(ev[k] == eval_steps for k in queries),
        f"{job_id}: evaluation counters differ from the saved evaluations",
    )
    require(
        col["env_steps"] == steps
        and col["episodes"] == episodes
        and all(col[k] == steps for k in queries)
        and done["retained_labels"]
        == (records[-1]["n_observations"] if records else 0),
        f"{job_id}: collection or label counters differ from rounds",
    )
    counted = acc["observed"]["env_steps"]
    unfinished = None if counted is None else counted - (steps + eval_steps)
    exact = rec.get("status") == "complete" and unfinished == 0
    flight, totals = acc["in_flight"], acc["totals"]
    require(
        (counted is None or (isinstance(counted, int) and unfinished >= 0))
        and totals["env_steps"] == counted
        and flight["env_steps"] == unfinished
        and acc["reconciled"] is (None if counted is None else unfinished == 0)
        and totals["exact"] is exact
        and all(totals[k] == (steps + eval_steps if exact else None) for k in queries)
        and all(
            flight[k] == (0 if exact else None)
            for k in ("expert_predict_calls", "learner_predict_calls")
        ),
        f"{job_id}: observed steps do not equal completed collection plus "
        "evaluation plus unfinished work, or exact totals are claimed without it",
    )
    if complete:
        require(exact, f"{job_id}: complete record without exact, reconciled totals")


def bc_reference(job_id, rec, data, exp, complete):
    """The one offline fit on the whole pool and its evaluation, or None."""
    records = rec["records"]
    if not records:
        require(not complete, f"{job_id}: complete offline BC without its record")
        # No fit and no evaluation: the published counters must still add up.
        check_accounting(job_id, rec, complete, [], [], "bc")
        return None, []
    require(
        len(records) == 1,
        f"{job_id}: offline BC has {len(records)} records; "
        "it is one fit and one evaluation, prefix fits are not accepted",
    )
    r = records[0]
    d = data["record"]["datasets"]["pool"]
    require(
        r["round"] == 0
        and r["n_observations"] == N_LABELS
        and r.get("reference") == "flat",
        f"{job_id}: record is not the round 0, {N_LABELS}-label flat reference",
    )
    require(
        "prefix_pairs_sha256" not in r,
        f"{job_id}: offline BC record carries a prefix digest",
    )
    require(
        r.get("trained_pairs_sha256")
        == rec.get("trained_pairs_sha256")
        == d["pairs_sha256"]
        == pairs_sha256(*data["arrays"]["pool"]),
        f"{job_id}: offline BC was not trained on the whole fixed pool",
    )
    require(
        r.get("logical") == pool_logical_cost(d["stats"]["episode_lengths"], N_LABELS),
        f"{job_id}: logical acquisition differs from the pool episodes",
    )
    fit = check_fit(job_id, r, N_LABELS, exp)
    p = point(job_id, r, rec["data"]["baselines"])
    check_accounting(job_id, rec, complete, [fit], [p], "bc")
    return {
        **p,
        "trained_on": f"the whole {N_LABELS:,}-label offline pool",
        "trained_pairs_sha256": r["trained_pairs_sha256"],
        "fit": fit,
        "fit_seconds": r["wall_seconds"].get("fit"),
        "eval_seconds": r["wall_seconds"].get("eval"),
    }, [fit]


_EVAL_KEYS = (
    "normalized_return",
    "d_eval_size",
    "episode_returns",
    "checkpoint",
    "rollout_cross_entropy",
    "expert_rollout_cross_entropy",
    "disagreement_rate",
)


def online_curve(job_id, rec, data, exp, complete):
    """Every round 0..k, evaluated exactly on the protocol schedule."""
    method, _ = split(job_id)
    records = rec["records"]
    require(
        [r["round"] for r in records] == list(range(len(records))),
        f"{job_id}: records do not cover every round from 0 in order",
    )
    require(len(records) <= N_LABELS + 1, f"{job_id}: more rounds than labels")
    if complete:
        require(
            len(records) == N_LABELS + 1,
            f"{job_id}: complete run does not record every round to {N_LABELS}",
        )
    stream = data["record"]["datasets"]["stream"]
    stream_steps = stream["stats"]["env_steps"]
    lengths = stream["stats"]["episode_lengths"]
    s_obs, s_acts = data["arrays"]["stream"]
    schedule = set(ONLINE_SCHEDULE)
    points, fits, prev_steps = [], [], 0
    for r in records:
        t = r["round"]
        where = f"{job_id}: round {t}"
        require(r["n_observations"] == t, f"{where} has {r['n_observations']} labels")
        evaluated = r.get("d_eval_size") is not None
        require(
            evaluated == (t in schedule),
            f"{where}: evaluation {'present' if evaluated else 'absent'} "
            "against the protocol schedule",
        )
        if evaluated:
            points.append(point(job_id, r, rec["data"]["baselines"]))
        else:
            require(
                all(r.get(k) is None for k in _EVAL_KEYS),
                f"{where}: non-evaluation round carries evaluation values",
            )
        if t == 0:
            require(
                r.get("train_cross_entropy") is None
                and r.get("inner_es_stop_epoch") is None,
                f"{where}: the untrained round 0 records a fit",
            )
            continue
        require(_finite(r.get("train_cross_entropy")), f"{where}: no training loss")
        fit = check_fit(job_id, r, t, exp)
        if evaluated:
            points[-1].update(
                train_cross_entropy=r["train_cross_entropy"], optimizer=fit
            )
        fits.append(fit)
        steps = r["collection_steps"]
        if method == "ftl":
            require(
                steps > prev_steps and r["trajectories_collected_this_round"] >= 1,
                f"{where}: FTL collected no new expert-labeled steps",
            )
            prev_steps = steps
        else:
            require(
                steps == stream_steps
                and r["trajectories_collected_this_round"] == 0
                and r.get("logical")
                == {"labels": t, "episodes": t, "env_steps": sum(lengths[:t])},
                f"{where}: BC-iid replay differs from the shared stream",
            )
            if evaluated:
                require(
                    r.get("prefix_pairs_sha256") == pairs_sha256(s_obs[:t], s_acts[:t]),
                    f"{where}: BC-iid did not train on the stream's first {t} labels",
                )
    if complete:
        trained = rec.get("trained_pairs_sha256")
        require(_is_sha(trained), f"{job_id}: complete run without trained digest")
        if method == "bc_iid":
            require(
                trained == stream["pairs_sha256"],
                f"{job_id}: BC-iid trained on data other than the stream",
            )
    check_accounting(job_id, rec, complete, fits, points, method)
    return points, fits


def optimizer_summary(fits, max_epochs):
    """Descriptive counts of what the inner optimizer did; no thresholds applied."""
    if not fits:
        return None
    held_out = [f for f in fits if f["fallback"] is None]
    return {
        "fits": len(fits),
        "epochs": sum(f["stop_epoch"] for f in fits),
        "fixed_budget_fallback_fits": len(fits) - len(held_out),
        "held_out_split_fits": len(held_out),
        "held_out_fits_at_epoch_cap": sum(
            f["stop_epoch"] == max_epochs for f in held_out
        ),
        "early_stopped_fits": sum(f["stop_epoch"] < max_epochs for f in fits),
        "early_stopped_rounds": [
            f["round"] for f in fits if f["stop_epoch"] < max_epochs
        ],
        "held_out_fits_lowest_val_nll_at_last_epoch": sum(
            bool(f["lowest_val_nll_at_last_epoch"]) for f in held_out
        ),
        "first_held_out_round": held_out[0]["round"] if held_out else None,
        "epoch_cap": max_epochs,
        "note": "descriptive only: reaching the epoch cap is not by itself a "
        "failure, and an early stop is not evidence of convergence",
    }


def matched_budget(curves):
    """Largest label count with a saved evaluation in all four online curves."""
    if any(not curves.get(j) for j in ONLINE_IDS):
        return None
    shared = set.intersection(*({p["labels"] for p in curves[j]} for j in ONLINE_IDS))
    return max(shared) if shared else None


# ----------------------------------------------------------------------- costs


def attempt_row(job_id, outcome, rec, sha, availability):
    att = outcome["attempt"] or {}
    term = att.get("termination") or {}
    row = {
        "job": job_id,
        "stage": outcome["stage"],
        "controller_state": outcome["state"],
        "state_reason": outcome["state_reason"],
        "attempts": 0 if outcome["attempt"] is None else 1,
        "attempt_id": att.get("attempt_id"),
        "exit_code": att.get("exit_code"),
        "failure_kind": att.get("failure_kind"),
        "stop_reason": term.get("stop_reason"),
        "controller_started_at_utc": att.get("started_at"),
        "controller_ended_at_utc": att.get("ended_at"),
        "controller_elapsed_seconds": att.get("elapsed_seconds"),
        "timeout_seconds": outcome["timeout_seconds"],
        "job_limit_seconds": outcome["job_limit_seconds"],
        "result": availability,
        "result_sha256": sha,
        "controller_result_sha256": outcome["controller_result_sha256"],
        "record_status": None if rec is None else rec.get("status"),
        "snapshot_final": None if rec is None else rec["snapshot"].get("final"),
        "snapshot_written_at_utc": (
            None if rec is None else rec["snapshot"].get("written_at_utc")
        ),
        "error_type": None if rec is None else (rec.get("error") or {}).get("type"),
        "counts_are": (
            "exact final totals"
            if outcome["state"] == "complete"
            else (
                "saved counts as of the last snapshot; lower bounds, later work unknown"
                if rec is not None
                else (
                    "unknown: result not accepted by the controller and not used"
                    if availability.startswith("rejected")
                    else "unknown: no readable record"
                )
            )
        ),
    }
    if rec is not None and "accounting" in rec:
        acc = rec["accounting"]
        done = acc["completed_records"]
        records = rec["records"]
        row.update(
            {
                "last_recorded_round": records[-1]["round"] if records else None,
                "observed": acc["observed"],
                "in_run_collection": {
                    k: done["collection"][k] for k in ("env_steps", "episodes")
                },
                "evaluation": {
                    k: done["evaluation"][k]
                    for k in ("evaluations", "episodes", "env_steps")
                },
                "training": done["training"],
                "retained_labels": done["retained_labels"],
                "totals": acc["totals"],
                "reconciled": acc["reconciled"],
                "in_flight_operation": acc["in_flight"]["operation"],
                "wall_seconds_recorded": acc["wall_seconds"],
            }
        )
    return row


def known_sum(items):
    """Sum of ``(value, exact)`` items; never labelled exact unless all are."""
    known = [(v, e) for v, e in items if v is not None]
    lower = any(not e for _, e in known)
    if len(known) < len(items):
        text = "known subtotal only; unknown values are not included"
    elif lower:
        text = (
            "sum of recorded values, not a total: incomplete runs contribute "
            "saved counts that are lower bounds"
        )
    else:
        text = "exact sum of the recorded values"
    if lower and len(known) < len(items):
        text += "; incomplete runs contribute lower bounds"
    return {
        "sum": sum(v for v, _ in known) if known else None,
        "known": len(known),
        "unknown": len(items) - len(known),
        "includes_lower_bounds": lower,
        "label": text,
    }


def new_execution_costs(rows):
    """Sums over launched attempts; a job never launched did no work."""
    launched = [j for j in ["audit"] + list(RUN_IDS) if rows[j]["attempts"]]
    runs = [j for j in launched if j != "audit"]

    def field(j, group, key):
        exact = rows[j]["controller_state"] == "complete"
        return (rows[j].get(group) or {}).get(key), exact

    return {
        "scope": "the audit and the run attempts executed for this pilot",
        "launched_attempts": len(launched),
        "not_launched": [
            j for j in ["audit"] + list(RUN_IDS) if not rows[j]["attempts"]
        ],
        # Controller wall time is measured externally, so it is exact when known.
        "controller_elapsed_seconds": known_sum(
            [(rows[j]["controller_elapsed_seconds"], True) for j in launched]
        ),
        "run_observed_env_steps": known_sum(
            [field(j, "observed", "env_steps") for j in runs]
        ),
        "ftl_in_run_collection_env_steps": known_sum(
            [
                field(j, "in_run_collection", "env_steps")
                for j in runs
                if split(j)[0] == "ftl"
            ]
        ),
        "evaluation_env_steps": known_sum(
            [field(j, "evaluation", "env_steps") for j in runs]
        ),
        "note": "worker wall time measured by the external controller, not CPU or "
        "GPU hours and not money; BC-iid collection_steps fields repeat the reused "
        "stream acquisition and are not counted here",
    }


def acquisition_costs(data):
    d, att = data["record"], data["attempt"]
    pool, stream = (d["datasets"][k]["stats"] for k in ("pool", "stream"))
    base = d["baselines_costs"]
    keep = (
        "elapsed_seconds",
        "env_steps",
        "episodes",
        "env_resets",
        "expert_predict_calls",
        "retained_labels",
    )
    return {
        "label": "historical protocol /1 data acquisition reused by exact pin; not "
        "new work of this pilot and charged once, not per run, method or condition",
        "data_result_sha256": data["sha256"],
        "producer_protocol": d["protocol"],
        "producer_source_combined_sha256": d["source"]["combined_sha256"],
        "acquisition_campaign_id": data["inventory"].get("campaign_id"),
        "acquisition_campaign_source_sha256": data["inventory"].get("source_sha256"),
        "controller_job": data["inventory_job"],
        "controller_started_at_utc": att.get("started_at"),
        "controller_ended_at_utc": att.get("ended_at"),
        "controller_elapsed_seconds": att.get("elapsed_seconds"),
        "fixed_bc_pool": {k: pool.get(k) for k in keep + ("overshoot_transitions",)},
        "bc_iid_stream": {k: stream.get(k) for k in keep},
        "normalization_baselines": {
            k: base.get(k)
            for k in (
                "elapsed_seconds",
                "env_steps_counted",
                "expert_episodes",
                "random_episodes",
                "expert_predict_calls",
            )
        },
        "note": "internal phase timers need not sum to the controller time",
    }


# --------------------------------------------------------------------- figures


def curves_figure(curves, refs, states, baselines):
    """Four online curves and two flat offline BC references; no smoothing."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.8), sharex=True)
    panels = [
        (
            "normalized_return",
            "Normalized return",
            r"$\widehat{J}_{\mathrm{norm}}(\pi_n)=(\widehat{J}(\pi_n)-J_{\mathrm{rand}})/(J_E-J_{\mathrm{rand}})$",
        ),
        (
            "disagreement_rate",
            "Disagreement with the expert's action",
            r"$\mathbb{E}_{s\sim\hat d_n}[\mathbf{1}\{a_n(s)\ne a_E(s)\}]$",
        ),
        (
            "rollout_cross_entropy",
            "Learner cross-entropy of the expert's action",
            r"$\mathbb{E}_{s\sim\hat d_n}[-\log\pi_n(a_E(s)\mid s)]$ (nats)",
        ),
        (
            "expert_rollout_cross_entropy",
            "Expert's own cross-entropy of its action, same states",
            r"$\mathbb{E}_{s\sim\hat d_n}[-\log\pi_E(a_E(s)\mid s)]$ (nats)",
        ),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        for job_id in ONLINE_IDS:
            m, r = split(job_id)
            pts = curves.get(job_id) or []
            if not pts:
                continue
            xs = [p["labels"] for p in pts]
            ys = [p[key] for p in pts]
            ax.plot(
                xs, ys, ls=STYLE[r], color=COLOR[m], lw=1.5, alpha=0.9, label=job_id
            )
            if states[job_id] != "complete":
                ax.plot(
                    xs[-1:],
                    ys[-1:],
                    ls="none",
                    marker="X",
                    ms=9,
                    color=COLOR[m],
                    mec="black",
                    mew=0.8,
                    zorder=5,
                )
        for job_id in BC_IDS:
            ref = refs.get(job_id)
            if ref is not None:
                ax.axhline(
                    ref[key],
                    ls=STYLE[split(job_id)[1]],
                    color=COLOR["bc"],
                    lw=1.5,
                    alpha=0.9,
                    zorder=1,
                    label=job_id,
                )
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-15, N_LABELS + 15)
        ax.grid(alpha=0.3)
    ax = axes[0, 0]
    ax.axhline(1.0, color="#555555", ls=":", lw=1.2, zorder=0)
    ax.axhline(0.0, color="#999999", ls="-.", lw=1.0, zorder=0)
    set_normalized_return_axis(ax)
    for ax in axes[1]:
        ax.set_xlabel(r"$n$: expert labels used for training")
    handles = [
        Line2D([], [], color=COLOR[m], lw=2.5, label=METHOD_LABEL[m]) for m in ONLINE
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=COLOR["bc"],
            lw=2.5,
            label=f"Offline BC: one fit on all {N_LABELS:,} pool "
            "labels,\nflat reference, not label matched",
        )
    )
    handles += [
        Line2D(
            [],
            [],
            color="#444444",
            ls=STYLE[r],
            lw=1.5,
            label=OBS_LABEL[r].capitalize(),
        )
        for r in RESTRICTIONS
    ]
    if any(states.get(j) != "complete" for j in ONLINE_IDS):
        handles.append(Line2D([], [], ls="none", marker="X", ms=9,
                              color="#999999", mec="black",
                              label="Incomplete run endpoint"))
    handles += [
        Line2D(
            [],
            [],
            color="#555555",
            ls=":",
            label=r"Expert reference ($J_E$): 1",
        ),
        Line2D(
            [],
            [],
            color="#999999",
            ls="-.",
            label=r"Random reference ($J_{\mathrm{rand}}$): 0",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.07),
    )
    fig.suptitle(
        "CartPole stronger-mask pilot, training seed 300: online curves "
        "and offline BC references",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.008,
        r"$\hat d_n$: empirical distribution of all pre-action states pooled over 100 deterministic rollouts of $\pi_n$; longer episodes contribute more states." "\n"
        r"Each method and checkpoint has its own $\hat d_n$. $a_n,a_E$: deterministic learner/expert actions; $\pi_n$ includes the observation mask." "\n"
        r"$\widehat{J}$: mean episode return. Offline BC uses its own rollout distribution and one fixed 1,000-label policy. One training seed; no across-seed bands.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.96))
    return fig


def costs_figure(rows, acquisition):
    """Controller wall time of new attempts; the reused acquisition apart."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    jobs = ["audit"] + list(RUN_IDS)
    ys = np.arange(len(jobs) + 1)[::-1]
    for y, job_id in zip(ys[:-1], jobs):
        row = rows[job_id]
        secs = row["controller_elapsed_seconds"]
        color = "#bbbbbb" if job_id == "audit" else COLOR[split(job_id)[0]]
        done = row["controller_state"] == "complete"
        if secs is not None:
            ax.barh(
                y,
                secs / 60,
                color=color,
                alpha=0.9 if done else 0.55,
                hatch=None if done else "//",
                edgecolor="black",
                lw=0.6,
            )
        if not row["attempts"]:
            text = f"not launched, {row['controller_state']}"
        elif secs is None:
            text = f"wall time unknown, {row['controller_state']}"
        else:
            text = f"{num(secs)} s, {row['controller_state']}"
        ax.text((secs or 0) / 60 + 0.5, y, text, va="center", fontsize=8.5)
    secs = acquisition["controller_elapsed_seconds"]
    ax.barh(
        ys[-1], (secs or 0) / 60, color="white", hatch="..", edgecolor="#555555", lw=0.6
    )
    ax.text(
        (secs or 0) / 60 + 0.5,
        ys[-1],
        f"{num(secs)} s, historical /1 acquisition, reused once, not new",
        va="center",
        fontsize=8.5,
    )
    longest = (
        max([rows[j]["controller_elapsed_seconds"] or 0 for j in jobs] + [secs or 0])
        / 60
    )
    ax.set_xlim(0, max(longest, 1.0) * 1.45)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        ["Mask audit"] + [label(j) for j in RUN_IDS] + ["Reused data acquisition"]
    )
    ax.set_xlabel(r"$t_{\mathrm{wall}}$: controller-measured worker time per attempt (minutes)")
    ax.set_ylabel("Attempt / acquisition")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(
        handles=[
            Patch(fc="#999999", ec="black", label="complete"),
            Patch(
                fc="#999999", ec="black", alpha=0.55, hatch="//", label="not complete"
            ),
            Patch(
                fc="white", ec="#555555", hatch="..", label="reused, not charged as new"
            ),
        ],
        loc="lower right",
        fontsize=8.5,
        frameon=False,
    )
    ax.set_title(
        "CartPole stronger-mask pilot: controller wall time",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- report


def _metrics_row(name, p):
    return (
        f"| {name} | {p['labels']:,} | {p['mean_return']:.2f} "
        f"({p['min_return']:.0f} to {p['max_return']:.0f}) | "
        f"{p['episodes_at_cap']}/{p['episodes']} | "
        + " | ".join(f"{p[k]:.6f}" for k in ("normalized_return",) + METRICS)
        + " |"
    )


_METRIC_HEADER = (
    "| Run | Labels | Mean raw return (min to max) | Episodes at cap | Normalized "
    "return | Learner cross-entropy | Disagreement | Expert cross-entropy |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|"
)


def report(s):
    rows, refs, curves = s["outcomes"], s["offline_bc_reference"], s["curves"]
    b = s["baselines"]
    out = [
        "# CartPole stronger-mask pilot: results",
        "",
        f"Producer protocol `{PROTOCOL}`, training seed {SEED}, mask `{MASK}` "
        "(cart position x and pole angular velocity theta_dot set to 0 before the "
        "learner's frozen features; the expert reads full observations). Methods: "
        f"offline BC (one fit on the whole {N_LABELS:,}-label pool), BC-iid and FTL. "
        "This report summarizes stored measurements only. It makes no claim beyond "
        "this one training seed.",
        "",
        "## Outcomes",
        "",
        "| Job | Stage | Controller state | Record | Last recorded round | "
        "Saved evaluations | Controller wall seconds |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for job_id in ["audit"] + list(RUN_IDS):
        r = rows[job_id]
        evals = (r.get("evaluation") or {}).get("evaluations")
        out.append(
            f"| {job_id} | {r['stage']} | {r['controller_state']}"
            + (f" ({r['state_reason']})" if r["state_reason"] else "")
            + f" | {r['result']}"
            + (f", status {r['record_status']}" if r["record_status"] else "")
            + (
                " | n/a | n/a | "
                if job_id == "audit"
                else f" | {num(r.get('last_recorded_round'))} | {num(evals)} | "
            )
            + f"{num(r['controller_elapsed_seconds'])} |"
        )
    out += [
        "",
        "Incomplete outcomes are kept: their saved evaluations are shown as "
        "recorded, their counts are lower bounds and unknown work stays unknown.",
        "",
        "## Learning curves",
        "",
        f"![Learning curves]({FIG_CURVES})",
        "",
        "Normalized return is the stored `normalized_return`, "
        f"(R - R_random) / (R_expert - R_random) with R_random = "
        f"{b['random_return']:g} and R_expert = {b['expert_return']:g} from the "
        f"reused acquisition, so random is 0 and expert is 1. Every stored value was "
        "checked against the raw episode returns. Learner cross-entropy "
        "(`rollout_cross_entropy`) is the learner's mean negative log probability "
        "of the expert's action; expert cross-entropy "
        "(`expert_rollout_cross_entropy`) is the expert's own value on the same "
        "states. Both, and the disagreement rate, are measured on each learner's own "
        "visited states, so methods are compared on different state distributions.",
        "",
        "## Offline BC references",
        "",
        _METRIC_HEADER,
    ]
    for job_id in BC_IDS:
        if refs.get(job_id) is not None:
            out.append(_metrics_row(label(job_id), refs[job_id]))
        else:
            out.append(
                f"| {label(job_id)} | no saved evaluation, no reference "
                "| | | | | | |"
            )
    out += [
        "",
        f"Each reference is one fit on all {N_LABELS:,} pool labels and one "
        f"{EVAL_EPISODES}-episode evaluation. It is drawn flat across the whole "
        "label axis and is not label matched to the online curves.",
        "",
        "## Matched label budget (online curves only)",
        "",
    ]
    mb = s["matched_budget"]
    if mb["labels"] is None:
        out.append(
            "Unavailable: the four online curves share no saved evaluation "
            "budget. Nothing is imputed."
        )
    else:
        out += [
            f"Largest label count evaluated by all four online curves: "
            f"{mb['labels']:,}.",
            "",
            _METRIC_HEADER,
        ]
        out += [
            _metrics_row(f"{label(j)} ({rows[j]['controller_state']})", mb["points"][j])
            for j in ONLINE_IDS
        ]
        unfinished = [
            label(j) for j in ONLINE_IDS if rows[j]["controller_state"] != "complete"
        ]
        out += [
            "",
            "This comparison is descriptive only: one training seed, no uncertainty "
            "estimate and no significance test."
            + (
                " Runs not marked complete contribute saved evaluations of an "
                f"unfinished run ({'; '.join(unfinished)})."
                if unfinished
                else ""
            )
            + (
                " Label count 0 is the untrained initial policy."
                if not mb["labels"]
                else ""
            ),
        ]
    out += ["", "## Last saved evaluation of each online run", "", _METRIC_HEADER]
    for job_id in ONLINE_IDS:
        pts = curves.get(job_id)
        name = f"{label(job_id)} ({rows[job_id]['controller_state']})"
        out.append(
            _metrics_row(name, pts[-1])
            if pts
            else f"| {name} | no saved evaluation | | | | | | |"
        )
    raw = s["return_cap"]
    out += [
        "",
        f"Evaluations with every episode at the {CAP}-step cap: "
        f"{raw['online_trained_at_cap']} of {raw['online_trained']} trained online "
        f"evaluations and {raw['offline_bc_at_cap']} of {raw['offline_bc']} offline "
        "BC references. The cap bounds this return measurement; reaching it, or "
        f"completing {N_LABELS:,} labels, is not evidence of convergence.",
        "",
        "## Mask audit",
        "",
    ]
    a = s["audit"]
    if a is None:
        out.append("No complete audit result; its outcome is listed under Outcomes.")
    else:
        out.append(
            f"Status `{a['audit_status']}`: {num(a['n_conflicting_pairs'])} of "
            f"{num(a['n_checked_pairs'])} checked paired states "
            f"({num(a['n_invalid_pairs'])} invalid) have identical masked "
            "observations and different expert labels. The pairs are fixed x-grid "
            f"witnesses on diagnostic seed {AUDIT_SEED}, never used for training. "
            "They show that conflicting labels exist under the mask. They are not an "
            "on-policy error floor for any learner and not a return ceiling. Every "
            "checked pair differs only in cart position x, so the audit does not "
            "certify any additional conflict caused by also zeroing pole angular "
            f"velocity. Recorded witness scope: {a['witness_scope']}"
        )
    out += [
        "",
        "## Inner optimizer diagnostics",
        "",
        "| Run | Fits | Epochs | Fixed-budget fallback fits | Held-out split fits "
        "| Split fits at epoch cap | Early-stopped fits | Split fits whose lowest "
        "validation NLL is at the last epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for job_id in RUN_IDS:
        o = s["optimizer"].get(job_id)
        if o is None:
            out.append(f"| {label(job_id)} | not recorded | | | | | | |")
            continue
        out.append(
            f"| {label(job_id)} | {o['fits']:,} | {o['epochs']:,} | "
            f"{o['fixed_budget_fallback_fits']:,} | {o['held_out_split_fits']:,} | "
            f"{o['held_out_fits_at_epoch_cap']:,} | {o['early_stopped_fits']:,} | "
            f"{o['held_out_fits_lowest_val_nll_at_last_epoch']:,} |"
        )
    out += [
        "",
        "Below the held-out minimum (fewer than "
        f"{EXPERIMENT_PINS['inner_early_stop_min_val_size']} validation labels) the "
        "trainer uses the original fixed budget of "
        f"{EXPERIMENT_PINS['bc_n_epochs']} epochs. These counts are descriptive. "
        "No threshold is applied: an epoch-cap fit is not by itself a failure, and "
        "an early stop is not evidence of convergence. Per-fit values are in the "
        "curve points file.",
        "",
        "## Costs",
        "",
        f"![Controller wall time]({FIG_COSTS})",
        "",
    ]
    out += [
        "| Job | Controller state | Controller wall s | Recorded setup s | Recorded "
        "total s | Collection and training s | Offline fit s | Evaluation s | FTL "
        "collection steps | Evaluation steps | Counts are |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for job_id in ["audit"] + list(RUN_IDS):
        r = rows[job_id]
        w = r.get("wall_seconds_recorded") or {}
        out.append(
            f"| {job_id} | {r['controller_state']} | "
            f"{num(r['controller_elapsed_seconds'])} | {num(w.get('setup'))} | "
            f"{num(w.get('total'))} | {num(w.get('collection_and_training'))} | "
            f"{num(w.get('fits'))} | {num(w.get('evaluation'))} | "
            f"{num((r.get('in_run_collection') or {}).get('env_steps'))} | "
            f"{num((r.get('evaluation') or {}).get('env_steps'))} | "
            f"{r['counts_are']} |"
        )
    out += [
        "",
        "Recorded timers are the producer's internal phase timers and need not sum "
        "to the controller time. Online fits are timed inside collection and "
        "training; offline BC has no in-run collection. BC-iid and offline BC "
        "collect nothing in their runs; their data come from the reused "
        "acquisition below.",
        "",
    ]
    new, acq = s["costs"]["new_execution"], s["costs"]["reused_acquisition"]
    for key, name in (
        ("controller_elapsed_seconds", "Controller wall seconds"),
        ("run_observed_env_steps", "Run environment steps observed"),
        ("ftl_in_run_collection_env_steps", "FTL in-run collection steps"),
        ("evaluation_env_steps", "Evaluation environment steps"),
    ):
        v = new[key]
        out.append(
            f"- {name}, new execution: {num(v['sum'])} ({v['label']}; "
            f"{v['known']} known, {v['unknown']} unknown)."
        )
    out += [
        f"- Reused acquisition: {num(acq['controller_elapsed_seconds'])} controller "
        f"seconds; pool {num(acq['fixed_bc_pool']['env_steps'])} expert steps, "
        f"stream {num(acq['bc_iid_stream']['env_steps'])} expert steps, baselines "
        f"{num(acq['normalization_baselines']['env_steps_counted'])} steps. "
        "Historical work, reused once by exact pin, not new and not charged per run "
        "or method.",
        "",
        "Wall time is controller-measured worker time, not CPU or GPU hours or "
        "money. Expert preparation, review and this analysis are outside the cost "
        "scope.",
        "",
        "## Provenance",
        "",
        f"- Execution source commit `{s['provenance']['execution_source_commit']}`, "
        f"campaign source hash `{s['provenance']['campaign_source_sha256']}`, "
        "scientific source fingerprint "
        f"`{s['provenance']['scientific_source_combined_sha256']}`.",
        f"- Snapshot verification is external: {s['provenance']['verification_note']}",
        f"- Reused data: `{DATA_SCHEMA}` result "
        f"`{s['provenance']['data_result_sha256']}`, produced by source fingerprint "
        f"`{acq['producer_source_combined_sha256']}` and accepted by every run only "
        "through that exact pin.",
        f"- Expert `{s['provenance']['expert_sha256']}`.",
        "- Input and analysis digests are in the summary JSON.",
        "",
        "## Limitations",
        "",
        "- One training seed: no across-seed confidence interval or significance "
        "statement is possible, and none is made. Spread across evaluation "
        "episodes is not training-seed uncertainty.",
        "- Evaluation episodes of different runs are not assumed to be paired.",
        "- Each method's cross-entropy and disagreement use its own visited states.",
        "- The mask is attributed jointly (x and theta_dot together).",
        "- A negative or null difference between methods is a valid outcome.",
        "- Whether fitting under the restriction is adequate, and whether any "
        "learning curve has reached a plateau, remain unresolved. Completing the "
        f"{N_LABELS:,}-label budget or hitting the inner epoch cap does not "
        "establish convergence. This report runs no diagnostic to settle either "
        "question.",
    ]
    return "\n".join(out) + "\n"


# -------------------------------------------------------------------- analysis


def forbidden_tokens(blobs, paths):
    tokens = {
        "/home/",
        "/Users/",
        "/opt/",
        "/private/",
        "/tmp/",
        "miniconda",
        chr(0x2014),
    }
    # Filesystem paths are banned as absolute paths: a relative argument such as
    # "out" is an ordinary word, not a machine detail.
    absolute = sorted(
        {str(pathlib.Path(p).absolute()) for p in paths}
        | {str(pathlib.Path(p).resolve()) for p in paths}
    )

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and o.startswith("/") and len(o) > 1:
            tokens.add(o)
            for m in re.finditer(r"/(?:home|Users)/([^/]+)", o):
                tokens.add(m.group(1))

    for blob in blobs + [absolute]:
        walk(blob)
    return tokens


def analyze(args, expected=PUBLIC_EXPECTED):
    """Validate every input and build the outputs in memory."""
    inputs = Inputs()
    evidence, _ = inputs.json("trusted_evidence", args.trusted_evidence)
    manifest_sha = check_evidence(evidence, expected)
    manifests, mjobs, canonical = {}, {}, {}
    for stage in STAGE_JOBS:
        manifests[stage], sha = inputs.json(
            f"{stage}/manifest", getattr(args, f"{stage}_manifest")
        )
        require(
            sha == manifest_sha[stage],
            f"{stage} manifest bytes differ from the trusted evidence",
        )
        canonical[stage] = config_digest(manifests[stage])
        mjobs[stage] = check_manifest(stage, manifests[stage], expected)
    flat = check_stage_union(manifests, mjobs)

    invs, outcomes = {}, {}
    for stage in STAGE_JOBS:
        invs[stage], _ = inputs.json(
            f"{stage}/inventory", getattr(args, f"{stage}_inventory")
        )
        outcomes.update(
            check_inventory(
                stage,
                invs[stage],
                manifests[stage],
                canonical[stage],
                mjobs[stage],
                expected,
            )
        )

    prep, prep_sha = inputs.json("preparation.json", args.preparation_record)
    acq_inv, _ = inputs.json("acquisition/inventory", args.acquisition_inventory)
    data = check_data(inputs, args.data_dir, acq_inv, expected)

    records, shas, avail = {}, {}, {}
    for job_id, outcome in outcomes.items():
        root = getattr(args, f"{outcome['stage']}_root")
        rec, shas[job_id], avail[job_id] = read_result(inputs, job_id, outcome, root)
        if rec is not None:
            reason = rejection_reason(outcome, rec)
            if reason is not None:
                avail[job_id] = f"rejected: {reason}"
                continue
            check_status(job_id, rec, outcome["state"])
            records[job_id] = rec
    require(records, "no readable audit or run record")
    ref = reference_identity(prep, prep_sha, records, expected)
    require(
        _json_equal(data["record"]["preparation"], ref["preparation"])
        and data["record"]["expert_sha256"] == ref["expert_sha256"],
        "pinned data was acquired with another expert",
    )
    for job_id, rec in records.items():
        schema = AUDIT_SCHEMA if job_id == "audit" else RUN_SCHEMA
        check_common(
            job_id,
            rec,
            schema,
            ref,
            flat[job_id],
            outcomes[job_id]["state"] == "complete",
        )

    audit = None
    if "audit" in records:
        a = records["audit"]
        cfg = a["config"]
        require(
            cfg["restriction_id"] == MASK and cfg["reset_seed"] == AUDIT_SEED,
            "audit: restriction or reset seed differs from the protocol",
        )
        if a["status"] == "complete":
            inner = a["audit"]
            require(
                a["audit_status"] == inner["status"] == "conflict_found"
                and inner["labels_used_for_training"] is False
                and all(
                    isinstance(inner[k], int)
                    for k in (
                        "n_checked_pairs",
                        "n_conflicting_pairs",
                        "n_invalid_pairs",
                    )
                ),
                "audit: complete record without a conflict-found result",
            )
            require(
                isinstance(cfg.get("witness_scope"), str) and cfg["witness_scope"],
                "audit: record does not state its witness scope",
            )
            audit = {
                k: inner[k]
                for k in (
                    "n_checked_pairs",
                    "n_conflicting_pairs",
                    "n_invalid_pairs",
                    "claim",
                )
            }
            audit.update(
                audit_status=a["audit_status"],
                expert_predict_calls=a["expert_predict_calls"],
                witness_scope=cfg["witness_scope"],
            )

    runs = {j: records[j] for j in RUN_IDS if j in records}
    exp = None
    for job_id, rec in runs.items():
        exp = check_run(
            job_id, rec, data, expected, outcomes[job_id]["job_limit_seconds"]
        )
    check_configs(runs)
    baselines = data["record"]["baselines"]
    refs, curves, fits = {}, {}, {}
    for job_id in RUN_IDS:
        rec = runs.get(job_id)
        if rec is None:
            continue
        complete = outcomes[job_id]["state"] == "complete"
        if split(job_id)[0] == "bc":
            refs[job_id], fits[job_id] = bc_reference(job_id, rec, data, exp, complete)
        else:
            curves[job_id], fits[job_id] = online_curve(
                job_id, rec, data, exp, complete
            )
    refs = {j: refs.get(j) for j in BC_IDS}
    rows = {
        j: attempt_row(j, outcomes[j], records.get(j), shas[j], avail[j])
        for j in outcomes
    }
    for job_id in ONLINE_IDS:
        pts = curves.get(job_id) or []
        rows[job_id]["last_evaluated_labels"] = pts[-1]["labels"] if pts else None
    budget = matched_budget(curves)
    trained = [p for j in ONLINE_IDS for p in curves.get(j, []) if p["labels"] > 0]
    present_refs = [p for p in refs.values() if p is not None]

    def at_cap(ps):
        return sum(p["episodes_at_cap"] == p["episodes"] for p in ps)

    max_epochs = EXPERIMENT_PINS["bc_n_epochs"]
    analysis_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()

    summary = {
        "schema": SUMMARY_SCHEMA,
        "scope": "derived summary of the stronger-mask CartPole pilot (audit and six "
        "runs); raw records, checkpoints and logs remain private",
        "provenance": {
            **dataclasses.asdict(expected),
            "verification_note": evidence["verification_note"],
            "source_identity_rule": "bound by public constants and the trusted "
            "evidence; deployed snapshot verified externally, not by this analysis",
            "manifest_file_sha256": manifest_sha,
            "manifest_canonical_sha256": canonical,
            "expert_sha256": ref["expert_sha256"],
            "preparation_record_sha256": prep_sha,
            "package_versions": ref["package_versions"],
            "result_source_files": ref["source"]["files"],
            "analysis_script": f"experiments/agnostic/{pathlib.Path(__file__).name}",
            "analysis_script_sha256": analysis_sha,
            "reused_helper_script": "experiments/agnostic/analyze_cartpole_pilot.py",
            "reused_helper_script_sha256": HIST_SHA256,
            "analysis_code_note": "analysis script digests identify this analysis "
            "code only, not the execution source",
        },
        "input_sha256": inputs.hashes(),
        "protocol": PROTOCOL,
        "data_protocol": DATA_PROTOCOL,
        "env_name": ENV_NAME,
        "seed": SEED,
        "restrictions": list(RESTRICTIONS),
        "episode_cap": CAP,
        "eval_episodes": EVAL_EPISODES,
        "n_labels": N_LABELS,
        "online_eval_schedule": ONLINE_SCHEDULE,
        "baselines": baselines,
        "stages": {s: stage_span(invs[s]) for s in STAGE_JOBS},
        "outcomes": rows,
        "audit": audit,
        "offline_bc_reference": refs,
        "matched_budget": {
            "labels": budget,
            "rule": "largest label count with a saved evaluation in all four online "
            "curves, counting saved evaluations of incomplete runs under their "
            "state; offline BC is not part of it; unavailable when none is shared",
            "points": (
                None
                if budget is None
                else {
                    j: next(p for p in curves[j] if p["labels"] == budget)
                    for j in ONLINE_IDS
                }
            ),
        },
        "curves": curves,
        "return_cap": {
            "online_trained": len(trained),
            "online_trained_at_cap": at_cap(trained),
            "offline_bc": len(present_refs),
            "offline_bc_at_cap": at_cap(present_refs),
        },
        "optimizer": {
            j: optimizer_summary(fits.get(j, []), max_epochs) for j in RUN_IDS
        },
        "costs": {
            "new_execution": new_execution_costs(rows),
            "reused_acquisition": acquisition_costs(data),
        },
        "notes": [
            "Null values are unknown, never zero.",
            "One training seed; no across-seed uncertainty is estimated.",
            "Metrics are on each learner's own visited states.",
            "Offline BC is one fit on the whole pool, drawn flat, never label matched.",
        ],
    }
    curve_points = {
        "schema": POINTS_SCHEMA,
        "note": "every saved online evaluation as stored, with raw episode returns "
        "and the fit behind it; offline BC is one flat reference per condition",
        "online": {
            j: {
                "controller_state": outcomes[j]["state"],
                "points": curves.get(j, []),
                "fits": fits.get(j, []),
            }
            for j in ONLINE_IDS
        },
        "offline_bc_reference": {
            j: {"controller_state": outcomes[j]["state"], "point": refs[j]}
            for j in BC_IDS
        },
    }
    # Curves live in the points file only.
    summary_view = {k: v for k, v in summary.items() if k != "curves"}
    texts = {
        REPORT: report(summary),
        SUMMARY: json.dumps(summary_view, indent=2) + "\n",
        POINTS: json.dumps(curve_points, indent=2) + "\n",
    }
    states = {j: outcomes[j]["state"] for j in outcomes}
    images = {
        FIG_CURVES: png(curves_figure(curves, refs, states, baselines)),
        FIG_COSTS: png(costs_figure(rows, summary["costs"]["reused_acquisition"])),
    }
    private = forbidden_tokens(
        list(manifests.values()) + list(invs.values()) + [acq_inv],
        [p for p, _ in inputs.files.values()]
        + [args.output_dir, args.data_dir]
        + [getattr(args, f"{s}_root") for s in STAGE_JOBS],
    )
    for name, text in texts.items():
        hits = sorted(t for t in private if t and t in text)
        require(not hits, f"{name}: refusing to write private or banned text {hits}")
    inputs.recheck()
    return {
        "texts": texts,
        "images": images,
        "summary": summary,
        "points": curve_points,
        "inputs": inputs,
    }


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trusted-evidence", required=True, type=pathlib.Path)
    for stage, opt in STAGE_OPTION.items():
        for part in ("manifest", "inventory", "root"):
            ap.add_argument(
                f"--{opt}-{part}",
                required=True,
                type=pathlib.Path,
                dest=f"{stage}_{part}",
            )
    ap.add_argument("--data-dir", required=True, type=pathlib.Path)
    ap.add_argument("--acquisition-inventory", required=True, type=pathlib.Path)
    ap.add_argument("--preparation-record", required=True, type=pathlib.Path)
    ap.add_argument("--output-dir", required=True, type=pathlib.Path)
    return ap


def main(argv=None, expected=PUBLIC_EXPECTED):
    args = build_parser().parse_args(argv)
    out = args.output_dir.resolve()
    require(not out.exists(), "output directory already exists; refusing to overwrite")
    require(out.parent.is_dir(), "output parent directory does not exist")
    dirs = [args.data_dir] + [getattr(args, f"{s}_root") for s in STAGE_JOBS]
    for d in dirs:
        d = d.resolve()
        require(
            out != d and d not in out.parents,
            "output directory must not be inside an input directory",
        )
    result = analyze(args, expected)
    out.mkdir(exist_ok=False)
    for name, text in result["texts"].items():
        with open(out / name, "x", encoding="utf-8") as fh:
            fh.write(text)
    for name, blob in result["images"].items():
        with open(out / name, "xb") as fh:
            fh.write(blob)
    print(
        f"wrote {len(result['texts']) + len(result['images'])} files to the new "
        "output directory"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PilotInputError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        sys.exit(2)
