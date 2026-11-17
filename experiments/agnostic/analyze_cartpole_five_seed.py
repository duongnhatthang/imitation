"""Results adapter for the five-seed CartPole restriction experiment.

Analysis only. It reads one locally collected bundle, never launches jobs,
never rolls out environments, never loads checkpoints and never contacts a
remote machine. Every check runs before anything is written; outputs go into a
new directory only::

    python experiments/agnostic/analyze_cartpole_five_seed.py \
        --bundle-root BUNDLE --frozen-design frozen-design.json \
        --output-dir NEW_DIR [--artifact-manifest SHA256SUMS]

Experiment (producer protocol ``agnostic-classical-restriction/1``, module
``imitation.experiments.agnostic.task_pilot``): five predeclared paired fresh
training seeds 3100..3104. Each seed has one shared data job (the 1,000-label
chronological fixed-BC pool and the 1,000-label BC-iid stream) and six learner
runs: ``bc``, ``bc_iid`` and ``ftl``, each under ``identity`` (full
observation) or ``cart_position_angular_velocity_zero`` (learner-only mask).
Every run normalizes with the one pinned reference data job named in the frozen
design (expert 500, random 22.95). Pilot seed 300 and audit seed 301 are never
aggregated.

Filesystem contract of ``BUNDLE`` (relative paths; anything else is ignored,
except that job-like directory names outside the frozen grid are refused)::

    control/frozen-design.json       byte-identical copy of --frozen-design
    control/data-manifest.json       executed campaign manifest, 5 data jobs
    control/data-inventory.json      that campaign's controller inventory.json
    control/run-manifest.json        executed campaign manifest, 30 run jobs
    control/run-inventory.json       that campaign's controller inventory.json
    data-seed<S>/result.json         one per seed S
    data-seed<S>/fixed_bc_pool.npz
    data-seed<S>/bc_iid_stream.npz
    run-<M>-<R>-seed<S>/result.json  M in bc, bc_iid, ftl; R as above
    run-<M>-<R>-seed<S>/checkpoints/<M>_CartPole-v1_seed<S>/round-<NNNNN>.pt
                                     every evaluated policy (BC: round-01000)

Job ids are exactly the directory names. A manifest's absolute
``expected_result`` and ``--output-dir`` values are machine paths of the
executing host; they are only required to end in ``<job id>/result.json`` and
``<job id>``, and every file is read from ``BUNDLE/<job id>/`` instead. No
absolute path is ever opened, and none is written to an output.

``--artifact-manifest`` (optional) is a transfer-integrity list in
``sha256sum`` format (``<64 hex>  <path relative to BUNDLE>``). When given, every
listed file must exist with that digest and every bundle file this adapter
reads, checkpoints included, must be listed. The producer records no
checkpoint digests, so checkpoint bytes are verified only through this list;
without it, checkpoints are checked for presence and the final ones are hashed
for the report.

Validation. Global inputs (design pin and semantics, manifests, inventories,
artifact list, unexpected job directories) refuse the whole analysis with exit
code 2 and no output. Per-job problems (a failed, timed out, cancelled, missing
or rejected job, or a complete record that disagrees with the design, its
same-seed data, the common expert, task, reference, source, configuration,
schedule or accounting) are kept in an inventory. Aggregate curves, means,
standard errors and paired differences are produced only when all 5 data jobs
and all 30 runs are controller-complete and pass every check; otherwise only
the inventory report is written (exit code 3). Nothing is imputed, replaced or
interpolated.

Statistics. A seed-level value is the producer's stored metric for one
evaluation (normalized return of the mean over 100 episodes; disagreement and
cross-entropy over the states pooled from that evaluation's rollouts). Bands
and standard errors are across the five training seeds only:
``std(ddof=1) / sqrt(5)``. Episode-level variability is reported separately and
never used as training-seed uncertainty; episodes are never pooled across
seeds. Paired differences are formed per seed before averaging.
"""

import argparse
import dataclasses
import hashlib
import io
import json
import math
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

# ------------------------------------------------------------ protocol constants

PROTOCOL = "agnostic-classical-restriction/1"
DATA_SCHEMA = PROTOCOL + "/data"
RUN_SCHEMA = PROTOCOL + "/run"
PRODUCER_MODULE = "imitation.experiments.agnostic.task_pilot"
ENV_NAME = "CartPole-v1"
CAP = 500
SEEDS = (3100, 3101, 3102, 3103, 3104)
EXCLUDED_SEEDS = (300, 301)
MASK = "cart_position_angular_velocity_zero"
RESTRICTIONS = ("identity", MASK)
METHODS = ("ftl", "bc_iid", "bc")
ONLINE = ("ftl", "bc_iid")
N_LABELS = 1000
EVAL_INTERVAL = 10
EVAL_EPISODES = 100
BASELINE_EPISODES = 500
MAX_JOB_LIMIT_SECONDS = 7200.0
# restriction_pilot.eval_budgets plus the untrained round 0.
ONLINE_SCHEDULE = [0, 1] + list(range(EVAL_INTERVAL, N_LABELS + 1, EVAL_INTERVAL))
RESULT_FILE = "result.json"
DATA_FILES = {"pool": "fixed_bc_pool.npz", "stream": "bc_iid_stream.npz"}
SHARED_DATASET = {"ftl": None, "bc": "pool", "bc_iid": "stream"}
BETA = {"ftl": 0.0, "bc_iid": 1.0, "bc": None}
SUBSAMPLE = {"ftl": "uniform", "bc_iid": "uniform", "bc": "prefix"}
# The producer rounds normalized_return to six decimals.
NORM_TOL = 0.5e-6 + 1e-12

# task_pilot.SOURCE_FILES: the files whose digests every record carries.
SOURCE_FILES = (
    "experiments/agnostic/restriction.py",
    "experiments/agnostic/restriction_pilot.py",
    "experiments/agnostic/classical.py",
    "experiments/agnostic/pilot.py",
    "experiments/agnostic/rollouts.py",
    "experiments/ftrl/run_experiment.py",
    "experiments/ftrl/policy_utils.py",
    "experiments/ftrl/eval_utils.py",
    "experiments/ftrl/env_utils.py",
    "experiments/ftrl/env_baselines.py",
    "algorithms/ftrl.py",
    "algorithms/dagger.py",
    "algorithms/bc.py",
    "data/rollout.py",
    "util/util.py",
    "experiments/agnostic/tasks.py",
    "experiments/agnostic/task_experts.py",
    "experiments/agnostic/task_audit.py",
    "experiments/agnostic/task_pilot.py",
)
# Producer rule strings (restriction_pilot and classical at the frozen source).
ACTION_RULE = "PPO.predict(obs, deterministic=True) on CPU"
POOL_RULE = (
    "complete expert episodes until >= budget transitions; first budget kept in "
    "order"
)
STREAM_RULE = (
    "budget independent complete expert episodes; one uniformly selected "
    "pre-action state per episode"
)
BATCH_RULE = (
    "Online methods: before each round's data are loaded, batch and minibatch "
    "are min(bc_batch_size, accumulated labels). Fixed BC: min(bc_batch_size, "
    "dataset size). When the held-out split runs and leaves fewer training "
    "examples than that, the batch shrinks to the training split. Loaders "
    "shuffle and drop the incomplete final batch every epoch (drop_last). "
    "Records give the loader actually trained on (train_batch_size, "
    "train_examples, train_batches_per_epoch, train_drop_last)."
)
FIXED_BC_RULE = (
    "Offline fixed BC: one cold fit on all n_rounds pool labels with "
    "run_experiment._fit_fixed_bc (torch reseeded first) and one evaluation. "
    "The frozen policy is a flat reference across the online label axis, not "
    "a function of it; it never fits prefixes."
)
POLICY_RULE = (
    "linear: expert clone, frozen features, reinitialized action_net; "
    "restriction applied before the frozen features"
)
RESTRICTION_RECORDS = {
    "identity": {
        "restriction_id": "identity",
        "zeroed_indices": [],
        "obs_shape": None,
        "description": "Full observation, unchanged.",
    },
    MASK: {
        "restriction_id": MASK,
        "zeroed_indices": [0, 3],
        "obs_shape": [4],
        "description": (
            "CartPole cart position x := 0 and pole angular velocity "
            "theta_dot := 0; x_dot and theta are kept."
        ),
    },
}
# Settings shared by every run (restriction_pilot._experiment_config and the
# ExperimentConfig defaults it inherits); seed, algo and sampling vary.
EXPERIMENT_PINS = {
    "env_name": ENV_NAME,
    "policy_mode": "linear",
    "n_rounds": N_LABELS,
    "samples_per_round": 1,
    "l2_lambda": 0.0,
    "l2_decay": False,
    "warm_start": False,
    "beta_rampdown": 0,
    "bc_n_epochs": 20,
    "eval_interval": EVAL_INTERVAL,
    "learning_rate": 1e-3,
    "outer_early_stop": False,
    "inner_early_stop": True,
    "inner_early_stop_val_frac": 0.1,
    "inner_early_stop_min_val_size": 32,
    "trajectories_per_round": 1,
    "bc_batch_size": 32,
    "grow_batch_with_data": True,
}
# Keys allowed to differ between paired runs.
EXPERIMENT_VARYING = {"seed", "algo", "subsample_strategy"}
RUN_CONFIG_VARYING = {
    "seed",
    "method",
    "restriction_id",
    "restriction",
    "restriction_sha256",
    "distinct_observations",
    "beta",
    "eval_budgets",
    "fixed_bc",
    "experiment",
}
RUN_CONFIG_KEYS = RUN_CONFIG_VARYING | {
    "policy",
    "eval_episodes",
    "eval_deterministic",
    "batch_rule",
    "mixture",
    "task_sha256",
    "allow_relaxed_expert",
    "reference",
}
DATA_CONFIG_KEYS = {
    "seed",
    "budget",
    "baseline_episodes",
    "pool",
    "stream",
    "expert_action_rule",
    "allow_relaxed_expert",
}
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
        "task",
        "task_sha256",
    ]
)
MANIFEST_KEYS = {
    "version",
    "campaign_id",
    "source_root",
    "source_sha256",
    "stage_deadline",
    "overall_deadline",
    "jobs",
}
COMMON_FLAGS = {"--env", "--expert-dir", "--output-dir", "--deadline", "--job-limit-seconds"}
DATA_FLAGS = COMMON_FLAGS | {"--seed"}
RUN_FLAGS = COMMON_FLAGS | {
    "--data-dir",
    "--seed",
    "--method",
    "--restriction",
    "--reference-data-dir",
    "--expected-reference-sha256",
}
RUN_OPTIONAL_FLAGS = {"--expected-data-sha256"}
TERMINAL_STATES = {"complete", "failed", "timed_out", "cancelled", "interrupted"}
REJECTING_FAILURES = {"validation", "source_changed"}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_JOB_DIR_RE = re.compile(r"^(data-seed\d+|run-.+-seed\d+)$")

PLOT_METRICS = ("normalized_return", "disagreement_rate", "rollout_cross_entropy")
METRICS = PLOT_METRICS + ("expert_rollout_cross_entropy",)
METHOD_LABEL = {"ftl": "FTL", "bc_iid": "BC-iid", "bc": "Fixed offline BC"}
OBS_LABEL = {
    "identity": "full observation",
    MASK: "learner-only mask (cart position and pole angular velocity zeroed)",
}
# Method colors of the earlier pilot figures; validated for color-vision deficiency.
COLOR = {"ftl": "#1b7837", "bc": "#2166ac", "bc_iid": "#b2182b"}
STYLE = {"identity": "--", MASK: "-"}

FIG_CURVES = "cartpole_five_seed_learning_curves.png"
SUMMARY = "cartpole_five_seed_summary.json"
POINTS = "cartpole_five_seed_curve_points.json"
REPORT = "CARTPOLE_FIVE_SEED_RESULTS.md"
INVENTORY = "cartpole_five_seed_inventory.json"
INVENTORY_REPORT = "CARTPOLE_FIVE_SEED_INVENTORY.md"
SUMMARY_SCHEMA = "cartpole-five-seed-analysis/1"
POINTS_SCHEMA = "cartpole-five-seed-curve-points/1"
INVENTORY_SCHEMA = "cartpole-five-seed-inventory/1"

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_INCOMPLETE = 3


@dataclasses.dataclass(frozen=True)
class Expected:
    """Identities the inputs must match, fixed before any main outcome existed."""

    # SHA-256 of the frozen-design.json bytes received before launch.
    design_sha256: str = (
        "01133f875e863f4f436db7a01ea183eeb6b8cf4932d4c8766e78a89caab2e576"
    )
    # restriction_pilot.source_identity(task_pilot.SOURCE_FILES)["combined_sha256"]
    # of the frozen snapshot (campaign source hash 290d72a5...).
    scientific_source_combined_sha256: str = (
        "393544acc3482deca403885a1d48fbc560c567995ff14c375442566073e58ae4"
    )


PUBLIC_EXPECTED = Expected()


# ------------------------------------------------------------------ small helpers


class AnalysisInputError(RuntimeError):
    """An input is inconsistent with the frozen design or with another input."""


def require(cond, msg):
    if not cond:
        raise AnalysisInputError(msg)


def config_digest(value):
    """Producer canonical digest (``pilot.config_digest``, ``tasks._digest``)."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_sha(v):
    return isinstance(v, str) and bool(_SHA_RE.match(v))


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _json_equal(a, b):
    """Exact JSON equality that does not conflate bool, int and float."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return a == b


def _strict_object(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate JSON keys {sorted({k for k in keys if keys.count(k) > 1})}")
    return dict(pairs)


def _reject_constant(name):
    raise ValueError(f"non-finite JSON constant {name}")


def parse_json(raw):
    return json.loads(
        raw,
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )


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
    raise AnalysisInputError(f"pool episodes cover fewer than {budget} transitions")


def data_id(seed):
    return f"data-seed{seed}"


def run_id(method, restriction, seed):
    return f"run-{method}-{restriction}-seed{seed}"


DATA_IDS = tuple(data_id(s) for s in SEEDS)
RUN_KEYS = tuple((m, r, s) for s in SEEDS for m in METHODS for r in RESTRICTIONS)
RUN_IDS = tuple(run_id(*k) for k in RUN_KEYS)


def num(v, digits=3):
    """Prose number; a missing value stays visibly missing, never zero."""
    if v is None:
        return "not recorded"
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:,.{digits}f}"


class Bundle:
    """Reads bundle files once by relative path, remembering each SHA-256.

    Files are re-hashed before anything is written, so inputs that change
    during the analysis are refused rather than silently mixed.
    """

    def __init__(self, root):
        self.root = pathlib.Path(root)
        require(self.root.is_dir(), "bundle root is not a directory")
        self.files: Dict[str, str] = {}

    def path(self, rel):
        parts = pathlib.PurePosixPath(rel).parts
        require(
            rel
            and not pathlib.PurePosixPath(rel).is_absolute()
            and ".." not in parts,
            f"{rel!r}: not a path inside the bundle",
        )
        return self.root.joinpath(*parts)

    def exists(self, rel):
        return self.path(rel).is_file()

    def read(self, rel):
        path = self.path(rel)
        require(path.is_file(), f"{rel}: missing from the bundle")
        raw = path.read_bytes()
        self.files[rel] = hashlib.sha256(raw).hexdigest()
        return raw

    def json(self, rel):
        raw = self.read(rel)
        try:
            return parse_json(raw), self.files[rel]
        except ValueError as exc:
            raise AnalysisInputError(f"{rel}: not valid strict JSON: {exc}")

    def recheck(self):
        for rel, sha in self.files.items():
            path = self.path(rel)
            require(
                path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha,
                f"{rel}: input changed during the analysis",
            )


# --------------------------------------------------------------- frozen design


def check_design(raw, expected):
    """The explicit frozen design: exact pinned bytes and the planned semantics."""
    require(
        hashlib.sha256(raw).hexdigest() == expected.design_sha256,
        "frozen design is not the pinned pre-launch design",
    )
    try:
        d = parse_json(raw)
    except ValueError as exc:
        raise AnalysisInputError(f"frozen design: not valid strict JSON: {exc}")
    want = {
        "version": 1,
        "env_name": ENV_NAME,
        "seeds": list(SEEDS),
        "excluded_seeds": list(EXCLUDED_SEEDS),
        "restrictions": list(RESTRICTIONS),
        "labels": N_LABELS,
        "eval_episodes": EVAL_EPISODES,
        "episode_cap": CAP,
        "bc_epochs": EXPERIMENT_PINS["bc_n_epochs"],
        "learning_rate": EXPERIMENT_PINS["learning_rate"],
        "warm_start": False,
        "mixture": False,
        "bc_prefix": False,
    }
    for key, value in want.items():
        require(_json_equal(d.get(key), value), f"frozen design: {key} differs")
    require(sorted(d.get("methods") or []) == sorted(METHODS), "frozen design: methods")
    require(d.get("ftl_beta") == 0, "frozen design: FTL beta is not 0")
    require(_finite(d.get("entropy_weight")), "frozen design: entropy weight")
    for key in ("reference_data_sha256", "expert_sha256", "source_sha256", "task_sha256"):
        require(_is_sha(d.get(key)), f"frozen design: {key} is not a SHA-256")
    ref = d.get("reference")
    require(
        isinstance(ref, dict)
        and set(ref) == {"expert_return", "random_return", "expert_self_ce"}
        and all(_finite(v) for v in ref.values())
        and ref["expert_return"] == 500.0
        and ref["random_return"] == 22.95,
        "frozen design: normalization reference is not expert 500 / random 22.95",
    )
    ident = d.get("expert_identity")
    require(
        isinstance(ident, dict) and ident.get("expert_sha256") == d["expert_sha256"],
        "frozen design: expert identity does not name the expert checkpoint",
    )
    return d


# ------------------------------------------------------ manifests and inventories


def parse_argv(job_id, argv):
    """Producer command and ``--flag value`` pairs of a manifest argv."""
    require(
        isinstance(argv, list) and all(isinstance(a, str) for a in argv) and "-m" in argv,
        f"{job_id}: argv does not run a module",
    )
    i = argv.index("-m")
    require(argv[i + 1 : i + 2] == [PRODUCER_MODULE], f"{job_id}: argv module")
    command, rest = (argv[i + 2] if len(argv) > i + 2 else None), argv[i + 3 :]
    require(len(rest) % 2 == 0, f"{job_id}: argv flags are not flag/value pairs")
    flags = {}
    for flag, value in zip(rest[::2], rest[1::2]):
        require(
            flag.startswith("--") and flag not in flags,
            f"{job_id}: malformed or repeated argv flag {flag}",
        )
        flags[flag] = value
    return command, flags


def _tail(path_text, n):
    return pathlib.PurePosixPath(path_text).parts[-n:]


def check_manifest(name, m, design, job_ids):
    """One executed campaign manifest: exact job set and producer arguments."""
    require(
        isinstance(m, dict) and set(m) == MANIFEST_KEYS and m["version"] == 1,
        f"{name}: unexpected manifest format",
    )
    require(
        m["source_sha256"] == design["source_sha256"],
        f"{name}: campaign source differs from the frozen source",
    )
    require(isinstance(m["jobs"], list), f"{name}: jobs is not a list")
    ids = [j.get("id") if isinstance(j, dict) else None for j in m["jobs"]]
    require(len(ids) == len(set(ids)), f"{name}: duplicate job ids")
    require(
        sorted(ids) == sorted(job_ids),
        f"{name}: job set differs from the frozen five-seed grid",
    )
    jobs = {}
    for job in m["jobs"]:
        job_id = job["id"]
        command, flags = parse_argv(job_id, job.get("argv"))
        is_run = job_id.startswith("run-")
        required = RUN_FLAGS if is_run else DATA_FLAGS
        optional = RUN_OPTIONAL_FLAGS if is_run else set()
        require(
            command == ("run" if is_run else "data")
            and required <= set(flags) <= required | optional,
            f"{job_id}: producer command or flags differ from the protocol",
        )
        out = flags["--output-dir"]
        require(
            _tail(out, 1) == (job_id,)
            and isinstance(job.get("expected_result"), str)
            and _tail(job["expected_result"], 2) == (job_id, RESULT_FILE)
            and pathlib.PurePosixPath(job["expected_result"]).parent
            == pathlib.PurePosixPath(out),
            f"{job_id}: expected result is not <output dir>/{job_id}/{RESULT_FILE}",
        )
        require(
            flags["--env"] == ENV_NAME and flags["--deadline"] == m["stage_deadline"],
            f"{job_id}: environment or producer deadline differ",
        )
        try:
            limit = float(flags["--job-limit-seconds"])
        except ValueError:
            limit = float("nan")
        require(
            math.isfinite(limit) and 0 < limit <= MAX_JOB_LIMIT_SECONDS,
            f"{job_id}: job limit outside (0, {MAX_JOB_LIMIT_SECONDS:g}]",
        )
        fields = job.get("expected_fields") or {}
        require(
            fields.get("status") == "complete"
            and fields.get("protocol") == PROTOCOL
            and fields.get("schema") == (RUN_SCHEMA if is_run else DATA_SCHEMA)
            and fields.get("expert_sha256", design["expert_sha256"])
            == design["expert_sha256"]
            and fields.get("task_sha256", design["task_sha256"]) == design["task_sha256"],
            f"{job_id}: manifest expected fields differ from the frozen design",
        )
        jobs[job_id] = dict(job, flags=flags, job_limit_seconds=limit)
    return jobs


def check_manifest_pair(dm, rm, djobs, rjobs, design):
    """The run campaign consumes exactly the data campaign's outputs."""
    for key in ("source_root", "source_sha256"):
        require(dm[key] == rm[key], f"data and run manifests differ in {key}")
    require(dm["campaign_id"] != rm["campaign_id"], "manifests share a campaign id")
    flat = dict(djobs, **rjobs)
    require(len({j["cwd"] for j in flat.values()}) == 1, "jobs differ in cwd")
    require(len({j["argv"][0] for j in flat.values()}) == 1, "jobs differ in interpreter")
    require(
        len({j["flags"]["--expert-dir"] for j in flat.values()}) == 1,
        "jobs use different expert directories",
    )
    for s in SEEDS:
        require(djobs[data_id(s)]["flags"]["--seed"] == str(s), f"{data_id(s)}: seed")
    refs = set()
    for (m, r, s), job_id in zip(RUN_KEYS, RUN_IDS):
        f = rjobs[job_id]["flags"]
        require(
            f["--method"] == m and f["--restriction"] == r and f["--seed"] == str(s),
            f"{job_id}: argv method, restriction or seed differ from the job id",
        )
        require(
            f["--data-dir"] == djobs[data_id(s)]["flags"]["--output-dir"],
            f"{job_id}: data directory is not the same-seed data job's output",
        )
        require(
            f["--expected-reference-sha256"] == design["reference_data_sha256"],
            f"{job_id}: reference pin differs from the frozen design",
        )
        refs.add(f["--reference-data-dir"])
    require(len(refs) == 1, "runs use different reference data directories")


def check_inventory(name, inv, manifest, jobs):
    """A terminal controller inventory; one outcome per manifest job."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    require(isinstance(inv, dict) and inv.get("version") == 1, f"{name}: version")
    require(
        inv.get("campaign_id") == manifest["campaign_id"]
        and inv.get("manifest_sha256")
        == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        and inv.get("source_root") == manifest["source_root"]
        and inv.get("source_sha256") == manifest["source_sha256"],
        f"{name}: does not belong to its executed manifest",
    )
    runs = inv.get("controller_runs")
    require(isinstance(runs, list) and runs, f"{name}: no controller run recorded")
    require(
        all(not r.get("retry_requests") for r in runs),
        f"{name}: retry requests recorded; no retries were authorized",
    )
    require(
        runs[-1].get("ended_at") is not None and runs[-1].get("exit_code") is not None,
        f"{name}: the controller has not finished; not terminal",
    )
    require(
        isinstance(inv.get("jobs"), dict) and set(inv["jobs"]) == set(jobs),
        f"{name}: job set differs from its manifest",
    )
    outcomes = {}
    for job_id, mj in jobs.items():
        row = inv["jobs"][job_id]
        state = row.get("state")
        require(state in TERMINAL_STATES, f"{job_id}: job is not terminal ({state})")
        require(
            row.get("expected_result") == mj["expected_result"],
            f"{job_id}: inventory expected result differs from the manifest",
        )
        attempts = row.get("attempts")
        require(isinstance(attempts, list), f"{job_id}: malformed attempts")
        att = None
        if not attempts:
            require(state == "cancelled", f"{job_id}: {state} without an attempt")
        else:
            require(len(attempts) == 1, f"{job_id}: retried; no retries were authorized")
            att = attempts[0]
            require(
                att.get("number") == 1
                and att.get("infrastructure_retry") is False
                and att.get("archived_prior_result") is None,
                f"{job_id}: attempt is a retry or replaced a prior result",
            )
            require(
                att.get("argv") == mj["argv"] and att.get("cwd") == mj["cwd"],
                f"{job_id}: attempt ran a different command than the manifest",
            )
        if state == "complete":
            val = att.get("validation") or {}
            require(
                att.get("state") == "complete"
                and att.get("exit_code") == 0
                and _is_sha(row.get("result_sha256"))
                and val.get("ok") is True
                and val.get("sha256") == row["result_sha256"]
                and _finite(att.get("elapsed_seconds")),
                f"{job_id}: complete without a matching controller validation",
            )
        elapsed = (att or {}).get("elapsed_seconds")
        outcomes[job_id] = {
            "state": state,
            "state_reason": row.get("state_reason"),
            "result_sha256": row.get("result_sha256"),
            "attempts": len(attempts),
            "exit_code": (att or {}).get("exit_code"),
            "failure_kind": (att or {}).get("failure_kind"),
            "stop_reason": ((att or {}).get("termination") or {}).get("stop_reason"),
            "controller_elapsed_seconds": elapsed if _finite(elapsed) else None,
            "controller_started_at_utc": (att or {}).get("started_at"),
            "controller_ended_at_utc": (att or {}).get("ended_at"),
        }
    return outcomes


def campaign_span(inv):
    return {
        "campaign_id": inv["campaign_id"],
        "controller_runs": len(inv["controller_runs"]),
        "controller_started_at_utc": inv["controller_runs"][0].get("started_at"),
        "controller_ended_at_utc": inv["controller_runs"][-1].get("ended_at"),
        "controller_exit_code": inv["controller_runs"][-1].get("exit_code"),
    }


# --------------------------------------------------------------- artifact list


def parse_artifact_manifest(raw):
    """``sha256sum`` lines: digest, two characters, bundle-relative path."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise AnalysisInputError("artifact manifest is not UTF-8 text")
    entries = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        m = re.match(r"^([0-9a-f]{64}) [ *](.+)$", line)
        require(m is not None, f"artifact manifest line {n}: not '<sha256>  <path>'")
        rel = m.group(2)
        if rel.startswith("./"):
            rel = rel[2:]
        p = pathlib.PurePosixPath(rel)
        require(
            not p.is_absolute() and ".." not in p.parts and rel not in entries,
            f"artifact manifest line {n}: absolute, escaping or duplicate path",
        )
        entries[rel] = m.group(1)
    require(entries, "artifact manifest lists no files")
    return entries


def check_artifacts(bundle, entries, used):
    """Every listed file matches; every file the analysis used is listed."""
    for rel, sha in sorted(entries.items()):
        path = bundle.path(rel)
        require(path.is_file(), f"artifact manifest: {rel} is missing")
        actual = bundle.files.get(rel) or hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == sha, f"artifact manifest: {rel} digest differs")
    missing = sorted(set(used) - set(entries))
    require(
        not missing,
        f"artifact manifest does not cover {len(missing)} used file(s), "
        f"e.g. {missing[:3]}",
    )


# ------------------------------------------------------------ record validation


def check_common(job_id, rec, schema, mj, design, expected, ctx):
    """Identity every complete record shares with the design and the others."""
    require(
        rec.get("status") == "complete"
        and (rec.get("snapshot") or {}).get("final") is True
        and rec.get("error") is None,
        f"{job_id}: record is not a final complete record",
    )
    require(
        rec.get("protocol") == PROTOCOL and rec.get("schema") == schema,
        f"{job_id}: protocol or schema differ",
    )
    require(
        rec.get("env_name") == ENV_NAME and rec.get("episode_cap") == CAP,
        f"{job_id}: environment or step cap differ",
    )
    require(
        rec.get("expert_sha256") == design["expert_sha256"]
        and _json_equal(rec.get("preparation"), design["expert_identity"]),
        f"{job_id}: expert or preparation identity differ from the frozen design",
    )
    task = rec.get("task")
    require(
        isinstance(task, dict)
        and rec.get("task_sha256") == design["task_sha256"] == config_digest(task)
        and task.get("env_name") == ENV_NAME
        and task.get("episode_cap") == CAP,
        f"{job_id}: task description differs from the frozen task",
    )
    src = rec.get("source") or {}
    files = src.get("files")
    require(
        isinstance(files, dict)
        and sorted(files) == sorted(SOURCE_FILES)
        and src.get("combined_sha256")
        == config_digest(files)
        == expected.scientific_source_combined_sha256,
        f"{job_id}: scientific source differs from the frozen snapshot",
    )
    pkgs = rec.get("package_versions")
    require(isinstance(pkgs, dict) and pkgs, f"{job_id}: no package versions")
    ctx.setdefault("source", src)
    ctx.setdefault("package_versions", pkgs)
    require(
        _json_equal(src, ctx["source"]) and _json_equal(pkgs, ctx["package_versions"]),
        f"{job_id}: source or package versions differ between records",
    )
    for key, value in (mj.get("expected_fields") or {}).items():
        if key != "status":
            require(
                _json_equal(rec.get(key), value),
                f"{job_id}: {key} differs from the manifest's expected field",
            )


def check_data(job_id, seed, rec, sha, bundle, mj, design, expected, ctx):
    check_common(job_id, rec, DATA_SCHEMA, mj, design, expected, ctx)
    cfg = rec.get("config") or {}
    require(
        rec.get("seed") == seed
        and set(cfg) == DATA_CONFIG_KEYS
        and cfg["seed"] == seed
        and cfg["budget"] == N_LABELS
        and cfg["baseline_episodes"] == BASELINE_EPISODES
        and cfg["pool"] == POOL_RULE
        and cfg["stream"] == STREAM_RULE
        and cfg["expert_action_rule"] == ACTION_RULE
        and cfg["allow_relaxed_expert"] is False
        and rec.get("config_sha256") == config_digest(cfg),
        f"{job_id}: data configuration differs from the protocol",
    )
    b = rec.get("baselines") or {}
    require(
        all(_finite(b.get(k)) for k in ("expert_return", "random_return", "expert_self_ce")),
        f"{job_id}: data baselines are not finite",
    )
    costs = rec.get("baselines_costs") or {}
    require(
        costs.get("expert_episodes") == costs.get("random_episodes") == BASELINE_EPISODES
        and _is_int(costs.get("env_steps_counted"))
        and costs["env_steps_counted"]
        == costs.get("expert_env_steps", -1) + costs.get("random_env_steps", -1),
        f"{job_id}: baseline measurement record is inconsistent",
    )
    arrays = {}
    for name, file_name in DATA_FILES.items():
        entry = (rec.get("datasets") or {}).get(name) or {}
        rel = f"{job_id}/{file_name}"
        raw = bundle.read(rel)
        require(
            entry.get("file") == file_name
            and hashlib.sha256(raw).hexdigest() == entry.get("file_sha256"),
            f"{rel}: file digest differs from the data record",
        )
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
                obs, acts = saved["obs"], saved["acts"]
                step_index = saved["step_index"]
        except Exception as exc:  # malformed archive
            raise AnalysisInputError(f"{rel}: unreadable dataset ({type(exc).__name__})")
        require(
            pairs_sha256(obs, acts) == entry.get("pairs_sha256")
            and entry.get("n") == len(acts) == N_LABELS
            and obs.shape == (N_LABELS, 4)
            and np.all(np.isfinite(obs))
            and set(np.unique(acts).tolist()) <= {0, 1},
            f"{rel}: content differs from the data record",
        )
        stats = entry.get("stats") or {}
        require(
            stats.get("retained_labels") == N_LABELS
            and isinstance(stats.get("episode_lengths"), list)
            and all(_is_int(v) and 1 <= v <= CAP for v in stats["episode_lengths"])
            and _is_int(stats.get("env_steps"))
            and stats["env_steps"] == sum(stats["episode_lengths"])
            and stats.get("episodes") == len(stats["episode_lengths"]),
            f"{job_id}: {name} acquisition statistics are inconsistent",
        )
        if name == "stream":
            lengths = stats["episode_lengths"]
            require(
                len(lengths) == N_LABELS
                and all(0 <= k < n for k, n in zip(step_index.tolist(), lengths)),
                f"{job_id}: stream is not one state from each of {N_LABELS} episodes",
            )
        else:
            logical = pool_logical_cost(stats["episode_lengths"], N_LABELS)
            require(
                logical["episodes"] == len(stats["episode_lengths"])
                and stats.get("overshoot_transitions") == logical["overshoot_transitions"],
                f"{job_id}: pool does not stop at the first episode reaching "
                f"{N_LABELS} labels",
            )
        arrays[name] = (obs, acts)
    return {"record": rec, "sha256": sha, "arrays": arrays}


def check_reference(job_id, ref, design):
    """The run normalized with the pinned common reference, not its own data."""
    require(
        isinstance(ref, dict)
        and ref.get("source") == "pinned reference data job"
        and ref.get("data_result_sha256")
        == ref.get("expected_sha256")
        == design["reference_data_sha256"],
        f"{job_id}: normalization reference is not the pinned common reference",
    )
    require(
        _json_equal(ref.get("baselines"), design["reference"]),
        f"{job_id}: reference baselines differ from expert 500 / random 22.95",
    )


def check_fit(where, r, n, exp):
    """One recorded fit: the batch and held-out split the trainer actually used."""
    max_epochs = exp["bc_n_epochs"]
    stop, traj = r.get("inner_es_stop_epoch"), r.get("inner_es_val_nll_trajectory")
    fallback, best = r.get("inner_es_fallback"), r.get("inner_es_val_nll_best")
    require(
        _is_int(stop) and 1 <= stop <= max_epochs and isinstance(traj, list),
        f"{where}: malformed optimizer record",
    )
    n_val = int(exp["inner_early_stop_val_frac"] * n)
    if n_val < exp["inner_early_stop_min_val_size"]:
        require(
            fallback == "dataset_too_small" and stop == max_epochs and traj == []
            and best is None,
            f"{where}: {n} labels must use the fixed-budget fallback",
        )
        examples = n
    else:
        require(
            fallback is None and len(traj) == stop and all(_finite(v) for v in traj)
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
        f"{where}: loader is not batch min(32, labels) on the training split",
    )
    return {"labels": n, "stop_epoch": stop, "fallback": fallback}


def eval_point(job_id, method, seed, r, reference, bundle, checkpoints):
    """One saved evaluation: stored metrics as recorded, checked for consistency."""
    where = f"{job_id}: round {r.get('round')}"
    ret = r.get("episode_returns")
    require(
        isinstance(ret, list)
        and len(ret) == EVAL_EPISODES
        and all(_finite(v) and 0 < v <= CAP and float(v).is_integer() for v in ret),
        f"{where}: evaluation is not {EVAL_EPISODES} finite capped episode returns",
    )
    arr = np.asarray(ret, dtype=float)
    require(
        r.get("d_eval_size") == int(arr.sum()),
        f"{where}: pooled visited states differ from the episode lengths",
    )
    span = reference["expert_return"] - reference["random_return"]
    stored = r.get("normalized_return")
    require(
        _finite(stored)
        and abs(stored - (arr.mean() - reference["random_return"]) / span) <= NORM_TOL,
        f"{where}: normalized return is not (mean return - 22.95) / (500 - 22.95)",
    )
    # Normalized return may be negative (below random); only the formula binds it.
    require(
        all(_finite(r.get(k)) for k in METRICS)
        and 0 <= r["disagreement_rate"] <= 1
        and r["rollout_cross_entropy"] >= 0
        and r["expert_rollout_cross_entropy"] >= 0,
        f"{where}: non-finite or invalid metrics",
    )
    labels = int(r["n_observations"])
    ckpt = f"checkpoints/{method}_{ENV_NAME}_seed{seed}/round-{labels:05d}.pt"
    require(r.get("checkpoint") == ckpt, f"{where}: checkpoint is not {ckpt}")
    rel = f"{job_id}/{ckpt}"
    require(bundle.exists(rel), f"{where}: saved checkpoint is missing from the bundle")
    checkpoints.append(rel)
    episode_se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    return {
        "labels": labels,
        "round": int(r["round"]),
        **{k: float(r[k]) for k in METRICS},
        "mean_return": float(arr.mean()),
        "episodes_at_cap": int((arr == CAP).sum()),
        "within_eval_episode_se_return": episode_se,
        "within_eval_episode_se_normalized": episode_se / span,
        "eval_env_steps": int(r["d_eval_size"]),
    }


def check_accounting(job_id, rec, fits, evals, method):
    """Recompute the producer's accounting; a complete record must reconcile."""
    acc = rec.get("accounting") or {}
    done = acc.get("completed_records") or {}
    records = rec["records"]
    epochs = [f["stop_epoch"] for f in fits]
    require(
        done.get("training")
        == {
            "fits": len(fits),
            "epochs": sum(epochs),
            "early_stopped_fits": sum(e < EXPERIMENT_PINS["bc_n_epochs"] for e in epochs),
            "fixed_budget_fallback_fits": sum(f["fallback"] is not None for f in fits),
        },
        f"{job_id}: training counters differ from the rounds",
    )
    eval_steps = sum(p["eval_env_steps"] for p in evals)
    if method == "ftl":
        steps = int(records[-1].get("collection_steps", 0))
        episodes = sum(int(r.get("trajectories_collected_this_round", 0)) for r in records)
    else:
        steps = episodes = 0
    queries = ("expert_predict_calls", "expert_action_entries", "learner_predict_calls")
    ev, col = done.get("evaluation") or {}, done.get("collection") or {}
    require(
        ev.get("evaluations") == len(evals)
        and ev.get("episodes") == EVAL_EPISODES * len(evals)
        and ev.get("env_steps") == eval_steps
        and all(ev.get(k) == eval_steps for k in queries)
        and col.get("env_steps") == steps
        and col.get("episodes") == episodes
        and all(col.get(k) == steps for k in queries)
        and done.get("retained_labels") == N_LABELS,
        f"{job_id}: collection, evaluation or label counters differ from the rounds",
    )
    total = steps + eval_steps
    obs, totals = acc.get("observed") or {}, acc.get("totals") or {}
    require(
        obs.get("env_steps") == total
        and totals.get("env_steps") == total
        and totals.get("exact") is True
        and acc.get("reconciled") is True
        and all(totals.get(k) == total for k in queries)
        and (acc.get("in_flight") or {}).get("env_steps") == 0,
        f"{job_id}: observed steps do not reconcile with collection plus evaluation",
    )
    return {
        "observed_env_steps": total,
        "in_run_collection_env_steps": steps,
        "evaluation_env_steps": eval_steps,
        "fits": len(fits),
        "epochs": sum(epochs),
    }


def check_run(job_id, key, rec, sha, data, bundle, mj, design, expected, ctx):
    """A complete run linked to its same-seed data and the common reference."""
    method, restriction, seed = key
    check_common(job_id, rec, RUN_SCHEMA, mj, design, expected, ctx)
    require(
        rec.get("seed") == seed
        and rec.get("method") == method
        and rec.get("restriction_id") == restriction,
        f"{job_id}: seed, method or restriction differ from the job",
    )
    cfg = rec.get("config") or {}
    require(
        set(cfg) - {"normalization_reference"} == RUN_CONFIG_KEYS
        and rec.get("config_sha256") == config_digest(cfg),
        f"{job_id}: configuration keys or digest differ",
    )
    want_restriction = RESTRICTION_RECORDS[restriction]
    require(
        cfg["seed"] == seed
        and cfg["method"] == method
        and cfg["restriction_id"] == restriction
        and _json_equal(cfg["restriction"], want_restriction)
        and cfg["restriction_sha256"] == config_digest(want_restriction),
        f"{job_id}: observation restriction differs from the frozen mask",
    )
    schedule = [N_LABELS] if method == "bc" else ONLINE_SCHEDULE
    require(
        _json_equal(cfg["beta"], BETA[method])
        and cfg["mixture"] is False
        and cfg["eval_episodes"] == EVAL_EPISODES
        and cfg["eval_deterministic"] is True
        and cfg["eval_budgets"] == schedule
        and cfg["fixed_bc"] == (FIXED_BC_RULE if method == "bc" else None)
        and cfg["batch_rule"] == BATCH_RULE
        and cfg["policy"] == POLICY_RULE
        and cfg["task_sha256"] == design["task_sha256"]
        and cfg["allow_relaxed_expert"] is False,
        f"{job_id}: method semantics, schedule or evaluation settings differ",
    )
    dist = cfg["distinct_observations"]
    require(
        isinstance(dist, dict)
        and set(dist) == {"pool", "masked_pool"}
        and all(_is_int(v) and v > 1 for v in dist.values())
        and (restriction != "identity" or dist["pool"] == dist["masked_pool"]),
        f"{job_id}: distinct-observation record is malformed",
    )
    check_reference(job_id, cfg["reference"], design)
    if "normalization_reference" in cfg:
        normalization = cfg["normalization_reference"]
        require(
            isinstance(normalization, dict)
            and normalization.get("type") == "uniform_random"
            and normalization.get("lower_return") == cfg["reference"]["baselines"]["random_return"],
            f"{job_id}: normalization does not match the pinned random reference",
        )
    exp = cfg["experiment"]
    require(
        isinstance(exp, dict)
        and exp.get("algo") == method
        and exp.get("seed") == seed
        and exp.get("subsample_strategy") == SUBSAMPLE[method]
        and all(_json_equal(exp.get(k), v) for k, v in EXPERIMENT_PINS.items()),
        f"{job_id}: training settings differ from the protocol",
    )
    d = data["record"]
    rd = rec.get("data") or {}
    pool, stream = (d["datasets"][k]["pairs_sha256"] for k in ("pool", "stream"))
    require(
        rd.get("data_result_sha256") == data["sha256"]
        and rd.get("pool_pairs_sha256") == pool
        and rd.get("stream_pairs_sha256") == stream
        and _json_equal(rd.get("baselines"), d["baselines"]),
        f"{job_id}: not linked to the same-seed data job",
    )
    acc = rd.get("acceptance") or {}
    pin = mj["flags"].get("--expected-data-sha256")
    require(
        acc.get("rule") == "same_source"
        and acc.get("expected_data_sha256") == pin
        and pin in (None, data["sha256"])
        and acc.get("data_result_sha256") == data["sha256"]
        and acc.get("checks") in (ACCEPTANCE_CHECKS, sorted([*ACCEPTANCE_CHECKS, "baseline_reference"]))
        and acc.get("producer_protocol") == PROTOCOL
        and acc.get("producer_schema") == DATA_SCHEMA
        and _json_equal(acc.get("producer_source"), d["source"])
        and acc.get("current_source_combined_sha256")
        == expected.scientific_source_combined_sha256,
        f"{job_id}: data acceptance is not the same-source same-seed data job",
    )
    shared = (rec.get("accounting") or {}).get("shared_acquisition")
    kind = SHARED_DATASET[method]
    if kind is None:
        require(shared is None, f"{job_id}: FTL claims a shared acquisition")
    else:
        require(
            isinstance(shared, dict)
            and shared.get("dataset") == kind
            and shared.get("charged_to_this_run") is False
            and shared.get("pairs_sha256") == d["datasets"][kind]["pairs_sha256"],
            f"{job_id}: shared acquisition is not the same-seed {kind}",
        )
    require(
        (rec.get("deadline") or {}).get("job_limit_seconds") == mj["job_limit_seconds"],
        f"{job_id}: job limit differs from the manifest",
    )
    records = rec.get("records")
    require(isinstance(records, list) and records, f"{job_id}: no round records")
    ref = design["reference"]
    checkpoints: List[str] = []
    if method == "bc":
        require(
            len(records) == 1,
            f"{job_id}: fixed BC has {len(records)} records; it is one full-pool "
            "fit and one evaluation, prefix fits are not accepted",
        )
        r = records[0]
        entry = d["datasets"]["pool"]
        require(
            r.get("round") == 0
            and r.get("n_observations") == N_LABELS
            and r.get("reference") == "flat"
            and "prefix_pairs_sha256" not in r,
            f"{job_id}: record is not the single {N_LABELS}-label flat reference",
        )
        require(
            r.get("trained_pairs_sha256")
            == rec.get("trained_pairs_sha256")
            == entry["pairs_sha256"]
            == pairs_sha256(*data["arrays"]["pool"]),
            f"{job_id}: fixed BC was not trained on the entire same-seed pool",
        )
        require(
            r.get("logical") == pool_logical_cost(entry["stats"]["episode_lengths"], N_LABELS),
            f"{job_id}: logical acquisition differs from the pool episodes",
        )
        fits = [check_fit(f"{job_id}: fit", r, N_LABELS, exp)]
        points = [eval_point(job_id, method, seed, r, ref, bundle, checkpoints)]
    else:
        require(
            [r.get("round") for r in records] == list(range(N_LABELS + 1)),
            f"{job_id}: complete run does not record every round 0..{N_LABELS}",
        )
        entry = d["datasets"]["stream"]
        lengths = entry["stats"]["episode_lengths"]
        s_obs, s_acts = data["arrays"]["stream"]
        evaluated_at = set(ONLINE_SCHEDULE)
        points, fits, prev = [], [], 0
        for r in records:
            t = r["round"]
            where = f"{job_id}: round {t}"
            require(r.get("n_observations") == t, f"{where}: label count differs")
            evaluated = r.get("d_eval_size") is not None
            require(
                evaluated == (t in evaluated_at),
                f"{where}: evaluation does not follow the 0, 1, every 10, final schedule",
            )
            if evaluated:
                points.append(eval_point(job_id, method, seed, r, ref, bundle, checkpoints))
            else:
                require(
                    all(r.get(k) is None for k in METRICS + ("episode_returns",))
                    and "checkpoint" not in r,
                    f"{where}: non-evaluation round carries evaluation values",
                )
            if t == 0:
                require(
                    r.get("inner_es_stop_epoch") is None
                    and r.get("train_cross_entropy") is None,
                    f"{where}: the untrained round 0 records a fit",
                )
                continue
            require(_finite(r.get("train_cross_entropy")), f"{where}: no training loss")
            fits.append(check_fit(where, r, t, exp))
            steps = r.get("collection_steps")
            if method == "ftl":
                require(
                    _is_int(steps)
                    and steps > prev
                    and _is_int(r.get("trajectories_collected_this_round"))
                    and r["trajectories_collected_this_round"] >= 1,
                    f"{where}: FTL collected no new learner-state labels",
                )
                prev = steps
            else:
                require(
                    steps == entry["stats"]["env_steps"]
                    and r.get("trajectories_collected_this_round") == 0
                    and r.get("logical")
                    == {"labels": t, "episodes": t, "env_steps": sum(lengths[:t])},
                    f"{where}: BC-iid replay differs from the same-seed stream",
                )
                if evaluated:
                    require(
                        r.get("prefix_pairs_sha256") == pairs_sha256(s_obs[:t], s_acts[:t]),
                        f"{where}: BC-iid did not train on the stream's first {t} labels",
                    )
        trained = rec.get("trained_pairs_sha256")
        require(_is_sha(trained), f"{job_id}: no trained-pairs digest")
        if method == "bc_iid":
            require(
                trained == entry["pairs_sha256"],
                f"{job_id}: BC-iid trained on data other than the same-seed stream",
            )
    require(
        [p["labels"] for p in points] == schedule,
        f"{job_id}: saved evaluations differ from the protocol schedule",
    )
    costs = check_accounting(job_id, rec, fits, points, method)
    final_ckpt = checkpoints[-1]
    return {
        "points": points,
        "costs": costs,
        "checkpoints": checkpoints,
        "final_checkpoint": final_ckpt[len(job_id) + 1 :],
        "final_checkpoint_sha256": hashlib.sha256(
            bundle.path(final_ckpt).read_bytes()
        ).hexdigest(),
        "trained_pairs_sha256": rec.get("trained_pairs_sha256"),
        "config_sha256": rec["config_sha256"],
        "fixed_budget_fallback_fits": sum(f["fallback"] is not None for f in fits),
        "early_stopped_fits": sum(
            f["stop_epoch"] < EXPERIMENT_PINS["bc_n_epochs"] for f in fits
        ),
        "fits": len(fits),
        "wall_seconds_recorded": (rec.get("accounting") or {}).get("wall_seconds"),
    }


def check_paired_configs(runs):
    """Paired conditions differ only in seed, method, mask and their consequences."""

    def strip(rec):
        cfg = {k: v for k, v in rec["config"].items() if k not in RUN_CONFIG_VARYING}
        exp = {
            k: v for k, v in rec["config"]["experiment"].items()
            if k not in EXPERIMENT_VARYING
        }
        return cfg, exp

    ref = strip(next(iter(runs.values())))
    for job_id, rec in runs.items():
        require(
            _json_equal(strip(rec), ref),
            f"{job_id}: configuration differs from its paired conditions",
        )
    for s in SEEDS:
        pools = {runs[run_id(m, r, s)]["config"]["distinct_observations"]["pool"]
                 for m in METHODS for r in RESTRICTIONS}
        require(len(pools) == 1, f"seed {s}: runs saw different pools")
        for m in METHODS:
            a, b = (runs[run_id(m, r, s)]["config"]["experiment"] for r in RESTRICTIONS)
            require(
                _json_equal(a, b),
                f"{run_id(m, MASK, s)}: full and masked runs differ beyond the mask",
            )


# ------------------------------------------------------------------- inventory


def read_job(bundle, job_id, outcome):
    """The job's record, or None with a visible reason when there is none."""
    rel = f"{job_id}/{RESULT_FILE}"
    state = outcome["state"]
    if not bundle.exists(rel):
        require(state != "complete", f"{job_id}: complete job without its result")
        return None, None, "missing" if outcome["attempts"] else "none, never launched"
    require(outcome["attempts"], f"{job_id}: a result exists for a job never launched")
    try:
        rec, sha = bundle.json(rel)
    except AnalysisInputError:
        require(state != "complete", f"{job_id}: complete job with unreadable result")
        return None, bundle.files[rel], "unreadable"
    require(isinstance(rec, dict), f"{job_id}: result is not an object")
    if state == "complete":
        require(
            sha == outcome["result_sha256"],
            f"{job_id}: result bytes differ from the controller's validated hash",
        )
    return rec, sha, "read"


def inventory_row(job_id, outcome, rec, sha, availability):
    records = (rec or {}).get("records")
    error = (rec or {}).get("error")
    return {
        "job": job_id,
        "controller_state": outcome["state"],
        "state_reason": outcome["state_reason"],
        "attempts": outcome["attempts"],
        "exit_code": outcome["exit_code"],
        "failure_kind": outcome["failure_kind"],
        "stop_reason": outcome["stop_reason"],
        "controller_elapsed_seconds": outcome["controller_elapsed_seconds"],
        "result": availability,
        "result_sha256": sha,
        "record_status": None if rec is None else rec.get("status"),
        "snapshot_final": None if rec is None else (rec.get("snapshot") or {}).get("final"),
        "last_recorded_round": (
            records[-1].get("round")
            if isinstance(records, list) and records and isinstance(records[-1], dict)
            else None
        ),
        # Error messages may hold machine paths; only the type is kept.
        "error_type": error.get("type") if isinstance(error, dict) else None,
        "usable": False,
        "problems": [],
    }


def known_sum(values):
    """Sum of known values; a missing value is never counted as zero."""
    known = [v for v in values if v is not None]
    return {
        "sum": sum(known) if len(known) == len(values) else None,
        "known_subtotal": sum(known) if known else None,
        "known": len(known),
        "missing": len(values) - len(known),
    }


# ------------------------------------------------------------------ statistics


def seed_stats(values):
    """Mean and training-seed standard error, sample std(ddof=1) / sqrt(n).

    ``values`` holds one stored metric per training seed, in seed order. Any
    within-evaluation (episode) spread is deliberately not an input.
    """
    a = np.asarray(values, dtype=float)
    require(a.ndim == 1 and a.size == len(SEEDS), "one value per training seed needed")
    require(bool(np.all(np.isfinite(a))), "non-finite seed-level value")
    sd = float(a.std(ddof=1))
    return {
        "mean": float(a.mean()),
        "sd": sd,
        "se": sd / math.sqrt(a.size),
        "n_seeds": int(a.size),
        "per_seed": [float(v) for v in a],
    }


def aggregate(curves, bc):
    """Across-seed means and SEs: online per budget, fixed BC as one flat value.

    ``curves[(method, restriction)][seed]`` is the list of evaluation points of
    an online run; ``bc[restriction][seed]`` is the single fixed-BC point.
    """
    online = {}
    for m in ONLINE:
        for r in RESTRICTIONS:
            per_seed = [curves[(m, r)][s] for s in SEEDS]
            budgets = [p["labels"] for p in per_seed[0]]
            require(
                all([p["labels"] for p in pts] == budgets for pts in per_seed),
                f"{m}/{r}: seeds differ in evaluation budgets",
            )
            online[(m, r)] = [
                {
                    "labels": n,
                    **{
                        k: seed_stats([pts[i][k] for pts in per_seed])
                        for k in METRICS + ("mean_return",)
                    },
                }
                for i, n in enumerate(budgets)
            ]
    flat = {
        r: {
            "labels": N_LABELS,
            **{
                k: seed_stats([bc[r][s][k] for s in SEEDS])
                for k in METRICS + ("mean_return",)
            },
        }
        for r in RESTRICTIONS
    }
    return online, flat


def paired_differences(curves, bc):
    """Per-seed differences first, then mean and training-seed SE.

    FTL minus BC-iid at every shared budget; FTL minus the same-seed fixed-BC
    value (flat, 1,000 labels) at every FTL budget.
    """
    out = {}
    for r in RESTRICTIONS:
        ftl = [curves[("ftl", r)][s] for s in SEEDS]
        iid = [curves[("bc_iid", r)][s] for s in SEEDS]
        rows_iid, rows_bc = [], []
        for i, p in enumerate(ftl[0]):
            n = p["labels"]
            require(all(pts[i]["labels"] == n for pts in ftl + iid), "budget mismatch")
            rows_iid.append(
                {
                    "labels": n,
                    **{
                        k: seed_stats([f[i][k] - b[i][k] for f, b in zip(ftl, iid)])
                        for k in METRICS
                    },
                }
            )
            rows_bc.append(
                {
                    "labels": n,
                    **{
                        k: seed_stats(
                            [ftl[j][i][k] - bc[r][s][k] for j, s in enumerate(SEEDS)]
                        )
                        for k in METRICS
                    },
                }
            )
        out[r] = {"ftl_minus_bc_iid": rows_iid, "ftl_minus_fixed_bc": rows_bc}
    return out


# --------------------------------------------------------------------- figure

PANELS = (
    (
        "normalized_return",
        "Normalized return",
        r"$\widehat{J}_{\mathrm{norm}}(\pi_n)="
        r"(\widehat{J}(\pi_n)-J_{\mathrm{rand}})/(J_E-J_{\mathrm{rand}})$",
    ),
    (
        "disagreement_rate",
        "Disagreement with the expert, learner's own states",
        r"$\mathbb{E}_{s\sim\hat d_{\pi_n}}[\mathbf{1}\{a_n(s)\ne a_E(s)\}]$",
    ),
    (
        "rollout_cross_entropy",
        "Learner cross-entropy of the expert action, learner's own states",
        r"$\mathbb{E}_{s\sim\hat d_{\pi_n}}[-\log\pi_n(a_E(s)\mid s)]$ (nats)",
    ),
)


def curves_figure(online, flat):
    """Three panels: mean over training seeds with a +/- 1 training-seed SE band."""
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.4), sharex=True)
    for ax, (key, title, ylabel) in zip(axes, PANELS):
        for r in RESTRICTIONS:
            row = flat[r][key]
            lo, hi = row["mean"] - row["se"], row["mean"] + row["se"]
            ax.fill_between(
                [0, N_LABELS], [lo, lo], [hi, hi], color=COLOR["bc"],
                alpha=0.10 if r == "identity" else 0.16, lw=0, zorder=1,
            )
            ax.plot(
                [0, N_LABELS], [row["mean"]] * 2, ls=STYLE[r], color=COLOR["bc"],
                lw=1.8, zorder=2,
            )
        for m in ONLINE:
            for r in RESTRICTIONS:
                rows = online[(m, r)]
                xs = np.array([p["labels"] for p in rows], dtype=float)
                mean = np.array([p[key]["mean"] for p in rows])
                se = np.array([p[key]["se"] for p in rows])
                ax.fill_between(
                    xs, mean - se, mean + se, color=COLOR[m],
                    alpha=0.12 if r == "identity" else 0.20, lw=0, zorder=3,
                )
                ax.plot(xs, mean, ls=STYLE[r], color=COLOR[m], lw=1.8, zorder=4)
        if key == "normalized_return":
            ax.axhline(1.0, color="#555555", ls=":", lw=1.2, zorder=0)
            set_normalized_return_axis(ax)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"$n$: expert labels used for training")
        ax.set_xlim(-15, N_LABELS + 15)
        ax.grid(alpha=0.3)
    handles = [
        Line2D([], [], color=COLOR["ftl"], lw=2.5, label="FTL"),
        Line2D([], [], color=COLOR["bc_iid"], lw=2.5, label="BC-iid"),
        Line2D(
            [], [], color=COLOR["bc"], lw=2.5,
            label=f"Fixed offline BC, {N_LABELS:,} training labels\n"
            "(one fit per seed, drawn flat over $n$)",
        ),
        Line2D([], [], color="#444444", ls=STYLE["identity"], lw=1.6,
               label="Full observation (dashed)"),
        Line2D([], [], color="#444444", ls=STYLE[MASK], lw=1.6,
               label="Learner-only mask: x and $\\dot\\theta$ zeroed (solid)"),
        Patch(facecolor="#888888", alpha=0.3,
              label=f"$\\pm 1$ SE across {len(SEEDS)} training seeds"),
        Line2D([], [], color="#555555", ls=":", lw=1.2,
               label="Expert reference: normalized return 1"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.105))
    fig.suptitle(
        f"CartPole-v1, {len(SEEDS)} paired training seeds "
        f"({SEEDS[0]} to {SEEDS[-1]}): mean and training-seed standard error",
        fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, 0.008,
        r"$\hat d_{\pi_n}$: empirical distribution of all pre-action states pooled "
        r"over 100 deterministic rollouts of the evaluated learner $\pi_n$ itself "
        r"(its own on-policy occupancy), not expert-state test data. Each method, "
        r"mask, seed and $n$ has its own $\hat d_{\pi_n}$." "\n"
        r"$a_n, a_E$: deterministic learner and expert actions; $\pi_n$ includes "
        r"its observation mask. $\widehat{J}$: mean return over the 100 episodes; "
        r"$J_E=500$, $J_{\mathrm{rand}}=22.95$ (common pinned reference)." "\n"
        r"Lines: mean over training seeds of each seed's stored value; bands: "
        r"sample s.d. (ddof 1) / $\sqrt{5}$ across seeds, not episode spread. "
        r"Points at $n$ = 0, 1, 10, 20, ..., 1000, joined by straight segments; "
        r"no smoothing. Pilot seeds 300 and 301 excluded.",
        ha="center", va="bottom", fontsize=8.5, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.23, 1, 0.95))
    return fig


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, metadata={"Software": None})
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------- report


TABLE_OBS = {"identity": "full", MASK: "masked (x, theta_dot zeroed)"}


def _cond(m, r):
    return f"{METHOD_LABEL[m]}, {TABLE_OBS[r]}"


def _pm(stat, digits=3):
    return f"{stat['mean']:.{digits}f} ± {stat['se']:.{digits}f}"


def final_rows(online, flat):
    rows = {}
    for m in METHODS:
        for r in RESTRICTIONS:
            row = flat[r] if m == "bc" else online[(m, r)][-1]
            require(row["labels"] == N_LABELS, "final point is not at 1,000 labels")
            rows[f"{m}/{r}"] = {k: row[k] for k in METRICS + ("mean_return",)}
    return rows


LIMITATIONS = [
    "Five training seeds give descriptive standard-error bands. They do not "
    "establish an algorithmic advantage, and nothing here measures an agnostic "
    "error floor or return ceiling.",
    "The user accepted the original training budget with fitting and return "
    "convergence unresolved. Early stops and epoch-cap fits are descriptive "
    "only, not convergence evidence.",
    "Uncertainty is conditional on the one frozen expert and the one pinned "
    "normalization reference; reference uncertainty is not propagated.",
    "Disagreement and cross-entropy are measured on each learner's own rollout "
    "states, so conditions are compared on different state distributions.",
    "The BC entropy weight (0.001 in the frozen design) is an inherited trainer "
    "default that the producer does not record; it is not verified here.",
    "The producer records no checkpoint digests. Checkpoint bytes are verified "
    "only through the optional transfer artifact list; final-checkpoint digests "
    "are reported as observed in the bundle.",
]


def report(s):
    L = []
    w = L.append
    w("# CartPole-v1 five-seed restriction results")
    w("")
    w(f"Status: **complete grid**, {s['n_seeds']} of {s['n_seeds']} predeclared "
      f"training seeds ({', '.join(map(str, s['seeds']))}), all 5 data jobs and "
      "all 30 runs controller-complete and validated. Pilot seed 300 and audit "
      "seed 301 are excluded from every aggregate.")
    w("")
    w(f"Producer protocol `{PROTOCOL}`. Original normalization for every run: "
      f"expert return {s['reference']['expert_return']}, random return "
      f"{s['reference']['random_return']}, from the pinned reference data record "
      f"`{s['identities']['reference_data_sha256']}`. The expert reference is "
      "normalized return 1.")
    w("")
    w(f"![learning curves]({FIG_CURVES})")
    w("")
    w("Figure: lines are means across the five training seeds of each seed's "
      "stored metric; shading is plus or minus one standard error across "
      "training seeds, sample s.d. (ddof 1) / sqrt(5). Disagreement and "
      "cross-entropy are averages over the empirical pooled states visited by "
      "each evaluated learner in its own 100 deterministic rollouts, not over "
      "expert-state test data. Fixed offline BC is one fit on all 1,000 pool "
      "labels per seed, drawn as a flat mean and flat band over the label axis.")
    w("")
    w("## Final outcomes at 1,000 labels (mean ± training-seed SE, n = 5)")
    w("")
    w("| Condition | Normalized return | Mean return | Disagreement | "
      "Learner CE (nats) | Expert CE, same states (nats) |")
    w("|---|---|---|---|---|---|")
    for key, row in s["final"].items():
        m, r = key.split("/")
        w(f"| {_cond(m, r)} | {_pm(row['normalized_return'])} | "
          f"{_pm(row['mean_return'], 1)} | {_pm(row['disagreement_rate'])} | "
          f"{_pm(row['rollout_cross_entropy'])} | "
          f"{_pm(row['expert_rollout_cross_entropy'])} |")
    w("")
    w("## Paired differences at 1,000 labels")
    w("")
    w("Each difference is formed within a training seed (same data job, same "
      "mask) before averaging, so shared seed effects cancel; SE is the "
      "training-seed SE of the five differences. These terminal 1,000-label "
      "paired contrasts are the predeclared primary endpoint; the learning "
      "curves are descriptive.")
    w("")
    w("| Mask | Contrast | Normalized return | Disagreement | Learner CE (nats) |")
    w("|---|---|---|---|---|")
    for r in RESTRICTIONS:
        for name, label in (("ftl_minus_bc_iid", "FTL minus BC-iid"),
                            ("ftl_minus_fixed_bc", "FTL minus fixed offline BC")):
            row = s["paired_final"][r][name]
            w(f"| {TABLE_OBS[r]} | {label} | {_pm(row['normalized_return'])} | "
              f"{_pm(row['disagreement_rate'])} | {_pm(row['rollout_cross_entropy'])} |")
    w("")
    w("All budgets are in the curve-point file.")
    w("")
    w("## Per-seed final outcomes")
    w("")
    w("Within-evaluation episode SE (normalized, 100 episodes) describes one "
      "evaluation's sampling noise for one trained policy. It is not "
      "training-seed uncertainty and is not used in any band.")
    w("")
    w("| Seed | Condition | Normalized return | Episode SE (normalized) | "
      "Episodes at cap | Disagreement | Learner CE |")
    w("|---|---|---|---|---|---|---|")
    for row in s["per_seed_final"]:
        w(f"| {row['seed']} | {_cond(row['method'], row['restriction'])} | "
          f"{row['normalized_return']:.3f} | "
          f"{row['within_eval_episode_se_normalized']:.3f} | "
          f"{row['episodes_at_cap']} | {row['disagreement_rate']:.4f} | "
          f"{row['rollout_cross_entropy']:.4f} |")
    w("")
    c = s["costs"]
    w("## Costs")
    w("")
    w("New execution in this experiment (controller-measured worker wall time; "
      "not CPU hours or money):")
    w("")
    w(f"- 30 learner runs: {num(c['new_runs']['controller_elapsed_seconds']['sum'], 0)} s; "
      f"run environment steps {num(c['new_runs']['observed_env_steps']['sum'])}, "
      f"of which evaluation {num(c['new_runs']['evaluation_env_steps']['sum'])} and FTL "
      f"in-run collection {num(c['new_runs']['ftl_in_run_collection_env_steps']['sum'])}; "
      f"{num(c['new_runs']['fits']['sum'])} fits, {num(c['new_runs']['epochs']['sum'])} epochs.")
    w(f"- 5 per-seed data jobs: {num(c['new_data']['controller_elapsed_seconds']['sum'], 0)} s; "
      f"pool steps {num(c['new_data']['pool_env_steps']['sum'])}, stream steps "
      f"{num(c['new_data']['stream_env_steps']['sum'])}, own baseline steps "
      f"{num(c['new_data']['baseline_env_steps']['sum'])} (measured but not used "
      "for normalization).")
    w("")
    w("Reused, not new work and not charged to any run: the qualified expert "
      "(preparation cost not in this bundle: not recorded) and the pinned "
      f"reference acquisition ({num(c['reused_reference'].get('env_steps_counted'))} "
      "baseline environment steps as recorded by the reference).")
    w("")
    w("## Inventory")
    w("")
    w("| Job | Controller state | Record status | Usable |")
    w("|---|---|---|---|")
    for row in s["inventory"]:
        w(f"| {row['job']} | {row['controller_state']} | {row['record_status']} | "
          f"{'yes' if row['usable'] else 'no'} |")
    w("")
    w("## Reproducibility identities")
    w("")
    for k, v in s["identities"].items():
        if isinstance(v, (str, int)) or v is None:
            w(f"- {k}: `{v}`")
    w("")
    w("Per-seed dataset digests, run configuration digests and final checkpoint "
      "digests are in the summary JSON.")
    w("")
    w("## Limitations")
    w("")
    for item in LIMITATIONS:
        w(f"- {item}")
    w("")
    return "\n".join(L)


def inventory_report(inv):
    L = [
        "# CartPole-v1 five-seed restriction: incomplete or invalid grid",
        "",
        f"Status: **{inv['status']}**. {inv['usable_data_jobs']} of 5 data jobs and "
        f"{inv['usable_runs']} of 30 runs are usable. No aggregate curve, mean, "
        "standard error or paired difference is published from an incomplete "
        "five-seed grid, and no seed is imputed or replaced.",
        "",
        "| Job | Controller state | Record status | Last round | Usable | Problems |",
        "|---|---|---|---|---|---|",
    ]
    for row in inv["jobs"]:
        L.append(
            f"| {row['job']} | {row['controller_state']} | {row['record_status']} | "
            f"{row['last_recorded_round']} | {'yes' if row['usable'] else 'no'} | "
            f"{'; '.join(row['problems']) or 'none'} |"
        )
    L.append("")
    L.append(f"Controller wall time of launched attempts: "
             f"{num(inv['controller_elapsed_seconds']['known_subtotal'], 0)} s known "
             f"over {inv['controller_elapsed_seconds']['known']} attempt(s); "
             f"{inv['controller_elapsed_seconds']['missing']} not recorded "
             "(not counted as zero).")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------- privacy


def forbidden_tokens(blobs, paths):
    tokens = {"/home/", "/Users/", "/opt/", "/private/", "/tmp/", "miniconda", chr(0x2014)}
    for p in paths:
        tokens.add(str(pathlib.Path(p).absolute()))
        tokens.add(str(pathlib.Path(p).resolve()))

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

    for blob in blobs:
        walk(blob)
    return {t for t in tokens if t}


def write_outputs(out_dir, outputs, tokens):
    texts = {}
    for name, value in outputs.items():
        data = value if isinstance(value, bytes) else value.encode("utf-8")
        for t in tokens:
            require(
                t.encode("utf-8") not in data,
                f"{name}: would contain a private machine detail; refusing to write",
            )
        texts[name] = data
    out_dir.mkdir(parents=True)
    for name, data in texts.items():
        (out_dir / name).write_bytes(data)


def dumps(obj):
    return json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"


# ------------------------------------------------------------------------ main


def analyze(args, expected=PUBLIC_EXPECTED):
    out_dir = pathlib.Path(args.output_dir)
    require(not out_dir.exists(), "output directory already exists; give a new one")
    design_raw = pathlib.Path(args.frozen_design).read_bytes()
    design = check_design(design_raw, expected)
    bundle = Bundle(args.bundle_root)
    require(
        bundle.read("control/frozen-design.json") == design_raw,
        "bundle frozen design differs from the explicit frozen design",
    )
    expected_dirs = set(DATA_IDS) | set(RUN_IDS)
    strays = sorted(
        p.name for p in bundle.root.iterdir()
        if p.is_dir() and _JOB_DIR_RE.match(p.name) and p.name not in expected_dirs
    )
    require(not strays, f"job directories outside the frozen grid: {strays}")

    dm, _ = bundle.json("control/data-manifest.json")
    rm, _ = bundle.json("control/run-manifest.json")
    djobs = check_manifest("data manifest", dm, design, DATA_IDS)
    rjobs = check_manifest("run manifest", rm, design, RUN_IDS)
    check_manifest_pair(dm, rm, djobs, rjobs, design)
    dinv, _ = bundle.json("control/data-inventory.json")
    rinv, _ = bundle.json("control/run-inventory.json")
    outcomes = dict(
        check_inventory("data inventory", dinv, dm, djobs),
        **check_inventory("run inventory", rinv, rm, rjobs),
    )
    jobs = dict(djobs, **rjobs)

    ctx: Dict[str, Any] = {}
    rows: Dict[str, Dict[str, Any]] = {}
    records: Dict[str, Any] = {}
    for job_id in DATA_IDS + RUN_IDS:
        rec, sha, availability = read_job(bundle, job_id, outcomes[job_id])
        rows[job_id] = inventory_row(job_id, outcomes[job_id], rec, sha, availability)
        records[job_id] = (rec, sha)

    def problem(job_id, text):
        rows[job_id]["problems"].append(text)

    def usable_state(job_id):
        o, (rec, _) = outcomes[job_id], records[job_id]
        if o["state"] != "complete":
            problem(job_id, f"controller state {o['state']}"
                    + (f" ({o['state_reason']})" if o["state_reason"] else ""))
            if o["failure_kind"] in REJECTING_FAILURES:
                problem(job_id, f"result rejected by the controller ({o['failure_kind']})")
            return False
        return rec is not None

    data: Dict[int, Any] = {}
    for s in SEEDS:
        job_id = data_id(s)
        if not usable_state(job_id):
            continue
        rec, sha = records[job_id]
        try:
            data[s] = check_data(job_id, s, rec, sha, bundle, jobs[job_id], design,
                                 expected, ctx)
            rows[job_id]["usable"] = True
        except AnalysisInputError as exc:
            problem(job_id, str(exc))

    runs: Dict[Tuple[str, str, int], Any] = {}
    for key, job_id in zip(RUN_KEYS, RUN_IDS):
        if not usable_state(job_id):
            continue
        if key[2] not in data:
            problem(job_id, f"same-seed data job {data_id(key[2])} is not usable")
            continue
        rec, sha = records[job_id]
        try:
            runs[key] = check_run(job_id, key, rec, sha, data[key[2]], bundle,
                                  jobs[job_id], design, expected, ctx)
            rows[job_id]["usable"] = True
        except AnalysisInputError as exc:
            problem(job_id, str(exc))

    if len(data) == len(SEEDS) and len(runs) == len(RUN_KEYS):
        try:
            check_paired_configs({run_id(*k): records[run_id(*k)][0] for k in RUN_KEYS})
        except AnalysisInputError as exc:
            for job_id in RUN_IDS:
                problem(job_id, f"paired configuration check failed: {exc}")
                rows[job_id]["usable"] = False
            runs = {}

    used = set(bundle.files)
    for res in runs.values():
        used.update(res["checkpoints"])
    if args.artifact_manifest:
        entries = parse_artifact_manifest(pathlib.Path(args.artifact_manifest).read_bytes())
        check_artifacts(bundle, entries, used)

    complete = len(data) == len(SEEDS) and len(runs) == len(RUN_KEYS)
    tokens = forbidden_tokens(
        [dm, rm, dinv, rinv] + [r for r, _ in records.values() if r is not None],
        [args.bundle_root, args.frozen_design, args.output_dir]
        + ([args.artifact_manifest] if args.artifact_manifest else []),
    )
    inv_rows = [rows[j] for j in DATA_IDS + RUN_IDS]
    bundle.recheck()

    if not complete:
        inv = {
            "schema": INVENTORY_SCHEMA,
            "status": "incomplete_grid",
            "protocol": PROTOCOL,
            "seeds": list(SEEDS),
            "excluded_seeds": list(EXCLUDED_SEEDS),
            "usable_data_jobs": len(data),
            "usable_runs": len(runs),
            "aggregates": None,
            "aggregates_note": "withheld: the five-seed grid is not complete and valid",
            "campaigns": {"data": campaign_span(dinv), "runs": campaign_span(rinv)},
            "controller_elapsed_seconds": known_sum(
                [rows[j]["controller_elapsed_seconds"] for j in rows if rows[j]["attempts"]]
            ),
            "input_sha256": dict(sorted(bundle.files.items())),
            "jobs": inv_rows,
        }
        write_outputs(out_dir, {INVENTORY: dumps(inv), INVENTORY_REPORT: inventory_report(inv)},
                      tokens)
        return EXIT_INCOMPLETE

    curves = {(m, r): {s: runs[(m, r, s)]["points"] for s in SEEDS}
              for m in ONLINE for r in RESTRICTIONS}
    bc = {r: {s: runs[("bc", r, s)]["points"][0] for s in SEEDS} for r in RESTRICTIONS}
    online, flat = aggregate(curves, bc)
    paired = paired_differences(curves, bc)

    def flat_costs(key):
        return known_sum([runs[k]["costs"][key] for k in RUN_KEYS])

    def data_stat(name, key):
        return known_sum([data[s]["record"]["datasets"][name]["stats"].get(key) for s in SEEDS])

    # Every run carries the same verified reference block.
    ref_block = records[RUN_IDS[0]][0]["config"]["reference"]
    costs = {
        "new_runs": {
            "controller_elapsed_seconds": known_sum(
                [outcomes[j]["controller_elapsed_seconds"] for j in RUN_IDS]
            ),
            "observed_env_steps": flat_costs("observed_env_steps"),
            "evaluation_env_steps": flat_costs("evaluation_env_steps"),
            "ftl_in_run_collection_env_steps": known_sum(
                [runs[k]["costs"]["in_run_collection_env_steps"] for k in RUN_KEYS
                 if k[0] == "ftl"]
            ),
            "fits": flat_costs("fits"),
            "epochs": flat_costs("epochs"),
            "note": "BC-iid collection_steps repeat the shared stream acquisition and "
            "are not run work; fixed BC collects nothing in the run",
        },
        "new_data": {
            "controller_elapsed_seconds": known_sum(
                [outcomes[j]["controller_elapsed_seconds"] for j in DATA_IDS]
            ),
            "pool_env_steps": data_stat("pool", "env_steps"),
            "stream_env_steps": data_stat("stream", "env_steps"),
            "pool_expert_predict_calls": data_stat("pool", "expert_predict_calls"),
            "stream_expert_predict_calls": data_stat("stream", "expert_predict_calls"),
            "baseline_env_steps": known_sum(
                [data[s]["record"]["baselines_costs"].get("env_steps_counted")
                 for s in SEEDS]
            ),
            "note": "charged once per seed and shared by its six runs; each data job's "
            "own baselines are measured but runs normalize with the common reference",
        },
        "reused_expert_preparation": {
            "cost": None,
            "note": "not recorded in this bundle; reused, not new work",
        },
        "reused_reference": {
            "data_result_sha256": ref_block["data_result_sha256"],
            "producer_seed": ref_block.get("producer_seed"),
            "env_steps_counted": (ref_block.get("baselines_costs") or {}).get(
                "env_steps_counted"
            ),
            "elapsed_seconds": (ref_block.get("baselines_costs") or {}).get(
                "elapsed_seconds"
            ),
            "note": "pinned earlier acquisition reused for normalization; not new work",
        },
    }
    per_seed_final = []
    for s in SEEDS:
        for m in METHODS:
            for r in RESTRICTIONS:
                p = runs[(m, r, s)]["points"][-1]
                per_seed_final.append({"seed": s, "method": m, "restriction": r, **p})
    identities = {
        "frozen_design_sha256": expected.design_sha256,
        "campaign_source_sha256": design["source_sha256"],
        "scientific_source_combined_sha256": ctx["source"]["combined_sha256"],
        "source_git_revision": ctx["source"].get("git_revision"),
        "source_git_dirty": ctx["source"].get("git_dirty"),
        "expert_sha256": design["expert_sha256"],
        "expert_policy_state_sha256": design["expert_identity"].get("policy_state_sha256"),
        "task_sha256": design["task_sha256"],
        "reference_data_sha256": design["reference_data_sha256"],
        "data_campaign_id": dm["campaign_id"],
        "run_campaign_id": rm["campaign_id"],
        "package_versions": ctx["package_versions"],
        "per_seed_data": {
            str(s): {
                "data_result_sha256": data[s]["sha256"],
                "config_sha256": data[s]["record"]["config_sha256"],
                "pool_pairs_sha256": data[s]["record"]["datasets"]["pool"]["pairs_sha256"],
                "stream_pairs_sha256": data[s]["record"]["datasets"]["stream"]["pairs_sha256"],
            }
            for s in SEEDS
        },
        "per_run": {
            run_id(*k): {
                "result_sha256": records[run_id(*k)][1],
                "config_sha256": runs[k]["config_sha256"],
                "trained_pairs_sha256": runs[k]["trained_pairs_sha256"],
                "final_checkpoint": runs[k]["final_checkpoint"],
                "final_checkpoint_sha256_observed": runs[k]["final_checkpoint_sha256"],
            }
            for k in RUN_KEYS
        },
        "artifact_manifest_verified": bool(args.artifact_manifest),
    }
    optimizer = {
        run_id(*k): {
            "fits": runs[k]["fits"],
            "early_stopped_fits": runs[k]["early_stopped_fits"],
            "fixed_budget_fallback_fits": runs[k]["fixed_budget_fallback_fits"],
        }
        for k in RUN_KEYS
    }

    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "complete_grid",
        "protocol": PROTOCOL,
        "n_seeds": len(SEEDS),
        "seeds": list(SEEDS),
        "excluded_seeds": list(EXCLUDED_SEEDS),
        "excluded_note": "pilot seed 300 and audit seed 301 are not aggregated",
        "reference": design["reference"],
        "uncertainty": "training-seed standard error: sample std (ddof=1) of the five "
        "seed-level values divided by sqrt(5); episode-level spread is reported "
        "separately and never used as training-seed uncertainty",
        "final": final_rows(online, flat),
        "paired_final": {
            r: {name: paired[r][name][-1] for name in paired[r]} for r in RESTRICTIONS
        },
        "per_seed_final": per_seed_final,
        "costs": costs,
        "optimizer": optimizer,
        "campaigns": {"data": campaign_span(dinv), "runs": campaign_span(rinv)},
        "inventory": inv_rows,
        "identities": identities,
        "input_sha256": dict(sorted(bundle.files.items())),
        "convergence": design.get("convergence"),
        "limitations": LIMITATIONS,
    }
    points = {
        "schema": POINTS_SCHEMA,
        "n_seeds": len(SEEDS),
        "seeds": list(SEEDS),
        "online": {f"{m}/{r}": online[(m, r)] for m in ONLINE for r in RESTRICTIONS},
        "fixed_bc_flat": {r: flat[r] for r in RESTRICTIONS},
        "paired_differences": paired,
        "note": "no smoothing or interpolation; each row is one saved evaluation "
        "budget present in all five seeds",
    }
    fig_bytes = png(curves_figure(online, flat))
    write_outputs(
        out_dir,
        {
            FIG_CURVES: fig_bytes,
            SUMMARY: dumps(summary),
            POINTS: dumps(points),
            REPORT: report(summary),
        },
        tokens,
    )
    return EXIT_OK


def build_parser():
    p = argparse.ArgumentParser(
        description="Five-seed CartPole restriction results adapter (analysis only)."
    )
    p.add_argument("--bundle-root", required=True, type=pathlib.Path)
    p.add_argument("--frozen-design", required=True, type=pathlib.Path)
    p.add_argument("--output-dir", required=True, type=pathlib.Path)
    p.add_argument("--artifact-manifest", type=pathlib.Path, default=None)
    return p


def main(argv: Optional[Sequence[str]] = None, expected=PUBLIC_EXPECTED) -> int:
    args = build_parser().parse_args(argv)
    try:
        code = analyze(args, expected)
    except (AnalysisInputError, OSError) as exc:
        # OSError text can hold machine paths, so only its type is printed.
        text = str(exc) if isinstance(exc, AnalysisInputError) else type(exc).__name__
        print(f"Refused: {text}", file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps({"exit_code": code,
                      "status": "complete_grid" if code == EXIT_OK else "incomplete_grid"}))
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
