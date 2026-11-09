"""Fail-closed analysis of a frozen classical agnostic confirmation study.

Usage::

    python -m imitation.experiments.agnostic.analyze_classical \\
        --manifest MANIFEST.json --results-root ROOT --audit-root AUDITS \\
        --output-dir FRESH_DIR

``ROOT`` holds one ``ENV/REP/seed-N/result.json`` per learning cell written by
``classical.run_cell``; ``AUDITS`` holds one ``ENV/result.json`` (with its
diagnostic sample file) per environment written by ``classical.audit``.
``FRESH_DIR`` must not exist. The analysis writes ``summary.json`` (strict
finite JSON, relative paths only), ``report.md`` and, only when the analysis is
``complete``, ``figures/*.png|pdf``. Exit codes: 0 (``complete``), 2 (invalid
invocation or manifest, nothing written), 3 (``incomplete``: inventory and
reasons only) and 4 (``failed``: provenance or consistency failure; inventory
and reasons only). No estimate or figure is produced unless every declared
cell and audit is present, complete and consistent; nothing is dropped or
imputed.

The manifest is the scientific protocol frozen before confirmation results are
inspected. It holds exactly these keys and no job argv, host or machine data.
Example (hashes elided; ``seeds`` must be written out in full and strictly
increasing, the ellipsis is documentation only)::

    {
     "schema": "agnostic-classical-confirmation-manifest-v1",
     "study": "Agnostic classical confirmation",
     "cell_protocol": "agnostic-classical-cell/1",
     "audit_protocol": "agnostic-classical-audit/1",
     "source_stage2_files": {"classical.py": "<sha256>", "quantized.py": "<sha256>"},
     "source_stage1_sha256": "<sha256>",
     "package_versions": {"python": "3.8.20", "numpy": "1.24.4", "torch": "2.4.1",
                          "stable_baselines3": "2.2.1", "gymnasium": "0.29.1",
                          "imitation": "<version>"},
     "candidate_env_order": ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"],
     "selection_rule": "first two candidates, in order, that qualified, have at
                        least one positive audit bound and are runtime feasible",
     "envs": [
      {"env_name": "CartPole-v1", "n_actions": 2,
       "expert_sha256": "<sha256>", "preparation_config_sha256": "<sha256>",
       "quantizer_sha256": {"mild": "<sha256>", "severe": "<sha256>"},
       "audit": {"result_sha256": "<sha256>", "samples_sha256": "<sha256>",
                 "seed": 500, "episodes": 4096, "delta": 0.008333333333333333}},
      ...
     ],
     "representations": ["mild", "severe"],
     "seeds": [1000, 1001, ..., 1019],
     "excluded_pilot_seeds": [0, 1, 2],
     "budget": 4096, "batch": 16,
     "checkpoints": [128, 256, 512, 1024, 2048, 4096],
     "eval_episodes": 100,
     "analysis": {"confidence_level": 0.95, "bootstrap_resamples": 10000,
                  "bootstrap_seed": 20260929, "auc_budget_range": [128, 4096]}
    }

``quantizer_sha256`` is ``pilot.config_digest(quantizer.describe())`` and must
match the predeclared quantizer. ``source_stage2_files`` must equal the hashes
of the ``classical.py`` and ``quantized.py`` this analyzer imports, since it
regenerates their seeds, streams and quantizers; otherwise the manifest is
invalid. ``source_stage1_sha256`` must equal the imported
``pilot.source_fingerprint()["combined_sha256"]``. ``package_versions`` pins
the producers' Python and library versions (``python``, ``numpy``, ``torch``,
``stable_baselines3``, ``gymnasium``, ``imitation``) and is frozen before
confirmation runs; the analysis itself may run under other versions. Every
cell and audit record must carry exactly these versions and this stage 1
digest, whose file map must hash to it; any mismatch is ``failed_checks``.
``envs`` lists the environments the operator selected before
confirmation, in candidate order; the analyzer reports, but never performs,
that selection. Every declared seed is required.

The Bonferroni family is every declared (env, representation) condition, so
its size is ``len(envs) * len(representations)`` and is fixed by the manifest.
The production manifest (two environments by ``mild`` and ``severe``) freezes
a family of 4, giving family intervals at ``1 - (1 - level) / 4`` (0.9875 for
level 0.95). Smaller manifests are accepted, for tiny fixtures, and report
their own family size.

Artifact statuses: ``missing``, ``incomplete`` (producer ``running`` or
``partial``), ``failed`` (producer ``failed``; kept in the inventory with its
reasons), ``invalid`` (unreadable or malformed), ``failed_checks`` (provenance
or consistency) and ``complete``. The analysis is ``failed`` if any artifact
is ``failed``, ``invalid`` or ``failed_checks``, else ``incomplete`` unless all
are ``complete``. Every record, whatever its status, must carry aware UTC
timestamps with the claim before the start and the start before the finish,
and an effective deadline equal to the requested one clamped to
``classical.HARD_CAP``; a complete record must also finish by its effective
deadline. A partial audit fails rather than being incomplete when its bytes
differ from the pinned ``result_sha256``.

Operational precondition: the analyzer sees only the two roots, not the job
queue. Before invoking it, the operator must independently verify that every
queued job exited 0 with a validated complete record and that each
``result.json`` hash matches the queue's record; the analyzer does not and
cannot check this.

Producer clock note: the producer's final deadline check and its
``finished_at_utc`` read are two clock calls. A run that passes the check
within that sub-millisecond window before the deadline records a finish
after it and is reported ``failed_checks``. This can only reject an honest
record, never accept a late one, and the record is kept. A wall clock that
steps backwards can likewise break the chronological checks.
"""

import argparse
import dataclasses
import json
import math
import pathlib
import re
import sys
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from imitation.experiments.agnostic import (
    analyze_toy,
    classical,
    pilot,
    quantized,
    rollouts,
)
from imitation.experiments.agnostic.analyze_toy import (
    FIGURE_COLORS,
    _canonical_hash,
    _is_int,
    _is_real,
    _sha256,
    _strict_loads,
    _utc,
    bootstrap_mean,
)

MANIFEST_SCHEMA = "agnostic-classical-confirmation-manifest-v1"
SUMMARY_SCHEMA = "agnostic-classical-analysis-v1"
EXIT_OK, EXIT_INVALID, EXIT_INCOMPLETE, EXIT_FAILED = 0, 2, 3, 4
MANIFEST_KEYS = {
    "schema",
    "study",
    "cell_protocol",
    "audit_protocol",
    "source_stage2_files",
    "source_stage1_sha256",
    "package_versions",
    "candidate_env_order",
    "selection_rule",
    "envs",
    "representations",
    "seeds",
    "excluded_pilot_seeds",
    "budget",
    "batch",
    "checkpoints",
    "eval_episodes",
    "analysis",
}
ENV_KEYS = {
    "env_name",
    "n_actions",
    "expert_sha256",
    "preparation_config_sha256",
    "quantizer_sha256",
    "audit",
}
AUDIT_KEYS = {"result_sha256", "samples_sha256", "seed", "episodes", "delta"}
ANALYSIS_KEYS = {
    "confidence_level",
    "bootstrap_resamples",
    "bootstrap_seed",
    "auc_budget_range",
}
CELL_CONFIG_KEYS = {
    "schema",
    "env_name",
    "representation",
    "seed",
    "budget",
    "batch",
    "checkpoints",
    "eval_episodes",
    "quantizer",
    "learner",
    "initial_policy",
    "arms",
    "estimand",
    "pairing",
    "preparation_config_sha256",
    "expert_sha256",
}
SOURCE_FILES = ("classical.py", "quantized.py")
PACKAGE_KEYS = (
    "python",
    "numpy",
    "torch",
    "stable_baselines3",
    "gymnasium",
    "imitation",
)
# Producer modules whose streams, seeds and quantizers the analysis reuses.
PINNED_MODULES = {"classical.py": classical, "quantized.py": quantized}
# Every module whose code the analysis runs directly, hashed into the summary.
ANALYSIS_MODULES = (
    sys.modules[__name__],
    analyze_toy,
    pilot,
    rollouts,
    classical,
    quantized,
)
PRODUCER_STATUSES = ("running", "partial", "failed", "complete")
# Errors a malformed record raises inside the per-artifact validation boundary.
MALFORMED = (KeyError, TypeError, ValueError, IndexError, AttributeError)
AUDIT_ARRAYS = {
    "obs": np.floating,
    "label": np.integer,
    "expert_behavior": np.bool_,
    "length": np.integer,
    "index": np.integer,
    "reset_seed": np.integer,
}
ARMS = ("ftl_final", "bc_iid_final", "ftl_mixture")
ARM_LABELS = {
    "ftl_final": "FTL-DAgger final post-update policy",
    "bc_iid_final": "BC-iid final policy",
    "fixed_bc": "fixed BC (alias: same table and evaluation as BC-iid)",
    "ftl_mixture": "FTL episode-level mixture (sampled, pi_0..pi_{R-1})",
}
INT_COSTS = tuple(
    f.name for f in dataclasses.fields(classical.Costs) if f.name != "elapsed_seconds"
)
SEED_DIR = re.compile(r"^seed-(\d+)$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_REPORTED_FAILURES = 200


class ManifestError(ValueError):
    pass


def _close(a: Any, b: Any) -> bool:
    return _is_real(a) and _is_real(b) and abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _is_hex(x: Any) -> bool:
    return isinstance(x, str) and bool(HEX64.match(x))


def quantizer_sha256(env_name: str, representation: str) -> str:
    return pilot.config_digest(
        quantized.get_quantizer(env_name, representation).describe(),
    )


def _module_sha256(module: Any) -> str:
    return pilot.sha256_file(pathlib.Path(module.__file__))


def loaded_stage2_files() -> Dict[str, str]:
    """Hashes of the producer modules this analyzer actually imported."""
    return {name: _module_sha256(mod) for name, mod in PINNED_MODULES.items()}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def validate_manifest(m: Any) -> Dict[str, Any]:
    """Return the manifest after checking every field; raise ManifestError."""

    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ManifestError(msg)

    need(isinstance(m, dict), "manifest must be a JSON object")
    need(
        set(m) == MANIFEST_KEYS,
        "manifest keys must be exactly {}".format(sorted(MANIFEST_KEYS)),
    )
    need(m["schema"] == MANIFEST_SCHEMA, "unknown manifest schema")
    need(isinstance(m["study"], str) and bool(m["study"]), "study must be text")
    need(isinstance(m["selection_rule"], str), "selection_rule must be text")
    need(m["cell_protocol"] == classical.CELL_SCHEMA, "unsupported cell protocol")
    need(m["audit_protocol"] == classical.AUDIT_SCHEMA, "unsupported audit protocol")
    src = m["source_stage2_files"]
    need(
        isinstance(src, dict)
        and set(src) == set(SOURCE_FILES)
        and all(_is_hex(v) for v in src.values()),
        "source_stage2_files must give a sha256 for {}".format(SOURCE_FILES),
    )
    need(
        src == loaded_stage2_files(),
        "source_stage2_files differ from the classical.py and quantized.py "
        "this analyzer imports for streams, seeds and quantizers",
    )
    need(
        _is_hex(m["source_stage1_sha256"])
        and m["source_stage1_sha256"] == pilot.source_fingerprint()["combined_sha256"],
        "source_stage1_sha256 must be the imported stage 1 source fingerprint",
    )
    versions = m["package_versions"]
    need(
        isinstance(versions, dict)
        and set(versions) == set(PACKAGE_KEYS)
        and all(
            isinstance(v, str) and v and v != "unavailable" for v in versions.values()
        ),
        "package_versions must give a version string for each of {}".format(
            PACKAGE_KEYS
        ),
    )
    order = m["candidate_env_order"]
    need(
        isinstance(order, list)
        and order
        and all(isinstance(n, str) for n in order)
        and len(set(order)) == len(order),
        "candidate_env_order must be a non-empty list of distinct names",
    )
    reps = m["representations"]
    need(
        isinstance(reps, list)
        and reps
        and all(isinstance(r, str) for r in reps)
        and len(set(reps)) == len(reps)
        and all(r in quantized.REPRESENTATIONS for r in reps),
        "representations must be distinct entries of {}".format(
            quantized.REPRESENTATIONS
        ),
    )
    envs = m["envs"]
    need(isinstance(envs, list) and envs, "envs must be a non-empty list")
    names = []
    for e in envs:
        need(
            isinstance(e, dict) and set(e) == ENV_KEYS,
            "env keys must be {}".format(sorted(ENV_KEYS)),
        )
        name = e["env_name"]
        need(
            isinstance(name, str) and name in order,
            "{!r} is not a declared candidate".format(name),
        )
        names.append(name)
        need(_is_int(e["n_actions"]) and e["n_actions"] >= 2, "n_actions >= 2")
        need(_is_hex(e["expert_sha256"]), "expert_sha256 must be sha256 hex")
        need(_is_hex(e["preparation_config_sha256"]), "preparation sha256 hex")
        qs = e["quantizer_sha256"]
        need(isinstance(qs, dict) and set(qs) == set(reps), "quantizer_sha256 per rep")
        for rep in reps:
            try:
                expected = quantizer_sha256(name, rep)
            except ValueError as exc:
                raise ManifestError(str(exc))
            need(
                qs[rep] == expected,
                "{}/{} quantizer differs from the predeclared "
                "specification".format(name, rep),
            )
        a = e["audit"]
        need(
            isinstance(a, dict) and set(a) == AUDIT_KEYS,
            "audit keys must be {}".format(sorted(AUDIT_KEYS)),
        )
        need(
            _is_hex(a["result_sha256"]) and _is_hex(a["samples_sha256"]),
            "audit hashes must be sha256 hex",
        )
        need(_is_int(a["seed"]) and a["seed"] >= 0, "audit seed must be >= 0")
        need(_is_int(a["episodes"]) and a["episodes"] > 0, "audit episodes > 0")
        need(_is_real(a["delta"]) and 0 < a["delta"] < 1, "audit delta in (0, 1)")
    need(len(set(names)) == len(names), "duplicate env")
    need(names == [n for n in order if n in names], "envs must follow candidate order")
    seeds, pilots = m["seeds"], m["excluded_pilot_seeds"]
    need(
        isinstance(seeds, list)
        and seeds
        and all(_is_int(s) and s >= 0 for s in seeds)
        and all(b > a for a, b in zip(seeds, seeds[1:])),
        "seeds must be non-empty, non-negative and strictly increasing",
    )
    need(
        isinstance(pilots, list) and all(_is_int(s) for s in pilots),
        "excluded_pilot_seeds must be integers",
    )
    need(not set(seeds) & set(pilots), "pilot seeds must be excluded from seeds")
    for key in ("budget", "batch", "eval_episodes"):
        need(_is_int(m[key]), "{} must be an integer".format(key))
    cps = m["checkpoints"]
    need(isinstance(cps, list) and all(_is_int(c) for c in cps), "checkpoints ints")
    need(all(b > a for a, b in zip(cps, cps[1:])), "checkpoints strictly increasing")
    try:
        classical.validate_cell_args(
            reps[0], 0, m["budget"], m["batch"], cps, m["eval_episodes"]
        )
    except ValueError as exc:
        raise ManifestError(str(exc))
    an = m["analysis"]
    need(
        isinstance(an, dict) and set(an) == ANALYSIS_KEYS,
        "analysis keys must be " "{}".format(sorted(ANALYSIS_KEYS)),
    )
    need(
        _is_real(an["confidence_level"]) and 0 < an["confidence_level"] < 1,
        "confidence_level in (0, 1)",
    )
    need(
        _is_int(an["bootstrap_resamples"]) and an["bootstrap_resamples"] >= 1,
        "bootstrap_resamples >= 1",
    )
    need(
        _is_int(an["bootstrap_seed"]) and an["bootstrap_seed"] >= 0,
        "bootstrap_seed >= 0",
    )
    rng = an["auc_budget_range"]
    need(
        isinstance(rng, list)
        and len(rng) == 2
        and all(_is_int(r) and r in cps for r in rng)
        and rng[0] < rng[1],
        "auc_budget_range must be two increasing checkpoints",
    )
    return m


# ---------------------------------------------------------------------------
# Exact recomputation helpers
# ---------------------------------------------------------------------------


def erm_table(counts: np.ndarray) -> np.ndarray:
    """Majority label per bin; ties and empty bins take the lowest action."""
    return np.argmax(np.asarray(counts), axis=1).astype(np.int64)


def count_matrix(bins: Sequence[int], labels: Sequence[int], k: int, a: int):
    counts = np.zeros((k, a), dtype=np.int64)
    for b, y in zip(bins, labels):
        counts[b, y] += 1
    return counts


def reference_disagreement(table: Sequence[int], counts: np.ndarray) -> float:
    """Empirical disagreement of ``table`` on the frozen audit sample."""
    counts = np.asarray(counts, dtype=np.int64)
    n = int(counts.sum())
    agree = int(counts[np.arange(counts.shape[0]), np.asarray(table)].sum())
    return (n - agree) / n


def reference_floor(counts: np.ndarray, delta: float) -> Dict[str, Any]:
    """Recompute the audit's uniform bound; ``K`` counts every bin, seen or not."""
    counts = np.asarray(counts, dtype=np.int64)
    k, a = counts.shape
    n = int(counts.sum())
    empirical = (n - int(counts.max(axis=1).sum())) / n
    slack = math.sqrt((k * math.log(a) + math.log(1.0 / delta)) / (2.0 * n))
    bound = max(0.0, empirical - slack)
    return {
        "n": n,
        "K": int(k),
        "A": int(a),
        "delta": float(delta),
        "observed_bins": int(np.count_nonzero(counts.sum(axis=1))),
        "empirical_min_disagreement": empirical,
        "slack": slack,
        "lower_bound": bound,
        "positive_certificate": bound > 0.0,
    }


def _int_costs(c: Any) -> Optional[Dict[str, int]]:
    if not isinstance(c, dict) or not all(_is_int(c.get(k)) for k in INT_COSTS):
        return None
    if not _is_real(c.get("elapsed_seconds")) or c["elapsed_seconds"] < 0:
        return None
    return {k: c[k] for k in INT_COSTS}


def _add(*items: Dict[str, int]) -> Dict[str, int]:
    return {k: sum(i[k] for i in items) for k in INT_COSTS}


# ---------------------------------------------------------------------------
# Provenance and consistency of one record
# ---------------------------------------------------------------------------


def shared_provenance(rec: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, bool]:
    """Source and library pins every cell and audit must share with the manifest."""
    source = rec.get("source") or {}
    stage1 = source.get("stage1") or {}
    files = stage1.get("files")
    return {
        "source": source.get("stage2_files") == manifest["source_stage2_files"],
        "source_stage1": stage1.get("combined_sha256")
        == manifest["source_stage1_sha256"]
        and isinstance(files, dict)
        and pilot.config_digest(files) == manifest["source_stage1_sha256"],
        "package_versions": rec.get("package_versions") == manifest["package_versions"],
    }


def time_problems(rec: Dict[str, Any]) -> List[str]:
    """Check a producer record's timestamps; applies whatever its status.

    Every timestamp must be aware UTC ISO. The effective deadline must be the
    requested one clamped to the producer's fixed hard cap, the claim must
    precede the start, and the finish (null only while running) must follow
    the start. A complete record must also finish by its effective deadline.
    """
    claim = rec.get("claim")
    status = rec.get("status")
    raw = {
        "claim.claimed_at_utc": (
            claim.get("claimed_at_utc") if isinstance(claim, dict) else None
        ),
        "requested_deadline_utc": rec.get("requested_deadline_utc"),
        "effective_deadline_utc": rec.get("effective_deadline_utc"),
        "started_at_utc": rec.get("started_at_utc"),
    }
    if status == "running":
        if rec.get("finished_at_utc") is not None:
            return ["running record has a finish time"]
    else:
        raw["finished_at_utc"] = rec.get("finished_at_utc")
    t = {name: _utc(value) for name, value in raw.items()}
    bad = sorted(name for name, value in t.items() if value is None)
    if bad:
        return ["not an aware UTC ISO timestamp: {}".format(bad)]
    p = []
    effective = t["effective_deadline_utc"]
    if effective != min(t["requested_deadline_utc"], classical.HARD_CAP):
        p.append("effective deadline is not the requested one clamped to the hard cap")
    if not t["claim.claimed_at_utc"] <= t["started_at_utc"]:
        p.append("started before its output directory was claimed")
    finished = t.get("finished_at_utc")
    if finished is not None and not t["started_at_utc"] <= finished:
        p.append("finished before it started")
    if status == "complete" and not finished <= effective:
        p.append("complete record finished after its effective deadline")
    return p


def producer_status(rec: Dict[str, Any], art: Dict[str, Any]) -> bool:
    """Record the producer's status on ``art``; True only if it is complete.

    A producer ``failed`` record is a failed artifact; ``running`` and
    ``partial`` are incomplete; any other value is invalid.
    """
    status = rec.get("status")
    art["producer_status"] = status
    if status == "complete":
        return True
    if status not in PRODUCER_STATUSES:
        art["status"] = "invalid"
        art["reasons"].append("unknown producer status {!r}".format(status))
        return False
    err = rec.get("error")
    art["status"] = "failed" if status == "failed" else "incomplete"
    art["reasons"].append(
        "producer status {!r}{}".format(
            status,
            ", error {}".format(err.get("type")) if isinstance(err, dict) else "",
        )
    )
    return False


def cell_provenance(
    rec: Any, env: Dict[str, Any], rep: str, seed: int, manifest: Dict[str, Any]
) -> List[str]:
    """Identity problems; checked for every record, whatever its status."""
    if not isinstance(rec, dict) or not isinstance(rec.get("config"), dict):
        return ["not a cell record"]
    cfg, prep = rec["config"], rec.get("preparation") or {}
    name = env["env_name"]
    checks = {
        "schema": rec.get("schema")
        == rec.get("protocol")
        == cfg.get("schema")
        == manifest["cell_protocol"],
        "identity": (rec.get("env_name"), rec.get("representation"), rec.get("seed"))
        == (cfg.get("env_name"), cfg.get("representation"), cfg.get("seed"))
        == (name, rep, seed),
        "config_keys": set(cfg) == CELL_CONFIG_KEYS,
        "config_sha256": rec.get("config_sha256") == pilot.config_digest(cfg),
        "grid": [
            cfg.get(k) for k in ("budget", "batch", "checkpoints", "eval_episodes")
        ]
        == [manifest[k] for k in ("budget", "batch", "checkpoints", "eval_episodes")],
        "quantizer": _is_hex(env["quantizer_sha256"][rep])
        and isinstance(cfg.get("quantizer"), dict)
        and pilot.config_digest(cfg["quantizer"]) == env["quantizer_sha256"][rep],
        "expert_sha256": rec.get("expert_sha256")
        == cfg.get("expert_sha256")
        == prep.get("expert_sha256")
        == env["expert_sha256"],
        "preparation_config_sha256": cfg.get("preparation_config_sha256")
        == prep.get("config_sha256")
        == env["preparation_config_sha256"],
        **shared_provenance(rec, manifest),
    }
    problems = [
        "provenance mismatch: {}".format(k) for k, ok in checks.items() if not ok
    ]
    return problems + time_problems(rec)


def _check_eval(
    ev: Any, where: str, eval_seeds: List[int], n_eval: int, p: List[str]
) -> Optional[Dict[str, int]]:
    if not isinstance(ev, dict):
        p.append("{}: missing evaluation".format(where))
        return None
    returns, lengths, dis = ev.get("returns"), ev.get("lengths"), ev.get("disagreement")
    if not (
        isinstance(returns, list)
        and isinstance(lengths, list)
        and isinstance(dis, list)
        and len(returns) == len(lengths) == len(dis) == n_eval
        and all(_is_real(x) for x in returns)
        and all(_is_int(x) and x >= 1 for x in lengths)
        and all(_is_real(x) and 0 <= x <= 1 for x in dis)
    ):
        p.append(
            "{}: per-episode arrays malformed or not {} long".format(where, n_eval)
        )
        return None
    if ev.get("reset_seeds") != eval_seeds:
        p.append("{}: evaluation reset seeds are not the shared seeds".format(where))
    if not _close(ev.get("mean_return"), float(np.mean(returns))):
        p.append("{}: mean_return differs from per-episode returns".format(where))
    if not _close(ev.get("mean_disagreement"), float(np.mean(dis))):
        p.append("{}: mean_disagreement differs from per-episode values".format(where))
    c = _int_costs(ev.get("costs"))
    steps = int(sum(lengths))
    expected = dict.fromkeys(INT_COSTS, 0)
    expected.update(
        env_steps=steps,
        env_resets=n_eval,
        completed_episodes=n_eval,
        behavior_predict_calls=steps,
        behavior_action_entries=steps,
        expert_predict_calls=n_eval,
        expert_action_entries=steps,
    )
    if c != expected:
        p.append("{}: evaluation counters do not match its episodes".format(where))
        return None
    return c


def cell_consistency(
    rec: Dict[str, Any],
    env: Dict[str, Any],
    rep: str,
    seed: int,
    manifest: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """Recompute every derivable quantity of a complete record.

    Returns:
        Problems (empty if consistent) and the extracted analysis data.
    """
    p: List[str] = []
    budget, batch, n_eval = (
        manifest["budget"],
        manifest["batch"],
        manifest["eval_episodes"],
    )
    points = manifest["checkpoints"]
    rounds = budget // batch
    k = quantized.get_quantizer(env["env_name"], rep).n_bins
    a = env["n_actions"]
    if rec.get("interruption") is not None or rec.get("error") is not None:
        p.append("complete record carries an interruption or error")
    if rec.get("rounds_completed") != rounds:
        p.append("rounds_completed is not budget / batch")
    retained = rec.get("retained") or {}
    data: Dict[str, Any] = {}
    for arm in ("ftl", "bc_iid"):
        r = retained.get(arm) or {}
        bins, labels, lengths = r.get("bins"), r.get("labels"), r.get("lengths")
        if not (
            isinstance(bins, list)
            and isinstance(labels, list)
            and isinstance(lengths, list)
            and len(bins) == len(labels) == len(lengths) == budget == r.get("n")
            and all(_is_int(b) and 0 <= b < k for b in bins)
            and all(_is_int(y) and 0 <= y < a for y in labels)
            and all(_is_int(x) and x >= 1 for x in lengths)
        ):
            return (
                p
                + ["retained {}: arrays malformed or not {} long".format(arm, budget)],
                {},
            )
        counts = count_matrix(bins, labels, k, a)
        if r.get("counts") != counts.tolist():
            p.append("retained {}: counts differ from retained arrays".format(arm))
        table = erm_table(counts)
        if r.get("table") != table.tolist() or r.get(
            "table_sha256"
        ) != quantized.table_sha256(table):
            p.append("retained {}: final table is not the exact ERM table".format(arm))
        data[arm] = (bins, labels, lengths)

    seeds = rec.get("seeds") or {}
    train_seeds = classical.reset_seeds(seed, "train", budget)
    eval_seeds = classical.reset_seeds(seed, "eval", n_eval)
    if seeds.get("train_reset_seeds") != train_seeds or any(
        retained[arm].get("reset_seeds") != train_seeds for arm in data
    ):
        p.append("training reset seeds are not the paired seed stream")
    if (
        seeds.get("train_selection_uniforms")
        != classical.selection_uniforms(seed, "train", budget).tolist()
    ):
        p.append("training selection uniforms are not the paired stream")
    if seeds.get("eval_reset_seeds") != eval_seeds:
        p.append("evaluation reset seeds are not the shared seed stream")

    fb, fl, _ = data["ftl"]
    history = rec.get("ftl_behavior_tables")
    if not (isinstance(history, list) and len(history) == rounds):
        return p + ["FTL behavior history is not one table per round"], {}
    if history[0] != [0] * k:
        p.append("FTL initial behavior policy is not the all-zero table")
    for r in range(1, rounds):
        if (
            history[r]
            != erm_table(count_matrix(fb[: r * batch], fl[: r * batch], k, a)).tolist()
        ):
            p.append("FTL behavior table {} is not the ERM refit".format(r))
            break
    if rec.get("ftl_behavior_table_sha256") != [
        quantized.table_sha256(np.asarray(t)) for t in history
    ]:
        p.append("FTL behavior table hashes differ")

    costs = rec.get("costs") or {}
    totals = {
        name: _int_costs(costs.get(name))
        for name in ("train_ftl", "train_bc_iid", "train_fixed_bc", "eval")
    }
    if any(v is None for v in totals.values()) or set(costs) != set(totals):
        return p + ["cost ledger malformed"], {}

    entries = rec.get("checkpoints")
    if not (
        isinstance(entries, list)
        and all(isinstance(e, dict) for e in entries)
        and [e.get("budget") for e in entries] == points
    ):
        return p + ["checkpoint list does not match the declared checkpoints"], {}
    eval_sum = dict.fromkeys(INT_COSTS, 0)
    extracted = []
    snapshot = None
    for idx, (entry, n) in enumerate(zip(entries, points)):
        where = "B={}".format(n)
        if entry.get("complete") is not True or entry.get("rounds") != n // batch:
            p.append("{}: checkpoint incomplete or wrong round count".format(where))
        tables = {}
        for arm in ("ftl", "bc_iid"):
            bins, labels, _ = data[arm]
            fin = entry.get("{}_final".format(arm)) or {}
            table = erm_table(count_matrix(bins[:n], labels[:n], k, a))
            if fin.get("table") != table.tolist() or fin.get(
                "table_sha256"
            ) != quantized.table_sha256(table):
                p.append(
                    "{}: {} table is not the ERM table of its first B labels".format(
                        where, arm
                    )
                )
            if fin.get("data_sha256") != quantized.data_sha256(bins[:n], labels[:n]):
                p.append("{}: {} data hash differs".format(where, arm))
            tables["{}_final".format(arm)] = table.tolist()
        if n < budget and tables["ftl_final"] != history[n // batch]:
            p.append(
                "{}: final FTL table is not the next behavior policy".format(where)
            )
        fixed = entry.get("fixed_bc") or {}
        bc = entry.get("bc_iid_final") or {}
        if (
            fixed.get("table_sha256") != bc.get("table_sha256")
            or fixed.get("data_sha256") != bc.get("data_sha256")
            or fixed.get("equals_bc_iid") is not True
            or fixed.get("eval_alias") != "bc_iid_final"
            or "eval" in fixed
        ):
            p.append("{}: fixed BC is not an evaluation alias of BC-iid".format(where))
        mix = entry.get("ftl_mixture") or {}
        # Regenerate the producer's per-checkpoint mixture draw.
        uniforms = classical.stream(seed, "eval", classical._MIXTURE, idx).random(
            n_eval
        )
        want_idxs = [classical.select_index(u, n // batch) for u in uniforms]
        if (
            mix.get("n_policies") != n // batch
            or mix.get("mixture_indices") != want_idxs
        ):
            p.append(
                "{}: mixture policy count or indices are not the frozen "
                "mixture stream".format(where)
            )
        evals = {}
        for arm in ARMS:
            ev = (entry.get(arm) or {}).get("eval")
            c = _check_eval(ev, "{} {}".format(where, arm), eval_seeds, n_eval, p)
            if c is not None:
                eval_sum = _add(eval_sum, c)
                evals[arm] = ev
        snap = {
            name: _int_costs(v)
            for name, v in (entry.get("train_costs_at_checkpoint") or {}).items()
        }
        if set(snap) != {"ftl", "bc_iid", "fixed_bc"} or None in snap.values():
            p.append("{}: training cost snapshot malformed".format(where))
            continue
        steps = {arm: int(sum(data[arm][2][:n])) for arm in data}
        zero = dict.fromkeys(INT_COSTS, 0)
        want_ftl = dict(
            zero,
            env_steps=steps["ftl"],
            env_resets=n,
            completed_episodes=n,
            behavior_predict_calls=steps["ftl"],
            behavior_action_entries=steps["ftl"],
            expert_predict_calls=n,
            expert_action_entries=n,
            retained_labels=n,
            fits=n // batch,
        )
        want_bc = dict(
            zero,
            env_steps=steps["bc_iid"],
            env_resets=n,
            completed_episodes=n,
            behavior_predict_calls=steps["bc_iid"],
            behavior_action_entries=steps["bc_iid"],
            expert_predict_calls=steps["bc_iid"],
            expert_action_entries=steps["bc_iid"],
            retained_labels=n,
            fits=n // batch,
        )
        if snap["ftl"] != want_ftl:
            p.append(
                "{}: FTL counters differ from its retained episodes (expert "
                "entries must equal B)".format(where)
            )
        if snap["bc_iid"] != want_bc:
            p.append(
                "{}: BC-iid counters differ from its expert-driven "
                "steps".format(where)
            )
        if snap["fixed_bc"] != dict(zero, fits=idx + 1):
            p.append("{}: fixed BC must add only its own fits".format(where))
        standalone = _int_costs(fixed.get("logical_standalone_costs"))
        if standalone != dict(snap["bc_iid"], fits=1):
            p.append(
                "{}: standalone fixed BC must be BC-iid acquisition plus one "
                "fit".format(where)
            )
        snapshot = snap
        extracted.append(
            {
                "budget": n,
                "rounds": n // batch,
                "evals": evals,
                "tables": tables,
                "history": history[: n // batch],
                "oracle_entries": {
                    "ftl": snap["ftl"]["expert_action_entries"],
                    "bc_iid": snap["bc_iid"]["expert_action_entries"],
                },
                "standalone": standalone,
            }
        )
    if snapshot is not None and (
        totals["train_ftl"] != snapshot["ftl"]
        or totals["train_bc_iid"] != snapshot["bc_iid"]
        or totals["train_fixed_bc"] != snapshot["fixed_bc"]
    ):
        p.append("final training counters differ from the last checkpoint")
    if totals["eval"] != eval_sum:
        p.append(
            "evaluation total is not the sum of the stored evaluations "
            "(an alias must not add cost)"
        )
    phys = rec.get("physical_costs") or {}
    train_sum = _add(
        totals["train_ftl"], totals["train_bc_iid"], totals["train_fixed_bc"]
    )
    if _int_costs(phys.get("training")) != train_sum or _int_costs(
        phys.get("all")
    ) != _add(train_sum, totals["eval"]):
        p.append(
            "physical totals must include fixed BC fits once and shared "
            "acquisition once"
        )
    if train_sum["retained_labels"] != 2 * budget:
        p.append("physical retained labels count shared BC data twice")
    return p, {
        "checkpoints": extracted,
        "costs": totals,
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
    }


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def audit_sample_problems(
    z: Dict[str, np.ndarray], spec: Dict[str, Any], n_actions: int, obs_dim: int
) -> List[str]:
    """Check the sample arrays' types, shapes and domains and their frozen streams.

    Each episode's behavior (expert or uniform random), its env reset seed and
    its selected time step ``select_index(u, length)`` are regenerated from the
    producer's reference streams.
    """
    n, seed = spec["episodes"], spec["seed"]
    if set(z) != set(AUDIT_ARRAYS):
        return ["audit sample arrays must be exactly {}".format(sorted(AUDIT_ARRAYS))]
    p = []
    for key, kind in AUDIT_ARRAYS.items():
        shape = (n, obs_dim) if key == "obs" else (n,)
        if not np.issubdtype(z[key].dtype, kind) or z[key].shape != shape:
            p.append(
                "audit {}: dtype {} shape {} is not {} {}".format(
                    key, z[key].dtype, z[key].shape, kind.__name__, shape
                )
            )
    if p:
        return p
    label, length = z["label"], z["length"]
    if not np.all(np.isfinite(z["obs"])):
        p.append("audit observations are not finite")
    if np.any(label < 0) or np.any(label >= n_actions):
        p.append("audit labels outside [0, {})".format(n_actions))
    if np.any(length < 1):
        p.append("audit episode lengths must be >= 1")
        return p
    behavior = (
        classical.stream(seed, "reference", classical._BEHAVIOR_CHOICE).random(n) < 0.5
    )
    if not np.array_equal(z["expert_behavior"], behavior):
        p.append("audit episode behaviors are not the frozen choice stream")
    uniforms = classical.selection_uniforms(seed, "reference", n)
    index = [classical.select_index(u, int(t)) for u, t in zip(uniforms, length)]
    if z["index"].tolist() != index:
        p.append("audit sample indices are not select_index(u, length)")
    if z["reset_seed"].tolist() != classical.reset_seeds(seed, "reference", n):
        p.append("audit reset seeds are not the reference stream")
    return p


def audit_cost_problems(costs: Any, z: Dict[str, np.ndarray]) -> List[str]:
    """Exact counters per behavior, derived from the samples' episodes.

    Expert episodes query the expert once per step and reuse the executed
    action as the label; random episodes query it once, at the selected state.
    """
    if not isinstance(costs, dict) or set(costs) != {
        "expert_episodes",
        "random_episodes",
    }:
        return ["audit cost ledger malformed"]
    p = []
    for kind, mask in (
        ("expert_episodes", z["expert_behavior"]),
        ("random_episodes", ~z["expert_behavior"]),
    ):
        episodes, steps = int(mask.sum()), int(z["length"][mask].sum())
        labels = steps if kind == "expert_episodes" else episodes
        want = dict.fromkeys(INT_COSTS, 0)
        want.update(
            env_steps=steps,
            env_resets=episodes,
            completed_episodes=episodes,
            behavior_predict_calls=steps,
            behavior_action_entries=steps,
            expert_predict_calls=labels,
            expert_action_entries=labels,
        )
        if _int_costs(costs[kind]) != want:
            p.append("audit {} counters differ from its samples".format(kind))
    return p


def load_audit(
    root: pathlib.Path, env: Dict[str, Any], manifest: Dict[str, Any]
) -> Dict[str, Any]:
    name = env["env_name"]
    rel = "{}/{}".format(name, classical.RESULT_FILE)
    art: Dict[str, Any] = {"env": name, "path": rel, "status": None, "reasons": []}
    path = root / name / classical.RESULT_FILE
    if not path.is_file():
        art.update(status="missing", reasons=["audit result missing"])
        return art
    try:
        raw = path.read_bytes()
        art["sha256"] = _sha256(raw)
        rec = _strict_loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        art.update(status="invalid", reasons=["unreadable: {}".format(exc)])
        return art
    try:
        _check_audit(rec, root, env, manifest, art)
    except MALFORMED as exc:
        art.pop("data", None)
        art["status"] = "invalid"
        art["reasons"].append(
            "malformed audit record: {}: {}".format(type(exc).__name__, exc)
        )
    return art


def _check_audit(
    rec: Any,
    root: pathlib.Path,
    env: Dict[str, Any],
    manifest: Dict[str, Any],
    art: Dict[str, Any],
) -> None:
    """Validate a parsed audit record and set ``art``'s status and data."""
    name = env["env_name"]
    spec = env["audit"]
    reasons = art["reasons"]
    if art["sha256"] != spec["result_sha256"]:
        reasons.append("audit result sha256 differs from the manifest")
    cfg = rec.get("config") if isinstance(rec, dict) else None
    if not isinstance(cfg, dict):
        art["status"] = "invalid"
        reasons.append("not an audit record")
        return
    prep = rec.get("preparation") or {}
    checks = {
        "schema": rec.get("schema")
        == rec.get("protocol")
        == cfg.get("schema")
        == manifest["audit_protocol"],
        "identity": rec.get("env_name") == cfg.get("env_name") == name
        and rec.get("seed") == cfg.get("seed") == spec["seed"],
        "episodes_delta": cfg.get("episodes") == spec["episodes"]
        and cfg.get("delta") == spec["delta"],
        "config_sha256": rec.get("config_sha256") == pilot.config_digest(cfg),
        "expert_sha256": rec.get("expert_sha256")
        == cfg.get("expert_sha256")
        == prep.get("expert_sha256")
        == env["expert_sha256"],
        "preparation_config_sha256": cfg.get("preparation_config_sha256")
        == prep.get("config_sha256")
        == env["preparation_config_sha256"],
        **shared_provenance(rec, manifest),
        "quantizers": [pilot.config_digest(q) for q in cfg.get("quantizers") or []]
        == [quantizer_sha256(name, r) for r in quantized.REPRESENTATIONS],
    }
    reasons += [
        "provenance mismatch: {}".format(k) for k, ok in checks.items() if not ok
    ]
    reasons += time_problems(rec)
    if reasons:
        art["status"] = "failed_checks"
        art["producer_status"] = rec.get("status")
        return
    if not producer_status(rec, art):
        return
    art["status"] = "failed_checks"
    if rec.get("interruption") is not None or rec.get("error") is not None:
        reasons.append("complete record carries an interruption or error")
        return
    # Recompute counts and bounds from the diagnostic samples themselves.
    sample_file = rec.get("samples_file") or {}
    spath = root / name / classical.AUDIT_DATA_FILE
    if sample_file.get("path") != classical.AUDIT_DATA_FILE or not spath.is_file():
        reasons.append("audit sample file missing")
        return
    digest = pilot.sha256_file(spath)
    if not digest == sample_file.get("sha256") == spec["samples_sha256"]:
        reasons.append("audit sample sha256 mismatch")
        return
    try:
        loaded = np.load(spath, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError("not an npz archive")
        with loaded as npz:
            z = {k: npz[k] for k in npz.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        reasons.append("audit sample file unreadable: {}".format(exc))
        return
    obs_dim = quantized.get_quantizer(name, manifest["representations"][0]).obs_dim
    reasons += audit_sample_problems(z, spec, env["n_actions"], obs_dim)
    if rec.get("completed_samples") != spec["episodes"]:
        reasons.append("audit completed_samples differs from declared episodes")
    if reasons:
        return
    obs, labels = z["obs"], z["label"].tolist()
    reps_out = {}
    for rep in manifest["representations"]:
        q = quantized.get_quantizer(name, rep)
        stored = (rec.get("representations") or {}).get(rep) or {}
        counts = count_matrix([q(o) for o in obs], labels, q.n_bins, env["n_actions"])
        if stored.get("counts") != counts.tolist():
            reasons.append("{}: audit counts differ from its samples".format(rep))
            continue
        floor = reference_floor(counts, spec["delta"])
        bound = stored.get("bound") or {}
        same = all(
            (
                _close(bound.get(key), floor[key])
                if isinstance(floor[key], float)
                else bound.get(key) == floor[key]
            )
            for key in floor
        )
        if not same:
            reasons.append("{}: stored bound differs from recomputation".format(rep))
            continue
        reps_out[rep] = {"counts": counts, "floor": floor}
    costs = rec.get("costs")
    reasons += audit_cost_problems(costs, z)
    if reasons:
        return
    art.update(
        status="complete",
        data={
            "reps": reps_out,
            "reset_seeds": set(z["reset_seed"].tolist()),
            "costs": {
                "expert_episodes": _int_costs(costs["expert_episodes"]),
                "random_episodes": _int_costs(costs["random_episodes"]),
                "elapsed_seconds": float(
                    sum(costs[k]["elapsed_seconds"] for k in costs)
                ),
            },
        },
    )


# ---------------------------------------------------------------------------
# Result inventory
# ---------------------------------------------------------------------------


def scan_root(
    root: pathlib.Path, manifest: Dict[str, Any]
) -> Tuple[Dict[Tuple, pathlib.Path], List[str], List[str]]:
    """Map declared keys to files; list layout problems and ignored entries."""
    found: Dict[Tuple, pathlib.Path] = {}
    problems: List[str] = []
    ignored: List[str] = []
    envs = {e["env_name"] for e in manifest["envs"]}
    seeds, pilots = set(manifest["seeds"]), set(manifest["excluded_pilot_seeds"])
    for env_dir in sorted(root.iterdir()):
        if env_dir.name.startswith("."):
            ignored.append(env_dir.name)
            continue
        if env_dir.name not in envs or not env_dir.is_dir():
            problems.append("undeclared entry {}".format(env_dir.name))
            continue
        for rep_dir in sorted(env_dir.iterdir()):
            rel = "{}/{}".format(env_dir.name, rep_dir.name)
            if rep_dir.name.startswith("."):
                ignored.append(rel)
                continue
            if rep_dir.name not in manifest["representations"] or not rep_dir.is_dir():
                problems.append("undeclared entry {}".format(rel))
                continue
            for seed_dir in sorted(rep_dir.iterdir()):
                srel = "{}/{}".format(rel, seed_dir.name)
                if seed_dir.name.startswith("."):
                    ignored.append(srel)
                    continue
                m = SEED_DIR.match(seed_dir.name)
                if not m or not seed_dir.is_dir():
                    problems.append("undeclared entry {}".format(srel))
                    continue
                seed = int(m.group(1))
                key = (env_dir.name, rep_dir.name, seed)
                if seed_dir.name != "seed-{}".format(seed):
                    problems.append("non-canonical seed directory {}".format(srel))
                elif seed in pilots:
                    problems.append("excluded pilot seed present: {}".format(srel))
                elif seed not in seeds:
                    problems.append("undeclared seed {}".format(srel))
                elif key in found:
                    problems.append("duplicate cell {}".format(srel))
                else:
                    found[key] = seed_dir / classical.RESULT_FILE
    return found, problems, ignored


def _check_cell(
    rec: Any,
    env: Dict[str, Any],
    rep: str,
    seed: int,
    manifest: Dict[str, Any],
    audits: Dict[str, Dict[str, Any]],
    art: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Validate a parsed cell record; set ``art``'s status; return its data."""
    art["reasons"] += cell_provenance(rec, env, rep, seed, manifest)
    if art["reasons"]:
        art["status"] = "failed_checks"
        art["producer_status"] = rec.get("status") if isinstance(rec, dict) else None
        return None
    if not producer_status(rec, art):
        art["rounds_completed"] = rec.get("rounds_completed")
        return None
    probs, cell = cell_consistency(rec, env, rep, seed, manifest)
    audit = audits.get(env["env_name"], {}).get("data")
    if (
        audit
        and not probs
        and audit["reset_seeds"] & (set(cell["train_seeds"]) | set(cell["eval_seeds"]))
    ):
        probs.append(
            "training or evaluation shares reset seeds with the diagnostic audit"
        )
    art["reasons"] += probs
    art["status"] = "failed_checks" if probs else "complete"
    return None if probs else cell


def load_cells(
    root: pathlib.Path, manifest: Dict[str, Any], audits: Dict[str, Dict[str, Any]]
):
    found, problems, ignored = scan_root(root, manifest)
    artifacts, data, by_hash = [], {}, {}
    for env in manifest["envs"]:
        for rep in manifest["representations"]:
            for seed in manifest["seeds"]:
                key = (env["env_name"], rep, seed)
                rel = "{}/{}/seed-{}/{}".format(*key, classical.RESULT_FILE)
                art: Dict[str, Any] = {
                    "env": key[0],
                    "representation": rep,
                    "seed": seed,
                    "path": rel,
                    "reasons": [],
                }
                artifacts.append(art)
                path = found.get(key)
                if path is None or not path.is_file():
                    art["status"] = "missing"
                    art["reasons"].append("result.json missing")
                    continue
                try:
                    raw = path.read_bytes()
                except OSError as exc:
                    art["status"] = "invalid"
                    art["reasons"].append("unreadable: {}".format(exc))
                    continue
                art["sha256"] = _sha256(raw)
                if art["sha256"] in by_hash:
                    art["reasons"].append(
                        "byte-identical to {}".format(by_hash[art["sha256"]])
                    )
                by_hash.setdefault(art["sha256"], rel)
                try:
                    rec = _strict_loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as exc:
                    art["status"] = "invalid"
                    art["reasons"].append("unreadable: {}".format(exc))
                    continue
                try:
                    cell = _check_cell(rec, env, rep, seed, manifest, audits, art)
                except MALFORMED as exc:
                    art["status"] = "invalid"
                    art["reasons"].append(
                        "malformed cell record: {}: {}".format(type(exc).__name__, exc)
                    )
                    continue
                if cell is not None:
                    data[key] = cell
    return artifacts, data, problems, ignored


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def seed_index(n: int, resamples: int, seed: int) -> Optional[np.ndarray]:
    """One seed-block resampling matrix shared by every condition."""
    if n < 2:
        return None
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def paired_estimate(
    ftl_eps: Sequence[Sequence[float]],
    bc_eps: Sequence[Sequence[float]],
    index: Optional[np.ndarray],
    levels: Sequence[float],
) -> Dict[str, Any]:
    """Seed-paired mean difference with an episode Monte Carlo diagnostic.

    Each training seed contributes one value, the difference of its two
    evaluation means; the bootstrap resamples whole seeds. The within-run
    standard error of the paired episode differences is a precision
    diagnostic only and is not added to the bootstrap.
    """
    diffs = np.array([np.mean(f) - np.mean(b) for f, b in zip(ftl_eps, bc_eps)])
    out = bootstrap_mean(diffs, index, levels)
    ses = []
    for f, b in zip(ftl_eps, bc_eps):
        d = np.asarray(f, float) - np.asarray(b, float)
        ses.append(float(np.std(d, ddof=1) / math.sqrt(d.size)) if d.size > 1 else None)
    out["per_seed_difference"] = diffs.tolist()
    out["seed_sd"] = float(np.std(diffs, ddof=1)) if diffs.size > 1 else None
    known = [s for s in ses if s is not None]
    out["episode_mc_se_per_seed"] = ses
    out["episode_mc_se_of_seed_mean"] = (
        math.sqrt(sum(s * s for s in known)) / len(ses)
        if len(known) == len(ses)
        else None
    )
    return out


def _levels(manifest: Dict[str, Any]) -> Tuple[float, float, int]:
    conf = manifest["analysis"]["confidence_level"]
    m = len(manifest["envs"]) * len(manifest["representations"])
    return conf, 1 - (1 - conf) / m, m


def _auc(budgets: Sequence[int], values: Sequence[float], lo: int, hi: int) -> float:
    pts = [(math.log2(b), v) for b, v in zip(budgets, values) if lo <= b <= hi]
    return float(
        sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(pts, pts[1:]))
    )


def _spread(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def analyze(
    data: Dict[Tuple, Any], audits: Dict[str, Any], manifest: Dict[str, Any]
) -> Dict[str, Any]:
    conf, family, m = _levels(manifest)
    an = manifest["analysis"]
    seeds = manifest["seeds"]
    index = seed_index(len(seeds), an["bootstrap_resamples"], an["bootstrap_seed"])
    budgets = manifest["checkpoints"]
    lo, hi = an["auc_budget_range"]
    key_of = "{:.6g}".format
    conditions = []
    for env in manifest["envs"]:
        name = env["env_name"]
        audit = audits[name]["data"]["reps"]
        for rep in manifest["representations"]:
            cells = [data[(name, rep, s)] for s in seeds]
            floor = audit[rep]["floor"]
            counts = audit[rep]["counts"]
            final = [c["checkpoints"][-1]["evals"] for c in cells]
            primary = paired_estimate(
                [f["ftl_final"]["returns"] for f in final],
                [f["bc_iid_final"]["returns"] for f in final],
                index,
                [conf, family],
            )
            mixture = paired_estimate(
                [f["ftl_mixture"]["returns"] for f in final],
                [f["ftl_final"]["returns"] for f in final],
                index,
                [conf],
            )
            curves: Dict[str, Any] = {
                arm: {
                    "mean_return": [],
                    "on_policy_disagreement": [],
                    "reference_disagreement": [],
                    "training_expert_action_entries": [],
                }
                for arm in ARMS
            }
            curves["fixed_bc"] = {
                "alias_of": "bc_iid_final",
                "logical_standalone_expert_action_entries": [],
                "logical_standalone_env_steps": [],
            }
            paired_curve = []
            per_seed_curve = {arm: [[] for _ in seeds] for arm in ARMS}
            for i, b in enumerate(budgets):
                cps = [c["checkpoints"][i] for c in cells]
                for arm in ARMS:
                    means = [cp["evals"][arm]["mean_return"] for cp in cps]
                    for s, v in enumerate(means):
                        per_seed_curve[arm][s].append(v)
                    cur = curves[arm]
                    cur["mean_return"].append(
                        bootstrap_mean(np.array(means), index, [conf])
                    )
                    cur["on_policy_disagreement"].append(
                        bootstrap_mean(
                            np.array(
                                [cp["evals"][arm]["mean_disagreement"] for cp in cps]
                            ),
                            index,
                            [conf],
                        )
                    )
                    if arm == "ftl_mixture":
                        # Exact average over the uniform policy draw on the
                        # audit sample, not over the sampled mixture indices.
                        ref = [
                            float(
                                np.mean(
                                    [
                                        reference_disagreement(t, counts)
                                        for t in cp["history"]
                                    ]
                                )
                            )
                            for cp in cps
                        ]
                    else:
                        ref = [
                            reference_disagreement(cp["tables"][arm], counts)
                            for cp in cps
                        ]
                    cur["reference_disagreement"].append(
                        bootstrap_mean(np.array(ref), index, [conf])
                    )
                    src = "bc_iid" if arm == "bc_iid_final" else "ftl"
                    cur["training_expert_action_entries"].append(
                        _spread([cp["oracle_entries"][src] for cp in cps])
                    )
                curves["fixed_bc"]["logical_standalone_expert_action_entries"].append(
                    _spread([cp["standalone"]["expert_action_entries"] for cp in cps])
                )
                curves["fixed_bc"]["logical_standalone_env_steps"].append(
                    _spread([cp["standalone"]["env_steps"] for cp in cps])
                )
                paired_curve.append(
                    bootstrap_mean(
                        np.array(
                            [
                                cp["evals"]["ftl_final"]["mean_return"]
                                - cp["evals"]["bc_iid_final"]["mean_return"]
                                for cp in cps
                            ]
                        ),
                        index,
                        [conf],
                    )
                )
            auc = {
                arm: bootstrap_mean(
                    np.array([_auc(budgets, v, lo, hi) for v in per_seed_curve[arm]]),
                    index,
                    [conf],
                )
                for arm in ARMS
            }
            auc["ftl_minus_bc_iid"] = bootstrap_mean(
                np.array(
                    [
                        _auc(budgets, f, lo, hi) - _auc(budgets, b, lo, hi)
                        for f, b in zip(
                            per_seed_curve["ftl_final"], per_seed_curve["bc_iid_final"]
                        )
                    ]
                ),
                index,
                [conf],
            )
            conditions.append(
                {
                    "env": name,
                    "representation": rep,
                    "eligibility": {
                        "positive_bound": floor["positive_certificate"],
                        "label": (
                            "agnostic on the reference distribution: positive "
                            "uniform lower bound (delta {:.4g})".format(floor["delta"])
                            if floor["positive_certificate"]
                            else "capacity-restricted; misspecification unverified"
                        ),
                        "reference_floor": floor,
                    },
                    "primary": {
                        "estimand": "FTL final post-update mean return minus BC-iid "
                        "mean return at B={}, paired by training seed "
                        "(native units, higher is better)".format(budgets[-1]),
                        "pointwise_level": conf,
                        "family_level": family,
                        "family_size": m,
                        "interval_keys": {
                            "pointwise": key_of(conf),
                            "family": key_of(family),
                        },
                        "seeds": seeds,
                        **primary,
                    },
                    "mixture_minus_final_at_final_budget": mixture,
                    "curves": {
                        "budgets": budgets,
                        "arms": curves,
                        "ftl_minus_bc_iid": paired_curve,
                    },
                    "auc": {
                        "budget_range": [lo, hi],
                        "units": "native return x log2(retained labels), trapezoid "
                        "over measured checkpoints; divide by {} for the mean "
                        "return over the range".format(math.log2(hi) - math.log2(lo)),
                        "estimates": auc,
                    },
                }
            )
    # Reported, not enforced: the operator selected envs before confirmation.
    selection = [
        {
            "env": e["env_name"],
            "any_positive_bound": any(
                c["eligibility"]["positive_bound"]
                for c in conditions
                if c["env"] == e["env_name"]
            ),
        }
        for e in manifest["envs"]
    ]
    return {
        "conditions": conditions,
        "env_selection_check": selection,
        "n_seeds": len(seeds),
        "bootstrap": {
            "resamples": an["bootstrap_resamples"],
            "seed": an["bootstrap_seed"],
            "unit": "whole training seed (one index matrix shared "
            "by every condition)",
        },
    }


def accounting(
    data: Dict[Tuple, Any], audits: Dict[str, Any], manifest: Dict[str, Any]
) -> Dict[str, Any]:
    zero = dict.fromkeys(INT_COSTS, 0)
    per_condition = []
    all_cells = zero
    for env in manifest["envs"]:
        for rep in manifest["representations"]:
            cells = [data[(env["env_name"], rep, s)] for s in manifest["seeds"]]
            parts = {
                name: _add(zero, *[c["costs"][name] for c in cells])
                for name in ("train_ftl", "train_bc_iid", "train_fixed_bc", "eval")
            }
            training = _add(
                parts["train_ftl"], parts["train_bc_iid"], parts["train_fixed_bc"]
            )
            standalone = _add(
                zero, *[c["checkpoints"][-1]["standalone"] for c in cells]
            )
            everything = _add(training, parts["eval"])
            all_cells = _add(all_cells, everything)
            per_condition.append(
                {
                    "env": env["env_name"],
                    "representation": rep,
                    "training": parts,
                    "physical_training": training,
                    "evaluation": parts["eval"],
                    "physical_all": everything,
                    "logical_standalone_fixed_bc_at_final_budget": standalone,
                }
            )
    reference = {
        name: _add(
            a["data"]["costs"]["expert_episodes"], a["data"]["costs"]["random_episodes"]
        )
        for name, a in audits.items()
    }
    return {
        "per_condition": per_condition,
        "reference_audits": reference,
        "known_recorded_total": _add(all_cells, *reference.values()),
        "expert_preparation": "unavailable: expert training counters are not "
        "recorded; not zero and not included in any total",
        "notes": [
            "Physical training counts fixed BC's own fits once and shares BC-iid "
            "acquisition, which is counted once.",
            "The logical standalone fixed BC cost is what a separate offline run "
            "at B would spend; it is not added to physical totals.",
            "Evaluation and reference-audit costs are diagnostic and separate "
            "from training.",
            "Totals are known recorded components only; pilots and expert "
            "preparation are excluded.",
        ],
    }


# ---------------------------------------------------------------------------
# Figures and report
# ---------------------------------------------------------------------------


def make_figures(
    results: Dict[str, Any], manifest: Dict[str, Any], out: pathlib.Path
) -> List[Dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf = manifest["analysis"]["confidence_level"]
    key = "{:.6g}".format(conf)
    envs = [e["env_name"] for e in manifest["envs"]]
    reps = manifest["representations"]
    style = {
        "ftl_final": dict(color=FIGURE_COLORS[0], marker="o", linestyle="-"),
        "bc_iid_final": dict(color=FIGURE_COLORS[1], marker="s", linestyle="-"),
        "fixed_bc": dict(
            color=FIGURE_COLORS[2],
            marker="D",
            linestyle="none",
            markerfacecolor="none",
            markersize=9,
        ),
        "ftl_mixture": dict(color=FIGURE_COLORS[3], marker="^", linestyle="--"),
    }
    by = {(c["env"], c["representation"]): c for c in results["conditions"]}
    written = []
    for stem, xlabel in (
        ("return_vs_retained_labels", "retained labeled training observations B"),
        (
            "return_vs_training_expert_entries",
            "training expert action entries actually spent (mean over seeds)",
        ),
    ):
        fig, axes = plt.subplots(
            len(envs),
            len(reps),
            squeeze=False,
            figsize=(4.4 * len(reps), 3.4 * len(envs)),
            constrained_layout=True,
        )
        for i, env in enumerate(envs):
            for j, rep in enumerate(reps):
                ax, c = axes[i][j], by[(env, rep)]
                curves = c["curves"]["arms"]
                for arm in ("ftl_final", "bc_iid_final", "fixed_bc", "ftl_mixture"):
                    src = "bc_iid_final" if arm == "fixed_bc" else arm
                    y = [e["estimate"] for e in curves[src]["mean_return"]]
                    if stem.startswith("return_vs_retained"):
                        x = c["curves"]["budgets"]
                    elif arm == "fixed_bc":
                        x = [
                            e["mean"]
                            for e in curves["fixed_bc"][
                                "logical_standalone_expert_action_entries"
                            ]
                        ]
                    else:
                        x = [
                            e["mean"]
                            for e in curves[arm]["training_expert_action_entries"]
                        ]
                    ax.plot(x, y, label=ARM_LABELS[arm], linewidth=2, **style[arm])
                    ivs = [e["intervals"] for e in curves[src]["mean_return"]]
                    if arm != "fixed_bc" and all(iv is not None for iv in ivs):
                        ax.fill_between(
                            x,
                            [iv[key][0] for iv in ivs],
                            [iv[key][1] for iv in ivs],
                            color=style[arm]["color"],
                            alpha=0.15,
                            linewidth=0,
                        )
                ax.set_xscale("log", base=2)
                ax.set_title(
                    "{} / {} ({})".format(
                        env,
                        rep,
                        (
                            "positive bound"
                            if c["eligibility"]["positive_bound"]
                            else "misspecification unverified"
                        ),
                    ),
                    fontsize=9,
                )
                ax.set_xlabel(xlabel, fontsize=8)
                ax.set_ylabel(
                    "mean episode return (native, higher is better)", fontsize=8
                )
                ax.grid(alpha=0.3, linewidth=0.5)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
        fig.suptitle(
            "Final policies and FTL mixture; bands are pointwise {:.0%} "
            "seed-bootstrap intervals. Points are measured checkpoints only.".format(
                conf
            ),
            fontsize=9,
        )
        for ext in ("png", "pdf"):
            path = out / "{}.{}".format(stem, ext)
            fig.savefig(path, dpi=150)
            written.append(
                {
                    "path": "figures/{}".format(path.name),
                    "sha256": pilot.sha256_file(path),
                }
            )
        plt.close(fig)
    return written


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else "{:.4g}".format(x)


def _iv(est: Dict[str, Any], key: str) -> str:
    ivs = est.get("intervals")
    if not ivs:
        return "n/a (fewer than 2 seeds)"
    lo, hi = ivs[key]
    return "[{}, {}]".format(_fmt(lo), _fmt(hi))


def write_report(summary: Dict[str, Any], path: pathlib.Path) -> None:
    man = summary["manifest"]["content"]
    lines = [
        "# Classical agnostic confirmation analysis",
        "",
        "Status: **{}**. Manifest sha256 `{}`.".format(
            summary["analysis_status"], summary["manifest"]["content_sha256"]
        ),
        "",
        "## Hypothesis and design",
        "",
        "Hypothesis under test: at the frozen final retained-label budget, "
        "FTL-DAgger (learner-driven collection, expert labels only at selected "
        "states) reaches higher task return than BC-iid (expert-driven "
        "collection) for the same restricted learner. An FTL advantage is not "
        "assumed; a null or negative difference is a legitimate outcome.",
        "",
        "Aliasing is imposed by a fixed, predeclared quantizer: the learner sees "
        "only a bin id, and every arm fits the exact majority table (0-1 ERM, ties "
        "to the lowest action, empty bins action 0). Mild keeps velocity signs; "
        "severe drops them. Fixed BC fits the shared BC-iid prefix and is an "
        "equality control and evaluation alias, not an independent replicate.",
        "",
        "Environments (selected by the operator before confirmation, in candidate "
        "order {}): {}. Selection rule: {}".format(
            man["candidate_env_order"],
            [e["env_name"] for e in man["envs"]],
            man["selection_rule"],
        ),
        "",
        "## Shared provenance",
        "",
        "Every analyzed cell and audit record must match these manifest pins: "
        "stage 2 source {}, stage 1 source `{}`, producer package versions {}.".format(
            man["source_stage2_files"],
            man["source_stage1_sha256"],
            man["package_versions"],
        ),
        "",
        "## Inventory",
        "",
        "Cells complete: {} of {}. Audits complete: {} of {}.".format(
            summary["inventory"]["cells_complete"],
            summary["inventory"]["cells_expected"],
            summary["inventory"]["audits_complete"],
            summary["inventory"]["audits_expected"],
        ),
        "",
    ]
    if summary["failures"]:
        lines += ["Problems (no estimates are reported):", ""]
        lines += [
            "- {}: {}".format(f["where"], f["detail"]) for f in summary["failures"]
        ]
        lines.append("")
    res = summary["results"]
    if res is None:
        lines += [
            "No primary estimates or figures are produced unless every "
            "declared cell and audit is complete and consistent.",
            "",
        ]
    else:
        lines += [
            "## Expert audit qualifications",
            "",
            "Each audit uses one frozen reference sample (half expert-driven, half "
            "uniform-random episodes, one uniform state per episode, expert "
            "labels). Its labels are diagnostic only and never reach training.",
            "",
            "| env | rep | K | n | empirical floor | uniform slack | lower bound | "
            "status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in res["conditions"]:
            f = c["eligibility"]["reference_floor"]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    c["env"],
                    c["representation"],
                    f["K"],
                    f["n"],
                    _fmt(f["empirical_min_disagreement"]),
                    _fmt(f["slack"]),
                    _fmt(f["lower_bound"]),
                    c["eligibility"]["label"],
                )
            )
        pk = res["conditions"][0]["primary"]["interval_keys"]
        lines += [
            "",
            "A non-positive bound means the condition is capacity-restricted with "
            "misspecification unverified, not agnostic. The uniform slack is the "
            "audit's own population uncertainty; the seed intervals below are "
            "conditional on the frozen reference sample and do not cover it.",
            "",
        ]
        for s in res["env_selection_check"]:
            if not s["any_positive_bound"]:
                lines += [
                    "Protocol note: {} has no positive bound in any analyzed "
                    "representation, which the declared selection rule "
                    "requires; all its results are capacity-restricted with "
                    "misspecification unverified.".format(s["env"]),
                    "",
                ]
        lines += [
            "## Primary paired estimates",
            "",
            "FTL final minus BC-iid mean return at B={} in native units (higher "
            "is better), {} training seeds. Intervals resample whole training "
            "seeds ({} resamples); evaluation episodes are never treated as "
            "training replicates. Family intervals are approximate Bonferroni over "
            "{} primary tests.".format(
                man["checkpoints"][-1],
                res["n_seeds"],
                man["analysis"]["bootstrap_resamples"],
                res["conditions"][0]["primary"]["family_size"],
            ),
            "",
            "| env | rep | estimate | pointwise {} | family {} | seed SD | "
            "episode MC SE of mean |".format(pk["pointwise"], pk["family"]),
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in res["conditions"]:
            p = c["primary"]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    c["env"],
                    c["representation"],
                    _fmt(p["estimate"]),
                    _iv(p, pk["pointwise"]),
                    _iv(p, pk["family"]),
                    _fmt(p["seed_sd"]),
                    _fmt(p["episode_mc_se_of_seed_mean"]),
                )
            )
            if p.get("degenerate"):
                lines.append("")
                lines.append(
                    "Note ({}/{}): {}".format(
                        c["env"], c["representation"], p["interval_note"]
                    )
                )
        lines += [
            "",
            "The episode Monte Carlo SE is a within-run precision diagnostic. The "
            "seed bootstrap already includes the observed variation in evaluation "
            "means; it does not isolate training-only uncertainty, and the two are "
            "not combined.",
            "",
            "## Mixture versus final policy",
            "",
            "| env | rep | FTL mixture minus FTL final at final B | pointwise |",
            "| --- | --- | --- | --- |",
        ]
        for c in res["conditions"]:
            mx = c["mixture_minus_final_at_final_budget"]
            lines.append(
                "| {} | {} | {} | {} |".format(
                    c["env"],
                    c["representation"],
                    _fmt(mx["estimate"]),
                    _iv(mx, pk["pointwise"]),
                )
            )
        lines += [
            "",
            "Mixture returns come from sampled policy indices per evaluation "
            "episode, so they estimate, not compute, the mixture value. Its "
            "common-reference disagreement in `summary.json` is the exact uniform "
            "average over its policies on the frozen audit sample.",
            "",
            "## Costs",
            "",
            "| env | rep | physical training expert entries | physical training "
            "env steps | fits | eval expert entries | standalone fixed BC expert "
            "entries at final B |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in summary["accounting"]["per_condition"]:
            t = c["physical_training"]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    c["env"],
                    c["representation"],
                    t["expert_action_entries"],
                    t["env_steps"],
                    t["fits"],
                    c["evaluation"]["expert_action_entries"],
                    c["logical_standalone_fixed_bc_at_final_budget"][
                        "expert_action_entries"
                    ],
                )
            )
        lines += [""] + ["- " + n for n in summary["accounting"]["notes"]]
        lines += [
            "- Expert preparation: {}.".format(
                summary["accounting"]["expert_preparation"]
            ),
            "",
        ]
        lines += [
            "Equal retained B does not mean equal oracle cost: FTL spends one "
            "expert entry per retained label, while BC-iid spends one per "
            "expert-driven step. The second figure plots measured points against "
            "entries actually spent; performance between points is not measured.",
            "",
        ]
    lines += [
        "## Limits and negative outcomes",
        "",
        "- No agnostic guarantee is claimed for FTL-DAgger, and no FTL win is "
        "assumed. A null, negative or representation-specific difference is "
        "reported as is.",
        "- Classical episodes have variable length; samples target the "
        "episode-normalized state distribution, not fixed-horizon occupancy.",
        "- Reference disagreements are empirical on the frozen audit sample; no "
        "population certainty is attached to them.",
        "- Only the declared environments, representations, seeds and budgets are "
        "analyzed; nothing was selected by effect.",
        "- Operator precondition, not checked here: the operator must verify "
        "queue completion, exit codes and result hashes outside this analysis, "
        "which cannot see the queue.",
        "",
    ]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_analysis(
    manifest_path: pathlib.Path,
    results_root: pathlib.Path,
    audit_root: pathlib.Path,
    output_dir: pathlib.Path,
) -> Tuple[int, Dict[str, Any]]:
    """Validate, analyze and publish. Raises ManifestError/OSError before writing."""
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = validate_manifest(_strict_loads(manifest_bytes.decode("utf-8")))
    except ManifestError:
        raise
    except (UnicodeDecodeError,) + MALFORMED as exc:
        raise ManifestError("{}: {}".format(type(exc).__name__, exc))
    for root in (results_root, audit_root):
        if not root.is_dir():
            raise ManifestError("{} is not a directory".format(root.name))
    if output_dir.exists() or output_dir.is_symlink():
        raise ManifestError("output directory must not exist")
    if not output_dir.parent.is_dir():
        raise ManifestError("output parent directory does not exist")

    audits = {
        e["env_name"]: load_audit(audit_root, e, manifest) for e in manifest["envs"]
    }
    audit_extra = sorted(
        p.name
        for p in audit_root.iterdir()
        if not p.name.startswith(".") and p.name not in audits
    )
    artifacts, data, problems, ignored = load_cells(results_root, manifest, audits)
    failures = [{"where": "results root", "detail": p} for p in problems]
    failures += [
        {"where": "audit root", "detail": "undeclared entry {}".format(n)}
        for n in audit_extra
    ]
    for art in list(audits.values()) + artifacts:
        if art["status"] in ("invalid", "failed_checks", "failed"):
            failures.append({"where": art["path"], "detail": "; ".join(art["reasons"])})
    n_complete = sum(a["status"] == "complete" for a in artifacts)
    a_complete = sum(a["status"] == "complete" for a in audits.values())
    if failures:
        status = "failed"
    elif n_complete < len(artifacts) or a_complete < len(audits):
        status = "incomplete"
    else:
        status = "complete"
    summary: Dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "analysis_status": status,
        "analysis_sources_sha256": {
            pathlib.Path(mod.__file__).name: _module_sha256(mod)
            for mod in ANALYSIS_MODULES
        },
        "manifest": {
            "file_sha256": _sha256(manifest_bytes),
            "content_sha256": _canonical_hash(manifest),
            "content": manifest,
        },
        "inventory": {
            "cells_expected": len(artifacts),
            "cells_complete": n_complete,
            "audits_expected": len(audits),
            "audits_complete": a_complete,
            "cells": artifacts,
            "audits": [
                {k: v for k, v in a.items() if k != "data"} for a in audits.values()
            ],
            "ignored_hidden_entries": ignored,
            "note": "Paths are relative to the results and audit roots. Only "
            "producer-complete, consistent records are analyzed; nothing is "
            "dropped or imputed.",
        },
        "failure_count": len(failures),
        "failures": failures[:MAX_REPORTED_FAILURES],
        "results": None,
        "accounting": None,
        "figures": [],
    }
    output_dir.mkdir()
    if status == "complete":
        summary["results"] = analyze(data, audits, manifest)
        summary["accounting"] = accounting(data, audits, manifest)
        fig_dir = output_dir / "figures"
        fig_dir.mkdir()
        summary["figures"] = make_figures(summary["results"], manifest, fig_dir)
    text = json.dumps(summary, allow_nan=False, indent=1) + "\n"
    (output_dir / "summary.json").write_text(text)
    write_report(summary, output_dir / "report.md")
    code = {"complete": EXIT_OK, "incomplete": EXIT_INCOMPLETE}.get(status, EXIT_FAILED)
    return code, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m imitation.experiments.agnostic.analyze_classical",
        description="Fail-closed analysis of a frozen classical agnostic study.",
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--results-root", required=True)
    p.add_argument("--audit-root", required=True)
    p.add_argument("--output-dir", required=True, help="fresh path; must not exist")
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if not exc.code else EXIT_INVALID
    try:
        code, summary = run_analysis(
            pathlib.Path(args.manifest),
            pathlib.Path(args.results_root),
            pathlib.Path(args.audit_root),
            pathlib.Path(args.output_dir),
        )
    except (ManifestError, OSError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_INVALID
    inv = summary["inventory"]
    print(
        "analysis {}: {} of {} cells and {} of {} audits complete".format(
            summary["analysis_status"],
            inv["cells_complete"],
            inv["cells_expected"],
            inv["audits_complete"],
            inv["audits_expected"],
        )
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
