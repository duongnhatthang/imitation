"""Fail-closed analysis of a frozen confirmation run of the agnostic DAgger toy.

Usage::

    python -m imitation.experiments.agnostic.analyze_toy \\
        --manifest MANIFEST.json --results-root ROOT --output-dir FRESH_DIR

``ROOT`` holds one ``seed-N/result.json`` per training seed, as written by
``imitation.experiments.agnostic.toy``. ``FRESH_DIR`` must not exist; the
analysis writes ``summary.json`` (strict finite JSON), ``report.md`` and, only
when the analysis is complete, ``figures/*.png|pdf``. Exit codes: 0 (analysis
``complete``), 2 (invalid invocation or manifest, nothing written), 3
(``incomplete``: inventory only) and 4 (``failed``: provenance, consistency or
control failure; inventory and reasons only).

The manifest is the scientific protocol frozen before any result is inspected.
It has exactly these keys (no job argv, hostnames or machine data). Example for
the declared confirmation study, whose pilot seeds 0..3 are excluded::

    {
     "schema": "agnostic-toy-confirmation-manifest-v1",
     "study": "Agnostic DAgger toy confirmation",
     "protocol": "agnostic-toy-v1",
     "toy_source_sha256": "<sha256 of toy.py at the frozen revision>",
     "seeds": [1000, 1001, ..., 1099],
     "excluded_pilot_seeds": [0, 1, 2, 3],
     "grid": {"horizons": [16, 64], "alphas": [0.0, 0.02, 0.1, 0.3],
              "kappas": [0.0, 1.0], "qs": [0, 1], "orientations": [0, 1]},
     "budget": 4096,
     "batch": 16,
     "checkpoints": [128, 256, 512, 1024, 2048, 4096],
     "annotation_mode": "deferred",
     "tie_order": [0, 1, 2, 3],
     "analysis": {"inference_orientation": 0, "confidence_level": 0.95,
                  "bootstrap_resamples": 10000, "bootstrap_seed": 20260929}
    }

``seeds`` must be written out in full and strictly increasing; the ellipsis
above is documentation only.

A seed enters inference only if its producer status is ``complete``, every
cell is ``complete``, no cell overran the deadline, there is no partial cell,
no error and no deadline overshoot, and it started, then ended before its own
configured deadline (deadlines may differ between seeds). Anything else is
reported as incomplete with its reasons, never promoted.

Operational precondition: the analyzer sees only ``ROOT``, not the job queue.
Before invoking it, the operator must independently verify that every queued
job reached a complete, exit-validated state and that each ``result.json``
hash matches the queue's record; the analyzer does not and cannot check this.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

MANIFEST_SCHEMA = "agnostic-toy-confirmation-manifest-v1"
SUMMARY_SCHEMA = "agnostic-toy-analysis-v1"
EXIT_OK, EXIT_INVALID, EXIT_INCOMPLETE, EXIT_FAILED = 0, 2, 3, 4
MANIFEST_KEYS = {
    "schema",
    "study",
    "protocol",
    "toy_source_sha256",
    "seeds",
    "excluded_pilot_seeds",
    "grid",
    "budget",
    "batch",
    "checkpoints",
    "annotation_mode",
    "tie_order",
    "analysis",
}
GRID_KEYS = ("horizons", "alphas", "kappas", "qs", "orientations")
ANALYSIS_KEYS = {
    "inference_orientation",
    "confidence_level",
    "bootstrap_resamples",
    "bootstrap_seed",
}
CONFIG_KEYS = set(GRID_KEYS) | {
    "seeds",
    "budget",
    "batch",
    "checkpoints",
    "annotation_mode",
    "deadline",
    "tie_order",
}
ANNOTATION_MODES = ("deferred", "full_trajectory")
SEED_DIR = re.compile(r"^seed-(\d+)$")
MAX_REPORTED_FAILURES = 200
FIGURE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # fixed order by alpha


class ManifestError(ValueError):
    pass


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_real(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _strict_loads(text: str) -> Any:
    def reject(token):
        raise ValueError("non-finite JSON constant {}".format(token))

    return json.loads(text, parse_constant=reject)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256(text.encode())


def _schedule_ok(budget: int, batch: int, checkpoints: Sequence[int]) -> bool:
    return (
        len(checkpoints) > 0
        and all(_is_int(c) for c in checkpoints)
        and checkpoints[0] >= 1
        and all(b > a for a, b in zip(checkpoints, checkpoints[1:]))
        and checkpoints[-1] == budget
        and all(c % batch == 0 for c in checkpoints[:-1])
    )


def validate_manifest(manifest: Any) -> Dict[str, Any]:
    """Validate the exact manifest schema and return it unchanged."""

    def need(ok: bool, message: str) -> None:
        if not ok:
            raise ManifestError(message)

    need(isinstance(manifest, dict), "manifest must be a JSON object")
    need(
        set(manifest) == MANIFEST_KEYS,
        "manifest keys must be exactly {}".format(sorted(MANIFEST_KEYS)),
    )
    need(manifest["schema"] == MANIFEST_SCHEMA, "unknown manifest schema")
    for key in ("study", "protocol"):
        value = manifest[key]
        need(isinstance(value, str) and 0 < len(value) <= 200, key + " invalid")
    sha = manifest["toy_source_sha256"]
    need(
        isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) is not None,
        "toy_source_sha256 must be 64 lowercase hex digits",
    )
    for key in ("seeds", "excluded_pilot_seeds"):
        seeds = manifest[key]
        need(
            isinstance(seeds, list)
            and all(_is_int(s) and s >= 0 for s in seeds)
            and all(b > a for a, b in zip(seeds, seeds[1:])),
            key + " must be strictly increasing nonnegative integers",
        )
    need(len(manifest["seeds"]) > 0, "seeds must be nonempty")
    need(
        not set(manifest["seeds"]) & set(manifest["excluded_pilot_seeds"]),
        "pilot seeds cannot be confirmation seeds",
    )
    grid = manifest["grid"]
    need(
        isinstance(grid, dict) and set(grid) == set(GRID_KEYS),
        "grid keys must be exactly {}".format(list(GRID_KEYS)),
    )
    checks = {
        "horizons": lambda v: _is_int(v) and v >= 1,
        "alphas": lambda v: _is_real(v) and 0 <= v < 0.5,
        "kappas": lambda v: _is_real(v) and 0 <= v <= 1,
        "qs": lambda v: _is_int(v) and v in (0, 1),
        "orientations": lambda v: _is_int(v) and v in (0, 1),
    }
    for key, ok in checks.items():
        values = grid[key]
        need(
            isinstance(values, list)
            and values
            and all(ok(v) for v in values)
            and all(b > a for a, b in zip(values, values[1:])),
            "grid.{} must be nonempty, valid and strictly increasing".format(key),
        )
    need(grid["qs"] == [0, 1], "both q variants are required for balanced q")
    need(grid["orientations"] == [0, 1], "both orientations are required")
    need(1 in grid["kappas"], "kappa 1 is required for the primary family")
    need(any(a > 0 for a in grid["alphas"]), "a positive alpha is required")
    budget, batch, checkpoints = (
        manifest["budget"],
        manifest["batch"],
        manifest["checkpoints"],
    )
    need(_is_int(budget) and budget >= 1, "budget must be a positive integer")
    need(_is_int(batch) and batch >= 1, "batch must be a positive integer")
    need(
        isinstance(checkpoints, list) and _schedule_ok(budget, batch, checkpoints),
        "checkpoints must increase to the budget in multiples of batch",
    )
    need(manifest["annotation_mode"] in ANNOTATION_MODES, "unknown annotation_mode")
    need(
        isinstance(manifest["tie_order"], list)
        and all(_is_int(p) for p in manifest["tie_order"])
        and sorted(manifest["tie_order"]) == [0, 1, 2, 3],
        "tie_order must be a permutation of 0..3",
    )
    analysis = manifest["analysis"]
    need(
        isinstance(analysis, dict) and set(analysis) == ANALYSIS_KEYS,
        "analysis keys must be exactly {}".format(sorted(ANALYSIS_KEYS)),
    )
    need(analysis["inference_orientation"] in (0, 1), "bad inference_orientation")
    need(
        _is_real(analysis["confidence_level"]) and 0 < analysis["confidence_level"] < 1,
        "confidence_level must be in (0, 1)",
    )
    need(
        _is_int(analysis["bootstrap_resamples"])
        and analysis["bootstrap_resamples"] >= 100,
        "bootstrap_resamples must be an integer >= 100",
    )
    need(
        _is_int(analysis["bootstrap_seed"]) and analysis["bootstrap_seed"] >= 0,
        "bootstrap_seed must be a nonnegative integer",
    )
    return manifest


# ---------------------------------------------------------------------------
# Independent reference: the contract's scalar recurrence, not the toy's DP.


def _canonical(p: int) -> Tuple[int, int]:
    return p >> 1, p & 1


def reference(horizon: int, alpha: float, kappa: float, q: int) -> Dict[str, Any]:
    """Class costs, occupancy and population losses F[b, p] by recurrence."""
    e = [alpha if _canonical(p)[0] == 0 else 1.0 - alpha for p in range(4)]
    w = [float(_canonical(p)[1] != q) for p in range(4)]
    costs, recovery_mean, rs = [], [], []
    for b in range(4):
        r, total, seq = 0.0, 0.0, []
        for _ in range(horizon):
            seq.append(r)
            total += e[b] * (1.0 - r) + w[b] * r
            r = kappa * e[b] * (1.0 - r) + w[b] * r
        costs.append(total)
        recovery_mean.append(sum(seq) / horizon)
        rs.append(seq)
    f = np.array(
        [
            [
                sum((1.0 - r) * e[p] + r * w[p] for r in rs[b]) / horizon
                for p in range(4)
            ]
            for b in range(4)
        ]
    )
    closed = None
    if kappa == 0:
        closed = horizon * alpha
    elif kappa == 1:
        closed = alpha * horizon / (1 + alpha) + alpha**2 / (1 + alpha) ** 2 * (
            1 - (-alpha) ** horizon
        )
    return {
        "costs": costs,
        "recovery_mean": recovery_mean,
        "F": f,
        "expert_floor": min(e),
        "closed_optimum": closed,
        "expert_disagreement": e,
    }


def reference_erm(counts: Any, orientation: int, tie_order: Sequence[int]) -> int:
    """Exact 0-1 ERM on counts[mode][physical label]; first in tie_order wins."""
    best, best_mistakes = None, None
    for p in tie_order:
        nominal, recovery = (x ^ orientation for x in _canonical(p))
        mistakes = counts[0][1 - nominal] + counts[1][1 - recovery]
        if best_mistakes is None or mistakes < best_mistakes:
            best, best_mistakes = p, mistakes
    return int(best)


# ---------------------------------------------------------------------------
# Validation of result files.


class Checks:
    """Named consistency checks; every comparison is counted, failures kept."""

    def __init__(self, tol_scale: float = 1e-9):
        self.tol_scale = tol_scale
        self.counts: Dict[str, int] = {}
        self.failures: List[Dict[str, Any]] = []

    def check(self, name: str, ok: bool, where: str, detail: str = "") -> bool:
        self.counts[name] = self.counts.get(name, 0) + 1
        if not ok:
            self.failures.append({"check": name, "where": where, "detail": detail})
        return ok

    def close(self, name: str, a: Any, b: Any, scale: float, where: str) -> bool:
        ok = _is_real(a) and _is_real(b)
        ok = ok and abs(a - b) <= self.tol_scale * max(1.0, scale)
        return self.check(name, ok, where, "{!r} != {!r}".format(a, b))

    def summary(self) -> Dict[str, Any]:
        failed: Dict[str, int] = {}
        for f in self.failures:
            failed[f["check"]] = failed.get(f["check"], 0) + 1
        return {
            name: {
                "comparisons": n,
                "failures": failed.get(name, 0),
                "status": "fail" if failed.get(name) else "pass",
            }
            for name, n in sorted(self.counts.items())
        }


def _key(h: int, alpha: float, kappa: float, q: int, u: int) -> Tuple:
    return (int(h), float(alpha), float(kappa), int(q), int(u))


def expected_cells(manifest: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
    """Cell identities in the toy's product order for one seed."""
    g = manifest["grid"]
    return [
        {"seed": seed, "horizon": h, "alpha": a, "kappa": k, "q": q, "orientation": u}
        for h in g["horizons"]
        for a in g["alphas"]
        for k in g["kappas"]
        for q in g["qs"]
        for u in g["orientations"]
    ]


def _same_numbers(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_numbers(x, y) for x, y in zip(a, b))
    if isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return type(a) is type(b) and a == b
    return _is_real(a) and _is_real(b) and a == b


def _provenance_problems(result: Any, manifest: Dict[str, Any], seed: int) -> List[str]:
    if not isinstance(result, dict):
        return ["result is not a JSON object"]
    problems = []
    if result.get("protocol") != manifest["protocol"]:
        problems.append("protocol mismatch")
    source = result.get("source")
    if result.get("source_sha256") != manifest["toy_source_sha256"] or not (
        isinstance(source, dict)
        and source.get("sha256") == manifest["toy_source_sha256"]
    ):
        problems.append("toy source sha256 mismatch")
    if not (_is_int(result.get("seed")) and result["seed"] == seed):
        problems.append("top-level seed does not match directory label")
    config = result.get("config")
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        return problems + ["config keys differ from the toy config schema"]
    expected = dict(manifest["grid"])
    expected.update(
        seeds=[seed],
        budget=manifest["budget"],
        batch=manifest["batch"],
        checkpoints=manifest["checkpoints"],
        annotation_mode=manifest["annotation_mode"],
        tie_order=manifest["tie_order"],
    )
    # The deadline is operational (when to stop), not part of the science.
    for key, value in expected.items():
        if not _same_numbers(config[key], value):
            problems.append("config.{} differs from the manifest".format(key))
    return problems


def _cell_where(label: str, c: Dict[str, Any]) -> str:
    return "{} H={} alpha={} kappa={} q={} orientation={}".format(
        label, c["horizon"], c["alpha"], c["kappa"], c["q"], c["orientation"]
    )


def _validate_cell(
    record: Dict[str, Any],
    cell: Dict[str, Any],
    manifest: Dict[str, Any],
    checks: Checks,
    where: str,
) -> None:
    h, alpha, kappa, q, u = (
        cell["horizon"],
        cell["alpha"],
        cell["kappa"],
        cell["q"],
        cell["orientation"],
    )
    batch, budget = manifest["batch"], manifest["budget"]
    tie_order = manifest["tie_order"]
    deferred = manifest["annotation_mode"] == "deferred"
    ref = reference(h, alpha, kappa, q)
    ex = record["exact"]
    costs = ex["class_costs"]
    optimum = min(ref["costs"])
    checks.check("structure", len(costs) == 4, where, "class_costs length")
    for p in range(4):
        checks.close("exact_dp", costs[p], ref["costs"][p], h, where)
        checks.close(
            "exact_dp",
            ex["recovery_occupancy_mean"][p],
            ref["recovery_mean"][p],
            1.0,
            where,
        )
        checks.check("certificate", costs[p] >= -1e-12, where, "negative class cost")
    checks.close("exact_dp", ex["class_optimum_cost"], optimum, h, where)
    opt_id = ex["class_optimum_policy_id"]
    checks.check("structure", _is_int(opt_id) and 0 <= opt_id <= 3, where, "opt id")
    checks.close("exact_dp", ref["costs"][opt_id], optimum, h, where)
    if ref["closed_optimum"] is not None:
        checks.close("closed_form", optimum, ref["closed_optimum"], h, where)
    checks.close("exact_dp", ex["expert_cost"], 0.0, h, where)
    checks.close("certificate", ex["expert_occupancy_floor"], alpha, 1.0, where)
    if alpha == 0:
        checks.close("alpha0_certificate", ex["expert_occupancy_floor"], 0.0, 1, where)
        checks.close("alpha0_certificate", ex["class_optimum_cost"], 0.0, h, where)
    else:
        checks.check(
            "certificate",
            ex["expert_occupancy_floor"] > 0 and ex["class_optimum_cost"] > 0,
            where,
            "positive alpha needs a positive floor",
        )

    rounds_total = -(-budget // batch)
    checks.check("structure", record["rounds_completed"] == rounds_total, where)
    checks.check("structure", record.get("in_flight") is None, where, "in_flight")
    snaps = record["checkpoints"]
    checks.check(
        "structure", len(snaps) == len(manifest["checkpoints"]), where, "checkpoints"
    )
    f = ref["F"]

    def arm_costs(arm: Dict[str, Any], pid: int, prefix: str, name: str) -> None:
        checks.close(name, arm[prefix + "cost"], ref["costs"][pid], h, where)
        checks.close(
            name, arm[prefix + "expert_relative_cost"], ref["costs"][pid], h, where
        )
        checks.close(
            name, arm[prefix + "class_excess"], ref["costs"][pid] - optimum, h, where
        )

    for j, (c, snap) in enumerate(zip(manifest["checkpoints"], snaps)):
        at = "{} B={}".format(where, c)
        n = -(-c // batch)
        checks.check(
            "structure",
            snap["retained_labels"] == c
            and snap["rounds"] == n
            and snap["is_checkpoint"],
            at,
            "checkpoint identity",
        )
        ftl, bc, fixed = snap["ftl"], snap["bc_iid"], snap["bc_fixed"]
        shared = snap["shared_expert_acquisition"]
        # Exact refits from stored retained counts.
        for name, arm, id_key in (
            ("ftl", ftl, "post_update_policy_id"),
            ("bc_iid", bc, "policy_id"),
            ("bc_fixed", fixed, "policy_id"),
        ):
            counts = arm["retained_counts"]
            checks.check(
                "erm_refit",
                sum(map(sum, counts)) == c
                and arm[id_key] == reference_erm(counts, u, tie_order),
                at,
                name + " policy is not the ERM of its retained counts",
            )
        pid = ftl["post_update_policy_id"]
        arm_costs(ftl, pid, "post_update_", "exact_dp")
        checks.close(
            "exact_dp", ftl["post_update_onpolicy_disagreement"], f[pid, pid], 1, at
        )
        checks.close("exact_dp", f[pid, pid] * h, ref["costs"][pid], h, at)
        arm_costs(bc, bc["policy_id"], "", "exact_dp")
        checks.close(
            "exact_dp",
            bc["expert_occupancy_disagreement"],
            ref["expert_disagreement"][bc["policy_id"]],
            1.0,
            at,
        )
        # Behavior mixture and the exact regret identity.
        behavior = np.asarray(ftl["behavior_policy_id_counts"], dtype=float)
        checks.check("mixture_identity", behavior.sum() == n, at, "behavior counts")
        mixture = float(behavior @ np.asarray(ref["costs"])) / n
        totals = behavior @ f
        approximation = float(totals.min()) / n
        regret = float(behavior @ np.diag(f)) - float(totals.min())
        checks.close("mixture_identity", ftl["behavior_mixture_cost"], mixture, h, at)
        checks.close(
            "mixture_identity",
            ftl["behavior_mixture_class_excess"],
            mixture - optimum,
            h,
            at,
        )
        checks.close(
            "mixture_identity",
            ftl["behavior_mixture_expert_relative_cost"],
            mixture,
            h,
            at,
        )
        checks.close(
            "mixture_identity", ftl["approximation_term"], approximation, 1, at
        )
        checks.close("mixture_identity", ftl["regret"], regret, n, at)
        checks.close(
            "mixture_identity",
            ftl["behavior_mixture_cost"],
            h * (ftl["approximation_term"] + ftl["regret"] / n),
            h,
            at,
        )
        # Resource ledger: counts implied by the protocol.
        want_ftl = dict(
            rounds=n,
            episodes=c,
            env_steps=c * h,
            retained_labels=c,
            expert_action_entries=c if deferred else c * h,
            fits=n,
        )
        want_shared = dict(
            episodes=c, env_steps=c * h, expert_action_entries=c * h, retained_labels=c
        )
        logical = dict(
            retained_labels=c,
            logical_episodes=c,
            logical_env_steps=c * h,
            logical_expert_action_entries=c * h,
        )
        checks.check("ledger", ftl["counters"] == want_ftl, at, "ftl counters")
        checks.check("ledger", shared == want_shared, at, "shared acquisition")
        checks.check(
            "ledger", bc["counters"] == dict(logical, rounds=n, fits=n), at, "bc_iid"
        )
        checks.check(
            "ledger", fixed["counters"] == dict(logical, fits=j + 1), at, "bc_fixed"
        )
        # Fixed BC equals BC-iid by construction (state control, not evidence).
        checks.check(
            "bc_fixed_equals_bc_iid",
            fixed["policy_id"] == bc["policy_id"]
            and fixed["retained_counts"] == bc["retained_counts"]
            and all(
                fixed[k] == bc[k]
                for k in ("cost", "expert_relative_cost", "class_excess")
            ),
            at,
            "fixed BC differs from BC-iid",
        )
    last = snaps[-1]
    checks.check(
        "structure", record["latest_round"] == last, where, "latest_round != final"
    )
    for arm, final in (
        ("ftl", last["ftl"]["counters"]),
        ("shared_expert", last["shared_expert_acquisition"]),
    ):
        ledger = record["acquisition"][arm]
        checks.check(
            "ledger",
            ledger["rollouts"] == rounds_total
            and all(
                ledger[k] == final[k]
                for k in ("episodes", "env_steps", "expert_action_entries")
            ),
            where,
            arm + " acquisition ledger differs from the final counters",
        )


def _cross_cell_checks(
    cells: Dict[Tuple, Dict[str, Any]], checks: Checks, label: str
) -> None:
    for key, record in cells.items():
        h, alpha, kappa, q, u = key
        where = "{} H={} alpha={} kappa={} q={}".format(label, h, alpha, kappa, q)
        if u == 0 and _key(h, alpha, kappa, q, 1) in cells:
            other = cells[_key(h, alpha, kappa, q, 1)]
            for a, b in zip(record["checkpoints"], other["checkpoints"]):
                flip = [[row[1], row[0]] for row in b["ftl"]["retained_counts"]]
                same = (
                    a["ftl"]["post_update_policy_id"]
                    == b["ftl"]["post_update_policy_id"]
                    and a["ftl"]["behavior_policy_id_counts"]
                    == b["ftl"]["behavior_policy_id_counts"]
                    and a["bc_iid"]["policy_id"] == b["bc_iid"]["policy_id"]
                    and a["ftl"]["retained_counts"] == flip
                    and a["ftl"]["counters"] == b["ftl"]["counters"]
                    and a["bc_iid"]["counters"] == b["bc_iid"]["counters"]
                )
                checks.check(
                    "orientation_invariance",
                    same,
                    "{} B={}".format(where, a["retained_labels"]),
                    "orientation 0 and 1 are not relabelings of each other",
                )
        if kappa == 0:
            for snap in record["checkpoints"]:
                ftl, bc = snap["ftl"], snap["bc_iid"]
                checks.check(
                    "kappa0_pathwise",
                    ftl["post_update_policy_id"] == bc["policy_id"]
                    and ftl["retained_counts"] == bc["retained_counts"]
                    and sum(ftl["retained_counts"][1]) == 0,
                    "{} u={} B={}".format(where, u, snap["retained_labels"]),
                    "no-shift FTL and BC-iid trained outputs differ",
                )


def _scan_root(
    root: pathlib.Path, manifest: Dict[str, Any]
) -> Tuple[Dict[int, str], List[Dict[str, str]], List[str]]:
    """Map seed -> directory name; list ignored entries; report duplicates."""
    found: Dict[int, str] = {}
    ignored, problems = [], []
    expected = set(manifest["seeds"])
    pilots = set(manifest["excluded_pilot_seeds"])
    for entry in sorted(os.listdir(str(root))):
        m = SEED_DIR.match(entry)
        if m is None:
            ignored.append({"entry": entry, "reason": "not a seed-N directory"})
            continue
        seed = int(m.group(1))
        if entry != "seed-{}".format(seed):
            problems.append("{}: non-canonical seed label".format(entry))
        elif seed in found:
            problems.append("{}: duplicate seed".format(entry))
        if seed in expected and entry == "seed-{}".format(seed):
            found[seed] = entry
        elif seed in pilots:
            ignored.append({"entry": entry, "reason": "excluded pilot seed"})
        elif seed not in expected:
            ignored.append({"entry": entry, "reason": "seed not in manifest"})
    return found, ignored, problems


def _partial_indicators(result: Dict[str, Any], n_cells: int) -> List[str]:
    """Every sign that a run is not a clean, on-time, fully finished run."""
    records, inv = result["cells"], result["inventory"]
    reasons = []
    if result["status"] != "complete":
        reasons.append("producer status " + result["status"])
    if result["error"] is not None:
        reasons.append("error recorded: {}".format(result["error"]))
    done = sum(r["status"] == "complete" for r in records)
    if done != n_cells:
        reasons.append("{} of {} cells complete".format(done, n_cells))
    other = sorted({r["status"] for r in records} - {"complete"})
    if other:
        reasons.append("cell statuses {}".format(other))
    if inv["cells_overran_deadline"]:
        reasons.append(
            "cells overran deadline {}".format(inv["cells_overran_deadline"])
        )
    if inv["partial_cell"] is not None:
        reasons.append("partial cell {}".format(inv["partial_cell"]))
    if result["deadline_overshoot"] is not None:
        reasons.append(
            "run ended past its deadline {}".format(result["deadline_overshoot"])
        )
    return reasons


def _utc(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        return None
    return parsed


def _date_problems(result: Dict[str, Any]) -> List[str]:
    """A complete run must have started, then ended before its own deadline,
    which is the configured one and within the campaign cap. Deadlines may
    differ between seeds."""
    names = ("started_utc", "ended_utc", "deadline_utc", "campaign_cap_utc")
    t = {name: _utc(result[name]) for name in names}
    configured = _utc(result["config"]["deadline"])
    bad = [name for name, value in t.items() if value is None]
    if configured is None:
        bad.append("config.deadline")
    if bad:
        return ["not a UTC ISO timestamp: {}".format(bad)]
    problems = []
    if t["deadline_utc"] != configured:
        problems.append("deadline_utc differs from config.deadline")
    if t["deadline_utc"] > t["campaign_cap_utc"]:
        problems.append("deadline is past the campaign cap")
    if not t["started_utc"] <= t["ended_utc"] < t["deadline_utc"]:
        problems.append("not started <= ended < deadline")
    return problems


def load_results(
    root: pathlib.Path, manifest: Dict[str, Any], checks: Checks
) -> Tuple[Dict[int, Dict[Tuple, Dict[str, Any]]], List[Dict[str, Any]], List[str]]:
    """Validate every manifest seed; return complete cells, artifacts, problems."""
    found, ignored, problems = _scan_root(root, manifest)
    artifacts: List[Dict[str, Any]] = []
    data: Dict[int, Dict[Tuple, Dict[str, Any]]] = {}
    n_cells = len(expected_cells(manifest, 0))
    for seed in manifest["seeds"]:
        label = "seed-{}".format(seed)
        art: Dict[str, Any] = {
            "seed_label": label,
            "path": label + "/result.json",
            "sha256": None,
            "status": "missing",
            "reasons": [],
        }
        artifacts.append(art)
        path = root / label / "result.json"
        if seed not in found or not path.is_file():
            art["reasons"].append("result.json not found")
            continue
        raw = path.read_bytes()
        art["sha256"] = _sha256(raw)
        try:
            result = _strict_loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            art.update(status="invalid", reasons=["malformed JSON: {}".format(exc)])
            continue
        reasons = _provenance_problems(result, manifest, seed)
        if reasons:
            art.update(status="invalid", reasons=reasons)
            continue
        status = result.get("status")
        records = result.get("cells")
        if status not in ("complete", "partial_deadline", "partial_error", "running"):
            art.update(status="invalid", reasons=["unknown run status"])
            continue
        if not isinstance(records, list) or len(records) > n_cells:
            art.update(status="invalid", reasons=["cells is not a list of the grid"])
            continue
        failures_before = len(checks.failures)
        try:
            cells = {}
            for index, (record, cell) in enumerate(
                zip(records, expected_cells(manifest, seed))
            ):
                if (
                    record["index"] != index
                    or not _same_numbers(
                        [record["cell"][k] for k in sorted(cell)],
                        [cell[k] for k in sorted(cell)],
                    )
                    or set(record["cell"]) != set(cell)
                ):
                    raise ValueError("cell {} identity or order mismatch".format(index))
                cells[
                    _key(
                        cell["horizon"],
                        cell["alpha"],
                        cell["kappa"],
                        cell["q"],
                        cell["orientation"],
                    )
                ] = (record, cell)
            if result["inventory"]["cells_planned"] != n_cells:
                raise ValueError("inventory cells_planned differs from the grid")
            partial = _partial_indicators(result, n_cells)
            if status != "complete":
                # Never promoted, even when every cell's data are present.
                art["status"] = "incomplete_" + status
                art["reasons"].extend(partial or ["producer status " + status])
                continue
            if partial:
                raise ValueError("status complete contradicts: " + "; ".join(partial))
            if result["inventory"]["cells_complete"] != list(range(n_cells)):
                raise ValueError("inventory does not list every cell exactly once")
            dates = _date_problems(result)
            if dates:
                raise ValueError("inconsistent dates: " + "; ".join(dates))
            for record, cell in cells.values():
                _validate_cell(record, cell, manifest, checks, _cell_where(label, cell))
            plain = {k: rc[0] for k, rc in cells.items()}
            _cross_cell_checks(plain, checks, label)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            art.update(
                status="invalid",
                reasons=["malformed result: {}: {}".format(type(exc).__name__, exc)],
            )
            continue
        if len(checks.failures) > failures_before:
            art.update(
                status="failed_checks",
                reasons=[
                    "{} check failures".format(len(checks.failures) - failures_before)
                ],
            )
            continue
        art["status"] = "complete"
        data[seed] = plain
    return data, artifacts, problems


# ---------------------------------------------------------------------------
# Statistics.


def bootstrap_mean(
    values: np.ndarray, index: Optional[np.ndarray], levels: Sequence[float]
) -> Dict[str, Any]:
    """Point estimate and paired percentile intervals over training-seed blocks."""
    values = np.asarray(values, dtype=float)
    out: Dict[str, Any] = {
        "n_seeds": int(values.size),
        "estimate": float(values.mean()),
    }
    if values.size < 2 or index is None:
        out["intervals"] = None
        out["interval_note"] = "unavailable: fewer than 2 training seeds"
        return out
    means = values[index].mean(axis=1)
    out["intervals"] = {}
    for level in levels:
        lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
        out["intervals"]["{:.6g}".format(level)] = [float(lo), float(hi)]
    out["degenerate"] = bool(np.ptp(values) == 0)
    if out["degenerate"]:
        out["interval_note"] = (
            "all training seeds gave the same value; the zero-width interval "
            "reflects this sample, not a universal result"
        )
    return out


METRICS = (
    "ftl_cost",
    "ftl_class_excess",
    "ftl_expert_relative_cost",
    "mixture_cost",
    "mixture_class_excess",
    "bc_cost",
    "bc_class_excess",
    "bc_expert_relative_cost",
    "diff_final",
    "diff_mixture",
    "regret",
    "approximation_term",
    "ftl_expert_action_entries",
    "ftl_env_steps",
    "ftl_episodes",
    "ftl_fits",
    "shared_expert_action_entries",
    "shared_env_steps",
    "shared_episodes",
    "bc_iid_fits",
    "bc_fixed_fits",
    "retained_labels",
)


def _metrics(snap: Dict[str, Any]) -> Dict[str, float]:
    ftl, bc, shared = snap["ftl"], snap["bc_iid"], snap["shared_expert_acquisition"]
    return {
        "ftl_cost": ftl["post_update_cost"],
        "ftl_class_excess": ftl["post_update_class_excess"],
        "ftl_expert_relative_cost": ftl["post_update_expert_relative_cost"],
        "mixture_cost": ftl["behavior_mixture_cost"],
        "mixture_class_excess": ftl["behavior_mixture_class_excess"],
        "bc_cost": bc["cost"],
        "bc_class_excess": bc["class_excess"],
        "bc_expert_relative_cost": bc["expert_relative_cost"],
        "diff_final": bc["cost"] - ftl["post_update_cost"],
        "diff_mixture": bc["cost"] - ftl["behavior_mixture_cost"],
        "regret": ftl["regret"],
        "approximation_term": ftl["approximation_term"],
        "ftl_expert_action_entries": ftl["counters"]["expert_action_entries"],
        "ftl_env_steps": ftl["counters"]["env_steps"],
        "ftl_episodes": ftl["counters"]["episodes"],
        "ftl_fits": ftl["counters"]["fits"],
        "shared_expert_action_entries": shared["expert_action_entries"],
        "shared_env_steps": shared["env_steps"],
        "shared_episodes": shared["episodes"],
        "bc_iid_fits": bc["counters"]["fits"],
        "bc_fixed_fits": snap["bc_fixed"]["counters"]["fits"],
        "retained_labels": snap["retained_labels"],
    }


def seed_table(
    data: Dict[int, Dict[Tuple, Dict[str, Any]]], manifest: Dict[str, Any]
) -> Dict[Tuple, np.ndarray]:
    """(H, alpha, kappa, stratum, B) -> array[seed, metric] at the inference
    orientation only (never pooling orientations). Stratum is q0, q1 or
    balanced, the within-seed average over q."""
    u = manifest["analysis"]["inference_orientation"]
    g = manifest["grid"]
    seeds = sorted(data)
    out: Dict[Tuple, np.ndarray] = {}
    for h in g["horizons"]:
        for a in g["alphas"]:
            for k in g["kappas"]:
                for j, c in enumerate(manifest["checkpoints"]):
                    per_q = {}
                    for q in g["qs"]:
                        rows = [
                            [
                                _metrics(
                                    data[s][_key(h, a, k, q, u)]["checkpoints"][j]
                                )[m]
                                for m in METRICS
                            ]
                            for s in seeds
                        ]
                        per_q[q] = np.asarray(rows, dtype=float)
                        out[(h, float(a), float(k), "q{}".format(q), c)] = per_q[q]
                    out[(h, float(a), float(k), "balanced", c)] = (
                        per_q[0] + per_q[1]
                    ) / 2.0
    return out


def analyze(
    data: Dict[int, Dict[Tuple, Dict[str, Any]]], manifest: Dict[str, Any]
) -> Dict[str, Any]:
    """Pointwise and primary-family paired bootstrap summaries."""
    an = manifest["analysis"]
    level = an["confidence_level"]
    n_seeds = len(data)
    index = None
    if n_seeds >= 2:
        rng = np.random.default_rng(an["bootstrap_seed"])
        index = rng.integers(0, n_seeds, size=(an["bootstrap_resamples"], n_seeds))
    table = seed_table(data, manifest)
    col = {m: i for i, m in enumerate(METRICS)}
    g = manifest["grid"]
    final = manifest["checkpoints"][-1]
    family = [(h, float(a)) for h in g["horizons"] for a in g["alphas"] if a > 0]
    bonferroni = 1 - (1 - level) / len(family)
    reference_costs = {}

    conditions = []
    for (h, a, k, stratum, c), arr in sorted(table.items(), key=lambda kv: str(kv[0])):
        if stratum == "balanced":
            opt = (
                min(reference(h, a, k, 0)["costs"])
                + min(reference(h, a, k, 1)["costs"])
            ) / 2
        else:
            opt = min(reference(h, a, k, int(stratum[1]))["costs"])
        reference_costs[(h, a, k, stratum)] = opt
        entry = {
            "horizon": h,
            "alpha": a,
            "kappa": k,
            "q": stratum,
            "retained_labels": c,
            "class_optimum_cost": opt,
            "means": {m: float(arr[:, col[m]].mean()) for m in METRICS},
            "bc_minus_ftl_final": bootstrap_mean(
                arr[:, col["diff_final"]], index, [level]
            ),
            "bc_minus_ftl_mixture": bootstrap_mean(
                arr[:, col["diff_mixture"]], index, [level]
            ),
        }
        conditions.append(entry)
    conditions.sort(
        key=lambda e: (
            e["horizon"],
            e["alpha"],
            e["kappa"],
            e["q"],
            e["retained_labels"],
        )
    )
    primary = []
    for h, a in family:
        arr = table[(h, a, 1.0, "balanced", final)]
        primary.append(
            {
                "horizon": h,
                "alpha": a,
                "kappa": 1.0,
                "q": "balanced",
                "retained_labels": final,
                "class_optimum_cost": reference_costs[(h, a, 1.0, "balanced")],
                "ftl_final_class_excess": float(arr[:, col["ftl_class_excess"]].mean()),
                "bc_iid_class_excess": float(arr[:, col["bc_class_excess"]].mean()),
                "bc_minus_ftl_final": bootstrap_mean(
                    arr[:, col["diff_final"]], index, [level, bonferroni]
                ),
            }
        )
    q_contrast = []
    for h in g["horizons"]:
        for a in g["alphas"]:
            for k in g["kappas"]:
                d1 = table[(h, float(a), float(k), "q1", final)][:, col["diff_final"]]
                d0 = table[(h, float(a), float(k), "q0", final)][:, col["diff_final"]]
                q_contrast.append(
                    {
                        "horizon": h,
                        "alpha": float(a),
                        "kappa": float(k),
                        "retained_labels": final,
                        "q1_minus_q0_of_bc_minus_ftl_final": bootstrap_mean(
                            d1 - d0, index, [level]
                        ),
                    }
                )
    return {
        "n_seeds": n_seeds,
        "seed_labels": ["seed-{}".format(s) for s in sorted(data)],
        "inference_orientation": an["inference_orientation"],
        "sign_convention": "bc_minus_ftl = BC-iid cost minus FTL cost; positive "
        "means FTL has lower (better) cost. Class excess differences equal cost "
        "differences because both arms share the exact class optimum.",
        "bootstrap": {
            "method": "percentile bootstrap of the mean over whole training-seed "
            "blocks; both arms and both q variants of a seed are resampled together",
            "resamples": an["bootstrap_resamples"],
            "seed": an["bootstrap_seed"],
            "evaluation_variance": "none: costs are exact dynamic programs",
        },
        "primary_family": {
            "definition": "balanced q, alpha > 0, kappa 1, each horizon, final "
            "retained budget, inference orientation only",
            "size": len(family),
            "pointwise_level": level,
            "bonferroni_level": bonferroni,
            "note": "Bonferroni-adjusted bootstrap intervals are approximate "
            "simultaneous intervals; exact coverage is not claimed.",
            "conditions": primary,
        },
        "q_contrast_secondary": q_contrast,
        "conditions": conditions,
    }


# ---------------------------------------------------------------------------
# Figures and report.


def _series(results: Dict[str, Any], h, a, k, stratum, metric):
    rows = [
        e
        for e in results["conditions"]
        if (e["horizon"], e["alpha"], e["kappa"], e["q"]) == (h, a, k, stratum)
    ]
    return rows, [e["means"][metric] for e in rows]


def make_figures(results: Dict[str, Any], manifest: Dict[str, Any], out: pathlib.Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    g = manifest["grid"]
    colors = {float(a): FIGURE_COLORS[i % 4] for i, a in enumerate(g["alphas"])}
    markers = ("o", "s", "^", "D")
    written = []

    def grid_axes(title: str):
        fig, axes = plt.subplots(
            len(g["horizons"]),
            len(g["kappas"]),
            figsize=(4.2 * len(g["kappas"]), 3.3 * len(g["horizons"])),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(title)
        return fig, axes

    def alpha_handles():
        return [
            Line2D(
                [],
                [],
                color=colors[float(a)],
                marker=markers[i % 4],
                label="alpha = {:g}".format(a),
            )
            for i, a in enumerate(g["alphas"])
        ]

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = out / "{}.{}".format(name, ext)
            fig.savefig(str(path), dpi=150)
            written.append("figures/" + path.name)
        plt.close(fig)

    for name, xlabel, xmetric in (
        ("class_excess_vs_retained_labels", "retained labeled observations B", None),
        (
            "class_excess_vs_expert_entries",
            "expert action entries actually requested",
            "entries",
        ),
    ):
        fig, axes = grid_axes(
            "Agnostic toy: class-relative excess cost vs {} "
            "(balanced q, orientation {})".format(
                "retained labels" if xmetric is None else "expert entries",
                results["inference_orientation"],
            )
        )
        for r, h in enumerate(g["horizons"]):
            for cix, k in enumerate(g["kappas"]):
                ax = axes[r][cix]
                for i, a in enumerate(g["alphas"]):
                    rows, ftl = _series(
                        results, h, float(a), float(k), "balanced", "ftl_class_excess"
                    )
                    _, bc = _series(
                        results, h, float(a), float(k), "balanced", "bc_class_excess"
                    )
                    if xmetric is None:
                        xf = xb = [e["retained_labels"] for e in rows]
                    else:
                        xf = [e["means"]["ftl_expert_action_entries"] for e in rows]
                        xb = [e["means"]["shared_expert_action_entries"] for e in rows]
                    style = dict(
                        color=colors[float(a)],
                        marker=markers[i % 4],
                        markersize=5,
                        linewidth=2,
                    )
                    ax.plot(xf, ftl, linestyle="-", **style)
                    ax.plot(xb, bc, linestyle="--", markerfacecolor="white", **style)
                ax.set_xscale("log", base=2)
                ax.set_title("H = {}, kappa = {:g}".format(h, k))
                ax.set_xlabel(xlabel)
                ax.set_ylabel("J(pi) - J_B* (lower is better)")
                ax.grid(alpha=0.25)
        arms = [
            Line2D([], [], color="#52514e", linestyle="-", label="FTL-DAgger final"),
            Line2D([], [], color="#52514e", linestyle="--", label="BC-iid"),
        ]
        fig.legend(
            handles=alpha_handles() + arms,
            loc="outside lower center",
            ncol=4,
            frameon=False,
        )
        save(fig, name)

    fig, axes = grid_axes(
        "Agnostic toy: paired BC-iid minus FTL-DAgger final cost by q "
        "(positive favors FTL)"
    )
    styles = {"q0": ":", "q1": "-.", "balanced": "-"}
    for r, h in enumerate(g["horizons"]):
        for cix, k in enumerate(g["kappas"]):
            ax = axes[r][cix]
            ax.axhline(0.0, color="#52514e", linewidth=1)
            for i, a in enumerate(g["alphas"]):
                for stratum, ls in styles.items():
                    rows, y = _series(
                        results, h, float(a), float(k), stratum, "diff_final"
                    )
                    x = [e["retained_labels"] for e in rows]
                    ax.plot(
                        x,
                        y,
                        color=colors[float(a)],
                        linestyle=ls,
                        marker=markers[i % 4] if stratum == "balanced" else None,
                        markersize=5,
                        linewidth=2 if stratum == "balanced" else 1.2,
                    )
                    if stratum == "balanced":
                        iv = [e["bc_minus_ftl_final"]["intervals"] for e in rows]
                        if all(v is not None for v in iv):
                            lo = [list(v.values())[0][0] for v in iv]
                            hi = [list(v.values())[0][1] for v in iv]
                            ax.fill_between(
                                x,
                                lo,
                                hi,
                                color=colors[float(a)],
                                alpha=0.15,
                                linewidth=0,
                            )
            ax.set_xscale("log", base=2)
            ax.set_title("H = {}, kappa = {:g}".format(h, k))
            ax.set_xlabel("retained labeled observations B")
            ax.set_ylabel("BC-iid cost - FTL cost")
            ax.grid(alpha=0.25)
    qs = [
        Line2D([], [], color="#52514e", linestyle=ls, label=s)
        for s, ls in styles.items()
    ]
    fig.legend(
        handles=alpha_handles() + qs, loc="outside lower center", ncol=4, frameon=False
    )
    save(fig, "q_mechanism_paired_difference")
    return written


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else "{:.4f}".format(x)


def _interval(b: Dict[str, Any], level: float) -> str:
    if b["intervals"] is None:
        return "unavailable (n < 2)"
    lo, hi = b["intervals"]["{:.6g}".format(level)]
    tag = " (degenerate)" if b.get("degenerate") else ""
    return "[{}, {}]{}".format(_fmt(lo), _fmt(hi), tag)


def write_report(summary: Dict[str, Any], path: pathlib.Path) -> None:
    m = summary["manifest"]["content"]
    lines = [
        "# {}: analysis report".format(m["study"]),
        "",
        "Analysis status: **{}**.".format(summary["analysis_status"]),
        "",
        "Hypothesis: at a frozen retained-label budget, FTL-DAgger has lower exact "
        "task cost than BC-iid when the learner class is misspecified (alpha > 0) and "
        "mistakes lead to a recoverable state (kappa = 1). The kappa = 0 (no shift) "
        "and alpha = 0 (realizable) conditions are controls.",
        "",
        "## Inventory",
        "",
        "Expected seeds: {}. Scientifically complete seeds: {}.".format(
            summary["inventory"]["seeds_expected"],
            summary["inventory"]["seeds_complete"],
        ),
        "",
    ]
    for status, labels in sorted(summary["inventory"]["by_status"].items()):
        lines.append(
            "- {}: {} ({})".format(
                status,
                len(labels),
                ", ".join(labels[:10]) + (", ..." if len(labels) > 10 else ""),
            )
        )
    incomplete = [
        a for a in summary["input_artifacts"] if a["status"].startswith("incomplete_")
    ]
    if incomplete:
        lines += ["", "## Incomplete seeds (excluded, not promoted)", ""]
        for a in incomplete[:30]:
            lines.append("- {}: {}".format(a["seed_label"], "; ".join(a["reasons"])))
    if summary["failures"]:
        lines += ["", "## Failures", ""]
        for f in summary["failures"][:30]:
            lines.append("- {}".format(json.dumps(f, sort_keys=True)))
        lines.append("")
        lines.append("Total recorded failures: {}.".format(summary["failure_count"]))
    lines += ["", "## Controls", ""]
    for name, c in sorted((summary["controls"] or {}).items()):
        lines.append(
            "- {}: {} ({} comparisons, {} failures)".format(
                name, c["status"], c["comparisons"], c["failures"]
            )
        )
    lines += [
        "",
        "Fixed BC equals BC-iid by construction (same shared expert stream and "
        "exact ERM); it is a state and scheduling control, not independent evidence.",
    ]
    res = summary["results"]
    if res is None:
        lines += [
            "",
            "No scientific summary is published because the analysis is "
            "not complete. This is fail-closed by design.",
        ]
        path.write_text("\n".join(lines) + "\n")
        return
    pf = res["primary_family"]
    lines += [
        "",
        "## Primary family (balanced q, kappa = 1, alpha > 0, final B)",
        "",
        "Estimate is the mean over {} training seeds of the same-seed paired "
        "difference BC-iid cost minus FTL-DAgger final post-update cost (positive "
        "favors FTL). Orientation {} only; orientation 1 was validated as an "
        "exact relabeling and is not counted as extra seeds.".format(
            res["n_seeds"], res["inference_orientation"]
        ),
        "",
        "| H | alpha | B | J_B* | FTL excess | BC-iid excess | estimate | "
        "{:g}% pointwise | Bonferroni {:.4g}% |".format(
            100 * pf["pointwise_level"], 100 * pf["bonferroni_level"]
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in pf["conditions"]:
        b = e["bc_minus_ftl_final"]
        lines.append(
            "| {} | {:g} | {} | {} | {} | {} | {} | {} | {} |".format(
                e["horizon"],
                e["alpha"],
                e["retained_labels"],
                _fmt(e["class_optimum_cost"]),
                _fmt(e["ftl_final_class_excess"]),
                _fmt(e["bc_iid_class_excess"]),
                _fmt(b["estimate"]),
                _interval(b, pf["pointwise_level"]),
                _interval(b, pf["bonferroni_level"]),
            )
        )
    lines += [
        "",
        pf["note"],
        "",
        "## q-stratified support at final B",
        "",
        "| H | alpha | kappa | q | FTL final cost | FTL mixture cost | BC-iid "
        "cost | expert-relative FTL | BC - FTL final | 95% pointwise | "
        "BC - FTL mixture |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    final = m["checkpoints"][-1]
    level = pf["pointwise_level"]
    for e in res["conditions"]:
        if e["retained_labels"] != final:
            continue
        mm = e["means"]
        lines.append(
            "| {} | {:g} | {:g} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                e["horizon"],
                e["alpha"],
                e["kappa"],
                e["q"],
                _fmt(mm["ftl_cost"]),
                _fmt(mm["mixture_cost"]),
                _fmt(mm["bc_cost"]),
                _fmt(mm["ftl_expert_relative_cost"]),
                _fmt(e["bc_minus_ftl_final"]["estimate"]),
                _interval(e["bc_minus_ftl_final"], level),
                _fmt(e["bc_minus_ftl_mixture"]["estimate"]),
            )
        )
    lines += [
        "",
        "Secondary q contrast (q1 minus q0 of the paired difference, final "
        "B), reported for the unsupported-state prior, not as a test:",
        "",
    ]
    for e in res["q_contrast_secondary"]:
        b = e["q1_minus_q0_of_bc_minus_ftl_final"]
        lines.append(
            "- H={} alpha={:g} kappa={:g}: {} {}".format(
                e["horizon"],
                e["alpha"],
                e["kappa"],
                _fmt(b["estimate"]),
                _interval(b, level),
            )
        )
    acc = summary["accounting"]
    lines += [
        "",
        "## Resource accounting (means per cell at final B)",
        "",
        "| H | arm | retained labels | expert action entries | env transitions "
        "| fits |",
        "|---|---|---|---|---|---|",
    ]
    for row in acc:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                row["horizon"],
                row["arm"],
                row["retained_labels"],
                row["expert_action_entries"],
                row["env_transitions"],
                row["fits"],
            )
        )
    lines += [
        "",
        "BC-iid and fixed BC share one physical expert acquisition; their logical "
        "costs are the data each consumes and are not additive. Equal retained "
        "labels are not equal total oracle entries, and no performance at "
        "unmeasured oracle budgets is interpolated.",
        "",
        "## Limitations",
        "",
        "- For alpha > 0 the class cannot represent the expert: expert-relative "
        "cost has an irreducible floor (class optimum J_B* > 0) that no data "
        "removes; zero expert gap is not expected and not claimed.",
        "- Differences depend on q through the data-free prior for the unseen "
        "recovery state (ties go to the smallest canonical policy ID). Both q "
        "variants are reported and balanced within seed.",
        "- This controlled toy does not show that FTL-DAgger is universally better "
        "than BC-iid, and no MFTPL-P guarantee is tested or implied.",
        "- Intervals are percentile bootstrap intervals over training seeds; "
        "Bonferroni intervals are approximate and exact coverage is not claimed.",
        "",
    ]
    if summary.get("figures"):
        lines += ["## Figures", ""] + ["- {}".format(f) for f in summary["figures"]]
    path.write_text("\n".join(lines) + "\n")


def _accounting(results: Dict[str, Any], manifest: Dict[str, Any]) -> List[Dict]:
    final = manifest["checkpoints"][-1]
    rows = []
    for h in manifest["grid"]["horizons"]:
        es = [
            e
            for e in results["conditions"]
            if e["horizon"] == h
            and e["q"] == "balanced"
            and e["retained_labels"] == final
        ]
        mean = lambda m: float(np.mean([e["means"][m] for e in es]))  # noqa: E731
        rows += [
            {
                "horizon": h,
                "arm": "FTL-DAgger",
                "retained_labels": final,
                "expert_action_entries": mean("ftl_expert_action_entries"),
                "env_transitions": mean("ftl_env_steps"),
                "fits": mean("ftl_fits"),
            },
            {
                "horizon": h,
                "arm": "shared expert acquisition (physical)",
                "retained_labels": final,
                "expert_action_entries": mean("shared_expert_action_entries"),
                "env_transitions": mean("shared_env_steps"),
                "fits": 0.0,
            },
            {
                "horizon": h,
                "arm": "BC-iid (logical)",
                "retained_labels": final,
                "expert_action_entries": mean("shared_expert_action_entries"),
                "env_transitions": mean("shared_env_steps"),
                "fits": mean("bc_iid_fits"),
            },
            {
                "horizon": h,
                "arm": "fixed BC (logical)",
                "retained_labels": final,
                "expert_action_entries": mean("shared_expert_action_entries"),
                "env_transitions": mean("shared_env_steps"),
                "fits": mean("bc_fixed_fits"),
            },
        ]
    return rows


def run_analysis(
    manifest_path: pathlib.Path, results_root: pathlib.Path, output_dir: pathlib.Path
) -> Tuple[int, Dict[str, Any]]:
    """Validate, analyze and publish. Raises ManifestError/OSError before writing."""
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = validate_manifest(_strict_loads(manifest_bytes.decode("utf-8")))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ManifestError(str(exc))
    if not results_root.is_dir():
        raise ManifestError("results root is not a directory")
    if output_dir.exists() or output_dir.is_symlink():
        raise ManifestError("output directory must not exist")
    if not output_dir.parent.is_dir():
        raise ManifestError("output parent directory does not exist")

    checks = Checks()
    data, artifacts, problems = load_results(results_root, manifest, checks)
    _, ignored, _ = _scan_root(results_root, manifest)
    by_status: Dict[str, List[str]] = {}
    for art in artifacts:
        by_status.setdefault(art["status"], []).append(art["seed_label"])
    complete = len(by_status.get("complete", []))
    failures: List[Dict[str, Any]] = [{"check": "root", "detail": p} for p in problems]
    for art in artifacts:
        if art["status"] in ("invalid", "failed_checks"):
            failures.append(
                {
                    "check": "artifact",
                    "where": art["seed_label"],
                    "detail": "; ".join(art["reasons"]),
                }
            )
    failures += checks.failures
    if failures:
        status = "failed"
    elif complete < len(manifest["seeds"]):
        status = "incomplete"
    else:
        status = "complete"
    summary: Dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "analysis_status": status,
        "analysis_source_sha256": _sha256(pathlib.Path(__file__).read_bytes()),
        "manifest": {
            "file_sha256": _sha256(manifest_bytes),
            "content_sha256": _canonical_hash(manifest),
            "content": manifest,
        },
        "input_artifacts": artifacts,
        "inventory": {
            "seeds_expected": len(manifest["seeds"]),
            "seeds_complete": complete,
            "by_status": by_status,
            "ignored_entries": ignored,
            "note": "Only producer-complete runs that ended before their own "
            "deadline are analyzed. Any partial or late run, even with every "
            "cell's data present, stays incomplete and is listed with its reasons.",
        },
        "tolerance": {"absolute": "1e-9 * max(1, scale), scale = H for costs"},
        "controls": checks.summary() if checks.counts else None,
        "failure_count": len(failures),
        "failures": failures[:MAX_REPORTED_FAILURES],
        "results": None,
        "accounting": None,
        "figures": [],
    }
    output_dir.mkdir()
    if status == "complete":
        results = analyze(data, manifest)
        summary["results"] = results
        summary["accounting"] = _accounting(results, manifest)
        fig_dir = output_dir / "figures"
        fig_dir.mkdir()
        summary["figures"] = make_figures(results, manifest, fig_dir)
    text = json.dumps(summary, allow_nan=False, indent=1, sort_keys=False) + "\n"
    (output_dir / "summary.json").write_text(text)
    write_report(summary, output_dir / "report.md")
    code = {"complete": EXIT_OK, "incomplete": EXIT_INCOMPLETE}.get(status, EXIT_FAILED)
    return code, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m imitation.experiments.agnostic.analyze_toy",
        description="Fail-closed analysis of a frozen agnostic toy confirmation.",
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--results-root", required=True)
    p.add_argument("--output-dir", required=True, help="fresh path; must not exist")
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if not exc.code else EXIT_INVALID
    try:
        code, summary = run_analysis(
            pathlib.Path(args.manifest),
            pathlib.Path(args.results_root),
            pathlib.Path(args.output_dir),
        )
    except (ManifestError, OSError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_INVALID
    print(
        "analysis {}: {} of {} seeds complete".format(
            summary["analysis_status"],
            summary["inventory"]["seeds_complete"],
            summary["inventory"]["seeds_expected"],
        )
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
