"""Build the CartPole restriction pilot report from a final local copy of its root.

Analysis only. Reads the audit, data and six run ``result.json`` files and the
three controller inventories; never launches jobs, rolls out environments or
loads checkpoints. Every check runs before anything is written, and outputs go
into a new directory only:

    python experiments/agnostic/analyze_cartpole_pilot.py \
        --input-root INPUT --output-dir OUTPUT

The controller inventory is authoritative for terminal state and wall time. A
timed-out run's ``result.json`` is its last saved snapshot (status ``running``,
``snapshot.final`` false), never a live job and never a complete result.

FTL and BC-iid are learning curves. Offline BC is the original ``_run_bc``
baseline: one policy per observation condition, trained once on the whole
offline pool and drawn flat. The historical BC jobs also fitted earlier pool
prefixes; only the stored full-budget fit and its evaluation are used here, and
the prefix values stay in the private raw records. The common label budget
comes from the four online curves only.
"""

import argparse
import datetime as dt
import hashlib
import io
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SOURCE_COMMIT = "2896ffe"
PROTOCOL = "agnostic-cartpole-restriction-pilot/1"
SEED = 300
METHODS = ("ftl", "bc", "bc_iid")
RESTRICTIONS = ("identity", "cart_position_zero")
RUN_IDS = tuple(f"run-{m}-{r}" for m in METHODS for r in RESTRICTIONS)
# Learning-curve methods. Offline BC is a flat reference, not a curve.
ONLINE = ("ftl", "bc_iid")
ONLINE_IDS = tuple(f"run-{m}-{r}" for m in ONLINE for r in RESTRICTIONS)
BC_IDS = tuple(f"run-bc-{r}" for r in RESTRICTIONS)
METHOD_LABEL = {"ftl": "FTL", "bc": "Offline BC", "bc_iid": "BC-iid"}
OBS_LABEL = {"identity": "full observation", "cart_position_zero": "cart position hidden"}
COLOR = {"ftl": "#1b7837", "bc": "#2166ac", "bc_iid": "#b2182b"}
STYLE = {"identity": "--", "cart_position_zero": "-"}
SHARED_DATASET = {"ftl": None, "bc": "pool", "bc_iid": "stream"}
RUN_STATES = {"complete", "timed_out"}
LIVE_STATES = {"pending", "running"}
METRICS = ("rollout_cross_entropy", "disagreement_rate", "expert_rollout_cross_entropy")
# The producer (run_experiment._compute_round_eval) rounds normalized_return to six
# decimals, so a faithful stored value is within half a unit of the sixth place.
NORM_TOL = 0.5e-6 + 1e-12
# Keys that legitimately differ between methods in config.experiment.
METHOD_KEYS = {"algo", "subsample_strategy"}

FIG_CURVES = "cartpole_pilot_learning_curves.png"
FIG_COSTS = "cartpole_pilot_costs.png"
REPORT = "CARTPOLE_PILOT_RESULTS.md"
SUMMARY = "cartpole_pilot_summary.json"
POINTS = "cartpole_pilot_curve_points.json"


class PilotInputError(RuntimeError):
    """The input root is not a terminal, consistent copy of the pilot."""


def require(cond, msg):
    if not cond:
        raise PilotInputError(msg)


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def utc(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


# Arizona does not observe daylight saving time, so Phoenix is always UTC-7.
PHOENIX = dt.timezone(dt.timedelta(hours=-7))


def phoenix(s):
    """Format a UTC timestamp for prose as Phoenix time with UTC in parentheses."""
    u = utc(s).astimezone(dt.timezone.utc)
    p = u.astimezone(PHOENIX)
    clock = f"{p.hour % 12 or 12}:{p:%M} {'am' if p.hour < 12 else 'pm'}"
    return (f"{p:%B} {p.day}, {p.year}, {clock} Phoenix time "
            f"({u:%B} {u.day}, {u:%H:%M} UTC)")


def config_digest(config):
    """Producer canonicalization (``pilot.config_digest``): sorted keys, compact."""
    text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def num(v, digits=2):
    """Format a number for prose; null stays visibly null, never zero."""
    if v is None:
        return "not recorded"
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:,.{digits}f}"


# ---------------------------------------------------------------- validation


def check_inventory(inv, stage, job_ids):
    """Validate one controller inventory and return its single-attempt rows."""
    require(inv.get("version") == 1, f"{stage} inventory: unexpected version")
    require(
        str(inv.get("campaign_id", "")).endswith(f"-{stage}"),
        f"{stage} inventory: unexpected campaign id",
    )
    require(set(inv["jobs"]) == set(job_ids), f"{stage} inventory: unexpected job set")
    root = pathlib.PurePosixPath(inv["source_root"])
    require(
        root.name == "src" and root.parent.name == SOURCE_COMMIT,
        f"{stage} inventory: source snapshot is not {SOURCE_COMMIT}",
    )
    for run in inv["controller_runs"]:
        require(not run.get("retry_requests"), f"{stage}: retry requests recorded")
    rows = {}
    for job_id in job_ids:
        job = inv["jobs"][job_id]
        state = job["state"]
        require(state not in LIVE_STATES, f"{job_id}: job is still {state}")
        allowed = RUN_STATES if stage == "runs" else {"complete"}
        require(state in allowed, f"{job_id}: unsupported terminal state {state}")
        require(len(job["attempts"]) == 1, f"{job_id}: expected exactly one attempt")
        att = job["attempts"][0]
        require(att["state"] == state, f"{job_id}: attempt and job state differ")
        require(not att["infrastructure_retry"], f"{job_id}: infrastructure retry")
        require(att["archived_prior_result"] is None, f"{job_id}: prior result archived")
        require(
            pathlib.PurePosixPath(att["cwd"]).name == SOURCE_COMMIT,
            f"{job_id}: attempt ran outside source snapshot {SOURCE_COMMIT}",
        )
        require(
            pathlib.PurePosixPath(job["expected_result"]).parts[-2:]
            == (job_id, "result.json"),
            f"{job_id}: expected result is not <job>/result.json",
        )
        require(
            att["elapsed_seconds"] is not None and att["ended_at"] is not None,
            f"{job_id}: terminal attempt without controller timing",
        )
        if state == "complete":
            require(att["exit_code"] == 0, f"{job_id}: complete with nonzero exit")
            require(job["result_sha256"], f"{job_id}: complete without result hash")
            val = att["validation"] or {}
            require(
                val.get("ok") is True and val.get("sha256") == job["result_sha256"],
                f"{job_id}: controller validation does not match result hash",
            )
        else:
            term = att["termination"] or {}
            require(
                job["state_reason"] == "job_timeout"
                and term.get("stop_reason") == "job_timeout",
                f"{job_id}: timed_out without a job_timeout stop reason",
            )
        rows[job_id] = (job, att)
    return rows


def check_common(results):
    """Source, expert, preparation and env identities must agree everywhere."""
    ref = results["audit"]
    for key in ("protocol", "env_name", "episode_cap", "expert_sha256", "preparation",
                "source", "package_versions"):
        for name, rec in results.items():
            require(rec.get(key) == ref.get(key), f"{name}: {key} differs from audit")
    require(ref["protocol"] == PROTOCOL, "unexpected protocol")
    require(
        ref["expert_sha256"] == ref["preparation"]["expert_sha256"],
        "expert digest differs from preparation record",
    )


def check_result(job_id, rec, sha, job, schema):
    require(rec["schema"] == f"{PROTOCOL}/{schema}", f"{job_id}: unexpected schema")
    final = rec["snapshot"]["final"]
    if job["state"] == "complete":
        require(rec["status"] == "complete" and final is True,
                f"{job_id}: controller says complete but record is not final")
        require(sha == job["result_sha256"],
                f"{job_id}: result hash differs from controller hash")
    else:
        # The kill left the last atomic snapshot; it must not claim to be final.
        require(rec["status"] == "running" and final is False,
                f"{job_id}: timed-out job has a final or non-running snapshot")


def check_run(job_id, rec, method, restriction, data, data_sha):
    require(rec["method"] == method and rec["restriction_id"] == restriction,
            f"{job_id}: method or restriction differs from job id")
    cfg = rec["config"]
    require(rec["seed"] == SEED and cfg["seed"] == SEED, f"{job_id}: unexpected seed")
    require(cfg["mixture"] is False, f"{job_id}: mixture enabled")
    require(cfg["restriction"]["restriction_id"] == restriction,
            f"{job_id}: config restriction differs")
    require(cfg["experiment"]["algo"] == method, f"{job_id}: config algo differs")
    d = rec["data"]
    require(d["data_result_sha256"] == data_sha, f"{job_id}: data result hash differs")
    require(d["pool_pairs_sha256"] == data["datasets"]["pool"]["pairs_sha256"],
            f"{job_id}: pool digest differs")
    require(d["stream_pairs_sha256"] == data["datasets"]["stream"]["pairs_sha256"],
            f"{job_id}: stream digest differs")
    require(d["baselines"] == data["baselines"], f"{job_id}: baselines differ")
    shared = rec["accounting"]["shared_acquisition"]
    kind = SHARED_DATASET[method]
    if kind is None:
        require(shared is None, f"{job_id}: FTL records shared acquisition")
    else:
        require(shared["dataset"] == kind and shared["charged_to_this_run"] is False
                and shared["pairs_sha256"] == data["datasets"][kind]["pairs_sha256"],
                f"{job_id}: shared acquisition does not match data job")
    if method == "bc_iid" and rec["status"] == "complete":
        require(rec["trained_pairs_sha256"] == data["datasets"]["stream"]["pairs_sha256"],
                f"{job_id}: BC-iid trained pairs differ from stream")


def check_configs(runs):
    """All six runs share one experiment config apart from method-specific keys."""
    def common(rec):
        cfg = rec["config"]
        exp = {k: v for k, v in cfg["experiment"].items() if k not in METHOD_KEYS}
        return exp, cfg["eval_budgets"], cfg["eval_episodes"], cfg["eval_deterministic"]

    ref = common(runs[RUN_IDS[0]])
    for job_id in RUN_IDS:
        require(common(runs[job_id]) == ref, f"{job_id}: config differs from others")
    for m in METHODS:
        a, b = (runs[f"run-{m}-{r}"]["config"] for r in RESTRICTIONS)
        require(a["experiment"] == b["experiment"],
                f"{m}: full and masked runs differ beyond the restriction")
    require(ref[3] is True, "evaluation is not deterministic")
    return ref


# ---------------------------------------------------------------- extraction


def check_normalized(job_id, r, ret, baselines):
    """The stored normalized_return must follow the inherited formula on raw returns.

    Only a check: the stored value is what gets reported and plotted.
    """
    expert_ret, random_ret = baselines["expert_return"], baselines["random_return"]
    require(abs(expert_ret - random_ret) >= 1e-8,
            f"{job_id}: degenerate normalization references")
    stored = r["normalized_return"]
    require(stored is not None, f"{job_id}: round {r['round']} has no normalized_return")
    expected = (ret.mean() - random_ret) / (expert_ret - random_ret)
    require(abs(stored - expected) <= NORM_TOL,
            f"{job_id}: round {r['round']} normalized_return {stored} differs from "
            f"(mean return - random) / (expert - random) = {expected}")


def evaluation_point(job_id, r, n_episodes, cap, baselines):
    """One saved evaluation, with the stored metrics reported as recorded."""
    ret = np.asarray(r["episode_returns"], dtype=float)
    require(ret.size == n_episodes, f"{job_id}: evaluation with {ret.size} episodes")
    check_normalized(job_id, r, ret, baselines)
    return {
        "labels": int(r["n_observations"]),
        "round": int(r["round"]),
        "mean_return": float(ret.mean()),
        "min_return": float(ret.min()),
        "max_return": float(ret.max()),
        "episodes": int(ret.size),
        "episodes_at_cap": int((ret == cap).sum()),
        "normalized_return": r["normalized_return"],
        **{k: r[k] for k in METRICS},
        "eval_env_steps": r["d_eval_size"],
    }


def check_rounds(job_id, rec):
    for r in rec["records"]:
        require(r["round"] == r["n_observations"],
                f"{job_id}: round {r['round']} has {r['n_observations']} labels")


def curve(job_id, rec, n_episodes, cap):
    """Every saved evaluation of an online (FTL or BC-iid) run."""
    check_rounds(job_id, rec)
    points = [evaluation_point(job_id, r, n_episodes, cap, rec["data"]["baselines"])
              for r in rec["records"] if r.get("episode_returns") is not None]
    labels = [p["labels"] for p in points]
    require(labels and labels == sorted(set(labels)), f"{job_id}: bad evaluation order")
    return points


def bc_reference(job_id, rec, n_episodes, cap, n_rounds, pool_pairs_sha256):
    """The offline BC reference: the stored fit on the whole pool, and nothing else.

    The historical job also fitted every smaller prefix. Those records are
    neither selected nor summarized; without a saved full-budget evaluation on
    the whole pool there is no reference and the analysis refuses.
    """
    check_rounds(job_id, rec)
    ends = [r for r in rec["records"] if r["n_observations"] == n_rounds]
    require(len(ends) == 1, f"{job_id}: expected one record at the full budget "
            f"{n_rounds}, found {len(ends)}; refusing to substitute a prefix fit")
    r = ends[0]
    require(r.get("episode_returns") is not None,
            f"{job_id}: full-budget record has no saved evaluation")
    require(r.get("prefix_pairs_sha256") == pool_pairs_sha256,
            f"{job_id}: full-budget fit was not on the whole offline pool")
    require(r.get("checkpoint"), f"{job_id}: full-budget fit has no checkpoint")
    point = evaluation_point(job_id, r, n_episodes, cap, rec["data"]["baselines"])
    wall = r["wall_seconds"]
    return {
        **point,
        "trained_on": "the whole offline pool",
        "pool_pairs_sha256": r["prefix_pairs_sha256"],
        "checkpoint": r["checkpoint"],
        "inner_es_stop_epoch": r["inner_es_stop_epoch"],
        "inner_es_fallback": r["inner_es_fallback"],
        "reference_fit_seconds": wall["fit"],
        "reference_eval_seconds": wall["eval"],
    }


def online_matched_budget(points):
    """Largest label count evaluated by all four online curves.

    Offline BC is not part of this rule: its one policy is trained on the whole
    pool and is reported separately, never as label matched.
    """
    shared = set.intersection(*({p["labels"] for p in points[j]} for j in ONLINE_IDS))
    require(shared, "no evaluation budget is shared by the four online runs")
    return max(shared)


def run_row(job_id, rec, sha, job, att, n_rounds):
    """Physical record of one job: state, timing and saved counts, as recorded."""
    method, restriction = job_id.split("-", 2)[1:]
    acc = rec["accounting"]
    done = acc["completed_records"]
    last_saved_round = max(r["round"] for r in rec["records"])
    evaluated = [r["n_observations"] for r in rec["records"]
                 if r.get("episode_returns") is not None]
    require(evaluated, f"{job_id}: no saved evaluation")
    if job["state"] == "complete":
        require(evaluated[-1] == n_rounds and last_saved_round == n_rounds,
                f"{job_id}: complete run does not end at {n_rounds}")
        require(acc["totals"]["exact"] is True, f"{job_id}: complete totals not exact")
    tail = (utc(att["ended_at"]) - utc(rec["snapshot"]["written_at_utc"])).total_seconds()
    return {
        "run": job_id,
        "method": method,
        "restriction_id": restriction,
        "controller_state": job["state"],
        "saved_record_status": rec["status"],
        "saved_snapshot_final": rec["snapshot"]["final"],
        "attempts": 1,
        "exit_code": att["exit_code"],
        "controller_elapsed_seconds": att["elapsed_seconds"],
        "controller_started_at_utc": att["started_at"],
        "controller_ended_at_utc": att["ended_at"],
        "snapshot_written_at_utc": rec["snapshot"]["written_at_utc"],
        "seconds_from_last_snapshot_to_controller_end": tail,
        "last_saved_training_round": last_saved_round,
        "last_evaluated_labels": evaluated[-1],
        "evaluations_saved": len(evaluated),
        "result_sha256": sha,
        "controller_result_sha256": job["result_sha256"],
        "controller_hash_check": "matched" if job["state"] == "complete"
        else "not available for a timed-out run; saved snapshot hash calculated",
        "counts_are": "exact final totals" if job["state"] == "complete"
        else "saved counts as of the last snapshot; lower bounds on the work done",
        "observed": {k: acc["observed"][k]
                     for k in ("env_steps", "env_resets", "finished_episodes")},
        "in_run_collection": {k: done["collection"][k] for k in ("env_steps", "episodes")},
        "evaluation": {k: done["evaluation"][k]
                       for k in ("evaluations", "episodes", "env_steps")},
        "training": dict(done["training"]),
        "retained_labels": done["retained_labels"],
        "totals": dict(acc["totals"]),
        "reconciled": acc["reconciled"],
        "wall_seconds_recorded": dict(acc["wall_seconds"]),
    }


def shared_costs(audit_att, data_att, data, audit):
    pool, stream = (data["datasets"][k]["stats"] for k in ("pool", "stream"))
    base = data["baselines_costs"]
    return {
        "audit": {
            "controller_elapsed_seconds": audit_att["elapsed_seconds"],
            "expert_predict_calls": audit["expert_predict_calls"],
            "checked_pairs": audit["audit"]["n_checked_pairs"],
            "conflicting_pairs": audit["audit"]["n_conflicting_pairs"],
            "invalid_pairs": audit["audit"]["n_invalid_pairs"],
            "audit_status": audit["audit_status"],
        },
        "data": {
            "controller_elapsed_seconds": data_att["elapsed_seconds"],
            "fixed_bc_pool": {k: pool[k] for k in (
                "elapsed_seconds", "env_steps", "episodes", "env_resets",
                "expert_predict_calls", "retained_labels", "overshoot_transitions")},
            "bc_iid_stream": {k: stream[k] for k in (
                "elapsed_seconds", "env_steps", "episodes", "env_resets",
                "expert_predict_calls", "retained_labels")},
            "normalization_baselines": {k: base[k] for k in (
                "elapsed_seconds", "env_steps_counted", "expert_episodes",
                "random_episodes", "expert_predict_calls")},
            "note": "internal phase timers; they need not sum to the controller time",
        },
        "charged": "once for the whole pilot, not per run or per paired condition",
    }


def campaign_times(invs, attempts):
    stages = {}
    for stage, inv in invs.items():
        began = inv["controller_runs"][0]["started_at"]
        ended = inv["controller_runs"][-1]["ended_at"]
        span = None if ended is None else (utc(ended) - utc(began)).total_seconds()
        stages[stage] = {"campaign_id": inv["campaign_id"],
                         "controller_started_at_utc": began,
                         "controller_ended_at_utc": ended,
                         "controller_span_seconds": span,
                         "source_sha256": inv["source_sha256"]}
    start = min(attempts, key=lambda a: utc(a["started_at"]))["started_at"]
    end = max(attempts, key=lambda a: utc(a["ended_at"]))["ended_at"]
    return {
        "stages": stages,
        "first_attempt_start_utc": start,
        "last_attempt_end_utc": end,
        "elapsed_seconds": (utc(end) - utc(start)).total_seconds(),
        "elapsed_note": "first attempt start to last attempt end; includes review "
        "gates and idle time between stages",
        "sum_attempt_worker_seconds": sum(a["elapsed_seconds"] for a in attempts),
    }


# ------------------------------------------------------------------- figures


def label(job_id):
    m, r = job_id.split("-", 2)[1:]
    return f"{METHOD_LABEL[m]}, {OBS_LABEL[r]}"


def curves_figure(points, refs, rows, baselines, cap, ceiling, n_episodes, n_rounds):
    """Online learning curves with the offline BC reference drawn flat.

    ``points`` holds the four online curves; ``refs`` the one offline BC
    evaluation per observation condition.
    """
    expert_return = baselines["expert_return"]
    random_return = baselines["random_return"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.8), sharex=True)
    panels = [
        ("normalized_return",
         f"Normalized return (mean of {n_episodes} deterministic episodes)"),
        ("rollout_cross_entropy", "Learner rollout cross-entropy"),
        ("disagreement_rate", "Disagreement rate with the expert"),
        ("expert_rollout_cross_entropy",
         "Expert rollout cross-entropy on the same learner-visited states"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        for job_id in ONLINE_IDS:
            m, r = job_id.split("-", 2)[1:]
            xs = [p["labels"] for p in points[job_id]]
            ys = [np.nan if p[key] is None else p[key] for p in points[job_id]]
            ax.plot(xs, ys, STYLE[r], color=COLOR[m], lw=1.5, alpha=0.9)
            if rows[job_id]["controller_state"] == "timed_out":
                ax.plot(xs[-1], ys[-1], "X", ms=9, color=COLOR[m], mec="black",
                        mew=0.8, zorder=5)
        # One fixed policy per condition: a flat line, not a curve over labels.
        for job_id in BC_IDS:
            r = job_id.split("-", 2)[2]
            if refs[job_id][key] is not None:
                ax.axhline(refs[job_id][key], ls=STYLE[r], color=COLOR["bc"], lw=1.5,
                           alpha=0.9, zorder=1)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(-15, n_rounds + 15)
        ax.grid(alpha=0.3)
    ax = axes[0, 0]
    ax.axhline(1.0, color="#555555", ls=":", lw=1.2, zorder=0)
    ax.axhline(0.0, color="#999999", ls="-.", lw=1.0, zorder=0)
    # Limits follow the stored values so nothing is clipped.
    norm = ([p["normalized_return"] for j in ONLINE_IDS for p in points[j]]
            + [refs[j]["normalized_return"] for j in BC_IDS])
    ax.set_ylim(min(0.0, min(norm)) - 0.08, max(1.0, max(norm)) + 0.08)
    same = "equals" if expert_return == cap else "differs from"
    ceiling_text = (
        f"Ceiling: normalized 1 is raw {expert_return:g}, the expert reference;\n"
        f"it {same} the {cap} step cap. "
        + ("All trained evaluations and both\noffline BC references sit on it and "
           "overlap;\nno jitter is added." if ceiling
           else "Lines that reach it\noverlap; no jitter is added.")
    )
    ax.text(n_rounds - 10, 0.93, ceiling_text, ha="right", va="top", fontsize=8.5,
            color="#333333")
    ax.set_ylabel("normalized return (random 0, expert 1)")
    axes[0, 1].set_ylabel("Cross-entropy (natural log)")
    axes[1, 0].set_ylabel("fraction of visited states")
    axes[1, 1].set_ylabel("Cross-entropy (natural log)")
    for ax in axes[1]:
        ax.set_xlabel("expert labels used for training (FTL and BC-iid round)")
    handles = [Line2D([], [], color=COLOR[m], lw=2.5, label=METHOD_LABEL[m])
               for m in ONLINE]
    handles.append(Line2D([], [], color=COLOR["bc"], lw=2.5,
                          label=f"{METHOD_LABEL['bc']}: one fit on all {n_rounds:,}\n"
                                "pool labels, flat reference"))
    handles += [Line2D([], [], color="#444444", ls=STYLE[r], lw=1.5,
                       label=OBS_LABEL[r].capitalize()) for r in RESTRICTIONS]
    handles += [
        Line2D([], [], ls="none", marker="X", ms=9, color="#999999", mec="black",
               label="Last saved evaluation of a timed-out run\n"
                     "(a saved evaluation, not the exact terminal training state)"),
        Line2D([], [], color="#555555", ls=":",
               label=f"Expert reference, normalized 1 (raw {expert_return:g})"),
        Line2D([], [], color="#999999", ls="-.",
               label=f"Random reference, normalized 0 (raw {random_return:g})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.075))
    fig.suptitle("CartPole restriction pilot, training seed 300: online learning "
                 "curves and the offline BC reference", fontsize=13, fontweight="bold")
    fig.text(
        0.5, 0.008,
        "Return: stored normalized_return, "
        f"(mean return - random {random_return:g}) / "
        f"(expert {expert_return:g} - random {random_return:g}), so raw "
        f"{random_return:g} is 0 and raw {expert_return:g} is 1.\n"
        "Cross-entropy: average negative natural log probability of the expert's "
        "deterministic action, per learner-visited state; lower is better.\n"
        "Cross-entropy and disagreement are measured on each learner's own visited "
        "states, not on a common distribution. Round 0 is the untrained initial head.\n"
        f"Offline BC: one policy per condition, trained once on all {n_rounds:,} pool "
        "labels and evaluated once; its line is flat and has no position on the x "
        "axis. Timed-out lines stop at their\nlast saved evaluation and are not "
        "extended. One training seed; per-episode cross-entropy and disagreement "
        "were not stored, so no bands are drawn.",
        ha="center", va="bottom", fontsize=8.5, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.135, 1, 0.96))
    return fig


def costs_figure(rows, shared, job_limit):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    names = list(RUN_IDS)
    ys = np.arange(len(names) + 1)[::-1]
    for y, job_id in zip(ys[:-1], names):
        row = rows[job_id]
        timed_out = row["controller_state"] == "timed_out"
        minutes = row["controller_elapsed_seconds"] / 60
        ax.barh(y, minutes, color=COLOR[row["method"]], alpha=0.55 if timed_out else 0.9,
                hatch="//" if timed_out else None, edgecolor="black", lw=0.6)
        state = "timed out" if timed_out else "complete"
        text = f"{row['controller_elapsed_seconds']:,.2f} s, {state}"
        if row["method"] == "bc":
            # The historical job's whole time is charged, not only the reference fit.
            text += (f"; {row['training']['fits']} fits, "
                     f"{row['evaluation']['evaluations']} evaluations")
        ax.text(minutes + 1, y, text, va="center", fontsize=8.5)
    audit_m = shared["audit"]["controller_elapsed_seconds"] / 60
    data_m = shared["data"]["controller_elapsed_seconds"] / 60
    y = ys[-1]
    ax.barh(y, audit_m, color="#bbbbbb", edgecolor="black", lw=0.6)
    ax.barh(y, data_m, left=audit_m, color="#777777", edgecolor="black", lw=0.6)
    ax.text(audit_m + data_m + 1, y,
            f"audit {shared['audit']['controller_elapsed_seconds']:,.2f} s + data "
            f"{shared['data']['controller_elapsed_seconds']:,.2f} s "
            "(acquisition and normalization), charged once",
            va="center", fontsize=8.5)
    ax.axvline(job_limit / 60, color="#444444", ls="--", lw=1)
    ax.text(job_limit / 60, ys[0] + 0.62, f"controller job limit {job_limit:,.0f} s",
            ha="center", fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([label(j) for j in names] + ["Shared audit and data jobs"])
    ax.set_ylim(ys[-1] - 0.7, ys[0] + 0.9)
    ax.set_xlim(0, job_limit / 60 * 1.45)
    ax.set_xlabel("controller-measured worker wall time per attempt (minutes)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[Patch(fc="#999999", ec="black", label="complete"),
                       Patch(fc="#999999", ec="black", alpha=0.55, hatch="//",
                             label="timed out at the limit")],
              loc="lower right", fontsize=8.5, frameon=False)
    ax.set_title("CartPole restriction pilot: external controller wall time",
                 fontsize=12, fontweight="bold")
    fig.text(
        0.5, 0.01,
        "Worker wall time measured by the external controller. Not CPU or GPU hours and "
        "not a monetary cost. Offline BC bars include every historical prefix fit; "
        "only the full-pool fit is the reference.\nOriginal expert preparation, local "
        "review and plotting are outside this cost scope. Work after a timed-out "
        "run's last snapshot is\nunrecorded; its wall time is still exact to the "
        "controller's measurement.",
        ha="center", va="bottom", fontsize=8, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, metadata={"Software": None})
    plt.close(fig)
    return buf.getvalue()


# -------------------------------------------------------------------- report


def metric_row(job_id, p):
    return (f"| {METHOD_LABEL[job_id.split('-', 2)[1]]} | "
            f"{OBS_LABEL[job_id.split('-', 2)[2]]} | {p['labels']:,} | "
            f"{p['mean_return']:.1f} ({p['min_return']:.0f} to {p['max_return']:.0f}) | "
            + " | ".join("not recorded" if p[k] is None else f"{p[k]:.4f}"
                         for k in METRICS) + " |")


def joined(values, fmt="{:,}"):
    """One value if all agree, else each value, so differences stay visible."""
    vals = list(dict.fromkeys(values))
    return " and ".join(fmt.format(v) for v in vals)


def report(s, analysis_sha):
    rows, shared, camp = s["runs"], s["shared_preparation"], s["campaign"]
    n_done = sum(r["controller_state"] == "complete" for r in rows.values())
    n_out = len(rows) - n_done
    B = s["matched_budget"]["labels"]
    refs, err = s["bc_reference"]["points"], s["erratum"]
    n_bc = s["n_rounds"]
    ceil = s["return_ceiling"]
    fp = s["provenance"]
    exp_ret, rnd_ret, cap = s["expert_return"], s["random_return"], s["episode_cap"]
    reference = (f"Normalized 1 here corresponds to raw {exp_ret:g} and remains a "
                 f"ceiling: the expert reference equals the {cap} step cap"
                 if exp_ret == cap else
                 f"Normalized 1 here corresponds to raw {exp_ret:g}; the step cap is "
                 f"{cap}")
    data_sh = shared["data"]
    lines = [
        "# CartPole restriction pilot: results",
        "",
        f"Records as of the last controller attempt end, "
        f"{phoenix(camp['last_attempt_end_utc'])}. "
        "This report covers the six learning runs of the approved paired pilot in "
        "[`CARTPOLE_PILOT_PROTOCOL.md`](CARTPOLE_PILOT_PROTOCOL.md); the audit and data "
        "gates are described in [`CARTPOLE_PILOT_STATUS.md`](CARTPOLE_PILOT_STATUS.md). "
        "It is analysis only. No job was relaunched, retried or extended for it, and no "
        "scientific setting changed. The raw records (results, checkpoints, logs) stay "
        "private; this report, its two figures and its two JSON files are derived "
        "summaries. FTL and BC-iid are learning curves; offline BC is one fixed "
        "policy per observation condition, drawn as a flat reference (see the "
        "erratum).",
        "",
        "## Outcome",
        "",
        f"- {n_done} of 6 runs completed; {n_out} were stopped by the external "
        f"{num(s['job_limit_seconds'], 0)} second job limit (`timed_out`). Timed-out "
        "runs are reported up to their last saved evaluation only.",
    ]
    if ceil["all_at_cap"]:
        lines.append(
            "- Every saved evaluation of a trained FTL or BC-iid policy, in both "
            f"observation conditions, scored {cap} in all {s['eval_episodes']} episodes, "
            "starting with the first one-label evaluation, and so did both offline BC "
            f"reference policies (stored normalized return 1, which corresponds to raw "
            f"{exp_ret:g}). Return is at its ceiling and cannot separate methods or "
            "observation conditions here.")
    else:
        lines.append(
            f"- {ceil['online_trained_evaluations_at_cap']} of "
            f"{ceil['online_trained_evaluations']} saved evaluations of trained FTL "
            f"and BC-iid policies, and {ceil['bc_reference_evaluations_at_cap']} of "
            f"{len(refs)} offline BC reference evaluations, scored {cap} in every "
            "episode; see the curve points file for the rest.")
    lines += [
        "- Cross-entropy and disagreement are reported as recorded; each is measured on "
        "that learner's own visited states.",
        "- Offline BC and BC-iid also differ in optimization (below), so their "
        "comparison does not isolate data acquisition.",
        "",
        "## Erratum: offline BC",
        "",
        "- The intended BC baseline is the original `_run_bc`: one policy trained once "
        "on the whole offline dataset and drawn as a flat reference. The executed "
        f"pilot instead fitted BC separately on {joined(err['historical_fits'])} pool "
        f"prefixes ({err['historical_fit_budgets']['first']:,} to "
        f"{err['historical_fit_budgets']['last']:,} labels) per condition. The "
        "previous version of this report drew those fits as a learning curve and "
        "used a prefix fit in the matched comparison. That was wrong.",
        f"- This version uses, per condition, only the stored fit on all {n_bc:,} pool "
        "labels and its saved evaluation, drawn flat. The smaller prefix fits are not "
        "a baseline. They are not plotted, tabulated or summarized here and remain "
        "only in the private raw records.",
        f"- The {n_bc:,}-label fit does not depend on the earlier fits. Before every "
        "fit the job reseeded torch, built a fresh policy, gave the trainer a fresh "
        "generator with the run seed, and seeded the validation split by the seed and "
        "round 0, on CPU. This was checked by reading source commit "
        f"`{SOURCE_COMMIT}`, not by a rerun. Its evaluation reused the run's "
        f"environment, so its {s['eval_episodes']} starting states followed "
        f"{joined(err['earlier_evaluation_episodes'])} earlier evaluation episodes. "
        "The values are the saved estimate for that final policy, not an exact "
        "standalone rerun.",
        "- Costs are unchanged. Each historical BC job's controller time covers all of "
        "its fits and evaluations and is charged in full; the reference fit's own "
        "timers are listed separately under Cost.",
        f"- Offline BC uses {n_bc:,} labels and the online curves are compared at "
        f"{B:,}, so offline BC is not label matched to them.",
        "",
        "## Run status",
        "",
        "| Method | Observation | Controller state | Controller elapsed (s) | "
        "Last saved training round | Last evaluated label count | Saved record |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for job_id in RUN_IDS:
        r = rows[job_id]
        saved = ("complete, final" if r["saved_snapshot_final"]
                 else f"`{r['saved_record_status']}`, `snapshot.final=false`")
        lines.append(
            f"| {METHOD_LABEL[r['method']]} | {OBS_LABEL[r['restriction_id']]} | "
            f"`{r['controller_state']}` | {r['controller_elapsed_seconds']:,.2f} | "
            f"{r['last_saved_training_round']:,} | {r['last_evaluated_labels']:,} | "
            f"{saved} |")
    lines += [
        "",
        "The controller inventory is authoritative for terminal state. A timed-out "
        "run's saved record still says `running` because the external kill left its "
        "last atomic snapshot; it is neither a live job nor a complete result. Its "
        "in-flight operation label can lag by a round and is not used to infer what was "
        "running at termination.",
        "",
        "## Learning curves",
        "",
        f"![Learning curves](results/{FIG_CURVES})",
        "",
        "- Every saved FTL and BC-iid evaluation is drawn, including round 0 (the "
        "untrained initial head). Lines are not smoothed.",
        "- Offline BC is drawn as one flat line per condition in its own color, dashed "
        "for full observation and solid for cart position hidden, at the value of its "
        "single saved evaluation. A flat line is one fixed policy evaluated once, not "
        "repeated measurements; it spans the axis for comparison only.",
        "- An X marks a timed-out run's last saved evaluation. It is a saved "
        "evaluation, not the exact training state at termination, and the line is not "
        "extended toward 1,000.",
        "- The return panel draws the stored `normalized_return`, the inherited "
        "(mean return - random return) / (expert return - random return), with the "
        f"random reference {rnd_ret:g} at 0 (dash-dot line) and the expert reference "
        f"{exp_ret:g} at 1 (dotted line). {reference}. Return curves that reach 1 "
        "coincide and overlap; no jitter is added.",
        "- Each stored `normalized_return` was checked against its raw episode mean "
        "and the stored references, within the producer's six-decimal rounding. The "
        "stored values are plotted as recorded: none was recomputed, replaced or "
        "clipped. Tables below and both JSON files keep raw returns.",
        "- The previous version of this figure plotted raw mean return. That was a "
        "presentation departure from the stored metric, not a different evaluation; "
        "this figure restores the stored normalized return.",
        "- Cross-entropy (natural log) is the average negative natural log probability "
        "of the expert's deterministic action, per learner-visited state; lower is "
        "better. The earlier axis label \"nats per state\" named the same quantity. "
        "This is a unit clarification, not a changed metric.",
        "- Learner cross-entropy, disagreement and expert cross-entropy are computed "
        "on the states each learner visited during its own evaluation episodes. They "
        "are not errors on a common state distribution, so a lower value does not mean "
        "lower error on the same states.",
        "",
        "## Matched-label comparison",
        "",
        "The largest evaluation budget shared by the four online curves (FTL and "
        f"BC-iid, both conditions) is **{B:,} labels**: the maximum of the "
        "intersection of their four sets of evaluated label counts, taken mechanically "
        "and not by effect size. Offline BC does not enter this rule. Values are "
        "descriptive for one training seed. No winner is selected and no test is run. "
        "Return is shown as the raw mean with the episode minimum to maximum. "
        "Cross-entropy columns use the natural log, as defined above.",
        "",
        "| Method | Observation | Labels | Return mean (min to max) | Learner CE | "
        "Disagreement | Expert CE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [metric_row(j, s["matched_budget"]["points"][j]) for j in ONLINE_IDS]
    lines += [
        "",
        f"Offline BC reference, one policy per condition trained on all {n_bc:,} pool "
        f"labels. Not label matched to the {B:,}-label rows above:",
        "",
        "| Method | Observation | Labels | Return mean (min to max) | Learner CE | "
        "Disagreement | Expert CE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [metric_row(j, refs[j]) for j in BC_IDS]
    lines += [
        "",
        "Each online run at its own last saved evaluation (timed-out runs end earlier, "
        "so these rows are not matched):",
        "",
        "| Method | Observation | Labels | Return mean (min to max) | Learner CE | "
        "Disagreement | Expert CE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [metric_row(j, s["own_endpoints"][j]) for j in ONLINE_IDS]
    lines += [
        "",
        "## Cost",
        "",
        f"![Controller wall time](results/{FIG_COSTS})",
        "",
        "Wall time is the external controller's `elapsed_seconds` per attempt, exact "
        "to its measurement. It is worker wall time, not CPU or GPU hours, and no "
        "monetary cost is implied. Each job had exactly one attempt.",
        "",
        "| Job | Controller state | Controller elapsed (s) |",
        "| --- | --- | ---: |",
        f"| audit (shared) | `complete` | "
        f"{shared['audit']['controller_elapsed_seconds']:,.2f} |",
        f"| data (shared) | `complete` | {data_sh['controller_elapsed_seconds']:,.2f} |",
    ]
    lines += [f"| {label(j)} | `{rows[j]['controller_state']}` | "
              f"{rows[j]['controller_elapsed_seconds']:,.2f} |" for j in RUN_IDS]
    run_sum = sum(r["controller_elapsed_seconds"] for r in rows.values())
    lines += [
        "",
        f"Run attempts total {run_sum:,.2f} worker-seconds; with the shared audit and "
        f"data jobs, {camp['sum_attempt_worker_seconds']:,.2f}. The campaign took "
        f"{camp['elapsed_seconds']:,.2f} seconds ({camp['elapsed_seconds'] / 3600:.2f} "
        "hours) from the first attempt start to the last attempt end, including review "
        "gates between stages.",
        "",
        "**Shared preparation, charged once.** The audit checked "
        f"{shared['audit']['checked_pairs']} pairs with "
        f"{num(shared['audit']['expert_predict_calls'])} expert predict calls. The "
        "data job's internal timers record: offline BC chronological pool "
        f"{data_sh['fixed_bc_pool']['elapsed_seconds']:,.2f} s for "
        f"{num(data_sh['fixed_bc_pool']['env_steps'])} expert steps "
        f"({data_sh['fixed_bc_pool']['episodes']} episodes); BC-iid stream "
        f"{data_sh['bc_iid_stream']['elapsed_seconds']:,.2f} s for "
        f"{num(data_sh['bc_iid_stream']['env_steps'])} expert steps "
        f"({num(data_sh['bc_iid_stream']['episodes'])} episodes); normalization "
        f"baselines {data_sh['normalization_baselines']['elapsed_seconds']:,.2f} s for "
        f"{num(data_sh['normalization_baselines']['env_steps_counted'])} steps "
        f"({data_sh['normalization_baselines']['expert_episodes']} expert and "
        f"{data_sh['normalization_baselines']['random_episodes']} random episodes). "
        "These phase timers need not sum to the controller time. The full and masked "
        "offline BC runs reuse the pool, and the full and masked BC-iid runs reuse the "
        "stream, so this acquisition is not charged per run or per paired condition. "
        "FTL acquires its own labels inside each run (in-run collection below).",
        "",
        "Saved counts per run. For a timed-out run these come from its last snapshot: "
        "they are observed lower bounds (environment steps) or saved counts (fits, "
        "epochs), not final totals. Null values are shown as not recorded.",
        "",
        "| Run | Counts | Env steps observed | In-run collection steps | Evaluations | "
        "Evaluation steps | Fits | Epochs | Expert predict calls (total) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for j in RUN_IDS:
        r = rows[j]
        kind = "final" if r["controller_state"] == "complete" else "saved, lower bound"
        lines.append(
            f"| {label(j)} | {kind} | {num(r['observed']['env_steps'])} | "
            f"{num(r['in_run_collection']['env_steps'])} | "
            f"{num(r['evaluation']['evaluations'])} | "
            f"{num(r['evaluation']['env_steps'])} | {num(r['training']['fits'])} | "
            f"{num(r['training']['epochs'])} | "
            f"{num(r['totals']['expert_predict_calls'])} |")
    lines += [
        "",
        "Recorded in-run phase timers (seconds). Round-loop methods time fitting "
        "together with collection; the historical offline BC job times all its fits "
        "separately and collects nothing in the run. For timed-out runs these cover "
        "work up to the last snapshot, and the recorded total is null.",
        "",
        "| Run | Setup | Collection and training | Fits | Evaluation | Recorded total | "
        "Last snapshot to controller end |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for j in RUN_IDS:
        r, w = rows[j], rows[j]["wall_seconds_recorded"]
        bc = r["method"] == "bc"
        lines.append(
            f"| {label(j)} | {num(w['setup'])} | "
            f"{'no in-run collection' if bc else num(w['collection_and_training'])} | "
            f"{num(w['fits']) if bc else 'within collection and training'} | "
            f"{num(w['evaluation'])} | {num(w['total'])} | "
            f"{r['seconds_from_last_snapshot_to_controller_end']:,.2f} |")
    lines += [
        "",
        f"Offline BC reference fit on all {n_bc:,} pool labels, from its own record "
        "(seconds). These are part of the historical job totals above, not additional "
        "cost:",
        "",
        "| Run | Fit | Evaluation | Epochs | Early-stopping fallback |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    lines += [f"| {label(j)} | {refs[j]['reference_fit_seconds']:,.2f} | "
              f"{refs[j]['reference_eval_seconds']:,.2f} | "
              f"{num(refs[j]['inner_es_stop_epoch'])} | "
              f"{refs[j]['inner_es_fallback'] or 'none'} |" for j in BC_IDS]
    lines += [
        "",
        "Original expert preparation and qualification, local review, and report and "
        "plot work are outside this cost scope.",
        "",
        "## Interpretation and limits",
        "",
        f"- **Datasets.** Offline BC is one fit on all {n_bc:,} labels of a "
        f"chronological pool ({data_sh['fixed_bc_pool']['episodes']} complete expert "
        "episodes, kept in order). BC-iid replays an independently reset stream, one "
        "uniformly selected "
        f"state from each of {num(data_sh['bc_iid_stream']['episodes'])} expert "
        "episodes, one state per round. The two datasets have different content "
        "hashes.",
        "- **Inherited optimizer mismatch.** As in the previous pipeline, FTL and "
        f"BC-iid train with minibatch 1 (`min({s['bc_batch_size']}, "
        "samples_per_round)`) and refit after every label, up to "
        f"{num(s['n_rounds'])} fits, while the offline BC reference is a single fit "
        f"with minibatches of up to {s['bc_batch_size']}. All six configs record "
        f"`bc_batch_size` {s['bc_batch_size']}; the effective sizes follow from source "
        "and were not separately instrumented. This changes optimization, not only "
        "speed, so offline BC versus BC-iid does not isolate data acquisition.",
        "- **Return ceiling.** Reaching the cap after one label (normalized return 1, "
        f"raw {exp_ret:g}) is a pilot finding for "
        "this learner, the CartPole reset distribution and the 500 step limit. It does "
        "not show that hiding cart position is harmless in general or that FTL cannot "
        "help.",
        f"- **Audit scope.** The audit found {shared['audit']['conflicting_pairs']} "
        f"conflicting of {shared['audit']['checked_pairs']} checked pairs: expert "
        "labels conflict under the mask. It does not establish a positive error floor "
        "on natural state distributions or a return gap.",
        "- **Frozen expert features** are a plausible explanation for the rapid control "
        "success, not a verified causal attribution.",
        "- **One training seed.** Training-seed uncertainty cannot be estimated from one "
        "training seed, and no general superiority claim is made. The episode minimum "
        "and maximum describe evaluation spread "
        "for one trained policy, not confidence about other training seeds. Per-episode "
        "cross-entropy and disagreement were not stored, so no uncertainty bands are "
        "drawn.",
        "- **Pairing.** Pairing across observation conditions is limited. Offline BC "
        "uses the same stored pool and BC-iid the same stored stream in both "
        "conditions. FTL data are not shared: each FTL run collects labels on its own "
        "visited states, which differ between conditions. Each method's full and "
        "masked runs share the seed setup and the initial head, and the offline BC "
        "reference fit starts from torch reseeded to the run seed in both conditions. "
        + ("The round 0 evaluation records of FTL and BC-iid are identical within each "
           "observation condition, consistent with this. "
           if s["pairing"]["round0_ftl_bc_iid_identical_within_condition"] else
           "The round 0 evaluation records of FTL and BC-iid differ within a "
           "condition; see the summary file. ")
        + "Evaluation episodes are not paired across runs, and later round-loop head "
        "redraws can diverge between conditions.",
        "- **Collection versus evaluation.** FTL collects with the stochastic learner "
        "and is evaluated deterministically, preserved from the previous pipeline.",
        "- **No changes.** Every job ran once under its original settings and limits; "
        "nothing was retried, extended or modified.",
        "",
        "## Provenance",
        "",
        f"- Source commit `{SOURCE_COMMIT}`, checked against the source snapshot name "
        "the controller recorded for every attempt.",
        "- Campaign source hash, as recorded by the controller and identical in all "
        f"three inventories: `{fp['campaign_source_sha256']}`.",
        "- Source fingerprint, as recorded and identical in all eight results: "
        f"`{fp['result_source_combined_sha256']}`.",
        f"- Expert `{fp['expert_sha256']}`; preparation record "
        f"`{fp['preparation_record_sha256']}`; policy state "
        f"`{fp['preparation_policy_state_sha256']}`. Recorded and identical in all "
        "eight results.",
        f"- Offline BC pool: file `{fp['pool_file_sha256']}`, pairs "
        f"`{fp['pool_pairs_sha256']}`. BC-iid stream: file "
        f"`{fp['stream_file_sha256']}`, pairs `{fp['stream_pairs_sha256']}`. Recorded "
        "by the data job; every run cites the same pairs digests and the data result "
        "hash below.",
        "- The config digests of the data job and all six runs were recomputed with "
        "the producer's canonical JSON and match. The source, expert, preparation and "
        "dataset digests above are reported as recorded: this analysis compares them "
        "across records but does not rehash the source, preparation or dataset files, "
        "which are not part of its input.",
        f"- Analysis script `experiments/agnostic/analyze_cartpole_pilot.py`, SHA-256 "
        f"`{analysis_sha}`.",
        "",
        "| Record | SHA-256 of the analyzed file | Controller hash |",
        "| --- | --- | --- |",
    ]
    for rel, h in s["input_hashes"].items():
        lines.append(f"| `{rel}` | `{h['sha256']}` | {h['controller_check']} |")
    lines += [
        "",
        "## Current status",
        "",
        "- This report remains the historical record of the x-only pilot (cart "
        "position hidden). Its records stay unchanged.",
        "- The user has approved a separate six-run follow-up: FTL, BC-iid and "
        "single-fit offline BC, with minibatches that grow with the data up to 32, "
        "each under the identity observation and with both cart position and pole "
        "angular velocity hidden; other settings unchanged. See "
        "[`CARTPOLE_STRONGER_MASK_PROTOCOL.md`](CARTPOLE_STRONGER_MASK_PROTOCOL.md).",
        "- Its implementation and independent reviews come before any execution. No "
        "follow-up results are included here.",
        "- Larger studies, tuning and other masks remain unapproved. Read-only "
        "diagnosis of already stored artifacts remains authorized.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------- main


def forbidden_tokens(invs, input_root, output_dir):
    tokens = {"/home/", "/Users/", "/opt/", "miniconda", chr(0x2014),
              str(input_root), str(output_dir)}
    blob = json.dumps(list(invs.values()))
    for m in re.finditer(r"/(?:home|Users)/([^/\"]+)", blob):
        tokens.add(m.group(1))
    return tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input-root", required=True, type=pathlib.Path)
    ap.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = ap.parse_args()
    root = args.input_root.resolve()
    out = args.output_dir.resolve()
    require(root.is_dir(), "input root does not exist")
    require(not out.exists(), "output directory already exists; refusing to overwrite")
    require(root != out and root not in out.parents,
            "output directory must not be inside the input root")

    invs, inv_sha = {}, {}
    for stage in ("audit", "data", "runs"):
        rel = f"control/{stage}-inventory/inventory.json"
        invs[stage], inv_sha[rel] = load(root / rel)
    rows_ctl = {**check_inventory(invs["audit"], "audit", ["audit"]),
                **check_inventory(invs["data"], "data", ["data"]),
                **check_inventory(invs["runs"], "runs", list(RUN_IDS))}
    sources = {inv["source_sha256"] for inv in invs.values()}
    require(len(sources) == 1, "inventories disagree on the campaign source hash")

    results, sha = {}, {}
    for job_id in ("audit", "data") + RUN_IDS:
        results[job_id], sha[job_id] = load(root / job_id / "result.json")
        schema = job_id if job_id in ("audit", "data") else "run"
        check_result(job_id, results[job_id], sha[job_id], rows_ctl[job_id][0], schema)
    check_common(results)
    config_sha = {}
    for job_id in ("data",) + RUN_IDS:
        recorded = results[job_id]["config_sha256"]
        require(config_digest(results[job_id]["config"]) == recorded,
                f"{job_id}: config_sha256 does not match the producer canonical digest")
        config_sha[job_id] = recorded
    audit, data = results["audit"], results["data"]
    require(audit["audit_status"] == "conflict_found", "audit status changed")
    require(data["seed"] == SEED, "data job seed differs")
    for kind in ("pool", "stream"):
        require(data["datasets"][kind]["n"] == data["config"]["budget"],
                f"data {kind} size differs from budget")
    runs = {j: results[j] for j in RUN_IDS}
    for job_id in RUN_IDS:
        m, r = job_id.split("-", 2)[1:]
        check_run(job_id, runs[job_id], m, r, data, sha["data"])
    exp_cfg, eval_budgets, n_episodes, _ = check_configs(runs)
    n_rounds = exp_cfg["n_rounds"]
    require(n_rounds == data["config"]["budget"], "run budget differs from data budget")
    cap = audit["episode_cap"]
    expert_return = data["baselines"]["expert_return"]
    job_limit = runs[RUN_IDS[0]]["deadline"]["job_limit_seconds"]
    for job_id in RUN_IDS:
        require(runs[job_id]["deadline"]["job_limit_seconds"] == job_limit,
                f"{job_id}: job limit differs")

    for j in RUN_IDS:
        evaluated = {r["n_observations"] for r in runs[j]["records"]
                     if r.get("episode_returns") is not None}
        extra = evaluated - set(eval_budgets) - {0}
        require(not extra, f"{j}: evaluations outside the configured budgets")
    points = {j: curve(j, runs[j], n_episodes, cap) for j in ONLINE_IDS}
    pool_pairs = data["datasets"]["pool"]["pairs_sha256"]
    refs = {j: bc_reference(j, runs[j], n_episodes, cap, n_rounds, pool_pairs)
            for j in BC_IDS}
    rows = {j: run_row(j, runs[j], sha[j], *rows_ctl[j], n_rounds) for j in RUN_IDS}

    B = online_matched_budget(points)
    trained = [p for j in ONLINE_IDS for p in points[j] if p["labels"] > 0]
    at_cap = [p for p in trained if p["episodes_at_cap"] == p["episodes"]]
    refs_at_cap = [p for p in refs.values() if p["episodes_at_cap"] == p["episodes"]]
    round0 = {}
    for r in RESTRICTIONS:
        a, b = (next((p for p in points[f"run-{m}-{r}"] if p["labels"] == 0), None)
                for m in ("ftl", "bc_iid"))
        round0[r] = a is not None and b is not None and all(
            a[k] == b[k] for k in a if k != "round")

    attempts = [att for _, att in rows_ctl.values()]
    shared = shared_costs(rows_ctl["audit"][1], rows_ctl["data"][1], data, audit)
    campaign = campaign_times(invs, attempts)
    analysis_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()

    input_hashes = {rel: {"sha256": h, "controller_check": "inventory file"}
                    for rel, h in inv_sha.items()}
    for job_id in ("audit", "data") + RUN_IDS:
        state = rows_ctl[job_id][0]["state"]
        input_hashes[f"{job_id}/result.json"] = {
            "sha256": sha[job_id],
            "controller_check": "matched" if state == "complete"
            else "none recorded (timed out); saved snapshot hash shown",
        }
    pool, stream = data["datasets"]["pool"], data["datasets"]["stream"]
    erratum = {
        "issue": "the executed pilot fitted BC separately at every evaluated pool "
        "prefix, and the previous report drew those fits as a learning curve and used "
        "a prefix fit in the matched comparison; the intended baseline is the original "
        "_run_bc, one fit on the whole offline dataset drawn flat",
        "correction": "only the stored fit on the whole pool and its saved evaluation "
        "are used, per condition; prefix fits are not a baseline and are neither "
        "plotted nor summarized; their values stay in the private raw records",
        "historical_fits": [rows[j]["training"]["fits"] for j in BC_IDS],
        "historical_fit_budgets": {"count": len(eval_budgets),
                                   "first": eval_budgets[0], "last": eval_budgets[-1]},
        "independence": "by reading the source: before every fit torch was reseeded, "
        "a fresh policy built, the trainer given a fresh generator with the run seed, "
        "and the validation split seeded by the seed and round 0, on CPU; the "
        "full-pool fit does not depend on earlier fits; not verified by a rerun",
        "evaluation_rng": "the reference evaluation reused the run's environment after "
        "the earlier evaluations, so it is the saved estimate for the final policy, "
        "not an exact standalone rerun",
        "earlier_evaluation_episodes": [
            rows[j]["evaluation"]["episodes"] - refs[j]["episodes"] for j in BC_IDS],
        "cost": "historical BC job costs are unchanged and charged in full; the "
        "reference fit and evaluation timers are part of them, not additional",
    }
    summary = {
        "schema": "cartpole-restriction-pilot-analysis/2",
        "scope": "derived summary of the six-run CartPole restriction pilot; raw "
        "records, checkpoints and logs remain private",
        "provenance": {
            "source_commit": SOURCE_COMMIT,
            "campaign_source_sha256": sources.pop(),
            "result_source_combined_sha256": audit["source"]["combined_sha256"],
            "result_source_files": audit["source"]["files"],
            "expert_sha256": audit["expert_sha256"],
            "preparation_record_sha256": audit["preparation"]["record_sha256"],
            "preparation_policy_state_sha256": audit["preparation"]["policy_state_sha256"],
            "preparation_config_sha256": audit["preparation"]["config_sha256"],
            "data_result_sha256": sha["data"],
            "pool_file_sha256": pool["file_sha256"],
            "pool_pairs_sha256": pool["pairs_sha256"],
            "stream_file_sha256": stream["file_sha256"],
            "stream_pairs_sha256": stream["pairs_sha256"],
            "analysis_script": "experiments/agnostic/analyze_cartpole_pilot.py",
            "analysis_script_sha256": analysis_sha,
            "config_sha256_recomputed_and_matched": config_sha,
            "checks": {
                "recomputed_here": [
                    "SHA-256 of every analyzed result and inventory file; complete "
                    "jobs matched against the controller result hash",
                    "config_sha256 of the data job and six runs, with the producer's "
                    "canonical JSON (sorted keys, compact separators)",
                    "data result hash cited by every run",
                    "normalized_return of every saved FTL and BC-iid evaluation and of "
                    "each offline BC reference evaluation checked against "
                    "(mean episode return - random_return) / (expert_return - "
                    "random_return) on its raw episode returns and stored references, "
                    "within the producer's six-decimal rounding; the stored value is "
                    "reported, never replaced; inactive BC prefix evaluations are not "
                    "read for metrics",
                    "offline BC reference: exactly one record at the full budget, "
                    "with a saved evaluation and checkpoint, whose prefix digest "
                    "equals the whole pool's pairs digest",
                ],
                "compared_for_equality_only": [
                    "campaign source hash across the three inventories",
                    "source fingerprint, expert, preparation, environment and step "
                    "cap across all eight results",
                    "pool and stream pairs digests cited by runs against the data job",
                ],
                "recorded_only": [
                    "source file digests and combined source fingerprint (source "
                    "files are not in the input)",
                    "expert, preparation record, policy state and preparation config "
                    "hashes (preparation files are not in the input)",
                    "pool and stream file and pairs digests (dataset files are not "
                    "in the input)",
                ],
            },
        },
        "input_hashes": input_hashes,
        "protocol": PROTOCOL,
        "env_name": audit["env_name"],
        "seed": SEED,
        "episode_cap": cap,
        "expert_return": expert_return,
        "random_return": data["baselines"]["random_return"],
        "eval_episodes": n_episodes,
        "n_rounds": n_rounds,
        "n_eval_budgets": len(eval_budgets),
        "bc_batch_size": exp_cfg["bc_batch_size"],
        "job_limit_seconds": job_limit,
        "runs": rows,
        "matched_budget": {
            "labels": B,
            "rule": "maximum of the intersection of the four online (FTL and BC-iid) "
            "sets of evaluated label counts; offline BC is not part of it",
            "points": {j: next(p for p in points[j] if p["labels"] == B)
                       for j in ONLINE_IDS},
        },
        "bc_reference": {
            "definition": "original _run_bc: one policy per observation condition "
            "trained once on the whole offline pool and drawn as a flat reference",
            "labels": n_rounds,
            "label_matched_to_online": False,
            "points": refs,
        },
        "own_endpoints": {j: points[j][-1] for j in ONLINE_IDS},
        "return_ceiling": {
            "online_trained_evaluations": len(trained),
            "online_trained_evaluations_at_cap": len(at_cap),
            "bc_reference_evaluations_at_cap": len(refs_at_cap),
            "all_at_cap": len(at_cap) == len(trained) and len(refs_at_cap) == len(refs),
        },
        "erratum": erratum,
        "pairing": {"round0_ftl_bc_iid_identical_within_condition":
                    all(round0.values()), "per_condition": round0},
        "shared_preparation": shared,
        "campaign": campaign,
        "notes": [
            "Cross-entropy and disagreement are on each learner's own visited states.",
            "Cross-entropy is the average negative natural log probability of the "
            "expert's deterministic action per learner-visited state; lower is better.",
            "The learning-curve figure plots the stored normalized_return (random 0, "
            "expert 1); normalized 1 here corresponds to raw expert_return and, with "
            "expert_return equal to the episode cap, remains a ceiling. Raw returns "
            "are kept in these files.",
            "Timed-out counts are saved counts or observed lower bounds, not totals.",
            "Null values are unknown, never zero.",
            "Controller wall time is worker wall time, not CPU or GPU hours or money.",
            "Original expert preparation, local review and plotting are out of scope.",
            "Round-loop fits are timed within collection_and_training; the historical "
            "offline BC job has no in-run collection, so those recorded 0.0 phase "
            "timers are structural.",
            "runs holds the physical record of all six jobs as recorded, including "
            "every historical BC prefix fit and evaluation in its counts and wall "
            "time; only bc_reference is the offline BC result.",
            "Offline BC is trained on the whole pool and is never label matched to "
            "the online curves.",
        ],
    }
    curve_points = {
        "schema": "cartpole-restriction-pilot-curve-points/2",
        "note": "every saved FTL and BC-iid evaluation; x is labels used for "
        "training; metrics are on each learner's own evaluation states; timed-out "
        "runs end at their last saved evaluation",
        "runs": {j: {"method": rows[j]["method"],
                     "restriction_id": rows[j]["restriction_id"],
                     "controller_state": rows[j]["controller_state"],
                     "points": points[j]} for j in ONLINE_IDS},
        "offline_bc_reference": {
            "note": "one fixed policy per condition trained on the whole pool and "
            "evaluated once; a flat reference, not a curve",
            "runs": {j: {"restriction_id": rows[j]["restriction_id"],
                         "point": refs[j]} for j in BC_IDS},
        },
    }

    md = report(summary, analysis_sha)
    texts = {REPORT: md,
             SUMMARY: json.dumps(summary, indent=2) + "\n",
             POINTS: json.dumps(curve_points, indent=2) + "\n"}
    banned = forbidden_tokens(invs, root, out)
    for name, text in texts.items():
        hits = sorted(t for t in banned if t and t in text)
        require(not hits, f"{name}: refusing to write private or banned text {hits}")
    images = {
        FIG_CURVES: png(curves_figure(points, refs, rows, data["baselines"], cap,
                                      summary["return_ceiling"]["all_at_cap"],
                                      n_episodes, n_rounds)),
        FIG_COSTS: png(costs_figure(rows, shared, job_limit)),
    }

    out.mkdir(parents=True, exist_ok=False)
    for name, text in texts.items():
        with open(out / name, "x", encoding="utf-8") as fh:
            fh.write(text)
    for name, blob in images.items():
        with open(out / name, "xb") as fh:
            fh.write(blob)
    print(f"wrote {len(texts) + len(images)} files to the new output directory")


if __name__ == "__main__":
    main()
