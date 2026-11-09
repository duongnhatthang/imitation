"""Behavior tests for the fail-closed agnostic toy analysis.

Inputs are real ``toy.run_experiment`` outputs on a tiny grid with a fixed fake
clock (so no test depends on today's date relative to the campaign cap).
"""

import copy
import datetime as dt
import json
import shutil

import numpy as np
import pytest

from imitation.experiments.agnostic import analyze_toy as an
from imitation.experiments.agnostic import toy

START = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
SEEDS = [1000, 1001, 1002]
GRID = {
    "horizons": [3],
    "alphas": [0.0, 0.2],
    "kappas": [0.0, 1.0],
    "qs": [0, 1],
    "orientations": [0, 1],
}
BUDGET, BATCH, CHECKPOINTS = 8, 2, [4, 8]


def fixed_clock():
    return START


def tiny_config(seed, **overrides):
    kwargs = dict(
        seeds=(seed,),
        horizons=tuple(GRID["horizons"]),
        alphas=tuple(GRID["alphas"]),
        kappas=tuple(GRID["kappas"]),
        qs=tuple(GRID["qs"]),
        orientations=tuple(GRID["orientations"]),
        budget=BUDGET,
        batch=BATCH,
        checkpoints=tuple(CHECKPOINTS),
    )
    kwargs.update(overrides)
    return toy.ToyConfig(**kwargs)


def tiny_manifest(**overrides):
    manifest = {
        "schema": an.MANIFEST_SCHEMA,
        "study": "Tiny test study",
        "protocol": toy.PROTOCOL_VERSION,
        "toy_source_sha256": toy.source_identity()["sha256"],
        "seeds": list(SEEDS),
        "excluded_pilot_seeds": [0],
        "grid": copy.deepcopy(GRID),
        "budget": BUDGET,
        "batch": BATCH,
        "checkpoints": list(CHECKPOINTS),
        "annotation_mode": "deferred",
        "tie_order": [0, 1, 2, 3],
        "analysis": {
            "inference_orientation": 0,
            "confidence_level": 0.95,
            "bootstrap_resamples": 200,
            "bootstrap_seed": 7,
        },
    }
    manifest.update(overrides)
    return manifest


@pytest.fixture(scope="module")
def base_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("results")
    for seed in SEEDS:
        (root / "seed-{}".format(seed)).mkdir()
        result = toy.run_experiment(tiny_config(seed), clock=fixed_clock)
        assert result["status"] == "complete"
        toy.publish_json(root / "seed-{}".format(seed) / "result.json", result)
    return root


@pytest.fixture
def study(tmp_path, base_root):
    root = tmp_path / "results"
    shutil.copytree(str(base_root), str(root))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(tiny_manifest()))

    def run(manifest=None):
        if manifest is not None:
            manifest_path.write_text(json.dumps(manifest))
        out = tmp_path / "analysis-{}".format(len(list(tmp_path.glob("analysis-*"))))
        code = an.main(
            [
                "--manifest",
                str(manifest_path),
                "--results-root",
                str(root),
                "--output-dir",
                str(out),
            ]
        )
        summary = None
        if (out / "summary.json").exists():
            summary = an._strict_loads((out / "summary.json").read_text())
        return code, summary, out

    return root, run


def edit_result(root, seed, fn):
    path = root / "seed-{}".format(seed) / "result.json"
    result = json.loads(path.read_text())
    fn(result)
    path.write_text(json.dumps(result))


def find_cell(result, **attrs):
    for record in result["cells"]:
        if all(record["cell"][k] == v for k, v in attrs.items()):
            return record
    raise AssertionError(attrs)


def failed_checks(summary):
    return {f["check"] for f in summary["failures"]}


def test_end_to_end_complete_analysis(study, tmp_path):
    root, run = study
    code, summary, out = run()
    assert code == an.EXIT_OK and summary["analysis_status"] == "complete"
    assert all(c["status"] == "pass" for c in summary["controls"].values())
    for name in (
        "bc_fixed_equals_bc_iid",
        "kappa0_pathwise",
        "orientation_invariance",
        "mixture_identity",
        "alpha0_certificate",
        "closed_form",
        "erm_refit",
        "ledger",
    ):
        assert summary["controls"][name]["comparisons"] > 0
    results = summary["results"]
    # Two orientations validated, but inference uses one: N is seeds, not 2x.
    assert results["n_seeds"] == len(SEEDS)
    primary = results["primary_family"]["conditions"]
    assert [(e["horizon"], e["alpha"]) for e in primary] == [(3, 0.2)]
    assert primary[0]["bc_minus_ftl_final"]["n_seeds"] == len(SEEDS)
    assert set(primary[0]["bc_minus_ftl_final"]["intervals"]) == {
        "0.95",
        "{:.6g}".format(results["primary_family"]["bonferroni_level"]),
    }
    # Accounting comes from the actual ledgers: deferred FTL asks one label per
    # retained state, the shared expert labels every visited state.
    acc = {r["arm"]: r for r in summary["accounting"]}
    assert acc["FTL-DAgger"]["expert_action_entries"] == BUDGET
    shared = acc["shared expert acquisition (physical)"]
    assert shared["expert_action_entries"] == BUDGET * GRID["horizons"][0]
    assert acc["BC-iid (logical)"]["fits"] == BUDGET // BATCH
    # Public outputs keep only relative paths and seed labels.
    text = (out / "summary.json").read_text() + (out / "report.md").read_text()
    assert str(tmp_path) not in text
    assert {a["path"] for a in summary["input_artifacts"]} == {
        "seed-{}/result.json".format(s) for s in SEEDS
    }
    assert all(len(a["sha256"]) == 64 for a in summary["input_artifacts"])
    assert len(summary["manifest"]["content_sha256"]) == 64
    assert sorted(p.name for p in (out / "figures").iterdir()) == sorted(
        summary["figures"][i].split("/")[1] for i in range(len(summary["figures"]))
    )
    assert len(summary["figures"]) == 6


def test_balanced_q_is_within_seed_average_and_paired(base_root):
    manifest = tiny_manifest()
    checks = an.Checks()
    data, _, _ = an.load_results(base_root, manifest, checks)
    assert not checks.failures
    table = an.seed_table(data, manifest)
    key = (3, 0.2, 1.0)
    d = an.METRICS.index("diff_final")
    q0, q1 = table[key + ("q0", BUDGET)], table[key + ("q1", BUDGET)]
    balanced = table[key + ("balanced", BUDGET)]
    assert balanced.shape[0] == len(SEEDS)
    np.testing.assert_allclose(balanced[:, d], (q0[:, d] + q1[:, d]) / 2)
    # Resampling whole seed blocks: the q contrast of seed s pairs q1 and q0 of
    # the same seed, so opposite per-seed effects cancel exactly.
    values = np.array([1.0, -1.0, 3.0])
    index = np.array([[0, 0, 0], [1, 1, 2]])
    out = an.bootstrap_mean(values, index, [0.5])
    assert out["estimate"] == 1.0 and out["degenerate"] is False
    np.testing.assert_allclose(
        out["intervals"]["0.5"], np.quantile([1.0, 1 / 3], [0.25, 0.75])
    )
    same = an.bootstrap_mean(np.zeros(4), np.zeros((5, 4), dtype=int), [0.95])
    assert same["degenerate"] and "not a universal" in same["interval_note"]
    one = an.bootstrap_mean(np.array([2.0]), None, [0.95])
    assert one["intervals"] is None
    json.dumps(one, allow_nan=False)


def test_missing_and_partial_seeds_are_incomplete_without_summary(study):
    root, run = study
    shutil.rmtree(str(root / "seed-1001"))
    partial = toy.run_experiment(tiny_config(1002, deadline=START), clock=fixed_clock)
    assert partial["status"] == "partial_deadline"
    toy.publish_json(root / "seed-1002" / "result.json", partial)
    code, summary, out = run()
    assert code == an.EXIT_INCOMPLETE
    assert summary["analysis_status"] == "incomplete"
    assert summary["results"] is None and not (out / "figures").exists()
    assert summary["inventory"]["by_status"]["missing"] == ["seed-1001"]
    assert summary["inventory"]["by_status"]["incomplete_partial_deadline"] == [
        "seed-1002"
    ]


def test_deadline_is_operational_not_scientific(study):
    root, run = study
    other = "2026-02-01T00:00:00+00:00"  # seeds may have different deadlines

    def later_deadline(result):
        result["config"]["deadline"] = other
        result["deadline_utc"] = other

    edit_result(root, 1000, later_deadline)
    (root / "seed-0").mkdir()  # an excluded pilot next to the study is ignored
    code, summary, _ = run()
    assert code == an.EXIT_OK
    assert {"entry": "seed-0", "reason": "excluded pilot seed"} in summary["inventory"][
        "ignored_entries"
    ]


def _overran_last_cell(result):
    # The producer's shape when the deadline passed during the last round of
    # the last cell: every cell's data present, but the run is partial.
    result["status"] = "partial_deadline"
    result["cells"][-1]["status"] = "overran_deadline"
    inv = result["inventory"]
    inv["cells_overran_deadline"].append(inv["cells_complete"].pop())


def test_late_run_with_all_data_is_never_promoted(study):
    root, run = study
    edit_result(root, 1000, _overran_last_cell)
    code, summary, out = run()
    assert code == an.EXIT_INCOMPLETE and summary["analysis_status"] == "incomplete"
    assert summary["results"] is None and not (out / "figures").exists()
    assert summary["inventory"]["seeds_complete"] == len(SEEDS) - 1
    art = {a["seed_label"]: a for a in summary["input_artifacts"]}["seed-1000"]
    assert art["status"] == "incomplete_partial_deadline"
    assert len(art["sha256"]) == 64
    assert any("overran deadline" in r for r in art["reasons"])
    assert "overran deadline" in (out / "report.md").read_text()


@pytest.mark.parametrize(
    "tamper",
    [
        # Producer-complete, but some partial or late indicator contradicts it.
        lambda r: (_overran_last_cell(r), r.update(status="complete")),
        lambda r: r["inventory"]["cells_overran_deadline"].append(0),
        lambda r: r["inventory"].update(partial_cell={"index": 0}),
        lambda r: r["cells"][2].update(status="overran_deadline"),
        lambda r: r.update(deadline_overshoot={"seconds_past_deadline": 1.0}),
        lambda r: r.update(error="RuntimeError: boom"),
        # Inconsistent dates fail closed.
        lambda r: r.update(ended_utc="2026-10-08T00:00:00+00:00"),
        lambda r: r.update(started_utc="2026-01-02T00:00:00+00:00"),
        lambda r: r.update(deadline_utc="2026-02-01T00:00:00+00:00"),
        lambda r: r.update(ended_utc="2026-01-01T00:00:00"),
        lambda r: r.pop("ended_utc"),
    ],
)
def test_contradictory_complete_status_fails_closed(study, tamper):
    root, run = study
    edit_result(root, 1000, tamper)
    code, summary, out = run()
    assert code == an.EXIT_FAILED and summary["results"] is None
    assert not (out / "figures").exists()
    art = {a["seed_label"]: a for a in summary["input_artifacts"]}["seed-1000"]
    assert art["status"] == "invalid" and art["reasons"]


@pytest.mark.parametrize(
    "tamper",
    [
        lambda r: r.update(source_sha256="0" * 64),
        lambda r: r.update(seed=1001),
        lambda r: r["config"].update(budget=16),
        lambda r: r["config"].update(annotation_mode="full_trajectory"),
        lambda r: r["config"].pop("tie_order"),
        lambda r: r.update(protocol="other"),
        lambda r: r["cells"][3].update(index=4),
        lambda r: r["cells"][0].pop("exact"),
        lambda r: r["cells"].append(copy.deepcopy(r["cells"][0])),
    ],
)
def test_bad_provenance_or_structure_fails_closed(study, tamper):
    root, run = study
    edit_result(root, 1000, tamper)
    code, summary, _ = run()
    assert code == an.EXIT_FAILED and summary["results"] is None
    art = {a["seed_label"]: a for a in summary["input_artifacts"]}["seed-1000"]
    assert art["status"] == "invalid" and art["reasons"]


def test_non_finite_json_and_duplicate_labels_fail(study):
    root, run = study
    path = root / "seed-1000" / "result.json"
    path.write_text(
        path.read_text().replace('"expert_cost": 0.0', '"expert_cost": NaN', 1)
    )
    shutil.copytree(str(root / "seed-1001"), str(root / "seed-01001"))
    code, summary, _ = run()
    assert code == an.EXIT_FAILED
    details = " ".join(f["detail"] for f in summary["failures"])
    assert "non-canonical" in details and "malformed JSON" in details


def _tamper_checkpoint(fn, **attrs):
    def tamper(result):
        fn(find_cell(result, **attrs)["checkpoints"][-1])

    return tamper


@pytest.mark.parametrize(
    "tamper, check",
    [
        (
            _tamper_checkpoint(
                lambda s: s["bc_fixed"].update(cost=s["bc_fixed"]["cost"] + 1.0),
                alpha=0.2,
                kappa=1.0,
                q=0,
                orientation=0,
            ),
            "bc_fixed_equals_bc_iid",
        ),
        (
            _tamper_checkpoint(
                lambda s: s["ftl"].update(regret=s["ftl"]["regret"] + 0.25),
                alpha=0.2,
                kappa=1.0,
                q=1,
                orientation=0,
            ),
            "mixture_identity",
        ),
        (
            _tamper_checkpoint(
                lambda s: s["ftl"]["counters"].update(expert_action_entries=3),
                alpha=0.2,
                kappa=1.0,
                q=0,
                orientation=1,
            ),
            "ledger",
        ),
        (
            _tamper_checkpoint(
                lambda s: s["ftl"].update(
                    behavior_policy_id_counts=s["ftl"]["behavior_policy_id_counts"][
                        ::-1
                    ]
                ),
                alpha=0.2,
                kappa=1.0,
                q=0,
                orientation=1,
            ),
            "orientation_invariance",
        ),
        (
            lambda r: find_cell(r, alpha=0.0, kappa=1.0, q=0, orientation=0)[
                "exact"
            ].update(class_optimum_cost=0.5),
            "alpha0_certificate",
        ),
    ],
)
def test_control_and_identity_violations_are_rejected(study, tamper, check):
    root, run = study
    edit_result(root, 1001, tamper)
    code, summary, _ = run()
    assert code == an.EXIT_FAILED and summary["results"] is None
    assert check in failed_checks(summary)
    assert summary["controls"][check]["status"] == "fail"


def test_kappa0_pathwise_mismatch_is_rejected(study):
    root, run = study

    def tamper(result):
        snap = find_cell(result, alpha=0.2, kappa=0.0, q=0, orientation=0)[
            "checkpoints"
        ][0]
        # A self-consistent but different BC-iid fit: recovery action flipped.
        bc = snap["bc_iid"]
        bc["policy_id"] ^= 1
        bc["retained_counts"][1] = [0, 0]
        snap["bc_fixed"].update(
            copy.deepcopy(bc), counters=snap["bc_fixed"]["counters"]
        )

    edit_result(root, 1002, tamper)
    code, summary, _ = run()
    assert code == an.EXIT_FAILED
    assert "kappa0_pathwise" in failed_checks(summary)


@pytest.mark.parametrize(
    "change",
    [
        dict(extra_field=1),
        dict(seeds=[0, 1000]),
        dict(seeds=[1001, 1000]),
        dict(grid=dict(GRID, orientations=[0])),
        dict(checkpoints=[3, 8]),
    ],
)
def test_bad_manifest_is_refused_without_output(study, tmp_path, change):
    _, run = study
    code, summary, out = run(tiny_manifest(**change))
    assert code == an.EXIT_INVALID and not out.exists()


def test_existing_output_directory_is_refused(study, tmp_path):
    root, _ = study
    manifest = tmp_path / "manifest.json"
    out = tmp_path / "taken"
    out.mkdir()
    argv = ["--manifest", str(manifest), "--results-root", str(root)]
    assert an.main(argv + ["--output-dir", str(out)]) == an.EXIT_INVALID
    assert list(out.iterdir()) == []
