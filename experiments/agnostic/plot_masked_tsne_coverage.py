"""Per-seed t-SNE coverage of masked offline BC, BC-iid and FTL (original design).

A thin adapter: inputs are verified exactly as in ``analyze_masked_coverage.py``
(its manifest, `Reader`, `_env_cells` and `mask_from_description`), and every
seed group is drawn with the original FTRL coverage code, unchanged:
`coverage_features.extract_features`, `tsne_coverage.subsample` and
`fit_shared_tsne` (default perplexity grid 15/30/50, t-SNE seed 0),
`plot_tsne_coverage.render_coverage_figure` (arrival-round heatmap) and
`plot_tsne_coverage.render_coverage_diff` (separate FTL vs BC-iid overlap map
on the same embedding). The analyzer's PCA and coordinate panels are not
produced here. Postprocessing only: no environment, model, expert, training,
network or remote machine is touched. Outputs go to a new directory only::

    PYTHONPATH=src python experiments/agnostic/plot_masked_tsne_coverage.py \\
        --manifest MANIFEST.json --output-dir NEW_DIR [--env CartPole-v1] \\
        [--verify-only]

Per seed 3100..3104 the outputs follow the original naming and cache scheme:
``{env}_seed{seed}.png``, ``{env}_seed{seed}_coverage_diff.png``,
``{env}_seed{seed}.npz`` (features, embedding, arrival rounds) and
``{env}_seed{seed}.json`` (original metrics plus provenance and hashes).

What is embedded. Each method's retained supervised training states for that
seed, after the recorded learner mask is applied to the original states the
expert labeled. Offline BC is one unchanged full fixed pool fit once; it has
no online arrival, so it is drawn at time 0 and labeled a fixed reference.
BC-iid (pinned stream row t) and FTL (scratch round-000 is 1) keep their
one-based label arrival.

Tasks and masks are those of the analyzer: an original task's ladder rung or a
bound calibration candidate (zeroing, alias, quantized or a random-offset
grid, whose footer gives only its K and seed), or a registered
variant task named explicitly, labeled as such in the footer and records and
never promoted to a main environment.

Status. A seed is rendered only if its data job and all three masked runs are
complete and the environment's controls, provenance and mask are verified;
any other seed is kept in the inventory with its reasons, never dropped.
Exit codes: 0 every selected seed rendered (or verified, with
``--verify-only``), 3 inventory only for at least one seed, 2 refused.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import os
import pathlib
import shutil
import sys
import tempfile
import textwrap
from typing import Any, Dict, Optional, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import numpy as np  # noqa: E402

from imitation.experiments.agnostic import pilot  # noqa: E402
from imitation.experiments.agnostic import tasks  # noqa: E402
from imitation.experiments.ftrl import coverage_data  # noqa: E402
from imitation.experiments.ftrl import coverage_features  # noqa: E402
from imitation.experiments.ftrl import plot_tsne_coverage  # noqa: E402
from imitation.experiments.ftrl import tsne_coverage  # noqa: E402


def _load_analyzer():
    path = pathlib.Path(__file__).resolve().parent / "analyze_masked_coverage.py"
    spec = importlib.util.spec_from_file_location("analyze_masked_coverage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


amc = _load_analyzer()

INVENTORY_SCHEMA = "agnostic-masked-tsne-coverage-inventory/1"
SEED_SCHEMA = "agnostic-masked-tsne-coverage-seed/1"
# Defaults of plot_tsne_coverage.main / build_and_plot.
PERPLEXITIES = (15, 30, 50)
TSNE_SEEDS = (0,)
CAP = 8000
# Offline BC has no online arrival: drawn at time 0 under a label that says so.
BC_LABEL = "bc (fixed reference, time 0)"
ALGO_LABEL = {"bc": BC_LABEL, "bc_iid": "bc_iid", "ftl": "ftl"}
ARRIVAL_RULE = {
    "bc": "0 for every row: one unchanged full fixed pool fit once, a fixed "
    "reference with no online arrival (no stream is invented)",
    "bc_iid": "one-based label arrival: pinned stream row t arrives at t",
    "ftl": "one-based label arrival: scratch round-000 arrives at 1",
}
LIMITATION = (
    "Retained labeled samples only: each method's supervised training states "
    "for this seed. Not final-policy occupancy; no evaluation rollouts or "
    "visited but unretained states. Coverage alone does not establish an error "
    "floor, a return ceiling or a causal explanation."
)


def _sha_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    head = f"{a.dtype.str}{list(a.shape)}".encode("utf-8")
    return hashlib.sha256(head + a.tobytes()).hexdigest()


def annotation(env_name, seed, rid, mask_info, manifest_sha) -> str:
    """Figure footer: identity, mask, arrival semantics, provenance, limitation.

    A calibration-candidate mask and a variant task say so; an original
    ladder rung of an original task keeps the original wording.
    """
    kind = mask_info.get("kind")
    if kind == "zeroing":
        hidden = "zeroed " + ", ".join(mask_info["zeroed_features"])
    elif kind == "random_grid":
        # Concise: K and seed only; digests stay in the seed JSON.
        hidden = mask_info["hidden"].rstrip(".")
    elif kind == "quantized":
        hidden = (
            f"quantized: {mask_info['hidden']}; descriptor "
            f"{mask_info['descriptor_sha256'][:12]}"
        )
    elif mask_info.get("alias_family") == "random_state_binning":
        binning = mask_info["random_binning"]
        hidden = (
            f"random state binning: {binning['n_bins']} balanced bins of the "
            f"reachable nonterminal states, partition seed {binning['seed']}, "
            f"{binning['generator']}, each bin seen as its first shuffled "
            f"member; partition {binning['partition_sha256'][:12]}, alias "
            f"table {mask_info['table_sha256'][:12]}"
        )
    else:
        hidden = f"alias table {mask_info.get('table_sha256', '')[:12]}"
    scope = ""
    if mask_info.get("registry") == "calibration_candidate":
        scope += (
            "The mask is an opt-in calibration candidate, not an original "
            "ladder rung. "
        )
    if amc.task_scope(env_name) == "variant":
        scope += (
            "The environment is a variant task (opt-in diagnostic, not a main "
            "environment). "
        )
    return (
        f"{env_name} seed {seed}. Learner mask {rid} ({hidden}) applied to the "
        "original expert-labeled states before feature extraction; one t-SNE "
        "shared by all methods of this seed. "
        + scope
        + f"{BC_LABEL}: offline BC, one unchanged full pool fit once, no online "
        "arrival. bc_iid (pinned stream) and ftl (retained scratch) colored by "
        "one-based label arrival. "
        + LIMITATION
        + f" Inputs verified against coverage manifest {manifest_sha[:12]}; "
        "per-file SHA256 in the seed JSON."
    )


@contextlib.contextmanager
def footer(text: str):
    """Add ``text`` below every figure saved inside the block.

    The original renderers build, save and close their own figures; this adds
    one footer line block at save time without touching their layout, markers,
    colors or legends (``bbox_inches="tight"`` keeps it in frame).
    """
    original = matplotlib.figure.Figure.savefig

    def savefig(fig, *args, **kwargs):
        width = int(fig.get_figwidth() * 14.5)
        fig.text(
            0.0,
            -0.01,
            "\n".join(textwrap.wrap(text, width)),
            ha="left",
            va="top",
            fontsize=7,
            color="#333333",
        )
        return original(fig, *args, **kwargs)

    matplotlib.figure.Figure.savefig = savefig
    try:
        yield
    finally:
        matplotlib.figure.Figure.savefig = original


def render_seed(env_name, seed, samples, mask, mask_info, rid, manifest_sha, out):
    """One seed group with the original pipeline; returns the seed record."""
    states = [
        coverage_data.CoverageStates(
            algo=ALGO_LABEL[m],
            obs=mask.apply(samples[m].obs),
            rounds=(
                np.zeros(len(samples[m].obs), dtype=int)
                if samples[m].arrival is None
                else samples[m].arrival.astype(int)
            ),
        )
        for m in amc.METHODS
    ]
    pooled = coverage_data.pool(states)
    feats_full = coverage_features.extract_features(pooled.obs, env_name)
    feats, algo, rounds, idx = tsne_coverage.subsample(
        feats_full, pooled.algo, pooled.rounds, cap=CAP, seed=TSNE_SEEDS[0]
    )
    sub = coverage_data.PooledStates(obs=feats, algo=algo, rounds=rounds)
    result = tsne_coverage.fit_shared_tsne(feats, PERPLEXITIES, TSNE_SEEDS)
    metrics_2d = tsne_coverage.coverage_metrics_2d(result.embedding, algo)
    metrics_hd = tsne_coverage.coverage_metrics_highdim(feats, algo)
    metrics_uniq = tsne_coverage.coverage_metrics_unique(feats, algo)
    n_rounds = int(pooled.rounds.max()) if len(pooled.rounds) else 0
    stem = f"{env_name.replace('/', '_')}_seed{seed}"
    png = out / f"{stem}.png"
    diff = out / f"{stem}_coverage_diff.png"
    title = f"{env_name} seed {seed}, learner mask {rid}"
    with footer(annotation(env_name, seed, rid, mask_info, manifest_sha)):
        plot_tsne_coverage.render_coverage_figure(
            sub,
            result,
            png,
            title,
            n_rounds,
            metrics_2d,
            metrics_hd,
            metrics_uniq=metrics_uniq,
        )
        plot_tsne_coverage.render_coverage_diff(result.embedding, algo, diff, title)
    original_obs = np.concatenate([samples[m].obs for m in amc.METHODS])[idx]
    expert_action = np.concatenate([samples[m].acts for m in amc.METHODS])[idx]
    cache = {
        # Keys of the original build_and_plot cache, then the sample rows.
        "embedding": result.embedding,
        "features": feats,
        "algo": algo.astype(str),
        "rounds": rounds,
        "perplexity": result.perplexity,
        "trustworthiness": result.trustworthiness,
        "dataset_seed": seed,
        "embedding_seed": result.seed,
        "search_seeds": np.asarray(TSNE_SEEDS),
        "search_perplexities": np.asarray(PERPLEXITIES),
        "masked_obs": pooled.obs[idx],
        "original_obs": original_obs,
        "expert_action": expert_action,
        "subsample_index": idx,
    }
    with open(out / f"{stem}.npz", "xb") as f:
        np.savez(f, **cache)
    return {
        "schema": SEED_SCHEMA,
        # Keys of the original metrics JSON.
        "dataset_seed": seed,
        "embedding_seed": result.seed,
        "perplexity": result.perplexity,
        "training_counts": {s.algo: len(s.obs) for s in states},
        "display_counts": {
            str(label): int(np.sum(algo == label)) for label in np.unique(algo)
        },
        "coverage_2d": metrics_2d,
        "coverage_highdim": metrics_hd,
        "coverage_unique": metrics_uniq,
        "trustworthiness": result.trustworthiness,
        # Identity, semantics and hashes.
        "env_name": env_name,
        "restriction_id": rid,
        "restriction": mask_info,
        "feature_mode": coverage_features.feature_mode(env_name),
        "features_from": "masked observations (recorded learner mask applied "
        "to the original states), extract_features on all methods pooled",
        "n_rounds_colorbar": n_rounds,
        "algo_labels": dict(ALGO_LABEL),
        "arrival_rule": ARRIVAL_RULE,
        "origin": {m: samples[m].origin for m in amc.METHODS},
        "excluded_final_observations": {
            m: samples[m].excluded_final_observations for m in amc.METHODS
        },
        "coverage_diff_sides": {
            k: list(v) for k, v in plot_tsne_coverage.COVERAGE_DIFF_SIDES.items()
        },
        "search": {
            "perplexities": list(PERPLEXITIES),
            "tsne_seeds": list(TSNE_SEEDS),
            "cap": CAP,
        },
        "annotation": annotation(env_name, seed, rid, mask_info, manifest_sha),
        "limitation": LIMITATION,
        "array_sha256": {
            k: _sha_array(np.asarray(cache[k]))
            for k in (
                "features",
                "embedding",
                "rounds",
                "masked_obs",
                "original_obs",
                "expert_action",
            )
        },
        "outputs": {
            "heatmap": png.name,
            "overlap_map": diff.name,
            "cache": f"{stem}.npz",
            "embedding_shared_by": [png.name, diff.name],
        },
    }


def plot_env(reader, env_name, env, out, manifest_sha, verify_only):
    """Verify one environment and render each complete seed."""
    task = tasks.get_task(env_name)
    rid = env["restriction_id"]
    control, provenance, _, seeds, samples, descriptions = amc._env_cells(
        reader, env_name, env
    )
    reasons = []
    mask = mask_info = None
    if coverage_features.feature_mode(env_name) != "scaler":
        reasons.append("CNN features need an expert model; not loaded here")
    if descriptions:
        if any(amc._norm(d) != amc._norm(descriptions[0]) for d in descriptions):
            reasons.append("runs record different restriction descriptions")
        else:
            try:
                mask, mask_info = amc.mask_from_description(descriptions[0], rid, task)
            except amc.Ineligible as exc:
                reasons.append(f"mask: {exc}")
    env_ok = control.status == provenance.status == "complete" and not reasons
    record = {
        "env_name": env_name,
        "task_scope": amc.task_scope(env_name),
        "restriction_id": rid,
        "reasons": reasons,
        "control": control.as_dict(),
        "expert_provenance": provenance.as_dict(),
        "seeds": {},
    }
    for seed in amc.SEEDS:
        cells = seeds[seed]
        complete = env_ok and all(c.status == "complete" for c in cells.values())
        complete = complete and set(samples[seed]) == set(amc.METHODS)
        entry = {
            "status": "complete" if complete else "inventory_only",
            "cells": {k: c.as_dict() for k, c in cells.items()},
        }
        if complete and not verify_only:
            seed_record = render_seed(
                env_name, seed, samples[seed], mask, mask_info, rid, manifest_sha, out
            )
            seed_record["manifest_sha256"] = manifest_sha
            seed_record["task_scope"] = amc.task_scope(env_name)
            seed_record["inputs"] = {
                "control": control.files,
                "expert_provenance": provenance.files,
                "cells": {k: c.files for k, c in cells.items()},
                "job_ids": {k: c.job_id for k, c in cells.items()},
            }
            name = seed_record["outputs"]["cache"].replace(".npz", ".json")
            amc._write_json(out / name, seed_record)
            entry["outputs"] = sorted(
                [name, seed_record["outputs"]["cache"]]
                + seed_record["outputs"]["embedding_shared_by"]
            )
        record["seeds"][str(seed)] = entry
    return record


def inventory_markdown(inventory: Dict[str, Any]) -> str:
    lines = [
        "# Masked t-SNE coverage (original design): inventory",
        "",
        f"Manifest SHA256 `{inventory['manifest_sha256']}`.",
        "",
        "| environment | seed | status | data | bc | bc_iid | ftl |",
        "|---|---|---|---|---|---|---|",
    ]
    for env_name, env in inventory["environments"].items():
        for seed, entry in env.get("seeds", {}).items():
            c = entry["cells"]
            lines.append(
                f"| {env_name} | {seed} | {entry['status']} | "
                + " | ".join(c[k]["status"] for k in ("data",) + amc.METHODS)
                + " |"
            )
    lines += ["", "## Problems", ""]
    for env_name, env in inventory["environments"].items():
        for reason in env.get("reasons", []):
            lines.append(f"- {env_name}: {reason}")
        for name in ("control", "expert_provenance"):
            for reason in (env.get(name) or {}).get("reasons", []):
                lines.append(f"- {env_name} {name}: {reason}")
        for seed, entry in env.get("seeds", {}).items():
            for name, cell in entry["cells"].items():
                if cell["status"] != "complete" or cell["reasons"]:
                    lines.append(
                        f"- {env_name} seed {seed} {name} ({cell['status']}): "
                        + "; ".join(cell["reasons"])
                    )
    lines += ["", "## Limitation", "", f"- {LIMITATION}"]
    return "\n".join(lines) + "\n"


def plot(manifest_path, output_dir, envs=None, verify_only=False) -> int:
    """Run the adapter; see the module docstring for exit codes."""
    output_dir = pathlib.Path(output_dir)
    try:
        if output_dir.exists():
            raise amc.Refused(f"{output_dir} exists; outputs go to a new directory")
        manifest, manifest_sha, bundle = amc.load_manifest(manifest_path)
        selected = list(envs) if envs else list(manifest["environments"])
        unknown = [e for e in selected if e not in manifest["environments"]]
        if unknown:
            raise amc.Refused(f"environments not in the manifest: {unknown}")
        index = amc.load_indexes(bundle, manifest)
        parent = output_dir.absolute().parent
        if not parent.is_dir():
            raise amc.Refused(f"{parent} does not exist")
        if bundle == output_dir.absolute() or bundle in output_dir.absolute().parents:
            raise amc.Refused("the output directory must be outside the input bundle")
    except amc.Refused as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return amc.EXIT_REFUSED
    strict = manifest.get("require_index_corroboration", False)
    reader = amc.Reader(bundle, index, strict)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        inventory: Dict[str, Any] = {
            "schema": INVENTORY_SCHEMA,
            "manifest_sha256": manifest_sha,
            "provenance_indexes": manifest.get("provenance_indexes", []),
            "provenance_index_entries": len(index),
            "require_index_corroboration": reader.strict,
            "verify_only": bool(verify_only),
            "selected_environments": selected,
            "environments": {
                env_name: plot_env(
                    reader,
                    env_name,
                    manifest["environments"][env_name],
                    staging,
                    manifest_sha,
                    verify_only,
                )
                for env_name in selected
            },
        }
        amc._write_json(staging / "tsne-coverage-inventory.json", inventory)
        with open(staging / "tsne-coverage-inventory.md", "x") as f:
            f.write(inventory_markdown(inventory))
        artifacts = {p.name: pilot.sha256_file(p) for p in sorted(staging.iterdir())}
        amc._write_json(
            staging / "tsne-coverage-artifacts.json",
            {"manifest_sha256": manifest_sha, "files": artifacts},
        )
        os.rename(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    done = all(
        entry["status"] == "complete"
        for env in inventory["environments"].values()
        for entry in env["seeds"].values()
    )
    return amc.EXIT_OK if done else amc.EXIT_INVENTORY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--env", action="append", choices=amc.known_envs())
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify inputs and write the inventory without rendering",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return plot(args.manifest, args.output_dir, args.env, args.verify_only)


if __name__ == "__main__":
    sys.exit(main())
