"""Per-seed masked t-SNE coverage adapter on synthetic inputs only.

The coherent synthetic bundles of ``test_agnostic_masked_coverage.py`` (small
label budget, every pin recomputed) are reused, rendered once per module and
broken in exactly one place where a failure is tested. No environment is
created, no model or expert is loaded and no real experiment record is read;
nothing produced here is scientific evidence.
"""

import copy
import hashlib
import importlib.util
import io
import json
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytest

from imitation.experiments.agnostic import restriction
from imitation.experiments.ftrl import (
    coverage_features,
    plot_tsne_coverage,
    tsne_coverage,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ADAPTER = REPO_ROOT / "experiments" / "agnostic" / "plot_masked_tsne_coverage.py"
BASE_TESTS = pathlib.Path(__file__).with_name("test_agnostic_masked_coverage.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tsne = _load("masked_tsne_coverage", ADAPTER)
base = _load("masked_coverage_bundles", BASE_TESTS)

N = base.N
CART, CART_MASK = base.CART, base.CART_MASK
SEEDS = tsne.amc.SEEDS
STEM = {seed: f"{CART}_seed{seed}" for seed in SEEDS}


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text())


@pytest.fixture(autouse=True, scope="module")
def small_budget():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsne.amc, "N_LABELS", N)
        base.narrow_registry(mp)
        yield


def run(root, **kwargs):
    out = pathlib.Path(root).parent / "out"
    code = tsne.plot(pathlib.Path(root) / "manifest.json", out, **kwargs)
    inventory = None
    if (out / "tsne-coverage-inventory.json").exists():
        inventory = read_json(out / "tsne-coverage-inventory.json")
    return code, out, inventory


@pytest.fixture(scope="module")
def cart_base(tmp_path_factory, small_budget):
    return base.build_bundle(tmp_path_factory.mktemp("cart"), CART)


@pytest.fixture
def cart(cart_base, tmp_path):
    shutil.copytree(cart_base, tmp_path / "case")
    return tmp_path / "case"


@pytest.fixture(scope="module")
def rendered(cart_base, tmp_path_factory, small_budget):
    """One complete render with spies on the original renderers and t-SNE."""
    calls = {"fit": [], "figure": [], "diff": []}
    with pytest.MonkeyPatch.context() as mp:
        for module, name, key in (
            (tsne_coverage, "fit_shared_tsne", "fit"),
            (plot_tsne_coverage, "render_coverage_figure", "figure"),
            (plot_tsne_coverage, "render_coverage_diff", "diff"),
        ):
            real = getattr(module, name)

            def spy(*args, _real=real, _key=key, **kwargs):
                result = _real(*args, **kwargs)
                calls[_key].append((args, kwargs, result))
                return result

            mp.setattr(module, name, spy)

        def no_pca(*args, **kwargs):
            raise AssertionError("superseded PCA/coordinate panels were drawn")

        mp.setattr(tsne.amc, "plot_continuous", no_pca)
        mp.setattr(tsne.amc, "pca", no_pca)
        root = tmp_path_factory.mktemp("render")
        shutil.copytree(cart_base, root / "case")
        before = {p: sha(p) for p in sorted((root / "case").rglob("*")) if p.is_file()}
        code, out, inventory = run(root / "case")
        after = {p: sha(p) for p in sorted((root / "case").rglob("*")) if p.is_file()}
    assert before == after  # inputs untouched
    return {"code": code, "out": out, "inventory": inventory, "calls": calls}


# ---------------------------------------------------------------------------
# Complete inputs: one original-design seed group per seed
# ---------------------------------------------------------------------------


def test_every_seed_gets_heatmap_overlap_map_and_cache(rendered):
    assert rendered["code"] == tsne.amc.EXIT_OK
    out, inventory = rendered["out"], rendered["inventory"]
    seeds = inventory["environments"][CART]["seeds"]
    assert list(seeds) == [str(s) for s in SEEDS]
    for seed in SEEDS:
        stem = STEM[seed]
        assert seeds[str(seed)]["status"] == "complete"
        assert seeds[str(seed)]["outputs"] == sorted(
            [
                f"{stem}.json",
                f"{stem}.npz",
                f"{stem}.png",
                f"{stem}_coverage_diff.png",
            ]
        )
        for name in seeds[str(seed)]["outputs"]:
            assert (out / name).stat().st_size > 0
    # Nothing from the superseded analyzer layout.
    assert not list(out.glob("coverage-*"))
    files = read_json(out / "tsne-coverage-artifacts.json")["files"]
    assert set(files) == {p.name for p in out.iterdir()} - {
        "tsne-coverage-artifacts.json"
    }
    for name, digest in files.items():
        assert sha(out / name) == digest


def test_original_tsne_grid_and_one_embedding_per_seed(rendered):
    calls = rendered["calls"]
    assert len(calls["fit"]) == len(calls["figure"]) == len(calls["diff"]) == 5
    for (f_args, _, result), (g_args, g_kw, _), (d_args, d_kw, _) in zip(
        calls["fit"], calls["figure"], calls["diff"]
    ):
        assert tuple(f_args[1]) == (15, 30, 50) and tuple(f_args[2]) == (0,)
        assert len(f_args[0]) == 3 * N  # all three methods in one embedding
        assert g_args[1] is result  # heatmap on this seed's embedding
        assert d_args[0] is result.embedding  # overlap map on the same one
        assert g_args[4] == N  # colorbar 0..N rounds
        assert g_kw.get("point_size") is None  # original auto marker size
        # Original comparison sides: no override of ftl vs bc_iid.
        assert "left_algos" not in d_kw and "right_algos" not in d_kw
        assert len(d_args) == 4
    assert plot_tsne_coverage.COVERAGE_DIFF_SIDES == {
        "left": ("ftl",),
        "right": ("bc_iid",),
    }


def test_features_are_original_extraction_of_masked_states(rendered):
    out = rendered["out"]
    mask = restriction.RESTRICTIONS[CART_MASK]
    for seed in SEEDS:
        cache = np.load(out / f"{STEM[seed]}.npz", allow_pickle=False)
        assert np.array_equal(cache["masked_obs"], mask.apply(cache["original_obs"]))
        expected = coverage_features.extract_features(cache["masked_obs"], CART)
        assert np.allclose(cache["features"], expected)
        assert np.all(cache["features"][:, [0, 3]] == 0)  # x, theta_dot hidden
        assert np.all(np.isfinite(cache["features"]))
        assert cache["embedding"].shape == (3 * N, 2)
        assert not np.any(cache["original_obs"] == base.SENTINEL)
        algo = cache["algo"]
        ftl = algo == "ftl"
        obs, acts = base.make_pairs(CART, seed, "ftl")
        assert np.array_equal(cache["original_obs"][ftl], obs)
        assert np.array_equal(cache["expert_action"][ftl], acts)
        assert int(cache["dataset_seed"]) == seed
        assert list(cache["search_perplexities"]) == [15, 30, 50]


def test_arrival_one_based_and_offline_bc_fixed_reference_at_time_zero(rendered):
    out = rendered["out"]
    for seed in SEEDS:
        cache = np.load(out / f"{STEM[seed]}.npz", allow_pickle=False)
        algo, rounds = cache["algo"], cache["rounds"]
        assert set(algo) == {tsne.BC_LABEL, "bc_iid", "ftl"}
        assert np.all(rounds[algo == tsne.BC_LABEL] == 0)
        for m in ("bc_iid", "ftl"):
            assert np.array_equal(rounds[algo == m], np.arange(1, N + 1))
        pool = base.rp.ExpertData.load(
            rendered["out"].parent / f"case/bundle/data-seed{seed}/fixed_bc_pool.npz"
        )
        assert np.array_equal(cache["original_obs"][algo == tsne.BC_LABEL], pool.obs)
    # Panel order bc, bc_iid, ftl is kept with the longer BC label.
    assert sorted([tsne.BC_LABEL, "bc_iid", "ftl"])[0] == tsne.BC_LABEL


def test_seed_json_keeps_original_metrics_and_provenance(rendered):
    out = rendered["out"]
    for seed in SEEDS:
        record = read_json(out / f"{STEM[seed]}.json")
        for key in (
            "dataset_seed",
            "embedding_seed",
            "perplexity",
            "training_counts",
            "display_counts",
            "coverage_2d",
            "coverage_highdim",
            "coverage_unique",
            "trustworthiness",
        ):
            assert key in record
        assert record["dataset_seed"] == seed
        assert record["display_counts"] == {tsne.BC_LABEL: N, "bc_iid": N, "ftl": N}
        assert record["restriction"]["zeroed_features"] == ["x", "theta_dot"]
        assert record["manifest_sha256"] == sha(out.parent / "case/manifest.json")
        cells = record["inputs"]["cells"]
        assert set(cells) == {"data", "bc", "bc_iid", "ftl"}
        assert cells["ftl"]["scratch_demos"]["tree_sha256"]
        assert record["inputs"]["job_ids"]["ftl"].endswith(f"seed{seed}")
        cache = np.load(out / f"{STEM[seed]}.npz", allow_pickle=False)
        for key, digest in record["array_sha256"].items():
            assert tsne._sha_array(cache[key]) == digest
        assert "Retained labeled samples only" in record["annotation"]
        assert "fixed reference" in record["annotation"]
        assert chr(0x2014) not in json.dumps(record, ensure_ascii=False)


def test_footer_annotates_every_saved_figure_and_restores_savefig():
    original = plt.Figure.savefig
    fig = plt.figure()
    with tsne.footer("provenance note"):
        fig.savefig(io.BytesIO(), format="png")
    assert plt.Figure.savefig is original
    assert [t.get_text() for t in fig.texts] == ["provenance note"]
    plt.close(fig)


# ---------------------------------------------------------------------------
# Incomplete inputs stay in the inventory
# ---------------------------------------------------------------------------


def test_missing_seed_is_inventory_only_and_others_render(cart):
    base.edit_manifest(cart, lambda m: m["environments"][CART]["seeds"].pop("3102"))
    code, out, inventory = run(cart)
    assert code == tsne.amc.EXIT_INVENTORY
    seeds = inventory["environments"][CART]["seeds"]
    assert seeds["3102"]["status"] == "inventory_only"
    assert all(c["status"] == "missing" for c in seeds["3102"]["cells"].values())
    assert not list(out.glob(f"{STEM[3102]}*"))
    assert (out / f"{STEM[3101]}.png").exists()
    table = (out / "tsne-coverage-inventory.md").read_text()
    assert "| 3102 | inventory_only |" in table


def test_forged_ftl_scratch_keeps_seed_inventory_only(cart):
    demos = cart / f"bundle/run-ftl-{CART_MASK}-seed3104/scratch"
    demos = demos / f"ftl_{CART}_seed3104/demos"
    shutil.rmtree(demos)
    obs, acts = base.make_pairs(CART, 3104, "ftl")
    base.write_scratch(demos, obs, 1 - acts)
    base.repin(cart)
    code, out, inventory = run(cart)
    assert code == tsne.amc.EXIT_INVENTORY
    entry = inventory["environments"][CART]["seeds"]["3104"]
    assert entry["status"] == "inventory_only"
    assert entry["cells"]["ftl"]["status"] == "ineligible"
    assert not list(out.glob(f"{STEM[3104]}*"))


def test_running_record_is_partial_and_not_rendered(cart):
    path = cart / f"bundle/run-bc_iid-{CART_MASK}-seed3100/result.json"
    base.edit_json(path, lambda r: r.update(status="running"))
    base.repin(cart)
    code, out, inventory = run(cart)
    assert code == tsne.amc.EXIT_INVENTORY
    entry = inventory["environments"][CART]["seeds"]["3100"]
    assert entry["cells"]["bc_iid"]["status"] == "partial"
    assert not (out / f"{STEM[3100]}.png").exists()


def test_inconsistent_mask_description_blocks_every_seed(cart):
    path = cart / f"bundle/run-ftl-{CART_MASK}-seed3101/result.json"
    configs = cart / "bundle/control/run-configs.json"
    job = f"run-ftl-{CART_MASK}-seed3101"

    def change(config):
        config["restriction"]["description"] = "edited"
        config["restriction_sha256"] = base.pilot.config_digest(config["restriction"])

    def record(r):
        change(r["config"])
        r["config_sha256"] = base.pilot.config_digest(r["config"])

    base.edit_json(path, record)
    base.edit_json(configs, lambda c: c.update({job: read_json(path)["config"]}))
    manifest = cart / "bundle/control/run-manifest.json"

    def expected(doc):
        for j in doc["jobs"]:
            if j["id"] == job:
                j["expected_fields"]["config_sha256"] = read_json(path)["config_sha256"]

    base.edit_json(manifest, expected)
    base.repin(cart)
    code, out, inventory = run(cart)
    assert code == tsne.amc.EXIT_INVENTORY
    env = inventory["environments"][CART]
    assert env["seeds"]["3101"]["cells"]["ftl"]["status"] == "complete"
    assert any("restriction descriptions" in r for r in env["reasons"])
    assert all(e["status"] == "inventory_only" for e in env["seeds"].values())
    assert not list(out.glob("*.png"))


def test_verify_only_writes_inventory_without_rendering(cart, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("rendered in verify-only mode")

    monkeypatch.setattr(tsne_coverage, "fit_shared_tsne", boom)
    code, out, inventory = run(cart, verify_only=True)
    assert code == tsne.amc.EXIT_OK
    assert inventory["verify_only"] is True
    assert sorted(p.name for p in out.iterdir()) == [
        "tsne-coverage-artifacts.json",
        "tsne-coverage-inventory.json",
        "tsne-coverage-inventory.md",
    ]


def test_refusals_write_nothing(cart):
    out = cart.parent / "out"
    out.mkdir()
    (out / "keep").write_text("x")
    assert tsne.plot(cart / "manifest.json", out) == tsne.amc.EXIT_REFUSED
    assert [p.name for p in out.iterdir()] == ["keep"]
    base.edit_manifest(cart, lambda m: m.update(labels=N + 1))
    other = cart.parent / "other"
    assert tsne.plot(cart / "manifest.json", other) == tsne.amc.EXIT_REFUSED
    assert not other.exists()


# ---------------------------------------------------------------------------
# No environment, model or checkpoint; originals untouched
# ---------------------------------------------------------------------------


def test_no_environment_model_or_checkpoint_is_touched(cart, monkeypatch):
    import gymnasium
    import stable_baselines3
    import torch

    from imitation.experiments.ftrl import env_utils

    def boom(*args, **kwargs):
        raise AssertionError("forbidden call")

    monkeypatch.setattr(gymnasium, "make", boom)
    monkeypatch.setattr(torch, "load", boom)
    monkeypatch.setattr(stable_baselines3.PPO, "load", boom)
    monkeypatch.setattr(env_utils, "make_env", boom)
    monkeypatch.setattr(coverage_features, "load_expert_policy", boom)
    monkeypatch.setattr(base.tasks, "_raw_env", boom)
    monkeypatch.setattr(base.tasks, "alias_table", boom)
    code, _, _ = run(cart, verify_only=True)
    assert code == tsne.amc.EXIT_OK


def test_adapter_calls_original_modules_and_no_loader():
    source = ADAPTER.read_text()
    for call in (
        "coverage_features.extract_features(",
        "tsne_coverage.subsample(",
        "tsne_coverage.fit_shared_tsne(",
        "plot_tsne_coverage.render_coverage_figure(",
        "plot_tsne_coverage.render_coverage_diff(",
        "amc._env_cells(",
        "amc.mask_from_description(",
    ):
        assert call in source
    for forbidden in ("PPO.load", "load_expert_policy", "gym.make", "torch.load"):
        assert forbidden not in source
    assert chr(0x2014) not in source


# ---------------------------------------------------------------------------
# Generic tasks: quantized candidate and registered variants, same design
# ---------------------------------------------------------------------------

MCAR, TOY_QUANT = base.MCAR, base.TOY_QUANT
VARIED, VARIED_MASK = base.VARIED, base.VARIED_MASK


@pytest.fixture(scope="module")
def mcar_rendered(tmp_path_factory, small_budget):
    root = base.build_bundle(tmp_path_factory.mktemp("mcar"), MCAR)
    code, out, inventory = run(root)
    return {"code": code, "out": out, "inventory": inventory}


def test_quantized_candidate_gets_the_unchanged_per_seed_design(mcar_rendered):
    assert mcar_rendered["code"] == tsne.amc.EXIT_OK, mcar_rendered["inventory"]
    out = mcar_rendered["out"]
    env = mcar_rendered["inventory"]["environments"][MCAR]
    assert env["task_scope"] == "original"
    toy = base.toy_quantized()
    for seed in SEEDS:
        stem = f"{MCAR}_seed{seed}"
        assert env["seeds"][str(seed)]["outputs"] == sorted(
            [f"{stem}.json", f"{stem}.npz", f"{stem}.png", f"{stem}_coverage_diff.png"]
        )
        cache = np.load(out / f"{stem}.npz", allow_pickle=False)
        assert np.array_equal(cache["masked_obs"], toy.apply(cache["original_obs"]))
        expected = coverage_features.extract_features(cache["masked_obs"], MCAR)
        assert np.allclose(cache["features"], expected)
        algo, rounds = cache["algo"], cache["rounds"]
        assert set(algo) == {tsne.BC_LABEL, "bc_iid", "ftl"}
        assert np.all(rounds[algo == tsne.BC_LABEL] == 0)
        for m in ("bc_iid", "ftl"):
            assert np.array_equal(rounds[algo == m], np.arange(1, N + 1))
        record = read_json(out / f"{stem}.json")
        assert record["restriction"]["kind"] == "quantized"
        assert record["restriction"]["registry"] == "calibration_candidate"
        assert record["task_scope"] == "original"
        assert "quantized" in record["annotation"]
        assert "calibration candidate" in record["annotation"]
        assert "0.25" in record["annotation"]
        assert record["restriction"]["descriptor_sha256"][:12] in record["annotation"]
        assert chr(0x2014) not in json.dumps(record, ensure_ascii=False)


def test_original_ladder_annotation_is_unchanged(rendered):
    record = read_json(rendered["out"] / f"{STEM[3100]}.json")
    assert record["task_scope"] == "original"
    assert record["restriction"]["registry"] == "ladder"
    assert "variant" not in record["annotation"]
    assert "calibration candidate" not in record["annotation"]
    assert "(zeroed x, theta_dot)" in record["annotation"]


def test_registered_variant_verifies_and_is_labeled_diagnostic(tmp_path):
    root = base.build_bundle(tmp_path / "varied", VARIED)
    code, out, inventory = run(root, verify_only=True)
    assert code == tsne.amc.EXIT_OK, inventory
    env = inventory["environments"][VARIED]
    assert env["task_scope"] == "variant"
    assert all(e["status"] == "complete" for e in env["seeds"].values())
    info = {"kind": "alias", "table_sha256": "a" * 64, "registry": "ladder"}
    text = tsne.annotation(VARIED, 3100, VARIED_MASK, info, "b" * 64)
    assert "variant task (opt-in diagnostic, not a main environment)" in text


def test_variant_selected_but_absent_is_refused_not_complete(cart):
    code, out, _ = run(cart, envs=[CART, VARIED])
    assert code == tsne.amc.EXIT_REFUSED
    assert not out.exists()


def test_cli_accepts_known_variants_only():
    parser = tsne.build_parser()
    args = parser.parse_args(["--manifest", "m", "--output-dir", "o", "--env", VARIED])
    assert args.env == [VARIED]
    with pytest.raises(SystemExit):
        parser.parse_args(["--manifest", "m", "--output-dir", "o", "--env", "Foo-v0"])


# ---------------------------------------------------------------------------
# Random state binnings: static provenance in the unchanged per-seed design
# ---------------------------------------------------------------------------

CLIFF, CLIFF_RANDOM = base.CLIFF, base.CLIFF_RANDOM


def test_random_binning_gets_the_unchanged_per_seed_design(tmp_path, monkeypatch):
    root = base.build_bundle(tmp_path / "cliff", CLIFF)
    base.with_random(monkeypatch)
    base.forbid_environment(monkeypatch)
    code, out, inventory = run(root)
    assert code == tsne.amc.EXIT_OK, inventory
    desc = base.random_description(CLIFF_RANDOM)
    mask, _ = tsne.amc.mask_from_description(
        desc, CLIFF_RANDOM, base.tasks.TASKS[CLIFF]
    )
    partition = base.tasks.random_partition_sha256(CLIFF_RANDOM)
    env = inventory["environments"][CLIFF]
    for seed in SEEDS:
        stem = f"{CLIFF}_seed{seed}"
        assert env["seeds"][str(seed)]["outputs"] == sorted(
            [f"{stem}.json", f"{stem}.npz", f"{stem}.png", f"{stem}_coverage_diff.png"]
        )
        cache = np.load(out / f"{stem}.npz", allow_pickle=False)
        assert np.array_equal(cache["masked_obs"], mask.apply(cache["original_obs"]))
        record = read_json(out / f"{stem}.json")
        info = record["restriction"]
        assert info["alias_family"] == "random_state_binning"
        assert info["registry"] == "calibration_candidate"
        text = record["annotation"]
        assert "random state binning: 20 balanced bins" in text
        assert f"partition seed {base.tasks.RANDOM_BINNING_SEED}" in text
        assert base.tasks.RANDOM_BINNING_GENERATOR in text
        assert f"partition {partition[:12]}" in text
        assert f"alias table {info['table_sha256'][:12]}" in text
        assert "first shuffled member" in text and "smallest" not in text
        assert "calibration candidate" in text and "variant" not in text
        assert chr(0x2014) not in json.dumps(record, ensure_ascii=False)


def test_random_binning_of_the_slippery_variant_is_labeled_diagnostic(monkeypatch):
    rid = "frozenlake_slippery_random_bins6_seed20261003"
    base.with_random(monkeypatch)
    task = base.tasks.get_task(base.SLIP)
    _, info = tsne.amc.mask_from_description(base.random_description(rid), rid, task)
    text = tsne.annotation(base.SLIP, 3100, rid, info, "b" * 64)
    assert "random state binning: 6 balanced bins" in text
    assert "variant task (opt-in diagnostic, not a main environment)" in text
    assert "calibration candidate" in text


# ---------------------------------------------------------------------------
# Random-offset grids: the unchanged per-seed design, footer K and seed only
# ---------------------------------------------------------------------------


def test_random_grid_gets_the_unchanged_per_seed_design(tmp_path, monkeypatch):
    rid = base.tasks.random_grid_id(CART, 8, base.tasks.RANDOM_GRID_SEED)
    record = json.loads(json.dumps(restriction.describe_restriction(rid)))
    original = base.description
    monkeypatch.setattr(
        base,
        "description",
        lambda env: copy.deepcopy(record) if env == CART else original(env),
    )
    root = base.build_bundle(tmp_path / "grid", CART)
    grid = restriction.get_restriction(rid)
    base.forbid_environment(monkeypatch)
    code, out, inventory = run(root)
    assert code == tsne.amc.EXIT_OK, inventory
    env = inventory["environments"][CART]
    for seed in SEEDS:
        stem = f"{CART}_seed{seed}"
        assert env["seeds"][str(seed)]["outputs"] == sorted(
            [f"{stem}.json", f"{stem}.npz", f"{stem}.png", f"{stem}_coverage_diff.png"]
        )
        cache = np.load(out / f"{stem}.npz", allow_pickle=False)
        assert np.array_equal(cache["masked_obs"], grid.apply(cache["original_obs"]))
        expected = coverage_features.extract_features(cache["masked_obs"], CART)
        assert np.allclose(cache["features"], expected)
        algo, rounds = cache["algo"], cache["rounds"]
        assert set(algo) == {tsne.BC_LABEL, "bc_iid", "ftl"}
        assert np.all(rounds[algo == tsne.BC_LABEL] == 0)
        for m in ("bc_iid", "ftl"):
            assert np.array_equal(rounds[algo == m], np.arange(1, N + 1))
        seed_record = read_json(out / f"{stem}.json")
        info = seed_record["restriction"]
        assert info["kind"] == "random_grid" and info["n_bins"] == 8
        text = seed_record["annotation"]
        assert "Random-offset grid: K=8 per axis (9 cells per scalar axis)" in text
        assert f"seed {base.tasks.RANDOM_GRID_SEED}" in text
        assert info["descriptor_sha256"][:12] not in text
        assert chr(0x2014) not in json.dumps(seed_record, ensure_ascii=False)
