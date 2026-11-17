"""Masked retained labeled-state coverage adapter on synthetic inputs only.

Coherent synthetic bundles (data jobs, masked runs, retained scratch demos,
campaign manifests and a pinned coverage manifest) are built once per module
with a small label budget, copied per test and broken in exactly one place.
No environment is created, no model or expert is loaded and no real
experiment record is read; nothing produced here is scientific evidence.
"""

import ast
import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import shutil

import datasets
import numpy as np
import pytest

from imitation.data import serialize, types
from imitation.experiments.agnostic import pilot, restriction
from imitation.experiments.agnostic import restriction_pilot as rp
from imitation.experiments.agnostic import tasks

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ANALYZER = REPO_ROOT / "experiments" / "agnostic" / "analyze_masked_coverage.py"


def _load():
    spec = importlib.util.spec_from_file_location("masked_coverage", ANALYZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cov = _load()
datasets.disable_progress_bar()

N = 12  # small synthetic label budget; the production constant is 1000
PROTO = "agnostic-classical-restriction/1"
EXPERT = "e" * 64
SOURCE = {"files": {"a.py": "1" * 64}}
SOURCE["combined_sha256"] = pilot.config_digest(SOURCE["files"])
CART = "CartPole-v1"
CART_MASK = "cart_position_angular_velocity_zero"
LAKE = "FrozenLake-v1"
LAKE_MASK = "frozenlake_column_pairs"
# Registered variant task (opt-in diagnostic, never a main environment).
VARIED = "FrozenLakeVariedStart-v0"
VARIED_MASK = "frozenlake_varied_start_column_pairs"
# Mocked variant and toy quantized candidate: only ever registered through
# `narrow_registry`, never in the public registries.
SLIP = "FrozenLakeSlippery-v0"
SLIP_MASK = "test_slippery_column_pairs"
MCAR = "MountainCar-v0"
TOY_QUANT = "test_mountaincar_bins_pos025_vel010"
ALIAS_MASK = {LAKE: LAKE_MASK, VARIED: VARIED_MASK, SLIP: SLIP_MASK}
# Random state binning (`tasks.RANDOM_ALIAS_SPECS`), bundled end to end on
# CliffWalking only; every other random id is checked through its record.
CLIFF = "CliffWalking-v0"
CLIFF_RANDOM = "cliffwalking_random_bins20_seed20261003"
RANDOM_MASK = {CLIFF: CLIFF_RANDOM}
SENTINEL = 99.0  # trailing (next/terminal) observation value; never a sample


def toy_quantized():
    """A toy MountainCar bin map built with the public descriptor classes."""
    low = [float(np.float32(-1.2)), float(np.float32(-0.07))]
    high = [float(np.float32(0.6)), float(np.float32(0.07))]
    return restriction.QuantizedRestriction(
        restriction_id=TOY_QUANT,
        env_name=MCAR,
        features=(
            restriction.QuantizedFeature(
                "position", 0, "round_to_step", 0.25, None, None, low[0], high[0]
            ),
            restriction.QuantizedFeature(
                "velocity", 1, "sign_dead_zone", None, 0.01, 0.035, low[1], high[1]
            ),
        ),
        obs_shape=(2,),
        description="toy test bins: position step 0.25, velocity dead zone 0.01",
    )


def narrow_registry(mp):
    """Register the toy candidate and the mocked variant on copies only.

    Calibration candidates are narrowed to the toy one, so concurrent edits
    of the public candidate lists by other owners cannot change these tests.
    """
    mp.setattr(
        restriction,
        "QUANTIZED_RESTRICTIONS",
        {**restriction.QUANTIZED_RESTRICTIONS, TOY_QUANT: toy_quantized()},
    )
    mp.setattr(tasks, "CALIBRATION_CANDIDATES", {MCAR: (TOY_QUANT,)})
    lake = tasks.TASKS[LAKE]
    slip = tasks.Task(
        env_name=SLIP,
        episode_cap=lake.episode_cap,
        observation=lake.observation,
        feature_names=lake.feature_names,
        blocks=lake.blocks,
        n_actions=lake.n_actions,
        ladder=(SLIP_MASK,),
        ladder_rule="nested",
    )
    mp.setattr(tasks, "VARIANT_TASKS", {**tasks.VARIANT_TASKS, SLIP: slip})
    rung = tasks.ALIAS_RUNGS[LAKE_MASK]
    mp.setattr(
        tasks,
        "ALIAS_RUNGS",
        {
            **tasks.ALIAS_RUNGS,
            SLIP_MASK: tasks.AliasRung(SLIP_MASK, SLIP, rung.hidden, rung.key),
        },
    )
    mp.setattr(
        tasks,
        "FROZEN_TABLE_SHA256",
        {
            **tasks.FROZEN_TABLE_SHA256,
            SLIP_MASK: tasks.FROZEN_TABLE_SHA256[LAKE_MASK],
        },
    )


def task_meta(env):
    """Synthetic recorded task metadata (no factory is built)."""
    task = tasks.get_task(env)
    meta = {
        "env_name": env,
        "episode_cap": task.episode_cap,
        "declared": {
            "observation": task.observation,
            "feature_names": list(task.feature_names),
            "blocks": list(task.blocks),
            "ladder": list(task.ladder),
            "ladder_rule": task.ladder_rule,
        },
        "synthetic": "test metadata, not a factory description",
    }
    support = tasks.DECLARED_RESET_SUPPORT.get(env)
    if support is not None:
        meta["reset_distribution"] = {
            "support": list(support),
            "probabilities": [
                1.0 / len(support) if s in support else 0.0 for s in range(task.obs_dim)
            ],
            "rule": "uniform over the safe nonterminal states",
        }
    return meta


def task_sha(env):
    return tasks._digest(task_meta(env))


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True))


def read_json(path):
    return json.loads(pathlib.Path(path).read_text())


@pytest.fixture(autouse=True, scope="module")
def small_budget():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cov, "N_LABELS", N)
        narrow_registry(mp)
        yield


# ---------------------------------------------------------------------------
# Synthetic task data
# ---------------------------------------------------------------------------


def lake_table():
    """frozenlake_column_pairs from the default 4x4 map, without an env."""
    cells = "SFFFFHFHFFFHHFFG"
    terminal = {i for i, c in enumerate(cells) if c in "HG"}
    seen, stack = {0}, [0]
    while stack:
        s = stack.pop()
        r, c = divmod(s, 4)
        for dr, dc in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            t = min(max(r + dr, 0), 3) * 4 + min(max(c + dc, 0), 3)
            if t not in seen and t not in terminal:
                seen.add(t)
                stack.append(t)
    table = list(range(16))
    for s in range(16):
        r, c = divmod(s, 4)
        cands = [
            t for t in seen if divmod(t, 4)[0] == r and divmod(t, 4)[1] // 2 == c // 2
        ]
        if cands:
            table[s] = min(cands)
    assert (
        tasks.table_sha256((16,), tuple(table)) == tasks.FROZEN_TABLE_SHA256[LAKE_MASK]
    )
    return table, sorted(seen)


def static_reachable(env):
    """Sorted reachable nonterminal states of a random-binning env, no env built.

    Each set is checked against `tasks.FROZEN_REACHABLE_SHA256`.
    """
    if env == CLIFF:
        states = list(range(37))  # the cliff and goal are 37..47
    elif env == SLIP:
        _, states = lake_table()
    elif env == "Taxi-v3":
        states = [s for s in range(500) if len(set(tasks._taxi_factors(s)[2:])) == 2]
    else:
        assert env == "Blackjack-v1"
        cards = range(1, 11)
        hands = {(a + b, a == 1 or b == 1) for a in cards for b in cards}
        stack = list(hands)
        while stack:
            raw, ace = stack.pop()
            for card in cards:
                nxt = (raw + card, ace or card == 1)
                if nxt[0] <= 21 and nxt not in hands:
                    hands.add(nxt)
                    stack.append(nxt)
        player = {(r + 10, 1) if a and r + 10 <= 21 else (r, 0) for r, a in hands}
        states = sorted(
            tasks.blackjack_index(s, d, a) for s, a in player for d in cards
        )
    pinned = tasks.FROZEN_REACHABLE_SHA256[env]
    assert len(states) == pinned["n_states"]
    assert tasks._digest(states) == pinned["states_sha256"]
    return states


def random_description(rid):
    """A run-config record of a random binning, built without an environment.

    It has the fields of `restriction.describe_restriction` (as read back from
    JSON); the reachability evidence is a stand-in, bound only through the
    static descriptor.
    """
    spec = tasks.RANDOM_ALIAS_SPECS[rid]
    blocks = tasks.get_task(spec.env_name).blocks
    n = int(np.prod(blocks))
    states = static_reachable(spec.env_name)
    groups = tasks.random_state_partition(states, spec.n_bins, spec.seed, n_states=n)
    table = list(tasks.random_alias_table(n, states, spec.n_bins, spec.seed))
    digest = tasks.table_sha256(blocks, tuple(table))
    assert digest == tasks.FROZEN_TABLE_SHA256[rid]
    return json.loads(
        json.dumps(
            {
                "restriction_id": rid,
                "env_name": spec.env_name,
                "blocks": list(blocks),
                "hidden": spec.hidden,
                "representative": table,
                "table_sha256": digest,
                "frozen_table_sha256": digest,
                "reachable_nonterminal_states": len(states),
                "reachable_classes": len(groups),
                "identity_classes_without_reachable_state": n - len(states),
                "reachability": {"synthetic": "stand-in evidence"},
                "rule": tasks.RANDOM_BINNING_RULE,
                "partition": tasks.random_partition_descriptor(rid),
                "partition_sha256": tasks.random_partition_sha256(rid),
                "groups": [list(g) for g in groups],
                "kind": "alias",
                "obs_shape": [int(sum(blocks))],
                "description": "synthetic copy of the frozen random binning",
            },
        ),
    )


def description(env):
    if env in (CART, MCAR):
        # As read back from a JSON record (lists, not tuples).
        rid = CART_MASK if env == CART else TOY_QUANT
        return json.loads(json.dumps(restriction.describe_restriction(rid)))
    if env in RANDOM_MASK:
        return random_description(RANDOM_MASK[env])
    table, _ = lake_table()
    digest = tasks.table_sha256((16,), tuple(table))
    return {
        "restriction_id": ALIAS_MASK[env],
        "env_name": env,
        "blocks": [16],
        "hidden": "column within pairs {0,1}, {2,3}",
        "representative": table,
        "table_sha256": digest,
        "frozen_table_sha256": digest,
        "kind": "alias",
        "obs_shape": [16],
        "description": "synthetic copy of the frozen table",
        "rule": "smallest-index reachable nonterminal state",
    }


def make_pairs(env, seed, method):
    rng = np.random.default_rng([seed, ("bc", "bc_iid", "ftl").index(method)])
    if env == CART:
        obs = rng.normal(0.0, [0.8, 0.5, 0.05, 0.5], size=(N, 4)).astype(np.float32)
        if method == "ftl":
            # Identical after the mask (x and theta_dot hidden), opposite labels.
            obs[0] = [0.5, 0.1, 0.02, 1.0]
            obs[1] = [-0.5, 0.1, 0.02, -1.0]
            # Nearby but not identical after the mask: never a conflict.
            obs[2] = [0.5, 0.1, 0.0200001, 1.0]
        acts = (obs[:, 2] + 0.5 * obs[:, 3] > 0).astype(np.int64)
        if method == "ftl":
            acts[:3] = [1, 0, 0]
        return obs, acts
    if env == MCAR:
        obs = np.stack(
            [rng.uniform(-1.1, 0.5, size=N), rng.uniform(-0.06, 0.06, size=N)], 1
        ).astype(np.float32)
        if method == "ftl":
            # One bin (position 0.25, velocity +0.035), opposite labels.
            obs[0] = [0.30, 0.02]
            obs[1] = [0.26, 0.03]
            # Next position bin: never a conflict with the pair above.
            obs[2] = [0.40, 0.02]
        acts = np.where(obs[:, 1] > 0, 2, 0).astype(np.int64)
        if method == "ftl":
            acts[:3] = [2, 0, 0]
        return obs, acts
    if env in RANDOM_MASK:
        groups = random_description(RANDOM_MASK[env])["groups"]
        states = rng.choice(static_reachable(env), size=N)
        if method == "ftl":
            states[:2] = groups[0][:2]  # one random bin, opposite labels below
        obs = np.eye(48, dtype=np.float32)[states]
        acts = (states % 4).astype(np.int64)
        if method == "ftl":
            acts[:2] = [0, 1]
        return obs, acts
    _, reachable = lake_table()
    states = rng.choice(reachable, size=N)
    if method == "ftl":
        states[:2] = [0, 1]  # one masked class under column pairs
    obs = np.eye(16, dtype=np.float32)[states]
    acts = (states % 4).astype(np.int64)
    if method == "ftl":
        acts[:2] = [0, 3]
    return obs, acts


def expert_data(obs, acts):
    n = len(acts)
    return rp.ExpertData(
        obs=obs,
        acts=acts,
        next_obs=obs.copy(),
        dones=np.zeros(n, dtype=bool),
        episode_index=np.arange(n),
        step_index=np.zeros(n, dtype=np.int64),
    )


def write_scratch(demos, obs, acts, terminal=False):
    for r in range(len(acts)):
        trailing = np.full((1, obs.shape[1]), SENTINEL, dtype=np.float32)
        traj = types.Trajectory(
            obs=np.concatenate([obs[r : r + 1], trailing]),
            acts=acts[r : r + 1],
            infos=None,
            terminal=terminal,
        )
        round_dir = demos / f"round-{r:03d}"
        round_dir.mkdir(parents=True)
        serialize.save(round_dir / "uniform-demo_0000.npz", [traj])


def run_record(env, seed, method, data_sha, pool, stream, trained, demos_rel):
    desc = description(env)
    config = {
        "seed": seed,
        "method": method,
        "restriction_id": desc["restriction_id"],
        "restriction": desc,
        "restriction_sha256": pilot.config_digest(desc),
        "task_sha256": task_sha(env),
        "eval_episodes": 3,
        "experiment": {
            "algo": method,
            "env_name": env,
            "seed": seed,
            "n_rounds": N,
            "samples_per_round": 1,
        },
    }
    metrics = {
        "normalized_return": 0.5,
        "disagreement_rate": 0.1,
        "episode_returns": [1.0, 2.0, 3.0],
    }
    if method == "bc":
        records = [
            {
                "round": 0,
                "n_observations": N,
                "reference": "flat",
                "trained_pairs_sha256": trained,
                **metrics,
            },
        ]
    else:
        records = [{"round": 0, "n_observations": 0, **metrics}]
        records += [{"round": N, "n_observations": N, **metrics}]
    return {
        "schema": PROTO + "/run",
        "protocol": PROTO,
        "status": "complete",
        "env_name": env,
        "seed": seed,
        "method": method,
        "restriction_id": desc["restriction_id"],
        "config": config,
        "config_sha256": pilot.config_digest(config),
        "expert_sha256": EXPERT,
        "episode_cap": tasks.get_task(env).episode_cap,
        "task": task_meta(env),
        "task_sha256": task_sha(env),
        "source": SOURCE,
        "data": {
            "data_result_sha256": data_sha,
            "pool_pairs_sha256": pool.pairs_sha256(),
            "stream_pairs_sha256": stream.pairs_sha256(),
            "acceptance": {"data_result_sha256": data_sha, "producer_source": SOURCE},
        },
        "scratch_demos": demos_rel,
        "trained_pairs_sha256": trained,
        "records": records,
    }


def build_bundle(
    root,
    env,
    *,
    bc_iid_scratch=True,
    ftl_terminal=False,
    scheduled_ids=False,
):
    """One complete synthetic environment and its pinned coverage manifest.

    With ``scheduled_ids`` the campaign job IDs differ from the result folder
    names (as in the non-CartPole cohorts); ``expected_result`` still names
    the folder.
    """
    root = pathlib.Path(root)
    bundle = root / "bundle"
    rid = description(env)["restriction_id"]
    run_jobs, data_jobs, configs = [], [], {}
    seeds = {}
    for seed in cov.SEEDS:
        data_id = f"stage3-data-{env}-seed{seed}" if scheduled_ids else None
        data_id = data_id or f"data-seed{seed}"
        ddir = bundle / f"data-seed{seed}"
        ddir.mkdir(parents=True)
        pool = expert_data(*make_pairs(env, seed, "bc"))
        stream = expert_data(*make_pairs(env, seed, "bc_iid"))
        pool.save(ddir / rp.POOL_FILE)
        stream.save(ddir / rp.STREAM_FILE)
        dconfig = {
            "seed": seed,
            "budget": N,
            "pool": rp.POOL_RULE,
            "stream": rp.STREAM_RULE,
        }
        drecord = {
            "schema": PROTO + "/data",
            "protocol": PROTO,
            "status": "complete",
            "env_name": env,
            "seed": seed,
            "config": dconfig,
            "config_sha256": pilot.config_digest(dconfig),
            "expert_sha256": EXPERT,
            "episode_cap": tasks.get_task(env).episode_cap,
            "task": task_meta(env),
            "task_sha256": task_sha(env),
            "source": SOURCE,
            "datasets": {
                name: {
                    "file": file_name,
                    "file_sha256": sha(ddir / file_name),
                    "pairs_sha256": data.pairs_sha256(),
                    "n": N,
                }
                for name, file_name, data in (
                    ("pool", rp.POOL_FILE, pool),
                    ("stream", rp.STREAM_FILE, stream),
                )
            },
        }
        write_json(ddir / "result.json", drecord)
        data_sha = sha(ddir / "result.json")
        data_jobs.append(
            {
                "id": data_id,
                "expected_fields": {
                    k: drecord[k]
                    for k in ("status", "protocol", "schema", "seed", "config_sha256")
                },
                "expected_result": f"/remote/data-seed{seed}/result.json",
            },
        )
        cells = {
            "data": {
                "job_id": data_id,
                "manifest": "data_manifest",
                "result": {"path": f"data-seed{seed}/result.json"},
                "pool": {"path": f"data-seed{seed}/{rp.POOL_FILE}"},
                "stream": {"path": f"data-seed{seed}/{rp.STREAM_FILE}"},
            },
        }
        for method in cov.METHODS:
            folder = f"run-{method}-{rid}-seed{seed}"
            job = f"main2-{env}-seed{seed}-{rid}-{method}" if scheduled_ids else folder
            rdir = bundle / folder
            demos_rel = f"scratch/{method}_{env}_seed{seed}/demos"
            if method == "bc":
                trained = pool.pairs_sha256(N)
                demos_rel = None
            elif method == "bc_iid":
                trained = stream.pairs_sha256(N)
                if bc_iid_scratch:
                    write_scratch(rdir / demos_rel, stream.obs, stream.acts)
            else:
                obs, acts = make_pairs(env, seed, "ftl")
                trained = rp.pairs_sha256(obs, acts)
                write_scratch(rdir / demos_rel, obs, acts, terminal=ftl_terminal)
            record = run_record(
                env,
                seed,
                method,
                data_sha,
                pool,
                stream,
                trained,
                demos_rel,
            )
            write_json(rdir / "result.json", record)
            configs[job] = record["config"]
            run_jobs.append(
                {
                    "id": job,
                    "expected_fields": {
                        k: record[k]
                        for k in (
                            "status",
                            "env_name",
                            "seed",
                            "method",
                            "restriction_id",
                            "config_sha256",
                            "expert_sha256",
                            "task_sha256",
                        )
                    },
                    "expected_result": f"/remote/{folder}/result.json",
                },
            )
            cell = {
                "job_id": job,
                "manifest": "run_manifest",
                "configs": "run_configs",
                "result": {"path": f"{folder}/result.json"},
            }
            if method == "ftl" or (method == "bc_iid" and bc_iid_scratch):
                cell["scratch_demos"] = {"path": f"{folder}/{demos_rel}"}
            cells[method] = cell
        seeds[str(seed)] = cells
    write_json(bundle / "control/run-manifest.json", {"jobs": run_jobs})
    write_json(bundle / "control/run-configs.json", configs)
    write_json(bundle / "control/data-manifest.json", {"jobs": data_jobs})
    manifest = {
        "schema": cov.MANIFEST_SCHEMA,
        "bundle_root": "bundle",
        "labels": N,
        "seeds": list(cov.SEEDS),
        "methods": list(cov.METHODS),
        "environments": {
            env: {
                "restriction_id": rid,
                "protocol": PROTO,
                "expert_sha256": EXPERT,
                "task_sha256": task_sha(env),
                "run_source_combined_sha256": SOURCE["combined_sha256"],
                "control": {
                    "run_manifest": {"path": "control/run-manifest.json"},
                    "run_configs": {"path": "control/run-configs.json"},
                    "data_manifest": {"path": "control/data-manifest.json"},
                },
                "expert_provenance": None,
                "seeds": seeds,
            },
        },
    }
    write_json(root / "manifest.json", manifest)
    repin(root)
    return root


def _pins(manifest):
    for env in manifest["environments"].values():
        yield from env["control"].values()
        for cells in env["seeds"].values():
            for cell in cells.values():
                for key, value in cell.items():
                    if isinstance(value, dict) and "path" in value:
                        yield value
    for entry in manifest.get("provenance_indexes", []):
        yield entry


def repin(root):
    """Recompute every manifest pin from the current bundle bytes."""
    root = pathlib.Path(root)
    manifest = read_json(root / "manifest.json")
    bundle = root / manifest["bundle_root"]
    for pin in _pins(manifest):
        path = bundle / pin["path"]
        if path.is_dir():
            pin["tree_sha256"] = cov.tree_sha256(cov.tree_entries(path))
        elif path.is_file():
            pin["sha256"] = sha(path)
        elif "tree_sha256" not in pin and "sha256" not in pin:
            pin["sha256"] = "0" * 64
    write_json(root / "manifest.json", manifest)


@pytest.fixture(scope="module")
def cart_base(tmp_path_factory, small_budget):
    return build_bundle(tmp_path_factory.mktemp("cart"), CART)


@pytest.fixture(scope="module")
def lake_base(tmp_path_factory, small_budget):
    return build_bundle(tmp_path_factory.mktemp("lake"), LAKE)


@pytest.fixture
def cart(cart_base, tmp_path):
    shutil.copytree(cart_base, tmp_path / "case")
    return tmp_path / "case"


@pytest.fixture
def lake(lake_base, tmp_path):
    shutil.copytree(lake_base, tmp_path / "case")
    return tmp_path / "case"


def run(root, **kwargs):
    out = pathlib.Path(root).parent / "out"
    code = cov.analyze(pathlib.Path(root) / "manifest.json", out, **kwargs)
    inventory = None
    if (out / "coverage-inventory.json").exists():
        inventory = read_json(out / "coverage-inventory.json")
    return code, out, inventory


def cell(inventory, seed, name, env=CART):
    return inventory["environments"][env]["seeds"][str(seed)][name]


def edit_manifest(root, fn):
    manifest = read_json(root / "manifest.json")
    fn(manifest)
    write_json(root / "manifest.json", manifest)


def edit_json(path, fn):
    value = read_json(path)
    fn(value)
    write_json(path, value)


# ---------------------------------------------------------------------------
# Complete inputs
# ---------------------------------------------------------------------------


def test_complete_continuous_renders_every_seed_and_method(cart):
    code, out, inventory = run(cart)
    assert code == cov.EXIT_OK
    env = inventory["environments"][CART]
    assert env["status"] == "complete"
    assert set(env["seeds"]) == {str(s) for s in cov.SEEDS}
    assert env["counts"]["complete"] == 2 + len(cov.SEEDS) * 4
    summary = read_json(out / f"coverage-{CART}-summary.json")
    assert set(summary["per_seed"]) == {str(s) for s in cov.SEEDS}
    for seed in cov.SEEDS:
        stats = summary["per_seed"][str(seed)]
        assert all(stats[m]["n"] == N for m in cov.METHODS)
        assert stats["bc"]["arrival"] is None
        assert stats["ftl"]["arrival"]["first"] == 1
        assert stats["ftl"]["arrival"]["last"] == N
        # The trailing observation of every scratch demo is excluded.
        assert stats["ftl"]["excluded_final_observations"] == N
    arrays = np.load(out / f"coverage-{CART}-samples.npz")
    assert np.all(np.isfinite(arrays["obs"])) and np.all(
        np.isfinite(arrays["masked_obs"])
    )
    assert not np.any(arrays["obs"] == SENTINEL)
    for i, m in enumerate(cov.METHODS):
        for seed in cov.SEEDS:
            rows = (arrays["method_index"] == i) & (arrays["seed"] == seed)
            assert rows.sum() == N  # no dropped seed or method
    files = read_json(out / "coverage-artifacts.json")["files"]
    for name, digest in files.items():
        assert sha(out / name) == digest
    assert (out / f"coverage-{CART}.png").stat().st_size > 0
    assert summary["inputs"]["seeds"]["3100"]["ftl"]["scratch_demos"]["tree_sha256"]


def test_ftl_reads_retained_scratch_with_one_based_arrival(cart):
    code, out, _ = run(cart)
    assert code == cov.EXIT_OK
    arrays = np.load(out / f"coverage-{CART}-samples.npz")
    ftl = (arrays["method_index"] == cov.METHODS.index("ftl")) & (
        arrays["seed"] == 3101
    )
    obs, acts = make_pairs(CART, 3101, "ftl")
    assert np.array_equal(arrays["obs"][ftl], obs)
    assert np.array_equal(arrays["expert_action"][ftl], acts)
    assert np.array_equal(arrays["arrival"][ftl], np.arange(1, N + 1))


def test_terminal_scratch_observation_is_never_counted(tmp_path):
    root = build_bundle(tmp_path / "t", CART, bc_iid_scratch=False, ftl_terminal=True)
    code, out, inventory = run(root)
    assert code == cov.EXIT_OK
    arrays = np.load(out / f"coverage-{CART}-samples.npz")
    assert not np.any(arrays["obs"] == SENTINEL)
    reasons = cell(inventory, 3100, "bc_iid")["reasons"]
    assert any("scratch not pinned" in r for r in reasons)


def test_offline_bc_is_the_full_fixed_pool_reference(cart):
    code, out, _ = run(cart)
    assert code == cov.EXIT_OK
    arrays = np.load(out / f"coverage-{CART}-samples.npz")
    bc = (arrays["method_index"] == 0) & (arrays["seed"] == 3102)
    pool = rp.ExpertData.load(cart / f"bundle/data-seed3102/{rp.POOL_FILE}")
    assert np.array_equal(arrays["obs"][bc], pool.obs)
    assert np.all(arrays["arrival"][bc] == -1)


def test_bc_trained_on_a_prefix_is_ineligible(cart):
    job = cart / f"bundle/run-bc-{CART_MASK}-seed3100/result.json"
    pool = rp.ExpertData.load(cart / f"bundle/data-seed3100/{rp.POOL_FILE}")

    def prefix(record):
        record["trained_pairs_sha256"] = pool.pairs_sha256(N - 1)
        record["records"][0]["trained_pairs_sha256"] = pool.pairs_sha256(N - 1)

    edit_json(job, prefix)
    repin(cart)
    code, out, inventory = run(cart)
    assert code == cov.EXIT_INVENTORY
    assert cell(inventory, 3100, "bc")["status"] == "ineligible"
    assert not (out / f"coverage-{CART}.png").exists()


def test_bc_iid_scratch_cross_check_against_stream(cart):
    code, _, inventory = run(cart)
    assert code == cov.EXIT_OK
    files = cell(inventory, 3103, "bc_iid")["files"]
    assert files["scratch_demos"]["cross_check"] == "equal to stream rows 1..1000"


def test_bc_iid_scratch_that_differs_from_stream_is_refused(cart):
    demos = cart / f"bundle/run-bc_iid-{CART_MASK}-seed3103/scratch"
    demos = demos / f"bc_iid_{CART}_seed3103/demos"
    shutil.rmtree(demos)
    stream = rp.ExpertData.load(cart / f"bundle/data-seed3103/{rp.STREAM_FILE}")
    obs = stream.obs.copy()
    obs[4, 1] += 0.25
    write_scratch(demos, obs, stream.acts)
    repin(cart)
    code, _, inventory = run(cart)
    assert code == cov.EXIT_INVENTORY
    c = cell(inventory, 3103, "bc_iid")
    assert c["status"] == "ineligible"
    assert any("differ from the pinned stream" in r for r in c["reasons"])


# ---------------------------------------------------------------------------
# Missing, partial, forged
# ---------------------------------------------------------------------------


def test_missing_seed_is_visible_and_blocks_the_figure(cart):
    edit_manifest(cart, lambda m: m["environments"][CART]["seeds"].pop("3102"))
    code, out, inventory = run(cart)
    assert code == cov.EXIT_INVENTORY
    for name in ("data",) + cov.METHODS:
        assert cell(inventory, 3102, name)["status"] == "missing"
    assert inventory["environments"][CART]["status"] == "inventory_only"
    assert not (out / f"coverage-{CART}.png").exists()
    assert not (out / f"coverage-{CART}-summary.json").exists()
    assert "3102" in (out / "coverage-inventory.md").read_text()


def test_absent_ftl_scratch_is_partial(cart):
    job = f"run-ftl-{CART_MASK}-seed3104"
    shutil.rmtree(cart / "bundle" / job / "scratch")
    code, _, inventory = run(cart)
    assert code == cov.EXIT_INVENTORY
    c = cell(inventory, 3104, "ftl")
    assert c["status"] == "partial"
    assert c["files"]["result"]["sha256"]


def test_running_record_is_partial(cart):
    path = cart / f"bundle/run-ftl-{CART_MASK}-seed3100/result.json"
    edit_json(path, lambda r: r.update(status="running"))
    repin(cart)
    code, _, inventory = run(cart)
    assert code == cov.EXIT_INVENTORY
    assert cell(inventory, 3100, "ftl")["status"] == "partial"


def test_unpinned_byte_change_is_ineligible(cart):
    path = cart / f"bundle/data-seed3101/{rp.POOL_FILE}"
    data = rp.ExpertData.load(path)
    data.obs[0, 0] += 1.0
    path.unlink()
    data.save(path)
    code, _, inventory = run(cart)
    assert code == cov.EXIT_INVENTORY
    assert cell(inventory, 3101, "data")["status"] == "ineligible"
    for m in cov.METHODS:
        assert cell(inventory, 3101, m)["status"] == "ineligible"


def test_repinned_forged_pool_breaks_data_lineage(cart):
    path = cart / f"bundle/data-seed3101/{rp.POOL_FILE}"
    data = rp.ExpertData.load(path)
    data.obs[0, 0] += 1.0
    path.unlink()
    data.save(path)
    repin(cart)
    code, _, inventory = run(cart)
    assert code == cov.EXIT_INVENTORY
    c = cell(inventory, 3101, "data")
    assert c["status"] == "ineligible"
    assert any("data pool" in r for r in c["reasons"])


def test_forged_ftl_scratch_breaks_trained_pairs(cart):
    demos = cart / f"bundle/run-ftl-{CART_MASK}-seed3102/scratch"
    demos = demos / f"ftl_{CART}_seed3102/demos"
    shutil.rmtree(demos)
    obs, acts = make_pairs(CART, 3102, "ftl")
    acts = 1 - acts
    write_scratch(demos, obs, acts)
    repin(cart)
    code, _, inventory = run(cart)
    assert code == cov.EXIT_INVENTORY
    c = cell(inventory, 3102, "ftl")
    assert c["status"] == "ineligible"
    assert any("trained_pairs_sha256" in r for r in c["reasons"])


# ---------------------------------------------------------------------------
# Scheduling IDs versus manifest-declared result targets
# ---------------------------------------------------------------------------

DATA_REL = "data-seed3100/result.json"
FTL_REL = f"run-ftl-{CART_MASK}-seed3100/result.json"


def edit_job(root, control, job_id, fn):
    """Change one job of a campaign manifest copy and repin the bundle."""

    def apply(manifest):
        (job,) = [j for j in manifest["jobs"] if j["id"] == job_id]
        fn(job)

    edit_json(pathlib.Path(root) / "bundle/control" / control, apply)
    repin(root)


def test_scheduling_ids_unlike_result_folders_verify_and_render(tmp_path):
    root = build_bundle(tmp_path / "s", CART, scheduled_ids=True)
    manifest = read_json(root / "manifest.json")
    for cells in manifest["environments"][CART]["seeds"].values():
        for entry in cells.values():
            assert entry["job_id"] != entry["result"]["path"].split("/")[0]
    code, out, inventory = run(root)
    assert code == cov.EXIT_OK, inventory
    env = inventory["environments"][CART]
    assert env["status"] == "complete"
    for seed in cov.SEEDS:
        for name in ("data",) + cov.METHODS:
            assert cell(inventory, seed, name)["status"] == "complete"
    assert cell(inventory, 3100, "ftl")["job_id"] == (
        f"main2-{CART}-seed3100-{CART_MASK}-ftl"
    )
    assert (out / f"coverage-{CART}.png").stat().st_size > 0


@pytest.fixture(scope="module")
def scheduled_base(tmp_path_factory, small_budget):
    return build_bundle(tmp_path_factory.mktemp("sched"), CART, scheduled_ids=True)


@pytest.fixture(params=["folder_ids", "scheduled_ids"])
def either(request, cart_base, scheduled_base, tmp_path):
    base = cart_base if request.param == "folder_ids" else scheduled_base
    shutil.copytree(base, tmp_path / "case")
    return tmp_path / "case"


def _job_id(root, seed, name):
    entry = read_json(root / "manifest.json")["environments"][CART]["seeds"]
    return entry[str(seed)][name]["job_id"]


def _declare(root, name, value):
    control = "data-manifest.json" if name == "data" else "run-manifest.json"
    if value is _POP:
        fn = lambda job: job.pop("expected_result")  # noqa: E731
    else:
        fn = lambda job: job.update(expected_result=value)  # noqa: E731
    edit_job(root, control, _job_id(root, 3100, name), fn)


_POP = object()
_WRONG_TARGETS = {
    # Another seed's real directory: same record fields, wrong target.
    "other_directory": {
        "data": "/remote/data-seed3101/result.json",
        "ftl": f"/remote/run-ftl-{CART_MASK}-seed3101/result.json",
    },
    # A longer folder name ending in the pinned one: the path boundary matters.
    "prefix_collision": {
        "data": f"/remote/x{DATA_REL}",
        "ftl": f"/remote/stage-{FTL_REL}",
    },
    "other_file": {
        "data": "/remote/data-seed3100/result.json.bak",
        "ftl": f"/remote/run-ftl-{CART_MASK}-seed3100/other.json",
    },
    "deeper_relative": {
        "data": "result.json",
        "ftl": "result.json",
    },
    "missing": {"data": _POP, "ftl": _POP},
    "empty": {"data": "", "ftl": ""},
    "not_a_string": {"data": 7, "ftl": ["/remote", FTL_REL]},
    "dot_segment": {
        "data": "/remote/data-seed3100/./result.json",
        "ftl": f"/remote/run-ftl-{CART_MASK}-seed3100//result.json",
    },
    "parent_segment": {
        "data": f"/remote/data-seed3101/../{DATA_REL}",
        "ftl": f"/remote/x/../{FTL_REL}",
    },
}


@pytest.mark.parametrize("name", ["data", "ftl"])
@pytest.mark.parametrize("kind", sorted(_WRONG_TARGETS))
def test_declared_result_target_must_be_the_pinned_result(either, name, kind):
    _declare(either, name, _WRONG_TARGETS[kind][name])
    code, out, inventory = run(either)
    assert code == cov.EXIT_INVENTORY
    c = cell(inventory, 3100, name)
    assert c["status"] == "ineligible"
    assert any("expected_result" in r for r in c["reasons"]), c["reasons"]
    if name == "data":
        runs = [cell(inventory, 3100, m)["status"] for m in cov.METHODS]
        assert runs == ["ineligible"] * len(cov.METHODS)
    assert not (out / f"coverage-{CART}.png").exists()


@pytest.mark.parametrize("name", ["data", "ftl"])
def test_relative_or_transported_declared_target_is_accepted(either, name):
    rel = DATA_REL if name == "data" else FTL_REL
    _declare(either, name, rel)
    code, out, inventory = run(either)
    assert code == cov.EXIT_OK, inventory
    shutil.rmtree(out)
    _declare(either, name, f"/somewhere/else/entirely/{rel}")
    code, _, inventory = run(either)
    assert code == cov.EXIT_OK, inventory


def test_job_binding_is_by_pinned_path_not_scheduling_id():
    doc = {
        "jobs": [
            {
                "id": "main-a",
                "expected_fields": {},
                "expected_result": "/srv/x/run-a/result.json",
            },
        ],
    }
    assert cov._job(doc, "main-a", "run-a/result.json", "run")["id"] == "main-a"
    assert cov._job(doc, "main-a", "x/run-a/result.json", "run")["id"] == "main-a"
    for rel in ("a/result.json", "run-b/result.json", "srv/x/run-a/result.json/"):
        with pytest.raises(cov.Ineligible, match="expected_result"):
            cov._job(doc, "main-a", rel, "run")
    with pytest.raises(cov.Ineligible, match="exactly once"):
        cov._job(doc, "run-a", "run-a/result.json", "run")


def test_run_config_differs_from_campaign_configs(cart):
    job = f"run-bc_iid-{CART_MASK}-seed3100"
    path = cart / "bundle/control/run-configs.json"
    edit_json(path, lambda c: c[job].update(eval_episodes=4))
    repin(cart)
    code, _, inventory = run(cart)
    assert code == cov.EXIT_INVENTORY
    c = cell(inventory, 3100, "bc_iid")
    assert c["status"] == "ineligible" and "run_configs" in c["reasons"][-1]


def test_unsealed_config_edit_is_ineligible(cart):
    path = cart / f"bundle/run-ftl-{CART_MASK}-seed3101/result.json"
    edit_json(path, lambda r: r["config"]["experiment"].update(n_rounds=N + 1))
    repin(cart)
    code, _, inventory = run(cart)
    assert code == cov.EXIT_INVENTORY
    reasons = cell(inventory, 3101, "ftl")["reasons"]
    assert any("config_sha256" in r for r in reasons)


def test_run_bound_to_another_data_job_is_ineligible(cart):
    other = sha(cart / "bundle/data-seed3104/result.json")
    path = cart / f"bundle/run-ftl-{CART_MASK}-seed3100/result.json"
    edit_json(path, lambda r: r["data"].update(data_result_sha256=other))
    repin(cart)
    code, _, inventory = run(cart)
    assert code == cov.EXIT_INVENTORY
    assert cell(inventory, 3100, "ftl")["status"] == "ineligible"


def test_wrong_environment_source_pin_is_ineligible(cart):
    edit_manifest(
        cart,
        lambda m: m["environments"][CART].update(run_source_combined_sha256="2" * 64),
    )
    code, _, inventory = run(cart)
    assert code == cov.EXIT_INVENTORY
    assert all(
        cell(inventory, s, m)["status"] == "ineligible"
        for s in cov.SEEDS
        for m in cov.METHODS
    )


def test_non_finite_metric_is_ineligible(cart):
    path = cart / f"bundle/run-bc-{CART_MASK}-seed3103/result.json"
    text = path.read_text().replace(
        '"normalized_return": 0.5', '"normalized_return": NaN'
    )
    path.write_text(text)
    repin(cart)
    code, _, inventory = run(cart)
    assert code == cov.EXIT_INVENTORY
    assert cell(inventory, 3103, "bc")["status"] == "ineligible"


def test_provenance_index_disagreement_is_ineligible(cart):
    rel = f"data-seed3100/{rp.POOL_FILE}"
    write_json(cart / "bundle/index.json", [{"path": rel, "sha256": "3" * 64}])

    def add(m):
        m["provenance_indexes"] = [
            {"path": "index.json", "sha256": sha(cart / "bundle/index.json")},
        ]

    edit_manifest(cart, add)
    code, _, inventory = run(cart)
    assert code == cov.EXIT_INVENTORY
    assert any(
        "provenance index" in r for r in cell(inventory, 3100, "data")["reasons"]
    )


def test_strict_index_corroboration_requires_every_file(cart):
    rel = f"data-seed3100/{rp.POOL_FILE}"
    digest = sha(cart / "bundle" / rel)
    write_json(cart / "bundle/index.json", {"files": [{"path": rel, "sha256": digest}]})

    def add(m):
        m["require_index_corroboration"] = True
        m["provenance_indexes"] = [
            {"path": "index.json", "sha256": sha(cart / "bundle/index.json")},
        ]

    edit_manifest(cart, add)
    code, _, inventory = run(cart)
    assert code == cov.EXIT_INVENTORY
    data = cell(inventory, 3100, "data")
    assert data["status"] == "ineligible"
    assert "not in any provenance index" in data["reasons"][0]


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(schema="other/1"),
        lambda m: m.update(labels=N + 1),
        lambda m: m.update(seeds=[3100, 3101, 3102, 3103]),
        lambda m: m.update(methods=["bc", "ftl"]),
        lambda m: m["environments"][CART]["control"]["run_manifest"].update(
            path="/abs/run-manifest.json",
        ),
        lambda m: m["environments"][CART]["control"]["run_manifest"].update(
            path="../outside.json",
        ),
        lambda m: m["environments"][CART].update(restriction_id="identity"),
        lambda m: m["environments"][CART]["seeds"]["3100"]["ftl"].pop("scratch_demos"),
    ],
)
def test_malformed_manifest_refuses_without_output(cart, change):
    edit_manifest(cart, change)
    code, out, _ = run(cart)
    assert code == cov.EXIT_REFUSED
    assert not out.exists()


def test_existing_output_directory_is_refused(cart):
    out = cart.parent / "out"
    out.mkdir()
    (out / "keep").write_text("x")
    assert cov.analyze(cart / "manifest.json", out) == cov.EXIT_REFUSED
    assert [p.name for p in out.iterdir()] == ["keep"]


def test_inputs_are_not_mutated(cart):
    before = {p: sha(p) for p in sorted(cart.rglob("*")) if p.is_file()}
    code, _, _ = run(cart)
    assert code == cov.EXIT_OK
    after = {p: sha(p) for p in sorted(cart.rglob("*")) if p.is_file()}
    assert before == after


def test_expected_environments_outside_manifest_are_listed(cart):
    code, _, inventory = run(cart, envs=[CART, LAKE])
    assert code == cov.EXIT_INVENTORY
    assert inventory["environments"][LAKE]["status"] == "inventory_only"
    assert LAKE in inventory["expected_environments_not_in_manifest"]


# ---------------------------------------------------------------------------
# Mask semantics, conflicts, projection and axes
# ---------------------------------------------------------------------------


def test_zeroing_mask_is_rebuilt_exactly_from_its_description():
    task = tasks.TASKS[CART]
    mask, info = cov.mask_from_description(description(CART), CART_MASK, task)
    obs = np.array([[1.5, -0.25, 0.1, 2.0], [-0.0, 0.5, -0.2, -3.0]], np.float32)
    original = obs.copy()
    masked = mask.apply(obs)
    assert np.array_equal(masked[:, [0, 3]], np.zeros((2, 2)))
    assert np.array_equal(masked[:, [1, 2]], obs[:, [1, 2]])
    assert np.array_equal(obs, original)  # input untouched
    assert info["zeroed_features"] == ["x", "theta_dot"]


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.pop("zeroed_indices"),
        lambda d: d.update(zeroed_indices=[0]),  # differs from the registry
        lambda d: d.update(zeroed_indices=[0, 9]),
        lambda d: d.update(obs_shape=[5]),
        lambda d: d.update(restriction_id="cart_position_zero"),
        lambda d: d.update(kind="mystery"),
    ],
)
def test_insufficient_or_inconsistent_zeroing_description_fails(change):
    desc = description(CART)
    change(desc)
    with pytest.raises(cov.Ineligible):
        cov.mask_from_description(desc, CART_MASK, tasks.TASKS[CART])


def test_alias_mask_maps_one_hot_states_to_exact_representatives():
    table, _ = lake_table()
    mask, info = cov.mask_from_description(
        description(LAKE), LAKE_MASK, tasks.TASKS[LAKE]
    )
    states = np.arange(16)
    masked = mask.apply(np.eye(16, dtype=np.float32)[states])
    assert np.array_equal(masked, np.eye(16, dtype=np.float32)[table])
    assert np.array_equal(mask.decode(masked), np.array(table))
    assert info["table_sha256"] == tasks.FROZEN_TABLE_SHA256[LAKE_MASK]
    with pytest.raises(ValueError):
        mask.apply(np.full((1, 16), 0.5, np.float32))


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["representative"].__setitem__(3, 3),
        lambda d: d.update(representative=d["representative"][:-1]),
        lambda d: d.update(table_sha256="4" * 64),
        lambda d: d.update(blocks=[4, 4]),
        lambda d: d.pop("representative"),
    ],
)
def test_forged_or_partial_alias_table_fails(change):
    desc = description(LAKE)
    change(desc)
    with pytest.raises(cov.Ineligible):
        cov.mask_from_description(desc, LAKE_MASK, tasks.TASKS[LAKE])


def test_exact_conflicts_count_identical_rows_only():
    orig = np.array(
        [
            [0.5, 0.1, 0.02, 1.0],
            [-0.5, 0.1, 0.02, -1.0],
            [0.5, 0.1, 0.0200001, 1.0],
            [0.7, 0.1, 0.02, 0.3],
            [-0.0, 0.3, 0.1, 0.0],
            [0.0, 0.3, 0.1, 0.0],
        ],
        np.float32,
    )
    acts = np.array([1, 0, 0, 1, 1, 1])
    masked = restriction.RESTRICTIONS[CART_MASK].apply(orig)
    stats = cov.conflict_stats(orig, masked, acts)
    assert stats["distinct_original"] == 5  # -0.0 equals 0.0
    assert stats["distinct_masked"] == 3
    assert stats["masked_groups_merging_originals"] == 1
    assert stats["masked_groups_with_label_conflict"] == 1
    assert stats["samples_in_conflicting_groups"] == 3
    assert stats["majority_label_misfit_samples"] == 1
    assert stats["original_groups_with_label_conflict"] == 0


def test_pca_is_deterministic_and_sign_fixed():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4)) * [3.0, 1.0, 0.1, 0.0]
    a, b = cov.pca(x), cov.pca(x.copy())
    assert a == b
    flipped = cov.pca(-x)
    assert np.allclose(flipped["components"], a["components"])
    assert sum(a["explained_variance_ratio"]) <= 1.0 + 1e-12
    assert a["scale"][3] == 1.0  # zero-variance column left unscaled


def test_shared_axes_and_fixed_display_counts(cart):
    code, out, _ = run(cart)
    assert code == cov.EXIT_OK
    layout = read_json(out / f"coverage-{CART}-summary.json")["display"]
    pairs = [r for r in layout if r["kind"] == "pair"]
    assert [r["features"] for r in pairs[::2]] == [
        ["x", "x_dot"],
        ["theta", "theta_dot"],
    ]
    for original, masked in zip(pairs[::2], pairs[1::2]):
        assert original["representation"] == "original"
        assert masked["representation"] == "masked"
        assert original["xlim"] == masked["xlim"] and original["ylim"] == masked["ylim"]
    for row in layout:
        assert row["display_counts"] == {m: N * len(cov.SEEDS) for m in cov.METHODS}
        assert all(np.isfinite(row["xlim"])) and all(np.isfinite(row["ylim"]))
    summary = read_json(out / f"coverage-{CART}-summary.json")
    ftl = summary["per_seed"]["3100"]["ftl"]
    assert ftl["masked_groups_with_label_conflict"] >= 1
    assert summary["projection"]["original"]["explained_variance_ratio"][0] > 0


def test_expert_action_coloring_is_explicit(cart):
    code, out, _ = run(cart, color_by="expert_action")
    assert code == cov.EXIT_OK
    assert (
        read_json(out / f"coverage-{CART}-summary.json")["color_by"] == "expert_action"
    )


def test_discrete_semantic_map_and_alias_table(lake):
    code, out, inventory = run(lake)
    assert code == cov.EXIT_OK, inventory
    summary = read_json(out / f"coverage-{LAKE}-summary.json")
    assert summary["restriction"]["kind"] == "alias"
    assert summary["display"]["shape"] == [4, 4]
    table = {row["masked_representative"]: row for row in summary["alias_table"]}
    assert table[0]["members_in_table"] == [0, 1]
    assert table[0]["label_conflict"]["ftl/seed3100"] is True
    assert summary["per_seed"]["3100"]["ftl"]["masked_groups_with_label_conflict"] >= 1
    assert summary["taxi_expert_provenance_caption"] is None


# ---------------------------------------------------------------------------
# Scratch reader, Taxi provenance, no environment
# ---------------------------------------------------------------------------


def test_reader_takes_exactly_one_thousand_labels(tmp_path):
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(1000, 4)).astype(np.float32)
    acts = rng.integers(0, 2, size=1000)
    write_scratch(tmp_path / "demos", obs, acts, terminal=True)
    s_obs, s_acts, arrival, excluded = cov.read_scratch(tmp_path / "demos", 1000)
    assert np.array_equal(s_obs, obs) and np.array_equal(s_acts, acts)
    assert np.array_equal(arrival, np.arange(1, 1001)) and excluded == 1000
    assert rp.pairs_sha256(s_obs, s_acts) == rp.pairs_sha256(
        *rp.trained_pairs(tmp_path / "demos"),
    )
    with pytest.raises(cov.Ineligible):
        cov.read_scratch(tmp_path / "demos", 999)


def test_reader_refuses_gaps_extra_labels_and_plain_npz(tmp_path):
    obs = np.zeros((3, 4), np.float32)
    acts = np.zeros(3, np.int64)
    write_scratch(tmp_path / "gap", obs, acts)
    shutil.rmtree(tmp_path / "gap/round-001")
    with pytest.raises(cov.Ineligible, match="rounds"):
        cov.read_scratch(tmp_path / "gap", 3)
    write_scratch(tmp_path / "two", obs, acts)
    serialize.save(
        tmp_path / "two/round-002/uniform-extra.npz",
        [
            types.Trajectory(
                obs=np.zeros((2, 4), np.float32),
                acts=np.array([1]),
                infos=None,
                terminal=False,
            )
        ],
    )
    with pytest.raises(cov.Ineligible, match="2 labeled states"):
        cov.read_scratch(tmp_path / "two", 3)
    write_scratch(tmp_path / "pickle", obs, acts)
    shutil.rmtree(tmp_path / "pickle/round-000/uniform-demo_0000.npz")
    np.savez(tmp_path / "pickle/round-000/uniform-demo_0000.npz", obs=obs)
    with pytest.raises(cov.Ineligible, match="dataset dir"):
        cov.read_scratch(tmp_path / "pickle", 3)


def _acceptance(tmp_path, method, checkpoint):
    write_json(
        tmp_path / "acceptance.json",
        {
            "schema": cov.ACCEPTANCE_SCHEMA,
            "env_name": cov.TAXI_ENV,
            "status": "complete",
            "release_allowed": True,
            "training_method": method,
            "files": {"checkpoint": {"path": "model.zip", "sha256": checkpoint}},
        },
    )
    return {
        "expert_sha256": EXPERT,
        "expert_provenance": {
            "acceptance": {
                "path": "acceptance.json",
                "sha256": sha(tmp_path / "acceptance.json"),
            },
        },
    }


def test_taxi_caption_requires_verified_planner_distilled_expert(tmp_path):
    reader = cov.Reader(tmp_path, {}, False)
    cell_, caption = cov.verify_expert_provenance(
        reader, cov.TAXI_ENV, {"expert_sha256": EXPERT}
    )
    assert cell_.status == "ineligible" and caption is None
    env = _acceptance(tmp_path, cov.TAXI_PLANNER_PREFIX + " into a PPO actor", EXPERT)
    cell_, caption = cov.verify_expert_provenance(reader, cov.TAXI_ENV, env)
    assert cell_.status == "complete" and caption == cov.TAXI_CAPTION
    env = _acceptance(tmp_path, cov.TAXI_PLANNER_PREFIX, "5" * 64)
    cell_, caption = cov.verify_expert_provenance(reader, cov.TAXI_ENV, env)
    assert cell_.status == "ineligible" and caption is None
    env = _acceptance(tmp_path, "PPO trained from scratch", EXPERT)
    cell_, caption = cov.verify_expert_provenance(reader, cov.TAXI_ENV, env)
    assert cell_.status == "complete" and caption is None


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
    monkeypatch.setattr(tasks, "_raw_env", boom)
    monkeypatch.setattr(tasks, "alias_table", boom)
    code, _, _ = run(cart)
    assert code == cov.EXIT_OK


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return type(node).__name__


def test_source_calls_no_environment_or_loader():
    tree = ast.parse(ANALYZER.read_text())
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called.add(_dotted(node.func))
    forbidden = {
        "gym.make",
        "gymnasium.make",
        "env_utils.make_env",
        "th.load",
        "torch.load",
        "PPO.load",
        "tasks.alias_table",
        "restriction.get_restriction",
        "restriction.describe_restriction",
        "np.load",
    }
    assert not called & forbidden - {"np.load"}
    loads = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _dotted(n.func) == "np.load"
    ]
    assert loads and all(
        any(
            k.arg == "allow_pickle"
            and isinstance(k.value, ast.Constant)
            and k.value.value is False
            for k in n.keywords
        )
        for n in loads
    )


def test_tree_digest_cli_matches_function(cart, capsys):
    demos = (
        cart / f"bundle/run-ftl-{CART_MASK}-seed3100/scratch/ftl_{CART}_seed3100/demos"
    )
    assert cov.main(["tree-digest", str(demos)]) == cov.EXIT_OK
    assert capsys.readouterr().out.strip() == cov.tree_sha256(cov.tree_entries(demos))


# ---------------------------------------------------------------------------
# Generic tasks: ladder rungs and bound calibration candidates
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mcar_base(tmp_path_factory, small_budget):
    return build_bundle(tmp_path_factory.mktemp("mcar"), MCAR)


@pytest.fixture(scope="module")
def varied_base(tmp_path_factory, small_budget):
    return build_bundle(tmp_path_factory.mktemp("varied"), VARIED)


@pytest.fixture
def mcar(mcar_base, tmp_path):
    shutil.copytree(mcar_base, tmp_path / "case")
    return tmp_path / "case"


@pytest.fixture
def varied(varied_base, tmp_path):
    shutil.copytree(varied_base, tmp_path / "case")
    return tmp_path / "case"


def test_permitted_restrictions_are_the_ladder_and_bound_candidates(monkeypatch):
    mcar_task = tasks.TASKS[MCAR]
    # Random-offset grids (`tasks.RANDOM_GRID_CANDIDATES`) follow the candidates.
    mcar_grids = tasks.RANDOM_GRID_CANDIDATES[MCAR]
    cart_grids = tasks.RANDOM_GRID_CANDIDATES[CART]
    assert cov.permitted_restrictions(mcar_task) == (
        mcar_task.ladder + (TOY_QUANT,) + mcar_grids
    )
    assert cov.permitted_restrictions(tasks.TASKS[CART]) == (
        tasks.TASKS[CART].ladder + cart_grids
    )
    assert cov.permitted_restrictions(tasks.get_task(VARIED)) == (
        tasks.get_task(VARIED).ladder
    )
    for task in list(tasks.TASKS.values()) + list(tasks.VARIANT_TASKS.values()):
        assert "identity" not in cov.permitted_restrictions(task)
    # A listed candidate must also be registered and bound to this env.
    monkeypatch.setattr(
        tasks,
        "CALIBRATION_CANDIDATES",
        {MCAR: (TOY_QUANT, LAKE_MASK, "unregistered_bins", "identity")},
    )
    assert cov.permitted_restrictions(mcar_task) == (
        mcar_task.ladder + (TOY_QUANT,) + mcar_grids
    )
    monkeypatch.setattr(tasks, "CALIBRATION_CANDIDATES", {CART: (TOY_QUANT,)})
    assert TOY_QUANT not in cov.permitted_restrictions(tasks.TASKS[CART])


@pytest.mark.parametrize(
    "rid",
    ["identity", "unregistered_bins", LAKE_MASK, TOY_QUANT, VARIED_MASK, None],
)
def test_manifest_restriction_must_be_permitted_for_its_env(cart, rid):
    edit_manifest(cart, lambda m: m["environments"][CART].update(restriction_id=rid))
    code, out, _ = run(cart)
    assert code == cov.EXIT_REFUSED
    assert not out.exists()


def test_quantized_candidate_renders_with_its_recorded_bins(mcar):
    code, out, inventory = run(mcar)
    assert code == cov.EXIT_OK, inventory
    env = inventory["environments"][MCAR]
    assert env["status"] == "complete"
    assert env["task_scope"] == "original"
    summary = read_json(out / f"coverage-{MCAR}-summary.json")
    info = summary["restriction"]
    assert info["kind"] == "quantized"
    assert info["registry"] == "calibration_candidate"
    assert info["descriptor_sha256"] == toy_quantized().descriptor_sha256
    assert info["binned_features"] == ["position", "velocity"]
    assert [b["step"] for b in info["bins"]] == [0.25, None]
    assert [b["dead_zone"] for b in info["bins"]] == [None, 0.01]
    assert "position" in info["hidden"] and "0.25" in info["hidden"]
    arrays = np.load(out / f"coverage-{MCAR}-samples.npz")
    assert np.array_equal(arrays["masked_obs"], toy_quantized().apply(arrays["obs"]))
    ftl = summary["per_seed"]["3100"]["ftl"]
    assert ftl["masked_groups_with_label_conflict"] >= 1


def test_quantized_mask_is_the_frozen_descriptor_in_numpy_and_torch():
    import torch

    mask, info = cov.mask_from_description(
        description(MCAR), TOY_QUANT, tasks.TASKS[MCAR]
    )
    assert mask == restriction.QUANTIZED_RESTRICTIONS[TOY_QUANT]
    rng = np.random.default_rng(3)
    obs = np.stack(
        [rng.uniform(-1.2, 0.6, 200), rng.uniform(-0.07, 0.07, 200)], 1
    ).astype(np.float32)
    # Bin edges and bounds, where rounding and clipping matter.
    obs[:6] = [
        [0.125, 0.01],
        [-0.125, -0.01],
        [0.6, 0.07],
        [-1.2, -0.07],
        [0.375, 0.0100001],
        [-0.0, -0.0],
    ]
    original = obs.copy()
    masked = mask.apply(obs)
    assert np.array_equal(obs, original)  # input untouched
    with torch.no_grad():
        torch_masked = restriction.ObservationMask(TOY_QUANT)(torch.as_tensor(obs))
    assert np.array_equal(masked, torch_masked.numpy())
    assert np.array_equal(mask.apply(masked), masked)  # idempotent
    with pytest.raises(ValueError):
        mask.apply(np.zeros((2, 3), np.float32))  # recorded obs shape is (2,)
    assert info["descriptor_sha256"] == mask.descriptor_sha256


def _retag(desc):
    """Recompute the descriptor digest so only the registry check can fail."""
    body = {k: v for k, v in desc.items() if k not in ("kind", "descriptor_sha256")}
    desc["descriptor_sha256"] = tasks._digest(body)


@pytest.mark.parametrize("rehash", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["features"][0].update(step=0.5),
        lambda d: d["features"][1].update(dead_zone=0.02),
        lambda d: d["features"][1].update(magnitude=0.07),
        lambda d: d["features"][0].update(low=-2.0),
        lambda d: d["features"][1].update(high=1.0),
        lambda d: d["features"][0].update(rule="sign_dead_zone"),
        lambda d: d["features"][0].update(name="velocity"),
        lambda d: d["features"][0].update(index=True),
        lambda d: d["features"].reverse(),
        lambda d: d["features"].pop(),
        lambda d: d["features"][1].pop("magnitude"),  # never inferred
        lambda d: d["features"][0].update(extra=1),
        lambda d: d.pop("description"),
        lambda d: d.update(description="edited"),
        lambda d: d.update(env_name="Acrobot-v1"),
        lambda d: d.update(obs_shape=[3]),
        lambda d: d.update(extra="field"),
        lambda d: d.update(kind="alias"),
        lambda d: d.pop("kind"),
        lambda d: d.update(restriction_id=LAKE_MASK),
    ],
)
def test_tampered_quantized_description_is_ineligible(change, rehash):
    desc = description(MCAR)
    change(desc)
    if rehash:
        _retag(desc)
    with pytest.raises(cov.Ineligible):
        cov.mask_from_description(desc, TOY_QUANT, tasks.TASKS[MCAR])


def test_quantized_descriptor_hash_and_env_binding_are_checked():
    desc = description(MCAR)
    desc["descriptor_sha256"] = "4" * 64
    with pytest.raises(cov.Ineligible):
        cov.mask_from_description(desc, TOY_QUANT, tasks.TASKS[MCAR])
    # The same frozen descriptor is refused for another environment's task.
    with pytest.raises(cov.Ineligible):
        cov.mask_from_description(description(MCAR), TOY_QUANT, tasks.TASKS[CART])


def test_unregistered_quantized_id_is_ineligible(monkeypatch):
    desc = description(MCAR)
    monkeypatch.setattr(
        restriction,
        "QUANTIZED_RESTRICTIONS",
        {k: v for k, v in restriction.QUANTIZED_RESTRICTIONS.items() if k != TOY_QUANT},
    )
    with pytest.raises(cov.Ineligible):
        cov.mask_from_description(desc, TOY_QUANT, tasks.TASKS[MCAR])


def test_quantized_offline_parse_builds_no_environment(mcar, monkeypatch):
    import gymnasium

    from imitation.experiments.ftrl import env_utils

    def boom(*args, **kwargs):
        raise AssertionError("forbidden call")

    monkeypatch.setattr(gymnasium, "make", boom)
    monkeypatch.setattr(env_utils, "make_env", boom)
    monkeypatch.setattr(tasks, "describe", boom)
    monkeypatch.setattr(tasks, "alias_table", boom)
    monkeypatch.setattr(restriction, "describe_restriction", boom)
    monkeypatch.setattr(restriction, "get_restriction", boom)
    code, _, inventory = run(mcar)
    assert code == cov.EXIT_OK, inventory


# ---------------------------------------------------------------------------
# Variant tasks: explicit and registered only, never promoted to main
# ---------------------------------------------------------------------------


def test_registered_variant_verifies_without_main_promotion(varied):
    code, out, inventory = run(varied)
    assert code == cov.EXIT_OK, inventory
    env = inventory["environments"][VARIED]
    assert env["status"] == "complete"
    assert env["task_scope"] == "variant"
    assert inventory["expected_environments"] == list(tasks.TASKS)
    assert VARIED not in inventory["expected_environments"]
    assert VARIED not in inventory["expected_environments_not_in_manifest"]
    assert inventory["variant_environments"] == [VARIED]
    summary = read_json(out / f"coverage-{VARIED}-summary.json")
    assert summary["task_scope"] == "variant"
    assert summary["restriction"]["kind"] == "alias"
    assert summary["restriction"]["registry"] == "ladder"


def test_variant_manifest_is_refused_unless_registered(varied, monkeypatch):
    monkeypatch.setattr(tasks, "VARIANT_TASKS", {})
    code, out, _ = run(varied)
    assert code == cov.EXIT_REFUSED
    assert not out.exists()


def test_mocked_new_variant_uses_the_generic_path(tmp_path):
    root = build_bundle(tmp_path / "slip", SLIP)
    code, out, inventory = run(root)
    assert code == cov.EXIT_OK, inventory
    assert inventory["environments"][SLIP]["task_scope"] == "variant"
    assert SLIP not in inventory["expected_environments"]
    summary = read_json(out / f"coverage-{SLIP}-summary.json")
    assert summary["restriction_id"] == SLIP_MASK
    # The original FrozenLake rung is not a rung of the variant.
    edit_manifest(
        root, lambda m: m["environments"][SLIP].update(restriction_id=LAKE_MASK)
    )
    code, _, _ = run(root)
    assert code == cov.EXIT_REFUSED


def test_selected_variant_absent_from_manifest_is_missing_not_complete(cart):
    code, out, inventory = run(cart, envs=[CART, VARIED])
    assert code == cov.EXIT_INVENTORY
    assert inventory["environments"][CART]["status"] == "complete"
    absent = inventory["environments"][VARIED]
    assert absent["status"] == "inventory_only"
    assert absent["reasons"] == ["not in the manifest (missing)"]
    assert VARIED not in inventory["expected_environments_not_in_manifest"]
    # A legacy run never gains the variant in its inventory.
    shutil.rmtree(out)
    code, _, inventory = run(cart)
    assert code == cov.EXIT_OK
    assert VARIED not in inventory["environments"]
    assert inventory["variant_environments"] == []


def test_cli_accepts_known_variants_only(cart):
    parser = cov.build_parser()
    args = parser.parse_args(
        ["analyze", "--manifest", "m", "--output-dir", "o", "--env", VARIED]
    )
    assert args.env == [VARIED]
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["analyze", "--manifest", "m", "--output-dir", "o", "--env", "Foo-v0"]
        )
    assert cov.EXPECTED_ENVS == tuple(tasks.TASKS)


# ---------------------------------------------------------------------------
# Recorded task metadata, original and variant, with matching episode caps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env", [CART, LAKE, VARIED])
def test_task_metadata_checks_accept_their_own_task(env):
    record = {"episode_cap": tasks.get_task(env).episode_cap, "task": task_meta(env)}
    record["task_sha256"] = task_sha(env)
    checks = cov.task_checks(record, env)
    assert checks and all(checks.values()), checks


def _variant_as_original(meta):
    meta.pop("reset_distribution")


def _original_as_variant(meta):
    meta["reset_distribution"] = task_meta(VARIED)["reset_distribution"]


@pytest.mark.parametrize(
    "env, change",
    [
        (VARIED, _variant_as_original),
        (LAKE, _original_as_variant),
        (VARIED, lambda m: m["declared"].update(ladder=list(tasks.TASKS[LAKE].ladder))),
        (LAKE, lambda m: m["declared"].update(ladder=[LAKE_MASK])),
        (CART, lambda m: m["declared"].update(blocks=[4])),
        (LAKE, lambda m: m.update(env_name=VARIED)),
        (VARIED, lambda m: m.update(episode_cap=200)),
        (VARIED, lambda m: m["reset_distribution"].update(support=[0])),
    ],
)
def test_task_metadata_of_another_task_fails_even_with_matching_cap(env, change):
    # FrozenLake-v1 and FrozenLakeVariedStart-v0 share the episode cap 100.
    assert tasks.TASKS[LAKE].episode_cap == tasks.get_task(VARIED).episode_cap
    meta = task_meta(env)
    change(meta)
    record = {
        "episode_cap": tasks.get_task(env).episode_cap,
        "task": meta,
        "task_sha256": tasks._digest(meta),  # internally consistent forgery
    }
    checks = cov.task_checks(record, env)
    assert not all(checks.values())


def test_task_metadata_digest_must_match(cart):
    record = {"episode_cap": 500, "task": task_meta(CART), "task_sha256": "7" * 64}
    assert not cov.task_checks(record, CART)["task_metadata_sha256"]
    record = {"episode_cap": 100, "task": task_meta(CART)}
    record["task_sha256"] = task_sha(CART)
    assert not cov.task_checks(record, CART)["episode_cap"]


@pytest.mark.parametrize("env", [CART, VARIED])
@pytest.mark.parametrize("name", ["data", "ftl"])
def test_record_with_wrong_episode_cap_is_ineligible(tmp_path, env, name):
    root = build_bundle(tmp_path / "cap", env)
    rid = description(env)["restriction_id"]
    folder = "data-seed3100" if name == "data" else f"run-ftl-{rid}-seed3100"
    edit_json(
        root / "bundle" / folder / "result.json",
        lambda r: r.update(episode_cap=r["episode_cap"] + 1),
    )
    repin(root)
    code, out, inventory = run(root)
    assert code == cov.EXIT_INVENTORY
    c = cell(inventory, 3100, name, env=env)
    assert c["status"] == "ineligible"
    assert any("episode_cap" in r for r in c["reasons"]), c["reasons"]
    assert not (out / f"coverage-{env}.png").exists()


# ---------------------------------------------------------------------------
# Random state binnings: static records only, no environment
# ---------------------------------------------------------------------------

RANDOM_IDS = tuple(rid for ids in tasks.RANDOM_ALIAS_CANDIDATES.values() for rid in ids)


def with_random(mp):
    """List every random binning as a calibration candidate, on a copy only."""
    mp.setattr(
        tasks,
        "CALIBRATION_CANDIDATES",
        {**tasks.CALIBRATION_CANDIDATES, **tasks.RANDOM_ALIAS_CANDIDATES},
    )


def forbid_environment(mp):
    """Make every environment, reachability or alias-table builder raise."""
    import gymnasium

    from imitation.experiments.ftrl import env_utils

    def boom(*args, **kwargs):
        raise AssertionError("forbidden call")

    mp.setattr(gymnasium, "make", boom)
    mp.setattr(env_utils, "make_env", boom)
    for name in (
        "describe",
        "_raw_env",
        "reachable_nonterminal",
        "_reachable",
        "_tabular_reachable",
        "_blackjack_reachable",
        "_check_encodings",
        "alias_table",
        "_alias",
        "_random_alias",
    ):
        mp.setattr(tasks, name, boom)
    for name in ("describe_restriction", "get_restriction", "_alias_restriction"):
        mp.setattr(restriction, name, boom)


def rehash(d):
    d["partition_sha256"] = tasks._digest(d["partition"])


def retable(d, table):
    d["representative"] = table
    digest = tasks.table_sha256(tuple(d["blocks"]), tuple(table))
    d["table_sha256"] = d["frozen_table_sha256"] = digest


def regroup(d):
    """Rebuild the table from the recorded groups, digests included."""
    table = list(range(int(np.prod(d["blocks"]))))
    for group in d["groups"]:
        for s in group:
            table[s] = group[0]
    retable(d, table)


def test_every_random_binning_is_bound_and_permitted_only_when_listed(
    monkeypatch,
):
    n_declared = sum(len(ks) for ks in tasks.RANDOM_BINNING_COUNTS.values())
    assert len(RANDOM_IDS) == len(set(RANDOM_IDS)) == n_declared
    assert set(RANDOM_IDS) == set(tasks.RANDOM_ALIAS_SPECS)
    assert set(RANDOM_IDS) <= set(tasks.FROZEN_TABLE_SHA256)
    for rid in RANDOM_IDS:
        spec = tasks.RANDOM_ALIAS_SPECS[rid]
        assert rid == tasks.random_binning_id(spec.env_name, spec.n_bins, spec.seed)
    # The narrowed module registry lists none, so none is permitted by default.
    for env in tasks.RANDOM_ALIAS_CANDIDATES:
        task = tasks.get_task(env)
        assert not set(RANDOM_IDS) & set(cov.permitted_restrictions(task))
    with_random(monkeypatch)
    for env, env_ids in tasks.RANDOM_ALIAS_CANDIDATES.items():
        task = tasks.get_task(env)
        permitted = cov.permitted_restrictions(task)
        assert permitted[: len(task.ladder)] == task.ladder
        assert permitted[len(permitted) - len(env_ids) :] == env_ids
        for rid in RANDOM_IDS:
            bound = rid in env_ids
            assert (cov.restriction_kind(rid, task) == "alias") is bound
    for task in (tasks.TASKS[CART], tasks.TASKS[MCAR], tasks.TASKS[LAKE]):
        assert not set(RANDOM_IDS) & set(cov.permitted_restrictions(task))
    # Slippery stays an explicit-only variant; the eight originals are unchanged.
    assert cov.EXPECTED_ENVS == tuple(tasks.TASKS) and len(cov.EXPECTED_ENVS) == 8
    assert SLIP not in cov.EXPECTED_ENVS and SLIP in cov.known_envs()
    assert cov.task_scope(SLIP) == "variant" and cov.task_scope(CLIFF) == "original"


@pytest.mark.parametrize("rid", RANDOM_IDS)
def test_random_binning_record_maps_every_encoding_in_numpy_and_torch(
    rid,
    monkeypatch,
):
    import torch as th

    spec = tasks.RANDOM_ALIAS_SPECS[rid]
    task = tasks.get_task(spec.env_name)
    desc = random_description(rid)
    with_random(monkeypatch)
    forbid_environment(monkeypatch)
    mask, info = cov.mask_from_description(desc, rid, task)
    assert info == {
        "kind": "alias",
        "alias_family": "random_state_binning",
        "hidden": spec.hidden,
        "table_sha256": tasks.FROZEN_TABLE_SHA256[rid],
        "rule": tasks.RANDOM_BINNING_RULE,
        "representative_rule": "first member of each bin in permuted order",
        "random_binning": {
            "seed": spec.seed,
            "n_bins": spec.n_bins,
            "generator": spec.generator,
            "partition_sha256": tasks.random_partition_sha256(rid),
            "frozen_table_sha256": tasks.FROZEN_TABLE_SHA256[rid],
            "n_reachable_nonterminal": len(static_reachable(spec.env_name)),
            "reachable_nonterminal_sha256": (
                tasks.FROZEN_REACHABLE_SHA256[spec.env_name]["states_sha256"]
            ),
            "reachability_evidence_sha256": (
                tasks.FROZEN_REACHABLE_SHA256[spec.env_name]["evidence_sha256"]
            ),
        },
        "registry": "calibration_candidate",
    }
    assert "smallest" not in json.dumps(info)
    table = np.asarray(desc["representative"])
    joint = np.arange(len(table))
    monkeypatch.setattr(restriction, "get_restriction", lambda _: mask)
    torch_mask = restriction.ObservationMask(rid)
    for dtype, th_dtype in ((np.float32, th.float32), (np.float64, th.float64)):
        obs = mask.encode(joint, dtype=dtype)
        assert obs.shape == (len(table), task.obs_dim)
        expected = mask.encode(table, dtype=dtype)
        masked = mask.apply(obs)
        assert masked.dtype == dtype and np.array_equal(masked, expected)
        assert np.array_equal(mask.decode(masked), table)
        out = torch_mask(th.as_tensor(obs, dtype=th_dtype))
        assert out.dtype == th_dtype
        assert np.array_equal(out.numpy(), expected)
    # Each bin collapses to its first shuffled member; other states keep identity.
    for group in desc["groups"]:
        assert set(table[group]) == {group[0]}
    outside = sorted(set(joint) - {s for g in desc["groups"] for s in g})
    assert np.array_equal(table[outside], outside)


def _spliced(field, other="cliffwalking_random_bins24_seed20261003"):
    def change(d):
        d[field] = random_description(other)[field]

    return change


def _partition(key, value):
    def change(d):
        d["partition"][key] = value
        rehash(d)

    return change


def _swap_members(d):
    g = d["groups"]
    g[0][1], g[1][1] = g[1][1], g[0][1]  # the table still matches the generator


def _move_member(d):
    g = d["groups"]
    g[-1].append(g[0].pop())
    regroup(d)


@pytest.mark.parametrize(
    "change",
    [
        _partition("seed", tasks.RANDOM_BINNING_SEED + 1),
        _partition("seed", float(tasks.RANDOM_BINNING_SEED)),
        _partition("n_bins", 24),
        _partition("n_bins", 20.0),
        _partition("generator", "random_state_binning/2"),
        _partition("restriction_id", "cliffwalking_random_bins24_seed20261003"),
        _partition("env_name", "Taxi-v3"),
        _partition("frozen_table_sha256", "1" * 64),
        _partition("reachable_nonterminal_sha256", "1" * 64),
        _partition("nested_across_bin_counts", True),
        _partition("representative", "smallest-index reachable nonterminal state"),
        _partition("extra", 1),
        lambda d: (d["partition"].pop("rule"), rehash(d)),
        lambda d: d["partition"].update(seed=tasks.RANDOM_BINNING_SEED + 1),
        lambda d: d.update(partition_sha256="2" * 64),
        lambda d: d.pop("partition"),
        lambda d: d.pop("partition_sha256"),
        lambda d: d.update(partition=json.dumps(d["partition"])),
        _spliced("partition"),
        lambda d: (_spliced("partition")(d), _spliced("partition_sha256")(d)),
        lambda d: d.pop("groups"),
        lambda d: d["groups"].pop(),
        lambda d: d["groups"][0].pop(),
        lambda d: d["groups"][1].append(d["groups"][0][0]),
        lambda d: d["groups"][-1].__setitem__(0, float(d["groups"][-1][0])),
        lambda d: d["groups"][-1].append(True),
        lambda d: d["groups"].append([47]),
        _swap_members,
        _move_member,
        _spliced("groups"),
        lambda d: d.update(groups=d["groups"][::-1]),
        lambda d: d.update(
            hidden=tasks.ALIAS_RUNGS["cliffwalking_column_pairs"].hidden
        ),
        lambda d: d.update(rule="smallest-index reachable nonterminal state"),
        lambda d: d.update(reachable_classes=21),
        lambda d: d.update(reachable_classes=20.0),
        lambda d: d.pop("reachable_nonterminal_states"),
        lambda d: d.update(identity_classes_without_reachable_state=0),
        lambda d: retable(
            d, [d["representative"][0]] + d["representative"][1:-1] + [0]
        ),
        lambda d: d["representative"].__setitem__(40, 0),
        lambda d: (
            _spliced("representative")(d),
            _spliced("table_sha256")(d),
            _spliced("frozen_table_sha256")(d),
        ),
        lambda d: d.update(table_sha256="4" * 64),
        lambda d: d.update(frozen_table_sha256="4" * 64),
        lambda d: d.update(restriction_id="cliffwalking_random_bins24_seed20261003"),
        lambda d: d.update(env_name="Taxi-v3"),
        lambda d: d.update(blocks=[4, 12]),
        lambda d: d.update(kind="quantized"),
        lambda d: d.pop("kind"),
        lambda d: d.update(obs_shape=[47]),
    ],
)
def test_tampered_random_binning_record_is_ineligible(change, monkeypatch):
    desc = random_description(CLIFF_RANDOM)
    with_random(monkeypatch)
    forbid_environment(monkeypatch)
    cov.mask_from_description(
        json.loads(json.dumps(desc)), CLIFF_RANDOM, tasks.TASKS[CLIFF]
    )
    change(desc)
    with pytest.raises(cov.Ineligible):
        cov.mask_from_description(desc, CLIFF_RANDOM, tasks.TASKS[CLIFF])


def _respec(mp, rid, **changes):
    spec = dataclasses.replace(tasks.RANDOM_ALIAS_SPECS[rid], **changes)
    mp.setattr(tasks, "RANDOM_ALIAS_SPECS", {**tasks.RANDOM_ALIAS_SPECS, rid: spec})


def test_unregistered_or_spliced_random_binnings_are_refused(monkeypatch):
    with_random(monkeypatch)
    cliff = tasks.TASKS[CLIFF]
    # Another env's registered binning, with its own valid record.
    taxi = "taxi_random_bins200_seed20261003"
    assert cov.restriction_kind(taxi, cliff) is None
    with pytest.raises(cov.Ineligible):
        cov.mask_from_description(random_description(taxi), taxi, cliff)
    # Ids that look like random binnings but are not registered.
    for rid in (
        "cliffwalking_random_bins21_seed20261003",
        "cliffwalking_random_bins20_seed20261004",
        "frozenlake_random_bins6_seed20261003",
    ):
        desc = random_description(CLIFF_RANDOM)
        desc["restriction_id"] = desc["partition"]["restriction_id"] = rid
        rehash(desc)
        assert cov.restriction_kind(rid, cliff) is None
        assert rid not in cov.permitted_restrictions(cliff)
        with pytest.raises(cov.Ineligible):
            cov.mask_from_description(desc, rid, cliff)
    # A random record under a factor rung id, and random fields on a factor rung.
    desc = random_description(CLIFF_RANDOM)
    desc["restriction_id"] = "cliffwalking_column_pairs"
    with pytest.raises(cov.Ineligible):
        cov.mask_from_description(desc, "cliffwalking_column_pairs", cliff)
    lake = description(LAKE)
    for key in cov._RANDOM_FIELDS:
        spliced = {**lake, key: random_description(CLIFF_RANDOM)[key]}
        with pytest.raises(cov.Ineligible):
            cov.mask_from_description(spliced, LAKE_MASK, tasks.TASKS[LAKE])
    # Registered but unlisted, unfrozen, rebound or ambiguous: not bound.
    valid = random_description(CLIFF_RANDOM)
    cov.mask_from_description(valid, CLIFF_RANDOM, cliff)
    for patch in (
        lambda mp: mp.setattr(
            tasks,
            "RANDOM_ALIAS_CANDIDATES",
            {**tasks.RANDOM_ALIAS_CANDIDATES, CLIFF: ()},
        ),
        lambda mp: mp.setattr(
            tasks,
            "FROZEN_TABLE_SHA256",
            {k: v for k, v in tasks.FROZEN_TABLE_SHA256.items() if k != CLIFF_RANDOM},
        ),
        lambda mp: mp.setattr(tasks, "FROZEN_REACHABLE_SHA256", {}),
        lambda mp: _respec(mp, CLIFF_RANDOM, env_name="Taxi-v3"),
        lambda mp: _respec(mp, CLIFF_RANDOM, restriction_id=taxi),
        lambda mp: mp.setattr(
            tasks,
            "ALIAS_RUNGS",
            {**tasks.ALIAS_RUNGS, CLIFF_RANDOM: tasks.ALIAS_RUNGS[LAKE_MASK]},
        ),
    ):
        with pytest.MonkeyPatch.context() as mp:
            patch(mp)
            assert cov.restriction_kind(CLIFF_RANDOM, cliff) is None
            assert CLIFF_RANDOM not in cov.permitted_restrictions(cliff)
            with pytest.raises(cov.Ineligible):
                cov.mask_from_description(valid, CLIFF_RANDOM, cliff)
    # A spec whose seed, count or generator drifts from the frozen record.
    for changes in ({"seed": 1}, {"n_bins": 24}, {"generator": "other/1"}):
        with pytest.MonkeyPatch.context() as mp:
            _respec(mp, CLIFF_RANDOM, **changes)
            with pytest.raises(cov.Ineligible):
                cov.mask_from_description(valid, CLIFF_RANDOM, cliff)


def test_factor_alias_metadata_is_unchanged():
    digest = tasks.FROZEN_TABLE_SHA256[LAKE_MASK]
    _, info = cov.mask_from_description(description(LAKE), LAKE_MASK, tasks.TASKS[LAKE])
    assert info == {
        "kind": "alias",
        "hidden": "column within pairs {0,1}, {2,3}",
        "table_sha256": digest,
        "rule": "smallest-index reachable nonterminal state",
        "registry": "ladder",
    }


@pytest.fixture(scope="module")
def cliff_base(tmp_path_factory, small_budget):
    return build_bundle(tmp_path_factory.mktemp("cliff"), CLIFF)


def test_random_binning_bundle_renders_with_static_provenance(
    cliff_base,
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "case"
    shutil.copytree(cliff_base, root)
    code, out, _ = run(root)
    assert code == cov.EXIT_REFUSED and not out.exists()  # unlisted by default
    with_random(monkeypatch)
    forbid_environment(monkeypatch)
    code, out, inventory = run(root)
    assert code == cov.EXIT_OK, inventory
    assert inventory["environments"][CLIFF]["status"] == "complete"
    summary = read_json(out / f"coverage-{CLIFF}-summary.json")
    info = summary["restriction"]
    assert info["alias_family"] == "random_state_binning"
    assert info["registry"] == "calibration_candidate"
    assert info["random_binning"]["seed"] == tasks.RANDOM_BINNING_SEED
    assert info["random_binning"]["n_bins"] == 20
    assert info["random_binning"]["generator"] == tasks.RANDOM_BINNING_GENERATOR
    assert info["random_binning"]["partition_sha256"] == (
        tasks.random_partition_sha256(CLIFF_RANDOM)
    )
    assert info["table_sha256"] == tasks.FROZEN_TABLE_SHA256[CLIFF_RANDOM]
    assert summary["display"]["shape"] == [4, 12]
    first = random_description(CLIFF_RANDOM)["groups"][0]
    table = {row["masked_representative"]: row for row in summary["alias_table"]}
    assert table[first[0]]["members_in_table"] == sorted(first)
    assert table[first[0]]["label_conflict"]["ftl/seed3100"] is True
    text = json.dumps(summary, ensure_ascii=False)
    assert "smallest" not in json.dumps(info) and chr(0x2014) not in text
    assert (out / f"coverage-{CLIFF}.png").exists()
