"""Retained labeled-state coverage of masked offline BC, BC-iid and FTL.

Postprocessing only. It reads pinned, locally received files, never launches
jobs, never creates an environment, never rolls out, never loads a checkpoint
or an expert, never trains and never contacts a remote machine. Outputs go
into a new directory only (built beside it and renamed into place)::

    PYTHONPATH=src python experiments/agnostic/analyze_masked_coverage.py \\
        analyze --manifest MANIFEST.json --output-dir NEW_DIR \\
        [--env CartPole-v1 ...] [--color-by method|expert_action]

    PYTHONPATH=src python experiments/agnostic/analyze_masked_coverage.py \\
        tree-digest SCRATCH_DEMOS_DIR

What is plotted. For every seed, each method's actual 1,000 supervised
(observation, expert action) training pairs: offline BC from the pinned
``fixed_bc_pool.npz`` (one fixed pool, fit once in full; a flat reference with
no arrival order, never a prefix curve), BC-iid from the pinned
``bc_iid_stream.npz`` and FTL from its retained scratch demonstrations. A
scratch demo stores ``obs`` with one more row than ``acts``; only ``obs[:-1]``
are labeled states, so the trailing (next or terminal) observation is never
counted. Scratch ``round-000`` is label arrival 1. When BC-iid scratch demos
are pinned, they must equal the stream's first 1,000 pairs in order.

Two representations are shown for each method on shared axes: the original
observation the expert labeled, and the learner's masked observation, rebuilt
from the restriction description recorded in the run configs (zeroed indices,
the frozen alias table checked against its digest, for a random state
binning also its static partition descriptor and full bin membership, the
frozen quantized bin descriptor checked against its registered descriptor and
digest, or a random-offset grid's full descriptor checked against its frozen
digest and registered grid).
Continuous tasks get
direct-coordinate panels for consecutive declared feature pairs and one
deterministic standardized PCA per representation pooled over all methods and
seeds; categorical tasks get exact semantic state maps. Exact masked label
conflicts are counted from the sample rows themselves.

Tasks and masks. An environment is looked up with `tasks.get_task`, so a
registered variant task (`tasks.VARIANT_TASKS`) is accepted when a manifest
or ``--env`` names it explicitly; it is labeled ``task_scope: variant`` and
is never added to the eight expected original environments. A manifest's
``restriction_id`` must be a masked ladder rung of its task or a calibration
candidate declared for that environment and bound to it in the registries
(`permitted_restrictions`), random state binnings and random-offset grids
included; identity is refused. Data and run records must
carry their task's recorded metadata (`task_checks`).

Status. Every expected environment, seed and method is reported as
``complete``, ``missing``, ``partial`` or ``ineligible`` with reasons. An
environment is rendered only when all 5 data jobs and all 15 masked runs are
complete; otherwise only its inventory is written. Exit codes: 0 every
selected environment rendered, 3 inventory only for at least one, 2 refused
(unreadable or malformed manifest, unsafe path, existing output).
"""

import argparse
import dataclasses
import hashlib
import io
import json
import math
import os
import pathlib
import re
import shutil
import sys
import tempfile
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from imitation.data import serialize  # noqa: E402
from imitation.experiments.agnostic import pilot  # noqa: E402
from imitation.experiments.agnostic import random_grid  # noqa: E402
from imitation.experiments.agnostic import restriction  # noqa: E402
from imitation.experiments.agnostic import tasks  # noqa: E402
from imitation.experiments.agnostic import restriction_pilot as rp  # noqa: E402

MANIFEST_SCHEMA = "agnostic-masked-coverage-manifest/1"
SUMMARY_SCHEMA = "agnostic-masked-coverage-summary/1"
INVENTORY_SCHEMA = "agnostic-masked-coverage-inventory/1"
SEEDS = (3100, 3101, 3102, 3103, 3104)
METHODS = ("bc", "bc_iid", "ftl")
N_LABELS = 1000
EXPECTED_ENVS = tuple(tasks.TASKS)
EXIT_OK, EXIT_REFUSED, EXIT_INVENTORY = 0, 2, 3

METHOD_LABEL = {
    "bc": "Offline BC (fixed pool, one full fit)",
    "bc_iid": "BC-iid (pinned stream)",
    "ftl": "FTL (retained scratch)",
}
# Method colors of the earlier pilot and five-seed figures, kept for
# cross-report consistency. Identity is never color alone: every panel is
# titled by its method column and overlays use per-method markers.
COLOR = {"ftl": "#1b7837", "bc": "#2166ac", "bc_iid": "#b2182b"}
MARKER = {"bc": "s", "bc_iid": "^", "ftl": "o"}
# Expert-action colors (validated for color-vision deficiency; legend and the
# numerical summary carry the same information).
ACTION_COLORS = ("#5e3c99", "#e08214", "#1b7837", "#c51b7d")

SAMPLED_DISTRIBUTION = {
    "bc": (
        "Offline BC: the pinned fixed_bc_pool.npz, i.e. "
        + rp.POOL_RULE
        + ". All 1000 rows are the one full-fit training set; a fixed "
        "reference without arrival order, never a growing prefix."
    ),
    "bc_iid": (
        "BC-iid: the pinned bc_iid_stream.npz, i.e. "
        + rp.STREAM_RULE
        + ". Label arrival t is stream row t (cross-checked against retained "
        "scratch when pinned)."
    ),
    "ftl": (
        "FTL: retained scratch demonstrations, one uniformly selected "
        "pre-action state per round from one episode run by the current "
        "learner (beta 0) and labeled by the expert; obs[:-1] only, so no "
        "trailing next or terminal observation is counted; scratch round-000 "
        "is label arrival 1."
    ),
}
CAPTION = (
    "Retained labeled-state coverage: the empirical distribution of each "
    "method's 1000 supervised training samples per seed (all seeds pooled, "
    "every sample shown). It is not final-policy occupancy, contains no "
    "evaluation trajectories or unretained visited states, and coverage alone "
    "does not establish an irreducible error floor, a return ceiling or a "
    "causal explanation of an FTL advantage. Offline BC is one fixed pool and "
    "one full fit. Nearby points in a projection are not identical masked "
    "observations; exact masked label conflicts are counted from sample rows "
    "only, and simulator-validity or reachability witnesses are a separate "
    "audit not used here."
)
TAXI_ENV = "Taxi-v3"
TAXI_PLANNER_PREFIX = "supervised distillation of an exact shortest-delivery planner"
TAXI_CAPTION = (
    "Taxi expert trained by supervised imitation of an exact planner. "
    "Students use its frozen feature extractor and train only the final layer."
)
ACCEPTANCE_SCHEMA = "agnostic-expert-recovery-acceptance/1"
UNITS = {
    "CartPole-v1": {
        "x": "cart position x (m)",
        "x_dot": "cart velocity x_dot (m/s)",
        "theta": "pole angle theta (rad)",
        "theta_dot": "pole angular velocity theta_dot (rad/s)",
    },
    "Acrobot-v1": {
        "theta1_dot": "theta1_dot (rad/s)",
        "theta2_dot": "theta2_dot (rad/s)",
    },
}
GRID_SHAPES = {"FrozenLake-v1": (4, 4), "CliffWalking-v0": (4, 12)}
LIMITATIONS = (
    "Only retained supervised samples are shown; visited but unretained "
    "states, evaluation rollouts and final-policy occupancy are not in the "
    "saved records.",
    "Exact label conflicts are counted within each method's own 1000-sample "
    "training set (and pooled across methods per seed); they are a property "
    "of the sample, not an irreducible error or a return bound.",
    "The PCA projection is fitted separately per representation, so areas "
    "are comparable across methods within one representation only, never "
    "between the original and masked views.",
    "Exact equality of continuous observations across methods is rare; "
    "projection proximity does not mean identical observations.",
    "Bytes are verified against the manifest pins (and any provenance index); "
    "the pins themselves come from the received provenance bundle and are "
    "trusted as such.",
)

_HEX = re.compile(r"[0-9a-f]{64}")
_ROUND = re.compile(r"round-(\d{3,})")
_RANK = {"complete": 0, "missing": 1, "partial": 2, "ineligible": 3}


class Refused(Exception):
    """The manifest or the request is unusable; nothing is written."""


class Ineligible(Exception):
    """A pinned input exists but fails a check."""


# ---------------------------------------------------------------------------
# Manifest and pinned files
# ---------------------------------------------------------------------------


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pin(entry: Any, where: str, key: str = "sha256") -> Dict[str, str]:
    if not isinstance(entry, dict):
        raise Refused(f"{where}: expected an object with path and {key}")
    path, digest = entry.get("path"), entry.get(key)
    if not isinstance(path, str) or not path:
        raise Refused(f"{where}: missing path")
    if path.startswith("/") or "\\" in path or ".." in path.split("/"):
        raise Refused(f"{where}: path must be relative inside the bundle: {path!r}")
    if not isinstance(digest, str) or not _HEX.fullmatch(digest):
        raise Refused(f"{where}: {key} must be 64 lowercase hex digits")
    return {"path": path, key: digest}


def load_manifest(path: pathlib.Path) -> Tuple[Dict[str, Any], str, pathlib.Path]:
    """Parse and structurally validate a coverage manifest.

    Args:
        path: The manifest file.

    Returns:
        The manifest, its SHA256 and the resolved bundle root.

    Raises:
        Refused: If the manifest is unreadable or malformed.
    """
    path = pathlib.Path(path)
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise Refused(f"Unreadable manifest {path}: {exc}") from None
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise Refused(f"Manifest schema must be {MANIFEST_SCHEMA}")
    if manifest.get("labels") != N_LABELS:
        raise Refused(f"Manifest labels must be {N_LABELS}")
    if manifest.get("seeds") != list(SEEDS):
        raise Refused(f"Manifest seeds must be exactly {list(SEEDS)}")
    if manifest.get("methods") != list(METHODS):
        raise Refused(f"Manifest methods must be exactly {list(METHODS)}")
    root = manifest.get("bundle_root")
    if not isinstance(root, str) or not root:
        raise Refused("Manifest needs bundle_root")
    bundle = (path.parent / root).resolve()
    if not bundle.is_dir():
        raise Refused(f"bundle_root {bundle} is not a directory")
    if not isinstance(manifest.get("require_index_corroboration", False), bool):
        raise Refused("require_index_corroboration must be a boolean")
    indexes = manifest.get("provenance_indexes", [])
    if not isinstance(indexes, list):
        raise Refused("provenance_indexes must be a list")
    for i, entry in enumerate(indexes):
        _pin(entry, f"provenance_indexes[{i}]")
        prefix = entry.get("prefix", "")
        if not isinstance(prefix, str) or prefix.startswith("/") or ".." in prefix:
            raise Refused(f"provenance_indexes[{i}].prefix must be relative")
    envs = manifest.get("environments")
    if not isinstance(envs, dict) or not envs:
        raise Refused("Manifest needs a non-empty environments mapping")
    for env_name, env in envs.items():
        _check_env_entry(env_name, env)
    return manifest, _sha(raw), bundle


def known_envs() -> Tuple[str, ...]:
    """Selectable environments: the eight original tasks, then registered variants.

    Only the originals (`EXPECTED_ENVS`) are expected in every inventory; a
    variant is analyzed only when a manifest or the caller names it.
    """
    return EXPECTED_ENVS + tuple(
        e for e in tasks.VARIANT_TASKS if e not in EXPECTED_ENVS
    )


def task_scope(env_name: str) -> str:
    """``original`` for the eight main tasks, ``variant`` for an opt-in variant."""
    return "original" if env_name in tasks.TASKS else "variant"


def restriction_kind(restriction_id: Any, task: tasks.Task) -> Optional[str]:
    """Registered kind of a masked restriction bound to ``task``, else None.

    Reads the static registries only (no environment, no alias table): an
    alias rung must be declared for this environment, a random state binning
    (`tasks.RANDOM_ALIAS_SPECS`) must be one of its environment's random
    candidates with a frozen table and reachable-set digest, a random-offset
    grid (`tasks.RANDOM_GRID_SPECS`) must be one of its continuous
    environment's grid candidates with a frozen descriptor digest, a quantized
    descriptor must name it, and a zeroing mask must fit its continuous
    observation.
    """
    if not isinstance(restriction_id, str) or restriction_id == "identity":
        return None
    grid = tasks.RANDOM_GRID_SPECS.get(restriction_id)
    if grid is not None:
        bound = (
            grid.restriction_id == restriction_id
            and grid.env_name == task.env_name
            and task.observation == "continuous"
            and restriction_id in tasks.RANDOM_GRID_CANDIDATES.get(task.env_name, ())
            and restriction_id in random_grid.FROZEN_GRID_SHA256
        )
        return "random_grid" if bound else None
    spec = tasks.RANDOM_ALIAS_SPECS.get(restriction_id)
    if spec is not None:
        bound = (
            restriction_id not in tasks.ALIAS_RUNGS
            and spec.restriction_id == restriction_id
            and spec.env_name == task.env_name
            and task.observation == "one_hot"
            and restriction_id in tasks.RANDOM_ALIAS_CANDIDATES.get(task.env_name, ())
            and restriction_id in tasks.FROZEN_TABLE_SHA256
            and task.env_name in tasks.FROZEN_REACHABLE_SHA256
        )
        return "alias" if bound else None
    rung = tasks.ALIAS_RUNGS.get(restriction_id)
    if rung is not None:
        bound = rung.env_name == task.env_name and task.observation == "one_hot"
        return "alias" if bound else None
    quantized = restriction.QUANTIZED_RESTRICTIONS.get(restriction_id)
    if quantized is not None:
        bound = quantized.env_name == task.env_name and task.observation == "continuous"
        return "quantized" if bound else None
    zeroing = restriction.RESTRICTIONS.get(restriction_id)
    if (
        zeroing is not None
        and task.observation == "continuous"
        and zeroing.obs_shape == (task.obs_dim,)
    ):
        return "zeroing"
    return None


def permitted_restrictions(task: tasks.Task) -> Tuple[str, ...]:
    """Masks a coverage manifest may name for ``task``, never identity.

    Its ladder rungs, then the calibration candidates declared for its own
    environment (`tasks.CALIBRATION_CANDIDATES`), then its random-offset
    grids (`tasks.RANDOM_GRID_CANDIDATES`); each must be registered and bound
    to the environment (`restriction_kind`).
    """
    ids = [rid for rid in task.ladder if restriction_kind(rid, task)]
    for entry in tasks.CALIBRATION_CANDIDATES.get(
        task.env_name, ()
    ) + tasks.RANDOM_GRID_CANDIDATES.get(task.env_name, ()):
        rid = entry if isinstance(entry, str) else getattr(entry, "restriction_id", 0)
        if rid not in ids and restriction_kind(rid, task):
            ids.append(rid)
    return tuple(ids)


def _check_env_entry(env_name: str, env: Any) -> None:
    where = f"environments[{env_name}]"
    try:
        task = tasks.get_task(env_name)
    except ValueError:
        raise Refused(
            f"{where}: not a registered task; expected one of {list(known_envs())}",
        ) from None
    if not isinstance(env, dict):
        raise Refused(f"{where}: expected an object")
    rid = env.get("restriction_id")
    if rid not in permitted_restrictions(task):
        raise Refused(
            f"{where}: restriction_id must be a masked ladder rung or a "
            f"registered calibration candidate of {env_name}",
        )
    for key in ("protocol",):
        if not isinstance(env.get(key), str):
            raise Refused(f"{where}: {key} must be a string")
    for key in ("expert_sha256", "task_sha256", "run_source_combined_sha256"):
        if not isinstance(env.get(key), str) or not _HEX.fullmatch(env[key]):
            raise Refused(f"{where}: {key} must be 64 lowercase hex digits")
    control = env.get("control")
    if not isinstance(control, dict) or not control:
        raise Refused(f"{where}: control must map names to pinned files")
    for name, entry in control.items():
        _pin(entry, f"{where}.control[{name}]")
    provenance = env.get("expert_provenance")
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise Refused(f"{where}: expert_provenance must be an object or null")
        _pin(provenance.get("acceptance"), f"{where}.expert_provenance.acceptance")
    seeds = env.get("seeds")
    if not isinstance(seeds, dict):
        raise Refused(f"{where}: seeds must be an object keyed by seed")
    unknown = sorted(set(seeds) - {str(s) for s in SEEDS})
    if unknown:
        raise Refused(f"{where}: unexpected seeds {unknown}")
    for seed, cells in seeds.items():
        if not isinstance(cells, dict):
            raise Refused(f"{where}.seeds[{seed}]: expected an object")
        extra = sorted(set(cells) - {"data", *METHODS})
        if extra:
            raise Refused(f"{where}.seeds[{seed}]: unexpected keys {extra}")
        for name, cell in cells.items():
            _check_cell_entry(f"{where}.seeds[{seed}].{name}", name, cell, control)


def _check_cell_entry(where: str, name: str, cell: Any, control: Dict) -> None:
    if not isinstance(cell, dict):
        raise Refused(f"{where}: expected an object")
    if not isinstance(cell.get("job_id"), str) or not cell["job_id"]:
        raise Refused(f"{where}: job_id must be a string")
    keys = ("manifest",) if name == "data" else ("manifest", "configs")
    for key in keys:
        if cell.get(key) not in control:
            raise Refused(f"{where}: {key} must name a control entry")
    _pin(cell.get("result"), f"{where}.result")
    if name == "data":
        _pin(cell.get("pool"), f"{where}.pool")
        _pin(cell.get("stream"), f"{where}.stream")
    if "scratch_demos" in cell:
        if name == "bc":
            raise Refused(f"{where}: offline BC writes no scratch demos")
        _pin(cell["scratch_demos"], f"{where}.scratch_demos", key="tree_sha256")
    elif name == "ftl":
        raise Refused(f"{where}: FTL needs pinned scratch_demos")


def _resolve(bundle: pathlib.Path, rel: str) -> pathlib.Path:
    path = (bundle / rel).resolve()
    if path != bundle and bundle not in path.parents:
        raise Refused(f"{rel!r} resolves outside the bundle")
    return path


def load_indexes(bundle: pathlib.Path, manifest: Dict) -> Dict[str, str]:
    """Bundle-relative path to SHA256 from the pinned provenance indexes.

    Accepted formats: a JSON list of ``{"path", "sha256"}`` objects, a JSON
    object whose ``files`` is such a list, or ``sha256sum`` text.

    Raises:
        Refused: If an index is missing, does not match its pin, is malformed
            or lists one path with two digests.
    """
    out: Dict[str, str] = {}
    for i, entry in enumerate(manifest.get("provenance_indexes", [])):
        path = _resolve(bundle, entry["path"])
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise Refused(f"provenance index {entry['path']}: {exc}") from None
        if _sha(raw) != entry["sha256"]:
            raise Refused(f"provenance index {entry['path']} does not match its pin")
        prefix = entry.get("prefix", "")
        pairs: List[Tuple[str, str]] = []
        try:
            data = json.loads(raw)
            rows = data.get("files") if isinstance(data, dict) else data
            if not isinstance(rows, list):
                raise ValueError("no file list")
            for row in rows:
                pairs.append((str(row["path"]), str(row["sha256"])))
        except (ValueError, KeyError, TypeError):
            for line in raw.decode("utf-8", "replace").splitlines():
                if not line.strip():
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2 or not _HEX.fullmatch(parts[0]):
                    raise Refused(f"provenance index {entry['path']} is malformed")
                pairs.append((parts[1].strip().lstrip("*"), parts[0]))
        for rel, digest in pairs:
            key = (pathlib.PurePosixPath(prefix) / rel).as_posix()
            if out.get(key, digest) != digest:
                raise Refused(f"provenance indexes disagree on {key}")
            out[key] = digest
        del i
    return out


def tree_entries(root: pathlib.Path) -> List[Tuple[str, str]]:
    """Sorted ``(relative posix path, file SHA256)`` of every file under root.

    Raises:
        Ineligible: If the tree contains a symlink or a special file.
    """
    out = []
    for path in sorted(pathlib.Path(root).rglob("*")):
        if path.is_symlink():
            raise Ineligible(f"symlink in scratch tree: {path.name}")
        if path.is_file():
            out.append((path.relative_to(root).as_posix(), pilot.sha256_file(path)))
        elif not path.is_dir():
            raise Ineligible(f"special file in scratch tree: {path.name}")
    return out


def tree_sha256(entries: Sequence[Tuple[str, str]]) -> str:
    """Digest of a file tree: SHA256 over sorted ``path NUL sha256 LF`` lines.

    It can be computed from a provenance index's per-file digests as well as
    from local files (`tree_entries`).
    """
    digest = hashlib.sha256()
    for rel, sha in sorted(entries):
        digest.update(f"{rel}\0{sha}\n".encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Cells: one data job or one masked run
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Cell:
    """Status, reasons and verified files of one data job or run."""

    name: str
    job_id: Optional[str]
    status: str = "complete"
    reasons: List[str] = dataclasses.field(default_factory=list)
    files: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
    absent: List[str] = dataclasses.field(default_factory=list)

    def mark(self, status: str, reason: str) -> None:
        self.reasons.append(reason)
        if _RANK[status] > _RANK[self.status]:
            self.status = status

    def finish_presence(self) -> None:
        """Missing if nothing pinned is present, partial if only some is."""
        if not self.absent:
            return
        status = "missing" if not self.files else "partial"
        self.mark(status, f"absent pinned files: {sorted(self.absent)}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "reasons": list(self.reasons),
            "files": self.files,
        }


class Reader:
    """Read pinned bytes once, verify them, and record what was verified."""

    def __init__(self, bundle: pathlib.Path, index: Dict[str, str], strict: bool):
        self.bundle = bundle
        self.index = index
        self.strict = strict

    def _corroborate(self, cell: Cell, name: str, rel: str, digest: str) -> bool:
        listed = self.index.get(rel)
        if listed is not None and listed != digest:
            raise Ineligible(f"{name}: provenance index lists {listed} for {rel}")
        if listed is None and self.strict:
            raise Ineligible(f"{name}: {rel} is not in any provenance index")
        return listed is not None

    def read(self, cell: Cell, name: str, pin: Dict[str, str]) -> Optional[bytes]:
        """Pinned bytes, or None (absent, or marked ineligible)."""
        path = _resolve(self.bundle, pin["path"])
        if not path.is_file():
            cell.absent.append(name)
            return None
        raw = path.read_bytes()
        actual = _sha(raw)
        try:
            if actual != pin["sha256"]:
                raise Ineligible(f"{name}: SHA256 {actual} is not the pinned one")
            listed = self._corroborate(cell, name, pin["path"], actual)
        except Ineligible as exc:
            cell.mark("ineligible", str(exc))
            return None
        cell.files[name] = {
            "path": pin["path"],
            "sha256": actual,
            "index_corroborated": listed,
        }
        return raw

    def tree(self, cell: Cell, name: str, pin: Dict[str, str]):
        """Verified scratch demos directory, or None."""
        path = _resolve(self.bundle, pin["path"])
        if not path.is_dir():
            cell.absent.append(name)
            return None
        try:
            entries = tree_entries(path)
            digest = tree_sha256(entries)
            if digest != pin["tree_sha256"]:
                raise Ineligible(f"{name}: tree SHA256 {digest} is not the pinned one")
            listed = [
                self._corroborate(
                    cell,
                    name,
                    (pathlib.PurePosixPath(pin["path"]) / rel).as_posix(),
                    sha,
                )
                for rel, sha in entries
            ]
        except Ineligible as exc:
            cell.mark("ineligible", str(exc))
            return None
        cell.files[name] = {
            "path": pin["path"],
            "tree_sha256": digest,
            "n_files": len(entries),
            "index_corroborated": bool(listed) and all(listed),
        }
        return path


def _json(raw: bytes, what: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise Ineligible(f"{what} is not JSON: {exc}") from None
    if not isinstance(value, dict):
        raise Ineligible(f"{what} is not a JSON object")
    return value


def _expert_data(raw: bytes, what: str) -> rp.ExpertData:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            data = rp.ExpertData(**{k: np.array(saved[k]) for k in rp._ARRAYS})
    except (OSError, ValueError, KeyError) as exc:
        raise Ineligible(f"{what} is not an ExpertData archive: {exc}") from None
    if not (len(data.obs) == len(data.acts) == len(data.next_obs)):
        raise Ineligible(f"{what}: arrays differ in length")
    return data


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require(checks: Dict[str, bool], what: str) -> None:
    failed = sorted(k for k, ok in checks.items() if not ok)
    if failed:
        raise Ineligible(f"{what} failed checks {failed}")


def _segments(path: str) -> Optional[List[str]]:
    """Path segments, or None if any is empty, ``.``, ``..`` or has a backslash."""
    parts = path[1:].split("/") if path.startswith("/") else path.split("/")
    if "\\" in path or any(p in ("", ".", "..") for p in parts):
        return None
    return parts


def _job(manifest_doc: Dict, job_id: str, result: str, what: str) -> Dict[str, Any]:
    """The campaign job ``job_id``, bound to the pinned result path ``result``.

    Scheduling IDs need not equal result folder names, so the binding is the
    job's own ``expected_result``: it must equal the bundle-relative pinned
    ``result`` or end with ``/`` plus all of its segments. A bundle may be
    transported, so the manifest's absolute root is not compared.
    """
    jobs = manifest_doc.get("jobs")
    found = [j for j in jobs if j.get("id") == job_id] if isinstance(jobs, list) else []
    if len(found) != 1:
        raise Ineligible(f"{what}: job {job_id} is not exactly once in its manifest")
    job = found[0]
    if not isinstance(job.get("expected_fields"), dict):
        raise Ineligible(f"{what}: job {job_id} has no expected_fields")
    expected = job.get("expected_result")
    declared = _segments(expected) if isinstance(expected, str) else None
    pinned = _segments(result)
    if (
        declared is None
        or pinned is None
        or result.startswith("/")
        or declared[-len(pinned) :] != pinned
    ):
        raise Ineligible(
            f"{what}: job {job_id} expected_result {expected!r} is not the "
            f"pinned result {result!r}",
        )
    return job


def task_checks(record: Dict, env_name: str) -> Dict[str, bool]:
    """Checks binding a data or run record's task metadata to its registered task.

    The recorded task description must name this task with its declared
    layout, ladder and episode cap, carry a variant's declared reset
    distribution (and an original task none), and digest to the record's
    ``task_sha256``. An original task and its variant can share an episode
    cap, so the cap alone never identifies the task.
    """
    task = tasks.get_task(env_name)
    meta = record.get("task") if isinstance(record.get("task"), dict) else {}
    declared = {
        "observation": task.observation,
        "feature_names": list(task.feature_names),
        "blocks": list(task.blocks),
        "ladder": list(task.ladder),
        "ladder_rule": task.ladder_rule,
    }
    support = tasks.DECLARED_RESET_SUPPORT.get(env_name)
    reset = meta.get("reset_distribution")
    if support is None:
        reset_ok = "reset_distribution" not in meta
    else:
        probabilities = [
            1.0 / len(support) if s in support else 0.0 for s in range(task.obs_dim)
        ]
        reset_ok = (
            isinstance(reset, dict)
            and reset.get("support") == list(support)
            and reset.get("probabilities") == probabilities
        )
    return {
        "episode_cap": record.get("episode_cap")
        == task.episode_cap
        == meta.get("episode_cap"),
        "task_identity": meta.get("env_name") == env_name
        and meta.get("declared") == declared,
        "task_reset_distribution": reset_ok,
        "task_metadata_sha256": bool(meta)
        and tasks._digest(meta) == record.get("task_sha256"),
    }


def _expected_fields(record: Dict, job: Dict, what: str) -> None:
    bad = sorted(k for k, v in job["expected_fields"].items() if record.get(k) != v)
    if bad:
        raise Ineligible(f"{what}: record differs from manifest expected_fields {bad}")


@dataclasses.dataclass
class DataJob:
    record: Dict[str, Any]
    sha256: str
    pool: rp.ExpertData
    stream: rp.ExpertData


def verify_data(reader, env_name, env, seed, entry, controls) -> Tuple[Cell, Any]:
    """Check one data job against its pins, manifest and env pins."""
    cell = Cell("data", entry.get("job_id") if entry else None)
    if entry is None:
        cell.mark("missing", "no manifest entry")
        return cell, None
    raw = reader.read(cell, "result", entry["result"])
    pool_raw = reader.read(cell, "pool", entry["pool"])
    stream_raw = reader.read(cell, "stream", entry["stream"])
    cell.finish_presence()
    if cell.status != "complete":
        return cell, None
    try:
        record = _json(raw, "data result")
        if record.get("status") != pilot.COMPLETE_STATUS:
            status = "partial" if record.get("status") == "running" else "ineligible"
            cell.mark(status, f"data status {record.get('status')!r}")
            return cell, None
        doc = controls.get(entry["manifest"])
        if doc is None:
            raise Ineligible(f"control {entry['manifest']} is not verified")
        job = _job(doc, cell.job_id, entry["result"]["path"], "data")
        _expected_fields(record, job, "data")
        folder = entry["result"]["path"].rsplit("/", 1)[0]
        if any(
            entry[k]["path"].rsplit("/", 1)[0] != folder for k in ("pool", "stream")
        ):
            raise Ineligible("pool and stream are not in the data job's directory")
        config = record.get("config") if isinstance(record.get("config"), dict) else {}
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        datasets = record.get("datasets")
        datasets = datasets if isinstance(datasets, dict) else {}
        _require(
            {
                "schema": record.get("schema") == env["protocol"] + "/data",
                "protocol": record.get("protocol") == env["protocol"],
                "env_name": record.get("env_name") == env_name,
                "seed": record.get("seed") == seed == config.get("seed"),
                "config_sha256": record.get("config_sha256")
                == pilot.config_digest(config),
                "config_rules": config.get("pool") == rp.POOL_RULE
                and config.get("stream") == rp.STREAM_RULE,
                "budget": config.get("budget") == N_LABELS,
                "expert_sha256": record.get("expert_sha256") == env["expert_sha256"],
                "task_sha256": record.get("task_sha256") == env["task_sha256"],
                **task_checks(record, env_name),
                "source_integrity": isinstance(source.get("files"), dict)
                and source.get("combined_sha256")
                == pilot.config_digest(source["files"]),
            },
            "data record",
        )
        loaded = {}
        for name, file_name, data_raw in (
            ("pool", rp.POOL_FILE, pool_raw),
            ("stream", rp.STREAM_FILE, stream_raw),
        ):
            meta = datasets.get(name) if isinstance(datasets.get(name), dict) else {}
            data = _expert_data(data_raw, file_name)
            _require(
                {
                    "file_name": meta.get("file") == file_name
                    and entry[name]["path"].split("/")[-1] == file_name,
                    "file_sha256": meta.get("file_sha256") == entry[name]["sha256"],
                    "pairs_sha256": meta.get("pairs_sha256") == data.pairs_sha256(),
                    "n": meta.get("n") == len(data) == N_LABELS,
                    "finite": bool(np.all(np.isfinite(data.obs))),
                },
                f"data {name}",
            )
            loaded[name] = data
    except Ineligible as exc:
        cell.mark("ineligible", str(exc))
        return cell, None
    return cell, DataJob(record, cell.files["result"]["sha256"], **loaded)


@dataclasses.dataclass
class Sample:
    """One method's retained supervised pairs for one seed."""

    method: str
    seed: int
    obs: np.ndarray
    acts: np.ndarray
    arrival: Optional[np.ndarray]
    excluded_final_observations: int
    origin: str
    joint: Optional[np.ndarray] = None


def read_scratch(demos_dir: pathlib.Path, n_labels: Optional[int] = None):
    """Retained pairs and one-based label arrival from round-loop scratch demos.

    Every entry of ``demos_dir`` must be a ``round-NNN`` directory, rounds
    must be exactly ``0 .. n_labels - 1`` and each must hold HuggingFace
    dataset directories named ``*.npz`` (read with `serialize.load`; plain
    files are refused, so no pickle is ever loaded) with exactly one labeled
    state. Only ``obs[:-1]`` of a trajectory are labeled states.

    Returns:
        ``(obs, acts, arrival, excluded_final_observations)`` in the order the
        trainer aggregated them (`restriction_pilot.trained_pairs`).

    Raises:
        Ineligible: On any deviation.
    """
    n_labels = N_LABELS if n_labels is None else int(n_labels)
    rounds = []
    for path in sorted(pathlib.Path(demos_dir).iterdir()):
        match = _ROUND.fullmatch(path.name)
        if match is None or not path.is_dir():
            raise Ineligible(f"unexpected scratch entry {path.name}")
        rounds.append((int(match.group(1)), path))
    rounds.sort()
    if [r for r, _ in rounds] != list(range(n_labels)):
        raise Ineligible(
            f"scratch rounds are not exactly 0..{n_labels - 1} "
            f"(found {len(rounds)})",
        )
    obs, acts, arrival = [], [], []
    excluded = 0
    for r, round_dir in rounds:
        count = 0
        for demo in sorted(round_dir.iterdir()):
            if not (demo.name.endswith(".npz") and demo.is_dir()):
                raise Ineligible(f"round-{r:03d}: {demo.name} is not a dataset dir")
            try:
                trajectories = list(serialize.load(demo))
            except Exception as exc:  # noqa: BLE001 (any reader failure)
                raise Ineligible(f"round-{r:03d}: unreadable {demo.name}: {exc}")
            for traj in trajectories:
                t_obs = np.asarray(traj.obs, dtype=np.float32)
                t_acts = np.asarray(traj.acts)
                if t_obs.ndim != 2 or len(t_obs) != len(t_acts) + 1:
                    raise Ineligible(f"round-{r:03d}: obs is not acts + 1 rows")
                obs.append(t_obs[:-1])
                acts.append(t_acts.astype(np.int64).reshape(-1))
                excluded += 1
                count += len(t_acts)
        if count != 1:
            raise Ineligible(f"round-{r:03d} holds {count} labeled states, not 1")
        arrival.append(r + 1)
    return (
        np.concatenate(obs).astype(np.float32),
        np.concatenate(acts),
        np.asarray(arrival, dtype=np.int64),
        excluded,
    )


def verify_run(reader, env_name, env, seed, method, entry, controls, data):
    """Check one masked run and return its cell and retained sample."""
    cell = Cell(method, entry.get("job_id") if entry else None)
    if entry is None:
        cell.mark("missing", "no manifest entry")
        return cell, None, None
    raw = reader.read(cell, "result", entry["result"])
    demos = None
    if "scratch_demos" in entry:
        demos = reader.tree(cell, "scratch_demos", entry["scratch_demos"])
    cell.finish_presence()
    if cell.status != "complete":
        return cell, None, None
    if data is None:
        cell.mark("ineligible", "its seed's data job is not verified")
        return cell, None, None
    try:
        record = _json(raw, "run result")
        if record.get("status") != pilot.COMPLETE_STATUS:
            status = "partial" if record.get("status") == "running" else "ineligible"
            cell.mark(status, f"run status {record.get('status')!r}")
            return cell, None, None
        doc = controls.get(entry["manifest"])
        configs = controls.get(entry["configs"])
        if doc is None or configs is None:
            raise Ineligible("its control files are not verified")
        job = _job(doc, cell.job_id, entry["result"]["path"], method)
        _expected_fields(record, job, method)
        config = record.get("config") if isinstance(record.get("config"), dict) else {}
        experiment = config.get("experiment")
        experiment = experiment if isinstance(experiment, dict) else {}
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        lineage = record.get("data") if isinstance(record.get("data"), dict) else {}
        acceptance = lineage.get("acceptance")
        acceptance = acceptance if isinstance(acceptance, dict) else {}
        description = config.get("restriction")
        records = (
            record.get("records") if isinstance(record.get("records"), list) else []
        )
        final = records[-1] if records and isinstance(records[-1], dict) else {}
        returns = final.get("episode_returns")
        demos_rel = f"scratch/{method}_{env_name}_seed{seed}/demos"
        _require(
            {
                "schema": record.get("schema") == env["protocol"] + "/run",
                "protocol": record.get("protocol") == env["protocol"],
                "env_name": record.get("env_name") == env_name
                and experiment.get("env_name") == env_name,
                "seed": record.get("seed") == seed == config.get("seed")
                and experiment.get("seed") == seed,
                "method": record.get("method") == method == config.get("method")
                and experiment.get("algo") == method,
                "restriction_id": record.get("restriction_id")
                == env["restriction_id"]
                == config.get("restriction_id"),
                "restriction_sha256": isinstance(description, dict)
                and config.get("restriction_sha256")
                == pilot.config_digest(description),
                "config_sha256": record.get("config_sha256")
                == pilot.config_digest(config),
                "run_configs": configs.get(cell.job_id) == config,
                "budget": experiment.get("n_rounds") == N_LABELS
                and experiment.get("samples_per_round") == 1,
                "expert_sha256": record.get("expert_sha256")
                == env["expert_sha256"]
                == data.record.get("expert_sha256"),
                "task_sha256": record.get("task_sha256")
                == env["task_sha256"]
                == config.get("task_sha256"),
                **task_checks(record, env_name),
                "source_pin": source.get("combined_sha256")
                == env["run_source_combined_sha256"]
                and isinstance(source.get("files"), dict)
                and source["combined_sha256"] == pilot.config_digest(source["files"]),
                "data_result_sha256": lineage.get("data_result_sha256")
                == data.sha256
                == acceptance.get("data_result_sha256"),
                "data_pairs": lineage.get("pool_pairs_sha256")
                == data.pool.pairs_sha256()
                and lineage.get("stream_pairs_sha256") == data.stream.pairs_sha256(),
                "data_source": (acceptance.get("producer_source") or {}).get(
                    "combined_sha256",
                )
                == (data.record.get("source") or {}).get("combined_sha256"),
                "final_labels": final.get("n_observations") == N_LABELS,
                "finite_metrics": all(
                    _finite(final.get(k))
                    for k in ("normalized_return", "disagreement_rate")
                )
                and isinstance(returns, list)
                and len(returns) == config.get("eval_episodes")
                and all(_finite(v) for v in returns),
                "scratch_path": record.get("scratch_demos")
                == (None if method == "bc" else demos_rel)
                and (
                    demos is None
                    or entry["scratch_demos"]["path"]
                    == f"{entry['result']['path'].rsplit('/', 1)[0]}/{demos_rel}"
                ),
            },
            f"{method} run",
        )
        if method == "bc":
            _require(
                {
                    "single_full_fit": len(records) == 1
                    and final.get("round") == 0
                    and final.get("reference") == "flat",
                    "trained_on_pool": record.get("trained_pairs_sha256")
                    == final.get("trained_pairs_sha256")
                    == data.pool.pairs_sha256(N_LABELS),
                },
                "bc run",
            )
            sample = Sample(
                method,
                seed,
                data.pool.obs[:N_LABELS].astype(np.float32),
                data.pool.acts[:N_LABELS].astype(np.int64),
                None,
                0,
                "fixed_bc_pool.npz (all rows)",
            )
        else:
            _require({"final_round": final.get("round") == N_LABELS}, f"{method} run")
            sample = _online_sample(record, method, seed, data, demos, cell)
        if not np.all(np.isfinite(sample.obs)):
            raise Ineligible("retained observations are not finite")
    except Ineligible as exc:
        cell.mark("ineligible", str(exc))
        return cell, None, None
    return cell, sample, description


def _online_sample(record, method, seed, data, demos, cell) -> Sample:
    trained = record.get("trained_pairs_sha256")
    if method == "bc_iid":
        if trained != data.stream.pairs_sha256(N_LABELS):
            raise Ineligible("BC-iid trained_pairs_sha256 is not the pinned stream's")
        obs = data.stream.obs[:N_LABELS].astype(np.float32)
        acts = data.stream.acts[:N_LABELS].astype(np.int64)
        arrival = np.arange(1, N_LABELS + 1, dtype=np.int64)
        excluded = 0
        if demos is None:
            cell.reasons.append(
                "BC-iid scratch not pinned: stream used, lineage via "
                "trained_pairs_sha256 only",
            )
        else:
            s_obs, s_acts, s_arrival, excluded = read_scratch(demos)
            if not (
                np.array_equal(s_obs, obs)
                and np.array_equal(s_acts, acts)
                and np.array_equal(s_arrival, arrival)
            ):
                raise Ineligible("BC-iid scratch pairs differ from the pinned stream")
            if rp.pairs_sha256(s_obs, s_acts) != trained:
                raise Ineligible(
                    "BC-iid scratch pairs differ from trained_pairs_sha256"
                )
            cell.files["scratch_demos"]["cross_check"] = "equal to stream rows 1..1000"
        return Sample(method, seed, obs, acts, arrival, excluded, "bc_iid_stream.npz")
    obs, acts, arrival, excluded = read_scratch(demos)
    if rp.pairs_sha256(obs, acts) != trained:
        raise Ineligible("FTL scratch pairs differ from trained_pairs_sha256")
    if tree_sha256(tree_entries(demos)) != cell.files["scratch_demos"]["tree_sha256"]:
        raise Ineligible("FTL scratch tree changed while it was read")
    return Sample(method, seed, obs, acts, arrival, excluded, "retained scratch demos")


# ---------------------------------------------------------------------------
# Masks rebuilt from recorded descriptions (no environment)
# ---------------------------------------------------------------------------


def _norm(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    return value


_QUANTIZED_FIELDS = sorted(
    f.name for f in dataclasses.fields(restriction.QuantizedRestriction)
)
_BIN_FIELDS = tuple(f.name for f in dataclasses.fields(restriction.QuantizedFeature))


def _bin_text(feature: Dict[str, Any]) -> str:
    name = feature["name"]
    if feature["rule"] == "round_to_step":
        return f"{name} rounded to a multiple of {feature['step']}"
    return (
        f"{name} := 0 if |{name}| <= {feature['dead_zone']}, else "
        f"sign * {feature['magnitude']}"
    )


def _quantized_mask(description: Dict[str, Any], restriction_id: str, task):
    """A recorded quantized mask, accepted only as its frozen registered descriptor.

    Every constructor field must be recorded (none is inferred or filled from
    current environment defaults), each bin must name the task coordinate it
    maps, the canonical digest of the recorded fields must equal the recorded
    and the registered ``descriptor_sha256``, and the rebuilt descriptor must
    equal the registered one (`restriction.QUANTIZED_RESTRICTIONS`, static).
    """
    body = {
        k: v for k, v in description.items() if k not in ("kind", "descriptor_sha256")
    }
    if sorted(body) != _QUANTIZED_FIELDS:
        raise Ineligible(f"quantized description fields are not {_QUANTIZED_FIELDS}")
    features = body["features"]
    if not (
        isinstance(features, list)
        and features
        and all(
            isinstance(f, dict) and sorted(f) == sorted(_BIN_FIELDS) for f in features
        )
    ):
        raise Ineligible("quantized description lacks complete bin parameters")
    indices = [f["index"] for f in features]
    if not (
        all(type(i) is int and 0 <= i < task.obs_dim for i in indices)
        and len(set(indices)) == len(indices)
        and all(f["name"] == task.feature_names[f["index"]] for f in features)
    ):
        raise Ineligible("quantized bins do not name distinct task coordinates")
    if not (
        body["env_name"] == task.env_name
        and body["obs_shape"] == [task.obs_dim]
        and task.observation == "continuous"
    ):
        raise Ineligible("quantized description is not for this task's observation")
    try:
        rebuilt = restriction.QuantizedRestriction(
            restriction_id=body["restriction_id"],
            env_name=body["env_name"],
            features=tuple(restriction.QuantizedFeature(**f) for f in features),
            obs_shape=tuple(body["obs_shape"]),
            description=body["description"],
        )
        digest = tasks._digest(body)
    except (TypeError, ValueError) as exc:
        raise Ineligible(f"quantized description is malformed: {exc}") from None
    registered = restriction.QUANTIZED_RESTRICTIONS[restriction_id]
    if not (
        digest
        == description.get("descriptor_sha256")
        == rebuilt.descriptor_sha256
        == registered.descriptor_sha256
    ):
        raise Ineligible(
            "quantized descriptor digest differs from the recorded or frozen one",
        )
    if rebuilt != registered:
        raise Ineligible("quantized descriptor differs from the registered one")
    binned = [f["name"] for f in features]
    info = {
        "kind": "quantized",
        "descriptor_sha256": digest,
        "bins": [{k: f[k] for k in _BIN_FIELDS} for f in features],
        "binned_features": binned,
        "kept_features": [n for n in task.feature_names if n not in binned],
        "hidden": "; ".join(_bin_text(f) for f in features),
        "clipping": "outputs clipped to each bin's [low, high]",
    }
    return rebuilt, info


_RANDOM_FIELDS = ("partition", "partition_sha256", "groups")


def _random_binning_info(description: Dict[str, Any], restriction_id: str, table):
    """Check a recorded random state binning against its static registry entry.

    The recorded ``partition`` must be the static descriptor
    (`tasks.random_partition_descriptor`) with equal canonical digests and
    ``partition_sha256``; ``groups`` must be exactly ``n_bins`` bins whose
    union is the frozen reachable set (`tasks.FROZEN_REACHABLE_SHA256`); the
    pinned generator rerun on that set must give the same bins and the same
    representative table; and the recorded rule, hidden text and class counts
    must be the registered ones. The reachability evidence is bound through
    the descriptor's ``reachability_evidence_sha256``. Nothing here builds an
    environment, enumerates reachability or reads data.
    """
    spec = tasks.RANDOM_ALIAS_SPECS[restriction_id]
    reach = tasks.FROZEN_REACHABLE_SHA256[spec.env_name]
    static = tasks.random_partition_descriptor(restriction_id)
    partition = description.get("partition")
    if not (
        isinstance(partition, dict)
        and tasks._digest(partition)
        == description.get("partition_sha256")
        == tasks._digest(static)
        == tasks.random_partition_sha256(restriction_id)
    ):
        raise Ineligible(
            "random partition descriptor differs from the recorded or frozen one",
        )
    if (
        description.get("hidden") != spec.hidden
        or description.get("rule") != tasks.RANDOM_BINNING_RULE
    ):
        raise Ineligible("random binning rule or hidden text is not the registered one")
    groups = description.get("groups")
    if not (
        isinstance(groups, list)
        and len(groups) == spec.n_bins
        and all(
            isinstance(g, list) and g and all(type(s) is int for s in g) for g in groups
        )
    ):
        raise Ineligible("random binning lacks its full bin membership")
    members = sorted(s for g in groups for s in g)
    if not (
        len(set(members)) == len(members) == reach["n_states"]
        and tasks._digest(members) == reach["states_sha256"]
    ):
        raise Ineligible("random bins do not partition the frozen reachable set")
    n = len(table)
    try:
        rebuilt = tasks.random_state_partition(
            members, spec.n_bins, spec.seed, n_states=n
        )
        rebuilt_table = tasks.random_alias_table(n, members, spec.n_bins, spec.seed)
    except ValueError as exc:
        raise Ineligible(f"random bins are malformed: {exc}") from None
    if [list(g) for g in rebuilt] != groups:
        raise Ineligible("random bins are not the pinned generator's bins")
    if list(rebuilt_table) != table:
        raise Ineligible("random representative table is not the pinned generator's")
    counts = {
        "reachable_nonterminal_states": reach["n_states"],
        "reachable_classes": spec.n_bins,
        "identity_classes_without_reachable_state": n - reach["n_states"],
    }
    if any(
        type(description.get(k)) is not int or description[k] != v
        for k, v in counts.items()
    ):
        raise Ineligible("random binning class counts are not the registered ones")
    return {
        "kind": "alias",
        "alias_family": "random_state_binning",
        "hidden": spec.hidden,
        "table_sha256": tasks.table_sha256(tuple(static["blocks"]), tuple(table)),
        "rule": tasks.RANDOM_BINNING_RULE,
        "representative_rule": static["representative"],
        "random_binning": {
            "seed": spec.seed,
            "n_bins": spec.n_bins,
            "generator": spec.generator,
            "partition_sha256": description["partition_sha256"],
            "frozen_table_sha256": static["frozen_table_sha256"],
            "n_reachable_nonterminal": reach["n_states"],
            "reachable_nonterminal_sha256": reach["states_sha256"],
            "reachability_evidence_sha256": reach["evidence_sha256"],
        },
    }


_GRID_FIELDS = sorted(
    f.name for f in dataclasses.fields(random_grid.RandomGridRestriction)
)
_GRID_AXIS_FIELDS = sorted(f.name for f in dataclasses.fields(random_grid.GridAxis))
_GRID_RECORD_FIELDS = sorted(
    _GRID_FIELDS + ["descriptor_sha256", "kind", "nominal_cells"]
)


def grid_text(n_bins: int, seed: int, axes: Sequence[Dict[str, Any]]) -> str:
    """Concise human description of a random-offset grid, from its axes."""
    kinds = {a["kind"] for a in axes}
    cells = []
    if kinds & {"linear", "unbounded"}:
        cells.append(f"{n_bins + 1} cells per scalar axis")
    if "circular" in kinds:
        cells.append(f"{n_bins} arcs per angle")
    text = f"Random-offset grid: K={n_bins} per axis"
    text += f" ({', '.join(cells)})" if cells else ""
    text += f"; seed {seed}"
    binary = [a["name"] for a in axes if a["kind"] == "binary"]
    if binary:
        text += (
            "; leg contacts unchanged"
            if all("contact" in n for n in binary)
            else f"; {', '.join(binary)} unchanged"
        )
    return text + "."


def _random_grid_mask(description: Dict[str, Any], restriction_id: str, task):
    """A recorded random-offset grid, accepted only as its frozen registered map.

    The record must be exactly the `restriction.describe_restriction` record
    (the full descriptor plus ``kind``, ``descriptor_sha256`` and
    ``nominal_cells``; nothing is filled in), name this task's environment,
    count, seed, generator and declared Box bits, and its canonical digest
    must equal the recorded and the frozen ``random_grid.FROZEN_GRID_SHA256``
    one. The NumPy map is rebuilt from the recorded axes and must equal the
    registered grid (`random_grid.get_grid`, built from frozen tables). No
    environment is constructed.
    """
    if sorted(description) != _GRID_RECORD_FIELDS:
        raise Ineligible(
            f"random grid description fields are not {_GRID_RECORD_FIELDS}"
        )
    body = {k: v for k, v in description.items() if k in _GRID_FIELDS}
    spec = tasks.RANDOM_GRID_SPECS[restriction_id]
    bits = random_grid.DECLARED_BOX_BITS.get(task.env_name, {})
    if not (
        body["restriction_id"] == restriction_id
        and body["env_name"] == spec.env_name == task.env_name
        and type(body["n_bins"]) is int
        and body["n_bins"] == spec.n_bins
        and type(body["seed"]) is int
        and body["seed"] == spec.seed
        and body["generator"] == spec.generator
        and body["obs_shape"] == [task.obs_dim]
        and task.observation == "continuous"
        and body["box_low_bits"] == list(bits.get("low", ()))
        and body["box_high_bits"] == list(bits.get("high", ()))
    ):
        raise Ineligible("random grid description is not this task's registered grid")
    digest = tasks._digest(body)
    frozen = random_grid.FROZEN_GRID_SHA256.get(restriction_id)
    if not digest == description.get("descriptor_sha256") == frozen:
        raise Ineligible(
            "random grid descriptor digest differs from the recorded or frozen one"
        )
    axes = body["axes"]
    if not (
        isinstance(axes, list)
        and axes
        and all(isinstance(a, dict) and sorted(a) == _GRID_AXIS_FIELDS for a in axes)
    ):
        raise Ineligible("random grid description lacks complete axes")
    try:
        rebuilt = random_grid.RandomGridRestriction(
            **{
                **{k: v for k, v in body.items() if k != "axes"},
                "obs_shape": tuple(body["obs_shape"]),
                "box_low_bits": tuple(body["box_low_bits"]),
                "box_high_bits": tuple(body["box_high_bits"]),
                "offsets": tuple(body["offsets"]),
                "axes": tuple(
                    random_grid.GridAxis(
                        **{
                            k: tuple(v) if isinstance(v, list) else v
                            for k, v in a.items()
                        },
                    )
                    for a in axes
                ),
            },
        )
        registered = random_grid.get_grid(restriction_id)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Ineligible(f"random grid description is malformed: {exc}") from None
    if not (
        rebuilt == registered
        and rebuilt.descriptor_sha256 == registered.descriptor_sha256 == frozen
    ):
        raise Ineligible("random grid map differs from the registered frozen grid")
    nominal = description.get("nominal_cells")
    if type(nominal) is not int or nominal != registered.nominal_cells:
        raise Ineligible("random grid nominal cell count is not the registered one")
    summary = [
        {"name": a["name"], "kind": a["kind"], "n_cells": rebuilt.axes[i].n_cells}
        for i, a in enumerate(axes)
    ]
    info = {
        "kind": "random_grid",
        "descriptor_sha256": digest,
        "n_bins": spec.n_bins,
        "seed": spec.seed,
        "generator": spec.generator,
        "axes": summary,
        "nominal_cells": nominal,
        "hidden": grid_text(spec.n_bins, spec.seed, summary),
    }
    return rebuilt, info


def mask_from_description(description: Any, restriction_id: str, task: tasks.Task):
    """Rebuild the learner mask from the description recorded in a run config.

    The id must be permitted for the task (`permitted_restrictions`) and the
    recorded kind must be its registered kind. Zeroing masks need
    ``zeroed_indices`` and ``obs_shape``; alias masks need ``blocks`` and a
    full ``representative`` table whose digest equals the recorded and frozen
    ``table_sha256``; a random state binning must also carry its complete
    static partition record (`_random_binning_info`), and a factor alias
    must carry none; quantized masks must be their frozen registered
    descriptor (`_quantized_mask`), and random-offset grids their frozen
    registered grid (`_random_grid_mask`). A zeroing mask must equal its
    registry entry. No environment is constructed (alias tables are never
    rebuilt through `tasks.alias_table`).

    Returns:
        ``(mask, info)`` where ``mask.apply`` maps original observations and
        ``info["registry"]`` is ``ladder`` or ``calibration_candidate``.

    Raises:
        Ineligible: If the description is insufficient or inconsistent.
    """
    if not isinstance(description, dict):
        raise Ineligible("no restriction description recorded")
    description = _norm(description)
    if description.get("restriction_id") != restriction_id:
        raise Ineligible("recorded restriction description names another id")
    kind = restriction_kind(restriction_id, task)
    if kind is None or restriction_id not in permitted_restrictions(task):
        raise Ineligible(
            f"{restriction_id!r} is not a permitted masked restriction of "
            f"{task.env_name}",
        )
    if description.get("kind", "zeroing") != kind:
        raise Ineligible(
            f"recorded restriction kind {description.get('kind')!r} is not the "
            f"registered {kind!r}",
        )
    registry = "ladder" if restriction_id in task.ladder else "calibration_candidate"
    if kind == "random_grid":
        mask, info = _random_grid_mask(description, restriction_id, task)
        info["registry"] = registry
        return mask, info
    if kind == "quantized":
        mask, info = _quantized_mask(description, restriction_id, task)
        info["registry"] = registry
        return mask, info
    dim = task.obs_dim
    if kind == "alias":
        if description.get("env_name") != task.env_name:
            raise Ineligible("alias description names another environment")
        blocks = description.get("blocks")
        table = description.get("representative")
        if not (
            isinstance(blocks, list)
            and blocks
            and all(isinstance(b, int) and b > 0 for b in blocks)
            and tuple(blocks) == task.blocks
        ):
            raise Ineligible("alias description lacks the task's one-hot blocks")
        n = int(np.prod(blocks))
        if not (
            isinstance(table, list)
            and len(table) == n
            and all(isinstance(s, int) and 0 <= s < n for s in table)
        ):
            raise Ineligible("alias description lacks a full representative table")
        digest = tasks.table_sha256(tuple(blocks), tuple(table))
        frozen = tasks.FROZEN_TABLE_SHA256.get(restriction_id)
        if not (
            digest
            == description.get("table_sha256")
            == description.get("frozen_table_sha256")
            == frozen
        ):
            raise Ineligible(
                "alias table digest differs from the recorded or frozen one"
            )
        if any(table[table[s]] != table[s] for s in range(n)):
            raise Ineligible("alias table is not idempotent")
        if description.get("obs_shape") != [dim]:
            raise Ineligible("alias description obs_shape is not the task's")
        if restriction_id in tasks.RANDOM_ALIAS_SPECS:
            info = _random_binning_info(description, restriction_id, table)
        elif any(k in description for k in _RANDOM_FIELDS):
            raise Ineligible("a factor alias description carries random bin fields")
        else:
            info = {
                "kind": "alias",
                "hidden": description.get("hidden"),
                "table_sha256": digest,
                "rule": description.get("rule"),
            }
        info["registry"] = registry
        mask = restriction.AliasRestriction(
            restriction_id=restriction_id,
            env_name=task.env_name,
            blocks=tuple(blocks),
            representative=tuple(table),
            table_sha256=digest,
            obs_shape=(dim,),
            description=str(description.get("description")),
        )
        return mask, info
    if "kind" in description:
        raise Ineligible(f"unknown restriction kind {description['kind']!r}")
    zeroed = description.get("zeroed_indices")
    if not (
        isinstance(zeroed, list)
        and zeroed
        and all(isinstance(i, int) and 0 <= i < dim for i in zeroed)
        and len(set(zeroed)) == len(zeroed)
        and description.get("obs_shape") == [dim]
        and task.observation == "continuous"
    ):
        raise Ineligible("zeroing description lacks valid zeroed_indices/obs_shape")
    registered = restriction.RESTRICTIONS[restriction_id]
    if _norm(dataclasses.asdict(registered)) != _norm(description):
        raise Ineligible("zeroing description differs from the registered mask")
    mask = restriction.Restriction(
        restriction_id=restriction_id,
        zeroed_indices=tuple(zeroed),
        obs_shape=(dim,),
        description=str(description.get("description")),
    )
    info = {
        "kind": "zeroing",
        "zeroed_features": [task.feature_names[i] for i in zeroed],
        "registry_agrees": True,
        "registry": registry,
    }
    return mask, info


# ---------------------------------------------------------------------------
# Expert provenance (Taxi caption)
# ---------------------------------------------------------------------------


def verify_expert_provenance(reader, env_name, env) -> Tuple[Cell, Optional[str]]:
    """Check a pinned expert acceptance record; return the required caption."""
    cell = Cell("expert_provenance", None)
    entry = env.get("expert_provenance")
    if entry is None:
        if env_name == TAXI_ENV:
            cell.mark(
                "ineligible",
                "Taxi expert provenance is not pinned, so the required Taxi "
                "caption cannot be decided",
            )
        return cell, None
    raw = reader.read(cell, "acceptance", entry["acceptance"])
    cell.finish_presence()
    if raw is None:
        return cell, None
    try:
        acc = _json(raw, "acceptance record")
        files = acc.get("files") if isinstance(acc.get("files"), dict) else {}
        checkpoint = files.get("checkpoint") if isinstance(files, dict) else None
        _require(
            {
                "schema": acc.get("schema") == ACCEPTANCE_SCHEMA,
                "env_name": acc.get("env_name") == env_name,
                "status": acc.get("status") == pilot.COMPLETE_STATUS,
                "release_allowed": acc.get("release_allowed") is True,
                "checkpoint_is_run_expert": isinstance(checkpoint, dict)
                and checkpoint.get("sha256") == env["expert_sha256"],
                "training_method": isinstance(acc.get("training_method"), str),
            },
            "expert acceptance",
        )
    except Ineligible as exc:
        cell.mark("ineligible", str(exc))
        return cell, None
    method = acc["training_method"]
    cell.files["acceptance"]["training_method"] = method
    planner = method.startswith(TAXI_PLANNER_PREFIX)
    return cell, (TAXI_CAPTION if env_name == TAXI_ENV and planner else None)


# ---------------------------------------------------------------------------
# Numerical summary
# ---------------------------------------------------------------------------


def _canon(obs: np.ndarray) -> np.ndarray:
    # + 0.0 maps -0.0 to 0.0 so exact grouping follows numeric equality.
    return np.ascontiguousarray(np.asarray(obs, dtype=np.float32) + np.float32(0.0))


def _row_keys(obs: np.ndarray) -> List[bytes]:
    rows = _canon(obs)
    return [rows[i].tobytes() for i in range(len(rows))]


def conflict_stats(orig: np.ndarray, masked: np.ndarray, acts: np.ndarray):
    """Exact masked-observation groups of one sample and their label conflicts.

    Returns:
        Counts over the actual rows: distinct original and masked
        observations, masked groups merging several originals, groups whose
        expert labels differ, the samples in those groups, and the number of
        samples a per-group majority label cannot fit (a property of this
        sample only).
    """
    _, m_inv = np.unique(_canon(masked), axis=0, return_inverse=True)
    _, o_inv = np.unique(_canon(orig), axis=0, return_inverse=True)
    m_inv, o_inv = m_inv.reshape(-1), o_inv.reshape(-1)
    collapsed = conflicts = conflict_samples = residual = 0
    for g in range(int(m_inv.max()) + 1 if len(m_inv) else 0):
        rows = m_inv == g
        labels = np.bincount(acts[rows])
        if len(np.unique(o_inv[rows])) > 1:
            collapsed += 1
        if np.count_nonzero(labels) > 1:
            conflicts += 1
            conflict_samples += int(rows.sum())
        residual += int(rows.sum() - labels.max())
    orig_conflicts = sum(
        np.count_nonzero(np.bincount(acts[o_inv == g])) > 1
        for g in range(int(o_inv.max()) + 1 if len(o_inv) else 0)
    )
    return {
        "n": int(len(acts)),
        "distinct_original": int(o_inv.max() + 1) if len(o_inv) else 0,
        "distinct_masked": int(m_inv.max() + 1) if len(m_inv) else 0,
        "masked_groups_merging_originals": collapsed,
        "masked_groups_with_label_conflict": conflicts,
        "samples_in_conflicting_groups": conflict_samples,
        "majority_label_misfit_samples": residual,
        "original_groups_with_label_conflict": int(orig_conflicts),
        "action_counts": np.bincount(acts).tolist(),
    }


def overlap_stats(samples: Dict[str, "Sample"], masked: Dict[str, np.ndarray]):
    """Exact distinct-row overlap between methods within one seed."""
    out: Dict[str, Any] = {}
    for rep, arrays in (
        ("original", {m: s.obs for m, s in samples.items()}),
        ("masked", masked),
    ):
        sets = {m: set(_row_keys(a)) for m, a in arrays.items()}
        pairs = {
            f"{a}&{b}": len(sets[a] & sets[b])
            for i, a in enumerate(METHODS)
            for b in METHODS[i + 1 :]
        }
        pairs["all_three"] = len(sets["bc"] & sets["bc_iid"] & sets["ftl"])
        pairs["ftl_only"] = len(sets["ftl"] - sets["bc"] - sets["bc_iid"])
        pairs["offline_only"] = len((sets["bc"] | sets["bc_iid"]) - sets["ftl"])
        out[rep] = {"distinct_row_intersections": pairs}
    return out


def pca(points: np.ndarray, k: int = 2) -> Dict[str, Any]:
    """Deterministic standardized PCA (SVD) with sign-fixed components.

    Columns are centered and divided by their pooled standard deviation
    (columns with zero variance are left unscaled). Each component's largest
    absolute loading is made positive.
    """
    x = np.asarray(points, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    total = float(np.sum(s**2))
    comps = np.zeros((k, x.shape[1]))
    ratio = [0.0] * k
    for i in range(min(k, len(s))):
        v = vt[i]
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        comps[i] = v
        ratio[i] = float(s[i] ** 2 / total) if total > 0 else 0.0
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "components": comps.tolist(),
        "explained_variance_ratio": ratio,
    }


def project(points: np.ndarray, fit: Dict[str, Any]) -> np.ndarray:
    z = (np.asarray(points, np.float64) - fit["mean"]) / np.asarray(fit["scale"])
    return z @ np.asarray(fit["components"]).T


def feature_pairs(names: Sequence[str]) -> List[Tuple[int, int]]:
    """Consecutive declared feature pairs (an odd last one pairs backward)."""
    n = len(names)
    if n < 2:
        return []
    pairs = [(i, i + 1) for i in range(0, n - 1, 2)]
    if n % 2:
        pairs.append((n - 2, n - 1))
    return pairs


def coordinate_stats(arr: np.ndarray, names: Sequence[str]) -> Dict[str, Any]:
    q = np.quantile(arr, [0.0, 0.01, 0.5, 0.99, 1.0], axis=0)
    return {
        name: {
            "min": float(q[0, i]),
            "q01": float(q[1, i]),
            "median": float(q[2, i]),
            "q99": float(q[3, i]),
            "max": float(q[4, i]),
            "mean": float(arr[:, i].mean()),
            "std": float(arr[:, i].std()),
        }
        for i, name in enumerate(names)
    }


def semantic_layout(env_name: str, n: int):
    """Exact 2-D layout of joint discrete states: ``(shape, pos, labels)``."""
    if env_name in GRID_SHAPES:
        rows, cols = GRID_SHAPES[env_name]
        return (rows, cols), (lambda s: divmod(s, cols)), ("column", "row")
    if env_name == TAXI_ENV:

        def pos(s):
            row, col, passenger, dest = tasks._taxi_factors(s)
            return passenger * 4 + dest, row * 5 + col

        return (
            (20, 25),
            pos,
            (
                "taxi cell (row*5 + col)",
                "passenger*4 + destination",
            ),
        )
    if env_name == "Blackjack-v1":

        def pos(s):
            player, dealer, ace = tasks.blackjack_factors(s)
            return dealer * 2 + ace, player

        return (22, 32), pos, ("player sum index", "dealer card*2 + usable ace")
    return (1, n), (lambda s: (0, s)), ("joint state index", "")


def alias_table(samples_by_seed, rep: np.ndarray) -> List[Dict[str, Any]]:
    """Observed masked classes with their member states and expert labels."""
    rows: Dict[int, Dict[str, Any]] = {}
    for seed, by_method in samples_by_seed.items():
        for method, s in by_method.items():
            for state, act in zip(s.joint.tolist(), s.acts.tolist()):
                cls = int(rep[state])
                row = rows.setdefault(
                    cls,
                    {
                        "masked_representative": cls,
                        "members_in_table": [
                            int(m) for m in np.flatnonzero(rep == cls)
                        ],
                        "observed": {},
                    },
                )
                key = f"{method}/seed{seed}"
                obs = row["observed"].setdefault(key, {})
                obs.setdefault(str(state), {})
                obs[str(state)][str(act)] = obs[str(state)].get(str(act), 0) + 1
    out = []
    for cls in sorted(rows):
        row = rows[cls]
        conflicts = {}
        for key, states in row["observed"].items():
            labels = {a for counts in states.values() for a in counts}
            conflicts[key] = len(labels) > 1
        row["label_conflict"] = conflicts
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _limits(values: Sequence[np.ndarray]) -> Tuple[float, float]:
    lo = float(min(np.min(v) for v in values))
    hi = float(max(np.max(v) for v in values))
    if hi - lo <= 0:
        return lo - 1.0, hi + 1.0
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def _caption(fig, text: str) -> None:
    width = int(fig.get_figwidth() * 14.5)
    fig.text(
        0.01,
        0.005,
        "\n".join(textwrap.wrap(text, width)),
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#333333",
    )


def plot_continuous(env_name, rid, pooled, masked, names, fits, color_by, captions):
    """Method columns; rows are coordinate pairs then PCA, original then masked.

    Returns:
        The figure and the display layout record (limits and counts).
    """
    pairs = feature_pairs(names)
    units = UNITS.get(env_name, {})
    label = {n: units.get(n, f"{n} (simulator units)") for n in names}
    rows = [("pair", p, rep) for p in pairs for rep in ("original", "masked")]
    rows += [("pca", None, rep) for rep in ("original", "masked")]
    projected = {
        rep: {m: project(src[m], fits[rep]) for m in METHODS}
        for rep, src in (("original", pooled), ("masked", masked))
    }
    fig, axes = plt.subplots(
        len(rows),
        len(METHODS),
        figsize=(4.2 * len(METHODS), 3.0 * len(rows) + 1.6),
        squeeze=False,
    )
    layout = []
    for r, (kind, pair, rep) in enumerate(rows):
        src = pooled if rep == "original" else masked
        if kind == "pair":
            i, j = pair
            both = [pooled[m] for m in METHODS] + [masked[m] for m in METHODS]
            xlim = _limits([a[:, i] for a in both])
            ylim = _limits([a[:, j] for a in both])
            xlab, ylab = label[names[i]], label[names[j]]
            points = {m: src[m][:, [i, j]] for m in METHODS}
        else:
            points = projected[rep]
            xlim = _limits([p[:, 0] for p in points.values()])
            ylim = _limits([p[:, 1] for p in points.values()])
            ev = fits[rep]["explained_variance_ratio"]
            xlab = f"{rep} PC1 ({100 * ev[0]:.1f}% var.)"
            ylab = f"{rep} PC2 ({100 * ev[1]:.1f}% var.)"
        for c, m in enumerate(METHODS):
            ax = axes[r][c]
            pts = points[m]
            if color_by == "expert_action":
                acts = pooled["acts"][m]
                colors = [ACTION_COLORS[a % len(ACTION_COLORS)] for a in acts]
            else:
                colors = COLOR[m]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=3,
                c=colors,
                marker=MARKER[m],
                alpha=0.3,
                linewidths=0,
                rasterized=True,
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(color="#e5e5e5", lw=0.6)
            ax.tick_params(labelsize=7)
            ax.set_xlabel(xlab, fontsize=8)
            ax.set_ylabel(ylab, fontsize=8)
            view = "original" if rep == "original" else f"masked ({rid})"
            ax.set_title(f"{METHOD_LABEL[m]}\n{view}, n={len(pts)}", fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOR[m])
        layout.append(
            {
                "row": r,
                "kind": kind,
                "representation": rep,
                "features": None if pair is None else [names[pair[0]], names[pair[1]]],
                "xlim": list(xlim),
                "ylim": list(ylim),
                "display_counts": {m: int(len(points[m])) for m in METHODS},
            },
        )
    if color_by == "expert_action":
        n_actions = int(max(pooled["acts"][m].max() for m in METHODS)) + 1
        handles = [
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                color=ACTION_COLORS[a % 4],
                label=f"expert action {a}",
            )
            for a in range(n_actions)
        ]
    else:
        handles = [
            Line2D(
                [], [], ls="", marker=MARKER[m], color=COLOR[m], label=METHOD_LABEL[m]
            )
            for m in METHODS
        ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        f"{env_name}: retained labeled-state coverage, original vs learner "
        f"mask {rid} (seeds {SEEDS[0]}..{SEEDS[-1]}, {N_LABELS} per seed)",
        fontsize=10,
        y=0.975,
    )
    _caption(fig, " ".join(captions))
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return fig, layout


def plot_discrete(env_name, rid, joint, rep, n_states, conflict_classes, captions):
    """Method columns; rows are the original and masked exact state maps."""
    shape, pos, (xlab, ylab) = semantic_layout(env_name, n_states)
    counts = {m: np.bincount(joint[m], minlength=n_states) for m in METHODS}
    class_counts = {m: np.bincount(rep[joint[m]], minlength=n_states) for m in METHODS}
    grids = {}
    for m in METHODS:
        for name, values in (
            ("original", counts[m]),
            ("masked", class_counts[m][rep]),
        ):
            grid = np.zeros(shape)
            for s in range(n_states):
                grid[pos(s)] = values[s]
            grids[(m, name)] = grid
    vmax = max(float(g.max()) for g in grids.values()) or 1.0
    norm = LogNorm(vmin=1, vmax=max(vmax, 2.0))
    panel_w = min(6.0, max(3.2, 0.2 * shape[1]))
    panel_h = min(4.5, max(2.6, 0.2 * shape[0] + 1.0))
    fig, axes = plt.subplots(
        2,
        len(METHODS),
        figsize=(panel_w * len(METHODS) + 1.4, 2 * panel_h + 2.6),
        squeeze=False,
    )
    image = None
    for c, m in enumerate(METHODS):
        for r, name in enumerate(("original", "masked")):
            ax = axes[r][c]
            grid = np.ma.masked_equal(grids[(m, name)], 0)
            image = ax.imshow(
                grid, cmap="Blues", norm=norm, aspect="auto", interpolation="nearest"
            )
            if name == "masked":
                marks = [
                    pos(s) for s in range(n_states) if rep[s] in conflict_classes[m]
                ]
                if marks:
                    ys, xs = zip(*marks)
                    ax.scatter(xs, ys, marker="x", s=12, c="#b2182b", lw=0.9)
            ax.set_xticks(range(0, shape[1], 1 if shape[1] <= 16 else 4))
            ax.set_yticks(range(0, shape[0], 1 if shape[0] <= 16 else 2))
            ax.set_xlabel(xlab, fontsize=8)
            ax.set_ylabel(ylab, fontsize=8)
            ax.tick_params(labelsize=7)
            view = "original state" if name == "original" else f"masked class ({rid})"
            ax.set_title(f"{METHOD_LABEL[m]}\n{view}", fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOR[m])
    fig.subplots_adjust(
        left=0.07, right=0.87, top=0.83, bottom=0.22, hspace=0.5, wspace=0.3
    )
    cax = fig.add_axes([0.9, 0.3, 0.015, 0.5])
    fig.colorbar(image, cax=cax, label="retained samples, all seeds (log)")
    title = (
        f"{env_name}: retained labeled-state coverage, exact states vs learner "
        f"mask {rid}. Masked cells show their class count; x marks a class "
        "with an exact expert-label conflict among one seed's samples of "
        "that method."
    )
    fig.suptitle(
        "\n".join(textwrap.wrap(title, int(fig.get_figwidth() * 11))), fontsize=9
    )
    _caption(fig, " ".join(captions))
    layout = {
        "shape": list(shape),
        "axes": [xlab, ylab],
        "color_scale": {"kind": "log", "vmin": 1, "vmax": vmax},
        "display_counts": {m: int(len(joint[m])) for m in METHODS},
    }
    return fig, layout


# ---------------------------------------------------------------------------
# Environment analysis
# ---------------------------------------------------------------------------


def _env_cells(reader, env_name, env):
    controls: Dict[str, Any] = {}
    control_cell = Cell("control", None)
    for name, pin in env["control"].items():
        raw = reader.read(control_cell, name, pin)
        if raw is not None:
            try:
                controls[name] = _json(raw, f"control {name}")
            except Ineligible as exc:
                control_cell.mark("ineligible", str(exc))
    control_cell.finish_presence()
    provenance_cell, taxi_caption = verify_expert_provenance(reader, env_name, env)
    seeds: Dict[int, Dict[str, Cell]] = {}
    samples: Dict[int, Dict[str, Sample]] = {}
    descriptions = []
    for seed in SEEDS:
        entries = env["seeds"].get(str(seed), {})
        data_cell, data = verify_data(
            reader,
            env_name,
            env,
            seed,
            entries.get("data"),
            controls,
        )
        seeds[seed] = {"data": data_cell}
        samples[seed] = {}
        for method in METHODS:
            cell, sample, description = verify_run(
                reader,
                env_name,
                env,
                seed,
                method,
                entries.get(method),
                controls,
                data,
            )
            seeds[seed][method] = cell
            if sample is not None:
                samples[seed][method] = sample
                descriptions.append(description)
    return control_cell, provenance_cell, taxi_caption, seeds, samples, descriptions


def analyze_env(reader, env_name, env, out_dir, color_by) -> Dict[str, Any]:
    """Verify one environment; render it only if every input is complete."""
    task = tasks.get_task(env_name)
    control, provenance, taxi_caption, seeds, samples, descriptions = _env_cells(
        reader,
        env_name,
        env,
    )
    reasons: List[str] = []
    mask = mask_info = None
    if descriptions:
        if any(_norm(d) != _norm(descriptions[0]) for d in descriptions):
            reasons.append("runs record different restriction descriptions")
        else:
            try:
                mask, mask_info = mask_from_description(
                    descriptions[0],
                    env["restriction_id"],
                    task,
                )
            except Ineligible as exc:
                reasons.append(f"mask: {exc}")
    cells = [control, provenance] + [c for s in seeds.values() for c in s.values()]
    complete = all(c.status == "complete" for c in cells) and not reasons
    inventory = {
        "env_name": env_name,
        "task_scope": task_scope(env_name),
        "restriction_id": env["restriction_id"],
        "status": "complete" if complete else "inventory_only",
        "reasons": reasons,
        "control": control.as_dict(),
        "expert_provenance": provenance.as_dict(),
        "seeds": {
            str(s): {k: c.as_dict() for k, c in v.items()} for s, v in seeds.items()
        },
        "counts": {status: sum(c.status == status for c in cells) for status in _RANK},
    }
    if not complete:
        return inventory
    try:
        inventory["outputs"] = render_env(
            env_name,
            env,
            task,
            samples,
            mask,
            mask_info,
            taxi_caption,
            out_dir,
            color_by,
            inventory,
        )
    except (Ineligible, ValueError) as exc:
        inventory["status"] = "inventory_only"
        inventory["reasons"].append(f"rendering refused: {exc}")
    return inventory


def render_env(
    env_name,
    env,
    task,
    samples,
    mask,
    mask_info,
    taxi_caption,
    out_dir,
    color_by,
    inventory,
):
    """Write the figure, numerical summary and sample arrays of one env."""
    names = list(task.feature_names)
    masked = {
        seed: {m: mask.apply(s.obs) for m, s in by.items()}
        for seed, by in samples.items()
    }
    per_seed = {}
    for seed in SEEDS:
        by = samples[seed]
        stats = {
            m: {
                **conflict_stats(by[m].obs, masked[seed][m], by[m].acts),
                "excluded_final_observations": by[m].excluded_final_observations,
                "origin": by[m].origin,
                "arrival": (
                    None
                    if by[m].arrival is None
                    else {
                        "first": int(by[m].arrival[0]),
                        "last": int(by[m].arrival[-1]),
                        "rule": "one-based label arrival (scratch round-000 is 1)",
                    }
                ),
            }
            for m in METHODS
        }
        stats["pooled_three_methods"] = conflict_stats(
            np.concatenate([by[m].obs for m in METHODS]),
            np.concatenate([masked[seed][m] for m in METHODS]),
            np.concatenate([by[m].acts for m in METHODS]),
        )
        stats["overlap"] = overlap_stats(by, masked[seed])
        per_seed[str(seed)] = stats
    captions = [CAPTION] + ([taxi_caption] if taxi_caption else [])
    summary: Dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "env_name": env_name,
        "task_scope": task_scope(env_name),
        "restriction_id": env["restriction_id"],
        "restriction": mask_info,
        "observation": task.observation,
        "feature_names": names,
        "seeds": list(SEEDS),
        "labels_per_seed": N_LABELS,
        "methods": list(METHODS),
        "sampled_distributions": SAMPLED_DISTRIBUTION,
        "representations": {
            "original": "the full observation the expert labeled",
            "masked": "the learner's observation after the recorded mask",
        },
        "per_seed": per_seed,
        "caption": captions,
        "taxi_expert_provenance_caption": taxi_caption,
        "color_by": color_by,
        "limitations": list(LIMITATIONS),
    }
    pooled = {m: np.concatenate([samples[s][m].obs for s in SEEDS]) for m in METHODS}
    pooled_masked = {m: np.concatenate([masked[s][m] for s in SEEDS]) for m in METHODS}
    pooled["acts"] = {
        m: np.concatenate([samples[s][m].acts for s in SEEDS]) for m in METHODS
    }
    prefix = f"coverage-{env_name}"
    if task.observation == "continuous":
        fits = {
            "original": pca(np.concatenate([pooled[m] for m in METHODS])),
            "masked": pca(np.concatenate([pooled_masked[m] for m in METHODS])),
        }
        summary["projection"] = {
            "method": "standardized PCA (numpy SVD, sign-fixed), fitted once "
            "per representation on all methods and seeds pooled",
            **fits,
        }
        summary["coordinates"] = {
            m: {
                "original": coordinate_stats(pooled[m], names),
                "masked": coordinate_stats(pooled_masked[m], names),
            }
            for m in METHODS
        }
        fig, layout = plot_continuous(
            env_name,
            env["restriction_id"],
            pooled,
            pooled_masked,
            names,
            fits,
            color_by,
            captions,
        )
        joint = None
    else:
        rep = np.asarray(mask.representative, dtype=np.int64)
        for seed in SEEDS:
            for m in METHODS:
                samples[seed][m].joint = mask.decode(samples[seed][m].obs)
        joint = {
            m: np.concatenate([samples[s][m].joint for s in SEEDS]) for m in METHODS
        }
        table = alias_table(samples, rep)
        summary["alias_table"] = table
        conflict_classes = {
            m: {
                row["masked_representative"]
                for row in table
                if any(
                    v for k, v in row["label_conflict"].items() if k.startswith(m + "/")
                )
            }
            for m in METHODS
        }
        summary["distinct_states_all_seeds"] = {
            m: {
                "original": int(len(np.unique(joint[m]))),
                "masked": int(len(np.unique(rep[joint[m]]))),
            }
            for m in METHODS
        }
        fig, layout = plot_discrete(
            env_name,
            env["restriction_id"],
            joint,
            rep,
            len(rep),
            conflict_classes,
            captions,
        )
    summary["display"] = layout
    summary["inputs"] = {
        "control": inventory["control"]["files"],
        "expert_provenance": inventory["expert_provenance"]["files"],
        "seeds": {
            s: {k: v["files"] for k, v in cells.items()}
            for s, cells in inventory["seeds"].items()
        },
    }
    png = out_dir / f"{prefix}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    arrays = {
        "method_index": np.concatenate(
            [np.full(N_LABELS, i) for _ in SEEDS for i, _m in enumerate(METHODS)]
        ),
        "seed": np.concatenate([np.full(N_LABELS, s) for s in SEEDS for _ in METHODS]),
        "arrival": np.concatenate(
            [
                (
                    np.full(N_LABELS, -1)
                    if samples[s][m].arrival is None
                    else samples[s][m].arrival
                )
                for s in SEEDS
                for m in METHODS
            ]
        ),
        "obs": np.concatenate([samples[s][m].obs for s in SEEDS for m in METHODS]),
        "masked_obs": np.concatenate([masked[s][m] for s in SEEDS for m in METHODS]),
        "expert_action": np.concatenate(
            [samples[s][m].acts for s in SEEDS for m in METHODS]
        ),
    }
    summary["samples_file"] = {
        "file": f"{prefix}-samples.npz",
        "method_names": list(METHODS),
        "arrival_rule": "-1 for offline BC (fixed pool, no arrival order); "
        "otherwise one-based label arrival",
    }
    with open(out_dir / f"{prefix}-samples.npz", "xb") as f:
        np.savez_compressed(f, **arrays)
    _write_json(out_dir / f"{prefix}-summary.json", summary)
    return [png.name, f"{prefix}-samples.npz", f"{prefix}-summary.json"]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _write_json(path: pathlib.Path, value: Any) -> None:
    with open(path, "x", encoding="utf-8") as f:
        json.dump(value, f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")


def inventory_markdown(inventory: Dict[str, Any]) -> str:
    lines = [
        "# Masked retained labeled-state coverage: inventory",
        "",
        f"Manifest SHA256 `{inventory['manifest_sha256']}`.",
        "",
        "| environment | status | complete | missing | partial | ineligible |",
        "|---|---|---|---|---|---|",
    ]
    for env_name, env in inventory["environments"].items():
        c = env.get("counts", {})
        scope = " (variant)" if env.get("task_scope") == "variant" else ""
        lines.append(
            f"| {env_name}{scope} | {env['status']} | {c.get('complete', 0)} | "
            f"{c.get('missing', 0)} | {c.get('partial', 0)} | "
            f"{c.get('ineligible', 0)} |",
        )
    lines += ["", "## Problems", ""]
    for env_name, env in inventory["environments"].items():
        for reason in env.get("reasons", []):
            lines.append(f"- {env_name}: {reason}")
        for name in ("control", "expert_provenance"):
            block = env.get(name) or {}
            for reason in block.get("reasons", []):
                lines.append(f"- {env_name} {name}: {reason}")
        for seed, cells in (env.get("seeds") or {}).items():
            for name, cell in cells.items():
                if cell["status"] != "complete" or cell["reasons"]:
                    lines.append(
                        f"- {env_name} seed {seed} {name} ({cell['status']}): "
                        + "; ".join(cell["reasons"]),
                    )
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in LIMITATIONS]
    return "\n".join(lines) + "\n"


def summary_markdown(env_name: str, summary: Dict[str, Any]) -> str:
    lines = [f"## {env_name} ({summary['restriction_id']})", ""]
    lines.append(
        "| seed | method | distinct orig | distinct masked | merged groups | "
        "conflict groups | samples in conflict | majority misfit |",
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for seed, stats in summary["per_seed"].items():
        for m in METHODS + ("pooled_three_methods",):
            s = stats[m]
            lines.append(
                f"| {seed} | {m} | {s['distinct_original']} | "
                f"{s['distinct_masked']} | {s['masked_groups_merging_originals']} | "
                f"{s['masked_groups_with_label_conflict']} | "
                f"{s['samples_in_conflicting_groups']} | "
                f"{s['majority_label_misfit_samples']} |",
            )
    lines += ["", " ".join(summary["caption"]), ""]
    return "\n".join(lines)


def analyze(manifest_path, output_dir, envs=None, color_by="method") -> int:
    """Run the analysis; see the module docstring for exit codes."""
    output_dir = pathlib.Path(output_dir)
    try:
        if output_dir.exists():
            raise Refused(f"{output_dir} exists; outputs go to a new directory only")
        manifest, manifest_sha, bundle = load_manifest(manifest_path)
        selected = list(envs) if envs else list(manifest["environments"])
        unknown = [e for e in selected if e not in known_envs()]
        if unknown:
            raise Refused(f"unknown environments {unknown}")
        index = load_indexes(bundle, manifest)
        parent = output_dir.absolute().parent
        if not parent.is_dir():
            raise Refused(f"{parent} does not exist")
        if bundle == output_dir.absolute() or bundle in output_dir.absolute().parents:
            raise Refused("the output directory must be outside the input bundle")
    except Refused as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    reader = Reader(bundle, index, manifest.get("require_index_corroboration", False))
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        inventory: Dict[str, Any] = {
            "schema": INVENTORY_SCHEMA,
            "manifest_sha256": manifest_sha,
            "provenance_indexes": manifest.get("provenance_indexes", []),
            "provenance_index_entries": len(index),
            "require_index_corroboration": reader.strict,
            "expected_environments": list(EXPECTED_ENVS),
            "selected_environments": selected,
            "variant_environments": [e for e in selected if e not in EXPECTED_ENVS],
            "environments": {},
        }
        markdown = []
        for env_name in selected:
            if env_name not in manifest["environments"]:
                inventory["environments"][env_name] = {
                    "task_scope": task_scope(env_name),
                    "status": "inventory_only",
                    "reasons": ["not in the manifest (missing)"],
                    "counts": {"missing": 1 + len(SEEDS) * (1 + len(METHODS))},
                }
                continue
            result = analyze_env(
                reader,
                env_name,
                manifest["environments"][env_name],
                staging,
                color_by,
            )
            inventory["environments"][env_name] = result
            if result["status"] == "complete":
                with open(staging / f"coverage-{env_name}-summary.json") as f:
                    markdown.append(summary_markdown(env_name, json.load(f)))
        absent = [e for e in EXPECTED_ENVS if e not in manifest["environments"]]
        inventory["expected_environments_not_in_manifest"] = absent
        _write_json(staging / "coverage-inventory.json", inventory)
        with open(staging / "coverage-inventory.md", "x") as f:
            f.write(inventory_markdown(inventory))
        if markdown:
            with open(staging / "coverage-summary.md", "x") as f:
                f.write("# Masked retained labeled-state coverage\n\n")
                f.write("\n".join(markdown))
        artifacts = {p.name: pilot.sha256_file(p) for p in sorted(staging.iterdir())}
        _write_json(
            staging / "coverage-artifacts.json",
            {"manifest_sha256": manifest_sha, "files": artifacts},
        )
        os.rename(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    done = all(inventory["environments"][e]["status"] == "complete" for e in selected)
    return EXIT_OK if done else EXIT_INVENTORY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="verify pinned inputs and render coverage")
    a.add_argument("--manifest", type=pathlib.Path, required=True)
    a.add_argument("--output-dir", type=pathlib.Path, required=True)
    a.add_argument("--env", action="append", choices=known_envs())
    a.add_argument("--color-by", choices=("method", "expert_action"), default="method")
    t = sub.add_parser("tree-digest", help="print a scratch tree's tree_sha256")
    t.add_argument("path", type=pathlib.Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "tree-digest":
        try:
            print(tree_sha256(tree_entries(args.path)))
        except Ineligible as exc:
            print(f"Refused: {exc}", file=sys.stderr)
            return EXIT_REFUSED
        return EXIT_OK
    return analyze(args.manifest, args.output_dir, args.env, args.color_by)


if __name__ == "__main__":
    sys.exit(main())
