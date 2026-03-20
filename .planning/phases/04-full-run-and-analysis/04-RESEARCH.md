# Phase 4: Full Run and Analysis - Research

**Researched:** 2026-03-20
**Domain:** Scientific data analysis, publication-quality visualization, Sacred output parsing
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
None — no locked decisions from discussion.

### Claude's Discretion
All implementation choices are at Claude's discretion. Requirements are fully specified by REQUIREMENTS.md (EVAL-01 through EVAL-07). Key constraint: figures MUST support incremental generation from partial results (user requirement for boss updates every 2-3 days).

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-01 | Normalized scores `(agent - random) / (expert - random)` for each game | `compute_normalized_score` in `atari_helpers.py` already implemented; random baselines cached in `baselines/atari_random_scores.pkl`; expert scores loaded fresh per game |
| EVAL-02 | Learning curves: normalized return vs environment interactions per game | Sacred `metrics.json` contains `normalized_score` per round with `steps` field; round-to-timesteps conversion via `total_timesteps / n_rounds`; matplotlib `fill_between` for seed variance bands |
| EVAL-03 | Aggregate metrics: mean and IQM with 95% CI across all 7 games | `rliable` library provides `metrics.aggregate_iqm`, `rly.get_interval_estimates` with stratified bootstrap; score array shape `(num_runs, num_games)` |
| EVAL-04 | Figure generation works on partial results (incremental) | Sacred `run.json` status field (`COMPLETED`, `RUNNING`, `FAILED`, `INTERRUPTED`) enables filtering; analysis script runs on any available subset |
| EVAL-05 | Completion dashboard: done/running/pending per (algo, game, seed) | Walk `output/sacred/{algo}/{game}/{seed}/*/run.json`; cross-check expected 63 combinations (3 algos x 7 games x 3 seeds); terminal table via `rich` or `tabulate` |
| EVAL-06 | Publication-quality figures comparing FTL, FTRL, BC, Expert | matplotlib PDF output with rcParams for font sizes; per-game subplot grid (7 games); separate aggregate figure; Expert shown as horizontal reference line |
| EVAL-07 | Consistent colors/line styles across methods | Fixed color dict and linestyle dict keyed by algorithm name; defined once in a shared `plot_config.py` module |
</phase_requirements>

## Summary

Phase 4 produces a Python analysis script (`experiments/analyze_results.py`) that reads Sacred FileStorageObserver output and generates three types of figures: per-game learning curves, an aggregate bar chart (mean + IQM with 95% CI), and a completion dashboard. All figures must work on partial Sacred output, so the script is designed to run at any point mid-benchmark.

The Sacred output structure is fully known from Phase 3: each run writes to `output/sacred/{TIMESTAMP}/{algo}/{game}/{seed}/{run_id}/` with `run.json` (status), `config.json` (hyperparameters), and `metrics.json` (per-round scalars including `normalized_score`). The `run.json` status field distinguishes `COMPLETED`, `RUNNING`, `FAILED`, and `INTERRUPTED` runs, enabling incremental filtering without code changes.

For aggregate statistics, `rliable` (v1.2.0) provides IQM and stratified bootstrap confidence intervals — the standard approach in recent RL papers. Per-game learning curves use matplotlib with `fill_between` for seed variance bands. Consistent colors and line styles are defined once in a shared config and applied everywhere.

**Primary recommendation:** Implement a single `experiments/analyze_results.py` with three subcommands: `dashboard`, `curves`, and `aggregate`. Each subcommand reads whatever Sacred output exists and generates figures gracefully with missing data.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib | 3.7.5 (installed) | Figure generation, PDF/PNG output | Already in project environment |
| rliable | 1.2.0 | IQM + stratified bootstrap CI | NeurIPS 2021 gold standard for RL benchmarks |
| numpy | 1.24.4 (installed) | Score array manipulation | Already in project environment |
| sacred | 0.8.7 (installed) | Output format we read | Already used in Phase 3 |
| seaborn | 0.13.2 (installed) | Color palettes | Already in project environment |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | 1.4.3 (installed) | Tabular results for dashboard | Optional — can use plain dicts; useful for formatting |
| scipy | 1.9.0 (installed) | Fallback bootstrap if rliable has issues | Only if rliable unavailable |
| tabulate | latest | Terminal dashboard formatting | Simple text table for completion dashboard |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| rliable IQM | manual numpy IQM | rliable handles edge cases (few seeds, NaN, stratified bootstrap); hand-rolled is fragile |
| matplotlib | seaborn | seaborn wraps matplotlib; unnecessary abstraction for this use case |
| rliable plot_sample_efficiency_curve | manual learning curve plot | rliable's function requires specific data layout; custom matplotlib gives more control for per-game subplots |

**Installation:**
```bash
pip install rliable==1.2.0
# All other dependencies are already installed
```

**Version verification:** rliable 1.2.0 verified from PyPI (2026-03-20). matplotlib 3.7.5, numpy 1.24.4, seaborn 0.13.2, pandas 1.4.3 confirmed installed in project environment.

## Architecture Patterns

### Recommended Project Structure
```
experiments/
├── analyze_results.py       # Main CLI: dashboard / curves / aggregate subcommands
├── plot_config.py           # Shared COLOR_MAP, LINESTYLE_MAP, RCPARAMS (EVAL-07)
├── atari_helpers.py         # Already exists: compute_normalized_score, ATARI_GAMES
├── atari_smoke.py           # Already exists: load_random_baselines
figures/
├── learning_curves/         # Per-game PDF/PNG (EVAL-02, EVAL-06)
│   ├── Pong.pdf
│   └── ...
├── aggregate.pdf            # Mean + IQM figure (EVAL-03, EVAL-06)
└── dashboard.txt            # Completion table (EVAL-05)
```

### Pattern 1: Sacred Output Reader
**What:** Walk the output directory tree, parse `run.json` for status and `metrics.json` for scalars. Return a dict keyed by `(algo, game, seed)`.
**When to use:** Entry point for all three analysis modes.
**Example:**
```python
# Sacred FileStorageObserver creates: {obs_path}/{run_id}/
# where obs_path = output/sacred/{TIMESTAMP}/{algo}/{game}/{seed}
# metrics.json format (verified from IDSIA/sacred source):
# {
#   "normalized_score": {
#     "values": [0.1, 0.2, ...],    # one per round
#     "steps":  [0, 1, ...],        # round index
#     "timestamps": ["2026-...", ...]
#   }
# }
import json
from pathlib import Path

def load_run(run_dir: Path) -> dict | None:
    """Load a single Sacred run. Returns None if run is not COMPLETED."""
    run_json = run_dir / "run.json"
    metrics_json = run_dir / "metrics.json"
    if not run_json.exists():
        return None
    with open(run_json) as f:
        run_info = json.load(f)
    status = run_info.get("status", "UNKNOWN")
    # For incremental generation, include RUNNING runs with partial data
    if status not in ("COMPLETED", "RUNNING"):
        return None
    metrics = {}
    if metrics_json.exists():
        with open(metrics_json) as f:
            metrics = json.load(f)
    return {"status": status, "metrics": metrics, "config": run_info}
```

### Pattern 2: Score Matrix Assembly for rliable
**What:** Build `(num_runs, num_games)` matrix from collected results for aggregate metrics.
**When to use:** EVAL-03 — aggregate figure with IQM and mean.
**Example:**
```python
import numpy as np
from rliable import library as rly
from rliable import metrics

# scores_dict maps algo -> (num_seeds, num_games) array
# Use final normalized_score from each completed run
# Missing runs: use NaN, then mask before passing to rliable
scores_dict = {
    "DAgger": np.array([[...], [...], [...]]),  # shape (3 seeds, 7 games)
    "FTRL":   np.array([[...], [...], [...]]),
    "BC":     np.array([[...], [...], [...]]),
}

aggregate_func = lambda x: np.array([
    metrics.aggregate_mean(x),
    metrics.aggregate_iqm(x),
])
agg_scores, agg_cis = rly.get_interval_estimates(
    scores_dict, aggregate_func, reps=50000
)
```

### Pattern 3: Incremental Learning Curve from Partial Sacred Output
**What:** Interpolate per-round normalized scores onto a common round grid; plot mean and std band across seeds.
**When to use:** EVAL-02, EVAL-04 — learning curves runnable at any point mid-run.
**Example:**
```python
# BC has only round 0; DAgger/FTRL have rounds 0..N
# Collect scores per round for each seed; where a seed is still running,
# use only completed rounds (partial data is valid)
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(ax, algo_name, rounds_list, scores_list, color, linestyle):
    """Plot mean ± std band across seeds over rounds."""
    max_rounds = max(len(r) for r in rounds_list)
    common_rounds = np.arange(max_rounds)
    # Interpolate each seed's scores onto common_rounds
    interpolated = []
    for rounds, scores in zip(rounds_list, scores_list):
        interpolated.append(np.interp(common_rounds, rounds, scores))
    arr = np.array(interpolated)  # (n_seeds, max_rounds)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    ax.plot(common_rounds, mean, label=algo_name, color=color, linestyle=linestyle)
    ax.fill_between(common_rounds, mean - std, mean + std, alpha=0.2, color=color)
```

### Pattern 4: Completion Dashboard
**What:** Print a table showing status of all 63 expected (algo, game, seed) combinations.
**When to use:** EVAL-05 — quick check of benchmark progress.
**Example:**
```python
# Expected: 3 algos x 7 games x 3 seeds = 63 combinations
EXPECTED = [(a, g, s) for a in ALGOS for g in GAMES for s in SEEDS]

def make_dashboard(results: dict) -> str:
    rows = []
    for algo, game, seed in EXPECTED:
        key = (algo, game, seed)
        if key not in results:
            status = "PENDING"
        else:
            status = results[key].get("status", "UNKNOWN")
        rows.append((algo, game, seed, status))
    # Format as table; count completed/running/pending
    ...
```

### Anti-Patterns to Avoid
- **Loading all Sacred output into memory at once:** With 63 runs x 20 rounds, stay lazy — load per-run and discard.
- **Crashing on missing runs:** The script MUST handle missing directories, missing metrics.json, and NaN scores without raising exceptions (EVAL-04 requires incremental generation).
- **Hardcoding the Sacred run ID (e.g. "1"):** Each `obs_path` may contain only one run directory but its name is `1`. Use `glob(obs_path + "/*/run.json")` to find it robustly.
- **Calling rliable with NaN:** rliable does not handle NaN gracefully. Replace missing game scores with 0.0 and document clearly; or use only fully-populated columns.
- **Different color per figure file:** Define colors once in `plot_config.py`, import everywhere (EVAL-07).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| IQM computation | Custom trimmed mean | `rliable.metrics.aggregate_iqm` | Handles edge cases; stratified bootstrap CI is non-trivial |
| Bootstrap 95% CI | Manual percentile bootstrap | `rliable.library.get_interval_estimates` | Stratified bootstrap ensures all games represented; 50000 reps needed for stable CI |
| Sacred output parsing | Custom JSON walker | Standard `json.load` + `pathlib.glob` | Sacred format is simple; no framework needed |
| PDF figure saving | Custom rendering | `fig.savefig("out.pdf", bbox_inches="tight", dpi=300)` | matplotlib handles this natively |

**Key insight:** The only non-trivial statistics are IQM + stratified bootstrap CI. Everything else (loading JSON, plotting, dashboard) is straightforward Python. Use rliable for the statistics, matplotlib for the figures, and plain pathlib for file walking.

## Common Pitfalls

### Pitfall 1: Sacred Run ID Directory
**What goes wrong:** Code hardcodes `obs_path/1/metrics.json` — fails if Sacred assigns a different run ID.
**Why it happens:** Sacred assigns sequential integer IDs starting at 1 per observer directory. Since each `obs_path` is unique per `(algo, game, seed)`, ID=1 is almost always correct — but a failed+retried run would produce ID=2.
**How to avoid:** Use `sorted(Path(obs_path).glob("*/metrics.json"))[-1]` to get the latest run in the directory.
**Warning signs:** `FileNotFoundError` on `metrics.json` path.

### Pitfall 2: BC Has Only One Round Entry
**What goes wrong:** BC logs `normalized_score` at step=0 only (established in Phase 3 decision). Learning curve code that expects `n_rounds` entries will see only one point.
**Why it happens:** BC is a single-round algorithm; Sacred `log_scalar` with `step=0` is only called once.
**How to avoid:** In learning curve plotting, treat BC as a horizontal line at its single score. Do not interpolate or compute variance across rounds.
**Warning signs:** Learning curve for BC appears empty or crashes on empty arrays.

### Pitfall 3: Environment Interactions vs Rounds
**What goes wrong:** X-axis labeled "environment interactions" but `metrics.json` steps are round numbers (0, 1, 2...).
**Why it happens:** Sacred logs step as DAgger round number, not timestep count.
**How to avoid:** Convert: `timestep = round_num * (total_timesteps / n_rounds)`. The `config.json` in each run directory contains `total_timesteps` and `n_rounds`. Read them from config for exact conversion.
**Warning signs:** X-axis scale looks wrong relative to Lavington et al. figures.

### Pitfall 4: rliable Score Matrix Shape
**What goes wrong:** rliable throws an error or returns wrong CI because score matrix has wrong shape.
**Why it happens:** `get_interval_estimates` expects `(num_runs, num_games)` — but if you pass `(num_games, num_runs)` or a 1D array, it silently produces wrong results.
**How to avoid:** Assert shape before calling: `assert scores.shape == (len(SEEDS), len(ATARI_GAMES))`.
**Warning signs:** CIs are unreasonably narrow or wide; `aggregate_iqm` returns unexpected values.

### Pitfall 5: Missing Runs in Aggregate Figure
**What goes wrong:** Aggregate figure silently uses 0.0 for missing runs, making incomplete results look worse than they are.
**Why it happens:** Incremental generation requires handling missing data.
**How to avoid:** Count how many runs are included in the figure title or caption annotation: e.g., "Aggregate (42/63 runs complete)". Mark missing entries explicitly.
**Warning signs:** Bar chart heights seem artificially low for an algorithm with many pending runs.

### Pitfall 6: Timestamp vs Step in metrics.json
**What goes wrong:** Code iterates over `metric["timestamps"]` instead of `metric["steps"]` for round numbers.
**Why it happens:** Both arrays are present; `timestamps` are ISO datetime strings (not useful for x-axis).
**How to avoid:** Always use `metric["steps"]` for the x-axis round index.

### Pitfall 7: Output Directory Timestamp
**What goes wrong:** Analysis script hardcodes `output/sacred/` but the benchmark script writes to `output/sacred/{TIMESTAMP}/`.
**Why it happens:** Phase 3 benchmark adds a timestamp prefix to avoid overwriting runs.
**How to avoid:** Accept `--output-dir` argument that defaults to the most recently modified `output/sacred/*/` directory. Or let user pass the explicit path.
**Warning signs:** Script finds zero runs.

## Code Examples

Verified patterns from project code and official sources:

### Sacred metrics.json Loading
```python
# Source: IDSIA/sacred source (file_storage.py) + Sacred 0.8.4 docs
import json
from pathlib import Path

def load_sacred_metrics(obs_path: str | Path) -> dict:
    """Load metrics from Sacred FileStorageObserver directory.

    Args:
        obs_path: Path like output/sacred/{TIMESTAMP}/{algo}/{game}/{seed}
                  (the observer base, not the run subdirectory)

    Returns:
        Dict with keys: status, config, normalized_scores (list), rounds (list)
        Returns empty dict if no completed/running run found.
    """
    obs_path = Path(obs_path)
    # Find the latest run subdirectory (Sacred assigns sequential integer IDs)
    run_dirs = sorted(obs_path.glob("*/run.json"), key=lambda p: int(p.parent.name))
    if not run_dirs:
        return {}
    run_dir = run_dirs[-1].parent

    with open(run_dir / "run.json") as f:
        run_info = json.load(f)
    status = run_info.get("status", "UNKNOWN")
    if status not in ("COMPLETED", "RUNNING"):
        return {"status": status}

    with open(run_dir / "config.json") as f:
        config = json.load(f)

    metrics = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    # metrics["normalized_score"] = {"values": [...], "steps": [...], "timestamps": [...]}
    ns = metrics.get("normalized_score", {})
    return {
        "status": status,
        "config": config,
        "rounds": ns.get("steps", []),
        "normalized_scores": ns.get("values", []),
    }
```

### rliable IQM + Mean Aggregate Figure
```python
# Source: google-research/rliable README (verified 2026-03-20)
import numpy as np
from rliable import library as rly
from rliable import metrics, plot_utils

# Assemble score matrix: shape (n_seeds, n_games)
# Use final normalized_score from each completed run
def build_score_matrix(results, algo, games, seeds):
    """Build (n_seeds, n_games) score matrix; NaN for missing runs."""
    mat = np.full((len(seeds), len(games)), np.nan)
    for j, game in enumerate(games):
        for i, seed in enumerate(seeds):
            r = results.get((algo, game, seed))
            if r and r.get("normalized_scores"):
                mat[i, j] = r["normalized_scores"][-1]  # final round score
    return mat

scores_dict = {
    "DAgger": build_score_matrix(results, "dagger", GAMES, SEEDS),
    "FTRL":   build_score_matrix(results, "ftrl",   GAMES, SEEDS),
    "BC":     build_score_matrix(results, "bc",     GAMES, SEEDS),
}

# Replace NaN with 0.0 for rliable (document how many are missing)
for k in scores_dict:
    scores_dict[k] = np.nan_to_num(scores_dict[k], nan=0.0)

aggregate_func = lambda x: np.array([
    metrics.aggregate_mean(x),
    metrics.aggregate_iqm(x),
])
agg_scores, agg_cis = rly.get_interval_estimates(
    scores_dict, aggregate_func, reps=50000
)

fig, axes = plot_utils.plot_interval_estimates(
    agg_scores,
    agg_cis,
    metric_names=["Mean", "IQM"],
    algorithms=list(scores_dict.keys()),
    xlabel="Normalized Score",
)
fig.savefig("figures/aggregate.pdf", bbox_inches="tight", dpi=300)
```

### Publication-Quality rcParams
```python
# Source: matplotlib 3.10 docs (customizing.html)
import matplotlib.pyplot as plt
import matplotlib as mpl

RCPARAMS = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,         # screen preview
    "savefig.dpi": 300,        # saved PDF/PNG
    "savefig.format": "pdf",
    "pdf.fonttype": 42,        # embed fonts (required by most journals)
    "ps.fonttype": 42,
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
}

plt.rcParams.update(RCPARAMS)
```

### Consistent Color and Line Style Config (EVAL-07)
```python
# experiments/plot_config.py
import seaborn as sns

ALGORITHMS = ["BC", "DAgger", "FTRL", "Expert"]
_palette = sns.color_palette("colorblind", n_colors=4)

COLOR_MAP = dict(zip(ALGORITHMS, _palette))
LINESTYLE_MAP = {
    "BC":     "-",
    "DAgger": "--",
    "FTRL":   "-.",
    "Expert": ":",
}
LINEWIDTH_MAP = {
    "BC": 1.5, "DAgger": 2.0, "FTRL": 2.0, "Expert": 1.0,
}
```

### Completion Dashboard
```python
# Terminal table of (algo, game, seed) → status
from pathlib import Path
import json

ALGOS  = ["bc", "dagger", "ftrl"]
GAMES  = ["Pong", "Breakout", "BeamRider", "Enduro", "Qbert", "Seaquest", "SpaceInvaders"]
SEEDS  = [0, 1, 2]

def make_dashboard(output_dir: Path) -> None:
    done = running = pending = failed = 0
    rows = []
    for algo in ALGOS:
        for game in GAMES:
            for seed in SEEDS:
                obs = output_dir / algo / game / str(seed)
                run_jsons = sorted(obs.glob("*/run.json"))
                if not run_jsons:
                    status = "PENDING"
                    pending += 1
                else:
                    with open(run_jsons[-1]) as f:
                        status = json.load(f).get("status", "UNKNOWN")
                    if status == "COMPLETED": done += 1
                    elif status == "RUNNING":  running += 1
                    elif status in ("FAILED","INTERRUPTED"): failed += 1
                    else: pending += 1
                rows.append((algo, game, seed, status))
    # Print summary + table
    print(f"Done: {done}  Running: {running}  Failed: {failed}  Pending: {pending} / 63")
    # Use tabulate or manual formatting
    ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Median across games | IQM (interquartile mean) | NeurIPS 2021 (Agarwal et al.) | Less sensitive to outlier games; smaller CI with few seeds |
| Standard error (SEM) | Stratified bootstrap CI | NeurIPS 2021 | All games guaranteed representation in CI estimate |
| Point estimate at final step | Full learning curve with CI band | ~2021 onward | Reveals training stability and convergence speed |
| PNG at 72dpi | PDF (vector) + PNG at 300dpi | Continuous | PDF required for infinite-zoom in online readers |

**Deprecated/outdated:**
- Mean normalized score with SEM: replaced by IQM + stratified bootstrap. Still worth including as a secondary metric.
- Plotting raw episode returns without normalization: meaningless across games with different score scales.

## Open Questions

1. **Expert scores for normalization**
   - What we know: `run_bc` and `run_dagger` in `atari_smoke.py` compute `expert_score` live by rolling out the HuggingFace expert (10 episodes). This is done per-run, not cached globally.
   - What's unclear: Should expert scores be recomputed per analysis run (slow, requires GPU), or cached once per game?
   - Recommendation: Cache expert scores to `experiments/baselines/atari_expert_scores.pkl` in a one-time collection script (`collect_expert_scores.py`). Analysis script reads from cache. Falls back to Sacred `config` if cached value matches.

2. **Round-to-timestep conversion for x-axis**
   - What we know: Sacred logs `normalized_score` with `step = round_num`. The `config.json` contains `total_timesteps` and `n_rounds`.
   - What's unclear: Is `timesteps_per_round` constant (total_timesteps / n_rounds) or does it vary due to DAgger's interaction with environment?
   - Recommendation: Use `total_timesteps / n_rounds` as an approximation for the x-axis label "Environment Interactions". This is sufficient for comparison purposes. Note the approximation in figure caption.

3. **Lavington et al. Figure 4 exact style**
   - What we know: Lavington et al. arXiv 2208.00088 shows aggregate comparison across Atari games. Figure 4 is referenced as the style target.
   - What's unclear: We could not access the full PDF to verify exact visual style (bar chart vs. interval estimate, exact CI width).
   - Recommendation: Use `rliable`'s `plot_interval_estimates` which produces dot-with-CI-bar plots — the standard in recent RL benchmarking papers. This is consistent with how Agarwal et al. present IQM results which Lavington et al. follow.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (installed in project) |
| Config file | `setup.cfg` `[tool:pytest]` section |
| Quick run command | `pytest tests/test_analysis.py -x -q` |
| Full suite command | `pytest tests/test_analysis.py -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | `compute_normalized_score` returns correct value | unit | `pytest tests/test_analysis.py::test_normalized_score -x` | Wave 0 |
| EVAL-02 | Learning curve plot generated from partial data | unit | `pytest tests/test_analysis.py::test_learning_curve_partial -x` | Wave 0 |
| EVAL-03 | IQM + CI computed from score matrix | unit | `pytest tests/test_analysis.py::test_aggregate_metrics -x` | Wave 0 |
| EVAL-04 | Analysis script runs with 0, partial, and full Sacred output | integration | `pytest tests/test_analysis.py::test_incremental_generation -x` | Wave 0 |
| EVAL-05 | Dashboard counts correct done/running/pending | unit | `pytest tests/test_analysis.py::test_dashboard -x` | Wave 0 |
| EVAL-06 | Figures saved as PDF and PNG | unit | `pytest tests/test_analysis.py::test_figure_output_format -x` | Wave 0 |
| EVAL-07 | Consistent colors across figures | unit | `pytest tests/test_analysis.py::test_consistent_colors -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_analysis.py -x -q`
- **Per wave merge:** `pytest tests/test_analysis.py -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_analysis.py` — covers EVAL-01 through EVAL-07 (does not exist yet)
- [ ] `tests/fixtures/mock_sacred_output/` — mock Sacred directory structure for tests

*(Existing `tests/conftest.py` provides shared fixtures but no analysis-specific fixtures)*

## Sources

### Primary (HIGH confidence)
- IDSIA/sacred GitHub `file_storage.py` — exact metrics.json format with `values`, `steps`, `timestamps` arrays
- Sacred 0.8.4 readthedocs observers.html — run.json status values: QUEUED, RUNNING, COMPLETED, FAILED, INTERRUPTED
- google-research/rliable GitHub README — score array shape `(num_runs, num_games)`, API signatures for `get_interval_estimates`, `aggregate_iqm`, `plot_interval_estimates`, `plot_sample_efficiency_curve`
- matplotlib 3.10 docs (customizing.html) — rcParams for publication quality, PDF font embedding
- Project source: `experiments/atari_helpers.py` — `compute_normalized_score`, `ATARI_GAMES`
- Project source: `experiments/atari_smoke.py` — `load_random_baselines`
- Project source: `experiments/run_atari_experiment.py` — Sacred logging pattern, observer path structure

### Secondary (MEDIUM confidence)
- Agarwal et al. NeurIPS 2021 "Deep RL at the Statistical Precipice" — IQM definition, stratified bootstrap rationale (via rliable README)
- arXiv 2208.00088 (Lavington et al.) — confirmed paper exists; Figure 4 style reference not directly verified from PDF

### Tertiary (LOW confidence)
- WebSearch results on matplotlib publication-quality tips — consistent with official docs but from blog posts

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages installed and version-verified in project environment
- Sacred output format: HIGH — verified from IDSIA/sacred source code
- rliable API: HIGH — verified from official README
- Architecture patterns: HIGH — derived directly from existing project code
- Pitfalls: HIGH — derived from known Phase 3 decisions and Sacred behavior

**Research date:** 2026-03-20
**Valid until:** 2026-09-20 (stable libraries; rliable API changes infrequently)
