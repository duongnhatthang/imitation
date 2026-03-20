"""Shared plot configuration for consistent figure styling across all analysis figures.

Defines COLOR_MAP, LINESTYLE_MAP, LINEWIDTH_MAP, RCPARAMS, and ALGORITHMS
for use in analyze_results.py and any future figure generation scripts.
All figures MUST import from this module to ensure EVAL-07 compliance.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path (same pattern as atari_smoke.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Algorithm lists
# ---------------------------------------------------------------------------

# Display names (used as keys in COLOR_MAP, LINESTYLE_MAP, LINEWIDTH_MAP).
# "Expert" is a reference line, not a trained algorithm.
ALGORITHMS = ["BC", "DAgger", "FTRL", "Expert"]

# Sacred config keys (what the experiment scripts write to run.json/config.json).
# No "Expert" here — it is not a trained algorithm, just a reference level.
ALGO_KEYS = ["bc", "dagger", "ftrl"]

# Seeds used in the benchmark (3 seeds per combo).
SEEDS = [0, 1, 2]

# Mapping from Sacred config key -> display name for axis labels and legends.
ALGO_DISPLAY_NAMES = {
    "bc": "BC",
    "dagger": "DAgger",
    "ftrl": "FTRL",
}

# ---------------------------------------------------------------------------
# Color, line style, line width maps (EVAL-07)
# ---------------------------------------------------------------------------

# Use seaborn "colorblind" palette — visually distinct and accessible.
_palette = sns.color_palette("colorblind", n_colors=4)

COLOR_MAP = dict(zip(ALGORITHMS, _palette))
# COLOR_MAP = {"BC": ..., "DAgger": ..., "FTRL": ..., "Expert": ...}

LINESTYLE_MAP = {
    "BC": "-",
    "DAgger": "--",
    "FTRL": "-.",
    "Expert": ":",
}

LINEWIDTH_MAP = {
    "BC": 1.5,
    "DAgger": 2.0,
    "FTRL": 2.0,
    "Expert": 1.0,
}

# ---------------------------------------------------------------------------
# Publication-quality rcParams (EVAL-06)
# ---------------------------------------------------------------------------

RCPARAMS = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,          # screen preview quality
    "savefig.dpi": 300,         # saved PDF/PNG resolution
    "savefig.format": "pdf",
    "pdf.fonttype": 42,         # embed fonts (required by most journals)
    "ps.fonttype": 42,
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def apply_rcparams() -> None:
    """Apply RCPARAMS to the current matplotlib rcParams.

    Call this once at the top of any figure generation script or module
    to ensure all figures use consistent styling.
    """
    plt.rcParams.update(RCPARAMS)
