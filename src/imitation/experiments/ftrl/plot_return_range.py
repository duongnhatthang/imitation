"""Choose a linear return view from the later part of each learning curve."""

import numpy as np

RANGE_RULE = "second-half mean and uncertainty bands; 10% padding; minimum span 0.10"


def automatic_return_limits(series):
    """Enclose later curve bands without anchoring the view to expert return.

    Each entry holds annotation budgets, lower band edges, and upper band
    edges. A fixed baseline has a single budget. Early learning remains on
    the full x axis and may extend beyond this y view.
    """
    bounds = []
    for budgets, lower, upper in series:
        budgets = np.asarray(budgets, dtype=float)
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        valid = np.isfinite(budgets) & np.isfinite(lower) & np.isfinite(upper)
        if not valid.any():
            continue
        later = valid & (budgets >= 0.5 * budgets[valid].max())
        bounds.extend(lower[later])
        bounds.extend(upper[later])
    if not bounds:
        return (-0.5, 1.5)
    lo, hi = float(min(bounds)), float(max(bounds))
    span = max(hi - lo, 0.10)
    center = 0.5 * (lo + hi)
    return center - 0.6 * span, center + 0.6 * span
