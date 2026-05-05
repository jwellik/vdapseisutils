"""
Shared heatmap helpers that are backend-agnostic (NumPy only).
"""

from __future__ import annotations

import numpy as np


def heatmap_bin_step(extent_short: float, requested: float, floor: float, max_bins: int) -> float:
    """
    Choose bin width without forcing huge cells from extent-proportional floors.

    ``requested`` is honored when it already yields a reasonable number of bins; otherwise
    the step is clamped to ``[extent/max_bins, extent/2]``.
    """
    if extent_short <= 0 or not np.isfinite(extent_short):
        return max(float(requested), floor)
    req = max(float(requested), floor)
    min_step = max(extent_short / max_bins, floor)
    max_step = extent_short / 2.0
    return float(min(max(req, min_step), max_step))

