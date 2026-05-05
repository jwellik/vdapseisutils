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


def histogram_bin_edges(lo: float, hi: float, step: float) -> np.ndarray:
    """
    Build histogram edges that nominally space ``step`` apart and cover ``[lo, hi]``.

    ``numpy.arange(lo, hi, step)`` can omit ``hi`` when ``hi - lo`` is not a multiple of
    ``step``, dropping samples from ``histogram2d``.
    """
    step = float(step)
    lo, hi = float(lo), float(hi)
    if hi < lo:
        lo, hi = hi, lo
    if step <= 0:
        raise ValueError("step must be positive")
    if hi <= lo:
        return np.array([lo, hi], dtype=float)
    edges = np.arange(lo, hi + step, step, dtype=float)
    if edges[-1] < hi:
        edges = np.append(edges, hi)
    if edges.size < 2:
        return np.array([lo, hi], dtype=float)
    return edges

