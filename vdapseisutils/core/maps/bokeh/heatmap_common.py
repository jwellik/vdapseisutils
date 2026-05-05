"""
Shared helpers for Bokeh histogram-style heatmaps (bin sizing, palettes, color bar cleanup).

Author: Jay Wellik
"""

from __future__ import annotations

from typing import Any

import numpy as np
from bokeh.palettes import Cividis256, Inferno256, Magma256, Plasma256, Viridis256


def palette_for_heatmap_cmap(cmap_name: Any) -> list[str]:
    """Map matplotlib-style cmap names to Bokeh palette sequences."""
    key = str(cmap_name or "viridis").lower()
    rev = key.endswith("_r")
    base = key[:-2] if rev else key
    palettes = {
        "viridis": Viridis256,
        "plasma": Plasma256,
        "inferno": Inferno256,
        "magma": Magma256,
        "cividis": Cividis256,
    }
    pal = palettes.get(base, Viridis256)
    seq = list(pal)
    if rev:
        seq.reverse()
    return seq


def heatmap_bin_step(extent_short: float, requested: float, floor: float, max_bins: int) -> float:
    """
    Choose bin width without forcing huge cells (legacy ``extent * 0.1`` floors).

    ``requested`` is honored when it already yields a reasonable number of bins; otherwise
    the step is clamped to ``[extent/max_bins, extent/2]``.
    """
    if extent_short <= 0 or not np.isfinite(extent_short):
        return max(float(requested), floor)
    req = max(float(requested), floor)
    min_step = max(extent_short / max_bins, floor)
    max_step = extent_short / 2.0
    return float(min(max(req, min_step), max_step))


def remove_layout_annotation(fig: Any, annotation: Any) -> None:
    """Remove an annotation from the first figure side panel that contains it."""
    if annotation is None:
        return
    for side in ("right", "left", "below", "above"):
        panel = getattr(fig, side, None)
        if panel is None:
            continue
        try:
            if annotation in panel:
                panel.remove(annotation)
                return
        except (ValueError, TypeError):
            continue

