"""
Shared helpers for Bokeh histogram-style heatmaps (palettes, color bar cleanup).

Author: Jay Wellik
"""

from __future__ import annotations

from typing import Any

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

