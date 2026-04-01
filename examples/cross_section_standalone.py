#!/usr/bin/env python3
"""
Standalone ``CrossSection`` (origin + azimuth + radius), no ``VolcanoFigure``.

Elevation along the line may be skipped if the profile download fails offline.

Run from the repository root::

    python examples/cross_section_standalone.py

or::

    uv run python examples/cross_section_standalone.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from vdapseisutils.core.maps.cross_section import CrossSection


def main() -> None:
    xs = CrossSection(
        origin=(46.20, -122.25),
        azimuth=90.0,
        radius_km=8.0,
        depth_extent=(-12.0, 2.0),
        figsize=(5, 3),
        dpi=96,
    )
    try:
        # Synthetic hypocenters along the section (horizontal distance km, depth km)
        hd = np.linspace(2.0, 12.0, 8)
        dep = -3.0 - 0.4 * hd
        xs.ax.scatter(hd, dep, s=35, c="tab:blue", edgecolors="k", alpha=0.85, zorder=4)
        xs.figure.canvas.draw()
    finally:
        plt.close(xs.figure)


if __name__ == "__main__":
    main()
