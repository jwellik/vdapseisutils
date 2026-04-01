#!/usr/bin/env python3
"""
Minimal offline ``Map``: default or small extent, native scatter on Cartopy axes.

Run from the repository root::

    python examples/map_minimal.py

or::

    uv run python examples/map_minimal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from vdapseisutils.core.maps.map import Map


def main() -> None:
    m = Map(figsize=(5, 5), dpi=96)
    try:
        lon = np.array([-122.22, -122.18, -122.20], dtype=float)
        lat = np.array([46.18, 46.20, 46.22], dtype=float)
        m.ax.scatter(lon, lat, s=40, c="crimson", edgecolors="k", transform=ccrs.Geodetic(), zorder=5)
        m.figure.canvas.draw()
    finally:
        plt.close(m.figure)


if __name__ == "__main__":
    main()
