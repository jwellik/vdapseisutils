#!/usr/bin/env python3
"""
``VolcanoFigure`` layout: map + cross-sections + time series + magnitude legend.

Uses synthetic lon/lat/depth/time only (no FDSN, no map tiles).

Run from the repository root::

    python examples/volcano_figure_layout.py

or::

    uv run python examples/volcano_figure_layout.py
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
from obspy import UTCDateTime

from vdapseisutils.core.maps.legends import MagLegend
from vdapseisutils.core.maps.volcano_figure import VolcanoFigure


def main() -> None:
    vf = VolcanoFigure(figsize=(8, 8), dpi=96, hillshade=False)
    try:
        assert vf.map_obj is not None and vf.ts_obj is not None
        assert vf.xs1_obj is not None and vf.xs2_obj is not None

        n = 24
        rng = np.random.default_rng(0)
        lon = -122.20 + 0.04 * rng.standard_normal(n)
        lat = 46.20 + 0.03 * rng.standard_normal(n)
        dep_km = -2.0 - 8.0 * rng.random(n)
        t0 = UTCDateTime("2020-01-01T00:00:00")
        times = [t0 + float(i) * 3600.0 for i in range(n)]

        # Compose panels explicitly: ``Map.scatter`` uses (lat, lon, size, color); cross-sections
        # and time series use matplotlib-style ``s`` / ``c`` (``VolcanoFigure.scatter`` mixes both).
        vf.map_obj.scatter(list(lat), list(lon), 24, "0.2", transform=ccrs.Geodetic(), alpha=0.8)
        for xs in (vf.xs1_obj, vf.xs2_obj):
            xs.scatter(
                lat=list(lat),
                lon=list(lon),
                z=list(dep_km),
                z_dir="depth",
                z_unit="km",
                s=28,
                c="tab:blue",
                alpha=0.85,
                edgecolors="k",
                linewidths=0.3,
            )
        vf.ts_obj.scatter(times, list(dep_km), s=30, c="tab:orange", alpha=0.85, edgecolors="k", linewidths=0.3)

        leg_ax = vf.fig_leg.add_subplot(111)
        MagLegend().display(ax=leg_ax, include_counts=False)

        vf.canvas.draw()
    finally:
        plt.close(vf)


if __name__ == "__main__":
    main()
