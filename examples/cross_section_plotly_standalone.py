#!/usr/bin/env python3
"""
Standalone :class:`~vdapseisutils.core.maps.plotly.cross_section_plotly.CrossSectionPlotly`
(origin + azimuth + radius), no ``VolcanoFigure``.

Mirrors ``examples/cross_section_standalone.py`` inputs using the Plotly entry point.
Elevation along the line may be skipped if the profile download fails offline.

Run from the repository root::

    python examples/cross_section_plotly_standalone.py

or::

    uv run python examples/cross_section_plotly_standalone.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from vdapseisutils.core.maps.plotly.cross_section_plotly import CrossSectionPlotly


def main() -> None:
    xs = CrossSectionPlotly(
        origin=(46.20, -122.25),
        azimuth=90.0,
        radius_km=8.0,
        depth_extent=(-12.0, 2.0),
        layout_width=500,
        layout_height=320,
        label="A",
    )
    hd = np.linspace(2.0, 12.0, 8)
    dep = -3.0 - 0.4 * hd
    xs.scatter(
        x=hd,
        z=dep,
        z_dir="depth",
        z_unit="km",
        s=35,
        c="steelblue",
        edgecolors="black",
        alpha=0.85,
        name="events",
    )
    out = Path(__file__).resolve().with_name("cross_section_plotly_standalone.html")
    xs.save_html(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
