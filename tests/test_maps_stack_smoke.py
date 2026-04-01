"""
§10 smoke tests for the map / volcano stack (API v1 canonical).

Uses the Agg backend via ``tests/conftest.py``. Prefer artist counts and
structural checks over image comparison.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest

cartopy = pytest.importorskip("cartopy")
import cartopy.crs as ccrs  # noqa: E402


def _artist_count(fig: matplotlib.figure.Figure) -> int:
    n = 0
    for ax in fig.axes:
        n += (
            len(ax.collections)
            + len(ax.lines)
            + len(ax.patches)
            + len(ax.images)
            + len(ax.texts)
        )
    return n


def test_map_constructor_and_native_scatter():
    from vdapseisutils.core.maps.map import Map

    m = Map(figsize=(4, 4), dpi=72)
    try:
        try:
            from cartopy.mpl.geoaxes import GeoAxes

            assert isinstance(m.ax, GeoAxes)
        except ImportError:
            assert isinstance(m.ax, matplotlib.axes.Axes)

        lon = np.array([-122.20, -122.18], dtype=float)
        lat = np.array([46.19, 46.20], dtype=float)
        m.ax.scatter(lon, lat, s=20, transform=ccrs.Geodetic())
        assert len(m.ax.collections) >= 1
    finally:
        plt.close(m.figure)


def test_cross_section_smoke_offline_profile_fallback():
    """CrossSection builds axes; use origin mode so radius is defined if a profile loads."""
    from vdapseisutils.core.maps.cross_section import CrossSection

    xs = CrossSection(
        origin=(46.20, -122.25),
        azimuth=90.0,
        radius_km=5.0,
        figsize=(3, 2),
        dpi=72,
    )
    try:
        assert isinstance(xs.ax, matplotlib.axes.Axes)
        assert xs.properties["origin"] == (46.20, -122.25)
        xs.ax.plot([0, 1], [-5, -2], color="C0")
        assert len(xs.ax.lines) >= 1
    finally:
        plt.close(xs.figure)


def test_volcano_figure_panel_wiring_and_artist_floor():
    """
    Loose regression guard: default layout should keep map, cross-sections,
    time series, and legend subfigure wiring plus a minimum of drawn artists.
    Threshold 10: map grid/scale + cross-section labels/lines + ts spine setup;
    adjust only if layout intentionally changes.
    """
    from vdapseisutils.core.maps.volcano_figure import VolcanoFigure

    vf = VolcanoFigure(figsize=(5, 5), dpi=72)
    try:
        assert hasattr(vf, "map_obj") and vf.map_obj is not None
        assert hasattr(vf, "xs1_obj") and vf.xs1_obj is not None
        assert hasattr(vf, "xs2_obj") and vf.xs2_obj is not None
        assert hasattr(vf, "ts_obj") and vf.ts_obj is not None
        assert len(vf.subfigs) >= 5
        assert _artist_count(vf) >= 10
    finally:
        plt.close(vf)


def test_time_series_standalone_scatter():
    from vdapseisutils.core.maps.time_series import TimeSeries
    from obspy import UTCDateTime

    ts = TimeSeries(figsize=(4, 2), dpi=72, axis_type="depth")
    try:
        assert isinstance(ts.ax, matplotlib.axes.Axes)
        t0 = UTCDateTime("2020-01-01")
        times = [t0 + s for s in (0, 3600, 7200)]
        depths = [-2.0, -5.0, -8.0]
        ts.scatter(times, depths)
        assert len(ts.ax.collections) >= 1
    finally:
        plt.close(ts.figure)


def test_maglegend_mag2s_and_display():
    from vdapseisutils.core.maps.legends import MagLegend

    ml = MagLegend()
    sizes = ml.mag2s([0.0, 1.0, 2.0])
    assert sizes.shape == (3,)
    assert np.all(sizes >= 0)

    fig, ax = plt.subplots(figsize=(2, 3), dpi=72)
    try:
        out = ml.display(ax=ax, include_counts=False)
        assert out is ax
        assert len(ax.collections) >= 1
    finally:
        plt.close(fig)


def test_maps_shim_exports_match_core_package():
    """Sanity: public stack symbols stay reachable from the shim (§13.2)."""
    from vdapseisutils.core import maps as pkg_maps
    from vdapseisutils.core.maps import maps as shim_maps

    assert shim_maps.Map is pkg_maps.Map
    assert shim_maps.VolcanoFigure is pkg_maps.VolcanoFigure
    assert shim_maps.CrossSection is pkg_maps.CrossSection
    assert shim_maps.TimeSeries is pkg_maps.TimeSeries
    assert shim_maps.MagLegend is pkg_maps.MagLegend
