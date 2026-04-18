"""Tests for :class:`~vdapseisutils.core.maps.plotly.cross_section_plotly.CrossSectionPlotly`."""

from __future__ import annotations

from pathlib import Path

import pytest
from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Magnitude, Origin
from obspy.core.event.resourceid import ResourceIdentifier

pytest.importorskip("plotly")


def test_cross_section_plotly_methods_return_figure():
    from vdapseisutils.core.maps.plotly.cross_section_plotly import CrossSectionPlotly

    xs = CrossSectionPlotly(
        origin=(46.20, -122.25),
        azimuth=90.0,
        radius_km=8.0,
        depth_extent=(-12.0, 2.0),
        layout_height=320,
    )
    fig0 = xs.figure
    fig1 = xs.plot(x=[2.0, 6.0, 10.0], z=[-3.0, -4.0, -5.0], z_dir="depth", z_unit="km")
    assert fig1 is fig0
    fig2 = xs.scatter(
        lat=[46.20, 46.21],
        lon=[-122.24, -122.23],
        z=[-2.0, -3.0],
        z_dir="depth",
        z_unit="km",
        s=12,
        c="steelblue",
    )
    assert fig2 is fig0
    assert len(fig0.data) >= 2


def test_plot_catalog_returns_figure():
    from vdapseisutils.core.maps.plotly.cross_section_plotly import CrossSectionPlotly

    o = Origin(
        time=UTCDateTime(2020, 1, 1),
        latitude=46.20,
        longitude=-122.26,
        depth=3000.0,
    )
    o.resource_id = ResourceIdentifier(id="origin/o1")
    e = Event()
    e.origins = [o]
    e.preferred_origin_id = o.resource_id
    e.magnitudes = [
        Magnitude(mag=2.5, magnitude_type="ML", resource_id=ResourceIdentifier(id="mag/1"))
    ]
    cat = Catalog([e])

    xs = CrossSectionPlotly(
        origin=(46.20, -122.25),
        azimuth=90.0,
        radius_km=10.0,
        depth_extent=(-15.0, 2.0),
        layout_height=300,
    )
    fig = xs.plot_catalog(cat)
    assert fig is xs.figure
    assert len(fig.data) >= 1


def test_axis_titles_and_tick_styling():
    from vdapseisutils.core.maps.plotly.cross_section_plotly import CrossSectionPlotly

    xs = CrossSectionPlotly(
        origin=(46.20, -122.25),
        azimuth=90.0,
        radius_km=8.0,
        depth_extent=(-12.0, 2.0),
        layout_height=280,
    )
    layout = xs.figure.layout
    assert layout.xaxis.title.text == "Distance along profile (km)"
    assert layout.yaxis.title.text == "Depth (km)"
    assert layout.yaxis.side == "right"
    assert layout.xaxis.tickfont.color is not None


def test_save_html_writes_file(tmp_path: Path):
    from vdapseisutils.core.maps.plotly.cross_section_plotly import CrossSectionPlotly

    xs = CrossSectionPlotly(
        origin=(46.20, -122.25),
        azimuth=90.0,
        radius_km=8.0,
        depth_extent=(-10.0, 2.0),
    )
    p = tmp_path / "xs.html"
    xs.save_html(p)
    assert p.is_file()
    text = p.read_text(encoding="utf-8")
    assert "plotly" in text.lower()


def test_scatter_marker_sizes_use_matplotlib_area_semantics():
    """``s`` is matplotlib pt² area; Plotly diameters must be modest px (not raw s as px)."""
    from vdapseisutils.core.maps.plotly import cross_section_plotly as csp

    assert csp._mpl_scatter_area_to_plotly_diameter_px(49.0) < 15.0  # inventory default
    assert csp._mpl_scatter_area_to_plotly_diameter_px(64.0) < 15.0  # volcano/peak default
    arr = csp._mpl_scatter_area_to_plotly_diameter_px([36.0, 100.0])
    assert float(arr.max()) <= 80.0


def test_plot_catalog_time_colorbar_and_magnitude_legend():
    """Default catalog uses time colorbar + MagLegend-sized markers; Plotly adds M… legend."""
    from vdapseisutils.core.maps.plotly.cross_section_plotly import CrossSectionPlotly

    o = Origin(
        time=UTCDateTime(2020, 1, 1),
        latitude=46.20,
        longitude=-122.26,
        depth=3000.0,
    )
    o.resource_id = ResourceIdentifier(id="origin/o1")
    e = Event()
    e.origins = [o]
    e.preferred_origin_id = o.resource_id
    e.magnitudes = [
        Magnitude(mag=2.5, magnitude_type="ML", resource_id=ResourceIdentifier(id="mag/1"))
    ]
    cat = Catalog([e])

    xs = CrossSectionPlotly(
        origin=(46.20, -122.25),
        azimuth=90.0,
        radius_km=10.0,
        depth_extent=(-15.0, 2.0),
        layout_height=300,
    )
    xs.plot_catalog(cat)
    cat_tr = next(t for t in xs.figure.data if getattr(t, "name", None) == "catalog")
    assert cat_tr.marker.showscale is True
    assert cat_tr.marker.colorbar.title.text == "Time"
    mag_leg = [t for t in xs.figure.data if getattr(t, "legendgroup", None) == "vdap_mag_legend"]
    assert len(mag_leg) >= 2


def test_minimal_fake_arrays_instantiation():
    """Smoke test with in-axis coordinates only (no catalog / network)."""
    from vdapseisutils.core.maps.plotly.cross_section_plotly import CrossSectionPlotly

    xs = CrossSectionPlotly(
        points=[(46.19, -122.30), (46.21, -122.20)],
        depth_extent=(-5.0, 1.0),
        layout_width=400,
        layout_height=280,
    )
    xs.scatter(
        x=[1.0, 2.5, 4.0],
        z=[-2.0, -3.0, -1.5],
        z_dir="depth",
        z_unit="km",
        s=20,
        c="gray",
        showlegend=False,
    )
    assert len(xs.figure.data) >= 1
    xs.set_title("synthetic")
    assert xs.figure.layout.title.text == "synthetic"
