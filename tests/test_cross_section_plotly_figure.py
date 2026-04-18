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
