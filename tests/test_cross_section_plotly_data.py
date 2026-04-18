"""Smoke tests for matplotlib-free cross-section data (Plotly foundation, Part 1)."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Magnitude, Origin
from obspy.core.event.resourceid import ResourceIdentifier


def test_build_cross_section_data_origin_matches_example_ranges():
    """Aligns with ``examples/cross_section_standalone.py`` (origin + radius + depth)."""
    from vdapseisutils.core.maps.plotly.cross_section_data import build_cross_section_data

    d = build_cross_section_data(
        origin=(46.20, -122.25),
        azimuth=90.0,
        radius_km=8.0,
        depth_extent=(-12.0, 2.0),
    )
    assert d.horiz_extent_km == (0.0, 16.0)
    assert d.depth_extent == (-12.0, 2.0)
    assert d.depth_range == pytest.approx(14.0)
    assert d.properties["origin"] == (46.20, -122.25)
    assert d.label == "A"
    assert len(d.properties["points"]) == 2
    # Profile may be empty offline; arrays must still be valid numpy arrays
    assert d.profile_distance_km.shape == d.profile_elevation_km.shape


def test_build_cross_section_data_two_points_horizontal_extent():
    from vdapseisutils.core.maps.plotly.cross_section_data import build_cross_section_data

    p1 = (46.2, -122.3)
    p2 = (46.1, -122.1)
    d = build_cross_section_data(points=[p1, p2], depth_extent=(-50.0, 4.0), label="X")
    assert d.A1 == p1
    assert d.A2 == p2
    assert d.horiz_extent_km[0] == 0.0
    assert d.horiz_extent_km[1] > 0.0
    assert d.horiz_extent_km[1] == pytest.approx(float(d.properties["length"]) / 1000.0)


def test_project_latlon_to_cross_section():
    from vdapseisutils.core.maps.plotly.cross_section_data import (
        build_cross_section_data,
        project_latlon_to_cross_section,
    )

    d = build_cross_section_data(
        origin=(46.20, -122.25),
        azimuth=90.0,
        radius_km=5.0,
        depth_extent=(-10.0, 1.0),
    )
    # Point near origin — finite distance along line
    x_km, z_km = project_latlon_to_cross_section(
        [46.20],
        [-122.26],
        [3000.0],
        d.A1,
        d.A2,
        z_dir="depth",
        z_unit="m",
    )
    assert np.isfinite(x_km[0])
    assert np.isfinite(z_km[0])


def test_prep_catalog_for_cross_section():
    from vdapseisutils.core.maps.plotly.cross_section_data import (
        build_cross_section_data,
        prep_catalog_for_cross_section,
    )

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
    m = Magnitude(mag=2.5, magnitude_type="ML")
    m.resource_id = ResourceIdentifier(id="mag/1")
    e.magnitudes = [m]
    cat = Catalog([e])

    d = build_cross_section_data(
        origin=(46.20, -122.25),
        azimuth=90.0,
        radius_km=10.0,
        depth_extent=(-15.0, 2.0),
    )
    out = prep_catalog_for_cross_section(cat, d.A1, d.A2)
    assert len(out["x_km"]) == 1
    assert len(out["z_axes_km"]) == 1
    assert np.isfinite(out["x_km"][0])
    assert np.isfinite(out["z_axes_km"][0])


def test_empty_cross_section_figure_requires_plotly():
    plotly = pytest.importorskip("plotly")
    from vdapseisutils.core.maps.plotly import empty_cross_section_figure

    fig = empty_cross_section_figure()
    assert isinstance(fig, plotly.graph_objects.Figure)
