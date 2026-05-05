from __future__ import annotations

import numpy as np
from bokeh.models import ColorBar, ColumnDataSource
from obspy import Catalog, UTCDateTime
from obspy.core.event import Event, Magnitude, Origin
from obspy.core.inventory import Channel, Inventory, Network, Site, Station

from vdapseisutils.core.maps.bokeh import CrossSection, Map


def _sample_catalog() -> Catalog:
    cat = Catalog()
    for i in range(3):
        ev = Event()
        ev.origins = [
            Origin(
                time=UTCDateTime(2020, 1, 1) + i * 60,
                latitude=46.19 + i * 0.01,
                longitude=-122.26 + i * 0.01,
                depth=2000 + i * 1000,
            )
        ]
        ev.magnitudes = [Magnitude(mag=1.0 + i * 0.2)]
        cat.events.append(ev)
    return cat


def _sample_inventory() -> Inventory:
    station = Station(
        code="TEST",
        latitude=46.2,
        longitude=-122.2,
        elevation=1000.0,
        creation_date=UTCDateTime(2020, 1, 1),
        site=Site(name="TEST"),
    )
    station.channels.append(
        Channel(
            code="BHZ",
            location_code="",
            latitude=46.2,
            longitude=-122.2,
            elevation=1000.0,
            depth=0.0,
            azimuth=0.0,
            dip=-90.0,
            sample_rate=100.0,
        )
    )
    network = Network(code="XX", stations=[station])
    return Inventory(networks=[network], source="smoke")


def test_bokeh_map_smoke_methods():
    cat = _sample_catalog()
    inv = _sample_inventory()
    m = Map(map_extent=[-122.35, -122.05, 46.10, 46.30], figsize=(4, 4))
    m.add_terrain()
    m.plot(
        np.array([46.18, 46.21]),
        np.array([-122.28, -122.16]),
        "-k",
    )
    m.scatter(
        np.array([46.2]),
        np.array([-122.2]),
        size=25,
        color="r",
        hover_text=["pt"],
    )
    m.plot_catalog(cat, c="time", alpha=0.5)
    m.plot_inventory(inv)
    m.plot_volcano(46.20, -122.20, elev=1000)
    m.plot_peak(46.24, -122.15, elev=1200)
    m.plot_cross_section(origin=(46.2, -122.2), azimuth=90, radius_km=4, label="A")
    m.set_title("Map title")
    m.set_subtitle("Map subtitle")
    m.set_catalog_subtitle(cat)
    assert len(m.figure.renderers) >= 6
    assert m.figure.title.text == "Map title"


def test_bokeh_map_scatter_kwargs_alias_and_precedence():
    m = Map(map_extent=[-122.35, -122.05, 46.10, 46.30], figsize=(4, 4))
    renderer = m.scatter(
        lat=np.array([46.18, 46.19]),
        lon=np.array([-122.28, -122.21]),
        size=36,
        color="r",
        c="b",
        s=100,
        edgecolors="k",
        linewidths=2.5,
    )
    glyph = renderer.glyph
    assert glyph.size == 6.0  # sqrt(36): explicit size wins over alias s
    assert glyph.fill_color == "red"  # explicit color wins over alias c
    assert glyph.line_color == "black"  # edgecolors alias -> line_color
    assert glyph.line_width == 2.5  # linewidths alias -> line_width


def test_bokeh_map_scatter_vector_size_with_hover_source():
    m = Map(map_extent=[-122.35, -122.05, 46.10, 46.30], figsize=(4, 4))
    renderer = m.scatter(
        lat=np.array([46.18, 46.19, 46.20]),
        lon=np.array([-122.28, -122.21, -122.16]),
        size=np.array([16, 25, 36]),
        color="k",
        hover_text=["a", "b", "c"],
    )
    assert isinstance(renderer.data_source, ColumnDataSource)
    assert "size" in renderer.data_source.data
    np.testing.assert_allclose(renderer.data_source.data["size"], np.array([4.0, 5.0, 6.0]))


def test_bokeh_map_plot_catalog_color_precedence_and_time_colormap():
    cat = _sample_catalog()
    m_time = Map(map_extent=[-122.35, -122.05, 46.10, 46.30], figsize=(4, 4))
    r_time = m_time.plot_catalog(cat, c="time")
    assert "cval" in r_time.data_source.data

    m_color = Map(map_extent=[-122.35, -122.05, 46.10, 46.30], figsize=(4, 4))
    r_color = m_color.plot_catalog(cat, c="time", color="k")
    assert "cval" not in r_color.data_source.data
    assert r_color.glyph.fill_color == "black"


def test_bokeh_map_plot_heatmap_smoke():
    cat = _sample_catalog()
    m = Map(map_extent=[-122.35, -122.05, 46.10, 46.30], figsize=(4, 4))
    r_cat = m.plot_heatmap(cat, alpha=0.4)
    assert r_cat is not None
    assert "count" in r_cat.data_source.data
    bars = [x for x in m.figure.right if isinstance(x, ColorBar)]
    assert len(bars) == 1
    assert bars[0].title == "Event count"

    r_arrays = m.plot_heatmap(
        np.array([46.18, 46.20, 46.21]),
        np.array([-122.25, -122.22, -122.19]),
        np.array([1000, 2000, 3000]),
        grid_size=0.02,
    )
    assert r_arrays is not None
    assert "left" in r_arrays.data_source.data
    assert len([x for x in m.figure.right if isinstance(x, ColorBar)]) == 1


def test_bokeh_map_plot_heatmap_colorbar_disabled():
    cat = _sample_catalog()
    m = Map(map_extent=[-122.35, -122.05, 46.10, 46.30], figsize=(4, 4))
    m.plot_heatmap(cat, colorbar=False)
    assert not any(isinstance(x, ColorBar) for x in m.figure.right)


def test_bokeh_cross_section_plot_heatmap_colorbar_default():
    cat = _sample_catalog()
    cs = CrossSection(points=[(46.20, -122.26), (46.20, -122.14)], figsize=(5, 3))
    hm = cs.plot_heatmap(cat, alpha=0.5)
    assert hm is not None
    assert any(isinstance(x, ColorBar) for x in cs.figure.right)


def test_bokeh_cross_section_smoke_methods():
    cat = _sample_catalog()
    inv = _sample_inventory()
    cs = CrossSection(points=[(46.20, -122.26), (46.20, -122.14)], figsize=(5, 3))
    cs.plot(x=np.array([0, 5, 10]), z=np.array([1000, 3000, 2000]), color="black")
    cs.scatter(x=np.array([2, 8]), z=np.array([1500, 2500]), c="b", s=np.array([25, 36]))
    cs.plot_catalog(cat, c="time", alpha=0.6)
    cs.plot_inventory(inv, c="black", alpha=0.9)
    cs.plot_volcano(46.20, -122.20, elev=1000)
    cs.plot_peak(46.23, -122.16, elev=1400)
    hm = cs.plot_heatmap(cat, alpha=0.5)
    cs.set_titles(
        title_text="Cross-section title",
        subtitle_text="Cross-section subtitle",
        title_color="black",
        subtitle_color="gray",
    )
    cs.set_catalog_subtitle(cat)
    assert hm is not None
    assert len(cs.figure.renderers) >= 5
    assert cs.figure.title.text == "Cross-section title"


def test_bokeh_cross_section_alias_and_color_precedence():
    cs = CrossSection(points=[(46.20, -122.26), (46.20, -122.14)], figsize=(5, 3))
    r = cs.scatter(
        x=np.array([2, 8]),
        z=np.array([1500, 2500]),
        size=9,
        s=64,
        color="r",
        c="b",
        edgecolors="k",
        linewidths=1.25,
    )
    g = r.glyph
    assert g.size == 9
    assert g.fill_color == "red"
    assert g.line_color == "black"
    assert g.line_width == 1.25


def test_bokeh_cross_section_catalog_time_colormap_consistency():
    cat = _sample_catalog()
    cs_time = CrossSection(points=[(46.20, -122.26), (46.20, -122.14)], figsize=(5, 3))
    r_time = cs_time.plot_catalog(cat, c="time")
    assert "cval" in r_time.data_source.data

    cs_color = CrossSection(points=[(46.20, -122.26), (46.20, -122.14)], figsize=(5, 3))
    r_color = cs_color.plot_catalog(cat, c="time", color="k")
    assert "cval" not in r_color.data_source.data
    assert r_color.glyph.fill_color == "black"
