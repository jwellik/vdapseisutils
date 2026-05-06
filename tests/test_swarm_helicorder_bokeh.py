from __future__ import annotations

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.core.event import Catalog, Event, Origin, Pick, WaveformStreamID

from bokeh.models import HoverTool, Quad

from vdapseisutils.core.swarmmpl.bokeh import SwarmClipboardBk, SwarmHelicorderBk


def _trace(npts: int = 3600, sr: float = 1.0, **kwargs) -> Trace:
    header = {
        "network": "XX",
        "station": "H01",
        "location": "00",
        "channel": "EHZ",
        "starttime": UTCDateTime("2020-01-01T00:00:00"),
        "sampling_rate": sr,
        "npts": npts,
    }
    header.update(kwargs)
    x = np.linspace(0.0, 20.0 * np.pi, npts, dtype=float)
    data = (200.0 * np.sin(x)).astype(np.float64)
    return Trace(data=data, header=header)


def test_constructor_smoke_absolute_mode():
    st = Stream([_trace()])
    heli = SwarmHelicorderBk(st, interval=60)
    assert heli.interval == 3600
    assert heli.figure.x_range.start == 0.0
    assert heli.figure.x_range.end == 60.0
    assert heli.nlines >= 1


def test_interval_normalization_pathways():
    st = Stream([_trace()])
    assert SwarmHelicorderBk(st, interval=10).line_len_min == 15
    assert SwarmHelicorderBk(st, interval=20).line_len_min == 30
    assert SwarmHelicorderBk(st, interval=45).line_len_min == 60
    assert SwarmHelicorderBk(st, interval=61).line_len_min == 120


def test_one_bar_range_and_clip_threshold_pathways():
    st = Stream([_trace(npts=7200)])
    h_auto = SwarmHelicorderBk(st, one_bar_range=(95, "percentile"), clip_threshold="auto")
    assert h_auto.one_bar_range > 0.0
    assert h_auto.clip_threshold == 3.0 * h_auto.one_bar_range

    h_num = SwarmHelicorderBk(st, one_bar_range=100.0, clip_threshold=50.0)
    assert h_num.one_bar_range == 100.0
    assert h_num.clip_threshold == 50.0


def test_y_tick_label_formatting_presence():
    st = Stream([_trace(npts=24 * 3600, sr=1.0)])
    heli = SwarmHelicorderBk(st, interval=60)
    labels = heli.figure.yaxis[0].major_label_overrides
    assert isinstance(labels, dict)
    assert len(labels) > 0
    assert any(":" in str(v) or "/" in str(v) for v in labels.values())


def test_save_creates_nontrivial_html(tmp_path):
    st = Stream([_trace()])
    heli = SwarmHelicorderBk(st, interval=60)
    out = tmp_path / "heli_bokeh.html"
    heli.save(out)
    txt = out.read_text(encoding="utf-8")
    assert out.stat().st_size > 1000
    assert "Bokeh" in txt or "bk-root" in txt


def test_attach_clipboard_creates_default_wg_mode():
    st = Stream([_trace()])
    heli = SwarmHelicorderBk(st, interval=60)
    cb = heli.attach_clipboard()
    assert isinstance(cb, SwarmClipboardBk)
    assert cb.mode == "wg"


def test_cursor_xy_decode_to_expected_timestamp():
    st = Stream([_trace(npts=7200, sr=1.0)])
    heli = SwarmHelicorderBk(st, interval=60)
    row_idx = 1
    x_offset_min = 2.05
    y_value = heli.nlines - row_idx - 0.1
    got = heli._xy_to_time(x_offset_min, y_value)
    expected = heli.starttime + (row_idx * heli.interval) + (x_offset_min * 60.0)
    assert got is not None
    assert abs(float(got - expected)) < 1e-6


def test_focus_timestamp_updates_clipboard_window():
    st = Stream([_trace(npts=7200, sr=1.0)])
    heli = SwarmHelicorderBk(st, interval=60)
    cb = heli.attach_clipboard(window_s=120.0)
    assert cb is not None
    t_focus = heli.starttime + 600.0
    heli.set_focus_time(t_focus)
    xr = cb.all_axes[0].x_range
    expect_start = float((t_focus - 60.0).timestamp) * 1000.0
    expect_end = float((t_focus + 60.0).timestamp) * 1000.0
    assert abs(float(xr.start) - expect_start) < 1e-6
    assert abs(float(xr.end) - expect_end) < 1e-6


def test_plot_tags_adds_marker_renderer():
    st = Stream([_trace(npts=7200, sr=1.0)])
    heli = SwarmHelicorderBk(st, interval=60)
    t0 = heli.starttime + 300.0
    renderers = heli.plot_tags([t0], marker="circle", color="red")
    assert len(renderers) == 1
    assert renderers[0] in heli.figure.renderers


def test_plot_tags_registers_hover_tool():
    st = Stream([_trace(npts=7200, sr=1.0)])
    heli = SwarmHelicorderBk(st, interval=60)
    before = len([t for t in heli.figure.toolbar.tools if isinstance(t, HoverTool)])
    heli.plot_tags([heli.starttime + 300.0])
    after = len([t for t in heli.figure.toolbar.tools if isinstance(t, HoverTool)])
    assert after > before


def test_highlight_adds_quad_and_hover():
    st = Stream([_trace(npts=24 * 3600, sr=1.0)])
    heli = SwarmHelicorderBk(st, interval=60)
    t0 = heli.starttime + 3600.0
    t1 = heli.starttime + 12 * 3600.0
    renderers = heli.highlight([(t0, t1)])
    assert renderers
    assert any(isinstance(getattr(r, "glyph", None), Quad) for r in heli.figure.renderers)
    assert any(isinstance(t, HoverTool) for t in heli.figure.toolbar.tools)


def test_plot_catalog_smoke_origins_and_picks():
    st = Stream([_trace(station="H01")])
    heli = SwarmHelicorderBk(st, interval=60)
    et = heli.starttime + 120.0
    event = Event(
        origins=[Origin(time=et)],
        picks=[
            Pick(
                time=et + 5.0,
                phase_hint="P",
                waveform_id=WaveformStreamID(
                    network_code="XX",
                    station_code="H01",
                    location_code="00",
                    channel_code="EHZ",
                ),
            )
        ],
    )
    catalog = Catalog(events=[event])
    before = len(heli.figure.renderers)
    renderers = heli.plot_catalog(catalog, plot_picks=True, plot_origins=True)
    after = len(heli.figure.renderers)
    assert len(renderers) >= 1
    assert after > before


def test_decimation_caps_points_per_strip():
    st = Stream([_trace(npts=20000, sr=1.0)])
    heli = SwarmHelicorderBk(
        st,
        interval=60,
        decimate="envelope",
        max_points_per_strip=120,
    )
    line_renderer = next(
        r for r in heli.figure.renderers if hasattr(getattr(r, "data_source", None), "data")
    )
    xs_lines = line_renderer.data_source.data.get("xs", [])
    assert xs_lines
    assert max(len(line) for line in xs_lines) <= 120


def test_decimate_none_keeps_dense_lines():
    st = Stream([_trace(npts=20000, sr=1.0)])
    heli = SwarmHelicorderBk(
        st,
        interval=60,
        decimate="none",
        max_points_per_strip=40,
    )
    line_renderer = next(
        r for r in heli.figure.renderers if hasattr(getattr(r, "data_source", None), "data")
    )
    xs_lines = line_renderer.data_source.data.get("xs", [])
    assert xs_lines
    assert max(len(line) for line in xs_lines) > 40
