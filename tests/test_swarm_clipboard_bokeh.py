"""Tests for :mod:`vdapseisutils.core.swarmmpl.bokeh` (Phase 0 scaffold + Phase 1 ticks)."""

from __future__ import annotations

import numpy as np
import pytest
from bokeh.models import BasicTickFormatter, DatetimeTickFormatter
from obspy import Stream, Trace, UTCDateTime

from vdapseisutils.core.swarmmpl.bokeh import SwarmClipboardBk


def _long_trace(**kwargs) -> Trace:
    """Enough samples for :func:`compute_spectrogram` with default window."""
    header = {
        "network": "XX",
        "station": "T01",
        "location": "00",
        "channel": "HHZ",
        "starttime": UTCDateTime("2020-01-01T00:00:00"),
        "sampling_rate": 20.0,
        "npts": 800,
    }
    header.update(kwargs)
    rng = np.random.default_rng(42)
    data = rng.standard_normal(header["npts"]).astype(np.float64)
    return Trace(data=data, header=header)


def _tiny_trace(**kwargs) -> Trace:
    header = {
        "network": "XX",
        "station": "T01",
        "location": "00",
        "channel": "HHZ",
        "starttime": UTCDateTime("2020-01-01T00:00:00"),
        "sampling_rate": 20.0,
        "npts": 40,
    }
    header.update(kwargs)
    data = np.linspace(-1.0, 1.0, header["npts"])
    return Trace(data=data.astype("float64"), header=header)


def test_save_html_two_traces_absolute_sync(tmp_path):
    st = Stream(
        [
            _tiny_trace(station="A"),
            _tiny_trace(station="B", channel="HHN"),
        ]
    )
    cb = SwarmClipboardBk(data=st, sync_waves=True, mode="w", tick_type="absolute")
    out = tmp_path / "clipboard.html"
    cb.save(out)
    text = out.read_text(encoding="utf-8")
    assert out.stat().st_size > 500
    assert "bk-root" in text or "Bokeh" in text


def test_tick_type_datetime_alias_matches_absolute():
    st = Stream([_tiny_trace(), _tiny_trace(station="B", channel="HHN")])
    c1 = SwarmClipboardBk(data=st, sync_waves=True, tick_type="absolute")
    c2 = SwarmClipboardBk(data=st, sync_waves=True, tick_type="datetime")
    assert c1.figures[0].x_range.start == c2.figures[0].x_range.start
    assert isinstance(c1.figures[0].xaxis[0].formatter, DatetimeTickFormatter)
    assert isinstance(c2.figures[0].xaxis[0].formatter, DatetimeTickFormatter)


def test_no_data_save_absolute(tmp_path):
    cb = SwarmClipboardBk(data=None, mode="w", tick_type="absolute")
    cb.save(tmp_path / "empty.html")


def test_no_data_save_relative(tmp_path):
    cb = SwarmClipboardBk(data=None, mode="w", tick_type="relative")
    cb.save(tmp_path / "empty_rel.html")


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown mode"):
        SwarmClipboardBk(data=None, mode="wx")


def test_unknown_tick_type():
    with pytest.raises(ValueError, match="Unknown tick_type"):
        SwarmClipboardBk(data=None, tick_type="not_a_real_mode")


def test_relative_unsynced_independent_ranges():
    st = Stream(
        [
            _tiny_trace(starttime=UTCDateTime("2020-01-01T00:00:00")),
            _tiny_trace(
                starttime=UTCDateTime("2020-01-01T00:05:00"),
                station="B",
                channel="HHN",
            ),
        ]
    )
    cb = SwarmClipboardBk(data=st, sync_waves=False, tick_type="relative", mode="w")
    r0 = cb.figures[0].x_range
    r1 = cb.figures[1].x_range
    assert r0.start == 0.0 and r1.start == 0.0
    assert r0 is not r1
    assert isinstance(cb.figures[0].xaxis[0].formatter, BasicTickFormatter)


def test_relative_sync_misaligned_starts_shared_axis():
    """Two traces offset in absolute time; relative + sync uses seconds from t0 (min start)."""
    st = Stream(
        [
            _tiny_trace(starttime=UTCDateTime("2020-01-01T00:00:00")),
            _tiny_trace(
                starttime=UTCDateTime("2020-01-01T00:01:00"),
                station="B",
                channel="HHN",
            ),
        ]
    )
    cb = SwarmClipboardBk(data=st, sync_waves=True, tick_type="relative", mode="w")
    assert cb.figures[0].x_range is cb.figures[1].x_range
    # Global span: 60 s gap + 2 s waveforms -> ~62 s (ObsPy float duration)
    assert cb.figures[0].x_range.end > 60.0


def test_mode_g_spectrogram_html(tmp_path):
    st = Stream([_long_trace()])
    cb = SwarmClipboardBk(data=st, mode="g", tick_type="absolute")
    out = tmp_path / "spec.html"
    cb.save(out)
    html = out.read_text(encoding="utf-8")
    assert out.stat().st_size > 2000
    assert "ColorBar" in html or "color_mapper" in html


def test_mode_wg_stacked_wave_and_spec():
    st = Stream([_long_trace()])
    cb = SwarmClipboardBk(data=st, mode="wg", tick_type="absolute", sync_waves=True)
    panel = cb.figures[0]
    assert len(panel.children) == 2


def test_absolute_sync_mismatched_starts_same_range_object():
    st = Stream(
        [
            _tiny_trace(starttime=UTCDateTime("2020-01-01T00:00:00")),
            _tiny_trace(
                starttime=UTCDateTime("2020-01-01T00:01:00"),
                station="B",
                channel="HHN",
            ),
        ]
    )
    cb = SwarmClipboardBk(data=st, sync_waves=True, tick_type="absolute", mode="w")
    assert cb.figures[0].x_range is cb.figures[1].x_range
