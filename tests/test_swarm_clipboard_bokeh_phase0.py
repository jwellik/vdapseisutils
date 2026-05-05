"""Smoke tests for :mod:`vdapseisutils.core.swarmmpl.bokeh` Phase 0 scaffold."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from vdapseisutils.core.swarmmpl.bokeh import SwarmClipboardBk


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


def test_swarm_clipboard_bk_save_html_two_traces(tmp_path):
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


def test_swarm_clipboard_bk_no_data_save(tmp_path):
    cb = SwarmClipboardBk(data=None, mode="w", tick_type="absolute")
    out = tmp_path / "empty.html"
    cb.save(out)
    assert out.stat().st_size > 200


def test_swarm_clipboard_bk_unsupported_mode():
    with pytest.raises(NotImplementedError, match='mode="w"'):
        SwarmClipboardBk(data=None, mode="wg")


def test_swarm_clipboard_bk_unsupported_tick_type():
    with pytest.raises(NotImplementedError, match="absolute"):
        SwarmClipboardBk(data=None, tick_type="relative")

