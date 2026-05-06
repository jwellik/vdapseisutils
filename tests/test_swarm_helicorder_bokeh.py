from __future__ import annotations

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from vdapseisutils.core.swarmmpl.bokeh import SwarmHelicorderBk


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
