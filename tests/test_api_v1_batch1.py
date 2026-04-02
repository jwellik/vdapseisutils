"""Regression tests for API v1 Batch 1 (correctness fixes)."""

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from vdapseisutils.obspy_ext.time import VUTCDateTime
from vdapseisutils.core.swarmmpl.clipboard import ClipboardClass, TimeSeries


def test_vutcdatetime_matplotlib_date_matches_obspy():
    iso = "2020-01-15T12:30:45.123456Z"
    base = UTCDateTime(iso)
    v = VUTCDateTime(iso)
    assert float(v.matplotlib_date) == pytest.approx(float(base.matplotlib_date))
    assert v.to_format("matplotlib") == pytest.approx(float(base.matplotlib_date))


def _trace(start, minutes=1.0, npts=500, sampling_rate=50.0):
    data = np.zeros(int(npts))
    return Trace(
        data=data,
        header={
            "network": "XX",
            "station": "TEST",
            "location": "00",
            "channel": "HHZ",
            "starttime": UTCDateTime(start),
            "sampling_rate": sampling_rate,
            "npts": int(npts),
        },
    )


def test_clipboard_set_axes_sync_waves_false_datetime_no_attributeerror():
    st = Stream(
        [
            _trace("2020-01-01T00:00:00", npts=500),
            _trace("2020-01-01T01:00:00", npts=500),
        ]
    )
    fig = ClipboardClass(st=st, mode="w", sync_waves=False, tick_type="datetime")
    fig.plot()
    fig._set_axes()
    assert hasattr(fig, "time_extent")
    assert len(fig.time_extent) == 2
    for row in fig.time_extent:
        assert len(row) == 2
    # Per-trace xlims should span each trace's window (datetime axis)
    assert fig.taxis["xlim"][0][0] <= fig.taxis["xlim"][0][1]
    assert fig.taxis["xlim"][1][0] <= fig.taxis["xlim"][1][1]


def test_clipboard_set_axes_sync_waves_false_relative():
    st = Stream(
        [
            _trace("2020-01-01T00:00:00", npts=500),
            _trace("2020-01-01T01:00:00", npts=500),
        ]
    )
    fig = ClipboardClass(st=st, mode="w", sync_waves=False, tick_type="relative")
    fig.plot()
    fig._set_axes()
    assert fig.taxis["xlim"][0][0] == 0


def test_timeseries_axes_methods_no_recursion():
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = TimeSeries(fig, [0.1, 0.1, 0.8, 0.8])
    t = [UTCDateTime(2020, 1, 1) + i for i in range(10)]
    y = np.arange(10, dtype=float)
    ax.plot(t, y)
    assert len(ax.lines) >= 1
    ax.scatter(t, y, s=5)
    ax.axvline(t[3])
    ax.axvspan(t[2], t[5], alpha=0.2)
    img = np.zeros((2, 2))
    ax.imshow(t[:2], img)
    plt.close(fig)
