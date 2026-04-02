"""Canonical swarm API (API v1 §8)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime


def _trace(start: str, npts: int = 200, sampling_rate: float = 50.0) -> Trace:
    rng = np.random.default_rng(0)
    data = rng.standard_normal(int(npts)).astype(np.float64)
    return Trace(
        data=data,
        header={
            "network": "XX",
            "station": "TST",
            "location": "00",
            "channel": "HHZ",
            "starttime": UTCDateTime(start),
            "sampling_rate": sampling_rate,
            "npts": int(npts),
        },
    )


def test_canonical_swarm_all_exported() -> None:
    import vdapseisutils.plot.swarm as swarm

    for name in swarm.__all__:
        assert hasattr(swarm, name), name


def test_clipboard_and_swarmclipboard_distinct_names() -> None:
    from vdapseisutils.plot import swarm

    assert swarm.Clipboard is swarm.clipboard_figure
    assert swarm.SwarmFigure is swarm.ClipboardClass
    assert swarm.SwarmClipboard is not swarm.ClipboardClass


def test_smoke_helicorder_clipboard_swarmclipboard() -> None:
    from vdapseisutils.plot.swarm import Clipboard, Helicorder, SwarmClipboard

    st = Stream([_trace("2020-01-01T00:00:00")])

    h = Helicorder(st, interval=15)
    assert isinstance(h, matplotlib.figure.Figure)

    fig = Clipboard(st=st, mode="w")
    assert isinstance(fig, matplotlib.figure.Figure)

    sc = SwarmClipboard(data=None)
    assert hasattr(sc, "fig")
    assert isinstance(sc.fig, matplotlib.figure.Figure)
