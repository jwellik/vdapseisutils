"""Tests for ``vdapseisutils.plot.mpl.register_pyplot`` (API v1 §3)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from vdapseisutils.core.maps.map import Map
from vdapseisutils.core.maps.volcano_figure import VolcanoFigure
from vdapseisutils.core.swarmmpl.clipboard import ClipboardClass
from vdapseisutils.core.swarmmpl.heli import Helicorder
from vdapseisutils.plot.mpl import register_pyplot


def _trace(start, npts=200, sampling_rate=50.0):
    # ObsPy dayplot normalizes on data extrema; all-zero traces break that path.
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


@pytest.fixture(autouse=True)
def _reset_pyplot_state():
    """Avoid leaking patched names across tests."""
    names = ("helicorder", "clipboard", "eqmap", "volcano", "swarm")
    saved = {n: getattr(plt, n, None) for n in names}
    yield
    for n in names:
        if saved[n] is None:
            if hasattr(plt, n):
                delattr(plt, n)
        else:
            setattr(plt, n, saved[n])


def test_register_pyplot_adds_constructors_and_types():
    register_pyplot()

    for name in ("helicorder", "clipboard", "eqmap", "volcano", "swarm"):
        assert hasattr(plt, name)
        assert callable(getattr(plt, name))

    st = Stream([_trace("2020-01-01T00:00:00")])
    h = plt.helicorder(st, interval=15)
    assert isinstance(h, Helicorder)

    cb = plt.clipboard(st=st, mode="w")
    assert isinstance(cb, ClipboardClass)

    m = plt.eqmap()
    assert isinstance(m, Map)

    vf = plt.volcano(figsize=(4, 4), dpi=72)
    assert isinstance(vf, VolcanoFigure)

    sw = plt.swarm(st=st, mode="w")
    assert isinstance(sw, ClipboardClass)


def test_register_pyplot_idempotent():
    register_pyplot()
    h_ref = plt.helicorder
    c_ref = plt.clipboard
    register_pyplot()
    assert plt.helicorder is h_ref
    assert plt.clipboard is c_ref

    st = Stream([_trace("2020-01-02T00:00:00")])
    h2 = plt.helicorder(st, interval=15)
    assert isinstance(h2, Helicorder)
