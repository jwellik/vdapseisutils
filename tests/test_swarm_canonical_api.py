"""Canonical swarm API (API v1 §8) and legacy import deprecation."""

from __future__ import annotations

import importlib

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


@pytest.mark.parametrize(
    "mod",
    (
        "vdapseisutils.core.swarmmpl",
        "vdapseisutils.core.swarmmpl2",
        "vdapseisutils.core.swarmmpl3",
    ),
)
def test_legacy_swarm_package_emits_deprecation(mod: str) -> None:
    """``reload`` re-executes package ``__init__`` so the deprecation hook runs with test stack (§8).

    A plain ``import vdapseisutils.core.swarmmpl`` often does not warn: the parent
    ``vdapseisutils`` package may load ``swarmmpl`` first while ``vdapseisutils`` is
    still on the stack, which intentionally suppresses the warning.
    """
    m = importlib.import_module(mod)
    with pytest.warns(DeprecationWarning, match="plot.swarm"):
        importlib.reload(m)


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
