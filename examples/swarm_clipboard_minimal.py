#!/usr/bin/env python3
"""
Minimal offline Swarm-style plots: ``Helicorder`` and panel ``SwarmClipboard``.

Uses synthetic ObsPy traces only (no network). Run from the repository root::

    python examples/swarm_clipboard_minimal.py

or::

    uv run python examples/swarm_clipboard_minimal.py

See README **Swarm imports (API v1)** and ``vdapseisutils.plot.swarm`` for the
canonical import path. ``SwarmClipboard`` is the multi-panel clipboard; ``Helicorder``
is the dayplot-style figure. Legacy multi-trace ``Clipboard`` / ``ClipboardClass``
are documented in the same README section.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, Trace, UTCDateTime

from vdapseisutils.plot.swarm import Helicorder, SwarmClipboard


def _synthetic_trace(
    *,
    start: str,
    npts: int = 400,
    sampling_rate: float = 50.0,
    network: str = "XX",
    station: str = "TST",
    location: str = "00",
    channel: str = "HHZ",
) -> Trace:
    rng = np.random.default_rng(42)
    data = rng.standard_normal(int(npts)).astype(np.float64)
    return Trace(
        data=data,
        header={
            "network": network,
            "station": station,
            "location": location,
            "channel": channel,
            "starttime": UTCDateTime(start),
            "sampling_rate": sampling_rate,
            "npts": int(npts),
        },
    )


def main() -> None:
    # One trace: helicorder-style day plot (short window for a quick demo).
    st1 = Stream([_synthetic_trace(start="2020-01-01T00:00:00", npts=2500, channel="HHZ")])

    h = Helicorder(st1, interval=15, figsize=(6, 4), dpi=96)
    try:
        h.canvas.draw()
    finally:
        plt.close(h)

    # Two traces: panel clipboard (waveform-only mode keeps the example light).
    st2 = Stream(
        [
            _synthetic_trace(start="2020-01-01T00:00:00", channel="HHZ", station="TST"),
            _synthetic_trace(start="2020-01-01T00:00:00", channel="HHN", station="TST"),
        ]
    )
    cb = SwarmClipboard(
        data=st2,
        sync_waves=False,
        mode="w",
        figsize=(8, 6),
        tick_type="absolute",
    )
    try:
        cb.fig.canvas.draw()
    finally:
        plt.close(cb.fig)


if __name__ == "__main__":
    main()
