"""Swarm-style Matplotlib helpers (multi-trace clipboard, helicorder, TimeAxes / Panel stack).

Public API for new code: prefer :mod:`vdapseisutils.plot.swarm` (API v1 §8).

Implementation modules:

- :mod:`vdapseisutils.core.swarmmpl.clipboard` — legacy :class:`ClipboardClass` / :func:`Clipboard`,
  :class:`TimeSeries`, plotting helpers, and panel-based :class:`SwarmClipboard`.
- :mod:`vdapseisutils.core.swarmmpl.panel`, :mod:`vdapseisutils.core.swarmmpl.timeaxes` — v3 layout primitives.
- :mod:`vdapseisutils.core.swarmmpl.convenience` — ``swarmw``, ``swarmg``, ``swarmwg``, ``swarm_clipboard``.
"""

from __future__ import annotations

from . import colors
from .clipboard import (
    Clipboard,
    ClipboardClass,
    SwarmClipboard,
    TimeSeries,
    plot_spectrogram,
    plot_trace,
    plot_wave,
    t2axiscoords,
)
from .convenience import (
    swarm_clipboard,
    swarmg,
    swarmw,
    swarmwg,
)
from .heli import Helicorder
from .panel import Panel
from .timeaxes import TimeAxes

__all__ = [
    "Helicorder",
    "Clipboard",
    "ClipboardClass",
    "SwarmClipboard",
    "TimeSeries",
    "Panel",
    "TimeAxes",
    "colors",
    "plot_spectrogram",
    "plot_trace",
    "plot_wave",
    "swarm_clipboard",
    "swarmg",
    "swarmw",
    "swarmwg",
    "t2axiscoords",
]
