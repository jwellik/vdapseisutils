"""Legacy Swarm-style Matplotlib helpers (deprecated).

On-disk layout (API v1 §13.4 — single implementation tree):

- ``clipboard.py``, ``heli.py`` — legacy multi-trace clipboard and helicorder.
- :mod:`vdapseisutils.core.swarmmpl.v2` — v2 ``plot_trace`` / ``plot_clipboard`` helpers
  (legacy import path: ``vdapseisutils.core.swarmmpl2``, shim).
- :mod:`vdapseisutils.core.swarmmpl.v3` — time axes, panels, v3 clipboard
  (legacy import path: ``vdapseisutils.core.swarmmpl3``, shim).

Prefer :mod:`vdapseisutils.plot.swarm` (API v1 §8).
"""

from __future__ import annotations

from vdapseisutils.core._swarm_deprecation import warn_swarm_legacy_package

warn_swarm_legacy_package(legacy_qualname="vdapseisutils.core.swarmmpl")

from . import colors
from .clipboard import (
    Clipboard,
    ClipboardClass,
    TimeSeries,
    plot_spectrogram,
    plot_trace,
    plot_wave,
    t2axiscoords,
)
from .heli import Helicorder

__all__ = [
    "Helicorder",
    "Clipboard",
    "ClipboardClass",
    "TimeSeries",
    "colors",
    "plot_spectrogram",
    "plot_trace",
    "plot_wave",
    "t2axiscoords",
]
