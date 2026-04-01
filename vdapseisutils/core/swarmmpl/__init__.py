"""Legacy Swarm-style Matplotlib helpers (deprecated).

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
