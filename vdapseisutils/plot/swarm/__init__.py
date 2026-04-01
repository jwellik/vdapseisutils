"""
Canonical public API for Swarm-style plotting (multi-trace clipboard, helicorder,
v3 :class:`TimeAxes` / :class:`Panel` / panel :class:`SwarmClipboard`).

API v1 §8: use this package instead of ``vdapseisutils.core.swarmmpl*``.

**Name disambiguation**

- :func:`Clipboard` — legacy factory (same as ``matplotlib.pyplot.figure(..., FigureClass=ClipboardClass)``) that returns a :class:`ClipboardClass` instance.
- :class:`SwarmClipboard` — v3 panel-based multi-trace layout (``swarmmpl3``); not the same as the legacy factory.

See ``.local/api-v1-coord/API_V1_CANONICAL.md`` §8.
"""

from __future__ import annotations

from vdapseisutils.core.swarmmpl.clipboard import Clipboard, ClipboardClass
from vdapseisutils.core.swarmmpl.heli import Helicorder
from vdapseisutils.core.swarmmpl3.clipboard import Clipboard as SwarmClipboard
from vdapseisutils.core.swarmmpl3.panel import Panel
from vdapseisutils.core.swarmmpl3.timeaxes import TimeAxes

# Canonical §3 name for the legacy multi-trace figure class.
SwarmFigure = ClipboardClass

# Explicit alias for the legacy factory (same object as ``Clipboard``).
clipboard_figure = Clipboard

__all__ = [
    "Helicorder",
    "Clipboard",
    "clipboard_figure",
    "ClipboardClass",
    "SwarmFigure",
    "TimeAxes",
    "Panel",
    "SwarmClipboard",
]
