"""v3 time-axis, panel, and multi-trace clipboard stack.

Prefer :mod:`vdapseisutils.plot.swarm` (API v1 §8). The legacy package
``vdapseisutils.core.swarmmpl3`` re-exports from this subtree (deprecated).
"""

from __future__ import annotations

from .clipboard import Clipboard
from .convenience import swarm_clipboard, swarmg, swarmw, swarmwg
from .panel import Panel
from .timeaxes import TimeAxes

__all__ = [
    "Clipboard",
    "Panel",
    "TimeAxes",
    "swarm_clipboard",
    "swarmg",
    "swarmw",
    "swarmwg",
]
