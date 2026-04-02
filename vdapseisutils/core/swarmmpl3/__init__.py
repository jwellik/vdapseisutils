"""
Deprecated shim for the v3 time-series stack.

Implementation: :mod:`vdapseisutils.core.swarmmpl.v3`.
Prefer :mod:`vdapseisutils.plot.swarm` (API v1 §8); there, the v3 clipboard class is
:class:`vdapseisutils.plot.swarm.SwarmClipboard`.
"""

from __future__ import annotations

from vdapseisutils.core._swarm_deprecation import warn_swarm_legacy_package

warn_swarm_legacy_package(legacy_qualname="vdapseisutils.core.swarmmpl3")

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
