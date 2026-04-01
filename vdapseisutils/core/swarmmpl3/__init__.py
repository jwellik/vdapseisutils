"""
swarmmpl3: Enhanced time-series plotting for seismic data (deprecated).

Prefer :mod:`vdapseisutils.plot.swarm` (API v1 §8). There, the v3 panel clipboard
class is exposed as :class:`vdapseisutils.plot.swarm.SwarmClipboard`.

A more structured and object-oriented approach to plotting ObsPy Stream and Trace objects
with waveforms and spectrograms.

Classes:
    TimeAxes: Time-aware axes wrapper with flexible tick formatting
    Panel: Collection of TimeAxes sharing a time axis (e.g., waveform + spectrogram)
    Clipboard: Collection of Panels for multi-trace plotting

Convenience functions:
    swarmw: Plot waveform only
    swarmg: Plot spectrogram only  
    swarmwg: Plot waveform + spectrogram panel
"""

from __future__ import annotations

from vdapseisutils.core._swarm_deprecation import warn_swarm_legacy_package

warn_swarm_legacy_package(legacy_qualname="vdapseisutils.core.swarmmpl3")

from .timeaxes import TimeAxes
from .panel import Panel
from .clipboard import Clipboard
from .convenience import swarmw, swarmg, swarmwg, swarm_clipboard

__all__ = ['TimeAxes', 'Panel', 'Clipboard', 'swarmw', 'swarmg', 'swarmwg', 'swarm_clipboard']
