"""Legacy Swarm-style Matplotlib helpers, variant 2 (deprecated).

Prefer :mod:`vdapseisutils.plot.swarm` (API v1 §8).
"""

from __future__ import annotations

from vdapseisutils.core._swarm_deprecation import warn_swarm_legacy_package

warn_swarm_legacy_package(legacy_qualname="vdapseisutils.core.swarmmpl2")

from .clipboard import (
    plot_clipboard,
    plot_spectrogram,
    plot_swarmwg,
    plot_trace,
    plot_wave,
)

__all__ = [
    "plot_clipboard",
    "plot_spectrogram",
    "plot_swarmwg",
    "plot_trace",
    "plot_wave",
]
