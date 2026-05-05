"""
Bokeh-backed Swarm-style clipboard (multi-panel waveforms / spectrograms).

Import as::

    from vdapseisutils.core.swarmmpl.bokeh import SwarmClipboardBk

Matplotlib reference: :class:`vdapseisutils.core.swarmmpl.clipboard.SwarmClipboard`.
Plan: ``docs/plans/bokeh-swarm-clipboard-plan.md`` (modes **w** / **g** / **wg**, spectrogram
settings + **inferno_u** palette).
"""

from __future__ import annotations

from .clipboard import SwarmClipboardBk

__all__ = ["SwarmClipboardBk"]
