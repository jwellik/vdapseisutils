"""
Backward-compatible shim. Import from :mod:`vdapseisutils.obspy_ext` instead.

This module emits :exc:`DeprecationWarning` on import.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Imports from vdapseisutils.core.datasource.waveID are deprecated; "
    "use vdapseisutils.obspy_ext (e.g. VStreamID, parse_wave_id).",
    DeprecationWarning,
    stacklevel=2,
)

from vdapseisutils.obspy_ext.stream_id import VStreamID, parse_wave_id, waveID

__all__ = ["VStreamID", "waveID", "parse_wave_id"]
