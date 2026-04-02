"""
Deprecated: use :mod:`vdapseisutils.obspy_ext` for the unified client, waveform
fetch helpers, and stream identifiers.

This package re-exports symbols that previously lived under ``core.datasource``
so legacy imports keep working for one release cycle.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Imports from vdapseisutils.core.datasource are deprecated; "
    "use vdapseisutils.obspy_ext instead.",
    DeprecationWarning,
    stacklevel=2,
)

from vdapseisutils.obspy_ext.client import (
    DataSource,
    VClient,
    get_waveforms_bulk_from_client,
    get_waveforms_from_client,
)
from vdapseisutils.obspy_ext.stream_id import VStreamID, parse_wave_id, waveID

__all__ = [
    "DataSource",
    "VClient",
    "VStreamID",
    "get_waveforms_bulk_from_client",
    "get_waveforms_from_client",
    "parse_wave_id",
    "waveID",
]
