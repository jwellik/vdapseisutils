"""Deprecated: prefer :mod:`vdapseisutils.obspy_ext` for extended ObsPy types and client."""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from vdapseisutils.utils.obspyutils is deprecated; "
    "use vdapseisutils.obspy_ext instead.",
    DeprecationWarning,
    stacklevel=2,
)

from vdapseisutils.obspy_ext import (
    DataSource,
    VCatalog,
    VClient,
    VEvent,
    VInventory,
    VStream,
    VStreamID,
    VTrace,
    VUTCDateTime,
    parse_wave_id,
    read,
    read_events,
    read_inventory,
    vutcnow,
    vutcrange,
    waveID,
)

__all__ = [
    "DataSource",
    "VCatalog",
    "VClient",
    "VEvent",
    "VInventory",
    "VStream",
    "VStreamID",
    "VTrace",
    "VUTCDateTime",
    "parse_wave_id",
    "read",
    "read_events",
    "read_inventory",
    "vutcnow",
    "vutcrange",
    "waveID",
]
