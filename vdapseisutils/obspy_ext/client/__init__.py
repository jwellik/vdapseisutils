"""
Client facade (``VClient``) and ``DataSource`` alias.

Waveform fetch orchestration lives in ``_fetch``; this package re-exports the
public entry points used by legacy callers and ``core.datasource.DataSource``.
"""

from vdapseisutils.obspy_ext.client._fetch import (
    get_waveforms_bulk_from_client,
    get_waveforms_from_client,
)
from vdapseisutils.obspy_ext.client.vclient import DataSource, VClient

__all__: list[str] = [
    "DataSource",
    "VClient",
    "get_waveforms_bulk_from_client",
    "get_waveforms_from_client",
]
