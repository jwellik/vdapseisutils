"""
Client facade (``VClient``) and ``DataSource`` alias — implemented in T4+.

Waveform fetch orchestration lives in ``_fetch``; this module re-exports the
public entry points used by legacy ``DataSource`` and future ``VClient``.
"""

from vdapseisutils.obspy_ext.client._fetch import get_waveforms_from_client

__all__: list[str] = ["get_waveforms_from_client"]
