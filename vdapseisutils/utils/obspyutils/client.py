"""Deprecated shim: import ``VClient`` / ``DataSource`` from :mod:`vdapseisutils.obspy_ext`."""

from __future__ import annotations

from vdapseisutils.obspy_ext.client.vclient import DataSource, VClient

__all__ = ["DataSource", "VClient"]
