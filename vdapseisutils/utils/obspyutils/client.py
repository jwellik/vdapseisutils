"""Deprecated shim: import ``VClient`` / ``DataSource`` from ``obspy_ext.client`` instead."""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing VClient from vdapseisutils.utils.obspyutils.client is deprecated; "
    "use vdapseisutils.obspy_ext.client (or vdapseisutils.obspy_ext) instead.",
    DeprecationWarning,
    stacklevel=2,
)

from vdapseisutils.obspy_ext.client.vclient import DataSource, VClient

__all__ = ["DataSource", "VClient"]
