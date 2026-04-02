"""
Deprecated shim: the legacy ``DataSource`` class is :class:`~vdapseisutils.obspy_ext.client.VClient`
(also exposed as ``DataSource`` there). Import from :mod:`vdapseisutils.obspy_ext` instead.

Loading this submodule runs :mod:`vdapseisutils.core.datasource`, which emits
:class:`DeprecationWarning` once per process for the package.
"""

from __future__ import annotations

from vdapseisutils.obspy_ext.client import DataSource, VClient

__all__ = ["DataSource", "VClient"]
