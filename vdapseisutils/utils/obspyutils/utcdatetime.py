"""Deprecated shim; use :mod:`vdapseisutils.obspy_ext.time`."""

from vdapseisutils.obspy_ext.time import VUTCDateTime, vutcnow, vutcrange

__all__ = ["VUTCDateTime", "vutcnow", "vutcrange"]
