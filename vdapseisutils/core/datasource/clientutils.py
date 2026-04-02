"""Deprecated shim; waveform fetch lives in ``vdapseisutils.obspy_ext.client``."""

import warnings

warnings.warn(
    "vdapseisutils.core.datasource.clientutils is deprecated; import "
    "get_waveforms_from_client from vdapseisutils.obspy_ext.client instead.",
    DeprecationWarning,
    stacklevel=2,
)

from vdapseisutils.obspy_ext.client._fetch import (  # noqa: E402
    __true_npts,
    _safe_merge,
    get_waveforms_from_client,
)

__all__ = ["__true_npts", "_safe_merge", "get_waveforms_from_client"]
