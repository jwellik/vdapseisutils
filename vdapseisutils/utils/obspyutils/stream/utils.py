"""
Deprecated shim for stream helpers (duplicate of legacy ``streamutils``).

Use :mod:`vdapseisutils.obspy_ext.stream_ops` instead.
"""

import warnings

warnings.warn(
    "Importing vdapseisutils.utils.obspyutils.stream.utils is deprecated; "
    "use vdapseisutils.obspy_ext.stream_ops instead.",
    DeprecationWarning,
    stacklevel=2,
)

from vdapseisutils.obspy_ext.stream_ops import (  # noqa: E402
    SuperStream,
    align_streams,
    clip,
    createEmptyTrace,
    idselect,
    preprocess,
    removeWinstonGaps,
    replaceGapValue,
    same_data_type,
    sortStreamByNSLClist,
    winston_gap_value,
)

__all__ = [
    "SuperStream",
    "align_streams",
    "clip",
    "createEmptyTrace",
    "idselect",
    "preprocess",
    "removeWinstonGaps",
    "replaceGapValue",
    "same_data_type",
    "sortStreamByNSLClist",
    "winston_gap_value",
]
