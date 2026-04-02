"""
I/O helpers that return vdapseisutils extended types (``VStream``, ``VCatalog``,
``VInventory``).

Users who need plain ObsPy ``Stream``, ``Catalog``, or ``Inventory`` objects
should call :func:`obspy.read`, :func:`obspy.read_events`, or
:func:`obspy.read_inventory` directly.
"""

from __future__ import annotations

import obspy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vdapseisutils.utils.obspyutils.catalog import VCatalog
    from vdapseisutils.utils.obspyutils.inventory import VInventory
    from vdapseisutils.utils.obspyutils.stream.core import VStream

__all__ = ["read", "read_events", "read_inventory"]


def read(*args, **kwargs) -> VStream:
    """Read waveforms via ObsPy; returns ``VStream``."""
    from vdapseisutils.utils.obspyutils.stream.core import VStream

    return VStream(obspy.read(*args, **kwargs))


def read_events(*args, **kwargs) -> VCatalog:
    """Read events via ObsPy; returns ``VCatalog``."""
    from vdapseisutils.utils.obspyutils.catalog import VCatalog

    return VCatalog(obspy.read_events(*args, **kwargs))


def read_inventory(*args, **kwargs) -> VInventory:
    """Read station metadata via ObsPy; returns ``VInventory``."""
    from vdapseisutils.utils.obspyutils.inventory import VInventory

    return VInventory(obspy.read_inventory(*args, **kwargs))
