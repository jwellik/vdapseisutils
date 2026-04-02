"""
ObsPy extensions for vdapseisutils: unified client facade, typed streams/catalogs,
and I/O helpers. This package replaces scattered ``utils/obspyutils/`` and
``core/datasource/`` patterns over the migration (see ``docs/obspy_ext_subtasks.md``).

Team conventions (T0 — do not re-litigate in subtasks)
-------------------------------------------------------
1. **Layout:** New implementation code belongs under ``vdapseisutils/obspy_ext/``
   (sibling to ``core/``), not new modules under ``utils/obspyutils/`` or ad-hoc
   ``core/datasource`` growth for greenfield features.

2. **Client API:** There is a single facade class ``VClient``. The public name
   ``DataSource`` is an alias of ``VClient`` (same class), not a separate type.

3. **Catalog / events:** ``VCatalog.events`` stores real ``VEvent`` instances
   (subclasses of ObsPy's ``Event``). ``VCatalog.__getitem__(i)`` returns
   ``self.events[i]`` (and slice semantics return a ``VCatalog`` over the same
   objects). No proxy objects, no ``_original_event`` / hidden “real” event for
   indexing.

Subpackage ``client`` and module ``io`` (read helpers) are populated across T1–T6.
"""

from vdapseisutils.obspy_ext.catalog import VCatalog, VEvent
from vdapseisutils.obspy_ext.client import DataSource, VClient
from vdapseisutils.obspy_ext.inventory import VInventory
from vdapseisutils.obspy_ext.io import read, read_events, read_inventory
from vdapseisutils.obspy_ext.stream import VStream, VTrace
from vdapseisutils.obspy_ext.stream_id import VStreamID, parse_wave_id, waveID
from vdapseisutils.obspy_ext.time import VUTCDateTime, vutcnow, vutcrange

__all__: list[str] = [
    "DataSource",
    "VClient",
    "VCatalog",
    "VEvent",
    "VInventory",
    "VStream",
    "VTrace",
    "VUTCDateTime",
    "read",
    "read_events",
    "read_inventory",
    "vutcnow",
    "vutcrange",
    "VStreamID",
    "parse_wave_id",
    "waveID",
]
