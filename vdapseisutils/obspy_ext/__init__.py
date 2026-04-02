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

Subpackages ``client`` and ``io`` are populated in later tasks (T1–T6).
"""

from vdapseisutils.obspy_ext.stream_id import VStreamID, parse_wave_id, waveID

__all__: list[str] = ["VStreamID", "parse_wave_id", "waveID"]
