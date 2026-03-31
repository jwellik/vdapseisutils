"""
Backend-neutral compute helpers for plotting and dashboards.

Planned extraction of numeric preparation from matplotlib-specific code lives here;
see ``.local/api-v1-coord/API_V1_CANONICAL.md`` §7.

Submodules: ``catalog``, ``waveforms``, ``maps`` (map-adjacent pure helpers when migrated).
"""

from . import catalog, maps, waveforms
from .catalog import prepare_catalog_points, prepare_catalog_points_from_time_format

__all__ = [
    "catalog",
    "maps",
    "waveforms",
    "prepare_catalog_points",
    "prepare_catalog_points_from_time_format",
]
