"""
Plotly-oriented helpers for maps (optional ``[plotly]`` extra).

Part 1 provides a matplotlib-free **data layer** for cross-sections; interactive
figures arrive in later parts. Import ``cross_section_data`` for geometry and
catalog/inventory prep without pulling in Plotly.
"""

from __future__ import annotations

from .cross_section_data import (
    CrossSectionData,
    build_cross_section_data,
    inventory_station_arrays,
    prep_catalog_for_cross_section,
    prep_inventory_for_cross_section,
    project_latlon_to_cross_section,
)

__all__ = [
    "CrossSectionData",
    "build_cross_section_data",
    "empty_cross_section_figure",
    "inventory_station_arrays",
    "prep_catalog_for_cross_section",
    "prep_inventory_for_cross_section",
    "project_latlon_to_cross_section",
]


def empty_cross_section_figure():
    """
    Minimal Plotly figure to verify the optional dependency resolves.

    Requires the ``plotly`` package (``pip install 'vdapseisutils[plotly]'`` or ``uv sync --extra plotly``).
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly is not installed. Add the optional extra: pip install 'vdapseisutils[plotly]'"
        ) from exc
    return go.Figure()
