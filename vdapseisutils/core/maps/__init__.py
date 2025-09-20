"""
Maps module for vdapseisutils.

This module provides mapping functionality including Map, VolcanoFigure, CrossSection,
and TimeSeries classes, as well as various tile sources for background maps.
"""

from .maps import VolcanoFigure, Map, CrossSection, TimeSeries, MagLegend
from .map_tiles import (
    add_arcgis_terrain,
    add_google_terrain,
    add_google_street,
    add_google_satellite,
    _calculate_auto_zoom_arcgis,
    _calculate_auto_zoom_google
)

__all__ = [
    'VolcanoFigure',
    'Map', 
    'CrossSection',
    'TimeSeries',
    'MagLegend',
    'add_arcgis_terrain',
    'add_google_terrain',
    'add_google_street', 
    'add_google_satellite',
    '_calculate_auto_zoom_arcgis',
    '_calculate_auto_zoom_google'
]
