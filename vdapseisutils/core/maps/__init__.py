"""
Maps module for vdapseisutils.

This module provides mapping functionality including Map, VolcanoFigure, CrossSection,
TimeSeries, and VelocityModel1D classes, as well as various tile sources for
background maps.
"""

# Import classes from their individual modules for backwards compatibility
from .map import Map
from .cross_section import CrossSection
from .time_series import TimeSeries
from .volcano_figure import VolcanoFigure
from .velocity_model_1d import VelocityModel1D
from .legends import MagLegend

# Import utilities that users might need
from .map_tiles import (
    add_arcgis_terrain,
    add_google_terrain,
    add_google_street,
    add_google_satellite,
    _calculate_auto_zoom_arcgis,
    _calculate_auto_zoom_google
)

# Maintain backwards compatibility - all imports work exactly as before
__all__ = [
    'Map',
    'CrossSection', 
    'TimeSeries',
    'VolcanoFigure',
    'VelocityModel1D',
    'MagLegend',
    'add_arcgis_terrain',
    'add_google_terrain',
    'add_google_street', 
    'add_google_satellite',
    '_calculate_auto_zoom_arcgis',
    '_calculate_auto_zoom_google'
]
