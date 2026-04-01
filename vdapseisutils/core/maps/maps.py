"""
Deprecated compatibility re-exports for the split ``vdapseisutils.core.maps`` package.

**Do not import this module in new code.** Implementations live in ``map.py``,
``cross_section.py``, ``volcano_figure.py``, ``time_series.py``, ``legends.py``,
``utils.py``, ``map_tiles.py``, and ``defaults.py``. Use for example::

    from vdapseisutils.core.maps import Map, VolcanoFigure, CrossSection, TimeSeries, MagLegend
    from vdapseisutils.core.maps.utils import prep_catalog_data_mpl
    from vdapseisutils.core.maps.map import Map, add_hillshade_pygmt

This stub remains only so old ``from vdapseisutils.core.maps.maps import ...`` paths
still resolve; importing it emits :exc:`DeprecationWarning` and the module will be
removed in a future release.

See ``.local/api-v1-coord/API_V1_CANONICAL.md`` §13.5.
"""

from __future__ import annotations

import warnings

import matplotlib as mpl

warnings.warn(
    "vdapseisutils.core.maps.maps is deprecated: import from vdapseisutils.core.maps "
    "(package), or from vdapseisutils.core.maps.map, .utils, .cross_section, etc. "
    "This module will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from .cross_section import CrossSection
from .defaults import (
    AXES_DEFAULTS,
    CROSSSECTION_DEFAULTS,
    GRID_DEFAULTS,
    HEATMAP_DEFAULTS,
    PLOT_CATALOG_DEFAULTS,
    PLOT_INVENTORY_DEFAULTS,
    PLOT_PEAK_DEFAULTS,
    PLOT_VOLCANO_DEFAULTS,
    SUBTITLE_DEFAULTS,
    TICK_DEFAULTS,
    TITLE_DEFAULTS,
    WORLD_LOCATION_MAP_DEFAULTS,
    agung,
    cmap,
    default_volcano,
    hood,
    norm,
)
from .legends import ColorBar, MagLegend
from .map import Map, add_hillshade_pygmt
from .map_tiles import ShadedReliefESRI
from .time_series import TimeSeries
from .utils import choose_scale_bar_length, get_scale_length, prep_catalog_data_mpl
from .volcano_figure import VolcanoFigure

# Legacy font aliases (formerly defined alongside rc setup in the monolith)
titlefontsize = t1fs = TICK_DEFAULTS["axes_titlesize"]
subtitlefontsize = t2fs = TICK_DEFAULTS["axes_labelsize"]
axlabelfontsize = axlf = TICK_DEFAULTS["axes_labelsize"]
annotationfontsize = afs = mpl.rcParams["font.size"]
axlabelcolor = axlc = TICK_DEFAULTS["axes_labelcolor"]

__all__ = [
    "AXES_DEFAULTS",
    "CROSSSECTION_DEFAULTS",
    "ColorBar",
    "CrossSection",
    "GRID_DEFAULTS",
    "HEATMAP_DEFAULTS",
    "MagLegend",
    "Map",
    "PLOT_CATALOG_DEFAULTS",
    "PLOT_INVENTORY_DEFAULTS",
    "PLOT_PEAK_DEFAULTS",
    "PLOT_VOLCANO_DEFAULTS",
    "SUBTITLE_DEFAULTS",
    "ShadedReliefESRI",
    "TICK_DEFAULTS",
    "TITLE_DEFAULTS",
    "TimeSeries",
    "VolcanoFigure",
    "WORLD_LOCATION_MAP_DEFAULTS",
    "add_hillshade_pygmt",
    "afs",
    "agung",
    "annotationfontsize",
    "axlabelcolor",
    "axlabelfontsize",
    "axlc",
    "axlf",
    "choose_scale_bar_length",
    "cmap",
    "default_volcano",
    "get_scale_length",
    "hood",
    "norm",
    "prep_catalog_data_mpl",
    "subtitlefontsize",
    "t1fs",
    "t2fs",
    "titlefontsize",
]
