"""
Pure style and configuration constants for maps (no plotting-library import).

Import this module when you need dicts/volcano presets without touching rcParams.
Figure style registration lives in :mod:`vdapseisutils.core.maps.defaults`.
"""

# Heatmap defaults - change these values to update all plot_heatmap methods
HEATMAP_DEFAULTS = {
    "cmap": "plasma",
    "alpha": 0.7,
    "grid_size": 0.01,  # Default to 0.05 degrees (≈5km) for VolcanoFigure
    "vmin": None,
    "vmax": None,
}

# Comprehensive styling defaults - change these values to update all classes at once
TICK_DEFAULTS = {
    "labelcolor": "grey",
    "labelsize": "small",
    "tick_color": "grey",
    "tick_size": 3,
    "tick_width": 1.5,
    "tick_direction": "out",
    "tick_pad": 2,
    "axes_labelcolor": "grey",
    "axes_labelsize": "medium",
    "axes_titlesize": "large",
    "legend_fontsize": "medium",
}

AXES_DEFAULTS = {
    "spine_linewidth": 1.5,
    "spine_color": "black",
}

CROSSSECTION_DEFAULTS = {
    "profile_linewidth": 1.5,
    "profile_color": "k",
    "text_stroke_linewidth": 2,
    "text_stroke_color": "white",
    "ylabel_rotation": 270,
    "ylabel_pad": 15,
}

GRID_DEFAULTS = {
    "linewidth": 0,
    "color": "gray",
    "alpha": 0.5,
    "xlines": True,
    "ylabel_style": {"color": "grey", "rotation": 90, "size": "small"},
    "xlabel_style": {"color": "grey", "size": "small"},
}

cmap = "viridis_r"
norm = None

hood = {
    "name": "Hood",
    "synonyms": "Wy'east",
    "lat": 45.374,
    "lon": -121.695,
    "elev": 3426,
}

agung = {
    "name": "Agung",
    "synonyms": "Agung",
    "lat": -8.343,
    "lon": 115.508,
    "elev": 2997,
}

default_volcano = agung

PLOT_CATALOG_DEFAULTS = {
    "s": "magnitude",
    "c": "time",
    "color": None,
    "cmap": "viridis_r",
    "alpha": 0.5,
}

PLOT_INVENTORY_DEFAULTS = {
    "s": 49,
    "c": "black",
    "alpha": 1.0,
    "marker": "v",
    "edgecolors": "black",
}

PLOT_VOLCANO_DEFAULTS = {
    "marker": "^",
    "c": "orangered",
    "s": 64,
    "edgecolors": "black",
}

PLOT_PEAK_DEFAULTS = {
    "marker": "^",
    "c": "floralwhite",
    "s": 64,
    "edgecolors": "black",
}

TITLE_DEFAULTS = {
    "fontsize": "large",
    "fontweight": "bold",
    "color": "black",
    "ha": "center",
    "va": "top",
    "pad": 20,
    "y": None,
    "x": 0.5,
    "auto_spacing": True,
}

SUBTITLE_DEFAULTS = {
    "fontsize": "medium",
    "fontweight": "normal",
    "color": "black",
    "ha": "center",
    "va": "top",
    "pad": 10,
    "y": None,
    "x": 0.5,
    "auto_spacing": True,
}

WORLD_LOCATION_MAP_DEFAULTS = {
    "ocean_color": "lightgrey",
    "ocean_alpha": 0.8,
    "land_color": "white",
    "land_alpha": 1.0,
    "borders_color": "black",
    "borders_linewidth": 0.5,
    "borders_alpha": 0.8,
    "coastlines_color": "black",
    "coastlines_linewidth": 0.5,
    "coastlines_alpha": 0.8,
    "grid_linewidth": 0.5,
    "grid_color": "gray",
    "grid_alpha": 0.5,
    "grid_linestyle": "--",
    "marker_color": "black",
    "marker_size": 5,
    "marker_style": "s",
}
