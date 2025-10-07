"""
Default styling and configuration constants for maps module.

This module contains all the styling defaults, configuration dictionaries,
and default volcano information used across the maps classes.

Author: Jay Wellik, jwellik@vdap.org
Last updated: 2025 September 28
"""

import matplotlib.pyplot as plt
from vdapseisutils.style import load_custom_rc

# Load custom styling
load_custom_rc("swarmmplrc")

# Plotting styles and formatters for maps and cross-sections
plt.rcParams['svg.fonttype'] = 'none'

# Global font settings - only set base font size, let classes handle specific styling
plt.rcParams['font.size'] = 8  # Base font size (small for crisp appearance)

# Heatmap defaults - change these values to update all plot_heatmap methods
HEATMAP_DEFAULTS = {
    'cmap': 'plasma',
    'alpha': 0.7,
    'grid_size': 0.01,  # Default to 0.05 degrees (≈5km) for VolcanoFigure
    'vmin': None,
    'vmax': None
}

# Comprehensive styling defaults - change these values to update all classes at once
TICK_DEFAULTS = {
    # Tick and ticklabel styling
    'labelcolor': 'grey',
    'labelsize': 'small',  # 0.8x base = 6.4
    'tick_color': 'grey',
    'tick_size': 3,  # tick length
    'tick_width': 1.5,  # tick line width
    'tick_direction': 'out',
    'tick_pad': 2,  # padding between ticks and labels
    
    # Axis label styling
    'axes_labelcolor': 'grey',
    'axes_labelsize': 'medium',  # 1.0x base = 8
    'axes_titlesize': 'large',  # 1.2x base = 9.6
    
    # Legend styling
    'legend_fontsize': 'medium',  # 1.0x base = 8
}

# General axes customization defaults - applies to all axes
AXES_DEFAULTS = {
    # Spine styling
    'spine_linewidth': 1.5,  # thickness of all spines
    'spine_color': 'black',  # color of spines
}

# Cross-section specific defaults
CROSSSECTION_DEFAULTS = {
    # Profile line styling (for cross-sections)
    'profile_linewidth': 1.5,  # linewidth for topographic profiles
    'profile_color': 'k',  # color for topographic profiles
    
    # Text styling for labels and annotations
    'text_stroke_linewidth': 2,  # linewidth for text stroke effects
    'text_stroke_color': 'white',  # color for text stroke effects
    
    # Label positioning
    'ylabel_rotation': 270,  # rotation for y-axis labels
    'ylabel_pad': 15,  # padding for y-axis labels
}

# Grid line styling defaults for maps
GRID_DEFAULTS = {
    'linewidth': 0,
    'color': 'gray',
    'alpha': 0.5,
    'xlines': True,
    'ylabel_style': {'color': 'grey', 'rotation': 90, 'size': 'small'},  # 90 = vertical
    'xlabel_style': {'color': 'grey', 'size': 'small'}
}

# Default colormap and normalization
cmap = "viridis_r"
norm = None

# Default volcano configurations
hood = {
    'name': "Hood",
    'synonyms': "Wy'east",
    'lat': 45.374,
    'lon': -121.695,
    'elev': 3426,
}

agung = {
    'name': "Agung",
    'synonyms': "Agung",
    'lat': -8.343,
    'lon': 115.508,
    'elev': 2997,
}

default_volcano = agung

# Shared plotting method defaults for consistency across classes
PLOT_CATALOG_DEFAULTS = {
    's': "magnitude",
    'c': "time", 
    'color': None,
    'cmap': "viridis_r",
    'alpha': 0.5
}

PLOT_INVENTORY_DEFAULTS = {
    's': 49,  # Use 's' for size in scatter plots (6^2 = 36 for equivalent visual size)
    'c': 'black',  # Use 'c' for scatter plot color
    'alpha': 1.0,
    'marker': 'v',
    'edgecolors': 'black'  # Use 'edgecolors' (plural) in scatter plots
}

PLOT_VOLCANO_DEFAULTS = {
    'marker': '^',
    'c': 'orangered',  # Use 'c' for color in scatter plots
    's': 64,  # Use 's' for size in scatter plots (8^2 = 64 for equivalent visual size)
    'edgecolors': 'black'  # Use 'edgecolors' (plural) in scatter plots
}

PLOT_PEAK_DEFAULTS = {
    'marker': '^',
    'c': 'floralwhite',  # Use 'c' for color in scatter plots
    's': 64,  # Use 's' for size in scatter plots (8^2 = 64 for equivalent visual size)
    'edgecolors': 'black'  # Use 'edgecolors' (plural) in scatter plots
}

# Title and subtitle styling defaults
TITLE_DEFAULTS = {
    'fontsize': 'large',  # 1.2x base = 9.6
    'fontweight': 'bold',
    'color': 'black',
    'ha': 'center',
    'va': 'top',
    'pad': 20,  # padding in points
    'y': None,  # Will be calculated automatically
    'x': 0.5,   # default x position in figure coordinates
    'auto_spacing': True,  # Enable automatic spacing calculation
}

SUBTITLE_DEFAULTS = {
    'fontsize': 'medium',  # 1.0x base = 8
    'fontweight': 'normal',
    'color': 'black',
    'ha': 'center',
    'va': 'top',
    'pad': 10,  # padding in points
    'y': None,  # Will be calculated automatically
    'x': 0.5,   # default x position in figure coordinates
    'auto_spacing': True,  # Enable automatic spacing calculation
}

# World location map styling defaults
WORLD_LOCATION_MAP_DEFAULTS = {
    # Ocean styling
    'ocean_color': 'lightgrey',
    'ocean_alpha': 0.8,
    
    # Land styling  
    'land_color': 'white',
    'land_alpha': 1.0,
    
    # Country borders styling
    'borders_color': 'black',
    'borders_linewidth': 0.5,
    'borders_alpha': 0.8,
    
    # Coastlines styling
    'coastlines_color': 'black', 
    'coastlines_linewidth': 0.5,
    'coastlines_alpha': 0.8,
    
    # Grid styling
    'grid_linewidth': 0.5,
    'grid_color': 'gray',
    'grid_alpha': 0.5,
    'grid_linestyle': '--',
    
    # Center marker styling
    'marker_color': 'black',
    'marker_size': 5,
    'marker_style': 's'
}


def _test_defaults():
    """Simple test to verify defaults module loads correctly."""
    try:
        # Test that all required defaults are present
        required_defaults = [
            'HEATMAP_DEFAULTS', 'TICK_DEFAULTS', 'AXES_DEFAULTS',
            'CROSSSECTION_DEFAULTS', 'GRID_DEFAULTS', 'TITLE_DEFAULTS', 
            'SUBTITLE_DEFAULTS', 'default_volcano'
        ]
        
        for default in required_defaults:
            if default not in globals():
                raise ValueError(f"Missing required default: {default}")
        
        # Test that default_volcano has required keys
        required_keys = ['name', 'lat', 'lon', 'elev']
        for key in required_keys:
            if key not in default_volcano:
                raise ValueError(f"Missing key '{key}' in default_volcano")
        
        print("✓ All defaults loaded successfully")
        print(f"✓ Default volcano: {default_volcano['name']} at ({default_volcano['lat']}, {default_volcano['lon']})")
        return True
        
    except Exception as e:
        print(f"✗ Defaults test failed: {e}")
        return False


if __name__ == "__main__":
    _test_defaults()
