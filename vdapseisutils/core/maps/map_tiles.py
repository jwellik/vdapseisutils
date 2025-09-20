"""
Map tile functionality for vdapseisutils.

This module provides methods for adding various map tile sources to cartopy maps,
including ArcGIS terrain tiles and Google Maps tiles.

Author: Jay Wellik, jwellik@vdap.org
Last updated: 2025-01-27
"""

import os
import cartopy.io.img_tiles as cimgt


def _calculate_auto_zoom_arcgis(radius_km):
    """
    Calculate appropriate zoom level for ArcGIS tiles based on radius in kilometers.
    
    Parameters:
    -----------
    radius_km : float
        Radius in kilometers
        
    Returns:
    --------
    int
        Appropriate zoom level for ArcGIS tiles
    """
    if radius_km <= 1:
        return 15
    elif radius_km <= 5:
        return 15
    elif radius_km <= 10:
        return 13
    elif radius_km <= 25:
        return 11
    elif radius_km <= 50:
        return 10
    elif radius_km <= 100:
        return 9
    elif radius_km <= 200:
        return 8
    elif radius_km <= 500:
        return 6
    else:
        return 5


def _calculate_auto_zoom_google(radius_km):
    """
    Calculate appropriate zoom level for Google tiles based on radius in kilometers.
    
    Parameters:
    -----------
    radius_km : float
        Radius in kilometers
        
    Returns:
    --------
    int
        Appropriate zoom level for Google tiles
    """
    if radius_km <= 1:
        return 15
    elif radius_km <= 5:
        return 15
    elif radius_km <= 10:
        return 13
    elif radius_km <= 25:
        return 11
    elif radius_km <= 50:
        return 10
    elif radius_km <= 100:
        return 9
    elif radius_km <= 200:
        return 8
    elif radius_km <= 500:
        return 6
    else:
        return 5


def add_arcgis_terrain(ax, zoom='auto', style='terrain', cache=False, radial_extent_km=None):
    """
    Add world terrain background tiles from ArcGIS to the map.

    Parameters:
    -----------
    ax : matplotlib.axes
        The axes to add tiles to
    zoom : int or str, optional
        Zoom level for the tiles ('auto' for auto-detection, default: 'auto')
    style : str, optional
        Style of ArcGIS tiles ('terrain', 'street', 'satellite', default: 'terrain')
    cache : bool, optional
        Whether to cache tiles (default: False)
    radial_extent_km : float, optional
        Radial extent in km for auto zoom calculation (default: None)
    """
    # Calculate zoom level if auto
    if zoom == 'auto':
        if radial_extent_km:
            zoom_level = _calculate_auto_zoom_arcgis(radial_extent_km)
        else:
            # Fallback to a reasonable default
            zoom_level = 10
    else:
        zoom_level = zoom
    
    background_tile = cimgt.GoogleTiles(
        cache=cache,
        url=(
            'https://server.arcgisonline.com/ArcGIS/rest/services/'
            'World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg'
        ),
        style=style,
        desired_tile_form='RGB',
    )
    ax.add_image(background_tile, zoom_level)


def add_google_tile(ax, zoom='auto', style='terrain', cache=False, radial_extent_km=None, **kwargs):
    """
    Add Google background tiles to the map.
    
    Parameters:
    -----------
    ax : matplotlib.axes
        The axes to add tiles to
    zoom : int or str, optional
        Zoom level for the tiles ('auto' for auto-detection, default: 'auto')
    style : str, optional
        Style of Google tiles ('terrain', 'street', 'satellite', default: 'terrain')
    cache : bool, optional
        Whether to cache tiles (default: False)
    radial_extent_km : float, optional
        Radial extent in km for auto zoom calculation (default: None)
    """
    # Calculate zoom level if auto
    if zoom == 'auto':
        if radial_extent_km:
            zoom_level = _calculate_auto_zoom_google(radial_extent_km)
        else:
            # Fallback to a reasonable default
            zoom_level = 10
    else:
        zoom_level = zoom
    
    # Map style names to Google's lyrs parameter
    style_map = {
        'terrain': 'p',
        'street': 'm', 
        'satellite': 's'
    }
    
    lyrs = style_map.get(style, 'p')  # Default to terrain
    
    background_tile = cimgt.GoogleTiles(
        cache=cache,
        url=f'https://mt1.google.com/vt/lyrs={lyrs}&x={{x}}&y={{y}}&z={{z}}',
        desired_tile_form='RGB',
    )
    
    # Check if axes projection matches tile projection
    if hasattr(ax, 'projection') and ax.projection != background_tile.crs:
        print(f"Warning: Axes projection {ax.projection} doesn't match tile projection {background_tile.crs}")
        print("Consider using the same projection for both axes and tiles")
    
    ax.add_image(background_tile, zoom_level)


def add_google_terrain(ax, zoom='auto', cache=False, radial_extent_km=None, **kwargs):
    """Add Google terrain tiles to the map."""
    add_google_tile(ax, zoom, style='terrain', cache=cache, radial_extent_km=radial_extent_km, **kwargs)


def add_google_street(ax, zoom='auto', cache=False, radial_extent_km=None, **kwargs):
    """Add Google street tiles to the map."""
    add_google_tile(ax, zoom, style='street', cache=cache, radial_extent_km=radial_extent_km, **kwargs)


def add_google_satellite(ax, zoom='auto', cache=False, radial_extent_km=None, **kwargs):
    """Add Google satellite tiles to the map."""
    add_google_tile(ax, zoom, style='satellite', cache=cache, radial_extent_km=radial_extent_km, **kwargs)


def add_tile_with_projection_check(ax, tile_source, zoom, tile_name="tile"):
    """
    Add tiles to axes with proper projection handling.
    
    Parameters:
    -----------
    ax : matplotlib.axes
        The axes to add tiles to
    tile_source : cartopy.io.img_tiles
        The tile source (GoogleTiles, Stamen, etc.)
    zoom : int
        Zoom level for the tiles
    tile_name : str
        Name of tile source for error messages
    """
    try:
        # Check projection compatibility
        if hasattr(ax, 'projection') and hasattr(tile_source, 'crs'):
            if ax.projection != tile_source.crs:
                print(f"Warning: {tile_name} projection mismatch:")
                print(f"  Axes projection: {ax.projection}")
                print(f"  Tile projection: {tile_source.crs}")
                print("  This may cause distortion or errors")
        
        # Add the tiles
        ax.add_image(tile_source, zoom)
        
    except Exception as e:
        print(f"Error adding {tile_name} tiles: {e}")
        print("This might be due to projection mismatch or network issues")
        raise
