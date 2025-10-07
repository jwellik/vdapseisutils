"""
Map tile functionality for vdapseisutils.

This module provides methods for adding various map tile sources to cartopy maps,
including ArcGIS terrain tiles and Google Maps tiles.

Author: Jay Wellik, jwellik@vdap.org
Last updated: 2025-01-27
"""

import os
import ssl
import urllib.request
import cartopy.io.img_tiles as cimgt


def _create_ssl_context(ssl_verify=False):
    """
    Create an SSL context for tile requests.
    
    Parameters:
    -----------
    ssl_verify : bool
        Whether to verify SSL certificates (default: True)
        
    Returns:
    --------
    ssl.SSLContext or None
        SSL context for urllib requests, None if verification is enabled
    """
    if not ssl_verify:
        # Create unverified context for testing/debugging
        return ssl._create_unverified_context()
    else:
        # Use default context (with verification)
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            # Fall back to default context if certifi not available
            return ssl.create_default_context()


def _create_tile_with_ssl_context(url, cache=False, ssl_verify=False):
    """
    Create a GoogleTiles object with SSL context handling.
    
    Parameters:
    -----------
    url : str
        URL template for the tile source
    cache : bool
        Whether to cache tiles
    ssl_verify : bool
        Whether to verify SSL certificates
        
    Returns:
    --------
    cimgt.GoogleTiles
        Configured tile source
    """
    ssl_context = _create_ssl_context(ssl_verify)
    
    # Create custom opener if SSL context is needed
    if ssl_context:
        https_handler = urllib.request.HTTPSHandler(context=ssl_context)
        opener = urllib.request.build_opener(https_handler)
        # Set as default opener for urllib
        urllib.request.install_opener(opener)
    
    return cimgt.GoogleTiles(
        cache=cache,
        url=url,
        desired_tile_form='RGB'
    )


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


def add_arcgis_terrain(ax, zoom='auto', cache=False, radial_extent_km=None, verbose=False, ssl_verify=False):
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
    verbose : bool, optional
        Whether to print URLs and zoom levels to console (default: False)
    ssl_verify : bool, optional
        Whether to verify SSL certificates (default: False)
        Set to True for enhanced security in production
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

    if verbose:
        print(f"ArcGIS Terrain Tiles:")
        print(f"  Zoom level: {zoom_level}")
        print(f"  SSL verification: {ssl_verify}")
        print(f"  Terrain URL: https://services.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer/tile/{zoom_level}/{{y}}/{{x}}")
        print(f"  Overlay URL: https://tiles.basemaps.cartocdn.com/light_nolabels/{zoom_level}/{{x}}/{{y}}.png")

    # Add arcgis terrain and transparent shading
    # (I stole this two-pronged approach from Alicia Hotovec Ellis and REDPy, circa 2025 September)
    # - Terrain--Nice hillshade tile
    terrain_url = (
        'https://services.arcgisonline.com/arcgis/rest/services'
        '/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}'
    )
    terrain = _create_tile_with_ssl_context(terrain_url, cache=cache, ssl_verify=ssl_verify)
    ax.add_image(terrain, zoom_level)

    # - Overlay--shading provides contrast for land/sea
    overlay_url = (
        'https://tiles.basemaps.cartocdn.com/light_nolabels/'
        '{z}/{x}/{y}.png'
    )
    overlay = _create_tile_with_ssl_context(overlay_url, cache=cache, ssl_verify=ssl_verify)
    ax.add_image(overlay, zoom_level, alpha=0.5)  # Reduced alpha for better contrast


def add_google_tile(ax, zoom='auto', style='terrain', cache=False, radial_extent_km=None, verbose=False, ssl_verify=False, **kwargs):
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
    verbose : bool, optional
        Whether to print URLs and zoom levels to console (default: False)
    ssl_verify : bool, optional
        Whether to verify SSL certificates (default: False)
        Set to True for enhanced security in production
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
    
    tile_url = f'https://mt1.google.com/vt/lyrs={lyrs}&x={{x}}&y={{y}}&z={{z}}'
    
    if verbose:
        print(f"Google {style.capitalize()} Tiles:")
        print(f"  Zoom level: {zoom_level}")
        print(f"  SSL verification: {ssl_verify}")
        print(f"  URL template: {tile_url}")
    
    background_tile = _create_tile_with_ssl_context(tile_url, cache=cache, ssl_verify=ssl_verify)
    
    # Check if axes projection matches tile projection
    if hasattr(ax, 'projection') and ax.projection != background_tile.crs:
        print(f"Warning: Axes projection {ax.projection} doesn't match tile projection {background_tile.crs}")
        print("Consider using the same projection for both axes and tiles")
    
    ax.add_image(background_tile, zoom_level)


def add_google_terrain(ax, zoom='auto', cache=False, radial_extent_km=None, verbose=False, ssl_verify=False, **kwargs):
    """Add Google terrain tiles to the map."""
    add_google_tile(ax, zoom, style='terrain', cache=cache, radial_extent_km=radial_extent_km, verbose=verbose, ssl_verify=ssl_verify, **kwargs)


def add_google_street(ax, zoom='auto', cache=False, radial_extent_km=None, verbose=False, ssl_verify=False, **kwargs):
    """Add Google street tiles to the map."""
    add_google_tile(ax, zoom, style='street', cache=cache, radial_extent_km=radial_extent_km, verbose=verbose, ssl_verify=ssl_verify, **kwargs)


def add_google_satellite(ax, zoom='auto', cache=False, radial_extent_km=None, verbose=False, ssl_verify=False, **kwargs):
    """Add Google satellite tiles to the map."""
    add_google_tile(ax, zoom, style='satellite', cache=cache, radial_extent_km=radial_extent_km, verbose=verbose, ssl_verify=ssl_verify, **kwargs)


def add_tile_with_projection_check(ax, tile_source, zoom, tile_name="tile", verbose=False):
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
    verbose : bool, optional
        Whether to print URLs and zoom levels to console (default: False)
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
        if verbose:
            print(f"{tile_name.capitalize()} Tiles:")
            print(f"  Zoom level: {zoom}")
            if hasattr(tile_source, 'url'):
                print(f"  URL template: {tile_source.url}")
        
        ax.add_image(tile_source, zoom)
        
    except Exception as e:
        print(f"Error adding {tile_name} tiles: {e}")
        print("This might be due to projection mismatch or network issues")
        raise
