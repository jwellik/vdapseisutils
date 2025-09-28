"""
Utility functions shared across multiple map classes.

This module contains utility functions that are used by multiple classes
in the maps module, including data preparation and scale bar utilities.

Author: Jay Wellik, jwellik@vdap.org
Last updated: 2025 September 28
"""

import numpy as np
from vdapseisutils.utils.obspyutils.catalogutils import catalog2txyzm
import pandas as pd


def prep_catalog_data_mpl(catalog, s="magnitude", c="time", maglegend=None, time_format="matplotlib"):
    """
    PREPCATALOG Converts ObsPy Catalog object to DataFrame w fields appropriate for swarmmpl

    TODO Allow for custom MagLegends
    TODO Add color column
    TODO Filter catalog to extents and return nRemoved

    :return:
    """
    # Import here to avoid circular imports
    from .legends import MagLegend
    
    if maglegend is None:
        maglegend = MagLegend()

    ## Get info out of Events object
    # returns time(UTCDateTime), lat, lon, depth(km, positive below sea level), mag
    catdata = catalog2txyzm(catalog, time_format=time_format)
    catdata = pd.DataFrame(catdata).sort_values("time")
    catdata["depth"] *= -1  # below sea level values are negative for swarmmpl purposes
    catdata["size"] = maglegend.mag2s(catdata["mag"])  # converts magnitudes to point size for scatter plot
    return catdata


def get_scale_length(origin, distance_km):
    """
    Convert real-world distance in km to degrees at a given latitude.
    Uses the WGS84 ellipsoid for accuracy.
    """
    from pyproj import Geod

    lat, lon = origin
    geod = Geod(ellps="WGS84")
    end_lon, end_lat, _ = geod.fwd(lon, lat, 90, distance_km * 1000)  # Move eastward
    return abs(end_lon - lon)  # Return the degree difference


def choose_scale_bar_length(map_width_km, fraction=0.25):
    """
    Choose an appropriate scale bar length based on map width.
    
    Given the width of the map in km, return the scale bar length (in km) as the value
    from ALLOWED_SCALES that is the largest value less than or equal to fraction * map_width_km.
    
    Parameters:
    -----------
    map_width_km : float
        Width of the map in kilometers
    fraction : float, optional
        Fraction of map width to target for scale bar length (default: 0.25)
        
    Returns:
    --------
    int
        Scale bar length in kilometers from the predefined ALLOWED_SCALES list
        
    Notes:
    ------
    The function uses a predefined list of standard scale bar lengths:
    [0.1, 0.2, 0.5, 1, 5, 10, 20, 50, 100, 150, 200, 250, 500, 750, 1000, 5000, 10000] km
    """
    candidate = map_width_km * fraction
    # Choose the largest allowed scale that is less than or equal to candidate.
    ALLOWED_SCALES = [0.1, 0.2, 0.5, 1, 5, 10, 20, 50, 100, 150, 200, 250, 500, 750, 1000, 5000, 10000]
    valid_scales = [scale for scale in ALLOWED_SCALES if scale <= candidate]
    if valid_scales:
        scale = max(valid_scales)
    else:
        scale = min(ALLOWED_SCALES)  # Fallback to smallest scale if candidate is too small
    return scale


def _test_utils():
    """Simple test to verify utils module works correctly."""
    try:
        # Test choose_scale_bar_length
        scale_100km = choose_scale_bar_length(100)  # Should return 25 km (25% of 100)
        scale_10km = choose_scale_bar_length(10)    # Should return 2.5 -> 1 km
        
        print(f"✓ Scale bar for 100km map: {scale_100km} km")
        print(f"✓ Scale bar for 10km map: {scale_10km} km")
        
        # Test get_scale_length
        origin = (45.0, -122.0)  # Portland area
        degrees = get_scale_length(origin, 10)  # 10 km distance
        print(f"✓ 10 km at lat {origin[0]} = {degrees:.4f} degrees")
        
        print("✓ Utils module test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        return False


if __name__ == "__main__":
    _test_utils()